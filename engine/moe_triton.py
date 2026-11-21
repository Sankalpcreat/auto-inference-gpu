"""Sync-free MoE for a single decode token: router + top-8 stay on GPU; experts read in place by a grouped GEMV.

out[j, n] = sum_k W[ids[j], n, k] * x[j, k]   (x shared across slots when stride_xj == 0), fp32 accumulate.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .model_v0 import GATE_CAP, ROUTE_SCALE, TOPK, UP_CAP

BLOCK_N, BLOCK_K = 32, 128


@triton.jit
def _grouped_gemv(W, X, IDS, OUT, N, K, stride_we, stride_wn, stride_xj,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    j = tl.program_id(0)
    pn = tl.program_id(1)
    e = tl.load(IDS + j).to(tl.int64)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
    xbase = X + j * stride_xj
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(xbase + offs_k).to(tl.float32)
        w = tl.load(wbase + offs_k[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(OUT + j * N + offs_n, acc)


def grouped_gemv(W, x, ids, shared_x):
    """W (E, N, K) bf16 contiguous, x (K,) if shared_x else (S, K) bf16, ids (S,) int -> (S, N) fp32."""
    E, N, K = W.shape
    S = ids.shape[0]
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and W.is_contiguous() and x.is_contiguous()
    out = torch.empty(S, N, dtype=torch.float32, device=W.device)
    _grouped_gemv[(S, N // BLOCK_N)](W, x, ids, out, N, K, W.stride(0), W.stride(1), 0 if shared_x else K,
                                     BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out


def moe_t1(lw, x):
    """x (1, D) bf16 -> (1, D) bf16. Same math as InklingV0.moe, no host syncs."""
    logits = x.float() @ lw["mlp.gate.weight"].float().T
    _, ids = torch.topk(logits.sigmoid() + lw["mlp.gate.bias"].float(), TOPK, sorted=True)
    w = F.logsigmoid(logits.gather(1, ids)).softmax(-1) * ROUTE_SCALE            # (1, 8)
    e = ids[0].contiguous()
    gu = grouped_gemv(lw["w13"], x[0].contiguous(), e, True)                    # (8, 2I) fp32
    gt, up = gu[:, 0::2], gu[:, 1::2]
    act = GATE_CAP * torch.tanh(gt / GATE_CAP) * torch.sigmoid(gt) * UP_CAP * torch.tanh(up / UP_CAP)
    act = (act * w[0][:, None]).to(torch.bfloat16).contiguous()                  # (8, I)
    y = grouped_gemv(lw["w2"], act, e, False)                                   # (8, D) fp32
    return y.sum(0, keepdim=True).to(torch.bfloat16)
