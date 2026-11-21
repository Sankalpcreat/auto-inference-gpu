"""FP8-E4M3 weight-only dense GEMV for a single decode token (per-output-row fp32 scale, fp32 accumulate).

quantize(W) -> (q fp8 (N, K), s fp32 (N,), W' bf16) with s = amax_row / 448 and W' = bf16(q * s), the lossy weight
the prefill path uses so that every path (prefill, NLL, decode) sees the same quantized model.
fp8_gemv(x, q, s) -> bf16 (N,) = bf16(s * sum_k q[n, k] x[k]); registered as a triton_op so inductor-compiled regions
can call it.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

FP8 = torch.float8_e4m3fn


@triton.jit
def _fp8_gemv(X, W, S, OUT, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pn = tl.program_id(0)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    wbase = W + offs_n[:, None].to(tl.int64) * K
    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k).to(tl.float32)
        w = tl.load(wbase + offs_k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        acc += w * x[None, :]
    r = tl.sum(acc, 1) * tl.load(S + offs_n, mask=mn, other=0.0)
    tl.store(OUT + offs_n, r.to(tl.bfloat16), mask=mn)


CFG = {"small": (16, 256, 4), "big": (32, 256, 4)}   # (BLOCK_N, BLOCK_K, num_warps); big = unembed


def _cfg(N, K, cfg):
    BN, BK, nw = cfg or (CFG["big"] if N > 50000 else CFG["small"])
    while K % BK:
        BK //= 2
    return BN, BK, nw


@triton_op("inkling::fp8_gemv", mutates_args={})
def fp8_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    x = x.reshape(-1).contiguous()
    out = torch.empty(N, dtype=torch.bfloat16, device=q.device)
    BN, BK, nw = _cfg(N, K, None)
    wrap_triton(_fp8_gemv)[(triton.cdiv(N, BN),)](x, q, s, out, N, K, BLOCK_N=BN, BLOCK_K=BK, num_warps=nw)
    return out


def fp8_gemv_cfg(x, q, s, cfg):
    """Eager variant with an explicit config (for sweeps)."""
    N, K = q.shape
    out = torch.empty(N, dtype=torch.bfloat16, device=q.device)
    BN, BK, nw = _cfg(N, K, cfg)
    _fp8_gemv[(triton.cdiv(N, BN),)](x, q, s, out, N, K, BLOCK_N=BN, BLOCK_K=BK, num_warps=nw)
    return out


def quantize(W):
    """W (N, K) bf16 -> (q fp8 (N, K), s fp32 (N,), bf16 dequant (N, K))."""
    qs, ss, ds = [], [], []
    for c in range(0, W.shape[0], 32768):                      # chunked: the unembed is 201k rows
        w = W[c:c + 32768].float()
        s = (w.abs().amax(1) / 448.0).clamp(min=1e-12)
        q = (w / s[:, None]).to(FP8)
        qs.append(q); ss.append(s); ds.append((q.float() * s[:, None]).to(torch.bfloat16))
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous(), torch.cat(ds).contiguous()
