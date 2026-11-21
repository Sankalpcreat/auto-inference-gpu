"""Fused T=1 decode attention partials: per (kv head, key block) online-softmax pieces in fp32 (no tl.dot -> no TF32).
attn_partial(q (H,HD) fp32, bias (H,S) fp32 incl. -inf mask, K, V (HKV, cap, HD) bf16, S) -> m (H,NS), l (H,NS), acc (H,NS,HD).
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

BS = 64
G = 3          # q heads per kv head


@triton.jit
def _attn_part(Q, BIAS, K, V, M, L, ACC, S, sg, NS, scale, BS: tl.constexpr, HD: tl.constexpr, G: tl.constexpr):
    g = tl.program_id(0)
    ps = tl.program_id(1)
    offs_s = ps * BS + tl.arange(0, BS)
    offs_d = tl.arange(0, HD)
    ms = offs_s < S
    base = g * sg + offs_s[:, None] * HD + offs_d[None, :]
    k = tl.load(K + base, mask=ms[:, None], other=0.0).to(tl.float32)
    v = tl.load(V + base, mask=ms[:, None], other=0.0).to(tl.float32)
    for h in tl.static_range(G):
        hh = g * G + h
        q = tl.load(Q + hh * HD + offs_d)
        sc = tl.sum(k * q[None, :], 1) * scale + tl.load(BIAS + hh * S + offs_s, mask=ms, other=float("-inf"))
        m = tl.max(sc, 0)
        msafe = tl.where(m == float("-inf"), 0.0, m)
        p = tl.exp(sc - msafe)
        tl.store(M + hh * NS + ps, m)
        tl.store(L + hh * NS + ps, tl.sum(p, 0))
        tl.store(ACC + (hh * NS + ps) * HD + offs_d, tl.sum(v * p[:, None], 0))


@triton_op("inkling::attn_partial", mutates_args={})
def attn_partial(q: torch.Tensor, bias: torch.Tensor, K: torch.Tensor, V: torch.Tensor, S: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, HD = q.shape
    NS = triton.cdiv(S, BS)
    m = torch.empty(H, NS, dtype=torch.float32, device=q.device)
    l = torch.empty(H, NS, dtype=torch.float32, device=q.device)
    acc = torch.empty(H, NS, HD, dtype=torch.float32, device=q.device)
    wrap_triton(_attn_part)[(H // G, NS)](q, bias, K, V, m, l, acc, S, K.stride(0), NS, HD ** -0.5,
                                          BS=BS, HD=HD, G=G, num_warps=4)
    return m, l, acc
