"""T-row packed expert GEMV for a speculative verify step: each DISTINCT expert is read once for all T <= 16 rows.

ids (T, 8) top-8 per row. Programs run over the 8T sorted ids; a program whose id equals its predecessor exits, so the
weight bytes scale with the number of distinct experts, not 8T. The dequantized tile (same bf16 rounding as
moe_packed / weights.py) is tl.dot-ed against a 16-row block of x; rows not routed to the expert are masked.
Output (T*8, N) fp32 in (row, slot) order, i.e. row t*8+k = expert ids[t, k] applied to row t.
  w13: x (T, K), shared over the 8 slots of a row.   w2: x (T*8, K), one vector per (row, slot).
"""
import torch
import triton
import triton.language as tl

from .moe_packed import _e2m1


@triton.jit
def _route(SID, IDS, s, T, shared_x: tl.constexpr):
    e = tl.load(SID + s)
    prev = tl.load(SID + tl.maximum(s - 1, 0))
    dup = (s > 0) & (prev == e)
    rt = tl.arange(0, 16)
    kk = tl.arange(0, 8)
    ids2 = tl.load(IDS + rt[:, None] * 8 + kk[None, :], mask=rt[:, None] < T, other=-1)
    m = ids2 == e
    hit = tl.max(m.to(tl.int32), 1) > 0
    slot = tl.sum(tl.where(m, kk[None, :], 0), 1)
    orow = rt * 8 + slot
    xrow = rt if shared_x else orow
    return e.to(tl.int64), dup, hit, orow, xrow


@triton.jit
def _v_nvfp4(W, S, OUTER, X, SID, IDS, OUT, T, N, K, stride_we, stride_wn, stride_se, stride_sn,
             shared_x: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr):
    s = tl.program_id(0)
    pn = tl.program_id(1)
    e, dup, hit, orow, xrow = _route(SID, IDS, s, T, shared_x)
    if not dup:
        outer = tl.load(OUTER + e)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
        xbase = X + xrow[None, :].to(tl.int64) * K
        acc = tl.zeros([BLOCK_N, 16], dtype=tl.float32)
        for b0 in range(0, K // 2, BLOCK_B):
            offs_b = b0 + tl.arange(0, BLOCK_B)
            xe = tl.load(xbase + 2 * offs_b[:, None], mask=hit[None, :], other=0.0)
            xo = tl.load(xbase + 2 * offs_b[:, None] + 1, mask=hit[None, :], other=0.0)
            w = tl.load(wbase + offs_b[None, :]).to(tl.int32)
            sc = tl.load(sbase + (offs_b // 8)[None, :]).to(tl.float32) * 16384.0
            lo = ((_e2m1(w & 15) * sc) * outer).to(tl.bfloat16)
            hi = ((_e2m1(w >> 4) * sc) * outer).to(tl.bfloat16)
            acc = tl.dot(lo, xe, acc)
            acc = tl.dot(hi, xo, acc)
        tl.store(OUT + orow[None, :] * N + offs_n[:, None], acc, mask=hit[None, :])


@triton.jit
def _v_mxfp8(W, S, X, SID, IDS, OUT, T, N, K, stride_we, stride_wn, stride_se, stride_sn,
             shared_x: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    s = tl.program_id(0)
    pn = tl.program_id(1)
    e, dup, hit, orow, xrow = _route(SID, IDS, s, T, shared_x)
    if not dup:
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
        xbase = X + xrow[None, :].to(tl.int64) * K
        acc = tl.zeros([BLOCK_N, 16], dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(xbase + offs_k[:, None], mask=hit[None, :], other=0.0)
            w = tl.load(wbase + offs_k[None, :]).to(tl.float32)
            sc = tl.exp2(tl.load(sbase + (offs_k // 32)[None, :]).to(tl.float32) - 127.0)
            acc = tl.dot((w * sc).to(tl.bfloat16), x, acc)
        tl.store(OUT + orow[None, :] * N + offs_n[:, None], acc, mask=hit[None, :])


CFG = {"nvfp4": (32, 128, 4), "mxfp8": (32, 128, 4)}


def verify_gemv(P, x, ids, shared_x, cfg=None):
    """P packed expert dict; ids (T, 8) int64 cuda; x (T, K) if shared_x else (T*8, K) bf16 -> (T*8, N) fp32."""
    w, s = P["w"], P["s"]
    T = ids.shape[0]
    assert T <= 16 and ids.shape[1] == 8
    N = w.shape[1]
    ids = ids.contiguous()
    sid = ids.flatten().sort().values
    x = x.contiguous()
    out = torch.empty(T * 8, N, dtype=torch.float32, device=w.device)
    BN, BK, nw = cfg or CFG[P["kind"]]
    grid = (T * 8, N // BN)
    if P["kind"] == "nvfp4":
        K = w.shape[2] * 2
        _v_nvfp4[grid](w, s, P["outer"], x, sid, ids, out, T, N, K, w.stride(0), w.stride(1), s.stride(0), s.stride(1),
                       shared_x=shared_x, BLOCK_N=BN, BLOCK_B=BK, num_warps=nw)
    else:
        K = w.shape[2]
        _v_mxfp8[grid](w, s, x, sid, ids, out, T, N, K, w.stride(0), w.stride(1), s.stride(0), s.stride(1),
                       shared_x=shared_x, BLOCK_N=BN, BLOCK_K=BK, num_warps=nw)
    return out
