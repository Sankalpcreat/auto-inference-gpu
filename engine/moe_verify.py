"""Verify-step (T rows) packed expert GEMV, row-outer design (study #21).

Flat ids (T*8,) = (row t, slot k) -> j = t*8 + k. Program (j, pn) owns expert e = ids[j] only if j is the first
occurrence of e; it then loops (static) over every j2 with ids[j2] == e and runs packed_gemv's exact inner loop for
that row, so each distinct expert's tile is fetched from DRAM once and re-read from L1/L2 by the later rows.
Output (T*8, N) fp32 == torch.cat of per-row packed_gemv (bit-identical math per row).
"""
import torch
import triton
import triton.language as tl

from .moe_packed import _e2m1, CFG


@triton.jit
def _owner(IDS, j, NJ, JB: tl.constexpr):
    offs = tl.arange(0, JB)
    allids = tl.load(IDS + offs, mask=offs < NJ, other=-1)
    e = tl.load(IDS + j)
    dup = tl.sum(((allids == e) & (offs < j)).to(tl.int32), 0)
    return e, dup


@triton.jit
def _v_nvfp4(W, S, OUTER, X, IDS, OUT, N, K, NJ, stride_we, stride_wn, stride_se, stride_sn, XDIV,
             BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, JB: tl.constexpr):
    j = tl.program_id(0)
    pn = tl.program_id(1)
    e32, dup = _owner(IDS, j, NJ, JB)
    if dup == 0:
        e = e32.to(tl.int64)
        outer = tl.load(OUTER + e)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
        for j2 in tl.static_range(JB):
            if j2 < NJ:
                if tl.load(IDS + j2) == e32:
                    xbase = X + (j2 // XDIV) * K
                    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
                    for b0 in range(0, K // 2, BLOCK_B):
                        offs_b = b0 + tl.arange(0, BLOCK_B)
                        xe = tl.load(xbase + 2 * offs_b).to(tl.float32)
                        xo = tl.load(xbase + 2 * offs_b + 1).to(tl.float32)
                        w = tl.load(wbase + offs_b[None, :]).to(tl.int32)
                        s = tl.load(sbase + (offs_b // 8)[None, :]).to(tl.float32) * 16384.0
                        lo = ((_e2m1(w & 15) * s) * outer).to(tl.bfloat16).to(tl.float32)
                        hi = ((_e2m1(w >> 4) * s) * outer).to(tl.bfloat16).to(tl.float32)
                        acc += tl.sum(lo * xe[None, :] + hi * xo[None, :], 1)
                    tl.store(OUT + j2 * N + offs_n, acc)


@triton.jit
def _v_mxfp8(W, S, X, IDS, OUT, N, K, NJ, stride_we, stride_wn, stride_se, stride_sn, XDIV,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, JB: tl.constexpr):
    j = tl.program_id(0)
    pn = tl.program_id(1)
    e32, dup = _owner(IDS, j, NJ, JB)
    if dup == 0:
        e = e32.to(tl.int64)
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        wbase = W + e * stride_we + offs_n[:, None].to(tl.int64) * stride_wn
        sbase = S + e * stride_se + offs_n[:, None].to(tl.int64) * stride_sn
        for j2 in tl.static_range(JB):
            if j2 < NJ:
                if tl.load(IDS + j2) == e32:
                    xbase = X + (j2 // XDIV) * K
                    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
                    for k0 in range(0, K, BLOCK_K):
                        offs_k = k0 + tl.arange(0, BLOCK_K)
                        x = tl.load(xbase + offs_k).to(tl.float32)
                        w = tl.load(wbase + offs_k[None, :]).to(tl.float32)
                        s = tl.exp2(tl.load(sbase + (offs_k // 32)[None, :]).to(tl.float32) - 127.0)
                        w = (w * s).to(tl.bfloat16).to(tl.float32)
                        acc += tl.sum(w * x[None, :], 1)
                    tl.store(OUT + j2 * N + offs_n, acc)


def verify_gemv(P, x, ids, shared_x, cfg=None):
    """P packed dict; ids (T, 8); x (T, K) if shared_x (one x per row) else (T*8, K) -> (T*8, N) fp32."""
    w, s = P["w"], P["s"]
    N = w.shape[1]
    flat = ids.reshape(-1).to(torch.int32).contiguous()
    NJ = flat.numel()
    JB = triton.next_power_of_2(NJ)
    xdiv = 8 if shared_x else 1
    x = x.contiguous()
    out = torch.empty(NJ, N, dtype=torch.float32, device=w.device)
    BN, BK, nw = cfg or CFG[P["kind"]]
    if P["kind"] == "nvfp4":
        K = w.shape[2] * 2
        while (K // 2) % BK:
            BK //= 2
        _v_nvfp4[(NJ, N // BN)](w, s, P["outer"], x, flat, out, N, K, NJ, w.stride(0), w.stride(1),
                                s.stride(0), s.stride(1), xdiv, BLOCK_N=BN, BLOCK_B=BK, JB=JB, num_warps=nw)
    else:
        K = w.shape[2]
        while K % BK:
            BK //= 2
        _v_mxfp8[(NJ, N // BN)](w, s, x, flat, out, N, K, NJ, w.stride(0), w.stride(1),
                                s.stride(0), s.stride(1), xdiv, BLOCK_N=BN, BLOCK_K=BK, JB=JB, num_warps=nw)
    return out
