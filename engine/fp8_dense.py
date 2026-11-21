"""FP8-E4M3 weight-only dense GEMV for a single decode token (per-output-row fp32 scale, fp32 accumulate).

quantize(W) -> (q fp8 (N, K), s fp32 (N,), W' bf16) with s = amax_row / 448 and W' = bf16(q * s), the lossy weight
the prefill path uses so that every path (prefill, NLL, decode) sees the same quantized model.
fp8_gemv(x, q, s) -> bf16 (N,) = bf16(s * sum_k q[n, k] x[k]); registered as a triton_op so inductor-compiled regions
can call it. Small matrices use split-K (fp32 partials + a fixed-order reduce kernel, deterministic) so the grid fills
the 188 SMs.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

FP8 = torch.float8_e4m3fn


@triton.jit
def _fp8_gemv(X, W, S, OUT, PART, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """grid (cdiv(N, BLOCK_N), SPLIT_K). SPLIT_K == 1: OUT = bf16(s * W x). Else PART[split, n] = fp32 partial sums."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mn = offs_n < N
    wbase = W + offs_n[:, None].to(tl.int64) * K
    KS = K // SPLIT_K
    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k0 in range(ps * KS, ps * KS + KS, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + offs_k).to(tl.float32)
        w = tl.load(wbase + offs_k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        acc += w * x[None, :]
    r = tl.sum(acc, 1)
    if SPLIT_K == 1:
        r = r * tl.load(S + offs_n, mask=mn, other=0.0)
        tl.store(OUT + offs_n, r.to(tl.bfloat16), mask=mn)
    else:
        tl.store(PART + ps * N + offs_n, r, mask=mn)


@triton.jit
def _splitk_reduce(PART, S, OUT, N, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    acc = tl.load(PART + offs, mask=m, other=0.0)
    for i in tl.static_range(1, SPLIT_K):                       # fixed order -> deterministic
        acc += tl.load(PART + i * N + offs, mask=m, other=0.0)
    acc = acc * tl.load(S + offs, mask=m, other=0.0)
    tl.store(OUT + offs, acc.to(tl.bfloat16), mask=m)


# (BLOCK_N, BLOCK_K, num_warps, SPLIT_K) by (N, K); fallback by size. Chosen by scratch/check_splitk.py (in-graph, cold).
CFG = {(6552, 2304): (8, 128, 8, 1), (1024, 2304): (8, 128, 8, 1), (2304, 3072): (8, 128, 8, 1)}   # split-K never won
DEFAULT = {"small": (16, 256, 4, 1), "big": (32, 256, 4, 1)}   # big = unembed


def _cfg(N, K, cfg):
    BN, BK, nw, sk = cfg or CFG.get((N, K)) or (DEFAULT["big"] if N > 50000 else DEFAULT["small"])
    while K % (BK * sk):
        if BK > 32:
            BK //= 2
        else:
            sk, BK = 1, 256
    return BN, BK, nw, sk


def _launch(lg, lr, x, q, s, cfg):
    N, K = q.shape
    out = torch.empty(N, dtype=torch.bfloat16, device=q.device)
    BN, BK, nw, sk = _cfg(N, K, cfg)
    part = torch.empty((sk, N) if sk > 1 else (1,), dtype=torch.float32, device=q.device)
    lg(_fp8_gemv)[(triton.cdiv(N, BN), sk)](x, q, s, out, part, N, K, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=sk,
                                            num_warps=nw)
    if sk > 1:
        lr(_splitk_reduce)[(triton.cdiv(N, 1024),)](part, s, out, N, SPLIT_K=sk, BLOCK=1024, num_warps=4)
    return out


@triton_op("inkling::fp8_gemv", mutates_args={})
def fp8_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return _launch(wrap_triton, wrap_triton, x.reshape(-1).contiguous(), q, s, None)


def _ident(k):
    return k


def fp8_gemv_cfg(x, q, s, cfg):
    """Eager variant with an explicit (BLOCK_N, BLOCK_K, num_warps, SPLIT_K) config (for sweeps)."""
    return _launch(_ident, _ident, x.reshape(-1).contiguous(), q, s, cfg)


def quantize(W):
    """W (N, K) bf16 -> (q fp8 (N, K), s fp32 (N,), bf16 dequant (N, K))."""
    qs, ss, ds = [], [], []
    for c in range(0, W.shape[0], 32768):                      # chunked: the unembed is 201k rows
        w = W[c:c + 32768].float()
        s = (w.abs().amax(1) / 448.0).clamp(min=1e-12)
        q = (w / s[:, None]).to(FP8)
        qs.append(q); ss.append(s); ds.append((q.float() * s[:, None]).to(torch.bfloat16))
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous(), torch.cat(ds).contiguous()
