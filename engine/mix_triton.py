"""attn_res at T=1 as ONE Triton kernel (same math as model_v0.attn_res; reduction order differs).

sources: list of N (1, D) tensors (bf16 or fp32, unit last stride); queries (Q, D) fp32 pre-scaled.
mix(...) -> (Q, 1, D) fp32;  mix_norm(..., norm_w (Q|1, D)) -> (Q, 1, D) bf16 = rms(mix) * norm_w.
One program per query; online softmax over sources, full row in registers.
"""
import torch
import triton
import triton.language as tl

EPS = 1e-6


@triton.jit
def _mix(SRC, QRY, NW, sw, OUT, D, N: tl.constexpr, BLK: tl.constexpr, NORM: tl.constexpr, EPS: tl.constexpr):
    q = tl.program_id(0)
    o = tl.arange(0, BLK)
    mk = o < D
    qv = tl.load(QRY + q * D + o, mask=mk, other=0.0)
    m = -float("inf")
    l = 0.0
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for n in tl.static_range(N):
        x = tl.load(SRC[n] + o, mask=mk, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
        lg = tl.sum(qv * x, 0) * inv
        mn = tl.maximum(m, lg)
        a = tl.exp(m - mn)
        e = tl.exp(lg - mn)
        l = l * a + e
        acc = acc * a + (e * inv) * x
        m = mn
    y = acc / l
    if NORM:
        y = y * tl.rsqrt(tl.sum(y * y, 0) / D + EPS)
        w = tl.load(NW + q * sw + o, mask=mk, other=0.0).to(tl.float32)
        tl.store(OUT + q * D + o, (y * w).to(tl.bfloat16), mask=mk)
    else:
        tl.store(OUT + q * D + o, y, mask=mk)


def _run(sources, queries, norm_w):
    Q, D = queries.shape
    assert queries.is_contiguous() and queries.dtype == torch.float32
    for s in sources:
        assert s.shape[-1] == D and s.stride(-1) == 1 and s.numel() == D
    norm = norm_w is not None
    out = torch.empty(Q, 1, D, device=queries.device, dtype=torch.bfloat16 if norm else torch.float32)
    if norm:
        assert norm_w.stride(-1) == 1 and norm_w.shape[-1] == D
        nw, sw = norm_w, (norm_w.stride(0) if norm_w.shape[0] == Q and Q > 1 else 0)
    else:
        nw, sw = queries, 0
    _mix[(Q,)](tuple(sources), queries, nw, sw, out, D, N=len(sources), BLK=triton.next_power_of_2(D), NORM=norm,
               EPS=EPS, num_warps=8)
    return out


def mix(sources, queries):
    return _run(sources, queries, None)


def mix_norm(sources, queries, norm_w):
    return _run(sources, queries, norm_w)
