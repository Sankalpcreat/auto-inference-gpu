"""Loop-message blend at T=1 in 3 Triton kernels (same math as fused_t1.blend; reduction order / tanh formula differ).

dest/msg (S, 1, D) fp32 (any stride on dim 0, e.g. an expanded base); down (S, 64, 2D), gain/forget (S, D, 64) fp32.
"""
import torch
import triton
import triton.language as tl

HID = 64
BH = 2        # hidden rows per program in kernel A
BD = 64       # output rows per program in kernel B
BC = 256      # columns per program in kernel C


@triton.jit
def _tanh(x):
    e = tl.exp(-2.0 * tl.abs(x))
    t = (1.0 - e) / (1.0 + e)
    return tl.where(x < 0, -t, t)


@triton.jit
def _down(DST, MSG, sd, sm, DOWN, H, D, BH: tl.constexpr, BLK: tl.constexpr):
    s = tl.program_id(0)
    ph = tl.program_id(1)
    o = tl.arange(0, BLK)
    mk = o < D
    d = tl.load(DST + s * sd + o, mask=mk, other=0.0)
    m = tl.load(MSG + s * sm + o, mask=mk, other=0.0)
    dn = (d * tl.rsqrt(tl.sum(d * d, 0) / D + 1e-6)).to(tl.bfloat16).to(tl.float32)
    mn = (m * tl.rsqrt(tl.sum(m * m, 0) / D + 1e-6)).to(tl.bfloat16).to(tl.float32)
    for j in tl.static_range(BH):
        h = ph * BH + j
        row = DOWN + (s * 64 + h).to(tl.int64) * (2 * D)
        wd = tl.load(row + o, mask=mk, other=0.0)
        wm = tl.load(row + D + o, mask=mk, other=0.0)
        r = tl.sum(wd * dn + wm * mn, 0)
        tl.store(H + s * 64 + h, r.to(tl.bfloat16).to(tl.float32))


@triton.jit
def _gf(DST, MSG, sd, sm, H, GAIN, FORGET, Y, PS, D, NB, BD: tl.constexpr):
    s = tl.program_id(0)
    pb = tl.program_id(1)
    rows = pb * BD + tl.arange(0, BD)
    k = tl.arange(0, 64)
    h = tl.load(H + s * 64 + k)
    off = (s * D + rows).to(tl.int64)[:, None] * 64 + k[None, :]
    g = tl.sum(tl.load(GAIN + off) * h[None, :], 1)
    f = tl.sum(tl.load(FORGET + off) * h[None, :], 1)
    d = tl.load(DST + s * sd + rows)
    m = tl.load(MSG + s * sm + rows)
    y = (1.0 - _tanh(f)) * d + _tanh(g) * m
    tl.store(Y + s * D + rows, y)
    tl.store(PS + s * NB + pb, tl.sum(y * y, 0))


@triton.jit
def _norm(Y, PS, NW, sw, OUT, D, NB, BC: tl.constexpr, NBP: tl.constexpr, EPS: tl.constexpr):
    s = tl.program_id(0)
    pc = tl.program_id(1)
    b = tl.arange(0, NBP)
    ss = tl.sum(tl.load(PS + s * NB + b, mask=b < NB, other=0.0), 0)
    inv = tl.rsqrt(ss / D + EPS)
    c = pc * BC + tl.arange(0, BC)
    mk = c < D
    y = tl.load(Y + s * D + c, mask=mk, other=0.0)
    w = tl.load(NW + s * sw + c, mask=mk, other=0.0).to(tl.float32)
    tl.store(OUT + s * D + c, (y * inv * w).to(tl.bfloat16), mask=mk)


def blend(dest, msg, down, gain, forget, norm_w):
    S, _, D = dest.shape
    dev = dest.device
    nw = norm_w
    assert nw.stride(1) == 1 and dest.dtype == torch.float32 and msg.dtype == torch.float32
    h = torch.empty(S, HID, device=dev, dtype=torch.float32)
    NB = D // BD
    y = torch.empty(S, D, device=dev, dtype=torch.float32)
    ps = torch.empty(S, NB, device=dev, dtype=torch.float32)
    out = torch.empty(S, 1, D, device=dev, dtype=torch.bfloat16)
    sd, sm = dest.stride(0), msg.stride(0)
    assert dest.stride(2) == 1 and msg.stride(2) == 1 and D % BD == 0
    blk = triton.next_power_of_2(D)
    _down[(S, HID // BH)](dest, msg, sd, sm, down, h, D, BH=BH, BLK=blk, num_warps=8)
    _gf[(S, NB)](dest, msg, sd, sm, h, gain, forget, y, ps, D, NB, BD=BD, num_warps=4)
    _norm[(S, triton.cdiv(D, BC))](y, ps, nw, nw.stride(0), out, D, NB, BC=BC, NBP=triton.next_power_of_2(NB), EPS=1e-6,
                                   num_warps=4)
    return out
