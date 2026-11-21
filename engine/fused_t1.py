"""T=1 decode regions as clean tensor-only functions, compiled by inductor (fullgraph, static shapes, no cudagraphs).

Called inside the whole-step CUDA-graph capture of model_v1 (compiled during the eager warm-up step first).
Same math as model_v0 / model_v1 decode; only the kernel boundaries (fusion, reduction order) change.
"""
import functools
import torch
import torch.nn.functional as F

from .model_v0 import (ALPHA, DREL, EPS, GATE_CAP, H, HD, HKV, N_FLOOR, QK_CLIP, REL_CLIP, ROUTE_SCALE, TOPK,
                       UP_CAP, attn_res, l2_clip, rms)
from .fp8_dense import fp8_gemv
from .attn_t1 import attn_partial

for _k, _v in (("cache_size_limit", 512), ("accumulated_cache_size_limit", 4096)):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, _v)

compile_ = functools.partial(torch.compile, fullgraph=True, dynamic=False)
BF = torch.bfloat16


def _hist(x, hist):
    """x (1, C), hist (3, C) same dtype -> (4, C) fp32; hist <- last 3 rows (in place)."""
    full = torch.cat([hist, x])
    hist.copy_(full[1:])
    return full.float()


def _conv(x, hist, wT):
    """Static conv, wT (4, C) fp32 oldest tap first."""
    xp = _hist(x, hist)
    return (((xp * wT).sum(0, keepdim=True) + x.float()) * 5 ** -0.5).to(x.dtype)


def _dyn_conv(x, hist, static_w, dyn):
    """Dynamic conv, static_w (4, C) newest first, dyn (1, HKV, 4)."""
    C = x.shape[1]
    xp = _hist(x, hist)
    taps = static_w.float() + dyn[0].float().T.repeat_interleave(C // dyn.shape[1], 1)
    return (((taps * xp.flip(0)).sum(0, keepdim=True) + x.float()) * 9 ** -0.5).to(x.dtype)


def _lin(x, w, name):
    """FP8 weight-only GEMV: x (1, K) bf16 -> (1, N) bf16."""
    return fp8_gemv(x, w[name + "8"], w[name + "s"]).view(1, -1)


def _qkv(xq, xk, xv, w, k, v):
    y = _lin(xq, w, "qrgs")                                     # wq | wr | gate | sink, one FP8 GEMV
    q = y[:, :H * HD].view(1, H, HD)
    r = y[:, H * HD:H * HD + H * DREL].view(1, H, DREL)
    gate = y[:, H * HD + H * DREL:2 * H * HD + H * DREL]
    sinks = y[:, 2 * H * HD + H * DREL:].float()
    k = l2_clip(k.view(1, HKV, HD), QK_CLIP).to(BF)
    v = rms(v.view(1, HKV, HD).to(BF)).to(BF)
    qc = l2_clip(q * w["qscale"].float(), QK_CLIP).to(BF)
    rc = l2_clip(r * w["rscale"].float(), REL_CLIP).to(BF)
    return qc, rc, gate, sinks, k, v


def _attend(qc, rel_bias, K, V, S, gate, sinks, wo):
    """rel_bias (H, S) fp32 (-inf masked); K/V (HKV, cap, HD) bf16, first S keys used."""
    m, l, acc = attn_partial(qc.float().view(H, HD).contiguous(), rel_bias.contiguous(), K, V, S)
    M = m.amax(1)
    Ms = torch.where(M == float("-inf"), torch.zeros_like(M), M)
    w = torch.exp(m - Ms[:, None])
    L = (l * w).sum(1)
    out = ((acc * w[..., None]).sum(1) / L[:, None]).view(1, H, HD)
    lse = (Ms + torch.log(L)).view(1, H)
    out = out * torch.sigmoid(gate.float().view(1, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
    return _lin(out.to(BF).reshape(1, H * HD), wo, "wo")


@compile_
def attn_local(xq, xk, xv, w, hk, hv, ha, K, V, slot, ldist, ladd):
    """Local layer decode attention + attn-output conv. Returns attn_out (1, D) bf16."""
    k = _conv(_lin(xk, w, "wk"), hk, w["kT"])
    v = _conv(_lin(xv, w, "wv"), hv, w["vT"])
    qc, rc, gate, sinks, k, v = _qkv(xq, xk, xv, w, k, v)
    rel = torch.einsum("thd,de->the", rc.float(), w["relp"])
    K.index_copy_(1, slot, k.transpose(0, 1))
    V.index_copy_(1, slot, v.transpose(0, 1))
    bias = rel[0][:, ldist] + ladd
    a = _attend(qc, bias, K, V, bias.shape[1], gate, sinks, w)
    return _conv(a, ha, w["aT"])


@compile_
def attn_global(xq, xk, xv, w, hk, hv, ha, K, V, pos, gidx, gmul, gadd):
    """Global layer decode attention over a static bucket of B = gidx.shape[0] keys + attn-output conv."""
    B = gidx.shape[0]
    k = _dyn_conv(_lin(xk, w, "wk"), hk, w["k_static"], (xk @ w["k_proj"].T).view(1, HKV, 4))
    v = _dyn_conv(_lin(xv, w, "wv"), hv, w["v_static"], (xv @ w["v_proj"].T).view(1, HKV, 4))
    qc, rc, gate, sinks, k, v = _qkv(xq, xk, xv, w, k, v)
    tau = 1 + ALPHA * torch.log(torch.clamp((pos.float() + 1) / N_FLOOR, min=1.0))
    qc = (qc.float() * tau[:, None, None]).to(BF)
    rc = (rc.float() * tau[:, None, None]).to(BF)
    rel = torch.einsum("thd,de->the", rc.float(), w["relp"])
    K.index_copy_(1, pos, k.transpose(0, 1))
    V.index_copy_(1, pos, v.transpose(0, 1))
    bias = rel[0][:, gidx] * gmul + gadd
    a = _attend(qc, bias, K, V, B, gate, sinks, w)
    return _conv(a, ha, w["aT"])


@compile_
def mix(sources, queries):
    return attn_res(sources, queries)


@compile_
def mix_norm(sources, queries, norm_w):
    return attn_res(sources, queries, norm_w).to(BF)


@compile_
def blend(dest, msg, down, gain, forget, norm_w):
    """dest/msg fp32 (S, 1, D); down (S, 64, 2D), gain/forget (S, D, 64) fp32; norm_w (S, D) -> bf16 (S, 1, D)."""
    d, m = dest[:, 0], msg[:, 0]
    pair = torch.stack((d, m), 1)                                                     # (S, 2, D)
    pair = (pair * torch.rsqrt(pair.pow(2).mean(-1, keepdim=True) + 1e-6)).to(BF).float().flatten(1)
    h = torch.bmm(down, pair[:, :, None]).to(BF).float()                              # (S, 64, 1)
    g = torch.bmm(gain, h)[..., 0]
    f = torch.bmm(forget, h)[..., 0]
    y = (1 - torch.tanh(f)) * d + torch.tanh(g) * m
    return (rms(y) * norm_w.float()).to(BF)[:, None, :]


@compile_
def router(x, gw, gb):
    logits = x.float() @ gw.T
    _, ids = torch.topk(logits.sigmoid() + gb, TOPK, sorted=True)
    w = F.logsigmoid(logits.gather(1, ids)).softmax(-1) * ROUTE_SCALE
    return ids[0].contiguous(), w[0]


@compile_
def act(gu, w):
    gt, up = gu[:, 0::2], gu[:, 1::2]
    a = GATE_CAP * torch.tanh(gt / GATE_CAP) * torch.sigmoid(gt) * UP_CAP * torch.tanh(up / UP_CAP)
    return (a * w[:, None]).to(BF).contiguous()


@compile_
def moe_out_conv(y, hist, wT):
    return _conv(y.sum(0, keepdim=True).to(BF), hist, wT)


@compile_
def residual(partial, attn_out, mconv):
    return (partial.float() + attn_out.float() + mconv.float()).to(BF)


@compile_
def residual2(attn_out, mconv):
    return (attn_out.float() + mconv.float()).to(BF)


def prep(lw, i, is_global, is_core, streams):
    """Precompute per-layer fp32 / stacked weights used by the compiled T=1 functions (once, at load).
    FP8 dense weights lw["fp8"][name] = (q, s) come from InklingV1.quantize_fp8."""
    f8 = lw["fp8"]
    w = {"qrgs8": f8["qrgs"][0], "qrgss": f8["qrgs"][1], "wk8": f8["wk"][0], "wks": f8["wk"][1],
         "wv8": f8["wv"][0], "wvs": f8["wv"][1], "wo8": f8["wo"][0], "wos": f8["wo"][1], "qscale": lw["attn.q_l2_clip.scale"], "rscale": lw["attn.rel_l2_clip.scale"],
         "relp": lw["attn.rel_logits_proj.proj"].float(),
         "aT": lw["attn_sconv.weight"].squeeze(1).float().T.contiguous()}
    if is_global:
        for n in "kv":
            w[f"{n}_static"], w[f"{n}_proj"] = lw[f"{n}_static"], lw[f"{n}_proj"]
    else:
        w["kT"] = lw["k_conv"].float().T.contiguous()
        w["vT"] = lw["v_conv"].float().T.contiguous()
    t = {"attn": w, "mT": lw["mlp_sconv.weight"].squeeze(1).float().T.contiguous(),
         "gw": lw["mlp.gate.weight"].float().contiguous(), "gb": lw["mlp.gate.bias"].float().contiguous()}
    if is_core:
        for name, sts in (("ctl3", streams[:3]), ("ctl1", streams[3:])):
            t[name] = tuple(torch.stack([lw[f"ctl.{s}.{n}"].float() for s in sts]).contiguous()
                            for n in ("down", "gain", "forget"))
    return t
