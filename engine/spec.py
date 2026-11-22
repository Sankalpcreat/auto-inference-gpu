"""Graphed speculative decoding: DSpark draft -> T=K+1 verify on the T=1 kernel family -> on-device accept + commit.

One CUDA graph per (K, global bucket B) does the whole step. State is device-resident:
  m.tok (1,)  anchor = last accepted token at position m.pos;  m.pos (1,) its position.
Verify does not touch conv histories / local rings; it stores cat(hist, inputs) and the new local K/V in static
buffers, and the commit (inside the same graph) keeps the first n = accepted + 1 rows. Global K/V are written for all
T rows (rows past n are masked by every later step and rewritten).
Kernels mirror the T=1 decode: FP8 weight-only GEMV (M-row variant, weights read once), packed NVFP4/MXFP8 expert GEMV
per (row, expert) in one launch, same attention math per row.
"""
import functools
import torch
import torch.nn.functional as F

from . import model_v1 as M1
from . import fused_t1 as FT
from .fp8_dense import fp8_rows
from .moe_packed import packed_gemv
from .model_v0 import (ALPHA, BLOCK, CORE, EXT, GLOBAL, H, HD, HKV, DREL, LOGIT_CAP, N_FLOOR, QK_CLIP, REL_CLIP,
                       ROUTE_SCALE, STREAMS, TOPK, VOCAB, WINDOW, attn_res, l2_clip, rms)

BF = torch.bfloat16
G = H // HKV
FEAT_VISITS = (5, 23, 41)
compile_ = functools.partial(torch.compile, fullgraph=True, dynamic=False)


def _conv(full, wT):
    """full (T+3, C) = cat(hist, x); wT (4, C) fp32 oldest first."""
    x = full[3:]
    win = full.float().unfold(0, 4, 1)                         # (T, C, 4) oldest first
    return (((win * wT.T[None]).sum(-1) + x.float()) * 5 ** -0.5).to(x.dtype)


def _dyn_conv(full, static_w, dyn):
    """static_w (4, C) newest first, dyn (T, HKV, 4)."""
    x = full[3:]
    C = x.shape[1]
    win = full.float().unfold(0, 4, 1).flip(-1)                # (T, C, 4) newest first
    taps = static_w.float().T[None] + dyn.float().repeat_interleave(C // HKV, 1)   # (T, C, 4)
    return (((taps * win).sum(-1) + x.float()) * 9 ** -0.5).to(x.dtype)


def _qkv(xq, w, k, v, T):
    y = fp8_rows(xq, w["qrgs8"], w["qrgss"])
    q = y[:, :H * HD].view(T, H, HD)
    r = y[:, H * HD:H * HD + H * DREL].view(T, H, DREL)
    gate = y[:, H * HD + H * DREL:2 * H * HD + H * DREL]
    sinks = y[:, 2 * H * HD + H * DREL:].float()
    k = l2_clip(k.view(T, HKV, HD), QK_CLIP).to(BF)
    v = rms(v.view(T, HKV, HD).to(BF)).to(BF)
    qc = l2_clip(q * w["qscale"].float(), QK_CLIP).to(BF)
    rc = l2_clip(r * w["rscale"].float(), REL_CLIP).to(BF)
    return qc, rc, gate, sinks, k, v


def _attend(qc, rel, idx, mul, add, K, V, gate, sinks, w, T):
    S = idx.shape[1]
    bias = rel.gather(2, idx[:, None, :].expand(T, H, S)) * mul[:, None] + add[:, None]       # (T, H, S)
    q = qc.float().view(T, HKV, G, HD).permute(1, 2, 0, 3).reshape(HKV, G * T, HD)
    sc = (q @ K.transpose(1, 2)) * HD ** -0.5
    sc = sc + bias.view(T, HKV, G, S).permute(1, 2, 0, 3).reshape(HKV, G * T, S)
    lse = sc.logsumexp(-1).view(HKV, G, T).permute(2, 0, 1).reshape(T, H)
    out = (sc.softmax(-1) @ V).view(HKV, G, T, HD).permute(2, 0, 1, 3).reshape(T, H, HD)
    out = out * torch.sigmoid(gate.float().view(T, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
    return fp8_rows(out.to(BF).reshape(T, H * HD), w["wo8"], w["wos"])


@compile_
def attn_local(xq, xk, xv, w, hk, hv, ha, fk, fv, fa, K, V, kb, vb, idx, mul, add):
    T = xq.shape[0]
    full = torch.cat([hk, fp8_rows(xk, w["wk8"], w["wks"])]); fk.copy_(full); k = _conv(full, w["kT"])
    full = torch.cat([hv, fp8_rows(xv, w["wv8"], w["wvs"])]); fv.copy_(full); v = _conv(full, w["vT"])
    qc, rc, gate, sinks, k, v = _qkv(xq, w, k, v, T)
    rel = torch.einsum("thd,de->the", rc.float(), w["relp"])
    kt, vt = k.transpose(0, 1).float(), v.transpose(0, 1).float()
    kb.copy_(kt); vb.copy_(vt)
    a = _attend(qc, rel, idx, mul, add, torch.cat([K, kt], 1), torch.cat([V, vt], 1), gate, sinks, w, T)
    full = torch.cat([ha, a]); fa.copy_(full)
    return _conv(full, w["aT"])


@compile_
def attn_global(xq, xk, xv, w, hk, hv, ha, fk, fv, fa, K, V, qp, idx, mul, add):
    T, B = xq.shape[0], idx.shape[1]
    full = torch.cat([hk, fp8_rows(xk, w["wk8"], w["wks"])]); fk.copy_(full)
    k = _dyn_conv(full, w["k_static"], (xk @ w["k_proj"].T).view(T, HKV, 4))
    full = torch.cat([hv, fp8_rows(xv, w["wv8"], w["wvs"])]); fv.copy_(full)
    v = _dyn_conv(full, w["v_static"], (xv @ w["v_proj"].T).view(T, HKV, 4))
    qc, rc, gate, sinks, k, v = _qkv(xq, w, k, v, T)
    tau = 1 + ALPHA * torch.log(torch.clamp((qp.float() + 1) / N_FLOOR, min=1.0))
    qc = (qc.float() * tau[:, None, None]).to(BF)
    rc = (rc.float() * tau[:, None, None]).to(BF)
    rel = torch.einsum("thd,de->the", rc.float(), w["relp"])
    K.index_copy_(1, qp, k.transpose(0, 1).float())
    V.index_copy_(1, qp, v.transpose(0, 1).float())
    a = _attend(qc, rel, idx, mul, add, K[:, :B], V[:, :B], gate, sinks, w, T)
    full = torch.cat([ha, a]); fa.copy_(full)
    return _conv(full, w["aT"])


@compile_
def blend(dest, msg, down, gain, forget, norm_w):
    """dest/msg fp32 (S, T, D); down (S, 64, 2D); gain/forget (S, D, 64); norm_w (S, D) -> bf16 (S, T, D)."""
    pair = torch.stack((dest, msg), 2)                                                  # (S, T, 2, D)
    pair = (pair * torch.rsqrt(pair.pow(2).mean(-1, keepdim=True) + 1e-6)).to(BF).float().flatten(2)
    h = torch.bmm(pair, down.transpose(1, 2)).to(BF).float()                           # (S, T, 64)
    g = torch.bmm(h, gain.transpose(1, 2))
    f = torch.bmm(h, forget.transpose(1, 2))
    y = (1 - torch.tanh(f)) * dest + torch.tanh(g) * msg
    return (rms(y) * norm_w.float()[:, None, :]).to(BF)


@compile_
def router(x, gw, gb):
    logits = x.float() @ gw.T
    _, ids = torch.topk(logits.sigmoid() + gb, TOPK, sorted=True)
    w = F.logsigmoid(logits.gather(1, ids)).softmax(-1) * ROUTE_SCALE
    return ids.reshape(-1).contiguous(), w.reshape(-1).contiguous()


@compile_
def moe_out(y, hm, fm, wT):
    T = y.shape[0] // TOPK
    full = torch.cat([hm, y.view(T, TOPK, -1).sum(1).to(BF)])
    fm.copy_(full)
    return _conv(full, wT)


@compile_
def logits_argmax(h, q, s):
    lg = fp8_rows(h, q, s).float()[:, :VOCAB]
    return (LOGIT_CAP * torch.tanh(lg / LOGIT_CAP)).argmax(-1)


class Spec:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.graphs, self.bufs = {}, {}
        self.cap = 0

    def _sync_cap(self):
        if self.m.cache["cap"] != self.cap:
            self.cap = self.m.cache["cap"]
            self.d.reset(self.cap + 64)
            self.graphs = {}

    def _buffers(self, T):
        if T in self.bufs:
            return self.bufs[T]
        m, out = self.m, []
        for (i, _), c in zip(M1.VISITS, m.cache["visits"]):
            b = {s: torch.empty(T + 3, c[s].shape[1], dtype=c[s].dtype, device=m.dev) for s in STREAMS}
            if i not in GLOBAL:
                b["kb"] = torch.empty(HKV, T, HD, device=m.dev)
                b["vb"] = torch.empty(HKV, T, HD, device=m.dev)
            out.append(b)
        self.bufs[T] = out
        return out

    def _ctx(self, T, B):
        m, dev = self.m, self.m.dev
        pos = m.pos
        arT = torch.arange(T, device=dev)
        qp = pos + arT
        s = torch.arange(WINDOW, device=dev)
        kp = pos - 1 - torch.remainder(pos - 1 - s, WINDOW)
        dr = qp[:, None] - kp[None]
        vr = (kp >= 0)[None] & (dr <= WINDOW - 1)
        dn = arT[:, None] - arT[None]
        dist = torch.cat([dr, dn], 1)
        valid = torch.cat([vr, dn >= 0], 1)
        ninf, zero = torch.full((), float("-inf"), device=dev), torch.zeros((), device=dev)
        loc = (dist.clamp(0, EXT - 1), (dist < EXT).float(), torch.where(valid, zero, ninf))
        gd = qp[:, None] - m.ar[:B][None]
        glo = (gd.clamp(0, EXT - 1), (gd < EXT).float(), torch.where(gd >= 0, zero, ninf))
        return qp, arT, loc, glo

    def verify(self, toks, T, B):
        """toks (T,) at positions pos..pos+T-1 -> (argmax (T,), feats (T, 4, D))."""
        m = self.m
        bufs = self._buffers(T)
        qp, arT, loc, glo = self._ctx(T, B)
        x0 = (rms(m.embed[toks]) * m.embed_norm.float()).to(BF)
        stack, nb, prelude, bank, feats = [x0], 0, None, None, []
        for v, (i, p) in enumerate(M1.VISITS):
            lw, c, t, b = m.layers[i], m.cache["visits"][v], m.t1[i], bufs[v]
            if i == CORE.start and p == 0:
                prelude = (nb, stack[nb], list(stack[:nb]))
            elif i == CORE.start:
                pb, ppart, pblocks = prelude
                bank = [stack[pb + 1], stack[pb + 2]]
                stack, nb = pblocks + [ppart], pb
            partial = stack[nb]
            if v in FEAT_VISITS:
                feats.append(partial.to(BF))
            if p > 0:
                base = FT.mix(stack[:nb] + [partial], m.attn_q[i])
                msgs = FT.mix(bank, m.msg_q[i - CORE.start])
                if base.shape[0] == 1:
                    base = base.expand(3, -1, -1)
                nw = lw["attn_norm"] if lw["attn_norm"].shape[0] == 3 else lw["attn_norm"].expand(3, -1)
                xin = blend(base.contiguous(), msgs[:3].contiguous(), *t["ctl3"], nw.contiguous())
            elif v == 0:
                xin = (rms(partial)[None] * lw["attn_norm"].float()[:, None, :]).to(BF)
            else:
                xin = FT.mix_norm(stack[:nb] + [partial], m.attn_q[i], lw["attn_norm"])
            xq = xin[0].contiguous()
            xk = xin[1].contiguous() if xin.shape[0] == 3 else xq
            xv = xin[2].contiguous() if xin.shape[0] == 3 else xq
            if i in GLOBAL:
                attn_out = attn_global(xq, xk, xv, t["attn"], c["k"], c["v"], c["attn"], b["k"], b["v"], b["attn"],
                                       c["K"], c["V"], qp, *glo)
            else:
                attn_out = attn_local(xq, xk, xv, t["attn"], c["k"], c["v"], c["attn"], b["k"], b["v"], b["attn"],
                                      c["K"], c["V"], b["kb"], b["vb"], *loc)
            first = i % BLOCK == 0
            src = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [partial.float() + attn_out.float()]
            if p > 0:
                mlp_in = blend(FT.mix(src, m.mlp_q[i]).contiguous(), msgs[3:].contiguous(), *t["ctl1"],
                               lw["mlp_norm.weight"][None].contiguous())[0]
            else:
                mlp_in = FT.mix_norm(src, m.mlp_q[i], lw["mlp_norm.weight"][None])[0]
            ids, w = router(mlp_in, t["gw"], t["gb"])
            pk = m.packed[i]
            xr = mlp_in.repeat_interleave(TOPK, 0).contiguous()
            gu = packed_gemv(pk["w13"], xr, ids, False)
            y = packed_gemv(pk["w2"], FT.act(gu, w), ids, False)
            mconv = moe_out(y, c["mlp"], b["mlp"], t["mT"])
            if first:
                nb += 1
                new = FT.residual2(attn_out, mconv)
            else:
                new = FT.residual(partial, attn_out, mconv)
            stack = stack[:nb] + [new]
        h = FT.mix_norm(stack[:nb] + [stack[nb]], m.out_q, m.norm[None])[0]
        feats.append(h)
        return logits_argmax(h, *m.unembed8), torch.stack(feats, 1), arT

    def commit(self, n, T, arT):
        m = self.m
        idx3 = n + torch.arange(3, device=m.dev)
        slots = torch.remainder(m.pos + arT, WINDOW)
        keep = (arT < n)[None, :, None]
        for (i, _), c, b in zip(M1.VISITS, m.cache["visits"], self.bufs[T]):
            for s in STREAMS:
                c[s].copy_(b[s].index_select(0, idx3))
            if i not in GLOBAL:
                c["K"].index_copy_(1, slots, torch.where(keep, b["kb"], c["K"].index_select(1, slots)))
                c["V"].index_copy_(1, slots, torch.where(keep, b["vb"], c["V"].index_select(1, slots)))

    def _step(self, K, B):
        m, d = self.m, self.d
        T = K + 1
        dr = d.draft_g(m.tok, m.pos, K, B)                           # (K,)
        targ, feats, arT = self.verify(torch.cat([m.tok, dr]), T, B)
        a = torch.cumprod((dr == targ[:K]).long(), 0).sum()
        n = a + 1
        d.append_g(feats, m.pos)
        self.commit(n, T, arT)
        self.out[K][:T].copy_(targ)
        self.out[K][T] = n
        m.tok.copy_(targ.index_select(0, a.view(1)))
        m.pos.add_(n)

    @torch.inference_mode()
    def generate(self, ids, n_out, K):
        m, d = self.m, self.d
        P = ids.shape[0]
        m.ensure_cap(P + n_out + K + 2)
        self._sync_cap()
        logits, feats = prefill_feats(m, ids)
        d.n = 0
        d.append(feats)
        first = int(logits[-1].argmax())
        res = [first]
        if n_out == 1:
            return res
        if not hasattr(self, "out"):
            self.out = {}
        self.out.setdefault(K, torch.zeros(K + 2, dtype=torch.long, device=m.dev))
        m.tok.fill_(first)
        m.pos.fill_(P)
        pos = P
        while len(res) < n_out:
            B = min(self.cap, -(-(pos + K + 1) // M1.BUCKET) * M1.BUCKET)
            g = self.graphs.get((K, B))
            if g is None:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._buffers(K + 1)
                    self._step(K, B)
                torch.cuda.current_stream().wait_stream(side)
                o = self.out[K].tolist()
                snap = (m.tok.clone(), m.pos.clone())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._step(K, B)
                m.tok.copy_(snap[0]); m.pos.copy_(snap[1])
                self.graphs[(K, B)] = g
            else:
                g.replay()
                o = self.out[K].tolist()
            nacc = o[K + 1]
            res += o[:nacc]
            pos += nacc
        return res[:n_out]


_REC = {}


def prefill_feats(m, ids):
    """Prefill (m.forward) + DSpark features: partial block entering visits 5/23/41 and the final normed h."""
    orig_ar, orig_attn = M1.attn_res, M1.InklingV1.attention_c
    attn_q_ids = {id(q) for q in m.attn_q if q is not None}
    rec = {"b": []}

    def hook_ar(sources, queries, norm_w=None, norm_eps=1e-6):
        out = orig_ar(sources, queries, norm_w, norm_eps)
        if id(queries) in attn_q_ids:
            rec["pend"] = sources[-1].to(BF)
        elif queries is m.out_q:
            rec["final"] = out[0].to(BF)
        return out

    def hook_attn(self, i, c, xq, xk, xv, pos, dctx):
        rec["b"].append(rec.pop("pend", None))
        return orig_attn(self, i, c, xq, xk, xv, pos, dctx)

    M1.attn_res, M1.InklingV1.attention_c = hook_ar, hook_attn
    try:
        logits, _ = m.forward(ids)
    finally:
        M1.attn_res, M1.InklingV1.attention_c = orig_ar, orig_attn
    assert len(rec["b"]) == len(M1.VISITS)
    return logits, torch.stack([rec["b"][v] for v in FEAT_VISITS] + [rec["final"]], 1)
