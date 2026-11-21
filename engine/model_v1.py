"""Engine v1: v0 math + incremental decoding with a shape-static, CUDA-graphed decode step.

Cross-token state lives only in (a) attention K/V and (b) the last 3 inputs of each short conv
(streams k, v, attn-out, mlp-out). AttnRes mixes, loop messages and MoE are per token. Each of the 48 layer
visits (layer, pass) owns its own cache, exactly like the reference's per-pass KV and conv-state caches.

The cache is persistent (owned by the model, reset by every prefill):
- local layers: fp32 K/V ring of WINDOW slots (slot = pos % WINDOW), zero-initialised;
- global layers: fp32 K/V of `cap` slots; decode attends a static bucket of B keys (masked);
- conv histories: static (3, C) buffers updated in place.
Decode (T=1) reads the position from a device tensor, so one step = one CUDA graph per global bucket B.
"""
import torch
import torch.nn.functional as F

from .model_v0 import (ALPHA, BLOCK, CORE, DREL, EXT, GLOBAL, H, HD, HKV, L, LOGIT_CAP, N_FLOOR, PASSES, QK_CLIP,
                       REL_CLIP, STREAMS, VOCAB, WINDOW, InklingV0, attn_res, l2_clip, rms)
from .moe_triton import moe_t1

VISITS = [(i, 0) for i in range(CORE.start)] + [(i, p) for p in range(PASSES) for i in CORE] + \
         [(i, 0) for i in range(CORE.stop, L)]
BUCKET = 1024


def _with_hist(cache, key, x, decode):
    """Prepend the stored last-3 inputs of stream `key` to x (T, C); update the static (3, C) store in place.
    Prefill always starts from zero history. Returns (T+3, C)."""
    if decode:
        hist = cache[key]
        full = torch.cat([hist, x])
        hist.copy_(full[1:])
        return full
    full = torch.cat([x.new_zeros(3, x.shape[1]), x])
    if key in cache:
        cache[key].copy_(full[-3:])
    else:
        cache[key] = full[-3:].clone()
    return full


def _f32(lw, key):
    """Cached fp32 copy of a small weight."""
    k = "_f32." + key
    if k not in lw:
        lw[k] = lw[key].float()
    return lw[k]


def conv_c(cache, key, x, w, lw=None, wkey=None, decode=False):
    """Static causal conv with history: y = (sum_k w[c,k] x[t-3+k] + x[t]) / sqrt(5)."""
    T = x.shape[0]
    xp = _with_hist(cache, key, x, decode).float()
    if T == 1 and lw is not None:
        k = "_convT." + wkey
        if k not in lw:
            lw[k] = w.float().T.contiguous()                                # (4, C), oldest tap first
        return (((xp * lw[k]).sum(0, keepdim=True) + x.float()) * 5 ** -0.5).to(x.dtype)
    acc = sum(w[:, k].float() * xp[k:k + T] for k in range(4))
    return ((acc + x.float()) * 5 ** -0.5).to(x.dtype)


def dyn_conv_c(cache, key, x, static_w, dyn, decode=False):
    T, C = x.shape
    xp = _with_hist(cache, key, x, decode).float()
    if T == 1:
        taps = static_w.float() + dyn[0].float().T.repeat_interleave(C // dyn.shape[1], 1)   # (4, C), newest first
        return (((taps * xp.flip(0)).sum(0, keepdim=True) + x.float()) * 9 ** -0.5).to(x.dtype)
    acc = torch.zeros(T, C, device=x.device)
    for l in range(4):
        taps = static_w[l].float()[None] + dyn[:, :, l].float().repeat_interleave(C // dyn.shape[1], 1)
        acc += taps * xp[3 - l:3 - l + T]
    return ((acc + x.float()) * 9 ** -0.5).to(x.dtype)


class InklingV1(InklingV0):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cache = {"cap": 0, "visits": [dict() for _ in VISITS]}
        for (i, _), c in zip(VISITS, self.cache["visits"]):
            if i not in GLOBAL:
                c["K"] = torch.zeros(HKV, WINDOW, HD, device=self.dev)
                c["V"] = torch.zeros(HKV, WINDOW, HD, device=self.dev)
        self.tok = torch.zeros(1, dtype=torch.long, device=self.dev)      # static decode input / output
        self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.graphs = {}
        self.ensure_cap(4096)

    def ensure_cap(self, need):
        if need <= self.cache["cap"]:
            return
        cap = max(4096, 1 << (need - 1).bit_length())
        for (i, _), c in zip(VISITS, self.cache["visits"]):
            if i in GLOBAL:
                c["K"] = torch.zeros(HKV, cap, HD, device=self.dev)
                c["V"] = torch.zeros(HKV, cap, HD, device=self.dev)
        self.cache["cap"] = cap
        self.ar = torch.arange(cap, device=self.dev)
        self.graphs = {}                                                  # buffer pointers changed

    def moe(self, i, x):
        if x.shape[0] == 1:
            return moe_t1(self.layers[i], x)
        return super().moe(i, x)

    def attention_c(self, i, c, xq, xk, xv, pos, dctx):
        lw, T = self.layers[i], xq.shape[0]
        is_global, decode = i in GLOBAL, dctx is not None
        q = (xq @ lw["attn.wq_du.weight"].T).view(T, H, HD)
        r = (xq @ lw["attn.wr_du.weight"].T).view(T, H, DREL)
        gate = xq @ lw["attn.gate_attn.weight"].T
        sinks = (xq @ lw["attn.attn_sink_proj.weight"].T).float()
        k_raw, v_raw = xk @ lw["attn.wk_dv.weight"].T, xv @ lw["attn.wv_dv.weight"].T
        if is_global:
            k = dyn_conv_c(c, "k", k_raw, lw["k_static"], (xk @ lw["k_proj"].T).view(T, HKV, 4), decode)
            v = dyn_conv_c(c, "v", v_raw, lw["v_static"], (xv @ lw["v_proj"].T).view(T, HKV, 4), decode)
        else:
            k = conv_c(c, "k", k_raw, lw["k_conv"], lw, "k", decode)
            v = conv_c(c, "v", v_raw, lw["v_conv"], lw, "v", decode)
        k = l2_clip(k.view(T, HKV, HD), QK_CLIP).to(torch.bfloat16)
        v = rms(v.view(T, HKV, HD).to(torch.bfloat16)).to(torch.bfloat16)
        qc = l2_clip(q * lw["attn.q_l2_clip.scale"].float(), QK_CLIP).to(torch.bfloat16)
        rc = l2_clip(r * lw["attn.rel_l2_clip.scale"].float(), REL_CLIP).to(torch.bfloat16)
        if is_global and (decode or T > N_FLOOR):          # tau == 1 exactly while every pos + 1 <= N_FLOOR
            tau = 1 + ALPHA * torch.log(torch.clamp((pos.float() + 1) / N_FLOOR, min=1.0))
            qc = (qc.float() * tau[:, None, None]).to(torch.bfloat16)
            rc = (rc.float() * tau[:, None, None]).to(torch.bfloat16)
        rel = torch.einsum("thd,de->the", rc.float(), _f32(lw, "attn.rel_logits_proj.proj"))
        if decode:
            kt, vt = k.transpose(0, 1).float(), v.transpose(0, 1).float()                    # (HKV, 1, HD)
            if is_global:
                c["K"].index_copy_(1, pos, kt)
                c["V"].index_copy_(1, pos, vt)
                B = dctx["B"]
                Kw, Vw = c["K"][:, :B], c["V"][:, :B]
                bias = rel[0][:, dctx["gidx"]] * dctx["gmul"] + dctx["gadd"]                   # (H, B)
            else:
                c["K"].index_copy_(1, dctx["slot"], kt)
                c["V"].index_copy_(1, dctx["slot"], vt)
                Kw, Vw = c["K"], c["V"]
                bias = rel[0][:, dctx["ldist"]] + dctx["ladd"]                                 # (H, WINDOW)
            scores = (qc.float().view(HKV, H // HKV, HD) @ Kw.transpose(1, 2)) * HD ** -0.5  # (HKV, 3, S)
            scores = scores + bias.view(HKV, H // HKV, -1)
            lse = scores.logsumexp(-1).view(1, H)
            out = (scores.softmax(-1) @ Vw).view(1, H, HD)
        else:                                               # prefill from an empty cache (positions 0..T-1)
            kf, vf = k.float(), v.float()
            if is_global:
                c["K"][:, :T] = kf.transpose(0, 1)
                c["V"][:, :T] = vf.transpose(0, 1)
            else:
                W = min(T, WINDOW)
                slots = torch.arange(T - W, T, device=xq.device) % WINDOW
                c["K"][:, slots] = kf[T - W:].transpose(0, 1)
                c["V"][:, slots] = vf[T - W:].transpose(0, 1)
            dist = pos[:, None] - pos[None, :]
            bias = rel.gather(2, dist.clamp(0, EXT - 1)[:, None, :].expand(-1, H, -1))
            bias = bias.masked_fill(~((dist >= 0) & (dist < EXT))[:, None, :], 0.0)
            kk = kf.repeat_interleave(H // HKV, 1)
            vv = vf.repeat_interleave(H // HKV, 1)
            scores = torch.einsum("qhd,khd->hqk", qc.float(), kk) * HD ** -0.5 + bias.transpose(0, 1)
            mask = dist < 0
            if not is_global:
                mask = mask | (dist > WINDOW - 1)
            scores = scores.masked_fill(mask[None], float("-inf"))
            lse = scores.logsumexp(-1).T
            out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
        out = out * torch.sigmoid(gate.float().view(T, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
        return out.to(torch.bfloat16).reshape(T, H * HD) @ lw["attn.wo_ud.weight"].T

    def _decode_ctx(self, pos, B):
        """Per-step index/mask tensors shared by all visits (pos (1,) device)."""
        ninf = float("-inf")
        gd = pos - self.ar[:B]                                                            # (B,) key distance
        ld = torch.remainder(pos - self.ar[:WINDOW], WINDOW)                              # (WINDOW,) ring distance
        zero = torch.zeros((), device=self.dev)
        return {"B": B, "slot": torch.remainder(pos, WINDOW),
                "gidx": gd.clamp(0, EXT - 1),
                "gadd": torch.where(gd < 0, ninf, zero),                                  # future/unwritten
                "gmul": (gd < EXT).float(),                                               # rel-bias extent
                "ldist": ld, "ladd": torch.where(ld > pos, ninf, zero)}

    @torch.inference_mode()
    def forward(self, ids, cache=None, pos=None, B=None):
        """Prefill (pos None): ids (T,) from an empty cache -> (fp32 logits (T, VOCAB), cache).
        Decode (pos (1,) device tensor, B global bucket): appends 1 token."""
        decode = pos is not None
        T = ids.shape[0]
        if not decode:
            self.ensure_cap(T)
            pos = torch.arange(T, device=self.dev)
        cache = self.cache
        dctx = self._decode_ctx(pos, B) if decode else None
        x0 = (rms(self.embed[ids]) * self.embed_norm.float()).to(torch.bfloat16)
        stack, nb, prelude, bank = [x0], 0, None, None
        for v, (i, p) in enumerate(VISITS):
            lw, c = self.layers[i], cache["visits"][v]
            if i == CORE.start and p == 0:
                prelude = (nb, stack[nb].clone(), [s.clone() for s in stack[:nb]])
            elif i == CORE.start:
                pb, ppart, pblocks = prelude
                bank = [stack[pb + 1], stack[pb + 2]]
                stack, nb = pblocks + [ppart], pb
            partial = stack[nb]
            if p > 0:
                base = attn_res(stack[:nb] + [partial], self.attn_q[i])
                msgs = attn_res(bank, self.msg_q[i - CORE.start])
                if base.shape[0] == 1:
                    base = base.expand(3, -1, -1)
                nw = lw["attn_norm"] if lw["attn_norm"].shape[0] == 3 else lw["attn_norm"].expand(3, -1)
                xin = self.blend(i, STREAMS[:3], base, msgs[:3], nw)
            elif v == 0:
                xin = (rms(partial)[None] * lw["attn_norm"].float()[:, None, :]).to(torch.bfloat16)
            else:
                xin = attn_res(stack[:nb] + [partial], self.attn_q[i], lw["attn_norm"]).to(torch.bfloat16)
            xq = xin[0]
            xk = xin[1] if xin.shape[0] == 3 else xq
            xv = xin[2] if xin.shape[0] == 3 else xq
            a = self.attention_c(i, c, xq, xk, xv, pos, dctx)
            attn_out = conv_c(c, "attn", a, lw["attn_sconv.weight"].squeeze(1), lw, "attn", decode)
            first = i % BLOCK == 0
            src = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [partial.float() + attn_out.float()]
            if p > 0:
                mlp_in = self.blend(i, ("mlp",), attn_res(src, self.mlp_q[i]), msgs[3:], lw["mlp_norm.weight"][None])[0]
            else:
                mlp_in = attn_res(src, self.mlp_q[i], lw["mlp_norm.weight"][None]).to(torch.bfloat16)[0]
            mconv = conv_c(c, "mlp", self.moe(i, mlp_in), lw["mlp_sconv.weight"].squeeze(1), lw, "mlp", decode)
            if first:
                nb += 1
                new = (attn_out.float() + mconv.float()).to(torch.bfloat16)
            else:
                new = (partial.float() + attn_out.float() + mconv.float()).to(torch.bfloat16)
            stack = stack[:nb] + [new]
        h = attn_res(stack[:nb] + [stack[nb]], self.out_q, self.norm[None])[0].to(torch.bfloat16)
        logits = (h @ self.unembed.T).float()[:, :VOCAB]
        return LOGIT_CAP * torch.tanh(logits / LOGIT_CAP), cache

    def _step(self, B):
        """One decode step on the static tensors: tok <- argmax(next), pos += 1."""
        logits, _ = self.forward(self.tok, pos=self.pos, B=B)
        self.tok.copy_(logits[-1].argmax().view(1))
        self.pos.add_(1)

    @torch.inference_mode()
    def generate(self, ids, n, use_graph=True):
        P = ids.shape[0]
        self.ensure_cap(P + n)
        logits, _ = self.forward(ids)
        first = logits[-1].argmax()
        out = [first.view(1)]
        if n == 1:
            return out[0]
        self.tok.copy_(first.view(1))
        self.pos.fill_(P)
        for s in range(n - 1):
            B = min(self.cache["cap"], -(-(P + s + 1) // BUCKET) * BUCKET)
            g = self.graphs.get(B) if use_graph else None
            if g is not None:
                g.replay()
            elif not use_graph:
                self._step(B)
            else:                                       # first use of bucket B: real step eagerly, then capture
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._step(B)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):               # recording only: does not advance the state
                    self._step(B)
                self.graphs[B] = g
            out.append(self.tok.clone())
        return torch.cat(out)
