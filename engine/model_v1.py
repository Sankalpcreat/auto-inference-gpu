"""Engine v1: v0 math + incremental decoding.

Cross-token state lives only in (a) attention K/V and (b) the last 3 inputs of each short conv
(streams k, v, attn-out, mlp-out). AttnRes mixes, loop messages and MoE are per token. Each of the 48 layer
visits (layer, pass) owns its own cache, exactly like the reference's per-pass KV and conv-state caches.
`forward(ids, cache)` appends len(ids) tokens (prefill if the cache is empty, decode for 1 token).
"""
import torch
import torch.nn.functional as F

from .model_v0 import (ALPHA, BLOCK, CORE, DREL, EXT, GLOBAL, H, HD, HKV, L, LOGIT_CAP, N_FLOOR, PASSES, QK_CLIP,
                       REL_CLIP, STREAMS, VOCAB, WINDOW, InklingV0, attn_res, l2_clip, rms)
from .moe_triton import moe_t1

VISITS = [(i, 0) for i in range(CORE.start)] + [(i, p) for p in range(PASSES) for i in CORE] + \
         [(i, 0) for i in range(CORE.stop, L)]


def _with_hist(cache, key, x):
    """Prepend the stored last-3 inputs of stream `key` to x (T, C); update the store. Returns (T+3, C)."""
    hist = cache.get(key)
    if hist is None:
        hist = x.new_zeros(3, x.shape[1])
    full = torch.cat([hist, x])
    cache[key] = full[-3:] if x.shape[0] == 1 else full[-3:].clone()
    return full


def _f32(lw, key):
    """Cached fp32 copy of a small weight."""
    k = "_f32." + key
    if k not in lw:
        lw[k] = lw[key].float()
    return lw[k]


def conv_c(cache, key, x, w, lw=None, wkey=None):
    """Static causal conv with history: y = (sum_k w[c,k] x[t-3+k] + x[t]) / sqrt(5)."""
    T = x.shape[0]
    xp = _with_hist(cache, key, x).float()
    if T == 1 and lw is not None:
        k = "_convT." + wkey
        if k not in lw:
            lw[k] = w.float().T.contiguous()                                # (4, C), oldest tap first
        return (((xp * lw[k]).sum(0, keepdim=True) + x.float()) * 5 ** -0.5).to(x.dtype)
    acc = sum(w[:, k].float() * xp[k:k + T] for k in range(4))
    return ((acc + x.float()) * 5 ** -0.5).to(x.dtype)


def dyn_conv_c(cache, key, x, static_w, dyn):
    T, C = x.shape
    xp = _with_hist(cache, key, x).float()
    if T == 1:
        taps = static_w.float() + dyn[0].float().T.repeat_interleave(C // dyn.shape[1], 1)   # (4, C), newest first
        return (((taps * xp.flip(0)).sum(0, keepdim=True) + x.float()) * 9 ** -0.5).to(x.dtype)
    acc = torch.zeros(T, C, device=x.device)
    for l in range(4):
        taps = static_w[l].float()[None] + dyn[:, :, l].float().repeat_interleave(C // dyn.shape[1], 1)
        acc += taps * xp[3 - l:3 - l + T]
    return ((acc + x.float()) * 9 ** -0.5).to(x.dtype)


class InklingV1(InklingV0):
    def moe(self, i, x):
        if x.shape[0] == 1:
            return moe_t1(self.layers[i], x)
        return super().moe(i, x)

    def new_cache(self, cap):
        """cap = max total tokens; K/V live in preallocated fp32 (HKV, cap, HD) buffers per visit."""
        return {"len": 0, "cap": cap, "visits": [dict() for _ in VISITS]}

    def attention_c(self, i, c, xq, xk, xv, pos, start, cap):
        lw, T = self.layers[i], xq.shape[0]
        is_global = i in GLOBAL
        q = (xq @ lw["attn.wq_du.weight"].T).view(T, H, HD)
        r = (xq @ lw["attn.wr_du.weight"].T).view(T, H, DREL)
        gate = xq @ lw["attn.gate_attn.weight"].T
        sinks = (xq @ lw["attn.attn_sink_proj.weight"].T).float()
        k_raw, v_raw = xk @ lw["attn.wk_dv.weight"].T, xv @ lw["attn.wv_dv.weight"].T
        if is_global:
            k = dyn_conv_c(c, "k", k_raw, lw["k_static"], (xk @ lw["k_proj"].T).view(T, HKV, 4))
            v = dyn_conv_c(c, "v", v_raw, lw["v_static"], (xv @ lw["v_proj"].T).view(T, HKV, 4))
        else:
            k, v = conv_c(c, "k", k_raw, lw["k_conv"], lw, "k"), conv_c(c, "v", v_raw, lw["v_conv"], lw, "v")
        k = l2_clip(k.view(T, HKV, HD), QK_CLIP).to(torch.bfloat16)
        v = rms(v.view(T, HKV, HD).to(torch.bfloat16)).to(torch.bfloat16)
        if "K" not in c:
            c["K"] = torch.empty(HKV, cap, HD, device=xq.device)
            c["V"] = torch.empty(HKV, cap, HD, device=xq.device)
        S = start + T
        c["K"][:, start:S] = k.transpose(0, 1)
        c["V"][:, start:S] = v.transpose(0, 1)
        qc = l2_clip(q * lw["attn.q_l2_clip.scale"].float(), QK_CLIP).to(torch.bfloat16)
        rc = l2_clip(r * lw["attn.rel_l2_clip.scale"].float(), REL_CLIP).to(torch.bfloat16)
        if is_global and S > N_FLOOR:                      # tau == 1 exactly while every pos + 1 <= N_FLOOR
            tau = 1 + ALPHA * torch.log(torch.clamp((pos.float() + 1) / N_FLOOR, min=1.0))
            qc = (qc.float() * tau[:, None, None]).to(torch.bfloat16)
            rc = (rc.float() * tau[:, None, None]).to(torch.bfloat16)
        rel = torch.einsum("thd,de->the", rc.float(), _f32(lw, "attn.rel_logits_proj.proj"))
        if T == 1:
            lo = 0 if is_global else max(0, S - WINDOW)
            n = min(S - lo, EXT)
            Kw, Vw = c["K"][:, lo:S], c["V"][:, lo:S]                                     # (HKV, Sw, HD) fp32
            scores = (qc.float().view(HKV, H // HKV, HD) @ Kw.transpose(1, 2)) * HD ** -0.5  # (HKV, 3, Sw)
            scores[..., scores.shape[-1] - n:] += rel[0, :, :n].flip(-1).view(HKV, H // HKV, n)
            lse = scores.logsumexp(-1).view(1, H)
            out = (scores.softmax(-1) @ Vw).view(1, H, HD)
        else:
            K, V = c["K"][:, :S].transpose(0, 1), c["V"][:, :S].transpose(0, 1)
            kpos = torch.arange(S, device=xq.device)
            dist = pos[:, None] - kpos[None, :]
            bias = rel.gather(2, dist.clamp(0, EXT - 1)[:, None, :].expand(-1, H, -1))
            bias = bias.masked_fill(~((dist >= 0) & (dist < EXT))[:, None, :], 0.0)
            kk = K.repeat_interleave(H // HKV, 1)
            vv = V.repeat_interleave(H // HKV, 1)
            scores = torch.einsum("qhd,khd->hqk", qc.float(), kk) * HD ** -0.5 + bias.transpose(0, 1)
            mask = dist < 0
            if not is_global:
                mask = mask | (dist > WINDOW - 1)
            scores = scores.masked_fill(mask[None], float("-inf"))
            lse = scores.logsumexp(-1).T
            out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
        out = out * torch.sigmoid(gate.float().view(T, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
        return out.to(torch.bfloat16).reshape(T, H * HD) @ lw["attn.wo_ud.weight"].T

    @torch.inference_mode()
    def forward(self, ids, cache=None):
        """Append ids (T,) to `cache` (new if None); return (fp32 logits (T, VOCAB), cache)."""
        if cache is None:
            cache = self.new_cache(ids.shape[0])
        T = ids.shape[0]
        start = cache["len"]
        pos = torch.arange(start, start + T, device=self.dev)
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
            a = self.attention_c(i, c, xq, xk, xv, pos, start, cache["cap"])
            attn_out = conv_c(c, "attn", a, lw["attn_sconv.weight"].squeeze(1), lw, "attn")
            first = i % BLOCK == 0
            src = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [partial.float() + attn_out.float()]
            if p > 0:
                mlp_in = self.blend(i, ("mlp",), attn_res(src, self.mlp_q[i]), msgs[3:], lw["mlp_norm.weight"][None])[0]
            else:
                mlp_in = attn_res(src, self.mlp_q[i], lw["mlp_norm.weight"][None]).to(torch.bfloat16)[0]
            mconv = conv_c(c, "mlp", self.moe(i, mlp_in), lw["mlp_sconv.weight"].squeeze(1), lw, "mlp")
            if first:
                nb += 1
                new = (attn_out.float() + mconv.float()).to(torch.bfloat16)
            else:
                new = (partial.float() + attn_out.float() + mconv.float()).to(torch.bfloat16)
            stack = stack[:nb] + [new]
        cache["len"] = start + T
        h = attn_res(stack[:nb] + [stack[nb]], self.out_q, self.norm[None])[0].to(torch.bfloat16)
        logits = (h @ self.unembed.T).float()[:, :VOCAB]
        return LOGIT_CAP * torch.tanh(logits / LOGIT_CAP), cache

    @torch.inference_mode()
    def generate(self, ids, n):
        logits, cache = self.forward(ids, self.new_cache(ids.shape[0] + n))
        out = [logits[-1].argmax()]
        for _ in range(n - 1):
            logits, cache = self.forward(out[-1][None], cache)
            out.append(logits[-1].argmax())
        return torch.stack(out)
