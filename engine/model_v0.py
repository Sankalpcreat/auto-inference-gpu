"""Engine v0: Inkling 2 Mini, plain PyTorch, correctness first (no KV cache: full-sequence forward every step).

Math transcribed from the reference (vLLM PR #69028 `vllm/models/inkling2/` + its tests), read-only:
- embed -> embed_norm; block attention residuals (AttnRes) instead of `x + f(x)`, blocks of 6 layers
- loop: prelude 0-5, core 6-17 x 3 passes (each pass restarts from the prelude state; passes >= 1 blend gated
  messages from the previous pass's two core blocks), coda 18-23
- attention: q/r/gate/sink/k/v projections, causal short conv on k/v (dynamic taps on global layers),
  L2-clipped q (x q_scale) and k, RMS-normalized v, relative bias from r, sliding window 1024 (local) / full (global),
  log-scaling tau on global q/r, output * sigmoid(gate) * sigmoid(lse - sink), wo
- short conv on attention and MoE outputs: y = (sum_k w[c,k] x[t-3+k] + x[t]) / sqrt(5)
- MoE: ids = topk(sigmoid(l) + bias, 8); w = softmax(logsigmoid(l[ids])) * 8;
  act = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25) with (g, u) interleaved rows of w13
- final: AttnRes mix with output_query + norm -> unembed -> softcap 30
"""
import math
import torch
import torch.nn.functional as F

from .weights import dequant_mxfp8, dequant_nvfp4, open_checkpoint

D = 2304
L = 24
CORE = range(6, 18)
PASSES = 3
BLOCK = 6
GLOBAL = {5, 11, 17, 23}
H, HKV, HD, DREL, EXT, WINDOW = 24, 8, 128, 16, 1024, 1024
QK_CLIP, REL_CLIP, EPS = 22.627416997969522, 8.0, 1e-6
TOPK, ROUTE_SCALE, GATE_CAP, UP_CAP, LOGIT_CAP = 8, 8.0, 4.0, 25.0, 30.0
N_FLOOR, ALPHA = 8192, 0.1
VOCAB = 200058
STREAMS = ("attn", "k", "v", "mlp")


def rms(x, w=None, eps=EPS):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def attn_res(sources, queries, norm_w=None, norm_eps=EPS):
    """sources list of (T, D); queries (Q, D) fp32 pre-scaled by D**-0.5 -> (Q, T, D) fp32 (normed if norm_w (Q, D))."""
    x = torch.stack([s.float() for s in sources])                      # (N, T, D)
    inv = torch.rsqrt(x.pow(2).mean(-1) + 1e-6)                        # (N, T)
    logits = torch.einsum("qd,ntd->qnt", queries, x) * inv             # (Q, N, T)
    p = logits.softmax(1) * inv                                         # normalize_values
    mix = torch.einsum("qnt,ntd->qtd", p, x)
    if norm_w is None:
        return mix
    return rms(mix, None, norm_eps) * norm_w.float()[:, None, :]


def causal_conv(x, w):
    """Static short conv: x (T, C), w (C, 4) oldest tap first -> (T, C) in x.dtype."""
    T = x.shape[0]
    xp = F.pad(x.float(), (0, 0, 3, 0))
    acc = sum(w[:, k].float() * xp[k:k + T] for k in range(4))
    return ((acc + x.float()) * 5 ** -0.5).to(x.dtype)


def dyn_conv(x, static_w, dyn):
    """Dynamic short conv: x (T, C=8*128), static_w (4, C), dyn (T, 8, 4) newest tap first -> (T, C)."""
    T, C = x.shape
    xp = F.pad(x.float(), (0, 0, 3, 0))
    acc = torch.zeros(T, C, device=x.device)
    for l in range(4):
        taps = static_w[l].float()[None] + dyn[:, :, l].float().repeat_interleave(C // dyn.shape[1], 1)
        acc += taps * xp[3 - l:3 - l + T]
    return ((acc + x.float()) * 9 ** -0.5).to(x.dtype)


def l2_clip(x, thr, eps=1e-6):
    n = x.float().norm(dim=-1, keepdim=True)
    return x.float() * (thr / (n + eps)).clamp(max=1.0)


class InklingV0:
    def __init__(self, path="/tmp/inkling/model-00001-of-00001.safetensors", device="cuda"):
        self.dev = device
        f = open_checkpoint(path, device)
        g = lambda k: f.get_tensor("model.llm." + k)
        self.embed, self.unembed = g("embed.weight"), g("unembed.weight")
        self.embed_norm, self.norm = g("embed_norm.weight"), g("norm.weight")
        s = D ** -0.5
        self.attn_q = [None] + [g(f"attn_res.attn_queries.{i}").float().reshape(-1, D) * s for i in range(L - 1)]
        self.mlp_q = [g(f"attn_res.mlp_queries.{i}").float().reshape(1, D) * s for i in range(L)]
        self.out_q = g("attn_res.output_query").float().reshape(1, D) * s
        q = g("middle_cycle_cross_pass_queries").float().view(-1, 2, D)
        kv = g("middle_cycle_cross_pass_kv_queries").float().view(-1, 2, D)
        self.msg_q = torch.stack((q[:, 0], kv[:, 0], kv[:, 1], q[:, 1]), 1) * s      # (12, 4, D): attn, k, v, mlp
        self.layers = []
        for i in range(L):
            p = f"layers.{i}."
            lw = {k: g(p + k) for k in (
                "attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight", "attn.attn_sink_proj.weight",
                "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wo_ud.weight", "attn.rel_logits_proj.proj",
                "attn.q_l2_clip.scale", "attn.rel_l2_clip.scale", "attn_sconv.weight", "mlp_sconv.weight",
                "mlp_norm.weight", "mlp.gate.weight", "mlp.gate.bias")}
            if i in GLOBAL:
                lw["attn_norm"] = torch.stack([g(p + f"{n}_input_norm.weight") for n in "qkv"])
                for n in "kv":
                    lw[f"{n}_static"] = g(p + f"attn.{n}_sconv.static_w")
                    lw[f"{n}_proj"] = g(p + f"attn.{n}_sconv.weight_proj")
            else:
                lw["attn_norm"] = g(p + "attn_norm.weight")[None]
                for n in "kv":
                    lw[f"{n}_conv"] = g(p + f"attn.{n}_sconv.weight").squeeze(1)
            if i in CORE:
                for st in STREAMS:
                    for w in ("down", "gain", "forget"):
                        lw[f"ctl.{st}.{w}"] = g(p + f"controllers.{st}.{w}_weight")
            e = p + "mlp.experts."
            if i < 20:
                ws = [dequant_nvfp4(f.get_tensor("model.llm." + e + n + "_weight"),
                                    f.get_tensor("model.llm." + e + n + "_weight_scale"),
                                    f.get_tensor("model.llm." + e + n + "_weight_amax"),
                                    f.get_tensor("model.llm." + e + n + "_weight_outer_scale_denominator"))
                      for n in ("w13", "w2")]
            else:
                ws = [dequant_mxfp8(f.get_tensor("model.llm." + e + n + "_weight"),
                                    f.get_tensor("model.llm." + e + n + "_weight_scale")) for n in ("w13", "w2")]
            lw["w13"], lw["w2"] = ws                                     # (128, 2304, 2304), (128, 2304, 1152) bf16
            self.layers.append(lw)
        torch.cuda.synchronize()

    # ---------------- sublayers ----------------
    def attention(self, i, xq, xk, xv, pos):
        lw, T = self.layers[i], xq.shape[0]
        is_global = i in GLOBAL
        q = (xq @ lw["attn.wq_du.weight"].T).view(T, H, HD)
        r = (xq @ lw["attn.wr_du.weight"].T).view(T, H, DREL)
        gate = xq @ lw["attn.gate_attn.weight"].T
        sinks = (xq @ lw["attn.attn_sink_proj.weight"].T).float()
        k_raw, v_raw = xk @ lw["attn.wk_dv.weight"].T, xv @ lw["attn.wv_dv.weight"].T
        if is_global:
            dk = (xk @ lw["k_proj"].T).view(T, HKV, 4)
            dv = (xv @ lw["v_proj"].T).view(T, HKV, 4)
            k = dyn_conv(k_raw, lw["k_static"], dk)
            v = dyn_conv(v_raw, lw["v_static"], dv)
        else:
            k, v = causal_conv(k_raw, lw["k_conv"]), causal_conv(v_raw, lw["v_conv"])
        k = l2_clip(k.view(T, HKV, HD), QK_CLIP).to(torch.bfloat16)
        v = rms(v.view(T, HKV, HD).to(torch.bfloat16)).to(torch.bfloat16)
        qc = l2_clip(q * lw["attn.q_l2_clip.scale"].float(), QK_CLIP).to(torch.bfloat16)
        rc = l2_clip(r * lw["attn.rel_l2_clip.scale"].float(), REL_CLIP).to(torch.bfloat16)
        if is_global:
            tau = 1 + ALPHA * torch.log(torch.clamp((pos.float() + 1) / N_FLOOR, min=1.0))
            qc = (qc.float() * tau[:, None, None]).to(torch.bfloat16)
            rc = (rc.float() * tau[:, None, None]).to(torch.bfloat16)
        rel = torch.einsum("thd,de->the", rc.float(), lw["attn.rel_logits_proj.proj"].float())   # (T, H, EXT)
        dist = pos[:, None] - pos[None, :]                                                          # (T, T)
        bias = rel.gather(2, dist.clamp(0, EXT - 1)[:, None, :].expand(-1, H, -1))                 # (T, H, T)
        bias = bias.masked_fill(~((dist >= 0) & (dist < EXT))[:, None, :], 0.0)
        scale = HD ** -0.5
        kk = k.float().repeat_interleave(H // HKV, 1)
        vv = v.float().repeat_interleave(H // HKV, 1)
        qk = torch.einsum("qhd,khd->hqk", qc.float(), kk)
        scores = qk * scale + bias.transpose(0, 1)
        mask = dist < 0
        if not is_global:
            mask = mask | (dist > WINDOW - 1)
        scores = scores.masked_fill(mask[None], float("-inf"))
        lse = scores.logsumexp(-1).T                                                                # (T, H)
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)                                  # (T, H, HD)
        out = out * torch.sigmoid(gate.float().view(T, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
        return out.to(torch.bfloat16).reshape(T, H * HD) @ lw["attn.wo_ud.weight"].T

    def moe(self, i, x):
        lw = self.layers[i]
        logits = (x.float() @ lw["mlp.gate.weight"].float().T)
        _, ids = torch.topk(logits.sigmoid() + lw["mlp.gate.bias"].float(), TOPK, sorted=True)
        w = F.logsigmoid(logits.gather(1, ids)).softmax(-1) * ROUTE_SCALE
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in ids.unique().tolist():
            tok, slot = (ids == e).nonzero(as_tuple=True)
            gu = (x[tok] @ lw["w13"][e].T).float()
            gt, up = gu[:, 0::2], gu[:, 1::2]
            act = GATE_CAP * torch.tanh(gt / GATE_CAP) * torch.sigmoid(gt) * UP_CAP * torch.tanh(up / UP_CAP)
            act = (act * w[tok, slot][:, None]).to(torch.bfloat16)
            out.index_add_(0, tok, (act @ lw["w2"][e].T).float())
        return out.to(torch.bfloat16)

    def blend(self, i, stream_ids, dest, msg, norm_w):
        """dest/msg fp32 (S, T, D) -> bf16 (S, T, D) via the rank-64 controllers + RMSNorm."""
        lw, outs = self.layers[i], []
        for s, st in enumerate(stream_ids):
            pair = torch.stack((dest[s], msg[s]), 1)                                     # (T, 2, D)
            pair = (pair * torch.rsqrt(pair.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).flatten(1)
            h = (pair.float() @ lw[f"ctl.{st}.down"].float().T).to(torch.bfloat16)
            gain = h.float() @ lw[f"ctl.{st}.gain"].float().T
            forget = h.float() @ lw[f"ctl.{st}.forget"].float().T
            y = (1 - torch.tanh(forget)) * dest[s] + torch.tanh(gain) * msg[s]
            outs.append((rms(y) * norm_w[s].float()).to(torch.bfloat16))
        return torch.stack(outs)

    # ---------------- full forward ----------------
    @torch.inference_mode()
    def forward(self, ids):
        """ids (T,) -> fp32 logits (T, VOCAB) for every position."""
        T = ids.shape[0]
        pos = torch.arange(T, device=self.dev)
        x0 = (rms(self.embed[ids]) * self.embed_norm.float()).to(torch.bfloat16)
        stack = [x0]                     # stack[:nb] completed blocks, stack[nb] running partial
        nb = 0
        visits = [(i, 0) for i in range(CORE.start)] + [(i, p) for p in range(PASSES) for i in CORE] + \
                 [(i, 0) for i in range(CORE.stop, L)]
        prelude = None
        bank = None
        dests = None                     # fp32 mixes for the first core layer on passes >= 1
        for v, (i, p) in enumerate(visits):
            lw = self.layers[i]
            if i == CORE.start and p == 0:
                prelude = (nb, stack[nb].clone(), [s.clone() for s in stack[:nb]])
            elif i == CORE.start:
                pb, ppart, pblocks = prelude
                bank = [stack[pb + 1], stack[pb + 2]] if len(stack) > pb + 2 else None
                stack = pblocks + [ppart]
                nb = pb
                dests = None
            partial = stack[nb]
            # --- attention input ---
            if p > 0:
                base = attn_res(stack[:nb] + [partial], self.attn_q[i]) if (dests is None) else dests
                msgs = attn_res(bank, self.msg_q[i - CORE.start])                         # (4, T, D) fp32
                if base.shape[0] == 1:
                    base = base.expand(3, -1, -1)
                nw = lw["attn_norm"] if lw["attn_norm"].shape[0] == 3 else lw["attn_norm"].expand(3, -1)
                xin = self.blend(i, STREAMS[:3], base, msgs[:3], nw)
                xq, xk, xv = xin[0], xin[1], xin[2]
            else:
                if v == 0:
                    xin = (rms(partial)[None] * lw["attn_norm"].float()[:, None, :]).to(torch.bfloat16)
                else:
                    xin = attn_res(stack[:nb] + [partial], self.attn_q[i], lw["attn_norm"]).to(torch.bfloat16)
                xq = xin[0]
                xk = xin[1] if xin.shape[0] == 3 else xq
                xv = xin[2] if xin.shape[0] == 3 else xq
            a = self.attention(i, xq, xk, xv, pos)
            attn_out = causal_conv(a, lw["attn_sconv.weight"].squeeze(1))
            first = i % BLOCK == 0
            src = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [(partial.float() + attn_out.float())]
            if p > 0:
                mix = attn_res(src, self.mlp_q[i])
                mlp_in = self.blend(i, ("mlp",), mix, msgs[3:], lw["mlp_norm.weight"][None])[0]
            else:
                mlp_in = attn_res(src, self.mlp_q[i], lw["mlp_norm.weight"][None]).to(torch.bfloat16)[0]
            m = self.moe(i, mlp_in)
            mconv = causal_conv(m, lw["mlp_sconv.weight"].squeeze(1))
            if first:
                nb += 1
                new = (attn_out.float() + mconv.float()).to(torch.bfloat16)
            else:
                new = (partial.float() + attn_out.float() + mconv.float()).to(torch.bfloat16)
            stack = stack[:nb] + [new]
        h = attn_res(stack[:nb] + [stack[nb]], self.out_q, self.norm[None])[0].to(torch.bfloat16)
        logits = (h @ self.unembed.T).float()[:, :VOCAB]
        return LOGIT_CAP * torch.tanh(logits / LOGIT_CAP)

    @torch.inference_mode()
    def generate(self, ids, n):
        ids = ids.clone()
        for _ in range(n):
            nxt = self.forward(ids)[-1].argmax()
            ids = torch.cat([ids, nxt[None]])
        return ids
