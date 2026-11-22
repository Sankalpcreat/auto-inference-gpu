"""Eager DSpark drafter (port validated offline in study #22, scratch/dspark_accept.py).

Features per position: (4, HID) = partial block entering target visits 5/23/41 + final normed h.
Context K/V per drafter layer is kept per position and appended as tokens are accepted.
draft(anchor, p, K) drafts K tokens for positions p+1..p+K given context positions 0..p-1.
"""
import json
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .model_v0 import rms, VOCAB

NH, NKV, HD, HID = 24, 8, 128, 2304


class DSpark:
    def __init__(self, m, path="/tmp/inkling/mtp.safetensors", cfg="/tmp/inkling/config.json"):
        mc = json.load(open(cfg)).get("mtp_config") or {}
        self.eps = mc.get("rms_norm_eps", 1e-6)
        self.local = set(mc.get("local_layer_ids", [0, 2]))
        self.sw = mc.get("sliding_window_size", 1024)
        self.bs = mc.get("draft_block_size", mc.get("block_size", 8))
        R = mc.get("markov_rank", 256)
        self.R, self.cap = R, mc.get("final_logit_softcapping", 30.0)
        theta = mc.get("rope_theta", 1e6)
        nl = mc.get("num_hidden_layers", 3)
        raw = load_file(path, device="cuda")

        def g(suffix):
            hits = [k for k in raw if k == suffix or k.endswith("." + suffix)]
            assert len(hits) == 1, (suffix, hits)
            return raw[hits[0]]

        self.L = []
        for i in range(nl):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            self.L.append({"qkv": torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]),
                           "kvc": torch.cat([g(a + "k_proj_ctx.weight"), g(a + "v_proj_ctx.weight")]),
                           "o": g(a + "o_proj.weight"), "qn": g(a + "q_norm.weight"), "kn": g(a + "k_norm.weight"),
                           "knc": g(a + "k_norm_ctx.weight"), "in": g(p + "input_layernorm.weight"),
                           "post": g(p + "post_attention_layernorm.weight"), "dfl": g(p + "dflare_norm.weight"),
                           "gu": torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]),
                           "down": g(p + "mlp.down_proj.weight")})
        self.norm_w = g("norm.weight")
        mh = "markov_head."
        self.w1, self.rnn_w = g(mh + "markov_w1.weight"), g(mh + "rnn_proj.weight").float()
        self.rnn_b = g(mh + "rnn_proj.bias").float()
        hout_w, hout_b, self.mh_dfl = g(mh + "hidden_out.weight"), g(mh + "hidden_out.bias"), g(mh + "dflare_norm.weight")
        self.alpha = torch.softmax(g("fuse_weight").float(), -1)
        self.mask_embed = g("mask_embed")
        self.m = m
        lm = m.unembed
        self.fold_w = torch.empty(lm.shape[0], R, dtype=torch.bfloat16, device="cuda")
        self.fold_b = torch.empty(lm.shape[0], dtype=torch.bfloat16, device="cuda")
        for s in range(0, lm.shape[0], 8192):
            h = lm[s:s + 8192].double()
            self.fold_w[s:s + 8192] = (h @ hout_w.double()).bfloat16()
            self.fold_b[s:s + 8192] = (h @ hout_b.double()).bfloat16()
        self.inv_freq = theta ** (-torch.arange(0, HD, 2, device="cuda").float() / HD)
        self.reset(4096)

    def reset(self, cap):
        self.n = 0
        self.K = [torch.zeros(cap, NKV, HD, dtype=torch.bfloat16, device="cuda") for _ in self.L]
        self.V = [torch.zeros(cap, NKV, HD, dtype=torch.bfloat16, device="cuda") for _ in self.L]
        self.feat = torch.zeros(cap, 4, HID, dtype=torch.bfloat16, device="cuda")

    def rope(self, x, pos):
        f = pos.float()[:, None] * self.inv_freq[None]
        c, s = f.cos()[:, None], f.sin()[:, None]
        x1, x2 = x[..., :HD // 2].float(), x[..., HD // 2:].float()
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).bfloat16()

    def nrm(self, x, w):
        return (rms(x.float(), None, self.eps) * w.float()).bfloat16()

    def mix(self, feat, row, w):
        return self.nrm((feat.float() * self.alpha[row][None, :, None]).sum(1), w)

    @torch.inference_mode()
    def append(self, feat):
        """feat (T, 4, HID) for positions n..n+T-1."""
        T, n = feat.shape[0], self.n
        pos = torch.arange(n, n + T, device="cuda")
        self.feat[n:n + T] = feat
        for i, lw in enumerate(self.L):
            k, v = (self.mix(feat, i, lw["dfl"]) @ lw["kvc"].T).split(NKV * HD, -1)
            self.K[i][n:n + T] = self.rope(self.nrm(k.reshape(T, NKV, HD), lw["knc"]), pos)
            self.V[i][n:n + T] = v.reshape(T, NKV, HD)
        self.n += T

    @torch.inference_mode()
    def draft(self, anchor, K):
        """anchor: (1,) token at position p = self.n. Returns (K,) draft tokens for positions p+1..p+K."""
        m, BS, p = self.m, self.bs, self.n
        pos = torch.arange(p, p + BS, device="cuda")
        x = self.mask_embed[None].expand(BS, -1).clone()
        x[0] = (rms(m.embed[anchor[0]].float()) * m.embed_norm.float()).bfloat16()
        res = x
        x = self.nrm(res, self.L[0]["in"])
        for i, lw in enumerate(self.L):
            if i > 0:
                res = (res.float() + x.float()).bfloat16()
                x = self.nrm(res, lw["in"])
            q, k, v = (x @ lw["qkv"].T).split([NH * HD, NKV * HD, NKV * HD], -1)
            q = self.rope(self.nrm(q.reshape(BS, NH, HD), lw["qn"]), pos)
            k = self.rope(self.nrm(k.reshape(BS, NKV, HD), lw["kn"]), pos)
            st = max(0, p - self.sw) if i in self.local else 0
            Kc = torch.cat([self.K[i][st:p], k]).float().repeat_interleave(NH // NKV, 1)
            Vc = torch.cat([self.V[i][st:p], v.reshape(BS, NKV, HD)]).float().repeat_interleave(NH // NKV, 1)
            sc = torch.einsum("qhd,khd->hqk", q.float(), Kc) * HD ** -0.5
            o = torch.einsum("hqk,khd->qhd", sc.softmax(-1), Vc).bfloat16().reshape(BS, NH * HD)
            res = (res.float() + (o @ lw["o"].T).float()).bfloat16()
            x = self.nrm(res, lw["post"])
            gt, up = (x @ lw["gu"].T).float().chunk(2, -1)
            x = (F.silu(gt) * up).bfloat16() @ lw["down"].T
        res = (res.float() + x.float()).bfloat16()
        hs = self.nrm(res, self.norm_w)[1:1 + K]                      # mask rows 1..K
        base = hs @ m.unembed.T
        f_rnn = self.mix(self.feat[p - 1:p], -1, self.mh_dfl)
        R = self.R
        ctx = self.rnn_b + torch.cat([hs, f_rnn.expand_as(hs)], -1).float() @ self.rnn_w[:, 2 * R:].T
        state = torch.zeros(1, R, dtype=torch.bfloat16, device="cuda")
        prev, toks = anchor.view(1), []
        for j in range(K):
            inp = torch.cat([state, self.w1[prev]], -1).float()
            gt, cand, out = (ctx[j:j + 1] + inp @ self.rnn_w[:, :2 * R].T).bfloat16().chunk(3, -1)
            gate = torch.sigmoid(gt)
            state = gate * state + (1.0 - gate) * torch.tanh(cand)
            lg = (base[j:j + 1] + (self.fold_b + torch.tanh(out) @ self.fold_w.T)).bfloat16().float()
            prev = (self.cap * torch.tanh(lg / self.cap))[:, :VOCAB].argmax(-1)
            toks.append(prev)
        return torch.cat(toks)

    # ---------------- graph-safe variants (device positions, static shapes) ----------------
    @torch.inference_mode()
    def append_g(self, feat, p):
        """feat (T, 4, HID) for positions p..p+T-1 (p (1,) device). Rows past the accepted ones are rewritten later."""
        T = feat.shape[0]
        pos = p + torch.arange(T, device="cuda")
        self.feat.index_copy_(0, pos, feat)
        for i, lw in enumerate(self.L):
            k, v = (self.mix(feat, i, lw["dfl"]) @ lw["kvc"].T).split(NKV * HD, -1)
            self.K[i].index_copy_(0, pos, self.rope(self.nrm(k.reshape(T, NKV, HD), lw["knc"]), pos))
            self.V[i].index_copy_(0, pos, v.reshape(T, NKV, HD).contiguous())

    @torch.inference_mode()
    def draft_g(self, anchor, p, K, B):
        """anchor (1,) at position p (1,) device; context keys from a static bucket of B positions (masked)."""
        from .fp8_dense import fp8_rows
        m, BS = self.m, self.bs
        pos = p + torch.arange(BS, device="cuda")
        kpos = torch.arange(B, device="cuda")
        ninf, zero = torch.full((), float("-inf"), device="cuda"), torch.zeros((), device="cuda")
        add_g = torch.cat([torch.where(kpos < p, zero, ninf), torch.zeros(BS, device="cuda")])
        add_l = torch.cat([torch.where((kpos < p) & (kpos >= p - self.sw), zero, ninf), torch.zeros(BS, device="cuda")])
        e0 = (rms(m.embed.index_select(0, anchor).float()) * m.embed_norm.float()).bfloat16()
        x = torch.cat([e0, self.mask_embed[None].expand(BS - 1, -1)])
        res = x
        x = self.nrm(res, self.L[0]["in"])
        for i, lw in enumerate(self.L):
            if i > 0:
                res = (res.float() + x.float()).bfloat16()
                x = self.nrm(res, lw["in"])
            q, k, v = (x @ lw["qkv"].T).split([NH * HD, NKV * HD, NKV * HD], -1)
            q = self.rope(self.nrm(q.reshape(BS, NH, HD), lw["qn"]), pos)
            k = self.rope(self.nrm(k.reshape(BS, NKV, HD), lw["kn"]), pos)
            Kc = torch.cat([self.K[i][:B], k]).float().repeat_interleave(NH // NKV, 1)
            Vc = torch.cat([self.V[i][:B], v.reshape(BS, NKV, HD)]).float().repeat_interleave(NH // NKV, 1)
            sc = torch.einsum("qhd,khd->hqk", q.float(), Kc) * HD ** -0.5
            sc = sc + (add_l if i in self.local else add_g)
            o = torch.einsum("hqk,khd->qhd", sc.softmax(-1), Vc).bfloat16().reshape(BS, NH * HD)
            res = (res.float() + (o @ lw["o"].T).float()).bfloat16()
            x = self.nrm(res, lw["post"])
            gt, up = (x @ lw["gu"].T).float().chunk(2, -1)
            x = (F.silu(gt) * up).bfloat16() @ lw["down"].T
        res = (res.float() + x.float()).bfloat16()
        hs = self.nrm(res, self.norm_w)[1:1 + K].contiguous()
        base = fp8_rows(hs, *m.unembed8)
        f_rnn = self.mix(self.feat.index_select(0, p - 1), -1, self.mh_dfl)
        R = self.R
        ctx = self.rnn_b + torch.cat([hs, f_rnn.expand_as(hs)], -1).float() @ self.rnn_w[:, 2 * R:].T
        state = torch.zeros(1, R, dtype=torch.bfloat16, device="cuda")
        prev, toks = anchor.view(1), []
        for j in range(K):
            inp = torch.cat([state, self.w1[prev]], -1).float()
            gt, cand, out = (ctx[j:j + 1] + inp @ self.rnn_w[:, :2 * R].T).bfloat16().chunk(3, -1)
            gate = torch.sigmoid(gt)
            state = gate * state + (1.0 - gate) * torch.tanh(cand)
            lg = (base[j:j + 1] + (self.fold_b + torch.tanh(out) @ self.fold_w.T)).bfloat16().float()
            prev = (self.cap * torch.tanh(lg[:, :VOCAB] / self.cap)).argmax(-1)
            toks.append(prev)
        return torch.cat(toks)
