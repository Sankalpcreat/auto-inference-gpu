"""Stage A (eager): T-token verify step on the persistent v1 cache, with rollback.

verify(m, toks (T,), pos0) runs toks at positions pos0..pos0+T-1 against the cache and returns fp32 logits (T, VOCAB).
It records what it overwrote; commit(m, n) keeps the first n tokens and rolls the rest back:
- conv histories: hist <- full[n:n+3] of cat(old_hist, chunk inputs)
- local K/V rings: slots of rejected positions get their old contents back
- global K/V: nothing (positions > pos are masked by every later step and overwritten)
"""
import torch

from . import model_v1 as M1
from .model_v0 import (ALPHA, BLOCK, CORE, DREL, EXT, GLOBAL, H, HD, HKV, LOGIT_CAP, N_FLOOR, QK_CLIP, REL_CLIP,
                       STREAMS, VOCAB, WINDOW, attn_res, l2_clip, rms)

_LOG = []          # rollback records of the current verify


def _chunk_hist(cache, key, x, decode):
    hist = cache[key]
    full = torch.cat([hist, x])
    _LOG.append(("hist", hist, full))
    hist.copy_(full[-3:])
    return full


def _attention(m, i, c, xq, xk, xv, qp):
    lw, T = m.layers[i], xq.shape[0]
    is_global = i in GLOBAL
    q = (xq @ lw["attn.wq_du.weight"].T).view(T, H, HD)
    r = (xq @ lw["attn.wr_du.weight"].T).view(T, H, DREL)
    gate = xq @ lw["attn.gate_attn.weight"].T
    sinks = (xq @ lw["attn.attn_sink_proj.weight"].T).float()
    k_raw, v_raw = xk @ lw["attn.wk_dv.weight"].T, xv @ lw["attn.wv_dv.weight"].T
    if is_global:
        k = M1.dyn_conv_c(c, "k", k_raw, lw["k_static"], (xk @ lw["k_proj"].T).view(T, HKV, 4), True)
        v = M1.dyn_conv_c(c, "v", v_raw, lw["v_static"], (xv @ lw["v_proj"].T).view(T, HKV, 4), True)
    else:
        k = M1.conv_c(c, "k", k_raw, lw["k_conv"], None, None, True)
        v = M1.conv_c(c, "v", v_raw, lw["v_conv"], None, None, True)
    k = l2_clip(k.view(T, HKV, HD), QK_CLIP).to(torch.bfloat16)
    v = rms(v.view(T, HKV, HD).to(torch.bfloat16)).to(torch.bfloat16)
    qc = l2_clip(q * lw["attn.q_l2_clip.scale"].float(), QK_CLIP).to(torch.bfloat16)
    rc = l2_clip(r * lw["attn.rel_l2_clip.scale"].float(), REL_CLIP).to(torch.bfloat16)
    if is_global:
        tau = 1 + ALPHA * torch.log(torch.clamp((qp.float() + 1) / N_FLOOR, min=1.0))
        qc = (qc.float() * tau[:, None, None]).to(torch.bfloat16)
        rc = (rc.float() * tau[:, None, None]).to(torch.bfloat16)
    rel = torch.einsum("thd,de->the", rc.float(), M1._f32(lw, "attn.rel_logits_proj.proj"))
    kt, vt = k.transpose(0, 1).float(), v.transpose(0, 1).float()                         # (HKV, T, HD)
    pos0 = int(qp[0])
    if is_global:
        c["K"][:, pos0:pos0 + T] = kt
        c["V"][:, pos0:pos0 + T] = vt
        Kw, Vw = c["K"][:, :pos0 + T], c["V"][:, :pos0 + T]
        kpos = torch.arange(pos0 + T, device=xq.device)
    else:
        old = torch.arange(max(0, pos0 - WINDOW + 1), pos0, device=xq.device)
        Kw = torch.cat([c["K"][:, old % WINDOW], kt], 1)
        Vw = torch.cat([c["V"][:, old % WINDOW], vt], 1)
        kpos = torch.cat([old, qp])
        slots = qp % WINDOW
        _LOG.append(("ring", c, slots, c["K"][:, slots].clone(), c["V"][:, slots].clone()))
        c["K"][:, slots] = kt
        c["V"][:, slots] = vt
    dist = qp[:, None] - kpos[None, :]                                                     # (T, S)
    bias = rel.gather(2, dist.clamp(0, EXT - 1)[:, None, :].expand(-1, H, -1))
    bias = bias.masked_fill(~((dist >= 0) & (dist < EXT))[:, None, :], 0.0)              # (T, H, S)
    mask = dist < 0
    if not is_global:
        mask = mask | (dist > WINDOW - 1)
    G = H // HKV
    scores = torch.einsum("tgjd,gsd->gjts", qc.float().view(T, HKV, G, HD), Kw) * HD ** -0.5
    scores = scores + bias.view(T, HKV, G, -1).permute(1, 2, 0, 3)
    scores = scores.masked_fill(mask[None, None], float("-inf"))
    lse = scores.logsumexp(-1).permute(2, 0, 1).reshape(T, H)
    out = torch.einsum("gjts,gsd->tgjd", scores.softmax(-1), Vw).reshape(T, H, HD)
    out = out * torch.sigmoid(gate.float().view(T, H, HD)) * torch.sigmoid(lse - sinks)[..., None]
    return out.to(torch.bfloat16).reshape(T, H * HD) @ lw["attn.wo_ud.weight"].T


@torch.inference_mode()
def verify(m, ids, pos0):
    _LOG.clear()
    saved, M1._with_hist = M1._with_hist, _chunk_hist
    try:
        T = ids.shape[0]
        qp = torch.arange(pos0, pos0 + T, device=m.dev)
        cache = m.cache
        x0 = (rms(m.embed[ids]) * m.embed_norm.float()).to(torch.bfloat16)
        stack, nb, prelude, bank = [x0], 0, None, None
        for v, (i, p) in enumerate(M1.VISITS):
            lw, c = m.layers[i], cache["visits"][v]
            if i == CORE.start and p == 0:
                prelude = (nb, stack[nb].clone(), [s.clone() for s in stack[:nb]])
            elif i == CORE.start:
                pb, ppart, pblocks = prelude
                bank = [stack[pb + 1], stack[pb + 2]]
                stack, nb = pblocks + [ppart], pb
            partial = stack[nb]
            if p > 0:
                base = attn_res(stack[:nb] + [partial], m.attn_q[i])
                msgs = attn_res(bank, m.msg_q[i - CORE.start])
                if base.shape[0] == 1:
                    base = base.expand(3, -1, -1)
                nw = lw["attn_norm"] if lw["attn_norm"].shape[0] == 3 else lw["attn_norm"].expand(3, -1)
                xin = m.blend(i, STREAMS[:3], base, msgs[:3], nw)
            elif v == 0:
                xin = (rms(partial)[None] * lw["attn_norm"].float()[:, None, :]).to(torch.bfloat16)
            else:
                xin = attn_res(stack[:nb] + [partial], m.attn_q[i], lw["attn_norm"]).to(torch.bfloat16)
            xq = xin[0]
            xk = xin[1] if xin.shape[0] == 3 else xq
            xv = xin[2] if xin.shape[0] == 3 else xq
            a = _attention(m, i, c, xq, xk, xv, qp)
            attn_out = M1.conv_c(c, "attn", a, lw["attn_sconv.weight"].squeeze(1), None, None, True)
            first = i % BLOCK == 0
            src = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [partial.float() + attn_out.float()]
            if p > 0:
                mlp_in = m.blend(i, ("mlp",), attn_res(src, m.mlp_q[i]), msgs[3:], lw["mlp_norm.weight"][None])[0]
            else:
                mlp_in = attn_res(src, m.mlp_q[i], lw["mlp_norm.weight"][None]).to(torch.bfloat16)[0]
            mo = m.moe(i, mlp_in) if T > 1 else M1.InklingV0.moe(m, i, mlp_in)
            mconv = M1.conv_c(c, "mlp", mo, lw["mlp_sconv.weight"].squeeze(1), None, None, True)
            if first:
                nb += 1
                new = (attn_out.float() + mconv.float()).to(torch.bfloat16)
            else:
                new = (partial.float() + attn_out.float() + mconv.float()).to(torch.bfloat16)
            stack = stack[:nb] + [new]
        h = attn_res(stack[:nb] + [stack[nb]], m.out_q, m.norm[None])[0].to(torch.bfloat16)
        logits = (h @ m.unembed.T).float()[:, :VOCAB]
        return LOGIT_CAP * torch.tanh(logits / LOGIT_CAP)
    finally:
        M1._with_hist = saved


def commit(m, n, T):
    """Keep the first n of the T verified tokens."""
    for rec in _LOG:
        if rec[0] == "hist":
            rec[1].copy_(rec[2][n:n + 3])
        elif n < T:
            _, c, slots, ko, vo = rec
            c["K"][:, slots[n:]] = ko[:, n:]
            c["V"][:, slots[n:]] = vo[:, n:]
    _LOG.clear()


def ngram_draft(seq, K, n=1):
    key = seq[-n:]
    for s in range(len(seq) - n - 1, -1, -1):
        if seq[s:s + n] == key:
            return seq[s + n:s + n + K]
    return []


@torch.inference_mode()
def generate_spec(m, ids, n_out, K=4, drafter=ngram_draft):
    """Eager speculative greedy decode with exact verify. Returns (tokens, steps)."""
    P = ids.shape[0]
    m.ensure_cap(P + n_out + K + 1)
    logits, _ = m.forward(ids)
    seq = ids.tolist() + [int(logits[-1].argmax())]
    pos, steps = P, 0
    while len(seq) - P < n_out:
        d = drafter(seq, K)[:max(0, n_out - (len(seq) - P) - 1)]
        chunk = torch.tensor([seq[-1]] + d, device=m.dev)
        targ = verify(m, chunk, pos).argmax(-1).tolist()
        a = 0
        while a < len(d) and d[a] == targ[a]:
            a += 1
        commit(m, a + 1, len(chunk))
        seq += targ[:a + 1]
        pos += a + 1
        steps += 1
    return seq[P:P + n_out], steps
