# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) — facts for the engine

Released 2026-11-20, Apache 2.0. `model_type: inkling2_loop`. Checkpoint 19 GB (NVFP4 + MXFP8 experts, BF16 rest).
Reference code (read, never installed): vLLM PR #69028 `vllm/models/inkling2/` (8.8k lines), transformers PR #51050.

## Shape
- hidden 2304, 24 layers, vocab 201,024 (padded), **untied** embed + unembed (both BF16, 926 MB each)
- **Loop:** prelude layers 0–5 (×1) → core 6–17 (**×3 passes, shared weights**) → coda 18–23 (×1) = **48 layer visits/token**
- Attention: 20 local (sliding window 1024) + 4 global (layers 5, 11, 17, 23). 24 q heads, 8 KV heads, head_dim 128
  (projections are wider: wq 3072, gate 3072, wk/wv 1024, wo 3072→2304, plus a relative-logit branch wr 384, d_rel 16,
  rel_extent 1024), attention sink per head, q/rel L2-clip, log-scaling of logits by position (n_floor 8192, α 0.1)
- Short convs (kernel 4): on the attention stream, the MLP stream, and on k/v (global layers use a dynamic
  `weight_proj` 32×2304 conv)
- MoE: 128 experts, top-8, no shared expert, I = 1152, gate softcap 4, up softcap 25, route_scale 8, gate bias
  Layers 0–19 experts **NVFP4** (16-value blocks, FP8-E4M3 block scales, per-tensor amax outer scale);
  layers **20–23 experts MXFP8** (FP8 values, 32-value E8M0 block scales)
- **Block attention residuals (AttnRes):** no plain `x + f(x)`. Layers are grouped in blocks of 6; each sublayer's input is
  a softmax mix over the stack of completed block outputs + the running partial block, with learned fp32 pseudo-queries
  (`attn_res.attn_queries`, `mlp_queries`, `output_query`), values RMS-normalized
- **Loop messages:** from pass 2 on, each core sublayer input (q, k/v, attn, mlp streams) is blended with a "message"
  mixed from the previous pass's two blocks: rank-64 controller, `(1 − tanh(forget(h)))·dest + tanh(gain(h))·message`
- Each core layer keeps **its own KV cache and conv state per pass** (48 KV caches + 48 conv states)
- Final logit softcap 30. DSpark speculator in `mtp.safetensors` (3 layers, block 8, reads hidden states at layer visits 5, 23, 41, 48)

## Bytes read per decode token at batch 1 (from the safetensors header)
| Part | Per visit | Visits | Per token |
|---|---|---|---|
| Attention + norms + controllers (BF16, ~57 MB/layer) | ~55 MB | 48 | **~2.6 GB** |
| MoE NVFP4 experts (top-8 × 4.48 MB) | 35.8 MB | 44 | **~1.58 GB** |
| MoE MXFP8 experts (top-8 × 8.21 MB, layers 20–23) | 65.7 MB | 4 | 0.26 GB |
| Unembed (BF16 201,024 × 2304) | 926 MB | 1 | **0.93 GB** |
| **Total** | | | **≈ 5.4 GB/token** |
At ~1.3 TB/s (measured, lab 01) → **≈ 4.1 ms → ceiling ≈ 240 tok/s at batch 1** with the checkpoint's own precisions.
**Half the bytes are BF16 attention weights read 48 times.** Quantizing them (FP8 → ~1.3 GB, NVFP4 → ~0.7 GB) and the
unembed (FP8 → 0.46 GB) moves the ceiling to ~400–550 tok/s — if the quality gate holds.
