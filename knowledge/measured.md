# Measured facts (update as agents learn; dead ends at the bottom)

## NEW TARGET (2026-11-24): ninfer on THIS card (RTX PRO 6000 Server Edition), github.com/iamwavecut/ninfer-all docs/inkling2.md
| ninfer (C++/CUDA, ~4-bit GGUF UD-IQ4_XS incl. attention; NVFP4 experts) | plain 155.8–158 tok/s · DSpark 207.0 (1.33×, 2.08 tokens/round, text equal to plain) |
| llama.cpp, same GGUF, same card | 84.4 tok/s |
| Ours | plain 96.0 · DSpark 98.1 (verify step ~2.2× a T=1 step) |
Their proven techniques: (1) vector kernels that multiply up to 8 token rows reading each weight once (cheap DSpark verify);
(2) programmatic dependent launch in the decode graph (+8% on an RTX 5090: 199.4 → 216.0); (3) hand-fused kernels with an
explicit rounding order so DSpark text equals plain decode; (4) ~4-bit dense weights (fewer bytes than our FP8).

## THE TARGET TO BEAT (same GPU family, measured by hyudryu/Inkling-2-Mini-2xRTXPro6000, patched vLLM, TP1, 2026-11-21)
| vLLM TP1 | 1 user |
|---|---|
| decode step (verifies 1 + K=4 DSpark drafts) | **10.0 ms** (baseline vLLM 11.5 ms at K=7) |
| tok/s prose / coding / counting | **196 / 295 / 407** |
| their bytes/step | 54 MB dense per layer pass x 48 (BF16, = 2.6 GB) + ~4.5 MB per distinct expert; ~9 ms of 11.5 ms is bandwidth |
| their kernels | big kernels at 1.3–1.6 TB/s cold; tile tuning and L2 prefetch gave nothing; ~850 small kernels per step |
| DSpark acceptance by draft position (K=7) | 67 / 45 / 31 / 23 / 18 / 14 / 11 % |
vLLM keeps outputs EXACT (BF16 dense, native NVFP4/MXFP8 experts). Our harness allows lossy changes if the gates pass,
so our edges are: FP8/NVFP4 dense projections (2.6 → 1.3/0.7 GB), FP8 unembed, adaptive draft length, a batch-1-only engine.

## Progress (2026-11-21, 6 experiments): 6.1 → 64.52 tok/s (15.5 ms/token), eval 12/12, nll_ratio 1.0
- exp: sync-free Triton MoE GEMV 9.24; static KV + windowed attention 11.83; whole-step CUDA graph 31.74; torch.compile T=1 regions 64.52
- Profile at 31.7 tok/s (agent 5): 12,056 kernels/step, gaps only 1.6 ms; ~65% of time in 11.5k tiny kernels (<5 µs);
  bandwidth part ~10 ms = experts 5.2 ms (BF16-dequantized, ~6 GB/token, near 1.3 TB/s) + attention projections 4.3 ms + unembed 0.7 ms
- After torch.compile the tiny-kernel share dropped a lot; re-profile before choosing.

## Baseline (2026-11-21): engine v1, 6.1 tok/s, noise ±0.6%, eval 12/12, peak 56 GB
- Decode step ≈ 164 ms at batch 1. Ceiling with the BF16-dequantized experts v1 uses: experts 8 × 10.6 MB × 48 visits
  ≈ 6.1 GB + attention 2.6 GB + unembed 0.93 GB ≈ **9.6 GB/token → ~135 tok/s at 1.3 TB/s**. Keeping experts packed
  (NVFP4/MXFP8, 1.8 GB/token) → ~240 tok/s; + FP8 attention/unembed → ~400–550 tok/s.
- So at 6 tok/s we are at ~4% of even the BF16 ceiling: the step is **launch/host-overhead bound**, not memory-bound.
- Known overheads in v1 (by reading the code, not yet profiled):
  - `moe()`: `ids.unique().tolist()` + `nonzero()` per visit → **48 host-device syncs per token** + a Python loop of 8 tiny GEMMs.
  - attention: fp32 einsum over the whole cache, `repeat_interleave` of K/V, `torch.cat` growing K/V every step (realloc).
  - `blend()` runs 4 controllers in a Python loop; `attn_res` stacks sources every call.
  - Eager mode: likely > 2,000 kernel launches per token × ~22 µs (lab 01) ≈ 45+ ms of pure launch cost → CUDA graphs need
    static shapes (pre-allocated KV/conv buffers, no `.tolist()`, no data-dependent shapes).
- Prefill TTFT: 1.08 s for 1,261 tokens (long), ~0.4–0.5 s for short prompts.

## Dead ends
(none yet)
