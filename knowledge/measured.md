# Measured facts (update as agents learn; dead ends at the bottom)

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
