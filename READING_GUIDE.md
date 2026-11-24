# How to read this repo

**What it is:** our own inference engine for **Inkling 2 Mini** (Thinking Machines, 26B looped MoE) on one **RTX PRO 6000**,
made **16× faster (6.1 → 98.1 tok/s)** by AI agents running one experiment at a time under a referee they cannot change.

**Know first:** PyTorch basics, how a transformer decodes one token at a time, what a KV cache is. Triton/CUDA helps but is
not needed on day 1.

## Day 1 — the big picture (~1 h, no code)
| # | File | What you learn |
|---|---|---|
| 1 | `README.md` | The result, the kept steps, the charts, how the loop works |
| 2 | `knowledge/model.md` | The model: 24 layers, the middle 12 run 3× (48 visits per token), 128 experts, bytes per token, the speed ceiling |
| 3 | `knowledge/hardware.md` | GPU facts that matter: 1.3 TB/s, FP4/FP8 tensor cores, launch cost 22 µs vs 1 µs inside a CUDA graph |
| 4 | `DESIGN.md` | Phases, the score, the search space, which kernels matter |

Check yourself: why does one token read ~5 GB? Why is the ceiling ~240 tok/s?

## Day 2 — the engine, simplest first (~3 h)
| # | File | What you learn |
|---|---|---|
| 5 | `engine/weights.py` | How 4-bit NVFP4 and 8-bit MXFP8 weights are unpacked |
| 6 | `engine/model_v0.py` | **Read slowly.** The whole model in plain PyTorch: the loop, block residuals (AttnRes), relative-bias attention, MoE routing |
| 7 | `tests/test_v0.py` | How v0 was proven correct (perplexity 3.06, "Paris") |
| 8 | `ref/model_v1.py` | v0 + per-visit KV caches for step-by-step decoding (frozen reference) |

Check yourself: what carries over between tokens? (K/V and the last 3 inputs of each short convolution; everything else is per token.)

## Day 3 — how it got 16× faster (~3 h)
Read `research/results.tsv` (every run), then the files in the order the speedups happened:

| Speedup | File | Idea |
|---|---|---|
| 6.1 → 9.2 | `engine/moe_triton.py` | MoE with no CPU↔GPU syncs (Triton kernel over the top-8 experts) |
| → 31.7 | `engine/model_v1.py` (T=1 path) | Static buffers + the whole decode step as one CUDA graph |
| → 64.5 | `engine/fused_t1.py` | `torch.compile` fusing the small per-token ops |
| → 79.0 | `engine/moe_packed.py` | 4-bit experts read directly, unpacked inside the kernel |
| → 92.3 | `engine/fp8_dense.py` | FP8 attention projections and output layer (quality-gated) |
| → 96.0 | `engine/blend_triton.py` | Loop-message blend as 3 Triton kernels |
| → 98.1 | `engine/spec.py`, `engine/dspark.py` | DSpark speculative decoding, fully CUDA-graphed |

`engine/infer.py` is the entry point the benchmark calls.

## Day 4 — the autoresearch machinery (~2 h)
| # | File | What you learn |
|---|---|---|
| 1 | `prepare_gpu.py` | The referee: external timing, fixed token counts, quality gates |
| 2 | `decide.py` | KEEP only if faster than the best by more than max(1.5%, 2× noise) |
| 3 | `program.md` | The brief the agents read: ranked ideas, rules, one experiment per session |
| 4 | `loop.sh` | One fresh agent per experiment; reverts anything not kept; commits the research record |
| 5 | `run_remote.sh`, `tools/` | How a laptop drives the remote GPU (MoLab here; adapt to SSH) |

## Day 5 — the research story
| File | What you learn |
|---|---|
| `research/notes.md` | The agents' lab notebook: every hypothesis, measurement, failure and lesson, in order |
| `research/runs/*.json` | The full measurement of every benchmark run |
| `knowledge/measured.md` | Key numbers, dead ends, competitor targets (vLLM, ninfer) |
| `CHECKLIST.md` | 28 optimization areas: what's done, what's left |
| `discarded/*` branches | Code of every rejected experiment (`git log discarded/<name>`) |
| `PRODUCT.md` | Where this is going: autoresearch for any model on any local hardware |

Reference only: `reference/vllm_inkling2/` (vLLM's DSpark math, read-only), `launch/`, `golden/`, `data/`.

## Exercises
1. Run `engine/model_v0.py` on one prompt and print the perplexity.
2. Find the biggest single jump in `research/results.tsv` and explain why it was so big.
3. Pick one DISCARD in `research/notes.md` and explain why it didn't help.
4. Read "NEW TARGET" in `knowledge/measured.md`: what does ninfer do that we don't?
