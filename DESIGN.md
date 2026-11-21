# auto-inference-gpu — autoresearch for Inkling 2 Mini on one RTX PRO 6000

Goal: our **own** inference engine for Inkling 2 Mini (no vLLM/SGLang at runtime; PyTorch + Triton + CUDA only),
made fast by an autoresearch loop: agents edit code, a frozen harness measures, gates quality, keeps or reverts.
Lessons carried over from `auto-inference-mlx` (Mac, +23.5%): external timing, fixed work, noise-aware KEEP,
knowledge base first, fresh agent per experiment, shared `notes.md`, profiling-only experiments are valid.

## 1. The ceiling (see knowledge/model.md)
Batch 1 reads **≈ 5.4 GB/token** (2.6 GB BF16 attention weights × 48 visits, 1.8 GB experts, 0.93 GB unembed)
→ ~4.1 ms at 1.3 TB/s → **~240 tok/s ceiling** at checkpoint precision. With FP8/NVFP4 attention + FP8 unembed the
ceiling moves to **~400–550 tok/s**. At batch 32+ the experts stop being "top-8 of 128": most experts get touched, so
throughput becomes MoE-GEMM-bound — a different regime with different winners.

## 2. Phases
| Phase | Output | Done when |
|---|---|---|
| **0. Weights** | loader: safetensors → NVFP4/MXFP8 dequant → BF16 (~53 GB) or kept packed | every tensor accounted for |
| **1. Engine v0 (correct, slow)** | `engine/model.py`: plain PyTorch, one token at a time, BF16 everywhere, dense KV per pass | passes the correctness anchors below |
| **2. Harness (frozen)** | `prepare.py`: workloads, timing, gates, KEEP/DISCARD, `results.tsv`, dashboard | baseline recorded, noise < ±2% |
| **3. Engine loop** | agents edit `engine/` (one idea per run) | — |
| **4. Kernel loop** | AutoKernel-style per-kernel loop for the top kernels by Amdahl share | — |

### Correctness anchors (no reference engine installed)
1. **Unit checks per op** against straightforward PyTorch written from the reference code (AttnRes mix, conv, message
   blend, rel-bias attention, MoE routing, NVFP4 dequant).
2. **Model-level:** perplexity on a fixed text set must be sane (and frozen as `golden_ppl`), greedy GSM8K-style mini
   eval (vendor reports 97.5% GSM8K), coding/JSON items like the Mac harness.
3. Once v0 passes: freeze **golden logits** from v0 for 32 prompts → every later experiment must stay within
   **KL < τ** of golden and keep eval ≥ golden − 1 item. Lossy changes (FP8 attention, FP8 KV, fewer passes) are
   allowed only if they pass this gate.

## 3. The score
- **Primary:** decode tok/s, batch 1, geomean over 4 workloads (chat 256, long-context 8k-prompt 256, code 256, reasoning 512).
- **Secondary (logged, separate leaderboard):** throughput tok/s at batch 32 (agentic 4k contexts), TTFT at 8k prompt,
  peak memory, KV bytes/token.
- KEEP if primary > best × (1 + max(1.5%, 2×noise)) AND all gates pass. (Option: also KEEP a secondary win ≥ 10% with
  primary not worse than −1%.)

## 4. Search space ("PnC": the agent combines axes, one change per run)
| Axis | Values |
|---|---|
| Execution | eager → **CUDA graph per batch size** → torch.compile regions → persistent megakernel |
| Attention weights precision | BF16 / FP8 per-channel / NVFP4 (gated by KL) |
| Unembed | BF16 / FP8 / NVFP4, fused softcap + argmax |
| Experts | dequant-BF16 GEMM → native NVFP4/MXFP8 block-scaled MMA → grouped GEMM vs per-expert GEMV by batch size |
| KV cache | BF16 / FP8; layout (dense, paged); **cross-pass sharing for local layers** (research) |
| Fusions | AttnRes mix + RMSNorm, short conv + residual, message controller (pair-norm → rank-64 → tanh blend), router (softcap + bias + top-8) |
| Attention kernel | SDPA/FlexAttention → Triton split-KV decode with sink + rel-bias + log-scaling |
| Speculation | none → DSpark depth 1–7 (exact greedy verify) |
| Loop | 3 passes (fixed checkpoint); variable-loop research checkpoint later |

## 5. Kernels that matter on sm_120 (ranked by expected share of decode time)
1. **Attention projections GEMV** (2.6 GB/token BF16): memory-bound; FP8/NVFP4 weight-only + fused dequant is the biggest byte lever.
2. **MoE expert GEMV/GEMM, NVFP4** (top-8, I=1152): block-scaled FP4 MMA on sm_120 (`mma.sync` block_scale), or
   dequant-in-register GEMV at batch 1. Router + gather/scatter fused.
3. **Unembed GEMV + softcap + argmax** (0.93 GB/token): one fused kernel, FP8 weights.
4. **Decode attention**: sliding window 1024 (40 of 48 visits) + 8 global visits; rel-bias branch, sinks, gate, log-scaling. Split-KV for long context (lab 05 code).
5. **AttnRes mix** (softmax over ≤ 5 blocks + RMSNorm): tiny but on the critical path 96× per token → fuse.
6. **Short convs** (attn, mlp, k, v streams; kernel 4) with conv-state update: fuse with the mix (vLLM's native `conv_mix` is sm90/sm100-only → open target on sm_120).
7. **Loop message controller** (36 core visits × 4 streams): pair RMSNorm → 4608→64 → 64→2×2304 → tanh blend, one kernel.
8. **Whole-step CUDA graph** (launch 22 µs → 1.1 µs; a token is ~1,000+ kernels eager → graphs are mandatory).

## 6. Two loops
- **Engine loop** (like the Mac): agent edits `engine/`, harness scores end-to-end. Good for precision, graphs, fusion boundaries, speculation.
- **Kernel loop** (AutoKernel method): `profile.py` ranks kernels by GPU time × Amdahl → extract one op with its exact
  decode shapes → agent edits `kernels/<op>.py` → `bench_kernel.py` (5-stage correctness: smoke, shape sweep,
  numerics, determinism, edge cases + roofline %) → plug back in → engine harness confirms.

## 7. Infrastructure (MoLab)
- Agent runs on the Mac (`claude -p`), code lives in this repo; `remote.sh` syncs `engine/` to the sandbox and runs the
  harness there via marimo-pair (Popen + poll, results written to /tmp/runs/*.json after every workload, pulled back).
- Sandbox restarts: `setup_gpu.sh` re-creates the toolchain + downloads weights (19 GB in < 1 min).
