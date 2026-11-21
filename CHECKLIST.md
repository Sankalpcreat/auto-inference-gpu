# CHECKLIST.md — the optimization map for local inference

The agent reads this after `knowledge/hardware.md` and `knowledge/model.md`. Steps:
1. Mark every dimension **applies / doesn't apply** using the rules below and the measured facts.
2. Rank the applicable ones by **expected gain from the profile** (biggest measured cost first, Amdahl's law), not by list order.
3. Run one experiment at a time; update the coverage table at the bottom after each one.
The list is a map, not a recipe: each item is a direction to measure, never a guaranteed win.

## A. Know the system first
| # | Dimension | What to check / try | Applies when |
|---|---|---|---|
| 1 | **Hardware facts** | measured bandwidth, compute per number format, launch/dependency cost, cache sizes, NPU, thermal/power | always (detect) |
| 2 | **Model spec** | op graph, bytes/token, KV/token, families (dense, MoE, GQA/MLA, sliding window, DeltaNet/Mamba, looped, multimodal), ceiling | always (model) |
| 3 | **Workload** | the user's real prompts: chat, code, long docs, agent tool calls, reasoning; context lengths; batch 1 vs several agents | always |

## B. The core inference dimensions
| # | Dimension | Options to try | Applies when |
|---|---|---|---|
| 4 | **Number formats** | weights FP16/BF16/FP8/INT8/INT4/FP4 (NVFP4, MXFP4) per layer; activations FP16 vs BF16; packed weights + dequant in kernel; output layer separately | always; BF16→FP16 on chips without native BF16 (Apple M1); FP8 on sm_89+; FP4 on sm_120+/Blackwell; fewer bits ≠ faster when dequant is ALU-bound (M1) |
| 5 | **Memory system** | static buffers, no host syncs, wired memory (Mac), pinned memory, cache limits, memory-mapped loading | always |
| 6 | **Execution** | CUDA graphs / `mx.compile` / torch.compile, fewer dependent steps, megakernels | always; most important for small models and batch 1 |
| 7 | **Kernels** | fused norm+matmul, fused MoE (router + grouped GEMV), packed low-bit GEMV, split-K vs tile shape, decode attention (split-KV), Triton / CUDA / Metal / HIP | always; profile first (small GEMVs are often latency-bound, not bandwidth-bound) |
| 8 | **KV cache** | layout (paged, ring buffer for windows, per-pass for looped models), FP8/INT8 KV, prefix caching (agent loops), sharing across layers/passes, eviction for long chats | models with attention; tiny gains if few layers hold KV (hybrids) |
| 9 | **Speculative decoding** | n-gram / prompt lookup, the model's own MTP/drafter (DSpark, EAGLE), small draft model, adaptive draft length, tree drafts; needs a cheap multi-token verify (weights read once for K+1 tokens) + rollback of all state | when verify(K+1) ≲ 1.5× a single step; drafter acceptance decides the gain |
| 10 | **Prefill** | chunked prefill, flash attention variant, CPU+GPU co-execution (Apple), prompt caching | long prompts, agent loops, TTFT-sensitive use |

## C. Dimensions often missed
| # | Dimension | Options to try | Applies when |
|---|---|---|---|
| 11 | **Sparsity / conditional compute** | activation sparsity (read only weights big activations need, e.g. SpAx), fewer experts for unimportant tokens (MoBiLE), layer skipping / early exit, adaptive loop passes (looped models) | lossy → gated; ReLU-era tricks (PowerInfer) need ReLU-like activations |
| 12 | **Storage hierarchy** | stream weights from SSD/flash (LLM in a Flash, PowerInfer-2), hot experts in VRAM / cold in RAM/SSD, mmap | model larger than memory; MoE on laptops |
| 13 | **Heterogeneous compute** | NPU (Apple ANE, Intel/AMD/Qualcomm NPUs), CPU + iGPU + dGPU split (e.g. prefill on GPU, decode elsewhere) | hardware has idle units and a runtime that can reach them |
| 14 | **Model surgery (with healing)** | prune experts/layers/heads, trim vocabulary/output layer to the user's languages, distill, low-rank; **fine-tune the drafter on the user's own traces** | user accepts a modified model; needs a healing/eval budget |
| 15 | **Compiler / runtime** | TVM/MLC, ExecuTorch, ONNX Runtime, IREE: graph fusion, scheduling, autotuning | a second backend is available for the hardware |
| 16 | **Input / output side** | prompt compression, context summarization, structured-output (grammar mask) cost, tokenizer/detokenizer speed, sampling kernels | tool calls, long chats, JSON output |
| 17 | **Test-time compute policy** | reasoning effort, token budgets, stop rules: end-to-end time = tokens × time per token | reasoning models; often the biggest end-to-end lever |
| 18 | **Energy / thermal / power** | energy per token as a score, throttling, battery vs AC, power modes | laptops |
| 19 | **Local concurrency** | several agents at once, draft + main + embedding models sharing one device, shared prefix caches | multi-agent local setups |
| 20 | **Startup / load** | cold start, weight load time, compile caches, warm-up | users who restart often |
| 21 | **Multimodal path** | vision-encoder cost, fewer image tokens, cache encoded images | image/PDF/screenshot inputs |
| 22 | **Multi-device local** | several Macs over Thunderbolt RDMA, 2 GPUs over PCIe (two TP1 servers often beat one TP2) | more than one device |

## D. How the research itself is run
| # | Dimension | Rule |
|---|---|---|
| 23 | **Measurement trust** | honest baselines (AutoTuneBench: a "10.6×" kernel was 2.03× against an honest baseline); external timing; fixed work |
| 24 | **Search strategy** | profile → biggest cost first; stack near-miss ideas (two +2% changes can clear a bar together); record dead ends |
| 25 | **Knowledge transfer** | write general lessons ("BF16 is emulated on M1", "small GEMVs are latency-bound on RTX") into `knowledge/` / `tricks/` |
| 26 | **Cost budget** | agent tokens + machine hours per % gained; stop when the return per dollar drops |

## Coverage — this repo (Inkling 2 Mini, RTX PRO 6000, batch-1 decode), 2026-11-21
| # | Dimension | Applies? | Tried | Result | Status |
|---|---|---|---|---|---|
| 4 | Number formats | yes | packed NVFP4/MXFP8 experts; FP8 dense + unembed | 64.5→79.0 (KEEP), →92.3 (KEEP, NLL ratio 0.996) | done (FP8 KV untried) |
| 5 | Memory system | yes | sync-free MoE, static K/V buffers | 6.1→9.2→11.8 (KEEP) | done |
| 6 | Execution | yes | whole-step CUDA graph; torch.compile regions | 11.8→31.7→64.5 (KEEP) | done |
| 7 | Kernels | yes | Triton MoE GEMV, packed GEMV, FP8 GEMV retune, split-K, fused attention, Triton blend | blend 92.3→96.0 (KEEP); retune +3% / split-K / fused attention (DISCARD) | in progress |
| 8 | KV cache | yes | ring buffer (local), buckets (global), fp32→bf16 K/V | part of KEEPs; FP8 KV untried | partial |
| 9 | Speculative decoding | yes | n-gram simulation (1.33 tok/step); verify(5) projected 1.37×; verify + rollback built, exact 64/64 | DSpark port next | in progress |
| 10 | Prefill | yes (~20× behind vLLM) | — | not in the score yet | todo |
| 11 | Sparsity / adaptive passes | yes (varloop checkpoint) | — | — | todo |
| 17 | Test-time compute policy | yes (reasoning_effort) | — | not in the score (raw completions) | todo |
| 3 | Workload breadth | partial | 4 short workloads, batch 1 | long context, batch 8, prefill not measured | todo |
| 13, 12, 22 | NPU / storage / multi-device | no (96 GB GPU, single device) | — | — | skipped |
