# Product: autoresearch for local inference

**One sentence:** a repo that turns any coding agent (Claude Code, Codex, …) into an inference engineer for *your* model on
*your* hardware: it measures the machine, understands the model, then runs experiments one at a time, kept honest by a
referee it cannot change, until your local inference is as fast as it can get.

## Why local can beat vLLM / SGLang
vLLM and SGLang are built for many users on datacenter GPUs (batching, schedulers, paging, generic kernels).
A local user has **one model, one chip, one user, one workload**, which allows specialization those engines can't do:
kernels for this chip, number formats for this chip, speculation for this user's tasks, no generic layers.
Evidence from this project's two runs:
- Mac M1 Pro, Qwen3.5-0.8B (MLX): **+23.5%** (fp16 because M1 emulates bf16, + fused Metal kernels).
- RTX PRO 6000, Inkling 2 Mini (own engine): **6.1 → 96 tok/s (15.7×)**, quality held (see results.tsv).

## User flow
1. Clone the repo (or give its link to a coding agent).
2. `detect` – chip, memory, measured bandwidth, native number formats, launch cost, caches, NPU, kernel languages
   available (CUDA / Triton / Metal / ROCm-HIP / Vulkan / CPU SIMD), power + thermal state.
3. `init <model>` – download, build the op graph (bytes/token, KV/token, architecture family), compute the roofline ceiling,
   pick a backend, run the slow correct **reference** once and save **golden** outputs (logits/NLL/eval answers).
4. `workload add` – optional: the user's own prompts / agent traces (the score should reflect real use).
5. `baseline` – first score + the gap to the ceiling.
6. `research --agent claude|codex --runs N` (or `--one`) – the loop: one experiment per fresh agent session, guided by
   `CHECKLIST.md`, judged by the frozen harness, KEEP (commit) / DISCARD (revert, saved on a `discarded/` branch).
7. `report` – dashboard: kept changes, speedup, quality, cost (agent tokens + machine hours).
8. `export` / `serve` – the optimized engine or config, OpenAI-compatible endpoint.

## Components
| Component | Role |
|---|---|
| `detect` | hardware probe → `knowledge/hardware.md` (measured, not spec-sheet) |
| `model` | architecture profiler → `knowledge/model.md` (op graph, bytes, KV, families) |
| `ceiling` | roofline: decode = bytes/token ÷ bandwidth, prefill = FLOPs ÷ compute |
| `ref/` + `golden/` | slow correct reference + saved truth; never edited by agents |
| `harness` | frozen referee: external timing, fixed work, noise-aware KEEP rule, quality gates (KL/NLL, eval, determinism), power guard |
| `backends/` | starting engines: MLX (Mac), llama.cpp (any CPU/GPU), PyTorch+Triton (NVIDIA/AMD), "own engine" (write from the model's reference code) |
| `loop` | agent runner: fresh session per experiment, shared `notes.md`, enforced revert, stops on usage limit / lost hardware, no background jobs |
| `CHECKLIST.md` | the map of optimization dimensions + "applies?" rules + coverage table |
| `tricks/` (later) | shared lessons across hardware and models, so each new run starts smarter |
| `report` | dashboard, charts, exportable result |

## Rules that keep it honest (learned the hard way)
- Speed is measured **outside** the engine; never self-reported. Fixed work (exact token counts). Noise-aware keep bar.
- Quality is gated against golden outputs; lossy changes are allowed only if the gates pass. Grow the eval beyond a sanity set.
- One experiment per fresh agent; memory lives in files (`notes.md`, `knowledge/`, `CHECKLIST.md`).
- The loop, not the agent, enforces reverts; discarded work is kept on branches for later retries and stacking.
- No background jobs inside agent sessions (the session ends and the result is lost).
- Budget per run (max experiments / max cost); stop cleanly on usage limits or when the hardware disappears.
- Compare with vLLM / SGLang / llama.cpp only **on the same machine, same workloads**.

## Honest limits (put these in the README)
- The agent must run on the real hardware it optimizes.
- It costs agent usage (~$1–1.5 per experiment in our runs).
- Quality is only as good as the gates; small evals are sanity checks, not proof.
- Beating vLLM/SGLang is most likely for one user / one machine / one workload, not for multi-user serving.

## Build order
1. v0.1 – generalize the Mac project: `detect` + `model` + `harness` + `loop`, backends MLX + llama.cpp (Macs + any laptop).
2. v0.2 – PyTorch + Triton backend for NVIDIA (from this project's engine and harness).
3. v0.3 – speculation module, KV-cache module, shared `tricks/` library, energy-per-token score.
4. v0.4 – AMD/ROCm, Windows, NPU backends, `serve`.
