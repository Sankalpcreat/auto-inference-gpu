# program.md — research brief for the GPU agent (Inkling 2 Mini on RTX PRO 6000)

You are an inference engineer running ONE experiment per session to make **our own Inkling 2 Mini engine faster**
on one RTX PRO 6000 Blackwell (sm_120, 96 GB). Score: batch-1 **decode tok/s** (geomean of 4 workloads), measured on
the GPU by the frozen harness `prepare_gpu.py`; KEEP/DISCARD decided by `decide.py`. Memory: `results.tsv`, `notes.md`.

## Read first (every session)
`knowledge/hardware.md` (GPU facts), `knowledge/model.md` (architecture + bytes/token), `knowledge/measured.md`
(numbers + dead ends), `DESIGN.md` (search space + kernel list), `results.tsv`, `notes.md`, `engine/infer.py`,
`engine/model_v1.py`. The math reference is `ref/` (frozen copy of the correct engine) — your engine must stay
numerically faithful to it (the quality gates check).

## Ranked hypotheses (now at 64.5 tok/s; target: beat vLLM's 10.0 ms/step ≈ 200–400 tok/s with DSpark)
1. **Re-profile the current graph replay** if notes.md has no profile at 64.5 tok/s (kernel count, time by kernel class, bytes vs time).
2. **Keep experts packed (NVFP4 layers 0–19, MXFP8 20–23)**: Triton GEMV that dequantizes in registers (E2M1 LUT x FP8-E4M3
   block scale per 16 x amax/2688; MXFP8: E4M3 x 2^(s-127) per 32). Expert bytes 6 GB → 1.7 GB/token (expect ~−3.5 ms).
   Must match the BF16-dequant result (same dequant math, fp32 accumulate).
3. **FP8 dense projections** (wq, gate, wk, wv, wo, wr, sink; per-output-channel scale, weight-only, fp32 accumulate) and
   **FP8 unembed**: 2.6 → 1.3 GB and 0.93 → 0.46 GB. Lossy: gates decide (NLL +2%, eval ≥ 11/12).
4. **DSpark speculative decoding** (`/tmp/inkling/mtp.safetensors`, 3 layers, block 8, reads hidden states at visits 5, 23, 41 + final);
   exact greedy verify of 1+K tokens per step in one graph; needs a T>1 decode path. Then **adaptive K** (draft long only when confident).
5. Remaining small-kernel fusion (Triton) where the profile says.
6. Loop-aware ideas: expert overlap across passes, per-pass caches.

## How to run ONE experiment
1. Read the files above. Don't repeat a dead end without a new reason.
2. Optional: ≤ 3 WebSearch/WebFetch on one technique; profiling scripts in `scratch/` via `tools/gpu_script.sh`.
3. Write the hypothesis + expected gain in `notes.md` BEFORE running.
4. Edit only files under `engine/` (keep the `engine/infer.py` contract), then `git add engine && git commit -q -m "<idea>"`.
5. `./run_remote.sh "<one-line description>" > logs/last_out.log 2>&1`, then read `logs/last_out.log` (~5 min at baseline speed).
6. `DECISION: KEEP` → keep the commit. `DISCARD` / `FAIL-GATE` → `git reset --hard HEAD~1`.
   `CRASH` → fix once if trivial, else reset. `REMOTE START FAILED` / `TIMEOUT` → the sandbox is gone: write that in notes.md and STOP.
7. Append to `notes.md`: what, numbers, decision, what it teaches, next idea. Finish with a 3–5 line summary.

## Rules
- **Never run anything in the background and never use Monitor**: this session ends when you stop talking, so a background job's result is lost. Run `tools/gpu_script.sh` / `./run_remote.sh` in the foreground with the Bash `timeout` set to 600000 (10 min); keep sweeps under ~8 minutes.
- Edit only `engine/**`, `notes.md`, `scratch/**`. Never touch `prepare_gpu.py`, `decide.py`, `run_remote.sh`, `tools/`,
  `ref/`, `golden/`, `data/`, `knowledge/`, `program.md`, `results.tsv`, `runs/`.
- Greedy only, exact max_tokens, no caching by prompt content, no special-casing eval prompts. Faster engine, not a faster benchmark.
- Never load a second copy of the weights in the harness process (96 GB GPU; the engine uses ~56 GB).
