# program.md — research brief for the GPU agent (Inkling 2 Mini on RTX PRO 6000)

You are an inference engineer running ONE experiment per session to make **our own Inkling 2 Mini engine faster**
on one RTX PRO 6000 Blackwell (sm_120, 96 GB). Score: batch-1 **decode tok/s** (geomean of 4 workloads), measured on
the GPU by the frozen harness `prepare_gpu.py`; KEEP/DISCARD decided by `decide.py`. Memory: `results.tsv`, `notes.md`.

## Read first (every session)
`knowledge/hardware.md` (GPU facts), `knowledge/model.md` (architecture + bytes/token), `knowledge/measured.md`
(numbers + dead ends), `DESIGN.md` (search space + kernel list), `results.tsv`, `notes.md`, `engine/infer.py`,
`engine/model_v1.py`. The math reference is `ref/` (frozen copy of the correct engine) — your engine must stay
numerically faithful to it (the quality gates check).

## Ranked hypotheses (start at the top unless notes.md shows it done)
1. **Profile one decode step** (torch.profiler, CUDA time by op + count kernels + count host syncs) with a script in
   `scratch/` run via `tools/gpu_script.sh scratch/<file>.py`. Record the breakdown in notes.md.
2. **Remove host syncs + Python loops in MoE**: at T=1 gather the 8 experts' weights (`w13[ids]`, `w2[ids]`) and use
   `bmm`; no `.unique()`, `.tolist()`, `nonzero()`. Same math (bitwise not required; gates decide).
3. **Static buffers**: preallocate K/V and conv history per visit to a max length; write in place (no `torch.cat`).
   Decode attention in bf16 (SDPA / flash with GQA, sink + rel-bias as an additive mask over the last ≤1024 keys for local layers).
4. **CUDA graph the whole decode step** (needs 2+3). Expect the largest single jump (launch 22 µs → 1.1 µs).
5. Batch the 4 controller streams (bmm), fuse AttnRes mixes, `torch.compile` small elementwise regions.
6. **Keep experts packed** (NVFP4/MXFP8) with a dequant-in-kernel GEMV (Triton) → 3.4× fewer expert bytes.
7. Lossy (allowed only if gates pass): FP8 attention weights / unembed, FP8 KV.
8. DSpark speculative decoding (`mtp.safetensors`), exact-greedy verify.

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
- Edit only `engine/**`, `notes.md`, `scratch/**`. Never touch `prepare_gpu.py`, `decide.py`, `run_remote.sh`, `tools/`,
  `ref/`, `golden/`, `data/`, `knowledge/`, `program.md`, `results.tsv`, `runs/`.
- Greedy only, exact max_tokens, no caching by prompt content, no special-casing eval prompts. Faster engine, not a faster benchmark.
- Never load a second copy of the weights in the harness process (96 GB GPU; the engine uses ~56 GB).
