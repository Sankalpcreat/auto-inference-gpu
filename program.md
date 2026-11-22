# program.md — research brief for the GPU agent (Inkling 2 Mini on RTX PRO 6000)

You are an inference engineer running ONE experiment per session to make **our own Inkling 2 Mini engine faster**
on one RTX PRO 6000 Blackwell (sm_120, 96 GB). Score: batch-1 **decode tok/s** (geomean of 4 workloads), measured on
the GPU by the frozen harness `prepare_gpu.py`; KEEP/DISCARD decided by `decide.py`. Memory: `results.tsv`, `notes.md`.

## Read first (every session)
`knowledge/hardware.md` (GPU facts), `knowledge/model.md` (architecture + bytes/token), `knowledge/measured.md`
(numbers + dead ends), `DESIGN.md` (search space + kernel list), `results.tsv`, `notes.md`, `engine/infer.py`,
`engine/model_v1.py`, and `CHECKLIST.md` (the optimization map: mark what applies, update its coverage table after your experiment). The math reference is `ref/` (frozen copy of the correct engine) — your engine must stay
numerically faithful to it (the quality gates check).

## Ranked hypotheses (now 95.98 tok/s ≈ 10.4 ms/token; small-kernel tweaks are at diminishing returns: last 5 tries ≤ +4%)
**NOW (supervisor, 2026-11-22): finish speculative decoding — 2 experiments, then stop.**
DSpark works end to end (#23: 2.66 tok/step at K=4) but output drifts from greedy. Code: `git show discarded/1122-071403:engine/spec.py` and
`discarded/1122-071403:engine/dspark.py` (restore both).
1. Log the top-2 logit gap where outputs split (tiny → precision, large → bug; fix a bug first).
2. Build the verify step on the SAME kernels the T=1 decode uses (FP8 dense GEMV per row or small-batch FP8, `packed_gemv` per
   (row, expert) in one launch, same attention path), CUDA-graphed per K. Bit-exact greedy match is NOT required: #24 showed every split is a near-tie (gap <= 0.29) and even the
   normal graphed decode drifts from eager prefill; the harness gates (eval, NLL, determinism, fixed work) decide, as they did for the FP8 KEEP.
   Speed matters: reuse the T=1 kernels so verify(K+1) costs ~1.4-1.8x a step.
3. Run the harness with spec on (K=2 and K=4; pick the better). The gates still apply (golden prefix, eval, NLL, fixed work).
Stop rule: if spec decoding can't reach ≥ ~130 tok/s with the gates passing in this experiment, record why and stop.
**#1 is speculative decoding — read `knowledge/dspark.md` first.** Work through its stages A → B → C, one per experiment:
A. T = K+1 verify step with per-visit rollback (K/V write positions, local ring buffers, conv histories) + an n-gram/prompt-lookup
   drafter with exact greedy verify. Report T(K+1)/T(1), accepted tokens/step per workload. CUDA-graph the verify step per K.
B. Port the DSpark drafter (`reference/vllm_inkling2/`, `/tmp/inkling/mtp.safetensors`), eager first; report acceptance by position.
C. Graph-capture draft + verify; adaptive K.
Only if blocked on spec decoding: further kernel-count reduction (attn_res mix as one Triton kernel was in progress —
see the newest `discarded/` branch; FP8 GEMV retune (8,128,8) is +3% stackable).

## How to run ONE experiment
1. Read the files above. Don't repeat a dead end without a new reason.
2. Optional: ≤ 3 WebSearch/WebFetch on one technique; profiling scripts in `scratch/` via `tools/gpu_script.sh`.
3. Write the hypothesis + expected gain in `notes.md` BEFORE running.
4. Edit only files under `engine/` (keep the `engine/infer.py` contract), then `git add engine && git commit -q -m "<idea>"`.
5. `./run_remote.sh "<one-line description>" > logs/last_out.log 2>&1`, then read `logs/last_out.log` (~5 min at baseline speed).
6. `DECISION: KEEP` → keep the commit. `DISCARD` / `FAIL-GATE` → `git reset --hard HEAD~1` (the loop also reverts engine/ to the last KEEP after you exit and saves unkept work on a `discarded/` branch; always start from HEAD).
   `CRASH` → fix once if trivial, else reset. `REMOTE START FAILED` / `TIMEOUT` → the sandbox is gone: write that in notes.md and STOP.
7. Append to `notes.md`: what, numbers, decision, what it teaches, next idea. Finish with a 3–5 line summary.

## Rules
- **Never run anything in the background and never use Monitor**: this session ends when you stop talking, so a background job's result is lost. Run `tools/gpu_script.sh` / `./run_remote.sh` in the foreground with the Bash `timeout` set to 600000 (10 min); keep sweeps under ~8 minutes.
- Edit only `engine/**`, `notes.md`, `scratch/**`, and the coverage table in `CHECKLIST.md`. Never touch `prepare_gpu.py`, `decide.py`, `run_remote.sh`, `tools/`,
  `ref/`, `golden/`, `data/`, `knowledge/`, `program.md`, `results.tsv`, `runs/`.
- Greedy only, exact max_tokens, no caching by prompt content, no special-casing eval prompts. Faster engine, not a faster benchmark.
- Never load a second copy of the weights in the harness process (96 GB GPU; the engine uses ~56 GB).
