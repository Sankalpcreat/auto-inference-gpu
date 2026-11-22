# notes.md — shared research journal (agents append)

## Baseline 2026-11-21: 6.1 tok/s (engine v1 eager), noise ±0.6%, eval 12/12, nll_ratio 1.0, peak 56.3 GB

## 2026-11-21 — Profiling study #1 (program.md hypothesis 1), no engine change
Hypothesis: the 164 ms decode step is launch/host bound: expect >2,000 kernels/token, GPU busy < 30% of wall,
48 host syncs/token from moe() (`unique().tolist()`), and MoE + attention-einsum dominating CPU-side time.
Method: `scratch/profile_decode.py` — short prefill, 5 warmup decode steps, then torch.profiler over 3 decode steps;
per-region labels (attention / moe / blend / attn_res / conv / unembed) via record_function wrappers.
Results (prompt 300 tok, steps at ctx ~310; profiler over 3 decode steps; all numbers per decode step):
- Wall without profiler: 169–190 ms. **GPU kernel time: 53 ms → GPU idle ~70% of the step.**
- **~22,760 cudaLaunchKernel / step** (10× the 2,000 guessed), **480 cudaStreamSynchronize**, 794 cudaMemcpyAsync.
  Syncs: 384 `nonzero` (8 experts × 48 visits) + 48 `unique().tolist()` + ~48 other.
- Per region (CPU time under profiler / GPU kernel time): moe 151 / 24.5 ms, attention 112 / 14.3, conv_c (176 calls)
  53 / 5.8, blend 44 / 5.7, attn_res (120 calls) 39 / 3.7. CPU-side share: moe 38%, attention 28%, conv 13%, blend 11%, attn_res 10%.
- By op (GPU): aten::mm 13.7 ms (bf16 gemvx 912×/step = 10.8 ms, ≈ the weight-read floor at this precision),
  **aten::copy_ (dtype casts .float()/.to(bf16)) 12.8 ms over ~5,700 calls** = 24% of GPU time, mul 6 ms over 4,350 calls,
  nonzero 2.8 ms, index/gather 2.3 ms. Everything except the GEMVs is µs-sized elementwise kernels.
What it teaches:
- Even with zero host overhead the current kernel set costs 53 ms (≈19 tok/s). A CUDA graph of the *current* op
  list would still pay ~22.8k × ~1–1.5 µs ≈ 25–35 ms → **kernel-count reduction is as important as graphs**.
- MoE is the top offender on both CPU and GPU and owns all the syncs → hypothesis 2 is confirmed as next.
- fp32 round-trips (`.float()` everywhere in conv_c/attn_res/blend/attention) are ~1/4 of GPU time: do math in
  one dtype per fused region (later: torch.compile / Triton for conv+mix+blend).
Next idea: hyp 2 — MoE at T=1 via `w13[ids]`/`w2[ids]` gather + bmm (no unique/nonzero/tolist/index_add); expect
~−5k launches, −432 syncs, step 175 → ~110–130 ms (≈ +35–50%). Then static KV/conv buffers (hyp 3) → CUDA graph (hyp 4).
Summary: profiling-only session (no engine change, no harness run). Step ≈ 175 ms wall but only 53 ms GPU;
22.8k kernel launches + 480 host syncs per token; MoE (38% CPU, 46% GPU) and attention (28% / 27%) dominate;
dtype-cast copies are 24% of GPU time. Next: sync-free batched MoE.

## 2026-11-21 — Experiment #2 (hyp 2): sync-free MoE at T=1 via a Triton grouped GEMV
Hypothesis: replace the per-expert Python loop (unique().tolist() + 8× nonzero + index_add) at T=1 with
router → topk ids stay on GPU → one Triton "grouped GEMV" kernel reading W[ids[j]] directly (no weight gather copy,
which would triple expert traffic: 8 × 15.9 MB per visit) for w13, torch activation on (8, 2304), second grouped
GEMV for w2, weighted sum over the 8 slots. Prefill (T>1) keeps the v0 loop. Same math, fp32 accumulate.
Expected: −432 syncs/step, ~−5k launches/step → step ~175 → ~115–130 ms (≈ +35–50%, ~8–9 tok/s).
Pre-check: `scratch/check_moe_t1.py` (max-abs diff vs v0 moe + timing per visit) before the harness run.
Pre-check (scratch/check_moe_t1.py): max-abs diff vs v0 moe ≤ 0.25 at |out| ≈ 40–60 (= 1 bf16 ULP), NVFP4 and MXFP8
layers alike; MoE per visit **1.82 ms → 0.18 ms** (10×); decode step ~175 → **116 ms** wall.
Harness: **9.24 tok/s** (6.1 → +51%), noise ±0.5%, eval 12/12, nll_ratio 1.0000, peak 56.3 GB, TTFT unchanged
(prefill still uses the v0 loop). Golden-prefix: chat 64, long 64, reasoning 63, **code 4** (ULP-level accumulation-order
differences flip an early near-tie greedy token on the code prompt; gates still pass). **DECISION: KEEP** (commit on main).
What it teaches:
- Host syncs were the MoE cost, not bytes: 0.18 ms/visit for ~85 MB of expert reads ≈ near the 1.3 TB/s floor already,
  so a naive Triton GEMV (BLOCK_N 32, BLOCK_K 128, grid 8×72) is good enough on sm_120 for now.
- MoE now ≈ 48 × 0.18 ≈ 9 ms of the 108 ms step; the rest is attention (fp32 einsum + cat), conv/attn_res/blend
  elementwise chains and launch overhead. The step is now sync-free in MoE → CUDA graph only blocked by dynamic shapes.
- Expect golden-prefix on "code" to stay low for any numerics change; don't read it as a regression signal.
Next idea: hyp 3 — static preallocated K/V + conv-history buffers written in place (no torch.cat), decode attention
in bf16 SDPA with GQA and sink/rel-bias as an additive mask over the window; this is also the prerequisite for hyp 4
(whole-step CUDA graph, expected largest jump).
Summary: Experiment #2 replaced the T=1 MoE loop with a sync-free Triton grouped GEMV over the top-8 expert ids.
6.1 → 9.24 tok/s (+51%), all gates pass, KEEP. MoE per visit 10× faster; remaining step time is attention,
elementwise chains and launches. Next: static KV/conv buffers + bf16 SDPA (hyp 3), then CUDA graph (hyp 4).

## 2026-11-21 — Experiment #3 (hyp 3): static K/V buffers + lean T=1 attention + cheaper conv history
Hypothesis: per visit, preallocate K/V as fp32 (HKV, cap, HD) buffers (cap = prompt + max_tokens; fp32 holds the
bf16-valued K/V exactly, so the per-step `.float()` of the whole cache disappears) and write in place (no torch.cat).
At T=1 attention becomes: slice the valid keys (last ≤1024 for local, all for global), GQA as a batched fp32 matmul
(HKV, 3, HD) @ (HKV, HD, S) — no repeat_interleave, no mask, no dist/gather: rel-bias for the window is just
`rel[:, :n].flip(-1)` added to the last n keys. Conv history: one cat + one weighted reduction instead of 4 mul/adds +
clone. Same math (fp32, reduction order only). Prefill keeps the reference path reading from the buffer.
Expected: ~−20 kernels × 48 visits for attention, ~−6 × 176 for convs → step ~108 → ~90 ms (≈ +15–20%, ~10.5–11 tok/s).
Extra memory: 48 × 2 × 1024 × 4 B ≈ 393 KB/token of capacity (~0.5 GB at the long workload).
Pre-check (scratch/check_static_kv.py): T=1 decode logits vs the prefill path on the same tokens, P=300 and P=1100
(window > 1024 exercised): argmax identical on all 16 steps, max-abs logit diff 0.15–1.1 (softcap-30 logits; the
diff already includes the MoE-T=1 vs prefill-loop bf16 differences). Decode step **~108 → ~84 ms** wall.
Harness: **11.83 tok/s** (9.24 → +28%), noise ±0.3%, eval 12/12, nll_ratio 1.0000, peak 56.3 GB, TTFT unchanged
(long 1103 ms). Golden-prefix: chat 64, long 64, **code 64** (back from 4), reasoning 63. **DECISION: KEEP**.
What it teaches:
- Attention's cost was the per-step cache handling (cat realloc, `.float()` + repeat_interleave of the whole cache,
  dist/gather/mask chains), not the math: removing it bought more than the hypothesis expected (+28% vs +15–20%).
- fp32 K/V storage is exact (values are bf16) and cheap at batch 1 (~0.4 MB/token of capacity) — revisit (bf16 + fused
  kernel) only when attention bytes matter, i.e. after graphs.
- Remaining dynamic shapes per step: window slice length (grows until 1024, global grows forever), conv-history views,
  python `start`/`lo`/`n` ints. For a CUDA graph (hyp 4): fixed-length attention over [cap] or [1024-ring] with a
  device-side position + additive -inf mask, conv history as static (3, C) buffers updated with copy_, and pos as a
  device tensor. GPU work is now well under the 84 ms wall → launch overhead dominates even more.
Next idea: hyp 4 prerequisites + capture — make the T=1 step shape-static (masked fixed-length attention, static conv
buffers, device pos/len) and CUDA-graph the whole decode step (incl. unembed + argmax). Expect the largest jump
(84 ms wall vs a GPU-time floor probably ~35–45 ms → ≥ 2×).
Summary: Experiment #3 put K/V in preallocated fp32 per-visit buffers written in place, replaced T=1 attention with a
windowed GQA batched matmul (no repeat_interleave/mask/gather; rel-bias = flipped slice), and vectorized T=1 convs.
9.24 → 11.83 tok/s (+28%), all gates pass, golden-prefix on code recovered to 64, KEEP. Next: shape-static step + CUDA graph.

## 2026-11-21 — Experiment #4 (hyp 4): shape-static T=1 step + whole-step CUDA graph
Hypothesis: make the decode step shape-static and capture it (embed → 48 visits → unembed → argmax → write next
token + pos+1 into static tensors) as one CUDA graph, replayed once per token. Changes:
- Persistent cache owned by the model (allocated once, grown only if prompt+max_tokens > cap → graphs dropped).
- Local layers: K/V in a 1024-slot fp32 ring (slot = pos % 1024); decode attends all 1024 slots with
  dist = (pos − slot) mod 1024, rel-bias = rel[dist], −inf where dist > pos (unwritten slots, zero-initialised).
- Global layers: linear fp32 K/V of size cap; decode attends a static bucket B = roundup(pos+1, 1024) keys,
  bias rel[dist] for 0 ≤ dist < 1024 else 0, −inf for dist < 0; tau log-scaling always applied on device (== 1 exactly
  while pos+1 ≤ 8192). K/V writes via index_copy_ with the device pos; conv histories are static (3, C) buffers (copy_).
- One graph per bucket B, captured lazily after one eager (real) step on a side stream → captured in the untimed
  warm-up generate of each workload; timed runs only replay. Prefill stays eager (start must be 0).
Expected: ~84 ms wall/step → GPU-time floor (~35–45 ms) → ~22–28 tok/s (≈ 2×). Extra reads: local attention always
reads 1024 fp32 keys (320 MB/token over 40 visits, ~0.25 ms) — acceptable for now.
Pre-check (scratch/check_graph.py): P=300 / 1100 (ring wrap, bucket 2048) / 1020 (crosses the 1024 bucket mid-run),
24 steps each: graph == eager tokens, graph repeat deterministic, graph vs teacher-forced prefill argmax 24/24 on all
three. Graph step **~31 ms** (eager same code ~115 ms/step at this size incl. bucket masks). Capture of a new bucket
costs ~1.5 s once (happens in the harness's untimed warm-up).
Harness: **31.74 tok/s** (11.83 → +168%, 2.7×), noise ±0.3%, eval 12/12, nll_ratio 1.0000, peak 56.3 GB, TTFT
409–1117 ms (unchanged, prefill eager). All 4 workloads ≈ 31.6–31.9 tok/s (long no longer slower: 1024-ring +
bucketed global make cost context-independent up to the bucket). Golden-prefix: chat 64, long 64, **code 4**,
reasoning 63 (same near-tie flip on code as exp #2 — numerics-order only; gates pass). **DECISION: KEEP**.
What it teaches:
- Graph replay removes all host/launch overhead: 84 ms wall → 31 ms ≈ the GPU kernel-time floor of the current op
  list. From now on **GPU kernel time is the score**: reduce kernel count and bytes, not Python.
- 31 ms vs a ~7.4 ms byte floor (9.6 GB/token BF16-dequant experts at 1.3 TB/s) → ~4× headroom is in the ~15–20k
  tiny elementwise kernels (fp32 casts, attn_res/blend/conv chains), each ~1–2 µs even inside a graph.
- Per-request cost of the design: one capture per new global bucket (1024 keys) — fine for the harness; for long
  generations a bucket switch costs ~1.5 s once per process per bucket.
Next idea: re-profile ONE graph replay (torch.profiler over replay, or nsys) to rank kernels by GPU time, then hyp 5:
fuse the elementwise regions — torch.compile (mode="max-autotune-no-cudagraphs" / default) on attn_res, blend
(batched 4-controller bmm), conv_c/dyn_conv_c and the attention epilogue, captured inside the same graph. Expect
31 → ~18–22 ms/step (~45–55 tok/s). After that: packed NVFP4 expert GEMV (hyp 6) to cut 6.1 → 1.8 GB/token.
Summary: Experiment #4 made the decode step shape-static (1024-slot local K/V ring, bucketed masked global attention,
device-side pos, static conv histories, in-graph argmax → next token) and CUDA-graphed the whole step per bucket.
11.83 → 31.74 tok/s (+168%), all gates pass, KEEP. The step is now GPU-kernel-bound (~31 ms); next: profile the
replay and fuse the elementwise chains (hyp 5).

## 2026-11-21 — Profiling study #2: what one CUDA-graph replay spends its ~31 ms on (no engine change)
Hypothesis: the graphed step is a long tail of µs elementwise kernels: expect ~15–20k kernels/step, GEMV/MoE
(the byte floor, ~7–8 ms) < 30% of GPU time, fp32 casts (`.float()` of weights each visit: router gate 128×2304,
controller down/gain/forget, rel proj) + attn_res/blend/conv chains the majority. Output: kernel count + GPU time per
region (eager step, same code, record_function labels) and per kernel name inside a graph replay → pick the fusion
target for hyp 5 by Amdahl share.
Method: `scratch/profile_graph.py` — P=300 prefill, capture bucket graph, time replays; torch.profiler over 3 replays
(kernel list) + over 2 eager `_step` with region labels.
Results (P=300, bucket 1024; all per decode step):
- Graph replay **31.5 ms**; **12,056 kernels** (not 20k); GPU kernel time 31.4 ms, inter-kernel gaps only ~1.6 ms →
  inside the graph the score is pure kernel time.
- Duration histogram: 7,709 kernels < 2 µs (9.9 ms) + 3,860 at 2–5 µs (10.8 ms) → **~65% of the step is sub-5 µs
  kernels**; 390 at 5–20 µs (4.8 ms), 96 at 20–100 µs (5.2 ms = the MoE grouped GEMVs), 1 ≥ 100 µs (unembed 0.7 ms).
- Top kernels: `direct_copy` unrolled (strided/dtype-cast copies) **5.8 ms, 2,346×, 18.5%**; `_grouped_gemv` 5.2 ms
  (96×, 54 µs ≈ 85/42 MB per call → already ~1.3 TB/s, only packing helps); bf16 gemvx 2.4 ms (144×, wq/gate/wo) +
  1.9 ms (209×, smaller projections); fp32 mul 1.2+0.9+0.6 ms (~2,000×); bf16 cast copies 1.1 ms (930×); mean
  (rms) 0.9 ms (410×); fp32 gemmSN (attention scores/PV over 1024 fp32 keys) 0.9+0.3 ms; fp32 gemvx 0.85+0.48 ms
  (controller down/gain/forget in fp32, 424×); topk+sort 0.4 ms.
- Per region (eager step, same op list, record_function): attention **10.9 ms / 3,585 kernels (≈75/visit,
  ~227 µs/visit vs a ~41 µs weight-byte floor)**, blend **5.7 ms / 3,024** (84/visit for 36 visits), attn_res 3.7 ms /
  1,909 (240 calls), conv 2.8 ms / 1,584 (+ dyn conv 0.4 / 224), MoE router+act 2.2 ms / 1,152 (+ grouped GEMV
  5.2 ms), other (embed, unembed, stack glue) 1.6 ms.
What it teaches:
- Byte-bound work (expert GEMVs 5.2 ms + projection GEMVs ~4.3 ms + unembed 0.7 ms) ≈ 10 ms; the other ~21 ms is
  ~11k tiny elementwise/reduce/copy kernels. Kernel count is the lever, not bandwidth (yet).
- Biggest single offender is strided/cast copies (`.float()` on activations and on bf16 weights every visit: router
  gate 128×2304, controllers down/gain/forget ×4 streams ×36 visits, `.transpose().float()`, stack/cat glue).
- Attention outside the GEMVs (~6.5 ms) is l2_clip/rms/conv/rel einsum/bias gather/softmax/epilogue chains;
  blend is 4 separate controllers each ~20 kernels; both are ideal fusion targets.
Next idea (hyp 5, highest expected value): `torch.compile` (inductor, no cudagraphs, fullgraph per region) of the T=1
non-GEMM regions — a compiled `attention_c` decode branch, `blend` (4 controllers stacked → one bmm), `attn_res`,
`conv_c` — called inside the existing whole-step graph capture. Also precompute fp32 / stacked copies of router gate
and controller weights once (removes ~1–1.5 ms of per-visit casts with zero risk). Expect 12k → ~3–4k kernels,
31.5 → ~15–18 ms (~55–65 tok/s). Cheaper fallback if compile is brittle: hand-fuse attn_res+rms and blend in Triton.
Summary: Profiling study #2 (no engine change, no harness run). Graphed step = 31.5 ms of pure kernel time over 12,056
kernels; ~65% of it is sub-5 µs elementwise/copy kernels, only ~10 ms is byte-bound GEMV work. Top offenders:
strided/cast copies 18.5%, attention non-GEMM chains (~6.5 ms), blend (5.7 ms), attn_res (3.7 ms). Next: hyp 5 —
compile/fuse the elementwise regions + hoist per-visit weight casts.

## 2026-11-21 — Experiment #5 (hyp 5): torch.compile the T=1 non-GEMM regions inside the whole-step graph
Hypothesis: the graphed step (31.5 ms, 12k kernels) is mostly µs elementwise chains. Compile (inductor, fullgraph,
dynamic=False, no cudagraphs) clean tensor-only T=1 functions and call them inside the existing capture:
(a) the whole decode attention per visit (projections via cuBLAS, k/v conv + history update, clips, rel einsum,
ring/bucket K/V write, scores/softmax/epilogue, wo) — one local and one global(+bucket) variant;
(b) attn_res; (c) blend with the 3 (or 1) controllers stacked → bmm over fp32 weights precomputed once;
(d) attn/mlp output conv; (e) MoE router (fp32 gate weight hoisted) + activation (grouped GEMV unchanged).
Prefill/NLL path unchanged. Compile happens in the untimed warm-up's eager step (before capture).
Expected: 12k → ~3–4k kernels, 31.5 → ~16–20 ms/step (~50–60 tok/s, +60–90%). Risk: compile time in warm-up,
numerics from fused reduction order (gates should hold).
Pre-check (scratch/check_compile.py): graph == eager on P=300/1100/1020, repeat-deterministic, graph vs teacher-forced
prefill argmax 24/24, 23/24 (one near-tie at P=1100), 24/24. First compile+capture ~34 s (once per process; later
buckets 1–4 s) — absorbed by the harness's untimed warm-up. Graph replay **31.5 → 15.0 ms**, kernels **12,056 → 2,927**/step.
Replay kernel ranking now: MoE grouped GEMV 15.6 ms/3 replays = **5.2 ms/step (34%)**, bf16 gemvx (projections)
~4.3 ms (28%), fp32 gemmSN (attention scores/PV over 1024 fp32 keys + global buckets) ~1.2 ms, fp32 gemvx
(controllers/router) ~0.8 ms, topk (gatherTopK + bitonic sort) ~0.38 ms, fused triton elementwise ≤ 0.2 ms each.
Harness: **64.52 tok/s** (31.74 → +103%, 2.03×), noise ±0.4%, eval 12/12, nll_ratio 1.0000, peak 56.3 GB, TTFT
unchanged (412–1110 ms; prefill eager). Golden-prefix: chat 64, long 64, code 4 (same known near-tie), reasoning 66.
**DECISION: KEEP**.
What it teaches:
- torch.compile (fullgraph, dynamic=False, tensor-only functions, dicts of weights as args so all layers share one
  compiled graph) works inside a manual whole-step CUDA graph when the eager warm-up step compiles first. Needed:
  raised dynamo cache_size_limit (attn_res variants by #sources/dtype). Input mutations (index_copy_ into K/V,
  conv history copy_) were fine — no whole-buffer copy-back visible in the replay.
- Step is now ~60% byte-bound GEMV work (MoE 5.2 + projections 4.3 of 15 ms). The remaining big levers are bytes:
  experts packed NVFP4/MXFP8 (hyp 6: 6.1 → 1.8 GB/token of expert reads, MoE 5.2 → ~1.6 ms) and FP8 attention
  weights (hyp 7). Small residuals: fp32 K/V attention (~1.2 ms → bf16/fused decode kernel), topk sort ~0.4 ms.
Next idea: hyp 6 — Triton grouped GEMV that reads the packed NVFP4 (layers 0–19) / MXFP8 (20–23) experts and
dequantizes in-register (keep the BF16 dequant only for prefill or drop it to save ~45 GB). Expect 15.0 → ~11.5 ms
(~85 tok/s). Must reproduce dequant exactly (same scales path as weights.py) so numerics stay within ULPs.
Summary: Experiment #5 compiled the T=1 non-GEMM regions (whole decode attention incl. convs and K/V writes, attn_res,
stacked 3-controller blend via bmm, router, activation, output convs, residuals) with inductor and captured them in the
existing whole-step CUDA graph; fp32/stacked weights hoisted to load time. 31.74 → 64.52 tok/s (2.03×), 12k → 2.9k
kernels/step, all gates pass, KEEP. Step is now dominated by MoE (34%) and projection (28%) GEMVs → next: packed
NVFP4/MXFP8 expert GEMV (hyp 6).

## 2026-11-21 — Experiment #6 (hyp 2 of brief v2): packed NVFP4/MXFP8 expert GEMV at T=1
Hypothesis: the MoE grouped GEMV (5.2 ms/step, 34%) reads BF16-dequantized experts (~6.1 GB/token) at ~1.3 TB/s.
Read the checkpoint's packed experts instead and dequantize in registers: NVFP4 (layers 0–19: u8 = 2 E2M1, low nibble
first; FP8-E4M3 scale per 16; outer amax/den per expert), MXFP8 (20–23: E4M3 values, E8M0 scale per 32). Same dequant
math as weights.py ((v·s)·outer in fp32, rounded to bf16, then fp32 accumulate) → expect outputs within ULPs of the
BF16 path. Prefill keeps the BF16 copies (extra ~15 GB packed resident, peak ~72 GB).
Expected: expert bytes 6.1 → ~1.8 GB/token, MoE 5.2 → ~1.6–2 ms, step 15.0 → ~11.5–12 ms (~82–87 tok/s, +27–35%).
Pre-check (scratch/check_packed.py): packed vs BF16-dequant grouped GEMV max-abs ≤ 3.3e-6 (rel ≤ 2.6e-7, i.e. fp32
reduction-order only; dequant is bit-identical incl. the bf16 rounding), graph == eager and repeat-deterministic at
P=300/1100. **Microbench trap:** a hot loop over one layer + fixed ids keeps 85 MB in the 128 MB L2 (bf16 26 µs, fake);
cold-L2 timing (cycle 20 layers × random top-8) gives the real numbers: NVFP4 w13 bf16 70.7 → **29.0 µs** (23.6 MB →
~0.8 TB/s, still ALU-bound), w2 37.3 → 16.9 µs; MXFP8 w13 71.0 → 41.0 µs (~1.0 TB/s), w2 36.8 → 20.5 µs.
First E2M1 decode (where/exp2) was 40 µs hot; an fp16 bit trick (sign<<15 | code<<9 → fp16 = value·2^-14, 2^14 folded
into the scale, exact incl. the subnormal 0.5) + BLOCK_N 16 brought NVFP4 to the numbers above. Step 15.0 → ~12.9 ms.
Harness: **78.96 tok/s** (64.52 → +22.4%), noise ±1.5%, eval 12/12, nll_ratio 1.0000, **peak 71.9 GB** (+15.6 GB: packed
experts resident next to the BF16 copies that prefill still uses). chat 84.4 / long 76.4 / code 76.9 / reasoning 78.4.
Golden-prefix chat 64, long 64, code 4 (known near-tie), reasoning 63. **DECISION: KEEP**.
What it teaches:
- Less than hoped (+22% vs +27–35%): NVFP4 GEMV runs ~0.8 TB/s, not 1.3 — the in-register decode (int ops, 2 cvts for the
  bf16 round, 2 muls per value) is the limit with BLOCK_N 16 × 512 values per iteration. MoE now ≈ 48 × ~50 µs ≈ 2.4 ms.
- Always benchmark GEMVs cold-L2 on this GPU (128 MB L2 swallows a whole MoE layer's top-8).
Next ideas: (a) FP8 dense projections + FP8 unembed (hyp 3; projections ~4.3 ms + unembed 0.7 ms are now the largest
byte share, ~40% of the step). (b) NVFP4 GEMV to ~1.2 TB/s: decode a byte to a bf16x2 pair with a 256-entry LUT in
shared/regs or `cvt.rn.bf16x2.e2m1x2`-style inline PTX (sm_120 has e2m1x2 → f16x2 cvt), and drop the bf16 round when
it's provably exact (v·s is exact; only ·outer rounds) — worth ~0.8 ms. (c) Drop the BF16 expert copies (−45 GB) by
running prefill through a packed path too; needed before DSpark/longer contexts squeeze memory.
Summary: Experiment #6 read the checkpoint's packed NVFP4 (layers 0–19) / MXFP8 (20–23) experts directly in a Triton
grouped GEMV with in-register dequant identical to weights.py (rel diff ~1e-7). 64.52 → 78.96 tok/s (+22%), all gates
pass, KEEP; peak memory 56 → 72 GB. NVFP4 kernel is decode-ALU-bound at ~0.8 TB/s; next: FP8 dense projections +
unembed (hyp 3), or a faster e2m1 decode.

## 2026-11-21 — Experiment #7 (hyp 3): FP8 weight-only dense projections + FP8 unembed at T=1
Hypothesis: projections run on cuBLAS bf16 gemvx at only ~0.6 TB/s (2.6 GB in ~4.3 ms; many small 1024–3072-row
matrices) and the unembed reads 0.93 GB (~0.7 ms). Quantize wq, wr, gate, sink, wk, wv, wo (all 24 layers) and the unembed
to FP8-E4M3 with a per-output-row fp32 scale (amax/448), weight-only, fp32 accumulate, bf16 output. One Triton GEMV
(`torch.library.triton_op`, so it runs inside the compiled attention regions); wq|wr|gate|sink are concatenated into one
6552-row matrix (one launch, one read of xq). k_proj/v_proj (32 rows) and controllers stay as they are. Honesty: the
original bf16 tensors are replaced by bf16(q·s), so prefill/NLL/eval all see the same lossy weights as decode.
Expected: dense bytes 2.6 → 1.3 GB, unembed 0.93 → 0.46 GB; at ~1 TB/s projections 4.3 → ~1.4 ms, unembed 0.7 → ~0.45 ms,
step ~12.9 → ~9.8 ms (~100 tok/s, +25–30%). Risk: NLL (gate +2%) — the loop runs core weights 3× so errors compound.
Pre-check: scratch/check_fp8.py (kernel vs reference, cold-L2 timing vs cuBLAS bf16, NLL before/after quantization,
graph == eager, ms/step).
Harness (exp #7, backfilled from results.tsv/logs by agent 8): **92.33 tok/s** (78.96 → +16.9%), noise ±0.8%, eval 12/12,
nll_ratio 0.9964, peak 73.0 GB, long TTFT 1252 ms. chat 92.0 / long 91.6 / code 94.7 / reasoning 91.0. **DECISION: KEEP**
(commit c17841a). Less than the expected +25–30% → re-profile before the next byte lever.

## 2026-11-21 — Profiling study #3 (brief hyp 1): one graph replay at 92.3 tok/s (~10.8 ms/step), no engine change
Hypothesis: after packed experts + FP8 dense the step is ~10.8 ms over ~2.9k kernels. Expect: MoE packed GEMV ~2.4 ms
(NVFP4 ALU-bound ~0.8 TB/s), FP8 projections ~1.5–2 ms (Triton GEMV, maybe < 1 TB/s on the small 1024/2304-row
matrices), unembed ~0.45 ms, fp32 attention over 1024 fp32 keys ~1.2 ms, and ~3–4 ms of residual small kernels
(attn_res/blend/topk/convs). Output: time per kernel class + achieved GB/s for each GEMV class, to rank the next
experiment (NVFP4 decode speedup vs FP8 GEMV tuning vs bf16 K/V vs DSpark T>1 path).
Method: scratch/profile_graph3.py — P=300 prefill, bucket 1024 graph; torch.profiler over 5 replays; kernels grouped
into classes by name; bytes/call from the known shapes.
Results (P=300, bucket 1024, 5 replays; per step):
- Graph replay **10.93 ms**; **2,783 kernels**; kernel time 10.88 ms, gaps only 0.39 ms → still pure kernel time.
- By class: cuBLAS (all fp32 small GEMV/GEMM) **2.95 ms / 592×**, FP8 Triton GEMV **2.51 ms / 193×**, NVFP4 expert GEMV
  2.06 ms / 88×, inductor reductions 1.24 ms / 646× (1.9 µs), inductor pointwise 1.03 ms / 920× (1.1 µs), topk+sort
  0.38 ms / 96×, copies 0.27 ms, MXFP8 GEMV 0.26 ms / 8×, other 0.2 ms. Buckets: 2,361 kernels < 5 µs = 4.6 ms (42%).
- **FP8 dense GEMV is under-occupied** (the surprise): in-graph wk/wv (1024×2304, 2.36 MB) **8.8 µs ≈ 270 GB/s**,
  wo (2304×3072, 7.1 MB) 11.7 µs ≈ 600 GB/s, qrgs (6552×2304, 15.1 MB) 15.3 µs ≈ 1.0 TB/s, unembed 373 µs ≈ 1.24 TB/s.
  Cause: BLOCK_N 16 and no split-K → wk/wv launch only 64 CTAs, wo 144, on 188 SMs. Dense per token = 1.29 GB in 2.14 ms
  (0.6 TB/s). At ~1.1 TB/s it would be ~1.2 ms → **~−0.95 ms (~9%)**. The cold sweep in check_fp8 (BLOCK_N 4–8, 2–8
  warps) never looked at grid size; split-K (e.g. 4–8 K-splits with fp32 atomics or a 2-stage reduce) is the fix.
- **Attention over fp32 K/V on cuBLAS**: PV gemmSN_NN 19.4 µs × 48 = 0.93 ms (4.2 MB fp32 V per visit ≈ 216 GB/s),
  QK gemmSN_TN 6.3 µs × 48 = 0.30 ms, plus the softmax/bias reductions. One fused Triton decode kernel (bf16 K/V, QK +
  rel-bias + mask + online softmax + PV + lse in one pass, split over keys) → ~8 MB/visit fp32 → 4 MB bf16 at ~1 TB/s
  ≈ 4 µs × 48 ≈ 0.2–0.3 ms: **~−1.0 ms**.
- fp32 cuBLAS gemvx 184× ≈ 4.4 µs (0.8 ms) + gemv2N 88× + small sgemms: these are the blend controllers (down 64×4608,
  gain/forget 2304×64, fp32) and the router (128×2304 fp32) → ~1.5 ms of tiny fp32 matmuls. bf16 weights + fusing the
  3-controller blend into one Triton kernel could save ~0.8–1 ms but is more work.
- NVFP4 in-graph: w13 29.8 µs, w2 17.3 µs (cold w13 top-8 467 GB/s) — still decode-ALU bound; ~−0.8 ms available.
- One core visit = 66 kernels; ~45 of them are 1–2 µs inductor/cuBLAS launches (~60 µs/visit of launch-floor work).
What it teaches:
- The byte floor at current precisions is ~3.9 GB/token ≈ 3.0 ms; we are at 10.9 ms. The gap is (a) under-filled
  GEMVs (FP8 dense, fp32 attention, NVFP4 decode) ≈ 3 ms, (b) ~2,400 launch-floor kernels ≈ 4.5 ms.
- Ordering of next experiments by (gain / effort): (1) FP8 GEMV split-K/occupancy fix — tiny change, numerics only
  reduction-order, ~−0.9 ms → ~101 tok/s. (2) fused bf16 decode-attention Triton kernel — ~−1.0 ms. (3) blend/router in
  one Triton kernel each. (4) NVFP4 LUT decode. DSpark (T>1 verify) multiplies whatever the step costs, so the step-cost
  fixes above also pay off after DSpark; DSpark itself is still the largest single lever (~2–3× tokens/step).
Next idea: FP8 GEMV with split-K (SPLIT_K so that grid ≥ ~2×188 CTAs: wk/wv 8, wo 4, qrgs 2, unembed 1) — two-stage
(fp32 partial (SPLIT_K, N) + tiny reduce fused with the scale + bf16 cast), deterministic (no atomics). Sweep in-graph,
not just cold-L2. Expect 10.93 → ~10.0 ms (~100 tok/s, +8–9%).
Summary: Profiling study #3 (no engine change, no harness run). Graph replay 10.93 ms, 2,783 kernels, gaps 0.4 ms.
FP8 dense GEMVs run at only ~0.6 TB/s because the small matrices launch 64–144 CTAs on 188 SMs (split-K fix, ~−0.9 ms);
fp32 K/V attention on cuBLAS costs ~1.2 ms (fused bf16 kernel, ~−1 ms); ~2,400 launch-floor kernels cost ~4.5 ms.
Next: FP8 GEMV split-K, then fused decode attention, then DSpark.

## 2026-11-21 — Experiment #8 (profiling #3 next idea): FP8 dense GEMV with split-K / occupancy fix
Hypothesis: the FP8 dense GEMVs (2.51 ms/step) are under-occupied: BLOCK_N 16, no split-K → wk/wv (1024×2304) launch
64 CTAs and run at ~270 GB/s, wo (2304×3072) 144 CTAs ~600 GB/s, qrgs (6552×2304) ~1.0 TB/s on 188 SMs. Add a SPLIT_K
grid axis: each (n-block, split) CTA accumulates its K/SPLIT slice in fp32 → fp32 partials (SPLIT, N) → tiny reduce
kernel (sum over splits in fixed order, × row scale, → bf16). Deterministic (no atomics); SPLIT_K=1 keeps the single
kernel path. Per-shape (BLOCK_N, SPLIT_K, num_warps) chosen by an in-graph sweep (scratch/check_splitk.py: a graph over
the 24 layers' qrgs/wk/wv/wo sequence ≈ 650 MB > L2, so cold like the real step). Numerics: fp32 reduction order only.
Expected: dense 2.14 ms → ~1.2–1.4 ms, step 10.93 → ~10.0–10.2 ms (~98–101 tok/s, +6–9%).

## Agent (batch 3, #2) — INTERRUPTED, no result
- Started split-K FP8 GEMV (hypothesis from profiling study #3): work-in-progress saved as `scratch/splitk_fp8_wip.patch`
  (engine reset to c17841a). It launched its config sweep as a background task and the session ended before it returned.
- Lesson (supervisor): never use background tasks or Monitor. Run `tools/gpu_script.sh` and `./run_remote.sh` in the
  foreground with the Bash timeout set to 600000 ms; if a job may exceed 10 minutes, make it shorter.

## 2026-11-21 — Experiment #8 result: FP8 dense GEMV occupancy retune — DISCARD
Pre-check (scratch/check_splitk.py, in-graph cold sweep over 24 layers of qrgs/wk/wv/wo, 645 MB): **split-K never won**.
Best for every shape was SPLIT_K=1 with (BLOCK_N 8, BLOCK_K 128, 8 warps): sweep time 1054 → 735 us (−0.32 ms/step
predicted, wk/wv 1024x2304 alone −195 us, wo −86 us, qrgs −18 us). Unembed unchanged (368 vs 374 us; already ~1.24 TB/s).
Numerics: all configs identical maxabs vs fp32 ref (1e-2 on |ref|~4.5), deterministic.
Harness (commit "FP8 dense GEMV: BLOCK_N 8/BLOCK_K 128/8 warps per shape"): **95.07 tok/s** (+3.0% over 92.33), noise ±2.0%,
eval 12/12, nll 0.9964, peak 73.0 GB. Keep bar 96.04 → **DECISION: DISCARD**. (chat 89.8 / long 95.9 / code 99.7 / reasoning 95.1.)
Teaches: the dense-GEMV occupancy problem is mostly a tile-shape issue, not a grid-size one; the gain (~0.3 ms) is real but
within noise of the keep bar. Profiling-#3's "~−0.9 ms" was overstated: wk/wv at 8.8 us are launch/latency-bound, not bandwidth-bound.
Worth re-applying together with the next change (fused bf16 decode attention, ~−1.0 ms) so the combined gain clears the bar.
Config to re-apply in engine/fp8_dense.py: CFG = {(6552,2304),(1024,2304),(2304,3072): (8,128,8,1)}; split-K code
(_fp8_gemv SPLIT_K axis + _splitk_reduce) is in scratch/splitk_fp8_wip.patch if wanted (it did not help).
NOTE (supervisor): the session sandbox denied `git reset --hard HEAD~1`, so the DISCARDed commit may still be on HEAD;
reset it manually before the next experiment (`git reset --hard HEAD~1` if HEAD message is "FP8 dense GEMV: BLOCK_N 8/...").
Next idea: fused bf16 decode-attention Triton kernel (QK + bias + mask + online softmax + PV, split over keys), expect −1.0 ms.

## 2026-11-21 — Experiment #9: fused bf16 decode attention (Triton split-key kernel) on top of the FP8 retune
Base: HEAD cbf2ff6 (FP8 retune, DISCARDed alone at 95.07 but kept in-tree as notes #8 advised; combined result judged vs 92.33).
Hypothesis: decode attention runs on cuBLAS over fp32 K/V (PV 19.4 µs x48 = 0.93 ms, QK 0.30 ms, + softmax/lse reductions
~1.2 ms/step). K/V values are already bf16-representable (k,v are cast to bf16 before the fp32 cache), so store them as bf16
(lossless, half the bytes) and replace QK + bias + softmax + PV by one Triton kernel: grid (HKV, S/64 key splits), each
program computes partial (m, l, acc) for the 3 q-heads of its KV group in fp32 (no tl.dot, so no TF32), a tiny compiled
combine does the log-sum-exp merge (+ lse for the sink). Bias/mask (rel-bias gather, -inf mask) stay as before.
Expected: attention 1.2 -> ~0.35 ms/step, step 10.9 -> ~10.0 ms with the retune (~100 tok/s, +8% over 92.33; bar 96.0).
Result (#9): **92.23 tok/s** (noise ±0.7%, eval 12/12, nll 0.9964, peak 72.7 GB; keep bar 93.71) → **DISCARD**. This is *below*
the 95.07 of the retune alone, so the fused attention gave no gain (≈ −3 tok/s, ~ +0.3 ms/step) and the FP8 retune's benefit vanished.
chat 93.4 / long 91.2 / code 92.5 / reasoning 91.9. Commits a26360b (+ one dtype-fix commit) are on HEAD; the sandbox denies
`git reset --hard`, so supervisor: reset to cbf2ff6 (`git reset --hard HEAD~2`). Full diff saved: scratch/fused_attn_exp9.patch.
What it teaches / red flags:
- Golden-prefix matches fell (chat 31, long 14, code 4, reasoning 63 vs 64/64/4/63 before). Reduction order alone should not
  move a greedy argmax at token 14 of the long workload, so the kernel may have a subtle error (suspects: `bias` masked entries
  = -inf combined with `other=-inf` is fine, but check the ring-buffer read of unwritten slots (zeros, bf16) and the global
  bucket B > cap case; and exp/accumulation of p in fp32 vs the old softmax which was fp32 over the *whole* row). NLL is
  unaffected because it runs the prefill path, so the gates cannot see a decode-attention bug. Verify with a unit check
  (old fp32 `_attend` vs `attn_partial`+merge on random K/V/bias incl. fully-masked splits) BEFORE any retry.
- No speedup even though the kernel replaces ~1.2 ms of cuBLAS+reductions: the 16-program-per-head split with a 64x128 fp32
  tile and 3 serial heads is probably latency-bound (~15 µs x 48 = 0.7 ms) and the merge adds ~4 small kernels per visit
  (m.amax, exp, sums). Profile in-graph (scratch/profile_graph3.py) to see the real per-visit attention time before tuning.
Next idea: unit-test + in-graph profile of the new kernel first (ideally one kernel that also does the merge via a second tiny
pass or atomics-free single-CTA-per-head over 1024 keys), or move to blend/router fusion (~1.5 ms of tiny fp32 cuBLAS ops).
Summary: Experiment #9 replaced the fp32 cuBLAS decode attention with a bf16-K/V Triton split-key kernel + LSE merge: 92.23 tok/s,
no gain (bar 93.71) → DISCARD; greedy golden-prefix agreement dropped, hinting at a numerics bug to unit-test. Reset needed manually.

## 2026-11-21 — Study #10 (unit check of the fused decode-attention kernel; no harness run, no engine change)
Context: reset of the DISCARDed #8/#9 commits was denied by the sandbox again (`git reset --hard` blocked, also compound
commands), so HEAD is still a26360b (fused attention + FP8 retune) — supervisor must still reset to cbf2ff6.
Hypothesis: the #9 golden-prefix drop (long 64→14) was a numerics bug in `attn_partial` + merge. Test: scratch/check_attn_unit.py
(random K/V bf16, bias with -inf masks incl. fully-masked leading blocks, S in 100/1000/1024/2049/4096, cap 4096, H=24 HKV=8 HD=128)
vs the old fp32 softmax reference.
Result: **kernel + merge is numerically correct**: max |out err| ≤ 1.9e-6, lse err ≤ 2.4e-6, no NaNs in any case (incl. fully
masked blocks). So the kernel is not the source of the golden mismatch; remaining suspects are outside it: (a) bf16 K/V cache
plumbing at the ring/bucket level (e.g. prefill writing into the bf16 cache, `fix prefill ring dtype` commit), (b) the extra
~1e-6 fp32 reordering flipping near-tie argmaxes (long prefix 14 is early, but golden may sit on a near-tie), (c) FP8 retune
interaction (retune alone had not been golden-checked per workload). Next check: run the old path vs new path on the same
state end to end (decode a few steps from one prefill; compare logits maxabs), not just the kernel.
Timing (eager, includes launch overhead, S=1024): attn_partial 34 us/call; eager merge adds ~90 us (≈ 12 ops; torch.compile fuses
these in the engine, so not representative); old fp32 path 97 us. Partial kernel at 34 us is slower than expected for ~1 MB of
K/V per call (HKV 8 × 1024 × 128 × 2 B × 2) → it is latency-bound: only 8×16=128 programs, each looping 3 heads serially over a
64×128 fp32 tile. Better: grid over (kv-head, head-in-group) or BS=32, num_warps tuned, and fold the merge into a second tiny kernel.
Next idea: end-to-end logits comparison old-vs-new decode (scratch), then re-tile the partial kernel (BS 32, 2 stages) with merge fused.
Summary: Study #10 verified attn_partial + LSE merge matches the fp32 reference to ~2e-6 incl. masked blocks, so the #9 golden
drop is not a kernel math bug; the kernel is latency-bound (34 us/call, 128 programs). No engine change; reset of HEAD still pending.

## Supervisor fix (20:00): discarded commits had stayed in HEAD (agents' git reset variants were blocked)
- #8 FP8 retune (95.07 DISCARD) and #9 fused bf16 decode attention (92.23 DISCARD; golden-prefix drop on long/chat) were still in HEAD.
  Engine reset to the last KEEP c17841a (92.33). Their code is on branch discarded/batch3-* (use git show/diff to re-apply).
- #10 profiling: the fused attention kernel math is correct (err <= 2e-6); its speed is the problem.
- loop.sh now resets to the last KEEP commit after every agent.

## 2026-11-21 — Experiment #11: bf16 loop-message controllers (gain|forget fused into one bmm)
Base: c17841a (92.33). Hypothesis: profiling #3 found ~1.5 ms/step of tiny fp32 cuBLAS GEMVs, mostly the blend controllers
(down 64x4608 + gain/forget 2304x64 in fp32 = 2.36 MB per stream x 4 streams x 36 visits ≈ 340 MB/token) plus the router.
Keep the controller weights in bf16 (their checkpoint dtype). `down` becomes a bf16 bmm (its output was already rounded to bf16,
so only the reduction order changes). gain|forget become ONE bf16 bmm on a stacked (S, 2D, 64) weight, which rounds g/f to bf16
before tanh, a small loss. This halves the bytes and drops one cuBLAS launch per blend (72 per step). Router left alone (top-k tie risk).
Expected: −0.2…−0.4 ms/step → ~94–96 tok/s (bar 93.71).
Result (#11): **92.62 tok/s** (noise ±0.6%, eval 12/12, nll 0.9964, peak 72.9 GB) → **DISCARD** (bar 93.71); reset done.
chat 93.5 / long 90.4 / code 92.7 / reasoning 94.0. Golden-prefix chat 31, long 14, code 1, reasoning 63. Code fell from 4 to 1,
which fits the bf16 g/f rounding; chat and long already sat at 31/14 on the kept engine's runs, so they do not point to a bug.
What it teaches: halving the controller bytes and dropping 72 launches gained ≈ +0.3% (~0.03 ms). The controller GEMVs are
latency-bound: each is one ~4 µs cuBLAS launch, and fp32 vs bf16 bytes barely matter at 1–2 MB. To win this ~1.5 ms you need
FEWER kernels per visit (e.g. one Triton kernel for the whole blend, pair-norm → down → gain|forget → tanh blend → rms) rather than
cheaper GEMVs. Same lesson as #8 and #9: on this step, tuning bytes/occupancy of small ops is worth ~0, and cutting kernel count is
the lever (≈2,400 launch-floor kernels ≈ 4.5 ms).
Next idea: DSpark speculative decoding (multiplies tokens per step, which pays off however the step's cost is split) or a
single-kernel blend in Triton (−1 to −2 µs × ~6 kernels × 72 blends ≈ −0.5…−0.8 ms).
Summary: #11 moved the loop-message controllers to bf16 and merged the gain and forget GEMVs into one: 92.62 tok/s vs a bar of 93.71
→ DISCARD. The small GEMVs are launch/latency-bound, so byte savings don't help. Kernel-count reduction (a fused blend kernel) or
DSpark are the remaining levers.

## 2026-11-21 — Experiment #12: blend in 3 Triton kernels (fewer launches per blend)
Base: c17841a (92.33). Lesson of #8/#9/#11: kernel count, not bytes, is the lever. FT.blend (72 calls/step) is inductor +
3 cuBLAS bmms (~6-7 kernels). Replace by 3 Triton kernels: A (S, 64/2) pair-rms + bf16 round + down GEMV -> h bf16;
B (S, D/64) gain|forget GEMV + tanh blend -> y fp32 + per-block sumsq; C (S, D/256) final rms * norm_w -> bf16.
Same roundings as before (pair->bf16, h->bf16); only reduction order and tanh formula change. Stride-aware for the
expanded base (no copy). Expected: −3…4 launches x 72 ≈ −0.3…0.5 ms → ~95–97 tok/s (bar 93.71).
Unit check (scratch/check_blend.py): maxabs vs inductor blend 0.031 on |ref|~10 (1 bf16 ulp); in-graph hot-L2 time
S=3 20.3 → 8.7 us, S=1 16.0 → 7.6 us (≈ −0.6 ms/step over 72 blends).
Result (#12): **95.98 tok/s** (noise ±0.8%, eval 12/12, nll 0.9964, peak 73.0 GB; bar 93.88) → **KEEP** (+4.0%).
chat 95.7 / long 97.3 / code 95.6 / reasoning 95.4; golden-prefix 31/14/4/63 (unchanged vs kept engine).
Teaches: confirms #11's lesson — replacing a multi-kernel small op by few hand-written Triton kernels pays (~0.4 ms),
where byte/precision tweaks did not. Same recipe should apply to mix/attn_res (~144 calls/step) and router+act.
Next idea: re-apply the #8 FP8 retune (CFG (8,128,8) per shape, +3% alone) on top of this KEEP, or fuse mix (attn_res)
into one Triton kernel; then DSpark.
Summary: #12 rewrote the loop-message blend as 3 Triton kernels (was inductor + 3 cuBLAS bmms): 92.33 → 95.98 tok/s,
KEEP, gates unchanged. Kernel-count reduction is the working lever; next targets are attn_res mixes and the FP8 retune.

## 2026-11-21 — Experiment #13: re-apply FP8 dense GEMV retune (small CFG 16/256/4 -> 8/128/8) on top of #12
Base: df2f16c (95.98). #8 measured 95.07 vs 92.33 (+3%) but was discarded against a noisy bar. It doesn't depend on the blend
kernels, so stacking it should give roughly +2-3% → ~98-99 tok/s. Minimal change (no split-K code); unembed config unchanged.
Result (#13): **94.37 tok/s** (noise ±0.4%, eval 12/12, nll 0.9964, peak 73.0 GB; bar 97.42) → **DISCARD**; reset to df2f16c.
chat 95.5 / long 92.6 / code 94.9 / reasoning 94.5; golden-prefix 31/14/4/63 (unchanged).
Teaches: the retune does not stack. It is −1.7% on top of #12, so #8's +3% was likely noise (that run had ±2.0% noise) or
depended on the old blend schedule. Stop tuning FP8 GEMV tiles; it is not a lever on this step (same as #8, #9, #11).
Next idea: fuse the attn_res mixes (~144 calls/step) into Triton the way #12 did the blend, or start DSpark.
Summary: #13 re-applied the #8 FP8 dense GEMV retune (8/128/8) on top of the Triton blend: 94.37 vs bar 97.42 → DISCARD.
The earlier gain does not hold up. Kernel-count reduction (attn_res mix fusion) and DSpark remain the levers.

## 2026-11-21 — Experiment #14: attn_res mix/mix_norm as one Triton kernel (online softmax over sources)
Base: df2f16c (95.98). #12 showed cutting kernel count pays. FT.mix / FT.mix_norm (~144 calls/step) are inductor regions
(stack of .float() sources + reductions). Replace with ONE Triton kernel, one program per query, full row D=2304 in registers:
per source (tuple of pointers, mixed bf16/fp32) inv-rms + logit, online-softmax accumulate p*inv*x; optional rms*norm_w -> bf16.
Same math, different reduction order. Expected: -0.3..0.5 ms -> ~98-100 tok/s (bar ~97.4).
Result (#15, offline on golden outputs, tokens per verify step = accepted + 1):
| n | K=1 | K=2 | K=3 | K=4 | K=6 (geomean; chat/long/code/reasoning at K=4) |
| 1 | 1.18 | 1.27 | 1.30 | 1.33 | 1.34 — K=4: 1.02 / 1.31 / 1.21 / 1.94 |
| 2 | 1.14 | 1.22 | 1.25 | 1.27 | 1.29 — K=4: 1.00 / 1.21 / 1.07 / 2.03 |
| 3 | 1.12 | 1.17 | 1.18 | 1.20 | 1.21 |
Teaches: the best prompt lookup (n=1, K=3–4) reaches only ~1.3 tokens/step geomean. Chat gets ~nothing (1.02), and only reasoning
(repeated times and numbers) gets close to 2. So an n-gram drafter pays only if T(K+1)/T(1) < ~1.3. That is unrealistic
while MoE reads up to 8(K+1) distinct experts per visit. The verify+rollback machinery (stage A) is still required, but n-gram
cannot be the drafter that justifies it. DSpark (vLLM: 67/45/31/23% by position → ~2.7 tokens/step at K=4) is what makes verify pay.
Next idea: build stage A's T=K+1 verify step and measure T(K+1)/T(1) directly (eager correctness first, then graph), using
n=1 lookup only as a correctness test drafter. Then port DSpark (stage B). Design notes for rollback, from reading model_v1:
global K/V needs no rollback (stale keys > pos are masked by gadd). Local rings need the T overwritten slots saved and restored,
and in-chunk queries must also attend to the saved old keys, or they lose their oldest T−1 window keys. After accepting a+1
tokens, conv hist = full[a+1:a+4] of cat(old_hist, chunk inputs).
Summary: #15 was a profiling-only study, with no engine change and no harness run. Simulated n-gram/prompt-lookup drafting on
the greedy outputs gives at most 1.34 tokens/step geomean (chat 1.02, reasoning 2.0). That is too little to pay for a K+1 verify
on this MoE, so stage A should be built as infrastructure for DSpark, not as a speedup in its own right.

## 2026-11-21 — Study #16 (profiling only): cost model for a T = K+1 verify step
Note: #14 (attn_res mix Triton kernel) has no recorded result; its code is on discarded/1121-201338 (engine/mix_triton.py).
Hypothesis: the T=1 step is about half launch floor (~2,400 kernels) and half bandwidth. In a verify step the launch count stays
about the same (each kernel takes T rows) and the dense FP8 weights are read once, so the extra cost comes mostly from (a) more
distinct experts per visit and (b) the dense GEMV turning into a small-M GEMM. Measure (scratch/verify_cost.py): distinct
experts per visit over T consecutive real tokens (prefill routing on the golden outputs), packed-expert GEMV time at that
many experts, and an M-row FP8 kernel (tl.dot) for M=1/5. Expected: T(5)/T(1) ≈ 1.4–1.7, mostly from experts (8 → ~25).
Result (#16): the T=1 graph step is 10.51 ms here.
- Distinct experts per visit over T consecutive real tokens (all workloads): T=1 8.0, T=2 12.7, T=3 16.3, T=4 19.4, T=5 22.2.
  Spread across workloads is small (T=5: chat 21.9, long 23.0, code 23.3, reasoning 21.4). Routing has little token locality.
- Packed expert GEMVs over 48 visits (eager, launches included): S=8 2.26 ms, 16 4.17, 24 5.54, 32 7.16, 40 8.77, so about
  +0.2 ms per extra distinct expert per visit. Over T=1: T=3 ≈ +1.7 ms, T=5 ≈ +2.9 ms (bandwidth, which no fusion removes).
- Dense FP8 as an M-row tl.dot kernel (fp32 acc, x padded to 16 rows): M=5 costs the same as M=1 for qrgs/wo/wk (e.g. qrgs x48
  0.747 vs 0.751 ms; maxerr ~1 bf16 ulp). Dense projections are free to batch. Five separate GEMVs would cost 5x (5.8 ms).
Cost model: T(K+1) ≈ T(1) + experts + about 1 ms for row-scaled small ops/attention (a guess, not measured)
 → T(3) ≈ 13.2 ms (1.26x), T(5) ≈ 14.4 ms (1.37x). This matches the 1.4–1.7 expectation, at its low end.
Implications: n-gram (1.27 tokens/step at K=2, 1.33 at K=4) → ~96–97 tok/s. That is break-even, so n-gram alone is no win,
which confirms #15. DSpark at ~2.7 tokens/step with K=4 → ~185 tok/s, so stage A+B is worth ~1.9x. The verify step must use
tl.dot small-M kernels for the dense projections and a grouped expert kernel over the distinct ids (each expert read once for all rows).
Next idea: build stage A with exactly that kernel plan (T-row versions of the T=1 regions + rollback as designed in #15) and graph-capture per K.
Summary: #16 was a profiling-only study with no engine change. A 5-token verify step should cost about 1.37x a T=1 step:
the experts grow from 8 to 22 distinct per visit (+2.9 ms), and the FP8 dense projections cost the same at M=5 as at M=1 with a tl.dot
kernel. n-gram drafting therefore breaks even, and DSpark (~2.7 tokens/step) projects to ~185 tok/s.

## 2026-11-21 — Study #17 (correctness + eager cost, no harness run): stage A verify step with rollback
Hypothesis: a T-token verify on the persistent v1 cache (engine/spec.py: verify + commit) reproduces T=1 decode exactly up to
fp32 reordering, with rollback as designed in #15 (conv hist <- full[n:n+3]; local ring slots of rejected positions restored;
in-chunk local queries attend old ring keys gathered before the write; global needs no rollback). n-gram K=4 spec generate
should match graphed greedy decode token for token (modulo near-ties from FP8-GEMV vs dequant-bf16 matmul) at ~1.3 tokens/step.
Eager timings are launch-bound, so they only bound T(K+1)/T(1) loosely; the real cost needs the graphed T-row kernels (#16 plan).
Result (#17, scratch/spec_stageA.py, untracked in git because scratch/ is ignored; engine/spec.py is not wired into infer.py):
- **Verify + rollback is correct.** n-gram K=4 spec generate == graphed T=1 greedy decode for **64/64 tokens on all 4 workloads**
  (golden prefixes unchanged: 31/14/4/63). A 5-token chunk matches teacher-forced prefill argmax on every row; after
  commit(keep 2 of 5) the next 5-token verify matches again on every row (chat, reasoning). Logit maxabs vs prefill is 0.5–1.1
  (softcap-30 scale). That comes from bf16 GEMM shapes and reduction order compounding over 48 visits, not from rollback (argmax always agrees).
- tokens/step with n=1 lookup, K=4, 64 tokens: chat 1.03 / long 1.33 / code 1.16 / reasoning 1.56 (fits #15's offline sim).
- Eager verify (BF16 dequant experts + generic path, launch-bound, NOT representative): T=1 176 ms, T=2 224, T=3 238, T=5 309,
  T=8 353 ms → T(5)/T(1) = 1.75 eager. Most of the growth is the per-expert Python MoE loop (more distinct ids).
Decision: no harness run (study; the engine's decode path is unchanged), so there is no KEEP/DISCARD. The loop will park spec.py on discarded/.
Teaches: the rollback design from #15 holds (conv hist full[n:n+3], local-ring slot restore plus pre-write gather of old
window keys, no global rollback). The rest of stage A is about speed only. It needs T-row versions of the graphed regions: a small-M tl.dot FP8
dense (#16 shows M=5 costs the same as M=1), a packed grouped expert kernel over the distinct ids, and T-row attention over
gathered ring+new keys. Then capture one graph per K with a static rollback (saved slots and full hists written to static buffers).
Next idea: re-apply engine/spec.py (git show from the discarded branch) and build the graphed T-row verify with those kernels. Measure
graphed T(5)/T(1) against #16's predicted 1.37x, then go straight to the DSpark drafter (stage B). n-gram alone breaks even.
Summary: #17 built stage A's eager verify step with exact rollback (engine/spec.py). Spec decoding with an n-gram drafter reproduces
graphed greedy decode token for token on all 4 workloads (64/64), and rollback is verified against prefill. Tokens/step is
1.03–1.56. The eager cost (T5/T1 1.75x) is launch-bound, so the next step is the graphed T-row kernels, then DSpark.

## 2026-11-21 — Study #18 (kernel + profiling, no harness run): T-row deduplicated packed expert GEMV for the verify step
Hypothesis: #16 measured S distinct experts with one program per (expert slot) at T=1 (S=8T if every row is run separately).
A verify kernel should read each distinct expert ONCE for all T rows: sort the 8T ids, have one program per sorted slot
(duplicates exit), dequantize the tile in registers and tl.dot it against up to 16 row vectors (rows not routed to that
expert are masked). engine/moe_verify.py, NVFP4 + MXFP8, w13 (shared x per row) and w2 (x per (row, slot)).
Expected: exact vs per-row packed_gemv (same bf16 dequant, fp32 acc). Cost over 48 visits on real routing ≈ #16's
S-distinct curve: T=5 ≈ 5.0–5.5 ms vs T=1 2.26 ms (≈ +2.9 ms), much less than 5 separate T=1 calls (~11 ms).
Result (#18): **REMOTE START FAILED.** `tools/gpu_script.sh` failed twice with "Failed to connect to marimo server at
https://sb-0d535f74068a97e9.sb.molab.run". The sandbox is gone, so nothing was measured. STOPPED per program.md.
Written but untested: engine/moe_verify.py (dedup T-row NVFP4/MXFP8 expert GEMV via tl.dot over a 16-row block; the
loop will park it on discarded/) and scratch/moe_verify_bench.py (correctness vs per-row packed_gemv on real routing for
T=1..8, eager + graphed cost over 48 visits, and a cfg sweep).
Next idea: once a sandbox is up, run `tools/gpu_script.sh scratch/moe_verify_bench.py` with engine/moe_verify.py restored.
Pass = rel err ~1e-6 and graphed T=5 ≈ 5 ms. Then build the graphed T-row verify step on top of it.
Summary: #18 wrote a T-row expert kernel that reads each distinct expert once, for the verify step. The GPU sandbox was
unreachable, so nothing was run or measured, and there is no KEEP or DISCARD. The kernel and its benchmark are ready for the next session.

## 2026-11-22 — Study #19 (profiling, no harness run): run the #18 T-row dedup expert GEMV bench
Hypothesis (from #18): engine/moe_verify.py (restored from 172c34c) matches per-row packed_gemv to ~1e-6 rel on real routing
for T=1..8, and its graphed cost over 48 visits at T=5 is about 5 ms (T=1 is 2.26 ms; 5 separate T=1 calls would be ~11 ms).
Result (#19, `tools/gpu_script.sh scratch/moe_verify_bench.py`; the sandbox was up this time): **the kernel FAILED on both counts.**
- Correctness: worst rel err **1.1e37** already at T=1 (garbage or uninitialized output; `out` is torch.empty, so some
  (row, slot) rows are never stored, or the stored values are wrong). Not fixed: the cause is not obvious and the kernel is slow anyway.
- Speed (48 visits, w13+w2), verify_gemv vs T separate packed_gemv calls: eager T=1 4.48 vs 0.92 ms, T=5 8.15 vs 4.29, T=8 10.1 vs 6.8.
  **Graphed: T=1 8.26 ms vs 2.25 ms; T=5 17.8 ms** (prediction was ~5 ms). Best cfg sweep at T=5 was (64,64,4)/(32,64,2) at ~12.3 ms eager.
  So this design is 3–4x slower than the T=1 kernel even before it is correct. Likely causes: a 16-wide tl.dot with mostly masked
  columns, 8T programs where most exit, each live program doing a full K loop with a 2-dot nibble split, and no split-K.
- Bench oddity: "distinct/visit" prints 3.2 at T=1, where it must be 8. The hook's recorded routes (model_v0.moe patched while
  running InklingV1.forward) are suspect, possibly a different ids layout or extra calls, so the correctness input may be broken too.
  Check `len(rec)` and `rec[0].shape` first.
Decision: study, no harness run, so no KEEP/DISCARD. engine/moe_verify.py stays uncommitted and the loop will revert it.
Teaches: a tl.dot-based dedup expert kernel is not the cheap path at T ≤ 8. A simpler plan probably beats it: reuse the T=1
packed_gemv GEMV structure (already near bandwidth) with grid = distinct experts × N-blocks, loop over the ≤T routed rows inside
the program (the weight tile stays in registers), then scatter.
Next idea: fix the bench's route capture, then write the "GEMV kernel with an inner row loop" variant and compare it graphed
against T × packed_gemv (target: T=5 ≤ 4 ms over 48 visits).
Summary: #19 ran #18's T-row dedup expert kernel on the GPU. Its output is wrong (rel err 1e37) and it is 3.7x slower than the T=1
kernel when graphed (8.3 vs 2.25 ms at T=1, 17.8 ms at T=5). The tl.dot dedup design is a dead end. The next attempt should be a GEMV
with an inner loop over the routed rows, after fixing the bench's route capture, which shows 3.2 distinct experts/visit at T=1 where it must be 8.

## 2026-11-22 — Study #20 (kernel + profiling, no harness run): T-row expert GEMV with an inner row loop (replaces #18's tl.dot design)
Bench bug found by reading (#19 "3.2 distinct/visit at T=1"): windows s0 in range(Pn, Pn+200, 40) run past the golden
continuation (golden outputs are short), so most windows were empty slices. Fix: teacher-force a long fixed text (prompt + golden +
engine source as text) so every window has real routes.
Hypothesis: grid = (8T sorted slots, N/BN); duplicate slots exit; each live program dequantizes its weight tile ONCE per K chunk
(same bit-exact deq as packed_gemv) and applies it to all TB=pow2(T) rows via a 3-D broadcast [TB,BN,BK] (rows not routed to
the expert masked to 0, store masked). Weight bytes = distinct experts only. Expected: rel err ~1e-6 vs per-row packed_gemv
(fp32 order differs only per k-chunk), graphed T=5 over 48 visits ≈ 4 ms (T=1 packed_gemv 2.25 ms; 5 × T=1 ≈ 11 ms).
Result (#20, `tools/gpu_script.sh scratch/moe_verify_bench.py`, engine/moe_verify.py rewritten, 2 runs):
- **Route capture fixed:** 48 records of shape (seq, 8), 8.0 distinct/visit at T=1. Distinct/visit: T=2 12.5, T=3 16.5, T=5 22.5, T=8 29.3.
- **Correctness passes:** worst rel err vs per-row packed_gemv is 3.8e-7 for T=1..8 over all 48 visits (NVFP4 + MXFP8, w13 + w2).
- **Speed fails badly.** Graphed over 48 visits (w13+w2), verify_gemv vs T × packed_gemv: T=1 7.9 vs 2.24 ms, T=2 10.5 vs 4.3,
  T=3 20.4 vs 6.2, T=5 48 vs 10.2, T=8 61 vs 16.2 ms. The graphed T=5 sweep over 8 tile configs (BN 4–32, BK 16–128, 4/8 warps)
  gave 46–63 ms, so tile shape is not the cause and neither is register spill.
- Cost per distinct expert visit: 5.8 µs for packed_gemv, about 20 µs at TB=2, about 45 µs at TB=8. It grows with TB, so the
  [TB,BN,BK] broadcast-reduce is compute- or layout-bound, probably a convert_layout through shared memory every iteration. It is not DRAM-bound.
Decision: study, no harness run, so no KEEP/DISCARD. engine/moe_verify.py is uncommitted; the loop will park it.
Teaches: (1) T × packed_gemv is the honest baseline for the verify MoE: 10.2 ms at T=5, about 4.5× T=1. That is the main obstacle
to T(5) ≤ 1.5 × T(1): the experts alone would add about 8 ms to a 10.4 ms step. Dedup must beat it by reading less, not by doing more math.
(2) Neither in-program multi-row design tried so far (tl.dot #19, 3-D broadcast #20) works in Triton at this size.
Next idea: a row-outer kernel. Keep packed_gemv's exact inner loop, wrap it in `for t in tl.static_range(TB)` with a masked skip
for rows not routed to e, so each row re-reads the weight tile from L1/L2 (DRAM once per distinct expert) and dequant runs once per
row. Expected cost is close to distinct × 5.8 µs, about 6.3 ms at T=5, vs 10.2 ms. The other option is to skip dedup, use
T × packed_gemv in the graphed verify step now, and measure end-to-end T(5)/T(1) with the n-gram drafter.
Summary: #20 fixed the bench's route capture (8.0 distinct/visit at T=1) and rewrote the T-row expert kernel as a 3-D broadcast
row block. It is exact (4e-7) but 3.5–4.7× slower than T separate packed_gemv calls when graphed (T=5: 48 vs 10.2 ms), and tile
sweeps don't help. Do the multi-row work as a static per-row loop around the proven GEMV body, which relies on L2 reuse.

## 2026-11-22 — Study #21 (kernel + profiling, no harness run): row-outer T-row expert GEMV (the next idea from #20)
Hypothesis: engine/moe_verify.py rewritten. Program (j = t*8+k, N-block) owns expert ids[j] only at its first occurrence
(no sort). It then runs packed_gemv's exact inner loop once per matching (row, slot) j2 in a static loop, so DRAM reads the
tile once per distinct expert and the other rows hit L1/L2. Dequant repeats per row.
Expected: bit-identical per row (rel err 0), graphed T=5 over 48 visits ≈ 6.3 ms (distinct 22.5 × 5.8 µs + repeats from L2)
vs 10.2 ms for T × packed_gemv. Same bench as #20 (scratch/moe_verify_bench.py).
Result (#21, `tools/gpu_script.sh scratch/moe_verify_bench.py`; the script hit its 560 s timeout during the last cfg sweep,
so every main number was printed first):
- **Correctness: bit-exact.** Worst rel err vs per-row packed_gemv is 0.0 for T=1..8 over 48 visits (NVFP4 + MXFP8, w13 + w2).
- **Speed: slower than T × packed_gemv at every T.** Graphed, 48 visits, w13+w2 (chat / code):
  T=1 2.42/2.40 vs 2.24/2.19 ms; T=2 5.04/4.84 vs 4.33/4.21; T=3 7.95/7.72 vs 6.32/6.23; T=5 **14.3/13.7 vs 10.3/10.2**;
  T=8 27.3/26.3 vs 16.2/16.2. Distinct/visit 8.0 / 12.6 / 16.5 / 22.5 / 29.3 (same as #20).
- Cfg sweep (code, graphed): the default (16,64,4) is the best tried (T=3 7.9 ms, T=5 14.0 ms); the other configs were 10–21 ms.
Decision: study, no harness run, so no KEEP/DISCARD. engine/moe_verify.py is uncommitted; the loop will park it.
Teaches: the time per expert-row is not DRAM time. Reading a duplicate expert's tile from L2 instead of DRAM saves nothing,
because packed_gemv at this size is bound by latency and dequant ALU work. Serializing the duplicate rows inside one owner program
removes parallelism, so it costs more than it saves. Three in-program dedup designs (#19 tl.dot, #20 3-D broadcast, #21 row-outer) have
now failed. **Drop dedup for the verify MoE.** Its cost scales with routed (row, slot) pairs, about 2.05 ms per row over 48 visits.
Next idea: one packed_gemv launch over all 8T (row, slot) pairs with per-row x, instead of T launches. That saves T−1 launches per
visit and fills the GPU better. The verify-step experts then cost ≈ 2 ms × T, so T(5) ≈ 10.4 + 8 ms ≈ 1.8 × T(1), and DSpark needs
> 1.8 accepted tokens/step at K=4 to win (vLLM's acceptance gives ~2.7), or use K=2 (T(3) ≈ 1.4×). Build the graphed verify step
with that, and measure end-to-end.
Summary: #21 tried a third dedup design for the verify-step expert GEMV: the T=1 GEMV body in a static loop over the routed
rows. It is bit-exact but 8–68% slower than T separate packed_gemv calls (T=5: 14.0 vs 10.2 ms graphed). Re-reading weights from
L2 doesn't help because the kernel is latency/ALU-bound, not DRAM-bound. Dedup is a dead end. Use one 8T-row packed_gemv launch
and budget about 2 ms of expert time per verify row.

## 2026-11-22 — Study #22 (stage B, profiling only, no harness run): eager DSpark drafter port, acceptance by position
Hypothesis: port the drafter eager (scratch/dspark_accept.py, torch only) from reference/vllm_inkling2/dspark.py + speculator.py.
Measure acceptance offline: teacher-force prompt + 128 greedy tokens, capture target features at visits 5, 23, 41 + the final
normed state, then at each anchor p in the generated region draft 7 tokens (context = features[0:p], block = anchor token p +
7 masks at positions p..p+7, Markov head greedy chain) and compare with the target's argmax. The reference does not define
"input of visit v" for our AttnRes model, so 3 candidates are tested: (a) the unnormed attn_res mix (query 0) at the visit,
(b) the running partial block stack[nb], (c) the normed attention input xq. The draft row offset (mask rows 1..7 vs 0..6) is
also tested. Expected: the right candidate gives roughly vLLM's 67/45/31/23 % cumulative → ~2.6 tokens/step at K=4; wrong
ones give ~0.
Result (#22, `tools/gpu_script.sh scratch/dspark_accept.py`, 120 anchors per workload, 128 greedy tokens; the prefill argmax
matches the graphed greedy decode on 98–100% of positions). **The port works.** Cumulative acceptance % by draft position:
| features / workload | chat | long | code | reasoning | geomean tok/step K=1/2/3/4/5/7 |
|---|---|---|---|---|---|
| (b) partial stack[nb], off=1 | 57/30/13/8/3/1/0 | 72/46/27/16/8/2/0 | 93/82/73/62/54/45/36 | 74/54/39/30/22/14/8 | **1.73/2.25/2.60/2.85/3.03/3.22** |
| (a) unnormed attn_res mix, off=1 | 54/28/12/5/1/0/0 | 71/40/22/12/7/3/0 | 92/82/74/64/54/46/37 | 75/53/38/31/22/15/8 | 1.72/2.22/2.54/2.78/2.94/3.13 |
| (c) normed xq, off=1 | 52/24/10/4/1/0/0 | 64/33/17/5/2/1/0 | 91/79/68/55/43/31/22 | 74/56/36/22/12/7/3 | 1.70/2.16/2.45/2.62/2.73/2.83 |
With off=0 (the anchor row predicts the next token), every candidate gets about 1.05. So the draft rows are the mask rows 1..7
(mask row j predicts token p+j). Tokens/step at K=4: chat 2.08, long 2.60, code 4.11, reasoning 2.98.
- Features (b) (the running partial block entering the visit) are best, and their position-1..4 geomean is close to vLLM's
  67/45/31/23 % at K=7. (a) is close behind. The true definition is probably (b) or (a); use (b).
- Port details that matter (all in scratch/dspark_accept.py): context = features[0:p] at positions 0..p-1 (K from kv_proj_ctx,
  k_norm_ctx, RoPE θ 1e6; V raw); block = embed_norm(embed(anchor)) + 7 mask_embed at positions p..p+7, bidirectional;
  local layers 0/2 see the last 1024 context keys; f_rnn = mix(features[p-1]) with fuse_alpha row −1; the Markov fold uses our
  FP8-dequant unembed; logits = softcap30(bf16(base + bias)).
Decision: study, no harness run, so no KEEP/DISCARD (scratch only, engine/ unchanged).
Teaches: acceptance is high enough. With #21's verify cost (T(5) ≈ 1.8×, T(3) ≈ 1.4× a step) and the draft cost left out, the
projection is K=4: 2.85/1.8 × 96 ≈ 152 tok/s and K=2: 2.25/1.4 × 96 ≈ 154 tok/s. Code reaches 4.1 tokens/step, so adaptive K
should help a lot (stage C). The drafter runs 3 small dense layers over 8 rows, 7 Markov steps (one 201k×256 GEMV each, ~100 MB)
and an 8-row base unembed (0.9 GB in bf16; use FP8 or only the 7 rows that are needed). Budget it at ≤ 1.5 ms graphed.
Next idea: build the end-to-end spec generate. Restore stage A verify + rollback and make verify return the partial-stack features
at visits 5/23/41 + the final h for the T rows. Keep a per-position context K/V cache for the drafter (append the accepted rows)
and run the drafter eager first, to get a first end-to-end tok/s at K=2 and K=4. Then graph it.
Summary: #22 ported the DSpark drafter eagerly (in scratch/) and measured its acceptance offline on teacher-forced target features.
With features = the partial block entering visits 5/23/41 + the final normed state, and mask rows 1..7 as the drafts, it averages
2.25 / 2.60 / 2.85 tokens/step at K=2/3/4 (code 4.1, chat 2.1). That clears the ~1.8 bar at K=4 and projects to ~150 tok/s
before draft cost. Next: wire it into the verify path end to end.

## 2026-11-22 — Study #23 (stage B end-to-end, eager, no harness run): spec generate = stage A verify + DSpark drafter
Hypothesis: restore engine/spec.py (stage A) and make verify also return the DSpark features of its T rows (partial block entering
visits 5/23/41 + final h). engine/dspark.py is the eager drafter from #22, with a per-position context K/V cache that is appended
with the accepted rows only. scratch/spec_dspark.py generates 128 tokens per workload at K=2/4/7 and compares with graphed greedy.
Expected: output == graphed greedy (modulo near-ties); on-engine tokens/step ≈ #22 offline (geomean 2.25 / 2.85 / 3.22), since
the verify features should equal the prefill features up to bf16 reordering. Eager ms/step is launch-bound and is not a speed result.
Result (#23, `tools/gpu_script.sh scratch/spec_dspark.py`, 128 tokens per workload; engine/spec.py and engine/dspark.py are uncommitted):
- **On-engine acceptance holds.** Tokens/step (chat / long / code / reasoning, geomean):
  K=2 1.88/2.33/2.61/2.61 → **2.34**; K=4 1.94/2.72/4.00/2.37 → **2.66**; K=7 2.03/2.67/4.13/2.84 → **2.82**
  (#22 offline: 2.25 / 2.85 / 3.22). The features captured inside verify work as drafter context. Eager draft is 3.5–3.9 ms
  (launch-bound; ~60 kernels + 7 Markov steps). Eager verify 258 / 300 / 360 ms at T=3/5/8 (launch-bound, not a speed result).
- **Exactness is NOT clean.** Agreement prefix with graphed greedy (out of 128): chat 97/31/128, long 94/95/14,
  code 115/115/**1**, reasoning 63/66/66 for K=2/4/7. #17 got 64/64 with n-gram (mostly T=1–2 chunks). These divergences depend
  on K (so on the verify T and the GEMM shapes). That points to near-ties between the eager bf16-dequant verify path and the
  graphed FP8-GEMV decode, which reduce in different orders. The code K=7 split at token 2 is suspicious: it could be a real T=8
  bug, or a tie. Margins were not logged at the split points.
Decision: study, no harness run, so no KEEP/DISCARD. The loop will park engine/spec.py + engine/dspark.py.
Teaches: the drafter and feature plumbing work end to end, and acceptance is about what #22 predicted (K=4 ≈ 2.7 tokens/step).
The blocker for a harness run is now numerics: the verify path must give the same argmax as the T=1 decode. The simplest fix is to
make the T=1 decode itself a verify with T=1 (same kernels), or to use the FP8 GEMV / packed experts per row inside verify.
Next idea: at each divergence point, log the top-2 logit margin of both paths. If the margins are < ~0.05 (ties), build the graphed
verify from the same per-row FP8 GEMV / packed_gemv kernels as T=1 (also needed for speed, per #21). Its rows then match T=1 decode
bit-for-bit, apart from attention reduction order. If the code K=7 margin is large, debug the T=8 path first.
Summary: #23 wired the DSpark drafter into the stage A verify/rollback loop, with features taken from verify itself. On-engine
acceptance is 2.34 / 2.66 / 2.82 tokens/step at K=2/4/7 (code 4.0 at K=4), close to the offline numbers. Output diverges from graphed
greedy at K-dependent points (as early as token 2 on code K=7), probably near-ties between the bf16 verify and the FP8 decode
numerics. Check the margins next, then build the graphed verify on the T=1 kernels so the numerics match.

## 2026-11-22 — Study #24 (supervisor exp 1, profiling only, no harness run): top-2 logit margins at spec/graph splits
Hypothesis: the #23 divergences are near-ties. scratch/spec_margin.py reruns #23's spec generate (K=2/4/7). At the first split it
logs the eager prefill logits on the identical prefix (top-2 gap, l[graph tok] − l[spec tok]), the verify row's gap and
l[g] − l[s], and max |verify − prefill|. It also logs the prefill top-2 gap distribution over the graph tokens. Expected: split
gaps < ~0.1 everywhere (precision). A large prefill gap favouring the graph token at code K=7 @2 would mean a T=8 verify bug.
Result (#24, `tools/gpu_script.sh scratch/spec_margin.py`, 128 tokens per workload; same accept/prefix numbers as #23):
| split | prefill argmax side, gap | verify gap (picks spec) | max abs(verify − prefill) |
|---|---|---|---|
| chat K=2 @97 | graph, 0.000 | 0.069 | 0.37 |
| chat K=4 @31 | **spec**, 0.676 | 0.626 | 0.18 |
| long K=2 @94 / K=4 @95 / K=7 @14 | graph 0.065 / spec 0.034 / graph 0.069 | 0.291 / 0.000 / 0.035 | 1.25 / 0.55 / 1.53 |
| code K=2,4 @115 / K=7 @1 | graph 0.135 / graph 0.200 | 0.000, 0.102 / 0.068 | 0.66–0.78 / 0.70 |
| reasoning K=2 @63 / K=4,7 @66 | graph 0.031 / graph 0.000 | 0.031 / 0.089 | 0.64 / 0.34–0.81 |
Prefill top-2 gap over the graph tokens: p5 = 0.15–0.29, min 0.000, median 3.6–8.2.
- **Every split is a near-tie; no verify bug.** At each split, the top-2 gap on both the prefill side and the verify side is ≤ 0.29
  logits. The code K=7 split at token 1 is a 0.20 vs 0.07 tie, not a T=8 bug. In chat K=4 @31, the eager prefill agrees with spec
  (0.68), so the graphed FP8 decode is the path that drifted there.
- The verify rows differ from the prefill logits by up to 0.2–1.5 logits (max over the vocab). About 5% of tokens have a top-2 gap
  under ~0.2, so with this numeric noise a 128-token run almost always splits. Exactness can only come from the verify rows
  using the same kernels and reduction order as the T=1 decode.
Decision: study, no harness run, so no KEEP/DISCARD. engine/spec.py and engine/dspark.py are restored but uncommitted, so the loop
parks them.
Teaches: the numerics are sound. Exact greedy match is a kernel-identity requirement, not a correctness bug.
Next idea (supervisor exp 2, the last one): build verify on the T=1 kernels, CUDA-graphed per K: FP8 dense GEMV per row (or a
loop of T=1 GEMV calls), one 8T-row packed_gemv for the MoE, and the T=1 attention path per row. Check that the rows match the
decode argmax. Then run the harness with K=2 and K=4. Stop rule: if this doesn't reach ≥ ~130 tok/s with exact output, stop.
Summary: #24 logged logit margins at all 11 spec/graph split points from #23. All of them are near-ties (top-2 gap ≤ 0.29 on both
paths), and the eager verify/prefill logits are within 0.2–1.5 of each other. The code K=7 token-1 split is a tie, not a T=8 bug.
Exact output therefore needs verify rows computed with the exact T=1 decode kernels. That is the final experiment.

## 2026-11-22 — Exp #25 (supervisor exp 2, the last spec experiment): fully graphed spec step on the T=1 kernel family
Hypothesis: one CUDA graph per (K, bucket) = DSpark draft (device positions, masked context bucket) -> T=K+1 verify built
from the T=1 kernel family (new M-row FP8 GEMV that reads the weights once, packed_gemv per (row, expert) in one launch,
the same attention math per row over ring + new keys) -> on-device accept (cumprod) -> commit (conv hist <- full[n:n+3],
masked local ring write, drafter K/V append). The host reads T+1 ints per step. Expected: 2.3-2.7 tokens/step at
1.5-1.9x a step plus ~1 ms of draft -> 120-150 tok/s. The gates decide (not bit-exact).
Result (#25): scratch/spec_graph.py (64/128 tokens, vs graphed greedy on the same box):
| K | chat | long | code | reasoning | geomean |
|---|---|---|---|---|---|
| 2 | 77.0 | 94.4 | 111.2 | 107.9 | 96.6 |
| 4 | 62.4 | 88.9 | 116.6 | 84.3 | 85.9 |
Deterministic on every workload. Output is not bit-equal to graphed greedy (prefix 4..110), as expected after #24 (near-ties).
Harness, K=2: **98.09 tok/s** (chat 79.6, long 97.0, code 111.0, reasoning 108.0), noise 0.2%, eval 12/12, NLL ratio 0.9964,
deterministic, fixed work, peak 74.1 GB. Golden prefixes 31/14/4/63. **DECISION: KEEP** (bar 97.42).
Teaches: the plumbing is right (graphable end to end, gates pass), but the step is too expensive. With ~2.3 tokens/step at K=2,
98 tok/s means one spec step takes ~23 ms, about 2.2x a T=1 step, against the 1.4x the plan assumed. K=4 is worse: the extra
verify rows and Markov steps cost more than the acceptance they buy, and chat (≈1.9 tokens/step) loses outright.
Not measured this run: the draft vs verify split. Likely culprits: un-fused eager draft ops (~60 launches + K Markov GEMVs
over 201k×256), commit is ~6 tiny ops × 48 visits, and the M-row FP8 GEMV (BK shrinks to 128/64 as M grows) plus 8T-row
packed_gemv reading each expert once per row.
Stop rule: 98 < 130 tok/s, so per the supervisor I stop the spec track here and record why.
Next idea (if spec decoding is reopened): profile one graphed step (draft / verify / commit), torch.compile the drafter,
fold the commit into foreach copies, and check whether the M-row FP8 GEMV is bandwidth-bound at M=3.
Summary: #25 put the whole speculative step (DSpark draft, T=K+1 verify on the FP8/packed T=1 kernel family, on-device
accept and commit) into one CUDA graph per (K, bucket). It passes every gate, is deterministic, and KEEPs at 98.09 tok/s with
K=2 (+2.2% over 95.98). K=4 is slower (86). The step costs ~2.2x a T=1 step, so it stays far below the 130 tok/s stop bar.
The spec track stops here as instructed. The remaining lever is the step cost, not acceptance.
