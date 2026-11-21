# DSpark speculative decoding for Inkling 2 Mini — what we know

## Why it matters
vLLM on this GPU: one step verifies 1 + K=4 draft tokens in ~10.0 ms and averages ~2–4 tokens/step (196–407 tok/s).
We are at ~10.4 ms per **single** token (95.98 tok/s). Matching vLLM needs ~2–3 tokens per step.
Acceptance by draft position (vLLM, K=7): 67 / 45 / 31 / 23 / 18 / 14 / 11 % → K=3–4 captures most of the value.

## Exact greedy verify = output unchanged
Accept draft token j only if it equals the target's argmax at that position; on the first mismatch, take the target's token.
So the drafter can be anything (even wrong) — it only changes speed. Our gates (golden prefix, eval, NLL) still apply.

## The hard part in OUR engine: a T = K+1 decode step with rollback
- 48 visits × (K/V cache + conv history for streams k, v, attn, mlp). A verify step appends K+1 positions to every visit;
  after acceptance of a ≤ K tokens, positions a+1..K must be **rolled back**: K/V write positions reset (ring buffers for
  local layers!), conv histories restored to the state after the last accepted token.
- The AttnRes mixes, messages, MoE are per token → batch them over the K+1 rows (MoE: ≤ 8(K+1) distinct experts per visit).
- Cost of verify vs T=1 is THE number to measure first: weights are read once for all K+1 rows (dense FP8 GEMV → small GEMM),
  experts grow with distinct ids. Target: T(5) ≤ 1.5 × T(1).
- Must be CUDA-graph captured per K (static shapes), like the T=1 step.

## The drafter (reference/vllm_inkling2/dspark.py, ops_dspark.py, test_dspark.py; weights /tmp/inkling/mtp.safetensors, 0.6 GB)
- config `mtp_config`: hidden 2304, 3 layers (local ids 0, 2: sliding window 1024), 24 q / 8 kv heads, head_dim 128,
  intermediate 9216, rope_theta 1e6, block_size 8, markov_rank 256, markov_head_type rnn_hidden,
  target_layer_ids [5, 23, 41, 48] = the inputs of target layer VISITS 5, 23, 41 and the final normed state (visit index, not layer id).
- Mechanism: an anchor (last accepted token) + mask block runs through a small dense transformer that attends to K/V built
  from the target's features at those visits ("DFlare-mixed target-feature K/V"), bidirectional within the block; then a
  recurrent Markov head corrects the draft logits per step (shared LM head, folded to a (vocab, 256) projection).
- Draft quality only affects acceptance; port it carefully but test with acceptance rate, not bit-exactness.

## Staged plan (one stage per experiment, each must pass the harness)
A. **Verify path + rollback + a trivial drafter** (n-gram / prompt lookup over prompt+generated text, exact greedy verify).
   Measure T(K+1)/T(1) and tokens/step. Even n-gram gives wins on code/long workloads if verify is cheap.
B. Port the DSpark drafter (eager first), feed it target hidden states at visits 5, 23, 41 + final; measure acceptance.
C. CUDA-graph the draft + verify; then adaptive K (draft further only while the drafter is confident).
