# LinkedIn post (main)

A new 26B model came out on Thursday. Within days, AI agents had written its inference engine and made it 20× faster.

Thinking Machines released Inkling 2 Mini: a looped transformer whose middle 12 layers run three times per token, with 128 experts,
4-bit weights, and 48 separate KV caches. vLLM support was still an open pull request. Docker didn't run on my GPU sandbox.

So I built our own engine for one RTX PRO 6000 and let Claude Code agents optimize it, one experiment at a time:

→ 6.1 tokens/s: plain PyTorch, correct, slow
→ 9.2: the agent wrote a Triton MoE kernel with no CPU↔GPU syncs
→ 31.7: the whole decode step captured in one CUDA graph
→ 64.5: compiled 11,500 tiny kernels down
→ 79.0: reads the 4-bit experts natively, unpacking inside its own kernel
→ 92.3: FP8 weights, quality-gated
→ 98.1: DSpark speculative decoding, draft + verify + commit fully on the GPU
→ 105.1: the speculative commit batched from ~624 kernels into 2
→ 124.0: fused KV projections and residual mixing (confirmed over repeat runs)

The part I care about most is the referee. Speed is timed outside the engine, outputs must match the reference on quality
(12/12 eval, perplexity within 0.4%), and anything not clearly faster gets reverted. Agents can't fake a win.
Half the experiments failed, and each failure is in the log with the reason.

Honest status: one forward pass now costs about the same as vLLM's on this GPU family; vLLM with speculation is still 2–4× ahead
on single-user decode. The remaining gap is the speculative step's overhead, and it's measured.

Repo, charts, every run: https://github.com/Sankalpcreat/auto-inference-gpu
Results page: https://claude.ai/artifact/WXxhSvdg767hn6x79ZU7ii

#inference #LLM #GPU #AIagents #autoresearch #ClaudeCode

---

# Short version (for the video caption / X)

New model Thursday. Our own inference engine Friday. AI agents took it from 6.1 → 124 tok/s (20×) on one RTX PRO 6000,
quality held, every run logged. <link>

# First comment (optional, add yourself)
Chart pack + how the referee works: <link>
