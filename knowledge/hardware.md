# RTX PRO 6000 Blackwell Server Edition (MoLab) — measured in our lab 01

| Item | Value |
|---|---|
| Arch | Blackwell **sm_120** (consumer/workstation family, like DGX Spark sm_121 — NOT B200 sm_100) |
| SMs / L2 | 188 SMs, **128 MB L2** |
| Memory | 96 GB GDDR7, **~1.3 TB/s measured** (1.8 spec) |
| Tensor cores | BF16 ~407, FP8 ~777, **FP4 ~1,084 TFLOPS** (measured) |
| Launch | eager kernel launch ~22 µs vs **~1.1 µs inside a CUDA graph** → small-batch decode MUST be graph-captured |
| sm_120 limits | No TMEM / tcgen05 (those are sm_100), no 2-CTA MMA; uses `mma.sync` + block-scaled FP4/FP8 MMA. Many B200 kernels (FlashAttention-4, trtllm-gen MoE) don't run. vLLM's Inkling PR falls back to Triton on sm_120 |
| Software | torch 2.11 cu130, Triton 3.6, nvcc 13.0 pinned (see ~/Desktop/Inference/00_setup/setup_molab.sh) |
| Sandbox | ephemeral /tmp, long tool calls drop → run jobs with Popen + poll; save results after every step |
