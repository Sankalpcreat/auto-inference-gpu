# auto-inference-gpu

Autoresearch for **Inkling 2 Mini** (Thinking Machines, 26.4B looped MoE, Apache 2.0) on **one RTX PRO 6000 Blackwell (sm_120, 96 GB)**,
with our own engine (PyTorch + Triton + CUDA; no vLLM/SGLang at runtime). See [DESIGN.md](DESIGN.md) and `knowledge/`.

Status:
- [x] Phase 0: checkpoint format decoded and verified (NVFP4 + MXFP8 experts), bytes-per-token budget (≈ 5.4 GB/token → ~240 tok/s ceiling at batch 1)
- [ ] Phase 1: engine v0 (correct, BF16)
- [ ] Phase 2: frozen harness + golden logits
- [ ] Phase 3/4: engine loop + kernel loop

Sibling project: [auto-inference-mlx](https://github.com/Sankalpcreat/auto-inference-mlx) (Mac, Qwen3.5-0.8B, +23.5%).
