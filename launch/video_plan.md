# 16:9 launch video — plan (fill style from the reference videos)

Length 60–90 s · 1920×1080 · black world, the chart style from `launch/charts/` (bold title, monospace numbers, blue = ours,
brown = vLLM) · real footage from the logs, never mock data.

| t (s) | Shot | On screen | VO (draft) |
|---|---|---|---|
| 0–3 | **Hook**: the speed counter ticking 6.1 → 98.1 over a 26B model's name | "6.1 → 98.1 tok/s" | "A model came out Thursday. By Friday, AI agents had made its engine sixteen times faster." |
| 3–10 | The model: 24 layers, middle 12 looping 3× (animated loop), 48 KV caches | "Inkling 2 Mini · 26.4B · looped MoE" | "It's a looped transformer: the middle layers run three times per token. No engine supported it on our GPU." |
| 10–18 | Engine v0 coming alive: a terminal printing "ppl 3.06", "Paris." | "our own engine · PyTorch + Triton" | "So we wrote our own. Correct, but slow: six tokens a second." |
| 18–30 | **The loop**: brief → agent → GPU → referee → KEEP/DISCARD (diagram), live agent log scrolling | "frozen referee" | "Then agents took over. One experiment each. A referee they can't change decides what stays." |
| 30–50 | **The climb** (chart 01 drawing step by step), each step's label popping as a stamp | 9.2 · 31.7 · 64.5 · 79.0 · 92.3 · 98.1 | "A sync-free MoE kernel. One CUDA graph. Compiled kernels. Native 4-bit experts. FP8. Speculative decoding." |
| 50–58 | Kernel count bar collapsing 22,760 → 2,783 | "22,760 → 2,783 kernels / token" | "Twenty-two thousand GPU kernels per token, down to under three thousand." |
| 58–68 | Honest comparison (chart 02 range bars) | "vLLM still 2–4× ahead" | "vLLM with speculation is still ahead, and we know exactly why." |
| 68–75 | End card: repo link + results page | link | "Every run, every failure, open." |

Needed from the user: reference videos (pacing, typography, transitions, sound).
Assets ready: `launch/charts/*.png|svg`, `logs/agent_*.log` (real agent footage), `results.tsv`, `notes.md` quotes.
