"""Launch charts from the real run data (results.tsv, runs/*.json, notes, Mac project). Black, monospace ticks, thin white axes.
python3 launch/charts.py  ->  launch/charts/*.png (+ .svg)
"""
import csv, datetime as dt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "launch" / "charts"; OUT.mkdir(parents=True, exist_ok=True)

BG, INK, INK2, MUTED = "#000000", "#f2f2f2", "#bdbdbd", "#6e6e6e"
BLUE, BROWN, VIOLET, GREEN = "#5b8def", "#b5683a", "#9b7be8", "#4fa36b"   # validated: dark, all checks pass
SANS, MONO = "Arial", "Menlo"
from matplotlib.font_manager import FontProperties
BOLD = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=21)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.linewidth": 1.0, "font.family": SANS, "font.size": 13, "axes.grid": False,
                     "xtick.major.size": 6, "ytick.major.size": 6, "axes.spines.top": False, "axes.spines.right": False})


def mono_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily(MONO); t.set_fontsize(12)


def title(fig, text, sub=None):
    fig.text(0.04, 0.955, text, fontproperties=BOLD, ha="left", va="top")
    if sub:
        fig.text(0.04, 0.905, sub, fontsize=12.5, color=INK2, ha="left", va="top")


def note(fig, text, y=0.03):
    fig.text(0.04, y, text, fontsize=11.5, color=INK2, ha="left", va="bottom", wrap=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200); fig.savefig(OUT / f"{name}.svg"); plt.close(fig)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


gpu = rows(ROOT / "research" / "results.tsv")
SHORT = ["baseline: plain PyTorch", "sync-free MoE kernel", "static KV + windowed attn", "whole-step CUDA graph",
         "torch.compile regions", "packed NVFP4 experts", "FP8 dense + unembed", "Triton loop-blend", "DSpark spec decoding",
         "batched spec commit", "fused KV proj + residual mix"]

# 1 — the climb (log scale), every run; kept = blue stars on the best-so-far line, discarded = grey rings
fig, ax = plt.subplots(figsize=(11, 7.2)); fig.subplots_adjust(left=0.1, right=0.7, top=0.78, bottom=0.17)
title(fig, f"AI agents made our Inkling 2 Mini engine {max(float(r['score_tps']) for r in gpu if r['status']=='KEEP') / float(gpu[0]['score_tps']):.0f}× faster",
      "Decode tokens/s, 1 user, one RTX PRO 6000 · every benchmark run, measured by a frozen referee")
best, xs_b, ys_b, k = 0, [], [], 0
for i, r in enumerate(gpu):
    s = float(r["score_tps"])
    if r["status"] in ("BASELINE", "KEEP"):
        best = s; ax.scatter(i, s, marker="*", s=260, color=BLUE, zorder=4, edgecolors=BG, linewidths=1.2)
        off = {6: (-10, 26), 7: (12, -18), 8: (-10, 44), 9: (-10, 24), 10: (14, 0)}.get(k, (10, -4))
        lab = SHORT[k] + (f"  {s:.1f}" if k == len(SHORT) - 1 else "")
        ax.annotate(lab, (i, s), xytext=off, textcoords="offset points", fontsize=11, va="center",
                    color=BLUE if k == len(SHORT) - 1 else INK2, ha="right" if k in (6, 8, 9) else "left")
        k += 1
    else:
        ax.scatter(i, s, marker="o", s=70, facecolors="none", edgecolors=MUTED, linewidths=1.6, zorder=3)
    xs_b.append(i); ys_b.append(best)
ax.step(xs_b, ys_b, where="post", color=BLUE, lw=2, zorder=2)
ax.set_yscale("log"); ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100])); ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["5", "10", "20", "50", "100"]); ax.set_ylim(4.5, 190)
ax.set_xticks(range(len(gpu))); ax.set_xticklabels([str(i) for i in range(len(gpu))])
ax.set_xlabel("Benchmark run", labelpad=12); ax.set_ylabel("Decode tokens / s", labelpad=12); mono_ticks(ax)
leg = [Line2D([], [], marker="*", color=BLUE, lw=2, ms=15, label="Kept (beat the best by > max(1.5%, 2×noise), all quality gates pass)"),
       Line2D([], [], marker="o", color=MUTED, lw=0, ms=8, mfc="none", mew=1.6, label="Discarded (not faster enough)")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(-0.01, 1.13), frameon=False, fontsize=11.5, ncol=1)
note(fig, "Quality on every kept run: 12/12 eval answers, perplexity within 0.4% of the reference engine, deterministic, fixed work.")
save(fig, "01_climb")

# 2 — vs vLLM on the same GPU family (range = vLLM prose…counting; dot = midpoint)
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(left=0.3, right=0.95, top=0.74, bottom=0.2)
title(fig, "Where we stand against vLLM", "Single-user decode tokens/s on one RTX PRO 6000 Blackwell")
items = [("Our engine: start (plain PyTorch)", 6.1, 6.1, VIOLET), ("Our engine: best plain decode", 95.98, 95.98, BLUE),
         ("Our engine + DSpark (latest)", 102.1, 139.8, BLUE), ("vLLM + DSpark (community, patched)", 196, 407, BROWN)]
for y, (lab, lo, hi, c) in enumerate(reversed(items)):
    if hi > lo:
        ax.plot([lo, hi], [y, y], color=c, lw=9, solid_capstyle="butt", zorder=2)
    mid = (lo + hi) / 2
    ax.scatter(mid, y, s=90, color=BG, edgecolors=c if hi > lo else c, linewidths=2.4, zorder=3)
    if hi == lo:
        ax.scatter(mid, y, s=90, color=c, zorder=4)
    ax.text(hi + 8, y, f"{lo:g}" if hi == lo else f"{lo:g}–{hi:g}", va="center", fontfamily=MONO, fontsize=12, color=INK2)
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in reversed(items)]); ax.set_xlim(0, 450); ax.set_ylim(-0.6, len(items) - 0.4)
ax.set_xlabel("Decode tokens / s (range = across workloads)", labelpad=12)
for t in ax.get_xticklabels(): t.set_fontfamily(MONO)
note(fig, "vLLM numbers: hyudryu/Inkling-2-Mini-2xRTXPro6000 (Workstation Edition, prose/coding/counting prompts, DSpark K=4).\n"
          "Ours: Server Edition, our 4 workloads, mean of 2 confirmation runs (124.0 overall). Not a same-machine head-to-head yet.")
save(fig, "02_vs_vllm")

# 3 — speed vs hours of autoresearch (like cost curves)
t0 = dt.datetime.strptime(gpu[0]["id"], "%Y%m%d-%H%M%S")
pts, best = [], 0
for r in gpu:
    if r["status"] in ("BASELINE", "KEEP"):
        best = float(r["score_tps"])
        h = (dt.datetime.strptime(r["id"], "%Y%m%d-%H%M%S") - t0).total_seconds() / 3600
        pts.append((h, best))
fig, ax = plt.subplots(figsize=(11, 6.6)); fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.18)
title(fig, "Most of the gain came in the first hour; the last 25% took days", "Best kept decode tokens/s vs wall-clock hours since the baseline")
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2, marker="*", ms=15, mec=BG, mew=1.2)
ax.text(pts[-1][0] + 0.2, pts[-1][1], f"{pts[-1][1]:.0f}", color=BLUE, fontfamily=MONO, va="center")
ax.set_xlabel("Hours since baseline (includes a usage-limit pause and a sandbox restart)", labelpad=12)
ax.set_ylabel("Decode tokens / s", labelpad=12); ax.set_ylim(0, 140); mono_ticks(ax)
note(fig, f"{len(gpu)} benchmark runs + profiling studies; each experiment is one fresh coding-agent session (~$1–1.5 of usage).")
save(fig, "03_hours")

# 4 — kernels per decode step
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(left=0.1, right=0.95, top=0.78, bottom=0.16)
title(fig, "From 22,760 GPU kernels per token to 2,783", "Kernel launches in one decode step (profiled by the agents)")
labs = ["eager PyTorch\n(6.1 tok/s)", "CUDA graph\n(31.7 tok/s)", "compiled + fused\n(92.3 tok/s)"]
vals = [22760, 12056, 2783]
bars = ax.bar(labs, vals, color=[VIOLET, BROWN, BLUE], width=0.62, edgecolor=INK, linewidth=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 400, f"{v:,}", ha="center", fontfamily=MONO, color=INK)
ax.set_ylabel("Kernels per decode step", labelpad=12); ax.set_ylim(0, 25500); mono_ticks(ax)
for t in ax.get_xticklabels(): t.set_fontfamily(SANS); t.set_fontsize(12.5)
save(fig, "04_kernels")

# 5 — DSpark acceptance by draft position (study #22, cumulative %)
acc = {"code": [93, 82, 73, 62, 54, 45, 36], "reasoning": [74, 54, 39, 30, 22, 14, 8],
       "long": [72, 46, 27, 16, 8, 2, 0], "chat": [57, 30, 13, 8, 3, 1, 0]}
fig, ax = plt.subplots(figsize=(11, 6.8)); fig.subplots_adjust(left=0.1, right=0.86, top=0.74, bottom=0.19)
title(fig, "How often the DSpark drafter guesses right", "Cumulative acceptance by draft position, in our engine (greedy)")
for (w, v), c, m in zip(acc.items(), [BLUE, VIOLET, GREEN, BROWN], ["*", "o", "s", "D"]):
    ax.plot(range(1, 8), v, color=c, lw=2, marker=m, ms=11 if m == "*" else 7, label=w)
    if v[-1] > 0:
        ax.text(7.2, v[-1], f"{v[-1]}%", color=c, fontfamily=MONO, va="center", fontsize=12)
ax.set_xticks(range(1, 8)); ax.set_ylim(0, 100); ax.set_xlabel("Draft position", labelpad=12); ax.set_ylabel("Accepted (%)", labelpad=12)
ax.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.16), ncol=4, frameon=False, fontsize=12.5, handletextpad=0.4)
mono_ticks(ax)
note(fig, "2.25 / 2.60 / 2.85 tokens per step at draft length 2 / 3 / 4. Kept at 98.1 tok/s: the verify step still costs ~2.2× a normal step.")
save(fig, "05_dspark")

# 6 — where one token's 10.9 ms went (profile at 92 tok/s)
parts = [("small fp32 GEMVs (attention, controllers, router)", 2.95), ("FP8 dense projections", 2.51),
         ("~1,570 small compiled kernels", 2.27), ("packed NVFP4 experts", 2.06), ("top-k + sort", 0.38)]
fig, ax = plt.subplots(figsize=(11, 6)); fig.subplots_adjust(left=0.42, right=0.93, top=0.76, bottom=0.16)
title(fig, "Where one token's 10.9 ms goes", "GPU time per decode step at 92 tok/s (agent profiling study #3)")
for y, (lab, v) in enumerate(reversed(parts)):
    ax.barh(y, v, color=BLUE, height=0.58, edgecolor=INK, linewidth=0.8)
    ax.text(v + 0.05, y, f"{v:.2f} ms", va="center", fontfamily=MONO, color=INK2, fontsize=12)
ax.set_yticks(range(len(parts))); ax.set_yticklabels([p[0] for p in reversed(parts)]); ax.set_xlim(0, 3.6)
ax.set_xlabel("ms per token", labelpad=10)
for t in ax.get_xticklabels(): t.set_fontfamily(MONO)
save(fig, "06_breakdown")

print("charts:", sorted(p.name for p in OUT.glob("*.png")))
