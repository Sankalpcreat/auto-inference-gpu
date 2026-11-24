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

# 1 — the climb: one message, 4 labeled milestones, linear scale, nothing else
fig, ax = plt.subplots(figsize=(11, 6.6)); fig.subplots_adjust(left=0.08, right=0.86, top=0.76, bottom=0.1)
title(fig, "6.1 → 124 tokens per second", "AI agents tuning our own Inkling 2 Mini engine · one RTX PRO 6000 · quality held on every step")
best, xs_b, ys_b, kept = 0, [], [], []
for i, r in enumerate(gpu):
    sc = float(r["score_tps"])
    if r["status"] in ("BASELINE", "KEEP"):
        best = sc; kept.append((i, sc))
    else:
        ax.scatter(i, sc, marker="o", s=34, facecolors="none", edgecolors="#4a4a4f", linewidths=1.3, zorder=2)
    xs_b.append(i); ys_b.append(best)
cut = kept[-1][0] + 1
ax.fill_between(xs_b[:cut], ys_b[:cut], step="post", color=BLUE, alpha=0.10, lw=0, zorder=1)
ax.step(xs_b[:cut], ys_b[:cut], where="post", color=BLUE, lw=2.6, zorder=3)
for i, sc in kept:
    ax.scatter(i, sc, s=46, color=BLUE, zorder=4, edgecolors=BG, linewidths=1.5)
MILESTONES = {31.74: "one CUDA graph", 64.52: "compiled + fused", 92.33: "4-bit experts + FP8", 124.01: "fused KV + residual mix"}
for i, sc in kept:
    for v, lab in MILESTONES.items():
        if abs(sc - v) < 0.01:
            ax.annotate(lab, (i, sc), xytext=(-12, 10), textcoords="offset points", ha="right", va="bottom", fontsize=12.5, color=INK2)
last_i, last_v = kept[-1]
ax.text(last_i + 0.45, last_v + 2, f"{last_v:.0f}", color=BLUE, fontsize=34, fontweight="bold", va="center", ha="left")
ax.text(0, 6.1 + 5, "6.1", color=INK2, fontsize=13, fontfamily=MONO, va="bottom", ha="center")
ax.set_ylim(0, 140); ax.set_xlim(-0.6, len(gpu) + 0.6)
ax.set_yticks([0, 50, 100]); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
for yv in (50, 100):
    ax.axhline(yv, color="#1c1c20", lw=1, zorder=0)
ax.set_xlabel(f"{len(gpu)} benchmark runs →", labelpad=8, color=INK2, fontsize=12)
mono_ticks(ax)
fig.text(0.86, 0.80, "●  kept     ○  reverted", ha="right", fontsize=12, color=INK2)
save(fig, "01_climb")

# 2 — vs vLLM on the same GPU family (range = vLLM prose…counting; dot = midpoint)
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(left=0.3, right=0.95, top=0.74, bottom=0.2)
title(fig, "Where we stand", "Single-user decode tokens/s, RTX PRO 6000")
items = [("Ours, day 1", 6.1, 6.1, VIOLET), ("Ours, plain", 96, 96, BLUE),
         ("Ours + DSpark", 102, 140, BLUE), ("vLLM + DSpark", 196, 407, BROWN)]
for y, (lab, lo, hi, c) in enumerate(reversed(items)):
    if hi > lo:
        ax.plot([lo, hi], [y, y], color=c, lw=9, solid_capstyle="butt", zorder=2)
    mid = (lo + hi) / 2
    ax.scatter(mid, y, s=90, color=BG, edgecolors=c if hi > lo else c, linewidths=2.4, zorder=3)
    if hi == lo:
        ax.scatter(mid, y, s=90, color=c, zorder=4)
    ax.text(hi + 8, y, f"{lo:g}" if hi == lo else f"{lo:g}–{hi:g}", va="center", fontfamily=MONO, fontsize=12, color=INK2)
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in reversed(items)]); ax.set_xlim(0, 450); ax.set_ylim(-0.6, len(items) - 0.4)
ax.set_xlabel("tokens / s  (bar = range across workloads)", labelpad=12)
for t in ax.get_xticklabels(): t.set_fontfamily(MONO)
note(fig, "vLLM: community run on a Workstation Edition card, different prompts. Not a same-machine test yet.")
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
title(fig, "Most of the gain came in the first hour", "Best kept tokens/s over time")
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2, marker="*", ms=15, mec=BG, mew=1.2)
ax.text(pts[-1][0] + 0.2, pts[-1][1], f"{pts[-1][1]:.0f}", color=BLUE, fontfamily=MONO, va="center")
ax.set_xlabel("Hours since the first run", labelpad=12)
ax.set_ylabel("Decode tokens / s", labelpad=12); ax.set_ylim(0, 140); mono_ticks(ax)
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
