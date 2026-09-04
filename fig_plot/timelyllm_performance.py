#!/usr/bin/env python3
"""How TimelyLLM performs against the GPU compute it is given, and what each
SM-partitioning mechanism can actually deliver.

Three figures:

  1. gh200-latency-by-mechanism.png -- time to first action against share of the
     GPU, for all three mechanisms, over a strip showing where each one can
     legally operate. This is the headline: MPS has a hole through the middle of
     the range, and the other two do not.
  2. gh200-latency-spread.png -- median / p90 / p99 across the compute range,
     from the green-context sweep, which is the only complete one.
  3. gh200-delivered-compute.png -- what a partition of n SMs actually computes.
     Green contexts track the request; libsmctrl's TPC budget does not, which is
     the calibration problem a scheduler has to solve.

Colours are the dataviz reference palette, used unmodified. The validator is a
node script and node is not installed on this machine, so the palette is taken
as documented-and-validated rather than re-checked here; the three categorical
slots used are the three the reference states pass all-pairs in both modes.

Usage:  .venv/bin/python fig_plot/timelyllm_performance.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "plotmps", REPO / "scripts" / "plot-mps-latency.py")
plotmps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plotmps)

OUT = REPO / "results" / "mps"
TOTAL_SM = 132

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"        # blue, orange, aqua
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
RAMP = ("#86b6ef", "#2a78d6", "#104281")            # ordinal: p50 -> p99


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def load(manifest, reference):
    """pct -> (median, p90, p99, sane) for the rtllm arm."""
    ref = plotmps.measure(reference)["plans"]
    pooled = plotmps.collect(str(manifest), ref)["rtllm"]
    out = {}
    for pct, e in sorted(pooled.items()):
        d = e["delays"]
        if not d:
            continue
        sane = e["insane_runs"] == 0
        out[pct] = (plotmps.quantile(d, 0.5) * 1e3,
                    plotmps.quantile(d, 0.9) * 1e3,
                    plotmps.quantile(d, 0.99) * 1e3,
                    sane)
    return out


def skipped_pcts(manifest):
    with open(manifest) as fh:
        return sorted({int(r["pct"]) for r in csv.DictReader(fh)
                       if r.get("load_s") == "SKIPPED_NUMCHECK"})


# ------------------------------------------------------------------ data
REF = REPO / "timelyllm" / "logs" / "gh200-gctx-rtllm-sm132-r1.log"
mps = load(OUT / "gh200-full-manifest.csv", REF)
gctx = load(OUT / "gh200-gctx-manifest.csv", REF)
# Two libsmctrl configurations were measured at the same TPC budget: standalone,
# and as an MPS client with no thread percentage set (the arrangement it would
# use beside cuPHY). They agree to within a millisecond, so they are pooled.
_sm_runs = [load(OUT / m, REF) for m in
            ("gh200-smctrl-manifest.csv", "gh200-mps-smctrl-manifest.csv")
            if (OUT / m).exists()]
smctrl = {}
for pct in {p for r in _sm_runs for p in r}:
    vals = [r[pct] for r in _sm_runs if pct in r]
    smctrl[pct] = tuple(sum(v[i] for v in vals) / len(vals) for i in range(3)) \
        + (all(v[3] for v in vals),)
mps_skipped = skipped_pcts(OUT / "gh200-full-manifest.csv")


# =================================================== 1. latency by mechanism
# The three mechanisms agree wherever they can be compared, so one trend line
# (the green-context sweep, the only complete one) carries the shape and the
# other two ride on it as markers. Runs that produced garbage are deliberately
# NOT plotted on the latency axis -- a corrupted run's "latency" is not a
# latency -- they appear in the coverage strip instead.
fig = plt.figure(figsize=(11.5, 8.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.15], hspace=0.34,
                      left=0.175, right=0.965, top=0.800, bottom=0.135)
ax, strip = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
style(ax)
style(strip)

# The mechanisms agree wherever they overlap, so the curve is the pooled
# measurement -- drawn as chrome, not as a series -- and each mechanism keeps
# its own marker on top. Pooling also fills the 6-24% stretch, where the green
# context has no levels and a straight segment would invent the shape.
pooled_pts = {}
for data in (gctx, mps, smctrl):
    for pct, v in data.items():
        if v[3]:
            pooled_pts.setdefault(pct, []).append(v[0])
curve = sorted((p, sum(vs) / len(vs)) for p, vs in pooled_pts.items())
ax.plot([p for p, _ in curve], [y for _, y in curve], "-",
        color=BASELINE, linewidth=2.4, zorder=2)

for data, colour, marker, size, label in (
        (gctx, S1, "o", 8, "green context"),
        (mps, S2, "s", 7, "MPS thread %"),
        (smctrl, S3, "D", 7.5, "libsmctrl mask")):
    ok = sorted((p, v) for p, v in data.items() if v[3])
    if not ok:
        continue
    ax.plot([p for p, _ in ok], [v[0] for _, v in ok], linestyle="none",
            color=colour, marker=marker, markersize=size,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5, label=label)

ax.set_xlim(0, 104)
ax.set_ylim(50, 125)
ax.set_yticks([50, 60, 70, 80, 90, 100, 110, 120])
ax.set_ylabel("time to first action  (ms, median)", color=INK_2, fontsize=11)
ax.set_xticklabels([])
ax.legend(loc="upper right", frameon=False, fontsize=10.5, labelcolor=INK_2,
          handletextpad=0.6, borderaxespad=0.8)

ax.annotate("62 ms", xy=(100, gctx[100][0]), xytext=(9, -1),
            textcoords="offset points", color=INK_2, fontsize=10.5,
            fontweight="bold", va="center")
ax.annotate("110 ms", xy=(6, gctx[6][0]), xytext=(10, 3),
            textcoords="offset points", color=INK_2, fontsize=10.5,
            fontweight="bold", va="bottom")
ax.annotate("flat from 100% down to 24% of the GPU",
            xy=(64, 68.5), ha="center", fontsize=10, color=MUTED)

# ---- the strip: where each mechanism can legally operate
rows = [("green context", sorted(gctx), [], S1),
        ("MPS thread %", [p for p, v in mps.items() if v[3]],
         sorted(set(mps_skipped) | {p for p, v in mps.items() if not v[3]}), S2),
        ("libsmctrl mask", sorted(smctrl), [], S3)]
for i, (label, ok, dead, colour) in enumerate(rows):
    y = len(rows) - 1 - i
    strip.scatter(ok, [y] * len(ok), s=95, marker="o", color=GOOD,
                  edgecolor=SURFACE, linewidth=1.5, zorder=3)
    strip.scatter(dead, [y] * len(dead), s=100, marker="X", color=CRITICAL,
                  edgecolor=SURFACE, linewidth=1.0, zorder=3)
    strip.text(-3, y, label, ha="right", va="center", fontsize=10.5,
               color=colour, fontweight="bold", clip_on=False)
strip.set_ylim(-0.75, len(rows) - 0.25)
strip.set_yticks([])
strip.set_xlim(0, 104)
strip.set_xticks([0, 20, 40, 60, 80, 100])
strip.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
strip.set_xlabel("share of the GPU's 132 SMs the engine was given",
                 color=INK_2, fontsize=11)
strip.grid(True, axis="x", color=GRID, linewidth=0.8)
strip.spines["left"].set_visible(False)
strip.annotate("MPS is unusable from 28% to 89%", xy=(58, -0.62),
               ha="center", fontsize=9.5, color=CRITICAL)
strip.legend(handles=[
    Line2D([], [], marker="o", color=GOOD, linestyle="", markersize=8,
           markeredgecolor=SURFACE, label="ran, output correct"),
    Line2D([], [], marker="X", color=CRITICAL, linestyle="", markersize=9,
           markeredgecolor=SURFACE, label="wrong output, or refused to run"),
], loc="upper center", bbox_to_anchor=(0.5, 1.30), frameon=False,
    fontsize=9.5, labelcolor=INK_2, ncol=2, handletextpad=0.4,
    columnspacing=2.2)

fig.text(0.175, 0.945, "TimelyLLM only needs a quarter of the GPU",
         fontsize=17, color=INK, fontweight="bold")
fig.text(0.175, 0.885,
         "Time to first action is unchanged from 100% of the SMs down to 24%, and doubles only at 6%. "
         "All three\npartitioning mechanisms agree where they can be compared - but MPS cannot be used "
         "over most of the range.",
         fontsize=10.5, color=INK_2, linespacing=1.55, va="top")
fig.text(0.175, 0.022,
         "GH200 480GB, Llama-3-8B, 42 agents, 1200 s trace. Every plotted point served 100% of the trace with "
         "100% output fidelity.\nThe libsmctrl point pools two runs, standalone and as an MPS client, "
         "which agree to within 1 ms.",
         fontsize=9, color=MUTED)
fig.savefig(OUT / "gh200-latency-by-mechanism.png", dpi=160, facecolor=SURFACE)
plt.close(fig)


# ======================================================= 2. latency spread
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.90, top=0.775, bottom=0.13)
style(ax)
pts = sorted((p, v) for p, v in gctx.items() if v[3])
xs = [p for p, _ in pts]
for idx, (name, colour) in enumerate(zip(("median", "p90", "p99"), RAMP)):
    ys = [v[idx] for _, v in pts]
    ax.plot(xs, ys, "-", color=colour, linewidth=2.0, marker="o",
            markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    ax.annotate(name, xy=(xs[-1], ys[-1]), xytext=(8, 0),
                textcoords="offset points", color=colour, fontsize=10.5,
                fontweight="bold", va="center")
ax.set_xlim(0, 108)
ax.set_ylim(0, 210)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
ax.set_ylabel("time to first action  (ms)", color=INK_2, fontsize=11)
ax.set_xlabel("share of the GPU's 132 SMs  (green-context partition)",
              color=INK_2, fontsize=11)
ax.legend(handles=[Line2D([], [], color=c, linewidth=2.2, label=n)
                   for n, c in zip(("median", "p90", "p99"), RAMP)],
          loc="lower left", frameon=False, fontsize=10, labelcolor=INK_2,
          handletextpad=0.6, borderaxespad=0.9)
fig.text(0.09, 0.945, "Three quarters of the GPU can go before latency moves",
         fontsize=17, color=INK, fontweight="bold")
fig.text(0.09, 0.885,
         "Median, p90 and p99 are all unchanged from 100% of the SMs down to 24%. "
         "Only at 6% does the distribution\nlift, and it lifts as a block - "
         "the spread between median and p99 barely changes.",
         fontsize=10.5, color=INK_2, linespacing=1.55, va="top")
fig.savefig(OUT / "gh200-latency-spread.png", dpi=160, facecolor=SURFACE)
plt.close(fig)


# =================================================== 3. delivered compute
gc_tf = {8: 56.1, 16: 110.1, 18: 164.8, 30: 212.1, 36: 256.4, 44: 325.2,
         62: 380.8, 68: 435.6, 88: 491.9, 116: 548.3, 118: 550.6, 132: 814.9}
sm_tf = {}
with open(OUT / "gh200-smctrl-numcheck-map.csv") as fh:
    for r in csv.DictReader(fh):
        sm_tf[int(r["sm"])] = float(r["tflops"])

fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.88, top=0.765, bottom=0.13)
style(ax)
ideal = 814.9 / 132
ax.plot([0, 132], [0, 132 * ideal], "--", color=MUTED, linewidth=1.4, zorder=2)
for data, colour, label, marker in ((gc_tf, S1, "green context", "o"),
                                    (sm_tf, S3, "libsmctrl", "D")):
    xs = sorted(data)
    ys = [data[x] for x in xs]
    ax.plot(xs, ys, "-", color=colour, linewidth=2.0, marker=marker,
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(8, 0),
                textcoords="offset points", color=colour, fontsize=10.5,
                fontweight="bold", va="center")
ax.set_xlim(0, 138)
ax.set_ylim(0, 880)
ax.set_xlabel("SMs the partition was asked for", color=INK_2, fontsize=11)
ax.set_ylabel("delivered  (TFLOP/s, bf16 4096³ GEMM)", color=INK_2, fontsize=11)
ax.annotate("five of these switches\nbuy nothing at all", xy=(13, 56),
            xytext=(50, 105), textcoords="data", fontsize=9.5, color=S3,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=S3, lw=1.3, shrinkA=6,
                            shrinkB=5, connectionstyle="arc3,rad=0.18"))
ax.legend(handles=[Line2D([], [], color=S1, linewidth=2.2, marker="o",
                          markersize=6, markeredgecolor=SURFACE,
                          label="green context"),
                   Line2D([], [], color=S3, linewidth=2.2, marker="D",
                          markersize=5, markeredgecolor=SURFACE,
                          label="libsmctrl"),
                   Line2D([], [], color=MUTED, linewidth=1.4, linestyle="--",
                          label="perfectly linear scaling")],
          loc="upper left", frameon=False, fontsize=10, labelcolor=INK_2,
          handletextpad=0.7, borderaxespad=1.0)
fig.text(0.09, 0.945, "A partition size is not a compute budget",
         fontsize=17, color=INK, fontweight="bold")
fig.text(0.09, 0.885,
         "Green contexts track the request closely. libsmctrl's TPC budget does not - its ladder is a staircase "
         "with\nflat treads, because mask bits do not map to the SMs that exist. A scheduler has to calibrate it.",
         fontsize=10.5, color=INK_2, linespacing=1.55, va="top")
fig.savefig(OUT / "gh200-delivered-compute.png", dpi=160, facecolor=SURFACE)
plt.close(fig)

print("wrote:")
for n in ("gh200-latency-by-mechanism.png", "gh200-latency-spread.png",
          "gh200-delivered-compute.png"):
    print("  ", OUT / n)
