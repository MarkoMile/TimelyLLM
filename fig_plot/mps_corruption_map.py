#!/usr/bin/env python3
"""Which MPS thread percentages compute the wrong answer, and what that looks like.

Two things are plotted, because there are two distinct failure modes and only one
of them is catchable before you spend GPU time on it:

  - the numerics gate (scripts/mps-numcheck.py) catches percentages where cuBLAS
    returns wrong fp16/bf16 GEMM results outright;
  - a percentage can pass that gate and STILL generate wrong plans in a real run,
    which only exact-token comparison against an uncapped reference detects.

Colour here is a STATUS encoding, not series identity, so it uses the reserved
status palette and every state carries a hatch and a label as well as a hue.
"""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
MAP = REPO / "results/mps/gh200-numcheck-map.csv"
OUT = REPO / "results/mps/gh200-corruption-map.png"

# Status palette (reserved; never used for series identity) + chart ink.
GOOD, SERIOUS, CRITICAL = "#0ca30c", "#ec835a", "#d03b3b"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

# Percentages that pass the gate and still generate wrong plans, measured by
# fidelity against the 100% reference in results/mps/gh200-full-manifest.csv.
GATE_PASS_GARBAGE = {30: "37% fidelity", 33: "0% fidelity"}

# Representative output, extracted from the p30/p33 logs against the p100 reference.
EXAMPLES = [
    ("correct (100%)", "?s('scissors')==True{g('scissors')}->False"),
    ("at 30%",         "?_1!=False{g(_1)}"),
    ("correct (100%)", "md(40);?iv('toy')==True{g('toy')}->False"),
    ("at 30%",         "md(0.2);?iv('toy')==True{g('toy')}->False;"),
    ("correct (100%)", "?s('cat')==True{g('cat')};"),
    ("at 33%",         " other other other other other other other other ..."),
]

FP16_FLOOR = 4e-3   # fp16/bf16 GEMM rounding noise; anything above is not precision


def load():
    rows = []
    with open(MAP) as fh:
        for r in csv.DictReader(fh):
            pct, sm = int(r["pct"]), r["sm"]
            err = r["worst_rel_err"]
            err = float("inf") if err == "inf" else (float(err) if err != "NA" else None)
            gate_clean = r["verdict"] == "CLEAN"
            if not gate_clean:
                state = "corrupt"
            elif pct in GATE_PASS_GARBAGE:
                state = "garbage"
            else:
                state = "clean"
            rows.append(dict(pct=pct, sm=int(sm) if sm != "NA" else None,
                             err=err, state=state))
    return rows


def main():
    if not MAP.exists():
        sys.exit(f"no map at {MAP} -- run the dense numcheck sweep first")
    rows = load()
    colour = {"clean": GOOD, "garbage": SERIOUS, "corrupt": CRITICAL}
    hatch  = {"clean": None, "garbage": "..", "corrupt": "///"}

    fig = plt.figure(figsize=(11.5, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 3.6, 2.0], hspace=0.42,
                          left=0.085, right=0.965, top=0.80, bottom=0.06)
    fig.text(0.085, 0.962, "Which MPS thread percentages compute the wrong answer",
             fontsize=14, color=INK, fontweight="bold", va="top")
    fig.text(0.085, 0.925,
             "GH200 · CUDA 12.9 · driver 575.64.03 · Llama-3-8B layer shapes, "
             "enqueued without a device synchronise",
             fontsize=9.5, color=MUTED, va="top")
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2])
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor(SURFACE)

    # -- strip: the map itself -------------------------------------------------
    for r in rows:
        ax0.bar(r["pct"], 1.0, width=0.86, color=colour[r["state"]],
                hatch=hatch[r["state"]], edgecolor=SURFACE, linewidth=0.6)
    ax0.set_ylim(0, 1); ax0.set_yticks([])
    for s in ("top", "right", "left"): ax0.spines[s].set_visible(False)
    ax0.spines["bottom"].set_color(BASELINE)

    # -- magnitude -------------------------------------------------------------
    CEIL = 3.0
    for r in rows:
        if r["err"] is None: continue
        inf = r["err"] == float("inf")
        ax1.plot(r["pct"], CEIL if inf else r["err"],
                 marker="^" if inf else "o", markersize=7.5,
                 color=colour[r["state"]], markeredgecolor=SURFACE,
                 markeredgewidth=1.2, linestyle="none", zorder=3)
    ax1.axhline(FP16_FLOOR, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax1.text(68, FP16_FLOOR * 1.35, "fp16 rounding noise", va="bottom", ha="center",
             fontsize=8.5, color=MUTED)
    ax1.text(4, CEIL * 1.45, "▲  relative error = inf", va="bottom", ha="left",
             fontsize=8.5, color=MUTED)
    ax1.set_yscale("log"); ax1.set_ylim(8e-4, 8.0)
    ax1.set_ylabel("worst relative error", fontsize=10, color=INK2)
    ax1.set_xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", fontsize=10, color=INK2)
    ax1.set_xlim(0, 101)
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for s in ("top", "right"): ax1.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax1.spines[s].set_color(BASELINE)
    ax1.tick_params(colors=MUTED, labelsize=9)
    ax0.tick_params(colors=MUTED, labelsize=9, labelbottom=False)

    for pct, note in GATE_PASS_GARBAGE.items():
        r = next(x for x in rows if x["pct"] == pct)
        if r["err"] is None: continue
        tx, ty, ha = ((20, 0.055, "right") if pct == 30 else (44, 0.30, "left"))
        ax1.annotate(f"{pct}% passes the gate,\n{note} in a real run",
                     xy=(pct, r["err"] * 1.3), xytext=(tx, ty),
                     fontsize=8.6, color=INK2, ha=ha, va="center",
                     arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9,
                                     shrinkA=2, shrinkB=3))

    n_c = sum(r["state"] == "clean" for r in rows)
    n_g = sum(r["state"] == "garbage" for r in rows)
    n_x = sum(r["state"] == "corrupt" for r in rows)
    ax0.legend(handles=[
        Patch(facecolor=GOOD, label=f"computes correctly  ({n_c})"),
        Patch(facecolor=SERIOUS, hatch="..", label=f"passes the gate, wrong plans in practice  ({n_g})"),
        Patch(facecolor=CRITICAL, hatch="///", label=f"wrong GEMM results  ({n_x})")],
        loc="lower left", bbox_to_anchor=(0, 1.20), ncol=3, frameon=False,
        fontsize=9.2, labelcolor=INK2, handlelength=1.6, handleheight=1.0)

    # -- what the wrong answer looks like -------------------------------------
    ax2.axis("off")
    ax2.text(0, 1.0, "What the wrong answer looks like",
             fontsize=10.5, color=INK, fontweight="bold", va="top")
    ax2.text(0, 0.855,
             "30% emits valid MiniSpec with the wrong semantics — md(40) becomes md(0.2), "
             "a drone moving 0.2 units instead of 40.\n"
             "Nothing errors, and 30% lands at a plausible 73 ms right on the latency curve. "
             "33% collapses outright.",
             fontsize=9.0, color=INK2, va="top", linespacing=1.5)
    y = 0.56
    for label, text in EXAMPLES:
        bad = not label.startswith("correct")
        ax2.text(0.0, y, label, fontsize=8.4, family="monospace",
                 color=CRITICAL if bad else MUTED, va="top",
                 fontweight="bold" if bad else "normal")
        ax2.text(0.135, y, text, fontsize=8.4, family="monospace",
                 color=INK if bad else INK2, va="top")
        y -= 0.115
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"clean={n_c}  gate-pass-but-garbage={n_g}  corrupt={n_x}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
