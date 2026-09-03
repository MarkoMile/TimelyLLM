#!/usr/bin/env python3
"""Plot TimelyLLM's latency against the GPU compute it was given.

Reads the manifest that mps-sweep.sh writes, measures every run's logs, and
draws latency as a function of MPS active-thread-percentage -- the fraction of
the GPU's SMs the engine was allowed to use.

The headline metric is time to first action: from a task being submitted to the
drone starting to move. That is what TimelyLLM's segmented generation exists to
shorten, so it is the number that should degrade as compute is taken away.

The second panel is not decoration. The workload is an open-loop trace: requests
arrive on a fixed wall clock whether or not the engine can keep up. Under enough
starvation, tasks are dropped rather than served late, which makes the latency of
the survivors look better than the system deserves. Latency is only readable
alongside how much of the trace actually got served.

Needs matplotlib, so run it with the project interpreter rather than the
system one:

    .venv/bin/python scripts/plot-mps-latency.py \\
        --manifest results/mps/sweep1-manifest.csv
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT = re.compile(r"^Output for task (\d+): (.*), time: ([\d.]+)$")
ADD = re.compile(r"^Added task (\d+), time: ([\d.]+)$")
EXE = re.compile(r"^Start executing task (\d+) for agent \d+ on time ([\d.]+) with plan")

# Categorical slots 1 and 2 of the validated default palette (light mode).
BLUE, ORANGE = "#2a78d6", "#eb6834"

# A healthy segment is a MiniSpec statement, ~8 characters. Garbage runs to the
# 200-token cap, 400+ characters. Anything between is not something we have
# seen, so the threshold sits well clear of both.
SANE_SEGMENT_CHARS = 150

# Below this share of plans matching the reference, a run is corrupt. Healthy
# runs land at 99-100%: a few plans diverge at the tail because segmentation is
# timing-dependent. A corrupted run collapses to ~0%, so the floor is nowhere
# near either population.
FIDELITY_FLOOR = 0.90
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def measure(path):
    """Per-run stats: first-action delays, segment and task counts."""
    segments, submitted, started = defaultdict(list), {}, {}
    try:
        fh = open(path, errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            line = line.rstrip("\n")
            m = OUT.match(line)
            if m:
                segments[m.group(1)].append(m.group(2))
                continue
            m = ADD.match(line)
            if m:
                submitted.setdefault(m.group(1), float(m.group(2)))
                continue
            m = EXE.match(line)
            if m:
                started.setdefault(m.group(1), float(m.group(2)))
    if not segments:
        return None
    delays = sorted(started[t] - submitted[t] for t in submitted if t in started)

    # Is this run's output real plans, or token garbage? Under a corrupting MPS
    # SM cap the engine still runs and still logs timings -- it just generates
    # nonsense, which never hits a MiniSpec statement boundary and so runs to the
    # 200-token cap. Segment length separates the two cleanly: a healthy run's
    # median segment is ~8 characters ("tc(30);"), a corrupted one's is 400-1200.
    body = [t for v in segments.values() for t in v if t.strip()]
    lens = sorted(len(t) for t in body)
    median_len = lens[len(lens) // 2] if lens else 0
    non_ascii = sum(1 for t in body if any(ord(c) > 127 for c in t))
    sane = bool(lens) and median_len <= SANE_SEGMENT_CHARS and \
        non_ascii <= 0.02 * len(body)

    return {
        "plans": {t: "".join(v) for t, v in segments.items()},
        "delays": delays,
        "submitted": len(submitted),
        "acted": len(delays),
        "tasks": len(segments),
        "segments": sum(len(v) for v in segments.values()),
        "median_len": median_len,
        "sane": sane,
    }


def fidelity(stats, reference):
    """Fraction of shared tasks whose whole plan matches the reference run.

    Decoding is greedy, so a run computing correctly must produce exactly the
    same tokens as the uncapped run for the same task. Segmentation may differ
    -- the stop rule fires on a clock, so a slower engine cuts elsewhere -- but
    the concatenation cannot. This is exact, unlike the length heuristic, and it
    is the same standard compare-arms.py holds the port to.

    Returns (matched, shared) or None if there is nothing to compare.
    """
    if reference is None:
        return None
    shared = set(stats["plans"]) & set(reference)
    if not shared:
        return None
    # A plan cut short counts as matching. Being truncated is a legitimate
    # timing effect -- a slower engine misses a deadline and the task is retired
    # mid-plan -- whereas corruption makes the tokens themselves differ. Only
    # divergence in content is evidence the GPU computed the wrong answer.
    ok = sum(reference[t].startswith(stats["plans"][t]) or
             stats["plans"][t] == reference[t] for t in shared)
    return ok, len(shared)


def quantile(xs, q):
    if not xs:
        return None
    return xs[min(int(len(xs) * q), len(xs) - 1)]


def collect(manifest, reference=None):
    """pct -> pooled stats, per arm. Repeats of the same point are pooled."""
    pooled = defaultdict(lambda: defaultdict(
        lambda: {"delays": [], "submitted": 0, "acted": 0, "segments": 0,
                 "runs": 0, "loads": [], "shared": False,
                 "insane_runs": 0, "median_len": 0,
                 "fid_ok": 0, "fid_n": 0}))
    with open(manifest) as fh:
        for row in csv.DictReader(fh):
            if row.get("load_s") == "SKIPPED_NUMCHECK":
                print(f"  {row['arm']} {row['pct']}%: skipped -- the GPU fails "
                      f"the numerics check at that thread percentage",
                      file=sys.stderr)
                continue
            stats = measure(row["log"])
            if stats is None:
                print(f"  no usable log for {row['arm']} {row['pct']}% "
                      f"rep {row['repeat']}: {row['log']}", file=sys.stderr)
                continue
            e = pooled[row["arm"]][int(row["pct"])]
            e["delays"] += stats["delays"]
            e["submitted"] += stats["submitted"]
            e["acted"] += stats["acted"]
            e["segments"] += stats["segments"]
            e["runs"] += 1
            e["median_len"] = max(e["median_len"], stats["median_len"])
            if not stats["sane"]:
                e["insane_runs"] += 1
            fid = fidelity(stats, reference)
            if fid:
                e["fid_ok"] += fid[0]
                e["fid_n"] += fid[1]
                # Divergence from the reference is corruption that the length
                # heuristic can miss -- 66% generated short, ASCII, plausible
                # nonsense that only this check catches.
                if fid[0] < FIDELITY_FLOOR * fid[1]:
                    e["insane_runs"] += 1
            if row.get("load_s") not in (None, "", "NA"):
                e["loads"].append(float(row["load_s"]))
            if (row.get("others_before") or row.get("others_after") or "").strip():
                e["shared"] = True
    for arm in pooled:
        for pct, e in pooled[arm].items():
            e["delays"].sort()
    return pooled


def table(pooled):
    for arm, points in pooled.items():
        print(f"\n  arm: {arm}")
        print(f"  {'thread %':>9}{'runs':>6}{'submitted':>11}{'acted':>8}"
              f"{'served':>9}{'segs':>7}{'median':>10}{'p90':>10}{'p99':>10}"
              f"{'load':>9}{'fidelity':>10}  output")
        print("  " + "-" * 108)
        for pct in sorted(points):
            e = points[pct]
            d = e["delays"]
            served = f"{100 * e['acted'] / e['submitted']:.0f}%" if e["submitted"] else "-"
            load = f"{statistics.mean(e['loads']):.0f}s" if e["loads"] else "-"
            med = f"{statistics.median(d) * 1000:.0f} ms" if d else "-"
            p90 = f"{quantile(d, 0.90) * 1000:.0f} ms" if d else "-"
            p99 = f"{quantile(d, 0.99) * 1000:.0f} ms" if d else "-"
            out = "plans" if not e["insane_runs"] else \
                f"GARBAGE (median segment {e['median_len']} chars)"
            fid = f"{100 * e['fid_ok'] / e['fid_n']:.0f}%" if e["fid_n"] else "-"
            flag = "  <- shared GPU" if e["shared"] else ""
            print(f"  {pct:>8}%{e['runs']:>6}{e['submitted']:>11}{e['acted']:>8}"
                  f"{served:>9}{e['segments']:>7}{med:>10}{p90:>10}{p99:>10}"
                  f"{load:>9}{fid:>10}  {out}{flag}")
        if any(e["fid_n"] for e in points.values()):
            print(f"\n  'fidelity' is the share of tasks whose plan matches the "
                  f"reference run.\n"
                  f"  Decoding is greedy, so a correct run lands at 99-100%: the only\n"
                  f"  legitimate difference is a plan cut short by a missed deadline,\n"
                  f"  which is counted as a match. Below {FIDELITY_FLOOR:.0%} the GPU computed a\n"
                  f"  different -- wrong -- answer, and the row is rejected.")
        if any(e["insane_runs"] for e in points.values()):
            print("\n  GARBAGE rows are not measurements: the GPU computed the wrong\n"
                  "  answer at that thread percentage, so the engine emitted nonsense\n"
                  "  tokens. Their latencies are excluded from the plot.")


def plot(pooled, arm, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = pooled[arm]
    # A corrupted run still has timings; they just do not mean anything.
    pcts = [p for p in sorted(points) if not points[p]["insane_runs"]]
    if not pcts:
        print(f"  no valid points for {arm}; nothing to plot", file=sys.stderr)
        return
    med = [statistics.median(points[p]["delays"]) * 1000 if points[p]["delays"] else None
           for p in pcts]
    p90 = [quantile(points[p]["delays"], 0.90) * 1000 if points[p]["delays"] else None
           for p in pcts]
    served = [100 * points[p]["acted"] / points[p]["submitted"] if points[p]["submitted"]
              else 0 for p in pcts]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 5.8), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.14})
    fig.patch.set_facecolor("#fcfcfb")

    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.plot(pcts, p90, color=ORANGE, linewidth=2, marker="o", markersize=6,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="p90", zorder=3)
    ax.plot(pcts, med, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="median", zorder=4)

    # Log scale whenever the curve spans more than a decade, which it does as
    # soon as the engine is starved enough to fall behind the trace.
    finite = [v for v in (med + p90) if v]
    if finite and max(finite) / min(finite) > 12:
        ax.set_yscale("log")

    # Log x. The usable percentages bunch up at the low end -- which is also
    # where the curve actually moves -- so a linear axis crushes them together
    # and spreads the flat region out over most of the width.
    for a in (ax, ax2):
        a.set_xscale("log")

    ax.set_ylabel("time to first action (ms)", color=INK, fontsize=10)
    ax.set_title(f"TimelyLLM latency vs GPU compute{title_suffix}",
                 color=INK, fontsize=12, loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=MUTED)
    leg.set_zorder(5)

    # Direct labels on the endpoints, so identity survives without the legend.
    # Offset away from the line: down-left at the starved end where the curve
    # rises steeply, up-right at the flat end.
    for series, color in ((med, BLUE), (p90, ORANGE)):
        if series and series[-1]:
            ax.annotate(f"{series[-1]:.0f} ms", (pcts[-1], series[-1]),
                        textcoords="offset points", xytext=(-4, 10),
                        color=color, fontsize=9, ha="right", weight="bold")
        if len(pcts) > 1 and series and series[0]:
            ax.annotate(f"{series[0]:.0f} ms", (pcts[0], series[0]),
                        textcoords="offset points", xytext=(-9, 0),
                        color=color, fontsize=9, ha="right", va="center",
                        weight="bold")

    ax2.plot(pcts, served, color=MUTED, linewidth=2, marker="o", markersize=5,
             markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("trace served (%)", color=INK, fontsize=10)
    ax2.set_xlabel("MPS active thread percentage  (share of the GPU's SMs)",
                   color=INK, fontsize=10)
    # Label every measured point, but drop a label whose neighbour is within a
    # few percent on the log axis -- otherwise 8/10/12 overprint into mush.
    import math
    keep, last = [], None
    for v in pcts:
        if last is None or math.log10(v) - math.log10(last) > 0.055:
            keep.append(v)
            last = v
    ax.set_xlim(min(pcts) * 0.72, max(pcts) * 1.16)
    ax2.set_xticks(keep)
    ax2.set_xticklabels([f"{v}%" for v in keep])
    ax2.set_xticks(pcts, minor=True)
    ax2.set_xticklabels([], minor=True)
    ax2.tick_params(axis="x", which="minor", length=3, color=GRID)

    fig.text(0.008, 0.008,
             "Median over pooled runs. 'trace served' is the share of submitted "
             "tasks that reached a first action;\nlatency below 100% is the "
             "latency of the survivors.",
             color=MUTED, fontsize=7.5, va="bottom")
    fig.subplots_adjust(left=0.115, right=0.975, top=0.92, bottom=0.155)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"\n  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arm", default=None, help="which arm to plot (default: all)")
    ap.add_argument("--reference", default=None,
                    help="log of an uncapped (100%%) run; every other run's plans "
                         "must match it exactly, since decoding is greedy")
    ap.add_argument("--out", default=None, help="output PNG (default: beside the manifest)")
    args = ap.parse_args()

    reference = None
    if args.reference:
        ref_stats = measure(args.reference)
        if ref_stats is None:
            print(f"unusable reference log: {args.reference}", file=sys.stderr)
            return 2
        reference = ref_stats["plans"]
        print(f"\n  reference: {args.reference} ({len(reference)} tasks)")

    pooled = collect(args.manifest, reference)
    if not pooled:
        print("no usable runs in the manifest", file=sys.stderr)
        return 2
    table(pooled)

    manifest = Path(args.manifest)
    arms = [args.arm] if args.arm else list(pooled)
    for arm in arms:
        out = Path(args.out) if args.out else \
            manifest.with_name(f"{manifest.stem.replace('-manifest', '')}-{arm}-latency.png")
        plot(pooled, arm, out, title_suffix="" if arm == "rtllm" else f"  ({arm})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
