#!/usr/bin/env python3
"""Summarise the four-cell experiment grid across both engines and both arms.

The grid is 2x2: two engines (vLLM 0.5.4 upstream, 0.27.1 ported) crossed with
two run modes (base vLLM batching, TimelyLLM). Reading it:

  across a row   did the port change behaviour?   (engine effect)
  down a column  does TimelyLLM beat plain batching on that engine?
  the two column gaps compared   does the paper's effect survive the port?

The last one is the deliverable. The base-vLLM row exists so an improvement can
be attributed: without it, ported TimelyLLM beating original TimelyLLM could
just be vLLM having got faster between versions.

The headline number is time to first action -- from a task being submitted to
the drone starting to move. That is what segmenting exists to shorten: the
baseline cannot act until the whole plan exists, while TimelyLLM releases the
first executable fragment immediately.

Stdlib only; it never imports vllm. Compares logs, not live engines.

    ./scripts/summarize-runs.py                       # discover the four logs
    ./scripts/summarize-runs.py "label=path.log" ...  # or name them explicitly
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

V1_TREE = Path(__file__).resolve().parent.parent
DEFAULT_V0_TREE = V1_TREE.parent / f"{V1_TREE.name}-v0"

OUT = re.compile(r"^Output for task (\d+): (.*), time: ([\d.]+)$")
ADD = re.compile(r"^Added task (\d+), time: ([\d.]+)$")
EXE = re.compile(r"^Start executing task (\d+) for agent \d+ on time ([\d.]+) with plan")


def measure(path):
    """Per-run stats. Returns None if the log has no completed work."""
    segments, submitted, started = {}, {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = OUT.match(line)
            if m:
                segments.setdefault(m.group(1), []).append(m.group(2))
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
    # Submission to the drone's first movement, per task.
    delays = sorted(started[t] - submitted[t]
                    for t in submitted if t in started)
    total = sum(len(v) for v in segments.values())
    return {
        "tasks": len(segments),
        "segments": total,
        "per_task": total / len(segments),
        "median_ms": statistics.median(delays) * 1000 if delays else None,
        "p90_ms": delays[int(len(delays) * 0.9)] * 1000 if delays else None,
        "n_timed": len(delays),
    }


def discover(v0_tree, rtllm_tag, vllm_tag):
    """The four logs compare-arms.py leaves behind, in grid order."""
    return [
        ("0.5.4", "base vLLM", v0_tree / "timelyllm/logs" / f"{vllm_tag}-v0.log"),
        ("0.5.4", "TimelyLLM", v0_tree / "timelyllm/logs" / f"{rtllm_tag}-v0.log"),
        ("0.27.1", "base vLLM", V1_TREE / "timelyllm/logs" / f"{vllm_tag}-v1.log"),
        ("0.27.1", "TimelyLLM", V1_TREE / "timelyllm/logs" / f"{rtllm_tag}-v1.log"),
    ]


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("logs", nargs="*", metavar="label=path",
                    help="explicit logs; label as 'engine arm', e.g. "
                         "'0.27.1 TimelyLLM=logs/rtllm-v1.log'")
    ap.add_argument("--v0-tree", default=str(DEFAULT_V0_TREE))
    ap.add_argument("--rtllm-tag", default="rtllm")
    ap.add_argument("--vllm-tag", default="vllm")
    args = ap.parse_args()

    if args.logs:
        rows = []
        for spec in args.logs:
            label, path = spec.split("=", 1)
            engine, _, arm = label.partition(" ")
            rows.append((engine, arm or "?", Path(path)))
    else:
        rows = discover(Path(args.v0_tree), args.rtllm_tag, args.vllm_tag)

    results, missing = [], []
    for engine, arm, path in rows:
        if not path.exists():
            missing.append((engine, arm, path))
            continue
        stats = measure(path)
        if stats is None:
            missing.append((engine, arm, path))
            continue
        results.append((engine, arm, stats))

    if not results:
        print("no usable logs found. Run compare-arms.py for both presets first:",
              file=sys.stderr)
        print("  ./scripts/compare-arms.py --preset exp741_timelyllm_high --tag rtllm",
              file=sys.stderr)
        print("  ./scripts/compare-arms.py --preset exp741_vllm_high      --tag vllm",
              file=sys.stderr)
        return 2

    base = {e: s["median_ms"] for e, a, s in results if a == "base vLLM"}

    print()
    print(f"  {'engine':<9}{'arm':<12}{'tasks':>7}{'segs':>7}{'per task':>10}"
          f"{'1st move':>11}{'p90':>9}{'vs base':>10}")
    print("  " + "-" * 75)
    for engine, arm, s in results:
        med = f"{s['median_ms']:.0f} ms" if s["median_ms"] is not None else "-"
        p90 = f"{s['p90_ms']:.0f} ms" if s["p90_ms"] is not None else "-"
        ratio = "-"
        if arm != "base vLLM" and base.get(engine) and s["median_ms"]:
            ratio = f"{base[engine] / s['median_ms']:.2f}x"
        print(f"  {engine:<9}{arm:<12}{s['tasks']:>7}{s['segments']:>7}"
              f"{s['per_task']:>10.1f}{med:>11}{p90:>9}{ratio:>10}")

    speedups = {e: base[e] / s["median_ms"]
                for e, a, s in results
                if a == "TimelyLLM" and base.get(e) and s["median_ms"]}
    print()
    if len(speedups) == 2:
        old, new = speedups.get("0.5.4"), speedups.get("0.27.1")
        print(f"  TimelyLLM's speedup over plain batching: {old:.2f}x on 0.5.4, "
              f"{new:.2f}x on 0.27.1.")
        drift = abs(new - old) / old
        if drift < 0.15:
            print("  The effect survives the port; the two agree within 15%.")
        else:
            print(f"  These differ by {drift * 100:.0f}%. Check the plan-text diff from")
            print("  compare-arms.py before reading anything into it -- a port defect")
            print("  and a real engine difference look the same in this table.")
    elif speedups:
        print("  Only one engine measured, so the port cannot be attributed yet.")

    if missing:
        print("\n  not measured:")
        for engine, arm, path in missing:
            print(f"    {engine:<8}{arm:<12}{path}")

    print("\n  '1st move' is the median delay from a task being submitted to the")
    print("  drone starting to act. Indicative unless every run used the same")
    print("  exclusive GPU, the same model, and the same duration.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
