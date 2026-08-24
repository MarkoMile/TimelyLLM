#!/usr/bin/env bash
#
# Demonstrate the vLLM V1 port by running the paper's 7.4.1 workload.
#
#   ./scripts/demo.sh              # both arms, side by side  (~110s)
#   ./scripts/demo.sh rtllm        # TimelyLLM only           (~55s)
#   ./scripts/demo.sh vllm         # baseline only            (~55s)
#   ./scripts/demo.sh both 60      # both, 60s per arm
#   ./scripts/demo.sh rtllm -v     # verbose: every plan, with what was asked
#                                  # and a merged generation/execution timeline
#
# The two arms differ only in run_mode:
#
#   vllm    plain continuous batching. Each plan is generated whole, so the
#           drone cannot move until the entire plan is finished.
#   rtllm   TimelyLLM. Generation is cut at MiniSpec statement boundaries; each
#           fragment goes to the drone immediately and the task resumes
#           mid-plan later, under the same request id, while the GPU serves
#           other agents.
#
# 45s is about the floor. The workload is a five-minute trace whose first task
# does not arrive until t=34s, so a shorter run expires before anything is sent.
# Model loading (~25s) overlaps that warm-up, so it costs nothing extra.
#
set -euo pipefail

VERBOSE=0
POS=()
for a in "$@"; do
    case "$a" in
        -v|--verbose) VERBOSE=1 ;;
        *) POS+=("$a") ;;
    esac
done
ARM="${POS[0]:-both}"
DURATION="${POS[1]:-45}"
case "$ARM" in
    rtllm|vllm|both) ;;
    *) echo "usage: $0 [rtllm|vllm|both] [seconds] [-v]" >&2; exit 2 ;;
esac
if [ "$DURATION" -lt 40 ]; then
    echo "  refusing: ${DURATION}s is below the trace's 34s warm-up, so no task" >&2
    echo "  would ever be sent. Use 45 or more." >&2
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/timelyllm/logs"
PY="${TIMELYLLM_PYTHON:-python3}"
# 0.20 rather than the 0.8 default: this GPU is shared, and 0.20 also puts the
# KV cache near the size the paper's RTX 4090 had.
UTIL="${TIMELYLLM_DEMO_UTIL:-0.20}"

run_arm() {  # $1 = rtllm|vllm
    local arm="$1" preset
    [ "$arm" = rtllm ] && preset=exp741_timelyllm_high || preset=exp741_vllm_high
    echo "  running $arm ($preset), ${DURATION}s ..."
    "$REPO/scripts/run-gh200.sh" --preset "$preset" \
        --gpu-memory-utilization "$UTIL" --run-duration "$DURATION" \
        --log-name "demo-$arm" > "$LOGDIR/demo-$arm.console.txt" 2>&1 || true
    if [ ! -f "$LOGDIR/demo-$arm.log" ]; then
        echo "  $arm produced no log; see $LOGDIR/demo-$arm.console.txt" >&2
        return 1
    fi
}

echo
echo "  TimelyLLM  ·  vLLM 0.27.1 (V1 engine)  ·  Llama-3-8B  ·  aarch64 / GH200"
echo "  ------------------------------------------------------------------------"

ARGS=()
if [ "$ARM" = both ] || [ "$ARM" = vllm ]; then
    run_arm vllm  && ARGS+=("vllm=$LOGDIR/demo-vllm.log")
fi
if [ "$ARM" = both ] || [ "$ARM" = rtllm ]; then
    run_arm rtllm && ARGS+=("rtllm=$LOGDIR/demo-rtllm.log")
fi
[ ${#ARGS[@]} -eq 0 ] && { echo "  nothing to report" >&2; exit 1; }

"$PY" - "$VERBOSE" "$REPO/dataset/data_sample_1.json" "${ARGS[@]}" <<'PY'
import json, re, sys, collections

OUT = re.compile(r"^Output for task (\d+): (.*), time: ([\d.]+)$")
ADD = re.compile(r"^Added task (\d+), time: ([\d.]+)$")
EXE = re.compile(r"^(Start|Finish) executing task (\d+) for agent (\d+) "
                 r"on time ([\d.]+) with plan (.*)$")
LABEL = {"rtllm": "rtllm  (TimelyLLM)", "vllm": "vllm   (baseline)"}

VERBOSE = sys.argv[1] == "1"
try:
    ASKED = {str(t["job_id"]): (t["task_input"], t["agent_id"])
             for t in json.load(open(sys.argv[2]))}
except Exception:
    ASKED = {}


def read(path):
    """plans: task -> [segment text]; events: task -> [(t, kind, detail)]"""
    plans, events = collections.OrderedDict(), collections.defaultdict(list)
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = OUT.match(line)
        if m:
            plans.setdefault(m.group(1), []).append(m.group(2))
            events[m.group(1)].append((float(m.group(3)), "gen", m.group(2)))
            continue
        m = ADD.match(line)
        if m:
            events[m.group(1)].append((float(m.group(2)), "submit", ""))
            continue
        m = EXE.match(line)
        if m:
            kind = "exec" if m.group(1) == "Start" else "done"
            events[m.group(2)].append((float(m.group(4)), kind, m.group(5)))
    return plans, events


def timeline(tid, events):
    """Generation and drone execution on one clock, so the overlap is visible."""
    rows = sorted(events.get(tid, []))
    if not rows:
        return
    t0 = rows[0][0]
    for t, kind, detail in rows:
        stamp = f"    +{t - t0:6.3f}s"
        if kind == "submit":
            print(f"{stamp}  gen    submit")
        elif kind == "gen":
            print(f"{stamp}  gen    -> {repr(detail) if detail else '<eos>'}")
        elif kind == "exec":
            print(f"{stamp}  drone  start  {detail!r}")
        else:
            print(f"{stamp}  drone  done   {detail!r}")


def render(seg):
    # A resume that yields only EOS shows up as an empty segment: the plan was
    # already complete, and this output is what carries the "final" flag back so
    # the scheduler can retire the task.
    return seg if seg else "<eos>"


arms = []
for a in sys.argv[3:]:
    name, path = a.split("=", 1)
    plans, events = read(path)
    arms.append((name, plans, events))

for arm, plans, events in arms:
    total = sum(len(v) for v in plans.values())
    split = sum(1 for v in plans.values() if len(v) > 1)
    print(f"\n  {LABEL[arm]}   {total} segments across {len(plans)} tasks, "
          f"{split} split into more than one piece\n")

    if not VERBOSE:
        for tid, segs in sorted(plans.items(), key=lambda kv: int(kv[0])):
            print(f"    task {tid:<4} {len(segs)}  "
                  + " | ".join(render(s) for s in segs))
        continue

    for tid, segs in sorted(plans.items(), key=lambda kv: int(kv[0])):
        asked, agent = ASKED.get(tid, ("", "?"))
        print(f"  task {tid}   agent {agent}   {len(segs)} segment"
              f"{'s' if len(segs) != 1 else ''}")
        if asked:
            print(f"    asked  {asked}")
        print(f"    plan   {''.join(segs)}")
        timeline(tid, events)
        print()

def time_to_first_move(events):
    """Submission -> the drone's first movement, per task.

    This is what segmenting is for: the baseline cannot move until the whole
    plan exists, while TimelyLLM releases the first fragment as soon as it is
    executable.
    """
    out = []
    for tid, rows in events.items():
        submits = [t for t, k, _ in rows if k == "submit"]
        starts = [t for t, k, _ in rows if k == "exec"]
        if submits and starts:
            out.append(min(starts) - min(submits))
    return sorted(out)


if len(arms) == 2:
    print("\n  side by side")
    print("  ------------")
    print(f"    {'':22}{'segments':>10}{'tasks':>8}{'per task':>10}"
          f"{'to 1st move':>14}")
    for arm, plans, events in arms:
        total = sum(len(v) for v in plans.values())
        n = len(plans) or 1
        d = time_to_first_move(events)
        med = f"{d[len(d) // 2] * 1000:.0f} ms" if d else "-"
        print(f"    {LABEL[arm]:22}{total:>10}{len(plans):>8}{total / n:>10.1f}"
              f"{med:>14}")
    print("\n    'to 1st move' is the median delay from a task being submitted to")
    print("    the drone starting to act. Indicative only: one short run on a")
    print("    shared GPU, not a measurement.")

    # Same prompt under both arms makes the difference concrete.
    by_plan = {arm: {"".join(v): (t, v) for t, v in plans.items()}
               for arm, plans, _ in arms}
    shared = set(by_plan["vllm"]) & set(by_plan["rtllm"])
    if shared:
        key = max(shared, key=lambda k: len(by_plan["rtllm"][k][1]))
        print("\n  the same plan under each arm:\n")
        print(f"    vllm    {by_plan['vllm'][key][1][0]}")
        print("    rtllm   " + "\n            ".join(
            render(s) for s in by_plan["rtllm"][key][1]))
        print("\n  The baseline emits it whole: the drone cannot move until the")
        print("  entire plan exists. TimelyLLM releases each fragment the moment")
        print("  it completes, and then keeps generating -- the next fragment is")
        print("  produced while the drone is still flying the previous one, so")
        print("  generation and physical execution overlap instead of alternating.")

print(f"\n  logs: {', '.join(a.split('=', 1)[1] for a in sys.argv[3:])}\n")
PY
