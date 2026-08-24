#!/usr/bin/env bash
#
# Demonstrate the vLLM V1 port by running the paper's 7.4.1 workload.
#
#   ./scripts/demo.sh              # both arms, side by side  (~110s)
#   ./scripts/demo.sh rtllm        # TimelyLLM only           (~55s)
#   ./scripts/demo.sh vllm         # baseline only            (~55s)
#   ./scripts/demo.sh both 60      # both, 60s per arm
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

ARM="${1:-both}"
DURATION="${2:-45}"
case "$ARM" in
    rtllm|vllm|both) ;;
    *) echo "usage: $0 [rtllm|vllm|both] [seconds]" >&2; exit 2 ;;
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

"$PY" - "${ARGS[@]}" <<'PY'
import re, sys, collections

PATTERN = re.compile(r"^Output for task (\d+): (.*), time: [\d.]+$")
LABEL = {"rtllm": "rtllm  (TimelyLLM)", "vllm": "vllm   (baseline)"}


def read(path):
    plans = collections.OrderedDict()
    for line in open(path, errors="replace"):
        m = PATTERN.match(line.rstrip("\n"))
        if m:
            plans.setdefault(m.group(1), []).append(m.group(2))
    return plans


def render(seg):
    # A resume that yields only EOS shows up as an empty segment: the plan was
    # already complete, and this output is what carries the "final" flag back so
    # the scheduler can retire the task.
    return seg if seg else "<eos>"


arms = [(a.split("=", 1)[0], read(a.split("=", 1)[1])) for a in sys.argv[1:]]

for arm, plans in arms:
    total = sum(len(v) for v in plans.values())
    split = sum(1 for v in plans.values() if len(v) > 1)
    print(f"\n  {LABEL[arm]}   {total} segments across {len(plans)} tasks, "
          f"{split} split into more than one piece\n")
    for tid, segs in sorted(plans.items(), key=lambda kv: int(kv[0])):
        print(f"    task {tid:<4} {len(segs)}  " + " | ".join(render(s) for s in segs))

if len(arms) == 2:
    print("\n  side by side")
    print("  ------------")
    print(f"    {'':22}{'segments':>10}{'tasks':>8}{'per task':>10}")
    for arm, plans in arms:
        total = sum(len(v) for v in plans.values())
        n = len(plans) or 1
        print(f"    {LABEL[arm]:22}{total:>10}{len(plans):>8}{total / n:>10.1f}")

    # Same prompt under both arms makes the difference concrete.
    by_plan = {arm: {"".join(v): (t, v) for t, v in plans.items()}
               for arm, plans in arms}
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

print(f"\n  logs: {', '.join(a.split('=', 1)[1] for a in sys.argv[1:])}\n")
PY
