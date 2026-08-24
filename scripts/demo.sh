#!/usr/bin/env bash
#
# One-minute demonstration that TimelyLLM runs on the vLLM V1 engine.
#
# Runs the paper's 7.4.1 workload briefly, then prints the plans it produced,
# segment by segment. Each segment is one stop-and-resume cycle: generation was
# cut at a MiniSpec statement boundary, that fragment was handed to the robot to
# execute, and the task later resumed mid-plan under the same request id. That
# loop is TimelyLLM's contribution, and it is what this shows working.
#
#   ./scripts/demo.sh          # ~55s wall clock
#   ./scripts/demo.sh 60       # longer, more tasks
#
# 45s is about the floor. The workload is a five-minute trace whose first task
# does not arrive until t=34s, so a shorter run expires before anything is ever
# sent. Model loading (~25s) happens in parallel with that warm-up, so it costs
# nothing extra.
#
set -euo pipefail
DURATION="${1:-45}"

if [ "$DURATION" -lt 40 ]; then
    echo "  refusing: --run-duration ${DURATION}s is below the trace's 34s warm-up," >&2
    echo "  so no task would ever be sent. Use 45 or more." >&2
    exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$REPO/timelyllm/logs/demo.log"
CONSOLE="$REPO/timelyllm/logs/demo.console.txt"

# 0.20 rather than the 0.8 default: this GPU is shared, and 0.20 also puts the
# KV cache near the size the paper's RTX 4090 had.
UTIL="${TIMELYLLM_DEMO_UTIL:-0.20}"

echo
echo "  TimelyLLM  ·  vLLM 0.27.1 (V1 engine)  ·  Llama-3-8B  ·  aarch64 / GH200"
echo "  ---------------------------------------------------------------------"
echo "  workload: exp741_timelyllm_high, 42 agents, ${DURATION}s"
echo "  loading model and running ..."

"$REPO/scripts/run-gh200.sh" --preset exp741_timelyllm_high \
    --gpu-memory-utilization "$UTIL" --run-duration "$DURATION" \
    --log-name demo > "$CONSOLE" 2>&1 || true

if [ ! -f "$LOG" ]; then
    echo
    echo "  no log produced -- the run ended before any task was scheduled."
    echo "  console output: $CONSOLE"
    exit 1
fi

"${TIMELYLLM_PYTHON:-python3}" - "$LOG" <<'PY'
import re, sys, collections
log = sys.argv[1]
pat = re.compile(r"^Output for task (\d+): (.*), time: [\d.]+$")
plans = collections.OrderedDict()
for line in open(log, errors="replace"):
    m = pat.match(line.rstrip("\n"))
    if m:
        plans.setdefault(m.group(1), []).append(m.group(2))

total = sum(len(v) for v in plans.values())
print(f"\n  {total} segments generated across {len(plans)} tasks\n")

# Prefer tasks with the most segments, but skip repeats: the trace replays the
# same prompts across agents, so the top three by length are often one plan
# three times, which hides the variety.
chosen, seen = [], set()
for tid, segs in sorted(plans.items(), key=lambda kv: -len(kv[1])):
    key = "".join(segs)
    if key in seen:
        continue
    seen.add(key)
    chosen.append((tid, segs))
    if len(chosen) == 3:
        break

for tid, segs in chosen:
    label = f"  task {tid}"
    print(f"{label}{' ' * max(1, 12 - len(label))}{segs[0]}")
    for s in segs[1:]:
        print(f"{' ' * 12}{s}")
    print()

print("  Each line is one segment. Generation stopped at a MiniSpec statement")
print("  boundary, the fragment went to the robot to execute, and the task")
print("  resumed mid-plan under the same request id while the GPU served others.")
print(f"\n  full log: {log}\n")
PY
