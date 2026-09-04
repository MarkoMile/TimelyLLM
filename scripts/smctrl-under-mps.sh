#!/usr/bin/env bash
# Does a libsmctrl TPC mask still take effect when work is submitted through MPS?
#
# Why it matters.  TimelyLLM and cuPHY are separate processes, so without MPS the
# GPU time-slices between their CUDA contexts and cuPHY is suspended wholesale
# while TimelyLLM runs -- no amount of masking helps.  MPS funnels both into one
# server context so their kernels run concurrently.  But MPS caps *how many* SMs
# a client may use, not *which*, so on its own it leaves the two contending.
# Concurrency has to come from MPS and placement from the mask, which only works
# if the mask survives the trip through the MPS server.
#
# The test is a throughput measurement, because a mask that has stopped working
# fails silently: it does not error, the kernel just runs on the whole GPU.
# Known non-MPS values from results/mps/gh200-smctrl-numcheck-map.csv:
#
#     8 TPC -> 56 TFLOP/s   22 TPC -> 210   44 TPC -> 434   64 TPC -> 560
#
# so under MPS, those same numbers mean the mask works and ~810 means it is a
# no-op.  No CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is ever set: that is the setting
# that corrupts cuBLAS, and the whole point is to partition without it.
#
# Safety: this starts a *private* MPS daemon of our own at a pipe directory
# under /tmp, never touches the root-owned daemon Aerial uses, never changes the
# GPU's compute mode, and quits its daemon on every exit path.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
GPU=0
TPCS="8 22 44 64"
RTLLM_TPCS=""              # if set, also run a full TimelyLLM trace at this budget
UTIL=0.67
RUN_DURATION=1200
CORES="${TIMELYLLM_CORES:-0-3,65-71}"

MPSDIR="/tmp/mps-$USER-gpu$GPU"
PIPE="$MPSDIR/pipe"
MPSLOG="$MPSDIR/log"
while [ $# -gt 0 ]; do
    case "$1" in
        --tpcs)          TPCS="$2"; shift 2 ;;
        --rtllm)         RTLLM_TPCS="$2"; shift 2 ;;
        --run-duration)  RUN_DURATION="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"

others() {
    local pid owner out=""
    while IFS=', ' read -r uuid pid mem; do
        [ "$uuid" = "$GPU_UUID" ] || continue
        owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        [ -z "$owner" ] || [ "$owner" = "$USER" ] && continue
        out="$out $owner:$pid($mem)"
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
    echo "${out# }"
}

mps_procs() { ps -u "$USER" -o pid=,comm=,args= | awk '$2 ~ /^nvidia-cuda-mps/'; }

cleanup() {
    local rc=$? survivors
    if [ -e "$PIPE/control" ]; then
        echo
        echo "  shutting down our MPS daemon"
        CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
            bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
        sleep 2
    fi
    survivors="$(mps_procs)"
    [ -z "$survivors" ] || { echo "  WARNING: an MPS process of yours survives:" >&2; echo "$survivors" >&2; }
    exit $rc
}
trap cleanup EXIT INT TERM HUP

echo
echo "  libsmctrl under MPS -- does the TPC mask survive the MPS server?"
echo "  GPU $GPU ($GPU_UUID)"
BASELINE="$(others)"
echo "  baseline occupants (not ours): ${BASELINE:-none}"

if [ -n "$(mps_procs)" ]; then
    echo "  an MPS daemon of yours is already running; quitting it first"
    CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
        bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
    sleep 2
    if [ -n "$(mps_procs)" ]; then
        echo "  refusing: an MPS daemon of yours survives the quit:" >&2
        mps_procs >&2
        exit 1
    fi
fi

# ---------------------------------------------------------- baseline, no MPS
echo
echo "  --- without MPS (the reference) ---"
for t in $TPCS; do
    env -u CUDA_HOME CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORES" \
        "$PY" "$REPO/scripts/smctrl-numcheck.py" --tpcs "$t" || true
done

# ---------------------------------------------------------------- with MPS
rm -rf "$MPSDIR"; mkdir -p "$PIPE" "$MPSLOG"
CUDA_VISIBLE_DEVICES="$GPU" CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
    nvidia-cuda-mps-control -d
sleep 2
DEFAULT_PCT="$(CUDA_MPS_PIPE_DIRECTORY="$PIPE" bash -c 'echo get_default_active_thread_percentage | nvidia-cuda-mps-control' 2>/dev/null || true)"
if [ -z "$DEFAULT_PCT" ]; then
    echo "  MPS daemon did not come up. Log:" >&2
    cat "$MPSLOG/control.log" >&2 || true
    exit 1
fi
echo
echo "  --- as an MPS client, no thread percentage set (default $DEFAULT_PCT) ---"
for t in $TPCS; do
    env -u CUDA_HOME \
        CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
        CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORES" \
        "$PY" "$REPO/scripts/smctrl-numcheck.py" --tpcs "$t" || true
done

echo
echo "  Read the TFLOP/s column: matching the non-MPS numbers means the mask"
echo "  survives MPS; jumping to ~810 at every budget means it is a no-op."

# ------------------------------------------------- the whole engine, end to end
# A microbenchmark shows the mask reaches the hardware.  This shows TimelyLLM
# itself surviving the combination it would actually run in beside cuPHY: an MPS
# client for the shared context, a libsmctrl mask for placement, and no thread
# percentage anywhere.
if [ -n "$RTLLM_TPCS" ]; then
    name="gh200-mps-smctrl-tpc$RTLLM_TPCS"
    console="$REPO/results/mps/$name.console.txt"
    echo
    echo "  --- TimelyLLM as an MPS client, masked to $RTLLM_TPCS TPCs ---"
    cd "$REPO/timelyllm"
    set +e
    env -u CUDA_HOME \
        CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
        TIMELYLLM_TPC_COUNT="$RTLLM_TPCS" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        timeout --signal=TERM --kill-after=60 $((RUN_DURATION + 900)) \
        taskset -c "$CORES" \
        "$PY" rtllm.py --preset exp741_timelyllm_high --log-name "$name" \
            --wait-for-engine --gpu-memory-utilization "$UTIL" \
            --run-duration "$RUN_DURATION" > "$console" 2>&1
    rc=$?
    set -e
    log="$REPO/timelyllm/logs/$name.log"
    if [ -f "$log" ]; then
        echo "  $(grep -c '^Output for task' "$log") segments -> $log"
    else
        echo "  NO LOG (rc=$rc) -- see $console"
    fi
fi
