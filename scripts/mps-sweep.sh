#!/usr/bin/env bash
#
# Sweep NVIDIA MPS active-thread-percentage and record TimelyLLM's latency at
# each point, so latency can be plotted against the compute the engine is given.
#
#   ./scripts/mps-sweep.sh                          # TimelyLLM, 10..100%, GPU 3
#   ./scripts/mps-sweep.sh --arms "rtllm vllm"      # both arms at every point
#   ./scripts/mps-sweep.sh --pct "25 50 100" --repeats 2
#
# What MPS gives us that nothing else does: a hard cap on the fraction of SMs a
# process may use, applied per client at process start, with memory left alone.
# So compute is the only variable -- the KV cache is the same size at every
# point, and the workload trace is identical.
#
# Safety on a shared machine. Everything here is scoped so other users are not
# affected:
#   - No compute-mode change. `nvidia-smi -c EXCLUSIVE_PROCESS` is what would
#     lock other people out of the card; we never touch it and need root for it.
#   - The daemon starts with CUDA_VISIBLE_DEVICES=$GPU, so its server manages
#     that one GPU and no other.
#   - A private CUDA_MPS_PIPE_DIRECTORY means only our clients join the server.
#     MPS servers are per-UID, so another user's process cannot attach anyway.
#   - The daemon is shut down on exit, including on error or Ctrl-C.
# The script refuses to start if anyone else already holds the target GPU, and
# records the GPU's process list around every run so a run that got shared with
# a late arrival can be spotted and redone.
#
# The pipe directory must be a SHORT path: it holds a unix socket, and the
# 108-byte sun_path limit is easy to exceed. A long path fails with exit 1 and
# an empty log, which is not a fun half hour.
#
# CORRECTNESS GATE. On this H100, capping SMs with MPS silently corrupts fp16
# tensor-core GEMM at many thread percentages -- the GPU computes the wrong
# answer with no error anywhere. A corrupted run still fills a log with
# plausible latencies, so every point is gated on scripts/mps-numcheck.py
# before it runs, and skipped if the GPU cannot do arithmetic at that setting.
# See that script for the measured pattern. Check the plotter's output-sanity
# column too: the gate is necessary, not proven sufficient.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU=3
PCTS="10 20 30 40 50 60 70 80 90 100"
ARMS="rtllm"
REPEATS=1
RUN_DURATION=1200
UTIL=0.8
TAG="mps"
OUTDIR="$REPO/results/mps"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)          GPU="$2"; shift 2 ;;
        --pct)          PCTS="$2"; shift 2 ;;
        --arms)         ARMS="$2"; shift 2 ;;
        --repeats)      REPEATS="$2"; shift 2 ;;
        --run-duration) RUN_DURATION="$2"; shift 2 ;;
        --util)         UTIL="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --outdir)       OUTDIR="$2"; shift 2 ;;
        -h|--help)      sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

PY="${TIMELYLLM_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY" >&2; exit 2; }

MPSDIR="/tmp/mps-$USER-gpu$GPU"          # short on purpose; see the note above
PIPE="$MPSDIR/pipe"
MPSLOG="$MPSDIR/log"
LOGDIR="$REPO/timelyllm/logs"
mkdir -p "$LOGDIR" "$OUTDIR"

GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")"
MANIFEST="$OUTDIR/$TAG-manifest.csv"

# Other people's processes on the target GPU. Used to refuse a busy GPU, and to
# flag any run that ended up sharing the card with a late arrival. Our own
# nvidia-cuda-mps-server sits on the GPU for the whole sweep, so ownership --
# not PID -- is what separates "someone else is here" from "that is us".
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

# Our own MPS daemon/server processes. `pgrep -x` cannot match a name longer
# than 15 characters, and `pgrep -f` matches any shell whose command line merely
# quotes the string -- including the one running this check. `comm` is the
# kernel's truncated process name, so it matches the daemon and nothing else.
mps_procs() {
    ps -u "$USER" -o pid=,comm=,args= | awk '$2 ~ /^nvidia-cuda-mps/'
}

cleanup() {
    local rc=$?
    if [ -S "$PIPE/control" ] || [ -e "$PIPE/control" ]; then
        echo
        echo "  shutting down the MPS daemon on GPU $GPU"
        CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
            bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 || true
        sleep 2
    fi
    local survivors
    survivors="$(mps_procs)"
    if [ -n "$survivors" ]; then
        echo "  WARNING: an MPS process of yours survives:" >&2
        echo "$survivors" >&2
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- preflight

echo
echo "  MPS thread-percentage sweep"
echo "  GPU $GPU  ($GPU_UUID)"
echo "  arms: $ARMS    percentages: $PCTS    repeats: $REPEATS"
echo "  gpu_memory_utilization=$UTIL   run-duration=${RUN_DURATION}s"
echo

BUSY="$(others)"
if [ -n "$BUSY" ]; then
    echo "  refusing: GPU $GPU already has processes on it: $BUSY" >&2
    echo "  Pick an idle GPU with --gpu, or wait. Sharing the card would both" >&2
    echo "  corrupt the measurement and slow the other job down." >&2
    exit 1
fi

if [ -n "$(mps_procs)" ]; then
    echo "  refusing: you already have an MPS control daemon running." >&2
    echo "  Stop it first, or its pipe directory will be used instead of ours." >&2
    exit 1
fi

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
echo "  MPS daemon up on GPU $GPU (default thread percentage $DEFAULT_PCT)"

[ -f "$MANIFEST" ] || echo "arm,pct,repeat,log,started,ended,load_s,others_before,others_after" > "$MANIFEST"

# ---------------------------------------------------------------- the sweep

# rtllm.py resolves the prompt as os.getcwd() + prompt_path and the model path
# relatively, so it only works from inside timelyllm/.
cd "$REPO/timelyllm"

for rep in $(seq 1 "$REPEATS"); do
for arm in $ARMS; do
    case "$arm" in
        rtllm) preset=exp741_timelyllm_high ;;
        vllm)  preset=exp741_vllm_high ;;
        *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac
    for pct in $PCTS; do
        name="$TAG-$arm-p$pct-r$rep"
        console="$LOGDIR/$name.console.txt"
        before="$(others)"
        started="$(date -Is)"
        printf "  %-22s %3s%%  rep %s  " "$arm" "$pct" "$rep"

        # Refuse to spend six minutes measuring a GPU that is computing the
        # wrong answer. Cheap: a few matmuls, about two seconds.
        if ! numcheck="$(env -u CUDA_HOME \
                CUDA_MPS_PIPE_DIRECTORY="$PIPE" \
                CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
                CUDA_VISIBLE_DEVICES=0 \
                "$PY" "$REPO/scripts/mps-numcheck.py" 2>&1)"; then
            echo "SKIPPED -- ${numcheck##*: }"
            echo "$arm,$pct,$rep,,$started,$(date -Is),SKIPPED_NUMCHECK,\"$before\",\"\"" >> "$MANIFEST"
            continue
        fi

        set +e
        env -u CUDA_HOME \
            CUDA_MPS_PIPE_DIRECTORY="$PIPE" \
            CUDA_MPS_LOG_DIRECTORY="$MPSLOG" \
            CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$pct" \
            CUDA_VISIBLE_DEVICES=0 \
            VLLM_USE_FLASHINFER_SAMPLER=0 \
            VLLM_ENABLE_V1_MULTIPROCESSING=0 \
            "$PY" rtllm.py --preset "$preset" --log-name "$name" \
                --wait-for-engine \
                --gpu-memory-utilization "$UTIL" \
                --run-duration "$RUN_DURATION" \
                > "$console" 2>&1
        rc=$?
        set -e

        ended="$(date -Is)"
        after="$(others)"
        load_s="$(grep -oP 'Engine ready after \K[0-9.]+' "$console" | head -1 || true)"
        [ -n "$load_s" ] || load_s="NA"

        if [ -f "$LOGDIR/$name.log" ]; then
            n=$(grep -c '^Output for task' "$LOGDIR/$name.log" || echo 0)
            echo "load ${load_s}s, $n segments"
        else
            echo "NO LOG (rc=$rc) -- see $console"
        fi
        [ -n "$before$after" ] && echo "      note: GPU $GPU was shared during this run (before='$before' after='$after')"

        echo "$arm,$pct,$rep,$LOGDIR/$name.log,$started,$ended,$load_s,\"$before\",\"$after\"" >> "$MANIFEST"
    done
done
done

echo
echo "  manifest: $MANIFEST"
echo "  plot with: $PY $REPO/scripts/plot-mps-latency.py --manifest $MANIFEST"
