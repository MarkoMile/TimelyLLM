#!/usr/bin/env bash
# Sweep TimelyLLM's latency against the SMs it is given, using CUDA green
# contexts instead of MPS.
#
# Why this exists.  scripts/mps-sweep.sh has to skip most of its grid: an MPS SM
# cap silently corrupts cuBLAS GEMM at most SM counts (docs/MPS-CUBLAS-CORRUPTION.md),
# so the numcheck gate rejects the point before it runs.  A green context
# partitions SMs without changing the multiProcessorCount cuBLAS keys its kernel
# choice on, and is numerically clean at every partition size
# (scripts/greenctx-numcheck.py).  It also needs no control daemon, so nothing is
# left running on a shared machine.
#
# SMs come in groups of 8 on this GH200 (132 total), so the useful ladder is
# 8, 16, 24, ... 128, 132.
#
# Usage:
#   ./scripts/greenctx-sweep.sh --tag gh200-gctx --sms "16 32 62 88 118 132"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
TAG="gh200-gctx"
SMS="8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 132"
ARMS="rtllm"
REPEATS=1
UTIL=0.67
RUN_DURATION=1200
GPU=0
CORES="${TIMELYLLM_CORES:-0-3,65-71}"
POINT_TIMEOUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)           TAG="$2"; shift 2 ;;
        --sms)           SMS="$2"; shift 2 ;;
        --arms)          ARMS="$2"; shift 2 ;;
        --repeats)       REPEATS="$2"; shift 2 ;;
        --util)          UTIL="$2"; shift 2 ;;
        --run-duration)  RUN_DURATION="$2"; shift 2 ;;
        --point-timeout) POINT_TIMEOUT="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
POINT_TIMEOUT="${POINT_TIMEOUT:-$((RUN_DURATION + 900))}"

TOTAL_SM=$("$PY" -c "import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)")
LOGDIR="$REPO/results/mps"
MANIFEST="$LOGDIR/$TAG-manifest.csv"
mkdir -p "$LOGDIR"

others() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null \
        | awk -v me="$(id -u)" '{gsub(/,/,"");pid=$1;
            cmd="stat -c %u /proc/"pid" 2>/dev/null"; cmd|getline uid; close(cmd);
            if (uid != me && uid != "") printf "%s(%s %s) ", pid, $2, $3}' | sed 's/ $//'
}

echo "  green-context SM sweep"
echo "  GPU $GPU, $TOTAL_SM SMs total"
echo "  arms: $ARMS    SM counts: $SMS    repeats: $REPEATS"
echo "  gpu_memory_utilization=$UTIL   run-duration=${RUN_DURATION}s"
BASELINE="$(others)"
echo "  baseline occupants of GPU $GPU: ${BASELINE:-none}"
echo

[ -f "$MANIFEST" ] || echo "arm,pct,repeat,log,started,ended,load_s,others_before,others_after" > "$MANIFEST"

# rtllm.py resolves the prompt as os.getcwd() + prompt_path, so it only works
# from inside timelyllm/.
cd "$REPO/timelyllm"

FLAGGED=""
for rep in $(seq 1 "$REPEATS"); do
for arm in $ARMS; do
    case "$arm" in
        rtllm) preset=exp741_timelyllm_high ;;
        vllm)  preset=exp741_vllm_high ;;
        *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac
    for sm in $SMS; do
        # The plotter's x-axis is a percentage of the GPU, so record the
        # partition that way and keep the two sweeps directly comparable.
        pct=$(( (sm * 100 + TOTAL_SM / 2) / TOTAL_SM ))
        name="$TAG-$arm-sm$sm-r$rep"
        console="$LOGDIR/$name.console.txt"
        before="$(others)"
        started="$(date -Is)"
        printf "  %-8s %3d SM (%3d%%)  rep %s  " "$arm" "$sm" "$pct" "$rep"

        set +e
        env -u CUDA_HOME \
            TIMELYLLM_SM_COUNT="$sm" \
            CUDA_VISIBLE_DEVICES=0 \
            VLLM_USE_FLASHINFER_SAMPLER=0 \
            VLLM_ENABLE_V1_MULTIPROCESSING=0 \
            timeout --signal=TERM --kill-after=60 "$POINT_TIMEOUT" \
            taskset -c "$CORES" \
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
        log="$REPO/timelyllm/logs/$name.log"

        if [ -f "$log" ]; then
            n=$(grep -c '^Output for task' "$log" || echo 0)
            echo "load ${load_s}s, $n segments"
        else
            echo "NO LOG (rc=$rc) -- see $console"
        fi
        if [ "$before" != "$BASELINE" ] || [ "$after" != "$BASELINE" ]; then
            echo "      *** FLAGGED: neighbours on GPU $GPU changed during this run"
            FLAGGED="$FLAGGED $sm"
        fi
        echo "$arm,$pct,$rep,$log,$started,$ended,$load_s,\"$before\",\"$after\"" >> "$MANIFEST"
    done
done
done

echo
[ -z "$FLAGGED" ] || echo "  *** FLAGGED POINTS (re-run these):$FLAGGED"
echo "  manifest: $MANIFEST"
