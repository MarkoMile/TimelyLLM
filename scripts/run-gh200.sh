#!/usr/bin/env bash
#
# Launch TimelyLLM on devkit04 (GH200, aarch64).
#
# The settings below are requirements found during bring-up, not preferences.
# Running rtllm.py directly without them will fail or silently misbehave.
# Background in NOTES.md; rationale for the vLLM ones in PORT_PLAN.md.
#
#   ./scripts/run-gh200.sh --preset exp741_timelyllm_high --log-name run1
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# FlashInfer's JIT fails its CCCL compatibility check here: the venv carries
# nvcc 13.3.73 while the system headers are CUDA 13.0. The Torch sampler is
# unaffected, as is the prebuilt FlashAttention 3.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Setting CUDA_HOME makes vLLM find the mismatched toolkit and reintroduces the
# failure above, so make sure it is not inherited.
unset CUDA_HOME

# Keep EngineCore in this process. The frontend abort that ends a segment then
# lands synchronously inside step(), which matters because the next segment is
# resubmitted under the same request id and V1 silently reinterprets a live
# duplicate id as a streaming update (PORT_PLAN.md C5). It also keeps the V1
# scheduler reachable in-process as a debugging escape hatch.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Cores 4-64 are isolcpus, reserved for the Aerial cuBB RAN L1 workload sharing
# this GPU. Override with TIMELYLLM_CORES if the reservation changes.
CORES="${TIMELYLLM_CORES:-0-3,65-71}"

if [ -n "${TIMELYLLM_PYTHON:-}" ]; then
    PYTHON="$TIMELYLLM_PYTHON"
elif [ -x "$REPO/.venv/bin/python" ]; then
    PYTHON="$REPO/.venv/bin/python"
else
    PYTHON="python3"
fi

cd "$REPO/timelyllm"
echo "cores=$CORES python=$PYTHON" >&2
exec taskset -c "$CORES" "$PYTHON" rtllm.py "$@"
