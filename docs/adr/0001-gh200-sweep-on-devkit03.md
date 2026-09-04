---
status: accepted
---

# Run the GH200 MPS sweep on devkit03, with a cu129 build and util 0.67

The sweep was handed off targeting devkit04, which has a working venv
(vLLM 0.27.1 / torch 2.13.0+cu130 / Python 3.12) and driver 590.48.01. devkit04
is not reachable from devkit03 and none of that environment exists here, so the
sweep runs on devkit03: same GH200 hardware and same `isolcpus=4-64` CPU
reservation, but driver 575.64.03, which caps us at CUDA 12.9 and therefore
rules out every cu130 wheel. We install the official `+cu129` aarch64 build of
the *same* vLLM 0.27.1 and *same* torch 2.13.0 against Python 3.12.14, and run
at `gpu_memory_utilization=0.67` so the absolute memory budget matches the
H100's `0.8` of an 80 GB card.

## Considered options

Matching the H100 stack exactly was impossible: CUDA 13.x requires driver r580+.
Downgrading vLLM to a release whose default wheels are CUDA 12 was unnecessary
once we found that vLLM publishes a `+cu129` aarch64 wheel for every release,
including 0.27.1. Python 3.10 was considered and rejected: the port branch's
`pyproject.toml` requires `>=3.12`, because that is what vLLM 0.27.x and the
aarch64 torch build target. 3.12 via `uv` also matches devkit04's interpreter.

For memory, `NOTES.md` measured that Llama-3-8B fits at `0.22` on devkit04 — but
that was forced by 71-77 GB of other people's residency, not chosen. Running at
`0.22` here would shrink the KV cache to roughly a third of the H100's, and
because share-served governs where tasks start being dropped, that could move
the knee of the curve for a memory reason while we attribute it to compute.

## Consequences

The latency comparison against the H100 is close to clean: identical vLLM and
torch versions, matched absolute KV cache, same trace, same
`VLLM_USE_FLASHINFER_SAMPLER=0` sampler path.

The correctness comparison is weaker, and in the one place that matters most.
The H100 finding is a *cuBLAS* bug, and cuBLAS 12.9 and 13.0 are different
libraries. A clean result on devkit03 therefore confounds three axes at once —
aarch64 vs x86, CUDA 12.9 vs 13.0, driver 575 vs 580 — where devkit04 would have
isolated architecture alone. This must be stated in any writeup: devkit03 can
show that the corruption is not universal, but it cannot localise it.

`util=0.67` reserves roughly two thirds of a card shared by design with the
Aerial cuBB RAN L1 workload. It is only defensible while that neighbour is idle,
which is why sharing is re-checked before and after every point and flagged.

## Measured outcome (2026-09-03), which reverses the paragraph above

The pessimism above was wrong, and usefully so. The corruption reproduces on
devkit03, and the clean set is *identical* to the H100's:

    clean    5 8 10 12 25 30 33 50 90 95 100
    corrupt  20 40 60 66 70 75 80

Three passes at each percentage, deterministic. Because the same pattern
survives a change of architecture (aarch64 vs x86_64), CUDA version (12.9 vs
13.0) and driver (575.64.03 vs 580.159.03), those three are ruled out rather
than confounded. What did not change is the Hopper sm_90 large-tile tensor-core
path and its SM-count-dependent kernel selection, which is where the bug lives.

Running on devkit03 therefore localised the bug better than devkit04 would have.
