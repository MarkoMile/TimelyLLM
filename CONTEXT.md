# TimelyLLM — MPS compute sweep

The vocabulary for measuring how TimelyLLM's latency degrades as GPU compute is
taken away from it with NVIDIA MPS. Terms here are the ones that have been
ambiguous in conversation or that mean something narrower in this project than
they do in general use.

## The measurement

**Time to first action**:
The interval from a task being submitted to the robot beginning to move. The
headline metric; it is what segmented generation exists to shorten.
_Avoid_: latency, TTFT, response time

**Share served**:
The fraction of the workload trace that was served rather than dropped. The
trace is open-loop, so starvation drops tasks instead of delaying them, which
flatters the latency of the survivors. Never read a latency without it.
_Avoid_: throughput, completion rate

**Segment**:
A contiguous run of generated tokens ending where the stop rule fires, after
which the plan so far can be dispatched to the robot. Segment boundaries can
only land on token boundaries, so they depend on the tokenizer.

**Arm**:
One system under test — `rtllm` (TimelyLLM) or `vllm` (base vLLM) — run against
the same trace. A sweep may cover one or both.
_Avoid_: mode, variant, condition

**Point**:
One (arm, thread percentage, repeat) triple: a single run producing one row in
the manifest.
_Avoid_: sample, data point, run

**Sweep**:
A set of points sharing an engine configuration and a tag, differing only in
thread percentage. Memory is held constant across a sweep so that compute is
the only variable.

**Thread percentage**:
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` — the ceiling on the fraction of the GPU's
SMs one MPS client may use, fixed per process at startup.
_Avoid_: SM cap, compute fraction, GPU share

**Green context**:
A CUDA partition of the device's SMs made inside our own process, with no MPS
daemon. It is the *other* way to take compute away from TimelyLLM, and unlike a
thread percentage it leaves the SM count the device reports untouched — which is
why it does not trip the cuBLAS bug. See docs/SM-SCHEDULING.md.
_Avoid_: MPS alternative, SM mask, stream partition

**SM budget**:
How many SMs TimelyLLM is currently allowed to run on, whichever mechanism set
it. `TIMELYLLM_SM_COUNT` sets it for a run; `sm_budget.set_sm_count()` changes
it live. Requested counts are rounded up to a multiple of 8 by the driver, so
the *budget* is what we asked for and the *partition* is what we got.
_Avoid_: SM allocation, SM quota, thread percentage

**TPC**:
Two SMs, and the unit libsmctrl masks in. `TIMELYLLM_TPC_COUNT` sets the budget
in TPCs. Beware that on this GPU a TPC budget is *not* a compute budget: mask
bit indexes do not correspond to software-visible TPC IDs on Hopper, so the
delivered throughput has to be measured rather than derived from the count.
_Avoid_: SM pair, cluster, partition

## Correctness

**Clean / Corrupt**:
A thread percentage is *clean* if the GPU computes correct answers at that
setting and *corrupt* if it silently computes wrong ones. Corruption here is
wrong results with no error, warning, or timing anomaly — not precision loss.
_Avoid_: passing/failing, good/bad, stable/unstable

**Pre-run gate**:
The GEMM check run before each point, which skips percentages that already
compute wrong answers. It is a filter that saves GPU time, never a proof of
correctness: percentages have passed it and still generated wrong plans.
_Avoid_: correctness check, validation

**Fidelity**:
Exact agreement between a point's generated plan text and the uncapped
reference run's, token for token. Decoding is greedy, so a correct run must
emit identical tokens. This is the check that decides whether a point counts.
_Avoid_: accuracy, quality, similarity

**Garbage**:
A point whose fidelity collapsed. It is not a slow measurement; it is not a
measurement, and it is excluded rather than plotted.

## Sharing the machine

**Neighbour**:
Another user's live workload resident on the same GPU. On this host that is
NVIDIA Aerial cuBB, a RAN L1 job, which shares the card by design.
_Avoid_: other job, contention

**Parked server**:
The root-owned `nvidia-cuda-mps-server` resident on the GPU for another user's
benefit. It occupies the card without being a neighbour workload, so guards
asking "is anyone else here?" must distinguish it from one. It is started by
hand, not by systemd, so it can appear or vanish at any time and must never be
assumed stable — which is why contamination is judged by change against a
baseline rather than by presence.

**Busy**:
Having work already in flight on the GPU. This is a precondition for the
corruption, not a performance description — and a neighbour can satisfy it
without us, which is why a clean verdict on this host is a sample rather than
a property.

**Flagged**:
A point taken while a neighbour was resident. Flagging marks the measurement as
contaminated and due to be re-run, not merely annotated — a flagged point is not
a usable result.
