# SM scheduling for TimelyLLM: use green contexts, not MPS

**Status: solved, and cheaply.** The blocker was never SM partitioning as such.
It was the *mechanism* we were partitioning with. MPS corrupts cuBLAS; CUDA
green contexts do not, and they are a better fit for scheduling anyway.

Companion documents: [MPS-CUBLAS-CORRUPTION.md](MPS-CUBLAS-CORRUPTION.md) (why
MPS breaks), [MPS-COMPUTE-SWEEP.md](MPS-COMPUTE-SWEEP.md) (the sweep runbook).

---

## The one-line answer

An MPS thread-percentage cap changes the SM count the *device reports*, and
cuBLAS keys its GEMM kernel selection on that number. Capping to 44 SMs makes
cuBLAS pick a kernel configuration no physical Hopper part would ever get, and
some of those configurations compute garbage.

A green context partitions SMs **without changing the reported count**. The
device still says 132 SMs, cuBLAS picks exactly the kernel it picks on the whole
GPU — a kernel we know is correct — and the driver runs it on the SMs we asked
for. The bug is never reached.

## Evidence

`scripts/greenctx-numcheck.py`, run on this GH200 (132 SMs, torch 2.13.0+cu129):

```
 num_sms  reported  TFLOP/s  worst_rel  verdict
  (none)       132    813.4   3.51e-03  CLEAN
       8       132     56.1   3.51e-03  CLEAN
      16       132    110.1   3.51e-03  CLEAN
      18       132    164.8   3.51e-03  CLEAN
      30       132    212.1   3.51e-03  CLEAN
      36       132    256.4   3.51e-03  CLEAN
      44       132    325.2   3.51e-03  CLEAN
      62       132    380.8   3.51e-03  CLEAN
      68       132    435.6   3.51e-03  CLEAN
      88       132    491.9   3.51e-03  CLEAN
     116       132    548.3   3.51e-03  CLEAN
     118       132    550.6   3.51e-03  CLEAN
     132       132    814.9   3.51e-03  CLEAN
```

Three things to read off it.

1. **`reported` never moves.** That is the whole mechanism. Compare
   `results/mps/gh200-numcheck-map.csv`, where the reported count tracks the cap
   and the corruption tracks the reported count.
2. **Every count is clean**, including 18, 36, 44, 68, 88 and 116 — all of which
   are `CORRUPT` under MPS, and 44 of which is the SM count where cuBLAS returns
   `nan`/`inf` on identical inputs.
3. **The partition is really in force.** Throughput moves from 56 to 813
   TFLOP/s. It is not linear in SM count — the kernel was chosen for 132 SMs, so
   at small partitions it pays heavy wave quantisation — but that is a
   performance property, not a correctness one, and it is the honest cost of
   keeping the kernel choice fixed.

End to end: TimelyLLM ran a full trace at `TIMELYLLM_SM_COUNT=44`, the partition
size that MPS cannot run at all (its numcheck gate rejects 34 %), and produced
793 segments of valid MiniSpec — `tc(30);`, `?s('scissors')==True{`, `md(40);`
— not the 400-1200-character token garbage a corrupted run emits.

## What it costs to adopt

Two files, both already written:

- `timelyllm/rtengine/sm_budget.py` — reads `TIMELYLLM_SM_COUNT`, creates the
  green context, enters it. ~40 lines including the explanation.
- `timelyllm/rtengine/backend/v1.py` — two lines, calling `sm_budget.apply()`
  before `LLMEngine.from_engine_args`, so weight loading, the memory-profiling
  run and CUDA-graph capture all happen inside the partition.

No MPS control daemon, no `nvidia-smi` compute-mode change, nothing that
outlives the process, nothing that needs root. On a shared machine that is a
strict improvement on the MPS path regardless of the corruption.

`scripts/greenctx-sweep.sh` is the MPS sweep with the daemon and the numcheck
gate removed, writing a manifest that `scripts/plot-mps-latency.py` already
reads.

## The ladder

SMs are handed out in groups of 8 on this GH200, so a requested count is rounded
up to the next multiple of 8:

```
requested   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32
TFLOP/s    56  56  56  56 110 110 110 110 165 165 164 164 210 210 210 210
```

That gives 17 usable levels — 8, 16, 24, … 128, 132, i.e. 6 % to 100 % of the
GPU in 6 % steps. Compare the MPS path, which offers 2-SM granularity in
principle and, after the corruption gate, **13 usable levels clustered at the
extremes** with nothing at all between 33 % and 89 % of the GPU. The interesting
middle of the range is exactly what MPS cannot give us and green contexts can.

## Static cap versus dynamic scheduling

The two are not the same amount of work.

**Static (one partition per process): done.** This is what the sweep needs and
what is implemented above. A CUDA graph captured inside a partition stays bound
to it, so graph mode works unchanged — no `--enforce-eager` needed.

**Dynamic (change the partition while the engine runs): easy, with one real
constraint.** `sm_budget.set_sm_count(n)` switches the live partition, and the
natural place to call it is around `V1Backend.step()` — TimelyLLM's own code,
one chokepoint, in-process because the sweep already runs with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`.

The constraint is CUDA graphs. Measured:

```
replay with NO green context current
  graph captured @16 SM :   105.5 TFLOP/s
  graph captured @128 SM:   587.7 TFLOP/s
replay while the 128-SM context is current
  graph captured @16 SM :   105.5 TFLOP/s
```

A graph carries its partition with it. Switching partitions therefore means
either

- running the decode path eager (`enforce_eager=True`) — simplest, and costs
  whatever vLLM's graph speedup is worth on this model; or
- capturing one graph set per SM level — correct and fast, but multiplies
  capture time and graph memory by the number of levels, so a scheduler would
  want 3-4 levels rather than 17.

Neither is deep engineering. Estimate for a working dynamic scheduler on top of
what is already here: **1-3 days**, most of it deciding the policy (when to
change level) and measuring the switch cost, not making the mechanism work.

## What this does not fix

The cuBLAS bug is still there. Anything that goes through an MPS thread cap —
including `scripts/mps-sweep.sh` — still needs `scripts/mps-numcheck.py` in
front of it, and the H100 results already collected are still valid only at the
percentages that passed that gate. Green contexts route around the bug; they do
not repair it, and it is worth reporting to NVIDIA regardless.

Green contexts also partition *our own* process's SMs. They do not fence other
tenants off the GPU, and they never did under MPS either — the cap limits us,
not them. For a shared machine that is the behaviour we want.

Finally, `torch.cuda.green_contexts` is documented as beta in torch 2.13. The
underlying driver API (`cuGreenCtxCreate`, CUDA 12.4+) is not beta, so the risk
is an API rename, not a behaviour change.
