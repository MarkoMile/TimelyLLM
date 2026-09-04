# Three ways to take SMs away from TimelyLLM

Written for one decision: what should a scheduler that dynamically resizes
TimelyLLM's share of the GPU be built on?

The short version. **MPS cannot do it at all** — its cap is fixed at process
start, and on Hopper it silently corrupts cuBLAS. **Green contexts can do it
today**, and are what `timelyllm/rtengine/sm_budget.py` uses. **libsmctrl is the
better long-term mechanism** and is *not* blocked by the cuBLAS bug — but it
needs real porting work before it can be trusted on a GH200.

---

## The distinction that decides everything

There are two different things a mechanism can do when you ask for fewer SMs:

1. **Tell the program it is on a smaller GPU.** The device now reports a smaller
   `multiProcessorCount`, and every library that tunes itself to the device
   re-plans for the smaller machine.
2. **Leave the program's picture of the GPU alone and restrict where its kernels
   are allowed to run.** The device still reports 132 SMs; the driver simply
   does not schedule the kernel onto the SMs you excluded.

MPS does (1). Green contexts and libsmctrl do (2).

That is the whole story of the corruption. cuBLAS chooses its GEMM kernel — tile
shape, stage count, thread-block cluster shape, split-K decomposition — from the
device's reported SM count. Under MPS that number takes values no physical
Hopper part reports (44, 68, 88…), cuBLAS plans for a machine that does not
exist, and some of the plans it produces write garbage. Measured: at 34 %
(44 SMs) `torch.matmul` returns `nan`/`inf` where a Triton kernel on the same
inputs is correct to 6e-06. Full write-up in
[MPS-CUBLAS-CORRUPTION.md](MPS-CUBLAS-CORRUPTION.md).

Mechanisms in class (2) never trigger it, because cuBLAS keeps picking the
132-SM kernel — one we know computes correctly — and the driver just runs it in
a smaller box. This is measured for green contexts
(`scripts/greenctx-numcheck.py`: clean at every partition size, including all
six that MPS corrupts) and is a prediction, not yet a measurement, for
libsmctrl.

The price of class (2) is that the kernel is no longer tuned for the partition
you actually have, so small partitions pay wave quantisation: 8 SMs delivers
56 TFLOP/s where linear scaling from 132 SMs would predict 49, and 116 SMs
delivers 548 where it would predict 716. That is a real distortion of any
latency-versus-compute curve and has to be stated. It is also unavoidable: the
only way to get a kernel tuned for 44 SMs is to tell cuBLAS it has 44 SMs, which
is precisely what breaks.

## Side by side

| | MPS thread % | Green context | libsmctrl |
|---|---|---|---|
| Mechanism | MPS server builds the client's context with an SM limit | driver splits the device's SM resource into a second context | writes a TPC-disable bitmask into the kernel launch descriptor |
| `multiProcessorCount` cuBLAS sees | **changes** | 132, unchanged | 132, unchanged |
| Hits the cuBLAS bug | **yes**, at most SM counts | no (measured) | no (measured) |
| Granularity | 2 SMs nominal — but of 63 reachable SM counts only **22 survive the GEMM gate**, and the gate is not sufficient: 2 of the 11 percentages the sweep actually ran produced garbage anyway | 8 SMs → 17 levels | 1 TPC = 2 SMs, floor of 4 TPCs → 61 levels, but see "the count is not the compute" |
| When it can change | **process start only** | any time; per context, pre-created | any time; per stream, or per next launch |
| Who it partitions | across processes, via the daemon | your process | your process, one CUDA context |
| Needs root / a daemon | a control daemon (root-owned server already parked on this box) | no | no, except GPC topology, which needs the `nvdebug` kernel module |
| Support status | official, documented, buggy here | official CUDA ≥12.4; `torch.cuda.green_contexts` is beta | reverse-engineered; upstream lists cc 3.5–8.9 and CUDA 6.5–12.6 |
| Hopper (cc 9.0) | works, corrupts | measured working | global mask measured working; **mask bits are not TPC IDs** |
| Works under MPS (shared context) | is MPS | untested | **measured working**, to within 0.5 % |
| aarch64 | working here | measured working here | global mask measured working here; per-stream mask **aborts** on non-Jetson aarch64 |
| CUDA graphs | n/a — cap never changes | partition is **baked in at capture** (measured) | untested |

## Why the MPS thread percentage is disqualified even ignoring the bug

Separate the two things MPS does. As a **partitioning knob** it is out; as a
**context provider** it is not only fine but necessary, and that distinction
carries the whole cuPHY story below.

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is read when the client creates its CUDA
context. There is no call to change it afterwards. "Dynamically change the SMs
available to TimelyLLM" would mean restarting the vLLM engine — a ~9 s model
load plus KV-cache re-profiling — on every scheduling decision. As a
partitioning mechanism it is a way to run a *sweep*, one setting per process,
and it corrupts cuBLAS at most settings besides.

What MPS alone gives you, and nothing else does, is a **shared CUDA context**
across processes. Without it a GPU runs one context at a time and time-slices
between them, so a second process is not sharing the GPU with cuPHY — it is
taking turns with it. That is why the recommended arrangement is MPS *without* a
thread percentage, with placement supplied by a mask.

It needs a control daemon either way, which on a shared machine is the part that
requires notifying other users.

## Green contexts, concretely

`cuGreenCtxCreate` (CUDA 12.4+), wrapped as `torch.cuda.green_contexts`. You ask
for *n* SMs, the driver rounds up to a multiple of 8 on this GH200 and hands
back a context; work issued while it is current runs only on those SMs.

Measured on this box: clean at 8, 16, 18, 30, 36, 44, 62, 68, 88, 116, 118 and
132 SMs, with the reported count pinned at 132 throughout, and throughput moving
56 → 813 TFLOP/s, so the partition is genuinely in force. TimelyLLM ran a full
trace at 44 SMs — the size MPS cannot run at all — and at 32, 48 and 64 SMs,
producing 1713 segments of valid MiniSpec, the same count as the MPS sweep's
clean points.

Switching partitions live is `sm_budget.set_sm_count(n)`: pop the current
context, push another. Pre-create one per level; creation is not something you
want on the critical path.

**The one real constraint is CUDA graphs.** A graph is bound to the partition it
was captured under:

```
replay with NO green context current
  graph captured @16 SM :   105.5 TFLOP/s
  graph captured @128 SM:   587.7 TFLOP/s
replay while the 128-SM context is current
  graph captured @16 SM :   105.5 TFLOP/s     <- still 16 SMs
```

So a level-switching scheduler must either run the decode path eager, or capture
one graph set per level — which argues for 3-4 levels rather than 17.

## libsmctrl, concretely

Bakita & Anderson, RTAS 2023. It co-opts debug logic in `libcuda` to write a
TPC-disable bitmask into the launch descriptor. A **set bit disables** that TPC.

```c
void libsmctrl_set_global_mask(uint64_t mask);                  // all kernels
void libsmctrl_set_stream_mask(void* stream, uint64_t mask);    // per stream
void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask);
void libsmctrl_set_next_mask(uint64_t mask);                    // next launch only
```

This is the right shape for a scheduler: per-stream and per-launch masks change
the partition between kernels with no context switch and no re-capture, at 2-SM
granularity.

**The two masks are not equally portable, and that difference decides what is
easy.** The global and next-launch masks go through a debug callback on the QMD,
which is version-independent — it reads a TMD version byte and branches, with an
explicit Hopper (TMD V04_00) path. Per-stream masking reads a hardcoded offset
into CUDA's stream struct, and that table has entries for x86 up to CUDA 12.8
and for *Jetson* aarch64 up to 12.6. On a GH200 it reaches
`abort("Not supported on non-Jetson aarch64")`.

### Measured on this GH200

It builds against the venv's CUDA 12.9 headers and the system driver unmodified:

```
make libsmctrl.so CUDA=.../site-packages/nvidia/cuda_runtime
```

`scripts/smctrl-numcheck.py` runs the same GEMM battery as
`scripts/mps-numcheck.py` under a global mask. **Clean at every TPC count from 4
to 64** — 8 to 128 SMs — with `multiProcessorCount` reported as 132 throughout.
44 SMs, where MPS returns `nan`/`inf`, is clean. The prediction holds: a
mechanism that does not lie to cuBLAS does not trip the bug.

End to end, TimelyLLM runs under it: `TIMELYLLM_TPC_COUNT=22` goes through the
same `sm_budget` seam as the green context.

Three findings from the sweep matter more than the verdict.

**1. There is a hard floor at 4 TPCs.** Below it a point does not finish: 3 TPCs
was still running after three minutes and 1 TPC after eighteen, at 100 % GPU
utilisation, where 4 TPCs completes in 1.5 seconds. The likely cause is Hopper
thread-block clusters — cuBLAS picks a 132-SM kernel whose cluster needs more
CTAs co-resident in a GPC than two or six SMs can hold. The green context's
minimum partition is 8 SMs, exactly this floor, enforced by the driver.
libsmctrl will let you ask for less and hang. **A scheduler must clamp its own
lower bound.**

**2. The count you ask for is not the compute you get.** Upstream warns that on
cc 9.0 "mask bit indexes do not directly correlate to software-visible TPC/SM
IDs… they appear to correspond to on-chip units, including disabled ones". The
throughput column measures it. The ladder is a staircase with wide flat treads —
4 through 9 TPCs all deliver 56 TFLOP/s, 47 through 57 all deliver ~491 — and it
sits well below a green context at the same nominal SM count:

| nominal SMs | green context | libsmctrl |
|---|---|---|
| 8 | 56.1 | 55.9 |
| 16 | 110.1 | 56.0 |
| 44 | 325.2 | 209.9 |
| 88 | 491.9 | 434.4 |
| 128-132 | 814.9 | 559.7 |

So a TPC budget is not a compute budget on this GPU, and "give TimelyLLM half
the GPU" cannot be implemented by setting 33 bits. Building the map is the real
work. `libsmctrl_get_gpc_info` would give the topology but needs the `nvdebug`
kernel module, which is root and out of scope, so it has to be done empirically:
a probe kernel recording `%smid` per block under single-bit masks, or simply
calibrating against measured throughput as above.

**3. Only 128 of 132 SMs are reachable.** libsmctrl's Hopper path writes `-1`
into the upper half of the TMD mask unconditionally, pinning TPCs 64+ off until
`..._global_mask_ext` and `..._next_mask_ext` exist. The 64-bit global and
next-launch masks therefore top out at 64 of this GPU's 66 TPCs.

### Under MPS -- the cuPHY case

TimelyLLM and cuPHY are separate processes, so without MPS the GPU time-slices
between their CUDA contexts and cuPHY is suspended wholesale while TimelyLLM
runs; no amount of masking helps. MPS funnels both into one server context so
their kernels execute concurrently. But MPS caps *how many* SMs a client may
use, not *which*, so on its own it leaves the two contending for the same SMs --
and a GPU does not preempt, so a cuPHY kernel needing an SM held by a long LLM
kernel simply waits. Concurrency has to come from MPS and placement from the
mask, which only works if the mask survives the trip through the MPS server.

It does. `scripts/smctrl-under-mps.sh`, with no thread percentage set anywhere:

```
                    without MPS        as an MPS client
   8 TPC             56.0 TFLOP/s        56.0 TFLOP/s
  22 TPC            210.2               209.4
  44 TPC            433.7               433.0
  64 TPC            559.8               557.0
```

Identical to within half a percent, and numerically clean at every budget. A
mask that had stopped working would read ~810 at every row, because that failure
is silent -- the kernel simply runs on the whole GPU.

End to end, TimelyLLM ran a full 1200 s trace as an MPS client masked to 22 TPCs:
1709 segments, 530/530 tasks served, **100 % output fidelity**, 70 ms median time
to first action against 69 ms for the same mask standalone. MPS costs nothing
measurable here.

So the arrangement TimelyLLM would use beside cuPHY works today:

- **MPS** for the shared context, so cuPHY is never suspended;
- **a libsmctrl mask** for placement, so TimelyLLM's kernels cannot land on
  cuPHY's SMs;
- **no `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` anywhere**, which is the one setting
  that corrupts cuBLAS.

Two things this does not settle. Joining Aerial's own MPS server needs the
admin: MPS servers are per-UID, root's pipe directory is under `/var`, and we
must never send its daemon a command -- the test above used a private daemon of
our own. And per-process global masks are enough only while each process needs a
single budget; concurrent differing budgets *inside* one process still require
the per-stream port.

### What is left before a scheduler can use it

The correctness question — the one that blocked everything — is **answered, and
answering it was cheap**: build, ctypes, reuse the existing battery. An
afternoon. What remains is not correctness:

- **Per-stream masking needs porting to CUDA 12.9 on non-Jetson aarch64.** The
  global mask cannot do per-stream scheduling, and `_ext`, which the GPU's 66
  TPCs require, exists only in the per-stream form. Upstream documents a
  brute-force search for the struct offset (`for i in …; MASK_OFF=$i
  ./libsmctrl_test_stream_mask`), and the aarch64 branch needs a non-Jetson case
  added. Mechanical, ~1 day — but it is reverse-engineered driver-struct poking
  and it will break on driver upgrades.
- **The bit→compute calibration** above. ~2 days, and it has to be redone per
  GPU model.
- **Cluster behaviour under arbitrary masks.** Finding 1 shows the failure mode
  is a hang, not an error. Every mask a scheduler can emit needs to be proven
  safe before it is used in anger, and "proven" here means measured.

Estimate for a trustworthy dynamic libsmctrl scheduler on GH200: **1-2 weeks**,
with the part that looked riskiest already done and clean.

One structural constraint to plan around: libsmctrl partitions *within a single
CUDA context*, and upstream notes that sharing a context across address spaces
is "challenging to impossible". An external scheduler process cannot mask
TimelyLLM's kernels from outside. The scheduler either lives in the engine
process, or it sends TimelyLLM a level and TimelyLLM applies its own mask. The
second is the sane design and works identically for green contexts.

## What to build

Keep the mechanism behind one seam, which already exists:

```python
sm_budget.set_sm_count(n)   # timelyllm/rtengine/sm_budget.py
```

Both backends exist: `TIMELYLLM_SM_COUNT=n` takes the green-context path,
`TIMELYLLM_TPC_COUNT=n` the libsmctrl one. Default to green contexts — official,
8-SM granularity, no calibration needed, and the sweep it unblocks already
covers the whole range MPS could not reach. Use the libsmctrl path to develop
the scheduler, because it is the one that will eventually give per-stream
control, and because having both behind one call is the only way to find out
whether 2-SM granularity is worth the porting effort and the calibration.

## Sources

- [libsmctrl](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/about/) —
  README and `libsmctrl.h`, UNC Real-Time Systems Group
- J. Bakita and J. H. Anderson, ["Hardware Compute Partitioning on NVIDIA
  GPUs"](https://www.cs.unc.edu/~jbakita/rtas23.pdf), RTAS 2023
