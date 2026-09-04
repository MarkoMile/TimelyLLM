---
status: accepted
---

# Partition SMs with CUDA green contexts, not with MPS thread percentages

SM scheduling — changing how much of the GPU TimelyLLM is allowed to use —
was blocked by an MPS bug: capping a client with
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` silently corrupts Hopper tensor-core GEMM at
most of the resulting SM counts (docs/MPS-CUBLAS-CORRUPTION.md). We adopt CUDA
green contexts (`torch.cuda.green_contexts`, driver API `cuGreenCtxCreate`,
CUDA 12.4+) as the partitioning mechanism instead.

The mechanism is the reason it works, not luck. cuBLAS selects its GEMM kernel
from the device's reported `multiProcessorCount`. An MPS cap changes that number
to values no physical Hopper part reports, and some of the kernels cuBLAS then
picks are wrong. A green context leaves the reported count at 132 and restricts
which SMs the kernel is scheduled on, so cuBLAS picks the kernel it picks on the
whole GPU — one we know computes correctly — and the bug is never reached.

## Considered options

**Whitelist the clean MPS percentages.** Trivial to implement and unsound. The
clean set is a property of the *shapes tested*, not of the percentage: 33 %
passes `mps-numcheck.py` and still generates garbage in a real run. A whitelist
derived from a probe would silently promote any shape the probe misses into a
corrupt point that looks clean. It also leaves nothing usable between 33 % and
89 % of the GPU, which is the part of the range a scheduler most needs.

**Route GEMMs away from cuBLAS.** Triton is numerically correct at every SM
count, so replacing `F.linear` would work. But matching cuBLAS's performance
with a hand-written or autotuned Triton GEMM across every shape in the model is
a large, ongoing job, and it would change the thing we are trying to measure.

**Wait for a cuBLAS fix.** Not in user scope, not on a timeline we control, and
the driver and CUDA toolkit here are fixed by the machine's admin.

**Green contexts.** Numerically clean at every partition size measured,
including all six that MPS corrupts. Needs no control daemon, no compute-mode
change, and no root, so it removes the shared-machine risk that made the MPS
sweep delicate in the first place. It is also the only option of the four that
supports changing the partition *while the process runs*, which is what SM
scheduling actually means.

## Consequences

The mechanism costs 17 usable levels (SMs come in groups of 8) against MPS's
nominal 66 — but MPS's real number after the corruption gate was 13, clustered
at the extremes. This is a gain in coverage, not a loss.

Kernel choice is now fixed at the 132-SM choice regardless of partition size, so
small partitions pay wave quantisation: 8 SMs delivers 56 TFLOP/s where linear
scaling would predict 49, and 116 SMs delivers 548 where it would predict 716.
This is a real effect on the numbers a sweep reports and must be stated in any
writeup. It is the price of not letting cuBLAS re-plan for a device shape that
does not exist.

A CUDA graph is bound to the partition it was captured under. Static capping is
therefore free — capture happens inside the partition and graph mode works
unchanged — but a scheduler that switches levels must either run eager or
capture one graph set per level.

`scripts/mps-sweep.sh` and its numcheck gate stay as they are. The H100 results
already collected remain valid at the percentages that passed the gate, and the
cuBLAS bug is still worth reporting to NVIDIA.
