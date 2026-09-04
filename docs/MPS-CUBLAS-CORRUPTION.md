# Why an SM-capped MPS client computes the wrong GEMM

Measured 2026-09-03 on devkit03: GH200 480GB (132 SMs, sm_90), aarch64, driver
575.64.03, CUDA 12.9, cuBLAS 12.9.1, torch 2.13.0+cu129.

`docs/MPS-COMPUTE-SWEEP.md` establishes *that* capping SMs with
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` silently corrupts fp16/bf16 GEMM. This file
is *why*, as far as it can be determined without cuBLAS source.

## The answer in one line

The hardware is fine. **cuBLAS picks its GEMM kernel as a function of the
device's reported `multiProcessorCount`, MPS makes that number take values no
physical Hopper part ever reports, and some of the resulting kernel
configurations are wrong.**

Nothing about this is specific to TimelyLLM, vLLM, or Llama-3. It reproduces in
two tensors and a matmul.

## Evidence

### 1. The arithmetic is not corrupted -- cuBLAS is

The decisive result. Same inputs, same cap, same shapes; a hand-written Triton
tiled matmul against `torch.matmul`:

| shape (fp16) | cuBLAS @ 34% (44 SM) | Triton @ 34% (44 SM) |
|---|---|---|
| 512x4096x6144 | `nan` | 6.0e-06 |
| 1024x1024x1024 | `nan` | 1.6e-06 |
| 2048x4096x4096 | 9.5e-01 | 6.2e-06 |
| 512x14336x4096 | 9.8e-01 | 1.9e-05 |
| 256x256x256 | `inf` | 5.0e-07 |

Triton is correct to machine precision at *every* cap and shape tested. The
tensor cores, the memory system and MPS itself are all behaving. Only cuBLAS is
wrong. `scripts/mps-triton-vs-cublas.py`.

### 2. The kernel choice moves with the cap, and we can watch it

`CUBLASLT_LOG_LEVEL=4` prints the selected algo. For one fixed shape
(512x4096x6144) the choice changes with the SM count:

| pct | SM | tile | cluster | verdict |
|---|---|---|---|---|
| 10 | 12 | 128x256 | 4x1x1 | clean |
| 29 | 38 | 448x64 | 1x2x1 | clean |
| 33 | 42 | 320x128 | 2x4x1 | clean |
| 34 | 44 | 192x128 | 1x4x1 | **corrupt** |
| 38 | 50 | 128x256 | 4x2x1 | **corrupt** |
| 89 | 116 | 128x256 | 4x2x1 | clean |
| 100 | 132 | 192x128 | 1x2x1 | clean |

Tile, stage count and cluster shape are all functions of the reported SM count.
`scripts/mps-cluster-probe.py`.

### 3. It is a property of (SM count x shape), not of the percentage

33% is clean for every Llama-3-8B layer shape the gate tests and returns `inf`
for a 1024x1024x1024 GEMM. Conversely 52%, 60% and 89% fail the gate (some shape
among its 40 is wrong) yet are clean for 512x4096x6144. So "this percentage is
clean" is not a well-formed statement -- only "this percentage is clean for
these shapes" is. **This is why 30% and 33% pass the gate and still generate
wrong plans: the gate does not cover every shape the model runs.**

### 4. The failure is at output-tile granularity

At 34% on 512x4096x6144 the damage is not scattered. Exactly two contiguous
runs of bad columns, each exactly 576 wide, starting at columns 1536 and 3648;
all 512 rows affected; 18.70% of entries wrong. With the logged 192x128 tile
read as 128 rows x 192 columns, that is **24 of the 128 output tiles**, and
18.75% of the output -- matching the measured 18.70%.

So a specific subset of CTAs produced wrong output tiles while the rest were
correct. The bad values are a mixture -- 10,486 non-finite, 31,995 exactly zero,
95,684 more than 10x the true magnitude -- which is what reading uninitialised
or foreign memory looks like, **not** what a missing K-slice would look like (a
dropped split would give a consistently *smaller* magnitude, not `inf`).

### 5. What it is NOT

Each of these was tested and ruled out:

- **Not split-K / stream-K workspace.** `CUBLASLT_WORKSPACE_SIZE` of 0, 1, 32
  and 1024 KiB all give identical corruption. With a zero workspace cuBLASLt
  cannot use a workspace-backed reduction at all.
- **Not a race with other work in flight.** `torch.cuda.synchronize()`
  immediately before the GEMM does not fix it. **This contradicts
  `docs/MPS-COMPUTE-SWEEP.md`, which states a device synchronise makes the bug
  disappear** -- that does not reproduce here.
- **Not cuBLASLt versus legacy cuBLAS.** Both fail identically; they share the
  kernel library.
- **Not reduced-precision reduction.** `allow_fp16_reduced_precision_reduction`
  False still corrupts.
- **Not thread-block clusters as such.** A Triton kernel requested with
  `num_ctas` 2 and 4 stays correct at 34%. (Weak evidence: the errors were
  identical across `num_ctas`, so Triton may not have honoured the request.)
- **Not fp32/fp64, and not small GEMMs.** fp32 and fp64 are correct at every
  cap; fp16 squares at n=64 and n=128 are correct where n>=256 fails. Only the
  large-tile tensor-core path is affected.

### 6. Non-determinism

Five repeats of the identical GEMM at 34% gave relative errors
`[9.07e-01, nan, 9.07e-01, nan, 9.07e-01]` with differing checksums. Which
entries are wrong varies run to run; that the run is wrong does not.

## The mechanism, as far as the evidence reaches

cuBLAS's Hopper heuristic sizes its kernel -- tile shape, stage count, cluster
shape, and the CTA-to-tile schedule -- from `multiProcessorCount`. Real Hopper
parts report a handful of values (132 on this GH200, 114/132/144 elsewhere).
MPS reports the capped count instead, so cuBLAS is asked to emit schedules for
44, 46, 50, 54 ... SMs -- counts no shipping part has and no tuning sweep is
likely to have covered. Some of those configurations write a subset of output
tiles from memory that was never correctly produced.

The clean bands are consistent with this: 118-132 SMs (near-native) is clean,
as is the very low end where a simple kernel is chosen, while the broad middle
(68-116 SMs) is almost entirely corrupt.

**What is not established:** the specific defect inside the chosen kernel. That
needs cuBLAS source or `compute-sanitizer`, and the irregular band structure
(clean 8-16, 30-42 except 36, 62-66, 118-132) has no arithmetic explanation we
found -- no modulus separates the clean SM counts from the corrupt ones.

## Consequences for the experiment

1. The two gates in `MPS-COMPUTE-SWEEP.md` are still the right design, and the
   post-run fidelity check is still the one that decides. Nothing here weakens
   them.
2. `mps-numcheck.py` should be understood as covering *the shapes it tests*.
   Extending it to the attention and lm_head shapes would have caught 33%.
3. A Triton or CUTLASS GEMM path would sidestep the bug entirely, at the cost of
   no longer measuring the stack the paper describes. Not recommended for the
   sweep; worth knowing.
4. The finding is reportable upstream as an NVIDIA cuBLAS bug: fp16/bf16 GEMM
   returns wrong results when `multiProcessorCount` is reduced by MPS, with a
   five-line reproducer that needs no inference stack.

## Reproducers

- `scripts/mps-numcheck.py` -- the gate (Llama-3-8B layer shapes)
- `scripts/mps-diagnose.py` -- error structure, determinism, dtype/size sweep
- `scripts/mps-triton-vs-cublas.py` -- the decisive cuBLAS-vs-Triton comparison
- `scripts/mps-cluster-probe.py` -- logs the selected algo per SM count
- `scripts/mps-cluster-controlled.py` -- Triton with explicit cluster sizes
- `results/mps/gh200-numcheck-map.csv` -- all 100 percentages
