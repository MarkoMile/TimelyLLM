# Handoff: run the MPS compute sweep on the GH200

You are picking up work that was done on the H100 (`results/mps/`). The goal on
devkit04 is the same measurement: how TimelyLLM's time to first action degrades
as you take SMs away from it, using MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to
vary the compute.

Read `docs/MPS-COMPUTE-SWEEP.md` first. It is the runbook and it explains why
each guard exists. This file is only the delta: what is different here, and what
you must not assume carries over.

## The one thing that is not portable

On the H100, capping SMs with MPS **silently corrupted fp16 and bf16 GEMM**.
cuBLAS returned wrong numbers, nothing errored, and the model generated fluent
nonsense while still logging plausible latencies. Only 5, 8, 10, 12, 25, 50, 90,
95 and 100 percent computed correctly.

That was driver 580.159.03 / CUDA 13.0 / torch 2.13.0+cu130 on x86_64. **Do not
assume it reproduces here, and do not assume it does not.** The first thing to
do on the GH200 is find out, because it decides whether you get a ten-point
curve or a nine-point one.

```bash
for p in 10 20 25 30 33 40 50 60 66 70 75 80 90 100; do
  CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$USER-gpu0/pipe \
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$p CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/mps-numcheck.py; echo "$p -> $?"
done
```

Run that with the daemon already up (see below). Exit 0 is clean, 1 is corrupt.
If every percentage passes here, say so explicitly in the results — a clean
GH200 alongside a corrupt H100 localises the bug to the driver or the x86
cuBLAS path, which is a more useful finding than the latency curve.

The gate is necessary but **not sufficient**. It covers cuBLAS GEMM only. On the
H100, 30 percent passed it and still produced syntactically valid, semantically
wrong plans — `md(40)` became `md(0.2)`, `tc(30)` became `tc(180)`, 337 of 530
plans diverged, at a completely believable 104 ms. Nothing in the timing data
gave it away. `scripts/plot-mps-latency.py` catches this by comparing generated
plan text against a known-good run, which is sound because decoding is greedy: a
correct run must emit identical tokens. Always run the plotter and always read
its fidelity column. A percentage that passes numcheck is not yet trustworthy.

## What you must change before running

**Two fixes are needed. Neither is optional.**

**1. `--gpu 0`.** `scripts/mps-sweep.sh` defaults to `GPU=3`, which was the idle
card on the H100 box. Pass `--gpu 0` here, or change the default.

**2. CPU pinning is missing.** `scripts/run-gh200.sh` pins to cores
`0-3,65-71` because **cores 4-64 are isolcpus, reserved for the Aerial cuBB RAN
L1 workload that shares this GPU**. `mps-sweep.sh` invokes `rtllm.py` directly
and does *not* `taskset`, because it was written on a machine with no such
reservation. Running it unmodified here puts vLLM's threads on cores that belong
to someone else's real-time workload. Add `taskset -c "${TIMELYLLM_CORES:-0-3,65-71}"`
in front of the `"$PY" rtllm.py` invocation (around line 200) before you run
anything. The other three GH200 requirements — `unset CUDA_HOME`,
`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0` — the sweep
already sets, so those are fine.

**The venv does not transfer.** `.venv` is gitignored and this box is aarch64.
Rebuild it, or point `TIMELYLLM_PYTHON` at an existing interpreter; the sweep
hard-fails with `no interpreter at ...` rather than falling back to system
python, which is deliberate.

**The H100 manifests carry absolute paths** (`/home/mm562/TimelyLLM/timelyllm/logs/...`).
The plotter will not find those logs here. That only matters if you try to
re-plot the old data; new sweeps write their own manifest.

## The GPU is shared here, and that changes the etiquette

This is the difference that matters most for safety. On the H100, GPU 3 was
idle and `mps-sweep.sh` could reasonably demand exclusive use — it refuses to
start if any other user's process is on the target GPU. On devkit04 the Aerial
cuBB L1 workload shares this GPU by design, so that check may refuse to start,
or may pass only because cuBB happens to be down.

Do not just delete the check. Work out first whether cuBB is running, and
whether the person who owns it is fine with you loading an 8B model and 0.8
memory utilisation onto the card. **The standing instruction on this project is
that the machine is shared and other users must not be disturbed — verify
before acting rather than acting and apologising.** If cuBB is live, the honest
options are to coordinate for a window, or to drop `--util` far enough that you
are demonstrably not competing for memory, and to record in the manifest that
the GPU was shared (the `others_before` / `others_after` columns exist for
exactly this, and the plotter flags those rows).

Two things that are never acceptable, on either machine:

- **Never** `nvidia-smi -c EXCLUSIVE_PROCESS`. It needs root and it locks every
  other user off the card. MPS works fine in `Default` mode; nothing here
  requires changing compute mode.
- **Never** leave an MPS daemon running. `mps-sweep.sh` traps `EXIT INT TERM`
  and tears it down; that trap has been verified to fire on kill. If you start a
  daemon by hand, quit it by hand:
  `echo quit | CUDA_MPS_PIPE_DIRECTORY=... nvidia-cuda-mps-control`.

Scope the daemon to one GPU with `CUDA_VISIBLE_DEVICES` at start, and give it a
private `CUDA_MPS_PIPE_DIRECTORY` so only your clients join it. Keep that path
**short** — `/tmp/mps-$USER-gpu0`, not somewhere under a scratch directory. Unix
socket paths are capped at 108 bytes and the daemon dies with an empty log if
you exceed it, which costs an hour to diagnose.

Also: `pgrep -x nvidia-cuda-mps-control` never matches, because `comm` is
truncated to 15 characters by the kernel, and `pgrep -f` matches your own
checking shell. Use `ps -u "$USER" -o pid=,comm=` and match the prefix.

## Running it

```bash
./scripts/mps-sweep.sh --gpu 0 --pct "10 25 50 100" --tag gh200-pilot
```

Start with a short pilot to confirm the plumbing before committing to the full
grid; each run is about six minutes plus load. Then widen `--pct` to whatever
numcheck said is clean here. `--repeats 2` on the low percentages is worth it —
on the H100 the run-to-run spread was about 2 ms, which is small enough that a
single run at 10 percent looked like a dip until it was repeated.

Then:

```bash
.venv/bin/python scripts/plot-mps-latency.py \
    --manifest results/mps/gh200-pilot-manifest.csv
```

It must be the venv interpreter — system python has no matplotlib.

## What the answer looked like on the H100

Median time to first action was flat from 100 percent down to 25 percent (74 to
77 ms, whole trace served) and rose below that, reaching 141 ms at 5 percent.
The reading: segmented generation is latency-bound rather than throughput-bound,
so it keeps its advantage on a small fraction of a card. If the GH200 curve has
the same shape, that conclusion is about the method rather than about one
machine, which is the point of running it twice.

Full numbers in `results/mps/combined-manifest.csv`, figure in
`results/mps/mps-latency.png`.

One thing that was offered on the H100 and never run, if you want it: the same
sweep on the **base vLLM arm** (`exp741_vllm_high`), which would show whether
segmentation's advantage grows or shrinks as compute shrinks. `--arms vllm`.
