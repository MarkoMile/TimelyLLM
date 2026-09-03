# Measuring TimelyLLM's latency against GPU compute, with NVIDIA MPS

The question: how much does time to first action depend on how much of the GPU
the engine gets? MPS answers it directly. `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
caps the fraction of the GPU's SMs a client may use, applied per process at
startup, with memory untouched -- so compute is the only variable. The KV cache
is the same size at every point and the workload trace is identical.

**Read the correctness section before running anything.** On this machine an MPS
SM cap silently makes the GPU compute the wrong answer at most percentages, and
a corrupted run still produces a log full of plausible latencies.

```bash
./scripts/mps-sweep.sh --pct "100 95 90 50 25 12 10 8 5" --arms rtllm
.venv/bin/python scripts/plot-mps-latency.py \
    --manifest results/mps/mps-manifest.csv \
    --reference timelyllm/logs/mps-rtllm-p100-r1.log
```

---

## Part A -- the correctness problem

Measured 2026-08-28: H100 80GB HBM3, driver 580.159.03, CUDA 13.0, torch
2.13.0+cu130, vLLM 0.27.1.

Under an MPS SM cap, cuBLAS returns **wrong results** for fp16/bf16 GEMM at most
thread percentages. Relative error is ~1.0 or `inf` -- not precision loss.
Nothing errors, nothing warns.

It reproduces in five lines of plain PyTorch with no inference stack involved,
and it is narrow:

| operation | result |
|---|---|
| elementwise add, reductions | correct |
| fp32 GEMM, any size | correct |
| fp16 GEMM, n=128 | correct |
| **fp16/bf16 GEMM, n>=512** | **wrong** |

So it is the large-tile Hopper tensor-core kernels specifically.

### The trigger is a busy GPU

The cap alone is not sufficient. A large fp16 GEMM launched onto an SM-capped
GPU **that already has work in flight** computes the wrong answer; the same GEMM
on an idle capped GPU is fine. Put a `.item()` -- a device synchronise --
immediately before the GEMM and the bug disappears.

This is why a single-prompt generation test passes at 66% while the real
workload produces garbage there: one prompt at a time never keeps the GPU busy,
and continuous batching always does. **Real serving is the worst case**, and any
check that synchronises before measuring will report a false OK.

### It is not monotonic in the cap

There is no safe range to stay inside, and no way to deduce one -- only to
measure. Model-shaped GEMMs, enqueued without synchronising:

```
clean      5  8  10  12        25     30*  33*  50           90  95  100
corrupt           15  18  20  22  28      35  40  45  55  60  65  66  70  75  80  85
```

`*` 30% and 33% pass the GEMM check and **still** generate wrong plans in a real
run -- see Part B. Everything else in 5..100 fails the check outright.

None of these help: `enforce_eager=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`DISABLE_ADDMM_CUDA_LT=1`.

---

## Part B -- why the two gates exist

### The pre-run gate saves time; it does not guarantee anything

`scripts/mps-numcheck.py` runs Llama-3-8B's own layer shapes across the batch
sizes prefill and decode produce, enqueued without a sync, and exits 1 if the
answers are wrong. `mps-sweep.sh` calls it before every point and skips the ones
that fail. That is worth roughly an hour of GPU time per sweep.

It is a filter, not a proof. Square matmuls miss cases the model's own shapes
catch (66%, 20%, 15% all pass a square test and still generate garbage), and
even the shape-aware check passes 30% and 33%, which are corrupt in practice.
It covers cuBLAS GEMM, not attention or normalisation.

### The post-run check is the one that decides

Decoding is greedy, so a run computing correctly must emit *the same tokens* as
the uncapped run. `plot-mps-latency.py --reference <100% log>` compares every
task's whole plan against that reference. This is exact, and it is the same
standard `compare-arms.py` holds the port to.

It is prefix-tolerant: a plan cut short by a missed deadline is a legitimate
timing effect, while diverging content is not. Healthy runs land at 99-100%; a
corrupted run collapses. Below 90% the row is rejected and excluded from the plot.

**Why exactness is necessary.** At 30% the model emitted syntactically perfect
MiniSpec that was semantically wrong -- 337 of 530 plans diverged:

| correct (100%) | at 30% |
|---|---|
| `?s('scissors')==True{g('scissors')}->False` | `?_1!=False{g(_1)}` |
| `tc(30);` | `?iv('dog')==True{tc(180);ml(50)}->False` |
| `md(40);?iv('toy')...` | `md(0.2);?iv('toy')...` |

`md(40)` becoming `md(0.2)` is a drone moving 0.2 units instead of 40. These are
valid programs a robot would execute, just the wrong ones. Every heuristic
passes that run -- ASCII output, median segment 11 characters, 3.5 segments per
task, 100% of the trace served, a believable 104 ms median. It would have
plotted as a plausible point on the curve. Only exact comparison caught it.

---

## Part C -- the harness change this needed

`rtllm.py` started the request generator and the engine as sibling processes
with no barrier, and `read_request.py` started the trace clock at its own
process start. Model-load time therefore landed inside the measured latency.

At full speed that is invisible -- loading takes ~14 s against a trace that runs
to 303 s. Under an SM cap it is not, and worse, it varies per arm, so it would
have shown up as latency.

`--wait-for-engine` makes the engine signal an `Event` once weights are loaded
and CUDA graphs captured, and the generator waits on it before starting its
clock. Every point then replays an identical trace against a ready engine. It is
**off by default**: it changes when the trace starts, so runs made with it are
not directly comparable to runs made without it.

---

## Part D -- sharing the machine

There is no scheduler on this box, so GPU choice is manual. Nothing here needs
root and nothing affects other users:

- **No compute-mode change.** `nvidia-smi -c EXCLUSIVE_PROCESS` is what would
  lock other people out of a card. It is never used and is not needed -- MPS
  works in `Default` mode.
- The daemon starts with `CUDA_VISIBLE_DEVICES=<gpu>`, so its server manages
  that one GPU and no other.
- A private `CUDA_MPS_PIPE_DIRECTORY` means only our clients join the server.
  MPS servers are per-UID, so another user's process cannot attach regardless.
- The daemon is shut down on exit, including on error, Ctrl-C and SIGTERM.

`mps-sweep.sh` refuses to start if anyone else's process is on the target GPU,
and records the GPU's process list around every run so a point that ended up
shared can be spotted and redone.

Note that `gpu_memory_utilization=0.8` reserves ~64 GB for the whole sweep. On a
shared box that squats the card; lower it if someone else needs the memory.

### Two environment traps

- **The MPS pipe directory must be a short path.** It holds a unix socket, and
  the 108-byte `sun_path` limit is easy to exceed -- a session scratchpad path
  does. The daemon then fails with exit 1 and an *empty* log. Use
  `/tmp/mps-$USER-gpu<n>`.
- **`pgrep -x nvidia-cuda-mps-control` never matches.** `pgrep -x` cannot match a
  name longer than 15 characters, and `pgrep -f` matches the shell running the
  check. Use `ps -u "$USER" -o pid=,comm=` and match `^nvidia-cuda-mps`; the
  kernel truncates `comm` to exactly that.

---

## Part E -- reading the result

The headline metric is time to first action: from a task being submitted to the
drone starting to move. That is what segmented generation exists to shorten.

The second panel -- share of the trace served -- is not decoration. The workload
is open-loop: requests arrive on a fixed wall clock whether or not the engine
keeps up. Under enough starvation tasks are dropped rather than served late,
which flatters the latency of the survivors. Latency is only readable alongside
it.

A `GARBAGE` row is not a slow measurement, it is not a measurement at all.
