# Comparing the port against upstream on the H100

This walks through validating `port/vllm-v1` against unmodified upstream TimelyLLM,
side by side on the same machine. The GH200 cannot do this: vLLM 0.5.4 has no
aarch64 wheels, so there is no reference implementation to diff against there.
An x86 box with an H100 can run both.

**What this establishes.** TimelyLLM samples at `temperature=0`, and a task's
prompt depends only on its own input plus the plan accumulated so far — never on
what else is in the batch. So for a given task the generated plan text should be
identical on both engines. If it is, the port is faithful. Comparing plan *text*
sidesteps latency noise entirely, which matters because the two arms run different
engines on shared hardware and their timings have no reason to match.

**What it does not establish.** Nothing about performance. Do not read the timings
in these logs as a result.

---

## Step 0 — Verify the gating unknown first

Everything below assumes vLLM 0.5.4 installs on this machine. It is from August
2024 and pins `torch==2.4.0` (cu121), `vllm-flash-attn==2.6.1` and
`xformers==0.0.27.post2`. Those are the parts most likely to fight you on Hopper.
Establish this before investing in the rest:

```bash
cd /tmp && uv venv --python 3.10 probe-v0 && cd probe-v0
uv pip install --find-links https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 vllm==0.5.4
./bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__,
                 torch.cuda.get_device_name(0))"
```

If that fails and cannot be made to work, **stop**. Both comparison strategies
need a running 0.5.4, so the port cannot be validated against upstream anywhere,
and the fallback is to accept that only the relative TimelyLLM-vs-vLLM gap is
defensible with both arms on identical 0.27.1. That is a decision for your
advisor, not a workaround to improvise.

---

## Step 1 — Clone the fork and create both working trees

One clone, two working trees. `main` is an untouched mirror of upstream, which is
what makes it a valid reference.

```bash
git clone git@github.com:<you>/TimelyLLM.git
cd TimelyLLM
git checkout port/vllm-v1
git worktree add ../TimelyLLM-v0 main
```

You now have:

| path | branch | engine | python |
|---|---|---|---|
| `TimelyLLM` | `port/vllm-v1` | vLLM 0.27.1 | 3.12 |
| `TimelyLLM-v0` | `main` (upstream verbatim) | vLLM 0.5.4 | 3.10 |

Do not edit anything in `TimelyLLM-v0`. The moment you do, it stops being a
reference. The comparison script warns if that tree is dirty, on the wrong
branch, or behind `origin/main`.

---

## Step 2 — Build both environments

Each tree carries its own `pyproject.toml` and `uv.lock`, so each resolves to its
own stack. `uv` fetches the right Python for each.

```bash
cd ~/TimelyLLM-v0 && uv sync      # upstream pins: torch 2.4.0 / vLLM 0.5.4 / py3.10
cd ~/TimelyLLM    && uv sync      # ported pins:   torch 2.13 / vLLM 0.27.1 / py3.12
```

They cannot share a venv — `torch==2.4.0` and `torch 2.13` cannot coexist, and
conflicting pins of the same package cannot be expressed as optional-dependency
groups. Two environments is the whole reason for two trees.

---

## Step 3 — Get the model

Both arms must use the same weights. Download once, into the port tree; the
comparison script points both arms at it, so it is not duplicated. (`/model/*` is
gitignored, so the worktree has no copy and does not need one.)

```bash
cd ~/TimelyLLM
hf auth login                    # gated repo; needs your approved account
hf download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir model/Meta-Llama-3-8B-Instruct --exclude "original/*"
```

`--exclude "original/*"` skips the duplicate consolidated checkpoint and halves
the download to about 15 GB. If `HF_HOME` points somewhere with space, set it
before downloading.

---

## Step 4 — Preflight

```bash
cd ~/TimelyLLM
./scripts/compare-arms.py --check
```

This runs nothing. It resolves both trees and interpreters, prints the branch and
commit of each, warns if the upstream arm is not stock, and confirms the model
path exists. Every missing prerequisite is reported with the exact command that
creates it, so work through it until it is clean.

Expected output resembles:

```
working trees
  v0  /home/you/TimelyLLM-v0
      branch main @ 669f249
  v1  /home/you/TimelyLLM
      branch port/vllm-v1 @ <sha>

interpreters
  v0  /home/you/TimelyLLM-v0/.venv/bin/python
  v1  /home/you/TimelyLLM/.venv/bin/python

model  /home/you/TimelyLLM/model/Meta-Llama-3-8B-Instruct
preset exp741_timelyllm_high   run-duration 900s per arm
```

---

## Step 5 — Check the GPU is free enough

Both arms request `gpu_memory_utilization=0.8`, which is hardcoded upstream and is
the config default on the port branch. Matched on purpose — an unfair memory
split would be a confound. On an 80 GB H100 that is about 64 GB, so the GPU needs
to be mostly idle.

```bash
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv
```

The arms run **sequentially, on GPU 0**, not in parallel. Upstream sets
`CUDA_VISIBLE_DEVICES=0` by assignment at import time, overwriting anything passed
in from outside, so the arms cannot be pinned to different GPUs without editing
`main` — which would forfeit its status as a reference. If GPU 0 is occupied and
others are free, the honest options are to wait, or to accept editing that one
line and record that you did.

---

## Step 6 — Run the comparison

```bash
./scripts/compare-arms.py --preset exp741_timelyllm_high
```

It runs the upstream arm, then the ported arm, each capped at 900 s (the preset
default of 10000 s is far longer than a comparison needs). Console output from
each goes to `<tree>/timelyllm/logs/cmp-v{0,1}.console.txt`, which is where to
look if an arm dies.

To re-diff without re-running:

```bash
./scripts/compare-arms.py --skip-run --json report.json
```

---

## Step 7 — Read the result

```
tasks with output   v0=42  v1=42  common=41
identical plans     41
diverged plans      0
only in v0          1: 17
```

**`identical plans`** is the number that matters. Every task present in both arms
producing identical segment text means the port reproduces upstream.

**`only in v0` / `only in v1`** is usually a deadline-miss drop. TimelyLLM discards
tasks that miss their deadline, and which ones miss is timing-dependent, so a
handful appearing in one arm only is expected and is not by itself a port defect.
A large imbalance is worth investigating — it suggests one arm is systematically
slower.

**Exit status** is 0 when every common task matched, 1 on any divergence or if no
task produced output in both arms.

---

## Step 8 — If plans diverge

The script prints the first differing segment per task. Work through the suspects
in this order:

1. **`max_tokens` is now enforced.** Upstream's stop checkers comment out their
   `super().maybe_stop_sequence()` call and return early, which bypasses V0's
   length capping; the port does not restore that bypass, and V1 enforces the cap
   in EngineCore regardless. A divergence where the ported plan is *shorter* and
   ends at exactly 200 tokens is this. It is upstream's bug, not the port's.
2. **Attention-kernel float drift.** Kernels changed between 0.5.4 and 0.27.1, so
   greedy paths can separate after enough tokens. Diagnostic: divergences that
   appear only in *late* segments, with early ones matching. Diff the first
   segment of each task most carefully — that is where drift has had least chance
   to accumulate.
3. **Tokenizer differences.** Segment boundaries are token-quantized: the stop
   rule is consulted once per token, so a boundary interior to a token is
   invisible to it. If the two stacks tokenize identically this is a non-issue,
   but a differing `transformers` version could move boundaries. Symptom:
   segments split at different points while the concatenated plan is identical.

If the cause is still unclear, escalate to Strategy 2 in `PORT_PLAN.md` — a V0
backend behind the same interface, so one harness drives both stacks and the
engine becomes the only variable. Everything it needs is in git history at
`git show main:timelyllm/rtengine/vllm_llm_engine_usage.py`.

---

## Appendix — optional follow-up

**Exercise the memory constraint.** On an 80 GB card the KV cache is large enough
that the memory-side admission check may never bind, leaving that path untested.
The port can shrink it; upstream cannot without editing. Not a comparison run —
both arms must stay matched — but worth doing once on the port alone:

```bash
cd ~/TimelyLLM/timelyllm
../.venv/bin/python rtllm.py --preset exp741_timelyllm_high \
    --gpu-memory-utilization 0.25 --log-name mem-constrained
```

**Do not** try to demonstrate the `1376` sequential-threshold bug by running
`--run-mode sequential`. That mode is not in `infer_start`'s dispatch chain, so it
falls through every branch and returns without starting a scheduler. See the
correction in `PORT_PLAN.md` D1.
