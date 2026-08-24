# Comparing base vLLM, original TimelyLLM, and the port — on an x86 H100

The GH200 cannot run this comparison: vLLM 0.5.4 has no aarch64 wheels, so there
is no reference implementation to diff against. An x86 machine with an H100 can
run both engines, which is the only place the port can be validated.

---

## What the experiment is

Four runs, not three. Two engines crossed with two run modes:

|  | vLLM 0.5.4 (upstream tree) | vLLM 0.27.1 (port tree) |
|---|---|---|
| `exp741_vllm_high` | base vLLM, old engine | base vLLM, new engine |
| `exp741_timelyllm_high` | **original TimelyLLM** | **ported TimelyLLM** |

The base-vLLM row is the control, and it is what makes the rest interpretable.
Without it, ported TimelyLLM beating original TimelyLLM could just as easily be
vLLM having got faster in the eighteen months between versions.

- **Across a row** — did the port change behaviour? (engine effect)
- **Down a column** — does TimelyLLM beat plain batching on that engine? (the
  paper's claim)
- **The two column-gaps against each other** — does the paper's effect survive
  the port? That is the deliverable.

Two separate questions get checked two different ways:

- **Fidelity**: does the port produce the *same plans* as upstream? Compared as
  text, by `compare-arms.py`. Greedy decoding makes this exact and immune to
  timing noise.
- **Effect**: does TimelyLLM still shorten *time to first action*? Compared as
  timing, by `summarize-runs.py`.

---

# Part A — before you leave the GH200

### A1. Commit the staged work

```bash
cd /space/mm562/TimelyLLM
git status --short          # confirm what is staged
git commit -F commit-msg.txt
rm commit-msg.txt
```

### A2. Fork on GitHub and push both branches

The upstream repo is `Neawhen/TimelyLLM` and you cannot push to it. Fork it in
the web UI (or `gh auth login` then `gh repo fork --remote=false`), then:

```bash
git remote rename origin upstream
git remote add origin git@github.com:<you>/TimelyLLM.git
git push -u origin port/vllm-v1 main
```

Both branches matter: `port/vllm-v1` is the work, and `main` is the reference
arm. `.gitignore` covers `/model/*` and `*.log`, so the 15 GB of weights and the
run logs stay out.

### A3. Confirm `main` is still pristine

The upstream arm is only a valid reference if it is upstream, untouched:

```bash
git diff --stat main upstream/main     # must print nothing
```

### A4. Nothing else

`uv.lock` is a universal lock — it carries x86_64 wheels alongside the aarch64
ones — so it will resolve on the H100 without re-locking. Your Hugging Face
account is already approved for Llama-3, but the token lives on this machine
only; you will authenticate again over there.

---

# Part B — once you are on the H100

### B0. The gate: does vLLM 0.5.4 install at all?

Do this before anything else. Everything below depends on it, and it is the one
step with real risk: 0.5.4 is from August 2024 and pins `torch==2.4.0` (cu121),
`vllm-flash-attn==2.6.1` and `xformers==0.0.27.post2`. Those are what will fight
you on Hopper.

```bash
uname -m                    # expect x86_64
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

cd /tmp && uv venv --python 3.10 probe-v0 && cd probe-v0
uv pip install --find-links https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 vllm==0.5.4
./bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__, \
    torch.cuda.get_device_name(0))"
```

**If this cannot be made to work, stop.** Both comparison strategies need a
running 0.5.4. The fallback is that only the relative TimelyLLM-vs-vLLM gap is
defensible, with both arms on identical 0.27.1 — that is a conversation with
Zongshen about what the results can claim, not something to improvise around.

### B1. Clone and create both working trees

```bash
git clone git@github.com:<you>/TimelyLLM.git
cd TimelyLLM
git checkout port/vllm-v1
git worktree add ../TimelyLLM-v0 main
```

One clone, two trees, same history. **Never edit anything in `TimelyLLM-v0`** —
the moment you do it stops being a reference, and the tooling will warn you.

### B2. Build both environments

Each tree has its own `pyproject.toml` and `uv.lock`; `uv` fetches the right
Python for each. They cannot share a venv — `torch==2.4.0` and `torch 2.13`
cannot coexist.

```bash
cd ~/TimelyLLM-v0 && uv sync      # torch 2.4.0 / vLLM 0.5.4  / py3.10
cd ~/TimelyLLM    && uv sync      # torch 2.13  / vLLM 0.27.1 / py3.12
```

### B3. Get the model, once

Both arms use the same weights; the tooling points both at one directory.

```bash
cd ~/TimelyLLM
hf auth login
hf download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir model/Meta-Llama-3-8B-Instruct --exclude "original/*"
```

`--exclude "original/*"` skips the duplicate consolidated checkpoint and halves
the download to about 15 GB.

### B4. Preflight

```bash
./scripts/compare-arms.py --check
```

Runs nothing. Resolves both trees and interpreters, prints each branch and
commit, warns if the upstream arm is not stock, confirms the model path. Every
missing prerequisite is reported with the command that creates it — work through
it until clean.

### B5. Get the GPU to yourself

All four runs must share one card with nothing else resident, at matched
`gpu_memory_utilization` (0.8 is the default in both trees, so they match). On an
80 GB H100 that is about 64 GB.

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

The arms run **sequentially on GPU 0**, not in parallel: upstream sets
`CUDA_VISIBLE_DEVICES=0` by assignment at import time, overwriting anything
passed in, and changing that would forfeit its status as a reference.

### B6. Run the grid — two commands, four runs

```bash
cd ~/TimelyLLM
./scripts/compare-arms.py --preset exp741_timelyllm_high --tag rtllm
./scripts/compare-arms.py --preset exp741_vllm_high      --tag vllm
```

Each invocation runs the preset on both trees and diffs the plans. Budget about
25 minutes total: the trace spans 34–303 s, so a full run is ~6 minutes, and
there are four.

Each prints a fidelity verdict. What you want, twice:

```
identical plans     41
diverged plans      0
RESULT: every task present in both arms produced identical plans.
```

Exit status is 0 on a clean match, 1 on any divergence.

### B7. Summarise the grid

```bash
./scripts/summarize-runs.py
```

It finds the four logs the previous step left behind and prints:

```
  engine   arm           tasks   segs  per task   1st move      p90   vs base
  ---------------------------------------------------------------------------
  0.5.4    base vLLM        38     38       1.0     166 ms   240 ms         -
  0.5.4    TimelyLLM        32    100       3.1      73 ms   110 ms     2.27x
  0.27.1   base vLLM        37     37       1.0     171 ms   246 ms         -
  0.27.1   TimelyLLM        33    101       3.1      75 ms   112 ms     2.28x
```

### B8. Reading it

**The two `vs base` numbers are the result.** If TimelyLLM's speedup over plain
batching is the same on both engines, the effect survives the port. The script
says so explicitly when they agree within 15%.

Everything else is diagnostic:

- **`identical plans` from B6** is the fidelity check, and it is the stronger of
  the two — it is exact rather than statistical. If it passes and the timings
  still differ, the difference is the engine, not your port.
- **A task in only one arm** is usually a deadline-miss drop, which is
  timing-dependent and not by itself a defect. A large imbalance means one arm is
  systematically slower.
- **`per task`** should be ~1.0 for base vLLM (whole plans) and ~3 for TimelyLLM
  (segmented). A TimelyLLM row near 1.0 means the stop rule never fired — check
  that the run really used `run_mode='rtllm'`.

### B9. If plans diverge

`compare-arms.py` prints the first differing segment per task. In order of
likelihood:

1. **`max_tokens` is now enforced.** Upstream's stop checkers comment out their
   `super().maybe_stop_sequence()` call and return early, bypassing V0's length
   capping; the port does not restore that bypass. *Symptom:* the ported plan is
   shorter and ends at exactly 200 tokens. Upstream's bug, not the port's.
2. **Attention-kernel float drift.** Kernels changed between versions, so greedy
   paths can separate after enough tokens. *Symptom:* only late segments differ.
   Diff each task's first segment most carefully.
3. **Tokenizer differences.** Segment boundaries are token-quantized — the stop
   rule is consulted once per token, so a boundary inside a token is invisible to
   it. *Symptom:* different split points, identical concatenated plan.

If the cause stays unclear, escalate to Strategy 2 in `PORT_PLAN.md`: a V0
backend behind the same interface, so one harness drives both stacks and the
engine is the only variable. What it needs is in git history at
`git show main:timelyllm/rtengine/vllm_llm_engine_usage.py`.

---

## Quick reference

```bash
# on the GH200, before leaving
git commit -F commit-msg.txt && git push -u origin port/vllm-v1 main

# on the H100
uv venv --python 3.10 /tmp/probe-v0    # B0 gate: does 0.5.4 install?
git clone <fork> TimelyLLM && cd TimelyLLM
git worktree add ../TimelyLLM-v0 main
(cd ../TimelyLLM-v0 && uv sync) && uv sync
hf auth login && hf download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir model/Meta-Llama-3-8B-Instruct --exclude "original/*"
./scripts/compare-arms.py --check
./scripts/compare-arms.py --preset exp741_timelyllm_high --tag rtllm
./scripts/compare-arms.py --preset exp741_vllm_high      --tag vllm
./scripts/summarize-runs.py
```

A 90-second sanity check of the port alone, on either machine:
`./scripts/demo.sh` runs both arms and prints the same effect in miniature.
