# TimelyLLM → vLLM V1 port plan

Branch: `port/vllm-v1`
Target: vLLM 0.27.1 on GH200 (aarch64). Upstream pins vLLM 0.5.4 / torch 2.4.0,
which has no aarch64 wheels and cannot be installed on this machine.
Environment specifics (FlashInfer, CUDA_HOME, core pinning) live in `NOTES.md`.

**Scope order: GH200 first.** H100 access is available and will be used later for
side-by-side validation against upstream 0.5.4 (see "Validation", bottom), but no
step below blocks on it.

All claims marked [V] were verified by reading 0.27.1 source at
`/space/mm562/vllm-probe/lib/python3.12/site-packages/vllm`.
Claims marked [I] are inferences and need confirmation by measurement.

---

## Why this is tractable

[V] V1's `LLMEngine` keeps the entire drive loop TimelyLLM uses — `from_engine_args`,
`add_request`, `step()`, `has_unfinished_requests()`. `vllm/engine/llm_engine.py` is
now a 6-line alias to `vllm.v1.engine.llm_engine.LLMEngine`.

TimelyLLM never subclasses vLLM's scheduler. Its admission control works by
*withholding* `add_request` calls at application level. The scheduling logic — the
research contribution — does not touch vLLM internals and is not modified by this port.

Coupling is confined to `rtengine/vllm_llm_engine_usage.py` and
`rtengine/vllm_llm_scheduler.py`. `rtllm.py:24` is the only other vLLM-adjacent
import; "vllm" strings in `config.py` / `request/read_request.py` are run-mode names.

---

## A. Mechanical changes (no behavioral content)

**A1. Delete dead imports.** `vllm_llm_engine_usage.py:1-25` defines
`asyncio_timeout`, `ENGINE_ITERATION_TIMEOUT_S`, `AsyncEngineDeadError` — none
referenced anywhere in the repo. Same for `AsyncEngineArgs`
(`vllm_llm_scheduler.py:12`) and `random_uuid` (`:14`). Delete rather than port.

**A2. `EngineArgs` import.** [V] `vllm.engine.llm_engine` no longer re-exports it.
Use `vllm.engine.arg_utils.EngineArgs`.

**A3. Dependencies.** Two environments, not two extras — `torch==2.4.0` and
`torch 2.13.0` cannot coexist in one venv, and conflicting pins of the same package
cannot be expressed as optional-dependency groups. V1 stack becomes the default
`pyproject.toml`; the current one cannot resolve on aarch64 at all, so `uv sync` is
broken on the GH200 today. Preserve upstream pins verbatim for the H100 reference env.

**A4. Removed symbols.** [V] `vllm.sequence` now contains only `IntermediateTensors`;
`Sequence` / `SequenceStatus` are gone. The whole `vllm/engine/output_processor/`
package including `StopChecker` is gone. Both are only used by the code B1 replaces.

---

## B. The two real couplings

### B1. Custom stop rule → detokenizer subclass

[V] V1 moved text-based stopping to the frontend. EngineCore operates on token IDs
and has no text. The mechanism already exists for V1's own stop-string feature:

- `v1/engine/detokenizer.py:96` `BaseIncrementalDetokenizer.update()` accumulates
  `self.output_text`, returns a matched stop string or `None`.
- `v1/engine/output_processor.py:656` a truthy return sets
  `finish_reason = STOP`, `stop_reason = <string>`, and appends the id to
  `reqs_to_abort` because EngineCore does not know the request is done.
- `v1/engine/llm_engine.py:319` `step()` flushes `reqs_to_abort` to EngineCore.

Port: subclass `update()`, call `super()`, then evaluate the MiniSpec predicate on
`self.output_text` and return `"stop by checker"`.

Injection is one symbol: `output_processor.py:234` calls
`IncrementalDetokenizer.from_new_request()`, imported at module top — patch
`vllm.v1.engine.output_processor.IncrementalDetokenizer`.

Rewrite the three checkers (`vllm_llm_engine_usage.py:32-133`) as three predicates
over text. **Delete their hand-rolled EOS and `min_tokens` blocks** — [V] V1's
EngineCore does both (`v1/core/sched/utils.py:94`). Reimplementing them in the
detokenizer risks disagreeing with the engine about whether a request is finished.

### B2. Free-KV-block query → backend method

Three live sites, all identical:

| Line | Method |
|---|---|
| 450-451 | `schedule_timely` |
| 638-639 | `schedule_edf` |
| 878-879 | `schedule_timely_chatbot` |

[V] Replacement signal: `SchedulerStats` (`v1/metrics/stats.py:186`) carries
`kv_cache_usage: float` (0-1, from `kv_cache_manager.usage`) and `num_running_reqs`.
It arrives with every `step()` as `outputs.scheduler_stats` and crosses the process
boundary, so it works in both transport modes.

[V] `step()` discards it but passes it to
`output_processor.update_scheduler_stats()` first — wrap that one method on the
instance to capture per-step in ~5 lines. No `step()` reimplementation.

[V] Requires `log_stats=True`: `v1/core/sched/scheduler.py:2502` returns `None`
otherwise. TimelyLLM does not set `disable_log_stats`, so the default satisfies this.

[V] In-process fallback if ever needed:
`llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()`

**Two further copies are commented out** (`:572` in `schedule_timely_fixed_batch`,
`:810` in `schedule_chatbot`) — those two modes currently run with **no memory
constraint**. Leave them commented; uncommenting silently changes what those
experimental arms measure.

**`num_running()` must NOT come from `SchedulerStats`.** Found by the smoke test.
`num_running_reqs` is `len(scheduler.running)` sampled inside EngineCore's step,
but a segment stop is decided in the frontend *afterwards*: `step()` flushes the
abort only after `update_scheduler_stats` has already run
(`llm_engine.py:313-320`). So the final snapshot still counts a segment-stopped
request as running — and because no further step happens once the frontend has no
unfinished requests, it stays that way permanently. `sequential` mode would then
never admit. `engine.get_num_unfinished_requests()` has no such lag and is the
same source `has_unfinished_requests()` reads, so the backend uses that.

### B3. The `request_id` type contract [V, found by running]

Missed in the original analysis, and it broke every run mode. [V] V1 rejects any
request id that is not a `str` (`v1/engine/llm_engine.py:231`); TimelyLLM's task
ids are ints taken from the dataset's `job_id`, so the scheduler died on its very
first `add_request`. The original pass checked that `add_request` *existed*, not
what it accepts.

Fixed at the vLLM seam, not system-wide. Making ids strings throughout would break
`AgentTaskCache._get_task_by_job_id`, which matches `task_item['job_id'] == job_id`
against the dataset by value, and would change upstream's semantics everywhere —
contaminating the A/B, whose entire premise is that the arms differ only in the
engine. So the backend now owns `add_request` / `step` /
`has_unfinished_requests`, stringifies on the way in, and restores the submitted
type on the way out via a small in-flight map.

This revises a design claim made above: the interface originally exposed the raw
engine because proxying the drive loop "would only add a layer that could drift".
The id contract is precisely why that layer has to exist. Three differences
between the engines became four.

The smoke test did not catch this because it used string request ids
(`"task-0"`), so it never exercised the type the real code passes. Regression
tests now cover int ids end to end in `tests/test_segment_stop.py`.

---

## C. Semantics that must survive exactly

### C1. The stop predicate is a per-token-boundary test [V]

`MiniSpecProgram.parse` (`stop_rule.py:82-119`) resets `current_statement` (`:83`),
replays the entire accumulated text character by character, and fires only if both:

- `:94` `current_statement.parsed_code == code_instance` — the statement terminator
  is the **final character generated**
- `:95` `detect_action(code_instance)` — the text contains a recognized robot verb

The first gate is load-bearing. In V0 the checker ran once per token, so the gate was
evaluated at every boundary. In V1, `update()` receives a **list** of token ids. If
two tokens ever arrive together and the statement completed after the first,
evaluating only on the final text fails the equality gate and the sequence
**silently over-generates past its stop point** — no error, just a longer segment
and a missed deadline.

[V] Cannot currently happen: `speculative_config` defaults to `None`
(`config/vllm.py:362`) and `async_scheduling` defaults to `None`
(`config/scheduler.py:148`), so greedy decode yields exactly one output token per
request per step.

**Required:** assert `len(new_token_ids) == 1` in the ported detokenizer and fail
loudly. We depend on a vLLM default we do not control; a silent failure mode
conditional on that default is unacceptable in a deadline-driven system.
Independent argument for in-process mode.

Confirmed by measurement, not just reading. Feeding `mf(100);tc(90);` one
character at a time fires at positions 7, 8, 14, 15 — i.e. at both `)` and `;`,
since `Statement.parse:153` treats `)`, `;` and `}` alike, so a segment actually
ends at the first `)`. Evaluating the same rule once on the nine-character
prefix `mf(100);t` returns **False**: the boundary at 7 is invisible and
generation runs past it with no error. That is exactly the failure the assertion
prevents.

### C1b. Segment boundaries are token-quantized [V, measured]

The rule is consulted once per token, so a boundary *interior* to a token is
invisible to it. Measured on Llama-3-8B: the first segment of a real plan is
`?s('scissors')==True{`. At character granularity the rule fires at position 14,
on `?s('scissors')` — but the tokenizer bundles the `)` with the characters after
it, so the earliest boundary the engine can act on is position 21, seven tokens in.

This is faithful, not a porting artefact: V0 consulted its checker once per token
too. But it means **where a plan breaks into segments is partly a property of the
tokenizer**, and therefore moves if the model does. That bears directly on the
pending Qwen-vs-Llama substitution, and on `lmax`, which is denominated in tokens
(10 for TypeFly; the measured segments were 7 and 5 tokens).

Practical consequence for tooling: a validator that replays the rule at character
granularity will disagree with the engine and look like a port bug.
`scripts/smoke-test.py` replays at token boundaries for exactly this reason.

### C2. `MiniSpecProgram` is safe to make per-request [V]

V0 shares one interpreter across all concurrent sequences
(`vllm_llm_engine_usage.py:35`), which looks like cross-request contamination.
It is not: `current_statement` is reset every call (`:83`), `statements.append` is
commented out (`:101`), `depth`/`finished` are only written under `sub_flag=True`
(`:109-117`) which the top-level call never uses, and nothing on this path writes
`env`. Every mutable field is per-call or dead.

Per-request instances in V1 are therefore behavior-identical. State this in the
commit message — a reviewer will otherwise flag it as a behavior change.

### C3. `output.prompt_token_ids` is load-bearing [V]

`TaskDetails._update_input` (`:48-57`) builds the resume prompt as
`prompt_token_ids + token_ids` on the first segment, appending thereafter. This is
what makes segmented generation work. Confirmed present and populated on V1's
`RequestOutput`. Add an assertion on the first segment — if it were `None` the
failure would look like a model quality problem, not a plumbing bug.

### C4. The seven `stop_reason` checks change zero characters [V]

`:531, 593, 699, 764, 835, 937`. [V] `_new_completion_output` sets
`stop_reason=stop_reason if finished else None`, so V1 propagates the detokenizer's
returned string verbatim into `outputs[0].stop_reason`. This seam is what keeps the
entire scheduling layer untouched. Worth an explicit test.

### C5. Request-id reuse across segments [V]

`_process_seg_output` re-queues the same `task_id` for the next segment.
`OutputProcessor._finish_request` pops `request_states` synchronously during
`process_outputs`, so re-adding is safe. **But** `output_processor.py:534`
reinterprets a *live* duplicate id as a streaming-input update rather than erroring —
silent wrong behavior if the abort has not landed. In-process makes the abort
synchronous inside `step()`. Another argument for `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

---

## D. GH200 practicality

### D1. Replace `1376` with the semantic it encodes

Sites: `:456, 644, 882` — `gpu_free_threhold = 1376 if run_mode == "sequential" else 0`.

[I] Arithmetic: 4090 24GB, ~16GB weights, ~128KB KV/token for an 8B GQA model,
16 tokens/block → ≈1,500 total blocks. So `free > 1376` means "the cache is
essentially empty", i.e. the intent of `sequential` mode is **run one request at a
time**. Confirm by printing the actual block count at startup on both machines.

Partly confirmed: the GH200 reports **44,652 KV blocks** at
`gpu_memory_utilization=0.10` with a 0.5B model — 32x upstream's 1376 while using
a *tenth* of the memory fraction. Whatever 1376 meant on a 4090, it is
meaningless here, and a rescaled constant would have left `sequential` mode
silently batching. Still worth printing the count on the 4090-equivalent config to
close out the arithmetic.

Do **not** rescale the constant. Express the intent directly:

- sequential → `num_running_reqs == 0`  [V] available in `SchedulerStats`
- other modes → `free > 0` becomes `kv_cache_usage < 1.0`  — exactly equivalent,
  needs no total-block count, no rounding

This is the one place to change the *form* of the code in order to keep the *spirit*:
a transliterated `1376` on a 96GB GH200 would leave sequential mode silently
non-sequential — no crash, no warning, it would just stop being a sequential
baseline.

**Correction, found while writing the H100 runbook: the `sequential` branch is
unreachable.** `infer_start` dispatches on `run_mode` through a chain of
`elif`s covering `vllm{,-edf,-fixed,-stream,-chatbot,-chatbot-stream}` and
`rtllm{,-fixed,-edf,-edf-fixed,-fcfs-fixed,-chatbot,-chatbot-timely}` — and no
`sequential`. Passing `--run-mode sequential` falls through every branch and
returns without starting a scheduler, so `run_mode == "sequential"` inside
`_admit_memory_ok` can never be true.

That does not change the code: expressing the intent as `num_running() == 0` is
still the right form, and it is what a future reachable sequential mode should
use. But it does downgrade the practical stakes — the live path is the
`kv_has_free()` branch, which the three timely/EDF schedulers do exercise. And it
removes a proposed H100 experiment: you cannot demonstrate the `1376`
portability bug by running sequential mode, because that mode does not run.

It stands on its own as a finding about the published artifact: a run mode
referenced in the scheduler cannot be dispatched from any entry point.

### D2. `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` at `vllm_llm_scheduler.py:21`

Import-time global side effect. Prevents pinning around the Aerial cuBB workload on
the GH200 and would break the H100 side-by-side plan (both arms on GPU 0).
Respect an already-set value.

### D3. Launch wrapper

`VLLM_USE_FLASHINFER_SAMPLER=0`, unset `CUDA_HOME`, `taskset -c 0-3,65-71`, plus
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Commit these to the branch rather than relying
on shell history.

### D4. Hardcoded engine sizing

`gpu_memory_utilization=0.8` and `max_model_len=4000` at `:136-137`. On 96GB this
gives a far larger KV cache than the paper's regime, so the memory constraint may
never bind and that path goes untested. Make both configurable so the cache can be
dialed to 4090-equivalent size and the admission logic actually exercised.

Confirmed: with 44,652 blocks, a short request registers
`kv_cache_usage = 2.24e-05`. `kv_has_free()` will be true essentially always, so
the memory constraint is inert at this scale — it has to be dialed down before
that path means anything.

Separately, this GPU is **heavily shared**: 76.9 of 97.9 GB was already resident
in other processes during the smoke run, leaving ~20 GB. `gpu_memory_utilization`
is a fraction of *total* memory, so the 0.8 default would request ~78 GB and fail.
Real runs need either exclusive access or a utilization set against what is
actually free.

---

## E. Deliberately not changed

- **`max_tokens` enforcement** returns, because V1 checks it in EngineCore. Upstream
  bypassed it: every V0 checker comments out its `super().maybe_stop_sequence()` call
  and returns early, making the `max_model_len=1024` ctor arg inert. Do not restore
  the bypass — but log when the cap fires, to separate "hit length cap" from "stop
  rule fired" in results. [I] Not yet confirmed against 0.5.4's `StopChecker` source.
- **Chunked prefill** — [V] `enable_chunked_prefill` defaults `True` in V1
  (`config/scheduler.py:74`), was `False` in 0.5.4. It changes step granularity, which
  perturbs the `token_speed = finished_token_num / elapsed` estimator driving the
  latency constraint. Leave at V1 default, expose the flag, decide from H100
  measurements rather than by argument.
- **The commented-out memory constraints** in two modes (B2).
- **The scheduling math** — `_priority_calculate`, `_time_diff_calcualte`,
  `_update_queue_priority`, the segment re-submission loop. This is the research
  contribution and is untouched. The backend split exists so it stays byte-identical
  across both engine versions.

---

## Commit sequence (GH200-first)

1. **DONE — Backend interface + V1 implementation**, under `rtengine/backend/`.
   Interface designed so a V0 backend can be added later without restructuring,
   keeping the x86 A/B available at no extra cost.
2. **DONE — Threshold semantics (D1)**, folded into step 1: the free-block query
   cannot be replaced without simultaneously deciding what the threshold means, so
   `_admit_memory_ok()` encodes the intent rather than transliterating `1376`.
3. **DONE — Dependencies (A3) and config/environment plumbing (D2-D4).**
   `pyproject.toml` now declares only what TimelyLLM imports plus `vllm==0.27.1`;
   upstream's ~40 pinned transitive dependencies are what made it unresolvable.
   Upstream pins preserved under `reference/`. Engine sizing is configurable;
   `scripts/run-gh200.sh` carries the required environment.
4. **DONE — Tests, and the port verified on hardware.**
   `tests/test_segment_stop.py`: 15 checks, no GPU needed. `scripts/smoke-test.py`:
   11 checks, run on the GH200 against Qwen2.5-0.5B-Instruct — all passing. The
   detokenizer rebind lands (`stop_reason == "stop by checker"`, stopping at the
   rule rather than `max_tokens`), `SchedulerStats` arrives with real readings,
   `prompt_token_ids` is populated, and a request id can be reused to resume a
   second segment. The smoke test uses a synthetic "stop after N tokens" rule, so
   this exercises the plumbing independently of the model emitting valid MiniSpec.
   It also caught the `num_running()` lag documented in B2.

   The real MiniSpec rule has since been run against Llama-3-8B-Instruct on the
   real TypeFly prompt and a real dataset task (`--real-rule`). It produced
   `?s('scissors')==True{g('scissors')}->False`, segmented into
   `?s('scissors')==True{` and `g('scissors')`, each confirmed to stop on the
   token where the rule first fires, followed by a natural EOS finish carrying
   `stop_reason=None` — which also confirms the `stop_terminated` guard does not
   relabel a natural ending as a segment boundary.

   **The full scheduler loop now runs.** 42 agents, Llama-3-8B at
   `gpu_memory_utilization=0.20` (1.84 GiB KV cache, 15,040 tokens — close to the
   4090's regime, so the memory constraint can actually bind), 150 s: 611 segments
   across 190 tasks, between 2 and 5 segments each. Sample plan, in order:
   `md(40);` / `?iv('toy')==True{` / `g('toy')}` / `->False`. That is
   segment-wise generation with resumption working end to end through the real
   entry point.

   Getting there took two bugs, both found by running rather than reading — see
   B3 for the `request_id` contract, and NOTES.md for the pre-existing
   `--agent-num` limitation that it was hiding.
5. *(deferred)* **V0 backend** behind the same interface, for the x86 comparison.

Backend surface — only three things differ between V0 and V1; `add_request`,
`step()`, `has_unfinished_requests()` are already identical:

    class EngineBackend:
        engine                                # raw LLMEngine; drive loop calls directly
        def kv_has_free(self) -> bool         # replaces `free > 0`
        def num_running(self) -> int          # replaces the sequential threshold
        # stop rule installed at construction

---

## Validation (deferred, H100 access confirmed)

TimelyLLM samples at `temperature=0` — greedy, therefore deterministic given the same
model and prompt. So the port can be validated by **diffing segment text and
boundaries** against upstream 0.5.4 on identical hardware, rather than by comparing
latency distributions and arguing about noise. That directly tests C1, the piece
least verifiable by reading.

Caveat: attention kernels changed between 0.5.4 and 0.27.1, so floating-point drift
can eventually diverge a greedy path. Segments are short (`lmax` 10-20), so they
should match; treat divergence as something to investigate, and diff the *first*
segment most carefully.

(An earlier draft proposed demonstrating the `1376` bug by running `sequential`
mode on the H100. That mode is undispatchable — see the correction in D1 — so the
experiment does not exist. The block-count evidence from `smoke-test.py` stands on
its own.)

### Strategy 1 — branch-level A/B, no new code in either tree

`scripts/compare-arms.py` runs one preset through each branch's own entry point and
diffs the plans. TimelyLLM already logs every segment
(`Output for task <id>: <text>, time: <t>`), so the logs are the comparison surface
and neither tree needs a shared harness. Stdlib only — it never imports vllm.

It compares the **sequence of segment texts per task**, never timings. Sampling is
greedy and a task's prompt depends only on its own input plus the plan accumulated
so far — never on what else is in the batch — so the text should be reproducible
even though the scheduling is not.

    git clone <fork> TimelyLLM && cd TimelyLLM
    git worktree add ../TimelyLLM-v0 main      # upstream arm, verbatim
    # one venv per tree: 3.10 for v0, 3.12 for v1
    ./scripts/compare-arms.py --check          # preflight; names what is missing
    ./scripts/compare-arms.py --preset exp741_timelyllm_high

Preflight reports each missing prerequisite with the exact command to fix it, and
warns if the v0 tree is not stock upstream (wrong branch, behind origin/main, or
dirty) — the arm is only a valid reference if it is.

The arms run **sequentially, not in parallel**. Upstream sets
`CUDA_VISIBLE_DEVICES=0` by assignment at import time, overwriting anything passed
in, so they cannot be pinned to different GPUs without editing `main` — which would
stop it being stock upstream. Leave it; the runs are just serialized.

One model directory is shared by both arms via `--model-path`, so the weights are
not duplicated per tree (the worktree has none — `/model/*` is gitignored).

Interpreting the result: a task present in only one arm is usually a deadline-miss
drop, which is timing-dependent and not by itself a port defect. Diverging *plans*
for a task in both arms is the real signal, and the suspects in order are the
`max_tokens` cap now being enforced (E), attention-kernel float drift, and
tokenizer differences.

### Strategy 2 — shared harness via the V0 backend

Only needed if Strategy 1 diverges and the cause is unclear. Implement
`backend/v0.py` against `EngineBackend`, reusing the same `segment_rules.py`
predicates, so one harness drives both stacks and the engine is the only variable.
Everything it needs is in git history:
`git show main:timelyllm/rtengine/vllm_llm_engine_usage.py`.

Also set `gpu_memory_utilization` low enough on H100 to make the KV cache
4090-sized, or the memory constraint will not bind and that path stays untested.
