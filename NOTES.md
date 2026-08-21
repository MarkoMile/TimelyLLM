# Context
devkit04 = GH200, aarch64, 96GB HBM3e, driver 590.48.01 / CUDA 13.1.
Hosts NVIDIA Aerial cuBB (RAN L1 on the SAME GPU). isolcpus=4-64 reserved.
Usable cores: 0-3, 65-71. No sudo, no docker group.

# Working vLLM
/space/mm562/vllm-probe — vllm 0.27.1, torch 2.13.0+cu130, python 3.12
MUST SET: VLLM_USE_FLASHINFER_SAMPLER=0   (venv has nvcc 13.3.73 vs CUDA 13.0
headers; FlashInfer JIT fails on CCCL compat check)
MUST NOT SET: CUDA_HOME
Run under: taskset -c 0-3,65-71
FlashAttention 3 works (prebuilt, unaffected).

# The port — analysis 2026-08-21, verified against 0.27.1 source
TimelyLLM pins x86-only torch 2.4.0 / vLLM 0.5.4 — cannot install here.

## Verdict: shallower than it looked. No engine fork needed.
V1 LLMEngine keeps the entire drive loop TimelyLLM uses:
from_engine_args / add_request / step() / has_unfinished_requests().
vllm/engine/llm_engine.py is now a 6-line alias to
vllm.v1.engine.llm_engine.LLMEngine.
TimelyLLM never subclasses V0's Scheduler — its admission control works by
WITHHOLDING add_request calls at application level. That is why it ports.

Corrections to the earlier draft of this file:
  - scheduler_cls / SchedulerPolicy are NOT needed. EDF ordering is already
    app-level (queue.PriorityQueue of util.Task). No SchedulerInterface
    subclass required.
  - vllm_llm_scheduler.py is NOT "public API only" — see coupling #2 below.
Coupling is otherwise confined to the two rtengine files as thought;
rtllm.py:24 is the only other vLLM-adjacent import, and the "vllm" hits in
config.py / request/read_request.py are just run-mode strings.

## Import breakage in 0.27.1
DEAD CODE — just delete (vllm_llm_engine_usage.py):
  vllm.engine.async_timeout.asyncio_timeout   module gone, unused anyway
  envs.VLLM_ENGINE_ITERATION_TIMEOUT_S        still exists, unused
  AsyncEngineDeadError                         unused
  AsyncEngineArgs, vllm.utils.random_uuid      still exist, unused
REAL BREAKS:
  from vllm.engine.llm_engine import EngineArgs  -> no longer re-exported;
                                        use vllm.engine.arg_utils.EngineArgs
  vllm.sequence.Sequence / SequenceStatus        -> GONE. sequence.py now
                                        holds only IntermediateTensors.
  vllm.engine.output_processor.stop_checker.StopChecker -> GONE, whole
                                        engine/output_processor/ package removed
STILL VALID: every EngineArgs field in use (model, tensor_parallel_size,
dtype, gpu_memory_utilization, max_model_len, max_num_seqs,
enable_prefix_caching, disable_log_stats).

## Coupling #1 — custom stop checking.  OPEN QUESTION NOW ANSWERED.
V1 relocated text-based stopping to the FRONTEND detokenizer, not the engine.
  v1/engine/detokenizer.py:96  BaseIncrementalDetokenizer.update()
      accumulates self.output_text, returns matched stop string or None.
  v1/engine/output_processor.py:656  truthy return =>
      finish_reason = STOP, stop_reason = <that string>, and because
      EngineCore does not know the req is done, req_id is appended to
      reqs_to_abort.
  v1/engine/llm_engine.py:319  step() flushes reqs_to_abort to EngineCore.
=> Port: subclass update(), call super(), then run
   MiniSpecProgram.parse(self.output_text, True) and return "stop by checker".
=> _new_completion_output sets stop_reason=stop_reason if finished else None,
   so the app-level sentinel check at vllm_llm_scheduler.py:536
   (output.outputs[0].stop_reason == 'stop by checker') works VERBATIM.
=> Injection point is ONE symbol: output_processor.py:234 calls
   IncrementalDetokenizer.from_new_request(), imported at module top.
   Patch vllm.v1.engine.output_processor.IncrementalDetokenizer.
=> EOS + min_tokens handling that each V0 checker reimplemented by hand is
   now free (EngineCore check_stop, v1/core/sched/utils.py:94). The three
   checkers collapse to three predicates over output_text.

## Coupling #2 — free-KV-block admission control.  THE live V0-internals use.
vllm_llm_scheduler.py:449-451 (also 637, 877) reads
  sum(s.block_manager.get_num_free_gpu_blocks() for s in llm_engine.scheduler)
This gates the memory constraint — load-bearing, not incidental.
PREFERRED FIX: SchedulerStats.kv_cache_usage (0-1 float, v1/metrics/stats.py:198,
  set from kv_cache_manager.usage). Arrives on every step() as
  outputs.scheduler_stats. step() discards it BUT passes it to
  output_processor.update_scheduler_stats() first — wrap that one method on
  the instance to capture per-step in ~5 lines. No step() reimplementation.
  Works in BOTH transport modes.
  Requires log_stats=True (scheduler.py:2502 returns None otherwise).
  TimelyLLM does not set disable_log_stats, so the default already satisfies.
FALLBACK (in-process only):
  llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
           .block_pool.get_num_free_blocks()

## Transport mode
VLLM_ENABLE_V1_MULTIPROCESSING defaults to TRUE (envs.py:1340).
Recommend setting it to 0:
  - makes EngineCore abort synchronous inside step(), which matters for the
    segment-resubmission pattern (see below)
  - keeps the V1 scheduler reachable in-process as an escape hatch
Still source free-block signal from kv_cache_usage so the port is not welded
to in-process mode.

## Segment resubmission reuses request_id — safe in V1, with a caveat
_process_seg_output re-queues the SAME task_id with accumulated
prompt_token_ids for the next segment (TaskDetails._update_input:48).
V1 is safe: _finish_request pops request_states synchronously during
process_outputs, before the app can re-add.
CAVEAT: OutputProcessor.add_request:534 reinterprets a *live* duplicate id as
a streaming-input update rather than erroring — silent wrong behavior if the
abort has not landed. In multiproc mode that abort is async. Another reason
for in-process.
Note this pattern is why enable_prefix_caching=True matters; V1 has prefix
caching on by default.

## Behavioral deltas that will move the numbers
1. max_tokens is now ENFORCED. Every V0 checker comments out its
   super().maybe_stop_sequence() call and returns early, bypassing V0
   length-capping — the max_model_len=1024 ctor arg is inert dead weight.
   V1 enforces max_tokens / max_model_len in EngineCore regardless.
   Probably a latent bug being fixed. NOT yet confirmed against 0.5.4's
   StopChecker source — confirm before attributing any result delta.
2. gpu_free_threhold = 1376 (vllm_llm_scheduler.py:456, "sequential" mode)
   is an absolute block count tuned to a 4090. Meaningless on a 96GB GH200 at
   gpu_memory_utilization=0.8. Re-express as a fraction of total blocks —
   which kv_cache_usage gives natively.
3. Chunked prefill is ON by default in V1, OFF in 0.5.4. Changes step()
   granularity, which directly perturbs the
   token_speed = finished_token_num / elapsed estimator driving the latency
   constraint. Not a correctness break but it moves scheduling decisions.
   DECISION NEEDED: disable to match 0.5.4, or accept V1 defaults and
   re-baseline both arms.

## Validation problem (needs mentor input)
We cannot run vLLM 0.5.4 on aarch64, so there is NO ground-truth reference on
this machine to diff the port against. Either get x86 GPU time to run the
original once, or accept that only the RELATIVE TimelyLLM-vs-vLLM comparison
is defensible and both arms must run on the identical 0.27.1 engine.

# Model
Llama-3-8B-Instruct pending Meta approval. Using Qwen2.5-7B-Instruct meanwhile.
Note: model swap changes tokens-per-segment, so lmax (10 TypeFly / 20 FLTRNN)
may need retuning.
