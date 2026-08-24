#!/usr/bin/env python3
"""End-to-end smoke test for the vLLM V1 backend. Needs a GPU and a model.

Checks the three things that would otherwise fail silently:

  1. The segment stop rule reaches RequestOutput as stop_reason == STOP_MARKER.
     If the detokenizer rebind did not land, generation just runs to max_tokens
     and every segment looks like a completed plan.
  2. SchedulerStats actually arrives. If the update_scheduler_stats wrapper never
     fires, kv_has_free() returns True forever and the memory-side admission
     constraint is silently disabled.
  3. A request id can be reused for the next segment, which is how TimelyLLM
     resumes a partially generated plan.

The rule used here is synthetic ("stop after N tokens") rather than the MiniSpec
one, so the result does not depend on the model producing valid MiniSpec. The
MiniSpec predicate itself is covered by tests/test_segment_stop.py.

    ./scripts/smoke-test.py --model-path ../model/Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

# Must precede the vllm import: keeps EngineCore in this process so the abort
# ending a segment lands synchronously before the id is reused.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.pop("CUDA_HOME", None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timelyllm"))

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs

from rtengine.backend.base import STOP_MARKER
from rtengine.backend.segment_rules import rule_factory
from rtengine.backend.v1 import V1Backend, _install_segment_rule

REPO = Path(__file__).resolve().parent.parent
PROMPT = REPO / "timelyllm/rtengine/prompt/prompt_drone_typefly.txt"
DATASET = REPO / "dataset/data_sample_1.json"

STOP_AFTER = 5


def stop_after_n_tokens(n):
    def make_rule():
        def rule(output_text, num_output_tokens):
            return num_output_tokens >= n
        return rule
    return make_rule


def drain(engine, seen_running):
    """Step to completion, returning the final RequestOutput."""
    final = None
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.finished:
                final = out
        seen_running.append(True)
    return final


def build_typefly_prompt(task_input, cur_plan=""):
    """Mirror RequestScheduler._input_gen: Llama-3 chat template, MiniSpec
    instruction, and any plan text generated so far appended to the assistant
    turn so generation resumes mid-plan."""
    system = PROMPT.read_text(encoding="utf-8")
    command = ("Please only generate the response with a single sentence of "
               "MiniSpec program. 'Response':")
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + system + "<|eot_id|>\n\n"
        + "<|start_header_id|>user<|end_header_id|>\n\n"
        + task_input + command + "<|eot_id|>\n\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n\n"
        + cur_plan
    )


def first_fire_token(tokenizer, token_ids):
    """Index of the first token at which a fresh MiniSpec rule fires, replaying
    the same *token* boundaries the engine saw. None if it never fires.

    Character granularity is the wrong yardstick here. The rule is consulted once
    per token, so a boundary interior to a token is invisible to it: the engine
    cannot stop at a ')' that the tokenizer bundled together with the characters
    after it. That is equally true of the V0 checker, so it is faithful rather
    than a porting artefact -- but it does mean where a segment ends is partly a
    property of the tokenizer, and so moves if the model does.
    """
    rule = rule_factory("typefly")()
    for i in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:i], skip_special_tokens=True)
        if rule(text, i):
            return i
    return None


def run_real_rule(engine, params, max_segments):
    """Generate a real plan segment by segment and check each boundary."""
    import json
    tokenizer = engine.get_tokenizer()
    task_input = json.loads(DATASET.read_text())[0]["task_input"]
    print(f"\ntask: {task_input}\n")

    failures = 0
    cur_plan = ""
    prompt = build_typefly_prompt(task_input)

    for i in range(1, max_segments + 1):
        engine.add_request("plan-0", prompt, params)
        out = None
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    out = o
        if out is None:
            print(f"segment {i}: no output"); failures += 1
            break

        text = out.outputs[0].text
        stopped_by_rule = out.outputs[0].stop_reason == STOP_MARKER
        marker = "segment" if stopped_by_rule else "final"
        print(f"  {marker} {i}: {text!r}")

        if stopped_by_rule:
            ids = list(out.outputs[0].token_ids)
            fired = first_fire_token(tokenizer, ids)
            if fired == len(ids):
                print(f"      boundary confirmed: rule first fires on token "
                      f"{fired}/{len(ids)}, the last one")
            elif fired is None:
                print("      FAIL: engine stopped where the rule never fires")
                failures += 1
            else:
                print(f"      FAIL: rule fires at token {fired} but the engine ran "
                      f"to {len(ids)}")
                failures += 1
        else:
            print(f"      finished naturally (stop_reason={out.outputs[0].stop_reason!r},"
                  f" {len(out.outputs[0].token_ids)} tokens)")
            break

        cur_plan += text
        prompt = {"prompt_token_ids": list(out.prompt_token_ids)
                  + list(out.outputs[0].token_ids)}

    print(f"\naccumulated plan: {cur_plan!r}")
    if failures:
        print(f"FAILED: {failures} segment(s)")
        return 1
    print("all segment boundaries valid")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=4000)
    ap.add_argument("--real-rule", action="store_true",
                    help="drive the real MiniSpec rule on the real TypeFly prompt "
                         "instead of the synthetic stop-after-N rule")
    ap.add_argument("--segments", type=int, default=4)
    args = ap.parse_args()

    engine_args = EngineArgs(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        enable_prefix_caching=True,
    )

    # segment_stop=False so the real MiniSpec rule is not installed by the
    # constructor; whichever rule this run wants goes in afterwards. The rebind is
    # read at add_request time, so installing it after construction is fine.
    backend = V1Backend(engine_args, robot_system="typefly", segment_stop=False)
    engine = backend.engine
    params = SamplingParams(temperature=0, skip_special_tokens=True, max_tokens=200)

    if args.real_rule:
        _install_segment_rule(rule_factory("typefly"))
        sys.exit(run_real_rule(engine, params, args.segments))

    _install_segment_rule(stop_after_n_tokens(STOP_AFTER))
    failures = []

    def check(label, ok, detail=""):
        print(f"  {'pass' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    # Total block count, for the PORT_PLAN D1 estimate that upstream's 1376 was
    # approximately the whole cache on a 4090.
    try:
        pool = engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        print(f"\ntotal KV blocks on this GPU: {pool.num_gpu_blocks}")
    except AttributeError:
        print("\ntotal KV blocks: unavailable (EngineCore not in-process)")

    print("\nsegment 1")
    engine.add_request("task-0", "Fly forward and describe what you see.", params)
    ticks = []
    out = drain(engine, ticks)

    check("request finished", out is not None)
    if out is None:
        sys.exit(1)

    check("stop_reason is the segment marker",
          out.outputs[0].stop_reason == STOP_MARKER,
          f"got {out.outputs[0].stop_reason!r}")
    check("stopped at the rule, not max_tokens",
          len(out.outputs[0].token_ids) == STOP_AFTER,
          f"{len(out.outputs[0].token_ids)} tokens")
    check("prompt_token_ids populated (needed to resume)",
          out.prompt_token_ids is not None)

    stats = backend._stats.get("last")
    check("SchedulerStats captured", stats is not None)
    if stats is not None:
        check("kv_cache_usage is a real reading",
              0.0 < stats.kv_cache_usage <= 1.0,
              f"usage={stats.kv_cache_usage:.3e}")
        check("kv_has_free() reports capacity", backend.kv_has_free() is True)
        check("num_running() drained to zero", backend.num_running() == 0)

    print("\nsegment 2 (same request id, prompt + generated so far)")
    resume = list(out.prompt_token_ids) + list(out.outputs[0].token_ids)
    engine.add_request("task-0", {"prompt_token_ids": resume}, params)
    out2 = drain(engine, ticks)

    check("resumed under the reused id", out2 is not None)
    if out2 is not None:
        check("second segment generated tokens",
              len(out2.outputs[0].token_ids) > 0,
              f"{len(out2.outputs[0].token_ids)} tokens")
        check("second segment also hit the rule",
              out2.outputs[0].stop_reason == STOP_MARKER,
              f"got {out2.outputs[0].stop_reason!r}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
