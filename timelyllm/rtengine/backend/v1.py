"""vLLM V1 backend (tested against 0.27.1).

V1 split the engine in two: EngineCore schedules and runs the model on token
ids and has no text, while the frontend detokenizes and assembles outputs. A
stop rule that keys on generated *text* therefore has to live in the frontend,
which is where V1 already runs its own stop-string checking. The machinery for
"the frontend decided this request is done, tell EngineCore to release it" is
built and used by that feature:

  detokenizer.py:96          BaseIncrementalDetokenizer.update() accumulates
                             output_text and returns a stop string or None
  output_processor.py:656    a truthy return sets finish_reason=STOP and
                             stop_reason=<string>, and queues the id in
                             reqs_to_abort since EngineCore does not know
  llm_engine.py:319          step() flushes reqs_to_abort to EngineCore

So the segment rule slots in as a detokenizer subclass, and STOP_MARKER
propagates unchanged into RequestOutput.outputs[0].stop_reason.
"""

from vllm.engine.arg_utils import EngineArgs  # noqa: F401  (re-exported)
from vllm.v1.engine.llm_engine import LLMEngine

from rtengine import sm_budget
from rtengine.backend.base import STOP_MARKER, EngineBackend
from rtengine.backend.segment_rules import rule_factory


class _SegmentStopMixin:
    """Applies a TimelyLLM segment rule on top of normal V1 detokenization.

    Mixed in ahead of Fast/SlowIncrementalDetokenizer so super() reaches the
    real implementation first: stop strings and min_tokens keep working exactly
    as upstream, and the segment rule only gets a say when they did not fire.
    """

    # Set on the concrete subclasses built in _install_segment_rule.
    _make_rule = None

    def __init__(self, tokenizer, request):
        super().__init__(tokenizer, request)
        self._rule = type(self)._make_rule()
        sampling_params = request.sampling_params
        self._max_tokens = (
            None if sampling_params is None else sampling_params.max_tokens
        )

    def update(self, new_token_ids, stop_terminated):
        stop_string = super().update(new_token_ids, stop_terminated)
        if stop_string is not None:
            return stop_string
        if not new_token_ids:
            return None

        if stop_terminated:
            # EngineCore already stopped this on EOS or a stop token. The V0
            # checker tested EOS first and returned before reaching the custom
            # rule, so the segment rule must not relabel a natural ending as a
            # segment boundary -- that would re-queue a finished task.
            return None

        if len(new_token_ids) > 1:
            # The MiniSpec predicate fires only when the statement terminator is
            # the *final* character generated, so it has to be evaluated at every
            # token boundary. Greedy decode with speculative decoding and async
            # scheduling both off yields exactly one token per step, which is the
            # configuration TimelyLLM runs. If that ever stops holding, checking
            # only the last position would silently skip a segment boundary and
            # over-generate past the deadline, so fail loudly instead.
            raise RuntimeError(
                f"segment stop rule needs one token per step, got "
                f"{len(new_token_ids)}; speculative decoding or async "
                f"scheduling is enabled and the rule would miss boundaries"
            )

        num_output_tokens = self.num_output_tokens()
        if num_output_tokens < self.min_tokens:
            return None
        if self._max_tokens is not None and num_output_tokens >= self._max_tokens:
            # Length-capped by the engine this step. Not a segment boundary, and
            # labelling it as one would re-queue a task that cannot progress.
            return None

        if self._rule(self.output_text, num_output_tokens):
            return STOP_MARKER
        return None


def _install_segment_rule(make_rule):
    """Route new requests to a detokenizer carrying the segment rule.

    output_processor.py:234 constructs detokenizers through the module-level
    name IncrementalDetokenizer, so rebinding that one symbol is enough.

    The dispatch below mirrors IncrementalDetokenizer.from_new_request. That is
    the fragile part of this port: if upstream changes which detokenizer it
    picks, this copy goes stale silently. It is a five-line mirror of a stable
    factory, which is the least-bad option short of vendoring the class.
    """
    from transformers import TokenizersBackend

    from vllm.v1.engine import output_processor as _output_processor
    from vllm.v1.engine.detokenizer import (
        USE_FAST_DETOKENIZER,
        FastIncrementalDetokenizer,
        IncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    class _FastWithRule(_SegmentStopMixin, FastIncrementalDetokenizer):
        _make_rule = staticmethod(make_rule)

    class _SlowWithRule(_SegmentStopMixin, SlowIncrementalDetokenizer):
        _make_rule = staticmethod(make_rule)

    class _Factory(IncrementalDetokenizer):
        @classmethod
        def from_new_request(cls, tokenizer, request):
            if tokenizer is None:
                # No tokenizer means no text, so no segment rule is possible.
                return IncrementalDetokenizer()
            if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
                return _FastWithRule(tokenizer, request)
            return _SlowWithRule(tokenizer, request)

    _output_processor.IncrementalDetokenizer = _Factory


class V1Backend(EngineBackend):
    def __init__(self, engine_args, robot_system, segment_stop):
        if segment_stop:
            _install_segment_rule(rule_factory(robot_system))

        # Enter the SM partition before the engine is built, so weight
        # loading, memory profiling and CUDA-graph capture all happen
        # inside it.  No-op unless TIMELYLLM_SM_COUNT is set.
        sm_budget.apply()

        self.engine = LLMEngine.from_engine_args(engine_args)

        # SchedulerStats rides along with every step() and crosses the
        # frontend/EngineCore process boundary, so it works whether or not
        # EngineCore is in-process. step() drops it, but hands it to
        # update_scheduler_stats first (llm_engine.py:313), so wrapping that one
        # method captures it without reimplementing step().
        #
        # The closure holds this dict rather than self, so the engine does not
        # end up holding a reference back to the backend.
        self._stats = {}
        output_processor = self.engine.output_processor
        original = output_processor.update_scheduler_stats
        stats = self._stats

        def capture(scheduler_stats):
            if scheduler_stats is not None:
                stats["last"] = scheduler_stats
            return original(scheduler_stats)

        output_processor.update_scheduler_stats = capture

        # str(id) -> the id the caller submitted. V1 requires str request ids
        # (llm_engine.py:231) and TimelyLLM's are ints, so they are converted
        # here and converted back on the way out. Bounded by the number of
        # in-flight requests: entries are dropped as requests finish.
        self._ids = {}

    def add_request(self, request_id, prompt, params):
        key = str(request_id)
        self._ids[key] = request_id
        self.engine.add_request(key, prompt, params)

    def step(self):
        outputs = self.engine.step()
        for output in outputs:
            key = output.request_id
            if key in self._ids:
                output.request_id = self._ids[key]
                if output.finished:
                    del self._ids[key]
        return outputs

    def has_unfinished_requests(self):
        return self.engine.has_unfinished_requests()

    def kv_has_free(self):
        last = self._stats.get("last")
        if last is None:
            # Before the first step nothing is allocated.
            return True
        return last.kv_cache_usage < 1.0

    def num_running(self):
        # Deliberately not SchedulerStats.num_running_reqs. That is
        # len(scheduler.running) sampled inside EngineCore's step, and a segment
        # stop is decided in the frontend afterwards -- step() only flushes the
        # abort to EngineCore after update_scheduler_stats has already run
        # (llm_engine.py:313-320). So the last stats snapshot still counts a
        # segment-stopped request as running, and since no further step happens
        # once the frontend has no unfinished requests, it stays that way.
        #
        # The output processor's own count has no such lag: it drops a request
        # the moment the frontend finishes it, segment stops included. That is
        # also the view TimelyLLM's loop reasons about, since
        # has_unfinished_requests() reads the same source.
        return self.engine.get_num_unfinished_requests()
