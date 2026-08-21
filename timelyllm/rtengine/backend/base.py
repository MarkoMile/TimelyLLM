"""Engine-version-agnostic surface that the TimelyLLM scheduler needs from vLLM.

TimelyLLM drives vLLM through add_request / step / has_unfinished_requests,
which are identical between the V0 engine (<= 0.5.4) and the V1 engine
(>= 0.11), so the drive loop is not abstracted here at all. Only three things
actually differ between the two engines:

  * how the engine is constructed
  * how a text-based segment stop rule is installed
  * how KV-cache pressure is read back

Those are what this interface isolates. Keeping the surface this small is
deliberate: the scheduling policy in vllm_llm_scheduler.py is the research
contribution, and it has to stay byte-identical across engine versions so the
two can be compared directly on the same hardware. See PORT_PLAN.md.
"""

from abc import ABC, abstractmethod


# Sentinel written into RequestOutput.outputs[0].stop_reason when generation was
# cut short at a segment boundary rather than finishing on its own. The
# scheduler tests for this exact string to tell "execute this segment and resume
# later" apart from "this task is done" -- see vllm_llm_scheduler.py:531 and the
# six sibling sites. Both engine backends must produce it verbatim.
STOP_MARKER = "stop by checker"


class EngineBackend(ABC):
    """A constructed vLLM engine plus the two signals the scheduler reads.

    `engine` is the raw LLMEngine. The scheduler calls add_request / step /
    has_unfinished_requests on it directly, unwrapped, because those calls are
    already identical across engine versions and proxying them would only add a
    layer that could drift.
    """

    engine = None

    @abstractmethod
    def kv_has_free(self):
        """True if the KV cache has room to admit another request.

        Replaces upstream's `num_free_gpu > 0`.
        """

    @abstractmethod
    def num_running(self):
        """Requests currently running in the engine.

        Used by "sequential" mode, which admits only when nothing is running.
        """
