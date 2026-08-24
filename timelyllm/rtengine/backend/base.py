"""Engine-version-agnostic surface that the TimelyLLM scheduler needs from vLLM.

Four things differ between the V0 engine (<= 0.5.4) and the V1 engine (>= 0.11):

  * how the engine is constructed
  * how a text-based segment stop rule is installed
  * how KV-cache pressure is read back
  * the request_id type contract -- V1 rejects anything that is not a str,
    while TimelyLLM's ids are ints taken from the dataset's job_id

The last one is why the drive loop goes through this interface rather than
touching the engine directly. Converting ids here, at the one seam that faces
vLLM, keeps them ints everywhere else in TimelyLLM -- which matters because
`job_id` is matched against the dataset by value (AgentTaskCache), and because
the A/B against upstream is only meaningful if the two arms agree on
everything except the engine. Keeping the surface this small is
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

    `engine` is the raw LLMEngine, kept accessible for debugging, but the
    scheduler drives the loop through the methods below so request ids are
    translated on both edges.
    """

    engine = None

    @abstractmethod
    def add_request(self, request_id, prompt, params):
        """Enqueue a request under TimelyLLM's own id type."""

    @abstractmethod
    def step(self):
        """Advance one iteration.

        Returned RequestOutputs carry `request_id` in the type it was submitted
        with, so callers can use it to index their own bookkeeping.
        """

    @abstractmethod
    def has_unfinished_requests(self):
        """True while any submitted request is still in flight."""

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
