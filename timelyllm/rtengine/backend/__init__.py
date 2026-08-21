"""Engine backends for TimelyLLM.

Only the V1 backend (vLLM >= 0.11) exists today. A V0 backend for vLLM 0.5.4 is
planned behind the same interface so the port can be diffed against upstream on
x86 hardware; see PORT_PLAN.md, "Validation".
"""

from rtengine.backend.base import STOP_MARKER, EngineBackend


def make_backend(engine_args, robot_system, segment_stop):
    """Build an engine backend for the installed vLLM.

    segment_stop enables TimelyLLM's early-stop-at-segment-boundary rule. It is
    off for the plain-vLLM baseline run modes, which generate whole plans.
    """
    from rtengine.backend.v1 import V1Backend

    return V1Backend(engine_args, robot_system, segment_stop)


__all__ = ["STOP_MARKER", "EngineBackend", "make_backend"]
