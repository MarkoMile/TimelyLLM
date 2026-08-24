"""Tests for the segment stop rule and its V1 detokenizer wiring.

Runs without a GPU or a model. The rule predicates are pure functions, and the
detokenizer mixin is exercised against a stub standing in for
BaseIncrementalDetokenizer, so what gets tested is the ported logic rather than
vLLM's. One test does touch vLLM, to confirm the factory rebind lands.

    python tests/test_segment_stop.py      # or: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "timelyllm"))

from rtengine.backend.base import STOP_MARKER
from rtengine.backend.segment_rules import rule_factory
from rtengine.backend.v1 import _SegmentStopMixin


# --------------------------------------------------------------------------
# Rule predicates
# --------------------------------------------------------------------------

def _first_fire(text, system="typefly"):
    """Index of the first prefix at which the rule fires, feeding one char at a
    time -- the stand-in for one token per step."""
    rule = rule_factory(system)()
    for i in range(1, len(text) + 1):
        if rule(text[:i], i):
            return i
    return None


def test_typefly_fires_at_statement_terminator():
    # Statement.parse treats ')' as a terminator alongside ';' and '}', so the
    # segment ends at the closing paren, before the semicolon is generated.
    assert _first_fire("mf(100);tc(90);") == 7


def test_typefly_needs_an_action_verb():
    # A syntactically complete statement with no recognised robot verb is not an
    # executable segment (stop_rule.py:95).
    assert _first_fire("x=1;") is None


def test_typefly_does_not_fire_mid_statement():
    rule = rule_factory("typefly")()
    assert not rule("mf(10", 5)


def test_fltrnn_fires_on_semicolon():
    assert _first_fire("go to the table;", "fltrnn") == 16


def test_chatbot_needs_both_punctuation_and_length():
    rule = rule_factory("chatbot")()
    assert not rule("Hi.", 3)             # sentence ends, but too short
    assert not rule("no punctuation", 50)  # long enough, no sentence end
    assert rule("Hello there, this is fine.", 26)


def test_unknown_robot_system_rejected():
    try:
        rule_factory("nosuchrobot")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# Detokenizer mixin
# --------------------------------------------------------------------------

class _Params:
    def __init__(self, min_tokens=0, max_tokens=None):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens


class _Request:
    def __init__(self, **kw):
        self.sampling_params = _Params(**kw)


class _StubBase:
    """Minimal stand-in for BaseIncrementalDetokenizer.

    Token ids are character codes, so one token is one character and token
    counts line up with string offsets.
    """

    def __init__(self, tokenizer, request):
        self.output_text = ""
        self.token_ids = []
        self.min_tokens = request.sampling_params.min_tokens
        self.stop_string = None  # what upstream's stop-string check would return

    def update(self, new_token_ids, stop_terminated):
        self.token_ids.extend(new_token_ids)
        self.output_text += "".join(chr(t) for t in new_token_ids)
        return self.stop_string

    def num_output_tokens(self):
        return len(self.token_ids)


def _probe(rule, **kw):
    class Probe(_SegmentStopMixin, _StubBase):
        _make_rule = staticmethod(lambda: rule)

    return Probe(None, _Request(**kw))


def _feed(det, text, stop_terminated=False):
    """Feed text one token at a time; return the value from the final update."""
    result = None
    for ch in text:
        result = det.update([ord(ch)], stop_terminated)
    return result


def test_mixin_marks_segment_stop():
    det = _probe(rule_factory("typefly")())
    assert _feed(det, "mf(100)") == STOP_MARKER


def test_mixin_defers_to_upstream_stop_string():
    det = _probe(lambda text, n: True)
    det.stop_string = "</s>"
    # Upstream's own stop-string match wins; the segment rule does not override it.
    assert det.update([ord("a")], False) == "</s>"


def test_mixin_ignores_empty_update():
    det = _probe(lambda text, n: True)
    assert det.update([], False) is None


def test_mixin_does_not_relabel_engine_stop():
    # EngineCore already ended this on EOS. Calling it a segment boundary would
    # re-queue a task that is actually finished.
    det = _probe(lambda text, n: True)
    assert det.update([ord("a")], True) is None


def test_mixin_respects_min_tokens():
    det = _probe(lambda text, n: True, min_tokens=3)
    assert det.update([ord("a")], False) is None
    assert det.update([ord("b")], False) is None
    assert det.update([ord("c")], False) is STOP_MARKER


def test_mixin_does_not_relabel_length_cap():
    # Hitting max_tokens is a length cap, not a segment boundary.
    det = _probe(lambda text, n: True, max_tokens=2)
    assert det.update([ord("a")], False) == STOP_MARKER
    assert det.update([ord("b")], False) is None


def test_mixin_rejects_multi_token_step():
    # The rule keys on the final character, so a step carrying two tokens could
    # step over a boundary invisibly. Fail loudly instead.
    det = _probe(lambda text, n: False)
    try:
        det.update([ord("a"), ord("b")], False)
    except RuntimeError as exc:
        assert "one token per step" in str(exc)
        return
    raise AssertionError("expected RuntimeError on a multi-token step")


def test_rule_instances_are_per_request():
    factory = rule_factory("typefly")
    a, b = factory(), factory()
    assert a is not b


# --------------------------------------------------------------------------
# vLLM wiring
# --------------------------------------------------------------------------

def test_install_rebinds_detokenizer_factory():
    from vllm.v1.engine import output_processor as op
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    from rtengine.backend.v1 import _install_segment_rule

    original = op.IncrementalDetokenizer
    try:
        _install_segment_rule(rule_factory("typefly"))
        assert op.IncrementalDetokenizer is not original
        # With no tokenizer there is no text, so no rule can apply.
        plain = op.IncrementalDetokenizer.from_new_request(None, _Request())
        assert type(plain) is IncrementalDetokenizer
    finally:
        op.IncrementalDetokenizer = original


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
