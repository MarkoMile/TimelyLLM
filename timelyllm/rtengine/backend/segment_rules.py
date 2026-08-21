"""When is the text generated so far an executable segment?

These are the three stop conditions from the V0 CustomStopChecker family,
extracted as plain predicates over (output_text, num_output_tokens) so they can
be installed on either engine backend.

The EOS and min_tokens handling that each V0 checker hand-rolled is *not*
reproduced here. Both engines already apply those themselves before any custom
rule is consulted, and re-checking EOS in the rule risks disagreeing with the
engine about whether a request is finished.

Each rule is produced by a factory and gets one instance per request. Upstream
shared a single MiniSpecProgram across all concurrent sequences, which looks
like cross-request contamination but is not: MiniSpecProgram.parse resets
current_statement on entry (stop_rule.py:83), the statements.append is commented
out (:101), depth/finished are only written under sub_flag=True (:109-117) which
the top-level call never uses, and nothing on this path writes env. Every
mutable field is per-call or dead, so per-request instances are
behavior-identical and avoid relying on that remaining true.
"""

from rtengine.stop_rule import MiniSpecProgram


def _typefly():
    """Stop once the text ends in a complete, executable MiniSpec statement.

    MiniSpecProgram.parse fires only when the statement terminator is the final
    character generated *and* the text contains a recognised robot action verb
    (stop_rule.py:94-95). Because it keys on the final character, it has to be
    evaluated at every token boundary -- see the guard in the V1 detokenizer.
    """
    program = MiniSpecProgram()

    def rule(output_text, num_output_tokens):
        return bool(program.parse(output_text, True))

    return rule


def _fltrnn():
    """FLTRNN emits one action per statement, so a bare `;` ends a segment."""

    def rule(output_text, num_output_tokens):
        return ';' in output_text

    return rule


def _chatbot():
    """Speech output: break at a sentence end, once there is enough to say."""
    LENGTH_THRESHOLD = 10

    def rule(output_text, num_output_tokens):
        if not any(punct in output_text for punct in ('.', '!', '?')):
            return False
        return num_output_tokens > LENGTH_THRESHOLD

    return rule


_FACTORIES = {
    "typefly": _typefly,
    "fltrnn": _fltrnn,
    "chatbot": _chatbot,
}


def rule_factory(robot_system):
    """Return a zero-arg factory producing a fresh rule for one request."""
    try:
        return _FACTORIES[robot_system]
    except KeyError:
        raise ValueError(f"Unsupported robot system: {robot_system}")
