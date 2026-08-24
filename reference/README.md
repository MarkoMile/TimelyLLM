# Upstream reference environment

`pyproject-v0.toml` is the dependency set from upstream TimelyLLM, preserved
verbatim: vLLM 0.5.4 / torch 2.4.0 / Python 3.10, x86_64 only.

It cannot be installed on the GH200 — those wheels have no aarch64 build, which
is what the port exists to work around. It is kept for the side-by-side
validation run described in PORT_PLAN.md ("Validation"): on x86 hardware, build
this environment alongside the ported one and diff segment text and boundaries
between them. TimelyLLM samples at temperature=0, so generation is deterministic
and the two can be compared directly rather than statistically.

To build it, copy `pyproject-v0.toml` to a scratch directory as
`pyproject.toml` and sync there against a Python 3.10 interpreter, keeping it
out of this tree so it does not shadow the root project.

Set `gpu_memory_utilization` low enough that the KV cache is roughly
4090-sized, or the memory constraint will not bind and that path stays untested.
