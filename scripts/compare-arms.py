#!/usr/bin/env python3
"""Run one experiment preset on both engine arms and diff the plans they produce.

Strategy 1 from PORT_PLAN.md: rather than a shared in-process harness, run each
branch through its own real entry point and compare what came out. The upstream
arm is `main` (vLLM 0.5.4, unmodified) and the ported arm is this branch
(vLLM 0.27.1). TimelyLLM already logs every generated segment, so the logs are
the comparison surface and neither tree needs new code.

What is compared is the *sequence of segment texts per task*, not timings.
Sampling is greedy (temperature=0) and a task's prompt depends only on its own
input plus the plan accumulated so far, never on what else is in the batch, so
the text is expected to be reproducible even though the scheduling is not.
Timings are reported but never failed on: the two arms have different engines
and there is no reason for them to match.

Stdlib only, so it runs under any python3 -- it never imports vllm or torch.

    ./scripts/compare-arms.py --check                      # preflight, no runs
    ./scripts/compare-arms.py --preset exp741_timelyllm_high
    ./scripts/compare-arms.py --skip-run                   # re-diff existing logs
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

V1_TREE = Path(__file__).resolve().parent.parent
DEFAULT_V0_TREE = V1_TREE.parent / f"{V1_TREE.name}-v0"

OUTPUT_RE = re.compile(r"^Output for task (\d+): (.*), time: ([\d.]+)$")
ADDED_RE = re.compile(r"^Added task (\d+), time: ([\d.]+)$")


# ---------------------------------------------------------------- environment

class Missing(Exception):
    """A prerequisite is absent; the message says how to create it."""


def resolve_python(tree, override, label):
    if override:
        p = Path(override)
        if not p.exists():
            raise Missing(f"{label}: no interpreter at {p}")
        return p
    p = tree / ".venv" / "bin" / "python"
    if p.exists():
        return p
    raise Missing(
        f"{label}: no interpreter at {p}\n"
        f"  create it with:\n"
        f"    uv venv --python {'3.10' if label == 'v0' else '3.12'} {tree}/.venv\n"
        f"    cd {tree} && VIRTUAL_ENV=.venv uv pip install -e ."
    )


def resolve_v0_tree(override):
    if override:
        tree = Path(override).resolve()
        if not tree.is_dir():
            raise Missing(f"v0 tree not found at {tree}")
        return tree
    if DEFAULT_V0_TREE.is_dir():
        return DEFAULT_V0_TREE
    raise Missing(
        f"no upstream working tree at {DEFAULT_V0_TREE}\n"
        f"  create one from the untouched `main` branch with:\n"
        f"    git -C {V1_TREE} worktree add {DEFAULT_V0_TREE} main"
    )


def git(tree, *args):
    try:
        return subprocess.run(["git", "-C", str(tree), *args],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def describe_tree(tree, label):
    branch = git(tree, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    head = git(tree, "rev-parse", "--short", "HEAD") or "?"
    dirty = " (uncommitted changes)" if git(tree, "status", "--porcelain") else ""
    print(f"  {label:2}  {tree}")
    print(f"      branch {branch} @ {head}{dirty}")
    return branch


def warn_if_not_upstream(tree, branch):
    """The v0 arm is only a valid reference if it is upstream, unmodified."""
    if branch != "main":
        print(f"      WARNING: expected branch 'main', got '{branch}'. The v0 arm "
              f"is meant to be upstream verbatim.")
    upstream = git(tree, "rev-parse", "--short", "origin/main")
    head = git(tree, "rev-parse", "--short", "HEAD")
    if upstream and head and upstream != head:
        print(f"      WARNING: HEAD ({head}) differs from origin/main ({upstream}).")
    if git(tree, "status", "--porcelain"):
        print(f"      WARNING: working tree is dirty, so this is not stock upstream.")


# ---------------------------------------------------------------- running

def run_arm(label, tree, python, preset, log_name, model_path, run_duration, extra):
    """Run rtllm.py in one tree. Returns the path to its log file."""
    workdir = tree / "timelyllm"
    log_path = workdir / "logs" / f"{log_name}.log"
    console = workdir / "logs" / f"{log_name}.console.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(python), "rtllm.py", "--preset", preset, "--log-name", log_name,
           "--model-path", str(model_path)]
    if run_duration is not None:
        cmd += ["--run-duration", str(run_duration)]
    cmd += extra

    print(f"\n=== arm {label}: {' '.join(cmd)}")
    print(f"    cwd {workdir}")
    print(f"    console -> {console}")

    env = dict(os.environ)
    # Upstream sets CUDA_VISIBLE_DEVICES=0 at import time by assignment, which
    # overwrites anything passed in, so the arms cannot be pinned to different
    # GPUs from out here. They run one after another instead.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.pop("CUDA_HOME", None)

    with console.open("w") as fh:
        rc = subprocess.run(cmd, cwd=workdir, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"    arm {label} exited {rc}; see {console}")
    if not log_path.exists():
        raise Missing(f"arm {label} produced no log at {log_path}; see {console}")
    return log_path


# ---------------------------------------------------------------- comparison

def parse_log(path):
    """task_id -> {'segments': [text, ...], 'added': n, 'outputs': [(text, t)]}"""
    tasks = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = OUTPUT_RE.match(line)
            if m:
                tid, text, t = m.group(1), m.group(2), float(m.group(3))
                e = tasks.setdefault(tid, {"segments": [], "added": 0, "times": []})
                e["segments"].append(text)
                e["times"].append(t)
                continue
            m = ADDED_RE.match(line)
            if m:
                tid = m.group(1)
                tasks.setdefault(tid, {"segments": [], "added": 0, "times": []})
                tasks[tid]["added"] += 1
    return tasks


def compare(v0, v1):
    ids0, ids1 = set(v0), set(v1)
    both = sorted(ids0 & ids1, key=int)
    report = {
        "only_v0": sorted(ids0 - ids1, key=int),
        "only_v1": sorted(ids1 - ids0, key=int),
        "identical": [],
        "diverged": [],
    }
    for tid in both:
        a, b = v0[tid]["segments"], v1[tid]["segments"]
        if a == b:
            report["identical"].append(tid)
            continue
        first = next((i for i in range(max(len(a), len(b)))
                      if i >= len(a) or i >= len(b) or a[i] != b[i]), 0)
        report["diverged"].append({
            "task": tid,
            "n_v0": len(a), "n_v1": len(b),
            "first_divergence": first,
            "v0": a[first] if first < len(a) else None,
            "v1": b[first] if first < len(b) else None,
        })
    return report


def print_report(v0, v1, rep, detail):
    n_both = len(rep["identical"]) + len(rep["diverged"])
    print("\n" + "=" * 68)
    print(f"tasks with output   v0={len(v0)}  v1={len(v1)}  common={n_both}")
    print(f"identical plans     {len(rep['identical'])}")
    print(f"diverged plans      {len(rep['diverged'])}")
    if rep["only_v0"]:
        print(f"only in v0          {len(rep['only_v0'])}: "
              f"{', '.join(rep['only_v0'][:12])}"
              f"{' ...' if len(rep['only_v0']) > 12 else ''}")
    if rep["only_v1"]:
        print(f"only in v1          {len(rep['only_v1'])}: "
              f"{', '.join(rep['only_v1'][:12])}"
              f"{' ...' if len(rep['only_v1']) > 12 else ''}")

    for d in rep["diverged"][:detail]:
        print(f"\n  task {d['task']}: {d['n_v0']} segment(s) in v0, "
              f"{d['n_v1']} in v1; first differs at index {d['first_divergence']}")
        print(f"    v0: {d['v0']!r}")
        print(f"    v1: {d['v1']!r}")
    if len(rep["diverged"]) > detail:
        print(f"\n  ... {len(rep['diverged']) - detail} more (raise --detail)")

    print("\n" + "=" * 68)
    if rep["only_v0"] or rep["only_v1"]:
        print("NOTE: a task appearing in only one arm usually means it was dropped on")
        print("      a deadline miss, which is timing-dependent and not by itself a")
        print("      port defect. Compare the plans of the tasks present in both.")
    if rep["diverged"]:
        print("RESULT: plans diverge. Because sampling is greedy and a task's prompt")
        print("        does not depend on the batch, identical text was expected.")
        print("        Suspects, in order: the max_tokens cap now being enforced,")
        print("        attention-kernel float drift, and tokenizer differences.")
        return 1
    if not rep["identical"]:
        print("RESULT: inconclusive -- no task produced output in both arms.")
        return 1
    print("RESULT: every task present in both arms produced identical plans.")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--preset", default="exp741_timelyllm_high")
    ap.add_argument("--v0-tree", help=f"default: {DEFAULT_V0_TREE}")
    ap.add_argument("--v1-tree", default=str(V1_TREE))
    ap.add_argument("--v0-python", help="default: <v0-tree>/.venv/bin/python")
    ap.add_argument("--v1-python", help="default: <v1-tree>/.venv/bin/python")
    ap.add_argument("--model-path",
                    help="shared model dir; default <v1-tree>/model/"
                         "Meta-Llama-3-8B-Instruct. Passed to both arms so the "
                         "weights are not duplicated per tree.")
    ap.add_argument("--run-duration", type=int, default=900,
                    help="per-arm wall-clock cap in seconds (default 900; the "
                         "preset default of 10000 is far longer than a "
                         "comparison run needs)")
    ap.add_argument("--tag", default="cmp", help="log name prefix")
    ap.add_argument("--extra-v0", default="", help="extra args for the v0 arm")
    ap.add_argument("--extra-v1", default="", help="extra args for the v1 arm")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--skip-run", action="store_true",
                    help="diff the logs from a previous run")
    ap.add_argument("--detail", type=int, default=10,
                    help="how many diverged tasks to print in full")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    v1_tree = Path(args.v1_tree).resolve()
    try:
        v0_tree = resolve_v0_tree(args.v0_tree)
        v0_python = resolve_python(v0_tree, args.v0_python, "v0")
        v1_python = resolve_python(v1_tree, args.v1_python, "v1")
    except Missing as exc:
        print(f"setup incomplete:\n{exc}", file=sys.stderr)
        return 2

    model = Path(args.model_path).resolve() if args.model_path else \
        (v1_tree / "model" / "Meta-Llama-3-8B-Instruct")

    print("working trees")
    v0_branch = describe_tree(v0_tree, "v0")
    warn_if_not_upstream(v0_tree, v0_branch)
    describe_tree(v1_tree, "v1")
    print(f"\ninterpreters\n  v0  {v0_python}\n  v1  {v1_python}")
    print(f"\nmodel  {model}")
    if not model.exists():
        print("  WARNING: model path does not exist")
    print(f"preset {args.preset}   run-duration {args.run_duration}s per arm")

    if args.check:
        print("\npreflight only; nothing run.")
        return 0

    v0_log = v0_tree / "timelyllm" / "logs" / f"{args.tag}-v0.log"
    v1_log = v1_tree / "timelyllm" / "logs" / f"{args.tag}-v1.log"

    if not args.skip_run:
        # Sequential, not parallel: see the note in run_arm about upstream
        # overwriting CUDA_VISIBLE_DEVICES.
        try:
            v0_log = run_arm("v0", v0_tree, v0_python, args.preset,
                             f"{args.tag}-v0", model, args.run_duration,
                             args.extra_v0.split())
            v1_log = run_arm("v1", v1_tree, v1_python, args.preset,
                             f"{args.tag}-v1", model, args.run_duration,
                             args.extra_v1.split())
        except Missing as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2

    for p in (v0_log, v1_log):
        if not p.exists():
            print(f"missing log: {p}", file=sys.stderr)
            return 2

    print(f"\nparsing\n  v0  {v0_log}\n  v1  {v1_log}")
    t0, t1 = parse_log(v0_log), parse_log(v1_log)
    rep = compare(t0, t1)
    rc = print_report(t0, t1, rep, args.detail)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"preset": args.preset, "v0_log": str(v0_log), "v1_log": str(v1_log),
             **rep}, indent=2))
        print(f"\nfull report written to {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
