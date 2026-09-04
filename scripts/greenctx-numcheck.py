#!/usr/bin/env python3
"""Does CUDA green-context SM partitioning trip the same cuBLAS bug as MPS?

Background.  Capping a client's SMs with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
silently corrupts Hopper tensor-core GEMM at most SM counts (see
docs/MPS-CUBLAS-CORRUPTION.md).  That blocks SM scheduling for TimelyLLM.

Green contexts (CUDA 12.4+, exposed by torch.cuda.green_contexts) partition SMs
*inside one process, at run time*, with no MPS daemon involved.  If they give
the same arithmetic as the full device at every partition size, they are a
drop-in replacement for the MPS cap -- and a better one, since the partition can
be changed between requests instead of at process start.

Run with no arguments.  Exit 0 if every partition size is numerically clean.
"""
import sys
import torch
from torch.cuda.green_contexts import GreenContext, SUPPORTED

TOL = 2e-2                      # same threshold as scripts/mps-numcheck.py
LAYERS = [("qkv", 4096, 6144), ("o_proj", 4096, 4096),
          ("gate_up", 4096, 28672), ("down", 14336, 4096)]
MS = (1, 8, 64, 512, 2048)

# SM counts to probe.  Chosen to straddle the MPS clean/corrupt boundaries
# measured in results/mps/gh200-numcheck-map.csv, so a difference in behaviour
# between the two mechanisms shows up immediately.
SM_COUNTS = [8, 16, 18, 30, 36, 44, 62, 68, 88, 116, 118, 132]


def battery():
    """Worst relative error over Llama-3-8B's linear layers.  Returns
    (worst, description).  Identical arithmetic to mps-numcheck.py."""
    torch.manual_seed(0)
    worst, case = 0.0, ""
    for dt in (torch.float16, torch.bfloat16):
        for m in MS:
            for name, k, n in LAYERS:
                a = torch.randn(m, k, device="cuda", dtype=dt)
                b = torch.randn(k, n, device="cuda", dtype=dt)
                ref = (a.double() @ b.double()).float()
                rel = (((a @ b).float() - ref).abs().max()
                       / ref.abs().max().clamp(min=1e-9)).item()
                if rel > worst:
                    worst, case = rel, f"M={m} {name} {str(dt).split('.')[-1]}"
    return worst, case


def throughput():
    """TFLOP/s on one large GEMM, to prove the partition is actually in force."""
    m = k = n = 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(20):
        a @ b
    end.record()
    torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1e3 / 20
    return 2 * m * n * k / secs / 1e12


def main():
    if not SUPPORTED:
        print("green contexts not supported by this torch build", file=sys.stderr)
        return 2
    torch.cuda.init()
    total = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device reports {total} SMs, torch {torch.__version__}\n")

    print(f"{'num_sms':>8} {'reported':>9} {'TFLOP/s':>8} {'worst_rel':>10}  verdict")
    base_tf = throughput()
    base_worst, _ = battery()
    print(f"{'(none)':>8} {total:>9} {base_tf:>8.1f} {base_worst:>10.2e}  "
          f"{'CLEAN' if base_worst <= TOL else 'CORRUPT'}")

    bad = []
    for nsm in SM_COUNTS:
        try:
            ctx = GreenContext.create(num_sms=nsm, device_id=0)
        except Exception as exc:                       # unsupported count, etc.
            print(f"{nsm:>8} {'-':>9} {'-':>8} {'-':>10}  create failed: {exc}")
            continue
        ctx.set_context()
        try:
            reported = torch.cuda.get_device_properties(0).multi_processor_count
            tf = throughput()
            worst, case = battery()
        finally:
            ctx.pop_context()
        ok = worst <= TOL
        if not ok:
            bad.append((nsm, worst, case))
        print(f"{nsm:>8} {reported:>9} {tf:>8.1f} {worst:>10.2e}  "
              f"{'CLEAN' if ok else 'CORRUPT  ' + case}")

    print()
    if bad:
        print("CORRUPT at:", ", ".join(str(n) for n, _, _ in bad))
        return 1
    print("every partition size numerically clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
