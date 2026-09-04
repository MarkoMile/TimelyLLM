#!/usr/bin/env python3
"""Is fp16/bf16 matmul numerically correct under a libsmctrl TPC mask?

The libsmctrl analogue of scripts/mps-numcheck.py, and the same GEMM battery, so
the two are directly comparable.  The question it answers is whether libsmctrl is
a viable basis for SM scheduling on this GH200, given that the MPS thread-percentage
cap corrupts cuBLAS at most SM counts (docs/MPS-CUBLAS-CORRUPTION.md).

The prediction is that it is clean everywhere.  MPS corrupts because it changes
the SM count the device *reports*, which is what cuBLAS keys its GEMM kernel
choice on.  libsmctrl writes a TPC-disable mask into the kernel launch
descriptor and never touches that number, so cuBLAS keeps choosing the 132-SM
kernel -- one we know computes correctly -- exactly as it does under a green
context.  This script is here to check the prediction rather than trust it.

Only the *global* mask is used.  On this platform that is the one that works:
libsmctrl's per-stream masking reads a hardcoded offset into CUDA's stream
struct and has no entry for CUDA 12.9 on non-Jetson aarch64 (it aborts with
"Not supported on non-Jetson aarch64"), whereas the global mask goes through a
version-independent debug callback on the QMD.

Two consequences of that choice, both visible in the output:
  - libsmctrl forces TPCs 64+ off in the Hopper path until the _ext variants
    exist, so this reaches 64 of the GPU's 66 TPCs -- 2 to 128 of 132 SMs.
  - a TPC is 2 SMs, so the ladder is in 2-SM steps.

Usage:
    smctrl-numcheck.py --tpcs 22          # one point, exit 1 if corrupt
    smctrl-numcheck.py --sweep            # every TPC count, CSV on stdout

Each sweep point runs in its own subprocess: libsmctrl aborts the process on an
unsupported platform rather than returning an error, and one bad point should
not take the sweep with it.
"""
import argparse
import ctypes
import os
import subprocess
import sys

TOL = 2e-2          # same threshold as mps-numcheck.py: healthy ~3e-3, corrupt ~1e0
LIB = os.environ.get("LIBSMCTRL_SO", "/space/mm562/libsmctrl/libsmctrl.so")
TPCS_MASKABLE = 64  # libsmctrl's Hopper path pins TPCs 64+ off
# Below four TPCs the point does not finish.  Measured: 4 TPCs completes in
# 1.5 s, 3 and 2 were still running after 3 minutes and 1 after 18.  The
# suspected cause is Hopper thread-block clusters -- cuBLAS picks a 132-SM
# kernel whose cluster needs more CTAs co-resident in a GPC than two, four or
# six SMs can hold.  Green contexts have a minimum partition of 8 SMs, which is
# exactly this floor; libsmctrl will happily let you ask for less and hang.
TPCS_FLOOR = 4


_lib = None


def _libsmctrl():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB)
        _lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        _lib.libsmctrl_set_global_mask.restype = None
    return _lib


def set_tpc_budget(n_tpc):
    """Allow kernels to run on TPCs 0..n_tpc-1 only, or on all of them if
    `n_tpc` is None.

    libsmctrl masks are *disable* masks -- a set bit turns a TPC off -- so the
    budget is the complement.  Note that "all of them" still means 64 of this
    GPU's 66 TPCs: libsmctrl's Hopper path pins the top two off unconditionally
    once its launch callback is installed."""
    mask = 0 if n_tpc is None else (~((1 << n_tpc) - 1)) & 0xFFFFFFFFFFFFFFFF
    _libsmctrl().libsmctrl_set_global_mask(mask)
    return mask


def battery(torch, n_tpc):
    """Worst relative error over Llama-3-8B's linear layers, with the reduced-
    precision GEMM run under a TPC budget of `n_tpc`.

    Same shapes, same dtypes and same tolerance as mps-numcheck.py, so the two
    maps compare directly.  One deliberate difference: the fp64 reference is
    computed with the mask lifted, and so is the random input generation; only
    the GEMM under test runs masked.  Everything else costs far more under a
    mask than the thing being tested -- cuBLAS and curand still pick 132-SM
    kernels and then serialise them onto the two SMs left, which ran for over 18
    minutes on a single point at one TPC -- and none of it is what we are
    testing.  The fp16/bf16 GEMM is still enqueued straight after the reference
    with no synchronisation in between, which is the condition the corruption
    needs."""
    torch.manual_seed(0)
    layers = [("qkv", 4096, 6144), ("o_proj", 4096, 4096),
              ("gate_up", 4096, 28672), ("down", 14336, 4096)]
    worst, case = 0.0, ""
    for dt in (torch.float16, torch.bfloat16):
        for m in (1, 8, 64, 512, 2048):
            for name, k, n in layers:
                set_tpc_budget(None)
                a = torch.randn(m, k, device="cuda", dtype=dt)
                b = torch.randn(k, n, device="cuda", dtype=dt)
                ref = (a.double() @ b.double()).float()
                set_tpc_budget(n_tpc)
                out = (a @ b).float()
                set_tpc_budget(None)
                rel = ((out - ref).abs().max()
                       / ref.abs().max().clamp(min=1e-9)).item()
                if rel > worst:
                    worst, case = rel, f"M={m} {name} {str(dt).split('.')[-1]}"
    set_tpc_budget(n_tpc)
    return worst, case


def throughput(torch, n_tpc):
    """TFLOP/s on one large GEMM.  A mask that is not taking effect shows up
    here as full-GPU throughput at a small TPC budget.

    The repeat count follows the budget: a 132-SM kernel serialised onto two SMs
    takes long enough that a fixed 20 repeats dominate the whole point."""
    m = k = n = 4096
    reps = max(3, min(20, n_tpc))
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        a @ b
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        a @ b
    end.record()
    torch.cuda.synchronize()
    return 2 * m * n * k / (start.elapsed_time(end) / 1e3 / reps) / 1e12


def point(n_tpc, quiet=False):
    import torch
    if not torch.cuda.is_available():
        print("smctrl-numcheck: no CUDA device", file=sys.stderr)
        return 2
    torch.cuda.init()
    reported = torch.cuda.get_device_properties(0).multi_processor_count
    set_tpc_budget(n_tpc)
    tf = throughput(torch, n_tpc)
    worst, case = battery(torch, n_tpc)
    ok = worst <= TOL
    verdict = "CLEAN" if ok else "CORRUPT"
    if quiet:
        print(f"{n_tpc},{n_tpc * 2},{reported},{tf:.1f},{worst:.2e},"
              f"\"{case}\",{verdict}")
    else:
        print(f"smctrl-numcheck: tpcs={n_tpc} sm={n_tpc * 2} reported={reported} "
              f"{tf:.1f} TFLOP/s worst_rel_err={worst:.2e} ({case})  {verdict}")
    return 0 if ok else 1


def sweep():
    print("tpcs,sm,reported,tflops,worst_rel_err,shape,verdict")
    bad = []
    for n_tpc in range(TPCS_FLOOR, TPCS_MASKABLE + 1):
        res = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--tpcs", str(n_tpc), "--csv"],
            capture_output=True, text=True)
        line = res.stdout.strip()
        if not line:
            print(f"{n_tpc},{n_tpc * 2},,,,\"\",FAILED  # {res.stderr.strip()[:120]}")
            bad.append(n_tpc)
            continue
        print(line, flush=True)
        if line.endswith("CORRUPT"):
            bad.append(n_tpc)
    print(f"# corrupt or failed at: {bad}" if bad else "# clean at every TPC count",
          file=sys.stderr)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--tpcs", type=int, help="TPC budget for a single point")
    ap.add_argument("--csv", action="store_true", help="one CSV row instead of prose")
    ap.add_argument("--sweep", action="store_true", help="every TPC count, as CSV")
    args = ap.parse_args()
    if args.sweep:
        return sweep()
    if args.tpcs is None:
        ap.error("give --tpcs N or --sweep")
    return point(args.tpcs, quiet=args.csv)


if __name__ == "__main__":
    sys.exit(main())
