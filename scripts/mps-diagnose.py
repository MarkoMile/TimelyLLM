#!/usr/bin/env python3
"""Characterise HOW an SM-capped MPS client computes the wrong GEMM.

Run as an MPS client with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE already set.
Prints a report; does not exit non-zero on corruption.
"""
import os, sys, json
import torch

PCT = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
DEV = "cuda"

# The shape numcheck most often flags: M=512 qkv fp16.
M, K, N = 512, 4096, 6144


def make(dt=torch.float16, m=M, k=K, n=N, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(m, k, device=DEV, dtype=dt)
    b = torch.randn(k, n, device=DEV, dtype=dt)
    return a, b


def relerr(got, ref):
    return ((got.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-9)).item()


def busy_ref(a, b):
    """fp64 reference. Slow on purpose: this is what leaves work in flight."""
    return (a.double() @ b.double()).float()


def report(name, val):
    print(f"  {name:<52} {val}")


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*76}\nMPS diagnose  pct={PCT}  sm={props.multi_processor_count}"
          f"  ({props.name})\n{'='*76}")

    # ---- 1. does a device synchronise before the GEMM change the answer? ----
    print("\n[1] co-residency: same GEMM, busy vs drained GPU")
    a, b = make()
    ref = busy_ref(a, b)                 # enqueued, still running
    got_busy = a @ b                     # lands on a busy GPU
    e_busy = relerr(got_busy, ref)

    a2, b2 = make()
    ref2 = busy_ref(a2, b2)
    torch.cuda.synchronize()             # drain first
    got_drained = a2 @ b2
    e_drained = relerr(got_drained, ref2)
    report("rel err, GEMM enqueued behind the fp64 reference", f"{e_busy:.3e}")
    report("rel err, GEMM after torch.cuda.synchronize()", f"{e_drained:.3e}")

    # ---- 2. what do the wrong numbers look like? ---------------------------
    print("\n[2] structure of the error (busy case)")
    g, r = got_busy.float(), ref
    finite = torch.isfinite(g)
    report("non-finite entries in the fp16 result",
           f"{(~finite).sum().item()} / {g.numel()}")
    if finite.any():
        ratio = (g[finite] / r[finite].clamp(min=1e-9))
        ratio = ratio[torch.isfinite(ratio)]
        if ratio.numel():
            qs = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=DEV)
            q = torch.quantile(ratio.float(), qs).tolist()
            report("got/ref quantiles [1,25,50,75,99]%",
                   "[" + ", ".join(f"{v:.4f}" for v in q) + "]")
            report("fraction of entries within 1% of ref",
                   f"{((ratio-1).abs() < 0.01).float().mean().item():.3f}")
            report("fraction of entries that are exactly 0",
                   f"{(g[finite] == 0).float().mean().item():.3f}")
    # is the damage localised by row/column?
    bad = (g - r).abs() > 0.05 * r.abs().max()
    if bad.any():
        rows = bad.any(dim=1).sum().item(); cols = bad.any(dim=0).sum().item()
        report("rows touched / total", f"{rows} / {g.shape[0]}")
        report("cols touched / total", f"{cols} / {g.shape[1]}")
        rowfrac = bad.float().mean(dim=1)
        report("per-row bad fraction: min/med/max",
               f"{rowfrac.min().item():.3f} / {rowfrac.median().item():.3f} / "
               f"{rowfrac.max().item():.3f}")

    # ---- 3. run-to-run determinism ----------------------------------------
    print("\n[3] determinism: 5 repeats of the identical busy GEMM")
    errs, sigs = [], []
    for i in range(5):
        aa, bb = make(seed=0)
        rr = busy_ref(aa, bb)
        gg = aa @ bb
        errs.append(relerr(gg, rr))
        sigs.append(float(gg.float()[torch.isfinite(gg.float())].sum().item()))
    report("rel errs", "[" + ", ".join(f"{e:.2e}" for e in errs) + "]")
    report("result checksums identical across repeats",
           "yes" if len(set(f"{s:.6e}" for s in sigs)) == 1 else "NO (nondeterministic)")

    # ---- 4. split-K / reduced-precision-reduction knob ---------------------
    print("\n[4] cuBLAS knobs")
    old = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    for flag in (True, False):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
        aa, bb = make()
        rr = busy_ref(aa, bb)
        report(f"allow_fp16_reduced_precision_reduction={flag}", f"{relerr(aa@bb, rr):.3e}")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old
    report("CUBLASLT_WORKSPACE_SIZE (env, KiB)",
           os.environ.get("CUBLASLT_WORKSPACE_SIZE", "unset"))

    # ---- 5. dtype / size sensitivity --------------------------------------
    print("\n[5] which kernels are affected")
    for dt, label in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"),
                      (torch.float32, "fp32"), (torch.float64, "fp64")):
        aa, bb = make(dt=dt)
        rr = (aa.double() @ bb.double()).float()
        report(f"{label} GEMM {M}x{K}x{N}", f"{relerr(aa@bb, rr):.3e}")
    for n in (64, 128, 256, 512, 1024):
        aa, bb = make(m=n, k=n, n=n)
        rr = busy_ref(aa, bb)
        report(f"fp16 square GEMM n={n}", f"{relerr(aa@bb, rr):.3e}")


if __name__ == "__main__":
    main()
