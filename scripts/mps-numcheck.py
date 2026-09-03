#!/usr/bin/env python3
"""Is fp16/bf16 matmul numerically correct under the current MPS SM cap?

Run this as an MPS client, with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE already set.
Exit 0 if the GPU computes correctly, 1 if it does not.

Why this exists. On this H100 (driver 580.159.03, CUDA 13.0), capping a client's
SMs with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE silently corrupts cuBLAS fp16/bf16
tensor-core GEMM at many -- not all -- of the resulting SM counts. It is
deterministic and reproducible, it happens in plain torch.matmul with no
inference stack involved, and it is narrow:

    add / reductions        correct
    fp32 GEMM               correct at every size
    fp16 GEMM, n=128        correct
    fp16 GEMM, n>=512       WRONG (relative error ~1, often inf)

so it is the large-tile Hopper tensor-core kernels specifically. It does not
scale with the cap either, so no safe range can be reasoned out -- only measured.

The shapes matter as much as the cap. Square matmuls miss cases that the model's
own layer shapes catch: 66%, 20% and 15% all pass a square fp16 test and still
generate garbage in a real run. So this checks Llama-3-8B's actual linear layers
across the batch sizes prefill and decode produce. Measured clean on this box:

    5 8 10 12 25 30 33 50 90 95 100      (everything else in 5..100 is corrupt)

The failure is silent and it is dangerous for an experiment: a corrupted run
still produces a log full of plausible-looking latencies. It was found because
the model started emitting garbage tokens instead of plans, not because anything
errored.

This gate is necessary, NOT proven sufficient -- it covers cuBLAS GEMM, not the
attention or normalisation kernels. Always also check the output-sanity column
in plot-mps-latency.py, which reads what the model actually generated.
"""

import os
import sys

TOL = 2e-2  # rel error; healthy is ~3e-3, corrupt is ~1e0 or inf


def main():
    import torch
    pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
    if not torch.cuda.is_available():
        print("numcheck: no CUDA device", file=sys.stderr)
        return 2
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(0)

    # Llama-3-8B's linear layers, as (name, K, N). M is the number of tokens in
    # the batch: 1 and 8 are decode steps, 512 and 2048 are prefill.
    layers = [("qkv", 4096, 6144), ("o_proj", 4096, 4096),
              ("gate_up", 4096, 28672), ("down", 14336, 4096)]
    worst, worst_case = 0.0, ""
    for dt in (torch.float16, torch.bfloat16):
        for m in (1, 8, 64, 512, 2048):
            for name, k, n in layers:
                a = torch.randn(m, k, device="cuda", dtype=dt)
                b = torch.randn(k, n, device="cuda", dtype=dt)
                ref = (a.double() @ b.double()).float()
                # Enqueue the fp16 GEMM with other work still in flight -- no
                # .item() between, which would synchronise first. This matters:
                # the corruption only appears when the big GEMM lands on a GPU
                # that is already busy, which is why a single-prompt generation
                # test passes at 66% while the real batched workload fails there.
                rel = (((a @ b).float() - ref).abs().max()
                       / ref.abs().max().clamp(min=1e-9)).item()
                if rel > worst:
                    worst, worst_case = rel, \
                        f"M={m} {name} {str(dt).split('.')[-1]}"

    ok = worst <= TOL
    print(f"numcheck: pct={pct} sm={sm} worst_rel_err={worst:.2e} "
          f"({worst_case})  {'OK' if ok else 'CORRUPT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
