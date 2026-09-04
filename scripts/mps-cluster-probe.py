#!/usr/bin/env python3
"""For one GEMM shape, record which cuBLASLt kernel is chosen under an SM cap,
and whether that kernel computes the right answer.

cuBLASLt logs its chosen algo (tile, stages, cluster shape) at LOG_LEVEL 4. The
cluster shape is the interesting field: a Hopper thread-block cluster requires
its CTAs to be co-resident on one GPC.
"""
import os, re, sys, torch

PCT = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
LOG = os.environ["CUBLASLT_LOG_FILE"]
M, K, N = (int(x) for x in os.environ.get("SHAPE", "512,4096,6144").split(","))

torch.backends.cuda.preferred_blas_library("cublaslt")
sm = torch.cuda.get_device_properties(0).multi_processor_count
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
ref = (a.double() @ b.double()).float()
got = a @ b
err = ((got.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-9)).item()
torch.cuda.synchronize()

algo = tile = cluster = stages = "?"
try:
    for line in open(LOG):
        m = re.search(r"algo=\[algoId=(\d+) tile=MATMUL_TILE_(\S+) stages=MATMUL_STAGES_(\S+).*?"
                      r"clusterShape=CLUSTER_SHAPE_(\S+)\]", line)
        if m:
            algo, tile, stages, cluster = m.groups()
except FileNotFoundError:
    pass
print(f"{PCT},{sm},{algo},{tile},{stages},{cluster},{err:.3e},"
      f"{'CLEAN' if err < 2e-2 else 'CORRUPT'}")
