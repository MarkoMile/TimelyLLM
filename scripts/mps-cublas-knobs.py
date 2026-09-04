#!/usr/bin/env python3
"""Which cuBLAS path is broken under an SM cap, and can any user-space knob avoid it?"""
import os, torch
PCT = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")

def relerr(g, r):
    return ((g.float() - r).abs().max() / r.abs().max().clamp(min=1e-9)).item()

def trial(M, K, N, busy=True):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    r = (a.double() @ b.double()).float()
    if not busy:
        torch.cuda.synchronize()
    return relerr(a @ b, r)

sm = torch.cuda.get_device_properties(0).multi_processor_count
print(f"\n{'='*72}\ncuBLAS knobs  pct={PCT}  sm={sm}\n{'='*72}")
SHAPES = [(512, 4096, 6144), (1024, 1024, 1024)]

print(f"  {'backend':<28}" + "".join(f"{'x'.join(map(str,s)):>22}" for s in SHAPES))
for backend in ("cublas", "cublaslt"):
    try:
        torch.backends.cuda.preferred_blas_library(backend)
    except Exception as e:
        print(f"  {backend:<28} unavailable: {e}"); continue
    row = "".join(f"{trial(*s):>22.3e}" for s in SHAPES)
    print(f"  {backend:<28}{row}")
torch.backends.cuda.preferred_blas_library("cublas")

print(f"\n  {'condition':<28}" + "".join(f"{'x'.join(map(str,s)):>22}" for s in SHAPES))
print(f"  {'GPU busy (no sync)':<28}" + "".join(f"{trial(*s, busy=True):>22.3e}" for s in SHAPES))
print(f"  {'GPU drained (sync first)':<28}" + "".join(f"{trial(*s, busy=False):>22.3e}" for s in SHAPES))
print(f"  {'cublas ws  = ' + str(torch.backends.cuda.cublas_workspace_size()):<28}")
print(f"  {'cublasLt ws= ' + str(torch.backends.cuda.cublaslt_workspace_size()):<28}")
print(f"  env CUBLAS_WORKSPACE_CONFIG   = {os.environ.get('CUBLAS_WORKSPACE_CONFIG','unset')}")
print(f"  env CUBLASLT_WORKSPACE_SIZE   = {os.environ.get('CUBLASLT_WORKSPACE_SIZE','unset')}")
