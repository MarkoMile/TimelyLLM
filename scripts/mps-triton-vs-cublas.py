#!/usr/bin/env python3
"""Is the SM cap corrupting the ARITHMETIC, or is it corrupting cuBLAS's choice
of kernel?  Same shapes, same cap, two independent GEMM implementations:

  - torch.matmul  -> cuBLASLt, heuristic kernel selection using the device's
                     reported multiProcessorCount
  - a plain Triton tiled matmul -> our own kernel, fixed tiling, no heuristic,
                     no persistent/stream-K decomposition, no workspace

If Triton is correct where cuBLAS is wrong, the hardware is computing fine under
the cap and the fault is in cuBLAS's kernel selection.
"""
import os, sys
import torch, triton, triton.language as tl

PCT = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")


@triton.jit
def _mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A + offs_m[:, None] * sam + (k * BK + offs_k)[None, :] * sak,
                    mask=(offs_m[:, None] < M) & ((k * BK + offs_k)[None, :] < K), other=0.0)
        b = tl.load(B + (k * BK + offs_k)[:, None] * sbk + offs_n[None, :] * sbn,
                    mask=((k * BK + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_mm(a, b):
    M, K = a.shape; K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BM = BN = 64; BK = 32
    _mm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c


def relerr(got, ref):
    return ((got.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-9)).item()


def main():
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"\n{'='*70}\ncuBLAS vs Triton  pct={PCT}  sm={sm}\n{'='*70}")
    print(f"  {'shape':<22}{'cuBLAS':>14}{'Triton':>14}   verdict")
    for (M, K, N) in [(512, 4096, 6144), (1024, 1024, 1024), (2048, 4096, 4096),
                      (512, 14336, 4096), (256, 256, 256)]:
        torch.manual_seed(0)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = (a.double() @ b.double()).float()      # leaves work in flight
        e_cublas = relerr(a @ b, ref)
        ref2 = (a.double() @ b.double()).float()     # same busy precondition
        e_triton = relerr(triton_mm(a, b), ref2)
        bad_c, bad_t = not (e_cublas < 2e-2), not (e_triton < 2e-2)
        verdict = ("cuBLAS WRONG, Triton ok" if bad_c and not bad_t else
                   "both wrong" if bad_c and bad_t else
                   "Triton wrong only" if bad_t else "both ok")
        print(f"  {M}x{K}x{N:<12}{e_cublas:>14.3e}{e_triton:>14.3e}   {verdict}")


if __name__ == "__main__":
    main()
