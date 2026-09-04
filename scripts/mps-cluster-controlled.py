#!/usr/bin/env python3
"""Controlled test of the Hopper thread-block-cluster hypothesis.

Same Triton matmul, same shapes, same SM cap -- only the cluster size changes.
If num_ctas=1 is correct and num_ctas>1 is not, clusters are the mechanism.
"""
import os, torch, triton, triton.language as tl

PCT = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")

@triton.jit
def _mm(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A + offs_m[:, None]*sam + (k*BK+offs_k)[None, :]*sak,
                    mask=(offs_m[:, None] < M) & ((k*BK+offs_k)[None, :] < K), other=0.0)
        b = tl.load(B + (k*BK+offs_k)[:, None]*sbk + offs_n[None, :]*sbn,
                    mask=((k*BK+offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None]*scm + offs_n[None, :]*scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def tmm(a, b, num_ctas):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BM = BN = 128; BK = 64
    _mm[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK,
        num_warps=8, num_stages=3, num_ctas=num_ctas)
    return c

sm = torch.cuda.get_device_properties(0).multi_processor_count
print(f"\npct={PCT}  sm={sm}")
print(f"  {'shape':<20}{'cuBLAS':>12}{'triton c=1':>13}{'triton c=2':>13}{'triton c=4':>13}")
for (M, K, N) in [(512, 4096, 6144), (1024, 1024, 1024), (2048, 4096, 4096)]:
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = (a.double() @ b.double()).float()
    def e(g):
        return ((g.float()-ref).abs().max()/ref.abs().max().clamp(min=1e-9)).item()
    out = [f"{e(a@b):>12.2e}"]
    for nc in (1, 2, 4):
        try:
            out.append(f"{e(tmm(a, b, nc)):>13.2e}")
        except Exception as ex:
            out.append(f"{type(ex).__name__:>13}")
    print(f"  {M}x{K}x{N:<10}" + "".join(out))
