"""Cap the engine to a fixed number of SMs, using a CUDA green context.

Why not MPS.  Capping SMs with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE silently
corrupts Hopper tensor-core GEMM at most of the resulting SM counts -- see
docs/MPS-CUBLAS-CORRUPTION.md.  cuBLAS picks its kernel from the device's
reported multiProcessorCount, and MPS makes that number take values no physical
Hopper part reports; some of the kernels it then picks compute wrong results.

A green context (CUDA 12.4+) partitions SMs without touching that number: the
device still reports 132 SMs, so cuBLAS picks exactly the kernel it picks on the
whole GPU, and the driver simply runs it on fewer SMs.  Measured clean at every
partition size, including all the ones MPS corrupts
(scripts/greenctx-numcheck.py).

Two further differences from MPS, both in our favour: no control daemon, so
nothing is left behind on a shared machine; and the partition can be changed at
run time, in-process, which is what SM scheduling needs.

Usage.  Set TIMELYLLM_SM_COUNT=<n> in the engine process; `apply()` is called
before the engine is built, so weight loading, the memory-profiling run and
CUDA-graph capture all happen inside the partition.  SMs are allocated in groups
of 8 on this GH200, so n is rounded up by the driver.

Dynamic use.  `set_sm_count(n)` switches the live partition.  Note that a CUDA
graph is bound to the partition it was captured under, so an engine that is
going to switch levels must either run eager or capture one graph set per level.

libsmctrl.  `TIMELYLLM_TPC_COUNT=<n>` uses libsmctrl's global TPC mask instead.
It is the mechanism a libsmctrl-based scheduler would use, and it is here so the
two can be compared on the same workload.  It is equally clean numerically
(scripts/smctrl-numcheck.py: clean at every TPC count from 4 to 64), but three
things differ and matter:

  - a TPC is 2 SMs, so the ladder is 4x finer;
  - libsmctrl's Hopper path pins TPCs 64+ off, so it reaches 128 of 132 SMs;
  - the count you ask for is NOT the compute you get.  Mask bit indexes do not
    correspond to software-visible TPC IDs on compute capability 9.0, and the
    measured throughput ladder is a staircase with wide flat treads.  Calibrate
    against measured TFLOP/s before treating a TPC count as a compute budget.

Below 4 TPCs the GEMM does not finish at all -- 3 TPCs was still running after
three minutes, 1 TPC after eighteen -- which is the same 8-SM floor the green
context enforces by construction.
"""
import os

_current = None          # the green context we pushed, if any


def _create(num_sms):
    from torch.cuda.green_contexts import GreenContext, SUPPORTED
    if not SUPPORTED:
        raise RuntimeError("this torch build has no green-context support")
    return GreenContext.create(num_sms=num_sms, device_id=0)


def set_sm_count(num_sms):
    """Run all subsequent GPU work on `num_sms` SMs.  None restores the whole
    device.  Safe to call repeatedly."""
    global _current
    import torch
    torch.cuda.init()
    if _current is not None:
        _current.pop_context()
        _current = None
    if num_sms is None:
        return
    ctx = _create(num_sms)
    ctx.set_context()
    _current = ctx


LIBSMCTRL_SO = os.environ.get("LIBSMCTRL_SO", "/space/mm562/libsmctrl/libsmctrl.so")

_smctrl = None


def set_tpc_count(n_tpc):
    """Run all subsequent GPU work on TPCs 0..n_tpc-1, via libsmctrl's global
    mask.  None lifts the mask.  libsmctrl masks are *disable* masks, so the
    budget is the complement."""
    global _smctrl
    import ctypes
    if _smctrl is None:
        _smctrl = ctypes.CDLL(LIBSMCTRL_SO)
        _smctrl.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        _smctrl.libsmctrl_set_global_mask.restype = None
    mask = 0 if n_tpc is None else (~((1 << n_tpc) - 1)) & 0xFFFFFFFFFFFFFFFF
    _smctrl.libsmctrl_set_global_mask(mask)


def apply():
    """Honour TIMELYLLM_SM_COUNT or TIMELYLLM_TPC_COUNT, if either is set.
    Returns a description of what was applied, or None."""
    sms = os.environ.get("TIMELYLLM_SM_COUNT", "").strip()
    tpcs = os.environ.get("TIMELYLLM_TPC_COUNT", "").strip()
    if sms and tpcs:
        raise ValueError("set TIMELYLLM_SM_COUNT or TIMELYLLM_TPC_COUNT, not both")
    if sms:
        num_sms = int(sms)
        set_sm_count(num_sms)
        print(f"SM budget: green context with num_sms={num_sms} "
              f"(driver rounds up to a multiple of 8)")
        return f"greenctx:{num_sms}"
    if tpcs:
        n_tpc = int(tpcs)
        set_tpc_count(n_tpc)
        print(f"SM budget: libsmctrl global mask, TPCs 0..{n_tpc - 1} "
              f"(nominally {n_tpc * 2} SMs; measure the throughput, do not "
              f"assume it)")
        return f"smctrl:{n_tpc}"
    return None
