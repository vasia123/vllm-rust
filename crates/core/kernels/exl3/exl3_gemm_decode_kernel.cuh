#pragma once

#include "exl3_kernel_map.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

// Decode fast path for the EXL3 GEMM. Assumes:
//   1. `M <= 16` (a single 16-row tile). Caller (Rust dispatch in
//      `Exl3GemmInplaceOp::cuda_fwd`) guarantees this — prefill
//      batches fall through to the cooperative `exl3_gemm_kernel`.
//   2. `A` already holds the Hadamard-transformed activations
//      (computed by the standalone `had_hf_r_128_kernel` in
//      `hadamard.cu`, launched separately on the same stream right
//      before this kernel).
//
// Differences vs `exl3_gemm_kernel`:
//   - no input-Hadamard prologue (caller did it)
//   - no `cg::this_grid().sync()` calls anywhere
//   - no M-tile loop (only one 16-row tile)
//   - no `cooperative_groups` include or runtime dependency
//
// The output Hadamard (svh) is still applied — `exl3_gemm_kernel_inner`
// with `shmem_out_had=true` uses lock-based barriers (`barrier_acquire`
// / `barrier_release` in `ptx.cuh:103,121`) for column-wise reduction
// coordination, which does NOT require cooperative launch. So the
// kernel produces identical output to the cooperative path within
// fp16 noise.
//
// The `suh` and `A_had` parameters are retained for ABI compatibility
// with `EXL3_GEMM_ARGS` (mirrors `exl3_gemm_kernel`'s signature so the
// Rust dispatcher reuses one builder/arg layout). Both are ignored
// inside this kernel.
//
// Launched via standard `cuLaunchKernel` (NOT cooperative) — making
// the EXL3 decode path CUDA-Graph-capturable end-to-end.
template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_decode_kernel(EXL3_GEMM_ARGS)
{
    (void) suh;    // unused — caller pre-applied input Hadamard
    (void) A_had;  // unused — A already holds A_had

    exl3_gemm_kernel_inner
    <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
    (A, B, C, MIN(size_m, 16), size_k, size_n, locks, svh);
}
