// AWQ INT4 GEMV kernel for batch-size M ≤ 16 (decode hot path).
//
// Companion to `marlin_gemm_int4_zp_bf16` in `marlin_gemm.cu`. The legacy
// kernel is a 256-thread tiled scalar GEMM tuned for prefill (M ≥ 16); at
// M = 1 it leaves 240/256 threads idle and the same value is re-computed by
// 16 threads per output slot. This file provides a purpose-built GEMV that
// runs at memory bandwidth instead of compute pessimism.
//
// Memory layout (matches `marlin_gemm_int4_zp_bf16` arguments — same data
// pointers, same shapes — so dispatch happens transparently in
// `MarlinGemmOp::cuda_fwd` without any change to weight loading):
//
//   output  [M, N]            BF16
//   input   [M, K]            BF16
//   qweight [K/8, N]          U32   sequential nibbles along K (GPTQ-style)
//   scales  [G, N]            BF16
//   qzeros  [G, N/8]          U32   sequential nibbles along N
//   bias    [N]               BF16  (optional)
//
//   G = num_groups = K / group_size
//
// Strategy:
//   - One block produces one tile of N_TILE = 32 output columns × all M rows.
//   - One warp per block (32 threads), one thread per output column.
//   - Each thread reduces the entire K dimension for its column in FP32.
//   - Input row is loaded once into shared memory and re-used across the K
//     reduction; for M > 1 we re-load per row (M is at most ~16, so the
//     amortised loads are negligible vs. the weight stream).
//   - Scale + zero point are cached per group: AWQ checkpoints have
//     group_size % 8 == 0 (group_size = 128 in practice), so all 8 nibbles
//     of one packed word land in the same group — exactly one scale/zp
//     update per packed_k step on a group boundary.
//
// Activation reorder (`g_idx`) is not supported here; the dispatcher only
// routes to this kernel when no g_idx tensor is bound, otherwise it falls
// back to `marlin_gemm_int4_zp_bf16`.
//
// FP32 accumulation, BF16 output. Numerically equivalent to the legacy
// kernel up to BF16 round-off (≤ 1e-2 absolute on the shapes we test).

#include <cuda_bf16.h>

#define N_TILE 32
#define BLOCK_THREADS 32

extern "C" __global__ void awq_gemv_int4_bf16(
    __nv_bfloat16* __restrict__ output,         // [M, N]
    const __nv_bfloat16* __restrict__ input,    // [M, K]
    const unsigned int* __restrict__ qweight,   // [K/8, N]
    const __nv_bfloat16* __restrict__ scales,   // [G, N]
    const unsigned int* __restrict__ qzeros,    // [G, N/8]
    const __nv_bfloat16* __restrict__ bias,     // [N], optional
    int M,
    int N,
    int K,
    int group_size,
    int has_zp,
    int has_bias
) {
    const int tid = threadIdx.x;
    const int n   = blockIdx.x * N_TILE + tid;

    // Shared input cache for the current M row, sized via the launch's
    // `shared_mem_bytes` (host-side: K * sizeof(__nv_bfloat16)).
    extern __shared__ __nv_bfloat16 input_sh[];

    const int packed_k = K / 8;
    const int packed_n = N / 8;

    for (int m = 0; m < M; ++m) {
        // Cooperative load: input[m, :K] -> input_sh[:K].
        // K is small enough for the warp to stream it without tiling.
        for (int k = tid; k < K; k += BLOCK_THREADS) {
            input_sh[k] = input[m * K + k];
        }
        __syncthreads();

        if (n < N) {
            float acc = 0.0f;

            // Group cache. Initialised to a sentinel so the first iteration
            // forces a fetch.
            int   cached_group = -1;
            float cached_scale = 0.0f;
            float cached_zp    = 8.0f;  // legacy default (symmetric AWQ bias)

            for (int kp = 0; kp < packed_k; ++kp) {
                const int k_base   = kp * 8;
                const int group_id = k_base / group_size;

                if (group_id != cached_group) {
                    cached_scale = __bfloat162float(scales[group_id * N + n]);
                    if (has_zp) {
                        const int zp_pack_col = n >> 3;          // n / 8
                        const int zp_bit      = (n & 7) << 2;    // (n % 8) * 4
                        const unsigned int zw = qzeros[group_id * packed_n + zp_pack_col];
                        cached_zp = (float)((zw >> zp_bit) & 0xFu);
                    }
                    cached_group = group_id;
                }

                const unsigned int w = qweight[kp * N + n];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int   nib  = (int)((w >> (i * 4)) & 0xFu);
                    const float wval = ((float)nib - cached_zp) * cached_scale;
                    acc += __bfloat162float(input_sh[k_base + i]) * wval;
                }
            }

            if (has_bias) {
                acc += __bfloat162float(bias[n]);
            }
            output[m * N + n] = __float2bfloat16(acc);
        }
        __syncthreads();  // ready for next m
    }
}
