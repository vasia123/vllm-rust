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

// ─── Split-K variant for higher SM occupancy ────────────────────────────────
//
// The serial kernel above puts a single warp on each block (one thread per
// output column, K reduced sequentially). For Qwen3-class shapes (N = 2560)
// that leaves ~10% of SM warp slots active on Ada. The split-K variant fans
// each output column across `K_THREADS = 4` threads, so the same N tile uses
// a 4-warp block — still one block per N tile, but 4× more parallel work
// per block, raising occupancy to ~40% on RTX 4060 Laptop.
//
// Layout assumes:
//   - K  divisible by (8 * K_THREADS)         (e.g. K = 2560 → K/4 = 640, ok)
//   - K  divisible by group_size              (always for AWQ)
//   - K_THREADS divides packed_k              (for clean k-chunk boundaries)
//   - N  divisible by N_TILE                  (32)
// The dispatcher checks these and falls back to `awq_gemv_int4_bf16` if not.
//
// Within one warp (32 lanes) we pack 8 output columns × 4 K-chunks each.
// Lane (4*c + ck) holds the partial sum for column c, K-chunk ck. After two
// `__shfl_xor_sync` rounds the 4 partial sums of each column collapse into
// lane (4*c + 0), which writes the BF16 output. FP32 accumulation throughout.

#define SPLIT_K_THREADS 4
#define SPLIT_BLOCK_THREADS (N_TILE * SPLIT_K_THREADS)  // 128
#define SPLIT8_K_THREADS 8
#define SPLIT8_BLOCK_THREADS (N_TILE * SPLIT8_K_THREADS) // 256

extern "C" __global__ void awq_gemv_int4_split_k_bf16(
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
    const int tid     = threadIdx.x;
    const int col_idx = tid / SPLIT_K_THREADS;     // 0..31 → output column within block
    const int k_chunk = tid % SPLIT_K_THREADS;     // 0..3  → K-range within column
    const int n       = blockIdx.x * N_TILE + col_idx;

    extern __shared__ __nv_bfloat16 input_sh[];

    const int packed_k = K / 8;
    const int packed_n = N / 8;

    // packed_k must be divisible by SPLIT_K_THREADS (host-side dispatcher
    // enforces this; otherwise the serial kernel is used).
    const int kp_per_chunk = packed_k / SPLIT_K_THREADS;
    const int kp_start     = k_chunk * kp_per_chunk;
    const int kp_end       = kp_start + kp_per_chunk;

    for (int m = 0; m < M; ++m) {
        // Cooperative load of input[m, :K].
        for (int k = tid; k < K; k += SPLIT_BLOCK_THREADS) {
            input_sh[k] = input[m * K + k];
        }
        __syncthreads();

        float acc = 0.0f;
        if (n < N) {
            int   cached_group = -1;
            float cached_scale = 0.0f;
            float cached_zp    = 8.0f;

            for (int kp = kp_start; kp < kp_end; ++kp) {
                const int k_base   = kp * 8;
                const int group_id = k_base / group_size;

                if (group_id != cached_group) {
                    cached_scale = __bfloat162float(scales[group_id * N + n]);
                    if (has_zp) {
                        const int zp_pack_col = n >> 3;
                        const int zp_bit      = (n & 7) << 2;
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
        }

        // Reduce 4 partial sums per column across lanes (4*c..4*c+3) within
        // the same warp. SPLIT_K_THREADS = 4 → two XOR rounds suffice.
        // The full warp participates (mask = 0xFFFFFFFF) — inactive lanes
        // (n ≥ N) carry acc = 0 from above, so they do not corrupt sums.
        unsigned mask = 0xFFFFFFFFu;
        acc += __shfl_xor_sync(mask, acc, 1);
        acc += __shfl_xor_sync(mask, acc, 2);

        if (k_chunk == 0 && n < N) {
            float out_val = acc;
            if (has_bias) {
                out_val += __bfloat162float(bias[n]);
            }
            output[m * N + n] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }
}

// ─── Split-K (×8) variant — wider K split for higher SM occupancy ───────────
//
// Same shape as the ×4 variant but K is fanned across 8 threads instead of 4.
// 256-thread block (8 warps × 32 lanes), N_TILE = 32 columns. Within one warp
// (32 lanes) we pack 4 columns × 8 K-chunks. Three `__shfl_xor_sync` rounds
// (mask 1, 2, 4) collapse the 8 partial sums into lane (8*c + 0).
//
// Layout assumes (host-side gate):
//   - K  divisible by (8 * SPLIT8_K_THREADS) — packed_k % 8 == 0
//
// With BLOCK_THREADS = 256 = 8 warps, peak active warps per SM rises to ~26
// out of 48 (Ada) at Qwen3 N = 2560 — roughly twice the ×4 variant.

extern "C" __global__ void awq_gemv_int4_split_k8_bf16(
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
    const int tid     = threadIdx.x;
    const int col_idx = tid / SPLIT8_K_THREADS;     // 0..31 → output column
    const int k_chunk = tid % SPLIT8_K_THREADS;     // 0..7  → K-range
    const int n       = blockIdx.x * N_TILE + col_idx;

    extern __shared__ __nv_bfloat16 input_sh[];

    const int packed_k = K / 8;
    const int packed_n = N / 8;

    const int kp_per_chunk = packed_k / SPLIT8_K_THREADS;
    const int kp_start     = k_chunk * kp_per_chunk;
    const int kp_end       = kp_start + kp_per_chunk;

    for (int m = 0; m < M; ++m) {
        for (int k = tid; k < K; k += SPLIT8_BLOCK_THREADS) {
            input_sh[k] = input[m * K + k];
        }
        __syncthreads();

        float acc = 0.0f;
        if (n < N) {
            int   cached_group = -1;
            float cached_scale = 0.0f;
            float cached_zp    = 8.0f;

            for (int kp = kp_start; kp < kp_end; ++kp) {
                const int k_base   = kp * 8;
                const int group_id = k_base / group_size;

                if (group_id != cached_group) {
                    cached_scale = __bfloat162float(scales[group_id * N + n]);
                    if (has_zp) {
                        const int zp_pack_col = n >> 3;
                        const int zp_bit      = (n & 7) << 2;
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
        }

        // 3-stage warp reduce across 8 lanes per column.
        unsigned mask = 0xFFFFFFFFu;
        acc += __shfl_xor_sync(mask, acc, 1);
        acc += __shfl_xor_sync(mask, acc, 2);
        acc += __shfl_xor_sync(mask, acc, 4);

        if (k_chunk == 0 && n < N) {
            float out_val = acc;
            if (has_bias) {
                out_val += __bfloat162float(bias[n]);
            }
            output[m * N + n] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }
}

// ─── vec4 LDG variant on transposed qweight `[N, K/8]` ──────────────────────
//
// The layouts above (`[K/8, N]` row-major) leave each thread reading
// stride-N u32s along K — every packed_k step lands in a fresh cache line.
// Transposing qweight to `[N, K/8]` puts the K-axis in the contiguous
// dimension, so a single thread can issue `uint4` loads (16 bytes = 4 u32 =
// 32 packed int4 weights) per memory transaction. Quarters the number of
// LDG instructions at the same coalesced bandwidth — the decisive lever
// for pushing the AWQ GEMV past the previous ~86 GB/s effective into the
// 150-200 GB/s range on Ada.
//
// Layout requires (host-side gate):
//   - K  divisible by (32 * SPLIT_KT_THREADS)   → packed_k % (4 * splits) == 0
//   - N  divisible by N_TILE                    (32)
//   - K  divisible by group_size                (always for AWQ)
//
// Block shape mirrors the ×8 split-K variant: 256 threads (8 warps), each
// output column handled by 8 K-cooperating threads. Each of those 8 threads
// processes its slice in `uint4` chunks (4 packed_k = 32 K positions per
// LDG), reducing per-thread issue rate by 4× vs the row-major variants.
// Final reduce is the same 3-stage `__shfl_xor_sync` as `split_k8`.

#define SPLIT_KT_THREADS 8
#define SPLIT_KT_BLOCK_THREADS (N_TILE * SPLIT_KT_THREADS)  // 256

extern "C" __global__ void awq_gemv_int4_kt_bf16(
    __nv_bfloat16* __restrict__ output,         // [M, N]
    const __nv_bfloat16* __restrict__ input,    // [M, K]
    const unsigned int* __restrict__ qweight_kt,// [N, K/8]   transposed
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
    const int tid     = threadIdx.x;
    const int col_idx = tid / SPLIT_KT_THREADS;     // 0..31 → output column
    const int k_chunk = tid % SPLIT_KT_THREADS;     // 0..7  → K-range
    const int n       = blockIdx.x * N_TILE + col_idx;

    extern __shared__ __nv_bfloat16 input_sh[];

    const int packed_k     = K / 8;
    const int packed_n     = N / 8;
    const int kp_per_chunk = packed_k / SPLIT_KT_THREADS;
    const int kp_start     = k_chunk * kp_per_chunk;

    for (int m = 0; m < M; ++m) {
        for (int k = tid; k < K; k += SPLIT_KT_BLOCK_THREADS) {
            input_sh[k] = input[m * K + k];
        }
        __syncthreads();

        float acc = 0.0f;
        if (n < N) {
            int   cached_group = -1;
            float cached_scale = 0.0f;
            float cached_zp    = 8.0f;

            // Base pointer into qweight_kt for this output column.
            // Stride between consecutive packed_k values is 1 u32 — fully
            // contiguous along the K dimension after the load-time
            // transpose.
            const uint4* __restrict__ qw_base =
                reinterpret_cast<const uint4*>(qweight_kt + (size_t)n * packed_k + kp_start);

            const int vec_iters = kp_per_chunk / 4;
            #pragma unroll 1
            for (int v = 0; v < vec_iters; ++v) {
                const uint4 packed4 = qw_base[v];
                const unsigned int ws[4] = {packed4.x, packed4.y, packed4.z, packed4.w};

                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int kp       = kp_start + v * 4 + j;
                    const int k_base   = kp * 8;
                    const int group_id = k_base / group_size;

                    if (group_id != cached_group) {
                        cached_scale = __bfloat162float(scales[group_id * N + n]);
                        if (has_zp) {
                            const int zp_pack_col = n >> 3;
                            const int zp_bit      = (n & 7) << 2;
                            const unsigned int zw =
                                qzeros[group_id * packed_n + zp_pack_col];
                            cached_zp = (float)((zw >> zp_bit) & 0xFu);
                        }
                        cached_group = group_id;
                    }

                    const unsigned int w = ws[j];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int   nib  = (int)((w >> (i * 4)) & 0xFu);
                        const float wval = ((float)nib - cached_zp) * cached_scale;
                        acc += __bfloat162float(input_sh[k_base + i]) * wval;
                    }
                }
            }
        }

        unsigned mask = 0xFFFFFFFFu;
        acc += __shfl_xor_sync(mask, acc, 1);
        acc += __shfl_xor_sync(mask, acc, 2);
        acc += __shfl_xor_sync(mask, acc, 4);

        if (k_chunk == 0 && n < N) {
            float out_val = acc;
            if (has_bias) {
                out_val += __bfloat162float(bias[n]);
            }
            output[m * N + n] = __float2bfloat16(out_val);
        }
        __syncthreads();
    }
}
