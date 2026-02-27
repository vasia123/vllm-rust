// Marlin-style fused INT4/INT8 dequantize + GEMM kernel for quantized inference.
//
// Performs output = input @ dequant(weight).T + bias
// where weight is packed INT4 or INT8 in Marlin tiled format.
//
// Key optimizations:
// - Fused dequantization + matrix multiply (no intermediate materialization)
// - Shared memory tiling for input and dequantized weights
// - Coalesced global memory access patterns
// - BF16 accumulation with optional FP32 reduce for precision
//
// Kernel variants:
// - marlin_gemm_int4_bf16: GPTQ INT4 symmetric (uint4b8)
// - marlin_gemm_int8_bf16: GPTQ INT8 symmetric (uint8b128)
// - marlin_gemm_int4_zp_bf16: AWQ INT4 with zero points

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Tile sizes for the GEMM
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Warp size
#define WARP_SIZE 32

// Pack factors
#define INT4_PACK_FACTOR 8  // 8 INT4 values per U32
#define INT8_PACK_FACTOR 4  // 4 INT8 values per U32

// ─── Helper functions ───────────────────────────────────────────────────────

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Extract INT4 value from packed U32 at position idx (0-7)
__device__ __forceinline__ int extract_int4(unsigned int packed, int idx) {
    int shift = idx * 4;
    int val = (packed >> shift) & 0xF;
    // uint4b8: bias of 8 (symmetric around 8)
    return val - 8;
}

// Extract INT4 value with zero point (AWQ)
__device__ __forceinline__ int extract_int4_zp(unsigned int packed, int idx, int zero_point) {
    int shift = idx * 4;
    int val = (packed >> shift) & 0xF;
    return val - zero_point;
}

// Extract INT8 value from packed U32 at position idx (0-3)
__device__ __forceinline__ int extract_int8(unsigned int packed, int idx) {
    int shift = idx * 8;
    int val = (packed >> shift) & 0xFF;
    // uint8b128: bias of 128 (symmetric around 128)
    return val - 128;
}

// ─── INT4 GEMM kernel (GPTQ symmetric, uint4b8) ────────────────────────────

extern "C" __global__ void marlin_gemm_int4_bf16(
    __nv_bfloat16* __restrict__ output,    // [M, N]
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    const unsigned int* __restrict__ qweight, // [K/8, N] packed INT4
    const __nv_bfloat16* __restrict__ scales, // [num_groups, N]
    unsigned int* __restrict__ workspace,
    int M, int N, int K,
    int num_groups,
    int is_k_full,
    int use_fp32_reduce,
    int has_bias,
    int has_zp,
    int has_g_idx,
    const unsigned int* __restrict__ qzeros,  // unused for symmetric
    const __nv_bfloat16* __restrict__ bias,
    const unsigned int* __restrict__ g_idx,
    const unsigned int* __restrict__ g_idx_sort_indices
) {
    // Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
    int tile_n = blockIdx.x * TILE_N;
    int tile_m = blockIdx.y * TILE_M;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Group size for scale lookup
    int group_size = (num_groups > 0) ? (K / num_groups) : K;
    if (num_groups <= 0) num_groups = 1;

    // Shared memory for input tile and accumulator
    __shared__ float smem_input[TILE_M][TILE_K];
    __shared__ float smem_weight[TILE_K][TILE_N];

    // Per-thread accumulator
    float acc[TILE_M];
    int local_n = tid % TILE_N;  // Which N column this thread handles

    for (int m = 0; m < TILE_M; m++) {
        acc[m] = 0.0f;
    }

    // Packed weight dimension: K/8 rows, N columns
    int packed_k = K / INT4_PACK_FACTOR;

    // Iterate over K dimension in tiles
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load input tile [TILE_M, TILE_K] into shared memory
        // Each of 256 threads loads one or more elements
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = tile_m + m;
            int global_k = k_start + k;
            if (global_m < M && global_k < K) {
                smem_input[m][k] = bf16_to_float(input[global_m * K + global_k]);
            } else {
                smem_input[m][k] = 0.0f;
            }
        }

        // Load and dequantize weight tile [TILE_K, TILE_N] into shared memory
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_k = k_start + k;
            int global_n = tile_n + n;

            if (global_k < K && global_n < N) {
                // Determine which packed word and position within it
                int pack_row = global_k / INT4_PACK_FACTOR;
                int pack_idx = global_k % INT4_PACK_FACTOR;

                unsigned int packed = qweight[pack_row * N + global_n];
                int quant_val = extract_int4(packed, pack_idx);

                // Get scale for this group
                int group_id = global_k / group_size;
                if (group_id >= num_groups) group_id = num_groups - 1;
                float scale = bf16_to_float(scales[group_id * N + global_n]);

                smem_weight[k][n] = (float)quant_val * scale;
            } else {
                smem_weight[k][n] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial products
        if (local_n < TILE_N && (tile_n + local_n) < N) {
            for (int m = 0; m < TILE_M; m++) {
                if (tile_m + m < M) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TILE_K; k++) {
                        sum += smem_input[m][k] * smem_weight[k][local_n];
                    }
                    acc[m] += sum;
                }
            }
        }

        __syncthreads();
    }

    // Write output
    if (local_n < TILE_N && (tile_n + local_n) < N) {
        for (int m = 0; m < TILE_M; m++) {
            int global_m = tile_m + m;
            int global_n = tile_n + local_n;
            if (global_m < M && global_n < N) {
                float val = acc[m];
                if (has_bias && bias != nullptr) {
                    val += bf16_to_float(bias[global_n]);
                }
                output[global_m * N + global_n] = float_to_bf16(val);
            }
        }
    }
}

// ─── INT8 GEMM kernel (GPTQ symmetric, uint8b128) ──────────────────────────

extern "C" __global__ void marlin_gemm_int8_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const unsigned int* __restrict__ qweight,  // [K/4, N] packed INT8
    const __nv_bfloat16* __restrict__ scales,
    unsigned int* __restrict__ workspace,
    int M, int N, int K,
    int num_groups,
    int is_k_full,
    int use_fp32_reduce,
    int has_bias,
    int has_zp,
    int has_g_idx,
    const unsigned int* __restrict__ qzeros,
    const __nv_bfloat16* __restrict__ bias,
    const unsigned int* __restrict__ g_idx,
    const unsigned int* __restrict__ g_idx_sort_indices
) {
    int tile_n = blockIdx.x * TILE_N;
    int tile_m = blockIdx.y * TILE_M;
    int tid = threadIdx.x;

    int group_size = (num_groups > 0) ? (K / num_groups) : K;
    if (num_groups <= 0) num_groups = 1;

    __shared__ float smem_input[TILE_M][TILE_K];
    __shared__ float smem_weight[TILE_K][TILE_N];

    float acc[TILE_M];
    int local_n = tid % TILE_N;

    for (int m = 0; m < TILE_M; m++) {
        acc[m] = 0.0f;
    }

    int packed_k = K / INT8_PACK_FACTOR;

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load input tile
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = tile_m + m;
            int global_k = k_start + k;
            if (global_m < M && global_k < K) {
                smem_input[m][k] = bf16_to_float(input[global_m * K + global_k]);
            } else {
                smem_input[m][k] = 0.0f;
            }
        }

        // Load and dequantize INT8 weight tile
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_k = k_start + k;
            int global_n = tile_n + n;

            if (global_k < K && global_n < N) {
                int pack_row = global_k / INT8_PACK_FACTOR;
                int pack_idx = global_k % INT8_PACK_FACTOR;

                unsigned int packed = qweight[pack_row * N + global_n];
                int quant_val = extract_int8(packed, pack_idx);

                int group_id = global_k / group_size;
                if (group_id >= num_groups) group_id = num_groups - 1;
                float scale = bf16_to_float(scales[group_id * N + global_n]);

                smem_weight[k][n] = (float)quant_val * scale;
            } else {
                smem_weight[k][n] = 0.0f;
            }
        }

        __syncthreads();

        if (local_n < TILE_N && (tile_n + local_n) < N) {
            for (int m = 0; m < TILE_M; m++) {
                if (tile_m + m < M) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TILE_K; k++) {
                        sum += smem_input[m][k] * smem_weight[k][local_n];
                    }
                    acc[m] += sum;
                }
            }
        }

        __syncthreads();
    }

    if (local_n < TILE_N && (tile_n + local_n) < N) {
        for (int m = 0; m < TILE_M; m++) {
            int global_m = tile_m + m;
            int global_n = tile_n + local_n;
            if (global_m < M && global_n < N) {
                float val = acc[m];
                if (has_bias && bias != nullptr) {
                    val += bf16_to_float(bias[global_n]);
                }
                output[global_m * N + global_n] = float_to_bf16(val);
            }
        }
    }
}

// ─── INT4 GEMM with zero points (AWQ, uint4) ───────────────────────────────

extern "C" __global__ void marlin_gemm_int4_zp_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const unsigned int* __restrict__ qweight,  // [K/8, N] packed INT4
    const __nv_bfloat16* __restrict__ scales,
    unsigned int* __restrict__ workspace,
    int M, int N, int K,
    int num_groups,
    int is_k_full,
    int use_fp32_reduce,
    int has_bias,
    int has_zp,
    int has_g_idx,
    const unsigned int* __restrict__ qzeros,   // [num_groups, N/8] packed INT4
    const __nv_bfloat16* __restrict__ bias,
    const unsigned int* __restrict__ g_idx,
    const unsigned int* __restrict__ g_idx_sort_indices
) {
    int tile_n = blockIdx.x * TILE_N;
    int tile_m = blockIdx.y * TILE_M;
    int tid = threadIdx.x;

    int group_size = (num_groups > 0) ? (K / num_groups) : K;
    if (num_groups <= 0) num_groups = 1;

    __shared__ float smem_input[TILE_M][TILE_K];
    __shared__ float smem_weight[TILE_K][TILE_N];

    float acc[TILE_M];
    int local_n = tid % TILE_N;

    for (int m = 0; m < TILE_M; m++) {
        acc[m] = 0.0f;
    }

    // Zero points are packed: N/8 per group row
    int zp_n_packed = N / INT4_PACK_FACTOR;

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load input tile
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = tile_m + m;
            int global_k = k_start + k;
            if (global_m < M && global_k < K) {
                smem_input[m][k] = bf16_to_float(input[global_m * K + global_k]);
            } else {
                smem_input[m][k] = 0.0f;
            }
        }

        // Load and dequantize INT4 weight tile with zero points
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_k = k_start + k;
            int global_n = tile_n + n;

            if (global_k < K && global_n < N) {
                int pack_row = global_k / INT4_PACK_FACTOR;
                int pack_idx = global_k % INT4_PACK_FACTOR;

                unsigned int packed = qweight[pack_row * N + global_n];

                int group_id = global_k / group_size;
                if (group_id >= num_groups) group_id = num_groups - 1;
                float scale = bf16_to_float(scales[group_id * N + global_n]);

                // Get zero point for this group and column
                int zp = 8; // Default: symmetric bias
                if (has_zp && qzeros != nullptr) {
                    int zp_pack_col = global_n / INT4_PACK_FACTOR;
                    int zp_pack_idx = global_n % INT4_PACK_FACTOR;
                    unsigned int zp_packed = qzeros[group_id * zp_n_packed + zp_pack_col];
                    zp = (zp_packed >> (zp_pack_idx * 4)) & 0xF;
                }

                int quant_val = extract_int4_zp(packed, pack_idx, zp);
                smem_weight[k][n] = (float)quant_val * scale;
            } else {
                smem_weight[k][n] = 0.0f;
            }
        }

        __syncthreads();

        if (local_n < TILE_N && (tile_n + local_n) < N) {
            for (int m = 0; m < TILE_M; m++) {
                if (tile_m + m < M) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TILE_K; k++) {
                        sum += smem_input[m][k] * smem_weight[k][local_n];
                    }
                    acc[m] += sum;
                }
            }
        }

        __syncthreads();
    }

    if (local_n < TILE_N && (tile_n + local_n) < N) {
        for (int m = 0; m < TILE_M; m++) {
            int global_m = tile_m + m;
            int global_n = tile_n + local_n;
            if (global_m < M && global_n < N) {
                float val = acc[m];
                if (has_bias && bias != nullptr) {
                    val += bf16_to_float(bias[global_n]);
                }
                output[global_m * N + global_n] = float_to_bf16(val);
            }
        }
    }
}

// ─── AWQ nibble deinterleave (GPU) ──────────────────────────────────────────
//
// Parallel per-word transform: AWQ interleaved nibble order → GPTQ sequential.
// Each thread handles one U32 word, so the kernel is trivially parallel.
//
// AWQ: nibble positions [0..7] = [v0, v2, v4, v6, v1, v3, v5, v7]
// GPTQ: nibble positions [0..7] = [v0, v1, v2, v3, v4, v5, v6, v7]
// Permutation: output nibble i ← input nibble undo_pack[i]
//              undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}
//
// Matches CPU `awq_to_gptq_u32()` in `awq_marlin.rs`.

extern "C" __global__ void awq_marlin_repack_int4(
    unsigned int* __restrict__ output,       // [rows, cols] — GPTQ nibble order
    const unsigned int* __restrict__ input,  // [rows, cols] — AWQ nibble order
    int rows,   // K / 8 (8 INT4 values packed per U32)
    int cols    // N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;

    unsigned int w = input[row * cols + col];

    // Extract nibbles from AWQ interleaved ordering:
    //   positions 0..3 carry even values  [v0, v2, v4, v6]
    //   positions 4..7 carry odd values   [v1, v3, v5, v7]
    unsigned int n0 = (w)       & 0xFU;  // v0
    unsigned int n1 = (w >>  4) & 0xFU;  // v2
    unsigned int n2 = (w >>  8) & 0xFU;  // v4
    unsigned int n3 = (w >> 12) & 0xFU;  // v6
    unsigned int n4 = (w >> 16) & 0xFU;  // v1
    unsigned int n5 = (w >> 20) & 0xFU;  // v3
    unsigned int n6 = (w >> 24) & 0xFU;  // v5
    unsigned int n7 = (w >> 28) & 0xFU;  // v7

    // Pack into GPTQ sequential ordering: [v0, v1, v2, v3, v4, v5, v6, v7]
    output[row * cols + col] =
        n0 | (n4 << 4) | (n1 << 8) | (n5 << 12) |
        (n2 << 16) | (n6 << 20) | (n3 << 24) | (n7 << 28);
}

// ─── FP8 E4M3 GEMM for Ampere (software decode) ─────────────────────────────
//
// Performs output = input @ dequant(weight).T + bias
// where weight is FP8 E4M3 stored as U8 [N, K] with per-channel F32 scales.
//
// FP8 E4M3fn bit layout: [s(7) | e3(6) e2(5) e1(4) e0(3) | m2(2) m1(1) m0(0)]
//   Exponent bias: 7
//   exp == 0:  subnormal — value = (-1)^s * 2^(-6) * (mantissa / 8)
//   exp in [1..14]: normal — value = (-1)^s * 2^(exp-7) * (1 + mantissa / 8)
//   exp == 15: NaN (E4M3fn has no infinity representation)
//
// Software FP8 decode makes this compatible with Ampere (sm_80) which has no
// hardware FP8 support.

__device__ __forceinline__ float fp8_e4m3_to_float(unsigned char fp8) {
    unsigned int sign     = (unsigned int)(fp8 >> 7) & 1U;
    unsigned int exponent = (unsigned int)(fp8 >> 3) & 0xFU;
    unsigned int mantissa = (unsigned int)(fp8)       & 0x7U;
    float val;
    if (exponent == 15U) {
        // E4M3fn: exponent=15 is always NaN (no infinity in this variant)
        val = __int_as_float(0x7FC00000U);
    } else if (exponent == 0U) {
        // Subnormal: (-1)^s * 2^(-6) * (mantissa / 8)
        // 2^(-6) / 8 = 1/512
        val = (float)mantissa * (1.0f / 512.0f);
    } else {
        // Normal: (-1)^s * 2^(exp - 7) * (1 + mantissa / 8)
        val = ldexpf(1.0f + (float)mantissa * (1.0f / 8.0f), (int)exponent - 7);
    }
    return sign ? -val : val;
}

extern "C" __global__ void marlin_gemm_fp8_bf16(
    __nv_bfloat16* __restrict__ output,         // [M, N]
    const __nv_bfloat16* __restrict__ input,    // [M, K] activations (BF16)
    const unsigned char* __restrict__ qweight,  // [N, K] FP8 E4M3 as U8
    const float* __restrict__ weight_scale,     // [N]    per-channel F32 scales
    int M, int N, int K,
    int has_bias,
    const __nv_bfloat16* __restrict__ bias      // [N]    optional bias
) {
    int tile_n = blockIdx.x * TILE_N;
    int tile_m = blockIdx.y * TILE_M;
    int tid = threadIdx.x;

    __shared__ float smem_input[TILE_M][TILE_K];
    __shared__ float smem_weight[TILE_K][TILE_N];

    float acc[TILE_M];
    int local_n = tid % TILE_N;

    for (int m = 0; m < TILE_M; m++) {
        acc[m] = 0.0f;
    }

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load input tile [TILE_M, TILE_K]
        for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = tile_m + m;
            int global_k = k_start + k;
            if (global_m < M && global_k < K) {
                smem_input[m][k] = bf16_to_float(input[global_m * K + global_k]);
            } else {
                smem_input[m][k] = 0.0f;
            }
        }

        // Decode and dequantize FP8 weight tile [TILE_K, TILE_N]
        // Weight layout [N, K]: element [global_n, global_k] at global_n*K + global_k
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_k = k_start + k;
            int global_n = tile_n + n;
            if (global_k < K && global_n < N) {
                float w = fp8_e4m3_to_float(qweight[global_n * K + global_k]);
                smem_weight[k][n] = w * weight_scale[global_n];
            } else {
                smem_weight[k][n] = 0.0f;
            }
        }

        __syncthreads();

        if (local_n < TILE_N && (tile_n + local_n) < N) {
            for (int m = 0; m < TILE_M; m++) {
                if (tile_m + m < M) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < TILE_K; k++) {
                        sum += smem_input[m][k] * smem_weight[k][local_n];
                    }
                    acc[m] += sum;
                }
            }
        }

        __syncthreads();
    }

    if (local_n < TILE_N && (tile_n + local_n) < N) {
        for (int m = 0; m < TILE_M; m++) {
            int global_m = tile_m + m;
            int global_n = tile_n + local_n;
            if (global_m < M && global_n < N) {
                float val = acc[m];
                if (has_bias && bias != NULL) {
                    val += bf16_to_float(bias[global_n]);
                }
                output[global_m * N + global_n] = float_to_bf16(val);
            }
        }
    }
}
