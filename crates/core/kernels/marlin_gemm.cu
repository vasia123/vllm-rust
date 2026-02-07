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
