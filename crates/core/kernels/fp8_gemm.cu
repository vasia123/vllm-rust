// FP8 GEMM Kernels for Ada Lovelace+ GPUs (sm_89+)
//
// Implements fused FP8 weight dequantization with GEMM:
//   output = input @ (weight_fp8 * scale).T
//
// Where:
//   input:  [M, K] in BF16
//   weight: [N, K] in FP8 (stored row-major, transposed in matmul)
//   scale:  [1] or [N] in F32
//   output: [M, N] in BF16
//
// This is a reference implementation. For production, use:
// - cuBLASLt with FP8 data types
// - CUTLASS FP8 GEMM templates

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define BLOCK_SIZE 256

// Convert FP8 E4M3 to float
__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return float(val);
}

// ============================================================================
// Naive FP8 GEMM: Dequantize-fused matmul
// ============================================================================

// Simple implementation: each thread computes one output element
// Grid: (ceil(N/16), ceil(M/16))
// Block: (16, 16)
extern "C" __global__ void fp8_gemm_bf16_naive(
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const __nv_bfloat16* __restrict__ input, // [M, K]
    const __nv_fp8_e4m3* __restrict__ weight, // [N, K] (transposed access)
    const float* __restrict__ scale,          // [1] or [N]
    const int M,
    const int N,
    const int K,
    const bool per_channel_scale
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row >= M || col >= N) return;

    // Get scale for this output column
    const float weight_scale = per_channel_scale ? scale[col] : scale[0];

    // Compute dot product
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float input_val = __bfloat162float(input[row * K + k]);
        float weight_val = fp8_to_float(weight[col * K + k]) * weight_scale;
        acc += input_val * weight_val;
    }

    out[row * N + col] = __float2bfloat16(acc);
}

// ============================================================================
// Tiled FP8 GEMM with shared memory
// ============================================================================

// Uses shared memory tiling for better memory access patterns
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (16, 16)
extern "C" __global__ void fp8_gemm_bf16_tiled(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale,
    const int M,
    const int N,
    const int K,
    const bool per_channel_scale
) {
    // Tile indices
    const int tile_row = blockIdx.y;  // Which tile in M
    const int tile_col = blockIdx.x;  // Which tile in N

    // Thread indices within tile
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15

    // Global indices
    const int row = tile_row * 16 + ty;
    const int col = tile_col * 16 + tx;

    // Shared memory for tiles
    __shared__ float input_tile[16][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ float weight_tile[16][TILE_K + 1];

    // Get scale for this column
    const float weight_scale = (col < N && per_channel_scale) ? scale[col] :
                               (col < N ? scale[0] : 1.0f);

    // Accumulator
    float acc = 0.0f;

    // Loop over K in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load input tile: each thread loads multiple elements
        for (int k = tx; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (row < M) {
                input_tile[ty][k] = __bfloat162float(input[row * K + k_tile + k]);
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }

        // Load weight tile with dequantization
        for (int k = ty; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (col < N) {
                float w = fp8_to_float(weight[col * K + k_tile + k]);
                weight_tile[tx][k] = w * weight_scale;
            } else {
                weight_tile[tx][k] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot product
        const int k_end = min(TILE_K, K - k_tile);
        for (int k = 0; k < k_end; k++) {
            acc += input_tile[ty][k] * weight_tile[tx][k];
        }

        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        out[row * N + col] = __float2bfloat16(acc);
    }
}

// ============================================================================
// FP8 GEMM with bias addition
// ============================================================================

extern "C" __global__ void fp8_gemm_bf16_with_bias(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ bias,  // [N] or nullptr
    const int M,
    const int N,
    const int K,
    const bool per_channel_scale,
    const bool has_bias
) {
    // Tile indices
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * 16 + ty;
    const int col = tile_col * 16 + tx;

    __shared__ float input_tile[16][TILE_K + 1];
    __shared__ float weight_tile[16][TILE_K + 1];

    const float weight_scale = (col < N && per_channel_scale) ? scale[col] :
                               (col < N ? scale[0] : 1.0f);

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (int k = tx; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (row < M) {
                input_tile[ty][k] = __bfloat162float(input[row * K + k_tile + k]);
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }

        for (int k = ty; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (col < N) {
                float w = fp8_to_float(weight[col * K + k_tile + k]);
                weight_tile[tx][k] = w * weight_scale;
            } else {
                weight_tile[tx][k] = 0.0f;
            }
        }

        __syncthreads();

        const int k_end = min(TILE_K, K - k_tile);
        for (int k = 0; k < k_end; k++) {
            acc += input_tile[ty][k] * weight_tile[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (has_bias && bias != nullptr) {
            acc += __bfloat162float(bias[col]);
        }
        out[row * N + col] = __float2bfloat16(acc);
    }
}

// ============================================================================
// Vectorized FP8 Weight Dequantization (for use with cuBLAS)
// ============================================================================

// Dequantize FP8 weights to BF16 for use with standard GEMM
// Grid: (ceil(K/256), N)
// Block: (256,)
extern "C" __global__ void fp8_dequant_weights_bf16(
    __nv_bfloat16* __restrict__ out,         // [N, K]
    const __nv_fp8_e4m3* __restrict__ weight, // [N, K]
    const float* __restrict__ scale,          // [1] or [N]
    const int N,
    const int K,
    const bool per_channel_scale
) {
    const int row = blockIdx.y;  // Output channel
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // K dimension

    if (row >= N || col >= K) return;

    const float weight_scale = per_channel_scale ? scale[row] : scale[0];
    float val = fp8_to_float(weight[row * K + col]) * weight_scale;
    out[row * K + col] = __float2bfloat16(val);
}

// ============================================================================
// FP8 Input Quantization for mixed-precision GEMM
// ============================================================================

// Quantize BF16 input to FP8 with dynamic per-row scaling
// Useful for W8A8 (weight FP8, activation FP8) GEMM
extern "C" __global__ void fp8_quantize_input_bf16(
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    const int M,
    const int K
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const __nv_bfloat16* row_in = input + row * K;
    __nv_fp8_e4m3* row_out = out + row * K;

    // Find row max
    float thread_max = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        float val = fabsf(__bfloat162float(row_in[k]));
        thread_max = fmaxf(thread_max, val);
    }

    // Block reduction for max
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }

    if (lane_id == 0) {
        smem[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();

    float row_max = smem[0];

    // Compute scale: ensure we don't divide by zero
    __shared__ float row_scale;
    if (tid == 0) {
        row_scale = fmaxf(row_max / 448.0f, 1e-12f);  // 448 = FP8 E4M3 max
        scales[row] = row_scale;
    }
    __syncthreads();

    // Quantize
    const float inv_scale = 1.0f / row_scale;
    for (int k = tid; k < K; k += blockDim.x) {
        float val = __bfloat162float(row_in[k]) * inv_scale;
        row_out[k] = __nv_fp8_e4m3(val);
    }
}

// ============================================================================
// Scaled MM: W8A8 FP8 GEMM with per-tensor/per-channel scaling
// ============================================================================
//
// Implements: output = (a_scale * A_fp8) @ (b_scale * B_fp8).T
// where both A and B are in FP8 E4M3 format.
//
// This is the key operation for FP8 quantized inference (H100/Ada):
// - Both activations and weights are FP8
// - Scales are applied during accumulation in FP32
// - Output is BF16
//
// Supports:
// - Per-tensor scaling: a_scale [1], b_scale [1]
// - Per-channel scaling: a_scale [M] (per-row), b_scale [N] (per-col)
// - Per-token-per-channel: a_scale [M], b_scale [N]
//
// Grid: (ceil(N/16), ceil(M/16))
// Block: (16, 16)
extern "C" __global__ void fp8_scaled_mm_bf16(
    __nv_bfloat16* __restrict__ out,           // [M, N]
    const __nv_fp8_e4m3* __restrict__ A,       // [M, K] FP8
    const __nv_fp8_e4m3* __restrict__ B,       // [N, K] FP8 (transposed)
    const float* __restrict__ a_scale,         // [1] or [M]
    const float* __restrict__ b_scale,         // [1] or [N]
    const __nv_bfloat16* __restrict__ bias,    // [N] or nullptr
    const int M,
    const int N,
    const int K,
    const int a_scale_mode,  // 0=per-tensor, 1=per-row
    const int b_scale_mode,  // 0=per-tensor, 1=per-col
    const int has_bias
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * 16 + ty;
    const int col = tile_col * 16 + tx;

    __shared__ float a_tile[16][TILE_K + 1];
    __shared__ float b_tile[16][TILE_K + 1];

    float acc = 0.0f;

    // Pre-load scales
    const float s_a = (row < M) ? (a_scale_mode == 1 ? a_scale[row] : a_scale[0]) : 0.0f;
    const float s_b = (col < N) ? (b_scale_mode == 1 ? b_scale[col] : b_scale[0]) : 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile with dequant and scale
        for (int k = tx; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (row < M) {
                a_tile[ty][k] = fp8_to_float(A[row * K + k_tile + k]) * s_a;
            } else {
                a_tile[ty][k] = 0.0f;
            }
        }

        // Load B tile with dequant and scale
        for (int k = ty; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (col < N) {
                b_tile[tx][k] = fp8_to_float(B[col * K + k_tile + k]) * s_b;
            } else {
                b_tile[tx][k] = 0.0f;
            }
        }

        __syncthreads();

        const int k_end = min(TILE_K, K - k_tile);
        #pragma unroll 8
        for (int k = 0; k < k_end; k++) {
            acc += a_tile[ty][k] * b_tile[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (has_bias && bias != nullptr) {
            acc += __bfloat162float(bias[col]);
        }
        out[row * N + col] = __float2bfloat16(acc);
    }
}

// ============================================================================
// Scaled MM with FP32 output (for intermediate accumulation)
// ============================================================================
//
// Same as above but outputs F32 instead of BF16.
// Useful when the output feeds into another kernel (e.g., fused LayerNorm).
extern "C" __global__ void fp8_scaled_mm_f32(
    float* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_scale,
    const int M,
    const int N,
    const int K,
    const int a_scale_mode,
    const int b_scale_mode
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * 16 + ty;
    const int col = tile_col * 16 + tx;

    __shared__ float a_tile[16][TILE_K + 1];
    __shared__ float b_tile[16][TILE_K + 1];

    float acc = 0.0f;

    const float s_a = (row < M) ? (a_scale_mode == 1 ? a_scale[row] : a_scale[0]) : 0.0f;
    const float s_b = (col < N) ? (b_scale_mode == 1 ? b_scale[col] : b_scale[0]) : 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        for (int k = tx; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (row < M) {
                a_tile[ty][k] = fp8_to_float(A[row * K + k_tile + k]) * s_a;
            } else {
                a_tile[ty][k] = 0.0f;
            }
        }

        for (int k = ty; k < TILE_K && (k_tile + k) < K; k += 16) {
            if (col < N) {
                b_tile[tx][k] = fp8_to_float(B[col * K + k_tile + k]) * s_b;
            } else {
                b_tile[tx][k] = 0.0f;
            }
        }

        __syncthreads();

        const int k_end = min(TILE_K, K - k_tile);
        #pragma unroll 8
        for (int k = 0; k < k_end; k++) {
            acc += a_tile[ty][k] * b_tile[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}
