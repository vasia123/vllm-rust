// GPTQ Dequantization Kernels for INT4/INT8
//
// Unpacks quantized weights from packed INT32 format and applies
// per-group scales and zero points.
//
// Weight packing format (4-bit, 8 values per INT32):
//   bits[0:3]   = weight[0]
//   bits[4:7]   = weight[1]
//   bits[8:11]  = weight[2]
//   ...
//   bits[28:31] = weight[7]
//
// Dequantization formula:
//   weight_fp = (weight_int - zero_point) * scale

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// INT4 Dequantization (8 weights per INT32)
// ============================================================================

// Extract 4-bit value at position idx from packed INT32
__device__ __forceinline__ int extract_int4(uint32_t packed, int idx) {
    return (packed >> (idx * 4)) & 0xF;
}

// Extract zero point for a given output column (stored in packed format)
__device__ __forceinline__ int extract_zero_int4(
    const uint32_t* qzeros,
    int group_idx,
    int out_col,
    int num_groups
) {
    // qzeros layout: [num_groups, packed_out_features]
    // where packed_out_features = out_features / 8 (for 4-bit)
    int packed_idx = out_col / 8;
    int bit_idx = out_col % 8;
    uint32_t packed = qzeros[group_idx * ((num_groups > 1) ? (packed_idx + 1) : 1) + packed_idx];
    return extract_int4(packed, bit_idx);
}

// GPTQ INT4 dequantization to BF16
// Grid: (ceil(out_features/16), ceil(in_features/16))
// Block: (16, 16)
extern "C" __global__ void gptq_dequant_int4_bf16(
    __nv_bfloat16* __restrict__ out,    // [in_features, out_features]
    const uint32_t* __restrict__ qweight, // [in_features/8, out_features]
    const __nv_bfloat16* __restrict__ scales, // [num_groups, out_features]
    const uint32_t* __restrict__ qzeros,      // [num_groups, out_features/8]
    const int in_features,
    const int out_features,
    const int group_size,
    const int num_groups
) {
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_col >= out_features || in_row >= in_features) return;

    // Determine which group this row belongs to
    const int group_idx = (group_size > 0) ? (in_row / group_size) : 0;

    // Get scale for this group/column
    const float scale = __bfloat162float(scales[group_idx * out_features + out_col]);

    // Get zero point (packed format)
    const int zero_packed_idx = out_col / 8;
    const int zero_bit_idx = out_col % 8;
    const int zeros_per_row = (out_features + 7) / 8;
    const uint32_t zero_packed = qzeros[group_idx * zeros_per_row + zero_packed_idx];
    const int zero_point = extract_int4(zero_packed, zero_bit_idx);

    // Get quantized weight (packed format)
    const int weight_packed_idx = in_row / 8;
    const int weight_bit_idx = in_row % 8;
    const int weights_per_row = (in_features + 7) / 8;
    const uint32_t weight_packed = qweight[weight_packed_idx * out_features + out_col];
    const int weight_int = extract_int4(weight_packed, weight_bit_idx);

    // Dequantize: (weight - zero) * scale
    const float weight_fp = (float)(weight_int - zero_point) * scale;

    out[in_row * out_features + out_col] = __float2bfloat16(weight_fp);
}

// GPTQ INT4 dequantization to FP16
extern "C" __global__ void gptq_dequant_int4_fp16(
    __half* __restrict__ out,
    const uint32_t* __restrict__ qweight,
    const __half* __restrict__ scales,
    const uint32_t* __restrict__ qzeros,
    const int in_features,
    const int out_features,
    const int group_size,
    const int num_groups
) {
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_col >= out_features || in_row >= in_features) return;

    const int group_idx = (group_size > 0) ? (in_row / group_size) : 0;

    const float scale = __half2float(scales[group_idx * out_features + out_col]);

    const int zero_packed_idx = out_col / 8;
    const int zero_bit_idx = out_col % 8;
    const int zeros_per_row = (out_features + 7) / 8;
    const uint32_t zero_packed = qzeros[group_idx * zeros_per_row + zero_packed_idx];
    const int zero_point = extract_int4(zero_packed, zero_bit_idx);

    const int weight_packed_idx = in_row / 8;
    const int weight_bit_idx = in_row % 8;
    const uint32_t weight_packed = qweight[weight_packed_idx * out_features + out_col];
    const int weight_int = extract_int4(weight_packed, weight_bit_idx);

    const float weight_fp = (float)(weight_int - zero_point) * scale;

    out[in_row * out_features + out_col] = __float2half(weight_fp);
}

// ============================================================================
// INT8 Dequantization (4 weights per INT32)
// ============================================================================

__device__ __forceinline__ int extract_int8(uint32_t packed, int idx) {
    return (packed >> (idx * 8)) & 0xFF;
}

extern "C" __global__ void gptq_dequant_int8_bf16(
    __nv_bfloat16* __restrict__ out,
    const uint32_t* __restrict__ qweight, // [in_features/4, out_features]
    const __nv_bfloat16* __restrict__ scales,
    const uint32_t* __restrict__ qzeros,  // [num_groups, out_features/4]
    const int in_features,
    const int out_features,
    const int group_size,
    const int num_groups
) {
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_col >= out_features || in_row >= in_features) return;

    const int group_idx = (group_size > 0) ? (in_row / group_size) : 0;

    const float scale = __bfloat162float(scales[group_idx * out_features + out_col]);

    // INT8: 4 values per INT32
    const int zero_packed_idx = out_col / 4;
    const int zero_bit_idx = out_col % 4;
    const int zeros_per_row = (out_features + 3) / 4;
    const uint32_t zero_packed = qzeros[group_idx * zeros_per_row + zero_packed_idx];
    const int zero_point = extract_int8(zero_packed, zero_bit_idx);

    const int weight_packed_idx = in_row / 4;
    const int weight_bit_idx = in_row % 4;
    const uint32_t weight_packed = qweight[weight_packed_idx * out_features + out_col];
    const int weight_int = extract_int8(weight_packed, weight_bit_idx);

    const float weight_fp = (float)(weight_int - zero_point) * scale;

    out[in_row * out_features + out_col] = __float2bfloat16(weight_fp);
}

// ============================================================================
// Fused GPTQ GEMM: Dequant + Matmul (simple version)
// ============================================================================

// Naive fused kernel: each thread computes one output element
// Grid: (ceil(N/16), ceil(M/16))
// Block: (16, 16)
// Input: [M, K]  Weight: [K, N] (packed)  Output: [M, N]
extern "C" __global__ void gptq_gemm_int4_bf16_naive(
    __nv_bfloat16* __restrict__ out,        // [M, N]
    const __nv_bfloat16* __restrict__ input, // [M, K]
    const uint32_t* __restrict__ qweight,    // [K/8, N]
    const __nv_bfloat16* __restrict__ scales, // [num_groups, N]
    const uint32_t* __restrict__ qzeros,      // [num_groups, N/8]
    const __nv_bfloat16* __restrict__ bias,  // [N] or nullptr
    const int M,
    const int N,
    const int K,
    const int group_size,
    const int num_groups,
    const bool has_bias
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row >= M || col >= N) return;

    // Precompute zero points for each group affecting this column
    const int zeros_per_row = (N + 7) / 8;
    const int zero_packed_idx = col / 8;
    const int zero_bit_idx = col % 8;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        // Input value
        const float input_val = __bfloat162float(input[row * K + k]);

        // Determine group for this k
        const int group_idx = (group_size > 0) ? (k / group_size) : 0;

        // Get scale and zero point
        const float scale = __bfloat162float(scales[group_idx * N + col]);
        const uint32_t zero_packed = qzeros[group_idx * zeros_per_row + zero_packed_idx];
        const int zero_point = extract_int4(zero_packed, zero_bit_idx);

        // Get quantized weight
        const int weight_packed_idx = k / 8;
        const int weight_bit_idx = k % 8;
        const uint32_t weight_packed = qweight[weight_packed_idx * N + col];
        const int weight_int = extract_int4(weight_packed, weight_bit_idx);

        // Dequantize weight
        const float weight_fp = (float)(weight_int - zero_point) * scale;

        acc += input_val * weight_fp;
    }

    // Add bias if present
    if (has_bias && bias != nullptr) {
        acc += __bfloat162float(bias[col]);
    }

    out[row * N + col] = __float2bfloat16(acc);
}

// ============================================================================
// Tiled GPTQ GEMM with shared memory
// ============================================================================

#define TILE_K 32
#define BLOCK_M 16
#define BLOCK_N 16

extern "C" __global__ void gptq_gemm_int4_bf16_tiled(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ qweight,
    const __nv_bfloat16* __restrict__ scales,
    const uint32_t* __restrict__ qzeros,
    const __nv_bfloat16* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int group_size,
    const int num_groups,
    const bool has_bias
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * BLOCK_M + ty;
    const int col = tile_col * BLOCK_N + tx;

    // Shared memory for input tile
    __shared__ float input_tile[BLOCK_M][TILE_K + 1];

    // Zero point info for this column (precompute)
    const int zeros_per_row = (N + 7) / 8;
    const int zero_packed_idx = (col < N) ? (col / 8) : 0;
    const int zero_bit_idx = col % 8;

    float acc = 0.0f;

    // Tile over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load input tile
        for (int k = tx; k < TILE_K && (k_tile + k) < K; k += BLOCK_N) {
            if (row < M) {
                input_tile[ty][k] = __bfloat162float(input[row * K + k_tile + k]);
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial dot product with dequantization
        if (row < M && col < N) {
            const int k_end = min(TILE_K, K - k_tile);
            for (int k = 0; k < k_end; k++) {
                const int global_k = k_tile + k;
                const int group_idx = (group_size > 0) ? (global_k / group_size) : 0;

                const float scale = __bfloat162float(scales[group_idx * N + col]);
                const uint32_t zero_packed = qzeros[group_idx * zeros_per_row + zero_packed_idx];
                const int zero_point = extract_int4(zero_packed, zero_bit_idx);

                const int weight_packed_idx = global_k / 8;
                const int weight_bit_idx = global_k % 8;
                const uint32_t weight_packed = qweight[weight_packed_idx * N + col];
                const int weight_int = extract_int4(weight_packed, weight_bit_idx);

                const float weight_fp = (float)(weight_int - zero_point) * scale;

                acc += input_tile[ty][k] * weight_fp;
            }
        }
        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        if (has_bias && bias != nullptr) {
            acc += __bfloat162float(bias[col]);
        }
        out[row * N + col] = __float2bfloat16(acc);
    }
}
