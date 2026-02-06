// BitsAndBytes Fused Dequantization + Matrix Multiplication CUDA Kernels
//
// Fuses weight dequantization with GEMM to avoid materializing full-precision
// weight tensors in memory. Supports NF4 (4-bit NormalFloat) and INT8 modes.
//
// NF4 format:
//   - Two 4-bit indices packed per uint8 byte (low nibble first, high nibble second)
//   - Each index maps to one of 16 codebook values from the NF4 lookup table
//   - Per-block absmax scaling: dequantized = codebook[index] * absmax[block_idx]
//
// INT8 format:
//   - int8 values stored as uint8 (bit reinterpretation)
//   - Per-block absmax scaling: dequantized = (int8_val / 127.0) * absmax[block_idx]
//
// Kernel computes: output[M, N] = input[M, K] @ weight[N, K].T
//   where weight is stored in quantized form with layout [N * K / pack_factor]
//   for NF4, or [N, K] for INT8.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// NF4 codebook: 16 values from the QLoRA paper, optimized for normally
// distributed weights. Stored in shared memory for fast access.
__constant__ float NF4_CODEBOOK[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

// ============================================================================
// NF4 Fused Dequantize + GEMM (Naive)
// ============================================================================
//
// Each thread computes one output element by iterating over K.
// Grid:  (ceil(N / 16), ceil(M / 16))
// Block: (16, 16)
//
// Weight layout (NF4): packed uint8 array of length (N * K / 2).
//   The weight matrix is logically [N, K] (row-major), packed such that
//   element [n, k] is at flat index (n * K + k), and two consecutive
//   flat-index elements share one byte: even index in low nibble, odd in high.
//
// Parameters:
//   out:      [M, N] output in f32
//   input:    [M, K] input activations in f32
//   packed_w: [N * K / 2] packed NF4 weights
//   absmax:   [num_blocks] per-block absmax scales in f32
//   M, N, K:  matrix dimensions
//   block_size: quantization block size (elements per absmax entry)

extern "C" __global__ void bnb_nf4_gemm_f32_naive(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ absmax,
    const int M,
    const int N,
    const int K,
    const int block_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        // Flat index into the logical [N, K] weight matrix
        const int flat_idx = col * K + k;

        // Unpack the 4-bit NF4 index from the packed byte
        const int byte_idx = flat_idx / 2;
        const int nibble = flat_idx & 1;  // 0 = low nibble, 1 = high nibble
        const uint8_t packed_byte = packed_w[byte_idx];
        const int nf4_idx = nibble ? ((packed_byte >> 4) & 0x0F) : (packed_byte & 0x0F);

        // Look up codebook value and apply absmax scale
        const int blk_idx = flat_idx / block_size;
        const float scale = absmax[blk_idx];
        const float weight_val = NF4_CODEBOOK[nf4_idx] * scale;

        // Accumulate
        acc += input[row * K + k] * weight_val;
    }

    out[row * N + col] = acc;
}

// ============================================================================
// NF4 Fused Dequantize + GEMM (Tiled with shared memory)
// ============================================================================
//
// Uses shared memory for the input tile to improve memory access patterns.
// The codebook is in constant memory for broadcast reads.
//
// Grid:  (ceil(N / 16), ceil(M / 16))
// Block: (16, 16)

#define BNB_TILE_K 32
#define BNB_BLOCK_M 16
#define BNB_BLOCK_N 16

extern "C" __global__ void bnb_nf4_gemm_f32_tiled(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ absmax,
    const int M,
    const int N,
    const int K,
    const int block_size
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * BNB_BLOCK_M + ty;
    const int col = tile_col * BNB_BLOCK_N + tx;

    // Shared memory for input tile (avoid bank conflicts with +1 padding)
    __shared__ float input_tile[BNB_BLOCK_M][BNB_TILE_K + 1];

    float acc = 0.0f;

    // Tile over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += BNB_TILE_K) {
        // Collaboratively load input tile into shared memory
        for (int k = tx; k < BNB_TILE_K && (k_tile + k) < K; k += BNB_BLOCK_N) {
            if (row < M) {
                input_tile[ty][k] = input[row * K + k_tile + k];
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial dot product with on-the-fly NF4 dequantization
        if (row < M && col < N) {
            const int k_end = min(BNB_TILE_K, K - k_tile);
            for (int k = 0; k < k_end; k++) {
                const int global_k = k_tile + k;
                const int flat_idx = col * K + global_k;

                // Unpack NF4 index
                const int byte_idx = flat_idx / 2;
                const int nibble = flat_idx & 1;
                const uint8_t packed_byte = packed_w[byte_idx];
                const int nf4_idx = nibble ? ((packed_byte >> 4) & 0x0F)
                                           : (packed_byte & 0x0F);

                // Dequantize
                const int blk_idx = flat_idx / block_size;
                const float scale = absmax[blk_idx];
                const float weight_val = NF4_CODEBOOK[nf4_idx] * scale;

                acc += input_tile[ty][k] * weight_val;
            }
        }
        __syncthreads();
    }

    // Write output
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

// ============================================================================
// NF4 Fused Dequantize + GEMM with bias
// ============================================================================

extern "C" __global__ void bnb_nf4_gemm_f32_bias(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ packed_w,
    const float* __restrict__ absmax,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int block_size,
    const int has_bias
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * BNB_BLOCK_M + ty;
    const int col = tile_col * BNB_BLOCK_N + tx;

    __shared__ float input_tile[BNB_BLOCK_M][BNB_TILE_K + 1];

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BNB_TILE_K) {
        for (int k = tx; k < BNB_TILE_K && (k_tile + k) < K; k += BNB_BLOCK_N) {
            if (row < M) {
                input_tile[ty][k] = input[row * K + k_tile + k];
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }
        __syncthreads();

        if (row < M && col < N) {
            const int k_end = min(BNB_TILE_K, K - k_tile);
            for (int k = 0; k < k_end; k++) {
                const int global_k = k_tile + k;
                const int flat_idx = col * K + global_k;

                const int byte_idx = flat_idx / 2;
                const int nibble = flat_idx & 1;
                const uint8_t packed_byte = packed_w[byte_idx];
                const int nf4_idx = nibble ? ((packed_byte >> 4) & 0x0F)
                                           : (packed_byte & 0x0F);

                const int blk_idx = flat_idx / block_size;
                const float scale = absmax[blk_idx];
                const float weight_val = NF4_CODEBOOK[nf4_idx] * scale;

                acc += input_tile[ty][k] * weight_val;
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (has_bias && bias != nullptr) {
            acc += bias[col];
        }
        out[row * N + col] = acc;
    }
}

// ============================================================================
// INT8 Fused Dequantize + GEMM (Naive)
// ============================================================================
//
// Weight layout (INT8): [N, K] stored as uint8 (reinterpreted as int8)
// Dequantization: weight_fp = (int8_val / 127.0) * absmax[block_idx]
//
// Grid:  (ceil(N / 16), ceil(M / 16))
// Block: (16, 16)

extern "C" __global__ void bnb_int8_gemm_f32_naive(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ absmax,
    const int M,
    const int N,
    const int K,
    const int block_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        const float input_val = input[row * K + k];

        // Reinterpret uint8 as int8
        const int flat_idx = col * K + k;
        const int8_t int8_val = (int8_t)weight[flat_idx];

        // Dequantize with per-block absmax
        const int blk_idx = flat_idx / block_size;
        const float scale = absmax[blk_idx] / 127.0f;
        const float weight_val = (float)int8_val * scale;

        acc += input_val * weight_val;
    }

    out[row * N + col] = acc;
}

// ============================================================================
// INT8 Fused Dequantize + GEMM (Tiled)
// ============================================================================

extern "C" __global__ void bnb_int8_gemm_f32_tiled(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ absmax,
    const int M,
    const int N,
    const int K,
    const int block_size
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * BNB_BLOCK_M + ty;
    const int col = tile_col * BNB_BLOCK_N + tx;

    __shared__ float input_tile[BNB_BLOCK_M][BNB_TILE_K + 1];

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BNB_TILE_K) {
        for (int k = tx; k < BNB_TILE_K && (k_tile + k) < K; k += BNB_BLOCK_N) {
            if (row < M) {
                input_tile[ty][k] = input[row * K + k_tile + k];
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }
        __syncthreads();

        if (row < M && col < N) {
            const int k_end = min(BNB_TILE_K, K - k_tile);
            for (int k = 0; k < k_end; k++) {
                const int global_k = k_tile + k;
                const int flat_idx = col * K + global_k;

                const int8_t int8_val = (int8_t)weight[flat_idx];
                const int blk_idx = flat_idx / block_size;
                const float scale = absmax[blk_idx] / 127.0f;
                const float weight_val = (float)int8_val * scale;

                acc += input_tile[ty][k] * weight_val;
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

// ============================================================================
// INT8 Fused Dequantize + GEMM with bias
// ============================================================================

extern "C" __global__ void bnb_int8_gemm_f32_bias(
    float* __restrict__ out,
    const float* __restrict__ input,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ absmax,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int block_size,
    const int has_bias
) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = tile_row * BNB_BLOCK_M + ty;
    const int col = tile_col * BNB_BLOCK_N + tx;

    __shared__ float input_tile[BNB_BLOCK_M][BNB_TILE_K + 1];

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += BNB_TILE_K) {
        for (int k = tx; k < BNB_TILE_K && (k_tile + k) < K; k += BNB_BLOCK_N) {
            if (row < M) {
                input_tile[ty][k] = input[row * K + k_tile + k];
            } else {
                input_tile[ty][k] = 0.0f;
            }
        }
        __syncthreads();

        if (row < M && col < N) {
            const int k_end = min(BNB_TILE_K, K - k_tile);
            for (int k = 0; k < k_end; k++) {
                const int global_k = k_tile + k;
                const int flat_idx = col * K + global_k;

                const int8_t int8_val = (int8_t)weight[flat_idx];
                const int blk_idx = flat_idx / block_size;
                const float scale = absmax[blk_idx] / 127.0f;
                const float weight_val = (float)int8_val * scale;

                acc += input_tile[ty][k] * weight_val;
            }
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        if (has_bias && bias != nullptr) {
            acc += bias[col];
        }
        out[row * N + col] = acc;
    }
}
