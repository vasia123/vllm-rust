// MXFP4 E2M1 × BF16 tiled GEMM with blockwise E8M0 scales.
//
// Performs output = input @ dequant(weight).T + bias
// where weight is OCP MXFP4 E2M1 packed as U8 [N, K/2] (2 nibbles per byte,
// lower nibble = lower K index) and scales are E8M0 [N, K/32] U8 with one
// scale per 32-element block along K.
//
// FP4 E2M1 nibble format: [s(3) | e1(2) | e0(1) | m0(0)]
//   Positive values (s=0): 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
//   Negative values (s=1): -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
//
// E8M0 scale format: value = 2^(byte_value - 127)
//
// Software FP4 decode makes this compatible with Ampere (sm_80) and newer.

#include <cuda_bf16.h>

// Tile sizes — must match blockDim.x = TILE_M * TILE_K = TILE_K * TILE_N = 256
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// One E8M0 scale covers this many FP4 elements along K
#define MXFP4_BLOCK_SIZE 32

// FP4 E2M1 look-up table: 16 entries indexed by 4-bit nibble
__device__ __constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Decode an E8M0 scale byte to float: value = 2^(byte - 127)
__device__ __forceinline__ float e8m0_to_float(unsigned char e8m0) {
    return ldexpf(1.0f, (int)e8m0 - 127);
}

// Extract a FP4 nibble from a packed byte and return its float value.
// which=0: lower nibble (bits 3:0); which=1: upper nibble (bits 7:4).
__device__ __forceinline__ float unpack_fp4(unsigned char packed, int which) {
    unsigned char nibble = (which == 0)
        ? (packed & 0xFU)
        : ((packed >> 4) & 0xFU);
    return FP4_LUT[nibble];
}

// ─── MXFP4 GEMM kernel ───────────────────────────────────────────────────────

extern "C" __global__ void mxfp4_gemm_bf16(
    __nv_bfloat16* __restrict__ output,         // [M, N]
    const __nv_bfloat16* __restrict__ input,    // [M, K] activations (BF16)
    const unsigned char* __restrict__ qweight,  // [N, K/2] packed FP4 (lower nibble = lower K)
    const unsigned char* __restrict__ scales,   // [N, K/32] E8M0 block scales
    int M, int N, int K,
    int has_bias,
    const __nv_bfloat16* __restrict__ bias      // [N] optional bias
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

    // Precomputed strides
    int half_K    = K / 2;                    // bytes per row in qweight
    int scale_cols = K / MXFP4_BLOCK_SIZE;    // scale columns per row

    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // ── Load input tile [TILE_M, TILE_K] ──────────────────────────────
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

        // ── Decode and dequantize FP4 weight tile [TILE_K, TILE_N] ────────
        // Weight layout [N, K/2]: element at (global_n, global_k) is nibble
        //   global_k even → lower nibble of byte at global_n * half_K + global_k/2
        //   global_k odd  → upper nibble of same byte
        // Scale layout  [N, K/32]: scale at global_n * scale_cols + global_k/32
        for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_k = k_start + k;
            int global_n = tile_n + n;

            if (global_k < K && global_n < N) {
                int byte_idx  = global_n * half_K + global_k / 2;
                int which     = global_k & 1;            // 0=lower nibble, 1=upper
                float fp4_val = unpack_fp4(qweight[byte_idx], which);

                int scale_idx = global_n * scale_cols + global_k / MXFP4_BLOCK_SIZE;
                float scale   = e8m0_to_float(scales[scale_idx]);

                smem_weight[k][n] = fp4_val * scale;
            } else {
                smem_weight[k][n] = 0.0f;
            }
        }

        __syncthreads();

        // ── Accumulate partial products ────────────────────────────────────
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

    // ── Write output ────────────────────────────────────────────────────────
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
