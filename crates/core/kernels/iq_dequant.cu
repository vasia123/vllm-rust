// I-quant (IQ) dequantization kernels.
//
// candle 0.10 has no I-quant support, so GGUF checkpoints that use the
// Unsloth "UD" dynamic quants (IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS)
// cannot run through candle's quantized path. These kernels dequantize an
// I-quant weight (resident on-device as raw GGUF block bytes) to a dense
// f32 matrix; `IqLinear` then matmuls it. The math is a one-to-one port of
// ggml's `dequantize_row_iq*` (reference/llama.cpp/ggml/src/ggml-quants.c),
// pinned bit-for-bit against the CPU port (crates/core/.../iq/mod.rs), which
// is itself pinned against gguf-py golden vectors from the real model.
//
// Parallelism: one CUDA thread per GGML super-block (QK_K = 256 elements).
// A weight has millions of elements → hundreds of thousands of blocks, so
// block-level parallelism saturates the GPU. The per-thread inner loops
// mirror the CPU reference exactly for auditability; this is the simple,
// correct dequant-then-matmul path (a fused MMVQ kernel is a later perf
// optimisation, tracked separately).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "iq_tables.cuh"

#define QK_K 256

// Sign multiplier for output lane j given a packed sign byte (kmask bit j).
__device__ __forceinline__ float iq_sign(uint8_t signs, int j) {
    return (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
}

// ---- IQ2_XS: block = d(f16) + qs[u16 x32] + scales[u8 x8] = 74 bytes ----
__device__ void deq_block_iq2_xs(const uint8_t* __restrict__ b, float* __restrict__ y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(b + 2);
    const uint8_t* scales = b + 66;
    int yi = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint16_t q = qs[4 * ib32 + l];
            const uint64_t grid = iq2xs_grid[q & 511];
            const uint8_t signs = ksigns_iq2xs[q >> 9];
            const float dl = (l / 2 == 0) ? db0 : db1;
            const uint8_t* g = reinterpret_cast<const uint8_t*>(&grid);
            for (int j = 0; j < 8; ++j) {
                y[yi + j] = dl * (float)g[j] * iq_sign(signs, j);
            }
            yi += 8;
        }
    }
}

// ---- IQ2_S: block = d + qs[u8 x64] + qh[u8 x8] + scales[u8 x8] = 82 bytes ----
__device__ void deq_block_iq2_s(const uint8_t* __restrict__ b, float* __restrict__ y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const uint8_t* qs = b + 2;
    const uint8_t* signs = b + 2 + QK_K / 8; // qs + 32
    const uint8_t* qh = b + 66;
    const uint8_t* scales = b + 74;
    int yi = 0, qs_off = 0, signs_off = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        const float db0 = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f;
        const float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = (l / 2 == 0) ? db0 : db1;
            const int idx = qs[qs_off + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
            const uint64_t grid = iq2s_grid[idx];
            const uint8_t sgn = signs[signs_off + l];
            const uint8_t* g = reinterpret_cast<const uint8_t*>(&grid);
            for (int j = 0; j < 8; ++j) {
                y[yi + j] = dl * (float)g[j] * iq_sign(sgn, j);
            }
            yi += 8;
        }
        qs_off += 4;
        signs_off += 4;
    }
}

// ---- IQ3_XXS: block = d + qs[u8 x96] (last 32 B are scales+signs) = 98 B ----
__device__ void deq_block_iq3_xxs(const uint8_t* __restrict__ b, float* __restrict__ y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const uint8_t* qs = b + 2;
    const uint8_t* sas = b + 66; // qs + QK_K/4
    int yi = 0, qs_off = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32;
        memcpy(&aux32, sas + 4 * ib32, sizeof(uint32_t));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint32_t grid1 = iq3xxs_grid[qs[qs_off + 2 * l]];
            const uint32_t grid2 = iq3xxs_grid[qs[qs_off + 2 * l + 1]];
            const uint8_t* g1 = reinterpret_cast<const uint8_t*>(&grid1);
            const uint8_t* g2 = reinterpret_cast<const uint8_t*>(&grid2);
            for (int j = 0; j < 4; ++j) {
                y[yi + j] = db * (float)g1[j] * iq_sign(signs, j);
                y[yi + j + 4] = db * (float)g2[j] * iq_sign(signs, j + 4);
            }
            yi += 8;
        }
        qs_off += 8;
    }
}

// ---- IQ3_S: d + qs[u8 x64] + qh[u8 x8] + signs[u8 x32] + scales[u8 x4] = 110 B ----
__device__ void deq_block_iq3_s(const uint8_t* __restrict__ b, float* __restrict__ y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const uint8_t* qs = b + 2;
    const uint8_t* qh = b + 66;
    const uint8_t* signs = b + 74;
    const uint8_t* scales = b + 106;
    int yi = 0, qs_off = 0, signs_off = 0, qh_off = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
        const uint8_t sc = scales[ib32 / 2];
        const float db1 = d * (1 + 2 * (sc & 0xf));
        const float db2 = d * (1 + 2 * (sc >> 4));
        for (int l = 0; l < 4; ++l) {
            const uint32_t grid1 = iq3s_grid[qs[qs_off + 2 * l] | ((qh[qh_off] << (8 - 2 * l)) & 256)];
            const uint32_t grid2 = iq3s_grid[qs[qs_off + 2 * l + 1] | ((qh[qh_off] << (7 - 2 * l)) & 256)];
            const uint8_t sgn = signs[signs_off + l];
            const uint8_t* g1 = reinterpret_cast<const uint8_t*>(&grid1);
            const uint8_t* g2 = reinterpret_cast<const uint8_t*>(&grid2);
            for (int j = 0; j < 4; ++j) {
                y[yi + j] = db1 * (float)g1[j] * iq_sign(sgn, j);
                y[yi + j + 4] = db1 * (float)g2[j] * iq_sign(sgn, j + 4);
            }
            yi += 8;
        }
        qs_off += 8;
        signs_off += 4;
        for (int l = 0; l < 4; ++l) {
            const uint32_t grid1 = iq3s_grid[qs[qs_off + 2 * l] | ((qh[qh_off + 1] << (8 - 2 * l)) & 256)];
            const uint32_t grid2 = iq3s_grid[qs[qs_off + 2 * l + 1] | ((qh[qh_off + 1] << (7 - 2 * l)) & 256)];
            const uint8_t sgn = signs[signs_off + l];
            const uint8_t* g1 = reinterpret_cast<const uint8_t*>(&grid1);
            const uint8_t* g2 = reinterpret_cast<const uint8_t*>(&grid2);
            for (int j = 0; j < 4; ++j) {
                y[yi + j] = db2 * (float)g1[j] * iq_sign(sgn, j);
                y[yi + j + 4] = db2 * (float)g2[j] * iq_sign(sgn, j + 4);
            }
            yi += 8;
        }
        qh_off += 2;
        qs_off += 8;
        signs_off += 4;
    }
}

// ---- IQ4_XS: d + scales_h(u16) + scales_l[u8 x4] + qs[u8 x128] = 136 B ----
__device__ void deq_block_iq4_xs(const uint8_t* __restrict__ b, float* __restrict__ y) {
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    uint16_t scales_h;
    memcpy(&scales_h, b + 2, sizeof(uint16_t));
    const uint8_t* scales_l = b + 4;
    const uint8_t* qs = b + 8;
    int yi = 0;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        const uint8_t* q = qs + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            y[yi + j] = dl * (float)kvalues_iq4nl[q[j] & 0xf];
            y[yi + 16 + j] = dl * (float)kvalues_iq4nl[q[j] >> 4];
        }
        yi += 32;
    }
}

// One thread per super-block. grid.x * blockDim.x >= n_blocks.
#define IQ_DEQUANT_KERNEL(NAME, TYPE_SIZE, DEQ_FN)                                    \
    extern "C" __global__ void NAME(const uint8_t* __restrict__ blocks,              \
                                    float* __restrict__ out, int n_blocks) {         \
        int blk = blockIdx.x * blockDim.x + threadIdx.x;                             \
        if (blk >= n_blocks) return;                                                 \
        DEQ_FN(blocks + (long)blk * (TYPE_SIZE), out + (long)blk * QK_K);            \
    }

IQ_DEQUANT_KERNEL(dequantize_iq2_xs_f32, 74, deq_block_iq2_xs)
IQ_DEQUANT_KERNEL(dequantize_iq2_s_f32, 82, deq_block_iq2_s)
IQ_DEQUANT_KERNEL(dequantize_iq3_xxs_f32, 98, deq_block_iq3_xxs)
IQ_DEQUANT_KERNEL(dequantize_iq3_s_f32, 110, deq_block_iq3_s)
IQ_DEQUANT_KERNEL(dequantize_iq4_xs_f32, 136, deq_block_iq4_xs)

// ---- Fused IQ × dense activation (decode GEMV) ----------------------------
//
// Computes Y = X @ Wᵀ where W is the I-quant weight `[n_out, n_in]` (block
// bytes) and X is f32 `[M, n_in]`, producing Y f32 `[M, n_out]`. One CUDA
// block per output row `o`: its threads split the row's QK_K sub-blocks,
// dequantize each in place (reusing the verified deq_block_*), accumulate the
// dot against every one of the M activation rows, then block-reduce. The
// dense weight is NEVER materialized — a single pass over the I-quant bytes,
// unlike the dequant-then-matmul path. For decode M is tiny (≤ batch); larger
// M (prefill) uses the dequant+cuBLAS path on the Rust side.
//
// X must be contiguous `[M, n_in]` and M <= IQ_GEMV_MAX_M.
//
// One WARP per output row (warp-shuffle reduction, no shared mem / no
// __syncthreads); IQ_GEMV_ROWS_PER_BLOCK rows per CUDA block. A warp's 32
// lanes stride over the row's QK_K sub-blocks, each lane dequantizing its
// sub-blocks and dotting them against every activation row.
#define IQ_GEMV_MAX_M 16
#define IQ_GEMV_ROWS_PER_BLOCK 4
#define IQ_GEMV_THREADS (32 * IQ_GEMV_ROWS_PER_BLOCK)

#define IQ_GEMV_KERNEL(NAME, TYPE_SIZE, DEQ_FN)                                          \
    extern "C" __global__ void NAME(const uint8_t* __restrict__ W,                       \
                                    const float* __restrict__ X, float* __restrict__ Y,  \
                                    int n_out, int n_in, int M) {                         \
        const int lane = threadIdx.x & 31;                                               \
        const int warp = threadIdx.x >> 5;                                               \
        const int o = blockIdx.x * IQ_GEMV_ROWS_PER_BLOCK + warp;                        \
        if (o >= n_out) return;                                                          \
        const int nb = n_in / QK_K;                                                      \
        float deq[QK_K];                                                                 \
        float acc[IQ_GEMV_MAX_M];                                                        \
        for (int m = 0; m < M; ++m) acc[m] = 0.0f;                                       \
        for (int blk = lane; blk < nb; blk += 32) {                                      \
            DEQ_FN(W + ((long)o * nb + blk) * (TYPE_SIZE), deq);                         \
            const int base = blk * QK_K;                                                 \
            for (int m = 0; m < M; ++m) {                                                 \
                const float* __restrict__ xm = X + (long)m * n_in + base;                \
                float s = 0.0f;                                                          \
                for (int j = 0; j < QK_K; ++j) s += deq[j] * xm[j];                      \
                acc[m] += s;                                                             \
            }                                                                            \
        }                                                                                \
        for (int m = 0; m < M; ++m) {                                                     \
            float v = acc[m];                                                            \
            for (int off = 16; off > 0; off >>= 1)                                       \
                v += __shfl_down_sync(0xffffffff, v, off);                               \
            if (lane == 0) Y[(long)m * n_out + o] = v;                                   \
        }                                                                                \
    }

IQ_GEMV_KERNEL(gemv_iq2_xs_f32, 74, deq_block_iq2_xs)
IQ_GEMV_KERNEL(gemv_iq2_s_f32, 82, deq_block_iq2_s)
IQ_GEMV_KERNEL(gemv_iq3_xxs_f32, 98, deq_block_iq3_xxs)
IQ_GEMV_KERNEL(gemv_iq3_s_f32, 110, deq_block_iq3_s)
IQ_GEMV_KERNEL(gemv_iq4_xs_f32, 136, deq_block_iq4_xs)
