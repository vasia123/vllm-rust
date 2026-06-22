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
// Two paths share these device decoders:
//
//   1. Dequant kernels (one thread per QK_K=256 super-block) materialize a
//      dense f32 weight; the Rust side then does a cuBLAS GEMM. Used for
//      prefill (large M), where the materialization amortizes over the prompt.
//
//   2. MMVQ kernels (mul_mat_vec_q, the decode hot path) quantize the
//      activation row to q8_1 (int8 + per-32 scale) once, then dot the
//      I-quant weight against it with integer `__dp4a` — no dense weight, no
//      float multiplies in the inner loop. This is a one-to-one port of
//      ggml's `vec_dot_iq*_q8_1` (reference/llama.cpp/.../vecdotq.cuh) and is
//      what llama.cpp itself runs for I-quant decode.
//
// The per-thread inner loops mirror the CPU reference exactly for
// auditability (crates/core/.../iq/mod.rs), itself pinned against gguf-py.

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

// ===========================================================================
// MMVQ (mul_mat_vec_q): q8_1-quantized activation × I-quant weight, integer
// dot. The decode hot path. Computes Y = X @ Wᵀ for W `[n_out, n_in]`
// (I-quant block bytes) and X f32 `[M, n_in]` (M <= IQ_GEMV_MAX_M), via:
//
//   step 1  quantize_q8_1_rows: X → (Aq int8 `[M, n_in]`, Ad f32 `[M, n_in/32]`)
//           per 32-element block: d = amax/127, q = round(x/d). Done ONCE,
//           reused across every output row.
//   step 2  mmvq_iq*_f32: for each output row, sum int8 `__dp4a` dots of the
//           weight's codebook values (sign-flipped on the fly) against the
//           q8_1 quants, scaled by the per-block I-quant scale and the
//           weight/activation deltas. No dense weight is materialized.
//
// The math is a one-to-one port of ggml's `vec_dot_iq*_q8_1`. We stride over
// 32-element sub-blocks (`g`) so every one of the 32 warp lanes stays busy
// even for the narrow rows (n_in as small as 256·k); each lane re-derives its
// super-block base. One WARP per output row, IQ_GEMV_ROWS_PER_BLOCK rows per
// CUDA block, warp-shuffle reduction (no shared mem / no __syncthreads).
// ===========================================================================

#define IQ_GEMV_MAX_M 16
#define IQ_GEMV_ROWS_PER_BLOCK 4
#define IQ_GEMV_THREADS (32 * IQ_GEMV_ROWS_PER_BLOCK)

// SIMD int8x4 dot-product with accumulate (sm_61+). a, b each pack four int8.
__device__ __forceinline__ int iq_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

// Expand a 7-bit ggml sign index to a per-byte selector (mirrors
// vecdotq.cuh::unpack_ksigns): the 8th sign is the parity of the 7 bits.
__device__ __forceinline__ uint32_t iq_unpack_ksigns(uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

// Per-byte negate of the four int8 in `g` where the matching byte of `signs`
// is set (0xFF). signs comes from __vcmpne4(mask & selector, 0).
__device__ __forceinline__ int iq_apply_signs(int g, int signs) {
    return __vsub4(g ^ signs, signs);
}

// q4 packs eight 4-bit indices (4 bytes). Returns the table[] lookups as two
// int8x4: .x = even-nibble (low) indices, .y = odd-nibble (high) indices.
__device__ __forceinline__ int2 iq_table16(int q4, const int8_t* table) {
    const int lo = (q4 >> 0) & 0x0F0F0F0F;
    const int hi = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t* lob = reinterpret_cast<const int8_t*>(&lo);
    const int8_t* hib = reinterpret_cast<const int8_t*>(&hi);
    const char4 v0 = make_char4(table[lob[0]], table[lob[1]], table[lob[2]], table[lob[3]]);
    const char4 v1 = make_char4(table[hib[0]], table[hib[1]], table[hib[2]], table[hib[3]]);
    return make_int2(*reinterpret_cast<const int*>(&v0), *reinterpret_cast<const int*>(&v1));
}

// ---- q8_1 activation quantizer: one thread per 32-element block ----
// Aq holds int8 quants (stored in a u8 buffer); Ad holds the f32 delta d.
extern "C" __global__ void quantize_q8_1_rows(const float* __restrict__ X,
                                              uint8_t* __restrict__ Aq,
                                              float* __restrict__ Ad, int total_blocks) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= total_blocks) return;
    const float* __restrict__ x = X + (long)blk * 32;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f;
    Ad[blk] = d;
    int8_t* q = reinterpret_cast<int8_t*>(Aq) + (long)blk * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = (amax == 0.0f) ? (int8_t)0 : (int8_t)roundf(x[i] / d);
}

// ---- Per-(sub-block, activation-block) integer dot, one per I-quant type ----
// `b` points to the weight super-block, `ib` is the 32-lane sub-block (0..8),
// `u` are the eight int8x4 of the matching q8_1 activation block. Each returns
// the scaled integer partial sum; the caller folds in the f32 deltas.

__device__ __forceinline__ int dot_iq2_xs(const uint8_t* b, int ib, const int* u) {
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(b + 2);
    const uint8_t* scales = b + 66;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint16_t q = qs[4 * ib + k];
        const uint2 g = ((const uint2*)iq2xs_grid)[q & 0x1FF];
        const uint32_t signs = iq_unpack_ksigns(q >> 9);
        const int gl = iq_apply_signs(g.x, __vcmpne4(signs & 0x08040201, 0));
        const int gh = iq_apply_signs(g.y, __vcmpne4(signs & 0x80402010, 0));
        if (k < 2) {
            s0 = iq_dp4a(gl, u[2 * k], s0);
            s0 = iq_dp4a(gh, u[2 * k + 1], s0);
        } else {
            s1 = iq_dp4a(gl, u[2 * k], s1);
            s1 = iq_dp4a(gh, u[2 * k + 1], s1);
        }
    }
    const int ls0 = scales[ib] & 0x0F, ls1 = scales[ib] >> 4;
    return (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
}

__device__ __forceinline__ int dot_iq2_s(const uint8_t* b, int ib, const int* u) {
    const uint8_t* idx = b + 2 + 4 * ib;             // grid index low bytes
    const uint8_t* sgn = b + 2 + QK_K / 8 + 4 * ib;  // sign bytes (qs + 32)
    const uint8_t* qh = b + 66;
    const uint8_t* scales = b + 74;
    const int qhv = qh[ib];
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int l0 = 2 * k;
        const int gi = idx[k] | ((qhv << (8 - l0)) & 0x300);
        const int* gp = (const int*)(iq2s_grid + gi);
        const uint8_t sb = sgn[k];
        const int sg0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        const int sg1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        const int gl = iq_apply_signs(gp[0], sg0);
        const int gh = iq_apply_signs(gp[1], sg1);
        if (k < 2) {
            s0 = iq_dp4a(gl, u[2 * k], s0);
            s0 = iq_dp4a(gh, u[2 * k + 1], s0);
        } else {
            s1 = iq_dp4a(gl, u[2 * k], s1);
            s1 = iq_dp4a(gh, u[2 * k + 1], s1);
        }
    }
    const int ls0 = scales[ib] & 0x0F, ls1 = scales[ib] >> 4;
    return (s0 * ls0 + s1 * ls1 + (s0 + s1) / 2) / 4;
}

__device__ __forceinline__ int dot_iq3_xxs(const uint8_t* b, int ib, const int* u) {
    const uint8_t* qs = b + 2 + 8 * ib;
    uint32_t aux32;
    memcpy(&aux32, b + 2 + QK_K / 4 + 4 * ib, sizeof(uint32_t));
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t g0 = iq3xxs_grid[qs[2 * k]];
        const uint32_t g1 = iq3xxs_grid[qs[2 * k + 1]];
        const uint32_t signs = iq_unpack_ksigns(aux32 >> (7 * k));
        const int gl = iq_apply_signs(g0, __vcmpne4(signs & 0x08040201, 0));
        const int gh = iq_apply_signs(g1, __vcmpne4(signs & 0x80402010, 0));
        sumi = iq_dp4a(gl, u[2 * k], sumi);
        sumi = iq_dp4a(gh, u[2 * k + 1], sumi);
    }
    const int ls = aux32 >> 28;
    return (ls * sumi + sumi / 2) / 2;
}

__device__ __forceinline__ int dot_iq3_s(const uint8_t* b, int ib, const int* u) {
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t* qh = b + 66;
    const uint8_t* sgn = b + 74 + 4 * ib;
    const uint8_t* scales = b + 106;
    const int qhv = qh[ib];
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int l0 = 2 * k;
        const uint32_t g0 = iq3s_grid[qs[2 * k] | ((qhv << (8 - l0)) & 0x100)];
        const uint32_t g1 = iq3s_grid[qs[2 * k + 1] | ((qhv << (7 - l0)) & 0x100)];
        const uint8_t sb = sgn[k];
        const int sg0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        const int sg1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        const int gl = iq_apply_signs(g0, sg0);
        const int gh = iq_apply_signs(g1, sg1);
        sumi = iq_dp4a(gl, u[2 * k], sumi);
        sumi = iq_dp4a(gh, u[2 * k + 1], sumi);
    }
    const int nib = (scales[ib / 2] >> (4 * (ib & 1))) & 0x0F;
    return sumi * (1 + 2 * nib);
}

__device__ __forceinline__ int dot_iq4_xs(const uint8_t* b, int ib, const int* u) {
    uint16_t scales_h;
    memcpy(&scales_h, b + 2, sizeof(uint16_t));
    const uint8_t* scales_l = b + 4;
    const uint8_t* qs = b + 8 + 16 * ib;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int aux_q4;
        memcpy(&aux_q4, qs + 4 * j, sizeof(int));
        const int2 v = iq_table16(aux_q4, kvalues_iq4nl);
        sumi = iq_dp4a(v.x, u[j], sumi);
        sumi = iq_dp4a(v.y, u[j + 4], sumi);
    }
    const int iqs = 4 * ib;
    const int ls = ((scales_l[ib / 2] >> (iqs & 0x04)) & 0x0F) | (((scales_h >> (iqs / 2)) & 0x03) << 4);
    return sumi * (ls - 32);
}

#define MMVQ_KERNEL(NAME, TYPE_SIZE, DOT_FN)                                              \
    extern "C" __global__ void NAME(const uint8_t* __restrict__ W,                        \
                                    const uint8_t* __restrict__ Aq,                       \
                                    const float* __restrict__ Ad, float* __restrict__ Y,  \
                                    int n_out, int n_in, int M) {                          \
        const int lane = threadIdx.x & 31;                                                \
        const int warp = threadIdx.x >> 5;                                                \
        const int o = blockIdx.x * IQ_GEMV_ROWS_PER_BLOCK + warp;                         \
        if (o >= n_out) return;                                                           \
        const int nb256 = n_in / QK_K;                                                    \
        const int nblk32 = n_in / 32;                                                     \
        float acc[IQ_GEMV_MAX_M];                                                         \
        for (int m = 0; m < M; ++m) acc[m] = 0.0f;                                        \
        for (int g = lane; g < nblk32; g += 32) {                                         \
            const int sb = g >> 3;                                                        \
            const int ib = g & 7;                                                         \
            const uint8_t* wb = W + (long)(o * nb256 + sb) * (TYPE_SIZE);                 \
            const float w_d = __half2float(*reinterpret_cast<const __half*>(wb));         \
            for (int m = 0; m < M; ++m) {                                                 \
                const int* u = reinterpret_cast<const int*>(Aq + (long)m * n_in + (long)g * 32); \
                const int sub = DOT_FN(wb, ib, u);                                        \
                acc[m] += w_d * Ad[(long)m * nblk32 + g] * (float)sub;                    \
            }                                                                             \
        }                                                                                 \
        for (int m = 0; m < M; ++m) {                                                     \
            float v = acc[m];                                                             \
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off); \
            if (lane == 0) Y[(long)m * n_out + o] = v;                                    \
        }                                                                                 \
    }

MMVQ_KERNEL(mmvq_iq2_xs_f32, 74, dot_iq2_xs)
MMVQ_KERNEL(mmvq_iq2_s_f32, 82, dot_iq2_s)
MMVQ_KERNEL(mmvq_iq3_xxs_f32, 98, dot_iq3_xxs)
MMVQ_KERNEL(mmvq_iq3_s_f32, 110, dot_iq3_s)
MMVQ_KERNEL(mmvq_iq4_xs_f32, 136, dot_iq4_xs)
