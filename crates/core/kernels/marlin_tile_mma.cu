// Stage 15.C/D — Marlin tile-MMA kernel.
//
// Shape constraints (v1 software scaffold):
//   M           any ≥ 1
//   N           any multiple of 64    (tile_n_size; producer requirement)
//   K           any multiple of 16    (tile_k_size; producer requirement)
//   group_size  must divide K and be a multiple of 16 (so each group spans
//               a whole number of k-tiles). num_groups = K / group_size.
//
// BF16 activations, INT4 AWQ weights laid out per
// `awq_to_marlin_tile_repack_cpu` (Stage 15.B), per-group AWQ scales +
// qzeros. Dequantisation: `(nibble - zp_{g,n}) * scale_{g,n}` where
// `g = k_index / group_size` and the AWQ packing of zp matches qweight
// (8 zp per u32, undo_pack [0,4,1,5,2,6,3,7]).
//
// **Scaffold-first.** Currently a software dequant + dot-product path,
// not yet tensor cores. Validates the producer/consumer interface end-
// to-end while the tensor-core mma.m16n8k16 inline-PTX path is being
// written. See `docs/perf/marlin-tile-mma-step-15c-design.md` §3-§7.
//
// Inputs:
//   a_ptr   [M, K]                BF16 row-major activations.
//   b_ptr                         u32, length k_tiles × n_tiles × 128
//                                 (full Marlin tile output of
//                                 `awq_to_marlin_tile_repack_cpu(K, N)`).
//                                 Index for (k_tile, n_tile, warp_id, th_id):
//                                   b_ptr[k_tile * (n_tiles * 128)
//                                         + n_tile * 128
//                                         + th_id * 4 + warp_id]
//   s_ptr   [num_groups, N]       BF16 per-group scale.
//   z_ptr   [num_groups, N/8]     u32 — GPTQ-sequential qzeros (8 zp per
//                                 u32, packed in plain little-endian
//                                 order: zp_n at bit (n%8)*4 of word
//                                 z_ptr[g*(N/8) + n/8]). This matches
//                                 the layout MarlinLinear stores after
//                                 `repack_awq_nibbles` runs at load
//                                 time, so the hybrid path can reuse
//                                 the already-loaded qzeros tensor with
//                                 no extra repack.
//   m, n, k, group_size           int — see constraints above.
//
// Output:
//   c_ptr [M, N]      BF16 row-major output.
//
// Launch:  block_dim = (256, 1, 1); grid_dim = (ceildiv(M*N, 256), 1, 1).
//          Threads with linear index ≥ M*N early-return.
//
// Numeric contract: same as `dequant(b) * s` followed by
// `a [16, K] @ that [K, 16]` (a row-vector dot product per output cell,
// reduction over the full K).
//
// Reference algorithm for nibble decode (extracted from
// `reference/vllm/csrc/quantization/marlin/awq_marlin_repack.cu:75-150`):
//
//   For (k_in_tile ∈ 0..16, n_in_tile ∈ 0..16):
//     m_n        = n_in_tile % 16            // 0..15
//     warp_id    = 0                          // v1 single warp
//     val_high   = m_n >= 8                   // vals[4..7] vs vals[0..3]
//     tc_col     = m_n - (val_high ? 8 : 0)   // 0..7
//     tc_row     = ((k_in_tile / 2) % 4) * 2  // 0,2,4,6
//     i_inner    = (k_in_tile % 2) + (k_in_tile / 8) * 2  // 0..3
//     th_id      = tc_row / 2 + tc_col * 4    // 0..31
//     val_idx    = i_inner + (val_high ? 4 : 0)  // 0..7
//     pack_idx⁻¹ = [0,4,1,5,2,6,3,7]
//     bit_off    = pack_idx⁻¹[val_idx] * 4
//     nibble     = (b_ptr[k_tile * 32 + th_id] >> bit_off) & 0xF

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define V1_TILE_N 64
#define V1_TILE_K 16
#define V1_TILE_INTS 128  // tile_k * tile_n / 8 (INT4 pack)

// GPTQ-sequential nibble shift: zero-point at logical column n lives at bit
// (n & 7) * 4 of the corresponding qzeros u32. Matches the layout
// `repack_awq_nibbles` produces at AwqMarlinLinear load time.
__device__ __forceinline__ uint32_t gptq_zp_shift(int n) {
    return (uint32_t)((n & 7) * 4);
}

extern "C" __global__ void marlin_tile_mma_int4_bf16(
    const __nv_bfloat16* __restrict__ a_ptr,
    const uint32_t*      __restrict__ b_ptr,
    const __nv_bfloat16* __restrict__ s_ptr,
    const uint32_t*      __restrict__ z_ptr,
    __nv_bfloat16*       __restrict__ c_ptr,
    int m,
    int n,
    int k,
    int group_size) {
    const int inv_pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    int row = idx / n;  // 0..m-1
    int col = idx % n;  // 0..n-1

    int n_tile = col / V1_TILE_N;
    int n_in_tile = col % V1_TILE_N;     // 0..63
    int warp_id = n_in_tile / 16;        // 0..3
    int m_n = n_in_tile % 16;            // 0..15
    int val_high = (m_n >= 8) ? 1 : 0;
    int tc_col = m_n - (val_high ? 8 : 0);

    int k_tiles = k / V1_TILE_K;
    int n_tiles = n / V1_TILE_N;
    int row_stride_u32 = n_tiles * V1_TILE_INTS;
    int qz_stride_u32 = n / 8;
    uint32_t zp_shift = gptq_zp_shift(col);
    int zp_word_col = col / 8;

    float acc = 0.0f;

    for (int kt = 0; kt < k_tiles; kt++) {
        int b_tile_base = kt * row_stride_u32 + n_tile * V1_TILE_INTS;
        int k_base = kt * V1_TILE_K;
        // The whole k-tile (16 K-rows) lies within a single group because
        // group_size is a multiple of 16. So we look up scale and zp once
        // per k-tile for this column.
        int g = k_base / group_size;
        float scale = __bfloat162float(s_ptr[g * n + col]);
        uint32_t zp_packed = z_ptr[g * qz_stride_u32 + zp_word_col];
        int zp = (int)((zp_packed >> zp_shift) & 0xF);

        for (int k_in_tile = 0; k_in_tile < V1_TILE_K; k_in_tile++) {
            int tc_row = ((k_in_tile / 2) % 4) * 2;
            int i_inner = (k_in_tile % 2) + (k_in_tile / 8) * 2;
            int th_id = tc_row / 2 + tc_col * 4;
            int val_idx = i_inner + (val_high ? 4 : 0);
            int u32_idx = b_tile_base + th_id * 4 + warp_id;
            int bit_off = inv_pack_idx[val_idx] * 4;

            uint32_t packed = b_ptr[u32_idx];
            int nib = (int)((packed >> bit_off) & 0xF);

            float w_f = (float)(nib - zp) * scale;
            float a_f = __bfloat162float(a_ptr[row * k + k_base + k_in_tile]);
            acc += a_f * w_f;
        }
    }

    c_ptr[row * n + col] = __float2bfloat16_rn(acc);
}

// TODO(stage 15.D-body): replace the per-thread software dot product
// above with a tensor-core mma.m16n8k16 path:
//   1. cooperative load of A into shared memory + ldmatrix.x4 → FragA.
//   2. INT4 dequant via LOP3(MASK=0x000f000f, EX=0x43004300) + bf16 SUB
//      (per `dequant.h:174–215`).
//   3. Two mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 calls
//      (n=0..7 and n=8..15) with shared FragA. Inner K-tile loop reuses
//      the same FragA / FragC across k-tiles, accumulating in FP32.
//   4. Output store: FragC FP32 → BF16 with the m16n8 thread-mapping
//      from `marlin-tile-mma-step-15c-design.md` §3.
// Block dims become (32, 1, 1) (1 warp); this scaffold uses (256, 1, 1).

// ─── Stage 15.D-body.1 — INT4 → BF16 LOP3 dequant primitive ─────────────
//
// Two LOP3 calls + an `__hsub2` recover the raw nibble values from a
// packed INT4 u32, mirroring the reference at
// `reference/vllm/csrc/quantization/marlin/dequant.h:174-215`. One call
// produces 4 bf16 values from 4 of the 8 nibbles in the input u32; for
// our nibble pack order (`pack_idx = [0,2,4,6,1,3,5,7]`) the chosen 4 are
// at packed-bit positions 0/4/16/20 of `q`, which decode (in our logical
// ordering) to vals 0, 2, 1, 3 — i.e. the FIRST FOUR vals of the 8-nibble
// pack. A second call on `q >> 8` extracts vals 4, 5, 6, 7.
//
// Output layout: the 4 bf16 values populate `frag_b` (a 2-element
// `__nv_bfloat162` array) so the upper-half of `frag_b[0]` and the
// upper-half of `frag_b[1]` are produced by the OFFSET LOP3 (after the
// `q >>= 4` shift in the reference's tight clang-format-off block).
//
// Standalone correctness test: see
// `marlin_test_dequant_int4_bf16_lo4` below.

template <int LOP3_IMM = 0xea>
__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t out;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(out)
                 : "r"(a), "r"(b), "r"(c), "n"(LOP3_IMM));
    return out;
}

__device__ __forceinline__ void dequant_int4_bf16_lo4(uint32_t q,
                                                     __nv_bfloat162* frag_b) {
    // (a & MASK) | EX → emits bf16(128 + nibble) per byte half.
    constexpr uint32_t MASK = 0x000f000f;
    constexpr uint32_t EX = 0x43004300;
    constexpr uint32_t SUB = 0x43004300;  // bf16(128.0) duplicated

    // First LOP3: nibbles at q-bits [3:0] and [19:16]
    // (i.e. our pack_idx positions 0 and 1, which are vals[0] and vals[1]).
    uint32_t lo = lop3<0xea>(q, MASK, EX);
    // Shift q by 4 and LOP3 again: now picks nibbles at original q-bits [7:4]
    // and [23:20] (pack_idx positions 2 and 3, vals[2] and vals[3]).
    uint32_t q_shift = q >> 4;
    uint32_t hi = lop3<0xea>(q_shift, MASK, EX);

    __nv_bfloat162 lo_bf16 = *reinterpret_cast<__nv_bfloat162*>(&lo);
    __nv_bfloat162 hi_bf16 = *reinterpret_cast<__nv_bfloat162*>(&hi);
    __nv_bfloat162 sub_bf16 = *reinterpret_cast<const __nv_bfloat162*>(&SUB);

    frag_b[0] = __hsub2(lo_bf16, sub_bf16);
    frag_b[1] = __hsub2(hi_bf16, sub_bf16);
}

// Test kernel — Stage 15.D-body.1 standalone correctness.
//
// Reads `count` u32s from `q_ptr`, dequants the lower 4 nibbles of each
// (in pack_idx order vals[0..3]) into 4 bf16 values written contiguously
// to `out_ptr`. CPU reference + this kernel must agree.
//
// Launch: blockDim = (256,1,1); grid_dim = (ceildiv(count,256),1,1).
extern "C" __global__ void marlin_test_dequant_int4_bf16_lo4(
    const uint32_t*    __restrict__ q_ptr,
    __nv_bfloat16*     __restrict__ out_ptr,
    int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    __nv_bfloat162 frag_b[2];
    dequant_int4_bf16_lo4(q_ptr[idx], frag_b);

    // The 4 outputs correspond to nibbles at q-bit positions:
    //   out[0] = bits [3:0]    (nibble #0)
    //   out[1] = bits [19:16]  (nibble #4)
    //   out[2] = bits [7:4]    (nibble #1)
    //   out[3] = bits [23:20]  (nibble #5)
    // The Marlin tile-frag permutation: nibbles 16 bits apart packed
    // into a single LOP3 to produce a bf16x2 fragment in one instruction.
    int out_base = idx * 4;
    out_ptr[out_base + 0] = __low2bfloat16(frag_b[0]);
    out_ptr[out_base + 1] = __high2bfloat16(frag_b[0]);
    out_ptr[out_base + 2] = __low2bfloat16(frag_b[1]);
    out_ptr[out_base + 3] = __high2bfloat16(frag_b[1]);
}

// ─── Stage 15.D-body.2a — minimal bf16 mma.m16n8k16 kernel (no INT4) ────
//
// Single-warp, single-tile (M=16, N=8, K=16) probe of the
// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction.
// Each thread builds its own fragments directly from global memory using
// the PTX-spec thread layout (no ldmatrix yet — that's body.2c).
//
// Layout (PTX 8.0 spec, mma.m16n8k16 with .f16/.bf16 operands;
// group_id = lane/4, group_thread = lane%4):
//   FragA (M=16, K=16, row-major, 4 u32/thread):
//     a[0] (lo,hi) = A(gid,    gth*2),   A(gid,    gth*2+1)
//     a[1] (lo,hi) = A(gid+8,  gth*2),   A(gid+8,  gth*2+1)
//     a[2] (lo,hi) = A(gid,    gth*2+8), A(gid,    gth*2+9)
//     a[3] (lo,hi) = A(gid+8,  gth*2+8), A(gid+8,  gth*2+9)
//   FragB (K=16, N=8, col-major, 2 u32/thread):
//     b[0] (lo,hi) = B(gth*2,   gid), B(gth*2+1, gid)
//     b[1] (lo,hi) = B(gth*2+8, gid), B(gth*2+9, gid)
//   FragC (M=16, N=8, row-major, 4 f32/thread):
//     c[0] = C(gid,    gth*2)
//     c[1] = C(gid,    gth*2+1)
//     c[2] = C(gid+8,  gth*2)
//     c[3] = C(gid+8,  gth*2+1)
//
// B is laid out col-major: B(k, n) = b_ptr[n * 16 + k]. CPU reference
// in the dispatcher must match this convention.
//
// Launch: block_dim = (32, 1, 1); grid_dim = (1, 1, 1). Single warp.

extern "C" __global__ void marlin_test_mma_m16n8k16_bf16(
    const __nv_bfloat16* __restrict__ a_ptr,  // [16, 16] row-major
    const __nv_bfloat16* __restrict__ b_ptr,  // [8, 16] col-major (b[n*16+k])
    float* __restrict__ c_ptr) {              // [16, 8] row-major
    int lane = threadIdx.x;
    if (lane >= 32) return;
    int gid = lane / 4;
    int gth = lane % 4;

    auto pack = [](__nv_bfloat16 lo, __nv_bfloat16 hi) {
        __nv_bfloat162 v = __halves2bfloat162(lo, hi);
        return *reinterpret_cast<uint32_t*>(&v);
    };

    uint32_t a[4];
    a[0] = pack(a_ptr[gid       * 16 + gth * 2    ], a_ptr[gid       * 16 + gth * 2 + 1]);
    a[1] = pack(a_ptr[(gid + 8) * 16 + gth * 2    ], a_ptr[(gid + 8) * 16 + gth * 2 + 1]);
    a[2] = pack(a_ptr[gid       * 16 + gth * 2 + 8], a_ptr[gid       * 16 + gth * 2 + 9]);
    a[3] = pack(a_ptr[(gid + 8) * 16 + gth * 2 + 8], a_ptr[(gid + 8) * 16 + gth * 2 + 9]);

    uint32_t b[2];
    b[0] = pack(b_ptr[gid * 16 + gth * 2    ], b_ptr[gid * 16 + gth * 2 + 1]);
    b[1] = pack(b_ptr[gid * 16 + gth * 2 + 8], b_ptr[gid * 16 + gth * 2 + 9]);

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    c_ptr[gid       * 8 + gth * 2    ] = c[0];
    c_ptr[gid       * 8 + gth * 2 + 1] = c[1];
    c_ptr[(gid + 8) * 8 + gth * 2    ] = c[2];
    c_ptr[(gid + 8) * 8 + gth * 2 + 1] = c[3];
}
