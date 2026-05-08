// Stage 15.C — Minimum-viable Marlin tile-MMA kernel.
//
// Hard-coded shapes for v1: M = N = K = 16, BF16 activations, INT4 AWQ
// weights laid out per `awq_to_marlin_tile_repack_cpu` (Stage 15.B).
// Per-channel scales (group_size = K).  Single CTA, single warp.
//
// **Scaffold-first.** This file currently implements the kernel as a
// software dequant + matmul that follows the same I/O contract as the
// future tensor-core MMA kernel. It exists so that the build pipeline
// (cu → ptx → cudarc → CustomOp dispatch → GPU test → CPU reference)
// is exercised and validated *before* tensor-core PTX inline-asm work
// is layered in.  The tensor-core path lands in a follow-up iteration
// against the design at `docs/perf/marlin-tile-mma-step-15c-design.md`.
//
// Inputs:
//   a_ptr [M=16, K=16] BF16 row-major
//   b_ptr [32]         u32 — Marlin-tile-laid-out INT4 weights for one
//                            warp's worth of N=16 (i.e. the warp_id=0
//                            quarter of a 128-u32 K=16/N=64 tile).
//                            Per Stage 15.B's repack_awq_nibbles +
//                            awq_to_marlin_tile_repack_cpu pipeline.
//   s_ptr [N=16]       BF16 — per-channel scale.
//
// Output:
//   c_ptr [M=16, N=16] BF16 row-major.
//
// Numeric contract: same as `dequant(b) * s` followed by `a @ that`
// (a row-vector dot product per output cell, K = 16 reduction).
//
// Reference algorithm for nibble decode (extracted from the
// `awq_marlin_repack_kernel` producer at
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
//     u32_idx    = th_id * 4 + warp_id        // 0..127 (only *4+0 used here)
//     pack_idx   = [0,2,4,6,1,3,5,7]
//     bit_off    = (pack_idx⁻¹[val_idx]) * 4  // = [0,4,1,5,2,6,3,7][val_idx]
//     nibble     = (b_ptr[u32_idx] >> bit_off) & 0xF
//
// (`pack_idx⁻¹ = [0,4,1,5,2,6,3,7]`, derived in the test
// `marlin_tile_decode_position` for Stage 15.B.)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define V1_M 16
#define V1_N 16
#define V1_K 16

extern "C" __global__ void marlin_tile_mma_int4_bf16_m16n16k16(
    const __nv_bfloat16* __restrict__ a_ptr,
    const uint32_t*      __restrict__ b_ptr,
    const __nv_bfloat16* __restrict__ s_ptr,
    __nv_bfloat16*       __restrict__ c_ptr) {
    // Inverse of pack_idx [0,2,4,6,1,3,5,7]: position of vals[v] in res.
    // Same as Stage 15.B's INV_PACK_IDX. Static constant memory access.
    const int inv_pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    int t = threadIdx.x;
    if (t >= V1_M * V1_N) return;  // 256 threads cover all output cells

    int row = t / V1_N;  // 0..15  (m)
    int col = t % V1_N;  // 0..15  (n)

    // Decode the producer's tile mapping for (row=k_in_tile, col=n_in_tile).
    // For v1 single warp, warp_id is always 0 → u32_idx_in_tile = th_id * 4.

    // We need ALL k ∈ 0..16 nibbles at column `col`. For each k_in_tile:
    //   compute (th_id, val_idx, bit_offset) → read b_ptr[th_id*4] → extract nibble.
    float acc = 0.0f;

    int m_n = col;  // 0..15
    int val_high = (m_n >= 8) ? 1 : 0;
    int tc_col = m_n - (val_high ? 8 : 0);  // 0..7

    for (int k = 0; k < V1_K; k++) {
        int tc_row = ((k / 2) % 4) * 2;
        int i_inner = (k % 2) + (k / 8) * 2;
        int th_id = tc_row / 2 + tc_col * 4;  // 0..31
        int val_idx = i_inner + (val_high ? 4 : 0);
        // v1 wrapper supplies only the warp_id=0 quarter (32 u32) compacted
        // contiguously, so the array index here is `th_id`, NOT
        // `th_id * 4 + warp_id` from the full 128-u32 tile.
        int u32_idx = th_id;
        int bit_off = inv_pack_idx[val_idx] * 4;

        uint32_t packed = b_ptr[u32_idx];
        int nib = (int)((packed >> bit_off) & 0xF);

        // Multiply by per-channel scale and by activation, accumulate.
        float w_f = (float)nib * __bfloat162float(s_ptr[col]);
        float a_f = __bfloat162float(a_ptr[row * V1_K + k]);
        acc += a_f * w_f;
    }

    c_ptr[row * V1_N + col] = __float2bfloat16_rn(acc);
}

// TODO(stage 15.C-body): replace the per-thread software dot product
// above with a tensor-core mma.m16n8k16 path:
//   1. cooperative load of A into shared memory + ldmatrix.x4 → FragA.
//   2. INT4 dequant via LOP3(MASK=0x000f000f, EX=0x43004300) + bf16 SUB
//      (per `dequant.h:174–215`).
//   3. Two mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 calls
//      (n=0..7 and n=8..15) with shared FragA.
//   4. Output store: FragC FP32 → BF16 with the m16n8 thread-mapping
//      from `marlin-tile-mma-step-15c-design.md` §3.
// Block dims become (32, 1, 1) (1 warp); this scaffold uses (256, 1, 1).
