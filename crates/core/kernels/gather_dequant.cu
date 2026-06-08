// Per-row gather + dequantize for quantized embedding tables.
//
// Used by `QuantizedEmbedding` (GGUF path) so a huge quantized embedding
// table — e.g. Gemma 4's Per-Layer Embedding, [262144, 42*256] Q6_K,
// ~5.6 GB dense — can stay quantized-resident on the GPU while a forward
// materializes only the rows the batch's token ids select. The whole
// table is never dequantized.
//
// Layout: the table is the raw GGUF block bytes, row-major over
// `[num_embeddings, embedding_dim]`. Row r is a contiguous run of
// `blocks_per_row = embedding_dim / QK_K` quant blocks (Gemma PLE row =
// 10752 / 256 = 42 Q6_K blocks — exact block alignment).
//
// The Q6_K dequant math mirrors candle's CPU `BlockQ6K::to_float`
// (candle-core quantized/k_quants.rs) exactly, so the GPU gather is
// bit-faithful to the CPU reference.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define QK_K 256
// Q6_K block: ql[QK_K/2=128] + qh[QK_K/4=64] + scales[QK_K/16=16](int8) + d(f16)
#define Q6K_QL 128
#define Q6K_QH 64
#define Q6K_SCALES 16
#define Q6K_BLOCK_BYTES (Q6K_QL + Q6K_QH + Q6K_SCALES + 2) // 210

// One CUDA block dequantizes one Q6_K block (256 outputs) of one gathered
// row. grid.x = num_ids * blocks_per_row; blockDim.x = 64.
extern "C" __global__ void gather_dequant_q6k_f32(
    const uint8_t* __restrict__ blocks, // quantized table bytes
    const int32_t* __restrict__ ids,    // [num_ids] row indices
    float* __restrict__ out,            // [num_ids * embedding_dim]
    int num_ids,
    int blocks_per_row,
    int num_embeddings) {
    int global_block = blockIdx.x;
    int total_blocks = num_ids * blocks_per_row;
    if (global_block >= total_blocks) {
        return;
    }
    int out_row = global_block / blocks_per_row;
    int block_in_row = global_block % blocks_per_row;
    int id = ids[out_row];
    if (id < 0 || id >= num_embeddings) {
        return; // out-of-range id → leave zeros
    }

    long src_block = (long)id * blocks_per_row + block_in_row;
    const uint8_t* b = blocks + src_block * (long)Q6K_BLOCK_BYTES;
    const uint8_t* ql = b;
    const uint8_t* qh = b + Q6K_QL;
    const int8_t* sc = (const int8_t*)(b + Q6K_QL + Q6K_QH);
    half dh = *reinterpret_cast<const half*>(b + Q6K_QL + Q6K_QH + Q6K_SCALES);
    float d = __half2float(dh);

    int out_base = (out_row * blocks_per_row + block_in_row) * QK_K;

    // 64 threads: thread t handles the 128-chunk idx=t/32 and inner l=t%32,
    // writing 4 outputs (l, l+32, l+64, l+96) — mirrors the CPU loop.
    int t = threadIdx.x;
    if (t >= 64) {
        return;
    }
    int idx = t / 32;
    int l = t % 32;
    const int8_t* scn = sc + 8 * idx;
    const uint8_t* qln = ql + 64 * idx;
    const uint8_t* qhn = qh + 32 * idx;
    int is = l / 16;

    int q1 = (int)((qln[l] & 0xF) | ((qhn[l] & 3) << 4)) - 32;
    int q2 = (int)((qln[l + 32] & 0xF) | (((qhn[l] >> 2) & 3) << 4)) - 32;
    int q3 = (int)((qln[l] >> 4) | (((qhn[l] >> 4) & 3) << 4)) - 32;
    int q4 = (int)((qln[l + 32] >> 4) | (((qhn[l] >> 6) & 3) << 4)) - 32;

    float* yo = out + out_base + 128 * idx;
    yo[l] = d * (float)scn[is] * (float)q1;
    yo[l + 32] = d * (float)scn[is + 2] * (float)q2;
    yo[l + 64] = d * (float)scn[is + 4] * (float)q3;
    yo[l + 96] = d * (float)scn[is + 6] * (float)q4;
}
