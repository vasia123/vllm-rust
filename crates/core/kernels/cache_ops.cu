// KV cache CUDA kernels: reshape_and_cache, copy_blocks.
// Ported from vLLM's cache_kernels.cu.
//
// reshape_and_cache: Write K/V from model output into paged KV cache
// using slot_mapping to determine target position.
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Input layout: [num_tokens, num_kv_heads, head_dim]

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ==========================================================================
// reshape_and_cache — BF16
// ==========================================================================
//
// For each token, write its K and V vectors into the cache at the
// position indicated by slot_mapping[token_idx].
//
// slot_mapping provides a flat index into the cache:
//   block_idx = slot / block_size
//   offset_in_block = slot % block_size
//   cache address = cache[block_idx, offset_in_block, head_idx, dim]
//
// Grid: (num_tokens,)
// Block: (min(num_kv_heads * head_dim, 1024),)
extern "C" __global__ void reshape_and_cache_bf16(
    const __nv_bfloat16* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ value_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ slot_mapping,          // [num_tokens] — cache slot per token
    const int num_kv_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int slot = slot_mapping[token_idx];

    // Negative slot means padding token — skip
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int offset_in_block = slot % block_size;

    // Input stride: [num_tokens, num_kv_heads, head_dim]
    const int kv_stride = num_kv_heads * head_dim;
    const __nv_bfloat16* k_src = key + token_idx * kv_stride;
    const __nv_bfloat16* v_src = value + token_idx * kv_stride;

    // Cache stride: [num_blocks, block_size, num_kv_heads, head_dim]
    const int cache_block_stride = block_size * kv_stride;
    __nv_bfloat16* k_dst = key_cache + block_idx * cache_block_stride + offset_in_block * kv_stride;
    __nv_bfloat16* v_dst = value_cache + block_idx * cache_block_stride + offset_in_block * kv_stride;

    // Copy all (num_kv_heads * head_dim) elements
    for (int i = threadIdx.x; i < kv_stride; i += blockDim.x) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}

// ==========================================================================
// reshape_and_cache — FP16
// ==========================================================================
extern "C" __global__ void reshape_and_cache_fp16(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ slot_mapping,
    const int num_kv_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int offset_in_block = slot % block_size;
    const int kv_stride = num_kv_heads * head_dim;
    const int cache_block_stride = block_size * kv_stride;

    const __half* k_src = key + token_idx * kv_stride;
    const __half* v_src = value + token_idx * kv_stride;
    __half* k_dst = key_cache + block_idx * cache_block_stride + offset_in_block * kv_stride;
    __half* v_dst = value_cache + block_idx * cache_block_stride + offset_in_block * kv_stride;

    for (int i = threadIdx.x; i < kv_stride; i += blockDim.x) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}

// ==========================================================================
// reshape_and_cache HND — BF16
// ==========================================================================
//
// HND variant: cache layout is [num_blocks, num_kv_heads, block_size, head_dim].
// Each thread block writes one head of one token into the cache.
//
// Grid: (num_tokens, num_kv_heads)
// Block: (min(head_dim, 1024),)
extern "C" __global__ void reshape_and_cache_hnd_bf16(
    const __nv_bfloat16* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ key_cache,        // [num_blocks, num_kv_heads, block_size, head_dim]
    __nv_bfloat16* __restrict__ value_cache,      // [num_blocks, num_kv_heads, block_size, head_dim]
    const int* __restrict__ slot_mapping,          // [num_tokens]
    const int num_kv_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int offset_in_block = slot % block_size;

    // Input: [num_tokens, num_kv_heads, head_dim]
    const int src_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    // HND cache: [num_blocks, num_kv_heads, block_size, head_dim]
    const int head_stride = block_size * head_dim;
    const int block_stride = num_kv_heads * head_stride;
    const int dst_offset = block_idx * block_stride + head_idx * head_stride + offset_in_block * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[dst_offset + d] = key[src_offset + d];
        value_cache[dst_offset + d] = value[src_offset + d];
    }
}

// ==========================================================================
// reshape_and_cache HND — FP16
// ==========================================================================
extern "C" __global__ void reshape_and_cache_hnd_fp16(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ slot_mapping,
    const int num_kv_heads,
    const int head_dim,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int slot = slot_mapping[token_idx];

    if (slot < 0) return;

    const int block_idx = slot / block_size;
    const int offset_in_block = slot % block_size;

    const int src_offset = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    const int head_stride = block_size * head_dim;
    const int block_stride = num_kv_heads * head_stride;
    const int dst_offset = block_idx * block_stride + head_idx * head_stride + offset_in_block * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[dst_offset + d] = key[src_offset + d];
        value_cache[dst_offset + d] = value[src_offset + d];
    }
}

// ==========================================================================
// copy_blocks — BF16
// ==========================================================================
// Copy cache blocks between physical locations (used for COW / preemption).
//
// block_mapping: [num_pairs, 2] — (src_block, dst_block) pairs
// numel_per_block: block_size * num_kv_heads * head_dim
//
// Grid: (num_pairs,)
// Block: (min(numel_per_block, 1024),)
extern "C" __global__ void copy_blocks_bf16(
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int* __restrict__ block_mapping,    // [num_pairs * 2]: src0, dst0, src1, dst1, ...
    const int numel_per_block
) {
    const int pair_idx = blockIdx.x;
    const int src_block = block_mapping[2 * pair_idx];
    const int dst_block = block_mapping[2 * pair_idx + 1];

    const int src_offset = src_block * numel_per_block;
    const int dst_offset = dst_block * numel_per_block;

    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        key_cache[dst_offset + i] = key_cache[src_offset + i];
    }
    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        value_cache[dst_offset + i] = value_cache[src_offset + i];
    }
}
