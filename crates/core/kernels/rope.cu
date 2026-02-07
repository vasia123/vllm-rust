// Fused Rotary Position Embedding (RoPE) CUDA kernel.
// Ported from vLLM's pos_encoding_kernels.cu.
//
// Applies rotary embedding to Q and K tensors in-place using a precomputed
// cos/sin cache. Supports both NeoX style (split halves) and GPT-J style
// (interleaved pairs).
//
// Grid: (num_tokens,)
// Block: (min(num_heads * rot_dim / 2, 512),)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Helper: apply rotary embedding to a single element pair.
// NeoX style: x = arr[offset], y = arr[embed_dim + offset]
// GPT-J style: x = arr[2*offset], y = arr[2*offset + 1]
template <typename scalar_t, bool IS_NEOX>
__device__ __forceinline__ void apply_token_rotary(
    scalar_t* __restrict__ arr,
    const float* __restrict__ cos_ptr,
    const float* __restrict__ sin_ptr,
    int rot_offset,
    int embed_dim
) {
    int x_index, y_index;
    float cos_val, sin_val;

    if constexpr (IS_NEOX) {
        x_index = rot_offset;
        y_index = embed_dim + rot_offset;
        cos_val = cos_ptr[x_index];
        sin_val = sin_ptr[x_index];
    } else {
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
        cos_val = cos_ptr[x_index / 2];
        sin_val = sin_ptr[x_index / 2];
    }

    float x = (float)arr[x_index];
    float y = (float)arr[y_index];
    arr[x_index] = (scalar_t)(x * cos_val - y * sin_val);
    arr[y_index] = (scalar_t)(y * cos_val + x * sin_val);
}

// Apply rotary embedding to all heads of Q and K for one token.
template <typename scalar_t, bool IS_NEOX>
__device__ void apply_rotary_embedding(
    scalar_t* __restrict__ query,   // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,     // [num_tokens, num_kv_heads, head_size] or nullptr
    const float* cache_ptr,         // cos_sin for this position: [rot_dim]
    const int num_heads,
    const int num_kv_heads,
    const int rot_dim,
    const int token_idx,
    const int query_stride,         // stride in elements between tokens for Q
    const int key_stride,           // stride in elements between tokens for K
    const int head_size
) {
    const int embed_dim = rot_dim / 2;
    const float* cos_ptr = cache_ptr;
    const float* sin_ptr = cache_ptr + embed_dim;

    // Apply to Q heads
    const int nq = num_heads * embed_dim;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) {
        const int head_idx = i / embed_dim;
        const int token_head_offset = token_idx * query_stride + head_idx * head_size;
        const int rot_offset = i % embed_dim;
        apply_token_rotary<scalar_t, IS_NEOX>(
            query + token_head_offset, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }

    // Apply to K heads
    if (key != nullptr) {
        const int nk = num_kv_heads * embed_dim;
        for (int i = threadIdx.x; i < nk; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int token_head_offset = token_idx * key_stride + head_idx * head_size;
            const int rot_offset = i % embed_dim;
            apply_token_rotary<scalar_t, IS_NEOX>(
                key + token_head_offset, cos_ptr, sin_ptr, rot_offset, embed_dim);
        }
    }
}

// ==========================================================================
// NeoX-style RoPE — BF16 (used by Llama, Qwen, Mistral, etc.)
// ==========================================================================
// Q/K are modified IN-PLACE.
// positions: [num_tokens] i32 — position index for each token
// cos_sin_cache: [max_position, rot_dim] f32 — precomputed cos/sin
extern "C" __global__ void rotary_embedding_neox_bf16(
    const int* __restrict__ positions,            // [num_tokens]
    __nv_bfloat16* __restrict__ query,            // [num_tokens, num_heads * head_size]
    __nv_bfloat16* __restrict__ key,              // [num_tokens, num_kv_heads * head_size] or nullptr
    const float* __restrict__ cos_sin_cache,      // [max_position, rot_dim]
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int head_size,
    const int num_heads,
    const int num_kv_heads
) {
    const int token_idx = blockIdx.x;
    const int pos = positions[token_idx];
    const float* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<__nv_bfloat16, true>(
        query, key, cache_ptr,
        num_heads, num_kv_heads, rot_dim,
        token_idx, query_stride, key_stride, head_size);
}

// ==========================================================================
// GPT-J-style RoPE — BF16 (used by GPT-J, CodeGen)
// ==========================================================================
extern "C" __global__ void rotary_embedding_gptj_bf16(
    const int* __restrict__ positions,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    const float* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int head_size,
    const int num_heads,
    const int num_kv_heads
) {
    const int token_idx = blockIdx.x;
    const int pos = positions[token_idx];
    const float* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<__nv_bfloat16, false>(
        query, key, cache_ptr,
        num_heads, num_kv_heads, rot_dim,
        token_idx, query_stride, key_stride, head_size);
}

// ==========================================================================
// NeoX-style RoPE — FP16
// ==========================================================================
extern "C" __global__ void rotary_embedding_neox_fp16(
    const int* __restrict__ positions,
    __half* __restrict__ query,
    __half* __restrict__ key,
    const float* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int head_size,
    const int num_heads,
    const int num_kv_heads
) {
    const int token_idx = blockIdx.x;
    const int pos = positions[token_idx];
    const float* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<__half, true>(
        query, key, cache_ptr,
        num_heads, num_kv_heads, rot_dim,
        token_idx, query_stride, key_stride, head_size);
}

// ==========================================================================
// NeoX-style RoPE — F32
// ==========================================================================
extern "C" __global__ void rotary_embedding_neox_f32(
    const int* __restrict__ positions,
    float* __restrict__ query,
    float* __restrict__ key,
    const float* __restrict__ cos_sin_cache,
    const int rot_dim,
    const int query_stride,
    const int key_stride,
    const int head_size,
    const int num_heads,
    const int num_kv_heads
) {
    const int token_idx = blockIdx.x;
    const int pos = positions[token_idx];
    const float* cache_ptr = cos_sin_cache + pos * rot_dim;

    apply_rotary_embedding<float, true>(
        query, key, cache_ptr,
        num_heads, num_kv_heads, rot_dim,
        token_idx, query_stride, key_stride, head_size);
}
