// In-place "apply grammar bitmask" kernel for structured-output sampling.
//
// Sets logits[b, t] = -INFINITY when bit `t` of row `b` of the packed
// int32 bitmask is zero. Logits with bit=1 are left untouched.
//
// Bitmask layout: row-major [batch, words_per_row], int32. Bit
// (token_id % 32) of word (token_id / 32) is the "allowed" flag.
// words_per_row = ceil(vocab_size / 32). The kernel reads bits in the
// range [0, vocab_size); any padding in the last word is unused.
//
// Grid : (ceil(vocab_size/256), batch_size, 1)
// Block: (256, 1, 1)
//
// Three dtypes are emitted (f32, bf16, f16). The kernel logic is the
// same; only the write type differs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math_constants.h>

template <typename T>
__device__ __forceinline__ T neg_inf_value();

template <>
__device__ __forceinline__ float neg_inf_value<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ __nv_bfloat16 neg_inf_value<__nv_bfloat16>() {
    return __float2bfloat16(-CUDART_INF_F);
}

template <>
__device__ __forceinline__ __half neg_inf_value<__half>() {
    return __float2half(-CUDART_INF_F);
}

template <typename T>
__device__ __forceinline__ void apply_bitmask_inner(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int vocab_size,
    const int words_per_row
) {
    const int batch = blockIdx.y;
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= vocab_size) {
        return;
    }
    const int word = bitmask[batch * words_per_row + (token >> 5)];
    if (((word >> (token & 31)) & 1) == 0) {
        logits[batch * vocab_size + token] = neg_inf_value<T>();
    }
}

extern "C" __global__ void apply_grammar_bitmask_f32(
    float* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int vocab_size,
    const int words_per_row
) {
    apply_bitmask_inner<float>(logits, bitmask, vocab_size, words_per_row);
}

extern "C" __global__ void apply_grammar_bitmask_bf16(
    __nv_bfloat16* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int vocab_size,
    const int words_per_row
) {
    apply_bitmask_inner<__nv_bfloat16>(logits, bitmask, vocab_size, words_per_row);
}

extern "C" __global__ void apply_grammar_bitmask_f16(
    __half* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int vocab_size,
    const int words_per_row
) {
    apply_bitmask_inner<__half>(logits, bitmask, vocab_size, words_per_row);
}
