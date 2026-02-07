// Fused QK-RMSNorm + RoPE CUDA kernel.
// Eliminates an intermediate global memory round-trip by applying RMSNorm
// and rotary position embedding in a single kernel launch.
//
// Used by DeepSeek V3, Qwen2, and other models that normalize Q/K before RoPE.
//
// Grid: (total_heads,) where total_heads = batch * seq_len * (num_q_heads + num_kv_heads)
// Block: (min(head_dim / VEC_SIZE, 1024),)
//
// Each thread block processes one head vector:
//   Phase 1: compute sum of squares for RMSNorm (vectorized loads + block reduce)
//   Phase 2: normalize with RMSNorm weight, then apply NeoX-style RoPE rotation

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Warp-level reduce sum via shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum using shared memory
__device__ float block_reduce_sum(float val) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);

    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            warp_sums[0] = v;
        }
    }
    __syncthreads();

    return warp_sums[0];
}

// Fused RMSNorm + NeoX RoPE for a single head vector.
// Templated on scalar type to support both BF16 and FP16.
//
// NeoX-style RoPE rotates the first half and second half of head_dim separately:
//   out[i]            = x_norm[i]            * cos[pos, i] - x_norm[i + half] * sin[pos, i]
//   out[i + half]     = x_norm[i + half]     * cos[pos, i] + x_norm[i]        * sin[pos, i]
// where i in [0, head_dim/2).
template <typename scalar_t>
__device__ void fused_qknorm_rope_head(
    scalar_t* __restrict__ output,         // output for this head: [head_dim]
    const scalar_t* __restrict__ input,    // input for this head: [head_dim]
    const scalar_t* __restrict__ weight,   // RMSNorm weight: [head_dim]
    const float* __restrict__ cos_cache,   // [max_pos, head_dim/2]
    const float* __restrict__ sin_cache,   // [max_pos, head_dim/2]
    const int pos,                         // position index for this token
    const int head_dim,                    // dimension of each head
    const int half_dim                     // head_dim / 2 (precomputed for RoPE)
) {
    const float epsilon = 1e-6f;

    // Phase 1: compute sum of squares for RMSNorm
    float variance = 0.0f;

    const int vec_size = 4;
    const int num_vecs = head_dim / vec_size;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        const int base = i * vec_size;
        float x0 = (float)input[base + 0];
        float x1 = (float)input[base + 1];
        float x2 = (float)input[base + 2];
        float x3 = (float)input[base + 3];
        variance += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }
    for (int i = num_vecs * vec_size + threadIdx.x; i < head_dim; i += blockDim.x) {
        float x = (float)input[i];
        variance += x * x;
    }

    variance = block_reduce_sum(variance);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)head_dim + epsilon);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // Precompute cos/sin row pointer for this position
    const float* cos_row = cos_cache + pos * half_dim;
    const float* sin_row = sin_cache + pos * half_dim;

    // Phase 2: RMSNorm + RoPE in one pass
    // NeoX-style: pair (i, i + half_dim) for rotation
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        // Load both elements of the rotation pair
        float x_lo = (float)input[i];
        float x_hi = (float)input[i + half_dim];
        float w_lo = (float)weight[i];
        float w_hi = (float)weight[i + half_dim];

        // RMSNorm
        float normed_lo = x_lo * inv_rms * w_lo;
        float normed_hi = x_hi * inv_rms * w_hi;

        // RoPE rotation
        float cos_val = cos_row[i];
        float sin_val = sin_row[i];
        output[i]            = (scalar_t)(normed_lo * cos_val - normed_hi * sin_val);
        output[i + half_dim] = (scalar_t)(normed_hi * cos_val + normed_lo * sin_val);
    }
}

// ==========================================================================
// Fused QKNorm + RoPE — BF16
// ==========================================================================
//
// Processes all Q and K heads across (batch, seq_len). Each thread block
// handles one head. Block indices [0, batch*seq_len*num_q_heads) cover Q;
// the remainder covers K.
//
// Layout assumptions:
//   Q: [batch, num_q_heads, seq_len, head_dim]
//   K: [batch, num_kv_heads, seq_len, head_dim]
//   positions: [batch, seq_len]
extern "C" __global__ void fused_qknorm_rope_bf16(
    __nv_bfloat16* __restrict__ out_q,       // [batch, num_q_heads, seq_len, head_dim]
    __nv_bfloat16* __restrict__ out_k,       // [batch, num_kv_heads, seq_len, head_dim]
    const __nv_bfloat16* __restrict__ in_q,  // [batch, num_q_heads, seq_len, head_dim]
    const __nv_bfloat16* __restrict__ in_k,  // [batch, num_kv_heads, seq_len, head_dim]
    const __nv_bfloat16* __restrict__ q_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_weight,  // [head_dim]
    const float* __restrict__ cos_cache,     // [max_pos, head_dim/2]
    const float* __restrict__ sin_cache,     // [max_pos, head_dim/2]
    const int* __restrict__ positions,       // [batch, seq_len]
    const int batch,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int half_dim = head_dim / 2;
    const int total_q_heads = batch * num_q_heads * seq_len;
    const int block_id = blockIdx.x;

    if (block_id < total_q_heads) {
        // Q head: decompose block_id into (batch_idx, head_idx, seq_idx)
        const int seq_idx = block_id % seq_len;
        const int head_idx = (block_id / seq_len) % num_q_heads;
        const int batch_idx = block_id / (num_q_heads * seq_len);

        const int pos = positions[batch_idx * seq_len + seq_idx];
        const int offset = ((batch_idx * num_q_heads + head_idx) * seq_len + seq_idx) * head_dim;

        fused_qknorm_rope_head<__nv_bfloat16>(
            out_q + offset,
            in_q + offset,
            q_weight,
            cos_cache, sin_cache,
            pos, head_dim, half_dim);
    } else {
        // K head: block_id relative to K start
        const int k_id = block_id - total_q_heads;
        const int seq_idx = k_id % seq_len;
        const int head_idx = (k_id / seq_len) % num_kv_heads;
        const int batch_idx = k_id / (num_kv_heads * seq_len);

        const int pos = positions[batch_idx * seq_len + seq_idx];
        const int offset = ((batch_idx * num_kv_heads + head_idx) * seq_len + seq_idx) * head_dim;

        fused_qknorm_rope_head<__nv_bfloat16>(
            out_k + offset,
            in_k + offset,
            k_weight,
            cos_cache, sin_cache,
            pos, head_dim, half_dim);
    }
}

// ==========================================================================
// Fused QKNorm + RoPE — FP16
// ==========================================================================
extern "C" __global__ void fused_qknorm_rope_f16(
    __half* __restrict__ out_q,
    __half* __restrict__ out_k,
    const __half* __restrict__ in_q,
    const __half* __restrict__ in_k,
    const __half* __restrict__ q_weight,
    const __half* __restrict__ k_weight,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ positions,
    const int batch,
    const int num_q_heads,
    const int num_kv_heads,
    const int seq_len,
    const int head_dim
) {
    const int half_dim = head_dim / 2;
    const int total_q_heads = batch * num_q_heads * seq_len;
    const int block_id = blockIdx.x;

    if (block_id < total_q_heads) {
        const int seq_idx = block_id % seq_len;
        const int head_idx = (block_id / seq_len) % num_q_heads;
        const int batch_idx = block_id / (num_q_heads * seq_len);

        const int pos = positions[batch_idx * seq_len + seq_idx];
        const int offset = ((batch_idx * num_q_heads + head_idx) * seq_len + seq_idx) * head_dim;

        fused_qknorm_rope_head<__half>(
            out_q + offset,
            in_q + offset,
            q_weight,
            cos_cache, sin_cache,
            pos, head_dim, half_dim);
    } else {
        const int k_id = block_id - total_q_heads;
        const int seq_idx = k_id % seq_len;
        const int head_idx = (k_id / seq_len) % num_kv_heads;
        const int batch_idx = k_id / (num_kv_heads * seq_len);

        const int pos = positions[batch_idx * seq_len + seq_idx];
        const int offset = ((batch_idx * num_kv_heads + head_idx) * seq_len + seq_idx) * head_dim;

        fused_qknorm_rope_head<__half>(
            out_k + offset,
            in_k + offset,
            k_weight,
            cos_cache, sin_cache,
            pos, head_dim, half_dim);
    }
}
