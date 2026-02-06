// PagedAttention v1 kernel for decode — bf16, head_dim=128, block_size=16.
// Simplified from vLLM's paged_attention_v1_kernel.
//
// Grid: (num_heads, num_seqs, 1)
// Block: (128, 1, 1) — one thread per head dimension
// Dynamic shared memory: sizeof(float) * max_seq_len + sizeof(float) * 128 + sizeof(float) * 4

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

#define HEAD_DIM 128
#define BLOCK_SIZE 16
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / 32)

// Warp-level reduce sum via shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum: 4 warps → 1 value.
// Returns result in all threads (broadcast via shared memory).
__device__ float block_reduce_sum(float val, float* reduce_smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Intra-warp reduce
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        reduce_smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float v = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            reduce_smem[0] = v;
        }
    }
    __syncthreads();

    return reduce_smem[0];
}

// Block-level reduce max, result broadcast to all threads.
__device__ float block_reduce_max(float val, float* reduce_smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Intra-warp reduce max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    if (lane_id == 0) {
        reduce_smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) {
            reduce_smem[0] = v;
        }
    }
    __syncthreads();

    return reduce_smem[0];
}

// Shared implementation for paged attention v1 with optional ALiBi support.
//
// Each thread block handles one (head, sequence) pair.
// Each thread handles one dimension of the head (tid in [0, 127]).
//
// When alibi_slopes is non-null, the ALiBi bias is added to the scaled QK
// logits: logit[token_pos] += alibi_slope * (token_pos - (seq_len - 1))
// This makes distant past tokens receive more negative bias, matching the
// ALiBi positional encoding used by Bloom and MPT models.
//
// Dynamic shared memory layout:
//   float q_smem[HEAD_DIM]           -- query vector (128 floats = 512 bytes)
//   float reduce_smem[NUM_WARPS]     -- reduction workspace (4 floats = 16 bytes)
//   float logits[max_context_len]    -- QK logits (variable)
__device__ void paged_attention_v1_impl(
    __nv_bfloat16* __restrict__ out,           // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ q,       // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const float* __restrict__ alibi_slopes     // [num_heads] or nullptr
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // GQA: map query head to KV head
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // ALiBi slope for this head (0 if not using ALiBi)
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;

    // Shared memory layout (fixed offsets to avoid overlap):
    //   [0, HEAD_DIM)                        -- q vector (128 floats)
    //   [HEAD_DIM, HEAD_DIM + NUM_WARPS)     -- reduction workspace (4 floats)
    //   [HEAD_DIM + NUM_WARPS, ...)           -- logits array (seq_len floats)
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce_smem = smem + HEAD_DIM;
    float* logits = smem + HEAD_DIM + NUM_WARPS;

    // Cache strides for [num_blocks, block_size, num_kv_heads, head_dim] layout
    const int cache_stride_block = BLOCK_SIZE * num_kv_heads * HEAD_DIM;
    const int cache_stride_token = num_kv_heads * HEAD_DIM;

    // Load query vector into shared memory
    const int q_offset = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    q_smem[tid] = __bfloat162float(q[q_offset]);
    __syncthreads();

    // ---- Phase 1: Compute QK dot products ----
    const int num_blocks_used = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float qk_max = -FLT_MAX;

    // The query position is the last position: seq_len - 1 (decode produces 1 token).
    // ALiBi bias for key at position j: alibi_slope * (j - (seq_len - 1))
    const int query_pos = seq_len - 1;

    for (int block_idx = 0; block_idx < num_blocks_used; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int tokens_in_block = min(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE);

        for (int token = 0; token < tokens_in_block; token++) {
            // Load K value for this thread's dimension
            const int k_offset = physical_block * cache_stride_block
                               + token * cache_stride_token
                               + kv_head_idx * HEAD_DIM
                               + tid;
            const float k_val = __bfloat162float(k_cache[k_offset]);

            // Partial dot product: q[tid] * k[tid]
            float qk = q_smem[tid] * k_val;

            // Block-level reduce to get full dot product
            qk = block_reduce_sum(qk, reduce_smem);

            // Thread 0 stores the scaled logit (with optional ALiBi bias)
            if (tid == 0) {
                const int token_pos = block_idx * BLOCK_SIZE + token;
                float scaled_qk = qk * scale;
                // ALiBi bias: slope * (key_position - query_position)
                // For causal decode, query_position = seq_len - 1 and
                // key_position = token_pos, so bias = slope * (token_pos - query_pos)
                // which is non-positive for past tokens.
                scaled_qk += alibi_slope * (float)(token_pos - query_pos);
                logits[token_pos] = scaled_qk;
                qk_max = fmaxf(qk_max, scaled_qk);
            }
            __syncthreads();
        }
    }

    // Broadcast qk_max from thread 0 to all threads
    if (tid == 0) {
        reduce_smem[0] = qk_max;
    }
    __syncthreads();
    qk_max = reduce_smem[0];

    // ---- Phase 2: Softmax ----
    float exp_sum = 0.0f;
    for (int i = tid; i < seq_len; i += NUM_THREADS) {
        const float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    __syncthreads();

    exp_sum = block_reduce_sum(exp_sum, reduce_smem);

    const float inv_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int i = tid; i < seq_len; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation ----
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < num_blocks_used; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int tokens_in_block = min(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE);

        for (int token = 0; token < tokens_in_block; token++) {
            const int v_offset = physical_block * cache_stride_block
                               + token * cache_stride_token
                               + kv_head_idx * HEAD_DIM
                               + tid;
            const float v_val = __bfloat162float(v_cache[v_offset]);
            const float weight = logits[block_idx * BLOCK_SIZE + token];
            acc += weight * v_val;
        }
    }

    // ---- Write output ----
    const int out_offset = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    out[out_offset] = __float2bfloat16(acc);
}

// Original entry point without ALiBi (backward compatible).
extern "C" __global__ void paged_attention_v1_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq
) {
    paged_attention_v1_impl(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        nullptr  // no ALiBi
    );
}

// Entry point with ALiBi positional bias support.
// alibi_slopes is a float array of length num_heads containing the per-head
// slope values. The bias applied is: slope[head] * (key_pos - query_pos).
extern "C" __global__ void paged_attention_v1_bf16_alibi(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const float* __restrict__ alibi_slopes     // [num_heads]
) {
    paged_attention_v1_impl(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        alibi_slopes
    );
}
