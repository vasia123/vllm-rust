// PagedAttention v1 kernel for decode — bf16, configurable head_dim and block_size.
// Simplified from vLLM's paged_attention_v1_kernel.
//
// Grid: (num_heads, num_seqs, 1)
// Block: (NUM_THREADS, 1, 1) — threads cooperate across the head dimension
// Dynamic shared memory: sizeof(float) * (head_dim + NUM_WARPS + max_seq_len)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

// NUM_THREADS must be >= head_dim for correctness when head_dim <= 128,
// and threads loop for head_dim > 128.
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

// Block-level reduce sum: warps → 1 value.
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

// Shared implementation for paged attention v1 with configurable dimensions
// and optional ALiBi support.
//
// Each thread block handles one (head, sequence) pair.
// Threads cooperate across the head dimension: for head_dim <= NUM_THREADS,
// each thread handles one dimension; for head_dim > NUM_THREADS, threads loop.
//
// When alibi_slopes is non-null, the ALiBi bias is added to the scaled QK
// logits: logit[token_pos] += alibi_slope * (token_pos - (seq_len - 1))
//
// Dynamic shared memory layout:
//   float q_smem[head_dim]             -- query vector
//   float reduce_smem[NUM_WARPS]       -- reduction workspace
//   float logits[max_context_len]      -- QK logits (variable)
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
    const int head_dim,
    const int block_size,
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
    //   [0, head_dim)                          -- q vector
    //   [head_dim, head_dim + NUM_WARPS)       -- reduction workspace
    //   [head_dim + NUM_WARPS, ...)             -- logits array (seq_len floats)
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce_smem = smem + head_dim;
    float* logits = smem + head_dim + NUM_WARPS;

    // Cache strides for [num_blocks, block_size, num_kv_heads, head_dim] layout
    const int cache_stride_block = block_size * num_kv_heads * head_dim;
    const int cache_stride_token = num_kv_heads * head_dim;

    // Load query vector into shared memory (threads loop if head_dim > NUM_THREADS)
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        const int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        q_smem[d] = __bfloat162float(q[q_offset]);
    }
    __syncthreads();

    // ---- Phase 1: Compute QK dot products ----
    const int num_blocks_used = (seq_len + block_size - 1) / block_size;
    float qk_max = -FLT_MAX;

    // The query position is the last position: seq_len - 1 (decode produces 1 token).
    const int query_pos = seq_len - 1;

    for (int block_idx = 0; block_idx < num_blocks_used; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int tokens_in_block = min(block_size, seq_len - block_idx * block_size);

        for (int token = 0; token < tokens_in_block; token++) {
            // Partial dot product across head dimensions
            float qk = 0.0f;
            for (int d = tid; d < head_dim; d += NUM_THREADS) {
                const int k_offset = physical_block * cache_stride_block
                                   + token * cache_stride_token
                                   + kv_head_idx * head_dim
                                   + d;
                const float k_val = __bfloat162float(k_cache[k_offset]);
                qk += q_smem[d] * k_val;
            }

            // Block-level reduce to get full dot product
            qk = block_reduce_sum(qk, reduce_smem);

            // Thread 0 stores the scaled logit (with optional ALiBi bias)
            if (tid == 0) {
                const int token_pos = block_idx * block_size + token;
                float scaled_qk = qk * scale;
                // ALiBi bias: slope * (key_position - query_position)
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
    // Each thread accumulates for its subset of head dimensions
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;

        for (int block_idx = 0; block_idx < num_blocks_used; block_idx++) {
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
            const int tokens_in_block = min(block_size, seq_len - block_idx * block_size);

            for (int token = 0; token < tokens_in_block; token++) {
                const int v_offset = physical_block * cache_stride_block
                                   + token * cache_stride_token
                                   + kv_head_idx * head_dim
                                   + d;
                const float v_val = __bfloat162float(v_cache[v_offset]);
                const float weight = logits[block_idx * block_size + token];
                acc += weight * v_val;
            }
        }

        // Write output
        const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        out[out_offset] = __float2bfloat16(acc);
    }
}

// Standard entry point: configurable head_dim and block_size, no ALiBi.
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
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size
) {
    paged_attention_v1_impl(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size,
        nullptr  // no ALiBi
    );
}

// Entry point with ALiBi positional bias support.
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
    const int head_dim,
    const int block_size,
    const float* __restrict__ alibi_slopes     // [num_heads]
) {
    paged_attention_v1_impl(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size,
        alibi_slopes
    );
}

// ============================================================================
// PagedAttention V2: Split-K partitioned attention for long sequences
// ============================================================================
//
// V2 splits the KV sequence into partitions of PARTITION_SIZE tokens.
// Each thread block processes one partition independently, producing:
//   - tmp_out: partial weighted V accumulation
//   - max_logits: max QK logit within the partition (for log-sum-exp)
//   - exp_sums: sum of exp(logit - max) within the partition
//
// A separate reduce kernel merges partitions using numerically stable
// log-sum-exp arithmetic.
//
// Grid: (num_heads, num_seqs, max_num_partitions)
// Block: (NUM_THREADS, 1, 1)

#define PARTITION_SIZE 512

// V2 main kernel: compute attention for one partition of the sequence.
__device__ void paged_attention_v2_impl(
    float* __restrict__ tmp_out,               // [num_seqs, num_heads, max_num_partitions, head_dim]
    float* __restrict__ exp_sums,              // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits_out,        // [num_seqs, num_heads, max_num_partitions]
    const __nv_bfloat16* __restrict__ q,       // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ alibi_slopes     // [num_heads] or nullptr
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Partition boundaries in token space
    const int partition_start_token = partition_idx * PARTITION_SIZE;
    if (partition_start_token >= seq_len) return;  // No work for this partition
    const int partition_end_token = min(partition_start_token + PARTITION_SIZE, seq_len);
    const int partition_num_tokens = partition_end_token - partition_start_token;

    // GQA: map query head to KV head
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // ALiBi slope
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;
    const int query_pos = seq_len - 1;

    // Shared memory layout:
    //   [0, head_dim)                    -- q vector (f32)
    //   [head_dim, head_dim+NUM_WARPS)   -- reduction workspace
    //   [head_dim+NUM_WARPS, ...)        -- logits[PARTITION_SIZE] (f32)
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce_smem = smem + head_dim;
    float* logits = smem + head_dim + NUM_WARPS;

    // Cache strides
    const int cache_stride_block = block_size * num_kv_heads * head_dim;
    const int cache_stride_token = num_kv_heads * head_dim;

    // Load query into shared memory
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        const int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        q_smem[d] = __bfloat162float(q[q_offset]);
    }
    __syncthreads();

    // ---- Phase 1: Compute QK dot products for this partition ----
    // Convert partition token range to cache block range
    const int start_block_idx = partition_start_token / block_size;
    const int end_block_idx = (partition_end_token + block_size - 1) / block_size;

    float qk_max = -FLT_MAX;

    for (int blk = start_block_idx; blk < end_block_idx; blk++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + blk];

        // Tokens within this cache block that fall in our partition
        const int block_start_token = blk * block_size;
        const int token_lo = max(block_start_token, partition_start_token);
        const int token_hi = min(block_start_token + block_size, partition_end_token);

        for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
            const int token_in_block = abs_token - block_start_token;

            float qk = 0.0f;
            for (int d = tid; d < head_dim; d += NUM_THREADS) {
                const int k_offset = physical_block * cache_stride_block
                                   + token_in_block * cache_stride_token
                                   + kv_head_idx * head_dim + d;
                qk += q_smem[d] * __bfloat162float(k_cache[k_offset]);
            }

            qk = block_reduce_sum(qk, reduce_smem);

            if (tid == 0) {
                const int local_idx = abs_token - partition_start_token;
                float scaled_qk = qk * scale;
                scaled_qk += alibi_slope * (float)(abs_token - query_pos);
                logits[local_idx] = scaled_qk;
                qk_max = fmaxf(qk_max, scaled_qk);
            }
            __syncthreads();
        }
    }

    // Broadcast qk_max
    if (tid == 0) {
        reduce_smem[0] = qk_max;
    }
    __syncthreads();
    qk_max = reduce_smem[0];

    // ---- Phase 2: Softmax (local to partition) ----
    float local_exp_sum = 0.0f;
    for (int i = tid; i < partition_num_tokens; i += NUM_THREADS) {
        const float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        local_exp_sum += val;
    }
    __syncthreads();

    local_exp_sum = block_reduce_sum(local_exp_sum, reduce_smem);

    // Store partition-level max and exp_sum for the reduce kernel
    const int partition_meta_offset = seq_idx * num_heads * max_num_partitions
                                    + head_idx * max_num_partitions
                                    + partition_idx;
    if (tid == 0) {
        max_logits_out[partition_meta_offset] = qk_max;
        exp_sums[partition_meta_offset] = local_exp_sum;
    }

    // Normalize logits (divide by exp_sum) for weighted V accumulation
    const float inv_sum = __fdividef(1.0f, local_exp_sum + 1e-6f);
    for (int i = tid; i < partition_num_tokens; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation (stored as f32 in tmp_out) ----
    const int tmp_out_offset = seq_idx * num_heads * max_num_partitions * head_dim
                             + head_idx * max_num_partitions * head_dim
                             + partition_idx * head_dim;

    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;

        for (int blk = start_block_idx; blk < end_block_idx; blk++) {
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + blk];
            const int block_start_token = blk * block_size;
            const int token_lo = max(block_start_token, partition_start_token);
            const int token_hi = min(block_start_token + block_size, partition_end_token);

            for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
                const int token_in_block = abs_token - block_start_token;
                const int v_offset = physical_block * cache_stride_block
                                   + token_in_block * cache_stride_token
                                   + kv_head_idx * head_dim + d;
                const int local_idx = abs_token - partition_start_token;
                acc += logits[local_idx] * __bfloat162float(v_cache[v_offset]);
            }
        }

        tmp_out[tmp_out_offset + d] = acc;
    }
}

// V2 reduce kernel: merge partition results using log-sum-exp.
//
// Grid: (num_heads, num_seqs, 1)
// Block: (NUM_THREADS, 1, 1)
//
// For each (head, seq), combines tmp_out from all partitions into final output.
// Uses the identity:
//   softmax(concat(A,B)) = rescale(softmax(A), softmax(B))
// via the log-sum-exp trick for numerical stability.
extern "C" __global__ void paged_attention_v2_reduce_bf16(
    __nv_bfloat16* __restrict__ out,        // [num_seqs, num_heads, head_dim]
    const float* __restrict__ tmp_out,      // [num_seqs, num_heads, max_num_partitions, head_dim]
    const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
    const int* __restrict__ seq_lens,       // [num_seqs]
    const int num_heads,
    const int head_dim,
    const int max_num_partitions
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int num_partitions = (seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

    // Fast path: single partition → just copy (convert f32 → bf16)
    if (num_partitions == 1) {
        const int tmp_offset = seq_idx * num_heads * max_num_partitions * head_dim
                             + head_idx * max_num_partitions * head_dim;
        const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            out[out_offset + d] = __float2bfloat16(tmp_out[tmp_offset + d]);
        }
        return;
    }

    // Shared memory for partition metadata:
    //   [0, max_num_partitions)                    -- max logits per partition
    //   [max_num_partitions, 2*max_num_partitions) -- rescaled exp sums
    extern __shared__ float reduce_smem[];
    float* shared_max_logits = reduce_smem;
    float* shared_exp_sums = reduce_smem + max_num_partitions;

    // Step 1: Find global max logit across all partitions
    const int meta_offset = seq_idx * num_heads * max_num_partitions
                          + head_idx * max_num_partitions;

    float global_max = -FLT_MAX;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        float l = max_logits[meta_offset + p];
        shared_max_logits[p] = l;
        global_max = fmaxf(global_max, l);
    }
    __syncthreads();

    // Reduce global_max across threads
    // Warp-level reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        global_max = fmaxf(global_max, __shfl_xor_sync(0xffffffff, global_max, offset));
    }
    // Inter-warp reduce via shared memory (reuse end of shared_exp_sums temporarily)
    {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        // Use space after shared_exp_sums for warp reduction
        float* warp_reduce = reduce_smem + 2 * max_num_partitions;
        if (lane_id == 0) {
            warp_reduce[warp_id] = global_max;
        }
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? warp_reduce[lane_id] : -FLT_MAX;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
            }
            if (lane_id == 0) {
                warp_reduce[0] = v;
            }
        }
        __syncthreads();
        global_max = warp_reduce[0];
    }

    // Step 2: Rescale exp_sums by the global max (log-sum-exp correction)
    float global_exp_sum = 0.0f;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        float rescaled = exp_sums[meta_offset + p] * __expf(shared_max_logits[p] - global_max);
        shared_exp_sums[p] = rescaled;
        global_exp_sum += rescaled;
    }
    __syncthreads();

    // Reduce global_exp_sum across threads
    {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        // Warp-level reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            global_exp_sum += __shfl_xor_sync(0xffffffff, global_exp_sum, offset);
        }
        float* warp_reduce = reduce_smem + 2 * max_num_partitions;
        if (lane_id == 0) {
            warp_reduce[warp_id] = global_exp_sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? warp_reduce[lane_id] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane_id == 0) {
                warp_reduce[0] = v;
            }
        }
        __syncthreads();
        global_exp_sum = warp_reduce[0];
    }

    const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

    // Step 3: Aggregate weighted outputs from all partitions
    const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int tmp_base = seq_idx * num_heads * max_num_partitions * head_dim
                       + head_idx * max_num_partitions * head_dim;

    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;
        for (int p = 0; p < num_partitions; p++) {
            // tmp_out already has locally-normalized values (divided by local exp_sum).
            // We need to undo that normalization and apply global normalization:
            //   contribution = tmp_out[p][d] * local_exp_sum[p] * exp(local_max - global_max) / global_exp_sum
            //                = tmp_out[p][d] * shared_exp_sums[p] / global_exp_sum
            acc += tmp_out[tmp_base + p * head_dim + d] * shared_exp_sums[p];
        }
        out[out_offset + d] = __float2bfloat16(acc * inv_global_exp_sum);
    }
}

// V2 entry point: standard (no ALiBi).
extern "C" __global__ void paged_attention_v2_bf16(
    float* __restrict__ tmp_out,               // [num_seqs, num_heads, max_num_partitions, head_dim]
    float* __restrict__ exp_sums,              // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,            // [num_seqs, num_heads, max_num_partitions]
    const __nv_bfloat16* __restrict__ q,       // [num_seqs, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int max_num_partitions
) {
    paged_attention_v2_impl(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, max_num_partitions,
        nullptr  // no ALiBi
    );
}

// V2 entry point with ALiBi support.
extern "C" __global__ void paged_attention_v2_bf16_alibi(
    float* __restrict__ tmp_out,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ alibi_slopes     // [num_heads]
) {
    paged_attention_v2_impl(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, max_num_partitions,
        alibi_slopes
    );
}
