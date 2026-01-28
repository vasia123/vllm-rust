// Fused MoE token alignment kernel.
// Sorts tokens by expert assignment and pads to block boundaries.
//
// Grid: (2, 1, 1) - two blocks for parallel work
// Block: (1024, 1, 1) - maximum threads for prefix sum
//
// Based on vLLM's moe_align_block_size_kernel implementation.

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Block-level prefix sum using CUB
template <int BLOCK_THREADS>
__device__ void block_exclusive_sum(int32_t value, int32_t* result, int32_t* shared_temp) {
    using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).ExclusiveSum(value, *result);
}

// Main alignment kernel for standard (non-batched) MoE.
//
// Parameters:
// - topk_ids: [numel] flattened expert indices (num_tokens * top_k)
// - sorted_token_ids: [max_num_tokens_padded] output sorted indices
// - expert_ids: [max_num_blocks] output expert ID per block
// - total_tokens_post_pad: [1] output total padded token count
// - num_experts: number of experts
// - block_size: alignment block size for GEMM
// - numel: total number of expert assignments (num_tokens * top_k)
// - max_num_tokens_padded: maximum output size
extern "C" __global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t num_experts,
    const int32_t block_size,
    const int32_t numel,
    const int32_t max_num_tokens_padded,
    int32_t* __restrict__ cumsum_buffer
) {
    // Shared memory for expert token counts
    extern __shared__ int32_t shared_mem[];
    int32_t* expert_counts = shared_mem;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int max_num_blocks = CEILDIV(max_num_tokens_padded, block_size);

    // Block 0: Count tokens and compute alignment
    // Block 1: Fill sorted_token_ids with invalid marker
    if (blockIdx.x == 1) {
        // Initialize sorted_token_ids with numel (invalid marker)
        for (int i = tid; i < max_num_tokens_padded; i += stride) {
            sorted_token_ids[i] = numel;
        }
        return;
    }

    // Block 0: Main alignment logic

    // Initialize expert counts
    if (tid < num_experts) {
        expert_counts[tid] = 0;
    }
    __syncthreads();

    // Count tokens per expert
    for (int i = tid; i < numel; i += stride) {
        int expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // Compute padded counts and cumulative sum
    int32_t expert_count = 0;
    if (tid < num_experts) {
        expert_count = expert_counts[tid];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    // Block-level exclusive prefix sum
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    int32_t cumsum_val;
    BlockScan(scan_temp).ExclusiveSum(expert_count, cumsum_val);

    // Store cumsum for later use
    if (tid <= num_experts) {
        cumsum_buffer[tid] = cumsum_val;
    }

    // Thread num_experts stores total
    if (tid == num_experts) {
        total_tokens_post_pad[0] = cumsum_val;
    }
    __syncthreads();

    // Generate expert_ids for each block
    if (tid < num_experts) {
        int start_block = cumsum_buffer[tid] / block_size;
        int end_block = cumsum_buffer[tid + 1] / block_size;
        for (int b = start_block; b < end_block; ++b) {
            expert_ids[b] = tid;
        }
    }

    // Fill remaining expert_ids with -1
    int fill_start = cumsum_buffer[num_experts] / block_size + tid;
    for (int i = fill_start; i < max_num_blocks; i += stride) {
        expert_ids[i] = -1;
    }
    __syncthreads();
}

// Second pass: sort tokens into their positions
extern "C" __global__ void moe_sort_tokens_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    const int32_t num_experts,
    const int32_t numel,
    const int32_t max_num_tokens_padded
) {
    const int tid = blockIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.y;

    for (int i = tid; i < numel; i += stride) {
        int expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            // Atomically get position within expert's allocation
            int pos = atomicAdd(&cumsum_buffer[expert_id], 1);
            if (pos < max_num_tokens_padded) {
                sorted_token_ids[pos] = i;
            }
        }
    }
}

// Simplified kernel for small batches (< 1024 tokens, <= 64 experts)
// Combines counting and sorting in single pass.
extern "C" __global__ void moe_align_block_size_small_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    const int32_t num_experts,
    const int32_t block_size,
    const int32_t numel,
    const int32_t max_num_tokens_padded
) {
    // Small batch kernel uses simpler shared memory layout
    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = shared_mem + num_experts + 1;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int max_num_blocks = CEILDIV(max_num_tokens_padded, block_size);

    // First portion of threads fill sorted_token_ids
    const int FILL_THREADS = 256;
    if (tid < FILL_THREADS) {
        for (int i = tid; i < max_num_tokens_padded; i += FILL_THREADS) {
            sorted_token_ids[i] = numel;
        }
        // Wait for other threads
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
    }

    // Remaining threads do counting and sorting
    const int work_tid = tid - FILL_THREADS;
    const int work_stride = stride - FILL_THREADS;

    // Initialize per-thread counts
    for (int e = 0; e < num_experts; ++e) {
        tokens_cnts[(work_tid + 1) * num_experts + e] = 0;
    }

    // Count tokens per expert per thread
    for (int i = work_tid; i < numel; i += work_stride) {
        int expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            tokens_cnts[(work_tid + 1) * num_experts + expert_id] += 1;
        }
    }
    __syncthreads();

    // Reduce counts across threads
    if (work_tid < num_experts) {
        tokens_cnts[work_tid] = 0;
        for (int t = 1; t <= work_stride; ++t) {
            tokens_cnts[t * num_experts + work_tid] +=
                tokens_cnts[(t - 1) * num_experts + work_tid];
        }
    }
    __syncthreads();

    // Compute cumulative sum with padding
    if (work_tid == 0) {
        cumsum[0] = 0;
        for (int e = 1; e <= num_experts; ++e) {
            int count = tokens_cnts[work_stride * num_experts + e - 1];
            cumsum[e] = cumsum[e - 1] + CEILDIV(count, block_size) * block_size;
        }
        total_tokens_post_pad[0] = cumsum[num_experts];
    }
    __syncthreads();

    // Generate expert_ids
    if (work_tid < num_experts) {
        for (int b = cumsum[work_tid] / block_size; b < cumsum[work_tid + 1] / block_size; ++b) {
            expert_ids[b] = work_tid;
        }
    }

    // Fill remaining expert_ids with -1
    int fill_start = cumsum[num_experts] / block_size + work_tid;
    for (int b = fill_start; b < max_num_blocks; b += work_stride) {
        expert_ids[b] = -1;
    }

    // Scatter tokens to sorted positions
    for (int i = work_tid; i < numel; i += work_stride) {
        int expert_id = topk_ids[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            int pos = tokens_cnts[work_tid * num_experts + expert_id] + cumsum[expert_id];
            sorted_token_ids[pos] = i;
            tokens_cnts[work_tid * num_experts + expert_id] += 1;
        }
    }
}

// Reduction kernel: sum expert outputs weighted by routing weights
// output[token] = sum_k(routing_weight[token, k] * expert_output[sorted_pos[token, k]])
extern "C" __global__ void moe_sum_kernel(
    float* __restrict__ output,           // [num_tokens, hidden_size]
    const float* __restrict__ input,      // [num_tokens, top_k, hidden_size]
    const int hidden_size,
    const int top_k
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    for (int d = tid; d < hidden_size; d += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += input[token_idx * top_k * hidden_size + k * hidden_size + d];
        }
        output[token_idx * hidden_size + d] = sum;
    }
}

// BFloat16 version of sum kernel
extern "C" __global__ void moe_sum_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int hidden_size,
    const int top_k
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    for (int d = tid; d < hidden_size; d += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += __bfloat162float(input[token_idx * top_k * hidden_size + k * hidden_size + d]);
        }
        output[token_idx * hidden_size + d] = __float2bfloat16(sum);
    }
}
