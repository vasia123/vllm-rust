// Fused Top-K Softmax kernel for MoE routing.
//
// This kernel fuses softmax computation with top-k selection for efficient
// expert routing. Based on vLLM's implementation with Rust integration focus.
//
// Algorithm:
// 1. Compute softmax over expert logits using parallel reduction
// 2. Find top-k indices and values using iterative selection
// 3. Optionally renormalize weights to sum to 1
//
// Optimizations:
// - Warp-level reductions for small expert counts (<= 32)
// - Minimal shared memory usage for warp-optimized path

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Type conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Warp-level argmax: returns value, index pair with min index for ties
__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_xor_sync(FULL_MASK, val, offset);
        int other_idx = __shfl_xor_sync(FULL_MASK, idx, offset);
        // Take larger value, or smaller index on tie
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Fused top-k softmax kernel for float32 inputs using warp-level operations.
// Each warp processes one token. Optimized for num_experts <= 32.
extern "C" __global__ void topk_softmax_f32_warp(
    const float* __restrict__ logits,       // [batch_size, num_experts]
    float* __restrict__ topk_weights,       // [batch_size, k]
    int32_t* __restrict__ topk_indices,     // [batch_size, k]
    const int32_t batch_size,
    const int32_t num_experts,
    const int32_t k,
    const int32_t renormalize
) {
    // Each warp processes one token
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= batch_size) return;

    // Point to this token's data
    const float* token_logits = logits + warp_id * num_experts;
    float* out_weights = topk_weights + warp_id * k;
    int32_t* out_indices = topk_indices + warp_id * k;

    // Each lane loads one expert logit (up to 32 experts)
    float val = (lane_id < num_experts) ? token_logits[lane_id] : -FLT_MAX;

    // Step 1: Find max for numerical stability
    float row_max = warp_reduce_max(val);

    // Step 2: Compute exp and sum
    float exp_val = (lane_id < num_experts) ? expf(val - row_max) : 0.0f;
    float row_sum = warp_reduce_sum(exp_val);

    // Step 3: Normalize to softmax
    float softmax_val = exp_val / row_sum;

    // Step 4: Find top-k
    float selected_sum = 0.0f;

    for (int ki = 0; ki < k; ++ki) {
        // Find max using warp reduction
        float max_val = softmax_val;
        int max_idx = lane_id;
        warp_reduce_argmax(max_val, max_idx);

        // Lane 0 stores result
        if (lane_id == 0) {
            out_weights[ki] = max_val;
            out_indices[ki] = max_idx;
            selected_sum += max_val;
        }

        // Broadcast max_idx and zero it out
        max_idx = __shfl_sync(FULL_MASK, max_idx, 0);
        if (lane_id == max_idx) {
            softmax_val = -FLT_MAX;
        }
    }

    // Step 5: Renormalize if requested
    if (renormalize != 0 && lane_id == 0) {
        float norm_factor = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
        for (int ki = 0; ki < k; ++ki) {
            out_weights[ki] *= norm_factor;
        }
    }
}

// Float16 version using warp-level operations
extern "C" __global__ void topk_softmax_f16_warp(
    const __half* __restrict__ logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices,
    const int32_t batch_size,
    const int32_t num_experts,
    const int32_t k,
    const int32_t renormalize
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= batch_size) return;

    const __half* token_logits = logits + warp_id * num_experts;
    float* out_weights = topk_weights + warp_id * k;
    int32_t* out_indices = topk_indices + warp_id * k;

    float val = (lane_id < num_experts) ? to_float(token_logits[lane_id]) : -FLT_MAX;

    float row_max = warp_reduce_max(val);
    float exp_val = (lane_id < num_experts) ? expf(val - row_max) : 0.0f;
    float row_sum = warp_reduce_sum(exp_val);
    float softmax_val = exp_val / row_sum;

    float selected_sum = 0.0f;

    for (int ki = 0; ki < k; ++ki) {
        float max_val = softmax_val;
        int max_idx = lane_id;
        warp_reduce_argmax(max_val, max_idx);

        if (lane_id == 0) {
            out_weights[ki] = max_val;
            out_indices[ki] = max_idx;
            selected_sum += max_val;
        }

        max_idx = __shfl_sync(FULL_MASK, max_idx, 0);
        if (lane_id == max_idx) {
            softmax_val = -FLT_MAX;
        }
    }

    if (renormalize != 0 && lane_id == 0) {
        float norm_factor = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
        for (int ki = 0; ki < k; ++ki) {
            out_weights[ki] *= norm_factor;
        }
    }
}

// BFloat16 version using warp-level operations
extern "C" __global__ void topk_softmax_bf16_warp(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices,
    const int32_t batch_size,
    const int32_t num_experts,
    const int32_t k,
    const int32_t renormalize
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= batch_size) return;

    const __nv_bfloat16* token_logits = logits + warp_id * num_experts;
    float* out_weights = topk_weights + warp_id * k;
    int32_t* out_indices = topk_indices + warp_id * k;

    float val = (lane_id < num_experts) ? to_float(token_logits[lane_id]) : -FLT_MAX;

    float row_max = warp_reduce_max(val);
    float exp_val = (lane_id < num_experts) ? expf(val - row_max) : 0.0f;
    float row_sum = warp_reduce_sum(exp_val);
    float softmax_val = exp_val / row_sum;

    float selected_sum = 0.0f;

    for (int ki = 0; ki < k; ++ki) {
        float max_val = softmax_val;
        int max_idx = lane_id;
        warp_reduce_argmax(max_val, max_idx);

        if (lane_id == 0) {
            out_weights[ki] = max_val;
            out_indices[ki] = max_idx;
            selected_sum += max_val;
        }

        max_idx = __shfl_sync(FULL_MASK, max_idx, 0);
        if (lane_id == max_idx) {
            softmax_val = -FLT_MAX;
        }
    }

    if (renormalize != 0 && lane_id == 0) {
        float norm_factor = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
        for (int ki = 0; ki < k; ++ki) {
            out_weights[ki] *= norm_factor;
        }
    }
}

// Block-level max reduction using shared memory
__device__ float block_reduce_max_256(float val, float* shared) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int num_warps = 256 / WARP_SIZE;

    // First, reduce within each warp
    val = warp_reduce_max(val);

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
    }

    return val;
}

// Block-level sum reduction using shared memory
__device__ float block_reduce_sum_256(float val, float* shared) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int num_warps = 256 / WARP_SIZE;

    // First, reduce within each warp
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// Block-level argmax using shared memory
__device__ void block_reduce_argmax_256(float& val, int& idx, float* shared_val, int* shared_idx) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    constexpr int num_warps = 256 / WARP_SIZE;

    // First, reduce within each warp
    warp_reduce_argmax(val, idx);

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_val[warp_id] = val;
        shared_idx[warp_id] = idx;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? shared_val[lane_id] : -FLT_MAX;
        idx = (lane_id < num_warps) ? shared_idx[lane_id] : -1;
        warp_reduce_argmax(val, idx);
    }
}

// Generic kernel for larger expert counts (> 32).
// Each block processes one token using block-level reductions.
extern "C" __global__ void topk_softmax_f32_generic(
    const float* __restrict__ logits,       // [batch_size, num_experts]
    float* __restrict__ topk_weights,       // [batch_size, k]
    int32_t* __restrict__ topk_indices,     // [batch_size, k]
    const int32_t batch_size,
    const int32_t num_experts,
    const int32_t k,
    const int32_t renormalize
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= batch_size) return;

    const int tid = threadIdx.x;
    constexpr int BLOCK_SIZE = 256;
    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;

    // Shared memory layout
    extern __shared__ char smem[];
    float* shared_reduce = reinterpret_cast<float*>(smem);
    float* softmax_vals = shared_reduce + num_warps;  // [num_experts]
    int* argmax_idx = reinterpret_cast<int*>(softmax_vals + num_experts);

    // Point to this token's logits
    const float* token_logits = logits + token_idx * num_experts;

    // Step 1: Find max for numerical stability
    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_experts; i += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, token_logits[i]);
    }
    float row_max = block_reduce_max_256(thread_max, shared_reduce);
    __syncthreads();

    // Broadcast max to all threads
    if (tid == 0) {
        shared_reduce[0] = row_max;
    }
    __syncthreads();
    row_max = shared_reduce[0];

    // Step 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < num_experts; i += BLOCK_SIZE) {
        float exp_val = expf(token_logits[i] - row_max);
        softmax_vals[i] = exp_val;
        thread_sum += exp_val;
    }
    __syncthreads();

    float row_sum = block_reduce_sum_256(thread_sum, shared_reduce);
    __syncthreads();

    // Broadcast sum to all threads
    if (tid == 0) {
        shared_reduce[0] = row_sum;
    }
    __syncthreads();
    row_sum = shared_reduce[0];

    // Step 3: Normalize to get softmax probabilities
    const float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < num_experts; i += BLOCK_SIZE) {
        softmax_vals[i] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Find top-k using iterative selection
    float* out_weights = topk_weights + token_idx * k;
    int32_t* out_indices = topk_indices + token_idx * k;

    float selected_sum = 0.0f;

    for (int ki = 0; ki < k; ++ki) {
        // Find maximum among remaining values
        float thread_val = -FLT_MAX;
        int thread_idx = -1;

        for (int i = tid; i < num_experts; i += BLOCK_SIZE) {
            float val = softmax_vals[i];
            if (val > thread_val) {
                thread_val = val;
                thread_idx = i;
            }
        }

        // Reduce to find global max
        float max_val = thread_val;
        int max_idx = thread_idx;
        block_reduce_argmax_256(max_val, max_idx, shared_reduce, argmax_idx);
        __syncthreads();

        // Thread 0 broadcasts and stores result
        if (tid == 0) {
            shared_reduce[0] = max_val;
            argmax_idx[0] = max_idx;
        }
        __syncthreads();

        max_val = shared_reduce[0];
        max_idx = argmax_idx[0];

        if (tid == 0) {
            out_weights[ki] = max_val;
            out_indices[ki] = max_idx;
            selected_sum += max_val;
        }

        // Zero out selected expert for next iteration
        if (max_idx >= 0 && tid == (max_idx % BLOCK_SIZE)) {
            softmax_vals[max_idx] = -FLT_MAX;
        }
        __syncthreads();
    }

    // Step 5: Renormalize if requested
    if (renormalize != 0 && tid == 0) {
        float norm_factor = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
        for (int ki = 0; ki < k; ++ki) {
            out_weights[ki] *= norm_factor;
        }
    }
}
