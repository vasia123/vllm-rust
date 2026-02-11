// GPU-side sampling kernels for inference.
//
// Avoids GPU→CPU logits transfer for greedy (argmax) and top-k/top-p sampling.
//
// Kernels:
//   argmax_bf16 / argmax_f32: greedy sampling per sequence
//   top_k_top_p_sample: combined top-k/top-p with multinomial (uniform params)
//   top_k_top_p_sample_per_seq: per-sequence top-k/top-p/temperature
//   temperature_scale_bf16: in-place temperature scaling
//   softmax_to_probs / softmax_to_probs_f32: logits → f32 probabilities

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

#define NUM_THREADS 256

// Warp-level reduce max with index
__device__ __forceinline__ void warp_reduce_max_idx(float& val, int& idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// Warp-level reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Argmax kernel: greedy sampling
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
// Each block finds the argmax of one sequence's logit vector.
extern "C" __global__ void argmax_bf16(
    int* __restrict__ output_ids,            // [num_seqs]
    const __nv_bfloat16* __restrict__ logits, // [num_seqs, vocab_size]
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = NUM_THREADS / 32;

    const __nv_bfloat16* seq_logits = logits + seq_idx * vocab_size;

    // Each thread finds local max
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        float val = __bfloat162float(seq_logits[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp-level reduce
    warp_reduce_max_idx(local_max, local_idx);

    // Inter-warp reduce via shared memory
    __shared__ float smem_vals[8];  // max NUM_WARPS = 8
    __shared__ int smem_idxs[8];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        smem_vals[warp_id] = local_max;
        smem_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    // First warp does final reduction
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem_vals[lane_id] : -FLT_MAX;
        int idx = (lane_id < num_warps) ? smem_idxs[lane_id] : 0;
        warp_reduce_max_idx(v, idx);
        if (lane_id == 0) {
            output_ids[seq_idx] = idx;
        }
    }
}

// ============================================================================
// Temperature scaling kernel
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
// Scales logits in-place by 1/temperature.
extern "C" __global__ void temperature_scale_bf16(
    __nv_bfloat16* __restrict__ logits, // [num_seqs, vocab_size]
    const float inv_temperature,
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;

    __nv_bfloat16* seq_logits = logits + seq_idx * vocab_size;

    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        float val = __bfloat162float(seq_logits[i]) * inv_temperature;
        seq_logits[i] = __float2bfloat16(val);
    }
}

// ============================================================================
// Softmax kernel (bf16 input → f32 output probabilities)
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
extern "C" __global__ void softmax_to_probs(
    float* __restrict__ probs,                // [num_seqs, vocab_size] output
    const __nv_bfloat16* __restrict__ logits, // [num_seqs, vocab_size] input
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = NUM_THREADS / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const __nv_bfloat16* seq_logits = logits + seq_idx * vocab_size;
    float* seq_probs = probs + seq_idx * vocab_size;

    __shared__ float smem[8];

    // Pass 1: find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        local_max = fmaxf(local_max, __bfloat162float(seq_logits[i]));
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) smem[0] = v;
    }
    __syncthreads();
    float max_val = smem[0];

    // Pass 2: compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        float val = __expf(__bfloat162float(seq_logits[i]) - max_val);
        seq_probs[i] = val;
        local_sum += val;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) smem[0] = v;
    }
    __syncthreads();
    float total_sum = smem[0];

    // Pass 3: normalize
    float inv_sum = __fdividef(1.0f, total_sum + 1e-6f);
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        seq_probs[i] *= inv_sum;
    }
}

// ============================================================================
// Top-k filtering + multinomial sampling (single thread per sequence)
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (1, 1, 1)
//
// Applies top-k + top-p filtering and samples from the filtered distribution.
// Single-threaded per sequence for correctness; for batch_size >> 1 the
// parallelism comes from the grid dimension.
extern "C" __global__ void top_k_top_p_sample(
    int* __restrict__ output_ids,        // [num_seqs]
    const float* __restrict__ probs,     // [num_seqs, vocab_size]
    const float* __restrict__ rand_vals, // [num_seqs] uniform random in [0,1)
    const int vocab_size,
    const int top_k,                     // 0 = disabled
    const float top_p                    // 1.0 = disabled
) {
    const int seq_idx = blockIdx.x;
    const float* seq_probs = probs + seq_idx * vocab_size;
    const float rand_val = rand_vals[seq_idx];

    // Find top-k threshold via iterative approach
    // For typical top_k (1-100), this is acceptable
    float threshold = 0.0f;

    if (top_k > 0 && top_k < vocab_size) {
        // Find k-th largest probability
        float prev_max = FLT_MAX;
        for (int rank = 0; rank < top_k; rank++) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                float p = seq_probs[i];
                if (p < prev_max && p > best) {
                    best = p;
                }
            }
            if (best <= -FLT_MAX) break;
            prev_max = best;
            threshold = best;
        }
    }

    // Compute filtered sum
    float filtered_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (seq_probs[i] >= threshold) {
            filtered_sum += seq_probs[i];
        }
    }

    // Apply top-p: accumulate from highest prob down until cumsum >= top_p
    if (top_p < 1.0f && top_p > 0.0f) {
        float cumsum = 0.0f;
        float top_p_thresh = 0.0f;
        float prev_max = FLT_MAX;
        float target_sum = top_p * filtered_sum;

        while (cumsum < target_sum) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                float p = seq_probs[i];
                if (p >= threshold && p < prev_max && p > best) {
                    best = p;
                }
            }
            if (best <= -FLT_MAX) break;
            prev_max = best;
            top_p_thresh = best;

            for (int i = 0; i < vocab_size; i++) {
                if (seq_probs[i] == best && seq_probs[i] >= threshold) {
                    cumsum += best;
                    if (cumsum >= target_sum) break;
                }
            }
        }

        if (top_p_thresh > threshold) {
            threshold = top_p_thresh;
        }

        // Recompute filtered sum after top-p
        filtered_sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (seq_probs[i] >= threshold) {
                filtered_sum += seq_probs[i];
            }
        }
    }

    // Multinomial sample from filtered distribution
    float target = rand_val * filtered_sum;
    float cumsum = 0.0f;
    int selected = vocab_size - 1;

    for (int i = 0; i < vocab_size; i++) {
        if (seq_probs[i] >= threshold) {
            cumsum += seq_probs[i];
            if (cumsum > target) {
                selected = i;
                break;
            }
        }
    }

    output_ids[seq_idx] = selected;
}

// ============================================================================
// F32 Argmax kernel: greedy sampling for float32 logits
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
extern "C" __global__ void argmax_f32(
    int* __restrict__ output_ids,       // [num_seqs]
    const float* __restrict__ logits,   // [num_seqs, vocab_size]
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = NUM_THREADS / 32;

    const float* seq_logits = logits + seq_idx * vocab_size;

    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        float val = seq_logits[i];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    warp_reduce_max_idx(local_max, local_idx);

    __shared__ float smem_vals[8];
    __shared__ int smem_idxs[8];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (lane_id == 0) {
        smem_vals[warp_id] = local_max;
        smem_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem_vals[lane_id] : -FLT_MAX;
        int idx = (lane_id < num_warps) ? smem_idxs[lane_id] : 0;
        warp_reduce_max_idx(v, idx);
        if (lane_id == 0) {
            output_ids[seq_idx] = idx;
        }
    }
}

// ============================================================================
// F32 Softmax kernel (f32 input → f32 output probabilities)
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
extern "C" __global__ void softmax_to_probs_f32(
    float* __restrict__ probs,          // [num_seqs, vocab_size] output
    const float* __restrict__ logits,   // [num_seqs, vocab_size] input
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_warps = NUM_THREADS / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const float* seq_logits = logits + seq_idx * vocab_size;
    float* seq_probs = probs + seq_idx * vocab_size;

    __shared__ float smem[8];

    // Pass 1: find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        local_max = fmaxf(local_max, seq_logits[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) smem[0] = v;
    }
    __syncthreads();
    float max_val = smem[0];

    // Pass 2: compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        float val = __expf(seq_logits[i] - max_val);
        seq_probs[i] = val;
        local_sum += val;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) smem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) smem[0] = v;
    }
    __syncthreads();
    float total_sum = smem[0];

    // Pass 3: normalize
    float inv_sum = __fdividef(1.0f, total_sum + 1e-6f);
    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        seq_probs[i] *= inv_sum;
    }
}

// ============================================================================
// Per-sequence top-k/top-p/temperature sampling
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (1, 1, 1)
//
// Each sequence gets its own temperature, top_k, and top_p from arrays.
// Temperature is applied before softmax conversion (probs should be raw logits
// converted to probabilities beforehand with appropriate temperature).
extern "C" __global__ void top_k_top_p_sample_per_seq(
    int* __restrict__ output_ids,          // [num_seqs]
    const float* __restrict__ probs,       // [num_seqs, vocab_size]
    const float* __restrict__ rand_vals,   // [num_seqs] uniform random in [0,1)
    const int* __restrict__ top_k_arr,     // [num_seqs] per-sequence top_k
    const float* __restrict__ top_p_arr,   // [num_seqs] per-sequence top_p
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const float* seq_probs = probs + seq_idx * vocab_size;
    const float rand_val = rand_vals[seq_idx];
    const int top_k = top_k_arr[seq_idx];
    const float top_p = top_p_arr[seq_idx];

    // Find top-k threshold
    float threshold = 0.0f;

    if (top_k > 0 && top_k < vocab_size) {
        float prev_max = FLT_MAX;
        for (int rank = 0; rank < top_k; rank++) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                float p = seq_probs[i];
                if (p < prev_max && p > best) {
                    best = p;
                }
            }
            if (best <= -FLT_MAX) break;
            prev_max = best;
            threshold = best;
        }
    }

    // Compute filtered sum
    float filtered_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (seq_probs[i] >= threshold) {
            filtered_sum += seq_probs[i];
        }
    }

    // Apply top-p
    if (top_p < 1.0f && top_p > 0.0f) {
        float cumsum = 0.0f;
        float top_p_thresh = 0.0f;
        float prev_max = FLT_MAX;
        float target_sum = top_p * filtered_sum;

        while (cumsum < target_sum) {
            float best = -FLT_MAX;
            for (int i = 0; i < vocab_size; i++) {
                float p = seq_probs[i];
                if (p >= threshold && p < prev_max && p > best) {
                    best = p;
                }
            }
            if (best <= -FLT_MAX) break;
            prev_max = best;
            top_p_thresh = best;

            for (int i = 0; i < vocab_size; i++) {
                if (seq_probs[i] == best && seq_probs[i] >= threshold) {
                    cumsum += best;
                    if (cumsum >= target_sum) break;
                }
            }
        }

        if (top_p_thresh > threshold) {
            threshold = top_p_thresh;
        }

        filtered_sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (seq_probs[i] >= threshold) {
                filtered_sum += seq_probs[i];
            }
        }
    }

    // Multinomial sample
    float target = rand_val * filtered_sum;
    float cumsum = 0.0f;
    int selected = vocab_size - 1;

    for (int i = 0; i < vocab_size; i++) {
        if (seq_probs[i] >= threshold) {
            cumsum += seq_probs[i];
            if (cumsum > target) {
                selected = i;
                break;
            }
        }
    }

    output_ids[seq_idx] = selected;
}

// ============================================================================
// Per-sequence temperature scaling (f32, in-place)
// ============================================================================
// Grid: (num_seqs, 1, 1)
// Block: (NUM_THREADS, 1, 1)
// Scales each sequence's logits by its own 1/temperature.
extern "C" __global__ void temperature_scale_f32_per_seq(
    float* __restrict__ logits,             // [num_seqs, vocab_size]
    const float* __restrict__ inv_temps,    // [num_seqs]
    const int vocab_size
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float inv_t = inv_temps[seq_idx];

    float* seq_logits = logits + seq_idx * vocab_size;

    for (int i = tid; i < vocab_size; i += NUM_THREADS) {
        seq_logits[i] *= inv_t;
    }
}
