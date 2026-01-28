// Fused MoE GEMM kernel for expert computation.
//
// Implements the permute-compute pattern:
// 1. Load tokens according to sorted_token_ids (permutation)
// 2. Multiply with expert weights selected by expert_ids
// 3. Optionally apply routing weights
//
// Based on vLLM's fused_moe_kernel Triton implementation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

// Warp-level reduce sum via shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum
__device__ float block_reduce_sum(float val, float* smem, int warp_id, int lane_id, int num_warps) {
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            smem[0] = v;
        }
    }
    __syncthreads();
    return smem[0];
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ __nv_bfloat16 silu_bf16(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    return __float2bfloat16(silu(fx));
}

// Fused MoE GEMM kernel for single projection.
// Computes: C[sorted_idx] = A[token_idx] @ B[expert]
//
// Parameters:
// - A: Input activations [num_tokens, K]
// - B: Expert weights [num_experts, N, K] (transposed for efficiency)
// - C: Output [num_tokens_padded * top_k, N]
// - topk_weights: Routing weights [num_tokens * top_k]
// - sorted_token_ids: Permutation [num_tokens_padded * top_k]
// - expert_ids: Expert per block [num_blocks]
// - num_tokens_post_padded: Scalar, total padded tokens
// - N, K: Matrix dimensions
// - num_valid_tokens: Number of valid (non-padding) entries
// - top_k: Number of experts per token
// - mul_routed_weight: Whether to apply routing weights
//
// Grid: (num_pid_m * num_pid_n, 1, 1)
// Block: (BLOCK_SIZE_K, 1, 1) or custom
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, bool MUL_ROUTED_WEIGHT>
__global__ void fused_moe_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    const int N,
    const int K,
    const int num_valid_tokens,
    const int top_k,
    const int stride_am,  // A row stride
    const int stride_be,  // B expert stride
    const int stride_bk,  // B K stride
    const int stride_bn,  // B N stride
    const int stride_cm,  // C row stride
    const int stride_cn   // C column stride
) {
    // Map program id to output block
    const int pid = blockIdx.x;
    const int num_tokens_post_padded = *num_tokens_post_padded_ptr;
    const int num_pid_m = CEILDIV(num_tokens_post_padded, BLOCK_M);
    const int num_pid_n = CEILDIV(N, BLOCK_N);

    // Grouped ordering for L2 cache reuse
    const int GROUP_SIZE_M = 8;
    const int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_SIZE_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
    const int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    // Early exit for padding blocks
    if (pid_m * BLOCK_M >= num_tokens_post_padded) {
        return;
    }

    // Load expert id for this block
    const int expert_id = expert_ids[pid_m];
    if (expert_id == -1) {
        // Invalid expert - write zeros
        const int offs_m_start = pid_m * BLOCK_M;
        const int offs_n_start = pid_n * BLOCK_N;
        for (int m = threadIdx.y; m < BLOCK_M; m += blockDim.y) {
            int token_idx = offs_m_start + m;
            if (token_idx < num_tokens_post_padded) {
                int sorted_id = sorted_token_ids[token_idx];
                if (sorted_id < num_valid_tokens) {
                    for (int n = threadIdx.x; n < BLOCK_N; n += blockDim.x) {
                        int col = offs_n_start + n;
                        if (col < N) {
                            C[sorted_id * stride_cm + col * stride_cn] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }
        return;
    }

    // Shared memory for tiles
    extern __shared__ char smem[];
    __nv_bfloat16* A_tile = (__nv_bfloat16*)smem;
    __nv_bfloat16* B_tile = A_tile + BLOCK_M * BLOCK_K;

    // Accumulator
    float acc[BLOCK_M / 32][BLOCK_N / 32];
    for (int i = 0; i < BLOCK_M / 32; ++i) {
        for (int j = 0; j < BLOCK_N / 32; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Main GEMM loop over K dimension
    const int offs_m = pid_m * BLOCK_M;
    const int offs_n = pid_n * BLOCK_N;

    for (int k_block = 0; k_block < CEILDIV(K, BLOCK_K); ++k_block) {
        const int offs_k = k_block * BLOCK_K;

        // Load A tile (gather by sorted_token_ids)
        for (int m = threadIdx.y; m < BLOCK_M; m += blockDim.y) {
            int token_pos = offs_m + m;
            int token_idx = (token_pos < num_tokens_post_padded)
                ? sorted_token_ids[token_pos]
                : num_valid_tokens;
            int original_token = token_idx / top_k;

            for (int k = threadIdx.x; k < BLOCK_K; k += blockDim.x) {
                int k_idx = offs_k + k;
                float val = 0.0f;
                if (token_idx < num_valid_tokens && k_idx < K) {
                    val = __bfloat162float(A[original_token * stride_am + k_idx]);
                }
                A_tile[m * BLOCK_K + k] = __float2bfloat16(val);
            }
        }

        // Load B tile
        for (int k = threadIdx.y; k < BLOCK_K; k += blockDim.y) {
            int k_idx = offs_k + k;
            for (int n = threadIdx.x; n < BLOCK_N; n += blockDim.x) {
                int n_idx = offs_n + n;
                float val = 0.0f;
                if (k_idx < K && n_idx < N) {
                    val = __bfloat162float(B[expert_id * stride_be + k_idx * stride_bk + n_idx * stride_bn]);
                }
                B_tile[k * BLOCK_N + n] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        // Compute tile GEMM
        for (int m = 0; m < BLOCK_M; ++m) {
            for (int n = 0; n < BLOCK_N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < BLOCK_K; ++k) {
                    sum += __bfloat162float(A_tile[m * BLOCK_K + k]) *
                           __bfloat162float(B_tile[k * BLOCK_N + n]);
                }
                if (m / 32 < BLOCK_M / 32 && n / 32 < BLOCK_N / 32) {
                    acc[m / 32][n / 32] += sum;
                }
            }
        }
        __syncthreads();
    }

    // Apply routing weights and store results
    for (int m = threadIdx.y; m < BLOCK_M; m += blockDim.y) {
        int token_pos = offs_m + m;
        if (token_pos >= num_tokens_post_padded) continue;

        int token_idx = sorted_token_ids[token_pos];
        if (token_idx >= num_valid_tokens) continue;

        float weight = 1.0f;
        if (MUL_ROUTED_WEIGHT) {
            weight = topk_weights[token_idx];
        }

        for (int n = threadIdx.x; n < BLOCK_N; n += blockDim.x) {
            int n_idx = offs_n + n;
            if (n_idx < N) {
                float val = acc[m / 32][n / 32] * weight;
                C[token_idx * stride_cm + n_idx * stride_cn] = __float2bfloat16(val);
            }
        }
    }
}

// Simplified GEMM for small batches - one thread block per output row
template <bool MUL_ROUTED_WEIGHT>
__global__ void fused_moe_gemm_simple_kernel(
    const __nv_bfloat16* __restrict__ A,     // [num_tokens, K]
    const __nv_bfloat16* __restrict__ B,     // [num_experts, N, K]
    __nv_bfloat16* __restrict__ C,           // [num_tokens_padded * top_k, N]
    const float* __restrict__ topk_weights,  // [num_tokens * top_k]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    const int N,
    const int K,
    const int num_valid_tokens,
    const int top_k,
    const int num_experts
) {
    const int out_row = blockIdx.x;  // Output row (in sorted order)
    const int num_tokens_post_padded = *num_tokens_post_padded_ptr;

    if (out_row >= num_tokens_post_padded) return;

    // Get sorted token index
    const int token_idx = sorted_token_ids[out_row];
    if (token_idx >= num_valid_tokens) return;

    // Get expert for this block
    const int block_idx = out_row / 64;  // Assuming block_size=64
    const int expert_id = expert_ids[block_idx];
    if (expert_id < 0 || expert_id >= num_experts) return;

    // Original token index (before top_k expansion)
    const int original_token = token_idx / top_k;

    // Get routing weight
    float weight = 1.0f;
    if (MUL_ROUTED_WEIGHT) {
        weight = topk_weights[token_idx];
    }

    // Compute output for this row
    extern __shared__ float reduce_smem[];
    const int tid = threadIdx.x;
    const int NUM_THREADS = blockDim.x;

    for (int n = 0; n < N; ++n) {
        // Dot product A[original_token, :] @ B[expert_id, n, :]
        float sum = 0.0f;
        for (int k = tid; k < K; k += NUM_THREADS) {
            float a_val = __bfloat162float(A[original_token * K + k]);
            float b_val = __bfloat162float(B[expert_id * N * K + n * K + k]);
            sum += a_val * b_val;
        }

        // Block reduction
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = NUM_THREADS / WARP_SIZE;

        sum = block_reduce_sum(sum, reduce_smem, warp_id, lane_id, num_warps);

        // Thread 0 writes result
        if (tid == 0) {
            C[token_idx * N + n] = __float2bfloat16(sum * weight);
        }
        __syncthreads();
    }
}

// Gate + Up projection fused with SiLU activation.
// Computes: hidden = SiLU(gate_proj(x)) * up_proj(x)
//
// w13_weight layout: [num_experts, 2*intermediate_size, hidden_size]
// First half is gate_proj, second half is up_proj
__global__ void fused_moe_gate_up_silu_kernel(
    const __nv_bfloat16* __restrict__ A,        // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ W13,      // [num_experts, 2*intermediate, hidden]
    __nv_bfloat16* __restrict__ hidden,         // [num_tokens_padded * top_k, intermediate]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    const int hidden_size,
    const int intermediate_size,
    const int num_valid_tokens,
    const int top_k,
    const int block_size
) {
    const int out_row = blockIdx.x;
    const int num_tokens_post_padded = *num_tokens_post_padded_ptr;

    if (out_row >= num_tokens_post_padded) return;

    const int token_idx = sorted_token_ids[out_row];
    if (token_idx >= num_valid_tokens) return;

    const int block_idx = out_row / block_size;
    const int expert_id = expert_ids[block_idx];
    if (expert_id < 0) return;

    const int original_token = token_idx / top_k;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];

    // For each output dimension
    for (int i = 0; i < intermediate_size; ++i) {
        // Compute gate projection
        float gate_sum = 0.0f;
        for (int k = tid; k < hidden_size; k += blockDim.x) {
            float a_val = __bfloat162float(A[original_token * hidden_size + k]);
            float w_val = __bfloat162float(W13[expert_id * 2 * intermediate_size * hidden_size +
                                               i * hidden_size + k]);
            gate_sum += a_val * w_val;
        }

        // Reduce gate
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        gate_sum = block_reduce_sum(gate_sum, smem, warp_id, lane_id, blockDim.x / WARP_SIZE);
        float gate_val = smem[0];
        __syncthreads();

        // Apply SiLU to gate
        gate_val = silu(gate_val);

        // Compute up projection
        float up_sum = 0.0f;
        for (int k = tid; k < hidden_size; k += blockDim.x) {
            float a_val = __bfloat162float(A[original_token * hidden_size + k]);
            float w_val = __bfloat162float(W13[expert_id * 2 * intermediate_size * hidden_size +
                                               (intermediate_size + i) * hidden_size + k]);
            up_sum += a_val * w_val;
        }

        // Reduce up
        up_sum = block_reduce_sum(up_sum, smem, warp_id, lane_id, blockDim.x / WARP_SIZE);
        float up_val = smem[0];
        __syncthreads();

        // Write gate * up
        if (tid == 0) {
            hidden[token_idx * intermediate_size + i] = __float2bfloat16(gate_val * up_val);
        }
        __syncthreads();
    }
}

// Down projection with routing weight application.
// Computes: output[token] = routing_weight * down_proj(hidden)
__global__ void fused_moe_down_reduce_kernel(
    const __nv_bfloat16* __restrict__ hidden,    // [num_tokens_padded * top_k, intermediate]
    const __nv_bfloat16* __restrict__ W2,        // [num_experts, hidden_size, intermediate]
    __nv_bfloat16* __restrict__ output,          // [num_tokens, hidden_size]
    const float* __restrict__ topk_weights,      // [num_tokens * top_k]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    const int hidden_size,
    const int intermediate_size,
    const int num_valid_tokens,
    const int num_tokens,
    const int top_k,
    const int block_size
) {
    const int token_out = blockIdx.x;  // Original token index
    if (token_out >= num_tokens) return;

    const int tid = threadIdx.x;
    extern __shared__ float smem[];

    // For each output dimension
    for (int h = 0; h < hidden_size; ++h) {
        float total = 0.0f;

        // Sum over top_k experts for this token
        for (int k = 0; k < top_k; ++k) {
            int token_idx = token_out * top_k + k;

            // Find sorted position for this token_idx
            // (In practice, we'd have inverse mapping, here simplified)
            const int num_tokens_post_padded = *num_tokens_post_padded_ptr;

            for (int sorted_pos = 0; sorted_pos < num_tokens_post_padded; ++sorted_pos) {
                if (sorted_token_ids[sorted_pos] == token_idx) {
                    int block_idx = sorted_pos / block_size;
                    int expert_id = expert_ids[block_idx];
                    if (expert_id < 0) continue;

                    // Compute down projection
                    float down_sum = 0.0f;
                    for (int i = tid; i < intermediate_size; i += blockDim.x) {
                        float h_val = __bfloat162float(hidden[token_idx * intermediate_size + i]);
                        float w_val = __bfloat162float(W2[expert_id * hidden_size * intermediate_size +
                                                          h * intermediate_size + i]);
                        down_sum += h_val * w_val;
                    }

                    // Reduce
                    const int warp_id = tid / WARP_SIZE;
                    const int lane_id = tid % WARP_SIZE;
                    down_sum = block_reduce_sum(down_sum, smem, warp_id, lane_id, blockDim.x / WARP_SIZE);

                    if (tid == 0) {
                        total += smem[0] * topk_weights[token_idx];
                    }
                    __syncthreads();
                    break;
                }
            }
        }

        // Write output
        if (tid == 0) {
            output[token_out * hidden_size + h] = __float2bfloat16(total);
        }
        __syncthreads();
    }
}

// Instantiate kernels with common block sizes
template __global__ void fused_moe_gemm_kernel<64, 64, 32, true>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, int, int, int, int, int, int);

template __global__ void fused_moe_gemm_kernel<64, 64, 32, false>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, int, int, int, int, int, int);

template __global__ void fused_moe_gemm_simple_kernel<true>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, int);

template __global__ void fused_moe_gemm_simple_kernel<false>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    const float*, const int32_t*, const int32_t*, const int32_t*,
    int, int, int, int, int);
