// RMSNorm and Fused Add-RMSNorm CUDA kernels.
// Ported from vLLM's layernorm_kernels.cu with vectorized access.
//
// RMSNorm: output[i] = (input[i] / sqrt(mean(input^2) + eps)) * weight[i]
// Fused Add-RMSNorm: residual += input; output = rmsnorm(residual) * weight
//
// Grid: (num_tokens,)
// Block: (min(hidden_size / VEC_SIZE, 1024),)

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

    // Intra-warp reduction
    val = warp_reduce_sum(val);

    // Shared memory for cross-warp reduction (max 32 warps = 1024 threads)
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
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

// ==========================================================================
// RMSNorm kernel — BF16 input/output, vectorized (4 elements per thread)
// ==========================================================================
//
// Each thread block processes one token (row of the hidden dimension).
// Phase 1: compute sum of squares via vectorized loads
// Phase 2: normalize and scale via vectorized stores
extern "C" __global__ void rms_norm_bf16(
    __nv_bfloat16* __restrict__ out,           // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* input_row = input + token_idx * hidden_size;
    __nv_bfloat16* out_row = out + token_idx * hidden_size;

    // Phase 1: compute variance = sum(x^2) / hidden_size
    float variance = 0.0f;

    // Vectorized path: 4 bf16 elements per load (8 bytes = 64 bits)
    // This is safe because hidden_size is always a multiple of at least 64
    // for transformer models.
    const int vec_size = 4;
    const int num_vecs = hidden_size / vec_size;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        // Load 4 bf16 values
        const int base = i * vec_size;
        float x0 = __bfloat162float(input_row[base + 0]);
        float x1 = __bfloat162float(input_row[base + 1]);
        float x2 = __bfloat162float(input_row[base + 2]);
        float x3 = __bfloat162float(input_row[base + 3]);
        variance += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }

    // Handle remainder (hidden_size not divisible by 4)
    for (int i = num_vecs * vec_size + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __bfloat162float(input_row[i]);
        variance += x * x;
    }

    // Block reduce to get total sum of squares
    variance = block_reduce_sum(variance);

    // Compute normalization factor: 1 / sqrt(variance / hidden_size + epsilon)
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // Phase 2: normalize and scale
    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        const int base = i * vec_size;
        float x0 = __bfloat162float(input_row[base + 0]);
        float x1 = __bfloat162float(input_row[base + 1]);
        float x2 = __bfloat162float(input_row[base + 2]);
        float x3 = __bfloat162float(input_row[base + 3]);
        float w0 = __bfloat162float(weight[base + 0]);
        float w1 = __bfloat162float(weight[base + 1]);
        float w2 = __bfloat162float(weight[base + 2]);
        float w3 = __bfloat162float(weight[base + 3]);
        out_row[base + 0] = __float2bfloat16(x0 * inv_rms * w0);
        out_row[base + 1] = __float2bfloat16(x1 * inv_rms * w1);
        out_row[base + 2] = __float2bfloat16(x2 * inv_rms * w2);
        out_row[base + 3] = __float2bfloat16(x3 * inv_rms * w3);
    }

    for (int i = num_vecs * vec_size + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __bfloat162float(input_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(x * inv_rms * w);
    }
}

// ==========================================================================
// RMSNorm kernel — FP16 input/output
// ==========================================================================
extern "C" __global__ void rms_norm_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const float epsilon,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const __half* input_row = input + token_idx * hidden_size;
    __half* out_row = out + token_idx * hidden_size;

    float variance = 0.0f;
    const int vec_size = 4;
    const int num_vecs = hidden_size / vec_size;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        const int base = i * vec_size;
        float x0 = __half2float(input_row[base + 0]);
        float x1 = __half2float(input_row[base + 1]);
        float x2 = __half2float(input_row[base + 2]);
        float x3 = __half2float(input_row[base + 3]);
        variance += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    }
    for (int i = num_vecs * vec_size + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __half2float(input_row[i]);
        variance += x * x;
    }

    variance = block_reduce_sum(variance);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        const int base = i * vec_size;
        float x0 = __half2float(input_row[base + 0]);
        float x1 = __half2float(input_row[base + 1]);
        float x2 = __half2float(input_row[base + 2]);
        float x3 = __half2float(input_row[base + 3]);
        float w0 = __half2float(weight[base + 0]);
        float w1 = __half2float(weight[base + 1]);
        float w2 = __half2float(weight[base + 2]);
        float w3 = __half2float(weight[base + 3]);
        out_row[base + 0] = __float2half(x0 * inv_rms * w0);
        out_row[base + 1] = __float2half(x1 * inv_rms * w1);
        out_row[base + 2] = __float2half(x2 * inv_rms * w2);
        out_row[base + 3] = __float2half(x3 * inv_rms * w3);
    }
    for (int i = num_vecs * vec_size + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __half2float(input_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(x * inv_rms * w);
    }
}

// ==========================================================================
// RMSNorm kernel — F32 input/output
// ==========================================================================
extern "C" __global__ void rms_norm_f32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float epsilon,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const float* input_row = input + token_idx * hidden_size;
    float* out_row = out + token_idx * hidden_size;

    float variance = 0.0f;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = input_row[i];
        variance += x * x;
    }

    variance = block_reduce_sum(variance);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = input_row[i] * inv_rms * weight[i];
    }
}

// ==========================================================================
// Fused Add-RMSNorm — BF16
// residual[i] += input[i]; output[i] = rmsnorm(residual[i]) * weight[i]
// Saves one full pass over the data compared to separate add + rmsnorm.
// ==========================================================================
extern "C" __global__ void fused_add_rms_norm_bf16(
    __nv_bfloat16* __restrict__ output,      // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ input,  // [num_tokens, hidden_size]
    __nv_bfloat16* __restrict__ residual,     // [num_tokens, hidden_size] (in-place updated)
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* input_row = input + token_idx * hidden_size;
    __nv_bfloat16* residual_row = residual + token_idx * hidden_size;
    __nv_bfloat16* out_row = output + token_idx * hidden_size;

    float variance = 0.0f;

    // Phase 1: residual += input, accumulate sum of squares
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = __bfloat162float(residual_row[i]) + __bfloat162float(input_row[i]);
        residual_row[i] = __float2bfloat16(r);
        variance += r * r;
    }

    variance = block_reduce_sum(variance);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
    }
    __syncthreads();
    const float inv_rms = s_inv_rms;

    // Phase 2: normalize from residual (already updated)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float r = __bfloat162float(residual_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(r * inv_rms * w);
    }
}
