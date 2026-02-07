// Fused LayerNorm/RMSNorm + FP8 Quantization kernel.
//
// Combines normalization and quantization in a single kernel pass to avoid
// an extra global memory round-trip. Critical for the W8A8 FP8 pipeline.
//
// Variants:
// - fused_rmsnorm_fp8: RMSNorm(x) -> FP8 quantize (one pass)
// - fused_layernorm_fp8: LayerNorm(x) -> FP8 quantize (one pass)
//
// Both compute:
// 1. Reduction across hidden dim for stats (variance/mean)
// 2. Normalize in-place
// 3. Quantize to FP8 with per-row dynamic scaling
//
// This saves ~40% memory bandwidth vs separate norm + quantize for typical LLM layers.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// FP8 E4M3 max representable value
#define FP8_E4M3_MAX 448.0f

// ============================================================================
// Fused RMSNorm + FP8 Quantization
// ============================================================================
//
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
// Then quantize to FP8 with per-row dynamic scaling.
//
// Grid: (num_rows,)
// Block: (block_size,) — typically 256 or 512
extern "C" __global__ void fused_rmsnorm_fp8(
    __nv_fp8_e4m3* __restrict__ out,          // [num_rows, hidden_dim] FP8
    float* __restrict__ scales,                // [num_rows] per-row scale
    const __nv_bfloat16* __restrict__ input,   // [num_rows, hidden_dim] BF16
    const __nv_bfloat16* __restrict__ weight,  // [hidden_dim] BF16
    const int num_rows,
    const int hidden_dim,
    const float eps
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const int tid = threadIdx.x;
    const __nv_bfloat16* row_in = input + row * hidden_dim;
    __nv_fp8_e4m3* row_out = out + row * hidden_dim;

    // Phase 1: Compute sum of squares for RMS
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    if (lane_id == 0) {
        smem[warp_id] = sum_sq;
    }
    __syncthreads();

    // Cross-warp reduction
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(smem[0] / (float)hidden_dim + eps);

    // Phase 2: Compute normalized values and find max for FP8 scaling
    // Store normalized values temporarily in shared memory
    float thread_max = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float normed = val * rms * w;

        thread_max = fmaxf(thread_max, fabsf(normed));
    }

    // Reduce to find row max
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }

    if (lane_id == 0) {
        smem[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) {
            smem[0] = val;
        }
    }
    __syncthreads();

    float row_max = smem[0];
    float row_scale = fmaxf(row_max / FP8_E4M3_MAX, 1e-12f);
    float inv_scale = 1.0f / row_scale;

    // Phase 3: Normalize and quantize in one pass
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float normed = val * rms * w;
        row_out[i] = __nv_fp8_e4m3(normed * inv_scale);
    }

    // Write per-row scale
    if (tid == 0) {
        scales[row] = row_scale;
    }
}

// ============================================================================
// Fused LayerNorm + FP8 Quantization
// ============================================================================
//
// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
// Then quantize to FP8 with per-row dynamic scaling.
//
// Grid: (num_rows,)
// Block: (block_size,) — typically 256 or 512
extern "C" __global__ void fused_layernorm_fp8(
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,     // [hidden_dim] or nullptr
    const int num_rows,
    const int hidden_dim,
    const float eps,
    const int has_bias
) {
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const int tid = threadIdx.x;
    const __nv_bfloat16* row_in = input + row * hidden_dim;
    __nv_fp8_e4m3* row_out = out + row * hidden_dim;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    // Phase 1: Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        sum += __bfloat162float(row_in[i]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (lane_id == 0) smem[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();

    float mean = smem[0] / (float)hidden_dim;

    // Phase 2: Compute variance
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = __bfloat162float(row_in[i]) - mean;
        sum_sq += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();

    float inv_std = rsqrtf(smem[0] / (float)hidden_dim + eps);

    // Phase 3: Normalize, apply weight+bias, find max for FP8 scaling
    float thread_max = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float normed = (val - mean) * inv_std * w;
        if (has_bias && bias != nullptr) {
            normed += __bfloat162float(bias[i]);
        }
        thread_max = fmaxf(thread_max, fabsf(normed));
    }

    // Reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
    }
    if (lane_id == 0) smem[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();

    float row_max = smem[0];
    float row_scale = fmaxf(row_max / FP8_E4M3_MAX, 1e-12f);
    float fp8_inv_scale = 1.0f / row_scale;

    // Phase 4: Normalize and quantize
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        float w = __bfloat162float(weight[i]);
        float normed = (val - mean) * inv_std * w;
        if (has_bias && bias != nullptr) {
            normed += __bfloat162float(bias[i]);
        }
        row_out[i] = __nv_fp8_e4m3(normed * fp8_inv_scale);
    }

    if (tid == 0) {
        scales[row] = row_scale;
    }
}
