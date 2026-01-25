// FP8 Quantization Kernels for Ada Lovelace+ GPUs (sm_89+)
//
// Supports:
// - Static quantization with pre-computed scales
// - Dynamic per-tensor quantization
// - Dynamic per-token quantization
//
// FP8 E4M3 format: 1 sign, 4 exponent, 3 mantissa bits
// Range: [-448, 448], precision: ~0.5%

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// Utility Functions
// ============================================================================

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level max reduction
__device__ float block_reduce_max(float val, float* smem) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            smem[0] = v;
        }
    }
    __syncthreads();

    return smem[0];
}

// FP8 E4M3 maximum representable value
__device__ __forceinline__ float fp8_e4m3_max() {
    return 448.0f;
}

// Convert float to FP8 E4M3 with saturation
__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8_e4m3(float val) {
    return __nv_fp8_e4m3(val);
}

// Convert FP8 E4M3 to float
__device__ __forceinline__ float fp8_e4m3_to_float(__nv_fp8_e4m3 val) {
    return float(val);
}

// ============================================================================
// Static Quantization: BF16 -> FP8 with pre-computed scale
// ============================================================================

// Grid: (num_tokens,)
// Block: (BLOCK_SIZE,)
// Each block handles one token (row)
extern "C" __global__ void fp8_static_quant_bf16(
    __nv_fp8_e4m3* __restrict__ out,       // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ in,  // [num_tokens, hidden_size]
    const float* __restrict__ scale,        // [1] or [num_tokens]
    const int hidden_size,
    const bool per_token_scale              // if true, scale is [num_tokens]
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* token_in = in + token_idx * hidden_size;
    __nv_fp8_e4m3* token_out = out + token_idx * hidden_size;

    // Get scale for this token
    const float inv_scale = 1.0f / (per_token_scale ? scale[token_idx] : scale[0]);

    // Process elements with stride
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(token_in[i]);
        float scaled = val * inv_scale;
        token_out[i] = float_to_fp8_e4m3(scaled);
    }
}

// ============================================================================
// Dynamic Per-Tensor Quantization
// ============================================================================

// Phase 1: Compute absmax across entire tensor
// Grid: (num_tokens,)
// Block: (BLOCK_SIZE,)
extern "C" __global__ void fp8_compute_absmax_bf16(
    float* __restrict__ absmax,             // [1] output (atomicMax)
    const __nv_bfloat16* __restrict__ in,   // [num_tokens, hidden_size]
    const int hidden_size
) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* token_in = in + token_idx * hidden_size;

    // Thread-local max
    float thread_max = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = fabsf(__bfloat162float(token_in[i]));
        thread_max = fmaxf(thread_max, val);
    }

    // Block-level reduction
    float block_max = block_reduce_max(thread_max, smem);

    // Atomic update global max
    if (tid == 0) {
        atomicMax((int*)absmax, __float_as_int(block_max));
    }
}

// Phase 2: Quantize with computed scale
// Uses same kernel as static but with dynamically computed scale

// ============================================================================
// Dynamic Per-Token Quantization
// ============================================================================

// Compute per-token scales only (for two-pass quantization)
// Grid: (num_tokens,)
// Block: (BLOCK_SIZE,)
extern "C" __global__ void fp8_compute_scales_bf16(
    float* __restrict__ scales,              // [num_tokens] output scales
    const __nv_bfloat16* __restrict__ in,   // [num_tokens, hidden_size]
    const int hidden_size
) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* token_in = in + token_idx * hidden_size;

    // Compute per-token absmax
    float thread_max = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = fabsf(__bfloat162float(token_in[i]));
        thread_max = fmaxf(thread_max, val);
    }

    float absmax = block_reduce_max(thread_max, smem);

    // Compute scale: scale = absmax / fp8_max
    // Ensure minimum scale to avoid division by zero
    if (tid == 0) {
        float computed_scale = fmaxf(absmax / fp8_e4m3_max(), 1e-12f);
        scales[token_idx] = computed_scale;
    }
}

// Combined kernel: compute scale and quantize per token
// Grid: (num_tokens,)
// Block: (BLOCK_SIZE,)
extern "C" __global__ void fp8_dynamic_per_token_quant_bf16(
    __nv_fp8_e4m3* __restrict__ out,        // [num_tokens, hidden_size]
    float* __restrict__ scales,              // [num_tokens] output scales
    const __nv_bfloat16* __restrict__ in,   // [num_tokens, hidden_size]
    const int hidden_size
) {
    extern __shared__ float smem[];

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* token_in = in + token_idx * hidden_size;
    __nv_fp8_e4m3* token_out = out + token_idx * hidden_size;

    // ─── Phase 1: Compute per-token absmax ───────────────────────────────────
    float thread_max = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = fabsf(__bfloat162float(token_in[i]));
        thread_max = fmaxf(thread_max, val);
    }

    float absmax = block_reduce_max(thread_max, smem);

    // Compute scale: scale = absmax / fp8_max
    // Ensure minimum scale to avoid division by zero
    __shared__ float token_scale;
    if (tid == 0) {
        float computed_scale = fmaxf(absmax / fp8_e4m3_max(), 1e-12f);
        token_scale = computed_scale;
        scales[token_idx] = computed_scale;
    }
    __syncthreads();

    // ─── Phase 2: Quantize with computed scale ───────────────────────────────
    const float inv_scale = 1.0f / token_scale;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(token_in[i]);
        float scaled = val * inv_scale;
        token_out[i] = float_to_fp8_e4m3(scaled);
    }
}

// ============================================================================
// FP8 Dequantization: FP8 -> BF16 with scale
// ============================================================================

// Grid: (num_tokens,)
// Block: (BLOCK_SIZE,)
extern "C" __global__ void fp8_dequant_bf16(
    __nv_bfloat16* __restrict__ out,        // [num_tokens, hidden_size]
    const __nv_fp8_e4m3* __restrict__ in,   // [num_tokens, hidden_size]
    const float* __restrict__ scale,         // [1] or [num_tokens]
    const int hidden_size,
    const bool per_token_scale
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_fp8_e4m3* token_in = in + token_idx * hidden_size;
    __nv_bfloat16* token_out = out + token_idx * hidden_size;

    // Get scale for this token
    const float token_scale = per_token_scale ? scale[token_idx] : scale[0];

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = fp8_e4m3_to_float(token_in[i]);
        float dequantized = val * token_scale;
        token_out[i] = __float2bfloat16(dequantized);
    }
}

// ============================================================================
// Vectorized versions (4x throughput with float4)
// ============================================================================

// Vectorized static quantization (processes 4 elements per thread)
extern "C" __global__ void fp8_static_quant_bf16_vec4(
    __nv_fp8_e4m3* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ scale,
    const int hidden_size,
    const bool per_token_scale
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Ensure hidden_size is divisible by 4 for vectorized access
    const int vec_size = hidden_size / 4;

    const float2* token_in = reinterpret_cast<const float2*>(in + token_idx * hidden_size);
    uint32_t* token_out = reinterpret_cast<uint32_t*>(out + token_idx * hidden_size);

    const float inv_scale = 1.0f / (per_token_scale ? scale[token_idx] : scale[0]);

    for (int i = tid; i < vec_size; i += blockDim.x) {
        // Load 4 bf16 values (packed as 2 float2)
        float2 packed = token_in[i];

        // Unpack bf16 to float
        __nv_bfloat162 bf16_pair1, bf16_pair2;
        memcpy(&bf16_pair1, &packed.x, sizeof(float));
        memcpy(&bf16_pair2, &packed.y, sizeof(float));

        float v0 = __bfloat162float(bf16_pair1.x);
        float v1 = __bfloat162float(bf16_pair1.y);
        float v2 = __bfloat162float(bf16_pair2.x);
        float v3 = __bfloat162float(bf16_pair2.y);

        // Scale and convert to FP8
        __nv_fp8_e4m3 fp8_0 = float_to_fp8_e4m3(v0 * inv_scale);
        __nv_fp8_e4m3 fp8_1 = float_to_fp8_e4m3(v1 * inv_scale);
        __nv_fp8_e4m3 fp8_2 = float_to_fp8_e4m3(v2 * inv_scale);
        __nv_fp8_e4m3 fp8_3 = float_to_fp8_e4m3(v3 * inv_scale);

        // Pack 4 FP8 values into uint32
        uint32_t packed_out =
            (uint32_t(*reinterpret_cast<uint8_t*>(&fp8_0))) |
            (uint32_t(*reinterpret_cast<uint8_t*>(&fp8_1)) << 8) |
            (uint32_t(*reinterpret_cast<uint8_t*>(&fp8_2)) << 16) |
            (uint32_t(*reinterpret_cast<uint8_t*>(&fp8_3)) << 24);

        token_out[i] = packed_out;
    }
}

// Vectorized dequantization
extern "C" __global__ void fp8_dequant_bf16_vec4(
    __nv_bfloat16* __restrict__ out,
    const __nv_fp8_e4m3* __restrict__ in,
    const float* __restrict__ scale,
    const int hidden_size,
    const bool per_token_scale
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int vec_size = hidden_size / 4;

    const uint32_t* token_in = reinterpret_cast<const uint32_t*>(in + token_idx * hidden_size);
    float2* token_out = reinterpret_cast<float2*>(out + token_idx * hidden_size);

    const float token_scale = per_token_scale ? scale[token_idx] : scale[0];

    for (int i = tid; i < vec_size; i += blockDim.x) {
        // Load 4 packed FP8 values
        uint32_t packed_in = token_in[i];

        // Unpack to individual FP8 values
        __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
        uint8_t byte0 = packed_in & 0xFF;
        uint8_t byte1 = (packed_in >> 8) & 0xFF;
        uint8_t byte2 = (packed_in >> 16) & 0xFF;
        uint8_t byte3 = (packed_in >> 24) & 0xFF;

        memcpy(&fp8_0, &byte0, 1);
        memcpy(&fp8_1, &byte1, 1);
        memcpy(&fp8_2, &byte2, 1);
        memcpy(&fp8_3, &byte3, 1);

        // Convert to float and apply scale
        float v0 = fp8_e4m3_to_float(fp8_0) * token_scale;
        float v1 = fp8_e4m3_to_float(fp8_1) * token_scale;
        float v2 = fp8_e4m3_to_float(fp8_2) * token_scale;
        float v3 = fp8_e4m3_to_float(fp8_3) * token_scale;

        // Pack back to bf16
        __nv_bfloat162 bf16_pair1 = make_bfloat162(__float2bfloat16(v0), __float2bfloat16(v1));
        __nv_bfloat162 bf16_pair2 = make_bfloat162(__float2bfloat16(v2), __float2bfloat16(v3));

        float2 packed_out;
        memcpy(&packed_out.x, &bf16_pair1, sizeof(float));
        memcpy(&packed_out.y, &bf16_pair2, sizeof(float));

        token_out[i] = packed_out;
    }
}
