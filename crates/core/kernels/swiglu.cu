// Fused SwiGLU Activation Kernel
//
// Computes: output = silu(gate) * up = gate * sigmoid(gate) * up
// in a single kernel pass, saving memory bandwidth by avoiding
// materialization of intermediate results.
//
// Supports BF16 and FP16 data types with vectorized memory access.
//
// Reference: vLLM csrc/activation_kernels.cu

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
// Always compute in float for numerical stability
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// Check 16-byte alignment for vectorized int4 access
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// ============================================================================
// Fused SwiGLU Kernel - BF16
// ============================================================================
//
// Grid: (num_tokens,)
// Block: (min(hidden_size, 1024),)
//
// For separate gate/up tensors:
//   gate: [..., hidden_size]
//   up:   [..., hidden_size]
//   out:  [..., hidden_size]
//   output[i] = silu(gate[i]) * up[i]

extern "C" __global__ void fused_swiglu_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 8;  // 16 bytes / 2 bytes per bf16 = 8 elements
    const int64_t token_idx = blockIdx.x;

    const __nv_bfloat16* gate_ptr = gate + token_idx * hidden_size;
    const __nv_bfloat16* up_ptr = up + token_idx * hidden_size;
    __nv_bfloat16* out_ptr = out + token_idx * hidden_size;

    // Check alignment for 128-bit vectorized access
    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        // Fast path: 128-bit vectorized loop
        const int4* gate_vec = reinterpret_cast<const int4*>(gate_ptr);
        const int4* up_vec = reinterpret_cast<const int4*>(up_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            int4 g = gate_vec[i];
            int4 u = up_vec[i];
            int4 r;

            __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&g);
            __nv_bfloat16* up_data = reinterpret_cast<__nv_bfloat16*>(&u);
            __nv_bfloat16* rp = reinterpret_cast<__nv_bfloat16*>(&r);

            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float g_val = __bfloat162float(gp[j]);
                float u_val = __bfloat162float(up_data[j]);
                float result = silu_f32(g_val) * u_val;
                rp[j] = __float2bfloat16(result);
            }

            out_vec[i] = r;
        }

        // Scalar cleanup for remaining elements
        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __bfloat162float(gate_ptr[i]);
            float u_val = __bfloat162float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2bfloat16(result);
        }
    } else {
        // Scalar fallback for unaligned data or small hidden_size
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __bfloat162float(gate_ptr[i]);
            float u_val = __bfloat162float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2bfloat16(result);
        }
    }
}

// ============================================================================
// Fused SwiGLU Kernel - FP16
// ============================================================================

extern "C" __global__ void fused_swiglu_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 8;  // 16 bytes / 2 bytes per fp16 = 8 elements
    const int64_t token_idx = blockIdx.x;

    const __half* gate_ptr = gate + token_idx * hidden_size;
    const __half* up_ptr = up + token_idx * hidden_size;
    __half* out_ptr = out + token_idx * hidden_size;

    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        const int4* gate_vec = reinterpret_cast<const int4*>(gate_ptr);
        const int4* up_vec = reinterpret_cast<const int4*>(up_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            int4 g = gate_vec[i];
            int4 u = up_vec[i];
            int4 r;

            __half* gp = reinterpret_cast<__half*>(&g);
            __half* up_data = reinterpret_cast<__half*>(&u);
            __half* rp = reinterpret_cast<__half*>(&r);

            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float g_val = __half2float(gp[j]);
                float u_val = __half2float(up_data[j]);
                float result = silu_f32(g_val) * u_val;
                rp[j] = __float2half(result);
            }

            out_vec[i] = r;
        }

        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __half2float(gate_ptr[i]);
            float u_val = __half2float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2half(result);
        }
    } else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __half2float(gate_ptr[i]);
            float u_val = __half2float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2half(result);
        }
    }
}

// ============================================================================
// Fused SwiGLU Kernel - FP32
// ============================================================================

extern "C" __global__ void fused_swiglu_fp32(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 4;  // 16 bytes / 4 bytes per float = 4 elements
    const int64_t token_idx = blockIdx.x;

    const float* gate_ptr = gate + token_idx * hidden_size;
    const float* up_ptr = up + token_idx * hidden_size;
    float* out_ptr = out + token_idx * hidden_size;

    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        const float4* gate_vec = reinterpret_cast<const float4*>(gate_ptr);
        const float4* up_vec = reinterpret_cast<const float4*>(up_ptr);
        float4* out_vec = reinterpret_cast<float4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            float4 g = gate_vec[i];
            float4 u = up_vec[i];
            float4 r;

            r.x = silu_f32(g.x) * u.x;
            r.y = silu_f32(g.y) * u.y;
            r.z = silu_f32(g.z) * u.z;
            r.w = silu_f32(g.w) * u.w;

            out_vec[i] = r;
        }

        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = gate_ptr[i];
            float u_val = up_ptr[i];
            out_ptr[i] = silu_f32(g_val) * u_val;
        }
    } else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = gate_ptr[i];
            float u_val = up_ptr[i];
            out_ptr[i] = silu_f32(g_val) * u_val;
        }
    }
}

// ============================================================================
// Fused SwiGLU Kernel - Packed input (concatenated gate/up)
// ============================================================================
//
// For packed gate/up tensor (common in some model formats):
//   input: [..., 2 * hidden_size] where input = [gate, up] concatenated
//   out:   [..., hidden_size]
//   output[i] = silu(input[i]) * input[hidden_size + i]

extern "C" __global__ void fused_swiglu_packed_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 8;
    const int64_t token_idx = blockIdx.x;

    const __nv_bfloat16* gate_ptr = input + token_idx * 2 * hidden_size;
    const __nv_bfloat16* up_ptr = gate_ptr + hidden_size;
    __nv_bfloat16* out_ptr = out + token_idx * hidden_size;

    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        const int4* gate_vec = reinterpret_cast<const int4*>(gate_ptr);
        const int4* up_vec = reinterpret_cast<const int4*>(up_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            int4 g = gate_vec[i];
            int4 u = up_vec[i];
            int4 r;

            __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&g);
            __nv_bfloat16* up_data = reinterpret_cast<__nv_bfloat16*>(&u);
            __nv_bfloat16* rp = reinterpret_cast<__nv_bfloat16*>(&r);

            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float g_val = __bfloat162float(gp[j]);
                float u_val = __bfloat162float(up_data[j]);
                float result = silu_f32(g_val) * u_val;
                rp[j] = __float2bfloat16(result);
            }

            out_vec[i] = r;
        }

        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __bfloat162float(gate_ptr[i]);
            float u_val = __bfloat162float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2bfloat16(result);
        }
    } else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __bfloat162float(gate_ptr[i]);
            float u_val = __bfloat162float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2bfloat16(result);
        }
    }
}

extern "C" __global__ void fused_swiglu_packed_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 8;
    const int64_t token_idx = blockIdx.x;

    const __half* gate_ptr = input + token_idx * 2 * hidden_size;
    const __half* up_ptr = gate_ptr + hidden_size;
    __half* out_ptr = out + token_idx * hidden_size;

    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        const int4* gate_vec = reinterpret_cast<const int4*>(gate_ptr);
        const int4* up_vec = reinterpret_cast<const int4*>(up_ptr);
        int4* out_vec = reinterpret_cast<int4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            int4 g = gate_vec[i];
            int4 u = up_vec[i];
            int4 r;

            __half* gp = reinterpret_cast<__half*>(&g);
            __half* up_data = reinterpret_cast<__half*>(&u);
            __half* rp = reinterpret_cast<__half*>(&r);

            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float g_val = __half2float(gp[j]);
                float u_val = __half2float(up_data[j]);
                float result = silu_f32(g_val) * u_val;
                rp[j] = __float2half(result);
            }

            out_vec[i] = r;
        }

        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __half2float(gate_ptr[i]);
            float u_val = __half2float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2half(result);
        }
    } else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = __half2float(gate_ptr[i]);
            float u_val = __half2float(up_ptr[i]);
            float result = silu_f32(g_val) * u_val;
            out_ptr[i] = __float2half(result);
        }
    }
}

extern "C" __global__ void fused_swiglu_packed_fp32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const int hidden_size
) {
    constexpr int VEC_SIZE = 4;
    const int64_t token_idx = blockIdx.x;

    const float* gate_ptr = input + token_idx * 2 * hidden_size;
    const float* up_ptr = gate_ptr + hidden_size;
    float* out_ptr = out + token_idx * hidden_size;

    const bool aligned = is_16byte_aligned(gate_ptr) &&
                         is_16byte_aligned(up_ptr) &&
                         is_16byte_aligned(out_ptr);

    if (aligned && hidden_size >= VEC_SIZE) {
        const float4* gate_vec = reinterpret_cast<const float4*>(gate_ptr);
        const float4* up_vec = reinterpret_cast<const float4*>(up_ptr);
        float4* out_vec = reinterpret_cast<float4*>(out_ptr);
        const int num_vecs = hidden_size / VEC_SIZE;
        const int vec_end = num_vecs * VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            float4 g = gate_vec[i];
            float4 u = up_vec[i];
            float4 r;

            r.x = silu_f32(g.x) * u.x;
            r.y = silu_f32(g.y) * u.y;
            r.z = silu_f32(g.z) * u.z;
            r.w = silu_f32(g.w) * u.w;

            out_vec[i] = r;
        }

        for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = gate_ptr[i];
            float u_val = up_ptr[i];
            out_ptr[i] = silu_f32(g_val) * u_val;
        }
    } else {
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float g_val = gate_ptr[i];
            float u_val = up_ptr[i];
            out_ptr[i] = silu_f32(g_val) * u_val;
        }
    }
}
