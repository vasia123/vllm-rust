// Fused activation CUDA kernels: GELU, GeGLU, GELU-tanh, SiLU.
// Ported from vLLM's activation_kernels.cu.
//
// Two types of operations:
// 1. Element-wise activation: output[i] = act(input[i])
// 2. Gated activation: output[i] = act(gate[i]) * up[i]
//    where input = [gate, up] concatenated along last dim
//
// Grid: (num_tokens,)
// Block: (min(d, 1024),) where d = hidden_size (or hidden_size/2 for gated)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// ==========================================================================
// Activation function implementations (device-side)
// ==========================================================================

__device__ __forceinline__ float gelu_exact(float x) {
    // GELU with exact erf: x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));  // M_SQRT1_2
}

__device__ __forceinline__ float gelu_tanh(float x) {
    // GELU with tanh approximation (faster, used by some models)
    const float beta = 0.7978845608028654f;   // sqrt(2/pi)
    const float kappa = 0.044715f;
    float inner = beta * (x + kappa * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ==========================================================================
// Gated activation kernels: output = act(gate) * up
// Input layout: [num_tokens, 2 * d] where first d = gate, last d = up
// ==========================================================================

// GELU (exact) gated activation — BF16
extern "C" __global__ void gelu_and_mul_bf16(
    __nv_bfloat16* __restrict__ out,           // [num_tokens, d]
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, 2 * d]
    const int d
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* gate = input + token_idx * 2 * d;
    const __nv_bfloat16* up = gate + d;
    __nv_bfloat16* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out_ptr[i] = __float2bfloat16(gelu_exact(g) * u);
    }
}

// GELU (tanh approximation) gated activation — BF16
extern "C" __global__ void gelu_tanh_and_mul_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* gate = input + token_idx * 2 * d;
    const __nv_bfloat16* up = gate + d;
    __nv_bfloat16* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out_ptr[i] = __float2bfloat16(gelu_tanh(g) * u);
    }
}

// SiLU gated activation (same as SwiGLU but from concatenated input) — BF16
extern "C" __global__ void silu_and_mul_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* gate = input + token_idx * 2 * d;
    const __nv_bfloat16* up = gate + d;
    __nv_bfloat16* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        out_ptr[i] = __float2bfloat16(silu(g) * u);
    }
}

// ==========================================================================
// FP16 variants of gated activations
// ==========================================================================

extern "C" __global__ void gelu_and_mul_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __half* gate = input + token_idx * 2 * d;
    const __half* up = gate + d;
    __half* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        out_ptr[i] = __float2half(gelu_exact(g) * u);
    }
}

extern "C" __global__ void gelu_tanh_and_mul_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __half* gate = input + token_idx * 2 * d;
    const __half* up = gate + d;
    __half* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        out_ptr[i] = __float2half(gelu_tanh(g) * u);
    }
}

// ==========================================================================
// F32 variants
// ==========================================================================

extern "C" __global__ void gelu_and_mul_f32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const float* gate = input + token_idx * 2 * d;
    const float* up = gate + d;
    float* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = gelu_exact(gate[i]) * up[i];
    }
}

extern "C" __global__ void gelu_tanh_and_mul_f32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const float* gate = input + token_idx * 2 * d;
    const float* up = gate + d;
    float* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = gelu_tanh(gate[i]) * up[i];
    }
}

// ==========================================================================
// Element-wise activation kernels (no gating)
// ==========================================================================

// Element-wise GELU — BF16
extern "C" __global__ void gelu_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __nv_bfloat16* in_ptr = input + token_idx * d;
    __nv_bfloat16* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = __float2bfloat16(gelu_exact(__bfloat162float(in_ptr[i])));
    }
}

extern "C" __global__ void gelu_fp16(
    __half* __restrict__ out,
    const __half* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const __half* in_ptr = input + token_idx * d;
    __half* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = __float2half(gelu_exact(__half2float(in_ptr[i])));
    }
}

extern "C" __global__ void gelu_f32(
    float* __restrict__ out,
    const float* __restrict__ input,
    const int d
) {
    const int token_idx = blockIdx.x;
    const float* in_ptr = input + token_idx * d;
    float* out_ptr = out + token_idx * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        out_ptr[i] = gelu_exact(in_ptr[i]);
    }
}
