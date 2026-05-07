// AWQ-Marlin dequantization to BF16 for the M > GEMV-threshold prefill path.
//
// `AwqMarlinLinear::load_weights` stores `qweight` *transposed* as
// `[N, K/8]` so the decode-time `awq_gemv_int4_kt_bf16` kernel can issue
// vectorised (uint4) global loads along the contiguous K-axis. The same
// transposed layout makes the legacy GPTQ dequant kernel
// (`gptq_dequant_int4_bf16`, which reads `[K/8, N]`) unusable.
//
// This kernel reads the transposed qweight and emits a dense BF16 matrix
// with shape `[K, N]` row-major, suitable as the right-hand operand for a
// standard `bf16 @ bf16` cuBLAS GEMM `(M, K) @ (K, N) -> (M, N)`. Combined
// with candle's BF16 matmul this gives the prefill path the same
// asymptotic `M × K × N` cost as a real INT4 GEMM, without the
// `M ×` re-read of weights from HBM that the gemv path suffers.
//
// Layout summary:
//   qweight  [N, K/8]  u32   (transposed; nibbles in GPTQ order after
//                             AwqMarlinLinear::load_weights)
//   qzeros   [K/g, N/8] u32  (GPTQ-ordered nibbles per
//                             `repack_awq_nibbles`)
//   scales   [K/g, N]   bf16
//   out      [K, N]    bf16  (row-major; W in `Y = X @ W`)
//
// Group-size restriction: matches the rest of the Marlin/AWQ stack —
// `K % group_size == 0` and `N % 8 == 0`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ int extract_int4(uint32_t packed, int idx) {
    return (packed >> (idx * 4)) & 0xF;
}

extern "C" __global__ void awq_marlin_dequant_int4_bf16(
    __nv_bfloat16* __restrict__ out,          // [K, N]
    const uint32_t* __restrict__ qweight,     // [N, K/8] (transposed)
    const __nv_bfloat16* __restrict__ scales, // [K/g, N]
    const uint32_t* __restrict__ qzeros,      // [K/g, N/8]
    const int K,
    const int N,
    const int group_size,
    const int num_groups
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;  // N axis
    const int k = blockIdx.y * blockDim.y + threadIdx.y;  // K axis
    if (n >= N || k >= K) return;

    const int Kp8 = K / 8;
    const int g = (group_size > 0) ? (k / group_size) : 0;

    const float scale = __bfloat162float(scales[g * N + n]);

    const int zeros_per_row = (N + 7) / 8;
    const int z_pidx = n / 8;
    const int z_bidx = n % 8;
    const uint32_t z_packed = qzeros[g * zeros_per_row + z_pidx];
    const int zero = extract_int4(z_packed, z_bidx);

    // qweight is [N, K/8]: row n contains K/8 packed words, lane k%8 of
    // word k/8 holds the nibble for position k.
    const int w_pidx = k / 8;
    const int w_bidx = k % 8;
    const uint32_t w_packed = qweight[n * Kp8 + w_pidx];
    const int w_int = extract_int4(w_packed, w_bidx);

    const float wfp = (float)(w_int - zero) * scale;
    out[k * N + n] = __float2bfloat16(wfp);
}
