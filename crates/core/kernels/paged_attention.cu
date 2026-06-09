// PagedAttention v1 kernel for decode — bf16, configurable head_dim and block_size.
// Simplified from vLLM's paged_attention_v1_kernel.
//
// Grid: (num_heads, num_seqs, 1)
// Block: (NUM_THREADS, 1, 1) — threads cooperate across the head dimension
// Dynamic shared memory: sizeof(float) * (head_dim + NUM_WARPS + max_seq_len)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// Type-erased fp16/bf16 conversion helpers. The kernels operate in F32 internally
// (Q/K/V are upcast to float for arithmetic) and only the load/store boundary
// depends on the storage dtype.
template <typename T> __device__ __forceinline__ float to_f32(T x);
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_f32<__half>(__half x) {
    return __half2float(x);
}

template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <> __device__ __forceinline__ __half from_f32<__half>(float x) {
    return __float2half(x);
}

// ============================================================================
// KV cache load helper — uniform interface across fp16 / bf16 / fp8 / int8.
// ============================================================================
//
// `Cache_t` identifies the on-disk cache type. For full-precision native types
// (__half, __nv_bfloat16) the scale parameter is ignored and the value is just
// upcast to float. For quantized types (fp8_e4m3, fp8_e5m2, int8_t) the byte
// is dequantized via inline conversion and multiplied by a per-tensor scalar
// scale. Calling code reads the scale once per kernel (loaded into a uniform
// register) and passes it to every load — CUDA L1 caches the constant load.
//
// Wrappers around fp8 storage:
//   `fp8_e4m3_byte` / `fp8_e5m2_byte` exist purely as type markers so template
//   specialization can distinguish the two FP8 encodings that share the same
//   1-byte storage. Rust passes the raw U8 device pointer; the kernel parameter
//   is declared with the appropriate marker type and the per-byte read is
//   re-interpreted to the matching CUDA fp8 class for conversion.
struct fp8_e4m3_byte { uint8_t b; };
struct fp8_e5m2_byte { uint8_t b; };

template <typename Cache_t>
__device__ __forceinline__ float load_kv_to_f32(Cache_t v, float scale);

template <>
__device__ __forceinline__ float load_kv_to_f32<__half>(__half v, float /*scale*/) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float load_kv_to_f32<__nv_bfloat16>(
    __nv_bfloat16 v, float /*scale*/) {
    return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float load_kv_to_f32<fp8_e4m3_byte>(
    fp8_e4m3_byte v, float scale) {
    // The CUDA fp8 class encapsulates the byte and provides an explicit
    // `operator float()`. Reinterpret the storage byte as the e4m3 class
    // and convert.
    __nv_fp8_e4m3 packed;
    packed.__x = static_cast<__nv_fp8_storage_t>(v.b);
    return static_cast<float>(packed) * scale;
}
template <>
__device__ __forceinline__ float load_kv_to_f32<fp8_e5m2_byte>(
    fp8_e5m2_byte v, float scale) {
    __nv_fp8_e5m2 packed;
    packed.__x = static_cast<__nv_fp8_storage_t>(v.b);
    return static_cast<float>(packed) * scale;
}
template <>
__device__ __forceinline__ float load_kv_to_f32<int8_t>(int8_t v, float scale) {
    // INT8 storage is U8 on the Rust side (`quantize_int8` writes
    // `rounded + 128` to fit unsigned u8 — see
    // `crates/core/src/kv_cache/quantization.rs::quantize_int8`).
    // The kernel parameter is declared `int8_t*` purely to dispatch the
    // template specialization; here we reinterpret the byte as unsigned
    // and subtract 128 to recover the symmetric [-128, 127] range
    // before applying the scale. Equivalent to vLLM's
    // `int8::scaled_convert` flow.
    const uint8_t u = static_cast<uint8_t>(v);
    return (static_cast<float>(u) - 128.0f) * scale;
}

// NUM_THREADS must be >= head_dim for correctness when head_dim <= 128,
// and threads loop for head_dim > 128.
#define NUM_THREADS 128
#define NUM_WARPS (NUM_THREADS / 32)

// Warp-level reduce sum via shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduce sum: warps → 1 value.
// Returns result in all threads (broadcast via shared memory).
__device__ float block_reduce_sum(float val, float* reduce_smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Intra-warp reduce
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane_id == 0) {
        reduce_smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float v = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            reduce_smem[0] = v;
        }
    }
    __syncthreads();

    return reduce_smem[0];
}

// Block-level reduce max, result broadcast to all threads.
__device__ float block_reduce_max(float val, float* reduce_smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Intra-warp reduce max
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    if (lane_id == 0) {
        reduce_smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
        }
        if (lane_id == 0) {
            reduce_smem[0] = v;
        }
    }
    __syncthreads();

    return reduce_smem[0];
}

// Shared implementation for paged attention v1 with configurable dimensions
// and optional ALiBi support.
//
// Each thread block handles one (head, sequence) pair.
// Threads cooperate across the head dimension: for head_dim <= NUM_THREADS,
// each thread handles one dimension; for head_dim > NUM_THREADS, threads loop.
//
// When alibi_slopes is non-null, the ALiBi bias is added to the scaled QK
// logits: logit[token_pos] += alibi_slope * (token_pos - (seq_len - 1))
//
// Dynamic shared memory layout:
//   float q_smem[head_dim]             -- query vector
//   float reduce_smem[NUM_WARPS]       -- reduction workspace
//   float logits[max_context_len]      -- QK logits (variable)
template <typename Q_t, typename Cache_t>
__device__ void paged_attention_v1_impl(
    Q_t* __restrict__ out,                     // [num_seqs, num_heads, head_dim]
    const Q_t* __restrict__ q,                 // [num_seqs, num_heads, head_dim]
    const Cache_t* __restrict__ k_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    const Cache_t* __restrict__ v_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int window,                          // sliding window; 0 = full causal
    const float* __restrict__ alibi_slopes,    // [num_heads] or nullptr
    const float* __restrict__ k_scale_ptr,     // [1] or nullptr (1.0 default)
    const float* __restrict__ v_scale_ptr      // [1] or nullptr (1.0 default)
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // GQA: map query head to KV head
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Sliding window: the query (position seq_len-1) attends only to the
    // last `window` keys — positions (query_pos - window, query_pos], i.e.
    // [seq_len - window, seq_len). Matches the Rust-side decode
    // `sliding_window_mask` exactly. All cached positions are causal at
    // decode, so a start offset replaces masking. Logits are indexed
    // relative to `start_token`, bounding shared memory by the window.
    const int start_token = (window > 0 && seq_len > window) ? (seq_len - window) : 0;
    const int eff_len = seq_len - start_token;

    // ALiBi slope for this head (0 if not using ALiBi)
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;
    // Per-tensor KV scales — loaded once per kernel (cached in L1). Native-dtype
    // cache (fp16/bf16) ignores these in `load_kv_to_f32`; quantized cache
    // (fp8/int8) multiplies the dequantized byte by the scale.
    const float k_scale = (k_scale_ptr != nullptr) ? *k_scale_ptr : 1.0f;
    const float v_scale = (v_scale_ptr != nullptr) ? *v_scale_ptr : 1.0f;

    // Shared memory layout (fixed offsets to avoid overlap):
    //   [0, head_dim)                          -- q vector
    //   [head_dim, head_dim + NUM_WARPS)       -- reduction workspace
    //   [head_dim + NUM_WARPS, ...)             -- logits array (seq_len floats)
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce_smem = smem + head_dim;
    float* logits = smem + head_dim + NUM_WARPS;

    // Cache strides for [num_blocks, block_size, num_kv_heads, head_dim] layout
    const int cache_stride_block = block_size * num_kv_heads * head_dim;
    const int cache_stride_token = num_kv_heads * head_dim;

    // Load query vector into shared memory (threads loop if head_dim > NUM_THREADS)
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        const int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        q_smem[d] = to_f32<Q_t>(q[q_offset]);
    }
    __syncthreads();

    // ---- Phase 1: Compute QK dot products ----
    const int num_blocks_used = (seq_len + block_size - 1) / block_size;
    const int start_block_idx = start_token / block_size;
    float qk_max = -FLT_MAX;

    // The query position is the last position: seq_len - 1 (decode produces 1 token).
    const int query_pos = seq_len - 1;

    for (int block_idx = start_block_idx; block_idx < num_blocks_used; block_idx++) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        const int block_start_token = block_idx * block_size;
        const int token_lo = max(block_start_token, start_token);
        const int token_hi = min(block_start_token + block_size, seq_len);

        for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
            const int token = abs_token - block_start_token;
            // Partial dot product across head dimensions
            float qk = 0.0f;
            for (int d = tid; d < head_dim; d += NUM_THREADS) {
                const int k_offset = physical_block * cache_stride_block
                                   + token * cache_stride_token
                                   + kv_head_idx * head_dim
                                   + d;
                const float k_val = load_kv_to_f32<Cache_t>(k_cache[k_offset], k_scale);
                qk += q_smem[d] * k_val;
            }

            // Block-level reduce to get full dot product
            qk = block_reduce_sum(qk, reduce_smem);

            // Thread 0 stores the scaled logit (with optional ALiBi bias)
            if (tid == 0) {
                float scaled_qk = qk * scale;
                // ALiBi bias: slope * (key_position - query_position)
                scaled_qk += alibi_slope * (float)(abs_token - query_pos);
                logits[abs_token - start_token] = scaled_qk;
                qk_max = fmaxf(qk_max, scaled_qk);
            }
            __syncthreads();
        }
    }

    // Broadcast qk_max from thread 0 to all threads
    if (tid == 0) {
        reduce_smem[0] = qk_max;
    }
    __syncthreads();
    qk_max = reduce_smem[0];

    // ---- Phase 2: Softmax ----
    float exp_sum = 0.0f;
    for (int i = tid; i < eff_len; i += NUM_THREADS) {
        const float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    __syncthreads();

    exp_sum = block_reduce_sum(exp_sum, reduce_smem);

    const float inv_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int i = tid; i < eff_len; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation ----
    // Each thread accumulates for its subset of head dimensions
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;

        for (int block_idx = start_block_idx; block_idx < num_blocks_used; block_idx++) {
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
            const int block_start_token = block_idx * block_size;
            const int token_lo = max(block_start_token, start_token);
            const int token_hi = min(block_start_token + block_size, seq_len);

            for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
                const int token = abs_token - block_start_token;
                const int v_offset = physical_block * cache_stride_block
                                   + token * cache_stride_token
                                   + kv_head_idx * head_dim
                                   + d;
                const float v_val = load_kv_to_f32<Cache_t>(v_cache[v_offset], v_scale);
                const float weight = logits[abs_token - start_token];
                acc += weight * v_val;
            }
        }

        // Write output
        const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        out[out_offset] = from_f32<Q_t>(acc);
    }
}

// ============================================================================
// V1 entry points
// ============================================================================
//
// Existing fp16/bf16 entries (kv_cache_dtype == Auto, i.e. KV cache stored at
// the same dtype as Q) gain two trailing optional scale pointers — pass
// nullptr to skip scaling (existing Auto behaviour). FP8/INT8 entries cast
// the cache pointer to the appropriate marker type and require scales.

// --- Auto KV cache (Q dtype matches cache dtype) -----------------------------

extern "C" __global__ void paged_attention_v1_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v1_impl<__nv_bfloat16, __nv_bfloat16>(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, window,
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v1_bf16_alibi(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v1_impl<__nv_bfloat16, __nv_bfloat16>(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, /*window*/ 0,
        alibi_slopes, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v1_f16(
    __half* __restrict__ out,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v1_impl<__half, __half>(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, window,
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v1_f16_alibi(
    __half* __restrict__ out,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v1_impl<__half, __half>(
        out, q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, /*window*/ 0,
        alibi_slopes, k_scale_ptr, v_scale_ptr
    );
}

// --- Quantized KV cache (FP8 E4M3 / E5M2 / INT8) -----------------------------
//
// The KV cache is allocated as U8 on the Rust side; the kernel parameter type
// (`fp8_e4m3_byte*`, `fp8_e5m2_byte*`, `int8_t*`) selects the dequantization
// path inside `load_kv_to_f32`. Two macros — one for the non-ALiBi entry, one
// for the ALiBi entry — keep both forms in lock-step. Both call the same
// `paged_attention_v1_impl<QTYPE, CACHE_T>` template; only the
// `alibi_slopes` argument differs.
#define EXL3_DEFINE_V1_QUANT(QTYPE, QSUFFIX, KVMARKER, KVSUFFIX) \
extern "C" __global__ void paged_attention_v1_##QSUFFIX##_##KVSUFFIX( \
    QTYPE* __restrict__ out, \
    const QTYPE* __restrict__ q, \
    const KVMARKER* __restrict__ k_cache, \
    const KVMARKER* __restrict__ v_cache, \
    const int* __restrict__ block_tables, \
    const int* __restrict__ seq_lens, \
    const float scale, \
    const int num_heads, \
    const int num_kv_heads, \
    const int max_blocks_per_seq, \
    const int head_dim, \
    const int block_size, \
    const int window, \
    const float* __restrict__ k_scale_ptr, \
    const float* __restrict__ v_scale_ptr \
) { \
    paged_attention_v1_impl<QTYPE, KVMARKER>( \
        out, q, k_cache, v_cache, block_tables, seq_lens, \
        scale, num_heads, num_kv_heads, max_blocks_per_seq, \
        head_dim, block_size, window, \
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr \
    ); \
}

#define EXL3_DEFINE_V1_QUANT_ALIBI(QTYPE, QSUFFIX, KVMARKER, KVSUFFIX) \
extern "C" __global__ void paged_attention_v1_##QSUFFIX##_##KVSUFFIX##_alibi( \
    QTYPE* __restrict__ out, \
    const QTYPE* __restrict__ q, \
    const KVMARKER* __restrict__ k_cache, \
    const KVMARKER* __restrict__ v_cache, \
    const int* __restrict__ block_tables, \
    const int* __restrict__ seq_lens, \
    const float scale, \
    const int num_heads, \
    const int num_kv_heads, \
    const int max_blocks_per_seq, \
    const int head_dim, \
    const int block_size, \
    const float* __restrict__ alibi_slopes, \
    const float* __restrict__ k_scale_ptr, \
    const float* __restrict__ v_scale_ptr \
) { \
    paged_attention_v1_impl<QTYPE, KVMARKER>( \
        out, q, k_cache, v_cache, block_tables, seq_lens, \
        scale, num_heads, num_kv_heads, max_blocks_per_seq, \
        head_dim, block_size, /*window*/ 0, \
        alibi_slopes, k_scale_ptr, v_scale_ptr \
    ); \
}

EXL3_DEFINE_V1_QUANT(__nv_bfloat16, bf16, fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V1_QUANT(__nv_bfloat16, bf16, fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V1_QUANT(__nv_bfloat16, bf16, int8_t,        int8)
EXL3_DEFINE_V1_QUANT(__half,        f16,  fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V1_QUANT(__half,        f16,  fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V1_QUANT(__half,        f16,  int8_t,        int8)

// Phase 10: ALiBi + FP8/INT8 entry points. 6 new V1 symbols.
EXL3_DEFINE_V1_QUANT_ALIBI(__nv_bfloat16, bf16, fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V1_QUANT_ALIBI(__nv_bfloat16, bf16, fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V1_QUANT_ALIBI(__nv_bfloat16, bf16, int8_t,        int8)
EXL3_DEFINE_V1_QUANT_ALIBI(__half,        f16,  fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V1_QUANT_ALIBI(__half,        f16,  fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V1_QUANT_ALIBI(__half,        f16,  int8_t,        int8)

#undef EXL3_DEFINE_V1_QUANT
#undef EXL3_DEFINE_V1_QUANT_ALIBI

// ============================================================================
// PagedAttention V2: Split-K partitioned attention for long sequences
// ============================================================================
//
// V2 splits the KV sequence into partitions of PARTITION_SIZE tokens.
// Each thread block processes one partition independently, producing:
//   - tmp_out: partial weighted V accumulation
//   - max_logits: max QK logit within the partition (for log-sum-exp)
//   - exp_sums: sum of exp(logit - max) within the partition
//
// A separate reduce kernel merges partitions using numerically stable
// log-sum-exp arithmetic.
//
// Grid: (num_heads, num_seqs, max_num_partitions)
// Block: (NUM_THREADS, 1, 1)

// `partition_size` is supplied at launch time (see paged_attention_v2_cuda
// in cuda_kernels.rs). Smaller partitions ⇒ more grid blocks ⇒ higher SM
// occupancy at batch=1; larger partitions ⇒ fewer reduce_smem entries and
// less tmp_out write traffic at long context. The dispatch in Rust picks
// the value adaptively from `max_seq_len`.

// V2 main kernel: compute attention for one partition of the sequence.
template <typename Q_t, typename Cache_t>
__device__ void paged_attention_v2_impl(
    float* __restrict__ tmp_out,               // [num_seqs, num_heads, max_num_partitions, head_dim]
    float* __restrict__ exp_sums,              // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits_out,        // [num_seqs, num_heads, max_num_partitions]
    const Q_t* __restrict__ q,                 // [num_seqs, num_heads, head_dim]
    const Cache_t* __restrict__ k_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    const Cache_t* __restrict__ v_cache,       // [num_blocks, block_size, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,      // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int partition_size,
    const int max_num_partitions,
    const int window,                          // sliding window; 0 = full causal
    const float* __restrict__ alibi_slopes,    // [num_heads] or nullptr
    const float* __restrict__ k_scale_ptr,     // [1] or nullptr (1.0 default)
    const float* __restrict__ v_scale_ptr      // [1] or nullptr (1.0 default)
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Sliding window: attend only to the last `window` keys — positions
    // [seq_len - window, seq_len). Partitions are re-based at the window
    // start so per-sequence partition count is ⌈min(seq_len, window) /
    // partition_size⌉; the reduce kernel derives the same count from
    // (seq_len, window). See `paged_attention_v1_impl` for semantics.
    const int start_token = (window > 0 && seq_len > window) ? (seq_len - window) : 0;

    // Partition boundaries in token space (re-based at start_token)
    const int partition_start_token = start_token + partition_idx * partition_size;
    if (partition_start_token >= seq_len) return;  // No work for this partition
    const int partition_end_token = min(partition_start_token + partition_size, seq_len);
    const int partition_num_tokens = partition_end_token - partition_start_token;

    // GQA: map query head to KV head
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // ALiBi slope
    const float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[head_idx] : 0.0f;
    const int query_pos = seq_len - 1;

    // Shared memory layout:
    //   [0, head_dim)                    -- q vector (f32)
    //   [head_dim, head_dim+NUM_WARPS)   -- reduction workspace
    //   [head_dim+NUM_WARPS, ...)        -- logits[partition_size] (f32)
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce_smem = smem + head_dim;
    float* logits = smem + head_dim + NUM_WARPS;

    // Cache strides
    const int cache_stride_block = block_size * num_kv_heads * head_dim;
    const int cache_stride_token = num_kv_heads * head_dim;

    // Per-tensor KV scales (see V1 impl for rationale).
    const float k_scale = (k_scale_ptr != nullptr) ? *k_scale_ptr : 1.0f;
    const float v_scale = (v_scale_ptr != nullptr) ? *v_scale_ptr : 1.0f;

    // Load query into shared memory
    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        const int q_offset = seq_idx * num_heads * head_dim + head_idx * head_dim + d;
        q_smem[d] = to_f32<Q_t>(q[q_offset]);
    }
    __syncthreads();

    // ---- Phase 1: Compute QK dot products for this partition ----
    //
    // Strategy: each warp owns its own subset of tokens (stride NUM_WARPS), so
    // the per-token QK reduction collapses into a warp-shuffle (no
    // __syncthreads).  The whole K-pass runs with ONE __syncthreads at the
    // very end (to broadcast the global qk_max), instead of one per token in
    // the legacy block-reduce design.
    //
    // Requires head_dim divisible by 32 (warp size).  All current LLM head
    // dimensions (64, 96, 128, 192, 256) satisfy this.  Upper bound 512
    // comes from the unroll cap in the per-lane dim loop below
    // (dims_per_thread = head_dim/32, hard-capped at 16).  If a future
    // caller breaks any of those, the legacy block-reduce path below is
    // safe for any head_dim.
    const int start_block_idx = partition_start_token / block_size;
    const int end_block_idx = (partition_end_token + block_size - 1) / block_size;

    float qk_max = -FLT_MAX;

    if (head_dim % 32 == 0 && head_dim >= 32 && head_dim <= 512) {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int dims_per_thread = head_dim / 32;
        const int dim_base = lane_id * dims_per_thread;

        // Loop tokens by warp stride.  Each warp handles
        // local_idx ∈ {warp_id, warp_id+NUM_WARPS, warp_id+2*NUM_WARPS, ...}.
        for (int local_idx = warp_id; local_idx < partition_num_tokens;
             local_idx += NUM_WARPS) {
            const int abs_token = partition_start_token + local_idx;
            const int blk = abs_token / block_size;
            const int token_in_block = abs_token - blk * block_size;
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + blk];

            const int k_base = physical_block * cache_stride_block
                             + token_in_block * cache_stride_token
                             + kv_head_idx * head_dim;

            float qk = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) {
                if (dd >= dims_per_thread) break;  // compile-time-friendly bound
                const int d = dim_base + dd;
                qk += q_smem[d] * load_kv_to_f32<Cache_t>(k_cache[k_base + d], k_scale);
            }

            // Warp-level reduce; result lands in lane 0.
            qk = warp_reduce_sum(qk);

            if (lane_id == 0) {
                float scaled_qk = qk * scale;
                scaled_qk += alibi_slope * (float)(abs_token - query_pos);
                logits[local_idx] = scaled_qk;
                qk_max = fmaxf(qk_max, scaled_qk);
            }
        }

        // Each warp's qk_max lives in lane 0 register.  Reduce across warps
        // via shared memory.  This is the only __syncthreads in the K-pass.
        if (lane_id == 0) {
            reduce_smem[warp_id] = qk_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : -FLT_MAX;
            #pragma unroll
            for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
            }
            if (lane_id == 0) {
                reduce_smem[0] = v;
            }
        }
        __syncthreads();
        qk_max = reduce_smem[0];
    } else {
        // Legacy fallback for head dims not divisible by 32.
        for (int blk = start_block_idx; blk < end_block_idx; blk++) {
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + blk];
            const int block_start_token = blk * block_size;
            const int token_lo = max(block_start_token, partition_start_token);
            const int token_hi = min(block_start_token + block_size, partition_end_token);

            for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
                const int token_in_block = abs_token - block_start_token;

                float qk = 0.0f;
                for (int d = tid; d < head_dim; d += NUM_THREADS) {
                    const int k_offset = physical_block * cache_stride_block
                                       + token_in_block * cache_stride_token
                                       + kv_head_idx * head_dim + d;
                    qk += q_smem[d] * load_kv_to_f32<Cache_t>(k_cache[k_offset], k_scale);
                }

                qk = block_reduce_sum(qk, reduce_smem);

                if (tid == 0) {
                    const int local_idx = abs_token - partition_start_token;
                    float scaled_qk = qk * scale;
                    scaled_qk += alibi_slope * (float)(abs_token - query_pos);
                    logits[local_idx] = scaled_qk;
                    qk_max = fmaxf(qk_max, scaled_qk);
                }
                __syncthreads();
            }
        }

        // Broadcast qk_max via shared memory.
        if (tid == 0) {
            reduce_smem[0] = qk_max;
        }
        __syncthreads();
        qk_max = reduce_smem[0];
    }

    // ---- Phase 2: Softmax (local to partition) ----
    float local_exp_sum = 0.0f;
    for (int i = tid; i < partition_num_tokens; i += NUM_THREADS) {
        const float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        local_exp_sum += val;
    }
    __syncthreads();

    local_exp_sum = block_reduce_sum(local_exp_sum, reduce_smem);

    // Store partition-level max and exp_sum for the reduce kernel
    const int partition_meta_offset = seq_idx * num_heads * max_num_partitions
                                    + head_idx * max_num_partitions
                                    + partition_idx;
    if (tid == 0) {
        max_logits_out[partition_meta_offset] = qk_max;
        exp_sums[partition_meta_offset] = local_exp_sum;
    }

    // Normalize logits (divide by exp_sum) for weighted V accumulation
    const float inv_sum = __fdividef(1.0f, local_exp_sum + 1e-6f);
    for (int i = tid; i < partition_num_tokens; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Weighted V accumulation (stored as f32 in tmp_out) ----
    const int tmp_out_offset = seq_idx * num_heads * max_num_partitions * head_dim
                             + head_idx * max_num_partitions * head_dim
                             + partition_idx * head_dim;

    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;

        for (int blk = start_block_idx; blk < end_block_idx; blk++) {
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + blk];
            const int block_start_token = blk * block_size;
            const int token_lo = max(block_start_token, partition_start_token);
            const int token_hi = min(block_start_token + block_size, partition_end_token);

            for (int abs_token = token_lo; abs_token < token_hi; abs_token++) {
                const int token_in_block = abs_token - block_start_token;
                const int v_offset = physical_block * cache_stride_block
                                   + token_in_block * cache_stride_token
                                   + kv_head_idx * head_dim + d;
                const int local_idx = abs_token - partition_start_token;
                acc += logits[local_idx] * load_kv_to_f32<Cache_t>(v_cache[v_offset], v_scale);
            }
        }

        tmp_out[tmp_out_offset + d] = acc;
    }
}

// V2 reduce kernel: merge partition results using log-sum-exp.
//
// Grid: (num_heads, num_seqs, 1)
// Block: (NUM_THREADS, 1, 1)
//
// For each (head, seq), combines tmp_out from all partitions into final output.
// Uses the identity:
//   softmax(concat(A,B)) = rescale(softmax(A), softmax(B))
// via the log-sum-exp trick for numerical stability.
template <typename T>
__device__ void paged_attention_v2_reduce_impl(
    T* __restrict__ out,                    // [num_seqs, num_heads, head_dim]
    const float* __restrict__ tmp_out,      // [num_seqs, num_heads, max_num_partitions, head_dim]
    const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
    const int* __restrict__ seq_lens,       // [num_seqs]
    const int num_heads,
    const int head_dim,
    const int partition_size,
    const int max_num_partitions,
    const int window                        // sliding window; 0 = full causal
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    // Must mirror the Stage-1 re-based partitioning: with a window the
    // per-sequence token range is the last min(seq_len, window) tokens.
    const int eff_len = (window > 0 && seq_len > window) ? window : seq_len;
    const int num_partitions = (eff_len + partition_size - 1) / partition_size;

    // Fast path: single partition → just copy (convert f32 → output dtype)
    if (num_partitions == 1) {
        const int tmp_offset = seq_idx * num_heads * max_num_partitions * head_dim
                             + head_idx * max_num_partitions * head_dim;
        const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
        for (int d = tid; d < head_dim; d += NUM_THREADS) {
            out[out_offset + d] = from_f32<T>(tmp_out[tmp_offset + d]);
        }
        return;
    }

    // Shared memory for partition metadata:
    //   [0, max_num_partitions)                    -- max logits per partition
    //   [max_num_partitions, 2*max_num_partitions) -- rescaled exp sums
    extern __shared__ float reduce_smem[];
    float* shared_max_logits = reduce_smem;
    float* shared_exp_sums = reduce_smem + max_num_partitions;

    // Step 1: Find global max logit across all partitions
    const int meta_offset = seq_idx * num_heads * max_num_partitions
                          + head_idx * max_num_partitions;

    float global_max = -FLT_MAX;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        float l = max_logits[meta_offset + p];
        shared_max_logits[p] = l;
        global_max = fmaxf(global_max, l);
    }
    __syncthreads();

    // Reduce global_max across threads
    // Warp-level reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        global_max = fmaxf(global_max, __shfl_xor_sync(0xffffffff, global_max, offset));
    }
    // Inter-warp reduce via shared memory (reuse end of shared_exp_sums temporarily)
    {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        // Use space after shared_exp_sums for warp reduction
        float* warp_reduce = reduce_smem + 2 * max_num_partitions;
        if (lane_id == 0) {
            warp_reduce[warp_id] = global_max;
        }
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? warp_reduce[lane_id] : -FLT_MAX;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
            }
            if (lane_id == 0) {
                warp_reduce[0] = v;
            }
        }
        __syncthreads();
        global_max = warp_reduce[0];
    }

    // Step 2: Rescale exp_sums by the global max (log-sum-exp correction)
    float global_exp_sum = 0.0f;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        float rescaled = exp_sums[meta_offset + p] * __expf(shared_max_logits[p] - global_max);
        shared_exp_sums[p] = rescaled;
        global_exp_sum += rescaled;
    }
    __syncthreads();

    // Reduce global_exp_sum across threads
    {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        // Warp-level reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            global_exp_sum += __shfl_xor_sync(0xffffffff, global_exp_sum, offset);
        }
        float* warp_reduce = reduce_smem + 2 * max_num_partitions;
        if (lane_id == 0) {
            warp_reduce[warp_id] = global_exp_sum;
        }
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? warp_reduce[lane_id] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane_id == 0) {
                warp_reduce[0] = v;
            }
        }
        __syncthreads();
        global_exp_sum = warp_reduce[0];
    }

    const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

    // Step 3: Aggregate weighted outputs from all partitions
    const int out_offset = seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int tmp_base = seq_idx * num_heads * max_num_partitions * head_dim
                       + head_idx * max_num_partitions * head_dim;

    for (int d = tid; d < head_dim; d += NUM_THREADS) {
        float acc = 0.0f;
        for (int p = 0; p < num_partitions; p++) {
            // tmp_out already has locally-normalized values (divided by local exp_sum).
            // We need to undo that normalization and apply global normalization:
            //   contribution = tmp_out[p][d] * local_exp_sum[p] * exp(local_max - global_max) / global_exp_sum
            //                = tmp_out[p][d] * shared_exp_sums[p] / global_exp_sum
            acc += tmp_out[tmp_base + p * head_dim + d] * shared_exp_sums[p];
        }
        out[out_offset + d] = from_f32<T>(acc * inv_global_exp_sum);
    }
}

// Thin bf16 wrapper around the templated reduce.
extern "C" __global__ void paged_attention_v2_reduce_bf16(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ tmp_out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const int* __restrict__ seq_lens,
    const int num_heads,
    const int head_dim,
    const int partition_size,
    const int max_num_partitions,
    const int window
) {
    paged_attention_v2_reduce_impl<__nv_bfloat16>(
        out, tmp_out, exp_sums, max_logits, seq_lens,
        num_heads, head_dim, partition_size, max_num_partitions, window
    );
}

// F16 reduce wrapper.
extern "C" __global__ void paged_attention_v2_reduce_f16(
    __half* __restrict__ out,
    const float* __restrict__ tmp_out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const int* __restrict__ seq_lens,
    const int num_heads,
    const int head_dim,
    const int partition_size,
    const int max_num_partitions,
    const int window
) {
    paged_attention_v2_reduce_impl<__half>(
        out, tmp_out, exp_sums, max_logits, seq_lens,
        num_heads, head_dim, partition_size, max_num_partitions, window
    );
}

// ============================================================================
// V2 entry points
// ============================================================================

// --- Auto KV cache (Q dtype matches cache dtype) -----------------------------

extern "C" __global__ void paged_attention_v2_bf16(
    float* __restrict__ tmp_out,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int partition_size,
    const int max_num_partitions,
    const int window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v2_impl<__nv_bfloat16, __nv_bfloat16>(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, partition_size, max_num_partitions, window,
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v2_bf16_alibi(
    float* __restrict__ tmp_out,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int partition_size,
    const int max_num_partitions,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v2_impl<__nv_bfloat16, __nv_bfloat16>(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, partition_size, max_num_partitions, /*window*/ 0,
        alibi_slopes, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v2_f16(
    float* __restrict__ tmp_out,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int partition_size,
    const int max_num_partitions,
    const int window,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v2_impl<__half, __half>(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, partition_size, max_num_partitions, window,
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr
    );
}

extern "C" __global__ void paged_attention_v2_f16_alibi(
    float* __restrict__ tmp_out,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq,
    const int head_dim,
    const int block_size,
    const int partition_size,
    const int max_num_partitions,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ k_scale_ptr,
    const float* __restrict__ v_scale_ptr
) {
    paged_attention_v2_impl<__half, __half>(
        tmp_out, exp_sums, max_logits,
        q, k_cache, v_cache, block_tables, seq_lens,
        scale, num_heads, num_kv_heads, max_blocks_per_seq,
        head_dim, block_size, partition_size, max_num_partitions, /*window*/ 0,
        alibi_slopes, k_scale_ptr, v_scale_ptr
    );
}

// --- Quantized KV cache (FP8 E4M3 / E5M2 / INT8) -----------------------------
#define EXL3_DEFINE_V2_QUANT(QTYPE, QSUFFIX, KVMARKER, KVSUFFIX) \
extern "C" __global__ void paged_attention_v2_##QSUFFIX##_##KVSUFFIX( \
    float* __restrict__ tmp_out, \
    float* __restrict__ exp_sums, \
    float* __restrict__ max_logits, \
    const QTYPE* __restrict__ q, \
    const KVMARKER* __restrict__ k_cache, \
    const KVMARKER* __restrict__ v_cache, \
    const int* __restrict__ block_tables, \
    const int* __restrict__ seq_lens, \
    const float scale, \
    const int num_heads, \
    const int num_kv_heads, \
    const int max_blocks_per_seq, \
    const int head_dim, \
    const int block_size, \
    const int partition_size, \
    const int max_num_partitions, \
    const int window, \
    const float* __restrict__ k_scale_ptr, \
    const float* __restrict__ v_scale_ptr \
) { \
    paged_attention_v2_impl<QTYPE, KVMARKER>( \
        tmp_out, exp_sums, max_logits, \
        q, k_cache, v_cache, block_tables, seq_lens, \
        scale, num_heads, num_kv_heads, max_blocks_per_seq, \
        head_dim, block_size, partition_size, max_num_partitions, window, \
        /*alibi*/ nullptr, k_scale_ptr, v_scale_ptr \
    ); \
}

#define EXL3_DEFINE_V2_QUANT_ALIBI(QTYPE, QSUFFIX, KVMARKER, KVSUFFIX) \
extern "C" __global__ void paged_attention_v2_##QSUFFIX##_##KVSUFFIX##_alibi( \
    float* __restrict__ tmp_out, \
    float* __restrict__ exp_sums, \
    float* __restrict__ max_logits, \
    const QTYPE* __restrict__ q, \
    const KVMARKER* __restrict__ k_cache, \
    const KVMARKER* __restrict__ v_cache, \
    const int* __restrict__ block_tables, \
    const int* __restrict__ seq_lens, \
    const float scale, \
    const int num_heads, \
    const int num_kv_heads, \
    const int max_blocks_per_seq, \
    const int head_dim, \
    const int block_size, \
    const int partition_size, \
    const int max_num_partitions, \
    const float* __restrict__ alibi_slopes, \
    const float* __restrict__ k_scale_ptr, \
    const float* __restrict__ v_scale_ptr \
) { \
    paged_attention_v2_impl<QTYPE, KVMARKER>( \
        tmp_out, exp_sums, max_logits, \
        q, k_cache, v_cache, block_tables, seq_lens, \
        scale, num_heads, num_kv_heads, max_blocks_per_seq, \
        head_dim, block_size, partition_size, max_num_partitions, /*window*/ 0, \
        alibi_slopes, k_scale_ptr, v_scale_ptr \
    ); \
}

EXL3_DEFINE_V2_QUANT(__nv_bfloat16, bf16, fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V2_QUANT(__nv_bfloat16, bf16, fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V2_QUANT(__nv_bfloat16, bf16, int8_t,        int8)
EXL3_DEFINE_V2_QUANT(__half,        f16,  fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V2_QUANT(__half,        f16,  fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V2_QUANT(__half,        f16,  int8_t,        int8)

// Phase 10: ALiBi + FP8/INT8 entry points. 6 new V2 symbols.
EXL3_DEFINE_V2_QUANT_ALIBI(__nv_bfloat16, bf16, fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V2_QUANT_ALIBI(__nv_bfloat16, bf16, fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V2_QUANT_ALIBI(__nv_bfloat16, bf16, int8_t,        int8)
EXL3_DEFINE_V2_QUANT_ALIBI(__half,        f16,  fp8_e4m3_byte, fp8e4m3)
EXL3_DEFINE_V2_QUANT_ALIBI(__half,        f16,  fp8_e5m2_byte, fp8e5m2)
EXL3_DEFINE_V2_QUANT_ALIBI(__half,        f16,  int8_t,        int8)

#undef EXL3_DEFINE_V2_QUANT
#undef EXL3_DEFINE_V2_QUANT_ALIBI
