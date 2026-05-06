# ADR-0014: PagedAttention FP16 Support via Templated Kernels

## Status

Accepted

## Context

The PagedAttention CUDA kernels in `crates/core/kernels/paged_attention.cu`
were originally bf16-only:

- `paged_attention_v1_bf16`, `paged_attention_v1_bf16_alibi`
- `paged_attention_v2_bf16`, `paged_attention_v2_bf16_alibi`
- `paged_attention_v2_reduce_bf16`

The Rust-side CustomOp wrappers (`PagedAttnOp`, `PagedAttnV2Op`,
`PagedAttnAliBiOp`, `PagedAttnV2AliBiOp` in `crates/core/src/cuda_kernels.rs`)
hardcoded `CudaStorageSlice::BF16` matches and bailed on any other dtype:

```
"paged_attention expects bf16 Q tensor"
```

Practical consequence: starting the server with `--dtype fp16` produced a
hard failure as soon as the first attention kernel was invoked, even though
weights, activations, and the rest of the model would have run correctly in
fp16. Users running fp16 checkpoints had to manually downcast to bf16 or
use a different attention backend, and there was no path to use fp16 with
PagedAttention at all.

Other CUDA kernels in the repo (`reshape_and_cache.cu`, RMSNorm, RoPE,
activations) already supported both bf16 and fp16 — PagedAttention was the
sole gap.

## Decision

Templatize the device-side `_impl` functions over the storage dtype, then
provide thin `extern "C"` wrappers for both bf16 and fp16. The Rust dispatch
selects the kernel symbol at launch time based on the Q-tensor dtype.

### Kernel layer (`paged_attention.cu`)

1. Two type-erased helpers for the load/store boundary:

   ```cpp
   template <typename T> __device__ __forceinline__ float to_f32(T x);
   template <typename T> __device__ __forceinline__ T from_f32(float x);
   ```

   Specialized for `__nv_bfloat16` (`__bfloat162float` / `__float2bfloat16`)
   and `__half` (`__half2float` / `__float2half`). All arithmetic stays in
   F32 inside the kernel — only loads from Q/K/V cache and stores to the
   output buffer touch the parameterized type.

2. The three device-side implementations are converted to function templates:

   - `template<typename T> paged_attention_v1_impl(...)`
   - `template<typename T> paged_attention_v2_impl(...)`
   - `template<typename T> paged_attention_v2_reduce_impl(...)`

3. Five new `extern "C" __global__` wrappers instantiate the templates with
   `__half`:

   - `paged_attention_v1_f16`
   - `paged_attention_v1_f16_alibi`
   - `paged_attention_v2_f16`
   - `paged_attention_v2_f16_alibi`
   - `paged_attention_v2_reduce_f16`

   Existing bf16 wrappers explicitly instantiate with `__nv_bfloat16`. The
   `extern "C"` ABI is preserved — only kernel symbols are added, none renamed.

### Rust layer (`cuda_kernels.rs`)

Each of the four PagedAttention CustomOps (`PagedAttnOp`, `PagedAttnV2Op`,
`PagedAttnAliBiOp`, `PagedAttnV2AliBiOp`) is updated to:

- Accept `CudaStorageSlice::BF16(_)` **or** `CudaStorageSlice::F16(_)` for Q,
  with a single helper that picks the matching K and V slices and bails on
  dtype mismatch between Q/K/V (fail-loudly, no silent coercion).
- Choose the kernel symbol name based on the resolved dtype:
  `paged_attention_v1_bf16` ↔ `paged_attention_v1_f16`, and the matching
  `_alibi` / `_v2_reduce` variants.

### Build (`build.rs`)

No changes. `paged_attention.cu` is already in the kernel list with
`min_sm: 80`, which covers both bf16 (sm_80+ requirement) and fp16
(sm_53+).

## Alternatives Considered

### A. Convert weights to bf16 on load when `--dtype fp16` is requested

Rejected. This silently violates the user's dtype contract, hides the
underlying capability gap, and forfeits any precision-vs-throughput trade
the user may have wanted. fp16 has a wider exponent range than bf16 has
mantissa bits — for some models the choice is intentional.

### B. Gate `--dtype fp16` with a hard error

Rejected as a non-production-grade workaround. The repo's stated quality
bar is "Production-grade. Every line, every commit, every decision — as if
it ships to thousands of GPUs tomorrow." Returning "fp16 not supported"
when the only missing piece is a one-line C++ template instantiation does
not meet that bar.

### C. Move PagedAttention to CUTLASS or another templated GEMM library

Rejected. CUTLASS would bring a sizeable build-time dependency for a single
kernel family. The existing hand-rolled kernels are correct, well-tested
on the bf16 path, and the templatization is mechanical (~30 lines of
helper code + one `template <typename T>` per `_impl`).

### D. Duplicate each kernel by copy-paste with `s/__nv_bfloat16/__half/g`

Rejected. Doubles the kernel source, diverges over time, and any future
optimization has to be applied twice. The templated approach has a single
source of truth and adds zero runtime overhead — `nvcc` instantiates each
specialization independently, producing the same PTX as a hand-written
duplicate would.

## Consequences

### Positive

- `--dtype fp16` is now functional on PagedAttention paths (both V1 and V2,
  with and without ALiBi).
- Future dtype additions (e.g., FP8 storage) follow an established pattern:
  add a specialization of `to_f32`/`from_f32`, instantiate the impl, expose
  a new `extern "C"` symbol.
- Single source of truth for the algorithm — bf16 and f16 stay in lockstep
  by construction.

### Negative

- PTX file size increases (~2× for paged_attention.ptx) due to the second
  set of compiled specializations. This affects load time only, not runtime
  performance.
- One additional `#include <cuda_fp16.h>` adds ~30 ms to nvcc compile time
  for this translation unit.

### Neutral

- Quantized weight paths (AWQ, GPTQ, FP8 weights, Marlin, etc.) are
  unaffected. They live behind a dequant boundary in their respective
  `Linear` modules; Q/K/V tensors entering attention are already in the
  activation dtype (bf16 or f16).
- KV-cache quantization (`KVCacheDtype::Fp8E4m3` / `Int8`) is a separate
  path. The bf16-only constraint never extended to U8 storage; this ADR
  does not change that contract.
- MLA cache uses its own attention path and is not touched.

## Verification

- `cargo check -p vllm-core --features cuda-kernels` — recompiles
  `paged_attention.ptx`. Inspecting the resulting PTX confirms 10 entry
  points: `paged_attention_v{1,2}_{bf16,f16}{,_alibi}` (8) plus
  `paged_attention_v2_reduce_{bf16,f16}` (2).
- bf16 path regression: existing GPU tests on the bf16 path continue to
  pass with no behavior change (templatization is a pure refactor at the
  bf16 specialization).
- New GPU parity test (`paged_attention_f16_bf16_parity`, ADR-bound to the
  follow-up Commit 4) launches V1 and V2 with identical inputs in bf16 and
  fp16 and asserts output difference is within fp16-mantissa tolerance
  (~5e-2 absolute).
