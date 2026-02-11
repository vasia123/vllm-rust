# ADR-0007: GPU Sampling Kernels

## Status
Implemented

## Context
During decode, the engine transfers the full `[batch_size, vocab_size]` logit matrix
from GPU to CPU every step via `Tensor::to_vec2::<f32>()`. For vocab_size=128K and
batch_size=32, that's ~16 MB per step over PCIe — a significant bottleneck for
throughput. CPU-side sampling then performs argmax or top-k/top-p on this data.

## Decision
Implement custom CUDA kernels that perform the entire sampling pipeline on GPU,
transferring back only `[batch_size]` token IDs (128 bytes for batch 32) instead
of the full logit matrix.

### Architecture

```
GPU Path (eligible sequences):
  logits [B, V] → temp scale → softmax → top-k/top-p filter → sample → ids [B]
                                                                          ↓
                                                                     to_vec1() (B × 4 bytes)

CPU Path (fallback for complex sampling):
  logits [B, V] → to_vec2() (B × V × 4 bytes) → penalties → temp → ... → sample
```

### GPU Eligibility
A sequence uses the GPU fast path when ALL of:
- No repetition/frequency/presence penalties
- No min_p or typical_p filtering
- No logit bias or banned/allowed token IDs
- No beam search
- No logprobs needed
- No sampling constraints (structured output)
- No bad words

If any sequence in the batch requires CPU-only features, the entire batch falls
back to the CPU path. This is the common case optimization — most production
workloads use greedy or simple temperature+top_k+top_p.

### CUDA Kernels
Located in `crates/core/kernels/sampling.cu`:

1. **argmax_bf16 / argmax_f32**: Warp-level parallel reduction, one block per sequence.
   256 threads stride-access the vocab dimension, warp shuffle reduces, shared memory
   for inter-warp final reduction.

2. **softmax_to_probs / softmax_to_probs_f32**: 3-pass online softmax
   (find max, compute exp+sum, normalize). Warp shuffle + shared memory reductions.

3. **top_k_top_p_sample_per_seq**: Single-threaded per sequence (parallelism from grid).
   Per-sequence top_k/top_p arrays loaded from GPU buffers. Iterative k-th element
   finding, cumulative top-p filtering, multinomial sampling from filtered distribution.

4. **temperature_scale_f32_per_seq**: Per-sequence inverse temperature scaling.

### Rust Integration
Uses candle's `CustomOp1`/`CustomOp2` trait for CUDA kernel dispatch:
- `ArgmaxOp`: logits → token IDs via `apply_op1_no_bwd`
- `SoftmaxOp`: logits → F32 probabilities via `apply_op1_no_bwd`
- `TopKTopPSampleOp`: probs + rand → token IDs via `apply_op2_no_bwd`

Temperature scaling uses candle's `broadcast_mul` (no custom kernel needed).

The `gpu_sample_batch()` function orchestrates the pipeline with a fast path for
all-greedy batches (single argmax kernel, no softmax/sampling).

### Engine Wiring
In `engine/helpers.rs::execute_batched_decode_with_graph`, before the existing
`to_vec2()` transfer, we check GPU eligibility. If all sequences qualify, we
call `gpu_sample_batch()` and return early. Otherwise we fall through to the
existing CPU path.

Random values for stochastic sampling are pre-generated from each sequence's
`SamplerState` RNG on CPU (ensuring seed reproducibility) and passed as a parameter
vector to the GPU kernel.

## Consequences
- Decode throughput improved for GPU-eligible batches (no PCIe logit transfer)
- CPU path unchanged — zero regression for complex sampling
- `SamplingParams::gpu_eligible()` makes eligibility explicit and testable
- CUDA kernels only compiled when `cuda-kernels` feature is enabled
- All existing tests pass; 25+ new tests for GPU sampling and eligibility
- Single-threaded top-k/top-p kernel is O(k × V) — acceptable for typical k < 100,
  but future work could use radix select for O(V) regardless of k
