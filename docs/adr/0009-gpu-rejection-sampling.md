# ADR-0009: GPU Rejection Sampling for Speculative Decoding

## Status
Accepted

## Context

Speculative decoding verification requires comparing K draft tokens against
target model logits. The naive approach transfers K+1 full logit vectors
(`K+1 * vocab_size * 4 bytes`, e.g. 2.5 MB for K=5, vocab=128K) from GPU to
CPU, then runs sequential per-position verification.

## Decision

**Hybrid GPU-CPU verification** with two fast paths and a CPU fallback:

### GPU Greedy Path
When the request uses greedy sampling with no penalties/constraints:
1. Batch argmax all K+1 positions in one GPU kernel call
2. Transfer only K+1 `u32` token IDs to CPU (20 bytes for K=5)
3. CPU compares draft tokens with target argmax (trivially fast)

### GPU Stochastic Path
When the request uses sampling with no penalties/constraints:
1. Temperature scale all K+1 positions on GPU (single broadcast multiply)
2. Batch softmax on GPU (one kernel call)
3. Gather draft token probabilities via `Tensor::gather` (K+1 floats to CPU)
4. CPU rejection test: `u < target_prob[draft]` per position (trivially fast)
5. On rejection: pull single row of probs for recovered token sampling

### CPU Fallback
When penalties, constraints, logit_bias, or other per-token state is needed:
- Sequential per-position verification (existing code, unchanged)
- Required because penalties depend on `generated_token_ids` which updates
  per accepted token

## Alternatives Considered

### Full GPU Rejection Kernel (rejected)
3000+ LOC raw CUDA implementing the complete rejection algorithm including
penalty application and recovered token sampling on GPU. Rejected because:
- The actual bottleneck is data transfer, not compute
- CPU rejection test is O(K) comparisons (~5ns)
- Penalty application requires per-token generated history (CPU state)

### CPU-Only (status quo, improved upon)
Transfers K+1 * vocab_size logits to CPU. For K=5 and vocab=128K, this is
2.5 MB per verification step. The GPU paths reduce this to 20 bytes (greedy)
or ~24 bytes (stochastic) for the common case.

## Eligibility Check

GPU path is used when all of:
- Logits tensor is on CUDA device
- No repetition/frequency/presence penalties
- No logit_bias, banned/allowed token IDs, bad_words
- No sampling constraints (regex, json, choice)
- No min_tokens suppression active

## Consequences

- **Transfer reduction**: ~100,000x for greedy (20B vs 2.5MB), ~100,000x for
  stochastic (24B + rare full-row vs 2.5MB)
- **Kernel reuse**: No new CUDA kernels needed; composes existing `gpu_argmax`,
  `gpu_softmax`, `gpu_temperature_scale`, and `gpu_top_k_top_p_sample`
- **Correctness**: GPU and CPU paths produce identical results for eligible
  requests (verified by unit tests)
- **Fallback safety**: CPU path unchanged for complex sampling configurations
