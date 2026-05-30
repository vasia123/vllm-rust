# v1.9 — GPU sampler path for constrained (structured-output) requests

Model: turboderp/Qwen3-8B-exl3 @ 4.0bpw, --enforce-eager, fp8_e4m3 KV cache.
Bench: scripts/bench_constrained_decode.{py,sh}, greedy (temperature=0),
max_tokens=80, 5 runs (+1 warmup), per-(concurrency, mode).

Constrained = `response_format: json_schema` (xgrammar), unconstrained =
identical request without the schema.

| concurrency | metric          | unconstrained | constrained | slowdown |
|-------------|-----------------|---------------|-------------|----------|
| **v1.8 (CPU fallback baseline)**                                          |
| c=1         | med tps/req     | 23.43         | 18.83       | +24.5%   |
| c=4         | med tps/req     | 18.61         | 11.20       | +66.2%   |
| **v1.9 (native GPU apply_grammar_bitmask)**                               |
| c=1         | med tps/req     | 25.21         | 23.38       | **+7.8%**  |
| c=4         | med tps/req     | 19.03         | 15.00       | **+26.9%** |

Result: constrained throughput +24% (c=1) / +34% (c=4) vs v1.8. Slowdown vs
unconstrained cut from 1.24x→1.08x (c=1) and 1.66x→1.27x (c=4).

c=1 meets the ≤10% target. The c≥2 residual is dominated by SEQUENTIAL
per-request xgrammar `fill_next_token_bitmask` (token-trie walk × batch),
confirmed by profiling (`VLLM_PROFILE_GRAMMAR=1`): at c=4 the CPU fill phase
was the largest grammar-attributable cost and scales ~linearly with batch.

## BatchMatcher parallel-fill dispatch (Phase 2.5b)

`xgrammar_rs::BatchMatcher` (thread-pool fill; FFI in csrc/xgr_c_shim +
src/matcher.rs, parity-tested in tests/batched_fill.rs) is now dispatched
from `apply_grammar_bitmask_to_logits` whenever ≥2 constrained rows share a
decode step: their non-terminated matchers are locked together and filled in
one pooled call, then scattered into the per-row bitmask slots. Single rows,
non-xgrammar constraints, and terminated matchers (xgrammar aborts on a
post-stop fill) fall back to the per-request path. Toggle:
`VLLM_DISABLE_GRAMMAR_BATCH=1`.

Same-session A/B (back-to-back, identical GPU clock; unconstrained baseline
52.3 agg in both → equal state), c=4 constrained aggregate tps, median/8 runs:

| dispatch         | constrained agg tps |
|------------------|---------------------|
| per-request      | 49.1                |
| **BatchMatcher** | **58.3 (+18.7%)**   |

Cross-RUN slowdown ratios on this laptop GPU are unreliable (unconstrained
baseline drifts 19→25 tps between sessions via clock/thermal — see memory
`perf_baseline_environmental_drift`). The same-session A/B isolates the
dispatch change and is the trustworthy figure; the constrained throughput
itself is the stable signal, not the cross-run ratio.

## Correctness

- GPU constrained path (min_p=0 → gpu_eligible): 5/5 valid schema-conforming
  JSON (stochastic temp=0.9), incl. markdown-bait prompt — `**bold**` only
  ever appears INSIDE the string value, never breaking structure.
- CPU path (min_p=0.04 → gpu-ineligible): 5/5 (unchanged from v1.8).
- 4557 core + 431 server unit tests pass.

## Known issue (pre-existing, NOT a v1.9 regression)

Greedy (temperature=0) structured decode degenerates: the model's argmax
prefers whitespace over `:` at the key/value boundary, yielding e.g.
`{"foo" 42, "bar" "abc"}` (missing colons). Reproduced identically on the
v1.8 CPU path — it is a grammar/translator interaction at the greedy
operating point, independent of the sampler device. Stochastic decode
(temp>0) is unaffected (5/5). Tracked separately from the GPU-path work.
