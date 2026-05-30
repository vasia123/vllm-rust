# ADR 0020 — GPU apply of structured-output grammar bitmask

## Status

Accepted (v1.9, 2026-05-19).

## Context

v1.8 closed Bug 10 (xgrammar JSON-Schema enforcement) but routed **every**
constrained request through the CPU sampler. The engine gate at
`crates/core/src/engine/helpers.rs` excluded any sequence carrying a
`constraint`, forcing a full `logits → host` transfer (≈ vocab×4 bytes per
sequence per step), a per-sequence CPU `mask_logits` bitmask walk, and a CPU
softmax/sort. Measured slowdown vs the unconstrained GPU sampler: +24.5%
(c=1), +66.2% (c=4) on Qwen3-8B-exl3.

The trait-level blocker was `SamplingConstraint::mask_logits(&mut [f32], _)`,
which requires a host slice. We needed a way to keep the sampler on-device
while still honoring the per-step grammar bitmask.

## Decision

Add a native CUDA kernel that applies a packed-i32 "allowed token" bitmask to
a logits tensor **in place**, setting forbidden tokens to `-inf`:

- `crates/core/kernels/apply_grammar_bitmask.cu` — `f32` / `bf16` / `f16`
  variants. Grid `(ceil(vocab/256), batch)`, block 256. Each thread masks one
  token: `if bit==0 { logits[b,t] = -inf }`. ~80 LOC, sm_80+.
- `crates/core/src/sampling/gpu.rs::gpu_apply_grammar_bitmask` — a candle
  `InplaceOp2` (receiver = logits, second operand = bitmask). Dispatches on
  the logits dtype at the `CudaStorageSlice` boundary; requires both tensors
  contiguous at offset 0.

Trait extension (additive, no signature change to `mask_logits`):
`SamplingConstraint::supports_gpu()` + `fill_cpu_bitmask_for_gpu(&mut [i32])`.
`GrammarConstraintAdapter` overrides both (xgrammar produces packed-i32 rows
natively). All other constraints inherit the `false` / `None` default and
keep their pure-CPU path.

Engine integration (`execute_batched_decode_with_graph`): the
`constraint.is_none()` gate clause becomes `constraint.map_or(true, supports_gpu)`.
When any batch member is constrained, `apply_grammar_bitmask_to_logits`
builds one `[batch, words_per_row]` i32 buffer (constrained rows filled by
the matcher, unconstrained rows all-1s, padded-vocab tail zeroed so lm_head
padding tokens are forbidden), uploads it, and applies the kernel in place
before the existing GPU sampler runs. `accept_token` is invoked post-sample
in the GPU branch, mirroring the CPU path.

The async pipelined decoder (`execute_pipelined_multi_step_decode`,
`VLLM_ASYNC_SAMPLING`) defers the sampler-token DtoH past the next forward,
so it cannot advance the matcher before the next step's bitmask. For any
constrained batch it falls back to the synchronous serial decoder, which
runs `accept_token` inline. Correctness over the async throughput win.

## Unsafe

The kernel launch in `ApplyGrammarBitmaskOp::cuda_fwd` is `unsafe` (cudarc
`builder.launch`). Justification: the kernel writes only within
`[0, batch*vocab)` (bounds-checked per thread by `token >= vocab → return`),
reads only within `[0, batch*words_per_row)` (validated against
`ceil(vocab/32)` before launch), and the receiver tensor is held mutably by
candle's `inplace_op2` RwLock. No aliasing with the bitmask (distinct
allocations). Covered by `gpu.rs` GPU smoke tests (`f32`, `bf16`).

## Consequences

- Constrained throughput +24% (c=1) / +34% (c=4) vs v1.8; slowdown vs
  unconstrained cut to +7.8% (c=1) / +26.9% (c=4). See
  `docs/perf/bench-history/2026-05-19-v19-constrained-gpu-path.md`.
- c≥2 residual was dominated by sequential xgrammar bitmask fill (profiled
  via `VLLM_PROFILE_GRAMMAR=1`). The `xgrammar_rs::BatchMatcher` thread-pool
  fill is now dispatched from `apply_grammar_bitmask_to_logits` when ≥2
  non-terminated xgrammar matchers share a step (terminated matchers excluded
  — xgrammar aborts on a post-stop fill). Same-session A/B: constrained c=4
  aggregate +18.7% (49.1 → 58.3 tps). Toggle `VLLM_DISABLE_GRAMMAR_BATCH=1`.
  Validated 20/20 concurrent constrained requests, no crash.
- The kernel allocates a fresh GPU bitmask tensor per step (no OutputPool
  reuse yet); acceptable under `--enforce-eager`, revisit if CUDA-graph
  capture of the constrained path is pursued (the buffer would need a stable
  pooled address — see ADR 0019).
- `mask_logits` is unchanged; legacy constraints (Choice/Regex/JsonSchema)
  are unaffected.

## Addendum (v1.9.1, 2026-05-27) — clean EOS termination

Greedy structured decode emitted trailing garbage after a complete document
(`{"foo": 42}!`, then a JSON-validation failure with "trailing characters").
Root cause: `GrammarCompiler` built its xgrammar `TokenizerInfo` with an empty
`stop_token_ids`. xgrammar only marks a stop token *allowed* once the grammar
reaches an accepting state, so with none configured the model's real EOS
(Qwen3 `<|im_end|>` = 151645) was forbidden at the accepting state; the only
allowed tokens were xgrammar's auto-detected markers (`</s>`, `<|endoftext|>`),
which the engine's `check_finished` does not treat as stops. Greedy could not
emit a recognised EOS → garbage. (This was previously misfiled as a
"colon-drop / translator" bug; the grammar always enforces the colon — proven
by `diag_greedy_colon_allowed_set_real_qwen3`.)

Fix: `GrammarCompiler::with_stop_tokens(vocab_index, Vec<u32>)` threads the
model EOS into the matcher; the server (`chat.rs`, `completions.rs`,
`responses.rs`) passes `state.eos_token_id`. At the accepting state the EOS is
then the sole allowed token → greedy emits it and `check_finished` stops
cleanly. Guards: `diag_eos_termination_needs_stop_token_ids` (root-cause proof,
stop=[] vs [eos]) and `with_stop_tokens_terminates_on_model_eos` (production
wiring). E2E after fix: greedy `{"foo": 42}` finish_reason=stop; GPU stochastic
5/5, CPU path 5/5, concurrent c=8 8/8.

## Addendum (v1.9.3, 2026-05-28) — EBNF string-rule fill was ~700× too slow

Investigating constrained decode perf (to gate a possible GPU bitmask-fill
port), profiling showed per-step grammar fill at 8–47 ms on Qwen3's 151k vocab
— but xgrammar's NATIVE `compile_json_schema` fills the same string state in
~15 µs. The ~1000× gap was self-inflicted by our `schema_to_ebnf` string rule,
not inherent to xgrammar (`bench_fill_native_jsonschema_vs_ebnf_string`,
`bench_string_rule_variants`). Two causes, both defeating xgrammar's FSM
fast-path and forcing an O(vocab) pushdown recompute per step:

1. A rule-REFERENCE inside the content star (`strchar ::= class | json_escape`)
   and the `"\\u" [0-9A-Fa-f]{4}` unicode-escape sub-pattern.
2. For length-bounded strings, an escape alternation inside the `{m,n}` repeat
   — slow per fill AND, inlined, xgrammar expands it n× and the compiled
   grammar blows up (CUDA OOM on nested array-of-strings schemas).

Fix (`emit_string`): emit the string content as a regular FSM.
- Unbounded (`*`/`{m,}`): inline the escape alternation
  `("\"" ([class] | "\\" ["\\/bfnrt])* "\"")` — a star is one FSM state, so
  escapes ride along; ~50 µs/fill. The `\uXXXX` branch is dropped (raw UTF-8 is
  admitted directly, so non-ASCII is emitted literally, not escaped).
- Bounded (`{m,n}`): a PURE char-class repeat `"\"" [class]{m,n} "\""` — small
  counter-FSM, ~260 µs/fill, no grammar blowup. Bounded strings drop the
  `\`-escape branch (stricter, still valid JSON).

This makes the grammar more restrictive (cannot emit literal `\u`/`\`-escapes in
some positions), never less — every produced string is valid JSON, and models
emit UTF-8 directly. **It removes the motivation for a GPU fill port (Track C)
for the common string-schema case** — the bottleneck was our EBNF. Guards:
`emit_unbounded_string_inlines_escape_branch`,
`emit_bounded_string_uses_pure_class_no_escape`, plus the existing CONCEPT_SCHEMA
length/boundary enforcement tests. E2E: a nested array-of-bounded-strings schema
went from CUDA-OOM / 14 s-truncation to 2.54 s valid; unbounded string decode is
forward-bound (fill ~µs); GPU/CPU 5/5, greedy c=8 8/8, 0 shape/OOM. core 4476 +
61 grammar tests pass.
