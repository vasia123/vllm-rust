# ADR-0005: DraftProposer Abstraction for Speculative Decoding

## Status

Accepted

## Context

The speculative decoding execution strategy (`SpeculativeExecution`) was monolithic: it owned both the target and draft models, managed two KV caches, tracked per-request draft state on `ActiveRequest`, and performed the entire draft-verify-rollback loop inline. This made it impossible to swap draft strategies (GPU draft model, n-gram, suffix array, Eagle, Medusa) without duplicating the verification logic.

Python vLLM v1 addressed this with a proposer pattern (`SpecDecodeBaseProposer`) that separates draft token generation from verification. We need a similar abstraction in Rust.

## Decision

Introduce a `DraftProposer` trait that encapsulates draft token generation and its lifecycle:

```rust
pub trait DraftProposer: Send {
    fn init_request(&mut self, request_id: RequestId, prompt_tokens: &[u32]) -> Result<(), EngineError>;
    fn propose_for_request(&mut self, request_id: RequestId, last_token: u32, state: &mut SequenceState, tokenizer: &TokenizerWrapper) -> Result<Vec<u32>, EngineError>;
    fn on_tokens_verified(&mut self, request_id: RequestId, num_accepted: usize, original_offset: usize) -> Result<(), EngineError>;
    fn finish_request(&mut self, request_id: RequestId) -> Result<(), EngineError>;
    fn preempt_request(&mut self, request_id: RequestId) -> Result<(), EngineError>;
    fn num_speculative_tokens(&self) -> usize;
    fn name(&self) -> &str;
}
```

### Key design choices

1. **`&mut self` instead of interior mutability**: The engine loop is single-threaded, so `Mutex` is unnecessary overhead. `&mut self` makes ownership clear and avoids poisoned lock issues.

2. **Per-request lifecycle hooks**: `init_request` / `finish_request` / `preempt_request` give proposers control over resource allocation (KV cache blocks, suffix arrays, etc.) without leaking internal state to the engine.

3. **`SequenceState` access**: `propose_for_request` receives `&mut SequenceState` so proposers can apply sampling params (penalties, constraints, min_tokens) during draft generation, matching the verification pipeline.

4. **`Result`-based errors**: All methods return `Result<_, EngineError>` for proper error propagation, unlike the legacy `SpeculativeProposer` which silently returned empty vectors.

5. **Coexistence with legacy trait**: `SpeculativeProposer` remains for standalone usage and testing. Types can implement both traits (e.g., `NGramProposer` implements both).

### Implementations

| Proposer | Type | Has per-request state | KV cache |
|----------|------|----------------------|----------|
| `DraftModelDraftProposer` | GPU | Yes (block tables, offsets) | Own `KVCacheManager` |
| `NGramProposer` | CPU | No (lifecycle = no-ops) | None |
| `SuffixArrayProposer` | CPU | No (lifecycle = no-ops) | None |
| `EagleProposer` | GPU | Future | Future |
| `MedusaProposer` | GPU | Future | Future |

### What changed

- `SpeculativeExecution<M>` now takes `Box<dyn DraftProposer>` instead of owning a draft model and KV cache directly.
- `DraftState` removed from `ActiveRequest` — proposers manage their own per-request state internally.
- `start_engine_with_proposer()` added as the general entry point; `start_engine_with_draft()` constructs a `DraftModelDraftProposer` internally.

## Consequences

**Benefits:**
- Any `DraftProposer` implementation can be plugged into `SpeculativeExecution` without modifying the verification logic.
- CPU proposers (n-gram, suffix array) can now be used with the full speculative decode pipeline including target model verification.
- Draft model state is fully encapsulated — no cross-concern leakage to `ActiveRequest`.

**Trade-offs:**
- Two proposer traits coexist (`DraftProposer` and `SpeculativeProposer`). The legacy trait is retained for backward compatibility and standalone usage patterns.
- Eagle and Medusa adaptation is deferred — they require base model hidden state access that the current trait interface doesn't provide.

**Future work:**
- Adapt Eagle/Medusa to `DraftProposer` with hidden state passing.
- Batched proposal interface for parallel drafting (multiple requests in one forward pass).
- Async scheduling with CUDA stream coordination.
