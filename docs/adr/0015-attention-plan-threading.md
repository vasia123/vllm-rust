# ADR 0015 — Attention plan threading (Stage 14-A)

## Status

Proposed. Stage 13 closed without this work; the perf gap to Python
vLLM (5.5× at c=8 aggregate) is large enough that this is the next
ROI step before the much larger Marlin tile-MMA milestone.

## Context

Both halves of the FlashInfer attention path — prefill via
`BatchPrefillPlan` and decode via `BatchDecodePlan` — currently
rebuild their CPU-side plan on **every** attention layer:

- `paged_attention()` (`crates/core/src/layers/attention/mod.rs`)
  calls `select_backend()` (which constructs a fresh
  `Box<FlashInferBackend>` every call) then
  `backend.prefill_attention(...)`. Inside, `prefill_flashinfer`
  builds a fresh `PrefillWrapper` and calls `.run(...)`, which
  internally constructs a `BatchPrefillPlan` and immediately runs it.
  That plan-build does (a) a host CPU work-estimate, (b) a
  page-locked metadata allocation, (c) a `cudaMemcpyAsync` to stage
  metadata onto the device. For a 36-layer Qwen3-4B, that's 36×
  per prefill.
- `batched_paged_attention_decode()` (same file) does the same: fresh
  `Box<FlashInferBackend>`, then `backend.batched_decode_attention(...)`,
  which always rebuilds a `BatchDecodePlan`.
- `AttentionBackend` already exposes `prepare_decode_plan` +
  `batched_decode_attention_with_plan` (default impls in
  `backend.rs:152-200`, real impls for FlashInfer at
  `flashinfer/mod.rs:610` and `:635`). **No production caller uses
  them.** They're dormant infrastructure.

Stage 13-K profiled this on a 489-token prefill: paged_attn took
67.8 % of the forward wallclock; the dominant component inside it
was the per-layer plan rebuild plus its host syncs.

Stage 13-K-bis tried a singleton-static cache (one slot, keyed by
`(block_ids hash, seq_len, kv layout, dtype, head dims)`) inside
`prefill_flashinfer`. Short single-prefill cases (9-token prompt)
PASSED — cache trace showed 1 MISS + 35 HITS, output identical to
baseline. Sustained / long-context / concurrent paths FAILED with
`CUDA_ERROR_ILLEGAL_ADDRESS` during `BatchPrefillPlan::run`. Root
cause: a static cache survives across forwards, but the FlashInfer
workspace pool that the cached plan's internal metadata points into
gets rewritten by intervening decode plans between any two prefills.
The next matching-key prefill replays a plan whose workspace pointers
now point at garbage. Reverted; documented in
`docs/perf/qwen3-4b-awq-profile.md` § Stage 13-K-bis.

This ADR records the design that should land instead.

## Decision (proposal)

Plan caching is per-forward, not per-process. The engine builds the
plan **once before the model.forward call**, threads it through to
each attention layer, and drops it when the forward returns.

### Shape A — explicit threading via metadata

Add an opaque plan slot to the existing metadata structs that already
flow through every attention call:

```rust
// crates/core/src/layers/attention/backend.rs
pub struct PagedAttentionMetadata<'a> {
    pub block_ids: &'a [BlockId],
    pub slot_mapping: &'a [usize],
    pub seq_len: usize,
    pub seqlen_offset: usize,
    /// Pre-built `BatchPrefillPlan`, threaded from the engine.
    pub attention_plan: Option<&'a (dyn std::any::Any + Send + Sync)>,
}

pub struct BatchedDecodeMetadata<'a> {
    /* existing fields */
    pub attention_plan: Option<&'a (dyn std::any::Any + Send + Sync)>,
}
```

Engine code (`engine/standard.rs::prefill_step` and the analogous
decode path) builds the plan via
`backend.prepare_prefill_plan(...) / prepare_decode_plan(...)` once,
puts a borrowed reference in the metadata, then calls
`model.forward(...)`. The plan is dropped at end of step.

Inside the model — *no* code changes for the 76 callers. The two
helper wrappers (`paged_attention`, `batched_paged_attention_decode`
in `attention/mod.rs`) read `metadata.attention_plan` and dispatch
to `_with_plan` variants when present, falling back to the eager
path when `None` (which keeps the old behaviour for callers who
don't yet thread a plan).

The two helpers' public signatures don't change — they currently
build their own metadata internally. To carry the plan they instead
take a `metadata: &PagedAttentionMetadata` argument; this is a
breaking change touching ~76 call sites across `crates/core/src/models/*`.
Each call site changes from:

```rust
paged_attention(&q, &k, &v, mask, seqlen_offset, cache_engine,
                block_table.block_ids(), slot_mapping,
                num_heads, num_kv_heads, head_dim)
```

to:

```rust
let metadata = PagedAttentionMetadata::new(
    block_table.block_ids(), slot_mapping, seqlen_offset, q.dim(2)?,
    /* attention_plan = */ None,
);
paged_attention(&q, &k, &v, mask, cache_engine,
                &metadata, num_heads, num_kv_heads, head_dim)
```

… or we keep the eager helper unchanged and add a parallel
`paged_attention_with_metadata` for the new path. The latter is
non-breaking.

### Shape B — per-key plan with a private workspace slice

Don't change any APIs. Inside `FlashInferBackend`, maintain a
`Mutex<HashMap<CacheKey, CachedPlan>>` plus a small slab allocator
on top of the existing `WorkspaceBuffer`. Each `CachedPlan` owns a
*private* slice of workspace that no other plan writes to —
eliminating the cross-forward corruption that broke Stage 13-K-bis.

Cost: doubles workspace footprint (currently 128 MiB → potentially
2-4× under churn), and the slab allocator is non-trivial. Win:
zero-touch for engine and model code.

### Recommended

**Shape A (explicit threading).** Mirrors the established
`prepare_decode_plan` + `_with_plan` pattern, has no extra VRAM cost,
and the plan-lifetime story is provably correct (drop on forward
return = no aliasing with subsequent forwards' workspace use). The
~76-call-site breakage is bookkeeping; sed + tests verify mechanically.

If the call-site sweep proves to be a sustained maintenance burden
(e.g. it conflicts repeatedly with model adds in active development),
Shape B becomes the fallback.

## Implementation order

1. **Lock down dormant decode path** — add unit tests
   `test_decode_with_plan_matches_eager_gpu` and
   `test_decode_with_plan_none_falls_through_gpu`
   (`crates/core/src/layers/attention/flashinfer/mod.rs`,
   `gpu-test-small` feature). **Done — Stage 14-A.0 (commit 2e0e829).**
2. Add `prepare_prefill_plan` to `AttentionBackend` (default `None`),
   plus `prefill_attention_with_plan` (default falls through).
   Implement both in `FlashInferBackend` reusing the work already in
   `prefill_flashinfer`. **Done — Stage 14-A.0a (commit 0f7be25).**
2.5. **Backend-singleton precondition** — currently `select_backend()`
   constructs a fresh `Box<FlashInferBackend>` on every helper call,
   so the workspace inside it dies with the box. A plan returned by
   `prepare_prefill_plan` holds pointers into that workspace; once
   the backend drops the pointers are dangling.

   **Failed approach (Stage 14-A.0b, reverted commit ffc6331):** lift
   `workspace` to a `static OnceLock<Mutex<…>>`. Single-thread tests
   passed, but `cargo test --jobs N` runs FlashInfer GPU tests in
   parallel; concurrent `ensure_size` resizes deallocated the
   underlying CudaSlice while one test still held a plan pointing
   into it → 10/45 GPU tests crashed. The singleton is unsafe under
   concurrent access regardless of the plan-caching shape.

   The right fix is to make **the backend itself** a singleton via
   `OnceLock<Box<dyn AttentionBackend>>` returned by `select_backend()`
   — then the workspace stays per-backend (so each backend instance
   has its own workspace and tests don't collide), but the lone
   live instance outlives any plan that flows through it. Single-
   threaded engine forward gets serial access through the existing
   workspace `Mutex` inside the singleton backend.

   Implementation is small: change `select_backend()` return from
   `Box<dyn …>` to `&'static dyn …` (or `Arc<dyn …>`), update the
   two callers in `attention/mod.rs`. Existing per-backend workspace
   lock keeps the engine-forward path serialized; tests keep their
   instance-bound workspaces by constructing `FlashInferBackend::with_block_size(...)`
   directly (which still works — just doesn't go through the
   `select_backend()` singleton path). **Done — Stage 14-A.0c
   (commit ee7e85b).**

3. **Engine wiring via TLS (FAILED, reverted).** With the singleton
   in place, the natural plan-threading shape is a thread-local
   `PrefillPlanScope` RAII guard entered before `model.forward(...)`
   in `engine::helpers::execute_prefill`. The scope's first
   `paged_attention()` call lazily builds a plan and stows it; the
   remaining N-1 attention layers reuse it. Drop clears.

   Implementation landed cleanly (~150 lines in
   `layers/attention/mod.rs` + a 3-line wrap in
   `engine/helpers.rs::execute_prefill`). 4497 lib tests + 45
   FlashInfer GPU tests + 7/7 e2e correctness PASSED. Looks great
   on paper.

   **`bench_prefill.py` showed a 240 → 336 ms TTFT regression at
   `prompt_len=256` (5-run p50 stable).** The same prompt without
   the TLS scope active comes back at 240 ms baseline. So the win
   path the ADR predicted (saving 35× `BatchPrefillPlan::new` cost
   per forward → ~190 ms TTFT) is *backwards* on this stack: the
   plan-cached path is **slower**, not faster.

   The build/run split inside `BatchPrefillPlan` on the FlashInfer
   side must be cheaper than the model-forward analysis assumed —
   most of the per-layer wallclock isn't plan-build, it's the
   actual attention kernel + KV-cache write + Q/K/V projections.
   Saving the plan-build ~doesn't move the needle, and even the
   small overhead of TLS lookup + `Arc::clone` + downcast on every
   layer call eats it.

   **Reverted** the engine wiring (helpers.rs + the `PrefillPlanScope`
   plumbing in `attention/mod.rs`). Foundation kept:
   - `prepare_prefill_plan` / `prefill_attention_with_plan` API
     surface remains in the `AttentionBackend` trait + FlashInfer
     impl (Stage 14-A.0a).
   - `select_backend()` singleton stays (Stage 14-A.0c).
   - GPU lock-down tests for both prefill and decode plan paths
     stay locked-in.

   **Lesson recorded for future plan-caching attempts on this stack:**
   profile before believing the 13-K.1 audit's interpretation. The
   audit said "67.8 % of forward in `paged_attn`, dominated by
   per-layer plan build" — but the breakdown was measured under
   `VLLM_PROFILE_PREFILL=1` which adds a `cuda_stream::synchronize()`
   per measurement, inflating ratios non-uniformly. Real wall-clock
   shows the kernel itself dominates. Future ROI work in this area
   should target the kernel (Stage 13-L Marlin tile-MMA) or the KV-
   cache write path, not plan caching.

   Stage 14-A status: **CLOSED**, no engine wiring beyond foundation.
   Next time this is revisited, the foundation API surface can be
   used immediately — but only if a new measurement shows
   plan-build is actually meaningful in the workload.
3. Add `attention_plan: Option<&dyn Any>` to `PagedAttentionMetadata`
   and `BatchedDecodeMetadata`; default-construct as `None` everywhere
   so existing callers don't change.
4. In `paged_attention` / `batched_paged_attention_decode` helpers,
   route to `_with_plan` when metadata carries a plan, eager otherwise.
5. Wire engine: in `engine/standard.rs::prefill_step` (and the decode
   analogue), call `backend.prepare_prefill_plan(...)`, build
   `metadata` with the plan ref, pass into `model.forward(...)`.
6. Rust trait/method visibility + lifetimes: the plan is borrowed for
   the forward call only; engine owns it via local `let plan =` and
   passes `&plan` reference into metadata.
7. Bench: `bench_prefill.py` 64 / 256 / 1024 / 2048 prompts. Target:
   TTFT_256 252 → ≤ 200 ms (saving 35× plan-build cost on short
   prompts; smaller relative win on long prompts where the kernel
   itself dominates). Decode bench: c=8 aggregate, expect +5–10 %
   from removing 35× plan rebuild per token.

## Risks

- **Lifetime gymnastics.** Engine builds plan, hands `&plan` to model
  via metadata, model passes through to attention layer. The `'a`
  lifetime threading needs to flow through `forward_decode_batch_with_*`
  signatures cleanly. If it gets ugly, fall back to `Arc<dyn Any>`
  (slight runtime cost — atomic refcount per forward — but no lifetime
  wrestling).
- **Pipeline-parallel / TP boundaries.** Plan is bound to a single
  device's workspace. Multi-device plans need per-rank plan builds —
  fine because each rank has its own engine instance.
- **CUDA Graph capture.** If we re-enter capture mode in the future
  (Stage 13-A is V3-blocked but might unblock), the plan can be
  captured into the graph once and replayed — works in our favour.

## Out of scope

- Marlin tile-MMA kernel (Stage 13-L) — much bigger refactor.
- `prepare_prefill_plan` for the naive backend — fallback path
  always returns `None`, no plan to cache.
- CPU offload integration — plan is GPU-only; CPU offload wraps GPU
  cache movement, doesn't touch the attention plan.

## References

- Stage 13-K.1 audit, `docs/perf/qwen3-4b-awq-profile.md`.
- Stage 13-K-bis attempt + revert, same doc.
- Decode plan dormant infra: `backend.rs:152-200`,
  `flashinfer/mod.rs:610-680`.
- Decode plan correctness lock-down (this milestone):
  `flashinfer/mod.rs::test_decode_with_plan_*_gpu`.
