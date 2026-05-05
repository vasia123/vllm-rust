# ADR-0013: Defer Registry-Based Dispatch (Phase 9)

## Status

Accepted (deferral)

## Context

The original architecture refactor plan included a Phase 2 / Phase 9
"registry-based dispatch" that would replace the 11 `from_config*`
match-arm dispatch functions in `crates/core/src/models/mod.rs` (2.6k
LOC, ~190 model arms) with:

- An `ArchFactory` trait (`build`, `build_quant`, `build_with_tp`,
  `build_with_pp`, `build_with_lora`, `build_encoder_decoder`,
  `as_speculative`).
- A `phf::Map<&'static str, &'static dyn ArchFactory>` registry
  populated with one entry per architecture name (~560 entries
  including aliases).
- ~190 mechanical factory files in `crates/core/src/models/factories/`.

The plan estimated **+3050 net LOC** (the registry adds more code than
it removes from `mod.rs`) and ~30 PRs of mechanical work. The benefit:
`models/mod.rs` shrinks from 2.6k LOC to ≤300 LOC, and adding a new
model edits 3 files instead of 7.

## Decision

**Defer the registry-based dispatch migration indefinitely.** The
`model-registry-v2` feature flag and `phf` workspace dependency stay
in place — the path remains open if a future requirement justifies
the work — but no further migration will land under autonomous
execution.

### Reasoning

1. **No functional debt.** The existing match-arm dispatch is correct,
   well-tested (4400+ tests), and runs zero dynamic dispatch in the
   hot path. The proposed registry would dispatch through `&dyn
   ArchFactory` at construction only, matching the current factory
   pattern's runtime characteristics exactly.

2. **Net LOC churn is positive (+3050).** Unlike Phase 4 (−15000 LOC)
   or Phase 7 (−2500 to −3500 LOC), Phase 9 is structural reshuffling.
   Each factory file adds ~30–80 LOC of trait-impl boilerplate; the
   `mod.rs` shrinkage doesn't compensate.

3. **30+ PRs of mechanical review.** Reviewers face the same work as
   ~190 model migrations did in Phase 4b — but with no
   correctness-uncovering benefit (Phase 4b found a Molmo2 `o_proj`
   bug; Phase 9 wouldn't surface anything because the math is
   unchanged).

4. **Plugin-architecture motivation doesn't apply.** Registry dispatch
   shines when models can be registered from external crates (vLLM
   Python's plugin system). vllm-rust is monolithic; the model files
   are part of `vllm-core` and grow with the project. The registry
   would remain entirely intra-crate.

5. **The "3 places vs 7 places" win is partial.** The current
   add-a-model touches `models/<name>.rs`, `models/mod.rs`
   (`pub mod` + match arm), and registry aliases. That's 3 places
   already, just not 3 *factory* files. The Phase 9 design changes
   *which* files but not the count.

6. **Discipline-by-test already in place.** ADRs 0010, 0011, 0012 plus
   the `no_local_attention.rs`, `no_local_vision_tower.rs` guardrails
   give the project a "single source of truth for what's bespoke and
   why" without registry dispatch. The architectural-discipline goal
   is met.

### What stays

- The `model-registry-v2` feature flag in `crates/core/Cargo.toml`
  (currently inactive; populates 0 entries).
- The `phf` workspace dependency, gated behind the flag.
- `crates/core/tests/registry_completeness.rs` (currently `cfg`-gated
  and empty in the live build), as a placeholder for the future
  invariant test.
- The plan document
  (`/home/vasis/.claude/plans/bubbly-hugging-brooks.md`) records the
  full Phase 9 design under the "EXTENSION (2026-05-06)" section, so
  a future contributor can pick up the work without re-deriving it.

### What's removed

Nothing is being removed. The deferral is a no-op against the current
codebase.

## Consequences

**Benefits:**

- Avoids +3050 LOC of mechanical churn that would not improve
  correctness or performance.
- Avoids 30+ PRs of review fatigue on a refactor with no
  externally-visible benefit.
- Keeps the door open: the feature flag, dependency, and design doc
  are all intact for a future contributor who needs the registry
  (e.g., for an out-of-tree model plugin).

**Trade-offs:**

- `models/mod.rs` stays at 2.6k LOC. The original plan's "≤300 LOC"
  target for that file is not met. The file remains a long match-arm
  dispatch, which is verbose but readable and easy to grep.
- Adding a new model still requires editing `mod.rs` (one match-arm
  per dispatch fn the model needs to expose). The
  `docs/adding-a-model.md` guide already documents this.

**When to revisit:**

- If vllm-rust ever supports out-of-tree model plugins.
- If `models/mod.rs` becomes a merge-conflict hotspot (per-arm edits
  conflict; per-file factory entries don't).
- If the contributor base grows large enough that "fewer-files-to-touch
  per model" overcomes the +3050 LOC review cost. The current
  contributor count makes this trade-off favour the status quo.

## Pairs with

- ADR-0010 (AttentionBlock) — the structural refactor that *did*
  earn its LOC cost (−15000 LOC of duplicated attention scaffolding).
- ADR-0011 (MoE infrastructure) — the canonical MoE primitive
  contract.
- ADR-0012 (vision tower) — the analogous "discipline without
  forced collapse" decision for vision encoders.

Together, ADRs 0010–0013 record the project's stance on the
"shared-abstraction earns its cost only when it pays back" principle.
Phase 9 is the case where the cost would not pay back at the current
project shape.
