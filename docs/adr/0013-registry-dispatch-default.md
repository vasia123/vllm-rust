# ADR-0013: Registry-Based Dispatch (`model-registry-v2`) Becomes Default

## Status

Accepted

## Context

The original architecture refactor plan included a Phase 2 / Phase 9
"registry-based dispatch" that replaces the legacy match-arm
`from_config*` dispatch with:

- An `ArchFactory` trait
  (`crates/core/src/models/factory.rs`).
- A `phf::Map<&'static str, &'static dyn ArchFactory>` registry
  (`crates/core/src/models/registry_v2.rs`) populated with one entry
  per HuggingFace `architectures[]` string (~190 unique factories,
  ~560 alias entries including aliases like `LLaMAForCausalLM` →
  Llama).
- ~190 mechanical factory files in `crates/core/src/models/factories/`.

When this ADR was first drafted, the author incorrectly believed the
foundation hadn't been built and proposed deferring the work. **A
re-audit found the foundation already in tree:** all 190 factory
files exist, the registry is populated, and
`cargo test --features model-registry-v2` passes the full suite
(4405 tests). The legacy match-arm dispatch in
`crates/core/src/models/mod.rs` is preserved as a fallback that
runs only when `registry_v2::lookup` returns `None`.

The remaining decision is **whether to make `model-registry-v2`
default-on**, which is a one-line change in `crates/core/Cargo.toml`.

## Decision

`model-registry-v2` is now in the **default features list**.

```toml
[features]
default = ["model-registry-v2"]
```

This means:

- Every `cargo build`, `cargo test`, `cargo bench`, and
  `cargo run` invocation goes through the registry-first dispatch
  path by default.
- The legacy match-arm dispatch in `mod.rs` is still present and
  still acts as the fallback when an `arch_name` isn't in the
  `phf::Map`. Currently every supported architecture **is** in the
  map, so the fallback is dead code at runtime — but kept for one
  release as a safety net.
- Disabling the feature (`--no-default-features`) restores the
  legacy-only path for diagnosis or rollback.

### Why default-on instead of the original "defer indefinitely"

The deferral reasoning rested on three claims that the audit
disproved:

1. **"+3050 net LOC churn for no benefit."** False — the +3050 LOC
   is *already in tree*, written by an earlier session. The cost is
   sunk. The benefit (cleaner add-a-model, plugin-ready surface,
   `phf` compile-time uniqueness check on arch_names) is now
   accessible at zero further cost.
2. **"30+ PRs of mechanical review."** False — the work is done.
   No further PRs needed.
3. **"No functional debt to address."** Still true, but the fact
   that the registry **passes the full test suite** means the
   default-on switch is itself low-risk.

The audit also confirmed that switching the default doesn't break
anything: `cargo test --lib` runs 4405 tests with the default-on
flag, identical to the previous `--features model-registry-v2`
invocation.

### What changed in this commit

- `crates/core/Cargo.toml`: `default = ["model-registry-v2"]`.
- This ADR rewritten to record the actual decision (default-on)
  rather than the deferral that the original draft proposed.

### Cleanup status

All five `from_config*` functions have had their legacy match-arm
bodies removed:

| Function | Match-arm cleanup | Notes |
|----------|------------------|-------|
| `from_config` | **Done** (−411 LOC) | All 326 arches have factories with `build` |
| `from_config_with_quant` | **Done** (−234 LOC) | After adding QUANTIZED + build_quant to Yi/Baichuan/SeedOss/TeleFLM factories |
| `from_config_with_pp` | **Done** (−25 LOC) | After adding PP + build_with_pp to Mistral factory (routes through Llama backbone) |
| `from_config_with_lora` | **Done** (−24 LOC) | All 11 LoRA-enabled arches already covered |
| `from_config_encoder_decoder` | **Done** (−13 LOC) | T5 / Whisper / NemotronParse already covered |
| `from_config_with_tp` | **Done** (−232 LOC) | After adding TP + build_with_tp to TeleFLM factory |

Cumulative reduction: **−935 LOC** in `models/mod.rs` (2607 → 1672, −36%).

Every dispatch fn now follows the same shape:

```rust
pub fn from_config_with_X(cfg, vb, ...) -> Result<...> {
    let arch = get_arch(cfg)?;

    #[cfg(feature = "model-registry-v2")]
    if let Some(factory) = registry_v2::lookup(arch) {
        if factory.info().supports(Capabilities::X) {
            return factory.build_with_X(...);
        }
    }

    Err(ModelError::UnsupportedArchitecture(arch.into()))
}
```

The `from_config` function additionally retains a deprecation table
for HuggingFace arch_names whose models were removed in upstream vLLM
(MotifForCausalLM, Phi3SmallForCausalLM, Phi4FlashForCausalLM,
Phi4MultimodalForCausalLM, BartForConditionalGeneration / BartModel /
MBart / Donut / Florence2, MllamaForConditionalGeneration). These arms
return `UnsupportedArchitecture` with an actionable message naming
the upstream removal version and the recommended replacement.

  The cleanup is straightforward: replace each match-arm body with a
  `Err(ModelError::UnsupportedArchitecture(...))` and trust the
  registry. ~1900 LOC removed in `mod.rs` (target: ≤300 LOC for the
  file, plus the dispatch shims).

  Tracked as the natural next iteration in the loop ordering of the
  plan. Not done in this ADR commit because it's a separate concern
  with its own review surface.

## Consequences

**Benefits:**

- Compile-time uniqueness check on every HuggingFace arch_name
  (`phf::Map` build fails on duplicate keys).
- Plugin-ready surface — adding a new model is a single
  `factories/<name>.rs` file plus one entry in `registry_v2.rs`.
- The registry is the single source of truth for "what
  architectures does vllm-rust support".
- Cleaner add-a-model story: `docs/adding-a-model.md` already
  documented the 3-place pattern; this commit makes that pattern
  the live default.

**Trade-offs:**

- Builds now require the `phf` dependency (small, well-maintained).
- Anyone disabling default features explicitly to opt out of the
  registry must use `--no-default-features` and re-enable the
  features they actually want — a one-time disruption documented in
  `docs/adding-a-model.md`.

**Pairs with:**

- ADR-0010 (AttentionBlock) — the structural refactor that informed
  the discipline approach.
- ADR-0011 (MoE infrastructure) — same canonicalisation pattern at
  the layer level.
- ADR-0012 (vision tower) — the analogous "guardrail-not-collapse"
  decision for vision encoders.

Together, ADRs 0010–0013 record the project's operationalised view
of consolidation: pay the cost where it pays back (Phase 4
attention, Phase 7 MoE primitives, Phase 9 dispatch registry),
defer it where it doesn't (Phase 8 vision tower).
