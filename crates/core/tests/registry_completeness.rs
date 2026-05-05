//! Phase 2 invariant: every HuggingFace `architectures[]` string used
//! by the legacy match-arm dispatch in `crates/core/src/models/mod.rs`
//! must resolve through `models::registry_v2::lookup`.
//!
//! Catches drift between the two paths during the migration window:
//! if someone adds a new arm to `from_config*` without registering
//! the factory, this test fails — preventing the "silently routes
//! through legacy fallback" failure mode.
//!
//! Maintained as data, not as a regex over `mod.rs`: the expected
//! list lives below and is updated when arch_names are added or
//! removed deliberately. Phase 6 deletes the legacy match-arms and
//! this test in turn.

#![cfg(feature = "model-registry-v2")]

use vllm_core::models::registry_v2;

/// Sample of arch_names that exercised every dispatch function in
/// Phase 2. Not exhaustive — the full ~560-name list is harder to
/// keep stable across HF aliases — but covers each capability path.
const REQUIRED_ARCHES: &[&str] = &[
    // Plain build (any).
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma4ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Phi3ForCausalLM",
    // Llama aliases (each must point to the same factory).
    "LlamaModel",
    "LLaMAForCausalLM",
    "AquilaForCausalLM",
    "InternLMForCausalLM",
    "OlmoForCausalLM",
    "SmolLM3ForCausalLM",
    // Multimodal.
    "Gemma4ForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "LlavaForConditionalGeneration",
    // Encoder-decoder.
    "T5ForConditionalGeneration",
    "WhisperForConditionalGeneration",
    // Embedding / classification.
    "BertModel",
    "BgeM3EmbeddingModel",
];

#[test]
fn every_required_arch_resolves() {
    let mut missing = Vec::new();
    for arch in REQUIRED_ARCHES {
        if registry_v2::lookup(arch).is_none() {
            missing.push(*arch);
        }
    }
    assert!(
        missing.is_empty(),
        "registry_v2 missing {} required arch(es): {:?}",
        missing.len(),
        missing,
    );
}

#[test]
fn registry_has_a_meaningful_size() {
    // Phase 2 lifted ~560 unique aliases from the legacy dispatch.
    // The exact number drifts as we add / remove archs; just guard
    // against accidental wipe (e.g. a generator regression).
    let n = registry_v2::len();
    assert!(
        n >= 250,
        "registry_v2 has only {n} entries — expected ≥250 after Phase 2"
    );
}

#[test]
fn llama_aliases_share_the_same_factory() {
    // 17 Llama aliases all point at the canonical factory's
    // `&FACTORY`. Compare via TypeId on `as_any()`.
    let canonical =
        registry_v2::lookup("LlamaForCausalLM").expect("LlamaForCausalLM must be registered");
    let canonical_id = canonical.as_any().type_id();

    for alias in [
        "LlamaModel",
        "LLaMAForCausalLM",
        "AquilaModel",
        "AquilaForCausalLM",
        "InternLMForCausalLM",
        "InternLM3ForCausalLM",
        "SolarForCausalLM",
        "OrionForCausalLM",
        "OlmoForCausalLM",
        "SmolLM3ForCausalLM",
    ] {
        let f = registry_v2::lookup(alias)
            .unwrap_or_else(|| panic!("Llama alias '{alias}' must be registered"));
        assert_eq!(
            f.as_any().type_id(),
            canonical_id,
            "Llama alias '{alias}' resolves to a different factory than the canonical name",
        );
    }
}

#[test]
fn registry_iter_round_trips_through_lookup() {
    // Every entry exposed by `entries()` must resolve via `lookup()`.
    // Catches a divergence between the iterator and the map (would
    // hint at a phf API misuse).
    for (name, factory) in registry_v2::entries() {
        let resolved = registry_v2::lookup(name).expect("entry must lookup");
        assert_eq!(
            resolved.as_any().type_id(),
            factory.as_any().type_id(),
            "entries()/lookup() disagree on '{name}'",
        );
    }
}
