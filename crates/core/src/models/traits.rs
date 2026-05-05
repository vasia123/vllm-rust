//! Capability traits — runtime polymorphism for instantiated models.
//!
//! These mirror vLLM's Python `Protocol` interfaces (`SupportsMultiModal`,
//! `MixtureOfExperts`, `SupportsLoRA`, …). A constructed model
//! `Box<dyn ModelForward>` exposes additional behaviour to the engine /
//! server layer when callers downcast to one of these traits.
//!
//! The static `ArchInfo::capabilities` bitfield in `factory.rs` is the
//! source of truth for *routing decisions* ("is this multimodal?
//! → take the encoder-cache path"). The traits here are the source of
//! truth for *behaviour* ("call `process_multimodal_inputs` to embed
//! image patches"). Both must agree, enforced by
//! `tests/registry_consistency.rs` once Phase 2 lands.
//!
//! Phase 1 ships only the trait surface — implementations on concrete
//! models follow in Phase 2 alongside their `ArchFactory` impls. None
//! of these traits are wired into the engine yet; the existing
//! `ModelForward::supports_multimodal()` boolean continues to drive
//! routing until Phase 2 cuts over.

use candle_core::Tensor;

use crate::engine::ModelForward;
use crate::multimodal::MultimodalInputs;

use super::factory::CapabilityError;

// ─── SupportsMultimodal ─────────────────────────────────────────────────────

/// A model that can accept image / audio / video inputs in addition to
/// text tokens. Implementors merge multimodal embeddings into the text
/// embedding stream before the language-model forward.
pub trait SupportsMultimodal: ModelForward {
    /// Embed multimodal inputs into the language-model hidden space.
    /// Returns the merged embedding tensor `[B, L, hidden]` with image /
    /// audio / video patches substituted at their token positions.
    fn embed_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm: &MultimodalInputs,
    ) -> candle_core::Result<Tensor>;

    /// True if `mm` carries any multimodal payload; cheap pre-check
    /// callers use to skip embedding work on text-only requests.
    fn has_multimodal_inputs(&self, mm: &MultimodalInputs) -> bool {
        mm.has_images() || mm.has_audio() || mm.has_videos()
    }
}

// ─── MixtureOfExperts ───────────────────────────────────────────────────────

/// A model whose decoder layers route tokens through a sparse expert
/// pool. Exposes EPLB / token-dispatch hooks consumed by the MoE
/// subsystem.
pub trait MixtureOfExperts: ModelForward {
    /// Total experts across all MoE layers (sum, not per-layer).
    fn num_experts(&self) -> usize;
    /// Number of layers that actually contain MoE blocks (some Llama4 /
    /// DeepSeek variants intersperse dense + MoE layers).
    fn num_moe_layers(&self) -> usize;
    /// Top-k experts selected per token (model-wide constant in current
    /// architectures; the Python `MixtureOfExperts.num_expert_groups`
    /// generalisation is deferred until we have a model that needs it).
    fn experts_per_token(&self) -> usize;
}

// ─── SupportsLoRA ───────────────────────────────────────────────────────────

/// A model that can hot-swap LoRA adapters at runtime. The exact
/// adapter representation is owned by `lora::LoraAdapter`; this trait
/// only commits to the registration / activation lifecycle.
pub trait SupportsLoRA {
    /// Modules the adapter is allowed to target (e.g. `q_proj`,
    /// `k_proj`, `gate_proj`). Used by the loader to validate adapter
    /// keys before construction.
    fn lora_supported_modules(&self) -> &'static [&'static str];

    /// Maximum number of adapters that can be staged simultaneously.
    /// Engine uses this to size adapter slot tables.
    fn max_active_adapters(&self) -> usize {
        8
    }
}

// ─── SupportsTp / SupportsPp ───────────────────────────────────────────────

/// A model instantiated under tensor parallelism. Implementations
/// expose the rank topology to the engine for collective-op planning.
pub trait SupportsTp {
    fn tp_world_size(&self) -> usize;
    fn tp_rank(&self) -> usize;
}

/// A model instantiated as one stage of pipeline parallelism.
pub trait SupportsPp {
    fn pp_stage(&self) -> usize;
    fn pp_num_stages(&self) -> usize;
    /// Layer indices owned by this pipeline stage (inclusive ranges).
    fn pp_layer_range(&self) -> std::ops::Range<usize>;
}

// ─── SpeculativeDraft + SpeculativeFactory ─────────────────────────────────

/// Variant of speculative decoding the draft model implements. The
/// engine routes proposals to different verification kernels per
/// variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftKind {
    /// Multi-Token Prediction (DeepSeek MTP, Ernie MTP, …).
    Mtp,
    /// Eagle 1 (target-conditioned attention + lm_head).
    Eagle1,
    /// Eagle 3 (recurrent draft head with hidden-state passthrough).
    Eagle3,
    /// Medusa heads (parallel projection + tree decode).
    Medusa,
    /// MLP Speculator (IBM-style draft with stacked MLPs).
    MlpSpeculator,
}

/// Trait implemented by drafts after construction. The engine uses
/// `draft_kind()` to dispatch to the matching verification kernel.
pub trait SpeculativeDraft: ModelForward {
    fn draft_kind(&self) -> DraftKind;
}

/// Companion to `ArchFactory::as_speculative()`. A factory that
/// produces speculative drafts implements this so the dispatcher can
/// build the draft without hard-coding per-variant entry points.
///
/// Mirrors the existing `mtp_from_config` / `eagle1_from_config` /
/// etc. fan-out: each method covers a single draft variant. Default
/// methods return `Unsupported`; concrete factories override only
/// the variants they actually implement.
pub trait SpeculativeFactory: Sync + 'static {
    /// Which `DraftKind` this factory builds. Used by the dispatch
    /// shim to validate that the requested variant matches.
    fn kind(&self) -> DraftKind;

    /// Build a draft model. The signature stays loose (`&dyn Any`) for
    /// the factory-config arg so `SpeculativeFactory` doesn't drag
    /// every draft-config struct (`MtpConfig`, `Eagle3Config`, …) into
    /// the trait surface. The shim casts back to the concrete config
    /// type.
    #[allow(unused_variables)]
    fn build(
        &self,
        cfg: &crate::config::ModelConfig,
        vb: candle_nn::VarBuilder,
        draft_cfg: &dyn std::any::Any,
    ) -> Result<Box<dyn SpeculativeDraft>, CapabilityError> {
        Err(CapabilityError {
            arch: "speculative",
            capability: "build",
        })
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_kind_is_copy_eq() {
        let a = DraftKind::Eagle3;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(DraftKind::Mtp, DraftKind::Medusa);
    }

    #[test]
    fn capability_traits_are_object_safe() {
        // Compile-time assertion: dyn Trait must construct.
        fn _check_object_safe(
            _: &dyn SupportsMultimodal,
            _: &dyn MixtureOfExperts,
            _: &dyn SpeculativeDraft,
            _: &dyn SpeculativeFactory,
        ) {
        }
        // SupportsLoRA / SupportsTp / SupportsPp don't extend
        // ModelForward and don't need object safety beyond their own
        // method set; assert separately.
        fn _check_simple_traits(_: &dyn SupportsLoRA, _: &dyn SupportsTp, _: &dyn SupportsPp) {}
    }
}
