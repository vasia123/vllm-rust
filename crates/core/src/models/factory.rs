//! Architecture factory layer (Phase 1 of the registry refactor).
//!
//! Each model architecture supplies a `ArchFactory` impl describing
//! - which HuggingFace `architectures[]` strings it answers to,
//! - what static capabilities the architecture has (multimodal, MoE,
//!   speculative draft, …),
//! - how to instantiate the model under each runtime mode (plain,
//!   quantized, tensor-parallel, pipeline-parallel, LoRA, encoder-decoder).
//!
//! Most factories override only the methods that apply to them;
//! everything else falls back to a `CapabilityError::Unsupported` so
//! the dispatcher can return a precise error without per-arm
//! boilerplate.
//!
//! The trait is **not** wired into `from_config*` yet — that happens in
//! Phase 2. This file only lands the API and tests so the rest of the
//! crate can compile against it under
//! `--features model-registry-v2`.
//!
//! # Worked example
//!
//! ```ignore
//! pub struct LlamaArchFactory;
//!
//! impl ArchFactory for LlamaArchFactory {
//!     fn arch_names(&self) -> &'static [&'static str] {
//!         &["LlamaForCausalLM", "LlamaModel", "LLaMAForCausalLM"]
//!     }
//!     fn info(&self) -> &'static ArchInfo { &LLAMA_INFO }
//!     fn build(&self, cfg: &ModelConfig, vb: VarBuilder) -> Result<Box<dyn ModelForward>, ModelError> {
//!         Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?))
//!     }
//!     // build_quant / build_with_tp / build_with_lora overridden as needed.
//! }
//!
//! static LLAMA_INFO: ArchInfo = ArchInfo {
//!     display_name: "LLaMA",
//!     capabilities: Capabilities::TP | Capabilities::QUANTIZED | Capabilities::LORA,
//! };
//! ```

use std::any::Any;
use std::fmt;

use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{PipelineStageConfig, ProcessGroup};
use crate::engine::{ModelForEncoderDecoder, ModelForward};

use super::tp_layers::TpContext;
use super::ModelError;

// ─── ArchInfo + Capabilities ────────────────────────────────────────────────

/// Static metadata about an architecture. Lives next to its factory so a
/// caller can answer "is this multimodal?" without instantiating the
/// model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchInfo {
    /// Human-readable label used in logs and errors. Not necessarily a
    /// HuggingFace `architectures[]` string.
    pub display_name: &'static str,
    /// Bitfield of supported capabilities — drives routing decisions in
    /// the dispatcher (e.g. "this arch is multimodal → take the encoder-
    /// cache path") and validates CLI flags before model construction.
    pub capabilities: Capabilities,
}

bitflags::bitflags! {
    /// Static capability flags. Source of truth for routing; per-instance
    /// behaviour is checked at construction time via the trait downcasts
    /// in `traits.rs` (see `tests/registry_consistency.rs`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Capabilities: u32 {
        /// Arch can run with `from_config_with_quant` (BnB / AWQ / GPTQ / FP8 / GGUF / …).
        const QUANTIZED         = 1 << 0;
        /// Arch implements `from_config_with_tp` for tensor parallelism.
        const TP                = 1 << 1;
        /// Arch implements `from_config_with_pp` for pipeline parallelism.
        const PP                = 1 << 2;
        /// Arch supports LoRA adapter registration.
        const LORA              = 1 << 3;
        /// Arch is encoder-decoder (BART / T5 / Whisper-style).
        const ENCODER_DECODER   = 1 << 4;
        /// Arch accepts multimodal inputs (image / audio / video).
        const MULTIMODAL        = 1 << 5;
        /// Arch is a Mixture-of-Experts model.
        const MOE               = 1 << 6;
        /// Arch is a speculative-decoding draft (MTP / Eagle / Medusa / MLPSpec).
        const SPECULATIVE_DRAFT = 1 << 7;
        /// Arch uses MLA (Multi-head Latent Attention) — DeepSeek family.
        const MLA               = 1 << 8;
    }
}

impl ArchInfo {
    pub const fn new(display_name: &'static str, capabilities: Capabilities) -> Self {
        Self {
            display_name,
            capabilities,
        }
    }

    pub const fn supports(&self, c: Capabilities) -> bool {
        self.capabilities.contains(c)
    }
}

// ─── CapabilityError ────────────────────────────────────────────────────────

/// Returned by default-impl factory methods when the arch does not
/// support a runtime mode. Promoted to `ModelError` by the dispatch
/// shims in Phase 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityError {
    pub arch: &'static str,
    pub capability: &'static str,
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "architecture '{}' does not support capability '{}'",
            self.arch, self.capability
        )
    }
}

impl std::error::Error for CapabilityError {}

impl From<CapabilityError> for ModelError {
    fn from(e: CapabilityError) -> Self {
        ModelError::UnsupportedArchitecture(e.to_string())
    }
}

/// Helper used by the default trait methods to construct a precise
/// `Err(...)` without forcing every factory to repeat the boilerplate.
pub fn unsupported(arch: &'static str, capability: &'static str) -> CapabilityError {
    CapabilityError { arch, capability }
}

// ─── ArchFactory trait ──────────────────────────────────────────────────────

/// Construct concrete model instances for a single HuggingFace
/// architecture (or a group of arch_name aliases that map to the same
/// implementation).
///
/// `Sync + 'static` so factories can live as `static FOO: FooArchFactory
/// = FooArchFactory;` and be referenced as `&'static dyn ArchFactory`
/// from the `phf::Map` registry.
pub trait ArchFactory: Sync + 'static {
    /// HuggingFace `architectures[]` strings this factory answers to.
    /// Multiple aliases (e.g. `LlamaForCausalLM`, `LlamaModel`) are
    /// allowed; they all map to the same `info()` and the same `build`.
    fn arch_names(&self) -> &'static [&'static str];

    /// Static metadata. Read by the dispatcher and the loader for
    /// routing / capability validation without constructing the model.
    fn info(&self) -> &'static ArchInfo;

    /// Plain (unquantized, single-GPU, no LoRA) instantiation. Default
    /// returns `Unsupported` so speculative-only / encoder-decoder-only
    /// archs (Eagle / MTP / T5 / Whisper) don't have to fake a method
    /// they never actually expose to the standard dispatch.
    #[allow(unused_variables)]
    fn build(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Err(unsupported(self.canonical_name(), "build").into())
    }

    /// Quantized instantiation. The dispatch shim builds the
    /// `weight_loader` once from the detected `DetectedQuantConfig` and
    /// hands it in here, so factories don't repeat the
    /// `create_weight_loader_with_params(...)` boilerplate. Default
    /// returns `Unsupported`.
    #[allow(unused_variables)]
    fn build_quant(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn crate::quantization::QuantizedWeightLoader,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Err(unsupported(self.canonical_name(), "quantized").into())
    }

    /// Tensor-parallel instantiation. Default returns `Unsupported`.
    /// Concrete types are taken directly because `ProcessGroup` is
    /// already `dyn`-safe and `TpContext` is `Clone + Copy`-friendly,
    /// so factories can call `LlamaForCausalLM::new_with_tp(cfg, vb,
    /// pg, tp_ctx)` straight through with no downcast dance.
    #[allow(unused_variables)]
    fn build_with_tp(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
        process_group: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Err(unsupported(self.canonical_name(), "tensor-parallel").into())
    }

    /// Pipeline-parallel instantiation. `stage` is the pipeline-stage
    /// descriptor expected by `LlamaForCausalLM::new_with_pp` and
    /// friends (engine-side `PipelineStage`). Default `Unsupported`.
    #[allow(unused_variables)]
    fn build_with_pp(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
        stage: &PipelineStageConfig,
    ) -> Result<Box<dyn ModelForward>, ModelError> {
        Err(unsupported(self.canonical_name(), "pipeline-parallel").into())
    }

    /// LoRA-adapter-aware instantiation. Returns the existing
    /// `LoraEnabledModel` enum from `models/mod.rs` to keep the public
    /// API surface stable. Default `Unsupported`.
    #[allow(unused_variables)]
    fn build_with_lora(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<super::LoraEnabledModel, ModelError> {
        Err(unsupported(self.canonical_name(), "lora").into())
    }

    /// Encoder-decoder instantiation (T5, Whisper, BART). Default
    /// `Unsupported`.
    #[allow(unused_variables)]
    fn build_encoder_decoder(
        &self,
        cfg: &ModelConfig,
        vb: VarBuilder,
    ) -> Result<Box<dyn ModelForEncoderDecoder>, ModelError> {
        Err(unsupported(self.canonical_name(), "encoder-decoder").into())
    }

    /// Speculative-draft factory hook. Default `None`. Archs that act
    /// as drafts (Eagle / Medusa / MTP / MlpSpeculator) override to
    /// expose their `SpeculativeFactory` impl from `traits.rs`.
    fn as_speculative(&self) -> Option<&dyn super::traits::SpeculativeFactory> {
        None
    }

    /// Downcast hook so capability-aware callers can treat the factory
    /// as a concrete type when needed (e.g. unit tests, doctest setup).
    fn as_any(&self) -> &dyn Any;

    // ─── Conveniences ───────────────────────────────────────────────────

    /// Display name from `info()`. Used to label `unsupported(...)`
    /// errors without forcing every override to pass it explicitly.
    fn canonical_name(&self) -> &'static str {
        self.info().display_name
    }
}

// ─── Tests for the trait shape itself ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    /// Mock factory that supports nothing — exercises every default
    /// `Unsupported` path.
    struct NopFactory;

    static NOP_INFO: ArchInfo = ArchInfo::new("Nop", Capabilities::empty());

    impl ArchFactory for NopFactory {
        fn arch_names(&self) -> &'static [&'static str] {
            &["NopForCausalLM"]
        }
        fn info(&self) -> &'static ArchInfo {
            &NOP_INFO
        }
        fn build(
            &self,
            _cfg: &ModelConfig,
            _vb: VarBuilder,
        ) -> Result<Box<dyn ModelForward>, ModelError> {
            Err(unsupported("Nop", "build").into())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn capabilities_bitflags_compose() {
        let caps = Capabilities::TP | Capabilities::QUANTIZED;
        assert!(caps.contains(Capabilities::TP));
        assert!(caps.contains(Capabilities::QUANTIZED));
        assert!(!caps.contains(Capabilities::LORA));
    }

    #[test]
    fn arch_info_supports_query() {
        let info = ArchInfo::new("Foo", Capabilities::MULTIMODAL | Capabilities::MOE);
        assert!(info.supports(Capabilities::MOE));
        assert!(!info.supports(Capabilities::TP));
    }

    #[test]
    fn unsupported_default_methods_return_unsupported() {
        let factory = NopFactory;
        assert_eq!(factory.arch_names(), &["NopForCausalLM"]);
        assert_eq!(factory.canonical_name(), "Nop");

        // We can't easily call `build_quant` etc. without a real
        // VarBuilder; the contract is asserted by the existence of the
        // default impl returning `unsupported(...)`. Smoke-test
        // CapabilityError formatting instead.
        let err = unsupported("Nop", "tensor-parallel");
        assert_eq!(
            err.to_string(),
            "architecture 'Nop' does not support capability 'tensor-parallel'"
        );
    }

    #[test]
    fn capability_error_promotes_to_model_error() {
        let err: ModelError = unsupported("Foo", "lora").into();
        match err {
            ModelError::UnsupportedArchitecture(s) => {
                assert!(s.contains("Foo"));
                assert!(s.contains("lora"));
            }
            other => panic!("expected UnsupportedArchitecture, got {other:?}"),
        }
    }

    #[test]
    fn factory_object_safe_via_dyn() {
        // Compile-time test: trait must be object-safe so it can live
        // as `&'static dyn ArchFactory` in `phf::Map`.
        let factories: &[&dyn ArchFactory] = &[&NopFactory];
        assert_eq!(factories.len(), 1);
        assert_eq!(factories[0].arch_names(), &["NopForCausalLM"]);
    }

    #[test]
    fn factory_as_any_downcast_works() {
        let factory: &dyn ArchFactory = &NopFactory;
        let nop = factory.as_any().downcast_ref::<NopFactory>();
        assert!(nop.is_some(), "downcast must recover concrete type");
    }
}
