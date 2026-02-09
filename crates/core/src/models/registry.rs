//! Static catalog of supported model architectures and their capabilities.
//!
//! This module centralizes model metadata so that adding a new architecture
//! only requires a single catalog entry instead of updating multiple match arms.

/// Capability flags for a model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub supports_tp: bool,
    pub supports_quantization: bool,
    pub supports_lora: bool,
    pub supports_multimodal: bool,
    pub is_moe: bool,
    pub is_encoder_only: bool,
}

impl ModelCapabilities {
    const fn new() -> Self {
        Self {
            supports_tp: false,
            supports_quantization: false,
            supports_lora: false,
            supports_multimodal: false,
            is_moe: false,
            is_encoder_only: false,
        }
    }

    const fn with_tp(mut self) -> Self {
        self.supports_tp = true;
        self
    }

    const fn with_quantization(mut self) -> Self {
        self.supports_quantization = true;
        self
    }

    const fn with_lora(mut self) -> Self {
        self.supports_lora = true;
        self
    }

    const fn with_moe(mut self) -> Self {
        self.is_moe = true;
        self
    }

    const fn with_multimodal(mut self) -> Self {
        self.supports_multimodal = true;
        self
    }

    const fn with_encoder_only(mut self) -> Self {
        self.is_encoder_only = true;
        self
    }
}

/// Metadata for a supported model architecture.
#[derive(Debug, Clone, Copy)]
pub struct ArchitectureInfo {
    /// HuggingFace `config.json` architecture identifiers that map to this model.
    pub arch_names: &'static [&'static str],
    /// Human-readable name for logging and error messages.
    pub display_name: &'static str,
    /// Feature flags for this architecture.
    pub capabilities: ModelCapabilities,
}

// ─── Static Catalog ──────────────────────────────────────────────────────────

static ARCHITECTURES: &[ArchitectureInfo] = &[
    ArchitectureInfo {
        arch_names: &["BaichuanForCausalLM"],
        display_name: "Baichuan",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &[
            "BertModel",
            "BertForMaskedLM",
            "BertForSequenceClassification",
        ],
        display_name: "BERT",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["HF_ColBERT", "ColBERTModel"],
        display_name: "ColBERT",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["CohereForCausalLM"],
        display_name: "Command R",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"],
        display_name: "DeepSeek V2/V3",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["FalconForCausalLM"],
        display_name: "Falcon",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["BloomForCausalLM"],
        display_name: "BLOOM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GPT2LMHeadModel"],
        display_name: "GPT-2",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["GPTNeoXForCausalLM"],
        display_name: "GPT-NeoX",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GemmaForCausalLM"],
        display_name: "Gemma",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma2ForCausalLM"],
        display_name: "Gemma 2",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma3ForCausalLM"],
        display_name: "Gemma 3",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["InternLM2ForCausalLM"],
        display_name: "InternLM2",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &[
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
        ],
        display_name: "LLaVA",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["LlamaForCausalLM"],
        display_name: "Llama",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_lora(),
    },
    ArchitectureInfo {
        arch_names: &["MistralForCausalLM"],
        display_name: "Mistral",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["MixtralForCausalLM"],
        display_name: "Mixtral",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen2ForCausalLM"],
        display_name: "Qwen2",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen2MoeForCausalLM"],
        display_name: "Qwen2-MoE",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3ForCausalLM"],
        display_name: "Qwen3",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_lora(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3MoeForCausalLM"],
        display_name: "Qwen3-MoE",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["PhiForCausalLM"],
        display_name: "Phi",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Phi3ForCausalLM"],
        display_name: "Phi-3",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Starcoder2ForCausalLM"],
        display_name: "StarCoder2",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["VoyageQwen3BidirectionalEmbedModel"],
        display_name: "Voyage",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["YiForCausalLM"],
        display_name: "Yi",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Olmo2ForCausalLM"],
        display_name: "OLMo2",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GlmForCausalLM"],
        display_name: "GLM",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Glm4ForCausalLM"],
        display_name: "GLM-4",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["GlmMoeDsaForCausalLM"],
        display_name: "GLM-5",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Glm4MoeForCausalLM"],
        display_name: "GLM-4 MoE",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["JambaForCausalLM"],
        display_name: "Jamba",
        capabilities: ModelCapabilities::new().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["MambaForCausalLM", "FalconMambaForCausalLM"],
        display_name: "Mamba",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Mamba2ForCausalLM"],
        display_name: "Mamba-2",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["MptForCausalLM"],
        display_name: "MPT",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["PersimmonForCausalLM"],
        display_name: "Persimmon",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["ExaoneForCausalLM"],
        display_name: "Exaone",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["DbrxForCausalLM"],
        display_name: "DBRX",
        capabilities: ModelCapabilities::new().with_moe(),
    },
];

/// Returns the full catalog of supported architectures.
pub fn supported_architectures() -> &'static [ArchitectureInfo] {
    ARCHITECTURES
}

/// Looks up an architecture by its HuggingFace identifier string.
///
/// Returns `None` if the architecture is not in the catalog.
pub fn find_architecture(arch_name: &str) -> Option<&'static ArchitectureInfo> {
    ARCHITECTURES
        .iter()
        .find(|info| info.arch_names.contains(&arch_name))
}

/// Returns `true` if the given architecture identifier is recognized.
pub fn is_supported(arch_name: &str) -> bool {
    find_architecture(arch_name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_llama_by_arch_name() {
        let info = find_architecture("LlamaForCausalLM").expect("should find LlamaForCausalLM");
        assert_eq!(info.display_name, "Llama");
        assert!(info.capabilities.supports_tp);
        assert!(info.capabilities.supports_quantization);
        assert!(info.capabilities.supports_lora);
        assert!(!info.capabilities.is_moe);
    }

    #[test]
    fn find_deepseek_v2_and_v3() {
        let v2 =
            find_architecture("DeepseekV2ForCausalLM").expect("should find DeepseekV2ForCausalLM");
        let v3 =
            find_architecture("DeepseekV3ForCausalLM").expect("should find DeepseekV3ForCausalLM");
        assert_eq!(v2.display_name, v3.display_name);
        assert_eq!(v2.display_name, "DeepSeek V2/V3");
    }

    #[test]
    fn mixtral_is_moe() {
        let info = find_architecture("MixtralForCausalLM").expect("should find MixtralForCausalLM");
        assert!(info.capabilities.is_moe);
        assert!(info.capabilities.supports_tp);
    }

    #[test]
    fn find_bert_is_encoder_only() {
        let info = find_architecture("BertModel").expect("should find BertModel");
        assert_eq!(info.display_name, "BERT");
        assert!(info.capabilities.is_encoder_only);
        assert!(!info.capabilities.supports_tp);
        assert!(!info.capabilities.is_moe);

        // All BERT arch names should resolve to the same entry
        let info2 = find_architecture("BertForMaskedLM").expect("should find BertForMaskedLM");
        assert_eq!(info.display_name, info2.display_name);
    }

    #[test]
    fn unknown_arch_returns_none() {
        assert!(find_architecture("UnknownForCausalLM").is_none());
    }

    #[test]
    fn is_supported_matches_find() {
        assert!(is_supported("LlamaForCausalLM"));
        assert!(is_supported("DeepseekV3ForCausalLM"));
        assert!(!is_supported("UnknownForCausalLM"));
    }

    #[test]
    fn catalog_covers_all_expected_architectures() {
        let expected = [
            "BaichuanForCausalLM",
            "BertModel",
            "BertForMaskedLM",
            "BertForSequenceClassification",
            "BloomForCausalLM",
            "HF_ColBERT",
            "ColBERTModel",
            "CohereForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "FalconForCausalLM",
            "GPT2LMHeadModel",
            "GPTNeoXForCausalLM",
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "InternLM2ForCausalLM",
            "JambaForCausalLM",
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "LlamaForCausalLM",
            "Mamba2ForCausalLM",
            "MistralForCausalLM",
            "MixtralForCausalLM",
            "PhiForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Phi3ForCausalLM",
            "Olmo2ForCausalLM",
            "Starcoder2ForCausalLM",
            "VoyageQwen3BidirectionalEmbedModel",
            "YiForCausalLM",
            "GlmForCausalLM",
            "Glm4ForCausalLM",
            "GlmMoeDsaForCausalLM",
            "Glm4MoeForCausalLM",
            "MambaForCausalLM",
            "FalconMambaForCausalLM",
            "MptForCausalLM",
            "PersimmonForCausalLM",
            "ExaoneForCausalLM",
            "DbrxForCausalLM",
        ];
        for arch in &expected {
            assert!(
                is_supported(arch),
                "expected architecture {arch} to be in the catalog"
            );
        }
    }

    #[test]
    fn supported_architectures_returns_nonempty() {
        let archs = supported_architectures();
        assert!(
            !archs.is_empty(),
            "catalog should contain at least one entry"
        );
    }

    #[test]
    fn capabilities_builder_chain() {
        let caps = ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_lora()
            .with_moe();
        assert!(caps.supports_tp);
        assert!(caps.supports_quantization);
        assert!(caps.supports_lora);
        assert!(caps.is_moe);
        assert!(!caps.supports_multimodal);
        assert!(!caps.is_encoder_only);
    }
}
