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
        arch_names: &["BaichuanForCausalLM", "BaiChuanForCausalLM"],
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
        arch_names: &["CohereForCausalLM", "Cohere2ForCausalLM"],
        display_name: "Command R",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &[
            "DeepseekForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
        ],
        display_name: "DeepSeek",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["FalconForCausalLM", "RWForCausalLM"],
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
        arch_names: &[
            "LlamaForCausalLM",
            "LlamaModel",
            "LLaMAForCausalLM",
            "AquilaModel",
            "AquilaForCausalLM",
            "CwmForCausalLM",
            "InternLMForCausalLM",
            "InternLM3ForCausalLM",
            "IQuestCoderForCausalLM",
            "XverseForCausalLM",
        ],
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
        arch_names: &["Step3p5ForCausalLM"],
        display_name: "Step-3.5-Flash",
        capabilities: ModelCapabilities::new().with_moe(),
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
        arch_names: &["Olmo2ForCausalLM", "Olmo3ForCausalLM"],
        display_name: "OLMo2/3",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["OlmoForCausalLM"],
        display_name: "OLMo",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["OlmoeForCausalLM"],
        display_name: "OLMoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
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
        arch_names: &["MptForCausalLM", "MPTForCausalLM"],
        display_name: "MPT",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["ChatGLMModel", "ChatGLMForConditionalGeneration"],
        display_name: "ChatGLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma3nForCausalLM"],
        display_name: "Gemma 3n",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["GPTBigCodeForCausalLM"],
        display_name: "GPT-BigCode",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GPTJForCausalLM"],
        display_name: "GPT-J",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GraniteForCausalLM"],
        display_name: "Granite",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GraniteMoeForCausalLM"],
        display_name: "Granite-MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["OPTForCausalLM"],
        display_name: "OPT",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["PhiMoEForCausalLM"],
        display_name: "Phi-MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &[
            "MistralModel",
            "Gemma2Model",
        ],
        display_name: "Mistral/Gemma2 (base)",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen2Model"],
        display_name: "Qwen2 (base)",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3NextForCausalLM"],
        display_name: "Qwen3-Next",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &[
            "StableLMEpochForCausalLM",
            "StableLmForCausalLM",
        ],
        display_name: "StableLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["SolarForCausalLM"],
        display_name: "Solar",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["TeleChatForCausalLM", "TeleChat2ForCausalLM"],
        display_name: "TeleChat",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["MiniCPMForCausalLM", "MiniCPM3ForCausalLM"],
        display_name: "MiniCPM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["NemotronForCausalLM"],
        display_name: "Nemotron",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Glm4MoeLiteForCausalLM"],
        display_name: "GLM-4 MoE Lite",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Llama4ForCausalLM"],
        display_name: "Llama 4",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Exaone4ForCausalLM"],
        display_name: "Exaone 4",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["OrionForCausalLM"],
        display_name: "Orion",
        capabilities: ModelCapabilities::new().with_tp(),
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
    ArchitectureInfo {
        arch_names: &["ArcticForCausalLM"],
        display_name: "Arctic",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["BailingMoeForCausalLM", "BailingMoeV2ForCausalLM"],
        display_name: "BailingMoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["BambaForCausalLM"],
        display_name: "Bamba",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Ernie4_5ForCausalLM"],
        display_name: "ERNIE 4.5",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Ernie4_5_MoeForCausalLM"],
        display_name: "ERNIE 4.5 MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["ExaoneMoEForCausalLM"],
        display_name: "Exaone MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["FalconH1ForCausalLM"],
        display_name: "Falcon H1",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Grok1ModelForCausalLM", "Grok1ForCausalLM"],
        display_name: "Grok-1",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["HunYuanMoEV1ForCausalLM"],
        display_name: "HunYuan MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["HunYuanDenseV1ForCausalLM"],
        display_name: "HunYuan Dense",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["JAISLMHeadModel", "Jais2ForCausalLM"],
        display_name: "JAIS",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["KimiLinearForCausalLM"],
        display_name: "Kimi Linear",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["MistralLarge3ForCausalLM"],
        display_name: "Mistral Large 3",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &[
            "MiniMaxForCausalLM",
            "MiniMaxText01ForCausalLM",
            "MiniMaxM1ForCausalLM",
            "MiniMaxM2ForCausalLM",
        ],
        display_name: "MiniMax",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["MiMoForCausalLM", "MiMoV2FlashForCausalLM"],
        display_name: "MiMo",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"],
        display_name: "Nemotron-H",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Plamo2ForCausalLM", "Plamo3ForCausalLM"],
        display_name: "PLaMo",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["QWenLMHeadModel"],
        display_name: "Qwen (v1)",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["SeedOssForCausalLM"],
        display_name: "Seed-OSS",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Step1ForCausalLM", "Step3TextForCausalLM"],
        display_name: "Step",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Lfm2ForCausalLM"],
        display_name: "Lfm2",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Lfm2MoeForCausalLM"],
        display_name: "Lfm2 MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["GraniteMoeHybridForCausalLM", "GraniteMoeSharedForCausalLM"],
        display_name: "Granite MoE Hybrid",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Zamba2ForCausalLM"],
        display_name: "Zamba 2",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Dots1ForCausalLM"],
        display_name: "Dots1",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GptOssForCausalLM"],
        display_name: "GptOSS",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &[
            "PanguEmbeddedForCausalLM",
            "PanguProMoEV2ForCausalLM",
            "PanguUltraMoEForCausalLM",
        ],
        display_name: "Pangu",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["IQuestLoopCoderForCausalLM"],
        display_name: "IQuest LoopCoder",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Fairseq2LlamaForCausalLM"],
        display_name: "Fairseq2 Llama",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["FlexOlmoForCausalLM"],
        display_name: "FlexOLMo",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["LongcatFlashForCausalLM"],
        display_name: "Longcat Flash",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["OuroForCausalLM"],
        display_name: "Ouro",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["TeleFLMForCausalLM"],
        display_name: "TeleFLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["DeciLMForCausalLM"],
        display_name: "DeciLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["ArceeForCausalLM"],
        display_name: "Arcee",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["AfmoeForCausalLM"],
        display_name: "Afmoe",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["ApertusForCausalLM"],
        display_name: "Apertus",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["GritLM"],
        display_name: "GritLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["InternLM2VEForCausalLM"],
        display_name: "InternLM2-VE",
        capabilities: ModelCapabilities::new().with_tp(),
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
    fn find_deepseek_variants() {
        let v2 =
            find_architecture("DeepseekV2ForCausalLM").expect("should find DeepseekV2ForCausalLM");
        let v3 =
            find_architecture("DeepseekV3ForCausalLM").expect("should find DeepseekV3ForCausalLM");
        let v32 =
            find_architecture("DeepseekV32ForCausalLM").expect("should find DeepseekV32ForCausalLM");
        assert_eq!(v2.display_name, v3.display_name);
        assert_eq!(v2.display_name, v32.display_name);
        assert_eq!(v2.display_name, "DeepSeek");
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
            // Baichuan
            "BaichuanForCausalLM",
            "BaiChuanForCausalLM",
            // BERT
            "BertModel",
            "BertForMaskedLM",
            "BertForSequenceClassification",
            // ColBERT
            "HF_ColBERT",
            "ColBERTModel",
            // Cohere
            "CohereForCausalLM",
            "Cohere2ForCausalLM",
            // DeepSeek
            "DeepseekForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            // Falcon
            "FalconForCausalLM",
            "RWForCausalLM",
            // BLOOM
            "BloomForCausalLM",
            // GPT family
            "GPT2LMHeadModel",
            "GPTNeoXForCausalLM",
            "GPTBigCodeForCausalLM",
            "GPTJForCausalLM",
            // Gemma
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "Gemma3nForCausalLM",
            // InternLM
            "InternLM2ForCausalLM",
            // Llama family (aliases)
            "LlamaForCausalLM",
            "LlamaModel",
            "LLaMAForCausalLM",
            "AquilaModel",
            "AquilaForCausalLM",
            "CwmForCausalLM",
            "InternLMForCausalLM",
            "InternLM3ForCausalLM",
            "IQuestCoderForCausalLM",
            "XverseForCausalLM",
            "Llama4ForCausalLM",
            // LLaVA
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            // Mamba
            "MambaForCausalLM",
            "FalconMambaForCausalLM",
            "Mamba2ForCausalLM",
            // Mistral
            "MistralForCausalLM",
            "MixtralForCausalLM",
            // MiniCPM
            "MiniCPMForCausalLM",
            "MiniCPM3ForCausalLM",
            // MPT
            "MptForCausalLM",
            "MPTForCausalLM",
            // Nemotron
            "NemotronForCausalLM",
            // OLMo
            "OlmoForCausalLM",
            "Olmo2ForCausalLM",
            "Olmo3ForCausalLM",
            "OlmoeForCausalLM",
            // OPT
            "OPTForCausalLM",
            // Orion
            "OrionForCausalLM",
            // Phi
            "PhiForCausalLM",
            "Phi3ForCausalLM",
            "PhiMoEForCausalLM",
            // Qwen
            "Qwen2ForCausalLM",
            "Qwen2Model",
            "Qwen2MoeForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "Qwen3NextForCausalLM",
            // GLM
            "GlmForCausalLM",
            "Glm4ForCausalLM",
            "GlmMoeDsaForCausalLM",
            "Glm4MoeForCausalLM",
            "Glm4MoeLiteForCausalLM",
            "ChatGLMModel",
            "ChatGLMForConditionalGeneration",
            // Granite
            "GraniteForCausalLM",
            "GraniteMoeForCausalLM",
            // Jamba
            "JambaForCausalLM",
            // StarCoder
            "Starcoder2ForCausalLM",
            // StableLM
            "StableLMEpochForCausalLM",
            "StableLmForCausalLM",
            // Solar
            "SolarForCausalLM",
            // Step
            "Step3p5ForCausalLM",
            // TeleChat
            "TeleChatForCausalLM",
            "TeleChat2ForCausalLM",
            // Voyage
            "VoyageQwen3BidirectionalEmbedModel",
            // Yi
            "YiForCausalLM",
            // Exaone
            "ExaoneForCausalLM",
            "Exaone4ForCausalLM",
            "ExaoneMoEForCausalLM",
            // Other
            "PersimmonForCausalLM",
            "DbrxForCausalLM",
            "MistralModel",
            "Gemma2Model",
            // New architectures (2025-02 sync)
            "ArcticForCausalLM",
            "BailingMoeForCausalLM",
            "BailingMoeV2ForCausalLM",
            "BambaForCausalLM",
            "Ernie4_5ForCausalLM",
            "Ernie4_5_MoeForCausalLM",
            "FalconH1ForCausalLM",
            "Grok1ModelForCausalLM",
            "Grok1ForCausalLM",
            "HunYuanMoEV1ForCausalLM",
            "HunYuanDenseV1ForCausalLM",
            "JAISLMHeadModel",
            "Jais2ForCausalLM",
            "KimiLinearForCausalLM",
            "MistralLarge3ForCausalLM",
            "MiniMaxForCausalLM",
            "MiniMaxText01ForCausalLM",
            "MiniMaxM1ForCausalLM",
            "MiniMaxM2ForCausalLM",
            "MiMoForCausalLM",
            "MiMoV2FlashForCausalLM",
            "NemotronHForCausalLM",
            "NemotronHPuzzleForCausalLM",
            "Plamo2ForCausalLM",
            "Plamo3ForCausalLM",
            "QWenLMHeadModel",
            "SeedOssForCausalLM",
            "Step1ForCausalLM",
            "Step3TextForCausalLM",
            "Lfm2ForCausalLM",
            "Lfm2MoeForCausalLM",
            "GraniteMoeHybridForCausalLM",
            "GraniteMoeSharedForCausalLM",
            "Zamba2ForCausalLM",
            "Dots1ForCausalLM",
            "GptOssForCausalLM",
            "PanguEmbeddedForCausalLM",
            "PanguProMoEV2ForCausalLM",
            "PanguUltraMoEForCausalLM",
            "IQuestLoopCoderForCausalLM",
            "Fairseq2LlamaForCausalLM",
            "FlexOlmoForCausalLM",
            "LongcatFlashForCausalLM",
            "OuroForCausalLM",
            "TeleFLMForCausalLM",
            "DeciLMForCausalLM",
            "ArceeForCausalLM",
            "AfmoeForCausalLM",
            "ApertusForCausalLM",
            "GritLM",
            "InternLM2VEForCausalLM",
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
