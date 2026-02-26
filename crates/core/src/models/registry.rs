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
    pub is_encoder_decoder: bool,
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
            is_encoder_decoder: false,
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

    const fn with_encoder_decoder(mut self) -> Self {
        self.is_encoder_decoder = true;
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
        arch_names: &["BagelForConditionalGeneration"],
        display_name: "Bagel",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
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
            "BertForTokenClassification",
            "BertSpladeSparseEmbeddingModel",
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
        arch_names: &[
            "DeepSeekVLV2ForCausalLM",
            "DeepseekVLV2ForCausalLM",
            "DeepseekVLV2ForConditionalGeneration",
        ],
        display_name: "DeepSeek VL V2",
        capabilities: ModelCapabilities::new().with_multimodal(),
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
        arch_names: &["GPT2LMHeadModel", "GPT2ForSequenceClassification"],
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
        arch_names: &["Gemma3ForCausalLM", "Gemma3TextModel"],
        display_name: "Gemma 3",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma3ForConditionalGeneration"],
        display_name: "Gemma 3 VLM",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["InternLM2ForCausalLM"],
        display_name: "InternLM2",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["InternVLChatModel", "H2OVLChatModel", "SkyworkR1VChatModel"],
        display_name: "InternVL2",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["NVLM_D_Model", "NVLM_D"],
        display_name: "NVLM-D",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "MantisForConditionalGeneration",
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
        arch_names: &[
            "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3",
            "Eagle3Qwen2_5vlForCausalLM",
            "Eagle3Qwen3vlForCausalLM",
        ],
        display_name: "Eagle3 Llama",
        capabilities: ModelCapabilities::new(),
    },
    // ─── Speculative Decoding Draft Models ──────────────────────────────
    ArchitectureInfo {
        arch_names: &["EagleLlamaForCausalLM"],
        display_name: "Eagle Llama",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["EagleLlama4ForCausalLM"],
        display_name: "Eagle Llama 4",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["EagleMiniCPMForCausalLM"],
        display_name: "Eagle MiniCPM",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["EagleMistralLarge3ForCausalLM"],
        display_name: "Eagle Mistral Large 3",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["EagleDeepSeekMTPModel"],
        display_name: "Eagle DeepSeek MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["DeepSeekMTPModel"],
        display_name: "DeepSeek MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["ErnieMTPModel"],
        display_name: "ERNIE MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["ExaoneMoeMTP"],
        display_name: "Exaone MoE MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Glm4MoeMTPModel", "Glm4MoeLiteMTPModel"],
        display_name: "GLM-4 MoE MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["GlmOcrMTPModel"],
        display_name: "GLM OCR MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["LongCatFlashMTPModel"],
        display_name: "LongCat Flash MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["MedusaModel"],
        display_name: "Medusa",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["MiMoMTPModel"],
        display_name: "MiMo MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["OpenPanguMTPModel"],
        display_name: "OpenPangu MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3NextMTP"],
        display_name: "Qwen3 Next MTP",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["Step3p5MTP"],
        display_name: "Step-3.5 MTP",
        capabilities: ModelCapabilities::new(),
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
        arch_names: &["JambaForCausalLM", "JambaForSequenceClassification"],
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
        arch_names: &[
            "ChatGLMModel",
            "ChatGLMForConditionalGeneration",
            "ChatGLMForCausalLM",
        ],
        display_name: "ChatGLM",
        capabilities: ModelCapabilities::new().with_tp(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma3nForCausalLM"],
        display_name: "Gemma 3n",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["FuyuForCausalLM"],
        display_name: "Fuyu",
        capabilities: ModelCapabilities::new().with_multimodal(),
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
        arch_names: &["PhiMoEForCausalLM", "PhiMoeForCausalLM"],
        display_name: "Phi-MoE",
        capabilities: ModelCapabilities::new().with_tp().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["MistralModel", "E5MistralModel"],
        display_name: "Mistral (base)",
        capabilities: ModelCapabilities::new().with_tp().with_quantization(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma2Model"],
        display_name: "Gemma 2 (base)",
        capabilities: ModelCapabilities::new()
            .with_tp()
            .with_quantization()
            .with_encoder_only(),
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
        arch_names: &["StableLMEpochForCausalLM", "StableLmForCausalLM"],
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
        arch_names: &[
            "Llama4VLForConditionalGeneration",
            "MLlama4ForConditionalGeneration",
            "Llama4ForConditionalGeneration",
        ],
        display_name: "Llama 4 VL",
        capabilities: ModelCapabilities::new().with_multimodal().with_moe(),
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
        arch_names: &["Exaone4ForCausalLM"],
        display_name: "Exaone 4",
        capabilities: ModelCapabilities::new(),
    },
    ArchitectureInfo {
        arch_names: &["ExaoneMoeForCausalLM", "ExaoneMoEForCausalLM"],
        display_name: "Exaone MoE",
        capabilities: ModelCapabilities::new().with_moe(),
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
        arch_names: &[
            "AyaVisionForConditionalGeneration",
            "Cohere2VisionForConditionalGeneration",
        ],
        display_name: "AyaVision / Cohere2 Vision",
        capabilities: ModelCapabilities::new().with_multimodal(),
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
        arch_names: &[
            "Ernie4_5_MoeForCausalLM",
            "Ernie45MoeForCausalLM",
            "Ernie4_5MoeForCausalLM",
        ],
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
    // ─── Embedding / Cross-Encoder / Reward Models ──────────────────────────
    ArchitectureInfo {
        arch_names: &[
            "RobertaForMaskedLM",
            "RobertaModel",
            "XLMRobertaModel",
            "JinaRobertaModel",
            "RobertaForSequenceClassification",
            "XLMRobertaForSequenceClassification",
            "BgeRerankerModel",
        ],
        display_name: "RoBERTa",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["BgeM3EmbeddingModel"],
        display_name: "BGE-M3",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &[
            "ModernBertModel",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
        ],
        display_name: "ModernBERT",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &[
            "GteModel",
            "GteNewModel",
            "NomicBertModel",
            "GteNewForSequenceClassification",
            "SnowflakeGteNewModel",
        ],
        display_name: "GTE/Nomic",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &[
            "LlamaBidirectionalModel",
            "LlamaBidirectionalForSequenceClassification",
        ],
        display_name: "Llama Bidirectional",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["InternLM2ForRewardModel"],
        display_name: "InternLM2 Reward",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen2ForRewardModel", "Qwen2ForProcessRewardModel"],
        display_name: "Qwen2 Reward",
        capabilities: ModelCapabilities::new().with_encoder_only(),
    },
    ArchitectureInfo {
        arch_names: &["HCXVisionForCausalLM"],
        display_name: "HyperCLOVA X",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    // ─── Encoder-Decoder Models ────────────────────────────────────────────
    ArchitectureInfo {
        arch_names: &["T5ForConditionalGeneration", "T5Model"],
        display_name: "T5",
        capabilities: ModelCapabilities::new().with_encoder_decoder(),
    },
    // ─── Multimodal Embedding / Cross-Encoder ───────────────────────────────
    ArchitectureInfo {
        arch_names: &["CLIPModel"],
        display_name: "CLIP",
        capabilities: ModelCapabilities::new()
            .with_encoder_only()
            .with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["SiglipModel"],
        display_name: "SigLIP",
        capabilities: ModelCapabilities::new()
            .with_encoder_only()
            .with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Phi3VForCausalLM"],
        display_name: "Phi-3 Vision",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["PaliGemmaForConditionalGeneration"],
        display_name: "PaliGemma",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "LlavaOnevisionForConditionalGeneration",
            "LlavaNextVideoForConditionalGeneration",
            "RForConditionalGeneration",
            "BeeForConditionalGeneration",
        ],
        display_name: "LLaVA-OneVision",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["MolmoForCausalLM"],
        display_name: "Molmo",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["GLM4VForCausalLM", "Glm4VForConditionalGeneration"],
        display_name: "GLM-4V",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["MiniCPMV", "MiniCPMVForConditionalGeneration"],
        display_name: "MiniCPM-V",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Qwen2VLForConditionalGeneration",
            "Tarsier2ForConditionalGeneration",
        ],
        display_name: "Qwen2-VL",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen25VLForConditionalGeneration",
            "OpenCUAForConditionalGeneration",
        ],
        display_name: "Qwen2.5-VL",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3VLForConditionalGeneration"],
        display_name: "Qwen3-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3VLMoeForConditionalGeneration"],
        display_name: "Qwen3-VL-MoE",
        capabilities: ModelCapabilities::new().with_multimodal().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["PixtralForConditionalGeneration"],
        display_name: "Pixtral",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Mistral3ForConditionalGeneration"],
        display_name: "Mistral3",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["AriaForConditionalGeneration"],
        display_name: "Aria",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["OpenPanguVLForConditionalGeneration"],
        display_name: "OpenPangu-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["KeyeForConditionalGeneration"],
        display_name: "Keye",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["KeyeVL1_5ForConditionalGeneration"],
        display_name: "Keye-VL 1.5",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["IsaacForConditionalGeneration"],
        display_name: "Isaac",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["PaddleOCRVLForConditionalGeneration"],
        display_name: "PaddleOCR-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Idefics3ForConditionalGeneration"],
        display_name: "Idefics3",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["SmolVLMForConditionalGeneration"],
        display_name: "SmolVLM",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Blip2ForConditionalGeneration"],
        display_name: "BLIP-2",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Eagle2_5_VLForConditionalGeneration"],
        display_name: "Eagle 2.5 VL",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["ChameleonForConditionalGeneration"],
        display_name: "Chameleon",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Phi4MMForCausalLM"],
        display_name: "Phi-4 Multimodal",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Gemma3nForConditionalGeneration"],
        display_name: "Gemma 3n VLM",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Molmo2ForConditionalGeneration"],
        display_name: "Molmo 2",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["JinaVLForRanking"],
        display_name: "Jina VL",
        capabilities: ModelCapabilities::new()
            .with_encoder_only()
            .with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["PrithviGeoSpatialMAE", "Terratorch"],
        display_name: "Terratorch",
        capabilities: ModelCapabilities::new()
            .with_encoder_only()
            .with_multimodal(),
    },
    // ─── Audio Models ──────────────────────────────────────────────────────────
    ArchitectureInfo {
        arch_names: &["WhisperForConditionalGeneration"],
        display_name: "Whisper",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen2AudioForConditionalGeneration"],
        display_name: "Qwen2-Audio",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["UltravoxModel"],
        display_name: "Ultravox",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["VoxtralForConditionalGeneration"],
        display_name: "Voxtral",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["GlmAsrForConditionalGeneration"],
        display_name: "GLM-ASR",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["GraniteSpeechForConditionalGeneration"],
        display_name: "Granite Speech",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["FunAudioChatForConditionalGeneration"],
        display_name: "FunAudioChat",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["AudioFlamingo3ForConditionalGeneration"],
        display_name: "AudioFlamingo3",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["MusicFlamingoForConditionalGeneration"],
        display_name: "MusicFlamingo",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Qwen2_5OmniThinkerForConditionalGeneration",
            "Qwen2_5OmniModel",
            "Qwen2_5OmniForConditionalGeneration",
        ],
        display_name: "Qwen2.5-Omni Thinker",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Qwen3OmniMoeThinkerForConditionalGeneration",
            "Qwen3OmniMoeForConditionalGeneration",
        ],
        display_name: "Qwen3-Omni-MoE Thinker",
        capabilities: ModelCapabilities::new().with_multimodal().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["Qwen3ASRForConditionalGeneration"],
        display_name: "Qwen3-ASR",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    // ─── Additional VLMs ───────────────────────────────────────────────────────
    ArchitectureInfo {
        arch_names: &[
            "InternS1ForConditionalGeneration",
            "InternVLForConditionalGeneration",
        ],
        display_name: "InternS1",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["InternS1ProForConditionalGeneration"],
        display_name: "InternS1-Pro",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Step3VLForConditionalGeneration",
            "StepVLForConditionalGeneration",
        ],
        display_name: "Step3-VL",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Ernie4_5_VLMoeForConditionalGeneration"],
        display_name: "ERNIE 4.5 VL MoE",
        capabilities: ModelCapabilities::new().with_multimodal().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["KimiVLForConditionalGeneration"],
        display_name: "Kimi-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["KimiK25ForConditionalGeneration"],
        display_name: "Kimi-K2.5",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["KananaVForConditionalGeneration"],
        display_name: "Kanana-V",
        capabilities: ModelCapabilities::new().with_tp().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Ovis", "OvisForConditionalGeneration"],
        display_name: "Ovis",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Ovis2_5"],
        display_name: "Ovis2.5",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &[
            "Lfm2VlForConditionalGeneration",
            "Lfm2VLForConditionalGeneration",
        ],
        display_name: "LFM2-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["MiniCPMO"],
        display_name: "MiniCPM-O",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["HunYuanVLForConditionalGeneration"],
        display_name: "HunYuan-VL",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["GlmOcrForConditionalGeneration"],
        display_name: "GLM-OCR",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["Glm4vMoeForConditionalGeneration"],
        display_name: "GLM-4.1V MoE",
        capabilities: ModelCapabilities::new().with_multimodal().with_moe(),
    },
    ArchitectureInfo {
        arch_names: &["DeepseekOCR2ForCausalLM"],
        display_name: "DeepSeek-OCR2",
        capabilities: ModelCapabilities::new().with_multimodal(),
    },
    ArchitectureInfo {
        arch_names: &["DotsOCRForCausalLM"],
        display_name: "Dots-OCR",
        capabilities: ModelCapabilities::new().with_multimodal(),
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
        let v32 = find_architecture("DeepseekV32ForCausalLM")
            .expect("should find DeepseekV32ForCausalLM");
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
            // Bagel
            "BagelForConditionalGeneration",
            // Baichuan
            "BaichuanForCausalLM",
            "BaiChuanForCausalLM",
            // BERT
            "BertModel",
            "BertForMaskedLM",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertSpladeSparseEmbeddingModel",
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
            // DeepSeek VL V2
            "DeepSeekVLV2ForCausalLM",
            "DeepseekVLV2ForCausalLM",
            "DeepseekVLV2ForConditionalGeneration",
            // Falcon
            "FalconForCausalLM",
            "RWForCausalLM",
            // BLOOM
            "BloomForCausalLM",
            // GPT family
            "GPT2LMHeadModel",
            "GPT2ForSequenceClassification",
            "GPTNeoXForCausalLM",
            "GPTBigCodeForCausalLM",
            "GPTJForCausalLM",
            // Gemma
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
            "Gemma3ForCausalLM",
            "Gemma3TextModel",
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
            "Llama4VLForConditionalGeneration",
            "MLlama4ForConditionalGeneration",
            "Llama4ForConditionalGeneration",
            // LLaVA
            "LlavaForConditionalGeneration",
            "LlavaNextForConditionalGeneration",
            "MantisForConditionalGeneration",
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
            "PhiMoeForCausalLM",
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
            "ChatGLMForCausalLM",
            // Granite
            "GraniteForCausalLM",
            "GraniteMoeForCausalLM",
            // Jamba
            "JambaForCausalLM",
            "JambaForSequenceClassification",
            // StarCoder
            "Starcoder2ForCausalLM",
            // StableLM
            "StableLMEpochForCausalLM",
            "StableLmForCausalLM",
            // Solar
            "SolarForCausalLM",
            // Eagle3
            "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3",
            "Eagle3Qwen2_5vlForCausalLM",
            "Eagle3Qwen3vlForCausalLM",
            // Speculative decoding draft models
            "EagleLlamaForCausalLM",
            "EagleLlama4ForCausalLM",
            "EagleMiniCPMForCausalLM",
            "EagleMistralLarge3ForCausalLM",
            "EagleDeepSeekMTPModel",
            "DeepSeekMTPModel",
            "ErnieMTPModel",
            "ExaoneMoeMTP",
            "Glm4MoeMTPModel",
            "Glm4MoeLiteMTPModel",
            "GlmOcrMTPModel",
            "LongCatFlashMTPModel",
            "MedusaModel",
            "MiMoMTPModel",
            "OpenPanguMTPModel",
            "Qwen3NextMTP",
            "Step3p5MTP",
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
            "ExaoneMoeForCausalLM",
            "ExaoneMoEForCausalLM",
            // Other
            "PersimmonForCausalLM",
            "DbrxForCausalLM",
            "MistralModel",
            "E5MistralModel",
            "Gemma2Model",
            // New architectures (2025-02 sync)
            "ArcticForCausalLM",
            "AyaVisionForConditionalGeneration",
            "Cohere2VisionForConditionalGeneration",
            "BailingMoeForCausalLM",
            "BailingMoeV2ForCausalLM",
            "BambaForCausalLM",
            "Ernie4_5ForCausalLM",
            "Ernie4_5_MoeForCausalLM",
            "Ernie45MoeForCausalLM",
            "Ernie4_5MoeForCausalLM",
            "FalconH1ForCausalLM",
            "FuyuForCausalLM",
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
            // Embedding / Cross-Encoder / Reward models
            "RobertaForMaskedLM",
            "RobertaModel",
            "XLMRobertaModel",
            "JinaRobertaModel",
            "RobertaForSequenceClassification",
            "XLMRobertaForSequenceClassification",
            "BgeRerankerModel",
            "BgeM3EmbeddingModel",
            "ModernBertModel",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
            "GteModel",
            "GteNewModel",
            "NomicBertModel",
            "GteNewForSequenceClassification",
            "SnowflakeGteNewModel",
            "LlamaBidirectionalModel",
            "LlamaBidirectionalForSequenceClassification",
            "InternLM2ForRewardModel",
            "Qwen2ForRewardModel",
            "Qwen2ForProcessRewardModel",
            "HCXVisionForCausalLM",
            // Multimodal embedding / cross-encoder
            "CLIPModel",
            "SiglipModel",
            "Phi3VForCausalLM",
            "PaliGemmaForConditionalGeneration",
            "LlavaOnevisionForConditionalGeneration",
            "LlavaNextVideoForConditionalGeneration",
            "RForConditionalGeneration",
            "BeeForConditionalGeneration",
            "MolmoForCausalLM",
            "GLM4VForCausalLM",
            "Glm4VForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "Tarsier2ForConditionalGeneration",
            // Qwen2.5-VL / OpenCUA
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen25VLForConditionalGeneration",
            "OpenCUAForConditionalGeneration",
            // Qwen3-VL
            "Qwen3VLForConditionalGeneration",
            // Qwen3-VL-MoE
            "Qwen3VLMoeForConditionalGeneration",
            // Pixtral
            "PixtralForConditionalGeneration",
            // Mistral3
            "Mistral3ForConditionalGeneration",
            // Idefics3
            "Idefics3ForConditionalGeneration",
            // SmolVLM
            "SmolVLMForConditionalGeneration",
            // BLIP-2
            "Blip2ForConditionalGeneration",
            // Eagle 2.5 VL
            "Eagle2_5_VLForConditionalGeneration",
            // Chameleon
            "ChameleonForConditionalGeneration",
            // Phi-4 Multimodal
            "Phi4MMForCausalLM",
            // Gemma3n VLM
            "Gemma3nForConditionalGeneration",
            // Molmo2
            "Molmo2ForConditionalGeneration",
            "JinaVLForRanking",
            "PrithviGeoSpatialMAE",
            "Terratorch",
            // Audio models
            "WhisperForConditionalGeneration",
            "Qwen2AudioForConditionalGeneration",
            "UltravoxModel",
            "VoxtralForConditionalGeneration",
            "GlmAsrForConditionalGeneration",
            "GraniteSpeechForConditionalGeneration",
            "FunAudioChatForConditionalGeneration",
            "AudioFlamingo3ForConditionalGeneration",
            "MusicFlamingoForConditionalGeneration",
            "Qwen2_5OmniThinkerForConditionalGeneration",
            "Qwen2_5OmniModel",
            "Qwen3OmniMoeForConditionalGeneration",
            "Qwen3ASRForConditionalGeneration",
            // Additional VLMs
            "InternS1ForConditionalGeneration",
            "InternVLForConditionalGeneration",
            "InternS1ProForConditionalGeneration",
            "Step3VLForConditionalGeneration",
            "StepVLForConditionalGeneration",
            "Ernie4_5_VLMoeForConditionalGeneration",
            "KimiVLForConditionalGeneration",
            "KimiK25ForConditionalGeneration",
            "KananaVForConditionalGeneration",
            "Ovis",
            "OvisForConditionalGeneration",
            "Ovis2_5",
            "Lfm2VlForConditionalGeneration",
            "MiniCPMO",
            "HunYuanVLForConditionalGeneration",
            "GlmOcrForConditionalGeneration",
            "Glm4vMoeForConditionalGeneration",
            "DeepseekOCR2ForCausalLM",
            "DotsOCRForCausalLM",
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

    #[test]
    fn find_roberta_variants() {
        let names = [
            "RobertaForMaskedLM",
            "RobertaModel",
            "XLMRobertaModel",
            "RobertaForSequenceClassification",
            "XLMRobertaForSequenceClassification",
            "BgeRerankerModel",
        ];
        for name in &names {
            let info = find_architecture(name).unwrap_or_else(|| panic!("should find {name}"));
            assert_eq!(info.display_name, "RoBERTa");
            assert!(info.capabilities.is_encoder_only);
        }
    }

    #[test]
    fn find_modernbert_family() {
        let names = [
            "ModernBertModel",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
        ];
        for name in &names {
            let info = find_architecture(name).unwrap_or_else(|| panic!("should find {name}"));
            assert_eq!(info.display_name, "ModernBERT");
            assert!(info.capabilities.is_encoder_only);
        }
    }

    #[test]
    fn find_reward_models() {
        let info =
            find_architecture("InternLM2ForRewardModel").expect("should find InternLM2 reward");
        assert_eq!(info.display_name, "InternLM2 Reward");
        assert!(info.capabilities.is_encoder_only);

        let q1 = find_architecture("Qwen2ForRewardModel").expect("should find Qwen2 reward");
        let q2 = find_architecture("Qwen2ForProcessRewardModel")
            .expect("should find Qwen2 process reward");
        assert_eq!(q1.display_name, "Qwen2 Reward");
        assert_eq!(q1.display_name, q2.display_name);
        assert!(q1.capabilities.is_encoder_only);
    }

    #[test]
    fn find_speculative_decoding_models() {
        let spec_decode_archs = [
            "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3",
            "Eagle3Qwen2_5vlForCausalLM",
            "EagleLlamaForCausalLM",
            "EagleMistralLarge3ForCausalLM",
            "DeepSeekMTPModel",
            "MedusaModel",
            "MiMoMTPModel",
            "Step3p5MTP",
        ];
        for name in &spec_decode_archs {
            assert!(
                is_supported(name),
                "speculative decoding model {name} should be in registry"
            );
        }
    }

    #[test]
    fn eagle3_aliases_resolve_to_same_entry() {
        let e3 = find_architecture("Eagle3LlamaForCausalLM").unwrap();
        let alias1 = find_architecture("LlamaForCausalLMEagle3").unwrap();
        let alias2 = find_architecture("Eagle3Qwen2_5vlForCausalLM").unwrap();
        assert_eq!(e3.display_name, alias1.display_name);
        assert_eq!(e3.display_name, alias2.display_name);
    }

    #[test]
    fn embedding_models_are_encoder_only() {
        let encoder_only_archs = [
            "BertModel",
            "BertForMaskedLM",
            "BertSpladeSparseEmbeddingModel",
            "HF_ColBERT",
            "VoyageQwen3BidirectionalEmbedModel",
            "RobertaForMaskedLM",
            "XLMRobertaModel",
            "BgeM3EmbeddingModel",
            "ModernBertModel",
            "GteModel",
            "GteNewModel",
            "NomicBertModel",
            "SnowflakeGteNewModel",
            "LlamaBidirectionalModel",
            "InternLM2ForRewardModel",
            "Qwen2ForRewardModel",
            "Gemma2Model",
        ];
        for name in &encoder_only_archs {
            let info = find_architecture(name).unwrap_or_else(|| panic!("should find {name}"));
            assert!(
                info.capabilities.is_encoder_only,
                "{name} should be encoder_only"
            );
        }
    }

    #[test]
    fn tarsier2_is_alias_for_qwen2_vl() {
        let qwen2vl = find_architecture("Qwen2VLForConditionalGeneration").unwrap();
        let tarsier2 = find_architecture("Tarsier2ForConditionalGeneration").unwrap();
        assert_eq!(qwen2vl.display_name, tarsier2.display_name);
        assert!(tarsier2.capabilities.supports_multimodal);
    }

    #[test]
    fn mantis_is_alias_for_llava() {
        let llava = find_architecture("LlavaForConditionalGeneration").unwrap();
        let mantis = find_architecture("MantisForConditionalGeneration").unwrap();
        assert_eq!(llava.display_name, mantis.display_name);
        assert!(mantis.capabilities.supports_multimodal);
    }

    #[test]
    fn qwen25_vl_in_registry() {
        let info = find_architecture("Qwen2_5_VLForConditionalGeneration")
            .expect("should find Qwen2.5-VL");
        assert_eq!(info.display_name, "Qwen2.5-VL");
        assert!(info.capabilities.supports_multimodal);

        let alias = find_architecture("Qwen25VLForConditionalGeneration").unwrap();
        assert_eq!(info.display_name, alias.display_name);
    }

    #[test]
    fn pixtral_in_registry() {
        let info =
            find_architecture("PixtralForConditionalGeneration").expect("should find Pixtral");
        assert_eq!(info.display_name, "Pixtral");
        assert!(info.capabilities.supports_multimodal);
    }

    #[test]
    fn snowflake_gte_resolves_to_gte_nomic() {
        let gte = find_architecture("GteNewModel").unwrap();
        let snowflake = find_architecture("SnowflakeGteNewModel").unwrap();
        assert_eq!(gte.display_name, snowflake.display_name);
        assert!(snowflake.capabilities.is_encoder_only);
    }

    #[test]
    fn gemma2_model_separate_from_mistral() {
        let mistral = find_architecture("MistralModel").unwrap();
        let gemma2 = find_architecture("Gemma2Model").unwrap();
        // They should be in separate entries
        assert_ne!(mistral.display_name, gemma2.display_name);
        assert!(gemma2.capabilities.is_encoder_only);
    }

    #[test]
    fn e5_mistral_grouped_with_mistral_base() {
        let mistral = find_architecture("MistralModel").unwrap();
        let e5 = find_architecture("E5MistralModel").unwrap();
        assert_eq!(mistral.display_name, e5.display_name);
    }
}
