#[macro_use]
mod macros;
pub mod afmoe;
pub mod apertus;
pub mod apertus_quantized;
pub mod arcee;
pub mod arctic;
pub mod aria;
pub mod aya_vision;
pub mod bagel;
pub mod baichuan;
pub mod bailing_moe;
pub mod bamba;
pub mod bert;
pub mod bert_splade;
pub mod bge_reranker;
pub mod blip2;
pub mod bloom;
pub mod bloom_quantized;
pub mod chameleon;
pub mod chatglm;
pub mod chatglm_quantized;
pub mod cohere;
pub mod cohere_quantized;
pub mod colbert;
pub mod dbrx;
pub mod deepseek;
pub mod deepseek_lora;
pub mod deepseek_mtp;
pub mod deepseek_quantized;
pub mod deepseek_vl2;
pub mod dots1;
pub mod e5_mistral;
pub mod eagle2_5_vl;
pub mod eagle3;
pub mod eagle3_mistral_large3;
pub mod eagle_llama;
pub mod ernie45_moe;
pub mod ernie45_vl;
pub mod ernie_mtp;
pub mod exaone;
pub mod exaone4;
pub mod exaone4_quantized;
pub mod exaone_moe;
pub mod exaone_moe_mtp;
pub mod exaone_quantized;
pub mod falcon;
pub mod falcon_h1;
pub mod falcon_quantized;
pub mod flex_olmo;
pub mod fuyu;
pub mod gemma;
pub mod gemma2;
pub mod gemma2_lora;
pub mod gemma2_quantized;
pub mod gemma3;
pub mod gemma3_lora;
pub mod gemma3_quantized;
pub mod gemma3_vlm;
pub mod gemma3n;
pub mod gemma3n_vlm;
pub mod gemma_lora;
pub mod gemma_quantized;
pub mod glm;
pub mod glm4;
pub mod glm4_1v;
pub mod glm4_moe;
pub mod glm4_moe_mtp;
pub mod glm4_quantized;
pub mod glm4v;
pub mod glm_ocr;
pub mod glm_ocr_mtp;
pub mod glm_quantized;
pub mod gpt2;
pub mod gpt2_quantized;
pub mod gpt_bigcode;
pub mod gpt_bigcode_quantized;
pub mod gpt_j;
pub mod gpt_j_quantized;
pub mod gpt_neox;
pub mod gpt_neox_quantized;
pub mod gpt_oss;
pub mod granite;
pub mod granite_quantized;
pub mod granitemoe;
pub mod granitemoe_hybrid;
pub mod granitemoe_shared;
pub mod gritlm;
pub mod grok1;
pub mod gte;
pub mod hunyuan;
pub mod hunyuan_quantized;
pub mod hyperclovax_vision;
pub mod idefics3;
pub mod internlm2;
pub mod internlm2_quantized;
pub mod internlm2_reward;
pub mod internlm2_ve;
pub mod interns1;
pub mod interns1_pro;
pub mod internvl;
pub mod iquest_loopcoder;
pub mod jais;
pub mod jais2;
pub mod jais2_quantized;
pub mod jais_quantized;
pub mod jamba;
pub mod kanana_v;
pub mod kimi_k25;
pub mod kimi_linear;
pub mod kimi_vl;
pub mod lfm2;
pub mod llama;
pub mod llama4;
pub mod llama4_vl;
pub mod llama_bidirectional;
pub mod llama_lora;
pub mod llama_quantized;
pub mod llava;
pub mod llava_onevision;
pub mod longcat_flash;
pub mod longcat_flash_mtp;
pub mod mamba;
pub mod mamba2;
pub mod mimo_mtp;
pub mod mimo_v2_flash;
pub mod minicpm;
pub mod minicpm3;
pub mod minicpm_quantized;
pub mod minicpmv;
pub mod minimax_m2;
pub mod minimax_text01;
pub mod mistral;
pub mod mistral3;
pub mod mistral_lora;
pub mod mistral_quantized;
pub mod mixtral;
pub mod mixtral_lora;
pub mod mixtral_quantized;
pub mod mlp_speculator;
pub mod modernbert;
pub mod molmo;
pub mod molmo2;
pub mod moonvit;
pub mod mpt;
pub mod mpt_quantized;
pub mod mtp_base;
pub mod nemotron;
pub mod nemotron_h;
pub mod nemotron_quantized;
pub mod nvlm_d;
pub mod olmo2;
pub mod olmo2_lora;
pub mod olmo2_quantized;
pub mod olmoe;
pub mod openpangu_mtp;
pub mod opt;
pub mod opt_quantized;
pub mod ouro;
pub mod ovis;
pub mod paligemma;
pub mod pangu;
pub mod persimmon;
pub mod persimmon_quantized;
pub mod phi;
pub mod phi3;
pub mod phi3_lora;
pub mod phi3_quantized;
pub mod phi3v;
pub mod phi4mm;
pub mod phi_quantized;
pub mod phimoe;
pub mod pixtral;
pub mod plamo2;
pub mod plamo3;
pub mod plamo3_quantized;
pub mod qwen;
pub mod qwen2;
pub mod qwen2_5_vl;
pub mod qwen2_lora;
pub mod qwen2_moe;
pub mod qwen2_moe_quantized;
pub mod qwen2_quantized;
pub mod qwen2_reward;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_lora;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_next_mtp;
pub mod qwen3_quantized;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;
pub mod qwen_quantized;
pub mod registry;
pub mod seed_oss;
pub mod starcoder2;
pub mod starcoder2_quantized;
pub mod step1;
pub mod step1_quantized;
pub mod step3_text;
pub mod step3_vl;
pub mod step3p5;
pub mod step3p5_mtp;
pub mod t5;
pub mod tp_layers;
pub mod voyage;
pub mod yi;
pub mod zamba2;

// Re-export tensor parallelism abstractions
pub use tp_layers::{TpContext, TpEmbedding, TpGeGluMlp, TpGeluMlp, TpLinear, TpSwiGluMlp};

pub use afmoe::AfmoeForCausalLM;
pub use apertus::ApertusForCausalLM;
pub use apertus_quantized::QuantizedApertusForCausalLM;
pub use arctic::ArcticForCausalLM;
pub use aria::AriaForConditionalGeneration;
pub use aya_vision::AyaVisionForConditionalGeneration;
pub use bagel::BagelForConditionalGeneration;
pub use baichuan::BaichuanForCausalLM;
pub use bailing_moe::{BailingMoeForCausalLM, BailingMoeV2ForCausalLM};
pub use bamba::BambaForCausalLM;
pub use bert::BertForSequenceEmbedding;
pub use bert_splade::BertSpladeSparseEmbeddingModel;
pub use bge_reranker::BgeRerankerForClassification;
pub use blip2::Blip2ForConditionalGeneration;
pub use bloom::BloomForCausalLM;
pub use bloom_quantized::QuantizedBloomForCausalLM;
pub use chameleon::ChameleonForConditionalGeneration;
pub use chatglm::ChatGLMForCausalLM;
pub use chatglm_quantized::QuantizedChatGLMForCausalLM;
pub use cohere::CohereForCausalLM;
pub use cohere_quantized::QuantizedCohereForCausalLM;
pub use colbert::ColBERTForRetrieval;
pub use dbrx::DbrxForCausalLM;
pub use deepseek::{DeepSeekForCausalLM, GlmMoeDsaForCausalLM};
pub use deepseek_lora::DeepSeekWithLora;
pub use deepseek_mtp::DeepSeekMtpModel;
pub use deepseek_quantized::QuantizedDeepSeekForCausalLM;
pub use deepseek_vl2::DeepSeekVLV2ForConditionalGeneration;
pub use dots1::Dots1ForCausalLM;
pub use e5_mistral::E5MistralForEmbedding;
pub use eagle2_5_vl::Eagle25VLForConditionalGeneration;
pub use eagle3::{Eagle3DraftModel, Eagle3LlamaForCausalLM};
pub use eagle3_mistral_large3::Eagle3MistralLarge3ForCausalLM;
pub use eagle_llama::{Eagle1DraftModel, EagleLlamaForCausalLM};
pub use ernie45_moe::Ernie45MoeForCausalLM;
pub use ernie45_vl::Ernie4_5_VLForConditionalGeneration;
pub use ernie_mtp::ErnieMtpModel;
pub use exaone::ExaoneForCausalLM;
pub use exaone4::Exaone4ForCausalLM;
pub use exaone4_quantized::QuantizedExaone4ForCausalLM;
pub use exaone_moe::ExaoneMoeForCausalLM;
pub use exaone_moe_mtp::ExaoneMoeMtpModel;
pub use exaone_quantized::QuantizedExaoneForCausalLM;
pub use falcon::FalconForCausalLM;
pub use falcon_h1::FalconH1ForCausalLM;
pub use falcon_quantized::QuantizedFalconForCausalLM;
pub use flex_olmo::FlexOlmoForCausalLM;
pub use fuyu::FuyuForCausalLM;
pub use gemma::GemmaForCausalLM;
pub use gemma2::Gemma2ForCausalLM;
pub use gemma2_lora::Gemma2WithLora;
pub use gemma2_quantized::QuantizedGemma2ForCausalLM;
pub use gemma3::Gemma3ForCausalLM;
pub use gemma3_lora::Gemma3WithLora;
pub use gemma3_quantized::QuantizedGemma3ForCausalLM;
pub use gemma3_vlm::Gemma3ForConditionalGeneration;
pub use gemma3n::Gemma3nForCausalLM;
pub use gemma3n_vlm::Gemma3nForConditionalGeneration;
pub use gemma_lora::GemmaWithLora;
pub use gemma_quantized::QuantizedGemmaForCausalLM;
pub use glm::GlmForCausalLM;
pub use glm4::Glm4ForCausalLM;
pub use glm4_1v::Glm4vForConditionalGeneration;
pub use glm4_moe::Glm4MoeForCausalLM;
pub use glm4_moe_mtp::Glm4MoeMtpModel;
pub use glm4_quantized::QuantizedGlm4ForCausalLM;
pub use glm4v::Glm4VForConditionalGeneration;
pub use glm_ocr::GlmOcrForConditionalGeneration;
pub use glm_ocr_mtp::GlmOcrMtpModel;
pub use glm_quantized::QuantizedGlmForCausalLM;
pub use gpt2::GPT2LMHeadModel;
pub use gpt2_quantized::QuantizedGPT2LMHeadModel;
pub use gpt_bigcode::GPTBigCodeForCausalLM;
pub use gpt_bigcode_quantized::QuantizedGPTBigCodeForCausalLM;
pub use gpt_j::GPTJForCausalLM;
pub use gpt_j_quantized::QuantizedGPTJForCausalLM;
pub use gpt_neox::GPTNeoXForCausalLM;
pub use gpt_neox_quantized::QuantizedGPTNeoXForCausalLM;
pub use gpt_oss::GptOssForCausalLM;
pub use granite::GraniteForCausalLM;
pub use granite_quantized::QuantizedGraniteForCausalLM;
pub use granitemoe::GraniteMoeForCausalLM;
pub use granitemoe_hybrid::GraniteMoeHybridForCausalLM;
pub use granitemoe_shared::GraniteMoeSharedForCausalLM;
pub use gritlm::GritLM;
pub use grok1::Grok1ForCausalLM;
pub use gte::{GteNewForEmbedding, GteNewForSequenceClassification};
pub use hunyuan::{HunYuanDenseV1ForCausalLM, HunYuanMoEV1ForCausalLM};
pub use hunyuan_quantized::QuantizedHunYuanDenseForCausalLM;
pub use hyperclovax_vision::HCXVisionForCausalLM;
pub use idefics3::{Idefics3ForConditionalGeneration, SmolVLMForConditionalGeneration};
pub use internlm2::InternLM2ForCausalLM;
pub use internlm2_quantized::QuantizedInternLM2ForCausalLM;
pub use internlm2_reward::InternLM2ForRewardModel;
pub use internlm2_ve::InternLM2VEForCausalLM;
pub use interns1::InternS1ForConditionalGeneration;
pub use interns1_pro::InternS1ProForConditionalGeneration;
pub use internvl::InternVLChatModel;
pub use iquest_loopcoder::IQuestLoopCoderForCausalLM;
pub use jais::JAISLMHeadModel;
pub use jais2::Jais2ForCausalLM;
pub use jais2_quantized::QuantizedJais2ForCausalLM;
pub use jais_quantized::QuantizedJAISLMHeadModel;
pub use jamba::JambaForCausalLM;
pub use kanana_v::KananaVForConditionalGeneration;
pub use kimi_k25::KimiK25ForConditionalGeneration;
pub use kimi_linear::KimiLinearForCausalLM;
pub use kimi_vl::KimiVLForConditionalGeneration;
pub use lfm2::{Lfm2ForCausalLM, Lfm2MoeForCausalLM};
pub use llama::LlamaForCausalLM;
pub use llama4::Llama4ForCausalLM;
pub use llama4_vl::Llama4VLForConditionalGeneration;
pub use llama_bidirectional::LlamaBidirectionalModel;
pub use llama_lora::LlamaWithLora;
pub use llama_quantized::QuantizedLlamaForCausalLM;
pub use llava::LLaVAForConditionalGeneration;
pub use llava_onevision::LlavaOnevisionForConditionalGeneration;
pub use longcat_flash::LongcatFlashForCausalLM;
pub use longcat_flash_mtp::LongCatFlashMtpModel;
pub use mamba::MambaForCausalLM;
pub use mamba2::Mamba2ForCausalLM;
pub use mimo_mtp::MiMoMtpModel;
pub use mimo_v2_flash::MiMoV2FlashForCausalLM;
pub use minicpm::MiniCPMForCausalLM;
pub use minicpm3::MiniCPM3ForCausalLM;
pub use minicpm_quantized::QuantizedMiniCPMForCausalLM;
pub use minicpmv::MiniCPMVForConditionalGeneration;
pub use minimax_m2::MiniMaxM2ForCausalLM;
pub use minimax_text01::MiniMaxText01ForCausalLM;
pub use mistral::MistralForCausalLM;
pub use mistral3::Mistral3ForConditionalGeneration;
pub use mistral_lora::MistralWithLora;
pub use mistral_quantized::QuantizedMistralForCausalLM;
pub use mixtral::{MixtralForCausalLM, MixtralTpForCausalLM};
pub use mixtral_lora::MixtralWithLora;
pub use mixtral_quantized::QuantizedMixtralForCausalLM;
pub use mlp_speculator::MLPSpeculatorModel;
pub use modernbert::ModernBertForEmbedding;
pub use molmo::MolmoForCausalLM;
pub use molmo2::Molmo2ForConditionalGeneration;
pub use mpt::MptForCausalLM;
pub use mpt_quantized::QuantizedMptForCausalLM;
pub use mtp_base::MtpDraftModel;
pub use nemotron::NemotronForCausalLM;
pub use nemotron_h::NemotronHForCausalLM;
pub use nemotron_quantized::QuantizedNemotronForCausalLM;
pub use nvlm_d::NVLMDModel;
pub use olmo2::Olmo2ForCausalLM;
pub use olmo2_lora::Olmo2WithLora;
pub use olmo2_quantized::QuantizedOlmo2ForCausalLM;
pub use olmoe::OlmoeForCausalLM;
pub use openpangu_mtp::OpenPanguMtpModel;
pub use opt::OPTForCausalLM;
pub use opt_quantized::QuantizedOPTForCausalLM;
pub use ouro::OuroForCausalLM;
pub use ovis::OvisForConditionalGeneration;
pub use paligemma::PaliGemmaForConditionalGeneration;
pub use pangu::{PanguEmbeddedForCausalLM, PanguProMoEV2ForCausalLM, PanguUltraMoEForCausalLM};
pub use persimmon::PersimmonForCausalLM;
pub use persimmon_quantized::QuantizedPersimmonForCausalLM;
pub use phi::PhiForCausalLM;
pub use phi3::Phi3ForCausalLM;
pub use phi3_lora::Phi3WithLora;
pub use phi3_quantized::QuantizedPhi3ForCausalLM;
pub use phi3v::Phi3VForCausalLM;
pub use phi4mm::Phi4MMForCausalLM;
pub use phi_quantized::QuantizedPhiForCausalLM;
pub use phimoe::PhiMoeForCausalLM;
pub use pixtral::PixtralForConditionalGeneration;
pub use plamo2::Plamo2ForCausalLM;
pub use plamo3::Plamo3ForCausalLM;
pub use plamo3_quantized::QuantizedPlamo3ForCausalLM;
pub use qwen::QWenLMHeadModel;
pub use qwen2::Qwen2ForCausalLM;
pub use qwen2_5_vl::Qwen25VLForConditionalGeneration;
pub use qwen2_lora::Qwen2WithLora;
pub use qwen2_moe::Qwen2MoeForCausalLM;
pub use qwen2_moe_quantized::QuantizedQwen2MoeForCausalLM;
pub use qwen2_quantized::QuantizedQwen2ForCausalLM;
pub use qwen2_reward::{Qwen2ForProcessRewardModel, Qwen2ForRewardModel};
pub use qwen2_vl::Qwen2VLForConditionalGeneration;
pub use qwen3::Qwen3ForCausalLM;
pub use qwen3_lora::Qwen3WithLora;
pub use qwen3_moe::Qwen3MoeForCausalLM;
pub use qwen3_next::Qwen3NextForCausalLM;
pub use qwen3_next_mtp::Qwen3NextMtpModel;
pub use qwen3_quantized::QuantizedQwen3ForCausalLM;
pub use qwen3_vl::Qwen3VLForConditionalGeneration;
pub use qwen3_vl_moe::Qwen3VLMoeForConditionalGeneration;
pub use qwen_quantized::QuantizedQWenLMHeadModel;
pub use registry::{
    find_architecture, supported_architectures, ArchitectureInfo, ModelCapabilities,
};
pub use seed_oss::SeedOssForCausalLM;
pub use starcoder2::StarCoder2ForCausalLM;
pub use starcoder2_quantized::QuantizedStarCoder2ForCausalLM;
pub use step1::Step1ForCausalLM;
pub use step1_quantized::QuantizedStep1ForCausalLM;
pub use step3_text::Step3TextForCausalLM;
pub use step3_vl::Step3VLForConditionalGeneration;
pub use step3p5::Step3p5ForCausalLM;
pub use step3p5_mtp::Step3p5MtpModel;
pub use t5::T5ForConditionalGeneration;
pub use voyage::VoyageForEmbedding;
pub use yi::YiForCausalLM;
pub use zamba2::Zamba2ForCausalLM;

use std::path::Path;

use candle_core::Device;
use candle_nn::VarBuilder;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::ModelForward;
use crate::kv_cache::{CacheConfig, CacheError, KVCacheDtype, KVCacheManager, MLACacheConfig};
use crate::lora::LoraModel;
use crate::quantization::{
    create_weight_loader_with_params, detect_from_directory, DetectedQuantConfig,
    QuantizationMethod,
};

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("model load error: {0}")]
    Load(#[from] candle_core::Error),
}

/// Extract the architecture identifier from config, returning an error on empty list.
fn get_arch(cfg: &ModelConfig) -> Result<&str, ModelError> {
    cfg.architectures
        .first()
        .map(|s| s.as_str())
        .ok_or_else(|| ModelError::UnsupportedArchitecture("empty architectures list".into()))
}

/// Construct the appropriate model from config.architectures[0].
pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "DeepseekForCausalLM"
        | "DeepseekV2ForCausalLM"
        | "DeepseekV3ForCausalLM"
        | "DeepseekV32ForCausalLM"
        | "GlmMoeDsaForCausalLM"
        | "MistralLarge3ForCausalLM" => Ok(Box::new(DeepSeekForCausalLM::new(cfg, vb)?)),
        "GemmaForCausalLM" => Ok(Box::new(GemmaForCausalLM::new(cfg, vb)?)),
        "Gemma2ForCausalLM" | "Gemma2Model" => Ok(Box::new(Gemma2ForCausalLM::new(cfg, vb)?)),
        "Gemma3ForCausalLM" | "Gemma3TextModel" => Ok(Box::new(Gemma3ForCausalLM::new(cfg, vb)?)),
        "JambaForCausalLM" | "JambaForSequenceClassification" => {
            Ok(Box::new(JambaForCausalLM::new(cfg, vb)?))
        }
        "LlamaForCausalLM"
        | "LlamaModel"
        | "LLaMAForCausalLM"
        | "AquilaModel"
        | "AquilaForCausalLM"
        | "CwmForCausalLM"
        | "InternLMForCausalLM"
        | "InternLM3ForCausalLM"
        | "IQuestCoderForCausalLM"
        | "XverseForCausalLM"
        | "SolarForCausalLM"
        | "Fairseq2LlamaForCausalLM"
        | "OrionForCausalLM"
        | "TeleChatForCausalLM"
        | "TeleChat2ForCausalLM"
        | "TeleFLMForCausalLM"
        | "DeciLMForCausalLM"
        | "OlmoForCausalLM" => Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?)),
        "MistralForCausalLM" => Ok(Box::new(MistralForCausalLM::new(cfg, vb)?)),
        "MixtralForCausalLM" => Ok(Box::new(MixtralForCausalLM::new(cfg, vb)?)),
        "Qwen2ForCausalLM" | "Qwen2Model" => Ok(Box::new(Qwen2ForCausalLM::new(cfg, vb)?)),
        "Qwen2MoeForCausalLM" => Ok(Box::new(Qwen2MoeForCausalLM::new(cfg, vb)?)),
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new(cfg, vb)?)),
        "Qwen3MoeForCausalLM" => Ok(Box::new(Qwen3MoeForCausalLM::new(cfg, vb)?)),
        "Phi3ForCausalLM" => Ok(Box::new(Phi3ForCausalLM::new(cfg, vb)?)),
        "Phi3VForCausalLM" => Ok(Box::new(Phi3VForCausalLM::new(cfg, vb)?)),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Ok(Box::new(Olmo2ForCausalLM::new(cfg, vb)?)),
        "GPT2LMHeadModel" | "GPT2ForSequenceClassification" => {
            Ok(Box::new(GPT2LMHeadModel::new(cfg, vb)?))
        }
        "GlmForCausalLM" => Ok(Box::new(GlmForCausalLM::new(cfg, vb)?)),
        "Glm4ForCausalLM" => Ok(Box::new(Glm4ForCausalLM::new(cfg, vb)?)),
        "Glm4MoeForCausalLM" | "Glm4MoeLiteForCausalLM" => {
            Ok(Box::new(Glm4MoeForCausalLM::new(cfg, vb)?))
        }
        "BagelForConditionalGeneration" => {
            Ok(Box::new(BagelForConditionalGeneration::new(cfg, vb)?))
        }
        "BaichuanForCausalLM" | "BaiChuanForCausalLM" => {
            Ok(Box::new(BaichuanForCausalLM::new(cfg, vb)?))
        }
        "InternLM2ForCausalLM" => Ok(Box::new(InternLM2ForCausalLM::new(cfg, vb)?)),
        "InternLM2VEForCausalLM" => Ok(Box::new(InternLM2VEForCausalLM::new(cfg, vb)?)),
        "IQuestLoopCoderForCausalLM" => Ok(Box::new(IQuestLoopCoderForCausalLM::new(cfg, vb)?)),
        "LongcatFlashForCausalLM" => Ok(Box::new(LongcatFlashForCausalLM::new(cfg, vb)?)),
        "MiMoForCausalLM" | "MiMoV2FlashForCausalLM" => {
            Ok(Box::new(MiMoV2FlashForCausalLM::new(cfg, vb)?))
        }
        "CohereForCausalLM" | "Cohere2ForCausalLM" => {
            Ok(Box::new(CohereForCausalLM::new(cfg, vb)?))
        }
        "GPTNeoXForCausalLM" | "StableLMEpochForCausalLM" | "StableLmForCausalLM" => {
            Ok(Box::new(GPTNeoXForCausalLM::new(cfg, vb)?))
        }
        "SeedOssForCausalLM" => Ok(Box::new(SeedOssForCausalLM::new(cfg, vb)?)),
        "Starcoder2ForCausalLM" => Ok(Box::new(StarCoder2ForCausalLM::new(cfg, vb)?)),
        "BloomForCausalLM" => Ok(Box::new(BloomForCausalLM::new(cfg, vb)?)),
        "FalconForCausalLM" | "RWForCausalLM" => Ok(Box::new(FalconForCausalLM::new(cfg, vb)?)),
        "PhiForCausalLM" => Ok(Box::new(PhiForCausalLM::new(cfg, vb)?)),
        "YiForCausalLM" => Ok(Box::new(YiForCausalLM::new(cfg, vb)?)),
        "MptForCausalLM" | "MPTForCausalLM" => Ok(Box::new(MptForCausalLM::new(cfg, vb)?)),
        "PersimmonForCausalLM" => Ok(Box::new(PersimmonForCausalLM::new(cfg, vb)?)),
        "ExaoneForCausalLM" => Ok(Box::new(ExaoneForCausalLM::new(cfg, vb)?)),
        "DbrxForCausalLM" => Ok(Box::new(DbrxForCausalLM::new(cfg, vb)?)),
        "MambaForCausalLM" | "FalconMambaForCausalLM" => {
            Ok(Box::new(MambaForCausalLM::new(cfg, vb)?))
        }
        "Mamba2ForCausalLM" => Ok(Box::new(Mamba2ForCausalLM::new(cfg, vb)?)),
        "BambaForCausalLM" => Ok(Box::new(BambaForCausalLM::new(cfg, vb)?)),
        "FalconH1ForCausalLM" => Ok(Box::new(FalconH1ForCausalLM::new(cfg, vb)?)),
        "FuyuForCausalLM" => Ok(Box::new(FuyuForCausalLM::new(cfg, vb)?)),
        "Zamba2ForCausalLM" => Ok(Box::new(Zamba2ForCausalLM::new(cfg, vb)?)),
        "GraniteMoeHybridForCausalLM" => Ok(Box::new(GraniteMoeHybridForCausalLM::new(cfg, vb)?)),
        "LlavaForConditionalGeneration"
        | "LlavaNextForConditionalGeneration"
        | "MantisForConditionalGeneration" => Ok(Box::new(
            LLaVAForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "LlavaOnevisionForConditionalGeneration"
        | "LlavaNextVideoForConditionalGeneration"
        | "RForConditionalGeneration"
        | "BeeForConditionalGeneration" => Ok(Box::new(
            LlavaOnevisionForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "Qwen2VLForConditionalGeneration" | "Tarsier2ForConditionalGeneration" => Ok(Box::new(
            Qwen2VLForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "Qwen2_5_VLForConditionalGeneration"
        | "Qwen25VLForConditionalGeneration"
        | "OpenCUAForConditionalGeneration" => {
            Ok(Box::new(Qwen25VLForConditionalGeneration::new(cfg, vb)?))
        }
        "Qwen3VLForConditionalGeneration" => Ok(Box::new(
            Qwen3VLForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "Qwen3VLMoeForConditionalGeneration" => Ok(Box::new(
            Qwen3VLMoeForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "PixtralForConditionalGeneration" => {
            Ok(Box::new(PixtralForConditionalGeneration::new(cfg, vb)?))
        }
        "Mistral3ForConditionalGeneration" => {
            Ok(Box::new(Mistral3ForConditionalGeneration::new(cfg, vb)?))
        }
        "Idefics3ForConditionalGeneration" => {
            Ok(Box::new(Idefics3ForConditionalGeneration::new(cfg, vb)?))
        }
        "SmolVLMForConditionalGeneration" => {
            Ok(Box::new(SmolVLMForConditionalGeneration::new(cfg, vb)?))
        }
        "Blip2ForConditionalGeneration" => {
            Ok(Box::new(Blip2ForConditionalGeneration::new(cfg, vb)?))
        }
        "Eagle2_5_VLForConditionalGeneration" => {
            Ok(Box::new(Eagle25VLForConditionalGeneration::new(cfg, vb)?))
        }
        "ChameleonForConditionalGeneration" => {
            Ok(Box::new(ChameleonForConditionalGeneration::new(cfg, vb)?))
        }
        "Phi4MMForCausalLM" => Ok(Box::new(Phi4MMForCausalLM::new(cfg, vb)?)),
        "Gemma3nForConditionalGeneration" => {
            Ok(Box::new(Gemma3nForConditionalGeneration::new(cfg, vb)?))
        }
        "Molmo2ForConditionalGeneration" => {
            Ok(Box::new(Molmo2ForConditionalGeneration::new(cfg, vb)?))
        }
        "MiniCPMV" | "MiniCPMVForConditionalGeneration" => {
            Ok(Box::new(MiniCPMVForConditionalGeneration::new(cfg, vb)?))
        }
        "PaliGemmaForConditionalGeneration" => Ok(Box::new(
            PaliGemmaForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "Gemma3ForConditionalGeneration" => Ok(Box::new(
            Gemma3ForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "DeepSeekVLV2ForCausalLM"
        | "DeepseekVLV2ForCausalLM"
        | "DeepseekVLV2ForConditionalGeneration" => Ok(Box::new(
            DeepSeekVLV2ForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "InternS1ForConditionalGeneration" => Ok(Box::new(
            InternS1ForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "InternS1ProForConditionalGeneration" => Ok(Box::new(
            InternS1ProForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "InternVLChatModel" | "H2OVLChatModel" | "SkyworkR1VChatModel" => {
            Ok(Box::new(InternVLChatModel::from_model_config(cfg, vb)?))
        }
        "NVLM_D_Model" | "NVLM_D" => Ok(Box::new(NVLMDModel::new(cfg, vb)?)),
        "MolmoForCausalLM" => Ok(Box::new(MolmoForCausalLM::from_model_config(cfg, vb)?)),
        "Glm4vForConditionalGeneration" | "Glm4vMoeForConditionalGeneration" => {
            Ok(Box::new(Glm4vForConditionalGeneration::new(cfg, vb)?))
        }
        "GLM4VForCausalLM" | "Glm4VForConditionalGeneration" => {
            Ok(Box::new(Glm4VForConditionalGeneration::new(cfg, vb)?))
        }
        "GlmOcrForConditionalGeneration" => {
            Ok(Box::new(GlmOcrForConditionalGeneration::new(cfg, vb)?))
        }
        "BertModel"
        | "BertForMaskedLM"
        | "BertForSequenceClassification"
        | "BertForTokenClassification"
        | "RobertaModel"
        | "RobertaForMaskedLM" => Ok(Box::new(BertForSequenceEmbedding::new(cfg, vb)?)),
        "BertSpladeSparseEmbeddingModel" => {
            Ok(Box::new(BertSpladeSparseEmbeddingModel::new(cfg, vb)?))
        }
        "HF_ColBERT" | "ColBERTModel" => Ok(Box::new(ColBERTForRetrieval::new(cfg, vb)?)),
        "GteNewModel" | "GteModel" | "SnowflakeGteNewModel" => {
            Ok(Box::new(GteNewForEmbedding::new(cfg, vb)?))
        }
        "NomicBertModel" => Ok(Box::new(GteNewForEmbedding::new_nomic(cfg, vb)?)),
        "XLMRobertaModel"
        | "JinaRobertaModel"
        | "XLMRobertaForSequenceClassification"
        | "BgeM3EmbeddingModel" => Ok(Box::new(GteNewForEmbedding::new_jina(cfg, vb)?)),
        "GteNewForSequenceClassification" => {
            Ok(Box::new(GteNewForSequenceClassification::new(cfg, vb)?))
        }
        // Eagle3LlamaForCausalLM: loaded by speculative decode pipeline (not from_config)
        // because its forward pass requires hidden_states from the target model.
        "Step3p5ForCausalLM" => Ok(Box::new(Step3p5ForCausalLM::new(cfg, vb)?)),
        "LlamaBidirectionalModel" | "LlamaBidirectionalForSequenceClassification" => {
            Ok(Box::new(LlamaBidirectionalModel::new(cfg, vb)?))
        }
        "VoyageQwen3BidirectionalEmbedModel" => Ok(Box::new(VoyageForEmbedding::new(cfg, vb)?)),
        "MistralModel" | "E5MistralModel" => Ok(Box::new(E5MistralForEmbedding::new(cfg, vb)?)),
        "ModernBertModel"
        | "ModernBertForSequenceClassification"
        | "ModernBertForTokenClassification" => Ok(Box::new(ModernBertForEmbedding::new(cfg, vb)?)),
        "RobertaForSequenceClassification" | "BgeRerankerModel" => {
            Ok(Box::new(BgeRerankerForClassification::new(cfg, vb)?))
        }
        "InternLM2ForRewardModel" => Ok(Box::new(InternLM2ForRewardModel::new(cfg, vb)?)),
        "Qwen2ForRewardModel" => Ok(Box::new(Qwen2ForRewardModel::new(cfg, vb)?)),
        "Qwen2ForProcessRewardModel" => Ok(Box::new(Qwen2ForProcessRewardModel::new(cfg, vb)?)),
        "MiniMaxForCausalLM" | "MiniMaxText01ForCausalLM" | "MiniMaxM1ForCausalLM" => {
            Ok(Box::new(MiniMaxText01ForCausalLM::new(cfg, vb)?))
        }
        "MiniMaxM2ForCausalLM" => Ok(Box::new(MiniMaxM2ForCausalLM::new(cfg, vb)?)),
        "ApertusForCausalLM" => Ok(Box::new(ApertusForCausalLM::new(cfg, vb)?)),
        "Plamo2ForCausalLM" => Ok(Box::new(Plamo2ForCausalLM::new(cfg, vb)?)),
        "Plamo3ForCausalLM" => Ok(Box::new(Plamo3ForCausalLM::new(cfg, vb)?)),
        "AfmoeForCausalLM" => Ok(Box::new(AfmoeForCausalLM::new(cfg, vb)?)),
        "ArceeForCausalLM" => Ok(Box::new(arcee::new_arcee(cfg, vb)?)),
        "ArcticForCausalLM" => Ok(Box::new(ArcticForCausalLM::new(cfg, vb)?)),
        "AriaForConditionalGeneration" => Ok(Box::new(AriaForConditionalGeneration::new(cfg, vb)?)),
        "AyaVisionForConditionalGeneration" | "Cohere2VisionForConditionalGeneration" => {
            Ok(Box::new(AyaVisionForConditionalGeneration::new(cfg, vb)?))
        }
        "ChatGLMModel" | "ChatGLMForCausalLM" | "ChatGLMForConditionalGeneration" => {
            Ok(Box::new(ChatGLMForCausalLM::new(cfg, vb)?))
        }
        "Dots1ForCausalLM" => Ok(Box::new(Dots1ForCausalLM::new(cfg, vb)?)),
        "Exaone4ForCausalLM" => Ok(Box::new(Exaone4ForCausalLM::new(cfg, vb)?)),
        "ExaoneMoeForCausalLM" | "ExaoneMoEForCausalLM" => {
            Ok(Box::new(ExaoneMoeForCausalLM::new(cfg, vb)?))
        }
        "FlexOlmoForCausalLM" => Ok(Box::new(FlexOlmoForCausalLM::new(cfg, vb)?)),
        "Gemma3nForCausalLM" => Ok(Box::new(Gemma3nForCausalLM::new(cfg, vb)?)),
        "GPTBigCodeForCausalLM" => Ok(Box::new(GPTBigCodeForCausalLM::new(cfg, vb)?)),
        "GPTJForCausalLM" => Ok(Box::new(GPTJForCausalLM::new(cfg, vb)?)),
        "GptOssForCausalLM" => Ok(Box::new(GptOssForCausalLM::new(cfg, vb)?)),
        "GraniteForCausalLM" => Ok(Box::new(GraniteForCausalLM::new(cfg, vb)?)),
        "GraniteMoeForCausalLM" => Ok(Box::new(GraniteMoeForCausalLM::new(cfg, vb)?)),
        "GraniteMoeSharedForCausalLM" => Ok(Box::new(GraniteMoeSharedForCausalLM::new(cfg, vb)?)),
        "GritLM" => Ok(Box::new(GritLM::new(cfg, vb)?)),
        "Grok1ForCausalLM" | "Grok1ModelForCausalLM" => {
            Ok(Box::new(Grok1ForCausalLM::new(cfg, vb)?))
        }
        "HunYuanDenseV1ForCausalLM" => Ok(Box::new(HunYuanDenseV1ForCausalLM::new(cfg, vb)?)),
        "HunYuanMoEV1ForCausalLM" => Ok(Box::new(HunYuanMoEV1ForCausalLM::new(cfg, vb)?)),
        "HCXVisionForCausalLM" => Ok(Box::new(HCXVisionForCausalLM::new(cfg, vb)?)),
        "JAISLMHeadModel" => Ok(Box::new(JAISLMHeadModel::new(cfg, vb)?)),
        "Jais2ForCausalLM" => Ok(Box::new(Jais2ForCausalLM::new(cfg, vb)?)),
        "KananaVForConditionalGeneration" => {
            Ok(Box::new(KananaVForConditionalGeneration::new(cfg, vb)?))
        }
        "KimiK25ForConditionalGeneration" => Ok(Box::new(
            KimiK25ForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "KimiLinearForCausalLM" => Ok(Box::new(KimiLinearForCausalLM::new(cfg, vb)?)),
        "KimiVLForConditionalGeneration" => Ok(Box::new(
            KimiVLForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "Lfm2ForCausalLM" => Ok(Box::new(Lfm2ForCausalLM::new(cfg, vb)?)),
        "Lfm2MoeForCausalLM" => Ok(Box::new(Lfm2MoeForCausalLM::new(cfg, vb)?)),
        "Llama4ForCausalLM" => Ok(Box::new(Llama4ForCausalLM::new(cfg, vb)?)),
        "Llama4VLForConditionalGeneration"
        | "MLlama4ForConditionalGeneration"
        | "Llama4ForConditionalGeneration" => Ok(Box::new(
            Llama4VLForConditionalGeneration::from_model_config(cfg, vb)?,
        )),
        "MiniCPMForCausalLM" => Ok(Box::new(MiniCPMForCausalLM::new(cfg, vb)?)),
        "MiniCPM3ForCausalLM" => Ok(Box::new(MiniCPM3ForCausalLM::new(cfg, vb)?)),
        "NemotronForCausalLM" => Ok(Box::new(NemotronForCausalLM::new(cfg, vb)?)),
        "NemotronHForCausalLM" | "NemotronHPuzzleForCausalLM" => {
            Ok(Box::new(NemotronHForCausalLM::new(cfg, vb)?))
        }
        "OlmoeForCausalLM" => Ok(Box::new(OlmoeForCausalLM::new(cfg, vb)?)),
        "OPTForCausalLM" => Ok(Box::new(OPTForCausalLM::new(cfg, vb)?)),
        "OuroForCausalLM" => Ok(Box::new(OuroForCausalLM::new(cfg, vb)?)),
        "OvisForConditionalGeneration" => Ok(Box::new(OvisForConditionalGeneration::new(cfg, vb)?)),
        "PanguEmbeddedForCausalLM" => Ok(Box::new(PanguEmbeddedForCausalLM::new(cfg, vb)?)),
        "PanguProMoEV2ForCausalLM" => Ok(Box::new(PanguProMoEV2ForCausalLM::new(cfg, vb)?)),
        "PanguUltraMoEForCausalLM" => Ok(Box::new(PanguUltraMoEForCausalLM::new(cfg, vb)?)),
        "PhiMoEForCausalLM" | "PhiMoeForCausalLM" => Ok(Box::new(PhiMoeForCausalLM::new(cfg, vb)?)),
        "QWenLMHeadModel" => Ok(Box::new(QWenLMHeadModel::new(cfg, vb)?)),
        "Qwen3NextForCausalLM" => Ok(Box::new(Qwen3NextForCausalLM::new(cfg, vb)?)),
        "Step1ForCausalLM" => Ok(Box::new(Step1ForCausalLM::new(cfg, vb)?)),
        "Step3TextForCausalLM" => Ok(Box::new(Step3TextForCausalLM::new(cfg, vb)?)),
        "Step3VLForConditionalGeneration" => {
            Ok(Box::new(Step3VLForConditionalGeneration::new(cfg, vb)?))
        }
        "BailingMoeForCausalLM" | "BailingMoeV2ForCausalLM" => {
            Ok(Box::new(BailingMoeForCausalLM::new(cfg, vb)?))
        }
        "Ernie45MoeForCausalLM"
        | "Ernie4_5MoeForCausalLM"
        | "Ernie4_5_MoeForCausalLM"
        | "Ernie4_5ForCausalLM" => Ok(Box::new(Ernie45MoeForCausalLM::new(cfg, vb)?)),
        "Ernie4_5_VLMoeForConditionalGeneration" => {
            Ok(Box::new(Ernie4_5_VLForConditionalGeneration::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Construct an encoder-decoder model from config.architectures[0].
pub fn from_config_encoder_decoder(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn crate::engine::ModelForEncoderDecoder>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "T5ForConditionalGeneration" | "T5Model" => {
            Ok(Box::new(T5ForConditionalGeneration::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Construct a quantized model with automatic quantization detection.
///
/// This function detects the quantization method from the model directory
/// (by reading config.json and quantize_config.json) and returns the appropriate
/// quantized model variant.
///
/// If no quantization is detected, it falls back to the unquantized model.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `model_dir` - Path to the model directory for quantization detection
///
/// # Returns
/// A boxed model implementing ModelForward
pub fn from_config_quantized(
    cfg: &ModelConfig,
    vb: VarBuilder<'static>,
    model_dir: &Path,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let detected = detect_from_directory(model_dir);
    from_config_with_quant(cfg, vb, &detected)
}

/// Construct a model with explicit quantization configuration.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `quant_config` - Detected quantization configuration
pub fn from_config_with_quant(
    cfg: &ModelConfig,
    vb: VarBuilder<'static>,
    quant_config: &DetectedQuantConfig,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;

    if quant_config.method == QuantizationMethod::None {
        return from_config(cfg, vb);
    }

    let weight_loader = create_weight_loader_with_params(vb.clone(), quant_config);

    match arch {
        "Qwen3ForCausalLM" => Ok(Box::new(QuantizedQwen3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "LlamaForCausalLM"
        | "LlamaModel"
        | "LLaMAForCausalLM"
        | "AquilaModel"
        | "AquilaForCausalLM"
        | "CwmForCausalLM"
        | "InternLMForCausalLM"
        | "InternLM3ForCausalLM"
        | "IQuestCoderForCausalLM"
        | "XverseForCausalLM"
        | "SolarForCausalLM"
        | "Fairseq2LlamaForCausalLM"
        | "OrionForCausalLM"
        | "TeleChatForCausalLM"
        | "TeleChat2ForCausalLM"
        | "TeleFLMForCausalLM"
        | "DeciLMForCausalLM"
        | "OlmoForCausalLM"
        | "YiForCausalLM"
        | "BaichuanForCausalLM"
        | "BaiChuanForCausalLM"
        | "SeedOssForCausalLM" => Ok(Box::new(QuantizedLlamaForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GemmaForCausalLM" => Ok(Box::new(QuantizedGemmaForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Gemma2ForCausalLM" | "Gemma2Model" => Ok(Box::new(QuantizedGemma2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MistralForCausalLM" => Ok(Box::new(QuantizedMistralForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Qwen2ForCausalLM" | "Qwen2Model" => Ok(Box::new(QuantizedQwen2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Phi3ForCausalLM" => Ok(Box::new(QuantizedPhi3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Gemma3ForCausalLM" | "Gemma3TextModel" => Ok(Box::new(QuantizedGemma3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MixtralForCausalLM" => Ok(Box::new(QuantizedMixtralForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "DeepseekForCausalLM"
        | "DeepseekV2ForCausalLM"
        | "DeepseekV3ForCausalLM"
        | "DeepseekV32ForCausalLM"
        | "GlmMoeDsaForCausalLM"
        | "MistralLarge3ForCausalLM" => Ok(Box::new(QuantizedDeepSeekForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Ok(Box::new(QuantizedOlmo2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "CohereForCausalLM" | "Cohere2ForCausalLM" => Ok(Box::new(
            QuantizedCohereForCausalLM::new(cfg, vb, weight_loader.as_ref())?,
        )),
        "InternLM2ForCausalLM" => Ok(Box::new(QuantizedInternLM2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GPTNeoXForCausalLM" | "StableLMEpochForCausalLM" | "StableLmForCausalLM" => Ok(Box::new(
            QuantizedGPTNeoXForCausalLM::new(cfg, vb, weight_loader.as_ref())?,
        )),
        "Qwen2MoeForCausalLM" => Ok(Box::new(QuantizedQwen2MoeForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Starcoder2ForCausalLM" => Ok(Box::new(QuantizedStarCoder2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "FalconForCausalLM" | "RWForCausalLM" => Ok(Box::new(QuantizedFalconForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "ExaoneForCausalLM" => Ok(Box::new(QuantizedExaoneForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "PhiForCausalLM" => Ok(Box::new(QuantizedPhiForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "BloomForCausalLM" => Ok(Box::new(QuantizedBloomForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GPT2LMHeadModel" | "GPT2ForSequenceClassification" => Ok(Box::new(
            QuantizedGPT2LMHeadModel::new(cfg, vb, weight_loader.as_ref())?,
        )),
        "GPTBigCodeForCausalLM" => Ok(Box::new(QuantizedGPTBigCodeForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GPTJForCausalLM" => Ok(Box::new(QuantizedGPTJForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MptForCausalLM" | "MPTForCausalLM" => Ok(Box::new(QuantizedMptForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "OPTForCausalLM" => Ok(Box::new(QuantizedOPTForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "PersimmonForCausalLM" => Ok(Box::new(QuantizedPersimmonForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GraniteForCausalLM" => Ok(Box::new(QuantizedGraniteForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "ChatGLMModel" | "ChatGLMForCausalLM" | "ChatGLMForConditionalGeneration" => Ok(Box::new(
            QuantizedChatGLMForCausalLM::new(cfg, vb, weight_loader.as_ref())?,
        )),
        "JAISLMHeadModel" => Ok(Box::new(QuantizedJAISLMHeadModel::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "QWenLMHeadModel" => Ok(Box::new(QuantizedQWenLMHeadModel::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "GlmForCausalLM" => Ok(Box::new(QuantizedGlmForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Glm4ForCausalLM" => Ok(Box::new(QuantizedGlm4ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "NemotronForCausalLM" => Ok(Box::new(QuantizedNemotronForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Jais2ForCausalLM" | "ArceeForCausalLM" => Ok(Box::new(QuantizedJais2ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "MiniCPMForCausalLM" => Ok(Box::new(QuantizedMiniCPMForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "ApertusForCausalLM" => Ok(Box::new(QuantizedApertusForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "HunYuanDenseV1ForCausalLM" => Ok(Box::new(QuantizedHunYuanDenseForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Exaone4ForCausalLM" => Ok(Box::new(QuantizedExaone4ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Plamo3ForCausalLM" => Ok(Box::new(QuantizedPlamo3ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        "Step1ForCausalLM" => Ok(Box::new(QuantizedStep1ForCausalLM::new(
            cfg,
            vb,
            weight_loader.as_ref(),
        )?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Get the detected quantization method for a model directory.
///
/// This is useful for checking quantization before loading.
pub fn detect_quantization(model_dir: &Path) -> DetectedQuantConfig {
    detect_from_directory(model_dir)
}

/// Construct the appropriate MTP draft model from `cfg.architectures[0]`.
///
/// Used by the speculative decoding engine to load the correct MTP model type.
pub fn mtp_from_config(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn MtpDraftModel>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "DeepSeekMTPModel" | "EagleDeepSeekMTPModel" => {
            Ok(Box::new(DeepSeekMtpModel::new(cfg, vb)?))
        }
        "ErnieMTPModel" => Ok(Box::new(ErnieMtpModel::new(cfg, vb)?)),
        "ExaoneMoeMTP" => Ok(Box::new(ExaoneMoeMtpModel::new(cfg, vb)?)),
        "Glm4MoeMTPModel" | "Glm4MoeLiteMTPModel" => Ok(Box::new(Glm4MoeMtpModel::new(cfg, vb)?)),
        "GlmOcrMTPModel" => Ok(Box::new(GlmOcrMtpModel::new(cfg, vb)?)),
        "LongCatFlashMTPModel" => Ok(Box::new(LongCatFlashMtpModel::new(cfg, vb)?)),
        "MiMoMTPModel" => Ok(Box::new(MiMoMtpModel::new(cfg, vb)?)),
        "OpenPanguMTPModel" => Ok(Box::new(OpenPanguMtpModel::new(cfg, vb)?)),
        "Qwen3NextMTP" => Ok(Box::new(Qwen3NextMtpModel::new(cfg, vb)?)),
        "Step3p5MTP" => Ok(Box::new(Step3p5MtpModel::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Create a KVCacheManager appropriate for the given model configuration.
///
/// For MLA models (DeepSeek V2/V3), this creates a compressed MLA cache.
/// For standard models (Llama, Qwen, etc.), this creates a standard paged KV cache.
///
/// # Arguments
/// * `cfg` - Model configuration (determines cache type based on architecture)
/// * `block_size` - Tokens per block
/// * `num_blocks` - Total number of cache blocks
/// * `dtype` - Data type for cache storage
/// * `device` - Target device
///
/// # Returns
/// A KVCacheManager configured for the model type
pub fn create_cache_manager(
    cfg: &ModelConfig,
    block_size: usize,
    num_blocks: usize,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<KVCacheManager, CacheError> {
    create_cache_manager_with_tp(cfg, block_size, num_blocks, dtype, device, 1)
}

/// Create a KVCacheManager with tensor parallelism support.
///
/// For tensor-parallel execution, each GPU only needs to cache its local KV heads.
/// This function divides the head count by tp_size to allocate the correct amount
/// of cache per GPU.
///
/// # Arguments
/// * `cfg` - Model configuration (determines cache type based on architecture)
/// * `block_size` - Tokens per block
/// * `num_blocks` - Total number of cache blocks
/// * `dtype` - Data type for cache storage
/// * `device` - Target device
/// * `tp_size` - Tensor parallel world size (1 for single GPU)
///
/// # Returns
/// A KVCacheManager configured for the model type and TP configuration
///
/// # Panics
/// Panics if num_kv_heads is not divisible by tp_size
pub fn create_cache_manager_with_tp(
    cfg: &ModelConfig,
    block_size: usize,
    num_blocks: usize,
    dtype: candle_core::DType,
    device: &Device,
    tp_size: usize,
) -> Result<KVCacheManager, CacheError> {
    if cfg.is_mla_model() {
        let mla_dims = cfg
            .mla_dims()
            .expect("MLA model must have mla_dims in config");

        let local_num_heads = cfg.num_attention_heads / tp_size;
        assert_eq!(
            cfg.num_attention_heads % tp_size,
            0,
            "num_attention_heads ({}) must be divisible by tp_size ({})",
            cfg.num_attention_heads,
            tp_size
        );

        let mla_config = MLACacheConfig {
            kv_lora_rank: mla_dims.kv_lora_rank,
            qk_rope_head_dim: mla_dims.qk_rope_head_dim,
            qk_nope_head_dim: mla_dims.qk_nope_head_dim,
            v_head_dim: mla_dims.v_head_dim,
            num_heads: local_num_heads,
            block_size,
            num_blocks,
            num_layers: cfg.num_hidden_layers,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
        };
        KVCacheManager::new_mla(&mla_config)
    } else {
        let local_num_kv_heads = cfg.num_key_value_heads / tp_size;
        assert_eq!(
            cfg.num_key_value_heads % tp_size,
            0,
            "num_key_value_heads ({}) must be divisible by tp_size ({})",
            cfg.num_key_value_heads,
            tp_size
        );

        let cache_config = CacheConfig {
            block_size,
            num_blocks,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_num_kv_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cache_config)
    }
}

// ─── Tensor Parallelism Support ──────────────────────────────────────────────

/// Construct a model with tensor parallelism support.
///
/// This function creates a model that distributes computation across multiple GPUs
/// using tensor parallelism. Each GPU holds a shard of the model weights and
/// performs partial computation, with collective operations combining results.
///
/// # Arguments
/// * `cfg` - Model configuration
/// * `vb` - VarBuilder for loading weights
/// * `pg` - Process group defining the TP topology
/// * `tp_ctx` - Tensor parallelism context with communicator
///
/// # Returns
/// A boxed model implementing ModelForward
///
/// # Supported Architectures
/// Currently supports: LlamaForCausalLM, MistralForCausalLM, MixtralForCausalLM,
/// Qwen2ForCausalLM, Qwen2MoeForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM,
/// GlmForCausalLM, Glm4ForCausalLM, Glm4MoeForCausalLM, GemmaForCausalLM, Gemma2ForCausalLM,
/// Gemma3ForCausalLM, Phi3ForCausalLM, Olmo2ForCausalLM, BaichuanForCausalLM,
/// InternLM2ForCausalLM, CohereForCausalLM, GPTNeoXForCausalLM, Starcoder2ForCausalLM,
/// BloomForCausalLM, FalconForCausalLM, PhiForCausalLM, YiForCausalLM, GPT2LMHeadModel,
/// ExaoneForCausalLM, PersimmonForCausalLM, MptForCausalLM, DbrxForCausalLM
/// Other architectures fall back to single-GPU (warning logged)
pub fn from_config_with_tp(
    cfg: &ModelConfig,
    vb: VarBuilder,
    pg: &dyn ProcessGroup,
    tp_ctx: TpContext,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;

    if pg.world_size() == 1 {
        return from_config(cfg, vb);
    }

    match arch {
        "LlamaForCausalLM"
        | "LlamaModel"
        | "LLaMAForCausalLM"
        | "AquilaModel"
        | "AquilaForCausalLM"
        | "CwmForCausalLM"
        | "InternLMForCausalLM"
        | "InternLM3ForCausalLM"
        | "IQuestCoderForCausalLM"
        | "XverseForCausalLM"
        | "SolarForCausalLM"
        | "Fairseq2LlamaForCausalLM"
        | "OrionForCausalLM"
        | "TeleChatForCausalLM"
        | "TeleChat2ForCausalLM"
        | "TeleFLMForCausalLM"
        | "DeciLMForCausalLM"
        | "OlmoForCausalLM" => Ok(Box::new(LlamaForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MistralForCausalLM" => Ok(Box::new(MistralForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MixtralForCausalLM" => Ok(Box::new(MixtralTpForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2ForCausalLM" | "Qwen2Model" => Ok(Box::new(Qwen2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2MoeForCausalLM" => Ok(Box::new(Qwen2MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen3MoeForCausalLM" => Ok(Box::new(Qwen3MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GlmForCausalLM" => Ok(Box::new(GlmForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Glm4ForCausalLM" => Ok(Box::new(Glm4ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Glm4MoeForCausalLM" | "Glm4MoeLiteForCausalLM" => Ok(Box::new(
            Glm4MoeForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "GemmaForCausalLM" => Ok(Box::new(GemmaForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Gemma2ForCausalLM" | "Gemma2Model" => Ok(Box::new(Gemma2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Phi3ForCausalLM" => Ok(Box::new(Phi3ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => Ok(Box::new(Olmo2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "BaichuanForCausalLM" | "BaiChuanForCausalLM" => Ok(Box::new(
            BaichuanForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "InternLM2ForCausalLM" => Ok(Box::new(InternLM2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "CohereForCausalLM" | "Cohere2ForCausalLM" => Ok(Box::new(CohereForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GPTNeoXForCausalLM" | "StableLMEpochForCausalLM" | "StableLmForCausalLM" => Ok(Box::new(
            GPTNeoXForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "SeedOssForCausalLM" => Ok(Box::new(SeedOssForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Starcoder2ForCausalLM" => Ok(Box::new(StarCoder2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "BloomForCausalLM" => Ok(Box::new(BloomForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "FalconForCausalLM" | "RWForCausalLM" => Ok(Box::new(FalconForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PhiForCausalLM" => Ok(Box::new(PhiForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "YiForCausalLM" => Ok(Box::new(YiForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "GPT2LMHeadModel" | "GPT2ForSequenceClassification" => {
            Ok(Box::new(GPT2LMHeadModel::new_with_tp(cfg, vb, pg, tp_ctx)?))
        }
        "Gemma3ForCausalLM" | "Gemma3TextModel" => Ok(Box::new(Gemma3ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "ExaoneForCausalLM" => Ok(Box::new(ExaoneForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PersimmonForCausalLM" => Ok(Box::new(PersimmonForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MptForCausalLM" | "MPTForCausalLM" => {
            Ok(Box::new(MptForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?))
        }
        "DbrxForCausalLM" => Ok(Box::new(DbrxForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "AfmoeForCausalLM" => Ok(Box::new(AfmoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "ArceeForCausalLM" => Ok(Box::new(arcee::new_arcee_with_tp(cfg, vb, pg, tp_ctx)?)),
        "ChatGLMModel" | "ChatGLMForCausalLM" | "ChatGLMForConditionalGeneration" => Ok(Box::new(
            ChatGLMForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "FlexOlmoForCausalLM" => Ok(Box::new(FlexOlmoForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Gemma3nForCausalLM" => Ok(Box::new(Gemma3nForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GPTBigCodeForCausalLM" => Ok(Box::new(GPTBigCodeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GPTJForCausalLM" => Ok(Box::new(GPTJForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "GptOssForCausalLM" => Ok(Box::new(GptOssForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GraniteForCausalLM" => Ok(Box::new(GraniteForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GraniteMoeForCausalLM" => Ok(Box::new(GraniteMoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "GraniteMoeSharedForCausalLM" => Ok(Box::new(GraniteMoeSharedForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Grok1ForCausalLM" | "Grok1ModelForCausalLM" => Ok(Box::new(
            Grok1ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "HunYuanDenseV1ForCausalLM" => Ok(Box::new(HunYuanDenseV1ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "HunYuanMoEV1ForCausalLM" => Ok(Box::new(HunYuanMoEV1ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "JAISLMHeadModel" => Ok(Box::new(JAISLMHeadModel::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Jais2ForCausalLM" => Ok(Box::new(Jais2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "KimiLinearForCausalLM" => Ok(Box::new(KimiLinearForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Lfm2ForCausalLM" => Ok(Box::new(Lfm2ForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Lfm2MoeForCausalLM" => Ok(Box::new(Lfm2MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Llama4ForCausalLM" => Ok(Box::new(Llama4ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "NemotronForCausalLM" => Ok(Box::new(NemotronForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "OlmoeForCausalLM" => Ok(Box::new(OlmoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "OuroForCausalLM" => Ok(Box::new(OuroForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "PanguEmbeddedForCausalLM" => Ok(Box::new(PanguEmbeddedForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PanguProMoEV2ForCausalLM" => Ok(Box::new(PanguProMoEV2ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PanguUltraMoEForCausalLM" => Ok(Box::new(PanguUltraMoEForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "PhiMoEForCausalLM" | "PhiMoeForCausalLM" => Ok(Box::new(PhiMoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "QWenLMHeadModel" => Ok(Box::new(QWenLMHeadModel::new_with_tp(cfg, vb, pg, tp_ctx)?)),
        "Qwen3NextForCausalLM" => Ok(Box::new(Qwen3NextForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Step1ForCausalLM" => Ok(Box::new(Step1ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Step3TextForCausalLM" => Ok(Box::new(Step3TextForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "MiMoForCausalLM" | "MiMoV2FlashForCausalLM" => Ok(Box::new(
            MiMoV2FlashForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "BailingMoeForCausalLM" | "BailingMoeV2ForCausalLM" => Ok(Box::new(
            BailingMoeForCausalLM::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "Ernie45MoeForCausalLM"
        | "Ernie4_5MoeForCausalLM"
        | "Ernie4_5ForCausalLM"
        | "Ernie4_5_MoeForCausalLM" => Ok(Box::new(Ernie45MoeForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "IQuestLoopCoderForCausalLM" => Ok(Box::new(IQuestLoopCoderForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "LongcatFlashForCausalLM" => Ok(Box::new(LongcatFlashForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Step3p5ForCausalLM" => Ok(Box::new(Step3p5ForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "InternLM2VEForCausalLM" => Ok(Box::new(InternLM2VEForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "InternLM2ForRewardModel" => Ok(Box::new(InternLM2ForRewardModel::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2ForRewardModel" => Ok(Box::new(Qwen2ForRewardModel::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Qwen2ForProcessRewardModel" => Ok(Box::new(Qwen2ForProcessRewardModel::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Molmo2ForConditionalGeneration" => Ok(Box::new(
            Molmo2ForConditionalGeneration::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        "Phi4MMForCausalLM" => Ok(Box::new(Phi4MMForCausalLM::new_with_tp(
            cfg, vb, pg, tp_ctx,
        )?)),
        "Eagle2_5_VLForConditionalGeneration" => Ok(Box::new(
            Eagle25VLForConditionalGeneration::new_with_tp(cfg, vb, pg, tp_ctx)?,
        )),
        other => {
            tracing::warn!(
                architecture = other,
                "TP not yet implemented for this architecture, using single-GPU fallback"
            );
            from_config(cfg, vb)
        }
    }
}

/// Construct a model using default single-GPU tensor parallelism context.
///
/// This is a convenience function that wraps `from_config_with_tp` with
/// single-GPU defaults. Use this when you want TP-aware models but are
/// running on a single GPU.
pub fn from_config_tp_aware(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let pg = LocalProcessGroup::new();
    let tp_ctx = TpContext::single_gpu();
    from_config_with_tp(cfg, vb, &pg, tp_ctx)
}

/// Construct a LoRA-enabled model from config.architectures[0].
///
/// The returned model supports per-request adapter selection through LoraContext.
/// Call `register_lora()` on the model to add adapters before inference.
/// The model implements `ModelForward` and can be used with the engine.
pub fn from_config_with_lora(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<LoraEnabledModel, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "Qwen3ForCausalLM" => Ok(LoraEnabledModel::Qwen3(Qwen3WithLora::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(LoraEnabledModel::Llama(LlamaWithLora::new(cfg, vb)?)),
        "MistralForCausalLM" => Ok(LoraEnabledModel::Mistral(MistralWithLora::new(cfg, vb)?)),
        "Qwen2ForCausalLM" => Ok(LoraEnabledModel::Qwen2(Qwen2WithLora::new(cfg, vb)?)),
        "GemmaForCausalLM" => Ok(LoraEnabledModel::Gemma(GemmaWithLora::new(cfg, vb)?)),
        "Gemma2ForCausalLM" => Ok(LoraEnabledModel::Gemma2(Gemma2WithLora::new(cfg, vb)?)),
        "Gemma3ForCausalLM" | "Gemma3TextModel" => {
            Ok(LoraEnabledModel::Gemma3(Gemma3WithLora::new(cfg, vb)?))
        }
        "Phi3ForCausalLM" => Ok(LoraEnabledModel::Phi3(Phi3WithLora::new(cfg, vb)?)),
        "MixtralForCausalLM" => Ok(LoraEnabledModel::Mixtral(MixtralWithLora::new(cfg, vb)?)),
        "Olmo2ForCausalLM" | "Olmo3ForCausalLM" => {
            Ok(LoraEnabledModel::Olmo2(Olmo2WithLora::new(cfg, vb)?))
        }
        "DeepseekV2ForCausalLM"
        | "DeepseekV3ForCausalLM"
        | "DeepseekV32ForCausalLM"
        | "DeepseekForCausalLM"
        | "GlmMoeDsaForCausalLM"
        | "MistralLarge3ForCausalLM" => {
            Ok(LoraEnabledModel::DeepSeek(DeepSeekWithLora::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

delegate_model_forward! {
    /// A LoRA-enabled model that can be used with the engine.
    ///
    /// This enum wraps concrete LoRA-enabled model types and provides both
    /// `ModelForward` for engine compatibility and LoRA-specific methods
    /// for adapter registration.
    pub enum LoraEnabledModel {
        Llama(LlamaWithLora),
        Qwen3(Qwen3WithLora),
        Mistral(MistralWithLora),
        Mixtral(MixtralWithLora),
        Qwen2(Qwen2WithLora),
        Gemma(GemmaWithLora),
        Gemma2(Gemma2WithLora),
        Gemma3(Gemma3WithLora),
        Phi3(Phi3WithLora),
        Olmo2(Olmo2WithLora),
        DeepSeek(DeepSeekWithLora),
    }
}

impl LoraEnabledModel {
    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        match self {
            LoraEnabledModel::Llama(m) => m.register_lora(lora_model),
            LoraEnabledModel::Qwen3(m) => m.register_lora(lora_model),
            LoraEnabledModel::Mistral(m) => m.register_lora(lora_model),
            LoraEnabledModel::Mixtral(m) => m.register_lora(lora_model),
            LoraEnabledModel::Qwen2(m) => m.register_lora(lora_model),
            LoraEnabledModel::Gemma(m) => m.register_lora(lora_model),
            LoraEnabledModel::Gemma2(m) => m.register_lora(lora_model),
            LoraEnabledModel::Gemma3(m) => m.register_lora(lora_model),
            LoraEnabledModel::Phi3(m) => m.register_lora(lora_model),
            LoraEnabledModel::Olmo2(m) => m.register_lora(lora_model),
            LoraEnabledModel::DeepSeek(m) => m.register_lora(lora_model),
        }
    }

    /// Get the list of registered LoRA adapter names.
    pub fn lora_adapters(&self) -> Vec<String> {
        match self {
            LoraEnabledModel::Llama(m) => m.lora_adapters(),
            LoraEnabledModel::Qwen3(m) => m.lora_adapters(),
            LoraEnabledModel::Mistral(m) => m.lora_adapters(),
            LoraEnabledModel::Mixtral(m) => m.lora_adapters(),
            LoraEnabledModel::Qwen2(m) => m.lora_adapters(),
            LoraEnabledModel::Gemma(m) => m.lora_adapters(),
            LoraEnabledModel::Gemma2(m) => m.lora_adapters(),
            LoraEnabledModel::Gemma3(m) => m.lora_adapters(),
            LoraEnabledModel::Phi3(m) => m.lora_adapters(),
            LoraEnabledModel::Olmo2(m) => m.lora_adapters(),
            LoraEnabledModel::DeepSeek(m) => m.lora_adapters(),
        }
    }
}

/// Trait for models that support LoRA adapters.
///
/// This extends the base ModelForward trait with LoRA-specific methods.
pub trait ModelForwardWithLora: Send + Sync {
    /// Register a LoRA adapter with the model.
    fn register_lora(&mut self, lora_model: &crate::lora::LoraModel);

    /// Forward pass with optional LoRA adapter.
    fn forward_with_lora(
        &self,
        input_ids: &candle_core::Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
        block_table: &crate::kv_cache::BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &crate::lora::LoraContext,
    ) -> candle_core::Result<candle_core::Tensor>;

    /// Batched decode forward with optional LoRA adapter.
    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &candle_core::Tensor,
        sequences: &[crate::engine::DecodeSequenceMetadata],
        kv_cache_mgr: &mut crate::kv_cache::KVCacheManager,
        lora_ctx: &crate::lora::LoraContext,
    ) -> candle_core::Result<candle_core::Tensor>;

    /// Get the list of registered LoRA adapter names.
    fn lora_adapters(&self) -> Vec<String>;

    /// Get the device this model is on.
    fn device(&self) -> &candle_core::Device;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config(arch: &str) -> ModelConfig {
        ModelConfig {
            architectures: vec![arch.to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 128,
            head_dim: 32,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
            extra: {
                let mut m = serde_json::Map::new();
                m.insert("layer_norm_eps".to_string(), serde_json::Value::from(1e-12));
                m.insert("type_vocab_size".to_string(), serde_json::Value::from(2));
                m
            },
        }
    }

    #[test]
    fn test_from_config_reward_models() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        for arch in [
            "InternLM2ForRewardModel",
            "Qwen2ForRewardModel",
            "Qwen2ForProcessRewardModel",
        ] {
            let cfg = minimal_config(arch);
            let result = from_config(&cfg, vb.clone());
            assert!(
                result.is_ok(),
                "from_config should handle {arch}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_from_config_embedding_aliases() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        for arch in [
            "BertForTokenClassification",
            "RobertaModel",
            "RobertaForMaskedLM",
            "ModernBertForTokenClassification",
            "XLMRobertaForSequenceClassification",
        ] {
            let cfg = minimal_config(arch);
            let result = from_config(&cfg, vb.clone());
            assert!(
                result.is_ok(),
                "from_config should handle {arch}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_from_config_causal_aliases() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        // Qwen2Model should dispatch to Qwen2ForCausalLM
        let cfg = minimal_config("Qwen2Model");
        let result = from_config(&cfg, vb.clone());
        assert!(
            result.is_ok(),
            "Qwen2Model should dispatch: {:?}",
            result.err()
        );

        // GPT2ForSequenceClassification should dispatch to GPT2LMHeadModel
        let cfg = minimal_config("GPT2ForSequenceClassification");
        let result = from_config(&cfg, vb.clone());
        assert!(
            result.is_ok(),
            "GPT2ForSequenceClassification should dispatch: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_splade_and_bidirectional() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        for arch in [
            "BertSpladeSparseEmbeddingModel",
            "LlamaBidirectionalModel",
            "LlamaBidirectionalForSequenceClassification",
        ] {
            let cfg = minimal_config(arch);
            let result = from_config(&cfg, vb.clone());
            assert!(
                result.is_ok(),
                "from_config should handle {arch}: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_from_config_seed_oss() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let cfg = minimal_config("SeedOssForCausalLM");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle SeedOssForCausalLM: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_nvlm_d() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = minimal_config("NVLM_D_Model");
        cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "norm_type": "rms_norm"
            }),
        );
        cfg.extra
            .insert("downsample_ratio".to_string(), serde_json::json!(0.5));
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle NVLM_D_Model: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_aya_vision() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = minimal_config("AyaVisionForConditionalGeneration");
        cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "model_type": "siglip_vision_model"
            }),
        );
        cfg.extra
            .insert("downsample_factor".to_string(), serde_json::json!(2));
        cfg.extra.insert(
            "alignment_intermediate_size".to_string(),
            serde_json::json!(128),
        );
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle AyaVisionForConditionalGeneration: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_fuyu() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = minimal_config("FuyuForCausalLM");
        cfg.extra
            .insert("patch_size".to_string(), serde_json::json!(4));
        cfg.extra
            .insert("num_channels".to_string(), serde_json::json!(3));
        cfg.extra
            .insert("layer_norm_eps".to_string(), serde_json::json!(1e-5));
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle FuyuForCausalLM: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_bagel() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = minimal_config("BagelForConditionalGeneration");
        cfg.extra.insert(
            "vit_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "image_size": 28,
                "patch_size": 14,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
            }),
        );
        cfg.extra.insert(
            "vit_max_num_patch_per_side".to_string(),
            serde_json::json!(4),
        );
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle BagelForConditionalGeneration: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_opencua_alias() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("OpenCUAForConditionalGeneration");
        cfg.extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({ "mrope_section": [2, 6, 8] }),
        );
        cfg.extra
            .insert("image_token_id".to_string(), serde_json::json!(151655));
        cfg.extra
            .insert("video_token_id".to_string(), serde_json::json!(151656));
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle OpenCUAForConditionalGeneration: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_llava_next_video_alias() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("LlavaNextVideoForConditionalGeneration");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle LlavaNextVideoForConditionalGeneration: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_unsupported_arch_error() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let cfg = minimal_config("NonExistentArchitecture");
        let result = from_config(&cfg, vb);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            matches!(err, ModelError::UnsupportedArchitecture(_)),
            "expected UnsupportedArchitecture, got {err}"
        );
    }
}
