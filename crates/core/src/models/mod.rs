#[macro_use]
mod macros;

// ─── Architecture refactor (Phase 1) ────────────────────────────────
//
// `factory` and `traits` define the new dispatch surface. They land
// without any model migrating onto them — the legacy match-arm
// dispatch below is still the live path. `registry_v2` is the
// phf-backed lookup table that Phase 2 will populate; behind a
// feature flag for now.
#[cfg(feature = "model-registry-v2")]
pub mod factories;
pub mod factory;
#[cfg(feature = "model-registry-v2")]
pub mod registry_v2;
pub mod traits;

pub mod afmoe;
pub mod apertus;
pub mod apertus_quantized;
pub mod arcee;
pub mod arctic;
pub mod aria;
pub mod audioflamingo3;
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
pub mod clip;
pub mod cohere;
pub mod cohere_quantized;
pub mod colbert;
pub mod dbrx;
pub mod deepseek;
pub mod deepseek_lora;
pub mod deepseek_mtp;
pub mod deepseek_ocr;
pub mod deepseek_ocr2;
pub mod deepseek_quantized;
pub mod deepseek_vl2;
pub mod dots1;
pub mod dots_ocr;
pub mod e5_mistral;
pub mod eagle2_5_vl;
pub mod eagle3;
pub mod eagle3_mistral_large3;
pub mod eagle_deepseek;
pub mod eagle_llama;
pub mod eagle_llama4;
pub mod eagle_minicpm;
pub mod eagle_mistral_large3;
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
pub mod funaudiochat;
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
pub mod gemma4;
pub mod gemma4_quantized;
pub mod gemma4_vision;
pub mod gemma4_vlm;
pub mod gemma4_vlm_quantized;
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
pub mod glmasr;
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
pub mod granite_speech;
pub mod granitemoe;
pub mod granitemoe_hybrid;
pub mod granitemoe_shared;
pub mod gritlm;
pub mod grok1;
pub mod gte;
pub mod hunyuan;
pub mod hunyuan_quantized;
pub mod hunyuan_vision;
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
pub mod isaac;
pub mod jais;
pub mod jais2;
pub mod jais2_quantized;
pub mod jais_quantized;
pub mod jamba;
pub mod jina_vl;
pub mod kanana_v;
pub mod keye_vl;
pub mod kimi_k25;
pub mod kimi_linear;
pub mod kimi_vl;
pub mod lfm2;
pub mod lfm2_vl;
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
pub mod medusa;
pub mod midashenglm;
pub mod mimo_mtp;
pub mod mimo_v2_flash;
pub mod minicpm;
pub mod minicpm3;
pub mod minicpm_quantized;
pub mod minicpmo;
pub mod minicpmv;
pub mod minimax_m2;
pub mod minimax_text01;
pub mod minimax_vl_01;
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
pub mod musicflamingo;
pub mod nano_nemotron_vl;
pub mod nemotron;
pub mod nemotron_h;
pub mod nemotron_nas;
pub mod nemotron_parse;
pub mod nemotron_quantized;
pub mod nemotron_vl;
pub mod nvlm_d;
pub mod olmo2;
pub mod olmo2_lora;
pub mod olmo2_quantized;
pub mod olmoe;
pub mod openpangu_mtp;
pub mod openpangu_vl;
pub mod opt;
pub mod opt_quantized;
pub mod ouro;
pub mod ovis;
pub mod ovis2_5;
pub mod paddleocr_vl;
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
pub mod phi4mm_audio;
pub mod phi_quantized;
pub mod phimoe;
pub mod pixtral;
pub mod plamo2;
pub mod plamo3;
pub mod plamo3_quantized;
pub mod qwen;
pub mod qwen2;
pub mod qwen2_5_omni_thinker;
pub mod qwen2_5_vl;
pub mod qwen2_audio;
pub mod qwen2_lora;
pub mod qwen2_moe;
pub mod qwen2_moe_quantized;
pub mod qwen2_quantized;
pub mod qwen2_reward;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_asr;
pub mod qwen3_lora;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_next_mtp;
pub mod qwen3_omni_moe_thinker;
pub mod qwen3_quantized;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;
pub mod qwen_quantized;
pub mod qwen_vl;
pub mod radio;
pub mod registry;
pub mod seed_oss;
pub mod siglip;
pub mod starcoder2;
pub mod starcoder2_quantized;
pub mod step1;
pub mod step1_quantized;
pub mod step3_text;
pub mod step3_vl;
pub mod step3p5;
pub mod step3p5_mtp;
pub mod t5;
pub mod tarsier;
pub mod tp_layers;
pub mod ultravox;
pub mod voxtral;
pub mod voyage;
pub mod whisper;
pub mod yi;
pub mod zamba2;

// Re-export tensor parallelism abstractions
pub use tp_layers::{TpContext, TpEmbedding, TpGeGluMlp, TpGeluMlp, TpLinear, TpSwiGluMlp};

pub use afmoe::AfmoeForCausalLM;
pub use apertus::ApertusForCausalLM;
pub use apertus_quantized::QuantizedApertusForCausalLM;
pub use arctic::ArcticForCausalLM;
pub use aria::AriaForConditionalGeneration;
pub use audioflamingo3::AudioFlamingo3ForConditionalGeneration;
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
pub use clip::CLIPEmbeddingModel;
pub use cohere::CohereForCausalLM;
pub use cohere_quantized::QuantizedCohereForCausalLM;
pub use colbert::ColBERTForRetrieval;
pub use dbrx::DbrxForCausalLM;
pub use deepseek::{DeepSeekForCausalLM, GlmMoeDsaForCausalLM};
pub use deepseek_lora::DeepSeekWithLora;
pub use deepseek_mtp::DeepSeekMtpModel;
pub use deepseek_ocr::DeepseekOCRForCausalLM;
pub use deepseek_ocr2::DeepseekOCR2ForCausalLM;
pub use deepseek_quantized::QuantizedDeepSeekForCausalLM;
pub use deepseek_vl2::DeepSeekVLV2ForConditionalGeneration;
pub use dots1::Dots1ForCausalLM;
pub use dots_ocr::DotsOCRForCausalLM;
pub use e5_mistral::E5MistralForEmbedding;
pub use eagle2_5_vl::Eagle25VLForConditionalGeneration;
pub use eagle3::{Eagle3DraftModel, Eagle3LlamaForCausalLM};
pub use eagle3_mistral_large3::Eagle3MistralLarge3ForCausalLM;
pub use eagle_deepseek::EagleDeepSeekForCausalLM;
pub use eagle_llama::{Eagle1DraftModel, EagleLlamaForCausalLM};
pub use eagle_llama4::EagleLlama4ForCausalLM;
pub use eagle_minicpm::EagleMiniCPMForCausalLM;
pub use eagle_mistral_large3::EagleMistralLarge3ForCausalLM;
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
pub use funaudiochat::FunAudioChatForConditionalGeneration;
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
pub use gemma4::Gemma4ForCausalLM;
pub use gemma4_quantized::QuantizedGemma4ForCausalLM;
pub use gemma4_vlm::Gemma4ForConditionalGeneration;
pub use gemma4_vlm_quantized::QuantizedGemma4ForConditionalGeneration;
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
pub use glmasr::GlmAsrForConditionalGeneration;
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
pub use granite_speech::GraniteSpeechForConditionalGeneration;
pub use granitemoe::GraniteMoeForCausalLM;
pub use granitemoe_hybrid::GraniteMoeHybridForCausalLM;
pub use granitemoe_shared::GraniteMoeSharedForCausalLM;
pub use gritlm::GritLM;
pub use grok1::Grok1ForCausalLM;
pub use gte::{GteNewForEmbedding, GteNewForSequenceClassification};
pub use hunyuan::{HunYuanDenseV1ForCausalLM, HunYuanMoEV1ForCausalLM};
pub use hunyuan_quantized::QuantizedHunYuanDenseForCausalLM;
pub use hunyuan_vision::HunYuanVLForConditionalGeneration;
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
pub use isaac::IsaacForConditionalGeneration;
pub use jais::JAISLMHeadModel;
pub use jais2::Jais2ForCausalLM;
pub use jais2_quantized::QuantizedJais2ForCausalLM;
pub use jais_quantized::QuantizedJAISLMHeadModel;
pub use jamba::JambaForCausalLM;
pub use jina_vl::JinaVLForRanking;
pub use kanana_v::KananaVForConditionalGeneration;
pub use keye_vl::{KeyeForConditionalGeneration, KeyeVL1_5ForConditionalGeneration};
pub use kimi_k25::KimiK25ForConditionalGeneration;
pub use kimi_linear::KimiLinearForCausalLM;
pub use kimi_vl::KimiVLForConditionalGeneration;
pub use lfm2::{Lfm2ForCausalLM, Lfm2MoeForCausalLM};
pub use lfm2_vl::Lfm2VLForConditionalGeneration;
pub use llama::{LlamaForCausalLM, TeleFLMForCausalLM};
pub use llama4::Llama4ForCausalLM;
pub use llama4_vl::Llama4VLForConditionalGeneration;
pub use llama_bidirectional::LlamaBidirectionalModel;
pub use llama_lora::LlamaWithLora;
pub use llama_quantized::QuantizedLlamaForCausalLM;
pub use llava::LLaVAForConditionalGeneration;
pub use llava_onevision::{
    BeeForConditionalGeneration, LlavaOnevisionForConditionalGeneration, RForConditionalGeneration,
};
pub use longcat_flash::LongcatFlashForCausalLM;
pub use longcat_flash_mtp::LongCatFlashMtpModel;
pub use mamba::MambaForCausalLM;
pub use mamba2::Mamba2ForCausalLM;
pub use medusa::{MedusaDraftModel, MedusaModel};
pub use midashenglm::MiDashengLMModel;
pub use mimo_mtp::MiMoMtpModel;
pub use mimo_v2_flash::MiMoV2FlashForCausalLM;
pub use minicpm::MiniCPMForCausalLM;
pub use minicpm3::MiniCPM3ForCausalLM;
pub use minicpm_quantized::QuantizedMiniCPMForCausalLM;
pub use minicpmo::MiniCPMOForCausalLM;
pub use minicpmv::MiniCPMVForConditionalGeneration;
pub use minimax_m2::MiniMaxM2ForCausalLM;
pub use minimax_text01::MiniMaxText01ForCausalLM;
pub use minimax_vl_01::MiniMaxVL01ForConditionalGeneration;
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
pub use musicflamingo::MusicFlamingoForConditionalGeneration;
pub use nemotron::NemotronForCausalLM;
pub use nemotron_h::NemotronHForCausalLM;
pub use nemotron_nas::NemotronNasForCausalLM;
pub use nemotron_quantized::QuantizedNemotronForCausalLM;
pub use nemotron_vl::LlamaNemotronVLForConditionalGeneration;
pub use nvlm_d::NVLMDModel;
pub use olmo2::Olmo2ForCausalLM;
pub use olmo2_lora::Olmo2WithLora;
pub use olmo2_quantized::QuantizedOlmo2ForCausalLM;
pub use olmoe::OlmoeForCausalLM;
pub use openpangu_mtp::OpenPanguMtpModel;
pub use openpangu_vl::OpenPanguVLForConditionalGeneration;
pub use opt::OPTForCausalLM;
pub use opt_quantized::QuantizedOPTForCausalLM;
pub use ouro::OuroForCausalLM;
pub use ovis::OvisForConditionalGeneration;
pub use ovis2_5::Ovis2_5ForConditionalGeneration;
pub use paddleocr_vl::PaddleOCRVLForConditionalGeneration;
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
pub use qwen2_5_omni_thinker::Qwen2_5OmniThinkerForConditionalGeneration;
pub use qwen2_5_vl::Qwen25VLForConditionalGeneration;
pub use qwen2_audio::Qwen2AudioForConditionalGeneration;
pub use qwen2_lora::Qwen2WithLora;
pub use qwen2_moe::Qwen2MoeForCausalLM;
pub use qwen2_moe_quantized::QuantizedQwen2MoeForCausalLM;
pub use qwen2_quantized::QuantizedQwen2ForCausalLM;
pub use qwen2_reward::{Qwen2ForProcessRewardModel, Qwen2ForRewardModel};
pub use qwen2_vl::Qwen2VLForConditionalGeneration;
pub use qwen3::Qwen3ForCausalLM;
pub use qwen3_asr::Qwen3ASRForConditionalGeneration;
pub use qwen3_lora::Qwen3WithLora;
pub use qwen3_moe::Qwen3MoeForCausalLM;
pub use qwen3_next::Qwen3NextForCausalLM;
pub use qwen3_next_mtp::Qwen3NextMtpModel;
pub use qwen3_omni_moe_thinker::Qwen3OmniMoeThinkerForConditionalGeneration;
pub use qwen3_quantized::QuantizedQwen3ForCausalLM;
pub use qwen3_vl::Qwen3VLForConditionalGeneration;
pub use qwen3_vl_moe::Qwen3VLMoeForConditionalGeneration;
pub use qwen_quantized::QuantizedQWenLMHeadModel;
pub use qwen_vl::QwenVLForConditionalGeneration;
pub use registry::{
    find_architecture, supported_architectures, ArchitectureInfo, ModelCapabilities,
};
pub use seed_oss::SeedOssForCausalLM;
pub use siglip::SiglipEmbeddingModel;
pub use starcoder2::StarCoder2ForCausalLM;
pub use starcoder2_quantized::QuantizedStarCoder2ForCausalLM;
pub use step1::Step1ForCausalLM;
pub use step1_quantized::QuantizedStep1ForCausalLM;
pub use step3_text::Step3TextForCausalLM;
pub use step3_vl::Step3VLForConditionalGeneration;
pub use step3p5::Step3p5ForCausalLM;
pub use step3p5_mtp::Step3p5MtpModel;
pub use t5::T5ForConditionalGeneration;
pub use tarsier::TarsierForConditionalGeneration;
pub use ultravox::UltravoxModel;
pub use voxtral::VoxtralForConditionalGeneration;
pub use voyage::VoyageForEmbedding;
pub use whisper::WhisperForConditionalGeneration;
pub use yi::YiForCausalLM;
pub use zamba2::Zamba2ForCausalLM;

use std::path::Path;

use candle_core::Device;
use candle_nn::VarBuilder;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, PipelineStageConfig, ProcessGroup};
use crate::engine::{ModelForward, PipelineForward};
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
///
/// Routes through the [`registry_v2`] factory map (default-on per
/// ADR-0013). Architectures not in the registry fall through to the
/// deprecation table below for actionable error messages.
pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = get_arch(cfg)?;

    #[cfg(feature = "model-registry-v2")]
    {
        if let Some(factory) = registry_v2::lookup(arch) {
            return factory.build(cfg, vb);
        }
    }

    // Deprecated / removed architectures: actionable guidance instead
    // of a generic "unsupported" error. Listed by HuggingFace
    // arch_name + the vLLM version that removed support.
    match arch {
        // Architectures removed from vLLM at the given version; provide actionable guidance.
        "MotifForCausalLM" => Err(ModelError::UnsupportedArchitecture(
            "MotifForCausalLM was removed in vLLM v0.10.2 and is no longer supported".into(),
        )),
        "Phi3SmallForCausalLM" => Err(ModelError::UnsupportedArchitecture(
            "Phi3SmallForCausalLM was removed in vLLM v0.9.2 and is no longer supported".into(),
        )),
        "Phi4FlashForCausalLM" => Err(ModelError::UnsupportedArchitecture(
            "Phi4FlashForCausalLM was removed in vLLM v0.10.2; use Phi4MMForCausalLM instead"
                .into(),
        )),
        "Phi4MultimodalForCausalLM" => Err(ModelError::UnsupportedArchitecture(
            "Phi4MultimodalForCausalLM was removed in vLLM v0.12.0; use Phi4MMForCausalLM instead"
                .into(),
        )),
        "BartModel" | "BartForConditionalGeneration" | "MBartForConditionalGeneration"
        | "DonutForConditionalGeneration" | "Florence2ForConditionalGeneration" => {
            Err(ModelError::UnsupportedArchitecture(format!(
                "{arch} (encoder-decoder) was removed in vLLM v0.10.2 due to V0 engine deprecation"
            )))
        }
        "MllamaForConditionalGeneration" => Err(ModelError::UnsupportedArchitecture(
            "MllamaForConditionalGeneration was removed in vLLM v0.10.2; use Llama4ForConditionalGeneration instead".into(),
        )),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Construct a pipeline-parallel model stage from config.architectures[0].
///
/// Only architectures in the Llama family are supported for pipeline parallelism.
/// The returned model implements [`PipelineForward`] and covers only the layers
/// assigned by `stage`: global layers `[stage.first_layer, stage.first_layer +
/// stage.num_layers)` are loaded from `vb`; other layers are not loaded.
///
/// # Errors
/// Returns [`ModelError::UnsupportedArchitecture`] for architectures that have
/// not yet been ported to the `PipelineForward` interface.
pub fn from_config_with_pp(
    cfg: &ModelConfig,
    vb: candle_nn::VarBuilder,
    stage: &PipelineStageConfig,
) -> Result<Box<dyn PipelineForward>, ModelError> {
    let arch = get_arch(cfg)?;

    #[cfg(feature = "model-registry-v2")]
    {
        use factory::Capabilities;
        if let Some(factory) = registry_v2::lookup(arch) {
            if factory.info().supports(Capabilities::PP) {
                return factory.build_with_pp(cfg, vb, stage);
            }
        }
    }

    Err(ModelError::UnsupportedArchitecture(format!(
        "{arch} does not support pipeline parallelism yet; use --pipeline-parallel-size 1"
    )))
}

/// Construct an encoder-decoder model from config.architectures[0].
pub fn from_config_encoder_decoder(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn crate::engine::ModelForEncoderDecoder>, ModelError> {
    let arch = get_arch(cfg)?;

    #[cfg(feature = "model-registry-v2")]
    {
        use factory::Capabilities;
        if let Some(factory) = registry_v2::lookup(arch) {
            if factory.info().supports(Capabilities::ENCODER_DECODER) {
                return factory.build_encoder_decoder(cfg, vb);
            }
        }
    }

    Err(ModelError::UnsupportedArchitecture(arch.into()))
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

    // Registry path: every quantized arch in the registry advertises
    // `Capabilities::QUANTIZED` and overrides `build_quant`. The
    // legacy match arm fallback was removed in Phase 9.4.
    #[cfg(feature = "model-registry-v2")]
    {
        use factory::Capabilities;
        if let Some(factory) = registry_v2::lookup(arch) {
            if factory.info().supports(Capabilities::QUANTIZED) {
                return factory.build_quant(cfg, vb, weight_loader.as_ref());
            }
        }
    }

    Err(ModelError::UnsupportedArchitecture(arch.into()))
}

/// Get the detected quantization method for a model directory.
///
/// This is useful for checking quantization before loading.
pub fn detect_quantization(model_dir: &Path) -> DetectedQuantConfig {
    detect_from_directory(model_dir)
}

// ─── Speculative-draft routing ──────────────────────────────────────────────
//
// All five speculative protocols (MTP / Eagle-1 / Eagle-3 / Medusa /
// MLP Speculator) use the same `(ModelConfig, VarBuilder) -> Result<...>`
// shape but return different trait types because their decode-time
// protocols are genuinely different (MTP fixed target_hs vs Eagle chain
// vs Medusa head-bank vs MLPSpec separate prefix). The 5 dispatch
// functions are kept for API-level type safety, and `DraftKind` +
// `speculative_kind_for` provide one-shot introspection so callers
// (e.g. the server's draft-model loader) can route to the correct
// constructor without re-implementing the arch_name → protocol map.

/// Speculative-decoding draft protocol kind for a given architecture.
///
/// Returned by [`speculative_kind_for`] so callers can dispatch to the
/// correct `*_from_config` constructor without hard-coded arch_name
/// lists. `MlpSpeculator` is its own kind because its model type is
/// concrete (not boxed-trait).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftKind {
    /// Multi-Token Prediction — fixed target hidden state for all K steps.
    Mtp,
    /// Eagle-1 — chained hidden-state propagation with shared head.
    Eagle1,
    /// Eagle-3 — Llama backbone variant of Eagle.
    Eagle3,
    /// Medusa — head-bank predicting K tokens in parallel from one hidden state.
    Medusa,
    /// MLP Speculator — separate small MLP model with `speculator.*` weights.
    MlpSpeculator,
}

/// Look up the speculative protocol kind for a HuggingFace
/// architecture name. Returns `None` when the architecture is not a
/// speculative-decoding draft.
///
/// Mirrors the dispatch tables in `mtp_from_config` /
/// `eagle1_from_config` / `eagle3_from_config` / `medusa_from_config` /
/// `mlp_speculator_from_config` — the single source of truth for
/// "which protocol does this draft model use".
pub fn speculative_kind_for(arch: &str) -> Option<DraftKind> {
    match arch {
        // MTP
        "DeepSeekMTPModel"
        | "ErnieMTPModel"
        | "ExaoneMoeMTP"
        | "Glm4MoeMTPModel"
        | "Glm4MoeLiteMTPModel"
        | "GlmOcrMTPModel"
        | "LongCatFlashMTPModel"
        | "MiMoMTPModel"
        | "OpenPanguMTPModel"
        | "Qwen3NextMTP"
        | "Step3p5MTP" => Some(DraftKind::Mtp),
        // Eagle-1
        "EagleLlamaForCausalLM"
        | "EagleLlama4ForCausalLM"
        | "EagleMiniCPMForCausalLM"
        | "EagleDeepSeekMTPModel"
        | "EagleDeepseekV3ForCausalLM"
        | "EagleMistralLarge3ForCausalLM" => Some(DraftKind::Eagle1),
        // Eagle-3
        "Eagle3LlamaForCausalLM"
        | "LlamaForCausalLMEagle3"
        | "Eagle3Qwen2_5vlForCausalLM"
        | "Eagle3Qwen3vlForCausalLM"
        | "Eagle3MistralLarge3ForCausalLM" => Some(DraftKind::Eagle3),
        // Medusa
        "MedusaModel" => Some(DraftKind::Medusa),
        // MLP Speculator
        "MLPSpeculatorPreTrainedModel" => Some(DraftKind::MlpSpeculator),
        _ => None,
    }
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
        // NOTE: EagleDeepSeekMTPModel in the Python registry maps to
        // EagleDeepseekV3ForCausalLM, which uses the Eagle-1 protocol —
        // NOT the MTP protocol. It is handled in eagle1_from_config().
        "DeepSeekMTPModel" => Ok(Box::new(DeepSeekMtpModel::new(cfg, vb)?)),
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

/// Create an Eagle-1 draft model from a model configuration.
///
/// Dispatches to the correct Eagle-1 variant based on the architecture name:
/// - `EagleLlamaForCausalLM` → Llama-based Eagle-1
/// - `EagleLlama4ForCausalLM` → Llama-4-based Eagle-1
/// - `EagleMiniCPMForCausalLM` → MiniCPM-based Eagle-1
/// - `EagleDeepSeekMTPModel` → DeepSeek V2/V3 MLA Eagle-1 (Eagle-1 protocol
///   despite the name; see Python registry entry `deepseek_eagle.py`)
/// - `EagleMistralLarge3ForCausalLM` → MistralLarge3 target + DeepSeek V2 MLA
///   draft (simpler fc-fusion: no enorm/hnorm vs the DeepSeek variant)
///
/// Used by the speculative decoding engine to load the correct Eagle-1 model.
pub fn eagle1_from_config(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn Eagle1DraftModel>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "EagleLlamaForCausalLM" => Ok(Box::new(EagleLlamaForCausalLM::new(cfg, vb)?)),
        "EagleLlama4ForCausalLM" => Ok(Box::new(EagleLlama4ForCausalLM::new(cfg, vb)?)),
        "EagleMiniCPMForCausalLM" => Ok(Box::new(EagleMiniCPMForCausalLM::new(cfg, vb)?)),
        // EagleDeepseekV3ForCausalLM is the current Python class name (deepseek_eagle.py);
        // EagleDeepSeekMTPModel is the legacy checkpoint name for the same Eagle-1 DeepSeek variant.
        "EagleDeepSeekMTPModel" | "EagleDeepseekV3ForCausalLM" => {
            Ok(Box::new(EagleDeepSeekForCausalLM::new(cfg, vb)?))
        }
        "EagleMistralLarge3ForCausalLM" => {
            Ok(Box::new(EagleMistralLarge3ForCausalLM::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Create an Eagle-3 draft model from a model configuration.
///
/// Dispatches to the correct Eagle-3 variant based on the architecture name.
/// All current Python Eagle-3 variants (`Eagle3LlamaForCausalLM`,
/// `LlamaForCausalLMEagle3`, `Eagle3Qwen2_5vlForCausalLM`,
/// `Eagle3Qwen3vlForCausalLM`) use the same underlying Llama-based
/// `Eagle3LlamaForCausalLM` implementation.
///
/// Eagle-3 variants for non-Llama target models use the
/// `Eagle3MistralLarge3ForCausalLM` path (DeepSeek V2 layers + fc projection).
pub fn eagle3_from_config(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn Eagle3DraftModel>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        // All Llama-family Eagle-3 variants share the same implementation.
        "Eagle3LlamaForCausalLM"
        | "LlamaForCausalLMEagle3"
        | "Eagle3Qwen2_5vlForCausalLM"
        | "Eagle3Qwen3vlForCausalLM" => Ok(Box::new(Eagle3LlamaForCausalLM::new(cfg, vb)?)),
        "Eagle3MistralLarge3ForCausalLM" => {
            Ok(Box::new(Eagle3MistralLarge3ForCausalLM::new(cfg, vb)?))
        }
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Create a Medusa draft model from a model configuration.
///
/// Dispatches to [`MedusaModel`] for the `MedusaModel` architecture.
/// Returns a trait object implementing [`MedusaDraftModel`] which can be used
/// by the speculative decoding engine to generate draft tokens from target
/// model hidden states.
///
/// Config fields read from `ModelConfig::extra`:
/// - `num_heads` — number of Medusa prediction heads (default: 4)
/// - `medusa_fc_bias` — add bias to residual layers (default: false)
/// - `truncated_vocab_size` — reduced vocab when a token_map is active
///
/// `ModelConfig::num_hidden_layers` is the number of residual layers per head.
pub fn medusa_from_config(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn MedusaDraftModel>, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "MedusaModel" => Ok(Box::new(MedusaModel::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}

/// Create an MLP Speculator draft model from a model configuration.
///
/// Reads speculator hyperparameters from `ModelConfig::extra`:
/// - `emb_dim` — input dimension from target model (defaults to `hidden_size`)
/// - `inner_dim` — speculator hidden size (0 → same as `emb_dim`)
/// - `n_predict` — number of lookahead tokens (default 3)
/// - `tie_weights` — share emb/proj/ln weights across heads (default false)
/// - `scale_input` — L2-normalise input hidden states (default false)
///
/// HF checkpoints store weights under a `speculator.*` prefix which is
/// handled internally by [`MLPSpeculatorModel::from_config`].
pub fn mlp_speculator_from_config(
    cfg: &ModelConfig,
    vb: candle_nn::VarBuilder,
) -> Result<MLPSpeculatorModel, ModelError> {
    let arch = get_arch(cfg)?;
    match arch {
        "MLPSpeculatorPreTrainedModel" => Ok(MLPSpeculatorModel::from_config(cfg, vb)?),
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

    #[cfg(feature = "model-registry-v2")]
    {
        use factory::Capabilities;
        if let Some(factory) = registry_v2::lookup(arch) {
            if factory.info().supports(Capabilities::TP) {
                return factory.build_with_tp(cfg, vb, pg, tp_ctx);
            }
        }
    }

    // Registry entry without TP capability OR arch not in registry:
    // fall back to single-GPU construction with a warning so the user
    // knows TP wasn't applied.
    tracing::warn!(
        architecture = arch,
        "TP not yet implemented for this architecture, using single-GPU fallback"
    );
    from_config(cfg, vb)
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

    #[cfg(feature = "model-registry-v2")]
    {
        use factory::Capabilities;
        if let Some(factory) = registry_v2::lookup(arch) {
            if factory.info().supports(Capabilities::LORA) {
                return factory.build_with_lora(cfg, vb);
            }
        }
    }

    Err(ModelError::UnsupportedArchitecture(arch.into()))
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

    #[test]
    fn speculative_kind_routes_known_architectures() {
        // One representative per kind covers the dispatch table well
        // enough; the *_from_config functions exhaustively test the
        // arch_name lists in their own tests.
        assert_eq!(
            speculative_kind_for("DeepSeekMTPModel"),
            Some(DraftKind::Mtp)
        );
        assert_eq!(
            speculative_kind_for("EagleLlamaForCausalLM"),
            Some(DraftKind::Eagle1)
        );
        assert_eq!(
            speculative_kind_for("EagleDeepseekV3ForCausalLM"),
            Some(DraftKind::Eagle1),
            "Python registry alias for legacy EagleDeepSeekMTPModel"
        );
        assert_eq!(
            speculative_kind_for("Eagle3LlamaForCausalLM"),
            Some(DraftKind::Eagle3)
        );
        assert_eq!(speculative_kind_for("MedusaModel"), Some(DraftKind::Medusa));
        assert_eq!(
            speculative_kind_for("MLPSpeculatorPreTrainedModel"),
            Some(DraftKind::MlpSpeculator)
        );
        assert_eq!(speculative_kind_for("LlamaForCausalLM"), None);
        assert_eq!(speculative_kind_for("UnknownArch"), None);
    }

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
            bos_token_id: Some(0),
            eos_token_id: Some(0),
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

    #[test]
    fn test_from_config_hunyuan_vl() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("HunYuanVLForConditionalGeneration");
        cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "patch_size": 4,
                "spatial_merge_size": 2,
                "out_hidden_size": 32,
                "max_image_size": 8,
                "rms_norm_eps": 1e-5
            }),
        );
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "from_config should handle HunYuanVLForConditionalGeneration: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_intern_vl_alias() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("InternVLForConditionalGeneration");
        cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "image_size": 32,
                "patch_size": 8,
                "layer_norm_eps": 1e-6,
                "qkv_bias": true
            }),
        );
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "InternVLForConditionalGeneration should alias to InternS1: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_step_vl_alias() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("StepVLForConditionalGeneration");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "StepVLForConditionalGeneration should alias to Step3VL: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_dots_ocr() {
        use serde_json::json;
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("DotsOCRForCausalLM");
        cfg.extra.insert(
            "vision_config".into(),
            json!({
                "embed_dim": 8, "hidden_size": 16, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_channels": 3,
                "patch_size": 4, "spatial_merge_size": 1, "temporal_patch_size": 1,
                "rms_norm_eps": 1e-5, "use_bias": false, "post_norm": false
            }),
        );
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "DotsOCRForCausalLM should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_lighton_ocr() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("LightOnOCRForConditionalGeneration");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "LightOnOCRForConditionalGeneration should build via Mistral3: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_lfm2_vl() {
        use serde_json::json;
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("Lfm2VLForConditionalGeneration");
        cfg.extra.insert(
            "vision_config".to_string(),
            json!({
                "patch_size": 2, "num_channels": 1, "hidden_size": 8,
                "num_attention_heads": 2, "intermediate_size": 16,
                "num_hidden_layers": 1, "layer_norm_eps": 1e-6, "num_patches": 4
            }),
        );
        cfg.extra.insert("downsample_factor".to_string(), json!(2));
        cfg.extra
            .insert("projector_hidden_size".to_string(), json!(16));
        cfg.extra.insert("projector_bias".to_string(), json!(false));
        cfg.extra
            .insert("projector_use_layernorm".to_string(), json!(false));
        cfg.extra
            .insert("layer_types".to_string(), json!(["full_attention"]));
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "Lfm2VLForConditionalGeneration should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_bee() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("BeeForConditionalGeneration");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "BeeForConditionalGeneration should build with custom projector: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_rvl() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("RForConditionalGeneration");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "RForConditionalGeneration should build with custom projector: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_teleflm_no_mup() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let cfg = minimal_config("TeleFLMForCausalLM");
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "TeleFLMForCausalLM (no muP) should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_from_config_teleflm_with_mup() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let mut cfg = minimal_config("TeleFLMForCausalLM");
        cfg.extra
            .insert("use_mup".to_string(), serde_json::json!(true));
        cfg.extra
            .insert("input_mult".to_string(), serde_json::json!(2.0));
        cfg.extra
            .insert("output_mult".to_string(), serde_json::json!(4.0));
        cfg.extra
            .insert("mup_scale_factor".to_string(), serde_json::json!(2.0));
        let result = from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "TeleFLMForCausalLM (with muP) should build: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_mlp_speculator_from_config() {
        let device = candle_core::Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);

        let mut cfg = ModelConfig {
            architectures: vec!["MLPSpeculatorPreTrainedModel".to_string()],
            hidden_size: 64,
            vocab_size: 256,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 0,
            head_dim: 16,
            max_position_embeddings: 128,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: None,
            extra: serde_json::Map::new(),
        };
        cfg.extra
            .insert("emb_dim".to_string(), serde_json::json!(64));
        cfg.extra
            .insert("inner_dim".to_string(), serde_json::json!(32));
        cfg.extra
            .insert("n_predict".to_string(), serde_json::json!(3));

        let result = mlp_speculator_from_config(&cfg, vb);
        assert!(
            result.is_ok(),
            "mlp_speculator_from_config should build: {:?}",
            result.err()
        );
        let model = result.unwrap();
        assert_eq!(model.n_predict(), 3);
    }
}
