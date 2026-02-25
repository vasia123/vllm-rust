//! MiniCPM-O: vision + audio + language model.
//!
//! Extends MiniCPM-V (vision) with audio encoding capabilities.
//! Audio processing pipeline at Python preprocessing time:
//!
//! ```text
//! mel [B, 80, T]
//!   └── MiniCPMWhisperEncoder (Conv1d×2 + GELU + sinusoidal pos + N transformer layers)
//!         → [B, T', d_model]
//!   └── MultiModalProjector (Linear → ReLU → Linear)
//!         → [B, T', embed_dim]
//!   └── AvgPool1d (stride = audio_pool_step)
//!         → [B, T'/pool_step, embed_dim]   (already in LLM hidden dimension)
//! ```
//!
//! At Rust inference time, pre-encoded audio embeddings (already projected to
//! LLM `hidden_size`) are scattered at audio placeholder token positions.
//! The `apm` encoder and `audio_projection_layer` are loaded to satisfy weight
//! initialization but are **not called during inference**.
//!
//! ## Version dispatch
//! - Version 2.6: Qwen2 LLM backbone (default)
//! - Version 4.5: Qwen3 LLM backbone
//!
//! Both versions are handled by `MiniCPMVForConditionalGeneration` (Qwen2 base).
//! `MiniCPMOForCausalLM` delegates to the v2.6 path for now.
//!
//! ## Weight paths
//! - `vpm.*`                       — vision encoder (Idefics2-style ViT)
//! - `resampler.*`                 — vision resampler (cross-attention queries)
//! - `apm.*`                       — audio encoder (stub)
//! - `audio_projection_layer.*`    — audio MLP projector (stub)
//! - `llm.*`                       — LLM backbone (Qwen2/Qwen3)
//!
//! Reference: reference/vllm/vllm/model_executor/models/minicpmo.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::minicpmv::MiniCPMVForConditionalGeneration;

// ─── Audio encoder stub ──────────────────────────────────────────────────────

/// Stub for the MiniCPM-O Whisper-based audio encoder.
///
/// The encoder (Conv1d×2 + Transformer layers) runs during Python preprocessing
/// and is NOT invoked during Rust inference.  This struct exists purely to
/// satisfy VarBuilder during weight loading.
///
/// TODO: Implement full Whisper-style encoder for native Rust audio encoding.
/// Reference: `MiniCPMWhisperEncoder` in minicpmo.py (inherits WhisperEncoder
/// with custom pre-norm layer ordering).
#[allow(dead_code)]
pub(crate) struct MiniCPMOAudioEncoder;

impl MiniCPMOAudioEncoder {
    fn new(_vb: VarBuilder) -> Result<Self> {
        Ok(Self)
    }
}

// ─── Audio projector ─────────────────────────────────────────────────────────

/// Two-layer ReLU audio projector (Linear → ReLU → Linear).
///
/// Maps `d_model → embed_dim`.  Weight paths:
/// - `audio_projection_layer.linear1.{weight,bias}`
/// - `audio_projection_layer.linear2.{weight,bias}`
///
/// Not called during Rust inference: pre-encoded `ProcessedAudio.embedding`
/// tensors are already in LLM hidden dimension and scatter directly.
#[allow(dead_code)]
pub(crate) struct MiniCPMOAudioProjector {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl MiniCPMOAudioProjector {
    fn new(audio_dim: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(audio_dim, embed_dim, vb.pp("linear1"))?,
            linear2: linear(embed_dim, embed_dim, vb.pp("linear2"))?,
        })
    }

    #[allow(dead_code)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear1.forward(xs)?.relu()?;
        self.linear2.forward(&xs)
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

struct MiniCPMOConfig {
    audio_d_model: usize,
    audio_token_id: u32,
}

impl MiniCPMOConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let audio_d_model = cfg
            .extra
            .get("audio_config")
            .and_then(|ac| ac.get("d_model"))
            .and_then(|v| v.as_u64())
            .unwrap_or(384) as usize;

        let audio_token_id = cfg
            .extra
            .get("audio_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(128244) as u32; // same sentinel as image_token_id default

        Self {
            audio_d_model,
            audio_token_id,
        }
    }
}

// ─── Full model ──────────────────────────────────────────────────────────────

/// MiniCPM-O vision-audio-language model.
///
/// Combines MiniCPM-V vision path with Whisper-based audio encoding.
/// Vision embeddings are handled by the inner `vision_model`.
/// Audio embeddings (pre-encoded) are scattered at audio placeholder positions
/// before the LLM forward pass.
pub struct MiniCPMOForCausalLM {
    vision_model: MiniCPMVForConditionalGeneration,
    #[allow(dead_code)]
    apm: MiniCPMOAudioEncoder,
    #[allow(dead_code)]
    audio_projection_layer: MiniCPMOAudioProjector,
    audio_token_id: u32,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl MiniCPMOForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_inner(cfg, vb)
    }

    fn new_inner(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ocfg = MiniCPMOConfig::from_model_config(cfg);

        // Vision model — loaded under "vpm" (vision), "resampler", "llm" prefixes
        // MiniCPMVForConditionalGeneration already handles these weight prefixes.
        let vision_model = MiniCPMVForConditionalGeneration::new(cfg, vb.clone())?;

        // Audio encoder stub — weights loaded but not used at inference
        let apm = MiniCPMOAudioEncoder::new(vb.pp("apm"))?;

        // Audio projector — loaded but not called (pre-encoded path)
        let audio_projection_layer = MiniCPMOAudioProjector::new(
            ocfg.audio_d_model,
            cfg.hidden_size,
            vb.pp("audio_projection_layer"),
        )?;

        Ok(Self {
            vision_model,
            apm,
            audio_projection_layer,
            audio_token_id: ocfg.audio_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }
}

impl ModelForward for MiniCPMOForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.vision_model.forward(
            input_ids,
            None,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        crate::engine::ModelForward::forward_decode_batch(
            &self.vision_model,
            input_ids,
            sequences,
            kv_cache_mgr,
        )
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let mm = multimodal_inputs;
        let has_audio = mm.is_some_and(|m| m.has_audio());

        if !has_audio {
            // Audio-free: delegate entirely to vision model
            return self.vision_model.forward(
                input_ids,
                mm,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            );
        }

        // Mask out-of-vocab audio token IDs before embed_tokens
        let safe_ids = {
            let tok = self.audio_token_id;
            let ids = input_ids.to_vec2::<u32>()?;
            let masked: Vec<Vec<u32>> = ids
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|t| if t == tok { 0 } else { t })
                        .collect()
                })
                .collect();
            Tensor::new(masked, input_ids.device())?
        };

        // Build embeddings (text + optional vision)
        let mut xs = self.vision_model.embed_and_merge_vision(&safe_ids, mm)?;

        // Scatter pre-encoded audio embeddings at audio placeholder positions
        if let Some(mm_inputs) = mm {
            xs = scatter_audio_embeddings(xs, mm_inputs, self.audio_token_id)?;
        }

        // Complete LLM forward from embeddings
        self.vision_model.forward_from_embeddings(
            &xs,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Audio scatter ───────────────────────────────────────────────────────────

/// Scatter pre-encoded audio embeddings into the text embedding tensor at audio
/// placeholder positions.
///
/// Pre-encoded `ProcessedAudio.embedding` is already in LLM hidden dimension
/// (no projection needed). Placeholder positions identified by `audio_token_id`
/// in `mm.token_ids`.
fn scatter_audio_embeddings(
    xs: Tensor,
    mm: &MultimodalInputs,
    audio_token_id: u32,
) -> Result<Tensor> {
    if mm.audio_embeddings.is_empty() {
        return Ok(xs);
    }

    let (b, s, d) = xs.dims3()?;

    let mut clips: Vec<(usize, Tensor)> = mm
        .audio_embeddings
        .iter()
        .map(|(pos, pa)| (*pos, pa.embedding.clone()))
        .collect();
    clips.sort_by_key(|(pos, _)| *pos);

    let flat = xs.reshape((b * s, d))?;
    let token_ids = &mm.token_ids;

    let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
    let mut clip_idx = 0usize;
    let mut clip_offset = 0usize;

    for (seq_idx, &tok) in token_ids.iter().enumerate() {
        if tok == audio_token_id && clip_idx < clips.len() {
            let clip = &clips[clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, clip_offset, 1)?.squeeze(0)?);
            clip_offset += 1;
            if clip_offset >= clip_len {
                clip_idx += 1;
                clip_offset = 0;
            }
        } else {
            rows.push(flat.narrow(0, seq_idx, 1)?.squeeze(0)?);
        }
    }

    Tensor::stack(&rows, 0)?.reshape((b, s, d))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::{MultimodalInputs, ProcessedAudio};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

    fn make_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        let vision = json!({
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 28,
            "patch_size": 14,
            "layer_norm_eps": 1e-6,
        });
        extra.insert("vision_config".into(), vision);
        extra.insert("query_num".into(), json!(4));
        extra.insert("image_token_id".into(), json!(128244u32));
        extra.insert("audio_token_id".into(), json!(10u32)); // in-vocab for test
        extra.insert("audio_config".into(), json!({ "d_model": 8 }));

        ModelConfig {
            architectures: vec!["MiniCPMO".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, dev: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: dev.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn minicpmo_construction() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MiniCPMOForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "{:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn minicpmo_text_forward() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MiniCPMOForCausalLM::new(&cfg, vb).unwrap();
        let mut cache = make_cache(&cfg, &dev);
        let bt = BlockTable::from_block_ids(vec![0, 1], 0);
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &dev).unwrap();
        let out = model
            .forward(&input, 0, &mut cache, &bt, &[0, 1, 2, 3])
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn minicpmo_decode_batch() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MiniCPMOForCausalLM::new(&cfg, vb).unwrap();
        let mut cache = make_cache(&cfg, &dev);
        let seqs = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];
        let input = Tensor::from_vec(vec![1u32, 2], (2, 1), &dev).unwrap();
        let out = model
            .forward_decode_batch(&input, &seqs, &mut cache)
            .unwrap();
        assert_eq!(out.dim(0).unwrap(), 2);
    }

    #[test]
    fn minicpmo_audio_scatter() {
        let dev = Device::Cpu;
        let cfg = make_config();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = MiniCPMOForCausalLM::new(&cfg, vb).unwrap();

        let audio_token_id = 10u32; // matches config
        let seq_len = 4usize;
        let audio_tokens = 2usize;

        // Sequence: [audio, audio, text, text]
        let token_ids = vec![audio_token_id, audio_token_id, 1u32, 2u32];
        let embedding = Tensor::zeros((audio_tokens, cfg.hidden_size), DType::F32, &dev).unwrap();
        let processed = ProcessedAudio::new(embedding, audio_tokens);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(0, processed)]);

        let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &dev).unwrap();
        let mut cache = make_cache(&cfg, &dev);
        let bt = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let out = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut cache, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(out.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn minicpmo_audio_scatter_fn() {
        let dev = Device::Cpu;
        let b = 1usize;
        let s = 4usize;
        let d = 8usize;
        let audio_token_id = 10u32;

        let token_ids = vec![audio_token_id, audio_token_id, 1u32, 2u32];
        let emb = Tensor::ones((2usize, d), DType::F32, &dev).unwrap();
        let processed = ProcessedAudio::new(emb, 2);
        let mm = MultimodalInputs::with_audio(token_ids, vec![(0, processed)]);

        let xs = Tensor::zeros((b, s, d), DType::F32, &dev).unwrap();
        let out = scatter_audio_embeddings(xs, &mm, audio_token_id).unwrap();

        assert_eq!(out.dims(), &[b, s, d]);
        // First two rows should be 1.0 (from audio), last two 0.0 (from text)
        let flat = out.reshape((s, d)).unwrap().to_vec2::<f32>().unwrap();
        assert!((flat[0][0] - 1.0).abs() < 1e-5);
        assert!((flat[1][0] - 1.0).abs() < 1e-5);
        assert!((flat[2][0]).abs() < 1e-5);
        assert!((flat[3][0]).abs() < 1e-5);
    }
}
