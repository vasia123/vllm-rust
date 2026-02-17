//! BLIP-2 vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: BLIP ViT (CLIP-like, reused from multimodal/vision.rs)
//! - Q-Former: Cross-attention between learnable queries and vision features
//! - Language projection: Linear (qformer_hidden → text_hidden)
//! - Language model: LLaMA decoder layers (reused from llama.rs)
//!
//! Reference: reference/vllm/vllm/model_executor/models/blip2.py

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::llama::{LlamaDecoderLayer, TpContext};
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Q-Former Config ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QFormerConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    encoder_hidden_size: usize,
    cross_attention_frequency: usize,
    layer_norm_eps: f64,
}

impl QFormerConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl Default for QFormerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            encoder_hidden_size: 1408,
            cross_attention_frequency: 2,
            layer_norm_eps: 1e-12,
        }
    }
}

// ─── Q-Former Multi-Head Attention ──────────────────────────────────────────

struct QFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    output_ln: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl QFormerAttention {
    fn new(
        hidden_size: usize,
        key_value_size: usize,
        cfg: &QFormerConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let all_head_size = num_heads * head_dim;

        let query = linear(hidden_size, all_head_size, vb.pp("attention").pp("query"))?;
        let key = linear(key_value_size, all_head_size, vb.pp("attention").pp("key"))?;
        let value = linear(
            key_value_size,
            all_head_size,
            vb.pp("attention").pp("value"),
        )?;
        let output = linear(all_head_size, hidden_size, vb.pp("output").pp("dense"))?;
        let output_ln = layer_norm(
            hidden_size,
            cfg.layer_norm_eps,
            vb.pp("output").pp("LayerNorm"),
        )?;

        Ok(Self {
            query,
            key,
            value,
            output,
            output_ln,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        let q = self.query.forward(hidden_states)?;
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let kv_input = encoder_hidden_states.unwrap_or(hidden_states);
        let kv_len = kv_input.dim(1)?;

        let k = self.key.forward(kv_input)?;
        let k = k
            .reshape((batch, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let v = self.value.forward(kv_input)?;
        let v = v
            .reshape((batch, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((batch, seq_len, ()))?;

        let output = self.output.forward(&attn_output)?;
        self.output_ln.forward(&(output + hidden_states)?)
    }
}

// ─── Q-Former FFN ───────────────────────────────────────────────────────────

struct QFormerFFN {
    intermediate: Linear,
    output: Linear,
    output_ln: LayerNorm,
}

impl QFormerFFN {
    fn new(cfg: &QFormerConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("intermediate").pp("dense"),
        )?;
        let output = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("output").pp("dense"),
        )?;
        let output_ln = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("output").pp("LayerNorm"),
        )?;
        Ok(Self {
            intermediate,
            output,
            output_ln,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let xs = self.intermediate.forward(hidden_states)?;
        let xs = candle_nn::Activation::Gelu.forward(&xs)?;
        let xs = self.output.forward(&xs)?;
        self.output_ln.forward(&(xs + residual)?)
    }
}

// ─── Q-Former Layer ─────────────────────────────────────────────────────────

struct QFormerLayer {
    self_attn: QFormerAttention,
    cross_attn: Option<QFormerAttention>,
    ffn: QFormerFFN,
    cross_attn_ln: Option<LayerNorm>,
}

impl QFormerLayer {
    fn new(cfg: &QFormerConfig, has_cross_attn: bool, vb: VarBuilder) -> Result<Self> {
        let self_attn =
            QFormerAttention::new(cfg.hidden_size, cfg.hidden_size, cfg, vb.pp("attention"))?;

        let (cross_attn, cross_attn_ln) = if has_cross_attn {
            let ca = QFormerAttention::new(
                cfg.hidden_size,
                cfg.encoder_hidden_size,
                cfg,
                vb.pp("crossattention"),
            )?;
            let ca_ln = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm_cross_att"));
            // Cross-attn layer norm is optional in some configs
            (Some(ca), ca_ln.ok())
        } else {
            (None, None)
        };

        let ffn = QFormerFFN::new(cfg, vb)?;

        Ok(Self {
            self_attn,
            cross_attn,
            ffn,
            cross_attn_ln,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        query_length: usize,
    ) -> Result<Tensor> {
        // Self-attention on the full sequence (queries + optional text tokens)
        let mut xs = self.self_attn.forward(hidden_states, None)?;

        // Cross-attention (only on query tokens, not text tokens)
        if let (Some(cross_attn), Some(enc_hidden)) = (&self.cross_attn, encoder_hidden_states) {
            let query_states = xs.narrow(1, 0, query_length)?;
            let query_states = if let Some(ln) = &self.cross_attn_ln {
                ln.forward(&query_states)?
            } else {
                query_states
            };
            let cross_out = cross_attn.forward(&query_states, Some(enc_hidden))?;
            // Replace query positions with cross-attention output
            let seq_len = xs.dim(1)?;
            if seq_len > query_length {
                let text_states = xs.narrow(1, query_length, seq_len - query_length)?;
                xs = Tensor::cat(&[&cross_out, &text_states], 1)?;
            } else {
                xs = cross_out;
            }
        }

        // FFN on query tokens
        let query_ffn = self.ffn.forward(&xs.narrow(1, 0, query_length)?)?;
        let seq_len = xs.dim(1)?;
        if seq_len > query_length {
            let text_states = xs.narrow(1, query_length, seq_len - query_length)?;
            Ok(Tensor::cat(&[&query_ffn, &text_states], 1)?)
        } else {
            Ok(query_ffn)
        }
    }
}

// ─── Q-Former Model ─────────────────────────────────────────────────────────

struct Blip2QFormerModel {
    layers: Vec<QFormerLayer>,
    query_tokens: Tensor,
    layernorm: LayerNorm,
    num_query_tokens: usize,
}

impl Blip2QFormerModel {
    fn new(cfg: &QFormerConfig, num_query_tokens: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("encoder").pp("layer");
        for i in 0..cfg.num_hidden_layers {
            let has_cross_attn = i % cfg.cross_attention_frequency == 0;
            layers.push(QFormerLayer::new(cfg, has_cross_attn, vb_layers.pp(i))?);
        }

        let query_tokens = vb.get((1, num_query_tokens, cfg.hidden_size), "query_tokens")?;

        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;

        Ok(Self {
            layers,
            query_tokens,
            layernorm,
            num_query_tokens,
        })
    }

    /// Forward: cross-attend learnable queries to vision features.
    ///
    /// Returns: (batch, num_query_tokens, hidden_size)
    fn forward(&self, encoder_hidden_states: &Tensor) -> Result<Tensor> {
        let batch_size = encoder_hidden_states.dim(0)?;

        // Expand query tokens for batch
        let queries = self.query_tokens.broadcast_left(batch_size)?.reshape((
            batch_size,
            self.num_query_tokens,
            (),
        ))?;

        let mut hidden_states = queries;

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                Some(encoder_hidden_states),
                self.num_query_tokens,
            )?;
        }

        self.layernorm.forward(&hidden_states)
    }
}

// ─── BLIP2 Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Blip2Config {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    qformer_config: QFormerConfig,
    num_query_tokens: usize,
    image_token_id: u32,
}

impl Blip2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        // Vision config
        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
            VisionEncoderConfig {
                encoder_type: VisionEncoderType::Clip,
                hidden_size: vc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1408) as usize,
                intermediate_size: vc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(6144) as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(39) as usize,
                image_size: vc.get("image_size").and_then(|v| v.as_u64()).unwrap_or(224) as usize,
                patch_size: vc.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(14) as usize,
                num_channels: 3,
                layer_norm_eps: vc
                    .get("layer_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-6),
            }
        } else {
            VisionEncoderConfig {
                encoder_type: VisionEncoderType::Clip,
                hidden_size: 1408,
                intermediate_size: 6144,
                num_attention_heads: 16,
                num_hidden_layers: 39,
                image_size: 224,
                patch_size: 14,
                num_channels: 3,
                layer_norm_eps: 1e-6,
            }
        };

        // Q-Former config
        let qformer_config = if let Some(qc) = cfg.extra.get("qformer_config") {
            QFormerConfig {
                hidden_size: qc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(768) as usize,
                num_attention_heads: qc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(12) as usize,
                intermediate_size: qc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3072) as usize,
                num_hidden_layers: qc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(12) as usize,
                encoder_hidden_size: qc
                    .get("encoder_hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(vision_config.hidden_size as u64)
                    as usize,
                cross_attention_frequency: qc
                    .get("cross_attention_frequency")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2) as usize,
                layer_norm_eps: qc
                    .get("layer_norm_eps")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1e-12),
            }
        } else {
            QFormerConfig {
                encoder_hidden_size: vision_config.hidden_size,
                ..Default::default()
            }
        };

        let num_query_tokens = cfg
            .extra
            .get("num_query_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            qformer_config,
            num_query_tokens,
            image_token_id,
        }
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// BLIP-2 vision-language model.
///
/// Uses a Q-Former to cross-attend learnable queries to vision features,
/// producing a fixed number of visual tokens that are projected into the
/// language model's embedding space.
pub struct Blip2ForConditionalGeneration {
    // Vision
    #[allow(dead_code)]
    vision_model: VisionEncoder,
    qformer: Blip2QFormerModel,
    language_projection: Linear,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    // Config
    #[allow(dead_code)]
    config: Blip2Config,
    device: Device,
    dtype: DType,
}

impl Blip2ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Blip2Config::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision model
        let vision_model = VisionEncoder::new(&config.vision_config, vb.pp("vision_model"))?;

        // Q-Former
        let qformer = Blip2QFormerModel::new(
            &config.qformer_config,
            config.num_query_tokens,
            vb.pp("qformer"),
        )?;

        // Language projection (qformer_hidden → text_hidden)
        let language_projection = linear(
            config.qformer_config.hidden_size,
            cfg.hidden_size,
            vb.pp("language_projection"),
        )?;

        // Language model (LLaMA backbone)
        let vb_m = vb.pp("language_model").pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("language_model").pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_model,
            qformer,
            language_projection,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Process image features through Q-Former and project to LLM space.
    #[allow(dead_code)]
    fn process_image(&self, vision_features: &Tensor) -> Result<Tensor> {
        let query_output = self.qformer.forward(vision_features)?;
        self.language_projection.forward(&query_output)
    }

    /// Merge text embeddings with image embeddings at placeholder positions.
    fn merge_multimodal_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            // Run through Q-Former and project
            let vision_feats = processed_image.embedding.unsqueeze(0)?;
            let query_output = self.qformer.forward(&vision_feats)?;
            let projected = self.language_projection.forward(&query_output)?;
            let projected = projected.squeeze(0)?;
            let img_emb: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start_pos + i;
                if target_pos < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let text_embeddings = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            }
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }
}

impl crate::engine::ModelForward for Blip2ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
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
        self.forward_inner(
            input_ids,
            multimodal_inputs,
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 56,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }),
        );
        extra.insert(
            "qformer_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "encoder_hidden_size": 64,
                "cross_attention_frequency": 2,
                "layer_norm_eps": 1e-12
            }),
        );
        extra.insert("num_query_tokens".to_string(), serde_json::json!(4));
        extra.insert("image_token_id".to_string(), serde_json::json!(32000));

        ModelConfig {
            architectures: vec!["Blip2ForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Blip2ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_qformer_forward() {
        let device = Device::Cpu;
        let qformer_cfg = QFormerConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            intermediate_size: 64,
            num_hidden_layers: 2,
            encoder_hidden_size: 64,
            cross_attention_frequency: 2,
            layer_norm_eps: 1e-12,
        };
        let vb = VarBuilder::zeros(DType::F32, &device);
        let qformer = Blip2QFormerModel::new(&qformer_cfg, 4, vb).unwrap();

        // Vision features: (batch=1, num_patches=16, vision_hidden=64)
        let vision_feats = Tensor::zeros((1, 16, 64), DType::F32, &device).unwrap();
        let output = qformer.forward(&vision_feats).unwrap();

        // Output: (1, num_query_tokens=4, hidden_size=32)
        assert_eq!(output.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Blip2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Blip2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
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

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Blip2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }
}
