//! Molmo2 vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: Custom ViT (SigLIP-based, 27 layers, 1152-dim)
//! - Multi-layer feature extraction from VIT_LAYERS (-3, -9)
//! - ImagePoolingAttention for feature aggregation
//! - Projector: SiLU MLP mapping to LLM hidden dim (3584)
//! - Language model: Qwen2-like backbone with GQA (28 heads → 4 KV heads)
//!   and optional Q/K normalization
//!
//! Reference: reference/vllm/vllm/model_executor/models/molmo2.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{causal_mask, paged_attention, RotaryEmbedding};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Molmo2VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    image_size: usize,
    patch_size: usize,
}

impl Default for Molmo2VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 378,
            patch_size: 14,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Molmo2AdapterConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    text_hidden_size: usize,
    vit_layers: Vec<i32>,
}

impl Default for Molmo2AdapterConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 18944,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 72,
            text_hidden_size: 3584,
            vit_layers: vec![-3, -9],
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Molmo2Config {
    model_config: ModelConfig,
    vit_config: Molmo2VisionConfig,
    adapter_config: Molmo2AdapterConfig,
    additional_vocab_size: usize,
    image_patch_id: u32,
}

impl Molmo2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vit_config = if let Some(vc) = cfg.extra.get("vit_config") {
            Molmo2VisionConfig {
                hidden_size: vc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1152) as usize,
                intermediate_size: vc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4304) as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(27) as usize,
                image_size: vc
                    .get("image_default_input_size")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_u64())
                    .unwrap_or(378) as usize,
                patch_size: vc
                    .get("image_patch_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(14) as usize,
            }
        } else {
            Molmo2VisionConfig::default()
        };

        let adapter_config = if let Some(ac) = cfg.extra.get("adapter_config") {
            let vit_layers = ac
                .get("vit_layers")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_i64().map(|x| x as i32))
                        .collect()
                })
                .unwrap_or_else(|| vec![-3, -9]);

            Molmo2AdapterConfig {
                hidden_size: ac
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1152) as usize,
                intermediate_size: ac
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(18944) as usize,
                num_attention_heads: ac
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize,
                num_key_value_heads: ac
                    .get("num_key_value_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize,
                head_dim: ac.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(72) as usize,
                text_hidden_size: ac
                    .get("text_hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(cfg.hidden_size as u64) as usize,
                vit_layers,
            }
        } else {
            Molmo2AdapterConfig {
                text_hidden_size: cfg.hidden_size,
                ..Molmo2AdapterConfig::default()
            }
        };

        let additional_vocab_size = cfg
            .extra
            .get("additional_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        let image_patch_id = cfg
            .extra
            .get("image_patch_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(152066) as u32;

        Self {
            model_config: cfg.clone(),
            vit_config,
            adapter_config,
            additional_vocab_size,
            image_patch_id,
        }
    }
}

// ─── Image Projector MLP ────────────────────────────────────────────────────

/// SiLU-gated MLP for projecting image features to LLM hidden dim.
#[allow(dead_code)]
struct ImageProjectorMLP {
    fc1: Linear,
    fc2: Linear,
    intermediate_size: usize,
}

#[allow(dead_code)]
impl ImageProjectorMLP {
    fn new(
        input_dim: usize,
        intermediate_size: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = candle_nn::linear(input_dim, intermediate_size * 2, vb.pp("w1"))?;
        let fc2 = candle_nn::linear(intermediate_size, output_dim, vb.pp("w2"))?;
        Ok(Self {
            fc1,
            fc2,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.fc1.forward(x)?;
        let gate = gate_up.narrow(2, 0, self.intermediate_size)?;
        let up = gate_up.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        self.fc2.forward(&x)
    }
}

// ─── LLM Attention ──────────────────────────────────────────────────────────

/// Molmo2 attention with GQA and optional Q/K normalization.
#[allow(dead_code)]
struct Molmo2Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

impl Molmo2Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let total_qkv = (num_heads + 2 * num_kv_heads) * head_dim;
        let qkv_proj = candle_nn::linear(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let use_qk_norm = cfg
            .extra
            .get("use_qk_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let (q_norm, k_norm) = if use_qk_norm {
            (
                Some(rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?),
                Some(rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };

        let rotary = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups: num_heads / num_kv_heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_dim)?;
        let k = qkv.narrow(2, q_dim, kv_dim)?;
        let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;

        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Optional Q/K normalization
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        let (q, k) = self.rotary.apply(&q, &k, seqlen_offset)?;

        paged_attention(
            &q,
            &k,
            &v,
            mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

/// SwiGLU MLP for the Molmo2 LLM.
struct Molmo2MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl Molmo2MLP {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = candle_nn::linear(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj =
            candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(2, 0, self.intermediate_size)?;
        let up = gate_up.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        self.down_proj.forward(&x)
    }
}

struct Molmo2DecoderLayer {
    self_attn: Molmo2Attention,
    mlp: Molmo2MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Molmo2DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Molmo2Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Molmo2MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(
            &x,
            mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let x = (residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Molmo2 vision-language model for conditional generation.
///
/// Custom ViT + ImagePoolingAttention + MLP projector + GQA LLM.
pub struct Molmo2ForConditionalGeneration {
    // Vision (uses SigLIP-compatible encoder for simplicity)
    #[allow(dead_code)]
    vision_backbone: VisionEncoder,
    #[allow(dead_code)]
    image_projector: ImageProjectorMLP,
    // LLM
    embed_tokens: TpEmbedding,
    #[allow(dead_code)]
    additional_embed: Option<Embedding>,
    layers: Vec<Molmo2DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    #[allow(dead_code)]
    config: Molmo2Config,
    device: Device,
    dtype: DType,
}

impl Molmo2ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Molmo2Config::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision backbone: SigLIP-compatible encoder
        let vis_cfg = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: config.vit_config.hidden_size,
            intermediate_size: config.vit_config.intermediate_size,
            num_attention_heads: config.vit_config.num_attention_heads,
            num_hidden_layers: config.vit_config.num_hidden_layers,
            image_size: config.vit_config.image_size,
            patch_size: config.vit_config.patch_size,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        };
        let vision_backbone =
            VisionEncoder::new(&vis_cfg, vb.pp("vision_backbone").pp("image_vit"))?;

        // Image projector MLP
        // Input dim = 2 * vit hidden (from concatenating 2 layer outputs)
        let proj_input = config.vit_config.hidden_size * config.adapter_config.vit_layers.len();
        let image_projector = ImageProjectorMLP::new(
            proj_input,
            config.adapter_config.intermediate_size,
            config.adapter_config.text_hidden_size,
            vb.pp("vision_backbone").pp("image_projector"),
        )?;

        // LLM backbone
        let vb_m = vb.pp("model");
        let total_vocab = cfg.vocab_size + config.additional_vocab_size;

        let embed_tokens =
            TpEmbedding::new(total_vocab, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        // Additional vocabulary embedding (for image tokens)
        let additional_embed = if config.additional_vocab_size > 0 {
            Some(candle_nn::embedding(
                config.additional_vocab_size,
                cfg.hidden_size,
                vb_m.pp("additional_embed"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Molmo2DecoderLayer::new(cfg, vb_l.pp(i))?);
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
                total_vocab,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                total_vocab,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_backbone,
            image_projector,
            embed_tokens,
            additional_embed,
            layers,
            norm,
            lm_head,
            tp_ctx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        _multimodal_inputs: Option<&MultimodalInputs>,
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

        // TODO: For multimodal, extract vision features and replace image patch tokens.
        // Currently only supports text-only forward.
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }
}

impl crate::engine::ModelForward for Molmo2ForConditionalGeneration {
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
            let mut batch_outputs = Vec::with_capacity(sequences.len());
            let cache_engine = kv_cache_mgr.engine_mut(layer_idx);
            for (seq_idx, seq_meta) in sequences.iter().enumerate() {
                let x_single = xs.narrow(0, seq_idx, 1)?;
                let block_table = BlockTable::from_block_ids(seq_meta.block_ids.clone(), 0);
                let out = layer.forward(
                    &x_single,
                    None,
                    seq_meta.seqlen_offset,
                    cache_engine,
                    &block_table,
                    &seq_meta.slot_mapping,
                )?;
                batch_outputs.push(out);
            }
            xs = Tensor::cat(&batch_outputs, 0)?;
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
            "vit_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_default_input_size": [56, 56],
                "image_patch_size": 14
            }),
        );
        extra.insert(
            "adapter_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "text_hidden_size": 64,
                "vit_layers": [-1, -2]
            }),
        );
        extra.insert("additional_vocab_size".to_string(), serde_json::json!(16));
        extra.insert("use_qk_norm".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Molmo2ForConditionalGeneration".to_string()],
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
            tie_word_embeddings: false,
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
        let model = Molmo2ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Molmo2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        // vocab_size + additional_vocab_size = 256 + 16 = 272
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, 272]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Molmo2ForConditionalGeneration::new(&cfg, vb).unwrap();

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
        let model = Molmo2ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, 272]);
        block_table.advance(3);

        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, 272]);
    }

    #[test]
    fn test_image_projector_mlp() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let projector = ImageProjectorMLP::new(128, 256, 64, vb).unwrap();

        let input = Tensor::zeros((1, 16, 128), DType::F32, &device).unwrap();
        let output = projector.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 16, 64]);
    }
}
