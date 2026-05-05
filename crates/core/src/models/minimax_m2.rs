//! MiniMaxM2 model implementation.
//!
//! MiniMaxM2 is a Mixture-of-Experts model with:
//! - Q/K normalization (RMSNorm per head)
//! - Configurable rotary dimension (partial RoPE)
//! - MoE with optional e_score_correction_bias
//! - FP32 router with configurable scoring function
//!
//! Architecture:
//! ```text
//! Embedding -> [DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! DecoderLayer:
//!   RMSNorm -> Attention(QK-norm, partial RoPE) -> RMSNorm -> MoE
//! ```
//!
//! Config keys from extra:
//! - `num_local_experts`: number of experts
//! - `num_experts_per_tok`: top-k experts
//! - `rotary_dim`: rotary embedding dimension (may differ from head_dim)
//! - `use_routing_bias`: whether MoE uses e_score_correction_bias
//! - `scoring_func`: "softmax" or "sigmoid"
//! - `attention_bias`: whether QKV has bias

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBias, AttentionBlock, AttentionConfig, QkNormVariant};
use crate::layers::RotaryEmbedding;
use crate::moe::{MoELayer, MoELayerConfig, ScoringFunc};

use super::tp_layers::TpContext;

// ─── Config ─────────────────────────────────────────────────────────────────

struct MiniMaxM2Config {
    num_local_experts: usize,
    num_experts_per_tok: usize,
    rotary_dim: usize,
    use_routing_bias: bool,
    scoring_func: String,
    attention_bias: bool,
}

impl MiniMaxM2Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(8);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let rotary_dim = cfg
            .extra
            .get("rotary_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let use_routing_bias = cfg
            .extra
            .get("use_routing_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let scoring_func = cfg
            .extra
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();

        let attention_bias = cfg.attention_bias.unwrap_or(false);

        Self {
            num_local_experts,
            num_experts_per_tok,
            rotary_dim,
            use_routing_bias,
            scoring_func,
            attention_bias,
        }
    }
}

// ─── MoE ────────────────────────────────────────────────────────────────────

// MiniMax-M2 MoE = top-k routing (softmax or sigmoid) + optional
// `e_score_correction_bias` for sigmoid mode + Mixtral-style
// `w1`/`w2`/`w3` expert weight paths. The bespoke per-token CPU
// scalar loop has been replaced by the shared `MoELayer`, which uses
// the same `MoEExpert` weight path layout (`experts.{i}.w{1,2,3}`)
// and applies router-bias + sigmoid scoring through the existing
// `RouterConfig` knobs.
struct MiniMaxM2MoE {
    inner: MoELayer,
}

impl MiniMaxM2MoE {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let scoring_func = match m2_cfg.scoring_func.as_str() {
            "sigmoid" => ScoringFunc::Sigmoid,
            _ => ScoringFunc::Softmax,
        };

        let moe_cfg = MoELayerConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_experts: m2_cfg.num_local_experts,
            top_k: m2_cfg.num_experts_per_tok,
            renormalize: true,
            scoring_func,
            inplace: false,
            is_act_and_mul: true,
        };

        let bias = if m2_cfg.use_routing_bias {
            Some(
                vb.get(m2_cfg.num_local_experts, "e_score_correction_bias")?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };

        let inner = MoELayer::new_with_router_bias(moe_cfg, vb, bias)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

// MiniMax-M2 = fused QKV (optional bias on QKV; o_proj never biased) +
// per-head RMS QK-norm + (optional) partial RoPE based on `rotary_dim`.
struct MiniMaxM2Attention {
    inner: AttentionBlock,
    tp_ctx: TpContext,
}

impl MiniMaxM2Attention {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim;

        let bias = if m2_cfg.attention_bias {
            AttentionBias::QKV_ONLY
        } else {
            AttentionBias::NONE
        };

        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused()
        .with_bias(bias)
        .with_qk_norm(QkNormVariant::PerHead, cfg.rms_norm_eps);

        let rotary_emb = if m2_cfg.rotary_dim < head_dim {
            let partial_factor = m2_cfg.rotary_dim as f64 / head_dim as f64;
            RotaryEmbedding::new_partial(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                partial_factor,
                true, // neox style
                vb.dtype(),
                vb.device(),
            )?
        } else {
            RotaryEmbedding::new(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.dtype(),
                vb.device(),
            )?
        };

        let pg = LocalProcessGroup::new();
        let inner = AttentionBlock::new(&attn_cfg, vb, &pg, rotary_emb)?;
        Ok(Self {
            inner,
            tp_ctx: TpContext::single_gpu(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.inner.forward(
            xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            &self.tp_ctx,
        )
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        self.inner
            .forward_decode_batch(xs, sequences, cache_engine, &self.tp_ctx)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct MiniMaxM2DecoderLayer {
    self_attn: MiniMaxM2Attention,
    moe: MiniMaxM2MoE,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MiniMaxM2DecoderLayer {
    fn new(cfg: &ModelConfig, m2_cfg: &MiniMaxM2Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = MiniMaxM2Attention::new(cfg, m2_cfg, vb.pp("self_attn"))?;
        let moe = MiniMaxM2MoE::new(cfg, m2_cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Pre-norm + attention
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;

        // Pre-norm + MoE
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.moe.forward(&xs)?;
        let hidden_states = (xs + residual)?;

        let device = hidden_states.device().clone();
        Ok((hidden_states, Tensor::zeros(1, DType::F32, &device)?))
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_decode_batch(&xs, sequences, cache_engine)?;
        let xs = (xs + residual)?;

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.moe.forward(&xs)?;
        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniMaxM2ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxM2DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl MiniMaxM2ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let m2_cfg = MiniMaxM2Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiniMaxM2DecoderLayer::new(cfg, &m2_cfg, vb_layers.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }
}

impl crate::engine::ModelForward for MiniMaxM2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (hidden, _residual) = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
            xs = hidden;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
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

    fn test_m2_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("rotary_dim".to_string(), serde_json::json!(16));
        extra.insert("use_routing_bias".to_string(), serde_json::json!(false));
        extra.insert("scoring_func".to_string(), serde_json::json!("softmax"));

        ModelConfig {
            architectures: vec!["MiniMaxM2ForCausalLM".to_string()],
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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_m2_config_with_bias() -> ModelConfig {
        let mut cfg = test_m2_config();
        cfg.extra
            .insert("use_routing_bias".to_string(), serde_json::json!(true));
        cfg.extra
            .insert("scoring_func".to_string(), serde_json::json!("sigmoid"));
        cfg
    }

    fn test_m2_config_partial_rotary() -> ModelConfig {
        let mut cfg = test_m2_config();
        // head_dim=16, rotary_dim=8 -> partial RoPE
        cfg.extra
            .insert("rotary_dim".to_string(), serde_json::json!(8));
        cfg
    }

    fn create_cache(cfg: &ModelConfig) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_parsing_defaults() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert_eq!(m2_cfg.num_local_experts, 4);
        assert_eq!(m2_cfg.num_experts_per_tok, 2);
        assert_eq!(m2_cfg.rotary_dim, 16);
        assert!(!m2_cfg.use_routing_bias);
        assert_eq!(m2_cfg.scoring_func, "softmax");
        assert!(!m2_cfg.attention_bias);
    }

    #[test]
    fn test_config_parsing_with_bias() {
        let cfg = test_m2_config_with_bias();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert!(m2_cfg.use_routing_bias);
        assert_eq!(m2_cfg.scoring_func, "sigmoid");
    }

    #[test]
    fn test_config_partial_rotary() {
        let cfg = test_m2_config_partial_rotary();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);

        assert_eq!(m2_cfg.rotary_dim, 8);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_construction() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxM2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxM2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_construction_partial_rotary() {
        let cfg = test_m2_config_partial_rotary();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxM2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxM2ForCausalLM with partial rotary should construct: {:?}",
            model.err()
        );
    }

    // ─── Component Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_moe_forward() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let moe = MiniMaxM2MoE::new(&cfg, &m2_cfg, vb.pp("moe")).expect("moe");
        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_attention_qk_norm() {
        let cfg = test_m2_config();
        let m2_cfg = MiniMaxM2Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Smoke test: attention with fused QKV + QK-norm + (optional) partial
        // RoPE builds with this config. End-to-end behavior is exercised by
        // the forward tests below.
        MiniMaxM2Attention::new(&cfg, &m2_cfg, vb.pp("attn")).expect("attn");
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_forward_shape() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let batch_size = 1;
        let seq_len = 3;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits = model
            .forward(&next, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_model_forward_trait() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_device() {
        let cfg = test_m2_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        assert!(matches!(ModelForward::device(&model), Device::Cpu));
    }

    #[test]
    fn test_partial_rotary_forward() {
        let cfg = test_m2_config_partial_rotary();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxM2ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward with partial rotary");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
