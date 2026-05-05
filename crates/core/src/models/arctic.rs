//! Snowflake Arctic model implementation.
//!
//! Arctic is a hybrid dense-MoE model that alternates between standard MLP layers
//! and MoE layers. MoE layers optionally include a parallel residual MLP path
//! that is added to the MoE output before the residual connection.
//!
//! Key features:
//! - Alternating dense/MoE layers controlled by `moe_layer_frequency`
//! - Optional residual MLP parallel to MoE (controlled by `use_residual`)
//! - Standard RoPE, GQA, SwiGLU, RMSNorm

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBlock, AttentionConfig};
use crate::layers::{RotaryEmbedding, SwiGluMlp};

use super::tp_layers::TpContext;
use crate::moe::MoELayer;
use crate::moe::MoELayerConfig;

// ─── Arctic Config ──────────────────────────────────────────────────────────

struct ArcticConfig {
    num_local_experts: usize,
    num_experts_per_tok: usize,
    moe_layer_frequency: usize,
    use_residual: bool,
}

impl ArcticConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let moe_layer_frequency = cfg
            .extra
            .get("moe_layer_frequency")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let use_residual = cfg
            .extra
            .get("use_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            num_local_experts,
            num_experts_per_tok,
            moe_layer_frequency,
            use_residual,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_layer_frequency > 0 && (layer_idx + 1).is_multiple_of(self.moe_layer_frequency)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

// Arctic = vanilla GQA + RoPE, no bias, no QK norm.
struct ArcticAttention {
    inner: AttentionBlock,
    tp_ctx: TpContext,
}

impl ArcticAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        );
        let rotary_emb = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        let pg = LocalProcessGroup::new();
        let inner = AttentionBlock::new(&attn_cfg, vb, &pg, rotary_emb)?;
        Ok(Self {
            inner,
            tp_ctx: TpContext::single_gpu(),
        })
    }

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

// ─── Feed-Forward (Dense or MoE) ───────────────────────────────────────────

enum ArcticFeedForward {
    Dense(SwiGluMlp),
    Moe(MoELayer),
}

impl ArcticFeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct ArcticDecoderLayer {
    self_attn: ArcticAttention,
    feed_forward: ArcticFeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_layernorm: Option<RmsNorm>,
    residual_mlp: Option<SwiGluMlp>,
}

impl ArcticDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        arctic_cfg: &ArcticConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = ArcticAttention::new(cfg, vb.pp("self_attn"))?;

        let is_moe = arctic_cfg.is_moe_layer(layer_idx);

        let feed_forward = if is_moe {
            let moe_cfg = MoELayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size,
                num_experts: arctic_cfg.num_local_experts,
                top_k: arctic_cfg.num_experts_per_tok,
                renormalize: arctic_cfg.num_experts_per_tok > 1,
                inplace: true,
                is_act_and_mul: true,
            };
            ArcticFeedForward::Moe(MoELayer::new(moe_cfg, vb.pp("block_sparse_moe"))?)
        } else {
            let mlp = SwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("block_sparse_moe").pp("mlp"),
            )?;
            ArcticFeedForward::Dense(mlp)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        // Residual MLP only on MoE layers when use_residual is true
        let (residual_layernorm, residual_mlp) = if is_moe && arctic_cfg.use_residual {
            let rln = rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("residual_layernorm"),
            )?;
            // Residual MLP uses hidden_size as intermediate (smaller parallel path)
            let rmlp = SwiGluMlp::new(cfg.hidden_size, cfg.hidden_size, vb.pp("residual_mlp"))?;
            (Some(rln), Some(rmlp))
        } else {
            (None, None)
        };

        Ok(Self {
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
            residual_layernorm,
            residual_mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Pre-attention norm + attention + residual
        let residual_input = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward(
            &hidden,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let residual_attn = (residual_input + &hidden)?;

        if let (Some(ref res_ln), Some(ref res_mlp)) =
            (&self.residual_layernorm, &self.residual_mlp)
        {
            // MoE layer with residual path:
            // residual_mlp runs on post-attention hidden states
            // MoE runs on pre-attention residual (residual_input)
            let res_hidden = res_ln.forward(&residual_attn)?;
            let residual_mlp_out = res_mlp.forward(&res_hidden)?;
            let moe_input = self.post_attention_layernorm.forward(residual_input)?;
            let moe_out = self.feed_forward.forward(&moe_input)?;
            // Combine: residual_attn + residual_mlp_out + moe_out
            let combined = (residual_mlp_out + moe_out)?;
            &residual_attn + &combined
        } else {
            // Standard path (dense or MoE without residual)
            let hidden = self.post_attention_layernorm.forward(&residual_attn)?;
            let hidden = self.feed_forward.forward(&hidden)?;
            &residual_attn + &hidden
        }
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual_input = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward_decode_batch(
            &hidden,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let residual_attn = (residual_input + &hidden)?;

        if let (Some(ref res_ln), Some(ref res_mlp)) =
            (&self.residual_layernorm, &self.residual_mlp)
        {
            let res_hidden = res_ln.forward(&residual_attn)?;
            let residual_mlp_out = res_mlp.forward(&res_hidden)?;
            let moe_input = self.post_attention_layernorm.forward(residual_input)?;
            let moe_out = self.feed_forward.forward(&moe_input)?;
            let combined = (residual_mlp_out + moe_out)?;
            &residual_attn + &combined
        } else {
            let hidden = self.post_attention_layernorm.forward(&residual_attn)?;
            let hidden = self.feed_forward.forward(&hidden)?;
            &residual_attn + &hidden
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct ArcticForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<ArcticDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl ArcticForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let arctic_cfg = ArcticConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ArcticDecoderLayer::new(cfg, &arctic_cfg, i, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
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

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for ArcticForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
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
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        xs = self.norm.forward(&xs)?;
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
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".into(), serde_json::Value::from(4));
        extra.insert("num_experts_per_tok".into(), serde_json::Value::from(2));
        extra.insert("moe_layer_frequency".into(), serde_json::Value::from(2));
        extra.insert("use_residual".into(), serde_json::Value::from(true));

        ModelConfig {
            architectures: vec!["ArcticForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_arctic_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ArcticForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ArcticForCausalLM should construct: {:?}",
            model.err()
        );
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_arctic_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ArcticForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");

        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward pass");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_arctic_layer_types() {
        let cfg = ArcticConfig {
            num_local_experts: 128,
            num_experts_per_tok: 2,
            moe_layer_frequency: 2,
            use_residual: true,
        };
        // Layer 0: (0+1)%2 = 1 → not MoE (dense)
        assert!(!cfg.is_moe_layer(0));
        // Layer 1: (1+1)%2 = 0 → MoE
        assert!(cfg.is_moe_layer(1));
        // Layer 2: (2+1)%2 = 1 → not MoE
        assert!(!cfg.is_moe_layer(2));
        // Layer 3: (3+1)%2 = 0 → MoE
        assert!(cfg.is_moe_layer(3));
    }
}
