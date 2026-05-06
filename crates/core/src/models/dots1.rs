//! Dots1 model implementation (rednote-hilab).
//!
//! Dots1 is a Mixture of Experts model with:
//! - SharedMoE: topk routing + shared expert with routed_scaling_factor
//! - Q/K normalization layers (per-head RMSNorm)
//! - Merged QKV projection
//! - SiLU activation, RoPE, RMSNorm
//! - Configurable MoE layers via first_k_dense_replace and moe_layer_freq
//! - Grouped top-k expert selection with sigmoid/softmax scoring

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBias, AttentionBlock, AttentionConfig, QkNormVariant};
use crate::layers::RotaryEmbedding;
use crate::moe::{MoELayerWithShared, MoELayerWithSharedConfig, ScoringFunc};

use super::tp_layers::TpContext;

// ---- Dots1-specific config parsing ----------------------------------------

struct Dots1Config {
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: Option<usize>,
    first_k_dense_replace: usize,
    moe_layer_freq: usize,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    norm_topk_prob: bool,
    scoring_func: ScoringFunc,
    routed_scaling_factor: f64,
    attention_bias: bool,
}

impl Dots1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let n_routed_experts = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(6);

        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);

        let n_shared_experts = cfg.n_shared_experts();

        let first_k_dense_replace = cfg.first_k_dense_replace().unwrap_or(0);

        let moe_layer_freq = cfg
            .extra
            .get("moe_layer_freq")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let n_group = cfg.n_group();
        let topk_group = cfg.topk_group();
        let norm_topk_prob = cfg.norm_topk_prob();

        let scoring_func = cfg
            .extra
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "sigmoid" => ScoringFunc::Sigmoid,
                _ => ScoringFunc::Softmax,
            })
            .unwrap_or(ScoringFunc::Softmax);

        let routed_scaling_factor = cfg.routed_scaling_factor();

        let attention_bias = cfg.attention_bias.unwrap_or(false);

        Self {
            n_routed_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            n_shared_experts,
            first_k_dense_replace,
            moe_layer_freq,
            n_group,
            topk_group,
            norm_topk_prob,
            scoring_func,
            routed_scaling_factor,
            attention_bias,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_k_dense_replace && layer_idx.is_multiple_of(self.moe_layer_freq)
    }
}

// ---- MLP (gate_up merged, SiLU) -------------------------------------------

// Dots1's dense MLP is the canonical fused-gate-up SwiGLU pattern
// without bias. Use the shared `crate::layers::FusedSwiGluMlp`.
type Dots1MLP = crate::layers::FusedSwiGluMlp;

// ---- Attention (merged QKV, Q/K norm, RoPE) --------------------------------
//
// Dots1 = fused QKV linear (optional bias) + per-head RMSNorm on Q/K
// (head_dim) + RoPE. o_proj has no bias. Bias is uniform across QKV
// (single fused weight), matching `AttentionBias::QKV_ONLY`.

struct Dots1Attention {
    inner: AttentionBlock,
    tp_ctx: TpContext,
}

impl Dots1Attention {
    fn new(cfg: &ModelConfig, dots1_cfg: &Dots1Config, vb: VarBuilder) -> Result<Self> {
        let bias = if dots1_cfg.attention_bias {
            AttentionBias::QKV_ONLY
        } else {
            AttentionBias::NONE
        };

        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused()
        .with_bias(bias)
        .with_qk_norm(QkNormVariant::PerHead, cfg.rms_norm_eps);

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

// ---- FFN Variant (dense MLP or SharedMoE) ----------------------------------

enum FfnVariant {
    Dense(Dots1MLP),
    MoE(Box<MoELayerWithShared>),
}

impl FfnVariant {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            FfnVariant::Dense(mlp) => mlp.forward(xs),
            FfnVariant::MoE(moe) => moe.forward(xs),
        }
    }
}

// ---- Decoder Layer ---------------------------------------------------------

struct Dots1DecoderLayer {
    self_attn: Dots1Attention,
    ffn: FfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Dots1DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        dots1_cfg: &Dots1Config,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Dots1Attention::new(cfg, dots1_cfg, vb.pp("self_attn"))?;

        let ffn = if dots1_cfg.is_moe_layer(layer_idx) {
            let shared_intermediate = dots1_cfg
                .n_shared_experts
                .map(|n| n * dots1_cfg.moe_intermediate_size);

            let moe_config = MoELayerWithSharedConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: dots1_cfg.moe_intermediate_size,
                shared_expert_intermediate_size: shared_intermediate,
                num_experts: dots1_cfg.n_routed_experts,
                top_k: dots1_cfg.num_experts_per_tok,
                renormalize: dots1_cfg.norm_topk_prob,
                scoring_func: dots1_cfg.scoring_func,
                routed_scaling_factor: dots1_cfg.routed_scaling_factor,
                gated_shared_expert: false,
                use_grouped_topk: dots1_cfg.n_group.is_some(),
                num_expert_groups: dots1_cfg.n_group,
                topk_per_group: dots1_cfg.topk_group,
                inplace: false,
                is_act_and_mul: true,
            };
            FfnVariant::MoE(Box::new(MoELayerWithShared::new(moe_config, vb.pp("mlp"))?))
        } else {
            FfnVariant::Dense(Dots1MLP::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            ffn,
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
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.ffn.forward(&xs)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.ffn.forward(&xs)?;
        residual + xs
    }
}

// ---- Model -----------------------------------------------------------------

/// Dots1 model for causal language modeling.
///
/// SharedMoE architecture with Q/K normalization, grouped top-k routing,
/// and optional shared experts with routed scaling factor.
pub struct Dots1ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<Dots1DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Dots1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let dots1_cfg = Dots1Config::from_model_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Dots1DecoderLayer::new(cfg, &dots1_cfg, i, vb_l.pp(i))?);
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

impl crate::engine::ModelForward for Dots1ForCausalLM {
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
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("n_routed_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("n_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("first_k_dense_replace".to_string(), serde_json::json!(1));
        extra.insert("moe_layer_freq".to_string(), serde_json::json!(1));
        extra.insert("n_group".to_string(), serde_json::json!(2));
        extra.insert("topk_group".to_string(), serde_json::json!(1));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("scoring_func".to_string(), serde_json::json!("softmax"));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["Dots1ForCausalLM".to_string()],
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ---- Config Parsing Tests -----------------------------------------------

    #[test]
    fn test_dots1_config_parsing() {
        let cfg = test_config();
        let dots1_cfg = Dots1Config::from_model_config(&cfg);
        assert_eq!(dots1_cfg.n_routed_experts, 4);
        assert_eq!(dots1_cfg.num_experts_per_tok, 2);
        assert_eq!(dots1_cfg.moe_intermediate_size, 64);
        assert_eq!(dots1_cfg.n_shared_experts, Some(1));
        assert_eq!(dots1_cfg.first_k_dense_replace, 1);
        assert_eq!(dots1_cfg.moe_layer_freq, 1);
        assert_eq!(dots1_cfg.n_group, Some(2));
        assert_eq!(dots1_cfg.topk_group, Some(1));
        assert!(dots1_cfg.norm_topk_prob);
        assert_eq!(dots1_cfg.routed_scaling_factor, 1.0);
        assert!(!dots1_cfg.attention_bias);
    }

    #[test]
    fn test_dots1_config_defaults() {
        let cfg = ModelConfig::default();
        let dots1_cfg = Dots1Config::from_model_config(&cfg);
        assert_eq!(dots1_cfg.n_routed_experts, 64);
        assert_eq!(dots1_cfg.num_experts_per_tok, 6);
        assert_eq!(dots1_cfg.first_k_dense_replace, 0);
        assert_eq!(dots1_cfg.moe_layer_freq, 1);
        assert!(dots1_cfg.n_shared_experts.is_none());
    }

    #[test]
    fn test_dots1_is_moe_layer() {
        let cfg = test_config();
        let dots1_cfg = Dots1Config::from_model_config(&cfg);

        // first_k_dense_replace=1, moe_layer_freq=1
        assert!(!dots1_cfg.is_moe_layer(0)); // layer 0 is dense
        assert!(dots1_cfg.is_moe_layer(1)); // layer 1+ are MoE
    }

    #[test]
    fn test_dots1_is_moe_layer_with_frequency() {
        let mut cfg = test_config();
        cfg.extra
            .insert("first_k_dense_replace".to_string(), serde_json::json!(0));
        cfg.extra
            .insert("moe_layer_freq".to_string(), serde_json::json!(2));
        let dots1_cfg = Dots1Config::from_model_config(&cfg);

        assert!(dots1_cfg.is_moe_layer(0)); // 0 % 2 == 0
        assert!(!dots1_cfg.is_moe_layer(1)); // 1 % 2 != 0
        assert!(dots1_cfg.is_moe_layer(2)); // 2 % 2 == 0
    }

    #[test]
    fn test_dots1_scoring_func_sigmoid() {
        let mut cfg = test_config();
        cfg.extra
            .insert("scoring_func".to_string(), serde_json::json!("sigmoid"));
        let dots1_cfg = Dots1Config::from_model_config(&cfg);
        assert!(matches!(dots1_cfg.scoring_func, ScoringFunc::Sigmoid));
    }

    // ---- Construction Tests -------------------------------------------------

    #[test]
    fn test_dots1_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Dots1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Dots1ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_dots1_mixed_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Dots1ForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 = dense (first_k_dense_replace=1), Layer 1 = MoE
        assert!(matches!(model.layers[0].ffn, FfnVariant::Dense(_)));
        assert!(matches!(model.layers[1].ffn, FfnVariant::MoE(_)));
    }

    #[test]
    fn test_dots1_all_dense_when_no_experts() {
        let mut cfg = test_config();
        cfg.extra.remove("n_routed_experts");
        // Set first_k_dense_replace above num_hidden_layers so all layers are dense
        cfg.extra
            .insert("first_k_dense_replace".to_string(), serde_json::json!(100));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Dots1ForCausalLM::new(&cfg, vb).unwrap();
        for layer in &model.layers {
            assert!(matches!(layer.ffn, FfnVariant::Dense(_)));
        }
    }

    #[test]
    fn test_dots1_attention_qk_norm() {
        let cfg = test_config();
        let dots1_cfg = Dots1Config::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Smoke test: attention with fused QKV + QK-norm builds with this
        // config. Head/projection shapes are enforced by AttentionBlock and
        // exercised end-to-end by the forward tests below.
        Dots1Attention::new(&cfg, &dots1_cfg, vb.pp("self_attn")).unwrap();
    }

    #[test]
    fn test_dots1_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let mlp = Dots1MLP::new(64, 128, vb.pp("mlp")).unwrap();
        let input = Tensor::zeros((2, 64), DType::F32, &device).unwrap();
        let output = mlp.forward(&input);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }

    // ---- Forward Tests ------------------------------------------------------

    #[test]
    fn test_dots1_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Dots1ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 3;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = crate::engine::ModelForward::forward(
            &model,
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
    fn test_dots1_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Dots1ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = crate::engine::ModelForward::forward(
            &model,
            &prompt,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = crate::engine::ModelForward::forward(
            &model,
            &next_token,
            3,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_dots1_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Dots1ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_dots1_with_attention_bias() {
        let mut cfg = test_config();
        cfg.attention_bias = Some(true);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Dots1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Dots1 with attention bias should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_dots1_with_routed_scaling() {
        let mut cfg = test_config();
        cfg.extra
            .insert("routed_scaling_factor".to_string(), serde_json::json!(2.5));
        let dots1_cfg = Dots1Config::from_model_config(&cfg);
        assert!((dots1_cfg.routed_scaling_factor - 2.5).abs() < 1e-9);
    }
}
