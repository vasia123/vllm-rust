//! ExaoneMoE model implementation (LG AI Research K-EXAONE).
//!
//! ExaoneMoE is a Mixture of Experts variant of Exaone with:
//! - Q/K normalization (per-head RMSNorm, same as Exaone4)
//! - Pre-LN architecture (unlike Exaone4's post-LN)
//! - Selective MoE layers via `is_moe_layer` config list
//! - Grouped top-k routing with sigmoid scoring
//! - Shared experts with SwiGLU MLP
//! - Routed scaling factor

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding, SwiGluMlp};
use crate::moe::{MoELayerWithShared, MoELayerWithSharedConfig, ScoringFunc};

// ---- ExaoneMoE-specific config parsing ------------------------------------

struct ExaoneMoeConfig {
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    num_shared_experts: usize,
    is_moe_layer: Vec<bool>,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl ExaoneMoeConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_experts = cfg
            .extra
            .get("num_experts")
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

        let num_shared_experts = cfg
            .extra
            .get("num_shared_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        // is_moe_layer: list of booleans, one per layer
        let is_moe_layer = cfg
            .extra
            .get("is_moe_layer")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
            .unwrap_or_else(|| vec![true; cfg.num_hidden_layers]);

        let n_group = cfg.n_group();
        let topk_group = cfg.topk_group();
        let norm_topk_prob = cfg.norm_topk_prob();

        let routed_scaling_factor = cfg.routed_scaling_factor();

        Self {
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            num_shared_experts,
            is_moe_layer,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_moe_layer.get(layer_idx).copied().unwrap_or(false)
    }
}

// ---- Attention (Q/K norm, RoPE) --------------------------------------------

struct ExaoneMoeAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl ExaoneMoeAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rope_theta = cfg
            .extra
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn apply_qk_norm(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_shape = q.dims().to_vec();
        let k_shape = k.dims().to_vec();

        let mut q_new_shape: Vec<usize> = q_shape[..q_shape.len() - 1].to_vec();
        q_new_shape.extend_from_slice(&[self.num_heads, self.head_dim]);
        let q_normed = self.q_norm.forward(&q.reshape(q_new_shape.as_slice())?)?;
        let q_flat = q_normed.reshape(q_shape)?;

        let mut k_new_shape: Vec<usize> = k_shape[..k_shape.len() - 1].to_vec();
        k_new_shape.extend_from_slice(&[self.num_kv_heads, self.head_dim]);
        let k_normed = self.k_norm.forward(&k.reshape(k_new_shape.as_slice())?)?;
        let k_flat = k_normed.reshape(k_shape)?;

        Ok((q_flat, k_flat))
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
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (q, k) = self.apply_qk_norm(&q, &k)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        let attn_output = paged_attention(
            &q,
            &k,
            &v,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (q, k) = self.apply_qk_norm(&q, &k)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

            let attn_out = paged_attention(
                &q_i,
                &k_i,
                &v_i,
                None,
                seq.seqlen_offset,
                cache_engine,
                &seq.block_ids,
                &seq.slot_mapping,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?;
            outputs.push(attn_out);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output)
    }
}

// ---- FFN Variant (dense SwiGLU MLP or MoE) --------------------------------

enum FfnVariant {
    Dense(SwiGluMlp),
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

// ---- Decoder Layer (Pre-LN) ------------------------------------------------

struct ExaoneMoeDecoderLayer {
    self_attn: ExaoneMoeAttention,
    ffn: FfnVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl ExaoneMoeDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        moe_cfg: &ExaoneMoeConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = ExaoneMoeAttention::new(cfg, vb.pp("self_attn"))?;

        let ffn = if moe_cfg.is_moe_layer(layer_idx) {
            let shared_intermediate = if moe_cfg.num_shared_experts > 0 {
                Some(moe_cfg.moe_intermediate_size * moe_cfg.num_shared_experts)
            } else {
                None
            };

            let moe_config = MoELayerWithSharedConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: moe_cfg.moe_intermediate_size,
                shared_expert_intermediate_size: shared_intermediate,
                num_experts: moe_cfg.num_experts,
                top_k: moe_cfg.num_experts_per_tok,
                renormalize: moe_cfg.norm_topk_prob,
                scoring_func: ScoringFunc::Sigmoid,
                routed_scaling_factor: moe_cfg.routed_scaling_factor,
                gated_shared_expert: false,
                use_grouped_topk: moe_cfg.n_group.is_some(),
                num_expert_groups: moe_cfg.n_group,
                topk_per_group: moe_cfg.topk_group,
                inplace: false,
                is_act_and_mul: true,
            };
            FfnVariant::MoE(Box::new(MoELayerWithShared::new(moe_config, vb.pp("mlp"))?))
        } else {
            FfnVariant::Dense(SwiGluMlp::new(
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
        // Pre-LN: norm -> attention -> residual add
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

        // Pre-LN: norm -> FFN -> residual add
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

/// ExaoneMoE model for causal language modeling.
///
/// Based on Exaone4 attention (Q/K norm, RoPE) with selective MoE layers,
/// grouped top-k routing, sigmoid scoring, and shared experts.
pub struct ExaoneMoeForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<ExaoneMoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl ExaoneMoeForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let moe_cfg = ExaoneMoeConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ExaoneMoeDecoderLayer::new(cfg, &moe_cfg, i, vb_l.pp(i))?);
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

impl crate::engine::ModelForward for ExaoneMoeForCausalLM {
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
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("num_shared_experts".to_string(), serde_json::json!(1));
        extra.insert("n_group".to_string(), serde_json::json!(2));
        extra.insert("topk_group".to_string(), serde_json::json!(1));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("routed_scaling_factor".to_string(), serde_json::json!(1.0));
        // Layer 0 = dense, Layer 1 = MoE
        extra.insert("is_moe_layer".to_string(), serde_json::json!([false, true]));
        extra.insert("rope_theta".to_string(), serde_json::json!(1_000_000.0));

        ModelConfig {
            architectures: vec!["ExaoneMoeForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
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
    fn test_exaone_moe_config_parsing() {
        let cfg = test_config();
        let moe_cfg = ExaoneMoeConfig::from_model_config(&cfg);
        assert_eq!(moe_cfg.num_experts, 4);
        assert_eq!(moe_cfg.num_experts_per_tok, 2);
        assert_eq!(moe_cfg.moe_intermediate_size, 64);
        assert_eq!(moe_cfg.num_shared_experts, 1);
        assert_eq!(moe_cfg.n_group, Some(2));
        assert_eq!(moe_cfg.topk_group, Some(1));
        assert!(moe_cfg.norm_topk_prob);
        assert_eq!(moe_cfg.routed_scaling_factor, 1.0);
    }

    #[test]
    fn test_exaone_moe_config_defaults() {
        let cfg = ModelConfig::default();
        let moe_cfg = ExaoneMoeConfig::from_model_config(&cfg);
        assert_eq!(moe_cfg.num_experts, 64);
        assert_eq!(moe_cfg.num_experts_per_tok, 6);
        assert_eq!(moe_cfg.num_shared_experts, 0);
        // Default: all layers are MoE
        assert!(moe_cfg.is_moe_layer(0));
    }

    #[test]
    fn test_exaone_moe_is_moe_layer() {
        let cfg = test_config();
        let moe_cfg = ExaoneMoeConfig::from_model_config(&cfg);

        assert!(!moe_cfg.is_moe_layer(0)); // false per config
        assert!(moe_cfg.is_moe_layer(1)); // true per config
        assert!(!moe_cfg.is_moe_layer(2)); // out of bounds = false
    }

    // ---- Construction Tests -------------------------------------------------

    #[test]
    fn test_exaone_moe_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ExaoneMoeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_exaone_moe_mixed_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb).unwrap();

        // Layer 0 = dense, Layer 1 = MoE
        assert!(matches!(model.layers[0].ffn, FfnVariant::Dense(_)));
        assert!(matches!(model.layers[1].ffn, FfnVariant::MoE(_)));
    }

    #[test]
    fn test_exaone_moe_all_moe() {
        let mut cfg = test_config();
        cfg.extra
            .insert("is_moe_layer".to_string(), serde_json::json!([true, true]));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb).unwrap();
        for layer in &model.layers {
            assert!(matches!(layer.ffn, FfnVariant::MoE(_)));
        }
    }

    #[test]
    fn test_exaone_moe_all_dense() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "is_moe_layer".to_string(),
            serde_json::json!([false, false]),
        );
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb).unwrap();
        for layer in &model.layers {
            assert!(matches!(layer.ffn, FfnVariant::Dense(_)));
        }
    }

    #[test]
    fn test_exaone_moe_attention_qk_norm() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let attn = ExaoneMoeAttention::new(&cfg, vb.pp("self_attn")).unwrap();
        assert_eq!(attn.num_heads, cfg.num_attention_heads);
        assert_eq!(attn.num_kv_heads, cfg.num_key_value_heads);
        assert_eq!(attn.head_dim, cfg.head_dim);
    }

    #[test]
    fn test_exaone_moe_qk_norm_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let attn = ExaoneMoeAttention::new(&cfg, vb.pp("self_attn")).unwrap();

        let batch = 2;
        let seq = 3;
        let q = Tensor::zeros(
            (batch, seq, cfg.num_attention_heads * cfg.head_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let k = Tensor::zeros(
            (batch, seq, cfg.num_key_value_heads * cfg.head_dim),
            DType::F32,
            &device,
        )
        .unwrap();

        let (q_normed, k_normed) = attn.apply_qk_norm(&q, &k).unwrap();
        assert_eq!(q_normed.dims(), q.dims());
        assert_eq!(k_normed.dims(), k.dims());
    }

    // ---- Forward Tests ------------------------------------------------------

    #[test]
    fn test_exaone_moe_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ExaoneMoeForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
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
    fn test_exaone_moe_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ExaoneMoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_exaone_moe_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ExaoneMoeForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_exaone_moe_no_shared_experts() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_shared_experts".to_string(), serde_json::json!(0));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ExaoneMoE without shared experts should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_exaone_moe_with_tied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = true;
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ExaoneMoeForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ExaoneMoE with tied embeddings should construct: {:?}",
            model.err()
        );
    }
}
