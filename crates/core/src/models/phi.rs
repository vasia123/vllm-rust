//! Phi model implementation (Phi-1.5, Phi-2).
//!
//! Key differences from Llama:
//! - LayerNorm instead of RMSNorm
//! - Parallel attention-MLP (attention + mlp run in parallel, not sequentially)
//! - Bias in projections (q, k, v, o, fc1, fc2, lm_head)
//! - Single input_layernorm (no post_attention_layernorm)
//!
//! Reference: https://huggingface.co/microsoft/phi-1_5

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBias, AttentionBlock, AttentionConfig, ProjNames};
use crate::layers::RotaryEmbedding;

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ── MLP ──────────────────────────────────────────────────────────────────────

/// Phi MLP with GELU activation.
/// Uses fc1 and fc2 (not gate/up/down like SwiGLU).
struct PhiMlp {
    fc1: TpLinear,
    fc2: TpLinear,
}

impl PhiMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Phi uses bias in MLP layers
        let fc1 = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            true,  // bias
            false, // no gather
            vb.pp("fc1"),
            pg,
        )?;
        let fc2 = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            true, // bias
            true, // input is parallel
            vb.pp("fc2"),
            pg,
        )?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let xs = self.fc1.forward(xs, tp_ctx)?;
        let xs = xs.gelu_erf()?; // Phi uses GELU activation
        self.fc2.forward(&xs, tp_ctx)
    }
}

// ── Attention ────────────────────────────────────────────────────────────────

// Phi (Phi1/Phi2) = vanilla GQA + bias on ALL projections + 'dense' output proj.
struct PhiAttention {
    inner: AttentionBlock,
}

impl PhiAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_bias(AttentionBias::ALL)
        .with_proj_names(ProjNames {
            o: "dense",
            ..Default::default()
        });
        let rotary_emb = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        let inner = AttentionBlock::new(&attn_cfg, vb, pg, rotary_emb)?;
        Ok(Self { inner })
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
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner.forward(
            xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            tp_ctx,
        )
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner
            .forward_decode_batch(xs, sequences, cache_engine, tp_ctx)
    }
}

// ── Decoder Layer ────────────────────────────────────────────────────────────

struct PhiDecoderLayer {
    self_attn: PhiAttention,
    mlp: PhiMlp,
    input_layernorm: LayerNorm, // Phi uses LayerNorm, not RMSNorm
                                // NOTE: Phi has no post_attention_layernorm - uses parallel attn+mlp
}

impl PhiDecoderLayer {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let self_attn = PhiAttention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;

        // Phi intermediate size: default is 4 * hidden_size if not specified
        let intermediate_size = if cfg.intermediate_size > 0 {
            cfg.intermediate_size
        } else {
            4 * cfg.hidden_size
        };
        let mlp = PhiMlp::new(cfg.hidden_size, intermediate_size, vb.pp("mlp"), pg)?;

        // Phi uses layer_norm_eps (typically 1e-5)
        let layer_norm_eps = cfg.rms_norm_eps; // reuse rms_norm_eps field
        let input_layernorm =
            layer_norm(cfg.hidden_size, layer_norm_eps, vb.pp("input_layernorm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
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
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;

        // Single layernorm before both attention and MLP (parallel design)
        let xs = self.input_layernorm.forward(xs)?;

        // Attention output
        let attn_output = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // MLP output (from same normalized input)
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;

        // Parallel: residual + attention + mlp
        (residual + attn_output + mlp_output)?.contiguous()
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;

        (residual + attn_output + mlp_output)?.contiguous()
    }
}

// ── Model ────────────────────────────────────────────────────────────────────

pub struct PhiForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<PhiDecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: TpLinear, // Phi uses bias in lm_head
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl PhiForCausalLM {
    /// Create a new Phi model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Phi model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(PhiDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let layer_norm_eps = cfg.rms_norm_eps;
        let final_layernorm =
            layer_norm(cfg.hidden_size, layer_norm_eps, vb_m.pp("final_layernorm"))?;

        // Phi uses bias in lm_head (unlike Llama)
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
                true, // Phi lm_head has bias
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
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

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
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
        let xs = self.final_layernorm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for PhiForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        PhiForCausalLM::forward(
            self,
            input_ids,
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

        let xs = self.final_layernorm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["PhiForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // Phi typically uses MHA, not GQA
            num_hidden_layers: 2,
            intermediate_size: 256, // 4 * hidden_size
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5, // Phi uses layer_norm_eps
            rope_theta: 10000.0,
            tie_word_embeddings: false, // Phi doesn't tie embeddings
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(true), // Phi uses bias
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
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

    #[test]
    fn test_phi_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = PhiForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "PhiForCausalLM should construct with zero weights"
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_phi_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PhiForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
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

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_phi_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PhiForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_phi_mha_configuration() {
        // Phi typically uses MHA (all heads equal), not GQA
        let cfg = test_config();
        assert_eq!(
            cfg.num_attention_heads, cfg.num_key_value_heads,
            "Phi uses MHA, not GQA"
        );
    }

    #[test]
    fn test_phi_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = PhiForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode step with seqlen_offset=3
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = model
            .forward(
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
    fn test_phi_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = PhiForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "PhiForCausalLM should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }
}
