use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBlock, AttentionConfig};
use crate::layers::RotaryEmbedding;

// Re-export for public API
pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpGeGluMlp, TpLinear};

// ─── Gemma2 RMSNorm ──────────────────────────────────────────────────────────
//
// Same as Gemma: output = x * (1 + weight) / rms(x)

type Gemma2RmsNorm = crate::layers::RmsNorm;

#[inline]
fn gemma2_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<Gemma2RmsNorm> {
    crate::layers::rms_norm_gemma(size, eps, vb)
}

// ─── Soft Capping ────────────────────────────────────────────────────────────
//
// Gemma2 uses soft capping: cap * tanh(x / cap)
// This prevents extreme logits while preserving gradients

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

// ─── Attention ───────────────────────────────────────────────────────────────
//
// Gemma2 = vanilla GQA + attention-logit softcap + per-layer alternating
// sliding-window (even = sliding, odd = global) + custom q-scale derived
// from `query_pre_attn_scalar`. All four axes are expressed declaratively
// on `AttentionConfig`; the bespoke struct is now a thin shim over
// `AttentionBlock`.

struct Gemma2Attention {
    inner: AttentionBlock,
}

impl Gemma2Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;

        // Gemma2: q-scale = 1 / sqrt(query_pre_attn_scalar). Default
        // query_pre_attn_scalar = sqrt(head_dim), so default scale becomes
        // 1 / head_dim^(1/4) — distinct from the AttentionBlock default
        // 1 / sqrt(head_dim). Always set scale explicitly.
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or((head_dim as f64).sqrt());
        let scale = 1.0 / query_pre_attn_scalar.sqrt();

        // Even layers attend within a sliding window; odd layers are global.
        let sliding_window = if layer_idx.is_multiple_of(2) {
            cfg.sliding_window
        } else {
            None
        };

        let attn_logit_softcap = cfg
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        let mut attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            head_dim,
            cfg.hidden_size,
        )
        .with_scale(scale);
        if let Some(cap) = attn_logit_softcap {
            attn_cfg = attn_cfg.with_softcap(cap);
        }
        if let Some(window) = sliding_window {
            attn_cfg = attn_cfg.with_sliding_window(window);
        }

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
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

// ─── Decoder Layer ───────────────────────────────────────────────────────────
//
// Gemma2 has additional layernorms: pre_feedforward and post_feedforward

struct Gemma2DecoderLayer {
    self_attn: Gemma2Attention,
    mlp: TpGeGluMlp,
    input_layernorm: Gemma2RmsNorm,
    post_attention_layernorm: Gemma2RmsNorm,
    pre_feedforward_layernorm: Gemma2RmsNorm,
    post_feedforward_layernorm: Gemma2RmsNorm,
}

impl Gemma2DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::new_with_tp(cfg, layer_idx, vb.pp("self_attn"), pg)?;
        let mlp = TpGeGluMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;
        let input_layernorm =
            gemma2_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = gemma2_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = gemma2_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = gemma2_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
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
        // Pre-attention norm
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        // Attention
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // Post-attention norm (Gemma2 specific)
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // Residual connection
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        // Pre-feedforward norm (Gemma2 specific)
        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;

        // MLP
        let hidden_states = self.mlp.forward(&hidden_states, tp_ctx)?;

        // Post-feedforward norm (Gemma2 specific)
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        // Residual connection
        residual + hidden_states
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
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states, tp_ctx)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Gemma2ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Gemma2DecoderLayer>,
    norm: Gemma2RmsNorm,
    lm_head: TpLinear,
    hidden_size: usize,
    final_logit_softcap: Option<f64>,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Gemma2ForCausalLM {
    /// Create a new Gemma2 model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Gemma2 model with tensor parallelism.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for weight loading
    /// * `pg` - Process group for tensor parallelism
    /// * `tp_ctx` - Tensor parallelism context (holds communicator)
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
            layers.push(Gemma2DecoderLayer::new_with_tp(cfg, i, vb_l.pp(i), pg)?);
        }

        let norm = gemma2_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // LM head: output projection to vocabulary
        // Gemma2 always uses tied embeddings
        let lm_head = if world_size == 1 {
            // Single GPU with tied embeddings: reuse embedding weights
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else {
            // TP with tied embeddings: load from embed_tokens path
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true, // gather output to get full vocab logits
                vb_m.pp("embed_tokens"),
                pg,
            )?
        };

        // Final logit soft capping
        let final_logit_softcap = cfg
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            final_logit_softcap,
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

        // Gemma2 normalizes embeddings by sqrt(hidden_size)
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

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
        let mut logits = self.lm_head.forward(&xs, &self.tp_ctx)?;

        // Apply final logit soft capping
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the TP context.
    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Gemma2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Gemma2ForCausalLM::forward(
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
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

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
        let mut logits = self.lm_head.forward(&xs, &self.tp_ctx)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
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
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(256.0).unwrap()),
        );
        extra.insert(
            "attn_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(50.0).unwrap()),
        );
        extra.insert(
            "final_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(30.0).unwrap()),
        );

        ModelConfig {
            architectures: vec!["Gemma2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4, // Need 4 layers to test alternating sliding window
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: Some(256),
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

    #[test]
    fn test_gemma2_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma2ForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "Gemma2ForCausalLM should construct");

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_logit_softcap.is_some());
    }

    #[test]
    fn test_gemma2_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gemma2_soft_cap() {
        let device = Device::Cpu;

        // Test soft capping function
        let x = Tensor::from_vec(vec![-100.0f32, -10.0, 0.0, 10.0, 100.0], (5,), &device)
            .expect("input");

        let capped = soft_cap(&x, 30.0).expect("soft_cap");
        let capped_vec: Vec<f32> = capped.to_vec1().unwrap();

        // Values should be bounded by [-cap, cap]
        for &v in &capped_vec {
            assert!(v.abs() <= 30.0 + 0.01, "value {v} should be bounded by cap");
        }

        // 0 should stay 0
        assert!(capped_vec[2].abs() < 0.01, "soft_cap(0) should be ~0");

        // Large positive should approach cap
        assert!(capped_vec[4] > 29.0, "soft_cap(100) should be close to cap");
    }

    #[test]
    fn test_gemma2_alternating_sliding_window() {
        let cfg = test_config();

        // Verify even layers have sliding window, odd layers don't
        assert!(cfg.sliding_window.is_some());

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

        // Model should have correct number of layers
        assert_eq!(model.layers.len(), 4);
    }

    #[test]
    fn test_gemma2_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
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

        // Decode
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
    fn test_gemma2_extra_layernorms() {
        // Verify Gemma2 has 4 layernorms per layer
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // This tests that the layer constructs with all 4 norms
        let pg = LocalProcessGroup::new();
        let layer = Gemma2DecoderLayer::new_with_tp(&cfg, 0, vb, &pg).expect("build layer");

        // Forward should work with the extra norms
        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let x = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("input");
        let tp_ctx = TpContext::single_gpu();
        let result = layer.forward(
            &x,
            None,
            0,
            &mut kv_cache_mgr,
            0,
            &block_table,
            &slot_mapping,
            &tp_ctx,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemma2_query_pre_attn_scalar() {
        // Verify custom scaling is used
        let cfg = test_config();
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap();

        assert_eq!(query_pre_attn_scalar, 256.0);
    }

    // ─── Tensor Parallelism Tests ────────────────────────────────────────────────

    fn test_config_tp_compatible() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "query_pre_attn_scalar".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(256.0).unwrap()),
        );
        extra.insert(
            "attn_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(50.0).unwrap()),
        );
        extra.insert(
            "final_logit_softcapping".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(30.0).unwrap()),
        );

        // Config with heads divisible by 2 for TP=2 testing
        ModelConfig {
            architectures: vec!["Gemma2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4, // divisible by 2
            num_key_value_heads: 2, // divisible by 2
            num_hidden_layers: 4,   // Need 4 layers for alternating sliding window
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_gemma2_tp_construction_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Simulate TP with world_size=2 (ProcessGroup and TpContext must match)
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Gemma2ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Gemma2ForCausalLM should construct with TP=2: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_gemma2_tp_forward_world_size_2() {
        let cfg = test_config_tp_compatible();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Create model with TP=2 simulation (ProcessGroup and TpContext must match)
        let tp_size = 2;
        let pg = LocalProcessGroup::with_rank(0, tp_size);
        let tp_ctx = TpContext::mock_multi_gpu(0, tp_size);
        let model = Gemma2ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx).expect("build model");

        // Create cache with LOCAL kv_heads (divided by tp_size)
        let local_kv_heads = cfg.num_key_value_heads / tp_size;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: local_kv_heads, // Important: local heads, not global
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

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

        // With TP=2 and MockCommunicator's all_gather simulation,
        // the output should be gathered to full vocab_size
        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_gemma2_tp_heads_divisibility_check() {
        // Test that TP fails with error when heads aren't divisible
        let mut cfg = test_config();
        cfg.num_key_value_heads = 3; // Not divisible by 2

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Try TP=2 with 3 kv_heads - should return error
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = Gemma2ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);

        match result {
            Ok(_) => panic!("Should fail when num_kv_heads is not divisible by world_size"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("divisible"),
                    "Error should mention divisibility: {}",
                    err_msg
                );
            }
        }
    }
}
