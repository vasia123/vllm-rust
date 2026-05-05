use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBlock, AttentionConfig};
use crate::layers::RotaryEmbedding;

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Fused SwiGLU MLP ───────────────────────────────────────────────────────
//
// Phi-3 fuses gate_proj and up_proj into a single gate_up_proj weight.
// After projection, we split the output: first half is gate, second is up.

struct Phi3FusedSwiGluMlp {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
    intermediate_size: usize,
}

impl Phi3FusedSwiGluMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // gate_up_proj: [hidden_size, 2 * intermediate_size] (fused)
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            false,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false,
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        // TP splits the fused dim, so local intermediate_size = intermediate_size / world_size
        let local_intermediate = intermediate_size / pg.world_size();

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: local_intermediate,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let fused = self.gate_up_proj.forward(xs, tp_ctx)?;
        let gate = fused.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(
            candle_core::D::Minus1,
            self.intermediate_size,
            self.intermediate_size,
        )?;
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────
//
// Phi-3 uses fused QKV projection instead of separate Q/K/V.

// Phi3 = vanilla GQA + fused qkv_proj + LongRoPE (su-scaled rotary).
struct Phi3Attention {
    inner: AttentionBlock,
}

impl Phi3Attention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused();
        let rotary_emb = Self::build_rotary_embedding(cfg, cfg.head_dim, vb.dtype(), vb.device())?;
        let inner = AttentionBlock::new(&attn_cfg, vb, pg, rotary_emb)?;
        Ok(Self { inner })
    }

    /// Build the rotary embedding, selecting su-scaled (LongRoPE) when
    /// the model config contains `rope_scaling` with type "su" or "longrope".
    fn build_rotary_embedding(
        cfg: &ModelConfig,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<RotaryEmbedding> {
        let rope_scaling = cfg.extra.get("rope_scaling");
        let rope_scaling = match rope_scaling {
            Some(serde_json::Value::Object(map)) => Some(map),
            _ => None,
        };

        if let Some(scaling) = rope_scaling {
            let rope_type = scaling
                .get("type")
                .or_else(|| scaling.get("rope_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("default");

            if rope_type == "su" || rope_type == "longrope" {
                let parse_factors = |key: &str| -> Result<Vec<f64>> {
                    let arr = scaling.get(key).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "rope_scaling.{key} is required for su/longrope"
                        ))
                    })?;
                    let arr = arr.as_array().ok_or_else(|| {
                        candle_core::Error::Msg(format!("rope_scaling.{key} must be an array"))
                    })?;
                    arr.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64().ok_or_else(|| {
                                candle_core::Error::Msg(format!(
                                    "rope_scaling.{key}[{i}] is not a number"
                                ))
                            })
                        })
                        .collect()
                };

                let short_factor = parse_factors("short_factor")?;
                let long_factor = parse_factors("long_factor")?;

                let original_max_pos = scaling
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(cfg.max_position_embeddings as u64)
                    as usize;

                return RotaryEmbedding::new_su_scaled(
                    head_dim,
                    cfg.max_position_embeddings,
                    original_max_pos,
                    cfg.rope_theta,
                    &short_factor,
                    &long_factor,
                    dtype,
                    device,
                );
            }
        }

        RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype,
            device,
        )
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

struct Phi3DecoderLayer {
    self_attn: Phi3Attention,
    mlp: Phi3FusedSwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Phi3DecoderLayer {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let self_attn = Phi3Attention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;
        let mlp =
            Phi3FusedSwiGluMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;
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
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
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
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Phi3ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Phi3DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Phi3ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

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
            layers.push(Phi3DecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
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
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
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
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    /// Embed text tokens (for VLM use — embed only, no layers).
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids, &self.tp_ctx)
    }

    /// Forward with pre-computed embeddings (for VLM use — skips embedding layer).
    pub fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let seq_len = embeddings.dim(1)?;
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

        let mut xs = embeddings.clone();
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

    /// Forward decode batch with pre-computed embeddings (for VLM use).
    pub fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = embeddings.clone();
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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Phi3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Phi3ForCausalLM::forward(
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

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
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

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["Phi3ForCausalLM".to_string()],
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
    fn test_phi3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Phi3ForCausalLM should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_phi3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_phi3_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_phi3_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    // Note: the explicit fused QKV split test was removed when Phi3Attention
    // was migrated to AttentionBlock; the splitting logic now lives in
    // AttentionBlock::project_qkv and is validated by the model's full forward
    // tests below.

    #[test]
    fn test_phi3_fused_mlp_split() {
        // Verify that fused gate_up_proj output is correctly split
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let mlp =
            Phi3FusedSwiGluMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), &pg)
                .expect("build mlp");

        assert_eq!(mlp.intermediate_size, cfg.intermediate_size);

        let tp_ctx = TpContext::single_gpu();
        let input = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx).expect("forward");
        assert_eq!(output.dims(), &[1, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_phi3_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_phi3_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_phi3_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_phi3_gqa_configuration() {
        let cfg = test_config();
        let gqa_groups = cfg.num_attention_heads / cfg.num_key_value_heads;

        assert_eq!(gqa_groups, 2, "test config uses GQA with 2 groups");
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 2);
    }

    // ─── su-scaled RoPE tests ──────────────────────────────────────────

    fn su_rope_config() -> crate::config::ModelConfig {
        let half_dim = 8; // head_dim / 2
        let short_factor: Vec<serde_json::Value> =
            vec![1.0; half_dim].into_iter().map(|v| v.into()).collect();
        let long_factor: Vec<serde_json::Value> =
            vec![2.0; half_dim].into_iter().map(|v| v.into()).collect();

        let mut rope_scaling = serde_json::Map::new();
        rope_scaling.insert("type".into(), "su".into());
        rope_scaling.insert("short_factor".into(), short_factor.into());
        rope_scaling.insert("long_factor".into(), long_factor.into());
        rope_scaling.insert("original_max_position_embeddings".into(), 256.into());

        let mut cfg = test_config();
        cfg.max_position_embeddings = 512;
        cfg.extra.insert("rope_scaling".into(), rope_scaling.into());
        cfg
    }

    #[test]
    fn test_phi3_construction_with_su_rope() {
        let cfg = su_rope_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Phi3 should construct with su-scaled RoPE: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_phi3_construction_with_longrope_type() {
        let mut cfg = su_rope_config();
        if let Some(serde_json::Value::Object(ref mut map)) = cfg.extra.get_mut("rope_scaling") {
            map.remove("type");
            map.insert("rope_type".into(), "longrope".into());
        }

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Phi3 should accept rope_type=longrope: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_phi3_fallback_standard_rope_no_scaling() {
        let cfg = test_config(); // no rope_scaling in extra
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should fall back to standard RoPE when no rope_scaling: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_phi3_fallback_standard_rope_null_scaling() {
        let mut cfg = test_config();
        cfg.extra
            .insert("rope_scaling".into(), serde_json::Value::Null);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should fall back to standard RoPE when rope_scaling is null: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_phi3_fallback_standard_rope_default_type() {
        let mut cfg = test_config();
        let mut rope_scaling = serde_json::Map::new();
        rope_scaling.insert("type".into(), "default".into());
        cfg.extra.insert("rope_scaling".into(), rope_scaling.into());

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should fall back to standard RoPE when type=default: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_phi3_su_rope_forward_shape() {
        let cfg = su_rope_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_phi3_su_rope_missing_short_factor() {
        let mut cfg = su_rope_config();
        if let Some(serde_json::Value::Object(ref mut map)) = cfg.extra.get_mut("rope_scaling") {
            map.remove("short_factor");
        }

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3ForCausalLM::new(&cfg, vb);
        assert!(model.is_err(), "Should fail when short_factor is missing");
    }
}
