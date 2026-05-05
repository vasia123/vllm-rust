//! ChatGLM model implementation (THUDM ChatGLM2/3/4 series).
//!
//! Key architectural features:
//! - Packed QKV projection (single query_key_value linear)
//! - Packed gate+up projection (single dense_h_to_4h linear for SwiGLU)
//! - Partial RoPE (factor 0.5) with configurable neox style
//! - Configurable MQA/GQA via `multi_query_attention` config flag
//! - Configurable norm type (RMSNorm or LayerNorm)
//! - Optional `apply_residual_connection_post_layernorm`
//! - Optional `post_layer_norm` (final layer norm before output)
//!
//! Weight prefix: `transformer.encoder.layers.{i}.*` (not `model.layers.{i}.*`)
//!
//! Reference: vLLM's chatglm.py implementation.

use crate::layers::{rms_norm, RmsNorm};
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

// ─── Constants ───────────────────────────────────────────────────────────────

const CHATGLM_PARTIAL_ROTARY_FACTOR: f64 = 0.5;

// ─── Normalization Abstraction ──────────────────────────────────────────────

/// Abstracts over RMSNorm and LayerNorm so forward calls are uniform.
enum Norm {
    Rms(RmsNorm),
    Layer(LayerNorm),
}

impl Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Norm::Rms(n) => n.forward(xs),
            Norm::Layer(n) => n.forward(xs),
        }
    }
}

fn create_norm(hidden_size: usize, eps: f64, use_rmsnorm: bool, vb: VarBuilder) -> Result<Norm> {
    if use_rmsnorm {
        Ok(Norm::Rms(rms_norm(hidden_size, eps, vb)?))
    } else {
        Ok(Norm::Layer(layer_norm(hidden_size, eps, vb)?))
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

// ChatGLM = MQA/GQA + fused 'query_key_value' + 'dense' (output) + partial RoPE
// (rotary_dim = head_dim/2). The fused projection cannot have heterogeneous
// QKV/O bias, but ChatGLM is always uniform: add_bias_linear controls the
// output, add_qkv_bias adds bias on QKV (and add_bias_linear implies QKV bias
// too).
struct ChatGLMAttention {
    inner: AttentionBlock,
}

impl ChatGLMAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg
            .extra
            .get("kv_channels")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);

        let multi_query_attention = cfg
            .extra
            .get("multi_query_attention")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let num_kv_heads = if multi_query_attention {
            cfg.num_key_value_heads
        } else {
            num_heads
        };

        let add_bias_linear = cfg
            .extra
            .get("add_bias_linear")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let add_qkv_bias = cfg
            .extra
            .get("add_qkv_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let qkv_bias = add_bias_linear || add_qkv_bias;
        let bias = AttentionBias {
            q: qkv_bias,
            k: qkv_bias,
            v: qkv_bias,
            o: add_bias_linear,
        };

        let attn_cfg = AttentionConfig::gqa(num_heads, num_kv_heads, head_dim, cfg.hidden_size)
            .with_qkv_fused()
            .with_bias(bias)
            .with_proj_names(ProjNames {
                qkv: "query_key_value",
                o: "dense",
                ..Default::default()
            });

        // ChatGLM-specific RoPE: theta = 10000 * rope_ratio, partial rotation
        // covering half of head_dim, original_rope=true → split-style (neox=false).
        let rope_ratio = cfg
            .extra
            .get("rope_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_theta = 10000.0 * rope_ratio;
        let max_positions = cfg
            .extra
            .get("seq_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.max_position_embeddings);
        let original_rope = cfg
            .extra
            .get("original_rope")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let is_neox_style = !original_rope;

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            max_positions,
            rope_theta,
            CHATGLM_PARTIAL_ROTARY_FACTOR,
            is_neox_style,
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

// ─── MLP (SwiGLU with packed gate+up) ───────────────────────────────────────

struct ChatGLMMlp {
    dense_h_to_4h: TpLinear,
    dense_4h_to_h: TpLinear,
    ffn_hidden_size: usize,
}

impl ChatGLMMlp {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let ffn_hidden_size = cfg
            .extra
            .get("ffn_hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let add_bias_linear = cfg
            .extra
            .get("add_bias_linear")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Packed gate+up projection: output is 2 * ffn_hidden_size
        let dense_h_to_4h = TpLinear::column_parallel(
            cfg.hidden_size,
            2 * ffn_hidden_size,
            add_bias_linear,
            false,
            vb.pp("dense_h_to_4h"),
            pg,
        )?;

        let dense_4h_to_h = TpLinear::row_parallel(
            ffn_hidden_size,
            cfg.hidden_size,
            add_bias_linear,
            true,
            vb.pp("dense_4h_to_h"),
            pg,
        )?;

        // For TP, the local ffn_hidden_size is divided by world_size
        let local_ffn = ffn_hidden_size / pg.world_size();

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            ffn_hidden_size: local_ffn,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        // Packed gate+up: [batch, seq, 2*ffn] -> split into gate and up
        let intermediate = self.dense_h_to_4h.forward(xs, tp_ctx)?;

        // SiluAndMul: split in half, silu(gate) * up
        let gate = intermediate.narrow(candle_core::D::Minus1, 0, self.ffn_hidden_size)?;
        let up = intermediate.narrow(
            candle_core::D::Minus1,
            self.ffn_hidden_size,
            self.ffn_hidden_size,
        )?;
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;

        self.dense_4h_to_h.forward(&hidden, tp_ctx)
    }
}

// ─── Decoder Layer (GLMBlock) ───────────────────────────────────────────────

struct ChatGLMBlock {
    self_attention: ChatGLMAttention,
    mlp: ChatGLMMlp,
    input_layernorm: Norm,
    post_attention_layernorm: Norm,
    apply_residual_connection_post_layernorm: bool,
}

impl ChatGLMBlock {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let use_rmsnorm = cfg
            .extra
            .get("rmsnorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let eps = cfg
            .extra
            .get("layernorm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        let apply_residual_connection_post_layernorm = cfg
            .extra
            .get("apply_residual_connection_post_layernorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let self_attention = ChatGLMAttention::new_with_tp(cfg, vb.pp("self_attention"), pg)?;
        let mlp = ChatGLMMlp::new_with_tp(cfg, vb.pp("mlp"), pg)?;

        let input_layernorm =
            create_norm(cfg.hidden_size, eps, use_rmsnorm, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = create_norm(
            cfg.hidden_size,
            eps,
            use_rmsnorm,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attention,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            apply_residual_connection_post_layernorm,
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
        let layernorm_output = self.input_layernorm.forward(xs)?;

        // Self attention
        let attention_output = self.self_attention.forward(
            &layernorm_output,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // First residual connection
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;

        // Post-attention norm
        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;

        // Second residual connection
        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };

        let mlp_output = self.mlp.forward(&layernorm_output, tp_ctx)?;
        mlp_output + residual
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(xs)?;

        let attention_output = self.self_attention.forward_decode_batch(
            &layernorm_output,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            xs
        };
        let layernorm_input = (residual + attention_output)?;

        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;

        let residual = if self.apply_residual_connection_post_layernorm {
            &layernorm_output
        } else {
            &layernorm_input
        };

        let mlp_output = self.mlp.forward(&layernorm_output, tp_ctx)?;
        mlp_output + residual
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// ChatGLM model for causal language modeling (THUDM ChatGLM2/3/4 series).
///
/// Weight path structure:
/// - `transformer.embedding.word_embeddings` -> embedding
/// - `transformer.encoder.layers.{i}.*` -> transformer layers
/// - `transformer.encoder.final_layernorm` -> final norm
/// - `transformer.output_layer` -> lm_head
pub struct ChatGLMForCausalLM {
    embedding: TpEmbedding,
    layers: Vec<ChatGLMBlock>,
    final_layernorm: Option<Norm>,
    output_layer: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl ChatGLMForCausalLM {
    /// Create a new ChatGLM model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new ChatGLM model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_t = vb.pp("transformer");

        // padded_vocab_size overrides vocab_size if present
        let vocab_size = cfg
            .extra
            .get("padded_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

        let embedding = TpEmbedding::new(
            vocab_size,
            cfg.hidden_size,
            vb_t.pp("embedding").pp("word_embeddings"),
            pg,
        )?;

        let post_layer_norm = cfg
            .extra
            .get("post_layer_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let use_rmsnorm = cfg
            .extra
            .get("rmsnorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let eps = cfg
            .extra
            .get("layernorm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        let vb_enc = vb_t.pp("encoder");

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_enc.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ChatGLMBlock::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let final_layernorm = if post_layer_norm {
            Some(create_norm(
                cfg.hidden_size,
                eps,
                use_rmsnorm,
                vb_enc.pp("final_layernorm"),
            )?)
        } else {
            None
        };

        let output_layer = TpLinear::column_parallel(
            cfg.hidden_size,
            vocab_size,
            false,
            true, // gather output
            vb_t.pp("output_layer"),
            pg,
        )?;

        Ok(Self {
            embedding,
            layers,
            final_layernorm,
            output_layer,
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

        let mut xs = self.embedding.forward(input_ids, &self.tp_ctx)?;
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

        if let Some(ref norm) = self.final_layernorm {
            xs = norm.forward(&xs)?;
        }

        self.output_layer.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for ChatGLMForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        ChatGLMForCausalLM::forward(
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
        let mut xs = self.embedding.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        if let Some(ref norm) = self.final_layernorm {
            xs = norm.forward(&xs)?;
        }

        self.output_layer.forward(&xs, &self.tp_ctx)
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
            "multi_query_attention".into(),
            serde_json::Value::Bool(true),
        );
        extra.insert("padded_vocab_size".into(), serde_json::Value::from(256u64));
        extra.insert(
            "apply_residual_connection_post_layernorm".into(),
            serde_json::Value::Bool(false),
        );
        extra.insert("post_layer_norm".into(), serde_json::Value::Bool(true));
        extra.insert("rmsnorm".into(), serde_json::Value::Bool(true));
        extra.insert("layernorm_epsilon".into(), serde_json::Value::from(1e-5));
        extra.insert("ffn_hidden_size".into(), serde_json::Value::from(128u64));

        ModelConfig {
            architectures: vec!["ChatGLMForCausalLM".to_string()],
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
            rope_theta: 10000.0,
            tie_word_embeddings: false,
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

    #[test]
    fn test_chatglm_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = ChatGLMForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ChatGLMForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_layernorm.is_some());
    }

    #[test]
    fn test_chatglm_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb).expect("build model");

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

        let vocab_size = cfg
            .extra
            .get("padded_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

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
            &[batch_size, seq_len, vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_chatglm_no_post_layer_norm() {
        let mut cfg = test_config();
        cfg.extra
            .insert("post_layer_norm".into(), serde_json::Value::Bool(false));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb).expect("build model");
        assert!(model.final_layernorm.is_none());
    }

    #[test]
    fn test_chatglm_layernorm_mode() {
        // Test with LayerNorm instead of RMSNorm
        let mut cfg = test_config();
        cfg.extra
            .insert("rmsnorm".into(), serde_json::Value::Bool(false));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ChatGLMForCausalLM should construct with LayerNorm: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_chatglm_post_layernorm_residual() {
        // Test apply_residual_connection_post_layernorm=true
        let mut cfg = test_config();
        cfg.extra.insert(
            "apply_residual_connection_post_layernorm".into(),
            serde_json::Value::Bool(true),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let result = model.forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(
            result.is_ok(),
            "forward with post_layernorm residual: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_chatglm_mha_mode() {
        // Test with multi_query_attention=false (MHA: kv_heads == num_heads)
        let mut cfg = test_config();
        cfg.extra.insert(
            "multi_query_attention".into(),
            serde_json::Value::Bool(false),
        );
        // When MQA is off, kv_heads = num_heads
        cfg.num_key_value_heads = cfg.num_attention_heads;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "ChatGLMForCausalLM should construct in MHA mode: {:?}",
            model.err()
        );

        let model = model.unwrap();
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_attention_heads, // MHA: kv_heads == num_heads
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward in MHA mode");

        let vocab_size = cfg
            .extra
            .get("padded_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);
        assert_eq!(logits.dims(), &[1, 3, vocab_size]);
    }

    #[test]
    fn test_chatglm_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = ChatGLMForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let vocab_size = cfg
            .extra
            .get("padded_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, vocab_size]);
        block_table.advance(3);

        // Decode step
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
        assert_eq!(logits.dims(), &[1, 1, vocab_size]);
    }
}
