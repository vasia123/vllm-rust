//! Gemma 3 model with LoRA adapter support.
//!
//! Gemma 3 extends Gemma 2 with alternating local (sliding window) and global
//! attention layers. Each layer has its own RoPE configuration (different
//! `rope_theta` for local vs global layers). LoRA is applied to all attention
//! projections (q, k, v, o) and MLP projections (gate_proj, up_proj, down_proj).
//!
//! Architecture:
//! ```text
//! Embedding (* sqrt(hidden_size)) -> [Gemma3Layer x N] -> RMSNorm -> LM Head
//!
//! Gemma3Layer (4-norm pattern):
//!   InputLayerNorm -> Attention -> PostAttnNorm -> Residual
//!   PreFFNorm -> GeGLU MLP -> PostFFNorm -> Residual
//! ```

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;
use crate::lora::{LinearWithLora, LoraContext, LoraModel};

// ─── Gemma3 RMSNorm ────────────────────────────────────────────────────────
//
// Same as Gemma/Gemma2: output = x * (1 + weight) / rms(x)

struct Gemma3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Gemma3RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Gemma3RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let scale = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        xs_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

// ─── Soft Capping ───────────────────────────────────────────────────────────

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

// ─── Sliding Window Mask ────────────────────────────────────────────────────

fn sliding_window_mask(
    q_len: usize,
    kv_len: usize,
    seqlen_offset: usize,
    window_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![f32::NEG_INFINITY; q_len * kv_len];

    for i in 0..q_len {
        let query_pos = seqlen_offset + i;
        for j in 0..kv_len {
            let is_causal = j <= query_pos;
            let is_in_window = query_pos < window_size || j > query_pos - window_size;

            if is_causal && is_in_window {
                mask[i * kv_len + j] = 0.0;
            }
        }
    }

    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

// ─── Extra Config ───────────────────────────────────────────────────────────

struct Gemma3ExtraConfig {
    query_pre_attn_scalar: f64,
    attn_logit_softcap: Option<f64>,
    final_logit_softcap: Option<f64>,
    sliding_window_pattern: usize,
    rope_theta_local: f64,
}

impl Gemma3ExtraConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let query_pre_attn_scalar = cfg
            .extra
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .unwrap_or((cfg.head_dim as f64).sqrt());

        let attn_logit_softcap = cfg
            .extra
            .get("attn_logit_softcapping")
            .and_then(|v| v.as_f64());

        let final_logit_softcap = cfg
            .extra
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64());

        let sliding_window_pattern = cfg
            .extra
            .get("sliding_window_pattern")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let rope_theta_local = cfg
            .extra
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        Self {
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window_pattern,
            rope_theta_local,
        }
    }

    fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        if self.sliding_window_pattern == 0 {
            return false;
        }
        layer_idx.is_multiple_of(self.sliding_window_pattern)
    }
}

// ─── GeGLU MLP with LoRA ────────────────────────────────────────────────────

struct GeGluMlpWithLora {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
}

impl GeGluMlpWithLora {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = LinearWithLora::new(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = LinearWithLora::new(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = LinearWithLora::new(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, layer_lora: &LoraModel, layer_prefix: &str) {
        let gate_key = format!("{}.gate_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&gate_key) {
            self.gate_proj
                .register_adapter(adapter_name, adapter.clone());
        }

        let up_key = format!("{}.up_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&up_key) {
            self.up_proj.register_adapter(adapter_name, adapter.clone());
        }

        let down_key = format!("{}.down_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&down_key) {
            self.down_proj
                .register_adapter(adapter_name, adapter.clone());
        }
    }

    fn forward(&self, xs: &Tensor, lora_ctx: &LoraContext) -> Result<Tensor> {
        let adapter = lora_ctx.adapter_name();
        let gate = self
            .gate_proj
            .forward_with_lora(xs, adapter)?
            .apply(&candle_nn::Activation::Gelu)?;
        let up = self.up_proj.forward_with_lora(xs, adapter)?;
        let intermediate = (gate * up)?;
        self.down_proj.forward_with_lora(&intermediate, adapter)
    }
}

// ─── Attention with LoRA ─────────────────────────────────────────────────────

struct Gemma3AttentionWithLora {
    q_proj: LinearWithLora,
    k_proj: LinearWithLora,
    v_proj: LinearWithLora,
    o_proj: LinearWithLora,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    attn_logit_softcap: Option<f64>,
    sliding_window: Option<usize>,
}

impl Gemma3AttentionWithLora {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3ExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = LinearWithLora::new(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            LinearWithLora::new(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            LinearWithLora::new(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = LinearWithLora::new(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let is_local = extra_cfg.is_sliding_window_layer(layer_idx);
        let rope_theta = if is_local {
            extra_cfg.rope_theta_local
        } else {
            cfg.rope_theta
        };

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        let scaling = 1.0 / extra_cfg.query_pre_attn_scalar.sqrt();
        let sliding_window = if is_local { cfg.sliding_window } else { None };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            scaling,
            attn_logit_softcap: extra_cfg.attn_logit_softcap,
            sliding_window,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, layer_lora: &LoraModel, layer_prefix: &str) {
        let q_key = format!("{}.q_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&q_key) {
            self.q_proj.register_adapter(adapter_name, adapter.clone());
        }

        let k_key = format!("{}.k_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&k_key) {
            self.k_proj.register_adapter(adapter_name, adapter.clone());
        }

        let v_key = format!("{}.v_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&v_key) {
            self.v_proj.register_adapter(adapter_name, adapter.clone());
        }

        let o_key = format!("{}.o_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&o_key) {
            self.o_proj.register_adapter(adapter_name, adapter.clone());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let device = xs.device();
        let dtype = xs.dtype();
        let adapter = lora_ctx.adapter_name();

        let q = self.q_proj.forward_with_lora(xs, adapter)?;
        let k = self.k_proj.forward_with_lora(xs, adapter)?;
        let v = self.v_proj.forward_with_lora(xs, adapter)?;

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

        // Write new K, V to cache
        let k_for_cache = k.squeeze(0)?.contiguous()?;
        let v_for_cache = v.squeeze(0)?.contiguous()?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        // Read full K, V history
        let num_tokens = seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(block_table.block_ids(), num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        let kv_len = k_full.dim(2)?;

        // GQA expansion
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        let q = (q * self.scaling)?;

        let mut attn_weights = q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

        // Attention logit soft capping
        if let Some(cap) = self.attn_logit_softcap {
            attn_weights = soft_cap(&attn_weights, cap)?;
        }

        // Mask: sliding window for local layers, causal for global layers
        let mask = if let Some(window_size) = self.sliding_window {
            sliding_window_mask(q_len, kv_len, seqlen_offset, window_size, dtype, device)?
        } else {
            crate::layers::causal_mask(q_len, seqlen_offset, dtype, device)?
        };

        attn_weights = attn_weights.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_full)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward_with_lora(&attn_output, adapter)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let device = xs.device();
        let dtype = xs.dtype();
        let adapter = lora_ctx.adapter_name();

        let q = self.q_proj.forward_with_lora(xs, adapter)?;
        let k = self.k_proj.forward_with_lora(xs, adapter)?;
        let v = self.v_proj.forward_with_lora(xs, adapter)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let mut outputs = Vec::with_capacity(batch_size);

        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

            let k_for_cache = k_i.squeeze(0)?.contiguous()?;
            let v_for_cache = v_i.squeeze(0)?.contiguous()?;
            cache_engine
                .write(&k_for_cache, &v_for_cache, &seq.slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let kv_len = seq.seqlen_offset + 1;
            let (k_full, v_full) = cache_engine
                .read(&seq.block_ids, kv_len)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

            let k_full = repeat_kv(k_full, num_kv_groups)?;
            let v_full = repeat_kv(v_full, num_kv_groups)?;

            let q_i = (q_i * self.scaling)?;

            let mut attn_weights = q_i.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

            if let Some(cap) = self.attn_logit_softcap {
                attn_weights = soft_cap(&attn_weights, cap)?;
            }

            if let Some(window_size) = self.sliding_window {
                let mask = sliding_window_mask(
                    1,
                    kv_len,
                    seq.seqlen_offset,
                    window_size,
                    dtype,
                    device,
                )?;
                attn_weights = attn_weights.broadcast_add(&mask)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_full)?;

            let attn_output = attn_output
                .transpose(1, 2)?
                .reshape((1, 1, self.num_heads * self.head_dim))?;

            outputs.push(attn_output);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward_with_lora(&attn_output, adapter)
    }
}

// ─── Decoder Layer with LoRA ─────────────────────────────────────────────────

struct Gemma3DecoderLayerWithLora {
    self_attn: Gemma3AttentionWithLora,
    mlp: GeGluMlpWithLora,
    input_layernorm: Gemma3RmsNorm,
    post_attention_layernorm: Gemma3RmsNorm,
    pre_feedforward_layernorm: Gemma3RmsNorm,
    post_feedforward_layernorm: Gemma3RmsNorm,
}

impl Gemma3DecoderLayerWithLora {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3ExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn =
            Gemma3AttentionWithLora::new(cfg, extra_cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = GeGluMlpWithLora::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let input_layernorm =
            Gemma3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = Gemma3RmsNorm::new(
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

    fn register_lora(&mut self, adapter_name: &str, lora_model: &LoraModel, layer_idx: usize) {
        let attn_prefix = format!("layers.{}.self_attn", layer_idx);
        self.self_attn
            .register_lora(adapter_name, lora_model, &attn_prefix);

        let mlp_prefix = format!("layers.{}.mlp", layer_idx);
        self.mlp
            .register_lora(adapter_name, lora_model, &mlp_prefix);
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
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            lora_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states, lora_ctx)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            lora_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states, lora_ctx)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }
}

// ─── Model with LoRA ─────────────────────────────────────────────────────────

/// Gemma 3 model with LoRA adapter support.
///
/// Uses Gemma-specific (1+weight) RMSNorm, GeGLU MLP activation, alternating
/// local/global attention with per-layer RoPE, attention logit soft-capping,
/// and final logit soft-capping. Embeddings are scaled by sqrt(hidden_size).
pub struct Gemma3WithLora {
    embed_tokens: Embedding,
    layers: Vec<Gemma3DecoderLayerWithLora>,
    norm: Gemma3RmsNorm,
    lm_head: candle_nn::Linear,
    hidden_size: usize,
    final_logit_softcap: Option<f64>,
    device: Device,
    dtype: DType,
}

impl Gemma3WithLora {
    /// Create a new Gemma 3 model with LoRA support.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let extra_cfg = Gemma3ExtraConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma3DecoderLayerWithLora::new(
                cfg,
                &extra_cfg,
                i,
                vb_l.pp(i),
            )?);
        }

        let norm = Gemma3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            final_logit_softcap: extra_cfg.final_logit_softcap,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        let adapter_name = &lora_model.name;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.register_lora(adapter_name, lora_model, layer_idx);
        }
    }

    /// Forward pass with optional LoRA adapter.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
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

        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                lora_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        let mut logits = xs.apply(&self.lm_head)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    /// Batched decode forward with optional LoRA adapter.
    pub fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx, lora_ctx)?;
        }

        let xs = self.norm.forward(&xs)?;
        let mut logits = xs.apply(&self.lm_head)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

// ─── ModelForward implementation ─────────────────────────────────────────────

impl crate::engine::ModelForward for Gemma3WithLora {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            &LoraContext::none(),
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, &LoraContext::none())
    }

    fn supports_lora(&self) -> bool {
        true
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── ModelForwardWithLora implementation ─────────────────────────────────────

impl crate::models::ModelForwardWithLora for Gemma3WithLora {
    fn register_lora(&mut self, lora_model: &LoraModel) {
        Gemma3WithLora::register_lora(self, lora_model)
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn lora_adapters(&self) -> Vec<String> {
        if let Some(first_layer) = self.layers.first() {
            first_layer
                .self_attn
                .q_proj
                .adapter_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::lora::LoraAdapter;

    fn test_config() -> crate::config::ModelConfig {
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
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));

        crate::config::ModelConfig {
            architectures: vec!["Gemma3ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
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
    fn test_gemma3_with_lora_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma3WithLora::new(&cfg, vb);
        assert!(model.is_ok(), "Gemma3WithLora should construct");

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_logit_softcap.is_some());
        assert_eq!(model.final_logit_softcap.unwrap(), 30.0);
    }

    #[test]
    fn test_gemma3_with_lora_forward_no_adapter() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma3WithLora::new(&cfg, vb).unwrap();

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 5)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, 5);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::none(),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
    }

    #[test]
    fn test_gemma3_with_lora_register_and_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let mut model = Gemma3WithLora::new(&cfg, vb).unwrap();

        let mut lora_model = LoraModel::new("test-adapter", 1, 8, 16.0);

        let lora_a = Tensor::randn(0.0f32, 0.1, (8, cfg.hidden_size), &device).unwrap();
        let lora_b = Tensor::randn(
            0.0f32,
            0.1,
            (cfg.num_attention_heads * cfg.head_dim, 8),
            &device,
        )
        .unwrap();
        lora_model.add_adapter(
            "layers.0.self_attn.q_proj",
            LoraAdapter::new(lora_a, lora_b, 8, 16.0),
        );

        model.register_lora(&lora_model);

        let adapters = model.layers[0].self_attn.q_proj.adapter_names();
        assert!(
            adapters.contains(&"test-adapter"),
            "Adapter should be registered"
        );

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::with_adapter("test-adapter"),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
