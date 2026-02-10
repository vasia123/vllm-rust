//! Quantized Gemma3 model implementation.
//!
//! This module provides a quantized version of the Gemma3 model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.
//!
//! Gemma3-specific features preserved in the quantized path:
//! - Modified RMSNorm: (1 + weight) scaling
//! - Embedding normalization by sqrt(hidden_size)
//! - 4 layernorms per decoder layer (pre/post attention + pre/post feedforward)
//! - Attention logit soft capping
//! - Final logit soft capping
//! - Alternating local (sliding window) / global attention
//!   (configurable pattern via `sliding_window_pattern`)
//! - Separate local/global RoPE theta
//! - Custom query_pre_attn_scalar scaling

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Gemma3 RMSNorm ────────────────────────────────────────────────────────

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

// ─── Soft Capping ──────────────────────────────────────────────────────────

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

/// Create a sliding window causal mask.
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

// ─── Gemma3 Config Extraction ──────────────────────────────────────────────

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

// ─── Quantized GeGLU MLP ──────────────────────────────────────────────────

struct QuantizedGeGluMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedGeGluMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{prefix}.gate_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let up_proj = loader.load_linear(
            &format!("{prefix}.up_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{prefix}.down_proj"),
            intermediate_size,
            hidden_size,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = gate.gelu_erf()?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ─── Quantized Attention ───────────────────────────────────────────────────

struct QuantizedGemma3Attention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f64,
    attn_logit_softcap: Option<f64>,
    sliding_window: Option<usize>,
}

impl QuantizedGemma3Attention {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3ExtraConfig,
        loader: &dyn QuantizedWeightLoader,
        layer_idx: usize,
        prefix: &str,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

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
            loader.dtype(),
            loader.device(),
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

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

        // Read full K, V history from cache
        let num_tokens = seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(block_table.block_ids(), num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        let kv_len = k_full.dim(2)?;

        // GQA: repeat KV heads to match num_heads
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        let q = (q * self.scaling)?;

        let mut attn_weights = q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

        if let Some(cap) = self.attn_logit_softcap {
            attn_weights = soft_cap(&attn_weights, cap)?;
        }

        let mask = if let Some(window_size) = self.sliding_window {
            sliding_window_mask(q_len, kv_len, seqlen_offset, window_size, dtype, device)?
        } else {
            crate::layers::causal_mask(q_len, seqlen_offset, dtype, device)?
        };

        attn_weights = attn_weights.broadcast_add(&mask)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_full)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

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
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.i(i)?.unsqueeze(0)?;
            let k_i = k.i(i)?.unsqueeze(0)?;
            let v_i = v.i(i)?.unsqueeze(0)?;

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
                let mask =
                    sliding_window_mask(1, kv_len, seq.seqlen_offset, window_size, dtype, device)?;
                attn_weights = attn_weights.broadcast_add(&mask)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_full)?;

            let attn_output =
                attn_output
                    .transpose(1, 2)?
                    .reshape((1, 1, self.num_heads * self.head_dim))?;

            outputs.push(attn_output);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output)
    }
}

// ─── Quantized Decoder Layer ───────────────────────────────────────────────

struct QuantizedGemma3DecoderLayer {
    self_attn: QuantizedGemma3Attention,
    mlp: QuantizedGeGluMlp,
    input_layernorm: Gemma3RmsNorm,
    post_attention_layernorm: Gemma3RmsNorm,
    pre_feedforward_layernorm: Gemma3RmsNorm,
    post_feedforward_layernorm: Gemma3RmsNorm,
}

impl QuantizedGemma3DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3ExtraConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let self_attn = QuantizedGemma3Attention::new(
            cfg,
            extra_cfg,
            loader,
            layer_idx,
            &format!("{prefix}.self_attn"),
        )?;
        let mlp = QuantizedGeGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);
        let input_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = Gemma3RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_feedforward_layernorm"),
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
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        residual + hidden_states
    }
}

// ─── Quantized Model ───────────────────────────────────────────────────────

/// Quantized Gemma3 model supporting FP8, GPTQ, AWQ, and unquantized weights.
pub struct QuantizedGemma3ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedGemma3DecoderLayer>,
    norm: Gemma3RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    hidden_size: usize,
    final_logit_softcap: Option<f64>,
    device: Device,
    dtype: DType,
}

impl QuantizedGemma3ForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let extra_cfg = Gemma3ExtraConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedGemma3DecoderLayer::new(
                cfg,
                &extra_cfg,
                weight_loader,
                vb.clone(),
                i,
            )?);
        }

        let norm = Gemma3RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Gemma3 always uses tied embeddings
        let lm_head = Box::new(TiedEmbeddingHead {
            weight: embed_tokens.embeddings().clone(),
        }) as Box<dyn QuantizedLinear>;

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

        // Gemma3 normalizes embeddings by sqrt(hidden_size)
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
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        let mut logits = self.lm_head.forward(&xs)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper for tied embedding lm_head.
struct TiedEmbeddingHead {
    weight: Tensor,
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match x.dims().len() {
            3 => self.weight.broadcast_left(x.dim(0)?)?,
            _ => self.weight.clone(),
        };
        x.matmul(&w.t()?)
    }

    fn load_weights(&mut self, _weights: &std::collections::HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    fn has_bias(&self) -> bool {
        false
    }
}

impl crate::engine::ModelForward for QuantizedGemma3ForCausalLM {
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
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.norm.forward(&xs)?;
        let mut logits = self.lm_head.forward(&xs)?;

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
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

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
    fn test_quantized_gemma3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGemma3ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedGemma3ForCausalLM should construct with unquantized loader: {:?}",
            model.err()
        );

        let model = model.expect("construction succeeded");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_logit_softcap.is_some());
    }

    #[test]
    fn test_quantized_gemma3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
}
