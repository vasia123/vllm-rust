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
use crate::engine::{DecodeSequenceMetadata, PoolingStrategy};
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Gemma3 RMSNorm ────────────────────────────────────────────────────────

type Gemma3RmsNorm = crate::layers::RmsNorm;

/// Gemma 3 RMSNorm. `prefolded` = the stored weight already includes the `+1`
/// (llama.cpp's GGUF conversion adds 1 to every `*norm.weight`), so apply it
/// plainly; otherwise use the HF `(1 + weight)` ScalePlusOne form.
#[inline]
fn gemma3_rms_norm(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    prefolded: bool,
) -> Result<Gemma3RmsNorm> {
    if prefolded {
        crate::layers::rms_norm(size, eps, vb)
    } else {
        crate::layers::rms_norm_gemma(size, eps, vb)
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

/// Full bidirectional mask (encoder-only): every query attends to every key.
/// Used by EmbeddingGemma's bidirectional Gemma3 encoder (`attention.causal`
/// = false).
fn bidirectional_mask(
    q_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Tensor::zeros((1, 1, q_len, kv_len), DType::F32, device)?.to_dtype(dtype)
}

/// Symmetric sliding-window mask (encoder-only): a query at position `p`
/// attends to keys within `window_size` on EITHER side. For sequences shorter
/// than the window this is identical to full bidirectional attention.
///
/// NOTE: the exact long-context (> window) boundary convention may differ from
/// llama.cpp's bidirectional SWA; EmbeddingGemma inputs are typically well
/// under the 512-token window, where this is exact.
fn bidirectional_sliding_window_mask(
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
            if j.abs_diff(query_pos) < window_size {
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
    /// EmbeddingGemma: bidirectional encoder (`is_encoder_only` in extra).
    is_encoder_only: bool,
    /// Explicit per-layer sliding/global flags (true = sliding) from the GGUF
    /// `sliding_window_pattern` bool array. Takes precedence over the stride
    /// heuristic when present (Gemma 3 is 5 sliding : 1 global).
    layer_is_sliding: Vec<bool>,
    /// Norm weights already include the `+1` (GGUF convention) → apply plainly.
    norm_prefolded: bool,
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

        let is_encoder_only = cfg
            .extra
            .get("is_encoder_only")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let layer_is_sliding = cfg
            .extra
            .get("layer_is_sliding")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(true)).collect())
            .unwrap_or_default();

        let norm_prefolded = cfg
            .extra
            .get("gemma_norm_prefolded")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            query_pre_attn_scalar,
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window_pattern,
            rope_theta_local,
            is_encoder_only,
            layer_is_sliding,
            norm_prefolded,
        }
    }

    fn is_sliding_window_layer(&self, layer_idx: usize) -> bool {
        // Prefer the explicit per-layer flags from the GGUF when available.
        if let Some(&sliding) = self.layer_is_sliding.get(layer_idx) {
            return sliding;
        }
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
        // Gemma activation is gelu_pytorch_tanh (the tanh approximation), not
        // the exact erf gelu — matches llama.cpp/HF for EmbeddingGemma.
        let activated = candle_nn::Activation::GeluPytorchTanh
            .forward(&gate)?
            .mul(&up)?;
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
    /// EmbeddingGemma: bidirectional (encoder-only) attention instead of causal.
    is_encoder_only: bool,
    /// Gemma 3 QK-norm: per-head RMSNorm on query/key before RoPE. Mandatory
    /// for Gemma 3 (and EmbeddingGemma); the GGUF ships `attn_q_norm`/
    /// `attn_k_norm`. Omitting it scrambles attention.
    q_norm: Gemma3RmsNorm,
    k_norm: Gemma3RmsNorm,
}

impl QuantizedGemma3Attention {
    /// Per-head RMSNorm over the last (head_dim) axis of `[b, h, s, d]`.
    fn apply_per_head_norm(x: &Tensor, norm: &Gemma3RmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
    }

    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma3ExtraConfig,
        loader: &dyn QuantizedWeightLoader,
        layer_idx: usize,
        prefix: &str,
        vb: VarBuilder,
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

        // Gemma 3 QK-norm (per-head RMSNorm on q/k, applied before RoPE).
        let pf = extra_cfg.norm_prefolded;
        let q_norm = gemma3_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"), pf)?;
        let k_norm = gemma3_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"), pf)?;

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
            is_encoder_only: extra_cfg.is_encoder_only,
            q_norm,
            k_norm,
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

        // Gemma 3 QK-norm: per-head RMSNorm on q/k BEFORE RoPE.
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;
        let k = Self::apply_per_head_norm(&k, &self.k_norm)?;

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

        let mask = match (self.is_encoder_only, self.sliding_window) {
            // EmbeddingGemma: bidirectional, optionally windowed.
            (true, Some(window_size)) => bidirectional_sliding_window_mask(
                q_len,
                kv_len,
                seqlen_offset,
                window_size,
                dtype,
                device,
            )?,
            (true, None) => bidirectional_mask(q_len, kv_len, dtype, device)?,
            // Causal Gemma 3 chat model (unchanged).
            (false, Some(window_size)) => {
                sliding_window_mask(q_len, kv_len, seqlen_offset, window_size, dtype, device)?
            }
            (false, None) => crate::layers::causal_mask(q_len, seqlen_offset, dtype, device)?,
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

        // Gemma 3 QK-norm (per-head RMSNorm on q/k before RoPE).
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;
        let k = Self::apply_per_head_norm(&k, &self.k_norm)?;

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
        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);
        let self_attn = QuantizedGemma3Attention::new(
            cfg,
            extra_cfg,
            loader,
            layer_idx,
            &format!("{prefix}.self_attn"),
            vb_layer.pp("self_attn"),
        )?;
        let mlp = QuantizedGeGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        let pf = extra_cfg.norm_prefolded;
        let input_layernorm = gemma3_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
            pf,
        )?;
        let post_attention_layernorm = gemma3_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
            pf,
        )?;
        let pre_feedforward_layernorm = gemma3_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("pre_feedforward_layernorm"),
            pf,
        )?;
        let post_feedforward_layernorm = gemma3_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_feedforward_layernorm"),
            pf,
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
    /// EmbeddingGemma sentence-transformers projector applied after pooling:
    /// `dense.1(dense.0(pooled))`. Identity activation, no bias. `None` for a
    /// normal causal Gemma 3.
    st_projector: Option<(Box<dyn QuantizedLinear>, Box<dyn QuantizedLinear>)>,
    /// Native pooling strategy for embedding mode (mean for EmbeddingGemma).
    native_pooling: Option<PoolingStrategy>,
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

        let norm = gemma3_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_m.pp("norm"),
            extra_cfg.norm_prefolded,
        )?;

        // Gemma3 always uses tied embeddings
        let lm_head = Box::new(TiedEmbeddingHead {
            weight: embed_tokens.embeddings().clone(),
        }) as Box<dyn QuantizedLinear>;

        // EmbeddingGemma sentence-transformers projector: two bias-free dense
        // layers (Identity activation) at the GGUF top level (`dense.0`,
        // `dense.1`), applied after mean pooling. `hidden → mid → out`.
        let st_projector = if cfg
            .extra
            .get("has_st_projector")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let mid = cfg
                .extra
                .get("st_projector_mid")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.hidden_size as u64) as usize;
            let out = cfg
                .extra
                .get("st_projector_out")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.hidden_size as u64) as usize;
            let dense_0 = weight_loader.load_linear("dense.0", cfg.hidden_size, mid, false)?;
            let dense_1 = weight_loader.load_linear("dense.1", mid, out, false)?;
            Some((dense_0, dense_1))
        } else {
            None
        };

        let native_pooling = cfg
            .extra
            .get("embedding_pooling")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<PoolingStrategy>().ok());

        tracing::info!(
            is_encoder_only = extra_cfg.is_encoder_only,
            has_projector = st_projector.is_some(),
            ?native_pooling,
            query_pre_attn_scalar = extra_cfg.query_pre_attn_scalar,
            "Gemma 3 quantized model built"
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            final_logit_softcap: extra_cfg.final_logit_softcap,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            st_projector,
            native_pooling,
        })
    }

    /// Per-token hidden states for embeddings: full sequence, post-final-norm,
    /// pre-`lm_head`. Returns `[seq, hidden]`. Uses bidirectional attention
    /// when the model is an encoder (EmbeddingGemma); otherwise causal.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Per-layer attention builds its own mask (causal vs bidirectional via
        // `is_encoder_only`), so the passed mask is unused — mirror `forward`.
        let normalizer = (self.hidden_size as f64).sqrt();
        let mut xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                None,
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        xs.squeeze(0)
    }

    /// Native pooling strategy for embedding mode (`None` = caller decides).
    pub fn embedding_pooling(&self) -> Option<PoolingStrategy> {
        self.native_pooling
    }

    /// Apply the sentence-transformers projector to a pooled embedding
    /// `[batch, hidden]`. Identity if no projector is loaded.
    pub fn project_embedding(&self, pooled: &Tensor) -> Result<Tensor> {
        match &self.st_projector {
            Some((dense_0, dense_1)) => dense_1.forward(&dense_0.forward(pooled)?),
            None => Ok(pooled.clone()),
        }
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
        // Flatten 3D `[B, S, H]` to 2D so cuBLAS picks plain GEMM
        // instead of stride-0 batched GEMM. +40% e2e at c=8 on the
        // lm_head shape (Qwen3-4B-AWQ side-by-side, 2026-05-09).
        match x.dims().len() {
            3 => {
                let dims = x.dims();
                let (b, s, h) = (dims[0], dims[1], dims[2]);
                let v = self.weight.dims()[0];
                let x_flat = x.reshape((b * s, h))?;
                let y_flat = x_flat.matmul(&self.weight.t()?)?;
                y_flat.reshape((b, s, v))
            }
            _ => x.matmul(&self.weight.t()?),
        }
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

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn forward_hidden(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedGemma3ForCausalLM::forward_hidden(
            self,
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn embedding_pooling(&self) -> Option<PoolingStrategy> {
        QuantizedGemma3ForCausalLM::embedding_pooling(self)
    }

    fn project_embedding(&self, pooled: &Tensor) -> Result<Tensor> {
        QuantizedGemma3ForCausalLM::project_embedding(self, pooled)
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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
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

    #[test]
    fn test_quantized_gemma3_forward_hidden_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        assert!(
            <QuantizedGemma3ForCausalLM as crate::engine::ModelForward>::supports_embeddings(
                &model
            ),
            "gemma3 quantized must advertise embedding support"
        );

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);
        let seq_len = 5;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let hidden = model
            .forward_hidden(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward_hidden");
        assert_eq!(
            hidden.dims(),
            &[seq_len, cfg.hidden_size],
            "forward_hidden shape should be [seq_len, hidden_size]"
        );
    }

    /// EmbeddingGemma mode: bidirectional attention + native mean pooling.
    /// (No projector here — dense weights aren't in the zero VarBuilder; the
    /// projector path is covered end-to-end by the ollama golden check.)
    #[test]
    fn test_quantized_gemma3_encoder_mode() {
        let mut cfg = test_config();
        cfg.extra
            .insert("is_encoder_only".into(), serde_json::json!(true));
        cfg.extra
            .insert("embedding_pooling".into(), serde_json::json!("mean"));
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        assert_eq!(
            <QuantizedGemma3ForCausalLM as crate::engine::ModelForward>::embedding_pooling(&model),
            Some(crate::engine::PoolingStrategy::Mean),
            "encoder mode must report mean pooling"
        );

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);
        let seq_len = 6;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        // Runs the bidirectional-mask path without error and preserves shape.
        let hidden = model
            .forward_hidden(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("encoder forward_hidden");
        assert_eq!(hidden.dims(), &[seq_len, cfg.hidden_size]);
    }
}
