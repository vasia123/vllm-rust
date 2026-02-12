//! T5 encoder-decoder model for conditional generation.
//!
//! Key architectural features:
//! - Relative position bias (not RoPE or absolute embeddings)
//! - Pre-norm (RMS LayerNorm before attention and FFN)
//! - Bidirectional encoder, autoregressive decoder with cross-attention
//! - Gated SiLU FFN (T5 v1.1+) or standard GELU FFN (T5 v1.0)
//! - Shared input embeddings between encoder and decoder
//! - No bias in any linear projections

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{EncoderOutput, ModelForEncoderDecoder};
use crate::kv_cache::{BlockTable, KVCacheManager};

// ─── T5 RMS LayerNorm ──────────────────────────────────────────────────────

/// T5-style RMS LayerNorm (scale only, no bias, no mean subtraction).
struct T5LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl T5LayerNorm {
    fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs_normed = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        (xs_normed.broadcast_mul(&weight))?.to_dtype(dtype)
    }
}

// ─── Relative Position Bias ────────────────────────────────────────────────

/// Learned relative position bias for T5 attention.
///
/// Maps relative distances between query and key positions to bias values
/// using logarithmic bucketing to handle long distances efficiently.
struct RelativePositionBias {
    embeddings: Embedding,
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
}

impl RelativePositionBias {
    fn new(
        num_heads: usize,
        num_buckets: usize,
        max_distance: usize,
        bidirectional: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embeddings = embedding(num_buckets, num_heads, vb.pp("relative_attention_bias"))?;
        Ok(Self {
            embeddings,
            num_buckets,
            max_distance,
            bidirectional,
        })
    }

    /// Compute relative position bucket indices.
    ///
    /// Follows the HuggingFace T5 algorithm:
    /// - Bidirectional: split buckets in half, track sign separately
    /// - Small distances get exact buckets
    /// - Large distances get logarithmically spaced buckets
    fn relative_position_bucket(&self, relative_position: i64) -> usize {
        let mut bucket: usize = 0;
        let mut rel_pos = relative_position;

        if self.bidirectional {
            let half = self.num_buckets / 2;
            if rel_pos > 0 {
                bucket += half;
            }
            rel_pos = rel_pos.unsigned_abs() as i64;

            let max_exact = half / 2;
            if (rel_pos as usize) < max_exact {
                return bucket + rel_pos as usize;
            }

            let val = ((rel_pos as f64 / max_exact as f64).ln()
                / (self.max_distance as f64 / max_exact as f64).ln()
                * (half - max_exact) as f64) as usize;
            bucket + max_exact + val.min(half - max_exact - 1)
        } else {
            rel_pos = (-rel_pos).max(0);
            let max_exact = self.num_buckets / 2;

            if (rel_pos as usize) < max_exact {
                return rel_pos as usize;
            }

            let val = ((rel_pos as f64 / max_exact as f64).ln()
                / (self.max_distance as f64 / max_exact as f64).ln()
                * (self.num_buckets - max_exact) as f64) as usize;
            max_exact + val.min(self.num_buckets - max_exact - 1)
        }
    }

    /// Compute the relative position bias tensor.
    ///
    /// Returns `[1, num_heads, query_len, key_len]`.
    fn compute_bias(
        &self,
        query_len: usize,
        key_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Build bucket indices: [query_len, key_len]
        let mut bucket_indices = vec![0u32; query_len * key_len];
        for qi in 0..query_len {
            for ki in 0..key_len {
                let rel_pos = ki as i64 - qi as i64;
                bucket_indices[qi * key_len + ki] =
                    self.relative_position_bucket(rel_pos) as u32;
            }
        }

        let indices = Tensor::from_vec(bucket_indices, (query_len, key_len), device)?;

        // Look up bias values: [query_len, key_len, num_heads]
        let bias = self.embeddings.forward(&indices)?;

        // Permute to [num_heads, query_len, key_len] and add batch dim
        bias.permute((2, 0, 1))?.unsqueeze(0)
    }
}

// ─── T5 Self-Attention ─────────────────────────────────────────────────────

struct T5SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    relative_bias: Option<RelativePositionBias>,
}

impl T5SelfAttention {
    fn new(
        cfg: &ModelConfig,
        has_relative_bias: bool,
        bidirectional: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg
            .extra
            .get("d_kv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_heads);
        let inner_dim = num_heads * head_dim;

        let q_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("q"))?;
        let k_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("k"))?;
        let v_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("v"))?;
        let o_proj = linear_no_bias(inner_dim, hidden_size, vb.pp("o"))?;

        let num_buckets = cfg
            .extra
            .get("relative_attention_num_buckets")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let max_distance = cfg
            .extra
            .get("relative_attention_max_distance")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;

        let relative_bias = if has_relative_bias {
            Some(RelativePositionBias::new(
                num_heads,
                num_buckets,
                max_distance,
                bidirectional,
                vb,
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            relative_bias,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // T5 does NOT scale attention by 1/sqrt(d_k)
        let mut attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        // Add position bias
        let bias = if let Some(bias) = position_bias {
            attn_weights = attn_weights.broadcast_add(bias)?;
            Some(bias.clone())
        } else if let Some(ref rel_bias) = self.relative_bias {
            let bias = rel_bias.compute_bias(seq_len, seq_len, xs.device())?;
            let bias = bias.to_dtype(attn_weights.dtype())?;
            attn_weights = attn_weights.broadcast_add(&bias)?;
            Some(bias)
        } else {
            None
        };

        // Apply attention mask (causal or padding)
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = self.o_proj.forward(&attn_output)?;
        Ok((output, bias))
    }
}

// ─── T5 Cross-Attention ──────────────────────────────────────────────────

/// T5 cross-attention for decoder → encoder attention.
///
/// Differs from the generic CrossAttention layer:
/// - Uses T5 weight names (q/k/v/o, not q_proj/k_proj/v_proj/o_proj)
/// - Projections are hidden_size → inner_dim (not square)
/// - No 1/sqrt(d_k) scaling (consistent with T5 self-attention)
struct T5CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl T5CrossAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg
            .extra
            .get("d_kv")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_heads);
        let inner_dim = num_heads * head_dim;

        let q_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("q"))?;
        let k_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("k"))?;
        let v_proj = linear_no_bias(hidden_size, inner_dim, vb.pp("v"))?;
        let o_proj = linear_no_bias(inner_dim, hidden_size, vb.pp("o"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass. Returns (output, key, value) for optional caching.
    fn forward(
        &self,
        decoder_hidden: &Tensor,
        encoder_hidden: &Tensor,
        cached_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, tgt_len, _) = decoder_hidden.dims3()?;

        let q = self.q_proj.forward(decoder_hidden)?;
        let q = q
            .reshape((b_sz, tgt_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (k, v) = match cached_kv {
            Some((ck, cv)) => (ck.clone(), cv.clone()),
            None => {
                let (_, src_len, _) = encoder_hidden.dims3()?;
                let k = self.k_proj.forward(encoder_hidden)?;
                let v = self.v_proj.forward(encoder_hidden)?;
                let k = k
                    .reshape((b_sz, src_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = v
                    .reshape((b_sz, src_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                (k, v)
            }
        };

        // T5 does NOT scale by 1/sqrt(d_k)
        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            tgt_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = self.o_proj.forward(&attn_output)?;
        Ok((output, k, v))
    }
}

// ─── T5 FFN ────────────────────────────────────────────────────────────────

/// T5 feed-forward network.
///
/// Two variants:
/// - Standard: wi → relu/gelu → wo
/// - Gated (T5 v1.1+): wi_0 (gate) → silu, wi_1 (up), gate * up → wo
struct T5Ffn {
    wi: Option<Linear>,     // Standard FFN
    wi_0: Option<Linear>,   // Gated FFN gate projection
    wi_1: Option<Linear>,   // Gated FFN up projection
    wo: Linear,
    is_gated: bool,
}

impl T5Ffn {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let d_ff = cfg
            .extra
            .get("d_ff")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let is_gated = cfg
            .extra
            .get("is_gated_act")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let (wi, wi_0, wi_1) = if is_gated {
            let wi_0 = linear_no_bias(hidden_size, d_ff, vb.pp("wi_0"))?;
            let wi_1 = linear_no_bias(hidden_size, d_ff, vb.pp("wi_1"))?;
            (None, Some(wi_0), Some(wi_1))
        } else {
            let wi = linear_no_bias(hidden_size, d_ff, vb.pp("wi"))?;
            (Some(wi), None, None)
        };

        let wo = linear_no_bias(d_ff, hidden_size, vb.pp("wo"))?;

        Ok(Self {
            wi,
            wi_0,
            wi_1,
            wo,
            is_gated,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if self.is_gated {
            let gate = candle_nn::ops::silu(
                &self.wi_0.as_ref().expect("gated FFN requires wi_0").forward(xs)?,
            )?;
            let up = self.wi_1.as_ref().expect("gated FFN requires wi_1").forward(xs)?;
            self.wo.forward(&(gate * up)?)
        } else {
            let hidden = self.wi.as_ref().expect("standard FFN requires wi").forward(xs)?;
            let hidden = hidden.relu()?;
            self.wo.forward(&hidden)
        }
    }
}

// ─── T5 Encoder Layer ──────────────────────────────────────────────────────

struct T5EncoderLayer {
    self_attention: T5SelfAttention,
    norm1: T5LayerNorm,
    ffn: T5Ffn,
    norm2: T5LayerNorm,
}

impl T5EncoderLayer {
    fn new(cfg: &ModelConfig, has_relative_bias: bool, eps: f64, vb: VarBuilder) -> Result<Self> {
        let self_attention =
            T5SelfAttention::new(cfg, has_relative_bias, true, vb.pp("layer").pp("0").pp("SelfAttention"))?;
        let norm1 = T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("layer").pp("0").pp("layer_norm"))?;
        let ffn = T5Ffn::new(cfg, vb.pp("layer").pp("1").pp("DenseReluDense"))?;
        let norm2 = T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("layer").pp("1").pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            norm1,
            ffn,
            norm2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Pre-norm self-attention with residual
        let normed = self.norm1.forward(xs)?;
        let (attn_output, bias) = self.self_attention.forward(&normed, None, position_bias)?;
        let xs = (xs + attn_output)?;

        // Pre-norm FFN with residual
        let normed = self.norm2.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let xs = (xs + ffn_output)?;

        Ok((xs, bias))
    }
}

// ─── T5 Decoder Layer ──────────────────────────────────────────────────────

struct T5DecoderLayer {
    self_attention: T5SelfAttention,
    norm1: T5LayerNorm,
    cross_attention: T5CrossAttention,
    norm2: T5LayerNorm,
    ffn: T5Ffn,
    norm3: T5LayerNorm,
}

impl T5DecoderLayer {
    fn new(cfg: &ModelConfig, has_relative_bias: bool, eps: f64, vb: VarBuilder) -> Result<Self> {
        let self_attention =
            T5SelfAttention::new(cfg, has_relative_bias, false, vb.pp("layer").pp("0").pp("SelfAttention"))?;
        let norm1 = T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("layer").pp("0").pp("layer_norm"))?;

        let cross_attention =
            T5CrossAttention::new(cfg, vb.pp("layer").pp("1").pp("EncDecAttention"))?;
        let norm2 = T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("layer").pp("1").pp("layer_norm"))?;

        let ffn = T5Ffn::new(cfg, vb.pp("layer").pp("2").pp("DenseReluDense"))?;
        let norm3 = T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("layer").pp("2").pp("layer_norm"))?;

        Ok(Self {
            self_attention,
            norm1,
            cross_attention,
            norm2,
            ffn,
            norm3,
        })
    }

    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        xs: &Tensor,
        encoder_hidden: &Tensor,
        causal_mask: Option<&Tensor>,
        self_attn_bias: Option<&Tensor>,
        cached_cross_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<Tensor>, Tensor, Tensor)> {
        // Pre-norm self-attention with causal mask
        let normed = self.norm1.forward(xs)?;
        let (attn_output, bias) =
            self.self_attention.forward(&normed, causal_mask, self_attn_bias)?;
        let xs = (xs + attn_output)?;

        // Pre-norm cross-attention
        let normed = self.norm2.forward(&xs)?;
        let (cross_output, cross_k, cross_v) =
            self.cross_attention.forward(&normed, encoder_hidden, cached_cross_kv)?;
        let xs = (xs + cross_output)?;

        // Pre-norm FFN
        let normed = self.norm3.forward(&xs)?;
        let ffn_output = self.ffn.forward(&normed)?;
        let xs = (xs + ffn_output)?;

        Ok((xs, bias, cross_k, cross_v))
    }
}

// ─── Full T5 Model ─────────────────────────────────────────────────────────

/// T5 encoder-decoder model for conditional generation.
///
/// Implements `ModelForEncoderDecoder` for sequence-to-sequence tasks
/// (translation, summarization, etc.).
pub struct T5ForConditionalGeneration {
    shared_embeddings: Embedding,
    encoder_layers: Vec<T5EncoderLayer>,
    encoder_final_norm: T5LayerNorm,
    decoder_layers: Vec<T5DecoderLayer>,
    decoder_final_norm: T5LayerNorm,
    lm_head: Linear,
    decoder_start_token_id: u32,
    max_position_embeddings: usize,
    device: Device,
    dtype: DType,
}

impl T5ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6);

        let num_decoder_layers = cfg
            .extra
            .get("num_decoder_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_hidden_layers);

        let decoder_start_token_id = cfg
            .extra
            .get("decoder_start_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        // Shared embeddings
        let shared_embeddings =
            embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("shared"))?;

        // Encoder
        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let has_relative_bias = i == 0; // Only first layer computes bias
            encoder_layers.push(T5EncoderLayer::new(
                cfg,
                has_relative_bias,
                eps,
                vb.pp("encoder").pp(format!("block.{i}")),
            )?);
        }
        let encoder_final_norm =
            T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("encoder").pp("final_layer_norm"))?;

        // Decoder
        let mut decoder_layers = Vec::with_capacity(num_decoder_layers);
        for i in 0..num_decoder_layers {
            let has_relative_bias = i == 0;
            decoder_layers.push(T5DecoderLayer::new(
                cfg,
                has_relative_bias,
                eps,
                vb.pp("decoder").pp(format!("block.{i}")),
            )?);
        }
        let decoder_final_norm =
            T5LayerNorm::new(cfg.hidden_size, eps, vb.pp("decoder").pp("final_layer_norm"))?;

        // LM head (often tied to shared embeddings)
        let lm_head = if cfg.tie_word_embeddings {
            // Create from embedding weights (transpose)
            let weight = shared_embeddings.embeddings().clone();
            Linear::new(weight, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            shared_embeddings,
            encoder_layers,
            encoder_final_norm,
            decoder_layers,
            decoder_final_norm,
            lm_head,
            decoder_start_token_id,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Run the encoder stack.
    fn run_encoder(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.shared_embeddings.forward(input_ids)?;
        let mut position_bias: Option<Tensor> = None;

        for layer in &self.encoder_layers {
            let (h, bias) = layer.forward(&hidden, position_bias.as_ref())?;
            hidden = h;
            if position_bias.is_none() {
                position_bias = bias;
            }
        }

        self.encoder_final_norm.forward(&hidden)
    }

    /// Run the decoder stack with cross-attention to encoder output.
    fn run_decoder(
        &self,
        decoder_input_ids: &Tensor,
        encoder_hidden: &Tensor,
    ) -> Result<Tensor> {
        let (_, tgt_len) = decoder_input_ids.dims2()?;

        let mut hidden = self.shared_embeddings.forward(decoder_input_ids)?;

        // Causal mask for decoder self-attention
        let causal_mask = if tgt_len > 1 {
            Some(crate::layers::causal_mask(
                tgt_len,
                0,
                self.dtype,
                &self.device,
            )?)
        } else {
            None
        };

        let mut position_bias: Option<Tensor> = None;

        for layer in &self.decoder_layers {
            let (h, bias, _cross_k, _cross_v) = layer.forward(
                &hidden,
                encoder_hidden,
                causal_mask.as_ref(),
                position_bias.as_ref(),
                None,
            )?;
            hidden = h;
            if position_bias.is_none() {
                position_bias = bias;
            }
        }

        let hidden = self.decoder_final_norm.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }
}

impl ModelForEncoderDecoder for T5ForConditionalGeneration {
    fn encode(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<EncoderOutput> {
        let hidden = self.run_encoder(input_ids)?;
        EncoderOutput::new(hidden)
    }

    fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &EncoderOutput,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.run_decoder(decoder_input_ids, &encoder_output.hidden_states)
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_source_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn max_target_len(&self) -> usize {
        self.max_position_embeddings
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_t5_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::json!(1e-6),
        );
        extra.insert("d_kv".to_string(), serde_json::json!(32));
        extra.insert("d_ff".to_string(), serde_json::json!(128));
        extra.insert(
            "relative_attention_num_buckets".to_string(),
            serde_json::json!(32),
        );
        extra.insert(
            "relative_attention_max_distance".to_string(),
            serde_json::json!(128),
        );
        extra.insert("is_gated_act".to_string(), serde_json::json!(true));
        extra.insert("decoder_start_token_id".to_string(), serde_json::json!(0));
        extra.insert("num_decoder_layers".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["T5ForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 0,
            eos_token_id: 1,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn tiny_t5_standard_config() -> ModelConfig {
        let mut cfg = tiny_t5_config();
        cfg.extra
            .insert("is_gated_act".to_string(), serde_json::json!(false));
        cfg
    }

    // ─── Construction ─────────────────────────────────────────────────────

    #[test]
    fn test_t5_construction() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = T5ForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "T5 should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.encoder_layers.len(), 2);
        assert_eq!(model.decoder_layers.len(), 2);
    }

    #[test]
    fn test_t5_standard_ffn_construction() {
        let cfg = tiny_t5_standard_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = T5ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "T5 with standard FFN should construct");
    }

    // ─── RelativePositionBias ─────────────────────────────────────────────

    #[test]
    fn test_relative_position_bias_bidirectional() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let bias = RelativePositionBias::new(2, 32, 128, true, vb).unwrap();

        // Same position should be bucket 0
        assert_eq!(bias.relative_position_bucket(0), 0);

        // Small positive should be in upper half
        let bucket = bias.relative_position_bucket(1);
        assert!(bucket >= 16, "positive should be in upper half, got {bucket}");

        // Small negative should be in lower half
        let bucket = bias.relative_position_bucket(-1);
        assert!(bucket < 16, "negative should be in lower half, got {bucket}");
    }

    #[test]
    fn test_relative_position_bias_causal() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let bias = RelativePositionBias::new(2, 32, 128, false, vb).unwrap();

        // For causal, we negate and clamp to >= 0
        // relative_position = 0 -> bucket 0
        assert_eq!(bias.relative_position_bucket(0), 0);

        // Negative rel_pos (looking back) should map to positive buckets
        let bucket = bias.relative_position_bucket(-5);
        assert!(bucket > 0, "looking back should have positive bucket");

        // Positive rel_pos (looking forward, forbidden in causal) -> bucket 0
        assert_eq!(bias.relative_position_bucket(5), 0);
    }

    #[test]
    fn test_relative_position_bias_compute_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let num_heads = 4;

        let bias = RelativePositionBias::new(num_heads, 32, 128, true, vb).unwrap();

        let result = bias.compute_bias(5, 5, &device).unwrap();
        assert_eq!(result.dims(), &[1, num_heads, 5, 5]);
    }

    #[test]
    fn test_relative_position_bias_asymmetric() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let bias = RelativePositionBias::new(2, 32, 128, true, vb).unwrap();

        let result = bias.compute_bias(3, 7, &device).unwrap();
        assert_eq!(result.dims(), &[1, 2, 3, 7]);
    }

    // ─── T5LayerNorm ──────────────────────────────────────────────────────

    #[test]
    fn test_t5_layer_norm_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let norm = T5LayerNorm::new(64, 1e-6, vb).unwrap();
        let input = Tensor::ones((2, 5, 64), DType::F32, &device).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 5, 64]);
    }

    // ─── T5Ffn ────────────────────────────────────────────────────────────

    #[test]
    fn test_t5_gated_ffn_shape() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let ffn = T5Ffn::new(&cfg, vb).unwrap();
        assert!(ffn.is_gated);

        let input = Tensor::zeros((2, 5, 64), DType::F32, &device).unwrap();
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 5, 64]);
    }

    #[test]
    fn test_t5_standard_ffn_shape() {
        let cfg = tiny_t5_standard_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let ffn = T5Ffn::new(&cfg, vb).unwrap();
        assert!(!ffn.is_gated);

        let input = Tensor::zeros((2, 5, 64), DType::F32, &device).unwrap();
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 5, 64]);
    }

    // ─── Encoder ──────────────────────────────────────────────────────────

    #[test]
    fn test_t5_encode_shape() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 10), DType::U32, &device).unwrap();
        let encoder_output = model.encode(&input_ids, None).unwrap();

        assert_eq!(encoder_output.src_len, 10);
        assert_eq!(
            encoder_output.hidden_states.dims(),
            &[2, 10, cfg.hidden_size]
        );
    }

    #[test]
    fn test_t5_encode_single_token() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let encoder_output = model.encode(&input_ids, None).unwrap();

        assert_eq!(encoder_output.src_len, 1);
    }

    // ─── Decoder ──────────────────────────────────────────────────────────

    #[test]
    fn test_t5_decode_shape() {
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 32,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let block_table = BlockTable::new(cache_config.block_size);

        let src_ids = Tensor::zeros((1, 5), DType::U32, &device).unwrap();
        let encoder_output = model.encode(&src_ids, None).unwrap();

        let decoder_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let logits = model
            .decode(&decoder_ids, &encoder_output, 0, &mut kv_cache_mgr, &block_table, &[])
            .unwrap();

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    // ─── Full Forward ─────────────────────────────────────────────────────

    #[test]
    fn test_t5_forward_full() {
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 32,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let block_table = BlockTable::new(cache_config.block_size);

        let encoder_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let decoder_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();

        let (logits, encoder_output) = model
            .forward(
                &encoder_ids,
                &decoder_ids,
                None,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &[],
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        assert_eq!(encoder_output.src_len, 8);
    }

    // ─── Trait Methods ────────────────────────────────────────────────────

    #[test]
    fn test_t5_decoder_start_token() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert_eq!(model.decoder_start_token_id(), 0);
    }

    #[test]
    fn test_t5_device() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_t5_max_source_and_target_len() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert_eq!(model.max_source_len(), 512);
        assert_eq!(model.max_target_len(), 512);
    }

    // ─── Batch Processing ─────────────────────────────────────────────────

    #[test]
    fn test_t5_batch_encode() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let batch_size = 4;
        let src_len = 6;
        let input_ids = Tensor::zeros((batch_size, src_len), DType::U32, &device).unwrap();
        let encoder_output = model.encode(&input_ids, None).unwrap();

        assert_eq!(
            encoder_output.hidden_states.dims(),
            &[batch_size, src_len, cfg.hidden_size]
        );
    }

    #[test]
    fn test_t5_tied_embeddings() {
        let cfg = tiny_t5_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        // With tied embeddings, lm_head weight should be shared
        assert_eq!(model.lm_head.weight().dims(), &[cfg.vocab_size, cfg.hidden_size]);
    }

    #[test]
    fn test_t5_untied_embeddings() {
        let mut cfg = tiny_t5_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "T5 with untied embeddings should construct");
    }

    // ─── Position Bias Propagation ────────────────────────────────────────

    #[test]
    fn test_t5_encoder_only_first_layer_has_bias() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(
            model.encoder_layers[0].self_attention.relative_bias.is_some(),
            "first encoder layer should have relative bias"
        );
        assert!(
            model.encoder_layers[1].self_attention.relative_bias.is_none(),
            "second encoder layer should NOT have relative bias"
        );
    }

    #[test]
    fn test_t5_decoder_only_first_layer_has_bias() {
        let cfg = tiny_t5_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = T5ForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(
            model.decoder_layers[0].self_attention.relative_bias.is_some(),
            "first decoder layer should have relative bias"
        );
        assert!(
            model.decoder_layers[1].self_attention.relative_bias.is_none(),
            "second decoder layer should NOT have relative bias"
        );
    }

    // ─── Send bound ───────────────────────────────────────────────────────

    #[test]
    fn test_t5_is_send() {
        fn assert_send<T: Send + 'static>() {}
        assert_send::<T5ForConditionalGeneration>();
    }
}
