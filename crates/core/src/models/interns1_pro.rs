//! InternS1Pro vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: Qwen3VisionTransformer (identical to Qwen3-VL)
//! - Language model: transformer with FoPE (Fourier Rotary Embeddings) +
//!   QK-normalization + sparse MoE feed-forward blocks
//!
//! FoPE replaces standard RoPE with a learned linear transformation of the
//! base sinusoidal frequencies. Learnable cos_coef and sin_coef matrices
//! project filtered inv_freq components into the final positional encoding.
//!
//! Weight mapping (HF checkpoint → this implementation):
//! - `model.visual.*`          → vision encoder
//! - `model.language_model.*`  → LLM layers/embed/norm/rotary_emb
//! - `lm_head.*`               → language model head
//!
//! Reference: vllm/model_executor/models/interns1_pro.py

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, causal_mask, paged_attention, rms_norm, RmsNorm};
use crate::multimodal::MultimodalInputs;

use super::qwen3_vl::{Qwen3VLConfig, Qwen3VisionTransformer};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return (top_k_values, top_k_indices) along the last dimension.
fn topk_last_dim(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let last = x.dims().len() - 1;
    let n = x.dim(last)?;
    let k = k.min(n);
    let sorted_idx = x.arg_sort_last_dim(false)?;
    let top_idx = sorted_idx.narrow(last, 0, k)?.contiguous()?;
    let top_vals = x.contiguous()?.gather(&top_idx, last)?;
    Ok((top_vals, top_idx))
}

/// Neox-style rotate_half: [x1, x2] → [-x2, x1].
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(candle_core::D::Minus1)? / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
    let x2 = x.narrow(candle_core::D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
}

// ─── FoPE (Fourier Rotary Embeddings) ────────────────────────────────────────

/// Fourier Positional Embedding (FoPE).
///
/// Replaces standard RoPE with a learned linear transform of filtered
/// sinusoidal frequencies. The cos_coef and sin_coef parameters are loaded
/// from the checkpoint and used to project base frequencies into the
/// final position encoding.
///
/// Unlike standard RoPE, FoPE only uses a subset of the rotary frequencies
/// (controlled by `num_inv_freq`) and pads remaining dimensions with ones,
/// resulting in a partial rotation even at positions beyond the trained range.
struct FoPE {
    /// Precomputed position encoding: [max_pos, 2*head_size] (non-sep-head)
    /// or [max_pos, num_kv_heads, 2*head_size] (sep-head).
    cos_sin_cache: Tensor,
    fope_sep_head: bool,
    num_kv_heads: usize,
    head_size: usize,
}

impl FoPE {
    #[allow(clippy::too_many_arguments)]
    fn new(
        head_size: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        num_kv_heads: usize,
        fope_sep_head: bool,
        num_inv_freq: Option<usize>,
        cos_coef: &Tensor,
        sin_coef: &Tensor,
    ) -> Result<Self> {
        let cos_sin_cache = Self::build_cache(
            head_size,
            max_position_embeddings,
            rope_theta,
            num_kv_heads,
            fope_sep_head,
            num_inv_freq,
            cos_coef,
            sin_coef,
        )?;

        Ok(Self {
            cos_sin_cache,
            fope_sep_head,
            num_kv_heads,
            head_size,
        })
    }

    /// Compute the subset of rotary inverse frequencies used by FoPE.
    fn compute_inv_freq(
        rotary_dim: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        num_inv_freq: Option<usize>,
    ) -> Vec<f32> {
        let raw: Vec<f32> = (0..rotary_dim / 2)
            .map(|i| 1.0 / (rope_theta as f32).powf((2 * i) as f32 / rotary_dim as f32))
            .collect();

        if let Some(n) = num_inv_freq {
            raw.into_iter().take(n).collect()
        } else {
            let threshold = 2.0 * std::f32::consts::PI / max_position_embeddings as f32;
            raw.into_iter().filter(|&f| f > threshold).collect()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_cache(
        head_size: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        num_kv_heads: usize,
        fope_sep_head: bool,
        num_inv_freq: Option<usize>,
        cos_coef: &Tensor,
        sin_coef: &Tensor,
    ) -> Result<Tensor> {
        let inv_freq =
            Self::compute_inv_freq(head_size, rope_theta, max_position_embeddings, num_inv_freq);
        let input_dim = inv_freq.len().max(1);

        // freqs[t, i] = t * inv_freq[i] → [max_pos, input_dim]
        let mut freqs_data = vec![0.0f32; max_position_embeddings * input_dim];
        for t in 0..max_position_embeddings {
            for (i, &f) in inv_freq.iter().enumerate() {
                freqs_data[t * input_dim + i] = t as f32 * f;
            }
        }
        let device = cos_coef.device();
        let dtype = cos_coef.dtype();
        let freqs = Tensor::from_vec(freqs_data, (max_position_embeddings, input_dim), device)?
            .to_dtype(dtype)?;

        let pos_cos = freqs.cos()?; // [T, input_dim]
        let pos_sin = freqs.sin()?;

        // Apply learned linear transform.
        let (cos_out, sin_out) = if fope_sep_head {
            // cos_coef: [num_kv_heads, input_dim, output_dim]
            // einsum "htD, hDd -> thd": per-head position transform.
            let pos_cos_e = pos_cos.unsqueeze(0)?.broadcast_as((
                num_kv_heads,
                max_position_embeddings,
                input_dim,
            ))?;
            let pos_sin_e = pos_sin.unsqueeze(0)?.broadcast_as((
                num_kv_heads,
                max_position_embeddings,
                input_dim,
            ))?;
            // [H, T, input_dim] @ [H, input_dim, output_dim] → [H, T, output_dim]
            let cos_h = pos_cos_e.matmul(cos_coef)?;
            let sin_h = pos_sin_e.matmul(sin_coef)?;
            // Permute to [T, H, output_dim]
            (cos_h.permute((1, 0, 2))?, sin_h.permute((1, 0, 2))?)
        } else {
            // cos_coef: [input_dim, output_dim]
            // [T, input_dim] @ [input_dim, output_dim] → [T, output_dim]
            (pos_cos.matmul(cos_coef)?, pos_sin.matmul(sin_coef)?)
        };

        let out_dim = cos_out.dim(candle_core::D::Minus1)?;
        let half = head_size / 2;

        // Pad to head_size/2 with ones, then double to head_size.
        // Both cos and sin are padded with value=1 (FoPE convention — intentional;
        // the padded dimensions apply a non-identity but fixed transformation).
        let (cos_full, sin_full) = if out_dim < half {
            let pad_size = half - out_dim;
            let ones_shape: Vec<usize> = {
                let mut s = cos_out.dims().to_vec();
                *s.last_mut().unwrap() = pad_size;
                s
            };
            let ones = Tensor::ones(ones_shape, dtype, device)?;
            let cos_padded = Tensor::cat(&[&cos_out, &ones], candle_core::D::Minus1)?;
            let sin_padded = Tensor::cat(&[&sin_out, &ones], candle_core::D::Minus1)?;
            (
                Tensor::cat(&[&cos_padded, &cos_padded], candle_core::D::Minus1)?,
                Tensor::cat(&[&sin_padded, &sin_padded], candle_core::D::Minus1)?,
            )
        } else {
            // out_dim already >= half (no extra padding needed)
            (
                Tensor::cat(&[&cos_out, &cos_out], candle_core::D::Minus1)?,
                Tensor::cat(&[&sin_out, &sin_out], candle_core::D::Minus1)?,
            )
        };

        // Cache: cat(cos, sin) → [T, 2*head_size] or [T, H, 2*head_size]
        Tensor::cat(&[&cos_full, &sin_full], candle_core::D::Minus1)
    }

    /// Apply FoPE to query and key during prefill.
    ///
    /// q: [batch, num_heads, seq_len, head_dim]  (BHLD)
    /// k: [batch, num_kv_heads, seq_len, head_dim]
    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let positions: Vec<u32> =
            (seqlen_offset as u32..(seqlen_offset + seq_len) as u32).collect();
        let pos_t = Tensor::from_vec(positions, seq_len, q.device())?;
        self.apply_positions(q, k, &pos_t)
    }

    fn apply_decode_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let positions: Vec<u32> = seqlen_offsets.iter().map(|&s| s as u32).collect();
        let pos_t = Tensor::from_vec(positions, seqlen_offsets.len(), q.device())?;
        self.apply_positions(q, k, &pos_t)
    }

    fn apply_positions(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, num_heads, seq_len, hd) = q.dims4()?;
        let nkv = self.num_kv_heads;

        // Lookup: [seq_len, 2*head_size] or [seq_len, num_kv_heads, 2*head_size]
        let cos_sin = self.cos_sin_cache.index_select(positions, 0)?;
        let half = self.head_size;
        let cos = cos_sin.narrow(candle_core::D::Minus1, 0, half)?;
        let sin = cos_sin.narrow(candle_core::D::Minus1, half, half)?;

        if self.fope_sep_head {
            // cos: [seq_len, num_kv_heads, head_size]
            // For GQA: groups = num_heads / num_kv_heads
            let groups = num_heads / nkv;

            // q: [b, nkv*groups, seq_len, hd] → [b*seq_len*nkv, groups, hd]
            let q_r = q
                .reshape((b, nkv, groups, seq_len, hd))?
                .permute((0, 3, 1, 2, 4))? // [b, seq_len, nkv, groups, hd]
                .reshape((b * seq_len * nkv, groups, hd))?;

            // cos: [seq_len, nkv, head_size] → [b*seq_len*nkv, 1, head_size]
            let cos_e = cos
                .unsqueeze(0)?
                .broadcast_as((b, seq_len, nkv, half))?
                .reshape((b * seq_len * nkv, 1, half))?;
            let sin_e = sin
                .unsqueeze(0)?
                .broadcast_as((b, seq_len, nkv, half))?
                .reshape((b * seq_len * nkv, 1, half))?;

            let q_rot = rotate_half(&q_r)?;
            let q_scaled = (q_r.broadcast_mul(&cos_e)? + q_rot.broadcast_mul(&sin_e)?)?;
            let q_new = q_scaled
                .reshape((b, seq_len, nkv, groups, hd))?
                .permute((0, 2, 3, 1, 4))? // [b, nkv, groups, seq_len, hd]
                .reshape((b, num_heads, seq_len, hd))?;

            // k: [b, nkv, seq_len, hd] → [b*seq_len*nkv, 1, hd]
            let k_r = k
                .permute((0, 2, 1, 3))?
                .reshape((b * seq_len * nkv, 1, hd))?;
            let k_rot = rotate_half(&k_r)?;
            let k_new = (k_r.broadcast_mul(&cos_e)? + k_rot.broadcast_mul(&sin_e)?)?
                .reshape((b, seq_len, nkv, hd))?
                .permute((0, 2, 1, 3))?; // [b, nkv, seq_len, hd]

            Ok((q_new, k_new))
        } else {
            // cos: [seq_len, head_size]
            // Broadcast over batch and heads.
            let cos_q = cos
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((b, num_heads, seq_len, hd))?;
            let sin_q = sin
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((b, num_heads, seq_len, hd))?;
            let cos_k = cos
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((b, nkv, seq_len, hd))?;
            let sin_k = sin
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((b, nkv, seq_len, hd))?;

            let q_new = ((q * &cos_q)? + (rotate_half(q)? * &sin_q)?)?;
            let k_new = ((k * &cos_k)? + (rotate_half(k)? * &sin_k)?)?;
            Ok((q_new, k_new))
        }
    }
}

// ─── Dense MLP ───────────────────────────────────────────────────────────────

struct DenseMlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    half: usize,
}

impl DenseMlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // gate_up_proj packs [gate_proj | up_proj] into one matrix.
        let gate_up_proj =
            candle_nn::linear_no_bias(hidden_size, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            half: intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.half)?;
        let up = gate_up.narrow(candle_core::D::Minus1, self.half, self.half)?;
        self.down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

// ─── MoE ─────────────────────────────────────────────────────────────────────

struct MoEExpert {
    gate_up_proj: Linear,
    down_proj: Linear,
    half: usize,
}

impl MoEExpert {
    fn new(hidden_size: usize, moe_intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = candle_nn::linear_no_bias(
            hidden_size,
            2 * moe_intermediate_size,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj =
            candle_nn::linear_no_bias(moe_intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            half: moe_intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let gate = gate_up.narrow(candle_core::D::Minus1, 0, self.half)?;
        let up = gate_up.narrow(candle_core::D::Minus1, self.half, self.half)?;
        self.down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

/// MoE sparse block with optional grouped top-k routing.
///
/// When `num_groups > 0`, the experts are logically partitioned into groups
/// and `top_k / num_groups` experts are selected per group, then ids are
/// globally adjusted. This matches InternS1Pro's `_custom_routing_function`.
struct SparseMoeBlock {
    gate: Linear,
    experts: Vec<MoEExpert>,
    top_k: usize,
    num_groups: usize,
    renormalize: bool,
}

impl SparseMoeBlock {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(16);
        let top_k = cfg.num_experts_per_tok().unwrap_or(4);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let renormalize = cfg.norm_topk_prob();
        let num_groups = cfg
            .extra
            .get("router_n_groups")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let gate = candle_nn::linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_e = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MoEExpert::new(
                hidden_size,
                moe_intermediate_size,
                vb_e.pp(i),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            top_k,
            num_groups,
            renormalize,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        let logits = self.gate.forward(&xs_2d)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&logits)?;

        // Grouped top-k routing: select top-k/G experts per group.
        let (topk_weights, topk_ids) = if self.num_groups > 0
            && self.top_k.is_multiple_of(self.num_groups)
        {
            let num_experts = self.experts.len();
            let group_size = num_experts / self.num_groups;
            let per_group_k = self.top_k / self.num_groups;

            // Reshape: [T, num_groups, group_size]
            let rw_grouped = routing_weights.reshape((num_tokens, self.num_groups, group_size))?;
            // Top-k within each group: [T, num_groups, per_group_k]
            let (gw, gids) = topk_last_dim(&rw_grouped, per_group_k)?;

            // Convert group-local indices to global expert indices.
            let offsets_data: Vec<u32> = (0..self.num_groups)
                .flat_map(|g| std::iter::repeat_n((g * group_size) as u32, per_group_k))
                .collect();
            let offsets = Tensor::from_vec(
                offsets_data,
                (1usize, self.num_groups, per_group_k),
                xs.device(),
            )?
            .broadcast_as((num_tokens, self.num_groups, per_group_k))?;

            let global_ids = (gids + offsets)?;
            let gw_flat = gw.reshape((num_tokens, self.top_k))?;
            let ids_flat = global_ids.reshape((num_tokens, self.top_k))?;
            (gw_flat, ids_flat)
        } else {
            topk_last_dim(&routing_weights, self.top_k)?
        };

        let topk_weights = if self.renormalize {
            let sum = topk_weights.sum_keepdim(candle_core::D::Minus1)?;
            topk_weights.broadcast_div(&sum)?
        } else {
            topk_weights
        };

        let weights_f32 = topk_weights.to_dtype(DType::F32)?;

        let mut output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;
        for token_idx in 0..num_tokens {
            let token_in = xs_2d.narrow(0, token_idx, 1)?;
            let ids: Vec<u32> = topk_ids
                .narrow(0, token_idx, 1)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let weights: Vec<f32> = weights_f32
                .narrow(0, token_idx, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let mut token_out = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;
            for (k, &eid) in ids.iter().enumerate() {
                let eid = eid as usize;
                if eid < self.experts.len() {
                    let out = self.experts[eid].forward(&token_in)?;
                    token_out = (token_out + out.affine(weights[k] as f64, 0.0)?)?;
                }
            }

            let idx_t = Tensor::new(&[token_idx as u32], xs.device())?;
            output = output.index_add(&idx_t, &token_out, 0)?;
        }

        output.reshape(orig_shape)
    }
}

enum MlpVariant {
    Dense(DenseMlp),
    Sparse(SparseMoeBlock),
}

impl MlpVariant {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Sparse(m) => m.forward(xs),
        }
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct InternS1ProAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    fope: Arc<FoPE>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl InternS1ProAttention {
    fn new(cfg: &ModelConfig, fope: Arc<FoPE>, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj =
            candle_nn::linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            fope,
            num_heads,
            num_kv_heads,
            head_dim,
        })
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
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        let (q, k) = self.fope.apply(&q, &k, seqlen_offset)?;

        paged_attention(
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
        )
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

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        let seqlen_offsets: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
        let (q, k) = self.fope.apply_decode_batch(&q, &k, &seqlen_offsets)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let max_blocks = sequences
                .iter()
                .map(|s| s.block_ids.len())
                .max()
                .unwrap_or(1);
            let mut bt_data = vec![0u32; batch_size * max_blocks];
            for (i, seq) in sequences.iter().enumerate() {
                for (j, &bid) in seq.block_ids.iter().enumerate() {
                    bt_data[i * max_blocks + j] = bid as u32;
                }
            }
            let block_tables = Tensor::from_vec(bt_data, (batch_size, max_blocks), q.device())?;
            let seq_lens_data: Vec<u32> = sequences
                .iter()
                .map(|s| (s.seqlen_offset + 1) as u32)
                .collect();
            let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
            let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
            )?;
            self.o_proj.forward(&attn_output)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;
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
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct InternS1ProDecoderLayer {
    self_attn: InternS1ProAttention,
    mlp: MlpVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl InternS1ProDecoderLayer {
    fn new(cfg: &ModelConfig, layer_idx: usize, fope: Arc<FoPE>, vb: VarBuilder) -> Result<Self> {
        let self_attn = InternS1ProAttention::new(cfg, fope, vb.pp("self_attn"))?;

        let decoder_sparse_step = cfg.decoder_sparse_step().unwrap_or(1);
        let mlp_only_layers = cfg.mlp_only_layers();
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe = !mlp_only_layers.contains(&layer_idx)
            && num_experts > 0
            && (layer_idx + 1).is_multiple_of(decoder_sparse_step);

        let mlp_vb = vb.pp("mlp");
        let mlp = if is_moe {
            MlpVariant::Sparse(SparseMoeBlock::new(cfg, mlp_vb)?)
        } else {
            MlpVariant::Dense(DenseMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp_vb,
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
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward(
            &hidden,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let hidden = (hidden + residual)?;
        let residual = &hidden;
        let hidden = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
        residual + hidden
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward_decode_batch(
            &hidden,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let hidden = (hidden + residual)?;
        let residual = &hidden;
        let hidden = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&hidden)?)?;
        residual + hidden
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// FoPE configuration parsed from `rope_scaling` in the model config.
struct FoPEConfig {
    num_inv_freq: Option<usize>,
    fope_sep_head: bool,
}

impl FoPEConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let rope_scaling = cfg.extra.get("rope_scaling");
        let num_inv_freq = rope_scaling
            .and_then(|rs| rs.get("num_inv_freq"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let fope_sep_head = rope_scaling
            .and_then(|rs| rs.get("fope_sep_head"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Self {
            num_inv_freq,
            fope_sep_head,
        }
    }
}

/// InternS1Pro vision-language model.
///
/// Uses `Qwen3VisionTransformer` for vision encoding and a custom LLM backbone
/// with FoPE positional embeddings (replacing standard RoPE) and sparse MoE
/// feed-forward layers. Image embeddings are expected to be pre-encoded
/// and passed through `MultimodalInputs.image_embeddings`.
pub struct InternS1ProForConditionalGeneration {
    #[allow(dead_code)]
    visual: Qwen3VisionTransformer,
    embed_tokens: Embedding,
    layers: Vec<InternS1ProDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl InternS1ProForConditionalGeneration {
    pub fn new(cfg: &Qwen3VLConfig, vb: VarBuilder) -> Result<Self> {
        let mc = &cfg.model_config;
        let vb_visual = vb.pp("model").pp("visual");
        let vb_llm = vb.pp("model").pp("language_model");

        let visual = Qwen3VisionTransformer::new(&cfg.vision_config, vb_visual)?;

        let embed_tokens =
            candle_nn::embedding(mc.vocab_size, mc.hidden_size, vb_llm.pp("embed_tokens"))?;

        // Build FoPE from shared rotary_emb weights.
        let fope_cfg = FoPEConfig::from_model_config(mc);
        let head_dim = mc.head_dim;
        let num_kv_heads = mc.num_key_value_heads;
        // inv_dim: number of filtered frequencies = output dimension of cos/sin coef.
        let inv_dim = fope_cfg.num_inv_freq.unwrap_or(head_dim / 2).max(1);

        let vb_rope = vb_llm.pp("rotary_emb");
        let (cos_coef, sin_coef) = if fope_cfg.fope_sep_head {
            let shape = (num_kv_heads, inv_dim, inv_dim);
            (
                vb_rope.get(shape, "cos_coef")?,
                vb_rope.get(shape, "sin_coef")?,
            )
        } else {
            let shape = (inv_dim, inv_dim);
            (
                vb_rope.get(shape, "cos_coef")?,
                vb_rope.get(shape, "sin_coef")?,
            )
        };

        let fope = Arc::new(FoPE::new(
            head_dim,
            mc.max_position_embeddings,
            mc.rope_theta,
            num_kv_heads,
            fope_cfg.fope_sep_head,
            fope_cfg.num_inv_freq,
            &cos_coef,
            &sin_coef,
        )?);

        let vb_layers = vb_llm.pp("layers");
        let mut layers = Vec::with_capacity(mc.num_hidden_layers);
        for i in 0..mc.num_hidden_layers {
            layers.push(InternS1ProDecoderLayer::new(
                mc,
                i,
                Arc::clone(&fope),
                vb_layers.pp(i),
            )?);
        }

        let norm = rms_norm(mc.hidden_size, mc.rms_norm_eps, vb_llm.pp("norm"))?;

        let lm_head = if mc.tie_word_embeddings {
            let w = vb_llm
                .pp("embed_tokens")
                .get((mc.vocab_size, mc.hidden_size), "weight")?;
            Linear::new(w, None)
        } else {
            candle_nn::linear_no_bias(mc.hidden_size, mc.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            visual,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let qwen3vl_cfg = Qwen3VLConfig::from_model_config(cfg);
        Self::new(&qwen3vl_cfg, vb)
    }

    /// Encode raw image patches through the vision transformer.
    ///
    /// Input: `[num_patches, patch_input_dim]` (pre-flattened patches)
    /// Output: `[num_tokens, hidden_size]` (after PatchMerger)
    pub fn encode_image(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        self.visual.forward(patches, grid_h, grid_w)
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = position / seq_len;
            let start_pos = position % seq_len;
            for (i, row) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = row.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    fn run_llm(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask,
                seqlen_offset,
                kv_cache_mgr,
                i,
                block_table,
                slot_mapping,
            )?;
        }
        self.lm_head.forward(&self.norm.forward(&xs)?)
    }
}

impl crate::engine::ModelForward for InternS1ProForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let xs = self.embed_tokens.forward(input_ids)?;
        self.run_llm(
            xs,
            attention_mask.as_ref(),
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
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, i)?;
        }
        self.lm_head.forward(&self.norm.forward(&xs)?)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let text_embeddings = self.embed_tokens.forward(input_ids)?;
        let xs = if let Some(mm) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm)?
        } else {
            text_embeddings
        };

        self.run_llm(
            xs,
            attention_mask.as_ref(),
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use crate::config::ModelConfig;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
    use crate::multimodal::{MultimodalInputs, ProcessedImage};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 2,
                "num_heads": 2,
                "hidden_size": 32,
                "embed_dim": 32,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "out_hidden_size": 64,
                "mlp_ratio": 4.0,
                "spatial_merge_size": 2
            }),
        );
        extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({"fope_sep_head": false, "num_inv_freq": 4}),
        );
        extra.insert("image_token_id".to_string(), serde_json::json!(10));
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("decoder_sparse_step".to_string(), serde_json::json!(2));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("router_n_groups".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["InternS1ProForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 64,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn make_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
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
    fn test_interns1_pro_new() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ProForConditionalGeneration::from_model_config(&cfg, vb);
        assert!(
            model.is_ok(),
            "Model construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_fope_build_non_sep() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_size = 16;
        let max_pos = 32;
        let num_kv_heads = 2;
        let inv_dim = 4;

        let cos_coef = Tensor::zeros((inv_dim, inv_dim), dtype, &device).unwrap();
        let sin_coef = Tensor::zeros((inv_dim, inv_dim), dtype, &device).unwrap();

        let fope = FoPE::new(
            head_size,
            max_pos,
            10000.0,
            num_kv_heads,
            false,
            Some(inv_dim),
            &cos_coef,
            &sin_coef,
        )
        .unwrap();
        // Non-sep-head cache: [max_pos, 2*head_size]
        assert_eq!(fope.cos_sin_cache.dims(), &[max_pos, 2 * head_size]);
    }

    #[test]
    fn test_fope_build_sep_head() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let head_size = 16;
        let max_pos = 32;
        let num_kv_heads = 2;
        let inv_dim = 4;

        let cos_coef = Tensor::zeros((num_kv_heads, inv_dim, inv_dim), dtype, &device).unwrap();
        let sin_coef = Tensor::zeros((num_kv_heads, inv_dim, inv_dim), dtype, &device).unwrap();

        let fope = FoPE::new(
            head_size,
            max_pos,
            10000.0,
            num_kv_heads,
            true,
            Some(inv_dim),
            &cos_coef,
            &sin_coef,
        )
        .unwrap();
        // Sep-head cache: [max_pos, num_kv_heads, 2*head_size]
        assert_eq!(
            fope.cos_sin_cache.dims(),
            &[max_pos, num_kv_heads, 2 * head_size]
        );
    }

    #[test]
    fn test_text_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ProForConditionalGeneration::from_model_config(&cfg, vb).unwrap();

        let cache_cfg = make_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let slot_mapping: Vec<usize> = (0..4).collect();

        let out = model.forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping);
        assert!(out.is_ok(), "forward failed: {:?}", out.err());
        assert_eq!(out.unwrap().dims(), &[1, 4, 256]);
    }

    #[test]
    fn test_multimodal_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternS1ProForConditionalGeneration::from_model_config(&cfg, vb).unwrap();

        let cache_cfg = make_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        let image_tok = 10u32; // small ID to stay within test vocab_size=256
        let input_ids = Tensor::new(&[[0u32, image_tok, image_tok, 0u32]], &device).unwrap();

        // Pre-encoded image embeddings: [num_tokens=2, hidden=64]
        let img_emb = Tensor::zeros((2, 64), DType::F32, &device).unwrap();
        let processed = ProcessedImage::new(img_emb, 2);
        let mm_inputs =
            MultimodalInputs::with_images(vec![image_tok, image_tok], vec![(1, processed)]);

        let slot_mapping: Vec<usize> = (0..4).collect();
        let out = model.forward_multimodal(
            &input_ids,
            Some(&mm_inputs),
            0,
            &mut kv_cache,
            &block_table,
            &slot_mapping,
        );
        assert!(out.is_ok(), "multimodal forward failed: {:?}", out.err());
        assert_eq!(out.unwrap().dims(), &[1, 4, 256]);
    }
}
