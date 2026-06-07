//! Quantized Gemma 4 model implementation.
//!
//! Mirrors `crates/core/src/models/gemma4.rs` but routes all linear layers
//! through the `QuantizedWeightLoader` abstraction so checkpoints quantized
//! with AWQ, BitsAndBytes, GPTQ, FP8, GGUF, etc. load without going through
//! the full-precision path.
//!
//! Gemma 4 specifics preserved in the quantized path:
//! - `Gemma4RmsNorm` with `(1 + weight)` offset scaling
//! - `UnweightedRmsNorm` for V and router normalisation
//! - PLE pipeline (per-layer embeddings + projection + per-layer norm)
//! - MoE block running in parallel with dense MLP (custom Gemma 4 router:
//!   softmax-all → top-k → renormalize → fold `per_expert_scale`)
//! - QKV norms with per-head application
//! - `k_eq_v` laptop variant where V reuses K weights on full-attention layers
//! - Sliding / global attention alternation with separate RoPE base for local
//! - Final logit soft capping
//! - Per-layer `layer_scalar` multiplier
//!
//! Single-GPU MVP: no tensor-parallel variant. The TP path lives in the full
//! precision `gemma4.rs` and is not yet wired for quantized weights.
//!
//! Reference: `crates/core/src/models/gemma4.rs` (full precision) and
//! `crates/core/src/models/gemma3_quantized.rs` (closest quantized pattern).

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Gemma4 RMSNorm ────────────────────────────────────────────────────────
//
// Plain `weight * x` — Gemma 4 Python uses `RMSNorm`, not the Gemma 2/3
// `GemmaRMSNorm` with `(1 + weight)` offset. Using the offset here would
// silently break every normalization in the model.

// `Gemma4RmsNorm` is `Standard` and `UnweightedRmsNorm` is `Unweighted` —
// both implemented by `crate::layers::RmsNorm` since Phase 3a.

type Gemma4RmsNorm = crate::layers::RmsNorm;

#[inline]
fn gemma4_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<Gemma4RmsNorm> {
    crate::layers::rms_norm(size, eps, vb)
}

type UnweightedRmsNorm = crate::layers::RmsNorm;

#[inline]
fn unweighted_rms_norm(eps: f64) -> UnweightedRmsNorm {
    crate::layers::rms_norm_unweighted(eps)
}

// ─── Soft capping helper ───────────────────────────────────────────────────

fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

// ─── Sliding window causal mask ────────────────────────────────────────────

/// Prefill activations are padded up to a multiple of this so candle's
/// per-shape CUDA buffer cache sees at most `max_model_len / 32` distinct
/// prefill shapes instead of one per prompt length. Each distinct Gemma 4
/// prefill shape pins ~30 MB of cached activation buffers; unbounded
/// distinct lengths slowly exhaust an 8 GB card on a long-running server.
const PREFILL_LEN_BUCKET: usize = 32;

/// Bucketed prefill length: `real_len` rounded up to the bucket, capped at
/// `max_positions` (RoPE tables are precomputed only that far) and never
/// below `real_len`.
fn prefill_bucket_len(real_len: usize, max_positions: usize) -> usize {
    if real_len <= 1 {
        return real_len;
    }
    real_len
        .div_ceil(PREFILL_LEN_BUCKET)
        .saturating_mul(PREFILL_LEN_BUCKET)
        .min(max_positions)
        .max(real_len)
}

/// Attention mask for a bucket-padded prefill.
///
/// Rows are query positions `seqlen_offset + i` for `i < q_len` (the first
/// `real_q` rows are real tokens, the rest right-padding); columns are KV
/// positions `j < kv_len` (`real_kv` real cache entries, the rest zero
/// padding). Real rows follow the causal (+ optional sliding-window) rule
/// over real columns and mask everything else. Padding rows attend solely
/// to column 0 — their outputs are discarded, but an all-`-inf` row would
/// turn softmax into NaN, which the next layer's row-mixing ops could
/// surface.
#[allow(clippy::too_many_arguments)]
fn bucketed_prefill_mask(
    q_len: usize,
    kv_len: usize,
    real_q: usize,
    real_kv: usize,
    seqlen_offset: usize,
    window_size: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![f32::NEG_INFINITY; q_len * kv_len];
    for i in 0..q_len {
        if i >= real_q {
            mask[i * kv_len] = 0.0;
            continue;
        }
        let query_pos = seqlen_offset + i;
        for j in 0..real_kv.min(kv_len) {
            let is_causal = j <= query_pos;
            let is_in_window = window_size.is_none_or(|w| query_pos < w || j > query_pos - w);
            if is_causal && is_in_window {
                mask[i * kv_len + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

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

// ─── Extra config extraction (duplicated from gemma4.rs for independence) ─

struct Gemma4ExtraConfig {
    attn_logit_softcap: Option<f64>,
    final_logit_softcap: Option<f64>,
    sliding_window_pattern: usize,
    /// RoPE base for sliding-attention layers (from
    /// `rope_parameters.sliding_attention.rope_theta`, legacy fallback
    /// `rope_local_base_freq`, then `cfg.rope_theta`).
    rope_theta_local: f64,
    /// RoPE base for full-attention layers (from
    /// `rope_parameters.full_attention.rope_theta`). For Gemma 4 E2B this
    /// is 1e6 vs 10000 for sliding.
    rope_theta_full: f64,
    /// Partial rotary factor for full-attention layers (default 1.0).
    /// Gemma 4 E2B uses 0.25.
    partial_rotary_factor_full: f64,
    /// `rope_type` for full attention — `"proportional"` selects Gemma 4's
    /// head_dim-denominator variant.
    rope_type_full: String,
    layer_types: Vec<String>,
    enable_moe_block: bool,
    num_experts: usize,
    top_k_experts: usize,
    moe_intermediate_size: usize,
    hidden_size_per_layer_input: usize,
    vocab_size_per_layer_input: usize,
    attention_k_eq_v: bool,
    num_kv_shared_layers: usize,
    use_double_wide_mlp: bool,
    global_head_dim: Option<usize>,
    /// KV heads on full-attention layers (released 12B/31B use fewer than
    /// sliding layers; 12B: 1 vs 8). `None` → homogeneous E2B case.
    num_global_key_value_heads: Option<usize>,
}

impl Gemma4ExtraConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
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

        let rope_params = cfg.extra.get("rope_parameters");
        let rope_full = rope_params.and_then(|rp| rp.get("full_attention"));
        let rope_sliding = rope_params.and_then(|rp| rp.get("sliding_attention"));

        let rope_theta_local = rope_sliding
            .and_then(|sa| sa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("rope_local_base_freq")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(cfg.rope_theta);

        let rope_theta_full = rope_full
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        let partial_rotary_factor_full = rope_full
            .and_then(|fa| fa.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let rope_type_full = rope_full
            .and_then(|fa| fa.get("rope_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();

        let layer_types = cfg
            .extra
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("full_attention").to_string())
                    .collect()
            })
            .unwrap_or_else(|| {
                (0..cfg.num_hidden_layers)
                    .map(|i| {
                        if sliding_window_pattern > 0 && i.is_multiple_of(sliding_window_pattern) {
                            "sliding_attention".to_string()
                        } else {
                            "full_attention".to_string()
                        }
                    })
                    .collect()
            });

        let enable_moe_block = cfg
            .extra
            .get("enable_moe_block")
            .and_then(|v| v.as_bool())
            .or_else(|| {
                cfg.extra
                    .get("use_second_mlp_block")
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false);

        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let top_k_experts = cfg
            .extra
            .get("top_k_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let moe_intermediate_size = cfg
            .extra
            .get("moe_intermediate_size")
            .or_else(|| cfg.extra.get("expert_intermediate_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let hidden_size_per_layer_input = cfg
            .extra
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let vocab_size_per_layer_input = cfg
            .extra
            .get("vocab_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

        let attention_k_eq_v = cfg
            .extra
            .get("attention_k_eq_v")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let num_kv_shared_layers = cfg
            .extra
            .get("num_kv_shared_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let use_double_wide_mlp = cfg
            .extra
            .get("use_double_wide_mlp")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let global_head_dim = cfg
            .extra
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let num_global_key_value_heads = cfg
            .extra
            .get("num_global_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Self {
            attn_logit_softcap,
            final_logit_softcap,
            sliding_window_pattern,
            rope_theta_local,
            rope_theta_full,
            partial_rotary_factor_full,
            rope_type_full,
            layer_types,
            enable_moe_block,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            hidden_size_per_layer_input,
            vocab_size_per_layer_input,
            attention_k_eq_v,
            num_kv_shared_layers,
            use_double_wide_mlp,
            global_head_dim,
            num_global_key_value_heads,
        }
    }

    /// Construct the per-layer rotary embedding. Sliding layers use
    /// standard `RotaryEmbedding::new` with `rope_theta_local`; full
    /// layers dispatch on `rope_type_full` — `"proportional"` selects
    /// Gemma 4's proportional variant, anything else falls back to
    /// standard partial (or full) rotation.
    fn build_rotary_for_layer(
        &self,
        layer_idx: usize,
        head_dim: usize,
        max_position_embeddings: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<RotaryEmbedding> {
        if self.is_sliding_layer(layer_idx) {
            return RotaryEmbedding::new(
                head_dim,
                max_position_embeddings,
                self.rope_theta_local,
                dtype,
                device,
            );
        }
        match self.rope_type_full.as_str() {
            "proportional" => RotaryEmbedding::new_gemma4_proportional(
                head_dim,
                max_position_embeddings,
                self.rope_theta_full,
                self.partial_rotary_factor_full,
                dtype,
                device,
            ),
            _ if self.partial_rotary_factor_full < 1.0 => RotaryEmbedding::new_partial(
                head_dim,
                max_position_embeddings,
                self.rope_theta_full,
                self.partial_rotary_factor_full,
                true,
                dtype,
                device,
            ),
            _ => RotaryEmbedding::new(
                head_dim,
                max_position_embeddings,
                self.rope_theta_full,
                dtype,
                device,
            ),
        }
    }

    /// Target layer index for KV cache sharing — see the matching helper
    /// in `gemma4.rs` for the algorithm. Returns `Some(target)` for
    /// layers in the last `num_kv_shared_layers`, `None` otherwise.
    fn kv_sharing_target_layer(&self, layer_idx: usize, num_hidden_layers: usize) -> Option<usize> {
        if self.num_kv_shared_layers == 0 || num_hidden_layers == 0 {
            return None;
        }
        let first_shared = num_hidden_layers.saturating_sub(self.num_kv_shared_layers);
        if layer_idx < first_shared {
            return None;
        }
        let current_type = self.layer_types.get(layer_idx).cloned().unwrap_or_else(|| {
            if self.is_sliding_layer(layer_idx) {
                "sliding_attention".to_string()
            } else {
                "full_attention".to_string()
            }
        });
        for candidate in (0..first_shared).rev() {
            let candidate_type = self.layer_types.get(candidate).cloned().unwrap_or_else(|| {
                if self.is_sliding_layer(candidate) {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                }
            });
            if candidate_type == current_type {
                return Some(candidate);
            }
        }
        None
    }

    fn is_sliding_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.layer_types.len() {
            self.layer_types[layer_idx] == "sliding_attention"
        } else if self.sliding_window_pattern == 0 {
            false
        } else {
            layer_idx.is_multiple_of(self.sliding_window_pattern)
        }
    }

    fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        !self.is_sliding_layer(layer_idx)
    }

    fn head_dim_for_layer(&self, layer_idx: usize, default_head_dim: usize) -> usize {
        if self.is_full_attention_layer(layer_idx) {
            self.global_head_dim.unwrap_or(default_head_dim)
        } else {
            default_head_dim
        }
    }

    /// KV-head count for the given layer. Full-attention layers use
    /// `num_global_key_value_heads` when set; sliding layers use the base.
    fn kv_heads_for_layer(&self, layer_idx: usize, default_kv_heads: usize) -> usize {
        if self.is_full_attention_layer(layer_idx) {
            self.num_global_key_value_heads.unwrap_or(default_kv_heads)
        } else {
            default_kv_heads
        }
    }

    fn is_kv_shared_layer(&self, layer_idx: usize, num_hidden_layers: usize) -> bool {
        if self.num_kv_shared_layers == 0 {
            return false;
        }
        // saturating_sub: `num_kv_shared_layers` may exceed the model depth
        // when an integration test or smoke run reduces `num_hidden_layers`
        // while keeping the rest of a real production config intact.
        let first_shared = num_hidden_layers.saturating_sub(self.num_kv_shared_layers);
        layer_idx >= first_shared
    }

    fn layer_intermediate_size(
        &self,
        layer_idx: usize,
        base_intermediate_size: usize,
        num_hidden_layers: usize,
    ) -> usize {
        if self.use_double_wide_mlp && self.is_kv_shared_layer(layer_idx, num_hidden_layers) {
            base_intermediate_size * 2
        } else {
            base_intermediate_size
        }
    }

    /// Largest per-layer `head_dim` across all layers. Used to allocate a
    /// uniform KV cache width — per-layer reads/writes pad and slice
    /// back to their true `head_dim`.
    fn max_cache_head_dim(&self, default_head_dim: usize) -> usize {
        default_head_dim.max(self.global_head_dim.unwrap_or(default_head_dim))
    }

    /// Largest per-layer KV-head count across all layers — the KV cache
    /// stride. Full-attention layers with fewer heads pad up on write and
    /// slice back on read.
    fn max_kv_heads(&self, default_kv_heads: usize) -> usize {
        default_kv_heads.max(self.num_global_key_value_heads.unwrap_or(default_kv_heads))
    }
}

// ─── Quantized Gemma 4 Router ─────────────────────────────────────────────
//
// UnweightedRmsNorm → root_size scaling → learned scale → gate linear.
// The `scale` tensor and the gate linear are loaded separately: the scale
// is a small FP vector kept in the checkpoint dtype, the gate is routed
// through the quantized weight loader like every other linear projection.

struct QuantizedGemma4Router {
    norm: UnweightedRmsNorm,
    scale: Tensor,
    gate: Box<dyn QuantizedLinear>,
    root_size: f64,
}

impl QuantizedGemma4Router {
    fn new(
        hidden_size: usize,
        num_experts: usize,
        rms_norm_eps: f64,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let norm = unweighted_rms_norm(rms_norm_eps);
        let scale = vb.get(hidden_size, "scale")?;
        let gate =
            loader.load_linear(&format!("{prefix}.proj"), hidden_size, num_experts, false)?;
        let root_size = (hidden_size as f64).powf(-0.5);

        Ok(Self {
            norm,
            scale,
            gate,
            root_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(xs)?;
        let h = (h * self.root_size)?;
        let h = h.broadcast_mul(&self.scale)?;
        self.gate.forward(&h)
    }
}

// ─── Quantized MoE Expert ─────────────────────────────────────────────────
//
// gate_proj + up_proj + down_proj, GELU-Erf activation (not SiLU — critical
// for Gemma 4 vs Llama/Mistral family).

struct QuantizedGemma4MoEExpert {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedGemma4MoEExpert {
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.gelu_erf()?.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Quantized MoE block ──────────────────────────────────────────────────

struct QuantizedGemma4MoE {
    router: QuantizedGemma4Router,
    per_expert_scale: Tensor,
    experts: Vec<QuantizedGemma4MoEExpert>,
    num_experts: usize,
    top_k: usize,
}

impl QuantizedGemma4MoE {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        rms_norm_eps: f64,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let num_experts = extra_cfg.num_experts;
        let top_k = extra_cfg.top_k_experts;
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = extra_cfg.moe_intermediate_size;

        let router = QuantizedGemma4Router::new(
            hidden_size,
            num_experts,
            rms_norm_eps,
            loader,
            vb.pp("router"),
            &format!("{prefix}.router"),
        )?;

        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;

        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(QuantizedGemma4MoEExpert::new(
                hidden_size,
                moe_intermediate_size,
                loader,
                &format!("{prefix}.experts.{i}"),
            )?);
        }

        Ok(Self {
            router,
            per_expert_scale,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, router_input: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        let router_input_2d = router_input.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward(&router_input_2d)?;

        let per_expert_scale_f32 = self.per_expert_scale.to_dtype(DType::F32)?;
        let expert_scales: Vec<f32> = per_expert_scale_f32.to_vec1()?;

        let mut final_output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for token_idx in 0..num_tokens {
            let token_input = xs_2d.narrow(0, token_idx, 1)?;
            let token_logits = router_logits
                .narrow(0, token_idx, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?;

            // Gemma 4 routing: top-k indices come from raw logits,
            // dispatch weights come from softmax over *all* experts,
            // then renormalised across the chosen top-k.
            let logits_vec: Vec<f32> = token_logits.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk_indices: Vec<usize> = indexed.iter().take(self.top_k).map(|x| x.0).collect();

            let probs =
                candle_nn::ops::softmax_last_dim(&token_logits.unsqueeze(0)?)?.squeeze(0)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;

            let mut gate_weights = vec![0.0f32; self.num_experts];
            for &idx in &topk_indices {
                gate_weights[idx] = probs_vec[idx];
            }
            let renorm_factor: f32 = gate_weights.iter().sum();
            let renorm_factor = if renorm_factor > 0.0 {
                renorm_factor
            } else {
                1.0
            };

            let mut token_output = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;

            for &expert_idx in &topk_indices {
                if expert_idx < self.num_experts {
                    let weight =
                        (gate_weights[expert_idx] / renorm_factor) * expert_scales[expert_idx];
                    let expert_out = self.experts[expert_idx].forward(&token_input)?;
                    let weighted = expert_out.affine(weight as f64, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            final_output = final_output.index_add(&indices, &token_output, 0)?;
        }

        final_output.reshape(orig_shape)
    }
}

// ─── Quantized Attention ──────────────────────────────────────────────────

/// Pad the last dimension of `[b, heads, seq, dim]` from `from` to `to`
/// with zeros. Used to widen K/V from the per-layer `head_dim` up to the
/// shared `cache_head_dim` before writing to the paged KV cache.
fn pad_last_dim(x: &Tensor, from: usize, to: usize) -> Result<Tensor> {
    if from == to {
        return Ok(x.clone());
    }
    if to < from {
        return Err(candle_core::Error::Msg(format!(
            "pad_last_dim: to ({to}) must be >= from ({from})"
        )));
    }
    let (b, h, s, d) = x.dims4()?;
    if d != from {
        return Err(candle_core::Error::Msg(format!(
            "pad_last_dim: tensor last dim is {d}, expected {from}"
        )));
    }
    let zeros = Tensor::zeros((b, h, s, to - from), x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], 3)
}

/// Zero-pad the KV-head dimension (dim 1 of `[b, heads, seq, head_dim]`)
/// from `from` to `to`. Full-attention layers use fewer KV heads than the
/// shared cache stride; the extra heads are zero on write, sliced on read.
fn pad_kv_heads(x: &Tensor, from: usize, to: usize) -> Result<Tensor> {
    if from == to {
        return Ok(x.clone());
    }
    if to < from {
        return Err(candle_core::Error::Msg(format!(
            "pad_kv_heads: to ({to}) must be >= from ({from})"
        )));
    }
    let (b, h, s, d) = x.dims4()?;
    if h != from {
        return Err(candle_core::Error::Msg(format!(
            "pad_kv_heads: tensor head dim is {h}, expected {from}"
        )));
    }
    let zeros = Tensor::zeros((b, to - from, s, d), x.dtype(), x.device())?;
    Tensor::cat(&[x, &zeros], 1)
}

struct QuantizedGemma4Attention {
    q_proj: Box<dyn QuantizedLinear>,
    /// KV-shared layers (last `num_kv_shared_layers` in the stack) carry
    /// no `k_proj` / `v_proj` weights — those slots are `None` and the
    /// forward pass reads K/V straight from the target layer's cache.
    k_proj: Option<Box<dyn QuantizedLinear>>,
    v_proj: Option<Box<dyn QuantizedLinear>>,
    o_proj: Box<dyn QuantizedLinear>,
    q_norm: Gemma4RmsNorm,
    k_norm: Option<Gemma4RmsNorm>,
    v_norm: Option<UnweightedRmsNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    /// Layer-specific head_dim (256 for sliding, 512 for full in E2B).
    head_dim: usize,
    /// Shared KV cache head_dim = max across all layers in the model.
    cache_head_dim: usize,
    /// Shared KV cache KV-head stride = max across all layers. Full layers
    /// with fewer KV heads pad up on write and slice back on read.
    cache_num_kv_heads: usize,
    attn_logit_softcap: Option<f64>,
    sliding_window: Option<usize>,
    is_kv_shared: bool,
}

impl QuantizedGemma4Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        layer_idx: usize,
        cache_head_dim: usize,
        cache_num_kv_heads: usize,
        is_kv_shared: bool,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        prefix: &str,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = extra_cfg.kv_heads_for_layer(layer_idx, cfg.num_key_value_heads);
        let head_dim = extra_cfg.head_dim_for_layer(layer_idx, cfg.head_dim);

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;

        let (k_proj, v_proj, k_norm, v_norm) = if is_kv_shared {
            (None, None, None, None)
        } else {
            let k_proj = loader.load_linear(
                &format!("{prefix}.k_proj"),
                cfg.hidden_size,
                num_kv_heads * head_dim,
                false,
            )?;

            let use_k_eq_v =
                extra_cfg.attention_k_eq_v && extra_cfg.is_full_attention_layer(layer_idx);
            let v_proj = if use_k_eq_v {
                match loader.load_linear(
                    &format!("{prefix}.v_proj"),
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    false,
                ) {
                    Ok(v) => v,
                    Err(_) => loader.load_linear(
                        &format!("{prefix}.k_proj"),
                        cfg.hidden_size,
                        num_kv_heads * head_dim,
                        false,
                    )?,
                }
            } else {
                loader.load_linear(
                    &format!("{prefix}.v_proj"),
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    false,
                )?
            };

            let k_norm = gemma4_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            let v_norm = unweighted_rms_norm(cfg.rms_norm_eps);

            (Some(k_proj), Some(v_proj), Some(k_norm), Some(v_norm))
        };

        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        let q_norm = gemma4_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        let rotary_emb = extra_cfg.build_rotary_for_layer(
            layer_idx,
            head_dim,
            cfg.max_position_embeddings,
            loader.dtype(),
            loader.device(),
        )?;

        let sliding_window = if extra_cfg.is_sliding_layer(layer_idx) {
            cfg.sliding_window
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_head_dim,
            cache_num_kv_heads,
            attn_logit_softcap: extra_cfg.attn_logit_softcap,
            sliding_window,
            is_kv_shared,
        })
    }

    fn apply_per_head_norm(x: &Tensor, norm: &Gemma4RmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
    }

    fn apply_per_head_norm_unweighted(x: &Tensor, norm: &UnweightedRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        real_q_len: usize,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;

        if self.is_kv_shared {
            // Shared layer: rotate Q only, read K/V from the target layer's
            // cache (already populated by an earlier layer of the same type).
            let (q, _) = self.rotary_emb.apply(&q, &q, seqlen_offset)?;
            let num_tokens = seqlen_offset + real_q_len;
            let (k_full, v_full) = cache_engine
                .read(block_table.block_ids(), num_tokens)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;
            self.finalize_attention(&q, &k_full, &v_full, q_len, real_q_len, b_sz, seqlen_offset)
        } else {
            let k = self
                .k_proj
                .as_ref()
                .expect("non-shared layer must have k_proj")
                .forward(xs)?;
            let v = self
                .v_proj
                .as_ref()
                .expect("non-shared layer must have v_proj")
                .forward(xs)?;

            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;

            let k = Self::apply_per_head_norm(
                &k,
                self.k_norm.as_ref().expect("non-shared layer has k_norm"),
            )?;
            let v = Self::apply_per_head_norm_unweighted(
                &v,
                self.v_norm.as_ref().expect("non-shared layer has v_norm"),
            )?;

            let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

            // Bucketed prefill pads the sequence; the padding rows of K/V
            // are dropped here — nothing ever attends to them (real queries
            // are causally earlier, pad queries are discarded), so they
            // must not reach the cache.
            let (k, v) = if real_q_len < q_len {
                (k.narrow(2, 0, real_q_len)?, v.narrow(2, 0, real_q_len)?)
            } else {
                (k, v)
            };

            let k_padded = pad_last_dim(&k, self.head_dim, self.cache_head_dim)?;
            let v_padded = pad_last_dim(&v, self.head_dim, self.cache_head_dim)?;
            let k_padded = pad_kv_heads(&k_padded, self.num_kv_heads, self.cache_num_kv_heads)?;
            let v_padded = pad_kv_heads(&v_padded, self.num_kv_heads, self.cache_num_kv_heads)?;
            let k_for_cache = k_padded.squeeze(0)?.contiguous()?;
            let v_for_cache = v_padded.squeeze(0)?.contiguous()?;
            cache_engine
                .write(&k_for_cache, &v_for_cache, slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let num_tokens = seqlen_offset + real_q_len;
            let (k_full, v_full) = cache_engine
                .read(block_table.block_ids(), num_tokens)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;
            self.finalize_attention(&q, &k_full, &v_full, q_len, real_q_len, b_sz, seqlen_offset)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_attention(
        &self,
        q: &Tensor,
        k_full: &Tensor,
        v_full: &Tensor,
        q_len: usize,
        real_q_len: usize,
        b_sz: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let device = q.device();
        let dtype = q.dtype();

        // Slice the shared cache geometry back to this layer's KV-head
        // count (dim 1) before head_dim (dim 3).
        let k_full = if k_full.dim(1)? != self.num_kv_heads {
            k_full.narrow(1, 0, self.num_kv_heads)?
        } else {
            k_full.clone()
        };
        let v_full = if v_full.dim(1)? != self.num_kv_heads {
            v_full.narrow(1, 0, self.num_kv_heads)?
        } else {
            v_full.clone()
        };

        let k_full = if k_full.dim(3)? != self.head_dim {
            k_full.narrow(3, 0, self.head_dim)?
        } else {
            k_full.clone()
        };
        let v_full = if v_full.dim(3)? != self.head_dim {
            v_full.narrow(3, 0, self.head_dim)?
        } else {
            v_full.clone()
        };

        // Bucket the KV length to match the (possibly padded) query length
        // class, keeping the attention-score buffer shapes bounded. Zero
        // K/V columns beyond `real_kv` are masked off below.
        let real_kv = k_full.dim(2)?;
        let kv_len = if real_q_len < q_len {
            prefill_bucket_len(real_kv, usize::MAX)
        } else {
            real_kv
        };
        let (k_full, v_full) = if kv_len > real_kv {
            let (b, h, _, d) = k_full.dims4()?;
            let pad = Tensor::zeros((b, h, kv_len - real_kv, d), k_full.dtype(), device)?;
            (
                Tensor::cat(&[&k_full, &pad], 2)?,
                Tensor::cat(&[&v_full, &pad], 2)?,
            )
        } else {
            (k_full, v_full)
        };
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        let mut attn_weights = q.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;

        if let Some(cap) = self.attn_logit_softcap {
            attn_weights = soft_cap(&attn_weights, cap)?;
        }

        let mask = if real_q_len < q_len || kv_len > real_kv {
            bucketed_prefill_mask(
                q_len,
                kv_len,
                real_q_len,
                real_kv,
                seqlen_offset,
                self.sliding_window,
                dtype,
                device,
            )?
        } else if let Some(window_size) = self.sliding_window {
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
        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;

        let (k, v) = if self.is_kv_shared {
            (None, None)
        } else {
            let k = self
                .k_proj
                .as_ref()
                .expect("non-shared layer must have k_proj")
                .forward(xs)?;
            let v = self
                .v_proj
                .as_ref()
                .expect("non-shared layer must have v_proj")
                .forward(xs)?;
            let k = k
                .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = Self::apply_per_head_norm(
                &k,
                self.k_norm.as_ref().expect("non-shared layer has k_norm"),
            )?;
            let v = Self::apply_per_head_norm_unweighted(
                &v,
                self.v_norm.as_ref().expect("non-shared layer has v_norm"),
            )?;
            (Some(k), Some(v))
        };

        let mut outputs = Vec::with_capacity(batch_size);
        let num_kv_groups = self.num_heads / self.num_kv_heads;

        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;

            let q_i = if self.is_kv_shared {
                let (q_rot, _) = self.rotary_emb.apply(&q_i, &q_i, seq.seqlen_offset)?;
                q_rot
            } else {
                let k_i = k.as_ref().unwrap().narrow(0, i, 1)?;
                let v_i = v.as_ref().unwrap().narrow(0, i, 1)?;
                let (q_rot, k_rot) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;
                let k_padded = pad_last_dim(&k_rot, self.head_dim, self.cache_head_dim)?;
                let v_padded = pad_last_dim(&v_i, self.head_dim, self.cache_head_dim)?;
                let k_padded = pad_kv_heads(&k_padded, self.num_kv_heads, self.cache_num_kv_heads)?;
                let v_padded = pad_kv_heads(&v_padded, self.num_kv_heads, self.cache_num_kv_heads)?;
                let k_for_cache = k_padded.squeeze(0)?.contiguous()?;
                let v_for_cache = v_padded.squeeze(0)?.contiguous()?;
                cache_engine
                    .write(&k_for_cache, &v_for_cache, &seq.slot_mapping)
                    .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;
                q_rot
            };

            let kv_len = seq.seqlen_offset + 1;
            let (k_full, v_full) = cache_engine
                .read(&seq.block_ids, kv_len)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

            let k_full = if k_full.dim(1)? != self.num_kv_heads {
                k_full.narrow(1, 0, self.num_kv_heads)?
            } else {
                k_full
            };
            let v_full = if v_full.dim(1)? != self.num_kv_heads {
                v_full.narrow(1, 0, self.num_kv_heads)?
            } else {
                v_full
            };
            let k_full = if k_full.dim(3)? != self.head_dim {
                k_full.narrow(3, 0, self.head_dim)?
            } else {
                k_full
            };
            let v_full = if v_full.dim(3)? != self.head_dim {
                v_full.narrow(3, 0, self.head_dim)?
            } else {
                v_full
            };

            let k_full = repeat_kv(k_full, num_kv_groups)?;
            let v_full = repeat_kv(v_full, num_kv_groups)?;

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

// ─── Quantized GeGLU MLP ──────────────────────────────────────────────────

struct QuantizedGemma4Mlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedGemma4Mlp {
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.gelu_erf()?.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Quantized Decoder Layer ──────────────────────────────────────────────

struct QuantizedGemma4DecoderLayer {
    self_attn: QuantizedGemma4Attention,
    mlp: QuantizedGemma4Mlp,
    moe: Option<QuantizedGemma4MoE>,
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm_1: Option<Gemma4RmsNorm>,
    post_feedforward_layernorm_2: Option<Gemma4RmsNorm>,
    pre_feedforward_layernorm_2: Option<Gemma4RmsNorm>,
    per_layer_input_gate: Option<Box<dyn QuantizedLinear>>,
    per_layer_projection: Option<Box<dyn QuantizedLinear>>,
    post_per_layer_input_norm: Option<Gemma4RmsNorm>,
    layer_scalar: Tensor,
    /// Target layer idx for KV cache sharing, or `None` when this layer
    /// owns its own slot.
    kv_sharing_target: Option<usize>,
}

impl QuantizedGemma4DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        layer_idx: usize,
        cache_head_dim: usize,
        cache_num_kv_heads: usize,
        loader: &dyn QuantizedWeightLoader,
        vb_m: VarBuilder,
    ) -> Result<Self> {
        // `vb_m` is positioned at the inner `model.*` namespace so the
        // layer VarBuilder is `vb_m.pp("layers").pp(idx)` without an
        // extra `.pp("model")` (that was the standalone-only path and
        // now lives in `QuantizedGemma4ForCausalLM::new`). The loader
        // prefix still carries "model." — callers that need to remap
        // (VLM wrapper) wrap the loader in `RemappingWeightLoader`.
        let prefix = format!("model.layers.{layer_idx}");
        let vb_layer = vb_m.pp("layers").pp(layer_idx);

        let kv_sharing_target = extra_cfg.kv_sharing_target_layer(layer_idx, cfg.num_hidden_layers);
        let is_kv_shared = kv_sharing_target.is_some();

        let self_attn = QuantizedGemma4Attention::new(
            cfg,
            extra_cfg,
            layer_idx,
            cache_head_dim,
            cache_num_kv_heads,
            is_kv_shared,
            loader,
            vb_layer.pp("self_attn"),
            &format!("{prefix}.self_attn"),
        )?;

        let intermediate_size = extra_cfg.layer_intermediate_size(
            layer_idx,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
        );
        let mlp = QuantizedGemma4Mlp::new(
            cfg.hidden_size,
            intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        let moe = if extra_cfg.enable_moe_block && extra_cfg.num_experts > 0 {
            Some(QuantizedGemma4MoE::new(
                cfg,
                extra_cfg,
                cfg.rms_norm_eps,
                loader,
                vb_layer.pp("moe"),
                &format!("{prefix}.moe"),
            )?)
        } else {
            None
        };

        let input_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_feedforward_layernorm"),
        )?;

        let (
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
        ) = if moe.is_some() {
            (
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_layer.pp("post_feedforward_layernorm_1"),
                )?),
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_layer.pp("post_feedforward_layernorm_2"),
                )?),
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb_layer.pp("pre_feedforward_layernorm_2"),
                )?),
            )
        } else {
            (None, None, None)
        };

        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
            if extra_cfg.hidden_size_per_layer_input > 0 {
                (
                    Some(loader.load_linear(
                        &format!("{prefix}.per_layer_input_gate"),
                        cfg.hidden_size,
                        extra_cfg.hidden_size_per_layer_input,
                        false,
                    )?),
                    Some(loader.load_linear(
                        &format!("{prefix}.per_layer_projection"),
                        extra_cfg.hidden_size_per_layer_input,
                        cfg.hidden_size,
                        false,
                    )?),
                    Some(gemma4_rms_norm(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb_layer.pp("post_per_layer_input_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };

        let layer_scalar = vb_layer
            .get(1, "layer_scalar")
            .unwrap_or_else(|_| Tensor::ones(1, DType::F32, vb_m.device()).unwrap());

        Ok(Self {
            self_attn,
            mlp,
            moe,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar,
            kv_sharing_target,
        })
    }

    fn apply_ple(
        &self,
        hidden_states: &Tensor,
        per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let (Some(gate_proj), Some(proj), Some(norm), Some(pli)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let gate = gate_proj.forward(hidden_states)?;
            let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
            let gated = gate.broadcast_mul(pli)?;
            let contribution = proj.forward(&gated)?;
            let contribution = norm.forward(&contribution)?;
            hidden_states + contribution
        } else {
            Ok(hidden_states.clone())
        }
    }

    fn forward_ffn(&self, residual: &Tensor, hidden_states_mlp: Tensor) -> Result<Tensor> {
        if let (Some(moe), Some(pf_norm1), Some(pf_norm2), Some(pre_ff2)) = (
            &self.moe,
            &self.post_feedforward_layernorm_1,
            &self.post_feedforward_layernorm_2,
            &self.pre_feedforward_layernorm_2,
        ) {
            let h1 = pf_norm1.forward(&hidden_states_mlp)?;
            let h2 = pre_ff2.forward(residual)?;
            let h2 = moe.forward(&h2, residual)?;
            let h2 = pf_norm2.forward(&h2)?;
            Ok((h1 + h2)?)
        } else {
            Ok(hidden_states_mlp)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        real_q_len: usize,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let cache_layer_idx = self.kv_sharing_target.unwrap_or(layer_idx);
        let attn_out = self.self_attn.forward(
            &hidden_states,
            real_q_len,
            seqlen_offset,
            kv_cache_mgr.engine_mut(cache_layer_idx),
            block_table,
            slot_mapping,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&attn_out)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states_norm = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states_mlp = self.mlp.forward(&hidden_states_norm)?;

        let hidden_states = self.forward_ffn(residual, hidden_states_mlp)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        let hidden_states = self.apply_ple(&hidden_states, per_layer_input)?;
        hidden_states.broadcast_mul(&self.layer_scalar)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let cache_layer_idx = self.kv_sharing_target.unwrap_or(layer_idx);
        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(cache_layer_idx),
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states_norm = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states_mlp = self.mlp.forward(&hidden_states_norm)?;

        let hidden_states = self.forward_ffn(residual, hidden_states_mlp)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        let hidden_states = self.apply_ple(&hidden_states, per_layer_input)?;
        hidden_states.broadcast_mul(&self.layer_scalar)
    }
}

// ─── Tied Embedding Head ──────────────────────────────────────────────────
//
// Copied from gemma3_quantized.rs — lets the lm_head implement the
// QuantizedLinear trait so the rest of the forward path is agnostic of
// whether the checkpoint ties word embeddings or stores a dedicated head.

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

// ─── Quantized Model ──────────────────────────────────────────────────────

/// Quantized Gemma 4 text model.
///
/// Supports AWQ, BitsAndBytes, GPTQ, FP8, GGUF, and the pass-through
/// unquantized loader via the `QuantizedWeightLoader` trait.
pub struct QuantizedGemma4ForCausalLM {
    embed_tokens: Embedding,
    embed_tokens_per_layer: Option<Embedding>,
    per_layer_model_projection: Option<Box<dyn QuantizedLinear>>,
    per_layer_projection_norm: Option<Gemma4RmsNorm>,
    layers: Vec<QuantizedGemma4DecoderLayer>,
    norm: Gemma4RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    hidden_size: usize,
    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,
    final_logit_softcap: Option<f64>,
    /// RoPE tables are precomputed to this many positions; prefill
    /// bucketing must not pad past it.
    max_position_embeddings: usize,
    device: Device,
}

impl QuantizedGemma4ForCausalLM {
    /// Standalone Gemma 4 text checkpoint constructor.
    ///
    /// Assumes the safetensors store the transformer under a top-level
    /// `model.*` wrapper and the weight loader resolves `model.*` paths
    /// literally — i.e. the normal case for a text-only
    /// `Gemma4ForCausalLM` repository. Internally descends one level
    /// with `vb.pp("model")` and delegates to
    /// [`Self::new_at_model_root`].
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        Self::new_at_model_root(cfg, vb_m, weight_loader)
    }

    /// Construct the quantized Gemma 4 text model directly from a
    /// VarBuilder already positioned at the inner `model.*` namespace.
    ///
    /// The VLM wrapper (`QuantizedGemma4ForConditionalGeneration`) uses
    /// this to skip the standalone `.pp("model")` step — in an HF VLM
    /// checkpoint the language-model tensors live at
    /// `model.language_model.*`, so the wrapper passes a VarBuilder
    /// already scoped to `model.language_model` here.
    ///
    /// The `weight_loader` argument must resolve vLLM-style `"model.X"`
    /// load paths to the REAL tensor names in the checkpoint. In
    /// standalone mode that is the identity (`new` wraps without any
    /// remapping); in VLM mode the caller wraps the base loader in
    /// `RemappingWeightLoader` so `"model.X"` → `"model.language_model.X"`.
    pub fn new_at_model_root(
        cfg: &ModelConfig,
        vb_m: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let extra_cfg = Gemma4ExtraConfig::from_model_config(cfg);

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // PLE components: per-layer embedding stays as a regular embedding
        // table (embeddings are never quantized by AWQ/BnB/GGUF), only the
        // model-level projection is routed through the quantized loader.
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if extra_cfg.hidden_size_per_layer_input > 0 {
                let total_ple_dim = extra_cfg.hidden_size_per_layer_input * cfg.num_hidden_layers;

                let ple_embed = embedding(
                    extra_cfg.vocab_size_per_layer_input,
                    total_ple_dim,
                    vb_m.pp("embed_tokens_per_layer"),
                )?;

                let ple_proj = weight_loader.load_linear(
                    "model.per_layer_model_projection",
                    cfg.hidden_size,
                    total_ple_dim,
                    false,
                )?;

                let ple_norm = gemma4_rms_norm(
                    extra_cfg.hidden_size_per_layer_input,
                    cfg.rms_norm_eps,
                    vb_m.pp("per_layer_projection_norm"),
                )?;

                (Some(ple_embed), Some(ple_proj), Some(ple_norm))
            } else {
                (None, None, None)
            };

        let cache_head_dim = extra_cfg.max_cache_head_dim(cfg.head_dim);
        let cache_num_kv_heads = extra_cfg.max_kv_heads(cfg.num_key_value_heads);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedGemma4DecoderLayer::new(
                cfg,
                &extra_cfg,
                i,
                cache_head_dim,
                cache_num_kv_heads,
                weight_loader,
                vb_m.clone(),
            )?);
        }

        let norm = gemma4_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            // Released Gemma 4 EXL3 checkpoints set `tie_word_embeddings=true`
            // yet still ship a separately-quantized `lm_head` (its `head_bits`
            // differ from the body bits — e.g. 6-bit head over a 2-4-bit body)
            // while `embed_tokens` stays unquantized bf16. Prefer the real
            // quantized head when present. We only probe for EXL3 loaders: the
            // EXL3 `load_linear` errors cleanly when `lm_head.trellis` is
            // absent (it never zero-fills the trellis), whereas other loaders
            // may synthesise a zero weight, so for them we keep the tied path.
            let separate_head =
                if weight_loader.method() == crate::quantization::QuantizationMethod::Exl3 {
                    weight_loader
                        .load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)
                        .ok()
                } else {
                    None
                };
            separate_head.unwrap_or_else(|| {
                Box::new(TiedEmbeddingHead {
                    weight: embed_tokens.embeddings().clone(),
                })
            })
        } else {
            weight_loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
        };

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_size_per_layer_input: extra_cfg.hidden_size_per_layer_input,
            final_logit_softcap: extra_cfg.final_logit_softcap,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb_m.device().clone(),
        })
    }

    /// Compute PLE per-layer inputs from input_ids and hidden_states.
    ///
    /// Returns tensor of shape [..., num_layers, hidden_size_per_layer_input]
    /// or None when PLE is disabled.
    fn compute_per_layer_inputs(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
    ) -> Result<Option<Tensor>> {
        let (embed_pl, proj, norm) = match (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            (Some(e), Some(p), Some(n)) => (e, p, n),
            _ => return Ok(None),
        };

        let ple_dim = self.hidden_size_per_layer_input;
        let num_layers = self.num_hidden_layers;
        let embed_scale = (ple_dim as f64).sqrt();
        let proj_scale = (self.hidden_size as f64).powf(-0.5);
        let input_scale = (2.0_f64).powf(-0.5);

        let ple_embed = (embed_pl.forward(input_ids)? * embed_scale)?;
        let id_shape = input_ids.dims();
        let mut new_shape: Vec<usize> = id_shape.to_vec();
        new_shape.push(num_layers);
        new_shape.push(ple_dim);
        let ple_embed = ple_embed.reshape(&new_shape[..])?;

        let ple_proj = (proj.forward(hidden_states)? * proj_scale)?;
        let proj_shape = {
            let hs = hidden_states.dims();
            let mut s: Vec<usize> = hs[..hs.len() - 1].to_vec();
            s.push(num_layers);
            s.push(ple_dim);
            s
        };
        let ple_proj = ple_proj.reshape(&proj_shape[..])?;

        let ndim = ple_proj.dims().len();
        let layer_dim = ndim - 2;
        let mut normed_slices = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let slice = ple_proj.narrow(layer_dim, l, 1)?.squeeze(layer_dim)?;
            let normed = norm.forward(&slice)?;
            normed_slices.push(normed.unsqueeze(layer_dim)?);
        }
        let ple_proj_normed = Tensor::cat(&normed_slices, layer_dim)?;

        let per_layer_inputs = ((ple_proj_normed + ple_embed)? * input_scale)?;
        Ok(Some(per_layer_inputs))
    }

    fn extract_per_layer_input(
        per_layer_inputs: &Option<Tensor>,
        layer_idx: usize,
    ) -> Result<Option<Tensor>> {
        match per_layer_inputs {
            Some(pli) => {
                let ndim = pli.dims().len();
                let layer_dim = ndim - 2;
                let slice = pli.narrow(layer_dim, layer_idx, 1)?.squeeze(layer_dim)?;
                Ok(Some(slice))
            }
            None => Ok(None),
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
        let normalizer = (self.hidden_size as f64).sqrt();
        let xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;

        // Bucket the prefill length so candle's per-shape CUDA buffer cache
        // sees a bounded set of activation shapes (see PREFILL_LEN_BUCKET).
        // Right-padding is invisible to the real tokens: pads are strictly
        // in the causal future, their K/V rows are dropped before the cache
        // write, and their outputs are discarded by the final-position
        // slice. Skipped when PLE is active (per-layer inputs are sized to
        // the real length).
        let real_len = xs.dim(1)?;
        let bucket_len = if self.embed_tokens_per_layer.is_none() {
            prefill_bucket_len(
                real_len,
                self.max_position_embeddings.saturating_sub(seqlen_offset),
            )
        } else {
            real_len
        };
        let xs = if bucket_len > real_len {
            let (b, _, h) = xs.dims3()?;
            let pad = Tensor::zeros((b, bucket_len - real_len, h), xs.dtype(), xs.device())?;
            Tensor::cat(&[&xs, &pad], 1)?
        } else {
            xs
        };

        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;
            xs = layer.forward(
                &xs,
                pli.as_ref(),
                real_len,
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }

        // Only the final position's logits are consumed downstream (the
        // engine samples the next token from it; full-sequence logits are
        // needed solely for echo+prompt_logprobs, see engine/helpers.rs).
        // Slicing BEFORE the lm_head matters enormously for Gemma 4: its
        // 262 144-token vocab makes a full-sequence logits buffer (and the
        // EXL3 kernel's fp32 staging) ~300 MB per distinct prompt length,
        // all of which the CUDA allocator caches per shape — several
        // differently-sized requests then OOM an 8 GB card.
        let xs = if real_len > 1 || xs.dim(1)? > 1 {
            xs.narrow(1, real_len - 1, 1)?
        } else {
            xs
        };
        let xs = self.norm.forward(&xs)?;
        let mut logits = self.lm_head.forward(&xs)?;

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    /// Embed text tokens (for VLM use — embed only, no layers).
    /// Applies Gemma 4's sqrt(hidden_size) normalisation.
    pub(crate) fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let normalizer = (self.hidden_size as f64).sqrt();
        self.embed_tokens.forward(input_ids)? * normalizer
    }

    /// Forward with pre-computed embeddings (for VLM use — skips embedding).
    pub(crate) fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // PLE uses a dummy input_ids of zeros because the original token ids
        // aren't available after multimodal token replacement.
        let dummy_shape: Vec<usize> = embeddings.dims()[..embeddings.dims().len() - 1].to_vec();
        let dummy_ids = Tensor::zeros(&dummy_shape[..], DType::U32, &self.device)?;
        let per_layer_inputs = self.compute_per_layer_inputs(&dummy_ids, embeddings)?;

        // Bucketed prefill — see `forward` for the rationale.
        let real_len = embeddings.dim(1)?;
        let bucket_len = if self.embed_tokens_per_layer.is_none() {
            prefill_bucket_len(
                real_len,
                self.max_position_embeddings.saturating_sub(seqlen_offset),
            )
        } else {
            real_len
        };
        let mut xs = if bucket_len > real_len {
            let (b, _, h) = embeddings.dims3()?;
            let pad = Tensor::zeros(
                (b, bucket_len - real_len, h),
                embeddings.dtype(),
                embeddings.device(),
            )?;
            Tensor::cat(&[embeddings, &pad], 1)?
        } else {
            embeddings.clone()
        };
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;
            xs = layer.forward(
                &xs,
                pli.as_ref(),
                real_len,
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }

        // Last-position-only lm_head — see `forward` for the rationale.
        let xs = if real_len > 1 || xs.dim(1)? > 1 {
            xs.narrow(1, real_len - 1, 1)?
        } else {
            xs
        };
        let xs = self.norm.forward(&xs)?;
        let mut logits = self.lm_head.forward(&xs)?;
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }
        Ok(logits)
    }

    pub(crate) fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let dummy_shape: Vec<usize> = embeddings.dims()[..embeddings.dims().len() - 1].to_vec();
        let dummy_ids = Tensor::zeros(&dummy_shape[..], DType::U32, &self.device)?;
        let per_layer_inputs = self.compute_per_layer_inputs(&dummy_ids, embeddings)?;

        let mut xs = embeddings.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;
            xs = layer.forward_decode_batch(
                &xs,
                pli.as_ref(),
                sequences,
                kv_cache_mgr,
                layer_idx,
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

impl crate::engine::ModelForward for QuantizedGemma4ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QuantizedGemma4ForCausalLM::forward(
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
        let xs = (self.embed_tokens.forward(input_ids)? * normalizer)?;

        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;

        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;
            xs = layer.forward_decode_batch(
                &xs,
                pli.as_ref(),
                sequences,
                kv_cache_mgr,
                layer_idx,
            )?;
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

    fn test_config() -> ModelConfig {
        test_config_with_moe(false)
    }

    fn test_config_with_moe(enable_moe: bool) -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
        extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(16),
        );
        extra.insert(
            "final_logit_softcapping".to_string(),
            serde_json::json!(30.0),
        );

        if enable_moe {
            extra.insert("enable_moe_block".to_string(), serde_json::json!(true));
            extra.insert("num_experts".to_string(), serde_json::json!(4));
            extra.insert("top_k_experts".to_string(), serde_json::json!(2));
            extra.insert("moe_intermediate_size".to_string(), serde_json::json!(32));
        }

        ModelConfig {
            architectures: vec!["Gemma4ForCausalLM".to_string()],
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
    fn test_gemma4_extra_config_parsing() {
        let cfg = test_config_with_moe(true);
        let extra = Gemma4ExtraConfig::from_model_config(&cfg);

        assert_eq!(extra.hidden_size_per_layer_input, 16);
        assert_eq!(extra.final_logit_softcap, Some(30.0));
        assert!(extra.enable_moe_block);
        assert_eq!(extra.num_experts, 4);
        assert_eq!(extra.top_k_experts, 2);
        assert_eq!(extra.moe_intermediate_size, 32);
        assert_eq!(extra.sliding_window_pattern, 2);

        assert!(extra.is_sliding_layer(0));
        assert!(!extra.is_sliding_layer(1));
        assert!(extra.is_sliding_layer(2));
        assert!(!extra.is_sliding_layer(3));
    }

    /// Config mirroring the released Gemma 4 12B head geometry at toy
    /// dimensions: full-attention layers use a larger head_dim AND fewer KV
    /// heads than sliding layers. Exercises both heterogeneity axes.
    fn test_config_heterogeneous_heads() -> ModelConfig {
        let mut cfg = test_config();
        // 4 q-heads everywhere; sliding kv=2 @ head_dim 16, full kv=1 @ 32.
        cfg.num_key_value_heads = 2;
        cfg.head_dim = 16;
        cfg.extra
            .insert("global_head_dim".to_string(), serde_json::json!(32u64));
        cfg.extra.insert(
            "num_global_key_value_heads".to_string(),
            serde_json::json!(1u64),
        );
        cfg.extra.insert(
            "layer_types".to_string(),
            serde_json::json!([
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ]),
        );
        // PLE off keeps the test focused on the attention/cache path.
        cfg.extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(0u64),
        );
        cfg
    }

    #[test]
    fn test_kv_head_heterogeneity_config() {
        let cfg = test_config_heterogeneous_heads();
        let extra = Gemma4ExtraConfig::from_model_config(&cfg);

        // Layer 0 sliding: 2 kv heads @ head_dim 16.
        assert!(extra.is_sliding_layer(0));
        assert_eq!(extra.kv_heads_for_layer(0, cfg.num_key_value_heads), 2);
        assert_eq!(extra.head_dim_for_layer(0, cfg.head_dim), 16);
        // Layer 1 full: 1 kv head @ head_dim 32.
        assert!(extra.is_full_attention_layer(1));
        assert_eq!(extra.kv_heads_for_layer(1, cfg.num_key_value_heads), 1);
        assert_eq!(extra.head_dim_for_layer(1, cfg.head_dim), 32);
        // Shared cache stride = max across layers.
        assert_eq!(extra.max_kv_heads(cfg.num_key_value_heads), 2);
        assert_eq!(extra.max_cache_head_dim(cfg.head_dim), 32);
        // ModelConfig helpers used by the server to size the paged cache.
        assert_eq!(cfg.kv_cache_num_kv_heads(), 2);
        assert_eq!(cfg.kv_cache_head_dim(), 32);
    }

    #[test]
    fn test_prefill_bucket_len() {
        // Multiples of 32, capped at max_positions, never below real_len.
        assert_eq!(prefill_bucket_len(1, 1024), 1); // decode-sized: untouched
        assert_eq!(prefill_bucket_len(5, 1024), 32);
        assert_eq!(prefill_bucket_len(32, 1024), 32);
        assert_eq!(prefill_bucket_len(33, 1024), 64);
        assert_eq!(prefill_bucket_len(1000, 1024), 1024);
        // Cap below the bucket boundary: pad only as far as RoPE reaches.
        assert_eq!(prefill_bucket_len(5, 20), 20);
        // Cap below real_len must not shrink the input.
        assert_eq!(prefill_bucket_len(5, 3), 5);
    }

    #[test]
    fn test_bucketed_prefill_mask_matches_unpadded_semantics() {
        // The real-token region of the bucketed mask must agree exactly with
        // the legacy masks the unpadded path used.
        let device = Device::Cpu;
        let (real_q, real_kv, offset) = (3usize, 5usize, 2usize);
        let (q_len, kv_len) = (8usize, 8usize);

        for window in [None, Some(2usize)] {
            let bucketed = bucketed_prefill_mask(
                q_len,
                kv_len,
                real_q,
                real_kv,
                offset,
                window,
                DType::F32,
                &device,
            )
            .unwrap()
            .to_vec2_like_4d();
            let reference = match window {
                Some(w) => sliding_window_mask(real_q, real_kv, offset, w, DType::F32, &device)
                    .unwrap()
                    .to_vec2_like_4d(),
                None => crate::layers::causal_mask(real_q, offset, DType::F32, &device)
                    .unwrap()
                    .to_vec2_like_4d(),
            };
            for i in 0..real_q {
                for j in 0..real_kv.min(reference[i].len()) {
                    assert_eq!(
                        bucketed[i][j], reference[i][j],
                        "mismatch at ({i},{j}) window={window:?}"
                    );
                }
                // Padded KV columns must be masked for real rows.
                for j in real_kv..kv_len {
                    assert_eq!(bucketed[i][j], f32::NEG_INFINITY, "pad col ({i},{j})");
                }
            }
            // Padding rows: only column 0 open (NaN guard), rest -inf.
            for (i, row) in bucketed.iter().enumerate().take(q_len).skip(real_q) {
                assert_eq!(row[0], 0.0, "pad row {i} col 0 must be open");
                for (j, &v) in row.iter().enumerate().skip(1) {
                    assert_eq!(v, f32::NEG_INFINITY, "pad row ({i},{j})");
                }
            }
        }
    }

    /// Test helper: flatten a `[1,1,q,kv]` mask into rows.
    trait MaskRows {
        fn to_vec2_like_4d(&self) -> Vec<Vec<f32>>;
    }
    impl MaskRows for Tensor {
        fn to_vec2_like_4d(&self) -> Vec<Vec<f32>> {
            self.squeeze(0)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| t.to_vec2::<f32>())
                .expect("mask to rows")
        }
    }

    #[test]
    fn test_padded_cache_roundtrip_preserves_values() {
        // Isolates the head_dim-padding cache path used for Gemma 4's
        // heterogeneous layers: write K/V at head_dim < cache_head_dim
        // (zero-padded), read back, slice to the real head_dim, and assert
        // the real slice survived the paged-cache round-trip bit-for-bit.
        let device = Device::Cpu;
        let head_dim = 16usize;
        let cache_head_dim = 32usize; // global_head_dim > head_dim
        let kv_heads = 2usize;
        let tokens = 5usize;

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: kv_heads,
            head_dim: cache_head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&cache_config).expect("cache mgr");
        let mut block_table = BlockTable::new(cache_config.block_size);
        mgr.allocate_for_request(&mut block_table, tokens)
            .expect("alloc");
        let slot_mapping = block_table.slot_mapping(0, tokens);

        // Deterministic non-trivial K/V at the real head_dim: [1, kv_heads, tokens, head_dim].
        let k = Tensor::arange(0u32, (kv_heads * tokens * head_dim) as u32, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, kv_heads, tokens, head_dim))
            .unwrap();
        let v = (&k + 100.0).unwrap();

        let k_pad = pad_last_dim(&k, head_dim, cache_head_dim).unwrap();
        let v_pad = pad_last_dim(&v, head_dim, cache_head_dim).unwrap();
        let engine = mgr.engine_mut(0);
        engine
            .write(
                &k_pad.squeeze(0).unwrap().contiguous().unwrap(),
                &v_pad.squeeze(0).unwrap().contiguous().unwrap(),
                &slot_mapping,
            )
            .expect("write");
        let (k_full, v_full) = engine.read(block_table.block_ids(), tokens).expect("read");
        let k_back = k_full.narrow(3, 0, head_dim).unwrap();
        let v_back = v_full.narrow(3, 0, head_dim).unwrap();

        let kd = (k_back - &k)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let vd = (v_back - &v)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            kd < 1e-5,
            "padded-cache K round-trip corrupted real slice (max|Δ|={kd})"
        );
        assert!(
            vd < 1e-5,
            "padded-cache V round-trip corrupted real slice (max|Δ|={vd})"
        );
    }

    #[test]
    fn test_kv_head_heterogeneity_forward() {
        // End-to-end prefill + decode through a 4-layer model whose full
        // layers have fewer KV heads and a wider head_dim than sliding
        // layers. Proves the cache pad-on-write / slice-on-read path
        // (`pad_kv_heads` + `narrow`) keeps shapes consistent.
        let cfg = test_config_heterogeneous_heads();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &DetectedQuantConfig::default());
        let model = QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("build heterogeneous-head Gemma 4");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.kv_cache_num_kv_heads(),
            head_dim: cfg.kv_cache_head_dim(),
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 6;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input_ids");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len + 1)
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
            .expect("prefill forward with heterogeneous KV heads");
        // prefill returns last-position-only logits (262k-vocab memory)
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        // Prefill runs the bucketed path (6 tokens → bucket 32); padding
        // rows are NaN-guarded via the mask's column-0 escape — the real
        // position's logits must come out finite.
        let flat: Vec<f32> = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "bucketed prefill produced non-finite logits"
        );

        // One decode step — reads the padded cache back and slices to the
        // per-layer KV-head count.
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next id");
        let decode_slots = block_table.slot_mapping(seq_len, 1);
        let logits2 = model
            .forward(
                &next,
                seq_len,
                &mut kv_cache_mgr,
                &block_table,
                &decode_slots,
            )
            .expect("decode forward with heterogeneous KV heads");
        assert_eq!(logits2.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_gemma4_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedGemma4ForCausalLM should construct with unquantized loader: {:?}",
            model.err()
        );

        let model = model.expect("construction succeeded");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.final_logit_softcap.is_some());
        assert!(model.embed_tokens_per_layer.is_some());
        assert!(model.per_layer_model_projection.is_some());
    }

    #[test]
    fn test_quantized_gemma4_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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

        // prefill returns last-position-only logits (262k-vocab memory)
        assert_eq!(
            logits.dims(),
            &[batch_size, 1, cfg.vocab_size],
            "prefill logits shape should be [batch, 1, vocab_size]"
        );
    }

    #[test]
    fn test_quantized_gemma4_moe_forward() {
        let cfg = test_config_with_moe(true);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build MoE model");

        assert!(model.layers[0].moe.is_some());

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 3;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");
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
            .expect("MoE forward");
        // prefill returns last-position-only logits (262k-vocab memory)
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_gemma4_ple_pipeline() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        // PLE must be enabled and carry through a forward pass.
        assert_eq!(
            model.hidden_size_per_layer_input, 16,
            "PLE should be configured"
        );

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
        // prefill returns last-position-only logits (262k-vocab memory)
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_gemma4_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model =
            QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        // prefill returns last-position-only logits (262k-vocab memory)
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        block_table.advance(3);

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
    #[ignore = "requires HF download + CUDA GPU; set HF_TOKEN in env"]
    fn load_gemma4_e2b_real_config_smoke_on_gpu() {
        // End-to-end smoke test against the real `google/gemma-4-E2B-it`
        // config, exercising every Gemma 4 specific code path:
        //
        // - `loader::fetch_model_config_only` → flattens nested
        //   `text_config` / `rope_parameters` into a flat ModelConfig.
        // - `Gemma4ExtraConfig` → parses per-layer-type RoPE parameters,
        //   PLE dims, KV-sharing config, layer_types array, etc.
        // - Full 35-layer model construction with the REAL attention
        //   shape: hidden_size=1536, num_heads=8, num_kv_heads=1 (MQA),
        //   head_dim=256 (sliding) vs 512 (full), use_double_wide_mlp,
        //   num_kv_shared_layers=20, proportional RoPE for full
        //   attention layers, standard RoPE for sliding layers.
        // - KV cache padding from per-layer `head_dim` to the shared
        //   `cache_head_dim = max(256, 512) = 512`, with read-side slice
        //   back to the layer's real head_dim.
        // - KV sharing: layers 15..35 reuse layers 10..14's engines.
        // - PLE pipeline end-to-end.
        // - Corrected Gemma4RmsNorm (no +1 offset).
        //
        // Memory budget: the real E2B at BF16 would weigh ~9 GB which is
        // over a laptop 8 GB GPU. We shrink the three dimensions that do
        // not affect Gemma 4 specific logic — `intermediate_size`,
        // `vocab_size` and `vocab_size_per_layer_input` — to fit under
        // ~2 GB. All 35 layers with their mixed sliding/full attention,
        // KV sharing pattern and proportional RoPE are preserved.
        //
        // Run with:
        //   HF_TOKEN=hf_... cargo test -p vllm-core --lib --features cuda \
        //     gemma4_quantized::tests::load_gemma4_e2b_real_config_smoke_on_gpu \
        //     -- --ignored --nocapture
        use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

        let device = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            Ok(_) => {
                eprintln!("SKIP: no CUDA device available");
                return;
            }
            Err(e) => {
                eprintln!("SKIP: CUDA init failed: {e}");
                return;
            }
        };

        // Fetch the real HF config through the production code path so
        // we exercise `flatten_hf_model_config` + `fetch_model_config_only`.
        let mut cfg = crate::loader::fetch_model_config_only("google/gemma-4-E2B-it", "main")
            .expect("fetch config.json via flattener (check HF_TOKEN + gated-repo access)");

        // Override the VLM arch to the text-only CausalLM path for a
        // focused language-model forward. The rest of the config stays
        // exactly as HF ships it.
        cfg.architectures = vec!["Gemma4ForCausalLM".to_string()];

        // Shrink the dimensions that don't affect Gemma 4 specific math
        // but dominate memory — keeps the test under 8 GB VRAM.
        let original_hidden = cfg.hidden_size;
        let original_layers = cfg.num_hidden_layers;
        let original_head_dim = cfg.head_dim;
        cfg.intermediate_size = 512; // from 6144
        cfg.vocab_size = 1024; // from 262144
        cfg.max_position_embeddings = 4096; // from 131072 — keeps RoPE precompute bounded
        cfg.extra.insert(
            "vocab_size_per_layer_input".to_string(),
            serde_json::json!(512u64),
        );

        // Sanity-check that the real structural fields landed.
        assert_eq!(original_hidden, 1536, "expected E2B hidden_size=1536");
        assert_eq!(original_layers, 35, "expected E2B num_hidden_layers=35");
        assert_eq!(original_head_dim, 256, "expected E2B head_dim=256");

        let extra_cfg = Gemma4ExtraConfig::from_model_config(&cfg);
        assert_eq!(extra_cfg.global_head_dim, Some(512));
        assert_eq!(extra_cfg.num_kv_shared_layers, 20);
        assert!(extra_cfg.use_double_wide_mlp);
        assert_eq!(extra_cfg.partial_rotary_factor_full, 0.25);
        assert_eq!(extra_cfg.rope_type_full, "proportional");
        assert!((extra_cfg.rope_theta_full - 1_000_000.0).abs() < 1.0);
        assert!((extra_cfg.rope_theta_local - 10_000.0).abs() < 1.0);
        assert_eq!(extra_cfg.hidden_size_per_layer_input, 256);
        // KV sharing: layer 0 is not shared, layer 34 is shared.
        assert!(extra_cfg
            .kv_sharing_target_layer(0, cfg.num_hidden_layers)
            .is_none());
        assert!(extra_cfg
            .kv_sharing_target_layer(34, cfg.num_hidden_layers)
            .is_some());

        let num_sliding = extra_cfg
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "sliding_attention")
            .count();
        let num_full = extra_cfg
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count();
        eprintln!(
            "real config: hidden={} layers={} sliding={} full={} head_dim={} \
             global_head_dim={:?} vocab={}→1024 inter_size={}→512 \
             rope_full=proportional@1e6 partial=0.25 kv_shared_last={}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            num_sliding,
            num_full,
            cfg.head_dim,
            extra_cfg.global_head_dim,
            262144,
            6144,
            extra_cfg.num_kv_shared_layers
        );

        let vb = VarBuilder::zeros(DType::BF16, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &DetectedQuantConfig::default());

        let model = QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("build QuantizedGemma4ForCausalLM from real (reduced) Gemma 4 E2B config");

        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(
            model.embed_tokens_per_layer.is_some(),
            "PLE must be enabled"
        );

        // KV cache allocated at the shared `cache_head_dim`
        // (max head_dim across layers = 512). Each layer's forward pads
        // K/V to this width and slices back after the read.
        let cache_head_dim = extra_cfg.max_cache_head_dim(cfg.head_dim);
        assert_eq!(cache_head_dim, 512);

        let cache_config = crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cache_head_dim,
            dtype: DType::BF16,
            device: device.clone(),
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 8;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input_ids zeros");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill blocks");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward through all 35 real Gemma 4 layers");

        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.vocab_size], // prefill returns last-position-only logits (262k-vocab memory)
            "logits shape must be [batch, seq_len, vocab_size]"
        );

        // A follow-up single-token decode pass exercises the decode
        // code path (narrower sequence, KV cache read at longer kv_len).
        block_table.advance(seq_len);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode block");
        let decode_slot = block_table.slot_mapping(seq_len, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("decode input");
        let decode_logits = model
            .forward(
                &next_token,
                seq_len,
                &mut kv_cache_mgr,
                &block_table,
                &decode_slot,
            )
            .expect("decode forward");
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);

        // Surface final VRAM usage so the GPU-fit check is visible.
        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            eprintln!(
                "nvidia-smi after full 35-layer forward: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }

        eprintln!(
            "OK: {} layers ({}s sliding / {}f full), KV sharing active for last {}, \
             proportional RoPE on full layers, produced prefill {:?} and decode {:?}",
            cfg.num_hidden_layers,
            num_sliding,
            num_full,
            extra_cfg.num_kv_shared_layers,
            logits.dims(),
            decode_logits.dims(),
        );
    }
}
