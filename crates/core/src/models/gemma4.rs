//! Gemma 4 model architecture with MoE and PLE.
//!
//! Gemma 4 combines dense MLP + Mixture of Experts (parallel), Per-Layer
//! Embeddings (PLE), QKV norms, and sliding/global attention patterns.
//!
//! Key differences from Gemma3/Gemma3n:
//! - MoE: parallel dense MLP + MoE experts per layer (optional)
//! - Custom router: UnweightedRmsNorm + root_size + learned scale + gate
//! - QKV norms: Q/K with learned weight (offset +1), V without weight
//! - Scaling = 1.0 (norms handle magnitude, unlike query_pre_attn_scalar)
//! - k_eq_v: laptop variant where V = K for full-attention layers
//! - layer_scalar: per-layer learned scalar multiplier
//! - PLE: simpler than Gemma3n (no AltUp, no Laurel)
//!
//! Architecture:
//! ```text
//! Embedding (* sqrt(hidden_size)) + PLE Pipeline -> [Gemma4Layer x N] -> RMSNorm -> LM Head
//!
//! Gemma4Layer:
//!   InputLayerNorm -> Attention(Q/K/V norms) -> PostAttnNorm -> Residual
//!   PreFFNorm -> GeGLU MLP ─────────┐
//!   [Optional] Router(residual) ─┐  │ PostFFNorm1
//!               PreFFNorm2 ──> MoE ─┘ PostFFNorm2
//!                              Sum -> PostFFNorm -> Residual
//!   PerLayerGating -> PerLayerProjection -> PostPerLayerNorm -> Residual
//!   * layer_scalar
//! ```
//!
//! Reference: `reference/vllm/vllm/model_executor/models/gemma4.py`

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::repeat_kv;
use crate::layers::RotaryEmbedding;

use super::tp_layers::{TpContext, TpEmbedding, TpGeGluMlp, TpLinear};

// ─── Gemma4 RMSNorm ────────────────────────────────────────────────────────
//
// Unlike Gemma 2/3 which use `(1 + weight) * x`, Gemma 4 uses plain
// `weight * x` — the reference Python `gemma4.py` imports `RMSNorm`, not
// `GemmaRMSNorm`. Implemented by `crate::layers::RmsNormVariant::Standard`
// since Phase 3a; the type alias keeps `Gemma4RmsNorm` available for
// callers (notably `gemma4_vlm.rs` and `gemma4_quantized.rs`).

pub(crate) type Gemma4RmsNorm = crate::layers::RmsNorm;

#[inline]
pub(crate) fn gemma4_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<Gemma4RmsNorm> {
    crate::layers::rms_norm(size, eps, vb)
}

// ─── RMSNorm without learned weight (for v_norm and router norm) ──────────

pub(crate) type UnweightedRmsNorm = crate::layers::RmsNorm;

#[inline]
pub(crate) fn unweighted_rms_norm(eps: f64) -> UnweightedRmsNorm {
    crate::layers::rms_norm_unweighted(eps)
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

// ─── Config Extraction ─────────────────────────────────────────────────────

struct Gemma4ExtraConfig {
    attn_logit_softcap: Option<f64>,
    final_logit_softcap: Option<f64>,
    sliding_window_pattern: usize,
    /// RoPE base (`rope_theta`) used by sliding-attention layers. Sourced
    /// from `rope_parameters.sliding_attention.rope_theta` when present,
    /// with a fallback to the legacy flat `rope_local_base_freq` and
    /// finally `cfg.rope_theta`.
    rope_theta_local: f64,
    /// RoPE base for full-attention layers
    /// (`rope_parameters.full_attention.rope_theta`, fallback
    /// `cfg.rope_theta`). For Gemma 4 E2B this is 1e6 vs 10000 for sliding.
    rope_theta_full: f64,
    /// `partial_rotary_factor` applied to full-attention layers
    /// (default 1.0). For Gemma 4 E2B this is 0.25 — only the first
    /// 25% of `global_head_dim` participate in rotation.
    partial_rotary_factor_full: f64,
    /// `rope_type` for full-attention layers. Gemma 4 uses `"proportional"`
    /// which tweaks the `inv_freq` computation to use `head_dim` (not
    /// `rotary_dim`) as the exponent denominator.
    rope_type_full: String,
    layer_types: Vec<String>,
    // MoE
    enable_moe_block: bool,
    num_experts: usize,
    top_k_experts: usize,
    moe_intermediate_size: usize,
    // PLE
    hidden_size_per_layer_input: usize,
    vocab_size_per_layer_input: usize,
    // k_eq_v
    attention_k_eq_v: bool,
    // KV sharing
    num_kv_shared_layers: usize,
    // Double-wide MLP
    use_double_wide_mlp: bool,
    // Global head dim (may differ from default head_dim for full-attention layers)
    global_head_dim: Option<usize>,
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

        // Gemma 4 stores RoPE parameters under
        // `rope_parameters.{full,sliding}_attention`. We read those first
        // and fall back to the legacy flat fields for back-compat with
        // test configs.
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
        }
    }

    /// Construct the per-layer rotary embedding. Sliding layers use
    /// standard `RotaryEmbedding::new` with `rope_theta_local`. Full
    /// layers dispatch on `rope_type_full`: `"proportional"` → Gemma 4's
    /// proportional variant, anything else → standard (partial if
    /// `partial_rotary_factor_full < 1.0`).
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

    /// Target layer index for KV cache sharing. Returns `Some(target)` for
    /// layers in the last `num_kv_shared_layers`, pointing at the most
    /// recent non-shared layer of the same attention type. `None` means
    /// the layer owns its KV cache.
    fn kv_sharing_target_layer(&self, layer_idx: usize, num_hidden_layers: usize) -> Option<usize> {
        if self.num_kv_shared_layers == 0 || num_hidden_layers == 0 {
            return None;
        }
        let first_shared = num_hidden_layers.saturating_sub(self.num_kv_shared_layers);
        if layer_idx < first_shared {
            return None;
        }
        // Walk back through the non-shared layers for the latest layer
        // of the same attention type.
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

    fn is_kv_shared_layer(&self, layer_idx: usize, num_hidden_layers: usize) -> bool {
        if self.num_kv_shared_layers == 0 {
            return false;
        }
        // saturating_sub: a reduced-layer smoke test may keep a production
        // `num_kv_shared_layers` that exceeds the shrunk model depth.
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

    /// Largest per-layer `head_dim` used by any attention layer in the
    /// model. All layers share the same KV cache geometry, so the cache
    /// is allocated at this width and per-layer reads/writes slice back
    /// to their true `head_dim`.
    fn max_cache_head_dim(&self, default_head_dim: usize) -> usize {
        default_head_dim.max(self.global_head_dim.unwrap_or(default_head_dim))
    }
}

// ─── Gemma4 Router ──────────────────────────────────────────────────────────
//
// Custom router: UnweightedRmsNorm → root_size scaling → learned scale → gate

struct Gemma4Router {
    norm: UnweightedRmsNorm,
    scale: Tensor,
    gate: candle_nn::Linear,
    root_size: f64,
}

impl Gemma4Router {
    fn new(
        hidden_size: usize,
        num_experts: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = unweighted_rms_norm(rms_norm_eps);
        let scale = vb.get(hidden_size, "scale")?;
        let gate = candle_nn::linear_no_bias(hidden_size, num_experts, vb.pp("proj"))?;
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

// ─── MoE Expert ─────────────────────────────────────────────────────────────
//
// Fused gate_up_proj with GELU activation (not SiLU).

struct Gemma4MoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Gemma4MoEExpert {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("gate_proj"),
            pg,
        )?;
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false,
            false,
            vb.pp("up_proj"),
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

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs, tp_ctx)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = gate.gelu_erf()?.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── MoE Block ──────────────────────────────────────────────────────────────
//
// Custom Gemma4 routing: softmax-all → top-k → renormalize → fold per_expert_scale

struct Gemma4MoE {
    router: Gemma4Router,
    per_expert_scale: Tensor,
    experts: Vec<Gemma4MoEExpert>,
    num_experts: usize,
    top_k: usize,
}

impl Gemma4MoE {
    fn new(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        rms_norm_eps: f64,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_experts = extra_cfg.num_experts;
        let top_k = extra_cfg.top_k_experts;
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = extra_cfg.moe_intermediate_size;

        let router = Gemma4Router::new(hidden_size, num_experts, rms_norm_eps, vb.pp("router"))?;
        let per_expert_scale = vb.get(num_experts, "per_expert_scale")?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(Gemma4MoEExpert::new(
                hidden_size,
                moe_intermediate_size,
                vb_experts.pp(i),
                pg,
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

    fn forward(&self, xs: &Tensor, router_input: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router logits from the residual stream
        let router_input_2d = router_input.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward(&router_input_2d)?;

        // Gemma4 routing: topk on raw logits, softmax on all, then renormalize
        let per_expert_scale_f32 = self.per_expert_scale.to_dtype(DType::F32)?;

        let mut final_output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for token_idx in 0..num_tokens {
            let token_input = xs_2d.narrow(0, token_idx, 1)?;
            let token_logits = router_logits
                .narrow(0, token_idx, 1)?
                .flatten_all()?
                .to_dtype(DType::F32)?;

            // Top-k selection on raw logits
            let logits_vec: Vec<f32> = token_logits.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let topk_indices: Vec<usize> = indexed.iter().take(self.top_k).map(|x| x.0).collect();

            // Softmax over ALL experts
            let probs =
                candle_nn::ops::softmax_last_dim(&token_logits.unsqueeze(0)?)?.squeeze(0)?;
            let probs_vec: Vec<f32> = probs.to_vec1()?;

            // Build indicator and compute dispatch weights
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

            // Fold per_expert_scale
            let expert_scales: Vec<f32> = per_expert_scale_f32.to_vec1()?;

            let mut token_output = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;

            for &expert_idx in &topk_indices {
                if expert_idx < self.num_experts {
                    let weight =
                        (gate_weights[expert_idx] / renorm_factor) * expert_scales[expert_idx];
                    let expert_out = self.experts[expert_idx].forward(&token_input, tp_ctx)?;
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

// ─── Attention ──────────────────────────────────────────────────────────────

/// Pad a 4D `[b, heads, seq, dim]` tensor on its last dimension from `from`
/// to `to` with zeros. Used to widen K/V from a layer-specific head_dim
/// (256 for sliding layers, 512 for full in E2B) to the shared cache
/// head_dim. Zero-padded dimensions contribute identically to attention
/// as long as both sides (K and Q view from cache) are sliced back
/// symmetrically.
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

struct Gemma4Attention {
    q_proj: TpLinear,
    /// KV-shared layers (layers in the last `num_kv_shared_layers`) have
    /// no `k_proj` / `v_proj` weights in the checkpoint. For those layers
    /// these fields are `None` and the forward pass reads K/V straight
    /// from the target layer's cache engine.
    k_proj: Option<TpLinear>,
    v_proj: Option<TpLinear>,
    o_proj: TpLinear,
    q_norm: Gemma4RmsNorm,
    k_norm: Option<Gemma4RmsNorm>,
    v_norm: Option<UnweightedRmsNorm>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    /// Layer-specific head dimension (256 for sliding, 512 for full in E2B).
    head_dim: usize,
    /// Padded head dimension stored in the shared KV cache. Equal to
    /// `max(head_dim, global_head_dim)` across all layers.
    cache_head_dim: usize,
    attn_logit_softcap: Option<f64>,
    sliding_window: Option<usize>,
    /// `true` when this layer reuses another layer's KV cache — no K/V
    /// projection, no K/V norm, no K/V RoPE, no cache write.
    is_kv_shared: bool,
}

impl Gemma4Attention {
    #[allow(clippy::too_many_arguments)]
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        layer_idx: usize,
        cache_head_dim: usize,
        is_kv_shared: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = extra_cfg.head_dim_for_layer(layer_idx, cfg.head_dim);
        let world_size = pg.world_size();

        if world_size > 1 {
            if !num_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if !num_kv_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;

        // KV-shared layers carry no k_proj/v_proj/k_norm/v_norm. The
        // reference Python `Gemma4Attention` simply doesn't create those
        // modules and the weight loader never sees the corresponding
        // checkpoint keys.
        let (k_proj, v_proj, k_norm, v_norm) = if is_kv_shared {
            (None, None, None, None)
        } else {
            let k_proj = TpLinear::column_parallel(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                false,
                false,
                vb.pp("k_proj"),
                pg,
            )?;

            let use_k_eq_v =
                extra_cfg.attention_k_eq_v && extra_cfg.is_full_attention_layer(layer_idx);
            let v_proj = if use_k_eq_v {
                match TpLinear::column_parallel(
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    false,
                    false,
                    vb.pp("v_proj"),
                    pg,
                ) {
                    Ok(v) => v,
                    Err(_) => TpLinear::column_parallel(
                        cfg.hidden_size,
                        num_kv_heads * head_dim,
                        false,
                        false,
                        vb.pp("k_proj"),
                        pg,
                    )?,
                }
            } else {
                TpLinear::column_parallel(
                    cfg.hidden_size,
                    num_kv_heads * head_dim,
                    false,
                    false,
                    vb.pp("v_proj"),
                    pg,
                )?
            };

            let k_norm = gemma4_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            let v_norm = unweighted_rms_norm(cfg.rms_norm_eps);

            (Some(k_proj), Some(v_proj), Some(k_norm), Some(v_norm))
        };

        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

        let q_norm = gemma4_rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        let rotary_emb = extra_cfg.build_rotary_for_layer(
            layer_idx,
            head_dim,
            cfg.max_position_embeddings,
            vb.dtype(),
            vb.device(),
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
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            cache_head_dim,
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
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;

        if self.is_kv_shared {
            // Shared layers rotate only Q. K/V are read straight from the
            // target layer's cache (at `cache_head_dim`), no new write.
            let (q, _) = self.rotary_emb.apply(&q, &q, seqlen_offset)?;

            let num_tokens = seqlen_offset + q_len;
            let (k_full, v_full) = cache_engine
                .read(block_table.block_ids(), num_tokens)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;
            self.finalize_attention(&q, &k_full, &v_full, q_len, b_sz, seqlen_offset, tp_ctx)
        } else {
            // Non-shared: full K/V computation + cache write.
            let k = self
                .k_proj
                .as_ref()
                .expect("non-shared layer must have k_proj")
                .forward(xs, tp_ctx)?;
            let v = self
                .v_proj
                .as_ref()
                .expect("non-shared layer must have v_proj")
                .forward(xs, tp_ctx)?;

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

            // Pad K/V from layer head_dim to the shared cache head_dim.
            let k_padded = pad_last_dim(&k, self.head_dim, self.cache_head_dim)?;
            let v_padded = pad_last_dim(&v, self.head_dim, self.cache_head_dim)?;

            let k_for_cache = k_padded.squeeze(0)?.contiguous()?;
            let v_for_cache = v_padded.squeeze(0)?.contiguous()?;
            cache_engine
                .write(&k_for_cache, &v_for_cache, slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let num_tokens = seqlen_offset + q_len;
            let (k_full, v_full) = cache_engine
                .read(block_table.block_ids(), num_tokens)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

            let _ = (device, dtype);
            self.finalize_attention(&q, &k_full, &v_full, q_len, b_sz, seqlen_offset, tp_ctx)
        }
    }

    /// Shared tail: slice padded K/V back to `head_dim`, broadcast KV
    /// groups, apply optional soft cap + mask, softmax, compute the
    /// weighted sum and project out.
    #[allow(clippy::too_many_arguments)]
    fn finalize_attention(
        &self,
        q: &Tensor,
        k_full: &Tensor,
        v_full: &Tensor,
        q_len: usize,
        b_sz: usize,
        seqlen_offset: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let device = q.device();
        let dtype = q.dtype();

        // K/V are stored at cache_head_dim; trim back to this layer's
        // actual head_dim so the matmul shapes line up with Q.
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

        let kv_len = k_full.dim(2)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        // Gemma 4 uses scaling=1.0 — the Q/K norms control magnitude.
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

        self.o_proj.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let device = xs.device();
        let dtype = xs.dtype();

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = Self::apply_per_head_norm(&q, &self.q_norm)?;

        // Non-shared layers compute + cache K/V up front; shared layers
        // skip this entirely and read directly from the target engine.
        let (k, v) = if self.is_kv_shared {
            (None, None)
        } else {
            let k = self
                .k_proj
                .as_ref()
                .expect("non-shared layer must have k_proj")
                .forward(xs, tp_ctx)?;
            let v = self
                .v_proj
                .as_ref()
                .expect("non-shared layer must have v_proj")
                .forward(xs, tp_ctx)?;
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

            let (q_i, _k_rotated) = if self.is_kv_shared {
                self.rotary_emb.apply(&q_i, &q_i, seq.seqlen_offset)?
            } else {
                let k_i = k.as_ref().unwrap().narrow(0, i, 1)?;
                let v_i = v.as_ref().unwrap().narrow(0, i, 1)?;
                let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;
                let k_padded = pad_last_dim(&k_i, self.head_dim, self.cache_head_dim)?;
                let v_padded = pad_last_dim(&v_i, self.head_dim, self.cache_head_dim)?;
                let k_for_cache = k_padded.squeeze(0)?.contiguous()?;
                let v_for_cache = v_padded.squeeze(0)?.contiguous()?;
                cache_engine
                    .write(&k_for_cache, &v_for_cache, &seq.slot_mapping)
                    .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;
                (q_i, k_i)
            };

            let kv_len = seq.seqlen_offset + 1;
            let (k_full, v_full) = cache_engine
                .read(&seq.block_ids, kv_len)
                .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

            // Slice padded cache back to this layer's real head_dim.
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
        self.o_proj.forward(&attn_output, tp_ctx)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Gemma4DecoderLayer {
    self_attn: Gemma4Attention,
    mlp: TpGeGluMlp,
    // MoE (optional — parallel with MLP)
    moe: Option<Gemma4MoE>,
    // Norms
    input_layernorm: Gemma4RmsNorm,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
    // Extra MoE norms (only when MoE enabled)
    post_feedforward_layernorm_1: Option<Gemma4RmsNorm>,
    post_feedforward_layernorm_2: Option<Gemma4RmsNorm>,
    pre_feedforward_layernorm_2: Option<Gemma4RmsNorm>,
    // PLE components (optional — when hidden_size_per_layer_input > 0)
    per_layer_input_gate: Option<candle_nn::Linear>,
    per_layer_projection: Option<candle_nn::Linear>,
    post_per_layer_input_norm: Option<Gemma4RmsNorm>,
    // Layer scalar (per-layer multiplier)
    layer_scalar: Tensor,
    /// `Some(target_layer_idx)` for KV-shared layers — the attention
    /// forward reads K/V from the target layer's cache engine instead of
    /// owning its own.
    kv_sharing_target: Option<usize>,
}

impl Gemma4DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Gemma4ExtraConfig,
        layer_idx: usize,
        cache_head_dim: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let kv_sharing_target = extra_cfg.kv_sharing_target_layer(layer_idx, cfg.num_hidden_layers);
        let is_kv_shared = kv_sharing_target.is_some();

        let self_attn = Gemma4Attention::new_with_tp(
            cfg,
            extra_cfg,
            layer_idx,
            cache_head_dim,
            is_kv_shared,
            vb.pp("self_attn"),
            pg,
        )?;

        let intermediate_size = extra_cfg.layer_intermediate_size(
            layer_idx,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
        );
        let mlp = TpGeGluMlp::new(cfg.hidden_size, intermediate_size, vb.pp("mlp"), pg)?;

        // MoE block (optional)
        let moe = if extra_cfg.enable_moe_block && extra_cfg.num_experts > 0 {
            Some(Gemma4MoE::new(
                cfg,
                extra_cfg,
                cfg.rms_norm_eps,
                vb.pp("moe"),
                pg,
            )?)
        } else {
            None
        };

        let input_layernorm =
            gemma4_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = gemma4_rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        // Extra norms for MoE layers
        let (
            post_feedforward_layernorm_1,
            post_feedforward_layernorm_2,
            pre_feedforward_layernorm_2,
        ) = if moe.is_some() {
            (
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_1"),
                )?),
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("post_feedforward_layernorm_2"),
                )?),
                Some(gemma4_rms_norm(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    vb.pp("pre_feedforward_layernorm_2"),
                )?),
            )
        } else {
            (None, None, None)
        };

        // PLE components
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
            if extra_cfg.hidden_size_per_layer_input > 0 {
                (
                    Some(candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        extra_cfg.hidden_size_per_layer_input,
                        vb.pp("per_layer_input_gate"),
                    )?),
                    Some(candle_nn::linear_no_bias(
                        extra_cfg.hidden_size_per_layer_input,
                        cfg.hidden_size,
                        vb.pp("per_layer_projection"),
                    )?),
                    Some(gemma4_rms_norm(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_per_layer_input_norm"),
                    )?),
                )
            } else {
                (None, None, None)
            };

        // Layer scalar (buffer, not a trained weight — defaults to 1.0)
        let layer_scalar = vb
            .get(1, "layer_scalar")
            .unwrap_or_else(|_| Tensor::ones(1, DType::F32, vb.device()).unwrap());

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

    fn forward_ffn(
        &self,
        residual: &Tensor,
        hidden_states_mlp: Tensor,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        if let (Some(moe), Some(pf_norm1), Some(pf_norm2), Some(pre_ff2)) = (
            &self.moe,
            &self.post_feedforward_layernorm_1,
            &self.post_feedforward_layernorm_2,
            &self.pre_feedforward_layernorm_2,
        ) {
            // MoE enabled: MLP and MoE run in parallel
            let h1 = pf_norm1.forward(&hidden_states_mlp)?;

            // MoE sees the residual (pre-MLP state)
            let h2 = pre_ff2.forward(residual)?;
            let h2 = moe.forward(&h2, residual, tp_ctx)?;
            let h2 = pf_norm2.forward(&h2)?;

            Ok((h1 + h2)?)
        } else {
            Ok(hidden_states_mlp)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        _attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        // KV-shared layers read from the target layer's cache engine
        // instead of their own slot. The target was already populated
        // earlier in the forward pass because layers execute in order.
        let cache_layer_idx = self.kv_sharing_target.unwrap_or(layer_idx);
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            None,
            seqlen_offset,
            kv_cache_mgr.engine_mut(cache_layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        // Dense MLP
        let hidden_states_norm = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states_mlp = self.mlp.forward(&hidden_states_norm, tp_ctx)?;

        // FFN: MLP only or MLP + MoE
        let hidden_states = self.forward_ffn(residual, hidden_states_mlp, tp_ctx)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        // PLE gating
        let hidden_states = self.apply_ple(&hidden_states, per_layer_input)?;

        // Layer scalar
        hidden_states.broadcast_mul(&self.layer_scalar)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;

        let cache_layer_idx = self.kv_sharing_target.unwrap_or(layer_idx);
        let hidden_states = self.self_attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(cache_layer_idx),
            tp_ctx,
        )?;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        // Dense MLP
        let hidden_states_norm = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states_mlp = self.mlp.forward(&hidden_states_norm, tp_ctx)?;

        // FFN: MLP only or MLP + MoE
        let hidden_states = self.forward_ffn(residual, hidden_states_mlp, tp_ctx)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        // PLE gating
        let hidden_states = self.apply_ple(&hidden_states, per_layer_input)?;

        // Layer scalar
        hidden_states.broadcast_mul(&self.layer_scalar)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Gemma4ForCausalLM {
    embed_tokens: TpEmbedding,
    // PLE embeddings (optional — when hidden_size_per_layer_input > 0)
    embed_tokens_per_layer: Option<TpEmbedding>,
    per_layer_model_projection: Option<TpLinear>,
    per_layer_projection_norm: Option<Gemma4RmsNorm>,
    layers: Vec<Gemma4DecoderLayer>,
    norm: Gemma4RmsNorm,
    lm_head: TpLinear,
    hidden_size: usize,
    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,
    final_logit_softcap: Option<f64>,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Gemma4ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        // Standalone Gemma 4 text checkpoints store the inner
        // transformer under a `model.` wrapper, so we descend one level
        // here. The VLM wrapper (`Gemma4ForConditionalGeneration`) calls
        // `new_with_tp_at_root` directly with a VarBuilder that's
        // already positioned at the language-model root and skips the
        // extra `vb.pp("model")` step.
        Self::new_with_tp_at_root(cfg, vb.pp("model"), Some(vb), pg, tp_ctx)
    }

    /// Construct a Gemma 4 text model where `vb_m` already points at
    /// the inner `model.*` namespace (i.e. `vb_m.pp("embed_tokens")`
    /// resolves to a real tensor name without an extra `model.` prefix).
    ///
    /// `lm_head_root` is an optional VarBuilder positioned at the same
    /// root as the (untied) `lm_head` weight — typically the parent of
    /// `vb_m`. When `tie_word_embeddings` is true the parameter is
    /// ignored. The wrapper VLM passes `None` because Gemma 4 always
    /// ties embeddings.
    pub fn new_with_tp_at_root(
        cfg: &ModelConfig,
        vb_m: VarBuilder,
        lm_head_root: Option<VarBuilder>,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let extra_cfg = Gemma4ExtraConfig::from_model_config(cfg);
        let vb = vb_m.clone();
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        // PLE components
        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if extra_cfg.hidden_size_per_layer_input > 0 {
                let total_ple_dim = extra_cfg.hidden_size_per_layer_input * cfg.num_hidden_layers;

                let ple_embed = TpEmbedding::new(
                    extra_cfg.vocab_size_per_layer_input,
                    total_ple_dim,
                    vb_m.pp("embed_tokens_per_layer"),
                    pg,
                )?;

                let ple_proj = TpLinear::column_parallel(
                    cfg.hidden_size,
                    total_ple_dim,
                    false,
                    true, // gather output for PLE
                    vb_m.pp("per_layer_model_projection"),
                    pg,
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
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma4DecoderLayer::new_with_tp(
                cfg,
                &extra_cfg,
                i,
                cache_head_dim,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = gemma4_rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            if world_size == 1 {
                let emb_weights = embed_tokens
                    .embeddings()
                    .expect("single GPU should have accessible embeddings")
                    .clone();
                TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
            } else {
                TpLinear::column_parallel(
                    cfg.hidden_size,
                    cfg.vocab_size,
                    false,
                    true,
                    vb_m.pp("embed_tokens"),
                    pg,
                )?
            }
        } else {
            // Untied lm_head sits OUTSIDE the inner model namespace —
            // for the standalone Gemma 4 checkpoint that's `lm_head.weight`
            // at the safetensors root. The caller passes the right vb
            // via `lm_head_root`; if absent (e.g. VLM nested call) we
            // still try `vb_m.pp("lm_head")` as a defensive fallback so
            // the constructor doesn't crash for unusual layouts.
            let head_vb = lm_head_root.as_ref().unwrap_or(&vb_m);
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                head_vb.pp("lm_head"),
                pg,
            )?
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
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Compute PLE per-layer inputs from input_ids and hidden_states.
    ///
    /// Returns tensor of shape [..., num_layers, hidden_size_per_layer_input] or None.
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
        let input_scale = (2.0_f64).powf(-0.5); // 1/sqrt(2)

        // Per-layer embeddings: [T, total_ple_dim]
        let ple_embed = (embed_pl.forward(input_ids, &self.tp_ctx)? * embed_scale)?;
        let shape = input_ids.dims();
        let batch_seq: Vec<usize> = shape.to_vec();
        let mut new_shape = batch_seq;
        new_shape.push(num_layers);
        new_shape.push(ple_dim);
        let ple_embed = ple_embed.reshape(&new_shape[..])?;

        // Projection from hidden_states: [T, total_ple_dim]
        let ple_proj = (proj.forward(hidden_states, &self.tp_ctx)? * proj_scale)?;
        let proj_shape = {
            let hs = hidden_states.dims();
            let mut s: Vec<usize> = hs[..hs.len() - 1].to_vec();
            s.push(num_layers);
            s.push(ple_dim);
            s
        };
        let ple_proj = ple_proj.reshape(&proj_shape[..])?;

        // Normalize each per-layer slice of the projection
        let ndim = ple_proj.dims().len();
        let layer_dim = ndim - 2;
        let mut normed_slices = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let slice = ple_proj.narrow(layer_dim, l, 1)?.squeeze(layer_dim)?;
            let normed = norm.forward(&slice)?;
            normed_slices.push(normed.unsqueeze(layer_dim)?);
        }
        let ple_proj_normed = Tensor::cat(&normed_slices, layer_dim)?;

        // Combine: (projection + embedding) * 1/sqrt(2)
        let per_layer_inputs = ((ple_proj_normed + ple_embed)? * input_scale)?;
        Ok(Some(per_layer_inputs))
    }

    /// Extract per-layer input for a specific layer from the combined tensor.
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
        let (_b_size, seq_len) = input_ids.dims2()?;
        let normalizer = (self.hidden_size as f64).sqrt();
        let xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

        // Compute PLE per-layer inputs
        let per_layer_inputs = self.compute_per_layer_inputs(input_ids, &xs)?;

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

        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;

            xs = layer.forward(
                &xs,
                pli.as_ref(),
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

        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }

        Ok(logits)
    }

    /// Embed text tokens (for VLM use — embed only, no layers).
    /// Applies Gemma4's sqrt(hidden_size) normalization.
    pub(crate) fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let normalizer = (self.hidden_size as f64).sqrt();
        self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer
    }

    /// Forward with pre-computed embeddings (for VLM use — skips embedding layer).
    pub(crate) fn forward_with_embeddings(
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

        // PLE: use a dummy input_ids of zeros for the per-layer embedding lookup
        let dummy_shape: Vec<usize> = embeddings.dims()[..embeddings.dims().len() - 1].to_vec();
        let dummy_ids = Tensor::zeros(&dummy_shape[..], DType::U32, &self.device)?;
        let per_layer_inputs = self.compute_per_layer_inputs(&dummy_ids, embeddings)?;

        let mut xs = embeddings.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let pli = Self::extract_per_layer_input(&per_layer_inputs, layer_idx)?;

            xs = layer.forward(
                &xs,
                pli.as_ref(),
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
        if let Some(cap) = self.final_logit_softcap {
            logits = soft_cap(&logits, cap)?;
        }
        Ok(logits)
    }

    /// Forward decode batch with pre-computed embeddings (for VLM use).
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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Gemma4ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Gemma4ForCausalLM::forward(
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
        let xs = (self.embed_tokens.forward(input_ids, &self.tp_ctx)? * normalizer)?;

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
    fn test_gemma4_config() {
        let cfg = test_config();
        let extra_cfg = Gemma4ExtraConfig::from_model_config(&cfg);

        assert_eq!(extra_cfg.hidden_size_per_layer_input, 16);
        assert_eq!(extra_cfg.final_logit_softcap, Some(30.0));
        assert!(!extra_cfg.enable_moe_block);
        assert_eq!(extra_cfg.sliding_window_pattern, 2);

        assert!(extra_cfg.is_sliding_layer(0));
        assert!(!extra_cfg.is_sliding_layer(1));
        assert!(extra_cfg.is_sliding_layer(2));
        assert!(!extra_cfg.is_sliding_layer(3));
    }

    #[test]
    fn test_gemma4_config_moe() {
        let cfg = test_config_with_moe(true);
        let extra_cfg = Gemma4ExtraConfig::from_model_config(&cfg);

        assert!(extra_cfg.enable_moe_block);
        assert_eq!(extra_cfg.num_experts, 4);
        assert_eq!(extra_cfg.top_k_experts, 2);
        assert_eq!(extra_cfg.moe_intermediate_size, 32);
    }

    #[test]
    fn test_gemma4_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma4ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Gemma4ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert!(model.embed_tokens_per_layer.is_some());
        assert!(model.per_layer_model_projection.is_some());
    }

    #[test]
    fn test_gemma4_construction_with_moe() {
        let cfg = test_config_with_moe(true);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Gemma4ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Gemma4ForCausalLM with MoE should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.layers[0].moe.is_some());
    }

    #[test]
    fn test_gemma4_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gemma4_single_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gemma4_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gemma4_forward_with_moe() {
        let cfg = test_config_with_moe(true);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
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
            .expect("forward with MoE");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size],);
    }

    #[test]
    fn test_gemma4_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Gemma4ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_gemma4_router() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let router = Gemma4Router::new(64, 4, 1e-6, vb.pp("router")).expect("router");

        let xs = Tensor::randn(0.0f32, 1.0, (2, 64), &device).expect("input");
        let logits = router.forward(&xs).expect("forward");
        assert_eq!(logits.dims(), &[2, 4]);
    }

    #[test]
    fn test_gemma4_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = crate::distributed::LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = Gemma4ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Gemma4ForCausalLM should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }
}
