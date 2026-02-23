//! Aria vision-language model.
//!
//! Architecture:
//! - Vision tower: Idefics3/SigLIP vision transformer (post_layernorm replaced with identity)
//! - Projector: cross-attention (learnable queries × ViT features) + MLP
//! - Language model: LlamaModel with every MLP replaced by a mixture-of-experts layer
//!
//! # Weight paths (original HF checkpoint format)
//!
//! ```text
//! vision_tower.*                                — SigLIP ViT (no post_layernorm)
//! multi_modal_projector.query                   — learnable queries [max_q, vis_hidden]
//! multi_modal_projector.cross_attn.*            — AriaCrossAttention
//! multi_modal_projector.layer_norm.*            — LayerNorm before feed-forward
//! multi_modal_projector.feed_forward.*          — AriaProjectorMLP
//! language_model.model.embed_tokens.*           — token embeddings
//! language_model.model.layers.{i}.self_attn.*  — attention
//! language_model.model.layers.{i}.mlp.*        — MoE (router + shared_experts + experts)
//! language_model.model.norm.*                  — final RMSNorm
//! language_model.lm_head.*                     — LM head
//! ```
//!
//! For checkpoints saved after transformers v4.52, all paths are prefixed with `model.`:
//! apply the vLLM `WeightsMapper` to strip the prefix before loading.
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/aria.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    layer_norm, linear, linear_no_bias, ops::softmax_last_dim, LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::distributed::LocalProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::llama::LlamaAttention;
use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Config ───────────────────────────────────────────────────────────────────

fn read_int(v: Option<&serde_json::Value>, key: &str, default: usize) -> usize {
    v.and_then(|v| v.get(key))
        .and_then(|v| v.as_u64())
        .unwrap_or(default as u64) as usize
}

fn read_float(v: Option<&serde_json::Value>, key: &str, default: f64) -> f64 {
    v.and_then(|v| v.get(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

/// Vision encoder configuration (Idefics3/SigLIP-based).
#[derive(Debug, Clone)]
struct AriaVisionCfg {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    image_size: usize,
    patch_size: usize,
    num_channels: usize,
    layer_norm_eps: f64,
}

impl Default for AriaVisionCfg {
    fn default() -> Self {
        // Aria-8B uses SigLIP ViT with 980×980 images, patch_size=14
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 980,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        }
    }
}

impl AriaVisionCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg.extra.get("vision_config");
        let d = Self::default();
        Self {
            hidden_size: read_int(vc, "hidden_size", d.hidden_size),
            intermediate_size: read_int(vc, "intermediate_size", d.intermediate_size),
            num_attention_heads: read_int(vc, "num_attention_heads", d.num_attention_heads),
            num_hidden_layers: read_int(vc, "num_hidden_layers", d.num_hidden_layers),
            image_size: read_int(vc, "image_size", d.image_size),
            patch_size: read_int(vc, "patch_size", d.patch_size),
            num_channels: read_int(vc, "num_channels", d.num_channels),
            layer_norm_eps: read_float(vc, "layer_norm_eps", d.layer_norm_eps),
        }
    }
}

/// Text model (LlamaMoE) configuration.
#[derive(Debug, Clone)]
struct AriaTextCfg {
    hidden_size: usize,
    /// Per-expert intermediate size.
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
    moe_num_experts: usize,
    moe_topk: usize,
    moe_num_shared_experts: usize,
}

impl AriaTextCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        // Prefer text_config sub-object; fall back to top-level fields.
        let tc = cfg.extra.get("text_config");
        let hs = read_int(tc, "hidden_size", cfg.hidden_size);
        let na = read_int(tc, "num_attention_heads", cfg.num_attention_heads);
        let nkv = read_int(tc, "num_key_value_heads", na);

        // MoE fields: check text_config first, then top-level extra.
        let moe_int = |key: &str, default: usize| -> usize {
            tc.and_then(|v| v.get(key))
                .and_then(|v| v.as_u64())
                .or_else(|| cfg.extra.get(key).and_then(|v| v.as_u64()))
                .unwrap_or(default as u64) as usize
        };

        Self {
            hidden_size: hs,
            intermediate_size: read_int(tc, "intermediate_size", cfg.intermediate_size),
            num_hidden_layers: read_int(tc, "num_hidden_layers", cfg.num_hidden_layers),
            num_attention_heads: na,
            num_key_value_heads: nkv,
            vocab_size: read_int(tc, "vocab_size", cfg.vocab_size),
            max_position_embeddings: read_int(
                tc,
                "max_position_embeddings",
                cfg.max_position_embeddings,
            ),
            head_dim: hs / na,
            rms_norm_eps: read_float(tc, "rms_norm_eps", cfg.rms_norm_eps),
            rope_theta: read_float(tc, "rope_theta", cfg.rope_theta),
            tie_word_embeddings: cfg.tie_word_embeddings,
            moe_num_experts: moe_int("moe_num_experts", 8),
            moe_topk: moe_int("moe_topk", 2),
            moe_num_shared_experts: moe_int("moe_num_shared_experts", 2),
        }
    }

    /// Convert to a flat ModelConfig for use with LlamaAttention.
    fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            architectures: vec!["AriaForConditionalGeneration".to_string()],
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            head_dim: self.head_dim,
            hidden_act: "silu".to_string(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            tie_word_embeddings: self.tie_word_embeddings,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }
}

// ─── patch_to_query_dict parser ───────────────────────────────────────────────

fn parse_patch_to_query_dict(cfg: &ModelConfig) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    if let Some(serde_json::Value::Object(dict)) = cfg.extra.get("projector_patch_to_query_dict") {
        for (k, v) in dict {
            if let (Ok(patches), Some(queries)) = (k.parse::<usize>(), v.as_u64()) {
                map.insert(patches, queries as usize);
            }
        }
    }
    // Default Aria-8B values.
    if map.is_empty() {
        map.insert(1225, 128);
        map.insert(4900, 256);
    }
    map
}

// ─── AriaCrossAttention ───────────────────────────────────────────────────────

/// Cross-attention module: queries attend to vision features.
///
/// Replicates PyTorch's `nn.MultiheadAttention(batch_first=True)` with separate
/// pre-projections for Q (from query tensor) and KV (from vision features).
struct AriaCrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    /// MHA internal Q/K/V projection weights [H, H], sliced from in_proj_weight.
    mha_q_weight: Tensor,
    mha_k_weight: Tensor,
    mha_v_weight: Tensor,
    mha_q_bias: Tensor,
    mha_k_bias: Tensor,
    mha_v_bias: Tensor,
    /// MHA output projection weight [H, H] and bias [H].
    mha_out_weight: Tensor,
    mha_out_bias: Tensor,
    /// Final linear projection (with bias) applied after MHA.
    linear: Linear,
    /// LayerNorm applied to queries before q_proj.
    layer_norm: LayerNorm,
    /// LayerNorm applied to kv_states before k_proj/v_proj.
    layer_norm_kv: LayerNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AriaCrossAttention {
    fn new(in_features: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = in_features / num_heads;
        let q_proj = linear_no_bias(in_features, in_features, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(in_features, in_features, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(in_features, in_features, vb.pp("v_proj"))?;

        // PyTorch MHA stores Q/K/V weights concatenated: in_proj_weight [3H, H].
        let vb_mha = vb.pp("multihead_attn");
        let in_proj_w = vb_mha.get((3 * in_features, in_features), "in_proj_weight")?;
        let in_proj_b = vb_mha.get(3 * in_features, "in_proj_bias")?;
        let mha_q_weight = in_proj_w.narrow(0, 0, in_features)?;
        let mha_k_weight = in_proj_w.narrow(0, in_features, in_features)?;
        let mha_v_weight = in_proj_w.narrow(0, 2 * in_features, in_features)?;
        let mha_q_bias = in_proj_b.narrow(0, 0, in_features)?;
        let mha_k_bias = in_proj_b.narrow(0, in_features, in_features)?;
        let mha_v_bias = in_proj_b.narrow(0, 2 * in_features, in_features)?;

        let vb_out = vb_mha.pp("out_proj");
        let mha_out_weight = vb_out.get((in_features, in_features), "weight")?;
        let mha_out_bias = vb_out.get(in_features, "bias")?;

        let linear = linear(in_features, in_features, vb.pp("linear"))?;
        let ln_fn = layer_norm;
        let layer_norm = ln_fn(in_features, 1e-5, vb.pp("layer_norm"))?;
        let layer_norm_kv = ln_fn(in_features, 1e-5, vb.pp("layer_norm_kv"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            mha_q_weight,
            mha_k_weight,
            mha_v_weight,
            mha_q_bias,
            mha_k_bias,
            mha_v_bias,
            mha_out_weight,
            mha_out_bias,
            linear,
            layer_norm,
            layer_norm_kv,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// `kv_states`: [B, kv_len, H] — vision features (key/value source).
    /// `queries`:   [B, q_num, H]  — learnable queries.
    /// Returns      [B, q_num, H].
    fn forward(&self, kv_states: &Tensor, queries: &Tensor) -> Result<Tensor> {
        let (b, q_num, h) = queries.dims3()?;
        let kv_len = kv_states.dim(1)?;

        // Pre-projection (from external projections, applied before MHA).
        let q = self.q_proj.forward(&self.layer_norm.forward(queries)?)?; // [B, q_num, H]
        let kv_n = self.layer_norm_kv.forward(kv_states)?;
        let k = self.k_proj.forward(&kv_n)?; // [B, kv_len, H]
        let v = self.v_proj.forward(&kv_n)?; // [B, kv_len, H]

        // MHA internal projections: xW^T + b.
        let proj = |x: &Tensor, len: usize, w: &Tensor, bias: &Tensor| -> Result<Tensor> {
            (x.reshape((b * len, h))?
                .matmul(&w.t()?)?
                .broadcast_add(bias)?)
            .reshape((b, len, h))
        };
        let q2 = proj(&q, q_num, &self.mha_q_weight, &self.mha_q_bias)?;
        let k2 = proj(&k, kv_len, &self.mha_k_weight, &self.mha_k_bias)?;
        let v2 = proj(&v, kv_len, &self.mha_v_weight, &self.mha_v_bias)?;

        // Reshape to [B, heads, len, head_dim] for SDPA.
        let to_mh = |x: &Tensor, len: usize| -> Result<Tensor> {
            x.reshape((b, len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q2 = to_mh(&q2, q_num)?;
        let k2 = to_mh(&k2, kv_len)?;
        let v2 = to_mh(&v2, kv_len)?;

        // Scaled dot-product attention.
        let scores = (q2.matmul(&k2.transpose(2, 3)?)? * self.scale)?;
        let weights = softmax_last_dim(&scores)?;
        let out = weights.matmul(&v2)?; // [B, heads, q_num, head_dim]

        // Merge heads: [B, q_num, H].
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, q_num, h))?;

        // MHA output projection.
        let out2 = (out
            .reshape((b * q_num, h))?
            .matmul(&self.mha_out_weight.t()?)?
            .broadcast_add(&self.mha_out_bias)?)
        .reshape((b, q_num, h))?;

        // Final linear (dropout = 0 at inference).
        self.linear.forward(&out2)
    }
}

// ─── AriaProjectorMlp ─────────────────────────────────────────────────────────

struct AriaProjectorMlp {
    linear_in: Linear,
    linear_out: Linear,
}

impl AriaProjectorMlp {
    fn new(
        in_features: usize,
        hidden_features: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_in = linear_no_bias(in_features, hidden_features, vb.pp("linear_in"))?;
        let linear_out = linear_no_bias(hidden_features, output_dim, vb.pp("linear_out"))?;
        Ok(Self {
            linear_in,
            linear_out,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // gelu_new (tanh approximation) matches AriaProjectorMLP in Python.
        self.linear_out
            .forward(&self.linear_in.forward(xs)?.gelu()?)
    }
}

// ─── AriaProjector ────────────────────────────────────────────────────────────

/// Projects vision features to text hidden space via cross-attention + MLP.
///
/// Learnable queries cross-attend to vision features; an MLP then maps to text dim.
struct AriaProjector {
    /// Learnable query parameter [max_query_num, in_features].
    query: Tensor,
    cross_attn: AriaCrossAttention,
    /// LayerNorm applied to cross-attention output before feed_forward.
    layer_norm: LayerNorm,
    feed_forward: AriaProjectorMlp,
    in_features: usize,
    patch_to_query_dict: HashMap<usize, usize>,
}

impl AriaProjector {
    fn new(
        in_features: usize,
        num_heads: usize,
        hidden_features: usize,
        output_dim: usize,
        max_query_num: usize,
        patch_to_query_dict: HashMap<usize, usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let query = vb.get((max_query_num, in_features), "query")?;
        let cross_attn = AriaCrossAttention::new(in_features, num_heads, vb.pp("cross_attn"))?;
        let layer_norm = layer_norm(in_features, 1e-5, vb.pp("layer_norm"))?;
        let feed_forward = AriaProjectorMlp::new(
            in_features,
            hidden_features,
            output_dim,
            vb.pp("feed_forward"),
        )?;
        Ok(Self {
            query,
            cross_attn,
            layer_norm,
            feed_forward,
            in_features,
            patch_to_query_dict,
        })
    }

    /// `x`: [B, num_patches, in_features].  Returns [B, query_num, output_dim].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, num_patches, _) = x.dims3()?;

        // Pick query count based on patch count; fall back to first entry.
        let query_num = self
            .patch_to_query_dict
            .get(&num_patches)
            .copied()
            .unwrap_or_else(|| {
                self.patch_to_query_dict
                    .values()
                    .copied()
                    .next()
                    .unwrap_or(num_patches)
            });

        // Expand learnable queries: [max_q, H] → [B, query_num, H].
        let queries = self
            .query
            .narrow(0, 0, query_num)?
            .unsqueeze(0)?
            .broadcast_as((b, query_num, self.in_features))?
            .contiguous()?;

        let attention_out = self.cross_attn.forward(x, &queries)?;
        self.feed_forward
            .forward(&self.layer_norm.forward(&attention_out)?)
    }
}

// ─── AriaTextMoELayer ─────────────────────────────────────────────────────────

/// Mixture-of-experts layer replacing the standard MLP in each Aria decoder layer.
///
/// Combines always-active shared experts (LlamaMLP) with top-K routed sparse experts.
struct AriaTextMoELayer {
    /// Router projection weights [num_experts, hidden_size].
    router_weight: Tensor,
    /// Shared experts: standard SwiGLU MLP, always active.
    shared_experts: TpSwiGluMlp,
    /// Sparse expert gate+up weights [num_experts, hidden_size, 2*intermediate].
    experts_fc1: Tensor,
    /// Sparse expert down weights [num_experts, intermediate, hidden_size].
    experts_fc2: Tensor,
    num_experts: usize,
    top_k: usize,
    intermediate_size: usize,
}

impl AriaTextMoELayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        num_shared_experts: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Router: try vLLM-remapped name first, fall back to HF original.
        let router_weight = vb
            .get((num_experts, hidden_size), "router_weight")
            .or_else(|_| vb.pp("router").get((num_experts, hidden_size), "weight"))?;

        let shared_experts = TpSwiGluMlp::new(
            hidden_size,
            intermediate_size * num_shared_experts,
            vb.pp("shared_experts"),
            &LocalProcessGroup::new(),
        )?;

        // HF checkpoint shape: [E, hidden, 2*intermediate] (transposed vs standard weight format).
        let vb_exp = vb.pp("experts");
        let experts_fc1 = vb_exp.get(
            (num_experts, hidden_size, 2 * intermediate_size),
            "fc1.weight",
        )?;
        let experts_fc2 =
            vb_exp.get((num_experts, intermediate_size, hidden_size), "fc2.weight")?;

        Ok(Self {
            router_weight,
            shared_experts,
            experts_fc1,
            experts_fc2,
            num_experts,
            top_k,
            intermediate_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let xs_flat = xs.reshape((b * s, h))?;

        let shared_out = self
            .shared_experts
            .forward(&xs_flat, &TpContext::single_gpu())?;
        let routed_out = self.forward_routed(&xs_flat)?;

        (shared_out + routed_out)?.reshape((b, s, h))
    }

    /// Naive top-K sparse MoE: select top_k experts per token, sum scaled outputs.
    ///
    /// Routes on CPU (softmax scores) while expert matmuls remain on the original device.
    fn forward_routed(&self, xs: &Tensor) -> Result<Tensor> {
        let (t, _h) = xs.dims2()?;
        let _device = xs.device();

        let router_logits = xs.matmul(&self.router_weight.t()?)?;
        let router_probs = softmax_last_dim(&router_logits)?;
        let probs_cpu = router_probs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(t);
        for (token_idx, token_probs) in probs_cpu.iter().enumerate() {
            let mut indexed: Vec<(usize, f32)> = token_probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let selected = &indexed[..self.top_k.min(self.num_experts)];

            let x = xs.narrow(0, token_idx, 1)?; // [1, H]
            let mut token_out: Option<Tensor> = None;
            for &(eidx, score) in selected {
                // [1, H] @ [H, 2I] = [1, 2I]  (HF weights: [E, H, 2I] → transposed linear)
                let fc1 = self
                    .experts_fc1
                    .narrow(0, eidx, 1)?
                    .squeeze(0)?
                    .contiguous()?;
                let h2 = x.matmul(&fc1)?;
                let gate = candle_nn::ops::silu(&h2.narrow(1, 0, self.intermediate_size)?)?;
                let up = h2.narrow(1, self.intermediate_size, self.intermediate_size)?;
                let hidden = (gate * up)?;

                // [1, I] @ [I, H] = [1, H]
                let fc2 = self
                    .experts_fc2
                    .narrow(0, eidx, 1)?
                    .squeeze(0)?
                    .contiguous()?;
                let expert_out = (hidden.matmul(&fc2)? * score as f64)?;
                token_out = Some(match token_out {
                    None => expert_out,
                    Some(prev) => (prev + expert_out)?,
                });
            }
            outputs.push(token_out.ok_or_else(|| {
                candle_core::Error::Msg(format!("no experts selected for token {token_idx}"))
            })?);
        }

        Tensor::cat(&outputs, 0)
    }
}

// ─── AriaDecoderLayer ─────────────────────────────────────────────────────────

/// Llama-style decoder layer with MoE replacing the standard MLP.
struct AriaDecoderLayer {
    self_attn: LlamaAttention,
    moe: AriaTextMoELayer,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl AriaDecoderLayer {
    fn new(attn_cfg: &ModelConfig, moe_cfg: &AriaTextCfg, vb: VarBuilder) -> Result<Self> {
        let self_attn =
            LlamaAttention::new_with_tp(attn_cfg, vb.pp("self_attn"), &LocalProcessGroup::new())?;
        let moe = AriaTextMoELayer::new(
            moe_cfg.hidden_size,
            moe_cfg.intermediate_size,
            moe_cfg.moe_num_experts,
            moe_cfg.moe_topk,
            moe_cfg.moe_num_shared_experts,
            vb.pp("mlp"),
        )?;
        let input_layernorm = rms_norm(
            attn_cfg.hidden_size,
            attn_cfg.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            attn_cfg.hidden_size,
            attn_cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            moe,
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
            .moe
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
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
            .moe
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── AriaTextModel ────────────────────────────────────────────────────────────

struct AriaTextModel {
    embed_tokens: TpEmbedding,
    layers: Vec<AriaDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl AriaTextModel {
    fn new(cfg: &ModelConfig, moe_cfg: &AriaTextCfg, vb: VarBuilder) -> Result<Self> {
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();
        let vb_m = vb.pp("model");

        let embed_tokens = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &pg,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AriaDecoderLayer::new(
                cfg,
                moe_cfg,
                vb_m.pp("layers").pp(i),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb = embed_tokens
                .embeddings()
                .expect("single GPU: embeddings accessible")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb, None))
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                &pg,
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

    fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids, &self.tp_ctx)
    }

    fn forward_with_embeddings(
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

    fn forward_decode_batch_with_embeddings(
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
}

// ─── merge_multimodal ─────────────────────────────────────────────────────────

/// Merge pre-encoded image embeddings into text embeddings at image token positions.
///
/// Each `(position, ProcessedImage)` in `mm_inputs.image_embeddings` maps a flat
/// offset (`batch_idx * seq_len + start_pos`) to a `[num_tokens, hidden]` feature
/// tensor that replaces text embeddings at those positions.
fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }

    let (_b, seq_len, _d) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;

    for (position, processed) in &mm_inputs.image_embeddings {
        let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
        let batch_idx = position / seq_len;
        let start_pos = position % seq_len;
        for (i, emb) in emb_vec.iter().enumerate() {
            let target = start_pos + i;
            if target < seq_len && batch_idx < merged.len() {
                merged[batch_idx][target] = emb.clone();
            }
        }
    }

    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main Model ───────────────────────────────────────────────────────────────

/// Aria vision-language model for conditional generation.
pub struct AriaForConditionalGeneration {
    vision_tower: VisionEncoder,
    projector: AriaProjector,
    language_model: AriaTextModel,
    device: Device,
}

impl AriaForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = AriaVisionCfg::from_model_config(cfg);
        let text_cfg = AriaTextCfg::from_model_config(cfg);

        // Vision tower: SigLIP ViT without post_layernorm.
        let vis_enc_cfg = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: vis_cfg.hidden_size,
            intermediate_size: vis_cfg.intermediate_size,
            num_attention_heads: vis_cfg.num_attention_heads,
            num_hidden_layers: vis_cfg.num_hidden_layers,
            image_size: vis_cfg.image_size,
            patch_size: vis_cfg.patch_size,
            num_channels: vis_cfg.num_channels,
            layer_norm_eps: vis_cfg.layer_norm_eps,
        };
        let vision_tower = VisionEncoder::new(&vis_enc_cfg, vb.pp("vision_tower"))?;

        // Projector.
        let patch_to_query_dict = parse_patch_to_query_dict(cfg);
        let max_query_num = cfg
            .extra
            .get("max_value_projector_patch_to_query_dict")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| patch_to_query_dict.values().copied().max().unwrap_or(256) as u64)
            as usize;
        let projector = AriaProjector::new(
            vis_cfg.hidden_size,
            vis_cfg.num_attention_heads,
            text_cfg.hidden_size,
            text_cfg.hidden_size,
            max_query_num,
            patch_to_query_dict,
            vb.pp("multi_modal_projector"),
        )?;

        // Language model (LlamaMoE).
        let text_model_cfg = text_cfg.to_model_config();
        let language_model =
            AriaTextModel::new(&text_model_cfg, &text_cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_tower,
            projector,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode images: pixel_values [B, C, H, W] → projected features [B, query_num, text_hidden].
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let features = self.vision_tower.forward_no_post_norm(pixel_values)?;
        self.projector.forward(&features)
    }
}

impl crate::engine::ModelForward for AriaForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_with_embeddings(
            &embeddings,
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
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_decode_batch_with_embeddings(
            &embeddings,
            sequences,
            kv_cache_mgr,
        )
    }

    fn device(&self) -> &Device {
        &self.device
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
        let text_embeddings = self.language_model.embed_text(input_ids)?;
        let embeddings = if let Some(mm) = multimodal_inputs {
            merge_multimodal(&text_embeddings, mm, &self.device)?
        } else {
            text_embeddings
        };
        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
    use crate::multimodal::{MultimodalInputs, ProcessedImage};

    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        // Vision config (tiny: 1-channel 8×8 images, 4×4 patches → 2×2=4 patches).
        extra.insert(
            "vision_config".to_string(),
            json!({
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "image_size": 8,
                "patch_size": 4,
                "num_channels": 1,
                "layer_norm_eps": 1e-6
            }),
        );

        // Projector: 4 patches → 2 queries.
        extra.insert("projector_patch_to_query_dict".to_string(), json!({"4": 2}));
        extra.insert(
            "max_value_projector_patch_to_query_dict".to_string(),
            json!(2),
        );
        extra.insert("image_token_index".to_string(), json!(9));

        // MoE config at top level.
        extra.insert("moe_num_experts".to_string(), json!(2));
        extra.insert("moe_topk".to_string(), json!(1));
        extra.insert("moe_num_shared_experts".to_string(), json!(1));

        ModelConfig {
            architectures: vec!["AriaForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 64,
            max_position_embeddings: 512,
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

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 4,
            num_blocks: 32,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_aria_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AriaForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_aria_vision_encode() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AriaForConditionalGeneration::new(&cfg, vb).unwrap();

        // 1ch 8×8 → (8/4)² = 4 patches → [1, 4, 16]
        let pixel_values = Tensor::zeros((1usize, 1, 8, 8), DType::F32, &device).unwrap();
        let features = model.vision_tower.forward_no_post_norm(&pixel_values);
        assert!(
            features.is_ok(),
            "vision forward failed: {:?}",
            features.err()
        );
        assert_eq!(features.unwrap().dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_aria_projector() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AriaForConditionalGeneration::new(&cfg, vb).unwrap();

        // 4 patches → 2 queries, output_dim = text_hidden = 32.
        let features = Tensor::zeros((1usize, 4, 16), DType::F32, &device).unwrap();
        let projected = model.projector.forward(&features);
        assert!(projected.is_ok(), "projector failed: {:?}", projected.err());
        assert_eq!(projected.unwrap().dims(), &[1, 2, 32]);
    }

    #[test]
    fn test_aria_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AriaForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_aria_with_image() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AriaForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode 1 image: 1ch 8×8 → 4 patches → 2 projected tokens.
        let pixel_values = Tensor::zeros((1usize, 1, 8, 8), DType::F32, &device).unwrap();
        let img_feats = model.encode_images(&pixel_values).expect("encode failed");
        // img_feats: [1, 2, 32] → squeeze to [2, 32] for ProcessedImage.
        let img_feats_2d = img_feats.squeeze(0).expect("squeeze failed");
        let processed = ProcessedImage::new(img_feats_2d, 2);

        // Sequence: 4 tokens, image at positions 1 and 2.
        let mm = MultimodalInputs::with_images(vec![0u32, 9, 9, 0], vec![(1, processed)]);

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::from_slice(&[0u32, 9, 9, 0], (1usize, seq_len), &device).unwrap();
        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "multimodal forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }
}
