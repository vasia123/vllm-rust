//! Qwen3Next model architecture.
//!
//! Qwen3Next is a next-generation Qwen model featuring:
//! - Hybrid attention: mix of full softmax attention and linear (gated delta net) attention layers
//! - Sparse MoE with shared experts and gated expert routing
//! - Per-head QK normalization (Gemma-style with +1 offset)
//! - Optional attention output gating (sigmoid gate on Q)
//! - Optional per-layer scaling (learnable attn_layer_scale and ffn_layer_scale)
//!
//! The linear attention layers use Gated Delta Net (GDN), a recurrent mechanism
//! with conv1d + gated delta rule. Since GDN requires specialized CUDA kernels
//! (causal_conv1d, fused_recurrent_gated_delta_rule), this implementation uses
//! standard full attention for all layers. GDN layer support would be a future
//! extension requiring the linear attention kernel infrastructure.
//!
//! Architecture:
//! ```text
//! Embedding -> [Qwen3NextDecoderLayer x N] -> RMSNorm -> LM Head
//!
//! Qwen3NextDecoderLayer:
//!   InputLayerNorm -> (Full or Linear) Attention -> [AttnLayerScale] -> PostAttnNorm -> Residual
//!   MLP or SparseMoE -> [FFNLayerScale] -> Residual
//! ```
//!
//! Reference: `reference/vllm/vllm/model_executor/models/qwen3_next.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Gemma-style RMSNorm (offset by +1, for Q/K norms) ─────────────────────

pub(crate) struct Qwen3NextRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3NextRmsNorm {
    pub(crate) fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for Qwen3NextRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs_normed = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let scale = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        xs_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

// ─── Qwen3Next Config Extraction ────────────────────────────────────────────

struct Qwen3NextExtraConfig {
    layer_types: Vec<String>,
    moe_num_experts: usize,
    moe_top_k: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    decoder_sparse_step: usize,
    norm_topk_prob: bool,
    attn_output_gate: bool,
    layer_scale: bool,
}

impl Qwen3NextExtraConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let layer_types = cfg
            .extra
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("full_attention").to_string())
                    .collect()
            })
            .unwrap_or_else(|| vec!["full_attention".to_string(); cfg.num_hidden_layers]);

        let moe_num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let moe_top_k = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let moe_intermediate_size = cfg
            .extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        let shared_expert_intermediate_size = cfg
            .extra
            .get("shared_expert_intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(0);

        let decoder_sparse_step = cfg
            .extra
            .get("decoder_sparse_step")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let norm_topk_prob = cfg
            .extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let attn_output_gate = cfg
            .extra
            .get("attn_output_gate")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let layer_scale = cfg
            .extra
            .get("layer_scale")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            layer_types,
            moe_num_experts,
            moe_top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            decoder_sparse_step,
            norm_topk_prob,
            attn_output_gate,
            layer_scale,
        }
    }

    fn layer_type(&self, layer_idx: usize) -> &str {
        if layer_idx < self.layer_types.len() {
            &self.layer_types[layer_idx]
        } else {
            "full_attention"
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_num_experts > 0 && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────
//
// Full attention with:
// - Gemma-style QK RMSNorm (weight offset by +1)
// - Optional output gating: Q is split into [q, gate], gate = sigmoid(gate) * attn_output
// - Standard RoPE

struct Qwen3NextAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    q_norm: Qwen3NextRmsNorm,
    k_norm: Qwen3NextRmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_gate: bool,
    // When attn_output_gate=true, q_proj outputs 2x heads: half for Q, half for gate
    gate_proj: Option<TpLinear>,
}

impl Qwen3NextAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Qwen3NextExtraConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        if world_size > 1 {
            if !num_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if num_kv_heads > 1 && !num_kv_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = std::cmp::max(1, num_kv_heads / world_size);

        // When attn_output_gate is true, Q projection is doubled to produce both Q and gate
        let (q_proj, gate_proj) = if extra_cfg.attn_output_gate {
            let q_proj = TpLinear::column_parallel(
                cfg.hidden_size,
                num_heads * head_dim,
                false,
                false,
                vb.pp("q_proj"),
                pg,
            )?;
            let gate = TpLinear::column_parallel(
                cfg.hidden_size,
                num_heads * head_dim,
                false,
                false,
                vb.pp("q_gate_proj"),
                pg,
            )?;
            (q_proj, Some(gate))
        } else {
            let q_proj = TpLinear::column_parallel(
                cfg.hidden_size,
                num_heads * head_dim,
                false,
                false,
                vb.pp("q_proj"),
                pg,
            )?;
            (q_proj, None)
        };

        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let q_norm = Qwen3NextRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3NextRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            attn_output_gate: extra_cfg.attn_output_gate,
            gate_proj,
        })
    }

    /// Apply Gemma-style per-head RMSNorm (with +1 offset)
    fn apply_qk_norm(x: &Tensor, norm: &Qwen3NextRmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let flat = x.reshape((b * h * s, d))?;
        let normed = norm.forward(&flat)?;
        normed.reshape((b, h, s, d))
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
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        // Compute gate if needed
        let gate = if self.attn_output_gate {
            if let Some(ref gate_proj) = self.gate_proj {
                Some(gate_proj.forward(xs, tp_ctx)?)
            } else {
                None
            }
        } else {
            None
        };

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = Self::apply_qk_norm(&q, &self.q_norm)?;
        let k = Self::apply_qk_norm(&k, &self.k_norm)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        let mut attn_output = paged_attention(
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
        )?;

        // Apply output gate
        if let Some(gate) = gate {
            let gate = candle_nn::ops::sigmoid(&gate)?;
            attn_output = attn_output.broadcast_mul(&gate)?;
        }

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

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let gate = if self.attn_output_gate {
            if let Some(ref gate_proj) = self.gate_proj {
                Some(gate_proj.forward(xs, tp_ctx)?)
            } else {
                None
            }
        } else {
            None
        };

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = Self::apply_qk_norm(&q, &self.q_norm)?;
        let k = Self::apply_qk_norm(&k, &self.k_norm)?;

        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

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

        let mut attn_output = Tensor::cat(&outputs, 0)?;

        if let Some(gate) = gate {
            let gate = candle_nn::ops::sigmoid(&gate)?;
            attn_output = attn_output.broadcast_mul(&gate)?;
        }

        self.o_proj.forward(&attn_output, tp_ctx)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

enum Qwen3NextMlp {
    Dense(TpSwiGluMlp),
    Moe {
        moe: Box<MoELayer>,
        shared_expert: Option<TpSwiGluMlp>,
        shared_expert_gate: Option<candle_nn::Linear>,
    },
}

#[allow(dead_code)]
pub(crate) struct Qwen3NextDecoderLayer {
    self_attn: Option<Qwen3NextAttention>,
    mlp: Qwen3NextMlp,
    input_layernorm: Qwen3NextRmsNorm,
    post_attention_layernorm: Qwen3NextRmsNorm,
    attn_layer_scale: Option<Tensor>,
    ffn_layer_scale: Option<Tensor>,
    layer_type: String,
}

impl Qwen3NextDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        extra_cfg: &Qwen3NextExtraConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let layer_type = extra_cfg.layer_type(layer_idx).to_string();

        // NOTE: Linear attention (GDN) layers require specialized CUDA kernels.
        // All layers use full attention for now. The layer_type is preserved for
        // future GDN kernel integration.
        let self_attn = if layer_type == "full_attention" || layer_type == "linear_attention" {
            Some(Qwen3NextAttention::new_with_tp(
                cfg,
                extra_cfg,
                vb.pp("self_attn"),
                pg,
            )?)
        } else {
            None
        };

        let mlp = if extra_cfg.is_moe_layer(layer_idx) {
            let moe_cfg = MoELayerConfig {
                num_experts: extra_cfg.moe_num_experts,
                top_k: extra_cfg.moe_top_k,
                hidden_size: cfg.hidden_size,
                intermediate_size: extra_cfg.moe_intermediate_size,
                renormalize: extra_cfg.norm_topk_prob,
                inplace: false,
                is_act_and_mul: true,
            };
            let moe = Box::new(MoELayer::new(moe_cfg, vb.pp("mlp").pp("experts"))?);

            let shared_expert = if extra_cfg.shared_expert_intermediate_size > 0 {
                Some(TpSwiGluMlp::new(
                    cfg.hidden_size,
                    extra_cfg.shared_expert_intermediate_size,
                    vb.pp("mlp").pp("shared_expert"),
                    pg,
                )?)
            } else {
                None
            };

            let shared_expert_gate = if extra_cfg.shared_expert_intermediate_size > 0 {
                Some(candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    1,
                    vb.pp("mlp").pp("shared_expert_gate"),
                )?)
            } else {
                None
            };

            Qwen3NextMlp::Moe {
                moe,
                shared_expert,
                shared_expert_gate,
            }
        } else {
            Qwen3NextMlp::Dense(TpSwiGluMlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
                pg,
            )?)
        };

        let input_layernorm =
            Qwen3NextRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen3NextRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let attn_layer_scale = if extra_cfg.layer_scale {
            Some(vb.get((1, 1, cfg.hidden_size), "attn_layer_scale")?)
        } else {
            None
        };

        let ffn_layer_scale = if extra_cfg.layer_scale {
            Some(vb.get((1, 1, cfg.hidden_size), "ffn_layer_scale")?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attn_layer_scale,
            ffn_layer_scale,
            layer_type,
        })
    }

    pub(crate) fn new_for_mtp(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let extra_cfg = Qwen3NextExtraConfig::from_model_config(cfg);
        let pg = LocalProcessGroup::new();
        Self::new_with_tp(cfg, &extra_cfg, layer_idx, vb, &pg)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
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
        let hidden_states = self.input_layernorm.forward(xs)?;

        let mut hidden_states = if let Some(ref attn) = self.self_attn {
            attn.forward(
                &hidden_states,
                attention_mask,
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
                tp_ctx,
            )?
        } else {
            hidden_states
        };

        // Apply attention layer scale: hidden_states * (scale + 1)
        if let Some(ref scale) = self.attn_layer_scale {
            let scale_plus_one = (scale.to_dtype(hidden_states.dtype())? + 1.0)?;
            hidden_states = hidden_states.broadcast_mul(&scale_plus_one)?;
        }

        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let mut hidden_states = match &self.mlp {
            Qwen3NextMlp::Dense(mlp) => mlp.forward(&hidden_states, tp_ctx)?,
            Qwen3NextMlp::Moe {
                moe,
                shared_expert,
                shared_expert_gate,
            } => {
                let routed = moe.forward(&hidden_states)?;
                if let Some(shared) = shared_expert {
                    let shared_out = shared.forward(&hidden_states, tp_ctx)?;
                    // Apply shared expert gate if present
                    if let Some(gate) = shared_expert_gate {
                        let gate_out = candle_nn::ops::sigmoid(&gate.forward(&hidden_states)?)?;
                        let gated_shared = shared_out.broadcast_mul(&gate_out)?;
                        (routed + gated_shared)?
                    } else {
                        (routed + shared_out)?
                    }
                } else {
                    routed
                }
            }
        };

        // Apply FFN layer scale
        if let Some(ref scale) = self.ffn_layer_scale {
            let scale_plus_one = (scale.to_dtype(hidden_states.dtype())? + 1.0)?;
            hidden_states = hidden_states.broadcast_mul(&scale_plus_one)?;
        }

        residual + hidden_states
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
        let hidden_states = self.input_layernorm.forward(xs)?;

        let mut hidden_states = if let Some(ref attn) = self.self_attn {
            attn.forward_decode_batch(
                &hidden_states,
                sequences,
                kv_cache_mgr.engine_mut(layer_idx),
                tp_ctx,
            )?
        } else {
            hidden_states
        };

        if let Some(ref scale) = self.attn_layer_scale {
            let scale_plus_one = (scale.to_dtype(hidden_states.dtype())? + 1.0)?;
            hidden_states = hidden_states.broadcast_mul(&scale_plus_one)?;
        }

        let hidden_states = (hidden_states + residual)?;
        let residual = &hidden_states;

        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        let mut hidden_states = match &self.mlp {
            Qwen3NextMlp::Dense(mlp) => mlp.forward(&hidden_states, tp_ctx)?,
            Qwen3NextMlp::Moe {
                moe,
                shared_expert,
                shared_expert_gate,
            } => {
                let routed = moe.forward(&hidden_states)?;
                if let Some(shared) = shared_expert {
                    let shared_out = shared.forward(&hidden_states, tp_ctx)?;
                    if let Some(gate) = shared_expert_gate {
                        let gate_out = candle_nn::ops::sigmoid(&gate.forward(&hidden_states)?)?;
                        let gated_shared = shared_out.broadcast_mul(&gate_out)?;
                        (routed + gated_shared)?
                    } else {
                        (routed + shared_out)?
                    }
                } else {
                    routed
                }
            }
        };

        if let Some(ref scale) = self.ffn_layer_scale {
            let scale_plus_one = (scale.to_dtype(hidden_states.dtype())? + 1.0)?;
            hidden_states = hidden_states.broadcast_mul(&scale_plus_one)?;
        }

        residual + hidden_states
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct Qwen3NextForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Qwen3NextDecoderLayer>,
    norm: Qwen3NextRmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Qwen3NextForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let extra_cfg = Qwen3NextExtraConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3NextDecoderLayer::new_with_tp(
                cfg,
                &extra_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = Qwen3NextRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Qwen3NextForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Qwen3NextForCausalLM::forward(
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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // All layers are full_attention for testing (no GDN kernels)
        extra.insert(
            "layer_types".to_string(),
            serde_json::json!(["full_attention", "full_attention"]),
        );
        extra.insert("attn_output_gate".to_string(), serde_json::json!(false));
        extra.insert("layer_scale".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Qwen3NextForCausalLM".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_config_with_gate() -> ModelConfig {
        let mut cfg = test_config();
        cfg.extra
            .insert("attn_output_gate".to_string(), serde_json::json!(true));
        cfg
    }

    fn test_config_with_layer_scale() -> ModelConfig {
        let mut cfg = test_config();
        cfg.extra
            .insert("layer_scale".to_string(), serde_json::json!(true));
        cfg
    }

    fn test_config_moe() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "layer_types".to_string(),
            serde_json::json!(["full_attention", "full_attention"]),
        );
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("decoder_sparse_step".to_string(), serde_json::json!(1));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));
        extra.insert("attn_output_gate".to_string(), serde_json::json!(false));
        extra.insert("layer_scale".to_string(), serde_json::json!(false));

        ModelConfig {
            architectures: vec!["Qwen3NextForCausalLM".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
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
    fn test_qwen3_next_extra_config_parsing() {
        let cfg = test_config();
        let extra = Qwen3NextExtraConfig::from_model_config(&cfg);

        assert_eq!(extra.layer_types.len(), 2);
        assert_eq!(extra.layer_type(0), "full_attention");
        assert!(!extra.attn_output_gate);
        assert!(!extra.layer_scale);
        assert!(!extra.is_moe_layer(0));
    }

    #[test]
    fn test_qwen3_next_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen3NextForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen3NextForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen3_next_construction_with_gate() {
        let cfg = test_config_with_gate();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen3NextForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen3NextForCausalLM with gate should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen3_next_construction_with_layer_scale() {
        let cfg = test_config_with_layer_scale();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen3NextForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen3NextForCausalLM with layer_scale should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen3_next_construction_moe() {
        let cfg = test_config_moe();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Qwen3NextForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Qwen3NextForCausalLM (MoE) should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen3_next_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3NextForCausalLM::new(&cfg, vb).expect("build model");

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
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_qwen3_next_single_token() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3NextForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_qwen3_next_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3NextForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_qwen3_next_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3NextForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_qwen3_next_rms_norm_offset() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let norm = Qwen3NextRmsNorm::new(8, 1e-6, vb.pp("norm")).expect("norm");

        // With zero weights, scale = (0 + 1) = 1, so output = normalized input
        let x = Tensor::ones((1, 8), DType::F32, &device).expect("input");
        let out = norm.forward(&x).expect("forward");
        assert_eq!(out.dims(), &[1, 8]);
    }

    #[test]
    fn test_qwen3_next_moe_config() {
        let cfg = test_config_moe();
        let extra = Qwen3NextExtraConfig::from_model_config(&cfg);

        assert_eq!(extra.moe_num_experts, 4);
        assert_eq!(extra.moe_top_k, 2);
        assert!(extra.is_moe_layer(0));
        assert!(extra.is_moe_layer(1));
    }

    #[test]
    fn test_qwen3_next_moe_forward() {
        let cfg = test_config_moe();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3NextForCausalLM::new(&cfg, vb).expect("build moe model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 2), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 2)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 2);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");
        assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
    }
}
