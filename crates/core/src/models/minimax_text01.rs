//! MiniMaxText01 (hybrid linear attention + standard attention) model.
//!
//! MiniMaxText01 is a hybrid architecture where some layers use linear attention
//! (SSM-like recurrence) and others use standard multi-head attention. Some layers
//! also use Mixture-of-Experts (MoE) with optional shared experts.
//!
//! Architecture aliases:
//! - MiniMaxForCausalLM -> MiniMaxText01ForCausalLM
//! - MiniMaxM1ForCausalLM -> MiniMaxText01ForCausalLM
//! - MiniMaxText01ForCausalLM
//!
//! Architecture:
//! ```text
//! Embedding -> [DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! DecoderLayer (linear attention variant, attention_type=0):
//!   RMSNorm -> LinearAttention -> alpha/beta scale -> RMSNorm -> MLP/MoE -> scale
//!
//! DecoderLayer (full attention variant, attention_type=1):
//!   RMSNorm -> SelfAttention(RoPE) -> alpha/beta scale -> RMSNorm -> MLP/MoE -> scale
//! ```
//!
//! Config keys from extra:
//! - `attn_type_list` or `decoder_attention_types` or `layer_types`: per-layer attention type
//! - `num_local_experts`: number of MoE experts (can be int or list)
//! - `num_experts_per_tok`: top-k experts
//! - `shared_intermediate_size`: shared expert intermediate size (0 = no shared expert)
//! - `shared_moe_mode`: "softmax" or "sigmoid" for shared expert mixing
//! - `layernorm_linear_attention_alpha`, `layernorm_linear_attention_beta`: scaling for linear attn layers
//! - `layernorm_full_attention_alpha`, `layernorm_full_attention_beta`: scaling for full attn layers
//! - `layernorm_mlp_alpha`, `layernorm_mlp_beta`: scaling for MLP output
//! - `postnorm`: whether to use post-norm (default false)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Parsed per-layer config from the HuggingFace MiniMax config.
struct MiniMaxText01Config {
    /// Per-layer attention type: 0 = linear, 1 = full attention
    decoder_attention_types: Vec<u8>,
    /// Per-layer number of experts (1 = dense MLP)
    num_local_experts: Vec<usize>,
    /// Top-k experts per token
    num_experts_per_tok: usize,
    /// Per-layer shared expert intermediate size (0 = no shared expert)
    shared_intermediate_sizes: Vec<usize>,
    /// Shared MoE mode: "softmax" or "sigmoid"
    shared_moe_mode: String,
    /// Scaling factors for attention residual connections
    linear_attn_alpha: f64,
    linear_attn_beta: f64,
    full_attn_alpha: f64,
    full_attn_beta: f64,
    /// Scaling factors for MLP residual connections
    mlp_alpha: f64,
    mlp_beta: f64,
    /// Whether to use post-norm (default false)
    postnorm: bool,
}

impl MiniMaxText01Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_layers = cfg.num_hidden_layers;

        // Parse attention type list
        let decoder_attention_types = Self::parse_attention_types(cfg, num_layers);

        // Parse expert counts per layer
        let num_local_experts = Self::parse_expert_counts(cfg, num_layers);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        // Parse shared intermediate sizes
        let shared_intermediate_sizes = Self::parse_shared_intermediate(cfg, num_layers);

        let shared_moe_mode = cfg
            .extra
            .get("shared_moe_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
            .to_string();

        let linear_attn_alpha = cfg
            .extra
            .get("layernorm_linear_attention_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("linear_attn_alpha_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let linear_attn_beta = cfg
            .extra
            .get("layernorm_linear_attention_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("linear_attn_beta_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let full_attn_alpha = cfg
            .extra
            .get("layernorm_full_attention_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("full_attn_alpha_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let full_attn_beta = cfg
            .extra
            .get("layernorm_full_attention_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("full_attn_beta_factor")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        let mlp_alpha = cfg
            .extra
            .get("layernorm_mlp_alpha")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.extra.get("mlp_alpha_factor").and_then(|v| v.as_f64()))
            .unwrap_or(1.0);

        let mlp_beta = cfg
            .extra
            .get("layernorm_mlp_beta")
            .and_then(|v| v.as_f64())
            .or_else(|| cfg.extra.get("mlp_beta_factor").and_then(|v| v.as_f64()))
            .unwrap_or(1.0);

        let postnorm = cfg
            .extra
            .get("postnorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            decoder_attention_types,
            num_local_experts,
            num_experts_per_tok,
            shared_intermediate_sizes,
            shared_moe_mode,
            linear_attn_alpha,
            linear_attn_beta,
            full_attn_alpha,
            full_attn_beta,
            mlp_alpha,
            mlp_beta,
            postnorm,
        }
    }

    fn parse_attention_types(cfg: &ModelConfig, num_layers: usize) -> Vec<u8> {
        // Try attn_type_list first
        if let Some(list) = cfg.extra.get("attn_type_list").and_then(|v| v.as_array()) {
            return list.iter().map(|v| v.as_u64().unwrap_or(1) as u8).collect();
        }

        // Try decoder_attention_types
        if let Some(list) = cfg
            .extra
            .get("decoder_attention_types")
            .and_then(|v| v.as_array())
        {
            return list.iter().map(|v| v.as_u64().unwrap_or(1) as u8).collect();
        }

        // Try layer_types (HF format)
        if let Some(list) = cfg.extra.get("layer_types").and_then(|v| v.as_array()) {
            return list
                .iter()
                .map(|v| match v.as_str() {
                    Some("linear_attention") => 0,
                    Some("full_attention") => 1,
                    _ => 1,
                })
                .collect();
        }

        // Default: all full attention
        vec![1; num_layers]
    }

    fn parse_expert_counts(cfg: &ModelConfig, num_layers: usize) -> Vec<usize> {
        if let Some(list) = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_array())
        {
            return list
                .iter()
                .map(|v| v.as_u64().unwrap_or(1) as usize)
                .collect();
        }

        if let Some(n) = cfg.extra.get("num_local_experts").and_then(|v| v.as_u64()) {
            return vec![n as usize; num_layers];
        }

        vec![1; num_layers]
    }

    fn parse_shared_intermediate(cfg: &ModelConfig, num_layers: usize) -> Vec<usize> {
        if let Some(list) = cfg
            .extra
            .get("shared_intermediate_size")
            .and_then(|v| v.as_array())
        {
            let mut sizes: Vec<usize> = list
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            sizes.resize(num_layers, 0);
            return sizes;
        }

        if let Some(n) = cfg
            .extra
            .get("shared_intermediate_size")
            .and_then(|v| v.as_u64())
        {
            return vec![n as usize; num_layers];
        }

        vec![0; num_layers]
    }

    fn attn_alpha(&self, layer_idx: usize) -> f64 {
        if self.decoder_attention_types[layer_idx] == 0 {
            self.linear_attn_alpha
        } else {
            self.full_attn_alpha
        }
    }

    fn attn_beta(&self, layer_idx: usize) -> f64 {
        if self.decoder_attention_types[layer_idx] == 0 {
            self.linear_attn_beta
        } else {
            self.full_attn_beta
        }
    }

    fn is_linear_attention(&self, layer_idx: usize) -> bool {
        self.decoder_attention_types
            .get(layer_idx)
            .copied()
            .unwrap_or(1)
            == 0
    }

    fn expert_count(&self, layer_idx: usize) -> usize {
        self.num_local_experts.get(layer_idx).copied().unwrap_or(1)
    }

    fn shared_intermediate(&self, layer_idx: usize) -> usize {
        self.shared_intermediate_sizes
            .get(layer_idx)
            .copied()
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn num_attention_layers(&self) -> usize {
        self.decoder_attention_types
            .iter()
            .filter(|&&t| t == 1)
            .count()
    }
}

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct MiniMaxText01Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
}

impl MiniMaxText01Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // Merged gate+up: hidden_size -> 2*intermediate_size
        let gate_up_proj =
            linear_no_bias(hidden_size, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let hidden = gate.mul(&chunks[1])?;
        self.down_proj.forward(&hidden)
    }
}

// ─── MoE Layer ──────────────────────────────────────────────────────────────

struct MiniMaxText01MoE {
    gate: Linear,
    experts: Vec<MiniMaxText01MoEExpert>,
    num_experts: usize,
    top_k: usize,
}

struct MiniMaxText01MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MiniMaxText01MoEExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w1"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("w2"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("w3"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

impl MiniMaxText01MoE {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MiniMaxText01MoEExpert::new(
                hidden_size,
                intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router logits in FP32
        let router_logits = self
            .gate
            .forward(&xs_2d.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let routing_data: Vec<f32> = routing_weights
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let flat_data: Vec<f32> = xs_2d.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let mut output_data = vec![0.0f32; num_tokens * hidden_dim];

        for token_idx in 0..num_tokens {
            let weights =
                &routing_data[token_idx * self.num_experts..(token_idx + 1) * self.num_experts];

            let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Renormalize top-k weights
            let top_sum: f32 = indexed[..self.top_k].iter().map(|(_, w)| w).sum();

            let token_input = Tensor::from_vec(
                flat_data[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].to_vec(),
                (1, hidden_dim),
                xs.device(),
            )?;

            for &(expert_idx, weight) in indexed[..self.top_k].iter() {
                let norm_weight = if top_sum > 0.0 {
                    weight / top_sum
                } else {
                    1.0 / self.top_k as f32
                };
                let expert_out = self.experts[expert_idx].forward(&token_input)?;
                let expert_data: Vec<f32> =
                    expert_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                for j in 0..hidden_dim {
                    output_data[token_idx * hidden_dim + j] += norm_weight * expert_data[j];
                }
            }
        }

        Tensor::from_vec(
            output_data,
            candle_core::Shape::from_dims(&orig_shape),
            xs.device(),
        )?
        .to_dtype(xs.dtype())
    }
}

// ─── Simplified Linear Attention ─────────────────────────────────────────────
//
// The reference MiniMaxText01LinearAttention uses a recurrent SSM-like formulation.
// For inference without the full CUDA kernel, we implement a simplified version
// that performs the same Q*K*V computation pattern as a linear (non-softmax)
// attention. This captures the essential semantics while being feasible on CPU.

struct MiniMaxText01LinearAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl MiniMaxText01LinearAttention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv = q_size + 2 * kv_size;

        let qkv_proj = linear_no_bias(hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // GQA expansion: repeat K/V heads to match Q heads
        let repeat_factor = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeat_factor > 1 {
            let k = k
                .unsqueeze(2)?
                .expand((b_sz, self.num_kv_heads, repeat_factor, q_len, self.head_dim))?
                .reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let v = v
                .unsqueeze(2)?
                .expand((b_sz, self.num_kv_heads, repeat_factor, q_len, self.head_dim))?
                .reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            (k, v)
        } else {
            (k, v)
        };

        // Linear attention: simplified Q @ K^T @ V / sqrt(d) (no softmax on Q*K^T)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}

// ─── Standard Attention ─────────────────────────────────────────────────────

struct MiniMaxText01Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl MiniMaxText01Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let total_qkv = q_size + 2 * kv_size;

        let qkv_proj = linear_no_bias(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
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

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

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

        let attn_output = paged_attention(
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

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.qkv_proj.forward(xs)?;
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

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

        Tensor::cat(&outputs, 0)
    }
}

// ─── Attention Variant ──────────────────────────────────────────────────────

enum AttentionVariant {
    Linear(MiniMaxText01LinearAttention),
    Full(MiniMaxText01Attention),
}

// ─── Feed-forward Variant ───────────────────────────────────────────────────

enum FfnVariant {
    Dense(MiniMaxText01Mlp),
    MoE(MiniMaxText01MoE),
}

struct SharedExpert {
    mlp: MiniMaxText01Mlp,
    coefficient: Linear,
    mode: String,
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct MiniMaxText01DecoderLayer {
    self_attn: AttentionVariant,
    ffn: FfnVariant,
    shared_expert: Option<SharedExpert>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn_alpha: f64,
    attn_beta: f64,
    mlp_alpha: f64,
    mlp_beta: f64,
    postnorm: bool,
    /// Which KV cache layer index (only for full attention layers)
    cache_layer_idx: Option<usize>,
}

impl MiniMaxText01DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        minimax_cfg: &MiniMaxText01Config,
        layer_idx: usize,
        cache_layer_idx: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_linear = minimax_cfg.is_linear_attention(layer_idx);

        let self_attn = if is_linear {
            AttentionVariant::Linear(MiniMaxText01LinearAttention::new(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                vb.pp("self_attn"),
            )?)
        } else {
            AttentionVariant::Full(MiniMaxText01Attention::new(cfg, vb.pp("self_attn"))?)
        };

        let expert_count = minimax_cfg.expert_count(layer_idx);
        let ffn = if expert_count > 1 {
            FfnVariant::MoE(MiniMaxText01MoE::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                expert_count,
                minimax_cfg.num_experts_per_tok,
                vb.pp("block_sparse_moe"),
            )?)
        } else {
            FfnVariant::Dense(MiniMaxText01Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?)
        };

        let shared_size = minimax_cfg.shared_intermediate(layer_idx);
        let shared_expert = if shared_size > 0 {
            let shared_mlp =
                MiniMaxText01Mlp::new(cfg.hidden_size, shared_size, vb.pp("shared_mlp"))?;
            let coefficient = linear_no_bias(cfg.hidden_size, 1, vb.pp("coefficient"))?;
            Some(SharedExpert {
                mlp: shared_mlp,
                coefficient,
                mode: minimax_cfg.shared_moe_mode.clone(),
            })
        } else {
            None
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
            ffn,
            shared_expert,
            input_layernorm,
            post_attention_layernorm,
            attn_alpha: minimax_cfg.attn_alpha(layer_idx),
            attn_beta: minimax_cfg.attn_beta(layer_idx),
            mlp_alpha: minimax_cfg.mlp_alpha,
            mlp_beta: minimax_cfg.mlp_beta,
            postnorm: minimax_cfg.postnorm,
            cache_layer_idx,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(hidden_states)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            hidden_states.clone()
        };

        let attn_output = match &self.self_attn {
            AttentionVariant::Linear(lin_attn) => lin_attn.forward(&layernorm_output)?,
            AttentionVariant::Full(attn) => {
                let cache_idx = self
                    .cache_layer_idx
                    .expect("full attention layer should have cache index");
                attn.forward(
                    &layernorm_output,
                    attention_mask,
                    seqlen_offset,
                    kv_cache_mgr.engine_mut(cache_idx),
                    block_table,
                    slot_mapping,
                )?
            }
        };

        // Scale residual and attention output
        let residual = residual.affine(self.attn_alpha, 0.0)?;
        let attn_output = attn_output.affine(self.attn_beta, 0.0)?;

        let layernorm_input = (residual + attn_output)?;
        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            layernorm_input
        };

        let ffn_output = match &self.ffn {
            FfnVariant::Dense(mlp) => mlp.forward(&layernorm_output)?,
            FfnVariant::MoE(moe) => {
                let moe_output = moe.forward(&layernorm_output)?;
                self.apply_shared_expert(&layernorm_output, &moe_output)?
            }
        };

        let residual = residual.affine(self.mlp_alpha, 0.0)?;
        let ffn_output = ffn_output.affine(self.mlp_beta, 0.0)?;

        residual + ffn_output
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let layernorm_output = self.input_layernorm.forward(xs)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            xs.clone()
        };

        let attn_output = match &self.self_attn {
            AttentionVariant::Linear(lin_attn) => lin_attn.forward(&layernorm_output)?,
            AttentionVariant::Full(attn) => {
                let cache_idx = self
                    .cache_layer_idx
                    .expect("full attention layer should have cache index");
                attn.forward_decode_batch(
                    &layernorm_output,
                    sequences,
                    kv_cache_mgr.engine_mut(cache_idx),
                )?
            }
        };

        let residual = residual.affine(self.attn_alpha, 0.0)?;
        let attn_output = attn_output.affine(self.attn_beta, 0.0)?;

        let layernorm_input = (residual + attn_output)?;
        let layernorm_output = self.post_attention_layernorm.forward(&layernorm_input)?;
        let residual = if self.postnorm {
            layernorm_output.clone()
        } else {
            layernorm_input
        };

        let ffn_output = match &self.ffn {
            FfnVariant::Dense(mlp) => mlp.forward(&layernorm_output)?,
            FfnVariant::MoE(moe) => {
                let moe_output = moe.forward(&layernorm_output)?;
                self.apply_shared_expert(&layernorm_output, &moe_output)?
            }
        };

        let residual = residual.affine(self.mlp_alpha, 0.0)?;
        let ffn_output = ffn_output.affine(self.mlp_beta, 0.0)?;

        residual + ffn_output
    }

    /// Apply shared expert mixing if configured.
    fn apply_shared_expert(&self, input: &Tensor, moe_output: &Tensor) -> Result<Tensor> {
        match &self.shared_expert {
            Some(shared) => {
                let moe_fp32 = moe_output.to_dtype(DType::F32)?;
                let shared_output = shared.mlp.forward(input)?.to_dtype(DType::F32)?;
                let coef = shared.coefficient.forward(&input.to_dtype(DType::F32)?)?;

                let mixed = match shared.mode.as_str() {
                    "sigmoid" => {
                        let coef = candle_nn::ops::sigmoid(&coef)?;
                        let one_minus = coef.affine(-1.0, 1.0)?;
                        (moe_fp32.broadcast_mul(&one_minus)?
                            + shared_output.broadcast_mul(&coef)?)?
                    }
                    _ => {
                        // softmax (default)
                        let coef = candle_nn::ops::softmax_last_dim(&coef)?;
                        let one_minus = coef.affine(-1.0, 1.0)?;
                        (moe_fp32.broadcast_mul(&one_minus)?
                            + shared_output.broadcast_mul(&coef)?)?
                    }
                };

                mixed.to_dtype(input.dtype())
            }
            None => Ok(moe_output.clone()),
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniMaxText01ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<MiniMaxText01DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
    /// Number of attention layers (for KV cache sizing)
    num_attn_layers: usize,
}

impl MiniMaxText01ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let minimax_cfg = MiniMaxText01Config::from_model_config(cfg);
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        let mut attn_layer_count = 0;

        for i in 0..cfg.num_hidden_layers {
            let cache_layer_idx = if !minimax_cfg.is_linear_attention(i) {
                let idx = attn_layer_count;
                attn_layer_count += 1;
                Some(idx)
            } else {
                None
            };

            layers.push(MiniMaxText01DecoderLayer::new(
                cfg,
                &minimax_cfg,
                i,
                cache_layer_idx,
                vb_layers.pp(i),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_attn_layers: attn_layer_count,
        })
    }

    /// Get the number of full attention layers (for KV cache sizing).
    pub fn num_attention_layers(&self) -> usize {
        self.num_attn_layers
    }
}

impl crate::engine::ModelForward for MiniMaxText01ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

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

        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in &self.layers {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr)?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_minimax_text01_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // 4 layers: layers 0,2 are linear attention, layers 1,3 are full attention
        extra.insert(
            "attn_type_list".to_string(),
            serde_json::json!([0, 1, 0, 1]),
        );
        // All layers have 1 expert (dense MLP)
        extra.insert("num_local_experts".to_string(), serde_json::json!(1));

        ModelConfig {
            architectures: vec!["MiniMaxText01ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn test_minimax_text01_moe_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // 2 layers: layer 0 linear, layer 1 full
        extra.insert("attn_type_list".to_string(), serde_json::json!([0, 1]));
        // 4 experts per layer
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        // Shared expert on both layers
        extra.insert(
            "shared_intermediate_size".to_string(),
            serde_json::json!(64),
        );
        extra.insert("shared_moe_mode".to_string(), serde_json::json!("softmax"));

        ModelConfig {
            architectures: vec!["MiniMaxText01ForCausalLM".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn create_cache(cfg: &ModelConfig, num_attn_layers: usize) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: num_attn_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_parsing_attn_type_list() {
        let cfg = test_minimax_text01_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![0, 1, 0, 1]);
        assert!(minimax_cfg.is_linear_attention(0));
        assert!(!minimax_cfg.is_linear_attention(1));
        assert!(minimax_cfg.is_linear_attention(2));
        assert!(!minimax_cfg.is_linear_attention(3));
        assert_eq!(minimax_cfg.num_attention_layers(), 2);
    }

    #[test]
    fn test_config_parsing_layer_types_format() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.remove("attn_type_list");
        cfg.extra.insert(
            "layer_types".to_string(),
            serde_json::json!([
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention"
            ]),
        );
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_config_default_all_full_attention() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.remove("attn_type_list");
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.decoder_attention_types, vec![1, 1, 1, 1]);
        assert_eq!(minimax_cfg.num_attention_layers(), 4);
    }

    #[test]
    fn test_config_expert_counts_list() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.insert(
            "num_local_experts".to_string(),
            serde_json::json!([1, 4, 1, 8]),
        );
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.expert_count(0), 1);
        assert_eq!(minimax_cfg.expert_count(1), 4);
        assert_eq!(minimax_cfg.expert_count(2), 1);
        assert_eq!(minimax_cfg.expert_count(3), 8);
    }

    #[test]
    fn test_config_shared_intermediate() {
        let cfg = test_minimax_text01_moe_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.shared_intermediate(0), 64);
        assert_eq!(minimax_cfg.shared_intermediate(1), 64);
    }

    #[test]
    fn test_config_scaling_defaults() {
        let cfg = test_minimax_text01_config();
        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.linear_attn_alpha, 1.0);
        assert_eq!(minimax_cfg.linear_attn_beta, 1.0);
        assert_eq!(minimax_cfg.full_attn_alpha, 1.0);
        assert_eq!(minimax_cfg.full_attn_beta, 1.0);
        assert_eq!(minimax_cfg.mlp_alpha, 1.0);
        assert_eq!(minimax_cfg.mlp_beta, 1.0);
        assert!(!minimax_cfg.postnorm);
    }

    #[test]
    fn test_config_scaling_custom() {
        let mut cfg = test_minimax_text01_config();
        cfg.extra.insert(
            "layernorm_linear_attention_alpha".to_string(),
            serde_json::json!(0.5),
        );
        cfg.extra.insert(
            "layernorm_full_attention_beta".to_string(),
            serde_json::json!(2.0),
        );
        cfg.extra
            .insert("postnorm".to_string(), serde_json::json!(true));

        let minimax_cfg = MiniMaxText01Config::from_model_config(&cfg);

        assert_eq!(minimax_cfg.linear_attn_alpha, 0.5);
        assert_eq!(minimax_cfg.full_attn_beta, 2.0);
        assert!(minimax_cfg.postnorm);
    }

    // ─── MLP Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_mlp_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mlp = MiniMaxText01Mlp::new(64, 128, vb.pp("mlp")).expect("mlp");
        let input = Tensor::zeros((2, 3, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, 64]);
    }

    // ─── Linear Attention Tests ─────────────────────────────────────────────────

    #[test]
    fn test_linear_attention_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = MiniMaxText01LinearAttention::new(64, 4, 2, 16, vb.pp("attn")).expect("attn");
        let input = Tensor::zeros((1, 3, 64), DType::F32, &device).expect("input");
        let output = attn.forward(&input);
        assert!(output.is_ok(), "linear attn forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[1, 3, 64]);
    }

    // ─── MoE Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_moe_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let moe = MiniMaxText01MoE::new(64, 128, 4, 2, vb.pp("moe")).expect("moe");
        let input = Tensor::zeros((2, 3, 64), DType::F32, &device).expect("input");
        let output = moe.forward(&input);
        assert!(output.is_ok(), "MoE forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, 64]);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_construction_hybrid() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxText01ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxText01ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.expect("model");
        assert_eq!(model.layers.len(), 4);
        // 2 full attention layers (indices 1 and 3)
        assert_eq!(model.num_attn_layers, 2);
    }

    #[test]
    fn test_construction_moe() {
        let cfg = test_minimax_text01_moe_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = MiniMaxText01ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniMaxText01ForCausalLM MoE should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_layer_type_classification() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        // Layers 0,2 are linear; layers 1,3 are full attention
        assert!(matches!(
            model.layers[0].self_attn,
            AttentionVariant::Linear(_)
        ));
        assert!(matches!(
            model.layers[1].self_attn,
            AttentionVariant::Full(_)
        ));
        assert!(matches!(
            model.layers[2].self_attn,
            AttentionVariant::Linear(_)
        ));
        assert!(matches!(
            model.layers[3].self_attn,
            AttentionVariant::Full(_)
        ));

        assert!(model.layers[0].cache_layer_idx.is_none());
        assert_eq!(model.layers[1].cache_layer_idx, Some(0));
        assert!(model.layers[2].cache_layer_idx.is_none());
        assert_eq!(model.layers[3].cache_layer_idx, Some(1));
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_forward_shape() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
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
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");

        let logits = model
            .forward(&next, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_model_forward_trait() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_device() {
        let cfg = test_minimax_text01_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("model");

        assert!(matches!(ModelForward::device(&model), Device::Cpu));
    }

    // ─── MoE Forward Tests ──────────────────────────────────────────────────────

    #[test]
    fn test_moe_model_forward() {
        let cfg = test_minimax_text01_moe_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiniMaxText01ForCausalLM::new(&cfg, vb).expect("build moe model");

        // Only 1 full attention layer (layer 1)
        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg, model.num_attn_layers);

        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("moe forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
