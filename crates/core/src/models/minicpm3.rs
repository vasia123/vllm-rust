//! MiniCPM3 model implementation.
//!
//! MiniCPM3 extends MiniCPM with Multi-head Latent Attention (MLA) — the same
//! query/key factorization as DeepSeek V2, but using standard paged KV cache
//! (not compressed MLA cache). Key differences from MiniCPM v1/v2:
//!
//! - Low-rank query path: `q_a_proj -> q_a_layernorm -> q_b_proj`
//! - Latent KV: `kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj`
//! - RoPE applied only to the `qk_rope_head_dim` portion of Q and K
//! - V is zero-padded to `qk_head_dim` for attention, then sliced back
//! - MiniCPM-style scaling: `scale_emb`, `scale_depth`, `dim_model_base`
//!
//! Unlike DeepSeek V2/V3, MiniCPM3 caches the full expanded K/V tensors in
//! standard paged KV cache, so it uses `KVCacheManager::new()` (not `new_mla()`).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

// ─── MiniCPM3 Config ────────────────────────────────────────────────────────

struct MiniCPM3Config {
    scale_emb: f64,
    scale_depth: f64,
    dim_model_base: f64,
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_act: String,
    #[allow(dead_code)]
    hidden_act_param: f64,
    // MLA-specific
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
}

impl MiniCPM3Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let scale_emb = cfg
            .extra
            .get("scale_emb")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let scale_depth = cfg
            .extra
            .get("scale_depth")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let dim_model_base = cfg
            .extra
            .get("dim_model_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.hidden_size as f64);
        let num_experts = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let hidden_act = cfg.hidden_act.clone();
        let hidden_act_param = cfg
            .extra
            .get("hidden_act_param")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let q_lora_rank = cfg
            .extra
            .get("q_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize;
        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        Self {
            scale_emb,
            scale_depth,
            dim_model_base,
            num_experts,
            num_experts_per_tok,
            hidden_act,
            hidden_act_param,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
        }
    }

    fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn residual_scale(&self, num_layers: usize) -> f64 {
        self.scale_depth / (num_layers as f64).sqrt()
    }

    fn output_scale(&self, hidden_size: usize) -> f64 {
        hidden_size as f64 / self.dim_model_base
    }
}

// ─── MLP ────────────────────────────────────────────────────────────────────

enum MiniCPM3Activation {
    Silu,
    FatRelu(f64),
}

struct MiniCPM3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: MiniCPM3Activation,
}

impl MiniCPM3Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: MiniCPM3Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;

        let activated = match &self.activation {
            MiniCPM3Activation::Silu => (candle_nn::ops::silu(&gate)? * up)?,
            MiniCPM3Activation::FatRelu(threshold) => {
                let threshold_t = Tensor::full(*threshold as f32, gate.shape(), gate.device())?
                    .to_dtype(gate.dtype())?;
                let mask = gate.ge(&threshold_t)?;
                let dtype = gate.dtype();
                let gated = (gate * mask.to_dtype(dtype)?)?;
                (gated * up)?
            }
        };

        self.down_proj.forward(&activated)
    }
}

// ─── Feed Forward (Dense or MoE) ───────────────────────────────────────────

enum MiniCPM3FeedForward {
    Dense(MiniCPM3Mlp),
    Moe(MoELayer),
}

impl MiniCPM3FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

// ─── MLA-style Attention (standard KV cache) ───────────────────────────────

struct MiniCPM3Attention {
    // Low-rank query path
    q_a_proj: Linear,
    q_a_layernorm: RmsNorm,
    q_b_proj: Linear,
    // Latent KV
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    // Output
    o_proj: Linear,
    // RoPE (only for the rope portion)
    rotary_emb: RotaryEmbedding,
    // Dimensions
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
}

impl MiniCPM3Attention {
    fn new(cfg: &ModelConfig, mini_cfg: &MiniCPM3Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let qk_nope_head_dim = mini_cfg.qk_nope_head_dim;
        let qk_rope_head_dim = mini_cfg.qk_rope_head_dim;
        let qk_head_dim = mini_cfg.qk_head_dim();
        let v_head_dim = mini_cfg.v_head_dim;
        let q_lora_rank = mini_cfg.q_lora_rank;
        let kv_lora_rank = mini_cfg.kv_lora_rank;

        // Low-rank query
        let q_a_proj = linear_no_bias(cfg.hidden_size, q_lora_rank, vb.pp("q_a_proj"))?;
        let q_a_layernorm = rms_norm(q_lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
        let q_b_proj = linear_no_bias(q_lora_rank, num_heads * qk_head_dim, vb.pp("q_b_proj"))?;

        // Latent KV
        let kv_a_proj_with_mqa = linear_no_bias(
            cfg.hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm = rms_norm(kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = linear_no_bias(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            vb.pp("kv_b_proj"),
        )?;

        // Output
        let o_proj = linear_no_bias(num_heads * v_head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        // RoPE operates on the rope portion only
        let rotary_emb = RotaryEmbedding::new(
            qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
        })
    }

    /// Build Q, K, V from hidden states, applying MLA factorization and RoPE.
    ///
    /// Returns (q, k, v) shaped for paged_attention:
    /// - q: [b, num_heads, seq, qk_head_dim]
    /// - k: [b, num_heads, seq, qk_head_dim]
    /// - v: [b, num_heads, seq, qk_head_dim] (zero-padded from v_head_dim)
    fn project_qkv(&self, xs: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Query: low-rank path
        let q = self.q_a_proj.forward(xs)?;
        let q = self.q_a_layernorm.forward(&q)?;
        let q = self.q_b_proj.forward(&q)?;
        // [b, seq, num_heads, qk_head_dim]
        let q = q.reshape((b_sz, q_len, self.num_heads, self.qk_head_dim))?;

        // Split q into nope and pe parts
        let q_nope = q.narrow(3, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(3, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV: latent projection
        let latent_cache = self.kv_a_proj_with_mqa.forward(xs)?;
        // Split: [b, seq, kv_lora_rank] and [b, seq, qk_rope_head_dim]
        let kv_a = latent_cache.narrow(2, 0, self.kv_lora_rank)?;
        let k_pe_raw = latent_cache.narrow(2, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize latent
        let kv_a = self.kv_a_layernorm.forward(&kv_a.contiguous()?)?;

        // Expand latent to K_nope and V
        let kv = self.kv_b_proj.forward(&kv_a)?;
        // [b, seq, num_heads, qk_nope_head_dim + v_head_dim]
        let kv = kv.reshape((
            b_sz,
            q_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ))?;
        let k_nope = kv.narrow(3, 0, self.qk_nope_head_dim)?;
        let v = kv.narrow(3, self.qk_nope_head_dim, self.v_head_dim)?;

        // Apply RoPE to the rope portions of Q and K
        // RoPE expects [b, heads, seq, head_dim] layout
        let q_pe_rope = q_pe.transpose(1, 2)?; // [b, num_heads, seq, rope_dim]
                                               // k_pe is shared across heads (MQA-style): [b, 1, seq, rope_dim]
        let k_pe_rope = k_pe_raw.reshape((b_sz, 1, q_len, self.qk_rope_head_dim))?;

        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_rope, &k_pe_rope, seqlen_offset)?;

        // Back to [b, seq, heads, dim]
        let q_pe = q_pe_rotated.transpose(1, 2)?;
        // Broadcast k_pe to all heads: [b, seq, 1, rope_dim] -> [b, seq, num_heads, rope_dim]
        let k_pe = k_pe_rotated
            .transpose(1, 2)?
            .broadcast_as((b_sz, q_len, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Assemble full Q: [q_nope | q_pe]
        let q_full = Tensor::cat(&[&q_nope, &q_pe], 3)?;

        // Assemble full K: [k_nope | k_pe]
        let k_full = Tensor::cat(&[&k_nope, &k_pe], 3)?;

        // Pad V from v_head_dim to qk_head_dim with zeros
        let v_padded = if self.v_head_dim < self.qk_head_dim {
            let pad_size = self.qk_head_dim - self.v_head_dim;
            let zeros = Tensor::zeros(
                (b_sz, q_len, self.num_heads, pad_size),
                v.dtype(),
                v.device(),
            )?;
            Tensor::cat(&[&v, &zeros], 3)?
        } else {
            v.contiguous()?
        };

        // Transpose to [b, heads, seq, dim] for paged_attention
        let q_out = q_full.transpose(1, 2)?;
        let k_out = k_full.transpose(1, 2)?;
        let v_out = v_padded.transpose(1, 2)?;

        Ok((q_out, k_out, v_out))
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (q, k, v) = self.project_qkv(xs, seqlen_offset)?;

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
            self.num_heads, // num_kv_heads = num_heads for MLA-style
            self.qk_head_dim,
        )?;

        // Slice off the V padding: attn_output is [b, seq, num_heads * qk_head_dim]
        // Reshape to [b, seq, num_heads, qk_head_dim], narrow to v_head_dim, flatten
        let (b_sz, seq_len, _) = attn_output.dims3()?;
        let attn_output = attn_output
            .reshape((b_sz, seq_len, self.num_heads, self.qk_head_dim))?
            .narrow(3, 0, self.v_head_dim)?
            .reshape((b_sz, seq_len, self.num_heads * self.v_head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let x_i = xs.narrow(0, i, 1)?;
            let (q, k, v) = self.project_qkv(&x_i, seq.seqlen_offset)?;

            let attn_out = paged_attention(
                &q,
                &k,
                &v,
                None,
                seq.seqlen_offset,
                cache_engine,
                &seq.block_ids,
                &seq.slot_mapping,
                self.num_heads,
                self.num_heads,
                self.qk_head_dim,
            )?;

            // Slice off V padding
            let attn_out = attn_out
                .reshape((1, 1, self.num_heads, self.qk_head_dim))?
                .narrow(3, 0, self.v_head_dim)?
                .reshape((1, 1, self.num_heads * self.v_head_dim))?;

            let out = self.o_proj.forward(&attn_out)?;
            outputs.push(out);
        }

        Tensor::cat(&outputs, 0)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct MiniCPM3DecoderLayer {
    self_attn: MiniCPM3Attention,
    feed_forward: MiniCPM3FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_scale: f64,
}

impl MiniCPM3DecoderLayer {
    fn new(cfg: &ModelConfig, mini_cfg: &MiniCPM3Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = MiniCPM3Attention::new(cfg, mini_cfg, vb.pp("self_attn"))?;

        let feed_forward = if mini_cfg.num_experts > 0 {
            let moe_cfg = MoELayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size,
                num_experts: mini_cfg.num_experts,
                top_k: mini_cfg.num_experts_per_tok,
                renormalize: true,
                inplace: true,
                is_act_and_mul: true,
            };
            MiniCPM3FeedForward::Moe(MoELayer::new(moe_cfg, vb.pp("mlp"))?)
        } else {
            let activation = match mini_cfg.hidden_act.as_str() {
                "fatrelu" => MiniCPM3Activation::FatRelu(mini_cfg.hidden_act_param),
                _ => MiniCPM3Activation::Silu,
            };
            MiniCPM3FeedForward::Dense(MiniCPM3Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                activation,
                vb.pp("mlp"),
            )?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let residual_scale = mini_cfg.residual_scale(cfg.num_hidden_layers);

        Ok(Self {
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
            residual_scale,
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
        let xs = (residual + (hidden * self.residual_scale)?)?;

        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let hidden = self.feed_forward.forward(&hidden)?;
        residual + (hidden * self.residual_scale)?
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
        let xs = (residual + (hidden * self.residual_scale)?)?;

        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let hidden = self.feed_forward.forward(&hidden)?;
        residual + (hidden * self.residual_scale)?
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

pub struct MiniCPM3ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<MiniCPM3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    scale_emb: f64,
    output_scale: f64,
    device: Device,
    dtype: DType,
}

impl MiniCPM3ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mini_cfg = MiniCPM3Config::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiniCPM3DecoderLayer::new(cfg, &mini_cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let output_scale = mini_cfg.output_scale(cfg.hidden_size);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            scale_emb: mini_cfg.scale_emb,
            output_scale,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.embed_tokens.forward(input_ids)?;
        hidden * self.scale_emb
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for MiniCPM3ForCausalLM {
    fn forward(
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

        let mut xs = self.embed(input_ids)?;
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
        xs = self.norm.forward(&xs)?;
        xs = (xs / self.output_scale)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        xs = self.norm.forward(&xs)?;
        xs = (xs / self.output_scale)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("scale_emb".into(), serde_json::Value::from(12.0));
        extra.insert("scale_depth".into(), serde_json::Value::from(1.4));
        extra.insert("dim_model_base".into(), serde_json::Value::from(256.0));
        // MLA dimensions
        extra.insert("qk_nope_head_dim".into(), serde_json::Value::from(8));
        extra.insert("qk_rope_head_dim".into(), serde_json::Value::from(4));
        extra.insert("v_head_dim".into(), serde_json::Value::from(8));
        extra.insert("q_lora_rank".into(), serde_json::Value::from(32));
        extra.insert("kv_lora_rank".into(), serde_json::Value::from(16));

        ModelConfig {
            architectures: vec!["MiniCPM3ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            // head_dim for KV cache = qk_nope_head_dim + qk_rope_head_dim = 12
            head_dim: 12,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_minicpm3_config_parsing() {
        let cfg = test_config();
        let mini_cfg = MiniCPM3Config::from_model_config(&cfg);

        assert!((mini_cfg.scale_emb - 12.0).abs() < 1e-6);
        assert!((mini_cfg.scale_depth - 1.4).abs() < 1e-6);
        assert!((mini_cfg.dim_model_base - 256.0).abs() < 1e-6);
        assert_eq!(mini_cfg.qk_nope_head_dim, 8);
        assert_eq!(mini_cfg.qk_rope_head_dim, 4);
        assert_eq!(mini_cfg.v_head_dim, 8);
        assert_eq!(mini_cfg.q_lora_rank, 32);
        assert_eq!(mini_cfg.kv_lora_rank, 16);
        assert_eq!(mini_cfg.qk_head_dim(), 12);
    }

    #[test]
    fn test_minicpm3_scaling_factors() {
        let cfg = test_config();
        let mini_cfg = MiniCPM3Config::from_model_config(&cfg);

        // residual_scale = scale_depth / sqrt(num_layers) = 1.4 / sqrt(2)
        let rs = mini_cfg.residual_scale(2);
        assert!((rs - 1.4 / (2.0f64).sqrt()).abs() < 1e-6);

        // output_scale = hidden_size / dim_model_base = 64 / 256 = 0.25
        let os = mini_cfg.output_scale(64);
        assert!((os - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_minicpm3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MiniCPM3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniCPM3ForCausalLM should construct: {:?}",
            model.err()
        );
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert!((model.scale_emb - 12.0).abs() < 1e-6);
        assert!((model.output_scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_minicpm3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPM3ForCausalLM::new(&cfg, vb).expect("build model");

        // MiniCPM3 uses standard KV cache with qk_head_dim as head_dim and
        // num_kv_heads = num_heads (since MLA expands to all heads)
        let qk_head_dim = 12; // qk_nope(8) + qk_rope(4)
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_attention_heads, // MLA: kv_heads = q_heads
            head_dim: qk_head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");

        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward pass");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_minicpm3_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPM3ForCausalLM::new(&cfg, vb).expect("build model");

        let qk_head_dim = 12;
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_attention_heads,
            head_dim: qk_head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 4 tokens
        let seq_len = 4;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let prefill_out = crate::engine::ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(
            prefill_out.is_ok(),
            "Prefill should succeed: {:?}",
            prefill_out.err()
        );

        // Decode 1 token
        let decode_input = Tensor::zeros((1, 1), DType::U32, &device).expect("input");
        let sequences = vec![DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: seq_len,
            block_ids: block_table.block_ids().to_vec(),
            slot_mapping: vec![seq_len],
        }];

        let decode_out = model.forward_decode_batch(&decode_input, &sequences, &mut kv_cache_mgr);
        assert!(
            decode_out.is_ok(),
            "Decode should succeed: {:?}",
            decode_out.err()
        );
        let decode_out = decode_out.unwrap();
        assert_eq!(decode_out.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_minicpm3_implements_model_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MiniCPM3ForCausalLM::new(&cfg, vb).expect("build model");

        let _: &dyn crate::engine::ModelForward = &model;
    }

    #[test]
    fn test_minicpm3_with_moe_config() {
        let mut cfg = test_config();
        cfg.extra
            .insert("num_experts".into(), serde_json::Value::from(4));
        cfg.extra
            .insert("num_experts_per_tok".into(), serde_json::Value::from(2));

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MiniCPM3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MiniCPM3ForCausalLM with MoE should construct: {:?}",
            model.err()
        );
    }
}
