//! DeepSeek V2/V3 model implementation.
//!
//! DeepSeek uses Multi-head Latent Attention (MLA) which compresses the KV cache
//! using low-rank projections, significantly reducing memory usage.
//!
//! Key features:
//! - MLA: KV cache compression via `kv_lora_rank`
//! - MoE: Mixture of Experts with shared experts
//! - YaRN RoPE scaling for long contexts
//!
//! NOTE: MLA requires special KV cache handling. The cache stores compressed
//! latent vectors (`kv_lora_rank` dimensions) instead of full K/V tensors.
//! This implementation uses standard cache for compatibility, with TODO for
//! optimized MLA cache integration.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

// ─── MLA Attention ──────────────────────────────────────────────────────────
//
// Multi-head Latent Attention compresses KV through a bottleneck:
// - hidden -> kv_a_proj -> kv_lora_rank + qk_rope_head_dim
// - kv_lora_rank -> kv_b_proj -> num_heads * (qk_nope_head_dim + v_head_dim)

struct MLAAttention {
    // Query projections (optional low-rank)
    q_a_proj: Option<Linear>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Option<Linear>,
    q_proj: Option<Linear>,
    // KV projections
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    // Output projection
    o_proj: Linear,
    // RoPE
    rotary_emb: RotaryEmbedding,
    // Config
    num_heads: usize,
    num_kv_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    head_dim: usize,
    #[allow(dead_code)] // TODO: Use scaling in attention computation
    scaling: f64,
}

impl MLAAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        // Extract MLA-specific config from extra
        let qk_nope_head_dim = cfg.extra.get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let qk_rope_head_dim = cfg.extra.get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let v_head_dim = cfg.extra.get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let q_lora_rank = cfg.extra.get("q_lora_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let kv_lora_rank = cfg.extra.get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;

        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        // Query projection (low-rank or direct)
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_rank) = q_lora_rank {
            let q_a = linear_no_bias(cfg.hidden_size, q_rank, vb.pp("q_a_proj"))?;
            let q_a_ln = rms_norm(q_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
            let q_b = linear_no_bias(q_rank, num_heads * qk_head_dim, vb.pp("q_b_proj"))?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = linear_no_bias(cfg.hidden_size, num_heads * qk_head_dim, vb.pp("q_proj"))?;
            (None, None, None, Some(q))
        };

        // KV projection
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

        // Output projection
        let o_proj = linear_no_bias(num_heads * v_head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        // RoPE for the rope dimension only
        let rotary_emb = RotaryEmbedding::new(
            qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        // Scaling with optional YaRN mscale
        let mscale = cfg.extra.get("rope_scaling")
            .and_then(|v| v.get("mscale"))
            .and_then(|v| v.as_f64())
            .map(|mscale| {
                let factor = cfg.extra.get("rope_scaling")
                    .and_then(|v| v.get("factor"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                if factor <= 1.0 { 1.0 } else { 0.1 * mscale * factor.ln() + 1.0 }
            })
            .unwrap_or(1.0);

        let scaling = (qk_head_dim as f64).powf(-0.5) * mscale * mscale;

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            head_dim: qk_head_dim,
            scaling,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache: &mut crate::kv_cache::CacheEngine,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Split Q into nope and rope parts
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection through latent space
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize and project KV
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;
        let kv = self.kv_b_proj.forward(&kv_a)?;
        let kv = kv.reshape((batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))?;

        let k_nope = kv.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let v = kv.narrow(D::Minus1, self.qk_nope_head_dim, self.v_head_dim)?;

        // Apply RoPE to the rope parts
        // RoPE expects [batch, heads, seq, head_dim] format
        let q_pe = q_pe.transpose(1, 2)?;
        let k_pe_raw = k_pe_raw.reshape((batch_size, 1, seq_len, self.qk_rope_head_dim))?;
        let (q_pe, k_pe) = self.rotary_emb.apply(&q_pe, &k_pe_raw, seqlen_offset)?;
        let q_pe = q_pe.transpose(1, 2)?;

        // Broadcast k_pe to all heads
        let k_pe = k_pe.broadcast_as((batch_size, seq_len, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Concatenate nope and rope parts
        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;
        let k = Tensor::cat(&[&k_nope, &k_pe], D::Minus1)?;

        // Pad V to match head_dim for cache compatibility
        // TODO: Optimize with MLA-specific cache that stores compressed format
        let v_padded = if self.v_head_dim < self.head_dim {
            Tensor::cat(&[
                &v,
                &Tensor::zeros(
                    (batch_size, seq_len, self.num_heads, self.head_dim - self.v_head_dim),
                    v.dtype(),
                    v.device(),
                )?,
            ], D::Minus1)?
        } else {
            v.clone()
        };

        // Reshape for cache: [batch*seq, num_heads, head_dim]
        let k_for_cache = k.reshape((batch_size * seq_len, self.num_heads, self.head_dim))?;
        let v_for_cache = v_padded.reshape((batch_size * seq_len, self.num_heads, self.head_dim))?;

        // Use paged attention
        let attn_output = paged_attention(
            &q.reshape((batch_size * seq_len, self.num_heads, self.head_dim))?,
            &k_for_cache,
            &v_for_cache,
            attention_mask,
            seqlen_offset,
            cache,
            block_ids,
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )?;

        // Extract only v_head_dim from output and project
        let attn_output = attn_output.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let attn_output = attn_output.narrow(D::Minus1, 0, self.v_head_dim)?;
        let attn_output = attn_output.reshape((batch_size, seq_len, self.num_heads * self.v_head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache: &mut crate::kv_cache::CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);

        for (seq_idx, seq) in sequences.iter().enumerate() {
            let x = xs.i(seq_idx)?.unsqueeze(0)?;
            let attn_out = self.forward(
                &x,
                None,
                seq.seqlen_offset,
                cache,
                &seq.block_ids,
                &seq.slot_mapping,
            )?;
            outputs.push(attn_out.squeeze(0)?);
        }

        Tensor::stack(&outputs, 0)
    }

    fn project_query(&self, x: &Tensor) -> Result<Tensor> {
        if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_latent = q_a.forward(x)?;
            let q_latent = q_a_ln.forward(&q_latent)?;
            q_b.forward(&q_latent)
        } else if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)
        } else {
            candle_core::bail!("No query projection configured")
        }
    }
}

// ─── DeepSeek MLP ───────────────────────────────────────────────────────────

struct DeepSeekMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DeepSeekMLP {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl Module for DeepSeekMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct DeepSeekDecoderLayer {
    self_attn: MLAAttention,
    mlp: Option<DeepSeekMLP>,
    moe: Option<MoELayer>,
    shared_experts: Option<DeepSeekMLP>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    routed_scaling_factor: f64,
}

impl DeepSeekDecoderLayer {
    fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = MLAAttention::new(cfg, vb.pp("self_attn"))?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        let routed_scaling_factor = cfg.extra.get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // Check if this layer uses MoE
        let n_routed = cfg.extra.get("n_routed_experts").and_then(|v| v.as_u64()).map(|v| v as usize);
        let is_moe = n_routed.is_some() && layer_idx > 0;

        let (mlp, moe, shared_experts) = if is_moe {
            let n_routed = n_routed.unwrap();
            let n_shared = cfg.extra.get("n_shared_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let top_k = cfg.extra.get("num_experts_per_tok").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
            let moe_intermediate = cfg.extra.get("moe_intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.intermediate_size as u64) as usize;

            let layer_cfg = MoELayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: moe_intermediate,
                num_experts: n_routed,
                top_k,
                renormalize: true,
            };
            let moe_layer = MoELayer::new(layer_cfg, vb.pp("mlp"))?;

            let shared = if n_shared > 0 {
                Some(DeepSeekMLP::new(cfg.hidden_size, moe_intermediate * n_shared, vb.pp("mlp.shared_experts"))?)
            } else {
                None
            };

            (None, Some(moe_layer), shared)
        } else {
            let mlp = DeepSeekMLP::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
            (Some(mlp), None, None)
        };

        Ok(Self {
            self_attn,
            mlp,
            moe,
            shared_experts,
            input_layernorm,
            post_attention_layernorm,
            routed_scaling_factor,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(
            &x,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table.block_ids(),
            slot_mapping,
        )?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;

        let x = if let Some(moe) = &self.moe {
            let routed = (moe.forward(&x)? * self.routed_scaling_factor)?;
            if let Some(shared) = &self.shared_experts {
                (routed + shared.forward(&x)?)?
            } else {
                routed
            }
        } else if let Some(mlp) = &self.mlp {
            mlp.forward(&x)?
        } else {
            x
        };

        residual + x
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        let xs = if let Some(moe) = &self.moe {
            let routed = (moe.forward(&xs)? * self.routed_scaling_factor)?;
            if let Some(shared) = &self.shared_experts {
                (routed + shared.forward(&xs)?)?
            } else {
                routed
            }
        } else if let Some(mlp) = &self.mlp {
            mlp.forward(&xs)?
        } else {
            xs
        };

        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// DeepSeek V2/V3 model for causal language modeling.
///
/// Uses Multi-head Latent Attention (MLA) for KV cache compression
/// and Mixture of Experts (MoE) for capacity scaling.
pub struct DeepSeekForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl DeepSeekForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DeepSeekDecoderLayer::new(cfg, i, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
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
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(seq_len, seqlen_offset, xs.dtype(), &self.device)?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset, kv_cache_mgr, layer_idx, block_table, slot_mapping)?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl crate::engine::ModelForward for DeepSeekForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(input_ids, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("qk_nope_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("qk_rope_head_dim".into(), serde_json::Value::Number(8.into()));
        extra.insert("v_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("kv_lora_rank".into(), serde_json::Value::Number(32.into()));

        ModelConfig {
            architectures: vec!["DeepSeekForCausalLM".to_string()],
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 24, // qk_nope + qk_rope
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

    #[test]
    fn test_deepseek_creation() {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekForCausalLM::new(&cfg, vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_deepseek_mlp() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mlp = DeepSeekMLP::new(64, 128, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (2, 4, 64), &device).unwrap();
        let out = mlp.forward(&x).unwrap();

        assert_eq!(out.dims(), &[2, 4, 64]);
    }

    #[test]
    fn test_deepseek_moe_layer() {
        let device = Device::Cpu;
        let mut cfg = test_config();

        cfg.extra.insert("n_routed_experts".into(), serde_json::Value::Number(4.into()));
        cfg.extra.insert("num_experts_per_tok".into(), serde_json::Value::Number(2.into()));
        cfg.extra.insert("moe_intermediate_size".into(), serde_json::Value::Number(128.into()));

        let vb = VarBuilder::zeros(DType::F32, &device);

        // Layer 0 should not use MoE
        let layer0 = DeepSeekDecoderLayer::new(&cfg, 0, vb.pp("layer0")).unwrap();
        assert!(layer0.moe.is_none());
        assert!(layer0.mlp.is_some());

        // Layer 1 should use MoE
        let layer1 = DeepSeekDecoderLayer::new(&cfg, 1, vb.pp("layer1")).unwrap();
        assert!(layer1.moe.is_some());
        assert!(layer1.mlp.is_none());
    }
}
