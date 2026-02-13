//! DeepSeek V2/V3 model implementation with MLA cache.
//!
//! DeepSeek uses Multi-head Latent Attention (MLA) which compresses the KV cache
//! using low-rank projections, achieving 42x memory reduction.
//!
//! # Memory Comparison (DeepSeek V3, 128K context)
//! - Standard cache: ~6 GB
//! - MLA cache: ~144 MB
//! - MLA cache + FP8: ~72 MB
//!
//! Key features:
//! - MLA: KV cache compression via `kv_lora_rank`
//! - MoE: Mixture of Experts with shared experts
//! - YaRN RoPE scaling for long contexts
//!
//! # Usage
//!
//! ```ignore
//! use vllm_core::models::{DeepSeekForCausalLM, create_cache_manager};
//!
//! let model = DeepSeekForCausalLM::new(&cfg, vb)?;
//! let mut kv_cache_mgr = create_cache_manager(&cfg, block_size, num_blocks, &device)?;
//!
//! // Forward pass uses MLA cache automatically
//! let logits = model.forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)?;
//! ```

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::attention::MLAAttention;
use crate::layers::{causal_mask, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

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

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
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

/// DeepSeek decoder layer using MLA attention with compressed KV cache.
pub(crate) struct DeepSeekDecoderLayer {
    self_attn: MLAAttention,
    mlp: Option<DeepSeekMLP>,
    moe: Option<MoELayer>,
    shared_experts: Option<DeepSeekMLP>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    routed_scaling_factor: f64,
}

impl DeepSeekDecoderLayer {
    pub(crate) fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let vb_attn = vb.pp("self_attn");
        let num_heads = cfg.num_attention_heads;

        // Extract MLA-specific config
        let qk_nope_head_dim = cfg
            .extra
            .get("qk_nope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let qk_rope_head_dim = cfg
            .extra
            .get("qk_rope_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let v_head_dim = cfg
            .extra
            .get("v_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as usize;
        let q_lora_rank = cfg
            .extra
            .get("q_lora_rank")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let kv_lora_rank = cfg
            .extra
            .get("kv_lora_rank")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;

        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        // Query projection (low-rank or direct)
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) = if let Some(q_rank) = q_lora_rank {
            let q_a = linear_no_bias(cfg.hidden_size, q_rank, vb_attn.pp("q_a_proj"))?;
            let q_a_ln = rms_norm(q_rank, cfg.rms_norm_eps, vb_attn.pp("q_a_layernorm"))?;
            let q_b = linear_no_bias(q_rank, num_heads * qk_head_dim, vb_attn.pp("q_b_proj"))?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = linear_no_bias(
                cfg.hidden_size,
                num_heads * qk_head_dim,
                vb_attn.pp("q_proj"),
            )?;
            (None, None, None, Some(q))
        };

        // KV projection
        let kv_a_proj_with_mqa = linear_no_bias(
            cfg.hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            vb_attn.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm =
            rms_norm(kv_lora_rank, cfg.rms_norm_eps, vb_attn.pp("kv_a_layernorm"))?;
        let kv_b_proj = linear_no_bias(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            vb_attn.pp("kv_b_proj"),
        )?;

        // Output projection
        let o_proj = linear_no_bias(
            num_heads * v_head_dim,
            cfg.hidden_size,
            vb_attn.pp("o_proj"),
        )?;

        // RoPE
        let rotary_emb = RotaryEmbedding::new(
            qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        // Compute YaRN mscale
        let mscale = cfg
            .extra
            .get("rope_scaling")
            .and_then(|v| v.get("mscale"))
            .and_then(|v| v.as_f64())
            .map(|mscale| {
                let factor = cfg
                    .extra
                    .get("rope_scaling")
                    .and_then(|v| v.get("factor"))
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                if factor <= 1.0 {
                    1.0
                } else {
                    0.1 * mscale * factor.ln() + 1.0
                }
            })
            .unwrap_or(1.0);

        let q_scale = mscale * mscale;

        let self_attn = MLAAttention::new(
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
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            q_scale,
        );

        // Layer norms
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let routed_scaling_factor = cfg
            .extra
            .get("routed_scaling_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        // MoE setup
        let n_routed = cfg
            .extra
            .get("n_routed_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let is_moe = n_routed.is_some() && layer_idx > 0;

        let (mlp, moe, shared_experts) = if is_moe {
            let n_routed = n_routed.unwrap();
            let n_shared = cfg
                .extra
                .get("n_shared_experts")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let top_k = cfg
                .extra
                .get("num_experts_per_tok")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let moe_intermediate =
                cfg.extra
                    .get("moe_intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(cfg.intermediate_size as u64) as usize;

            let layer_cfg = MoELayerConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: moe_intermediate,
                num_experts: n_routed,
                top_k,
                renormalize: true,
                inplace: false,
                is_act_and_mul: true,
            };
            let moe_layer = MoELayer::new(layer_cfg, vb.pp("mlp"))?;

            let shared = if n_shared > 0 {
                Some(DeepSeekMLP::new(
                    cfg.hidden_size,
                    moe_intermediate * n_shared,
                    vb.pp("mlp.shared_experts"),
                )?)
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
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_prefill(
            &x,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
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

    pub(crate) fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_decode(
            &x,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
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
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// DeepSeek V2/V3 model for causal language modeling.
///
/// Uses Multi-head Latent Attention (MLA) with compressed KV cache,
/// achieving 42x memory reduction compared to standard attention.
///
/// # Requirements
///
/// This model requires a KVCacheManager created with `new_mla()`:
///
/// ```ignore
/// let kv_cache_mgr = KVCacheManager::new_mla(&mla_config)?;
/// ```
///
/// Using a standard KVCacheManager will panic at runtime.
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
    /// Create a new DeepSeek model.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration (must have kv_lora_rank in extra)
    /// * `vb` - VarBuilder for loading weights
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

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

    /// Embed text tokens (for VLM use — embed only, no layers).
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Forward with pre-computed embeddings (for VLM use — skips embedding layer).
    pub fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let seq_len = embeddings.dim(1)?;
        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                embeddings.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        let mut xs = embeddings.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table.block_ids(),
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    /// Forward decode batch with pre-computed embeddings (for VLM use).
    pub fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);

        for (seq_idx, seq) in sequences.iter().enumerate() {
            let mut xs = embeddings.i(seq_idx)?.unsqueeze(0)?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                xs = layer.forward_decode(
                    &xs,
                    seq.seqlen_offset,
                    kv_cache_mgr,
                    layer_idx,
                    &seq.block_ids,
                    &seq.slot_mapping,
                )?;
            }

            let xs = self.norm.forward(&xs)?;
            let logits = self.lm_head.forward(&xs)?;
            outputs.push(logits.squeeze(0)?);
        }

        Tensor::stack(&outputs, 0)
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
        let (_, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                xs.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table.block_ids(),
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
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);

        for (seq_idx, seq) in sequences.iter().enumerate() {
            let x = input_ids.i(seq_idx)?.unsqueeze(0)?;
            let mut xs = self.embed_tokens.forward(&x)?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                xs = layer.forward_decode(
                    &xs,
                    seq.seqlen_offset,
                    kv_cache_mgr,
                    layer_idx,
                    &seq.block_ids,
                    &seq.slot_mapping,
                )?;
            }

            let xs = self.norm.forward(&xs)?;
            let logits = self.lm_head.forward(&xs)?;
            outputs.push(logits.squeeze(0)?);
        }

        Tensor::stack(&outputs, 0)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// GLM-5 (GlmMoeDsaForCausalLM) uses the same architecture as DeepSeek V2/V3.
/// See reference vLLM PR #34124 — the Python implementation is an empty subclass.
pub type GlmMoeDsaForCausalLM = DeepSeekForCausalLM;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "qk_nope_head_dim".into(),
            serde_json::Value::Number(16.into()),
        );
        extra.insert(
            "qk_rope_head_dim".into(),
            serde_json::Value::Number(8.into()),
        );
        extra.insert("v_head_dim".into(), serde_json::Value::Number(16.into()));
        extra.insert("kv_lora_rank".into(), serde_json::Value::Number(32.into()));

        ModelConfig {
            architectures: vec!["DeepseekV2ForCausalLM".to_string()],
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

    fn create_mla_cache_manager(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let mla_config = MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            cfg.num_attention_heads,
            4,  // block_size
            16, // num_blocks
            cfg.num_hidden_layers,
            DType::F32,
            device.clone(),
        );
        KVCacheManager::new_mla(&mla_config).unwrap()
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
    fn test_glm_moe_dsa_is_deepseek_alias() {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        // GlmMoeDsaForCausalLM is a type alias for DeepSeekForCausalLM
        let model = GlmMoeDsaForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GLM-5 alias should construct: {:?}",
            model.err()
        );
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

        cfg.extra.insert(
            "n_routed_experts".into(),
            serde_json::Value::Number(4.into()),
        );
        cfg.extra.insert(
            "num_experts_per_tok".into(),
            serde_json::Value::Number(2.into()),
        );
        cfg.extra.insert(
            "moe_intermediate_size".into(),
            serde_json::Value::Number(128.into()),
        );

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

    #[test]
    fn test_deepseek_implements_model_forward() {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekForCausalLM::new(&cfg, vb).unwrap();

        // Verify it implements ModelForward
        let _: &dyn crate::engine::ModelForward = &model;
    }

    #[test]
    fn test_deepseek_forward_with_mla_cache() {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekForCausalLM::new(&cfg, vb).unwrap();
        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);

        // Verify manager is MLA type
        assert!(kv_cache_mgr.is_mla());

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(4);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..4).collect();

        let output = model.forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_deepseek_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekForCausalLM::new(&cfg, vb).unwrap();
        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);

        // Prefill with 3 tokens
        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(4);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..3).collect();

        let prefill_out = model.forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(prefill_out.is_ok());

        // Decode 1 token using forward_decode_batch
        let decode_input = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let sequences = vec![DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: 3,
            block_ids: block_table.block_ids().to_vec(),
            slot_mapping: vec![3],
        }];

        let decode_out = model.forward_decode_batch(&decode_input, &sequences, &mut kv_cache_mgr);
        assert!(decode_out.is_ok());
        let decode_out = decode_out.unwrap();
        assert_eq!(decode_out.dims(), &[1, 1, cfg.vocab_size]);
    }
}
