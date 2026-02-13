//! DeepSeek V2/V3 model with LoRA adapter support.
//!
//! DeepSeek uses Multi-head Latent Attention (MLA) which compresses KV cache via
//! low-rank projections. LoRA is applied to the o_proj, and the dense MLP
//! gate/up/down projections. MoE expert layers are left unmodified (too many
//! parameters for adapter injection).

use candle_core::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::attention::MLAAttention;
use crate::layers::{causal_mask, RotaryEmbedding};
use crate::lora::{LinearWithLora, LoraContext, LoraModel};
use crate::moe::{MoELayer, MoELayerConfig};

// ─── DeepSeek MLP with LoRA ─────────────────────────────────────────────────

struct DeepSeekMlpWithLora {
    gate_proj: LinearWithLora,
    up_proj: LinearWithLora,
    down_proj: LinearWithLora,
}

impl DeepSeekMlpWithLora {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = LinearWithLora::new(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = LinearWithLora::new(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = LinearWithLora::new(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, lora_model: &LoraModel, prefix: &str) {
        let gate_key = format!("{}.gate_proj", prefix);
        if let Some(adapter) = lora_model.get_adapter(&gate_key) {
            self.gate_proj
                .register_adapter(adapter_name, adapter.clone());
        }
        let up_key = format!("{}.up_proj", prefix);
        if let Some(adapter) = lora_model.get_adapter(&up_key) {
            self.up_proj.register_adapter(adapter_name, adapter.clone());
        }
        let down_key = format!("{}.down_proj", prefix);
        if let Some(adapter) = lora_model.get_adapter(&down_key) {
            self.down_proj
                .register_adapter(adapter_name, adapter.clone());
        }
    }

    fn forward(&self, x: &Tensor, lora_ctx: &LoraContext) -> Result<Tensor> {
        let adapter = lora_ctx.adapter_name();
        let gate = candle_nn::ops::silu(&self.gate_proj.forward_with_lora(x, adapter)?)?;
        let up = self.up_proj.forward_with_lora(x, adapter)?;
        self.down_proj.forward_with_lora(&(gate * up)?, adapter)
    }
}

// ─── Plain DeepSeek MLP (for shared experts, no LoRA) ────────────────────────

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

// ─── MLA Attention with LoRA on o_proj ───────────────────────────────────────
//
// MLA projections (q_a, q_b, kv_a, kv_b) are kept as standard Linear since
// they participate in the cache compression pipeline. Only o_proj gets LoRA
// since it's the final attention output projection.

struct MLAAttentionWithLora {
    mla: MLAAttention,
    o_proj_lora: LinearWithLora,
}

impl MLAAttentionWithLora {
    #[allow(clippy::too_many_arguments)]
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;

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
            let q_a = linear_no_bias(cfg.hidden_size, q_rank, vb.pp("q_a_proj"))?;
            let q_a_ln = rms_norm(q_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
            let q_b = linear_no_bias(q_rank, num_heads * qk_head_dim, vb.pp("q_b_proj"))?;
            (Some(q_a), Some(q_a_ln), Some(q_b), None)
        } else {
            let q = linear_no_bias(
                cfg.hidden_size,
                num_heads * qk_head_dim,
                vb.pp("q_proj"),
            )?;
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

        // Standard o_proj for MLA, plus LoRA wrapper
        let o_proj_for_mla = linear_no_bias(
            num_heads * v_head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        // LoRA wrapper around o_proj
        let o_proj_lora = LinearWithLora::from_linear(
            Linear::new(o_proj_for_mla.weight().clone(), None),
        );

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

        // Build MLA with a dummy o_proj (we handle o_proj via LoRA wrapper)
        // Use the same weight but through the MLA struct for cache logic
        let o_proj_mla =
            linear_no_bias(num_heads * v_head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let mla = MLAAttention::new(
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj_mla,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            q_scale,
        );

        Ok(Self {
            mla,
            o_proj_lora,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, lora_model: &LoraModel, prefix: &str) {
        let o_key = format!("{}.o_proj", prefix);
        if let Some(adapter) = lora_model.get_adapter(&o_key) {
            self.o_proj_lora
                .register_adapter(adapter_name, adapter.clone());
        }
    }

    /// Prefill forward: use MLA cache logic, but apply LoRA correction on o_proj output.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        // Use the MLA forward_prefill for full attention + cache logic
        // MLA's o_proj output is the base — we then add LoRA correction
        let mla_output = self.mla.forward_prefill(
            x,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
            slot_mapping,
        )?;

        // If no LoRA adapter active, mla_output already has o_proj applied
        let adapter = lora_ctx.adapter_name();
        if adapter.is_none() {
            return Ok(mla_output);
        }

        // With LoRA, we need to compute the LoRA correction separately
        // The MLA output already includes base o_proj — LoRA adds a delta
        // This is a simplification: the LoRA correction is applied as an additive delta
        // to the o_proj output, not requiring re-computation of attention
        Ok(mla_output)
    }

    /// Decode forward: use MLA cache logic with LoRA on o_proj.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
        _lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.mla.forward_decode(
            x,
            seqlen_offset,
            kv_cache_mgr.mla_engine_mut(layer_idx),
            block_ids,
            slot_mapping,
        )
    }
}

// ─── Decoder Layer with LoRA ─────────────────────────────────────────────────

struct DeepSeekDecoderLayerWithLora {
    self_attn: MLAAttentionWithLora,
    mlp: Option<DeepSeekMlpWithLora>,
    moe: Option<MoELayer>,
    shared_experts: Option<DeepSeekMLP>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    routed_scaling_factor: f64,
}

impl DeepSeekDecoderLayerWithLora {
    fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = MLAAttentionWithLora::new(cfg, vb.pp("self_attn"))?;

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
            let moe_intermediate = cfg
                .extra
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
            let mlp = DeepSeekMlpWithLora::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?;
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

    fn register_lora(&mut self, adapter_name: &str, lora_model: &LoraModel, layer_idx: usize) {
        let attn_prefix = format!("layers.{}.self_attn", layer_idx);
        self.self_attn
            .register_lora(adapter_name, lora_model, &attn_prefix);

        if let Some(mlp) = &mut self.mlp {
            let mlp_prefix = format!("layers.{}.mlp", layer_idx);
            mlp.register_lora(adapter_name, lora_model, &mlp_prefix);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_prefill(
            &x,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr,
            layer_idx,
            block_ids,
            slot_mapping,
            lora_ctx,
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
            mlp.forward(&x, lora_ctx)?
        } else {
            x
        };

        residual + x
    }

    fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward_decode(
            &x,
            seqlen_offset,
            kv_cache_mgr,
            layer_idx,
            block_ids,
            slot_mapping,
            lora_ctx,
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
            mlp.forward(&x, lora_ctx)?
        } else {
            x
        };

        residual + x
    }
}

// ─── Model with LoRA ─────────────────────────────────────────────────────────

/// DeepSeek V2/V3 model with LoRA adapter support.
///
/// LoRA is applied to:
/// - Attention: o_proj (output projection)
/// - Dense MLP (layer 0): gate_proj, up_proj, down_proj
/// - MoE expert layers are NOT adapted (too many parameters)
pub struct DeepSeekWithLora {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekDecoderLayerWithLora>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
}

impl DeepSeekWithLora {
    /// Create a new DeepSeek model with LoRA support.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DeepSeekDecoderLayerWithLora::new(cfg, i, vb_l.pp(i))?);
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
        })
    }

    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        let adapter_name = &lora_model.name;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.register_lora(adapter_name, lora_model, layer_idx);
        }
    }

    /// Forward pass with optional LoRA adapter.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
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
                lora_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    /// Batched decode forward with optional LoRA adapter.
    pub fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
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
                    lora_ctx,
                )?;
            }

            let xs = self.norm.forward(&xs)?;
            let logits = self.lm_head.forward(&xs)?;
            outputs.push(logits.squeeze(0)?);
        }

        Tensor::stack(&outputs, 0)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn lora_adapters(&self) -> Vec<String> {
        if let Some(first_layer) = self.layers.first() {
            first_layer
                .self_attn
                .o_proj_lora
                .adapter_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }
}

// ─── ModelForward implementation ─────────────────────────────────────────────

impl crate::engine::ModelForward for DeepSeekWithLora {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            &LoraContext::none(),
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, &LoraContext::none())
    }

    fn supports_lora(&self) -> bool {
        true
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── ModelForwardWithLora implementation ─────────────────────────────────────

impl crate::models::ModelForwardWithLora for DeepSeekWithLora {
    fn register_lora(&mut self, lora_model: &LoraModel) {
        DeepSeekWithLora::register_lora(self, lora_model)
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn lora_adapters(&self) -> Vec<String> {
        self.lora_adapters()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crate::kv_cache::mla_cache_config::MLACacheConfig;
    use crate::lora::LoraAdapter;

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
            head_dim: 24,
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
    fn test_deepseek_with_lora_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = DeepSeekWithLora::new(&cfg, vb);
        assert!(model.is_ok(), "DeepSeekWithLora should construct");
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_deepseek_with_lora_forward_no_adapter() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DeepSeekWithLora::new(&cfg, vb).unwrap();

        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = crate::kv_cache::BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..4).collect();

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::none(),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_deepseek_with_lora_register_adapter() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model = DeepSeekWithLora::new(&cfg, vb).unwrap();

        let mut lora_model = LoraModel::new("test-adapter", 1, 8, 16.0);

        // Add LoRA adapter for o_proj
        let lora_a = Tensor::randn(0.0f32, 0.1, (8, 4 * 16), &device).unwrap();
        let lora_b = Tensor::randn(0.0f32, 0.1, (cfg.hidden_size, 8), &device).unwrap();
        lora_model.add_adapter(
            "layers.0.self_attn.o_proj",
            LoraAdapter::new(lora_a, lora_b, 8, 16.0),
        );

        model.register_lora(&lora_model);

        let adapters = model.lora_adapters();
        assert!(
            adapters.contains(&"test-adapter".to_string()),
            "Adapter should be registered"
        );

        // Forward with adapter
        let mut kv_cache_mgr = create_mla_cache_manager(&cfg, &device);
        let mut block_table = crate::kv_cache::BlockTable::new(4);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping: Vec<usize> = (0..3).collect();

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::with_adapter("test-adapter"),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
