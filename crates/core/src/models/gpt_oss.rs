//! GptOss model implementation.
//!
//! GptOss is a complex distributed MoE model with:
//! - YaRN RoPE (Yet another RoPE extensioN) for extended context
//! - Attention sink mechanism (per-head learnable bias for sink tokens)
//! - MoE with FusedMoE-style routing (router has bias, uses renormalization)
//! - Alternating sliding window attention (even layers use sliding window)
//! - Standard pre-norm transformer with RMSNorm (eps=1e-5)

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::{AttentionBlock, AttentionConfig};
use crate::layers::RotaryEmbedding;

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── GptOss Config (parsed from ModelConfig.extra) ──────────────────────────

struct GptOssConfig {
    num_local_experts: usize,
    num_experts_per_tok: usize,
    /// YaRN RoPE parameters
    rope_theta: f64,
    #[allow(dead_code)]
    yarn_factor: f64,
    #[allow(dead_code)]
    yarn_beta_fast: f64,
    #[allow(dead_code)]
    yarn_beta_slow: f64,
    #[allow(dead_code)]
    yarn_original_max_position_embeddings: usize,
}

impl GptOssConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(16);

        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        // YaRN RoPE parameters from rope_parameters
        let rope_params = cfg.extra.get("rope_parameters");
        let rope_theta = rope_params
            .and_then(|p| p.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);
        let yarn_factor = rope_params
            .and_then(|p| p.get("factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let yarn_beta_fast = rope_params
            .and_then(|p| p.get("beta_fast"))
            .and_then(|v| v.as_f64())
            .unwrap_or(32.0);
        let yarn_beta_slow = rope_params
            .and_then(|p| p.get("beta_slow"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let yarn_original_max_position_embeddings = rope_params
            .and_then(|p| p.get("original_max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.max_position_embeddings);

        Self {
            num_local_experts,
            num_experts_per_tok,
            rope_theta,
            yarn_factor,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_max_position_embeddings,
        }
    }

    /// Even layers (0, 2, 4, ...) use sliding window attention.
    /// TODO: not yet wired into the migrated AttentionBlock path; sliding
    /// window support for GPT-OSS will be added once the alternating-layer
    /// pattern is generalized in the block.
    #[allow(dead_code)]
    fn uses_sliding_window(&self, layer_idx: usize) -> bool {
        layer_idx.is_multiple_of(2)
    }
}

// ─── OAI Attention ──────────────────────────────────────────────────────────

// GPT-OSS = vanilla GQA + fused qkv_proj. Per-head attention sinks and
// alternating sliding-window were defined but never applied in forward; that
// remains a TODO and is preserved here verbatim (sinks/sliding fields removed
// since they were both `#[allow(dead_code)]`).
struct OaiAttention {
    inner: AttentionBlock,
}

impl OaiAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        gpt_cfg: &GptOssConfig,
        _layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let attn_cfg = AttentionConfig::gqa(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.hidden_size,
        )
        .with_qkv_fused();
        // YaRN-style RoPE approximated with the YaRN rope_theta value (full
        // YaRN frequency correction is a separate TODO).
        let rotary_emb = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            gpt_cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;
        let inner = AttentionBlock::new(&attn_cfg, vb, pg, rotary_emb)?;
        Ok(Self { inner })
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
        self.inner.forward(
            xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            tp_ctx,
        )
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.inner
            .forward_decode_batch(xs, sequences, cache_engine, tp_ctx)
    }
}

// ─── MoE MLP Block ─────────────────────────────────────────────────────────

/// Expert layer using SwiGLU activation with bias.
///
/// GptOss uses `swigluoai` activation: silu(gate) * up, but the experts
/// have biases on gate_up and down projections.
struct GptOssMoEExpert {
    gate_proj: TpLinear,
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl GptOssMoEExpert {
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
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

struct GptOssMlpBlock {
    router: Linear,
    experts: Vec<GptOssMoEExpert>,
    num_experts: usize,
    #[allow(dead_code)]
    top_k: usize,
}

impl GptOssMlpBlock {
    fn new(
        cfg: &ModelConfig,
        gpt_cfg: &GptOssConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_experts = gpt_cfg.num_local_experts;
        let top_k = gpt_cfg.num_experts_per_tok;

        // Router with bias
        let router = linear(hidden_size, num_experts, vb.pp("router"))?;

        // Routed experts
        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(GptOssMoEExpert::new(
                hidden_size,
                cfg.intermediate_size,
                vb_experts.pp(i),
                pg,
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        // Router logits
        let router_logits = self.router.forward(&xs_2d)?;

        // Softmax for routing weights
        let routing_weights = candle_nn::ops::softmax(&router_logits, 1)?;

        // Top-k selection
        let top_k = self.top_k.min(self.num_experts);
        let routing_weights_f32 = routing_weights.to_dtype(DType::F32)?;

        let mut output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for token_idx in 0..num_tokens {
            let token_input = xs_2d.narrow(0, token_idx, 1)?;
            let token_weights: Vec<f32> = routing_weights_f32
                .narrow(0, token_idx, 1)?
                .flatten_all()?
                .to_vec1()?;

            // Find top-k experts
            let mut indexed_weights: Vec<(usize, f32)> =
                token_weights.iter().copied().enumerate().collect();
            indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed_weights.truncate(top_k);

            // Renormalize weights
            let weight_sum: f32 = indexed_weights.iter().map(|(_, w)| w).sum();
            let scale = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                1.0
            };

            let mut token_output = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;
            for (expert_idx, weight) in &indexed_weights {
                if *expert_idx < self.num_experts {
                    let expert_out = self.experts[*expert_idx].forward(&token_input, tp_ctx)?;
                    // Truncate expert output to hidden_dim (GptOss slices to hidden_size)
                    let expert_out = if expert_out.dim(1)? > hidden_dim {
                        expert_out.narrow(1, 0, hidden_dim)?
                    } else {
                        expert_out
                    };
                    let scaled_weight = (*weight * scale) as f64;
                    let weighted = expert_out.affine(scaled_weight, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            output = output.index_add(&indices, &token_output, 0)?;
        }

        output.reshape(orig_shape)
    }
}

// ─── Transformer Block ──────────────────────────────────────────────────────

struct GptOssTransformerBlock {
    attn: OaiAttention,
    mlp: GptOssMlpBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl GptOssTransformerBlock {
    fn new_with_tp(
        cfg: &ModelConfig,
        gpt_cfg: &GptOssConfig,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let attn = OaiAttention::new_with_tp(cfg, gpt_cfg, layer_idx, vb.pp("attn"), pg)?;
        let mlp = GptOssMlpBlock::new(cfg, gpt_cfg, vb.pp("mlp"), pg)?;

        // Fixed eps=1e-5 as in Python reference
        let input_layernorm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.hidden_size, 1e-5, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = if let Some(res) = residual {
            let combined = (xs + res)?;
            let normed = self.input_layernorm.forward(&combined)?;
            (normed, combined)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        let hidden_states = self.attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // Post-attention norm with residual add
        let combined = (&hidden_states + &residual)?;
        let residual = combined.clone();
        let hidden_states = self.post_attention_layernorm.forward(&combined)?;

        let output = self.mlp.forward(&hidden_states, tp_ctx)?;
        Ok((output, residual))
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = if let Some(res) = residual {
            let combined = (xs + res)?;
            let normed = self.input_layernorm.forward(&combined)?;
            (normed, combined)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        let hidden_states = self.attn.forward_decode_batch(
            &hidden_states,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let combined = (&hidden_states + &residual)?;
        let residual = combined.clone();
        let hidden_states = self.post_attention_layernorm.forward(&combined)?;

        let output = self.mlp.forward(&hidden_states, tp_ctx)?;
        Ok((output, residual))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// GptOss model for causal language modeling.
///
/// Distributed MoE with YaRN RoPE, attention sinks, and alternating
/// sliding window attention.
pub struct GptOssForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<GptOssTransformerBlock>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl GptOssForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let gpt_cfg = GptOssConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embedding"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GptOssTransformerBlock::new_with_tp(
                cfg,
                &gpt_cfg,
                i,
                vb_l.pp(i),
                pg,
            )?);
        }

        // Fixed eps=1e-5
        let norm = rms_norm(cfg.hidden_size, 1e-5, vb_m.pp("norm"))?;

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
                vb_m.pp("embedding"),
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
        let mut residual: Option<Tensor> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward(
                &xs,
                residual.as_ref(),
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
            xs = h;
            residual = Some(r);
        }

        // Final norm with residual
        let xs = if let Some(ref res) = residual {
            self.norm.forward(&(xs + res)?)?
        } else {
            self.norm.forward(&xs)?
        };

        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for GptOssForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        GptOssForCausalLM::forward(
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
        let mut residual: Option<Tensor> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (h, r) = layer.forward_decode_batch(
                &xs,
                residual.as_ref(),
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
            xs = h;
            residual = Some(r);
        }

        let xs = if let Some(ref res) = residual {
            self.norm.forward(&(xs + res)?)?
        } else {
            self.norm.forward(&xs)?
        };

        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_local_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert(
            "rope_parameters".to_string(),
            serde_json::json!({
                "rope_theta": 10000.0,
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 256,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            }),
        );

        ModelConfig {
            architectures: vec!["GptOssForCausalLM".to_string()],
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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: Some(128),
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

    // ─── Config Parsing Tests ───────────────────────────────────────────────────

    #[test]
    fn test_gptoss_config_parsing() {
        let cfg = test_config();
        let gpt_cfg = GptOssConfig::from_model_config(&cfg);
        assert_eq!(gpt_cfg.num_local_experts, 4);
        assert_eq!(gpt_cfg.num_experts_per_tok, 2);
        assert_eq!(gpt_cfg.rope_theta, 10000.0);
        assert_eq!(gpt_cfg.yarn_factor, 2.0);
        assert_eq!(gpt_cfg.yarn_beta_fast, 32.0);
        assert_eq!(gpt_cfg.yarn_beta_slow, 1.0);
        assert_eq!(gpt_cfg.yarn_original_max_position_embeddings, 256);
    }

    #[test]
    fn test_gptoss_config_defaults() {
        let cfg = ModelConfig::default();
        let gpt_cfg = GptOssConfig::from_model_config(&cfg);
        assert_eq!(gpt_cfg.num_local_experts, 16);
        assert_eq!(gpt_cfg.num_experts_per_tok, 2);
        assert_eq!(gpt_cfg.yarn_factor, 1.0);
    }

    #[test]
    fn test_gptoss_sliding_window_alternation() {
        let cfg = test_config();
        let gpt_cfg = GptOssConfig::from_model_config(&cfg);
        // Even layers use sliding window
        assert!(gpt_cfg.uses_sliding_window(0));
        assert!(!gpt_cfg.uses_sliding_window(1));
        assert!(gpt_cfg.uses_sliding_window(2));
        assert!(!gpt_cfg.uses_sliding_window(3));
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_gptoss_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GptOssForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GptOssForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    // NOTE: tests for the `sinks` tensor allocation and per-layer `sliding_window`
    // field assignment were removed during the AttentionBlock migration. Both
    // fields existed in the bespoke OaiAttention struct but were never applied
    // in forward (they were `#[allow(dead_code)]`). When the alternating
    // sliding-window pattern + attention sinks are actually wired into the
    // forward path, observable forward-output tests should be added instead of
    // probing struct internals.

    // ─── Forward Tests ──────────────────────────────────────────────────────────
    #[test]
    fn test_gptoss_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GptOssForCausalLM::new(&cfg, vb).expect("build model");

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

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_gptoss_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GptOssForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gptoss_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GptOssForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward via trait");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_gptoss_moe_mlp_forward() {
        let cfg = test_config();
        let gpt_cfg = GptOssConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = GptOssMlpBlock::new(&cfg, &gpt_cfg, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MoE MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_gptoss_no_sliding_window_config() {
        let mut cfg = test_config();
        cfg.sliding_window = None;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GptOssForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct without sliding window: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_gptoss_tp_context() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GptOssForCausalLM::new(&cfg, vb).unwrap();
        assert!(model.tp_context().is_single());
    }
}
