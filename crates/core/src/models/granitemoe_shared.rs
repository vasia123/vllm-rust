//! GraniteMoeShared model implementation.
//!
//! Extends GraniteMoe with shared expert MLP alongside routed MoE.
//! When `shared_intermediate_size > 0`, a shared SwiGLU MLP processes all tokens
//! in parallel with the MoE layer, and their outputs are summed.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/granitemoeshared.py`

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayer, MoELayerConfig};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Config ──────────────────────────────────────────────────────────────────

struct GraniteMoeSharedConfig {
    attention_multiplier: f64,
    residual_multiplier: f64,
    embedding_multiplier: f64,
    logits_scaling: f64,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    shared_intermediate_size: usize,
}

impl GraniteMoeSharedConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let attention_multiplier = cfg
            .extra
            .get("attention_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0 / (cfg.head_dim as f64).sqrt());
        let residual_multiplier = cfg
            .extra
            .get("residual_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let embedding_multiplier = cfg
            .extra
            .get("embedding_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let logits_scaling = cfg
            .extra
            .get("logits_scaling")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let shared_intermediate_size = cfg
            .extra
            .get("shared_intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        Self {
            attention_multiplier,
            residual_multiplier,
            embedding_multiplier,
            logits_scaling,
            num_local_experts,
            num_experts_per_tok,
            shared_intermediate_size,
        }
    }
}

// ─── Shared MLP ──────────────────────────────────────────────────────────────

/// Shared expert MLP using SwiGLU activation.
///
/// gate_proj and up_proj are separate linear layers (matching the weight layout
/// from `input_linear` which is a MergedColumnParallel in Python).
/// output_linear maps back to hidden_size.
struct GraniteMoeSharedMLP {
    gate_proj: Linear,
    up_proj: Linear,
    output_linear: Linear,
}

impl GraniteMoeSharedMLP {
    fn new(hidden_size: usize, shared_intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            hidden_size,
            shared_intermediate_size,
            vb.pp("input_linear.0"),
        )?;
        let up_proj = linear_no_bias(
            hidden_size,
            shared_intermediate_size,
            vb.pp("input_linear.1"),
        )?;
        let output_linear = linear_no_bias(
            shared_intermediate_size,
            hidden_size,
            vb.pp("output_linear"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            output_linear,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(gate_proj(x)) * up_proj(x), then output_linear
        let gate = self.gate_proj.forward(xs)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = (gate * up)?;
        self.output_linear.forward(&hidden)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// GraniteMoeShared attention -- identical to GraniteMoe attention.
/// Uses custom attention_multiplier scaling instead of 1/sqrt(head_dim).
#[allow(dead_code)]
struct GraniteMoeSharedAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f64,
}

impl GraniteMoeSharedAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        gmoe_cfg: &GraniteMoeSharedConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
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
            rotary_emb,
            num_heads: num_heads / world_size,
            num_kv_heads: num_kv_heads / world_size,
            head_dim,
            attention_multiplier: gmoe_cfg.attention_multiplier,
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
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

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

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let max_blocks_per_seq = sequences
                .iter()
                .map(|s| s.block_ids.len())
                .max()
                .unwrap_or(1);
            let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
            for (i, seq) in sequences.iter().enumerate() {
                for (j, &block_id) in seq.block_ids.iter().enumerate() {
                    bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                }
            }
            let block_tables =
                Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

            let seq_lens_data: Vec<u32> = sequences
                .iter()
                .map(|s| (s.seqlen_offset + 1) as u32)
                .collect();
            let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
            let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

            let scale = self.attention_multiplier as f32;

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
            )?;

            self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
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

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.o_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct GraniteMoeSharedDecoderLayer {
    self_attn: GraniteMoeSharedAttention,
    block_sparse_moe: MoELayer,
    shared_mlp: Option<GraniteMoeSharedMLP>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    residual_multiplier: f64,
}

impl GraniteMoeSharedDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        gmoe_cfg: &GraniteMoeSharedConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn =
            GraniteMoeSharedAttention::new_with_tp(cfg, gmoe_cfg, vb.pp("self_attn"), pg)?;

        let moe_config = MoELayerConfig {
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_experts: gmoe_cfg.num_local_experts,
            top_k: gmoe_cfg.num_experts_per_tok,
            renormalize: true,
            inplace: false,
            is_act_and_mul: true,
        };
        let block_sparse_moe = MoELayer::new(moe_config, vb.pp("block_sparse_moe"))?;

        let shared_mlp = if gmoe_cfg.shared_intermediate_size > 0 {
            Some(GraniteMoeSharedMLP::new(
                cfg.hidden_size,
                gmoe_cfg.shared_intermediate_size,
                vb.pp("shared_mlp"),
            )?)
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
            block_sparse_moe,
            shared_mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: gmoe_cfg.residual_multiplier,
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
        // Self attention with pre-norm
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
        let xs = (residual + (xs * self.residual_multiplier)?)?;

        // MLP with pre-norm
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        let mlp_output = match &self.shared_mlp {
            None => self.block_sparse_moe.forward(&xs)?,
            Some(shared) => {
                let moe_out = self.block_sparse_moe.forward(&xs)?;
                let shared_out = shared.forward(&xs)?;
                (moe_out + shared_out)?
            }
        };

        residual + (mlp_output * self.residual_multiplier)?
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        // Self attention with pre-norm
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (residual + (xs * self.residual_multiplier)?)?;

        // MLP with pre-norm
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;

        let mlp_output = match &self.shared_mlp {
            None => self.block_sparse_moe.forward(&xs)?,
            Some(shared) => {
                let moe_out = self.block_sparse_moe.forward(&xs)?;
                let shared_out = shared.forward(&xs)?;
                (moe_out + shared_out)?
            }
        };

        residual + (mlp_output * self.residual_multiplier)?
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// GraniteMoeShared model for causal language modeling.
///
/// Extends GraniteMoe with a shared expert MLP that runs in parallel
/// with the routed MoE layer. When `shared_intermediate_size > 0`, the
/// shared MLP output is summed with the MoE output.
pub struct GraniteMoeSharedForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<GraniteMoeSharedDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    embedding_multiplier: f64,
    logit_scale: f64,
}

impl GraniteMoeSharedForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let gmoe_cfg = GraniteMoeSharedConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GraniteMoeSharedDecoderLayer::new_with_tp(
                cfg,
                &gmoe_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

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

        let logit_scale = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            / gmoe_cfg.logits_scaling;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            embedding_multiplier: gmoe_cfg.embedding_multiplier,
            logit_scale,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for GraniteMoeSharedForCausalLM {
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

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        xs = (xs * self.embedding_multiplier)?;

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
        if (self.logit_scale - 1.0).abs() > f64::EPSILON {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        xs = (xs * self.embedding_multiplier)?;

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
        if (self.logit_scale - 1.0).abs() > f64::EPSILON {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
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
        test_config_with_shared(256)
    }

    fn test_config_with_shared(shared_intermediate_size: usize) -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "attention_multiplier".into(),
            serde_json::Value::from(0.08838834764831845),
        );
        extra.insert(
            "residual_multiplier".into(),
            serde_json::Value::from(0.22360679774997896),
        );
        extra.insert("embedding_multiplier".into(), serde_json::Value::from(12.0));
        extra.insert("logits_scaling".into(), serde_json::Value::from(13.0));
        extra.insert("num_local_experts".into(), serde_json::Value::from(4));
        extra.insert("num_experts_per_tok".into(), serde_json::Value::from(2));
        extra.insert(
            "shared_intermediate_size".into(),
            serde_json::Value::from(shared_intermediate_size),
        );

        ModelConfig {
            architectures: vec!["GraniteMoeSharedForCausalLM".to_string()],
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
            tie_word_embeddings: true,
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
    fn test_granitemoe_shared_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GraniteMoeSharedForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_granitemoe_shared_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
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
    fn test_granitemoe_shared_without_shared_mlp() {
        // When shared_intermediate_size=0, the model should behave like standard GraniteMoe
        let cfg = test_config_with_shared(0);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct without shared MLP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        // Verify shared_mlp is None for all layers
        for layer in &model.layers {
            assert!(
                layer.shared_mlp.is_none(),
                "shared_mlp should be None when shared_intermediate_size=0"
            );
        }
    }

    #[test]
    fn test_granitemoe_shared_with_shared_mlp() {
        // When shared_intermediate_size>0, the model should have shared MLPs
        let cfg = test_config_with_shared(256);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb).expect("build model");

        for layer in &model.layers {
            assert!(
                layer.shared_mlp.is_some(),
                "shared_mlp should be Some when shared_intermediate_size>0"
            );
        }
    }

    #[test]
    fn test_granitemoe_shared_config_extraction() {
        let cfg = test_config();
        let gmoe_cfg = GraniteMoeSharedConfig::from_model_config(&cfg);

        assert!((gmoe_cfg.attention_multiplier - 0.08838834764831845).abs() < 1e-10);
        assert!((gmoe_cfg.residual_multiplier - 0.22360679774997896).abs() < 1e-10);
        assert!((gmoe_cfg.embedding_multiplier - 12.0).abs() < 1e-10);
        assert!((gmoe_cfg.logits_scaling - 13.0).abs() < 1e-10);
        assert_eq!(gmoe_cfg.num_local_experts, 4);
        assert_eq!(gmoe_cfg.num_experts_per_tok, 2);
        assert_eq!(gmoe_cfg.shared_intermediate_size, 256);
    }

    #[test]
    fn test_granitemoe_shared_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 4), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping: Vec<usize> = (0..4).collect();

        let logits = crate::engine::ModelForward::forward(
            &model,
            &prompt,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("prefill");
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
        block_table.advance(4);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(4, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = crate::engine::ModelForward::forward(
            &model,
            &next_token,
            4,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_granitemoe_shared_without_shared_forward() {
        // Forward pass without shared MLP should also work
        let cfg = test_config_with_shared(0);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GraniteMoeSharedForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 3;
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
        .expect("forward");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_granitemoe_shared_logit_scaling() {
        let cfg = test_config();
        let gmoe_cfg = GraniteMoeSharedConfig::from_model_config(&cfg);

        // logit_scale = logit_scale_config(default 1.0) / logits_scaling(13.0)
        let expected = 1.0 / 13.0;
        let actual = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
            / gmoe_cfg.logits_scaling;

        assert!(
            (actual - expected).abs() < 1e-10,
            "logit_scale should be 1/13, got {actual}"
        );
    }
}
