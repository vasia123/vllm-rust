//! Ouro (OuroForCausalLM) model implementation.
//!
//! Multi-pass (UT: Unroll and Think) architecture:
//! - Multiple UT steps per forward pass — layers are re-used across steps
//! - Each UT step has its own KV cache (separate attention backend per step)
//! - Early exit gate mechanism (RowParallelLinear → scalar)
//! - Dual layernorms: input_layernorm + input_layernorm_2 around attention,
//!   post_attention_layernorm + post_attention_layernorm_2 around MLP
//! - SiLU activation (SwiGLU MLP), RoPE, RMSNorm
//!
//! The KV cache allocation is num_layers * total_ut_steps to give each
//! UT step its own cache state per layer.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Ouro Config (parsed from ModelConfig.extra) ────────────────────────────

struct OuroConfig {
    total_ut_steps: usize,
}

impl OuroConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let total_ut_steps = cfg
            .extra
            .get("total_ut_steps")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        Self { total_ut_steps }
    }
}

// ─── MLP ────────────────────────────────────────────────────────────────────

struct OuroMLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
}

impl OuroMLP {
    fn new_with_tp(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            false,
            false,
            vb.pp("gate_up_proj"),
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
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(xs, tp_ctx)?;
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = &chunks[0];
        let up = &chunks[1];
        let gate_act = candle_nn::ops::silu(gate)?;
        let intermediate = gate_act.mul(up)?;
        self.down_proj.forward(&intermediate, tp_ctx)
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct OuroAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Number of UT steps — determines how many separate KV cache layers this
    /// attention logically manages.
    total_ut_steps: usize,
}

impl OuroAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        ouro_cfg: &OuroConfig,
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

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

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
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            total_ut_steps: ouro_cfg.total_ut_steps,
        })
    }

    /// Compute the effective cache layer index for a given base layer and UT step.
    /// Each UT step gets its own set of cache layers:
    /// cache_layer = ut_step * num_base_layers + base_layer_idx
    #[allow(dead_code)]
    fn cache_layer_idx(
        &self,
        base_layer_idx: usize,
        current_ut: usize,
        num_layers: usize,
    ) -> usize {
        current_ut * num_layers + base_layer_idx
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

            let scale = 1.0 / (self.head_dim as f32).sqrt();

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

// ─── Decoder Layer ──────────────────────────────────────────────────────────

/// Ouro decoder layer with dual layernorms around both attention and MLP.
///
/// The dual layernorm pattern applies a second layernorm to the attention/MLP
/// output before adding to the residual, providing additional normalization
/// stability for the multi-pass UT architecture.
struct OuroDecoderLayer {
    self_attn: OuroAttention,
    mlp: OuroMLP,
    input_layernorm: RmsNorm,
    input_layernorm_2: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_attention_layernorm_2: RmsNorm,
}

impl OuroDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        ouro_cfg: &OuroConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = OuroAttention::new_with_tp(cfg, ouro_cfg, vb.pp("self_attn"), pg)?;

        let mlp = OuroMLP::new_with_tp(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let input_layernorm_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("input_layernorm_2"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let post_attention_layernorm_2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm_2"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            input_layernorm_2,
            post_attention_layernorm,
            post_attention_layernorm_2,
        })
    }

    /// Forward with residual tracking.
    ///
    /// Returns (hidden_states, residual) to support the norm-fused residual pattern.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        // Pre-attention layernorm with residual
        let (xs, residual) = if let Some(res) = residual {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, (xs + res)?)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        // Attention
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // Second input layernorm on attention output
        let xs = self.input_layernorm_2.forward(&xs)?;

        // Post-attention layernorm with residual merge
        let xs_plus_residual = (&xs + &residual)?;
        let xs = self.post_attention_layernorm.forward(&xs_plus_residual)?;
        let residual = xs_plus_residual;

        // MLP
        let xs = self.mlp.forward(&xs, tp_ctx)?;

        // Second post-attention layernorm on MLP output
        let xs = self.post_attention_layernorm_2.forward(&xs)?;

        Ok((xs, residual))
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        residual: Option<&Tensor>,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor)> {
        let (xs, residual) = if let Some(res) = residual {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, (xs + res)?)
        } else {
            let normed = self.input_layernorm.forward(xs)?;
            (normed, xs.clone())
        };

        let xs = self
            .self_attn
            .forward_decode_batch(&xs, sequences, cache_engine, tp_ctx)?;

        let xs = self.input_layernorm_2.forward(&xs)?;

        let xs_plus_residual = (&xs + &residual)?;
        let xs = self.post_attention_layernorm.forward(&xs_plus_residual)?;
        let residual = xs_plus_residual;

        let xs = self.mlp.forward(&xs, tp_ctx)?;
        let xs = self.post_attention_layernorm_2.forward(&xs)?;

        Ok((xs, residual))
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// OuroForCausalLM: multi-pass (UT) transformer for causal language modeling.
///
/// The model loops through all decoder layers `total_ut_steps` times per
/// forward call. Each UT step uses a separate KV cache partition, so the
/// total number of cache layers is `num_hidden_layers * total_ut_steps`.
#[allow(dead_code)]
pub struct OuroForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<OuroDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    early_exit_gate: Linear,
    total_ut_steps: usize,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl OuroForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let ouro_cfg = OuroConfig::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(OuroDecoderLayer::new_with_tp(
                cfg,
                &ouro_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Early exit gate: hidden_size -> 1 (with bias)
        let early_exit_gate = candle_nn::linear(cfg.hidden_size, 1, vb_m.pp("early_exit_gate"))?;

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
            early_exit_gate,
            total_ut_steps: ouro_cfg.total_ut_steps,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Total number of KV cache layers needed: base layers * UT steps.
    pub fn total_cache_layers(&self) -> usize {
        self.layers.len() * self.total_ut_steps
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
        let num_layers = self.layers.len();

        // Multi-pass: iterate UT steps, each passing through all layers
        for current_ut in 0..self.total_ut_steps {
            let mut residual: Option<Tensor> = None;

            for (base_layer_idx, layer) in self.layers.iter().enumerate() {
                let cache_layer_idx = current_ut * num_layers + base_layer_idx;
                let (new_xs, new_residual) = layer.forward(
                    &xs,
                    residual.as_ref(),
                    attention_mask.as_ref(),
                    seqlen_offset,
                    kv_cache_mgr.engine_mut(cache_layer_idx),
                    block_table,
                    slot_mapping,
                    &self.tp_ctx,
                )?;
                xs = new_xs;
                residual = Some(new_residual);
            }

            // Fuse final norm with residual after each UT step
            if let Some(ref res) = residual {
                xs = (&xs + res)?;
            }
            xs = self.norm.forward(&xs)?;
        }

        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for OuroForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        OuroForCausalLM::forward(
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
        let num_layers = self.layers.len();

        for current_ut in 0..self.total_ut_steps {
            let mut residual: Option<Tensor> = None;

            for (base_layer_idx, layer) in self.layers.iter().enumerate() {
                let cache_layer_idx = current_ut * num_layers + base_layer_idx;
                let (new_xs, new_residual) = layer.forward_decode_batch(
                    &xs,
                    residual.as_ref(),
                    sequences,
                    kv_cache_mgr.engine_mut(cache_layer_idx),
                    &self.tp_ctx,
                )?;
                xs = new_xs;
                residual = Some(new_residual);
            }

            if let Some(ref res) = residual {
                xs = (&xs + res)?;
            }
            xs = self.norm.forward(&xs)?;
        }

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
        extra.insert("total_ut_steps".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["OuroForCausalLM".to_string()],
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

    /// Ouro needs num_layers * total_ut_steps cache layers.
    fn create_cache_config(
        cfg: &ModelConfig,
        total_ut_steps: usize,
        device: &Device,
    ) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers * total_ut_steps,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ─── Config Tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_ouro_config_defaults() {
        let cfg = ModelConfig::default();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        assert_eq!(ouro_cfg.total_ut_steps, 4);
    }

    #[test]
    fn test_ouro_config_custom() {
        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        assert_eq!(ouro_cfg.total_ut_steps, 2);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_ouro_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = OuroForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "OuroForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.total_ut_steps, 2);
    }

    #[test]
    fn test_ouro_total_cache_layers() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).unwrap();

        // 2 layers * 2 UT steps = 4 cache layers
        assert_eq!(model.total_cache_layers(), 4);
    }

    #[test]
    fn test_ouro_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_ouro_forward_shape() {
        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, ouro_cfg.total_ut_steps, &device);
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
    fn test_ouro_prefill_then_decode() {
        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, ouro_cfg.total_ut_steps, &device);
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
    fn test_ouro_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, ouro_cfg.total_ut_steps, &device);
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

    // ─── Multi-pass UT Tests ────────────────────────────────────────────────────

    #[test]
    fn test_ouro_multi_ut_steps() {
        // Verify with 1 UT step vs 2 UT steps to ensure they both produce valid output
        for ut_steps in [1, 2, 4] {
            let mut cfg = test_config();
            cfg.extra
                .insert("total_ut_steps".to_string(), serde_json::json!(ut_steps));

            let device = Device::Cpu;
            let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
            let model = OuroForCausalLM::new(&cfg, vb).expect("build model");
            assert_eq!(model.total_ut_steps, ut_steps);
            assert_eq!(model.total_cache_layers(), cfg.num_hidden_layers * ut_steps);

            let cache_config = create_cache_config(&cfg, ut_steps, &device);
            let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
            let mut block_table = BlockTable::new(cache_config.block_size);

            let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
            kv_cache_mgr
                .allocate_for_request(&mut block_table, 3)
                .expect("allocate");
            let slot_mapping = block_table.slot_mapping(0, 3);

            let logits = model
                .forward(
                    &input_ids,
                    0,
                    &mut kv_cache_mgr,
                    &block_table,
                    &slot_mapping,
                )
                .expect("forward");
            assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        }
    }

    #[test]
    fn test_ouro_cache_layer_indexing() {
        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).unwrap();

        // With 2 layers and 2 UT steps, cache indices should be:
        // UT 0: layer 0 -> cache 0, layer 1 -> cache 1
        // UT 1: layer 0 -> cache 2, layer 1 -> cache 3
        let attn = &model.layers[0].self_attn;
        assert_eq!(attn.cache_layer_idx(0, 0, 2), 0);
        assert_eq!(attn.cache_layer_idx(1, 0, 2), 1);
        assert_eq!(attn.cache_layer_idx(0, 1, 2), 2);
        assert_eq!(attn.cache_layer_idx(1, 1, 2), 3);
    }

    // ─── Component Tests ────────────────────────────────────────────────────────

    #[test]
    fn test_ouro_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = OuroMLP::new_with_tp(64, 128, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }

    #[test]
    fn test_ouro_dual_layernorm_exists() {
        let cfg = test_config();
        let ouro_cfg = OuroConfig::from_model_config(&cfg);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let layer = OuroDecoderLayer::new_with_tp(&cfg, &ouro_cfg, vb.pp("layer"), &pg).unwrap();

        // Verify dual layernorms by checking forward doesn't panic
        let input = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: 1,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache");
        let mut block_table = BlockTable::new(16);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("alloc");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let tp_ctx = TpContext::single_gpu();

        let mask = crate::layers::causal_mask(3, 0, DType::F32, &device).ok();

        let result = layer.forward(
            &input,
            None,
            mask.as_ref(),
            0,
            kv_cache_mgr.engine_mut(0),
            &block_table,
            &slot_mapping,
            &tp_ctx,
        );
        assert!(result.is_ok(), "Decoder layer forward: {:?}", result.err());
        let (hidden, residual) = result.unwrap();
        assert_eq!(hidden.dims(), &[1, 3, cfg.hidden_size]);
        assert_eq!(residual.dims(), &[1, 3, cfg.hidden_size]);
    }

    #[test]
    fn test_ouro_early_exit_gate_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OuroForCausalLM::new(&cfg, vb).unwrap();

        // Early exit gate: hidden_size -> 1
        let input = Tensor::zeros((2, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let gate_out = model.early_exit_gate.forward(&input);
        assert!(gate_out.is_ok(), "Gate forward: {:?}", gate_out.err());
        assert_eq!(gate_out.unwrap().dims(), &[2, 3, 1]);
    }
}
