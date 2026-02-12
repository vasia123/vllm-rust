//! Step1 model implementation.
//!
//! Step1 is a decoder-only transformer with:
//! - ALiBi-based attention with sqrt distance scaling (no RoPE)
//! - SiLU (SwiGLU) activation in the MLP
//! - RMSNorm pre-norm
//! - GQA support via `num_attention_groups` config key
//!
//! The sqrt ALiBi variant computes:
//!   bias[h, i, j] = slope[h] * (-sqrt(i - j))  for j <= i (causal)
//! instead of the standard ALiBi's linear distance.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, AlibiAttentionBias};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Step1 Config Extraction ────────────────────────────────────────────────

/// Parsed Step1-specific configuration from the HuggingFace config.
struct Step1Config {
    attention_bias: bool,
    mlp_bias: bool,
}

impl Step1Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let mlp_bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            attention_bias,
            mlp_bias,
        }
    }
}

// ─── Sqrt ALiBi Bias ────────────────────────────────────────────────────────

/// Build ALiBi bias matrix with sqrt distance scaling.
///
/// Instead of the standard linear distance `slope * (j - i)`, this computes:
///   `slope * (-sqrt(i - j))` for past positions (j <= i)
///   `0` for future positions (j > i)
///
/// This matches the Step model's `use_alibi_sqrt=True` behavior.
fn build_sqrt_alibi_bias(
    alibi: &AlibiAttentionBias,
    seq_len: usize,
    kv_len: usize,
) -> Result<Tensor> {
    let device = alibi.slopes().device();
    let dtype = alibi.slopes().dtype();
    let num_heads = alibi.num_heads();

    let kv_offset = kv_len - seq_len;

    // Build sqrt distance matrix [seq_len, kv_len]
    let distances: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                let query_pos = kv_offset + i;
                let relative_pos = j as i64 - query_pos as i64;
                if relative_pos <= 0 {
                    // Past or current position: -sqrt(|distance|)
                    -((-relative_pos) as f32).sqrt()
                } else {
                    // Future position: 0 (will be masked by causal mask)
                    0.0
                }
            })
        })
        .collect();

    let distance_matrix =
        Tensor::from_vec(distances, (1, 1, seq_len, kv_len), device)?.to_dtype(dtype)?;

    // Expand slopes to [1, num_heads, 1, 1] for broadcasting
    let slopes_expanded = alibi.slopes().reshape((1, num_heads, 1, 1))?;

    // bias = slopes * sqrt_distances
    distance_matrix.broadcast_mul(&slopes_expanded)
}

// ─── Attention ──────────────────────────────────────────────────────────────

struct Step1Attention {
    qkv_proj: TpLinear,
    o_proj: TpLinear,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl Step1Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        step_cfg: &Step1Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();
        let rank = pg.rank();

        // GQA: num_kv_heads from num_attention_groups or num_key_value_heads
        let total_num_kv_heads = cfg
            .extra
            .get("num_attention_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_key_value_heads)
            .max(1);

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = if total_num_kv_heads >= world_size {
            total_num_kv_heads / world_size
        } else {
            1
        };
        let q_size = num_heads_per_gpu * head_dim;
        let kv_size = num_kv_heads_per_gpu * head_dim;

        // Merged QKV projection: hidden_size -> (q_dim + 2*kv_dim)
        let total_qkv = num_heads * head_dim + 2 * total_num_kv_heads * head_dim;
        let qkv_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            total_qkv,
            step_cfg.attention_bias,
            false, // no gather
            vb.pp("qkv_proj"),
            pg,
        )?;

        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            step_cfg.attention_bias,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        // ALiBi slopes for this device's heads
        let head_start = rank * num_heads_per_gpu;
        let head_end = head_start + num_heads_per_gpu;
        let alibi = AlibiAttentionBias::new_partial(
            num_heads,
            head_start,
            head_end,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            alibi,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
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
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs, tp_ctx)?;
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

        // Sqrt ALiBi bias
        let kv_len = q_len + seqlen_offset;
        let alibi_bias = build_sqrt_alibi_bias(&self.alibi, q_len, kv_len)?;

        // Combine causal mask with sqrt ALiBi bias
        let combined_mask = if let Some(causal_mask) = attention_mask {
            Some(causal_mask.broadcast_add(&alibi_bias)?)
        } else {
            Some(alibi_bias)
        };

        let attn_output = paged_attention(
            &q,
            &k,
            &v,
            combined_mask.as_ref(),
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

        let qkv = self.qkv_proj.forward(xs, tp_ctx)?;
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

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

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

            let attn_output = crate::cuda_kernels::paged_attention_cuda_alibi(
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
                self.alibi.slopes(),
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

                // Build sqrt ALiBi bias for this sequence's decode step
                let kv_len = seq.seqlen_offset + 1;
                let alibi_bias = build_sqrt_alibi_bias(&self.alibi, 1, kv_len)?;

                let attn_out = paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    Some(&alibi_bias),
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

// ─── MLP ────────────────────────────────────────────────────────────────────

struct Step1MLP {
    gate_up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Step1MLP {
    fn new_with_tp(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // Merged gate+up projection: hidden_size -> 2*intermediate_size
        let gate_up_proj = TpLinear::column_parallel(
            hidden_size,
            2 * intermediate_size,
            bias,
            false,
            vb.pp("gate_up_proj"),
            pg,
        )?;

        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            bias,
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
        // SiLU-and-Mul: split into gate and up, then silu(gate) * up
        let chunks = gate_up.chunk(2, gate_up.rank() - 1)?;
        let gate = candle_nn::ops::silu(&chunks[0])?;
        let intermediate = gate.mul(&chunks[1])?;
        self.down_proj.forward(&intermediate, tp_ctx)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Step1DecoderLayer {
    self_attn: Step1Attention,
    mlp: Step1MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Step1DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        step_cfg: &Step1Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Step1Attention::new_with_tp(cfg, step_cfg, vb.pp("self_attn"), pg)?;

        let mlp = Step1MLP::new_with_tp(
            cfg.hidden_size,
            cfg.intermediate_size,
            step_cfg.mlp_bias,
            vb.pp("mlp"),
            pg,
        )?;

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, tp_ctx)?;
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Step1 model for causal language modeling.
///
/// Uses ALiBi with sqrt distance scaling instead of RoPE for position encoding.
pub struct Step1ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Step1DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Step1ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let step_cfg = Step1Config::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Step1DecoderLayer::new_with_tp(
                cfg,
                &step_cfg,
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
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Step1ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Step1ForCausalLM::forward(
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
        extra.insert("num_attention_groups".to_string(), serde_json::json!(2));

        ModelConfig {
            architectures: vec!["Step1ForCausalLM".to_string()],
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
        // Step1 uses num_attention_groups for KV heads
        let num_kv_heads = cfg
            .extra
            .get("num_attention_groups")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.num_key_value_heads);

        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ─── Sqrt ALiBi Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_sqrt_alibi_bias_shape() {
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 4;
        let kv_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device).expect("create alibi");
        let bias = build_sqrt_alibi_bias(&alibi, seq_len, kv_len).expect("build bias");

        assert_eq!(bias.dims(), &[1, num_heads, seq_len, kv_len]);
    }

    #[test]
    fn test_sqrt_alibi_bias_diagonal_zero() {
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device).expect("create alibi");
        let bias = build_sqrt_alibi_bias(&alibi, seq_len, seq_len).expect("build bias");

        let data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        for h in 0..num_heads {
            for i in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + i;
                assert!(
                    data[idx].abs() < 1e-6,
                    "Diagonal should be zero at head {h}, pos ({i},{i}), got {}",
                    data[idx]
                );
            }
        }
    }

    #[test]
    fn test_sqrt_alibi_bias_past_positions_negative() {
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device).expect("create alibi");
        let bias = build_sqrt_alibi_bias(&alibi, seq_len, seq_len).expect("build bias");

        let data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..i {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    assert!(
                        data[idx] < 0.0,
                        "Past positions should have negative bias: head {h}, ({i},{j}), got {}",
                        data[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sqrt_alibi_bias_sqrt_distance() {
        let device = Device::Cpu;
        let num_heads = 8;
        let seq_len = 4;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device).expect("create alibi");
        let slopes: Vec<f32> = alibi.slopes().to_vec1().unwrap();

        let bias = build_sqrt_alibi_bias(&alibi, seq_len, seq_len).expect("build bias");
        let data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        // Verify: bias[h, i, j] = slope[h] * (-sqrt(i - j)) for j < i
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    let distance = j as i64 - i as i64;
                    let expected = if distance <= 0 {
                        slopes[h] * -((-distance) as f32).sqrt()
                    } else {
                        0.0
                    };
                    assert!(
                        (data[idx] - expected).abs() < 1e-5,
                        "head {h}, ({i},{j}): expected {expected}, got {}",
                        data[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sqrt_alibi_bias_with_cache() {
        let device = Device::Cpu;
        let num_heads = 4;
        let seq_len = 1; // decode
        let kv_len = 10;

        let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device).expect("create alibi");
        let bias = build_sqrt_alibi_bias(&alibi, seq_len, kv_len).expect("build bias");

        assert_eq!(bias.dims(), &[1, num_heads, seq_len, kv_len]);

        let slopes: Vec<f32> = alibi.slopes().to_vec1().unwrap();
        let data: Vec<f32> = bias.flatten_all().unwrap().to_vec1().unwrap();

        // Query at position kv_len - 1 = 9
        for h in 0..num_heads {
            for j in 0..kv_len {
                let idx = h * kv_len + j;
                let distance = j as i64 - 9;
                let expected = if distance <= 0 {
                    slopes[h] * -((-distance) as f32).sqrt()
                } else {
                    0.0
                };
                assert!(
                    (data[idx] - expected).abs() < 1e-5,
                    "head {h}, key {j}: expected {expected}, got {}",
                    data[idx]
                );
            }
        }
    }

    // ─── Config Tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_step1_config_defaults() {
        let cfg = ModelConfig::default();
        let step_cfg = Step1Config::from_model_config(&cfg);
        assert!(!step_cfg.attention_bias);
        assert!(!step_cfg.mlp_bias);
    }

    #[test]
    fn test_step1_config_custom() {
        let mut cfg = test_config();
        cfg.attention_bias = Some(true);
        cfg.extra
            .insert("mlp_bias".to_string(), serde_json::json!(true));
        let step_cfg = Step1Config::from_model_config(&cfg);
        assert!(step_cfg.attention_bias);
        assert!(step_cfg.mlp_bias);
    }

    // ─── Construction Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_step1_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Step1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Step1ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_step1_construction_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Step1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Step1 with untied embeddings should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_step1_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = Step1ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Step1 with TP should construct: {:?}",
            model.err()
        );
    }

    // ─── Forward Tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_step1_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_step1_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_step1_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_step1_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step1ForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_step1_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Step1ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_step1_no_attention_groups_fallback() {
        // When num_attention_groups is missing, fall back to num_key_value_heads
        let mut cfg = test_config();
        cfg.extra.remove("num_attention_groups");

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Step1ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Should construct without num_attention_groups: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_step1_mlp_forward() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let mlp = Step1MLP::new_with_tp(64, 128, false, vb.pp("mlp"), &pg).unwrap();

        let input = Tensor::zeros((2, 64), DType::F32, &device).expect("input");
        let output = mlp.forward(&input, &tp_ctx);
        assert!(output.is_ok(), "MLP forward: {:?}", output.err());
        assert_eq!(output.unwrap().dims(), &[2, 64]);
    }
}
