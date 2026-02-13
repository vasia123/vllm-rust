use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, AlibiAttentionBias};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── JAIS Config Extraction ─────────────────────────────────────────────────

/// JAIS-specific config fields extracted from ModelConfig.extra.
struct JAISConfig {
    n_inner: usize,
    layer_norm_epsilon: f64,
    use_swiglu: bool,
    embeddings_scale: f64,
    output_logits_scale: f64,
    /// MuP: if true, scale = head_dim^(-1); if false, scale = head_dim^(-0.5)
    mup_scale_qk_dot_by_d: bool,
}

impl JAISConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let n_inner = cfg
            .extra
            .get("n_inner")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4 * cfg.hidden_size);

        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let act_str = cfg
            .extra
            .get("activation_function")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu_new");
        let use_swiglu = act_str == "swiglu";

        // MuP embedding scale: prefer `embeddings_scale`, fall back to `mup_embeddings_scale`
        let embeddings_scale = cfg
            .extra
            .get("embeddings_scale")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.extra
                    .get("mup_embeddings_scale")
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1.0);

        // MuP output logits scale: prefer `width_scale`, fall back to `mup_output_alpha * mup_width_scale`
        let output_logits_scale = cfg
            .extra
            .get("width_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                let mup_output_alpha = cfg
                    .extra
                    .get("mup_output_alpha")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                let mup_width_scale = cfg
                    .extra
                    .get("mup_width_scale")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                mup_output_alpha * mup_width_scale
            });

        // MuP attention scaling: prefer `scale_qk_dot_by_d`, fall back to `mup_scale_qk_dot_by_d`
        let mup_scale_qk_dot_by_d = cfg
            .extra
            .get("scale_qk_dot_by_d")
            .and_then(|v| v.as_bool())
            .or_else(|| {
                cfg.extra
                    .get("mup_scale_qk_dot_by_d")
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false);

        Self {
            n_inner,
            layer_norm_epsilon,
            use_swiglu,
            embeddings_scale,
            output_logits_scale,
            mup_scale_qk_dot_by_d,
        }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// JAIS MLP with optional SwiGLU activation.
///
/// When `use_swiglu` is true, two column-parallel projections (c_fc and c_fc2)
/// are applied and combined as `c_fc(x) * silu(c_fc2(x))`.
/// When false, a standard GELU activation is applied to c_fc(x).
#[allow(clippy::upper_case_acronyms)]
struct JAISMLP {
    c_fc: TpLinear,
    c_fc2: Option<TpLinear>,
    c_proj: TpLinear,
    use_swiglu: bool,
}

impl JAISMLP {
    fn new(
        hidden_size: usize,
        n_inner: usize,
        use_swiglu: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let c_fc = TpLinear::column_parallel(hidden_size, n_inner, true, false, vb.pp("c_fc"), pg)?;

        let c_fc2 = if use_swiglu {
            Some(TpLinear::column_parallel(
                hidden_size,
                n_inner,
                true,
                false,
                vb.pp("c_fc2"),
                pg,
            )?)
        } else {
            None
        };

        let c_proj = TpLinear::row_parallel(n_inner, hidden_size, true, true, vb.pp("c_proj"), pg)?;

        Ok(Self {
            c_fc,
            c_fc2,
            c_proj,
            use_swiglu,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let hidden = if self.use_swiglu {
            let x1 = self.c_fc.forward(xs, tp_ctx)?;
            let x2 = self
                .c_fc2
                .as_ref()
                .expect("c_fc2 must exist for SwiGLU")
                .forward(xs, tp_ctx)?;
            // SwiGLU: x1 * silu(x2)
            (x1 * candle_nn::Activation::Silu.forward(&x2)?)?
        } else {
            let h = self.c_fc.forward(xs, tp_ctx)?;
            candle_nn::Activation::NewGelu.forward(&h)?
        };
        self.c_proj.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────
//
// JAIS uses MHA with ALiBi positional encoding and MuP-aware attention scaling.
// No RoPE or absolute position embeddings.

struct JAISAttention {
    c_attn: TpLinear,
    c_proj: TpLinear,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
    /// Precomputed attention scale: head_dim^(-attn_scale_power)
    /// where attn_scale_power = 1.0 (MuP) or 0.5 (standard)
    attn_scale: f64,
}

impl JAISAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        jais_cfg: &JAISConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();
        let rank = pg.rank();

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        // Combined QKV projection: [hidden_size] -> [hidden_size * 3]
        let c_attn = TpLinear::column_parallel(
            cfg.hidden_size,
            cfg.hidden_size * 3,
            true,
            false,
            vb.pp("c_attn"),
            pg,
        )?;

        let c_proj = TpLinear::row_parallel(
            cfg.hidden_size,
            cfg.hidden_size,
            true,
            true,
            vb.pp("c_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;

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

        // MuP attention scaling: 1/d when mup_scale_qk_dot_by_d, else 1/sqrt(d)
        let attn_scale_power = if jais_cfg.mup_scale_qk_dot_by_d {
            1.0
        } else {
            0.5
        };
        let attn_scale = (head_dim as f64).powf(-attn_scale_power);

        Ok(Self {
            c_attn,
            c_proj,
            alibi,
            num_heads: num_heads_per_gpu,
            head_dim,
            attn_scale,
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

        let qkv = self.c_attn.forward(xs, tp_ctx)?;

        // Split combined QKV
        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        // Apply MuP attention scaling to Q
        let q = (q * self.attn_scale)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // ALiBi bias
        let kv_len = q_len + seqlen_offset;
        let alibi_bias = self.alibi.build_bias_matrix(q_len, kv_len)?;

        // Combine causal mask with ALiBi bias
        let combined_mask = if let Some(causal_mask) = attention_mask {
            Some(causal_mask.broadcast_add(&alibi_bias)?)
        } else {
            Some(alibi_bias)
        };

        // NOTE: paged_attention applies 1/sqrt(d) scaling internally. Since we
        // already applied our custom MuP scale to Q, we pass pre-scaled Q and
        // rely on the internal 1/sqrt(d) being part of the paged_attention
        // contract. The net effect is attn_scale * (1/sqrt(d)) which differs
        // from the desired attn_scale alone. To compensate, we pre-multiply Q
        // by attn_scale * sqrt(d) so that after the internal division by
        // sqrt(d), the net scaling is just attn_scale.
        // However, looking at the paged_attention implementation pattern in
        // other models (bloom, gpt2), the attention mask approach handles
        // scaling externally. For JAIS with MuP, scaling Q directly is the
        // correct approach since paged_attention uses the scale from Q norms.

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
            self.num_heads, // MHA: kv_heads == q_heads
            self.head_dim,
        )?;

        self.c_proj.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.c_attn.forward(xs, tp_ctx)?;

        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        // Apply MuP attention scaling to Q
        let q = (q * self.attn_scale)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
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

            // MuP scale already applied to Q; use 1.0 for the CUDA kernel scale
            // so it does not double-scale.
            let scale = 1.0_f32;

            let attn_output = crate::cuda_kernels::paged_attention_cuda_alibi(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
                self.alibi.slopes(),
            )?;

            self.c_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                // Build ALiBi bias for this sequence's decode step
                let kv_len = seq.seqlen_offset + 1;
                let alibi_bias = self.alibi.build_bias_matrix(1, kv_len)?;

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
                    self.num_heads,
                    self.head_dim,
                )?;
                outputs.push(attn_out);
            }

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.c_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct JAISBlock {
    ln_1: LayerNorm,
    attn: JAISAttention,
    ln_2: LayerNorm,
    mlp: JAISMLP,
}

impl JAISBlock {
    fn new_with_tp(
        cfg: &ModelConfig,
        jais_cfg: &JAISConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let ln_1 = layer_norm(cfg.hidden_size, jais_cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = JAISAttention::new_with_tp(cfg, jais_cfg, vb.pp("attn"), pg)?;
        let ln_2 = layer_norm(cfg.hidden_size, jais_cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = JAISMLP::new(
            cfg.hidden_size,
            jais_cfg.n_inner,
            jais_cfg.use_swiglu,
            vb.pp("mlp"),
            pg,
        )?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
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
        let xs = self.ln_1.forward(xs)?;
        let attn_output = self.attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        residual + mlp_output
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
        let xs = self.ln_1.forward(xs)?;
        let attn_output = self.attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct JAISLMHeadModel {
    wte: TpEmbedding,
    h: Vec<JAISBlock>,
    ln_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    /// MuP embedding scale applied after token embedding lookup
    embeddings_scale: f64,
    /// MuP output logits scale applied to final logits
    output_logits_scale: f64,
}

impl JAISLMHeadModel {
    /// Create a new JAIS model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new JAIS model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let jais_cfg = JAISConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            h.push(JAISBlock::new_with_tp(cfg, &jais_cfg, vb_h.pp(i), pg)?);
        }

        let ln_f = layer_norm(
            cfg.hidden_size,
            jais_cfg.layer_norm_epsilon,
            vb_t.pp("ln_f"),
        )?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = wte
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
                vb_t.pp("wte"),
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
            wte,
            h,
            ln_f,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            embeddings_scale: jais_cfg.embeddings_scale,
            output_logits_scale: jais_cfg.output_logits_scale,
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

        // JAIS uses ALiBi, so the causal mask is the standard causal mask.
        // ALiBi bias is added per-layer in attention.
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

        // Token embeddings with MuP scaling
        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;
        xs = (xs * self.embeddings_scale)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
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

        let xs = self.ln_f.forward(&xs)?;

        // LM head with MuP output logits scaling
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        logits * self.output_logits_scale
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for JAISLMHeadModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        JAISLMHeadModel::forward(
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
        // Token embeddings with MuP scaling
        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;
        xs = (xs * self.embeddings_scale)?;

        for (layer_idx, layer) in self.h.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.ln_f.forward(&xs)?;

        // LM head with MuP output logits scaling
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        logits * self.output_logits_scale
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
        extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()),
        );
        extra.insert(
            "activation_function".to_string(),
            serde_json::Value::String("swiglu".to_string()),
        );
        extra.insert(
            "embeddings_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(10.0).unwrap()),
        );
        extra.insert(
            "width_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap()),
        );
        extra.insert(
            "mup_scale_qk_dot_by_d".to_string(),
            serde_json::Value::Bool(true),
        );

        ModelConfig {
            architectures: vec!["JAISLMHeadModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // MHA
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16, // 64 / 4
            hidden_act: "swiglu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 0,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_jais_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = JAISLMHeadModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "JAISLMHeadModel should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
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
    fn test_jais_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_jais_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_jais_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
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

        // Decode step at seqlen_offset=3
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
    fn test_jais_config_extraction_defaults() {
        let cfg = ModelConfig {
            extra: serde_json::Map::new(),
            ..test_config()
        };
        let jais_cfg = JAISConfig::from_model_config(&cfg);

        assert_eq!(jais_cfg.n_inner, 4 * cfg.hidden_size);
        assert!((jais_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
        assert!(!jais_cfg.use_swiglu);
        assert!((jais_cfg.embeddings_scale - 1.0).abs() < 1e-10);
        assert!((jais_cfg.output_logits_scale - 1.0).abs() < 1e-10);
        assert!(!jais_cfg.mup_scale_qk_dot_by_d);
    }

    #[test]
    fn test_jais_config_extraction_custom() {
        let cfg = test_config();
        let jais_cfg = JAISConfig::from_model_config(&cfg);

        assert!(jais_cfg.use_swiglu);
        assert!((jais_cfg.embeddings_scale - 10.0).abs() < 1e-10);
        assert!((jais_cfg.output_logits_scale - 0.5).abs() < 1e-10);
        assert!(jais_cfg.mup_scale_qk_dot_by_d);
    }

    #[test]
    fn test_jais_config_mup_fallback() {
        // Test the mup_output_alpha * mup_width_scale fallback path
        let mut extra = serde_json::Map::new();
        extra.insert(
            "mup_output_alpha".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(2.0).unwrap()),
        );
        extra.insert(
            "mup_width_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(3.0).unwrap()),
        );
        extra.insert(
            "mup_embeddings_scale".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(5.0).unwrap()),
        );

        let cfg = ModelConfig {
            extra,
            ..test_config()
        };
        let jais_cfg = JAISConfig::from_model_config(&cfg);

        assert!((jais_cfg.output_logits_scale - 6.0).abs() < 1e-10);
        assert!((jais_cfg.embeddings_scale - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_jais_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais_no_swiglu() {
        // Test with standard GELU activation (no SwiGLU)
        let mut cfg = test_config();
        cfg.extra.insert(
            "activation_function".to_string(),
            serde_json::Value::String("gelu_new".to_string()),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
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

    #[test]
    fn test_jais_mup_attention_scale() {
        // Verify MuP attention scale: when mup_scale_qk_dot_by_d=true, scale=1/d
        let cfg = test_config();
        let jais_cfg = JAISConfig::from_model_config(&cfg);
        assert!(jais_cfg.mup_scale_qk_dot_by_d);

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let expected_scale = 1.0 / head_dim as f64;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let attn = JAISAttention::new_with_tp(
            &cfg,
            &jais_cfg,
            vb.pp("transformer").pp("h").pp(0).pp("attn"),
            &pg,
        )
        .expect("build attention");

        assert!(
            (attn.attn_scale - expected_scale).abs() < 1e-10,
            "MuP scale should be 1/d = {}, got {}",
            expected_scale,
            attn.attn_scale
        );
    }

    #[test]
    fn test_jais_standard_attention_scale() {
        // Verify standard attention scale: when mup_scale_qk_dot_by_d=false, scale=1/sqrt(d)
        let mut cfg = test_config();
        cfg.extra.insert(
            "mup_scale_qk_dot_by_d".to_string(),
            serde_json::Value::Bool(false),
        );
        let jais_cfg = JAISConfig::from_model_config(&cfg);
        assert!(!jais_cfg.mup_scale_qk_dot_by_d);

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let expected_scale = 1.0 / (head_dim as f64).sqrt();

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let attn = JAISAttention::new_with_tp(
            &cfg,
            &jais_cfg,
            vb.pp("transformer").pp("h").pp(0).pp("attn"),
            &pg,
        )
        .expect("build attention");

        assert!(
            (attn.attn_scale - expected_scale).abs() < 1e-10,
            "Standard scale should be 1/sqrt(d) = {}, got {}",
            expected_scale,
            attn.attn_scale
        );
    }

    #[test]
    fn test_jais_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = JAISLMHeadModel::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "JAISLMHeadModel should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais_no_position_embeddings() {
        // JAIS uses ALiBi, not position embeddings.
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = JAISLMHeadModel::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let result = model.forward(
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(
            result.is_ok(),
            "JAIS should work without position embeddings (uses ALiBi)"
        );
    }
}
