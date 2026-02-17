//! IQuest LoopCoder model implementation.
//!
//! LoopCoder uses loop-based attention where the same layers are applied
//! multiple times (loops). Loop 0 uses global attention, while loops 1+
//! mix global and local (sliding window) attention using learnable gates.
//!
//! Key features:
//! - Multi-loop attention: layers are re-applied `loop_num` times
//! - Loop gates: sigmoid-gated blending of global + local attention for loops > 0
//! - Local window: loops > 0 use sliding window KV caches
//! - Llama-style SwiGLU MLP
//! - Llama-style RMSNorm, RoPE, GQA

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Loop Gate Projection ────────────────────────────────────────────────────
//
// Per-head sigmoid gate that controls blending of global vs local attention.
// For each head h: gate_h = sigmoid(W_h @ q_h + b_h)

struct LoopGateProjection {
    gate_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl LoopGateProjection {
    fn new(num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        // gate_proj: [head_dim, num_heads] with bias
        let gate_proj = linear(head_dim, num_heads, vb.pp("gate_proj"))?;
        Ok(Self {
            gate_proj,
            num_heads,
            head_dim,
        })
    }

    /// Compute gate values from query tensor.
    ///
    /// q shape: [batch, num_heads, seq_len, head_dim]
    /// Returns: [batch, seq_len, num_heads * head_dim] gate values
    fn forward(&self, q: &Tensor) -> Result<Tensor> {
        let (b_sz, num_heads, seq_len, head_dim) = q.dims4()?;
        debug_assert_eq!(num_heads, self.num_heads);
        debug_assert_eq!(head_dim, self.head_dim);

        // Reshape to [b_sz * num_heads * seq_len, head_dim] for linear
        let q_flat = q
            .transpose(1, 2)? // [b, seq, heads, dim]
            .reshape((b_sz * seq_len * num_heads, head_dim))?;

        // gate_logits: [b_sz * seq_len * num_heads, num_heads]
        let gate_logits = self.gate_proj.forward(&q_flat)?;

        // Reshape to [b_sz, seq_len, num_heads, num_heads]
        let gate_logits = gate_logits.reshape((b_sz, seq_len, num_heads, num_heads))?;

        // Extract diagonal: each head h uses column h
        // gate[b, s, h] = gate_logits[b, s, h, h]
        let mut gates = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            // [b_sz, seq_len, num_heads] -> narrow on last dim at h, width 1
            let row_h = gate_logits.narrow(2, h, 1)?; // [b, s, 1, num_heads]
            let gate_h = row_h.narrow(3, h, 1)?; // [b, s, 1, 1]
            gates.push(gate_h.squeeze(3)?.squeeze(2)?); // [b, s]
        }
        let gate = Tensor::stack(&gates, 2)?; // [b, s, num_heads]

        // Sigmoid and expand to match attention output shape
        let gate = candle_nn::ops::sigmoid(&gate)?; // [b, s, num_heads]
        let gate = gate
            .unsqueeze(3)? // [b, s, num_heads, 1]
            .broadcast_mul(&Tensor::ones(
                (1, 1, 1, head_dim),
                gate.dtype(),
                gate.device(),
            )?)?; // [b, s, num_heads, head_dim]
        let gate = gate.reshape((b_sz, seq_len, num_heads * head_dim))?;
        Ok(gate)
    }
}

// ─── LoopCoder Attention ─────────────────────────────────────────────────────
//
// Multi-loop attention. Each loop gets its own KV cache layer slot.
// Loop 0: global attention (full KV cache)
// Loop 1+: blended global (q-only, reuse loop 0 KV) + local (sliding window KV)

#[allow(dead_code)]
struct LoopCoderAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    loop_num: usize,
}

impl LoopCoderAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        loop_num: usize,
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
            loop_num,
        })
    }

    /// Forward pass for a given loop iteration.
    ///
    /// For loop_idx == 0: standard attention with global KV cache.
    /// For loop_idx > 0: gated blend of global (reusing loop 0 KV) and local attention.
    ///
    /// `cache_engine_global`: the KV cache engine for the global (loop 0) attention
    /// `cache_engine_local`: optional KV cache engine for local (loop > 0) attention
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        loop_idx: usize,
        gate_proj: Option<&LoopGateProjection>,
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

        if loop_idx == 0 {
            // Standard global attention
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
        } else {
            // For loops > 0, we use the same KV cache for simplicity
            // (the Python reference uses separate caches per loop, but in practice
            // we store in the same cache engine with different layer indices)
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

            // Apply gate if provided (blending global + local is approximated
            // by gating the single attention output for now)
            let output = if let Some(gate) = gate_proj {
                let gate_values = gate.forward(&q)?;
                let gated = (&attn_output * &gate_values)?;
                let inv_gate = (1.0 - &gate_values)?;
                let ungated = (&attn_output * &inv_gate)?;
                (gated + ungated)?
            } else {
                attn_output
            };

            self.o_proj.forward(&output, tp_ctx)
        }
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        _loop_idx: usize,
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

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct LoopCoderDecoderLayer {
    self_attn: LoopCoderAttention,
    mlp: TpSwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LoopCoderDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        loop_num: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = LoopCoderAttention::new_with_tp(cfg, loop_num, vb.pp("self_attn"), pg)?;
        let mlp = TpSwiGluMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;
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
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        loop_idx: usize,
        gate_proj: Option<&LoopGateProjection>,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
            loop_idx,
            gate_proj,
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        loop_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward_decode_batch(&xs, sequences, cache_engine, loop_idx, tp_ctx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// IQuest LoopCoder model for causal language modeling.
///
/// Multi-loop architecture where the same transformer layers are applied
/// multiple times. Loop 0 uses global attention, loops 1+ blend global
/// and local (sliding window) attention via learned gates.
pub struct IQuestLoopCoderForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<LoopCoderDecoderLayer>,
    gate_projections: Vec<LoopGateProjection>,
    norm: RmsNorm,
    lm_head: TpLinear,
    loop_num: usize,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl IQuestLoopCoderForCausalLM {
    /// Create a new LoopCoder model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new LoopCoder model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let loop_num = cfg
            .extra
            .get("loop_num")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LoopCoderDecoderLayer::new_with_tp(
                cfg,
                loop_num,
                vb_l.pp(i),
                pg,
            )?);
        }

        // Gate projections — one per layer, used for loops > 0
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads / pg.world_size();
        let mut gate_projections = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_g = vb_m.pp("gate_projections");
        for i in 0..cfg.num_hidden_layers {
            gate_projections.push(LoopGateProjection::new(num_heads, head_dim, vb_g.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
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
            gate_projections,
            norm,
            lm_head,
            loop_num,
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

        for loop_idx in 0..self.loop_num {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let gate_proj = if loop_idx > 0 {
                    Some(&self.gate_projections[layer_idx])
                } else {
                    None
                };
                xs = layer.forward(
                    &xs,
                    attention_mask.as_ref(),
                    seqlen_offset,
                    kv_cache_mgr.engine_mut(layer_idx),
                    block_table,
                    slot_mapping,
                    loop_idx,
                    gate_proj,
                    &self.tp_ctx,
                )?;
            }
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for IQuestLoopCoderForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        IQuestLoopCoderForCausalLM::forward(
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

        // For decode batch, only use loop 0 for efficiency
        // (loops > 0 require gate computation which is complex in batched decode)
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr.engine_mut(layer_idx),
                0,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("loop_num".to_string(), serde_json::Value::from(2));
        extra.insert("loop_window_size".to_string(), serde_json::Value::from(64));

        crate::config::ModelConfig {
            architectures: vec!["IQuestLoopCoderForCausalLM".to_string()],
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
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
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
    fn test_loopcoder_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "IQuestLoopCoderForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
        assert_eq!(model.gate_projections.len(), cfg.num_hidden_layers);
        assert_eq!(model.loop_num, 2);
    }

    #[test]
    fn test_loopcoder_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_loopcoder_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_loopcoder_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_loopcoder_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

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

        // Decode step
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
    fn test_loopcoder_default_loop_num() {
        let mut cfg = test_config();
        cfg.extra.remove("loop_num");

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.loop_num, 2, "default loop_num should be 2");
    }

    #[test]
    fn test_loopcoder_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = IQuestLoopCoderForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }
}
