//! Falcon model implementation.
//!
//! Key differences from Llama:
//! - Multi-query attention (MQA) for older versions (num_kv_heads=1)
//! - Grouped-query attention (GQA) for newer versions (Falcon-40B, Falcon-180B)
//! - Parallel attention-MLP (attention + mlp run in parallel)
//! - LayerNorm instead of RMSNorm
//! - RoPE or ALiBi positional embeddings (we support RoPE here)
//! - GELU activation
//!
//! Reference: https://huggingface.co/tiiuae/falcon-7b

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── MLP ──────────────────────────────────────────────────────────────────────

/// Falcon MLP with GELU activation.
struct FalconMlp {
    dense_h_to_4h: TpLinear, // hidden -> 4*hidden
    dense_4h_to_h: TpLinear, // 4*hidden -> hidden
}

impl FalconMlp {
    fn new(hidden_size: usize, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let intermediate_size = 4 * hidden_size;
        // Falcon uses bias in MLP
        let dense_h_to_4h = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            true,  // bias
            false, // no gather
            vb.pp("dense_h_to_4h"),
            pg,
        )?;
        let dense_4h_to_h = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            true, // bias
            true, // input is parallel
            vb.pp("dense_4h_to_h"),
            pg,
        )?;
        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let xs = self.dense_h_to_4h.forward(xs, tp_ctx)?;
        let xs = xs.gelu_erf()?; // Falcon uses GELU
        self.dense_4h_to_h.forward(&xs, tp_ctx)
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

struct FalconAttention {
    query_key_value: TpLinear, // fused QKV projection
    dense: TpLinear,           // output projection
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
}

impl FalconAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        if world_size > 1 {
            if num_heads % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if num_kv_heads % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let qkv_size = q_size + 2 * kv_size;

        // Falcon uses fused QKV with bias - column parallel splits output
        let query_key_value = TpLinear::column_parallel(
            cfg.hidden_size,
            qkv_size,
            true,  // bias
            false, // no gather
            vb.pp("query_key_value"),
            pg,
        )?;
        let dense = TpLinear::row_parallel(
            q_size,
            cfg.hidden_size,
            true, // bias
            true, // input is parallel
            vb.pp("dense"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;
        let q_size_per_gpu = num_heads_per_gpu * head_dim;
        let kv_size_per_gpu = num_kv_heads_per_gpu * head_dim;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            query_key_value,
            dense,
            rotary_emb,
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
            q_size: q_size_per_gpu,
            kv_size: kv_size_per_gpu,
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

        // Fused QKV projection
        let qkv = self.query_key_value.forward(xs, tp_ctx)?;
        let (q, k, v) = self.split_qkv(&qkv)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
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

        self.dense.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.query_key_value.forward(xs, tp_ctx)?;
        let (q, k, v) = self.split_qkv(&qkv)?;

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
            )?;

            self.dense.forward(&attn_output, tp_ctx)?.unsqueeze(1)
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
            self.dense.forward(&attn_output, tp_ctx)
        }
    }

    /// Split fused QKV tensor into Q, K, V.
    fn split_qkv(&self, qkv: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len, _) = qkv.dims3()?;

        // QKV is packed as [Q, K, V] along last dimension
        let q = qkv.narrow(2, 0, self.q_size)?;
        let k = qkv.narrow(2, self.q_size, self.kv_size)?;
        let v = qkv.narrow(2, self.q_size + self.kv_size, self.kv_size)?;

        Ok((
            q.reshape((b_sz, seq_len, self.q_size))?,
            k.reshape((b_sz, seq_len, self.kv_size))?,
            v.reshape((b_sz, seq_len, self.kv_size))?,
        ))
    }
}

// ─── Decoder Layer ────────────────────────────────────────────────────────────

struct FalconDecoderLayer {
    self_attention: FalconAttention,
    mlp: FalconMlp,
    input_layernorm: LayerNorm,
    // Falcon with parallel_attn uses a single layernorm
    // (no post_attention_layernorm)
}

impl FalconDecoderLayer {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let self_attention = FalconAttention::new_with_tp(cfg, vb.pp("self_attention"), pg)?;
        let mlp = FalconMlp::new(cfg.hidden_size, vb.pp("mlp"), pg)?;

        let layer_norm_eps = cfg.rms_norm_eps; // reuse rms_norm_eps for layer_norm_epsilon
        let input_layernorm =
            layer_norm(cfg.hidden_size, layer_norm_eps, vb.pp("input_layernorm"))?;

        Ok(Self {
            self_attention,
            mlp,
            input_layernorm,
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

        // Single layernorm before both attention and MLP (parallel design)
        let xs = self.input_layernorm.forward(xs)?;

        // Attention output
        let attn_output = self.self_attention.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        // MLP output (from same normalized input - parallel attention)
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;

        // Parallel: residual + attention + mlp
        (residual + attn_output + mlp_output)?.contiguous()
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

        let attn_output = self.self_attention.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;

        (residual + attn_output + mlp_output)?.contiguous()
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

pub struct FalconForCausalLM {
    word_embeddings: TpEmbedding,
    layers: Vec<FalconDecoderLayer>,
    ln_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl FalconForCausalLM {
    /// Create a new Falcon model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Falcon model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let word_embeddings = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_t.pp("word_embeddings"),
            pg,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            layers.push(FalconDecoderLayer::new_with_tp(cfg, vb_h.pp(i), pg)?);
        }

        let layer_norm_eps = cfg.rms_norm_eps;
        let ln_f = layer_norm(cfg.hidden_size, layer_norm_eps, vb_t.pp("ln_f"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = word_embeddings
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
                vb_t.pp("word_embeddings"),
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
            word_embeddings,
            layers,
            ln_f,
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

        let mut xs = self.word_embeddings.forward(input_ids, &self.tp_ctx)?;
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
        let xs = self.ln_f.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for FalconForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        FalconForCausalLM::forward(
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
        let mut xs = self.word_embeddings.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.ln_f.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config_mqa() -> crate::config::ModelConfig {
        // Falcon-7B style config with MQA (num_kv_heads=1)
        crate::config::ModelConfig {
            architectures: vec!["FalconForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 1, // MQA
            num_hidden_layers: 2,
            intermediate_size: 256, // 4 * hidden_size
            vocab_size: 256,
            max_position_embeddings: 2048,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5, // layer_norm_epsilon
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
        }
    }

    fn test_config_gqa() -> crate::config::ModelConfig {
        // Falcon-40B style config with GQA
        let mut cfg = test_config_mqa();
        cfg.num_key_value_heads = 2; // GQA
        cfg.tie_word_embeddings = false;
        cfg
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
    fn test_falcon_mqa_construction() {
        let cfg = test_config_mqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = FalconForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "FalconForCausalLM should construct with zero weights"
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_falcon_gqa_construction() {
        let cfg = test_config_gqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = FalconForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "Falcon with GQA should construct");
    }

    #[test]
    fn test_falcon_forward_shape() {
        let cfg = test_config_mqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = FalconForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_falcon_single_token_forward() {
        let cfg = test_config_mqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = FalconForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_falcon_mqa_configuration() {
        let cfg = test_config_mqa();
        assert_eq!(cfg.num_key_value_heads, 1, "MQA uses single KV head");
        assert_eq!(
            cfg.num_attention_heads / cfg.num_key_value_heads,
            4,
            "Each KV head serves 4 query heads"
        );
    }

    #[test]
    fn test_falcon_prefill_then_decode() {
        let cfg = test_config_mqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = FalconForCausalLM::new(&cfg, vb).expect("build model");

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

        // Decode step with seqlen_offset=3
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
    fn test_falcon_tp_construction() {
        let cfg = test_config_gqa();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = FalconForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "FalconForCausalLM should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }
}
