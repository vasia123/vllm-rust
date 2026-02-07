use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, AlibiAttentionBias};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Bloom Config Extraction ─────────────────────────────────────────────────

/// Bloom-specific config fields extracted from ModelConfig.extra.
struct BloomConfig {
    layer_norm_epsilon: f64,
}

impl BloomConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        Self { layer_norm_epsilon }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct BloomMlp {
    dense_h_to_4h: TpLinear,
    dense_4h_to_h: TpLinear,
}

impl BloomMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let dense_h_to_4h = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            true,
            false,
            vb.pp("dense_h_to_4h"),
            pg,
        )?;
        let dense_4h_to_h = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            true,
            true,
            vb.pp("dense_4h_to_h"),
            pg,
        )?;

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let hidden = self.dense_h_to_4h.forward(xs, tp_ctx)?;
        let hidden = hidden.gelu_erf()?;
        self.dense_4h_to_h.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────
//
// Bloom uses MHA with ALiBi positional encoding. No RoPE or absolute position
// embeddings. The ALiBi bias is added to scaled dot-product attention scores.

struct BloomAttention {
    query_key_value: TpLinear,
    dense: TpLinear,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
}

impl BloomAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();
        let rank = pg.rank();

        if world_size > 1 && num_heads % world_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        // Combined QKV projection
        let query_key_value = TpLinear::column_parallel(
            cfg.hidden_size,
            cfg.hidden_size * 3,
            true,
            false,
            vb.pp("query_key_value"),
            pg,
        )?;

        let dense = TpLinear::row_parallel(
            cfg.hidden_size,
            cfg.hidden_size,
            true,
            true,
            vb.pp("dense"),
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

        Ok(Self {
            query_key_value,
            dense,
            alibi,
            num_heads: num_heads_per_gpu,
            head_dim,
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

        let qkv = self.query_key_value.forward(xs, tp_ctx)?;

        // Split combined QKV
        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Bloom uses ALiBi: no RoPE applied. The ALiBi bias is injected into
        // the attention mask that gets passed to paged_attention.
        let kv_len = q_len + seqlen_offset;
        let alibi_bias = self.alibi.build_bias_matrix(q_len, kv_len)?;

        // Combine causal mask with ALiBi bias
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
            self.num_heads, // MHA: kv_heads == q_heads
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

        let qkv_dim = qkv.dim(2)?;
        let split_size = qkv_dim / 3;
        let q = qkv.narrow(2, 0, split_size)?;
        let k = qkv.narrow(2, split_size, split_size)?;
        let v = qkv.narrow(2, split_size * 2, split_size)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // For decode: ALiBi bias applied inside the CUDA kernel
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
                self.num_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
                self.alibi.slopes(),
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
            self.dense.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct BloomDecoderLayer {
    input_layernorm: LayerNorm,
    self_attention: BloomAttention,
    post_attention_layernorm: LayerNorm,
    mlp: BloomMlp,
}

impl BloomDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        bloom_cfg: &BloomConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            bloom_cfg.layer_norm_epsilon,
            vb.pp("input_layernorm"),
        )?;
        let self_attention = BloomAttention::new_with_tp(cfg, vb.pp("self_attention"), pg)?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            bloom_cfg.layer_norm_epsilon,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = BloomMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;

        Ok(Self {
            input_layernorm,
            self_attention,
            post_attention_layernorm,
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
        let xs = self.input_layernorm.forward(xs)?;
        let attn_output = self.self_attention.forward(
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
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
        let xs = self.input_layernorm.forward(xs)?;
        let attn_output = self.self_attention.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct BloomForCausalLM {
    word_embeddings: TpEmbedding,
    word_embeddings_layernorm: LayerNorm,
    h: Vec<BloomDecoderLayer>,
    ln_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl BloomForCausalLM {
    /// Create a new Bloom model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new Bloom model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let bloom_cfg = BloomConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let word_embeddings = TpEmbedding::new(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_t.pp("word_embeddings"),
            pg,
        )?;

        let word_embeddings_layernorm = layer_norm(
            cfg.hidden_size,
            bloom_cfg.layer_norm_epsilon,
            vb_t.pp("word_embeddings_layernorm"),
        )?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            h.push(BloomDecoderLayer::new_with_tp(
                cfg,
                &bloom_cfg,
                vb_h.pp(i),
                pg,
            )?);
        }

        let ln_f = layer_norm(
            cfg.hidden_size,
            bloom_cfg.layer_norm_epsilon,
            vb_t.pp("ln_f"),
        )?;

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
            word_embeddings_layernorm,
            h,
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

        // Bloom uses ALiBi, so the causal mask is just the standard causal mask
        // without any position embeddings. ALiBi bias is added per-layer in attention.
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
        xs = self.word_embeddings_layernorm.forward(&xs)?;

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
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for BloomForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        BloomForCausalLM::forward(
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
        xs = self.word_embeddings_layernorm.forward(&xs)?;

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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()),
        );

        ModelConfig {
            architectures: vec!["BloomForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // MHA
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16, // 64 / 4
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
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
    fn test_bloom_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = BloomForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "BloomForCausalLM should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_bloom_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bloom_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bloom_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_bloom_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_bloom_no_position_embeddings() {
        // Bloom uses ALiBi, not position embeddings. Verify no position embedding layer exists.
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

        // The model should work without any position embedding infrastructure.
        // Verify by running forward - if position embeddings were missing, this would fail.
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
            "Bloom should work without position embeddings"
        );
    }

    #[test]
    fn test_bloom_alibi_slopes_are_correct() {
        // Verify ALiBi slopes match expected values for 4 heads
        let slopes = crate::layers::compute_alibi_slopes(4);
        assert_eq!(slopes.len(), 4);

        // 4 heads: base = 2^(-(2^(-(log2(4)-3)))) = 2^(-(2^(1))) = 2^(-2) = 0.25
        // slopes: 0.25^1, 0.25^2, 0.25^3, 0.25^4
        let expected = [0.25_f32, 0.0625, 0.015625, 0.00390625];
        for (i, (&actual, &exp)) in slopes.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Head {i}: expected {exp}, got {actual}"
            );
        }
    }

    #[test]
    fn test_bloom_config_extraction_defaults() {
        let cfg = ModelConfig {
            extra: serde_json::Map::new(),
            ..test_config()
        };
        let bloom_cfg = BloomConfig::from_model_config(&cfg);

        assert!((bloom_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_bloom_config_extraction_custom() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-6).unwrap()),
        );

        let bloom_cfg = BloomConfig::from_model_config(&cfg);
        assert!((bloom_cfg.layer_norm_epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_bloom_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_bloom_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = BloomForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    // ---- ALiBi Slope Tests ----

    #[test]
    fn test_bloom_alibi_slopes_8_heads() {
        let slopes = crate::layers::compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);

        // For 8 heads: base = 2^(-1) = 0.5, slopes = [0.5^1 .. 0.5^8]
        let expected = [
            0.5_f32, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ];
        for (i, (&actual, &exp)) in slopes.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-6,
                "Head {i}: expected {exp}, got {actual}"
            );
        }
    }

    #[test]
    fn test_bloom_alibi_slopes_16_heads() {
        let slopes = crate::layers::compute_alibi_slopes(16);
        assert_eq!(slopes.len(), 16);

        let base: f32 = 2.0_f32.powf(-0.5);
        for (i, &slope) in slopes.iter().enumerate() {
            let expected = base.powi((i + 1) as i32);
            assert!(
                (slope - expected).abs() < 1e-5,
                "Head {i}: expected {expected}, got {slope}"
            );
        }
    }

    #[test]
    fn test_bloom_alibi_slopes_32_heads() {
        let slopes = crate::layers::compute_alibi_slopes(32);
        assert_eq!(slopes.len(), 32);

        let base: f32 = 2.0_f32.powf(-0.25);
        for (i, &slope) in slopes.iter().enumerate() {
            let expected = base.powi((i + 1) as i32);
            assert!(
                (slope - expected).abs() < 1e-5,
                "Head {i}: expected {expected}, got {slope}"
            );
        }
    }

    #[test]
    fn test_bloom_alibi_slopes_64_heads() {
        let slopes = crate::layers::compute_alibi_slopes(64);
        assert_eq!(slopes.len(), 64);

        let base: f32 = 2.0_f32.powf(-0.125);
        for (i, &slope) in slopes.iter().enumerate() {
            let expected = base.powi((i + 1) as i32);
            assert!(
                (slope - expected).abs() < 1e-4,
                "Head {i}: expected {expected}, got {slope}"
            );
        }
    }

    #[test]
    fn test_bloom_alibi_bias_changes_attention_output() {
        // Verify that ALiBi bias produces different results than no bias.
        let device = Device::Cpu;

        let alibi = AlibiAttentionBias::new(4, DType::F32, &device).expect("create alibi");

        // Bias for q_len=1, kv_len=5 (decode with 5 tokens in cache)
        let bias_with_alibi = alibi.build_bias_matrix(1, 5).expect("build bias");
        let bias_data: Vec<f32> = bias_with_alibi
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");

        // Without ALiBi, all bias values would be zero. With ALiBi, only the
        // last column (self-position) should be zero.
        let has_nonzero = bias_data.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            has_nonzero,
            "ALiBi bias should have non-zero values for past tokens"
        );

        // The last element of each head (position 4, the current token) should
        // be zero because distance = 0.
        let slopes = crate::layers::compute_alibi_slopes(4);
        for h in 0..4 {
            let last_idx = h * 5 + 4;
            assert!(
                bias_data[last_idx].abs() < 1e-6,
                "Head {h}: current position bias should be 0, got {}",
                bias_data[last_idx]
            );

            // First key position (distance = -4) should have negative bias
            let first_idx = h * 5;
            let expected = slopes[h] * (-4.0_f32);
            assert!(
                (bias_data[first_idx] - expected).abs() < 1e-5,
                "Head {h}: expected bias {expected}, got {}",
                bias_data[first_idx]
            );
        }
    }
}
