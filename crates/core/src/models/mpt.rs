//! MPT (MosaicML Pretrained Transformer) model implementation.
//!
//! Key architectural features:
//! - ALiBi attention (no RoPE, no positional embeddings)
//! - LayerNorm (some variants use low_precision_layernorm)
//! - GELU activation (new GELU approximation)
//! - No bias in attention or MLP projections
//! - Tied word embeddings (lm_head uses embedding weights)
//! - Fused QKV projection (attn.Wqkv)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, AlibiAttentionBias};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── MPT Config Extraction ──────────────────────────────────────────────────

struct MptConfig {
    layer_norm_epsilon: f64,
}

impl MptConfig {
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
//
// MPT uses a simple two-layer MLP with GELU activation:
//   up_proj (hidden -> intermediate), GELU, down_proj (intermediate -> hidden)

struct MptMlp {
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl MptMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let up_proj = TpLinear::column_parallel(
            hidden_size,
            intermediate_size,
            false, // no bias
            false,
            vb.pp("up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            intermediate_size,
            hidden_size,
            false, // no bias
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let hidden = self.up_proj.forward(xs, tp_ctx)?;
        let hidden = hidden.gelu_erf()?;
        self.down_proj.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────
//
// MPT uses MHA with ALiBi positional encoding and a fused QKV projection.
// No RoPE or absolute position embeddings.

struct MptAttention {
    wqkv: TpLinear,
    out_proj: TpLinear,
    alibi: AlibiAttentionBias,
    num_heads: usize,
    head_dim: usize,
}

impl MptAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();
        let rank = pg.rank();

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        // Fused QKV projection (no bias)
        let wqkv = TpLinear::column_parallel(
            cfg.hidden_size,
            cfg.hidden_size * 3, // Q + K + V
            false,
            false,
            vb.pp("Wqkv"),
            pg,
        )?;

        let out_proj = TpLinear::row_parallel(
            cfg.hidden_size,
            cfg.hidden_size,
            false, // no bias
            true,
            vb.pp("out_proj"),
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
            wqkv,
            out_proj,
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

        let qkv = self.wqkv.forward(xs, tp_ctx)?;

        // Split fused QKV
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

        // ALiBi bias (no RoPE)
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

        self.out_proj.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let qkv = self.wqkv.forward(xs, tp_ctx)?;

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

            self.out_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

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
            self.out_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct MptDecoderLayer {
    norm_1: LayerNorm,
    attn: MptAttention,
    norm_2: LayerNorm,
    ffn: MptMlp,
}

impl MptDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        mpt_cfg: &MptConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let norm_1 = layer_norm(cfg.hidden_size, mpt_cfg.layer_norm_epsilon, vb.pp("norm_1"))?;
        let attn = MptAttention::new_with_tp(cfg, vb.pp("attn"), pg)?;
        let norm_2 = layer_norm(cfg.hidden_size, mpt_cfg.layer_norm_epsilon, vb.pp("norm_2"))?;
        let ffn = MptMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("ffn"), pg)?;

        Ok(Self {
            norm_1,
            attn,
            norm_2,
            ffn,
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
        let xs = self.norm_1.forward(xs)?;
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
        let xs = self.norm_2.forward(&xs)?;
        let mlp_output = self.ffn.forward(&xs, tp_ctx)?;
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
        let xs = self.norm_1.forward(xs)?;
        let attn_output = self.attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (attn_output + residual)?;

        let residual = &xs;
        let xs = self.norm_2.forward(&xs)?;
        let mlp_output = self.ffn.forward(&xs, tp_ctx)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct MptForCausalLM {
    wte: TpEmbedding,
    blocks: Vec<MptDecoderLayer>,
    norm_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl MptForCausalLM {
    /// Create a new MPT model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new MPT model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let mpt_cfg = MptConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_t.pp("blocks");
        for i in 0..cfg.num_hidden_layers {
            blocks.push(MptDecoderLayer::new_with_tp(cfg, &mpt_cfg, vb_b.pp(i), pg)?);
        }

        let norm_f = layer_norm(
            cfg.hidden_size,
            mpt_cfg.layer_norm_epsilon,
            vb_t.pp("norm_f"),
        )?;

        // MPT uses tied embeddings by default
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
            blocks,
            norm_f,
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

        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
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

        let xs = self.norm_f.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for MptForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        MptForCausalLM::forward(
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
        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm_f.forward(&xs)?;
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
            architectures: vec!["MptForCausalLM".to_string()],
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
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_mpt_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = MptForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MptForCausalLM should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mpt_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mpt_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mpt_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_mpt_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mpt_no_rope() {
        // MPT uses ALiBi, not RoPE. Verify the model works without any rotary embedding.
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

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
        assert!(result.is_ok(), "MPT should work with ALiBi (no RoPE)");
    }

    #[test]
    fn test_mpt_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mpt_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_mpt_config_extraction() {
        let cfg = test_config();
        let mpt_cfg = MptConfig::from_model_config(&cfg);
        assert!((mpt_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_mpt_config_extraction_defaults() {
        let cfg = ModelConfig {
            extra: serde_json::Map::new(),
            ..test_config()
        };
        let mpt_cfg = MptConfig::from_model_config(&cfg);
        assert!((mpt_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_mpt_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = MptForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_mpt_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = MptForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "MptForCausalLM should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    // ---- ALiBi Slope Tests ----

    #[test]
    fn test_mpt_alibi_slopes_8_heads() {
        let slopes = crate::layers::compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);

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
    fn test_mpt_alibi_slopes_16_heads() {
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
    fn test_mpt_alibi_slopes_32_heads() {
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
    fn test_mpt_alibi_slopes_64_heads() {
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
    fn test_mpt_alibi_bias_changes_attention_output() {
        let device = Device::Cpu;

        let alibi = AlibiAttentionBias::new(4, DType::F32, &device).expect("create alibi");

        // Bias for q_len=1, kv_len=10 (decode with 10 tokens in cache)
        let bias = alibi.build_bias_matrix(1, 10).expect("build bias");
        let bias_data: Vec<f32> = bias
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1");

        // Verify non-zero biases exist (ALiBi is active)
        let has_nonzero = bias_data.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            has_nonzero,
            "ALiBi bias should have non-zero values for past tokens"
        );

        // Current position (position 9) should have zero bias
        let slopes = crate::layers::compute_alibi_slopes(4);
        for h in 0..4 {
            let current_idx = h * 10 + 9;
            assert!(
                bias_data[current_idx].abs() < 1e-6,
                "Head {h}: current position bias should be 0, got {}",
                bias_data[current_idx]
            );

            // First position (distance = -9) should have negative bias
            let first_idx = h * 10;
            let expected = slopes[h] * (-9.0_f32);
            assert!(
                (bias_data[first_idx] - expected).abs() < 1e-5,
                "Head {h}: expected bias {expected}, got {}",
                bias_data[first_idx]
            );
        }
    }
}
