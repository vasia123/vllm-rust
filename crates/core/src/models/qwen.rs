//! QWen v1 (original QWen) model implementation.
//!
//! QWen v1 differs from Qwen2 in several key ways:
//! - Fused QKV projection via `c_attn` (single linear producing Q+K+V) with bias
//! - MLP uses `w1`/`w2`/`c_proj` naming (not gate_proj/up_proj/down_proj)
//! - MLP intermediate_size is halved (config stores 2x, model uses intermediate_size / 2)
//! - MHA only (num_kv_heads == num_heads, no GQA)
//! - Layer naming: `ln_1`, `ln_2` (not input_layernorm/post_attention_layernorm)
//! - Model prefix: `transformer.wte`, `transformer.h.{i}`, `transformer.ln_f`
//! - Uses `layer_norm_epsilon` (not rms_norm_eps) from config
//! - RoPE applied to Q and K after splitting fused QKV

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── QWen Config Extraction ─────────────────────────────────────────────────

/// QWen v1-specific config fields extracted from ModelConfig.extra.
struct QWenConfig {
    layer_norm_epsilon: f64,
}

impl QWenConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps);

        Self { layer_norm_epsilon }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// QWen v1 MLP: SwiGLU with w1 (gate), w2 (up), c_proj (down).
///
/// forward(x) = c_proj(silu(w1(x)) * w2(x))
///
/// The config's intermediate_size is 2x the actual dimension used,
/// so the MLP uses intermediate_size / 2 for w1 and w2 output.
struct QWenMLP {
    w1: TpLinear,
    w2: TpLinear,
    c_proj: TpLinear,
}

impl QWenMLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        // QWen v1 halves the intermediate_size for the gate/up projections
        let half_intermediate = intermediate_size / 2;

        let w1 = TpLinear::column_parallel(
            hidden_size,
            half_intermediate,
            false, // no bias on w1
            false, // no gather
            vb.pp("w1"),
            pg,
        )?;
        let w2 = TpLinear::column_parallel(
            hidden_size,
            half_intermediate,
            false, // no bias on w2
            false,
            vb.pp("w2"),
            pg,
        )?;
        let c_proj = TpLinear::row_parallel(
            half_intermediate,
            hidden_size,
            false, // no bias on c_proj
            true,  // input is parallel
            vb.pp("c_proj"),
            pg,
        )?;

        Ok(Self { w1, w2, c_proj })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let gate = self.w1.forward(xs, tp_ctx)?;
        let up = self.w2.forward(xs, tp_ctx)?;
        // SwiGLU: silu(gate) * up
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.c_proj.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// QWen v1 attention: fused QKV via `c_attn`, MHA (no GQA), RoPE.
///
/// - c_attn: [hidden_size] -> [3 * hidden_size] with bias
/// - c_proj: [hidden_size] -> [hidden_size] without bias
struct QWenAttention {
    c_attn: TpLinear,
    c_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl QWenAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        // Fused QKV projection: [hidden_size] -> [3 * num_heads * head_dim]
        // QWen v1 uses bias=True on c_attn
        let c_attn = TpLinear::column_parallel(
            cfg.hidden_size,
            3 * num_heads * head_dim,
            true, // bias on fused QKV
            false,
            vb.pp("c_attn"),
            pg,
        )?;

        // Output projection: no bias
        let c_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false, // no bias
            true,  // input is parallel
            vb.pp("c_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            c_attn,
            c_proj,
            rotary_emb,
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

        let qkv = self.c_attn.forward(xs, tp_ctx)?;

        // Split fused QKV into Q, K, V along last dimension
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

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        // Paged attention (MHA: num_kv_heads == num_heads)
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
            self.num_heads, // MHA
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

            // Batched RoPE with per-sequence positions
            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            // Write new K/V to cache
            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            // Build block_tables
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
                self.num_heads, // MHA
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
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
                    self.num_heads, // MHA
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

struct QWenBlock {
    ln_1: RmsNorm,
    attn: QWenAttention,
    ln_2: RmsNorm,
    mlp: QWenMLP,
}

impl QWenBlock {
    fn new_with_tp(
        cfg: &ModelConfig,
        qwen_cfg: &QWenConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let ln_1 = rms_norm(cfg.hidden_size, qwen_cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = QWenAttention::new_with_tp(cfg, vb.pp("attn"), pg)?;
        let ln_2 = rms_norm(cfg.hidden_size, qwen_cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = QWenMLP::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;

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
        let xs = self.attn.forward(
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
        let xs = self.attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.ln_2.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        residual + mlp_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// QWen v1 (original QWen) model for causal language modeling.
///
/// Key architectural differences from Qwen2:
/// - Fused QKV projection (`c_attn`) with bias
/// - MHA only (no GQA)
/// - SwiGLU MLP with `w1`/`w2`/`c_proj` naming
/// - `intermediate_size / 2` for MLP hidden dim
/// - `layer_norm_epsilon` config field (in `extra`)
/// - Weight prefix: `transformer.wte`, `transformer.h`, `transformer.ln_f`
pub struct QWenLMHeadModel {
    wte: TpEmbedding,
    h: Vec<QWenBlock>,
    ln_f: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl QWenLMHeadModel {
    /// Create a new QWen v1 model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new QWen v1 model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let qwen_cfg = QWenConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            h.push(QWenBlock::new_with_tp(cfg, &qwen_cfg, vb_h.pp(i), pg)?);
        }

        let ln_f = rms_norm(
            cfg.hidden_size,
            qwen_cfg.layer_norm_epsilon,
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

impl crate::engine::ModelForward for QWenLMHeadModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        QWenLMHeadModel::forward(
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
        // QWen v1 uses MHA (num_kv_heads == num_attention_heads)
        ModelConfig {
            architectures: vec!["QWenLMHeadModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // MHA
            num_hidden_layers: 2,
            intermediate_size: 256, // model uses 256/2 = 128
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16, // 64 / 4
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 151643,
            eos_token_id: 151643,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
        }
    }

    fn test_config_with_layer_norm_epsilon() -> ModelConfig {
        let mut cfg = test_config();
        cfg.extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-6).unwrap()),
        );
        cfg
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
    fn test_qwen_v1_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = QWenLMHeadModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "QWenLMHeadModel should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.expect("model construction failed");
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen_v1_construction_with_layer_norm_epsilon() {
        let cfg = test_config_with_layer_norm_epsilon();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = QWenLMHeadModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "QWenLMHeadModel should construct with layer_norm_epsilon in extra: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen_v1_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_qwen_v1_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_qwen_v1_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_qwen_v1_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_qwen_v1_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen_v1_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen_v1_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_qwen_v1_mha_configuration() {
        // QWen v1 uses multi-head attention (no GQA)
        let cfg = test_config();
        assert_eq!(
            cfg.num_attention_heads, cfg.num_key_value_heads,
            "QWen v1 uses MHA: num_kv_heads == num_heads"
        );
    }

    #[test]
    fn test_qwen_v1_intermediate_size_halved() {
        // QWen v1 config stores 2x the actual MLP intermediate dimension.
        // The model should work correctly with the halved value.
        let cfg = test_config();
        assert_eq!(cfg.intermediate_size, 256);
        // Internal MLP uses 256/2 = 128

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = QWenLMHeadModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Model should construct with intermediate_size correctly halved"
        );
    }

    #[test]
    fn test_qwen_v1_config_extraction_defaults() {
        // When layer_norm_epsilon is not in extra, falls back to rms_norm_eps
        let cfg = test_config();
        let qwen_cfg = QWenConfig::from_model_config(&cfg);
        assert!(
            (qwen_cfg.layer_norm_epsilon - cfg.rms_norm_eps).abs() < 1e-10,
            "should fall back to rms_norm_eps when layer_norm_epsilon not in extra"
        );
    }

    #[test]
    fn test_qwen_v1_config_extraction_custom() {
        let cfg = test_config_with_layer_norm_epsilon();
        let qwen_cfg = QWenConfig::from_model_config(&cfg);
        assert!(
            (qwen_cfg.layer_norm_epsilon - 1e-6).abs() < 1e-10,
            "should use layer_norm_epsilon from extra when present"
        );
    }

    #[test]
    fn test_qwen_v1_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = QWenLMHeadModel::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "QWenLMHeadModel should construct with TP: {:?}",
            model.err()
        );

        let model = model.expect("model construction failed");
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_qwen_v1_tp_heads_divisibility_check() {
        let mut cfg = test_config();
        cfg.num_attention_heads = 3; // Not divisible by 2
        cfg.num_key_value_heads = 3;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = QWenLMHeadModel::new_with_tp(&cfg, vb, &pg, tp_ctx);

        match result {
            Ok(_) => panic!("Should fail when num_heads is not divisible by world_size"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("divisible"),
                    "Error should mention divisibility: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn test_qwen_v1_differs_from_qwen2() {
        // Document key architectural differences between QWen v1 and Qwen2:
        // 1. QWen v1 uses MHA (no GQA)
        // 2. QWen v1 uses fused QKV (c_attn) with bias
        // 3. QWen v1 MLP: w1/w2/c_proj (intermediate_size / 2)
        // 4. QWen v1 norms: ln_1/ln_2 (not input_layernorm/post_attention_layernorm)
        // 5. QWen v1 weight prefix: transformer.wte/h/ln_f (not model.embed_tokens/layers/norm)

        let cfg = test_config();
        assert_eq!(
            cfg.num_attention_heads, cfg.num_key_value_heads,
            "QWen v1 uses MHA"
        );
    }
}
