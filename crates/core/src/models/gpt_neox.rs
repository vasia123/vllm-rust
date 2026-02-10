use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── GPT-NeoX Config Extraction ──────────────────────────────────────────────

/// GPT-NeoX-specific config fields extracted from ModelConfig.extra.
struct GptNeoXConfig {
    use_parallel_residual: bool,
    rotary_pct: f64,
    layer_norm_eps: f64,
}

impl GptNeoXConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let use_parallel_residual = cfg
            .extra
            .get("use_parallel_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let rotary_pct = cfg
            .extra
            .get("rotary_pct")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25);

        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        Self {
            use_parallel_residual,
            rotary_pct,
            layer_norm_eps,
        }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct GptNeoXMlp {
    dense_h_to_4h: TpLinear,
    dense_4h_to_h: TpLinear,
}

impl GptNeoXMlp {
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

struct GptNeoXAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    dense: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl GptNeoXAttention {
    fn new_with_tp(
        cfg: &ModelConfig,
        neox_cfg: &GptNeoXConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            true,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            true,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            true,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let dense = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            true,
            true,
            vb.pp("dense"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;

        // GPT-NeoX uses partial RoPE: only rotary_pct of head_dim is rotated
        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            neox_cfg.rotary_pct,
            true, // neox-style interleaved rotation
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
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

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Partial RoPE
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

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

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
                self.num_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
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

struct GptNeoXDecoderLayer {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GptNeoXAttention,
    mlp: GptNeoXMlp,
    use_parallel_residual: bool,
}

impl GptNeoXDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        neox_cfg: &GptNeoXConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let attention = GptNeoXAttention::new_with_tp(cfg, neox_cfg, vb.pp("attention"), pg)?;
        let mlp = GptNeoXMlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"), pg)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attention,
            mlp,
            use_parallel_residual: neox_cfg.use_parallel_residual,
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
        let ln_out = self.input_layernorm.forward(xs)?;
        let attn_output = self.attention.forward(
            &ln_out,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;

        if self.use_parallel_residual {
            // Parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
            let ln2_out = self.post_attention_layernorm.forward(xs)?;
            let mlp_output = self.mlp.forward(&ln2_out, tp_ctx)?;
            let result = (xs + &attn_output)?;
            &result + mlp_output
        } else {
            // Sequential residual: standard pre-norm
            let xs = (xs + attn_output)?;
            let ln2_out = self.post_attention_layernorm.forward(&xs)?;
            let mlp_output = self.mlp.forward(&ln2_out, tp_ctx)?;
            &xs + mlp_output
        }
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let ln_out = self.input_layernorm.forward(xs)?;
        let attn_output = self.attention.forward_decode_batch(
            &ln_out,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;

        if self.use_parallel_residual {
            let ln2_out = self.post_attention_layernorm.forward(xs)?;
            let mlp_output = self.mlp.forward(&ln2_out, tp_ctx)?;
            let result = (xs + &attn_output)?;
            &result + mlp_output
        } else {
            let xs = (xs + attn_output)?;
            let ln2_out = self.post_attention_layernorm.forward(&xs)?;
            let mlp_output = self.mlp.forward(&ln2_out, tp_ctx)?;
            &xs + mlp_output
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct GPTNeoXForCausalLM {
    embed_in: TpEmbedding,
    layers: Vec<GptNeoXDecoderLayer>,
    final_layer_norm: LayerNorm,
    embed_out: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl GPTNeoXForCausalLM {
    /// Create a new GPT-NeoX model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new GPT-NeoX model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let neox_cfg = GptNeoXConfig::from_model_config(cfg);
        let vb_m = vb.pp("gpt_neox");
        let world_size = pg.world_size();

        let embed_in = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_in"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GptNeoXDecoderLayer::new_with_tp(
                cfg,
                &neox_cfg,
                vb_l.pp(i),
                pg,
            )?);
        }

        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            neox_cfg.layer_norm_eps,
            vb_m.pp("final_layer_norm"),
        )?;

        let embed_out = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_in
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
                vb_m.pp("embed_in"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("embed_out"),
                pg,
            )?
        };

        Ok(Self {
            embed_in,
            layers,
            final_layer_norm,
            embed_out,
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

        let mut xs = self.embed_in.forward(input_ids, &self.tp_ctx)?;
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
        let xs = self.final_layer_norm.forward(&xs)?;
        self.embed_out.forward(&xs, &self.tp_ctx)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for GPTNeoXForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        GPTNeoXForCausalLM::forward(
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
        let mut xs = self.embed_in.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.final_layer_norm.forward(&xs)?;
        self.embed_out.forward(&xs, &self.tp_ctx)
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
            "use_parallel_residual".to_string(),
            serde_json::Value::Bool(true),
        );
        extra.insert(
            "rotary_pct".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.25).unwrap()),
        );
        extra.insert(
            "layer_norm_eps".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-5).unwrap()),
        );

        ModelConfig {
            architectures: vec!["GPTNeoXForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // MHA
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
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
    fn test_gpt_neox_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GPTNeoXForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GPTNeoXForCausalLM should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_gpt_neox_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gpt_neox_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gpt_neox_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_gpt_neox_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gpt_neox_parallel_residual_flag() {
        let cfg = test_config();
        let neox_cfg = GptNeoXConfig::from_model_config(&cfg);
        assert!(
            neox_cfg.use_parallel_residual,
            "default config should use parallel residual"
        );
    }

    #[test]
    fn test_gpt_neox_sequential_residual() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "use_parallel_residual".to_string(),
            serde_json::Value::Bool(false),
        );

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_gpt_neox_config_extraction_defaults() {
        let cfg = ModelConfig {
            extra: serde_json::Map::new(),
            ..test_config()
        };
        let neox_cfg = GptNeoXConfig::from_model_config(&cfg);

        assert!(neox_cfg.use_parallel_residual);
        assert!((neox_cfg.rotary_pct - 0.25).abs() < 1e-10);
        assert!((neox_cfg.layer_norm_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_gpt_neox_config_extraction_custom() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "use_parallel_residual".to_string(),
            serde_json::Value::Bool(false),
        );
        cfg.extra.insert(
            "rotary_pct".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(0.5).unwrap()),
        );
        cfg.extra.insert(
            "layer_norm_eps".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-6).unwrap()),
        );

        let neox_cfg = GptNeoXConfig::from_model_config(&cfg);

        assert!(!neox_cfg.use_parallel_residual);
        assert!((neox_cfg.rotary_pct - 0.5).abs() < 1e-10);
        assert!((neox_cfg.layer_norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_gpt_neox_tied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = true;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTNeoXForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }
}
