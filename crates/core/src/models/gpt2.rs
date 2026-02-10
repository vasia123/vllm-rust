use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, Embedding, LayerNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::paged_attention;

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── GPT-2 Config Extraction ─────────────────────────────────────────────────

/// GPT-2-specific config fields extracted from ModelConfig.extra.
struct GPT2Config {
    n_inner: usize,
    layer_norm_epsilon: f64,
    activation: candle_nn::Activation,
}

impl GPT2Config {
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

        let activation = match act_str {
            "gelu_new" => candle_nn::Activation::NewGelu,
            "gelu" => candle_nn::Activation::Gelu,
            "relu" => candle_nn::Activation::Relu,
            "silu" | "swish" => candle_nn::Activation::Silu,
            _ => candle_nn::Activation::NewGelu,
        };

        Self {
            n_inner,
            layer_norm_epsilon,
            activation,
        }
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct GPT2MLP {
    c_fc: TpLinear,
    c_proj: TpLinear,
    activation: candle_nn::Activation,
}

impl GPT2MLP {
    fn new(
        hidden_size: usize,
        n_inner: usize,
        activation: candle_nn::Activation,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let c_fc = TpLinear::column_parallel(
            hidden_size,
            n_inner,
            true, // GPT-2 uses bias
            false,
            vb.pp("c_fc"),
            pg,
        )?;
        let c_proj = TpLinear::row_parallel(n_inner, hidden_size, true, true, vb.pp("c_proj"), pg)?;

        Ok(Self {
            c_fc,
            c_proj,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let hidden = self.c_fc.forward(xs, tp_ctx)?;
        let hidden = self.activation.forward(&hidden)?;
        self.c_proj.forward(&hidden, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct GPT2Attention {
    c_attn: TpLinear,
    c_proj: TpLinear,
    num_heads: usize,
    head_dim: usize,
}

impl GPT2Attention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size / num_heads;
        let world_size = pg.world_size();

        if world_size > 1 && !num_heads.is_multiple_of(world_size) {
            return Err(candle_core::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )));
        }

        // Combined QKV projection: [hidden_size] -> [hidden_size * 3]
        let c_attn = TpLinear::column_parallel(
            cfg.hidden_size,
            cfg.hidden_size * 3,
            true, // GPT-2 uses bias
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

        Ok(Self {
            c_attn,
            c_proj,
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

        // Split combined QKV into Q, K, V along the last dimension
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

        // GPT-2 uses absolute position embeddings (added before layers),
        // so no RoPE application here.

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
            self.num_heads, // GPT-2 uses MHA (kv_heads == q_heads)
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

            self.c_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

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
            self.c_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct GPT2Block {
    ln_1: LayerNorm,
    attn: GPT2Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    fn new_with_tp(
        cfg: &ModelConfig,
        gpt2_cfg: &GPT2Config,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let ln_1 = layer_norm(cfg.hidden_size, gpt2_cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = GPT2Attention::new_with_tp(cfg, vb.pp("attn"), pg)?;
        let ln_2 = layer_norm(cfg.hidden_size, gpt2_cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = GPT2MLP::new(
            cfg.hidden_size,
            gpt2_cfg.n_inner,
            gpt2_cfg.activation,
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

pub struct GPT2LMHeadModel {
    wte: TpEmbedding,
    wpe: Embedding,
    h: Vec<GPT2Block>,
    ln_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl GPT2LMHeadModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let gpt2_cfg = GPT2Config::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        // Position embeddings are not sharded across TP ranks
        let wpe =
            candle_nn::embedding(cfg.max_position_embeddings, cfg.hidden_size, vb_t.pp("wpe"))?;

        let mut h = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_h = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            h.push(GPT2Block::new_with_tp(cfg, &gpt2_cfg, vb_h.pp(i), pg)?);
        }

        let ln_f = layer_norm(
            cfg.hidden_size,
            gpt2_cfg.layer_norm_epsilon,
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
            wpe,
            h,
            ln_f,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Build absolute position IDs tensor for the given sequence parameters.
    fn position_ids(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let positions: Vec<u32> = (0..seq_len).map(|i| (seqlen_offset + i) as u32).collect();
        Tensor::from_vec(positions, (1, seq_len), &self.device)
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

        let token_embeds = self.wte.forward(input_ids, &self.tp_ctx)?;
        let pos_ids = self.position_ids(seq_len, seqlen_offset)?;
        let pos_embeds = self.wpe.forward(&pos_ids)?;

        let mut xs = token_embeds.broadcast_add(&pos_embeds)?;

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

impl crate::engine::ModelForward for GPT2LMHeadModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        GPT2LMHeadModel::forward(
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
        // Build position IDs for the batch (each sequence at its own offset)
        let batch_size = sequences.len();
        let pos_ids_data: Vec<u32> = sequences.iter().map(|s| s.seqlen_offset as u32).collect();
        let pos_ids = Tensor::from_vec(pos_ids_data, (batch_size, 1), &self.device)?;

        let token_embeds = self.wte.forward(input_ids, &self.tp_ctx)?;
        let pos_embeds = self.wpe.forward(&pos_ids)?;
        let mut xs = (token_embeds + pos_embeds)?;

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
        ModelConfig {
            architectures: vec!["GPT2LMHeadModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4, // GPT-2 uses MHA
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16, // 64 / 4
            hidden_act: "gelu_new".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
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
    fn test_gpt2_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GPT2LMHeadModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GPT2LMHeadModel should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_gpt2_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_gpt2_position_embeddings() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

        // Verify position_ids generates correct values
        let pos_ids = model.position_ids(5, 0).expect("position_ids");
        let pos_data: Vec<u32> = pos_ids
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("vec");
        assert_eq!(pos_data, vec![0, 1, 2, 3, 4]);

        // With offset
        let pos_ids = model.position_ids(3, 10).expect("position_ids with offset");
        let pos_data: Vec<u32> = pos_ids
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("vec");
        assert_eq!(pos_data, vec![10, 11, 12]);

        // Single token decode
        let pos_ids = model.position_ids(1, 7).expect("single token position");
        let pos_data: Vec<u32> = pos_ids
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("vec");
        assert_eq!(pos_data, vec![7]);
    }

    #[test]
    fn test_gpt2_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_gpt2_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_gpt2_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

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
    fn test_gpt2_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_gpt2_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPT2LMHeadModel::new(&cfg, vb).expect("build model");

        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_gpt2_config_extraction_defaults() {
        let cfg = test_config();
        let gpt2_cfg = GPT2Config::from_model_config(&cfg);

        assert_eq!(gpt2_cfg.n_inner, 4 * cfg.hidden_size);
        assert!((gpt2_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
        assert!(matches!(
            gpt2_cfg.activation,
            candle_nn::Activation::NewGelu
        ));
    }

    #[test]
    fn test_gpt2_config_extraction_custom() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "n_inner".to_string(),
            serde_json::Value::Number(serde_json::Number::from(128)),
        );
        cfg.extra.insert(
            "layer_norm_epsilon".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(1e-6).unwrap()),
        );
        cfg.extra.insert(
            "activation_function".to_string(),
            serde_json::Value::String("gelu".to_string()),
        );

        let gpt2_cfg = GPT2Config::from_model_config(&cfg);

        assert_eq!(gpt2_cfg.n_inner, 128);
        assert!((gpt2_cfg.layer_norm_epsilon - 1e-6).abs() < 1e-10);
        assert!(matches!(gpt2_cfg.activation, candle_nn::Activation::Gelu));
    }

    #[test]
    fn test_gpt2_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = GPT2LMHeadModel::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "GPT2LMHeadModel should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.h.len(), cfg.num_hidden_layers);
    }
}
