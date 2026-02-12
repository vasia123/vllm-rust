use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::moe::{MoELayerWithShared, MoELayerWithSharedConfig};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Config ──────────────────────────────────────────────────────────────────

struct Llama4Config {
    num_local_experts: usize,
    num_experts_per_tok: usize,
    interleave_moe_layer_step: usize,
    no_rope_layers: Vec<bool>,
}

impl Llama4Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let num_local_experts = cfg
            .extra
            .get("num_local_experts")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let num_experts_per_tok = cfg
            .extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let interleave_moe_layer_step = cfg
            .extra
            .get("interleave_moe_layer_step")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let no_rope_layers = cfg
            .extra
            .get("no_rope_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_u64().unwrap_or(0) == 1)
                    .collect()
            })
            .unwrap_or_default();

        Self {
            num_local_experts,
            num_experts_per_tok,
            interleave_moe_layer_step,
            no_rope_layers,
        }
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if self.interleave_moe_layer_step == 0 {
            return false;
        }
        (layer_idx + 1).is_multiple_of(self.interleave_moe_layer_step)
    }

    fn has_rope(&self, layer_idx: usize) -> bool {
        if layer_idx < self.no_rope_layers.len() {
            !self.no_rope_layers[layer_idx]
        } else {
            true
        }
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Llama4Attention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_rope: bool,
}

impl Llama4Attention {
    fn new_with_tp(
        cfg: &ModelConfig,
        use_rope: bool,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

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
            num_heads: num_heads / world_size,
            num_kv_heads: num_kv_heads / world_size,
            head_dim,
            use_rope,
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
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Only apply RoPE if this layer uses it
        let (q, k) = if self.use_rope {
            self.rotary_emb.apply(&q, &k, seqlen_offset)?
        } else {
            (q, k)
        };

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
            let (q, k) = if self.use_rope {
                self.rotary_emb.apply_varlen(&q, &k, &positions)?
            } else {
                (q, k)
            };

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

                let (q_i, k_i) = if self.use_rope {
                    self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?
                } else {
                    (q_i, k_i)
                };

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

// ─── Feed-Forward ────────────────────────────────────────────────────────────

enum Llama4FeedForward {
    Dense(TpSwiGluMlp),
    Moe(MoELayerWithShared),
}

impl Llama4FeedForward {
    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        match self {
            Llama4FeedForward::Dense(mlp) => mlp.forward(xs, tp_ctx),
            Llama4FeedForward::Moe(moe) => moe.forward(xs),
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct Llama4DecoderLayer {
    self_attn: Llama4Attention,
    feed_forward: Llama4FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Llama4DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        l4_cfg: &Llama4Config,
        layer_idx: usize,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let use_rope = l4_cfg.has_rope(layer_idx);
        let self_attn = Llama4Attention::new_with_tp(cfg, use_rope, vb.pp("self_attn"), pg)?;

        let feed_forward = if l4_cfg.is_moe_layer(layer_idx) {
            let shared_intermediate = cfg
                .extra
                .get("shared_expert_intermediate_size")
                .or_else(|| cfg.extra.get("intermediate_size_mlp"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);

            let moe_config = MoELayerWithSharedConfig {
                hidden_size: cfg.hidden_size,
                intermediate_size: cfg.intermediate_size,
                shared_expert_intermediate_size: shared_intermediate,
                num_experts: l4_cfg.num_local_experts,
                top_k: l4_cfg.num_experts_per_tok,
                renormalize: true,
                scoring_func: crate::moe::ScoringFunc::Sigmoid,
                routed_scaling_factor: 1.0,
                gated_shared_expert: false,
                use_grouped_topk: false,
                num_expert_groups: None,
                topk_per_group: None,
                inplace: false,
                is_act_and_mul: true,
            };
            Llama4FeedForward::Moe(MoELayerWithShared::new(moe_config, vb.pp("feed_forward"))?)
        } else {
            let mlp_intermediate = cfg
                .extra
                .get("intermediate_size_mlp")
                .and_then(|v| v.as_u64())
                .unwrap_or(cfg.intermediate_size as u64)
                as usize;
            Llama4FeedForward::Dense(TpSwiGluMlp::new(
                cfg.hidden_size,
                mlp_intermediate,
                vb.pp("feed_forward"),
                pg,
            )?)
        };

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            feed_forward,
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
        let xs = self
            .feed_forward
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
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
        let xs = self
            .feed_forward
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Llama4ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Llama4DecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Llama4ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let l4_cfg = Llama4Config::from_model_config(cfg);
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Llama4DecoderLayer::new_with_tp(
                cfg, &l4_cfg, i, vb_l.pp(i), pg,
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

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for Llama4ForCausalLM {
    fn forward(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config(with_moe: bool) -> ModelConfig {
        let mut extra = serde_json::Map::new();
        if with_moe {
            extra.insert("num_local_experts".into(), serde_json::Value::from(4));
            extra.insert("num_experts_per_tok".into(), serde_json::Value::from(1));
            extra.insert(
                "interleave_moe_layer_step".into(),
                serde_json::Value::from(2),
            );
            extra.insert(
                "no_rope_layers".into(),
                serde_json::json!([0, 1, 0, 0]),
            );
        }

        ModelConfig {
            architectures: vec!["Llama4ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
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

    #[test]
    fn test_llama4_dense_construction() {
        let cfg = test_config(false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Llama4ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Llama4ForCausalLM (dense) should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), 4);
    }

    #[test]
    fn test_llama4_moe_construction() {
        let cfg = test_config(true);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Llama4ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Llama4ForCausalLM (moe) should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_llama4_forward_shape() {
        let cfg = test_config(false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Llama4ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");

        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward pass");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_llama4_config_moe_layer_detection() {
        let cfg = test_config(true);
        let l4_cfg = Llama4Config::from_model_config(&cfg);
        // interleave_moe_layer_step=2: layers 1, 3 are MoE
        assert!(!l4_cfg.is_moe_layer(0));
        assert!(l4_cfg.is_moe_layer(1));
        assert!(!l4_cfg.is_moe_layer(2));
        assert!(l4_cfg.is_moe_layer(3));
    }

    #[test]
    fn test_llama4_config_rope_detection() {
        let cfg = test_config(true);
        let l4_cfg = Llama4Config::from_model_config(&cfg);
        // no_rope_layers = [0, 1, 0, 0]: layer 1 has no rope
        assert!(l4_cfg.has_rope(0));
        assert!(!l4_cfg.has_rope(1));
        assert!(l4_cfg.has_rope(2));
        assert!(l4_cfg.has_rope(3));
    }
}
