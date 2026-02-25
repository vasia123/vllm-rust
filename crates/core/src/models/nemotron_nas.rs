use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear, TpSwiGluMlp};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn find_multiple(n: usize, k: usize) -> usize {
    if n.is_multiple_of(k) {
        n
    } else {
        n + k - (n % k)
    }
}

fn ffn_mult_to_intermediate_size(ffn_mult: f64, hidden_size: usize) -> usize {
    let intermediate = (2.0 * ffn_mult * hidden_size as f64 / 3.0) as usize;
    find_multiple(intermediate, 256)
}

// ─── Block config ─────────────────────────────────────────────────────────────

struct BlockAttnConfig {
    no_op: bool,
    // Global (un-sharded) KV heads for this layer.
    num_kv_heads: usize,
}

struct BlockFfnConfig {
    no_op: bool,
    intermediate_size: usize,
}

struct BlockConfig {
    attn: BlockAttnConfig,
    ffn: BlockFfnConfig,
}

fn parse_block_configs(cfg: &ModelConfig) -> Vec<BlockConfig> {
    let num_heads = cfg.num_attention_heads;
    let hidden_size = cfg.hidden_size;
    let default_kv_heads = cfg.num_key_value_heads;
    let default_intermediate = cfg.intermediate_size;

    let arr = cfg.extra.get("block_configs").and_then(|v| v.as_array());

    match arr {
        None => (0..cfg.num_hidden_layers)
            .map(|_| BlockConfig {
                attn: BlockAttnConfig {
                    no_op: false,
                    num_kv_heads: default_kv_heads,
                },
                ffn: BlockFfnConfig {
                    no_op: false,
                    intermediate_size: default_intermediate,
                },
            })
            .collect(),
        Some(arr) => arr
            .iter()
            .map(|bc| {
                let attn_json = bc.get("attention").unwrap_or(&serde_json::Value::Null);
                let no_op_attn = attn_json
                    .get("no_op")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let n_heads_in_group = attn_json
                    .get("n_heads_in_group")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(0);
                // n_heads_in_group == 0 means use the global default
                let num_kv_heads = if n_heads_in_group == 0 {
                    default_kv_heads
                } else {
                    num_heads / n_heads_in_group
                };

                let ffn_json = bc.get("ffn").unwrap_or(&serde_json::Value::Null);
                let no_op_ffn = ffn_json
                    .get("no_op")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let intermediate_size =
                    if let Some(sz) = ffn_json.get("intermediate_size").and_then(|v| v.as_u64()) {
                        sz as usize
                    } else if let Some(mult) = ffn_json.get("ffn_mult").and_then(|v| v.as_f64()) {
                        ffn_mult_to_intermediate_size(mult, hidden_size)
                    } else {
                        default_intermediate
                    };

                BlockConfig {
                    attn: BlockAttnConfig {
                        no_op: no_op_attn,
                        num_kv_heads,
                    },
                    ffn: BlockFfnConfig {
                        no_op: no_op_ffn,
                        intermediate_size,
                    },
                }
            })
            .collect(),
    }
}

// ─── Attention ────────────────────────────────────────────────────────────────

// Per-layer attention accepting explicit dimensions; mirrors LlamaAttention but
// does not read from &ModelConfig so each layer can have a different num_kv_heads.
struct NasAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl NasAttention {
    #[allow(clippy::too_many_arguments)]
    fn new_with_tp(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
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
            hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            hidden_size,
            num_kv_heads * head_dim,
            false,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            hidden_size,
            false,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            max_position_embeddings,
            rope_theta,
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
}

// ─── Decoder Layer ────────────────────────────────────────────────────────────

struct NasDecoderLayer {
    self_attn: Option<NasAttention>,
    input_layernorm: Option<RmsNorm>,
    mlp: Option<TpSwiGluMlp>,
    post_attention_layernorm: Option<RmsNorm>,
    /// Index into the KV cache manager; None when attention is no-op.
    kv_layer_idx: Option<usize>,
}

impl NasDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        block: &BlockConfig,
        kv_layer_idx: Option<usize>,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let (self_attn, input_layernorm) = if block.attn.no_op {
            (None, None)
        } else {
            let attn = NasAttention::new_with_tp(
                cfg.hidden_size,
                cfg.num_attention_heads,
                block.attn.num_kv_heads,
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                vb.pp("self_attn"),
                pg,
            )?;
            let ln = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
            (Some(attn), Some(ln))
        };

        let (mlp, post_attention_layernorm) = if block.ffn.no_op {
            (None, None)
        } else {
            let m = TpSwiGluMlp::new(
                cfg.hidden_size,
                block.ffn.intermediate_size,
                vb.pp("mlp"),
                pg,
            )?;
            let ln = rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?;
            (Some(m), Some(ln))
        };

        Ok(Self {
            self_attn,
            input_layernorm,
            mlp,
            post_attention_layernorm,
            kv_layer_idx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let mut hidden = xs.clone();

        if let (Some(attn), Some(ln), Some(kv_idx)) =
            (&self.self_attn, &self.input_layernorm, self.kv_layer_idx)
        {
            let residual = hidden.clone();
            let normed = ln.forward(&hidden)?;
            let attn_out = attn.forward(
                &normed,
                attention_mask,
                seqlen_offset,
                kv_cache_mgr.engine_mut(kv_idx),
                block_table,
                slot_mapping,
                tp_ctx,
            )?;
            hidden = (attn_out + &residual)?;
        }

        if let (Some(mlp), Some(ln)) = (&self.mlp, &self.post_attention_layernorm) {
            let residual = hidden.clone();
            let normed = ln.forward(&hidden)?;
            let mlp_out = mlp.forward(&normed, tp_ctx)?;
            hidden = (mlp_out + &residual)?;
        }

        Ok(hidden)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let mut hidden = xs.clone();

        if let (Some(attn), Some(ln), Some(kv_idx)) =
            (&self.self_attn, &self.input_layernorm, self.kv_layer_idx)
        {
            let residual = hidden.clone();
            let normed = ln.forward(&hidden)?;
            let attn_out = attn.forward_decode_batch(
                &normed,
                sequences,
                kv_cache_mgr.engine_mut(kv_idx),
                tp_ctx,
            )?;
            hidden = (attn_out + &residual)?;
        }

        if let (Some(mlp), Some(ln)) = (&self.mlp, &self.post_attention_layernorm) {
            let residual = hidden.clone();
            let normed = ln.forward(&hidden)?;
            let mlp_out = mlp.forward(&normed, tp_ctx)?;
            hidden = (mlp_out + &residual)?;
        }

        Ok(hidden)
    }

    #[cfg(test)]
    fn is_attention(&self) -> bool {
        self.self_attn.is_some()
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

/// DeciLM / NemotronNAS: Llama-based architecture with per-layer optional
/// attention and FFN blocks. No-op blocks are skipped entirely; only active
/// attention layers consume KV cache slots (indexed 0..num_kv_layers).
pub struct NemotronNasForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<NasDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
    /// Number of KV-cache layers (== count of non-no-op attention layers).
    num_kv_layers: usize,
}

impl NemotronNasForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let block_configs = parse_block_configs(cfg);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut num_kv_layers = 0usize;
        let vb_l = vb_m.pp("layers");
        for (i, block) in block_configs.iter().enumerate() {
            let kv_idx = if block.attn.no_op {
                None
            } else {
                let idx = num_kv_layers;
                num_kv_layers += 1;
                Some(idx)
            };
            layers.push(NasDecoderLayer::new_with_tp(
                cfg,
                block,
                kv_idx,
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
            num_kv_layers,
        })
    }

    /// Number of active KV-cache layers (for sizing the KVCacheManager).
    pub fn num_kv_layers(&self) -> usize {
        self.num_kv_layers
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
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
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
}

impl crate::engine::ModelForward for NemotronNasForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        NemotronNasForCausalLM::forward(
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
        for layer in &self.layers {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, &self.tp_ctx)?;
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
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn base_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["DeciLMForCausalLM".to_string()],
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
            extra: serde_json::Map::new(),
        }
    }

    /// Block configs where layer 1 has no-op attention and layer 3 has no-op FFN.
    /// All attention layers use n_heads_in_group=2 → num_kv_heads = 4/2 = 2.
    fn mixed_block_configs() -> serde_json::Value {
        serde_json::json!([
            {"attention": {"no_op": false, "n_heads_in_group": 2}, "ffn": {"no_op": false, "intermediate_size": 128}},
            {"attention": {"no_op": true,  "n_heads_in_group": 2}, "ffn": {"no_op": false, "intermediate_size": 128}},
            {"attention": {"no_op": false, "n_heads_in_group": 2}, "ffn": {"no_op": false, "intermediate_size": 128}},
            {"attention": {"no_op": false, "n_heads_in_group": 2}, "ffn": {"no_op": true,  "intermediate_size": 128}}
        ])
    }

    fn create_kv_cache(num_kv_layers: usize, num_kv_heads: usize) -> (KVCacheManager, BlockTable) {
        let cc = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: num_kv_layers,
            num_kv_heads,
            head_dim: 16,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cc).expect("cache manager");
        let bt = BlockTable::new(cc.block_size);
        (mgr, bt)
    }

    #[test]
    fn test_nemotron_nas_construction_no_block_configs() {
        // Without block_configs in extra, all layers are standard full layers.
        let cfg = base_config();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = NemotronNasForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "should construct: {:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 4);
        assert_eq!(model.num_kv_layers(), 4, "all layers have attention");
        assert!(model.layers.iter().all(|l| l.is_attention()));
    }

    #[test]
    fn test_nemotron_nas_mixed_noop_construction() {
        let mut cfg = base_config();
        cfg.extra
            .insert("block_configs".to_string(), mixed_block_configs());
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = NemotronNasForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "should construct: {:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 4);
        // Layer 1 is no-op attention → 3 KV layers (0, 2, 3)
        assert_eq!(model.num_kv_layers(), 3);
        assert!(model.layers[0].is_attention());
        assert!(!model.layers[1].is_attention());
        assert!(model.layers[2].is_attention());
        assert!(model.layers[3].is_attention());
    }

    #[test]
    fn test_nemotron_nas_ffn_mult_parsing() {
        // ffn_mult=3.0, hidden=64 → intermediate = int(2*3*64/3) = 128 → find_multiple(128,256) = 256
        assert_eq!(ffn_mult_to_intermediate_size(3.0, 64), 256);
        // ffn_mult=2.0, hidden=512 → intermediate = int(2*2*512/3) = 682 → find_multiple(682,256) = 768
        assert_eq!(ffn_mult_to_intermediate_size(2.0, 512), 768);
        // Exact multiple: int(2*3*768/3) = 1536 → already multiple of 256
        assert_eq!(ffn_mult_to_intermediate_size(3.0, 768), 1536);
        // find_multiple(0, 256) = 0
        assert_eq!(find_multiple(0, 256), 0);
        // find_multiple(257, 256) = 512
        assert_eq!(find_multiple(257, 256), 512);
    }

    #[test]
    fn test_nemotron_nas_forward_shape() {
        let mut cfg = base_config();
        cfg.extra
            .insert("block_configs".to_string(), mixed_block_configs());
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = NemotronNasForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_mgr, mut bt) = create_kv_cache(model.num_kv_layers(), cfg.num_key_value_heads);

        let batch = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch, seq_len), DType::U32, &device).expect("input_ids");

        kv_mgr
            .allocate_for_request(&mut bt, seq_len)
            .expect("allocate");
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch, seq_len, cfg.vocab_size],
            "logits shape mismatch"
        );
    }

    #[test]
    fn test_nemotron_nas_decode_batch_shape() {
        let mut cfg = base_config();
        cfg.extra
            .insert("block_configs".to_string(), mixed_block_configs());
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = NemotronNasForCausalLM::new(&cfg, vb).expect("model");

        let (mut kv_mgr, mut bt) = create_kv_cache(model.num_kv_layers(), cfg.num_key_value_heads);

        // Prefill one sequence of length 3, then decode one new token.
        let prefill_len = 3;
        kv_mgr
            .allocate_for_request(&mut bt, prefill_len + 1)
            .expect("allocate");

        let sequences = vec![DecodeSequenceMetadata {
            request_id: 0,
            seqlen_offset: prefill_len,
            block_ids: bt.block_ids().to_vec(),
            slot_mapping: bt.slot_mapping(prefill_len, 1),
        }];

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input_ids");
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_mgr)
            .expect("decode_batch");

        assert_eq!(
            logits.dims(),
            &[1, 1, cfg.vocab_size],
            "decode logits shape mismatch"
        );
    }
}
