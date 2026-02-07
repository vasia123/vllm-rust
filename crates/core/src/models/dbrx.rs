//! DBRX (Databricks) model implementation.
//!
//! Key architectural features:
//! - MoE (Mixture of Experts) model
//! - Fused QKV projection (Wqkv)
//! - LayerNorm
//! - GELU activation
//! - MoE with top-k routing (typically top-2 out of 16 experts)
//! - Weight naming: transformer.blocks.{i}, transformer.norm_f
//! - Attention: norm_attn_norm.attn.Wqkv, norm_attn_norm.attn.out_proj
//! - Norms: norm_attn_norm.norm_1, norm_attn_norm.norm_2
//! - MoE router: ffn.router.layer.weight
//! - MoE experts: ffn.experts.mlp.{expert_id}.v1 (gate), w1 (up), w2 (down)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear_no_bias, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

pub use super::tp_layers::TpContext;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── DBRX Config Extraction ─────────────────────────────────────────────────

struct DbrxConfig {
    layer_norm_epsilon: f64,
    num_experts: usize,
    top_k: usize,
    ffn_hidden_size: usize,
}

impl DbrxConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let layer_norm_epsilon = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let num_experts = cfg
            .extra
            .get("ffn_config")
            .and_then(|v| v.get("moe_num_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;

        let top_k = cfg
            .extra
            .get("ffn_config")
            .and_then(|v| v.get("moe_top_k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;

        let ffn_hidden_size = cfg
            .extra
            .get("ffn_config")
            .and_then(|v| v.get("ffn_hidden_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.intermediate_size);

        Self {
            layer_norm_epsilon,
            num_experts,
            top_k,
            ffn_hidden_size,
        }
    }
}

// ─── Single Expert MLP ──────────────────────────────────────────────────────
//
// Each expert is a GeGLU MLP: gelu(gate(x)) * up(x), then down.
// Weight naming: v1 (gate), w1 (up), w2 (down)

struct DbrxExpertMlp {
    v1: Linear, // gate
    w1: Linear, // up
    w2: Linear, // down
}

impl DbrxExpertMlp {
    fn new(hidden_size: usize, ffn_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let v1 = linear_no_bias(hidden_size, ffn_hidden_size, vb.pp("v1"))?;
        let w1 = linear_no_bias(hidden_size, ffn_hidden_size, vb.pp("w1"))?;
        let w2 = linear_no_bias(ffn_hidden_size, hidden_size, vb.pp("w2"))?;
        Ok(Self { v1, w1, w2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.v1.forward(xs)?.gelu_erf()?;
        let up = self.w1.forward(xs)?;
        let hidden = (gate * up)?;
        self.w2.forward(&hidden)
    }
}

// ─── MoE Layer ──────────────────────────────────────────────────────────────

struct DbrxMoE {
    router: Linear,
    experts: Vec<DbrxExpertMlp>,
    num_experts: usize,
    top_k: usize,
}

impl DbrxMoE {
    fn new(
        hidden_size: usize,
        ffn_hidden_size: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = linear_no_bias(hidden_size, num_experts, vb.pp("router").pp("layer"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_e = vb.pp("experts").pp("mlp");
        for i in 0..num_experts {
            experts.push(DbrxExpertMlp::new(
                hidden_size,
                ffn_hidden_size,
                vb_e.pp(i),
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = xs.dims3()?;
        let xs_flat = xs.reshape((b_sz * seq_len, hidden_size))?;

        // Router: compute expert selection
        let router_logits = self.router.forward(&xs_flat)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, 1)?;

        // Select top-k experts per token
        let router_probs_data: Vec<f32> = router_probs.to_vec2()?.into_iter().flatten().collect();

        let num_tokens = b_sz * seq_len;
        let mut output_data = vec![0.0f32; num_tokens * hidden_size];

        for token_idx in 0..num_tokens {
            // Find top-k expert indices and weights
            let probs = &router_probs_data
                [token_idx * self.num_experts..(token_idx + 1) * self.num_experts];
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_k_experts: Vec<(usize, f32)> =
                indexed_probs.into_iter().take(self.top_k).collect();

            // Renormalize top-k weights
            let weight_sum: f32 = top_k_experts.iter().map(|(_, w)| w).sum();
            let norm_factor = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                0.0
            };

            let token_input = xs_flat.narrow(0, token_idx, 1)?;

            for (expert_idx, weight) in &top_k_experts {
                if *expert_idx < self.experts.len() {
                    let expert_output = self.experts[*expert_idx].forward(&token_input)?;
                    let expert_data: Vec<f32> = expert_output.flatten_all()?.to_vec1()?;
                    let normalized_weight = weight * norm_factor;

                    for (j, val) in expert_data.iter().enumerate() {
                        output_data[token_idx * hidden_size + j] += val * normalized_weight;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, (b_sz, seq_len, hidden_size), xs.device())?
            .to_dtype(xs.dtype())?;

        Ok(output)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct DbrxAttention {
    wqkv: TpLinear,
    out_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl DbrxAttention {
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

        // Fused QKV
        let qkv_size = (num_heads + 2 * num_kv_heads) * head_dim;
        let wqkv =
            TpLinear::column_parallel(cfg.hidden_size, qkv_size, false, false, vb.pp("Wqkv"), pg)?;

        let out_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("out_proj"),
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
            wqkv,
            out_proj,
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

        let qkv = self.wqkv.forward(xs, tp_ctx)?;

        // Split fused QKV: Q has num_heads, K and V each have num_kv_heads
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

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

        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

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

            self.out_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
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
            self.out_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct DbrxDecoderLayer {
    norm_1: LayerNorm,
    attn: DbrxAttention,
    norm_2: LayerNorm,
    ffn: DbrxMoE,
}

impl DbrxDecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        dbrx_cfg: &DbrxConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let norm_1 = layer_norm(
            cfg.hidden_size,
            dbrx_cfg.layer_norm_epsilon,
            vb.pp("norm_attn_norm").pp("norm_1"),
        )?;
        let attn = DbrxAttention::new_with_tp(cfg, vb.pp("norm_attn_norm").pp("attn"), pg)?;
        let norm_2 = layer_norm(
            cfg.hidden_size,
            dbrx_cfg.layer_norm_epsilon,
            vb.pp("norm_attn_norm").pp("norm_2"),
        )?;
        let ffn = DbrxMoE::new(
            cfg.hidden_size,
            dbrx_cfg.ffn_hidden_size,
            dbrx_cfg.num_experts,
            dbrx_cfg.top_k,
            vb.pp("ffn"),
        )?;

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
        let moe_output = self.ffn.forward(&xs)?;
        residual + moe_output
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
        let moe_output = self.ffn.forward(&xs)?;
        residual + moe_output
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct DbrxForCausalLM {
    wte: TpEmbedding,
    blocks: Vec<DbrxDecoderLayer>,
    norm_f: LayerNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl DbrxForCausalLM {
    /// Create a new DBRX model for single GPU.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    /// Create a new DBRX model with tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let dbrx_cfg = DbrxConfig::from_model_config(cfg);
        let vb_t = vb.pp("transformer");
        let world_size = pg.world_size();

        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb_t.pp("blocks");
        for i in 0..cfg.num_hidden_layers {
            blocks.push(DbrxDecoderLayer::new_with_tp(
                cfg,
                &dbrx_cfg,
                vb_b.pp(i),
                pg,
            )?);
        }

        let norm_f = layer_norm(
            cfg.hidden_size,
            dbrx_cfg.layer_norm_epsilon,
            vb_t.pp("norm_f"),
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

impl crate::engine::ModelForward for DbrxForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        DbrxForCausalLM::forward(
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

        // MoE config in ffn_config
        let mut ffn_config = serde_json::Map::new();
        ffn_config.insert(
            "moe_num_experts".to_string(),
            serde_json::Value::Number(serde_json::Number::from(4)),
        );
        ffn_config.insert(
            "moe_top_k".to_string(),
            serde_json::Value::Number(serde_json::Number::from(2)),
        );
        ffn_config.insert(
            "ffn_hidden_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(128)),
        );
        extra.insert(
            "ffn_config".to_string(),
            serde_json::Value::Object(ffn_config),
        );

        ModelConfig {
            architectures: vec!["DbrxForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
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
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_dbrx_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = DbrxForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "DbrxForCausalLM should construct with zero weights: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_dbrx_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_dbrx_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_dbrx_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_dbrx_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_dbrx_config_extraction() {
        let cfg = test_config();
        let dbrx_cfg = DbrxConfig::from_model_config(&cfg);
        assert_eq!(dbrx_cfg.num_experts, 4);
        assert_eq!(dbrx_cfg.top_k, 2);
        assert_eq!(dbrx_cfg.ffn_hidden_size, 128);
        assert!((dbrx_cfg.layer_norm_epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_dbrx_config_extraction_defaults() {
        let mut cfg = test_config();
        cfg.extra = serde_json::Map::new();
        let dbrx_cfg = DbrxConfig::from_model_config(&cfg);
        assert_eq!(dbrx_cfg.num_experts, 4);
        assert_eq!(dbrx_cfg.top_k, 2);
        assert_eq!(dbrx_cfg.ffn_hidden_size, cfg.intermediate_size);
    }

    #[test]
    fn test_dbrx_moe_routing() {
        // Verify MoE layer routes and combines expert outputs correctly
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let dbrx_cfg = DbrxConfig::from_model_config(&cfg);
        let moe = DbrxMoE::new(
            cfg.hidden_size,
            dbrx_cfg.ffn_hidden_size,
            dbrx_cfg.num_experts,
            dbrx_cfg.top_k,
            vb.pp("moe"),
        )
        .expect("build MoE");

        assert_eq!(moe.experts.len(), dbrx_cfg.num_experts);
        assert_eq!(moe.top_k, dbrx_cfg.top_k);

        // Run forward through MoE
        let input = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let output = moe.forward(&input).expect("MoE forward");
        assert_eq!(
            output.dims(),
            &[1, 3, cfg.hidden_size],
            "MoE output should preserve shape"
        );
    }

    #[test]
    fn test_dbrx_tied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = true;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_dbrx_untied_embeddings() {
        let cfg = test_config();
        assert!(!cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_dbrx_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = DbrxForCausalLM::new(&cfg, vb).expect("build model");

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
    fn test_dbrx_gqa_configuration() {
        let cfg = test_config();
        let gqa_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(gqa_groups, 2, "test config uses GQA with 2 groups");
    }

    #[test]
    fn test_dbrx_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();
        let tp_ctx = TpContext::single_gpu();

        let model = DbrxForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "DbrxForCausalLM should construct with TP: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.blocks.len(), cfg.num_hidden_layers);
    }
}
