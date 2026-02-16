//! Chameleon unified token model implementation.
//!
//! Architecture:
//! - Image tokenization: Frozen VQVAE encoder + vector quantizer
//! - Unified vocabulary: Images encoded as discrete tokens in shared BPE space
//! - LLM: Llama-like transformer with Q/K RMSNorm per head
//! - Optional Swin-style layer normalization
//!
//! Reference: reference/vllm/vllm/model_executor/models/chameleon.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{causal_mask, paged_attention, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ChameleonConfig {
    model_config: ModelConfig,
    swin_norm: bool,
    logit_scale: f64,
}

impl ChameleonConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let swin_norm = cfg
            .extra
            .get("swin_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let logit_scale = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            model_config: cfg.clone(),
            swin_norm,
            logit_scale,
        }
    }
}

// ─── Per-Head RMSNorm (ChameleonLayerNorm) ──────────────────────────────────

/// Per-head RMSNorm applied to Q and K projections.
///
/// Unlike standard RMSNorm, this applies normalization per head independently.
struct ChameleonQKNorm {
    norms: Vec<RmsNorm>,
    num_heads: usize,
    head_dim: usize,
}

impl ChameleonQKNorm {
    fn new(num_heads: usize, head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let mut norms = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            norms.push(rms_norm(head_dim, eps, vb.pp(i))?);
        }
        Ok(Self {
            norms,
            num_heads,
            head_dim,
        })
    }

    /// Apply per-head normalization.
    ///
    /// Input shape: (batch, seq_len, num_heads * head_dim)
    /// Output shape: same
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Reshape to (batch, seq_len, num_heads, head_dim)
        let x = x.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        let mut head_outputs = Vec::with_capacity(self.num_heads);
        for (h, norm) in self.norms.iter().enumerate() {
            // Extract head h: (batch, seq_len, head_dim)
            let head = x.narrow(2, h, 1)?.squeeze(2)?;
            let normed = norm.forward(&head)?;
            head_outputs.push(normed.unsqueeze(2)?);
        }

        let result = Tensor::cat(&head_outputs, 2)?;
        result.reshape((batch, seq_len, self.num_heads * self.head_dim))
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct ChameleonAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: ChameleonQKNorm,
    k_norm: ChameleonQKNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

impl ChameleonAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let total_qkv = (num_heads + 2 * num_kv_heads) * head_dim;
        let qkv_proj = candle_nn::linear(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = ChameleonQKNorm::new(num_heads, head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm =
            ChameleonQKNorm::new(num_kv_heads, head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rotary = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups: num_heads / num_kv_heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_dim)?;
        let k = qkv.narrow(2, q_dim, kv_dim)?;
        let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;

        // Per-head Q/K normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Reshape: (batch, seq_len, heads, head_dim) → (batch, heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // RoPE
        let (q, k) = self.rotary.apply(&q, &k, seqlen_offset)?;

        // Paged attention with KV cache
        paged_attention(
            &q,
            &k,
            &v,
            mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

// ─── MLP ────────────────────────────────────────────────────────────────────

/// SwiGLU MLP: gate_up_proj → silu(gate) * up → down_proj
struct ChameleonMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl ChameleonMLP {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = candle_nn::linear(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj =
            candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(2, 0, self.intermediate_size)?;
        let up = gate_up.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        self.down_proj.forward(&x)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct ChameleonDecoderLayer {
    self_attn: ChameleonAttention,
    mlp: ChameleonMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl ChameleonDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = ChameleonAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = ChameleonMLP::new(cfg, vb.pp("mlp"))?;
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

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(
            &x,
            mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let x = (residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Chameleon unified token model.
///
/// Uses VQVAE to tokenize images into discrete tokens within a shared vocabulary.
/// The LLM is Llama-like with per-head Q/K RMSNorm on attention projections.
pub struct ChameleonForConditionalGeneration {
    embed_tokens: TpEmbedding,
    layers: Vec<ChameleonDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    #[allow(dead_code)]
    config: ChameleonConfig,
    #[allow(dead_code)]
    logit_scale: f64,
    device: Device,
    dtype: DType,
}

impl ChameleonForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = ChameleonConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ChameleonDecoderLayer::new(cfg, vb_l.pp(i))?);
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
            logit_scale: config.logit_scale,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        _multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        // In Chameleon, images are already tokenized as discrete BPE tokens
        // via VQVAE + vocabulary mapping, so they go through embed_tokens directly.
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;

        // Apply logit scaling
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        if (self.logit_scale - 1.0).abs() > 1e-6 {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
    }
}

impl crate::engine::ModelForward for ChameleonForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            None,
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

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut batch_outputs = Vec::with_capacity(sequences.len());
            let cache_engine = kv_cache_mgr.engine_mut(layer_idx);
            for (seq_idx, seq_meta) in sequences.iter().enumerate() {
                let x_single = xs.narrow(0, seq_idx, 1)?;
                let block_table = BlockTable::from_block_ids(seq_meta.block_ids.clone(), 0);
                let out = layer.forward(
                    &x_single,
                    None,
                    seq_meta.seqlen_offset,
                    cache_engine,
                    &block_table,
                    &seq_meta.slot_mapping,
                )?;
                batch_outputs.push(out);
            }
            xs = Tensor::cat(&batch_outputs, 0)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        if (self.logit_scale - 1.0).abs() > 1e-6 {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // TODO: Implement VQVAE image tokenization for raw pixel input.
        // Currently, images must be pre-tokenized into discrete BPE tokens
        // before being passed as input_ids.
        self.forward_inner(
            input_ids,
            multimodal_inputs,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("swin_norm".to_string(), serde_json::json!(false));
        extra.insert("logit_scale".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["ChameleonForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
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
    fn test_qk_norm() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = ChameleonQKNorm::new(4, 16, 1e-6, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 3, 64), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 3, 64]);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }
}
