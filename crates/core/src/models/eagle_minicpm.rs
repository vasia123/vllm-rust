//! Eagle-1 speculative decoding draft model for MiniCPM.
//!
//! Eagle MiniCPM uses the same µP-scaled decoder layers as regular MiniCPM,
//! with two differences from Eagle Llama:
//!  1. Both embeddings and target hidden states are normalised before fc-fusion:
//!     `fc(cat(norm1(embed * scale_emb), norm2(hidden)))` → layers
//!  2. Final output is divided by `scale_width = hidden_size / dim_model_base`
//!     instead of using a separate RMSNorm.
//!
//! Weight paths: `model.embed_tokens`, `model.fc`, `model.input_norm{1,2}`,
//! `model.eagle_layers.{i}.*`, `lm_head`.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/minicpm_eagle.py`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, rms_norm, RmsNorm};

use super::eagle_llama::Eagle1DraftModel;
use super::minicpm::{MiniCPMConfig, MiniCPMDecoderLayer};

// ─── Eagle MiniCPM Model ───────────────────────────────────────────────────

/// Eagle-1 speculative decoding draft model for MiniCPM.
///
/// Implements µP scaling: embeddings are multiplied by `scale_emb`, both
/// normalised with dedicated norms, fused via fc, passed through MiniCPM
/// decoder layers (with `scale_depth`-scaled residuals), then divided by
/// `scale_width` before the lm_head.
pub struct EagleMiniCPMForCausalLM {
    embed_tokens: Embedding,
    /// Normalise token embeddings before fc-fusion.
    input_norm1: RmsNorm,
    /// Normalise target hidden states before fc-fusion.
    input_norm2: RmsNorm,
    /// fc: `cat(norm1(embed), norm2(hidden))` [B, S, 2*H] → [B, S, H]
    fc: Linear,
    layers: Vec<MiniCPMDecoderLayer>,
    lm_head: Linear,
    scale_emb: f64,
    /// `hidden_size / dim_model_base` — divides output before lm_head.
    output_scale: f64,
    device: Device,
}

impl EagleMiniCPMForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mini_cfg = MiniCPMConfig::from_model_config(cfg);

        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let input_norm1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("input_norm1"))?;
        let input_norm2 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("input_norm2"))?;

        let fc = linear_no_bias(2 * cfg.hidden_size, cfg.hidden_size, vb_m.pp("fc"))?;

        let vb_l = vb_m.pp("eagle_layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(MiniCPMDecoderLayer::new(cfg, &mini_cfg, vb_l.pp(i))?);
        }

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            input_norm1,
            input_norm2,
            fc,
            layers,
            lm_head,
            scale_emb: mini_cfg.scale_emb,
            output_scale: mini_cfg.output_scale(cfg.hidden_size),
            device: vb.device().clone(),
        })
    }
}

impl Eagle1DraftModel for EagleMiniCPMForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len) = input_ids.dims2()?;

        // Embed and scale, then normalise both inputs before fusion.
        let embeds = (self.embed_tokens.forward(input_ids)? * self.scale_emb)?;
        let embeds = self.input_norm1.forward(&embeds)?;
        let target_hs = self.input_norm2.forward(hidden_states)?;

        let combined = Tensor::cat(&[&embeds, &target_hs], 2)?;
        let mut hs = self.fc.forward(&combined)?;

        let mask = if seq_len > 1 {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                hs.dtype(),
                &self.device,
            )?)
        } else {
            None
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hs = layer.forward(
                &hs,
                mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }

        // Eagle MiniCPM: no separate final RMSNorm — just divide by scale_width.
        // Both postnorm and prenorm return the same scaled value.
        let out = (hs / self.output_scale)?;
        Ok((out.clone(), out))
    }

    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager};
    use candle_core::DType;

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("scale_emb".to_string(), serde_json::json!(1.0));
        extra.insert("scale_depth".to_string(), serde_json::json!(1.4));
        extra.insert("dim_model_base".to_string(), serde_json::json!(256.0));

        ModelConfig {
            architectures: vec!["EagleMiniCPMForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
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
            attention_bias: None,
            extra,
        }
    }

    fn create_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_eagle_minicpm_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleMiniCPMForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        let model = model.unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_eagle_minicpm_forward_prefill() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleMiniCPMForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_cache(&cfg, &device);
        let mut bt = BlockTable::new(16);

        let seq_len = 4;
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slots = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, seq_len, 64), DType::F32, &device).unwrap();

        let (postnorm, prenorm) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();

        assert_eq!(postnorm.dims(), &[1, seq_len, 64]);
        assert_eq!(prenorm.dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_eagle_minicpm_forward_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleMiniCPMForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = create_cache(&cfg, &device);
        let mut bt = BlockTable::new(16);

        kv.allocate_for_request(&mut bt, 1).unwrap();
        let slots = bt.slot_mapping(0, 1);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = Tensor::zeros((1, 1, 64), DType::F32, &device).unwrap();

        let (postnorm, _) = model
            .forward(&input_ids, &hidden, 0, &mut kv, &bt, &slots)
            .unwrap();
        assert_eq!(postnorm.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_eagle_minicpm_output_scale() {
        // scale_width = hidden_size / dim_model_base = 64 / 256 = 0.25
        // output after division should have smaller magnitude than raw output
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleMiniCPMForCausalLM::new(&cfg, vb).unwrap();
        assert!((model.output_scale - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_eagle_minicpm_compute_logits() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = EagleMiniCPMForCausalLM::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 3, 64), DType::F32, &device).unwrap();
        let logits = model.compute_logits(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 3, 256]);
    }
}
