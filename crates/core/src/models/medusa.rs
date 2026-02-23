//! Medusa speculative decoding draft model.
//!
//! Medusa attaches K independent prediction heads to a base model's hidden
//! states. Each head is a residual MLP (`x += act(W*x)` per layer) that
//! predicts one future token position. All K heads run in parallel from the
//! same hidden states — no autoregressive generation.
//!
//! Reference: "Medusa: Simple LLM Inference Acceleration Framework with
//! Multiple Decoding Heads" (Cai et al., 2024)
//!
//! Architecture name in checkpoints: `MedusaModel`
//!
//! Weight paths (Python canoncial, after `medusa_heads.` prefix strip):
//!   `blocks.{i}.layers.{j}.{weight,bias}` — residual hidden layers per head
//!   `lm_heads.{i}.weight`                 — per-head vocabulary projection

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;

// ─── MedusaDraftModel trait ──────────────────────────────────────────────────

/// Draft model trait for Medusa speculative decoding.
///
/// Medusa runs K independent residual blocks on the base model's last hidden
/// states, then projects each block's output to vocabulary logits. Each block
/// predicts one future token position.
pub trait MedusaDraftModel: Send {
    /// Run all K residual blocks on the base model hidden states.
    ///
    /// `hidden_states`: `[batch, hidden_size]` — last-position activations from
    /// the target model. If 3D `[batch, seq, hidden]`, the caller is responsible
    /// for slicing to the last position before calling this method.
    ///
    /// Returns K tensors of shape `[batch, hidden_size]`.
    fn forward_heads(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>>;

    /// Project K block outputs to vocabulary logits.
    ///
    /// `head_hidden_states`: slice of K tensors `[batch, hidden_size]`, one per
    /// head (as returned by [`forward_heads`]).
    ///
    /// Returns K tensors of shape `[batch, vocab_size]`.
    fn compute_logits(&self, head_hidden_states: &[Tensor]) -> Result<Vec<Tensor>>;

    /// Number of prediction heads (= number of speculative positions proposed).
    fn num_heads(&self) -> usize;

    fn device(&self) -> &Device;
}

// ─── MedusaResidualBlock ─────────────────────────────────────────────────────

/// Single Medusa residual block.
///
/// Applies `x = x + act(W * x)` for each hidden layer in sequence.
/// Matches Python `ResidualBlock` with `act = SiLU`.
struct MedusaResidualBlock {
    layers: Vec<Linear>,
}

impl MedusaResidualBlock {
    fn new(hidden_size: usize, num_layers: usize, use_bias: bool, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            let layer = if use_bias {
                linear(hidden_size, hidden_size, vb_l.pp(j))?
            } else {
                linear_no_bias(hidden_size, hidden_size, vb_l.pp(j))?
            };
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for layer in &self.layers {
            h = (h.clone() + candle_nn::ops::silu(&layer.forward(&h)?)?)?;
        }
        Ok(h)
    }
}

// ─── MedusaModel ─────────────────────────────────────────────────────────────

/// Medusa speculative decoding model.
///
/// Contains K prediction heads — each a `MedusaResidualBlock` plus a linear
/// vocabulary projection — that predict K future token positions in parallel.
///
/// Config fields read from `ModelConfig::extra`:
/// - `num_heads` (u64, default 4): number of Medusa heads
/// - `medusa_fc_bias` (bool, default false): add bias to residual linear layers
/// - `truncated_vocab_size` (u64): reduced vocab for lm_heads (default: full vocab)
///
/// `ModelConfig::num_hidden_layers` controls the number of residual layers
/// per head (typically 1).
pub struct MedusaModel {
    blocks: Vec<MedusaResidualBlock>,
    lm_heads: Vec<Linear>,
    device: Device,
}

impl MedusaModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg
            .extra
            .get("num_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(4) as usize;
        let num_layers = cfg.num_hidden_layers;
        let use_bias = cfg
            .extra
            .get("medusa_fc_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Use truncated vocab when a token_map will filter to top-k tokens.
        let vocab_size = cfg
            .extra
            .get("truncated_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.vocab_size);

        let vb_blocks = vb.pp("blocks");
        let vb_lm = vb.pp("lm_heads");

        let mut blocks = Vec::with_capacity(num_heads);
        let mut lm_heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            blocks.push(MedusaResidualBlock::new(
                cfg.hidden_size,
                num_layers,
                use_bias,
                vb_blocks.pp(i),
            )?);
            // LM heads are always bias-free in the Python reference.
            lm_heads.push(linear_no_bias(cfg.hidden_size, vocab_size, vb_lm.pp(i))?);
        }

        Ok(Self {
            blocks,
            lm_heads,
            device: vb.device().clone(),
        })
    }
}

impl MedusaDraftModel for MedusaModel {
    fn forward_heads(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        self.blocks
            .iter()
            .map(|b| b.forward(hidden_states))
            .collect()
    }

    fn compute_logits(&self, head_hidden_states: &[Tensor]) -> Result<Vec<Tensor>> {
        head_hidden_states
            .iter()
            .zip(&self.lm_heads)
            .map(|(h, lm)| lm.forward(h))
            .collect()
    }

    fn num_heads(&self) -> usize {
        self.blocks.len()
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn test_config(num_heads: usize, num_layers: usize) -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("num_heads".to_string(), serde_json::json!(num_heads));

        ModelConfig {
            architectures: vec!["MedusaModel".to_string()],
            hidden_size: 32,
            num_hidden_layers: num_layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            vocab_size: 128,
            max_position_embeddings: 64,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    #[test]
    fn test_medusa_construction() {
        let cfg = test_config(3, 1);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MedusaModel::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert_eq!(model.unwrap().num_heads(), 3);
    }

    #[test]
    fn test_medusa_forward_heads_shape() {
        let cfg = test_config(3, 1);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MedusaModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let outputs = model.forward_heads(&hidden).unwrap();

        assert_eq!(outputs.len(), 3);
        for out in &outputs {
            assert_eq!(out.dims(), &[1, 32]);
        }
    }

    #[test]
    fn test_medusa_compute_logits_shape() {
        let cfg = test_config(3, 1);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MedusaModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let head_outputs = model.forward_heads(&hidden).unwrap();
        let logits = model.compute_logits(&head_outputs).unwrap();

        assert_eq!(logits.len(), 3);
        for l in &logits {
            assert_eq!(l.dims(), &[1, 128]); // vocab_size
        }
    }

    #[test]
    fn test_medusa_multi_layer_heads() {
        // 3 residual layers per head
        let cfg = test_config(2, 3);
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MedusaModel::new(&cfg, vb).unwrap();

        assert_eq!(model.num_heads(), 2);
        let hidden = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let outputs = model.forward_heads(&hidden).unwrap();
        assert_eq!(outputs.len(), 2);
        for out in &outputs {
            assert_eq!(out.dims(), &[1, 32]);
        }
    }

    #[test]
    fn test_medusa_truncated_vocab() {
        let mut cfg = test_config(2, 1);
        cfg.extra
            .insert("truncated_vocab_size".to_string(), serde_json::json!(64));

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MedusaModel::new(&cfg, vb).unwrap();

        let hidden = Tensor::zeros((1, 32), DType::F32, &device).unwrap();
        let head_outputs = model.forward_heads(&hidden).unwrap();
        let logits = model.compute_logits(&head_outputs).unwrap();

        assert_eq!(logits.len(), 2);
        for l in &logits {
            assert_eq!(l.dims(), &[1, 64]); // truncated_vocab_size
        }
    }
}
