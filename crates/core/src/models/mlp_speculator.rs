//! MLP Speculator draft model for speculative decoding.
//!
//! A lightweight MLP-based draft model that generates K speculative tokens
//! from the target model's hidden states. Each prediction head sequentially
//! embeds the previous predicted token, projects the hidden state, combines
//! them, and predicts the next token's logits.
//!
//! Unlike Medusa (independent parallel heads), MLP Speculator chains
//! predictions sequentially — each head's output feeds the next — capturing
//! inter-token dependencies without autoregressive attention.
//!
//! Reference: "Accelerating Production LLMs with Combined Token/Embedding
//! Speculators" (https://arxiv.org/pdf/2404.19124)
//!
//! Trained speculators available at: https://huggingface.co/ibm-granite

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

use crate::config::ModelConfig;

// ─── L2 Layer Norm ──────────────────────────────────────────────────────────

/// Custom L2 normalization with optional learnable scale and shift.
///
/// Normalizes input by its L2 norm (RMS-style), then optionally applies
/// elementwise affine transform: `y = weight * (x / rms(x)) + bias`.
struct MLPSpeculatorLayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl MLPSpeculatorLayerNorm {
    fn new(dim: usize, elementwise_scale_and_shift: bool, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if elementwise_scale_and_shift {
            let w = vb.get(dim, "weight")?;
            let b = vb.get(dim, "bias")?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // L2 normalization: x * rsqrt(mean(x^2) + eps)
        let mean_sq = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let norm = (mean_sq + self.eps)?.sqrt()?;
        let mut out = x.broadcast_div(&norm)?;

        if let (Some(w), Some(b)) = (&self.weight, &self.bias) {
            out = out.broadcast_mul(w)?;
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }
}

// ─── MLP Speculator Model ───────────────────────────────────────────────────

/// MLP Speculator draft model.
///
/// Generates K speculative tokens from the target model's last hidden states.
/// Each head sequentially:
/// 1. Embeds the previously predicted token
/// 2. Projects the hidden state
/// 3. Combines embedding and projection with decay weights
/// 4. Normalizes with GELU activation
/// 5. Projects to vocabulary logits
///
/// The `state_weight` decays with prediction depth to reduce confidence
/// in later predictions, while `emb_weight` balances the embedding contribution.
pub struct MLPSpeculatorModel {
    /// Token embeddings per head (shared if tie_weights).
    emb: Vec<Embedding>,
    /// Hidden state projections per head (first may differ in size).
    proj: Vec<Linear>,
    /// L2 layer norms per head.
    ln: Vec<MLPSpeculatorLayerNorm>,
    /// LM heads per head (always separate for logit diversity).
    head: Vec<Linear>,
    /// Optional input scaling norm (L2 without affine).
    ln0: Option<MLPSpeculatorLayerNorm>,
    /// Decay weight for hidden state contribution.
    state_weight: f64,
    /// Scale weight for embedding contribution.
    emb_weight: f64,
    /// Number of speculative tokens to generate.
    n_predict: usize,
    device: Device,
    dtype: DType,
}

impl MLPSpeculatorModel {
    /// Create a new MLP Speculator from config values.
    ///
    /// Config fields read from `ModelConfig.extra`:
    /// - `emb_dim`: embedding dimension of target model (defaults to hidden_size)
    /// - `inner_dim`: internal dimension (defaults to emb_dim if 0)
    /// - `n_predict`: number of lookahead tokens (defaults to 3)
    /// - `tie_weights`: share weights across heads (defaults to false)
    /// - `scale_input`: apply L2 norm to input hidden states (defaults to false)
    pub fn new(
        vocab_size: usize,
        emb_dim: usize,
        inner_dim: usize,
        n_predict: usize,
        tie_weights: bool,
        scale_input: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = if inner_dim == 0 { emb_dim } else { inner_dim };

        let mut emb = Vec::with_capacity(n_predict);
        let mut proj = Vec::with_capacity(n_predict);
        let mut ln = Vec::with_capacity(n_predict);
        let mut head = Vec::with_capacity(n_predict);

        if tie_weights && n_predict > 1 {
            // Shared embedding (single set of weights reused)
            let shared_emb = embedding(vocab_size, inner_dim, vb.pp("emb.0"))?;
            for _ in 0..n_predict {
                emb.push(shared_emb.clone());
            }

            // First projection: emb_dim → inner_dim
            let proj_first = linear_no_bias(emb_dim, inner_dim, vb.pp("proj.0"))?;
            // Subsequent projections: inner_dim → inner_dim (shared)
            let proj_tied = linear_no_bias(inner_dim, inner_dim, vb.pp("proj.1"))?;
            proj.push(proj_first);
            for _ in 1..n_predict {
                proj.push(proj_tied.clone());
            }

            // Shared layer norm
            let shared_ln = MLPSpeculatorLayerNorm::new(inner_dim, true, vb.pp("ln.0"))?;
            // First head uses the shared ln, but we need to store n_predict refs
            // Since MLPSpeculatorLayerNorm doesn't impl Clone, create separate instances
            // that share the same weight path for tied weights
            ln.push(shared_ln);
            for i in 1..n_predict {
                ln.push(MLPSpeculatorLayerNorm::new(
                    inner_dim,
                    true,
                    vb.pp(format!("ln.{i}")),
                )?);
            }
        } else {
            // Independent weights per head
            for i in 0..n_predict {
                emb.push(embedding(vocab_size, inner_dim, vb.pp(format!("emb.{i}")))?);

                let in_dim = if i == 0 { emb_dim } else { inner_dim };
                proj.push(linear_no_bias(
                    in_dim,
                    inner_dim,
                    vb.pp(format!("proj.{i}")),
                )?);

                ln.push(MLPSpeculatorLayerNorm::new(
                    inner_dim,
                    true,
                    vb.pp(format!("ln.{i}")),
                )?);
            }
        }

        // LM heads are always separate (for output diversity)
        for i in 0..n_predict {
            head.push(linear_no_bias(
                inner_dim,
                vocab_size,
                vb.pp(format!("head.{i}")),
            )?);
        }

        // Optional input scaling
        let ln0 = if scale_input {
            Some(MLPSpeculatorLayerNorm::new(emb_dim, false, vb.pp("ln0"))?)
        } else {
            None
        };

        // Decay weights from paper
        let state_weight = 0.5f64.powf(0.5 / n_predict as f64);
        let emb_weight = ((1.0 - state_weight * state_weight) * (inner_dim as f64 / 2.0)).sqrt();

        Ok(Self {
            emb,
            proj,
            ln,
            head,
            ln0,
            state_weight,
            emb_weight,
            n_predict,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Generate speculative logits from target model hidden states.
    ///
    /// - `hidden_states`: last hidden states from target model `[batch, hidden_size]`
    /// - `last_token_id`: the last accepted token ID (used to start the chain)
    ///
    /// Returns a Vec of `n_predict` logit tensors, each `[batch, vocab_size]`.
    pub fn forward(&self, hidden_states: &Tensor, last_token_id: u32) -> Result<Vec<Tensor>> {
        let sqrt2 = std::f64::consts::SQRT_2;

        // Optionally scale input
        let mut states = if let Some(ln0) = &self.ln0 {
            (ln0.forward(hidden_states)? / sqrt2)?
        } else {
            hidden_states.clone()
        };

        // Unsqueeze for consistent 3D [batch, 1, dim] processing
        if states.dims().len() == 2 {
            states = states.unsqueeze(1)?;
        }

        let mut last_token = last_token_id;
        let mut logits_list = Vec::with_capacity(self.n_predict);

        for i in 0..self.n_predict {
            // Embed previous token: [1] → [1, 1, inner_dim]
            let token_tensor = Tensor::new(&[last_token], &self.device)?;
            let z = self.emb[i].forward(&token_tensor)?.unsqueeze(0)?; // [1, 1, inner_dim]

            // Project hidden state: [batch, 1, emb_dim] → [batch, 1, inner_dim]
            let projected = self.proj[i].forward(&states)?;

            // Weighted combination: state_weight * projected + emb_weight * z
            let scale = self.emb_weight / self.state_weight;
            let combined = (projected + (z * scale)?)?;

            // Normalize + GELU activation
            let normed = self.ln[i].forward(&combined)?;
            states = normed.gelu_erf()?;

            // Project to vocabulary: [batch, 1, inner_dim] → [batch, 1, vocab_size]
            let head_logits = self.head[i].forward(&states)?;

            // Squeeze sequence dim: [batch, vocab_size]
            let head_logits = head_logits.squeeze(1)?;
            logits_list.push(head_logits.clone());

            // Greedy sample for the next token in the chain
            let predicted = head_logits.argmax(candle_core::D::Minus1)?;
            last_token = predicted.to_vec1::<u32>()?[0];
        }

        Ok(logits_list)
    }

    /// Generate speculative token IDs by greedy argmax at each head.
    pub fn propose(&self, hidden_states: &Tensor, last_token_id: u32) -> Result<Vec<u32>> {
        let logits_list = self.forward(hidden_states, last_token_id)?;

        let mut tokens = Vec::with_capacity(logits_list.len());
        for logits in &logits_list {
            let token = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Build from a `ModelConfig`, reading speculator params from `extra`.
    ///
    /// HF checkpoints store all weights under a `speculator.*` prefix, so this
    /// passes `vb.pp("speculator")` to the inner constructor.
    ///
    /// Config fields from `ModelConfig::extra`:
    /// - `emb_dim` — input dim from target model (defaults to `hidden_size`)
    /// - `inner_dim` — speculator hidden dim (0 → same as `emb_dim`)
    /// - `n_predict` — draft tokens per step (default 3)
    /// - `tie_weights` — share emb/proj/ln across heads (default false)
    /// - `scale_input` — L2-normalise input before projection (default false)
    pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let get_usize = |key: &str, default: usize| -> usize {
            cfg.extra
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_bool = |key: &str| -> bool {
            cfg.extra
                .get(key)
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        };

        let emb_dim = get_usize("emb_dim", cfg.hidden_size);
        let inner_dim = get_usize("inner_dim", 0);
        let n_predict = get_usize("n_predict", 3);
        let tie_weights = get_bool("tie_weights");
        let scale_input = get_bool("scale_input");

        // HF checkpoints prefix all weights with "speculator."
        Self::new(
            cfg.vocab_size,
            emb_dim,
            inner_dim,
            n_predict,
            tie_weights,
            scale_input,
            vb.pp("speculator"),
        )
    }

    pub fn n_predict(&self) -> usize {
        self.n_predict
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB_SIZE: usize = 256;
    const EMB_DIM: usize = 64;
    const INNER_DIM: usize = 32;
    const N_PREDICT: usize = 3;

    fn test_vb() -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_construction() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb);
        assert!(
            model.is_ok(),
            "MLPSpeculatorModel should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_construction_tied_weights() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, true, false, vb);
        assert!(
            model.is_ok(),
            "Tied-weight model should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_construction_with_scale_input() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, true, vb);
        assert!(
            model.is_ok(),
            "Scale-input model should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_construction_inner_dim_zero() {
        let vb = test_vb();
        // inner_dim=0 should default to emb_dim
        let model = MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, 0, N_PREDICT, false, false, vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_returns_n_predict_logits() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb)
                .expect("build");

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let logits = model.forward(&hidden, 0).expect("forward");

        assert_eq!(logits.len(), N_PREDICT, "should return N logit tensors");
        for (i, l) in logits.iter().enumerate() {
            assert_eq!(
                l.dims(),
                &[1, VOCAB_SIZE],
                "head {i} logits should be [1, vocab_size]"
            );
        }
    }

    #[test]
    fn test_forward_tied_weights() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, true, false, vb)
                .expect("build");

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let logits = model.forward(&hidden, 0).expect("forward");
        assert_eq!(logits.len(), N_PREDICT);
    }

    #[test]
    fn test_forward_with_scale_input() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, true, vb)
                .expect("build");

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let logits = model.forward(&hidden, 0).expect("forward");
        assert_eq!(logits.len(), N_PREDICT);
    }

    #[test]
    fn test_propose_returns_token_ids() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb)
                .expect("build");

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let tokens = model.propose(&hidden, 0).expect("propose");

        assert_eq!(tokens.len(), N_PREDICT);
        for &t in &tokens {
            assert!(
                (t as usize) < VOCAB_SIZE,
                "token {t} should be < {VOCAB_SIZE}"
            );
        }
    }

    #[test]
    fn test_decay_weights() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb)
                .expect("build");

        // state_weight should be 0.5^(0.5/N) which is between 0 and 1
        assert!(model.state_weight > 0.0 && model.state_weight < 1.0);
        assert!(model.emb_weight > 0.0);
    }

    #[test]
    fn test_n_predict_accessor() {
        let vb = test_vb();
        let model = MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, 5, false, false, vb)
            .expect("build");
        assert_eq!(model.n_predict(), 5);
    }

    #[test]
    fn test_3d_hidden_states_input() {
        let vb = test_vb();
        let model =
            MLPSpeculatorModel::new(VOCAB_SIZE, EMB_DIM, INNER_DIM, N_PREDICT, false, false, vb)
                .expect("build");

        // 3D input should work (unsqueeze is handled)
        let hidden = Tensor::zeros((1, 1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let logits = model.forward(&hidden, 0).expect("forward");
        assert_eq!(logits.len(), N_PREDICT);
    }

    fn mlp_speculator_config() -> crate::config::ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "emb_dim".into(),
            serde_json::Value::Number(serde_json::Number::from(EMB_DIM)),
        );
        extra.insert(
            "inner_dim".into(),
            serde_json::Value::Number(serde_json::Number::from(INNER_DIM)),
        );
        extra.insert(
            "n_predict".into(),
            serde_json::Value::Number(serde_json::Number::from(N_PREDICT)),
        );
        crate::config::ModelConfig {
            architectures: vec!["MLPSpeculatorPreTrainedModel".to_string()],
            hidden_size: EMB_DIM,
            vocab_size: VOCAB_SIZE,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 0,
            head_dim: 16,
            max_position_embeddings: 512,
            hidden_act: "gelu".to_string(),
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
    fn test_from_config_construction() {
        let cfg = mlp_speculator_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = MLPSpeculatorModel::from_config(&cfg, vb);
        assert!(
            model.is_ok(),
            "from_config should construct: {:?}",
            model.err()
        );
        let model = model.unwrap();
        assert_eq!(model.n_predict(), N_PREDICT);
    }

    #[test]
    fn test_from_config_forward() {
        let cfg = mlp_speculator_config();
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = MLPSpeculatorModel::from_config(&cfg, vb).expect("build");

        let hidden = Tensor::zeros((1, EMB_DIM), DType::F32, &Device::Cpu).expect("hidden");
        let logits = model.forward(&hidden, 0).expect("forward");
        assert_eq!(logits.len(), N_PREDICT);
        for (i, l) in logits.iter().enumerate() {
            assert_eq!(
                l.dims(),
                &[1, VOCAB_SIZE],
                "head {i} should be [1, vocab_size]"
            );
        }
    }
}
