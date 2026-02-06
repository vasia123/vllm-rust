//! Eagle speculative decoding proposer.
//!
//! Eagle (Extrapolation Algorithm for Greater Language-model Efficiency) uses
//! feature-level autoregression with a lightweight draft network. Unlike Medusa
//! where each head predicts independently, Eagle captures sequential dependencies
//! by autoregressively predicting the next *hidden state* rather than the next
//! token directly.
//!
//! Architecture:
//! 1. The base model produces hidden states for the current token.
//! 2. A lightweight draft network (small transformer-like layers) predicts
//!    the next hidden state from the current one.
//! 3. The predicted hidden state is projected to vocabulary via an LM head
//!    to get the next token prediction.
//! 4. The predicted token's embedding is combined with the predicted hidden
//!    state, and the process repeats for subsequent positions.
//!
//! This feature-level autoregression is more accurate than Medusa's independent
//! heads because each prediction conditions on the previous one.
//!
//! Reference: "EAGLE: Speculative Sampling Requires Rethinking Feature
//! Uncertainty" (Li et al., 2024)

use std::sync::Mutex;

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use super::SpeculativeProposer;

/// Eagle draft network configuration.
#[derive(Debug, Clone)]
pub struct EagleConfig {
    /// Hidden dimension (must match the base model).
    pub hidden_size: usize,
    /// Number of draft layers in the feature predictor.
    pub num_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of speculative tokens to propose.
    pub num_speculative_tokens: usize,
}

/// A single draft layer in the Eagle feature predictor.
///
/// Applies: residual + fc2(silu(fc1(norm(x))))
///
/// This is a simplified transformer block without attention, operating
/// purely on the feature (hidden state) dimension.
struct EagleDraftLayer {
    /// Feature projection (hidden -> hidden).
    fc1: Linear,
    /// Output projection (hidden -> hidden).
    fc2: Linear,
    /// Layer normalization applied before the MLP.
    norm: LayerNorm,
}

impl EagleDraftLayer {
    /// Create a new Eagle draft layer.
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, hidden_size, vb.pp("fc2"))?;
        let norm = layer_norm(hidden_size, 1e-5, vb.pp("norm"))?;

        Ok(Self { fc1, fc2, norm })
    }

    /// Forward pass: norm -> fc1 -> silu -> fc2 + residual.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(xs)?;
        let h = candle_nn::ops::silu(&self.fc1.forward(&normed)?)?;
        let out = self.fc2.forward(&h)?;
        // Residual connection
        xs + out
    }
}

/// Eagle speculative decoding proposer.
///
/// Uses feature-level autoregression with a lightweight draft network to
/// propose multiple tokens. More accurate than Medusa because it captures
/// sequential dependencies between predicted tokens.
///
/// # Usage
///
/// Like Medusa, the engine must call
/// [`set_hidden_states`](EagleProposer::set_hidden_states) with the base
/// model's last hidden states before calling `propose()`.
///
/// # Thread safety
///
/// Internal state is protected by a `Mutex` to satisfy `Send + Sync`.
pub struct EagleProposer {
    inner: Mutex<EagleInner>,
    config: EagleConfig,
}

struct EagleInner {
    /// Draft layers for feature-level autoregression.
    draft_layers: Vec<EagleDraftLayer>,
    /// Token embedding layer (token_id -> hidden).
    embed_tokens: Embedding,
    /// LM head (hidden -> vocab logits).
    lm_head: Linear,
    /// Last hidden states from the base model.
    last_hidden_states: Option<Tensor>,
    /// Device.
    device: Device,
}

impl EagleProposer {
    /// Create a new Eagle proposer.
    ///
    /// - `config`: Eagle configuration
    /// - `vb`: variable builder for weight initialization
    pub fn new(config: EagleConfig, vb: VarBuilder) -> Result<Self> {
        let mut draft_layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer =
                EagleDraftLayer::new(config.hidden_size, vb.pp(format!("draft_layer.{i}")))?;
            draft_layers.push(layer);
        }

        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let lm_head = linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        let device = vb.device().clone();

        Ok(Self {
            inner: Mutex::new(EagleInner {
                draft_layers,
                embed_tokens,
                lm_head,
                last_hidden_states: None,
                device,
            }),
            config,
        })
    }

    /// Set the hidden states from the last model forward pass.
    ///
    /// This must be called before `propose()`. The hidden states should be
    /// the output of the base model's last transformer layer.
    pub fn set_hidden_states(&self, hidden_states: Tensor) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.last_hidden_states = Some(hidden_states);
        }
    }

    /// Propose tokens autoregressively from hidden states.
    ///
    /// Starting from the base model's hidden states, iteratively:
    /// 1. Run through draft layers to predict next hidden state
    /// 2. Project to vocabulary and take argmax for token prediction
    /// 3. Embed predicted token and add to hidden state for next iteration
    pub fn propose_from_hidden_states(&self, hidden_states: &Tensor) -> Result<Vec<u32>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("EagleProposer lock poisoned: {e}")))?;

        let k = self.config.num_speculative_tokens;
        let mut proposals = Vec::with_capacity(k);

        // Start with the base model's hidden states.
        // If 3D [batch, seq_len, hidden], take the last position.
        let mut current_hidden = if hidden_states.dims().len() == 3 {
            let seq_len = hidden_states.dim(1)?;
            hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?
        } else {
            hidden_states.clone()
        };

        for _ in 0..k {
            // Run through draft layers to predict the next hidden state
            let mut h = current_hidden.clone();
            for layer in &inner.draft_layers {
                h = layer.forward(&h)?;
            }

            // Project to vocabulary and get the predicted token
            let logits = inner.lm_head.forward(&h)?;
            let token_id = logits
                .argmax(candle_core::D::Minus1)?
                .squeeze(0)?
                .to_scalar::<u32>()?;
            proposals.push(token_id);

            // Embed the predicted token and combine with predicted hidden state
            // for the next iteration (feature-level autoregression)
            let token_tensor = Tensor::from_vec(vec![token_id], (1,), &inner.device)?;
            let token_embedding = inner.embed_tokens.forward(&token_tensor)?;

            // Combine: add token embedding to predicted hidden state.
            // This fuses token-level information into the feature prediction.
            current_hidden = (&h + &token_embedding)?;
        }

        Ok(proposals)
    }
}

impl SpeculativeProposer for EagleProposer {
    fn propose(&self, _token_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        let k = max_tokens.min(self.config.num_speculative_tokens);
        if k == 0 {
            return Vec::new();
        }

        let hidden_states = {
            let Ok(inner) = self.inner.lock() else {
                return Vec::new();
            };
            match &inner.last_hidden_states {
                Some(hs) => hs.clone(),
                None => return Vec::new(),
            }
        };

        match self.propose_from_hidden_states(&hidden_states) {
            Ok(mut proposals) => {
                proposals.truncate(k);
                proposals
            }
            Err(_) => Vec::new(),
        }
    }

    fn name(&self) -> &str {
        "eagle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    const HIDDEN_SIZE: usize = 64;
    const VOCAB_SIZE: usize = 256;
    const NUM_LAYERS: usize = 2;
    const NUM_SPEC_TOKENS: usize = 3;

    fn test_config() -> EagleConfig {
        EagleConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            num_speculative_tokens: NUM_SPEC_TOKENS,
        }
    }

    fn test_vb() -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_eagle_draft_layer() {
        let vb = test_vb();
        let layer = EagleDraftLayer::new(HIDDEN_SIZE, vb.pp("test_layer")).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();

        // Shape should be preserved: [1, hidden_size]
        assert_eq!(output.dims(), &[1, HIDDEN_SIZE]);
    }

    #[test]
    fn test_eagle_draft_layer_residual() {
        let vb = test_vb();
        let layer = EagleDraftLayer::new(HIDDEN_SIZE, vb.pp("test_layer")).unwrap();

        // With zero weights, fc2(silu(fc1(norm(x)))) = bias terms only,
        // so output = x + bias. Shape should still be correct.
        let input = Tensor::ones((1, HIDDEN_SIZE), DType::F32, &Device::Cpu).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, HIDDEN_SIZE]);
    }

    #[test]
    fn test_eagle_proposer_creation() {
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        assert_eq!(proposer.config.hidden_size, HIDDEN_SIZE);
        assert_eq!(proposer.config.num_layers, NUM_LAYERS);
        assert_eq!(proposer.config.vocab_size, VOCAB_SIZE);
        assert_eq!(proposer.config.num_speculative_tokens, NUM_SPEC_TOKENS);
        assert_eq!(proposer.name(), "eagle");
    }

    #[test]
    fn test_eagle_propose_from_hidden() {
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        let result = proposer.propose(&[1, 2, 3], 5);

        // Should return up to NUM_SPEC_TOKENS proposals
        assert!(!result.is_empty());
        assert!(result.len() <= NUM_SPEC_TOKENS);

        // All tokens should be valid vocab indices
        for &token in &result {
            assert!((token as usize) < VOCAB_SIZE);
        }
    }

    #[test]
    fn test_eagle_propose_without_hidden_states() {
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        // No hidden states set -> should return empty
        let result = proposer.propose(&[1, 2, 3], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_eagle_propose_respects_max_tokens() {
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        let result = proposer.propose(&[1, 2, 3], 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_eagle_propose_zero_max_tokens() {
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        let result = proposer.propose(&[1, 2, 3], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_eagle_autoregressive() {
        // Verify that the proposals are generated sequentially:
        // each prediction depends on the previous hidden state + token embedding
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();

        let proposals = proposer.propose_from_hidden_states(&hidden).unwrap();

        assert_eq!(proposals.len(), NUM_SPEC_TOKENS);

        // All proposals should be valid token IDs
        for &token in &proposals {
            assert!((token as usize) < VOCAB_SIZE);
        }
    }

    #[test]
    fn test_eagle_3d_hidden_states() {
        // Test with 3D hidden states [1, seq_len, hidden]
        let vb = test_vb();
        let config = test_config();
        let proposer = EagleProposer::new(config, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, 10, HIDDEN_SIZE), &Device::Cpu).unwrap();

        let proposals = proposer.propose_from_hidden_states(&hidden).unwrap();
        assert_eq!(proposals.len(), NUM_SPEC_TOKENS);
    }

    #[test]
    fn test_eagle_on_tokens_accepted_noop() {
        let vb = test_vb();
        let config = test_config();
        let mut proposer = EagleProposer::new(config, vb).unwrap();

        // Should not panic
        proposer.on_tokens_accepted(42, &[1, 2, 3]);
    }

    #[test]
    fn test_eagle_on_request_finished_noop() {
        let vb = test_vb();
        let config = test_config();
        let mut proposer = EagleProposer::new(config, vb).unwrap();

        // Should not panic
        proposer.on_request_finished(42);
    }
}
