//! Medusa speculative decoding proposer.
//!
//! Medusa attaches multiple prediction heads to the base model's hidden
//! states. Each head is a small MLP that independently predicts one future
//! token position. Combined, the heads propose K tokens simultaneously.
//!
//! Unlike the draft-model approach, Medusa does not require a separate model
//! or autoregressive generation — all K positions are predicted in parallel
//! from a single set of hidden states. The trade-off is that each head's
//! prediction is independent, so it cannot capture sequential dependencies
//! between the speculated tokens (Eagle addresses this limitation).
//!
//! Reference: "Medusa: Simple LLM Inference Acceleration Framework with
//! Multiple Decoding Heads" (Cai et al., 2024)

use std::sync::Mutex;

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::SpeculativeProposer;

/// A single Medusa prediction head.
///
/// Each head is a small MLP (typically 1 hidden layer + vocabulary projection)
/// that maps the base model's hidden states to logits over the vocabulary.
/// The head predicts the token at one specific future position.
pub struct MedusaHead {
    /// Hidden-to-hidden linear layers.
    layers: Vec<Linear>,
    /// Final projection to vocabulary size.
    lm_head: Linear,
}

impl MedusaHead {
    /// Create a new Medusa head.
    ///
    /// - `hidden_size`: dimension of the base model's hidden states
    /// - `vocab_size`: vocabulary size for the output projection
    /// - `num_layers`: number of hidden layers before the final projection (typically 1)
    /// - `vb`: variable builder for loading/initializing weights
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = linear(hidden_size, hidden_size, vb.pp(format!("layer.{i}")))?;
            layers.push(layer);
        }
        let lm_head = linear(hidden_size, vocab_size, vb.pp("lm_head"))?;

        Ok(Self { layers, lm_head })
    }

    /// Forward pass through the Medusa head.
    ///
    /// Takes hidden states of shape `[batch, hidden_size]` or `[1, seq_len, hidden_size]`
    /// and returns logits of shape `[batch, vocab_size]` or `[1, vocab_size]`.
    ///
    /// Only the last token's hidden state is used when the input has a sequence dimension.
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // If 3D [batch, seq_len, hidden], take the last position
        let mut h = if hidden_states.dims().len() == 3 {
            let seq_len = hidden_states.dim(1)?;
            hidden_states.narrow(1, seq_len - 1, 1)?.squeeze(1)?
        } else {
            hidden_states.clone()
        };

        // Pass through hidden layers with SiLU activation
        for layer in &self.layers {
            h = candle_nn::ops::silu(&layer.forward(&h)?)?;
        }

        // Project to vocabulary
        self.lm_head.forward(&h)
    }
}

/// Medusa speculative decoding proposer.
///
/// Uses multiple prediction heads to propose K tokens simultaneously from
/// a single set of hidden states. Each head predicts one position ahead
/// independently, so the predictions do not capture inter-token dependencies.
///
/// # Usage
///
/// The engine must call [`set_hidden_states`](MedusaProposer::set_hidden_states)
/// with the base model's last hidden states before calling `propose()`.
/// If hidden states are not available, `propose()` returns an empty vec.
///
/// # Thread safety
///
/// Internal state is protected by a `Mutex` to satisfy `Send + Sync`.
pub struct MedusaProposer {
    inner: Mutex<MedusaInner>,
    num_speculative_tokens: usize,
}

struct MedusaInner {
    /// The Medusa prediction heads (one per speculative position).
    heads: Vec<MedusaHead>,
    /// Top-k candidates per head for tree construction.
    top_k: usize,
    /// Last hidden states from the base model.
    last_hidden_states: Option<Tensor>,
}

impl MedusaProposer {
    /// Create a new Medusa proposer.
    ///
    /// - `num_heads`: number of Medusa heads (= number of speculative positions)
    /// - `hidden_size`: hidden dimension of the base model
    /// - `vocab_size`: vocabulary size
    /// - `num_layers_per_head`: number of hidden layers in each head (typically 1)
    /// - `top_k`: number of top candidates per head for tree construction
    /// - `vb`: variable builder for weight initialization
    pub fn new(
        num_heads: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_layers_per_head: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            let head = MedusaHead::new(
                hidden_size,
                vocab_size,
                num_layers_per_head,
                vb.pp(format!("head.{i}")),
            )?;
            heads.push(head);
        }

        Ok(Self {
            inner: Mutex::new(MedusaInner {
                heads,
                top_k,
                last_hidden_states: None,
            }),
            num_speculative_tokens: num_heads,
        })
    }

    /// Set the hidden states from the last model forward pass.
    ///
    /// This must be called before `propose()`. The hidden states should be
    /// the output of the base model's last transformer layer, before the
    /// LM head projection.
    pub fn set_hidden_states(&self, hidden_states: Tensor) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.last_hidden_states = Some(hidden_states);
        }
    }

    /// Run all Medusa heads on the given hidden states and return the top-1
    /// prediction from each head.
    ///
    /// Returns a vec of length `num_heads`, one token per speculative position.
    pub fn propose_from_hidden_states(&self, hidden_states: &Tensor) -> Result<Vec<u32>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("MedusaProposer lock poisoned: {e}")))?;

        let mut proposals = Vec::with_capacity(inner.heads.len());
        for head in &inner.heads {
            let logits = head.forward(hidden_states)?;
            let token = logits.argmax(candle_core::D::Minus1)?.squeeze(0)?;
            let token_id = token.to_scalar::<u32>()?;
            proposals.push(token_id);
        }

        Ok(proposals)
    }

    /// Run all Medusa heads and return top-k candidates per head.
    ///
    /// Returns a vec of length `num_heads`, each containing `top_k` candidate
    /// token IDs. This is used for tree-based verification.
    pub fn propose_top_k_from_hidden_states(
        &self,
        hidden_states: &Tensor,
    ) -> Result<Vec<Vec<u32>>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("MedusaProposer lock poisoned: {e}")))?;

        let mut all_candidates = Vec::with_capacity(inner.heads.len());
        for head in &inner.heads {
            let logits = head.forward(hidden_states)?;
            let logits_1d = logits.squeeze(0)?;
            let logits_vec: Vec<f32> = logits_1d.to_vec1()?;
            let k = inner.top_k.min(logits_vec.len());

            // Partial sort on CPU to find top-k indices
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let candidates: Vec<u32> = indexed[..k].iter().map(|(idx, _)| *idx as u32).collect();
            all_candidates.push(candidates);
        }

        Ok(all_candidates)
    }
}

impl SpeculativeProposer for MedusaProposer {
    fn propose(&self, _token_ids: &[u32], max_tokens: usize) -> Vec<u32> {
        let k = max_tokens.min(self.num_speculative_tokens);
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
        "medusa"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    const HIDDEN_SIZE: usize = 64;
    const VOCAB_SIZE: usize = 256;
    const NUM_HEADS: usize = 3;
    const TOP_K: usize = 2;

    fn test_vb() -> VarBuilder<'static> {
        VarBuilder::zeros(DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_medusa_head_forward_shape() {
        let vb = test_vb();
        let head = MedusaHead::new(HIDDEN_SIZE, VOCAB_SIZE, 1, vb.pp("test_head")).unwrap();

        // Input: [1, hidden_size]
        let input = Tensor::zeros((1, HIDDEN_SIZE), DType::F32, &Device::Cpu).unwrap();
        let output = head.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, VOCAB_SIZE]);
    }

    #[test]
    fn test_medusa_head_forward_3d_input() {
        let vb = test_vb();
        let head = MedusaHead::new(HIDDEN_SIZE, VOCAB_SIZE, 1, vb.pp("test_head")).unwrap();

        // Input: [1, 5, hidden_size] — should take last position
        let input = Tensor::zeros((1, 5, HIDDEN_SIZE), DType::F32, &Device::Cpu).unwrap();
        let output = head.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, VOCAB_SIZE]);
    }

    #[test]
    fn test_medusa_head_multi_layer() {
        let vb = test_vb();
        let head = MedusaHead::new(HIDDEN_SIZE, VOCAB_SIZE, 3, vb.pp("test_head")).unwrap();

        let input = Tensor::zeros((1, HIDDEN_SIZE), DType::F32, &Device::Cpu).unwrap();
        let output = head.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, VOCAB_SIZE]);
    }

    #[test]
    fn test_medusa_proposer_creation() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        assert_eq!(proposer.num_speculative_tokens, NUM_HEADS);
        assert_eq!(proposer.name(), "medusa");
    }

    #[test]
    fn test_medusa_propose_with_hidden_states() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        // Create some hidden states
        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        let result = proposer.propose(&[1, 2, 3], 5);

        // Should return up to NUM_HEADS proposals
        assert!(!result.is_empty());
        assert!(result.len() <= NUM_HEADS);

        // All tokens should be valid vocab indices
        for &token in &result {
            assert!((token as usize) < VOCAB_SIZE);
        }
    }

    #[test]
    fn test_medusa_propose_without_hidden_states() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        // No hidden states set -> should return empty
        let result = proposer.propose(&[1, 2, 3], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_medusa_propose_respects_max_tokens() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        // Request fewer tokens than num_heads
        let result = proposer.propose(&[1, 2, 3], 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_medusa_propose_zero_max_tokens() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();
        proposer.set_hidden_states(hidden);

        let result = proposer.propose(&[1, 2, 3], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_medusa_multiple_heads() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        // Use non-zero hidden states so heads may produce different outputs
        // (with zero weights they'll all produce the same, but the structure is correct)
        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();

        let proposals = proposer.propose_from_hidden_states(&hidden).unwrap();

        // Should have one proposal per head
        assert_eq!(proposals.len(), NUM_HEADS);

        // All proposals should be valid token IDs
        for &token in &proposals {
            assert!((token as usize) < VOCAB_SIZE);
        }
    }

    #[test]
    fn test_medusa_top_k_candidates() {
        let vb = test_vb();
        let proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        let hidden = Tensor::randn(0.0f32, 1.0, (1, HIDDEN_SIZE), &Device::Cpu).unwrap();

        let candidates = proposer.propose_top_k_from_hidden_states(&hidden).unwrap();

        assert_eq!(candidates.len(), NUM_HEADS);
        for head_candidates in &candidates {
            assert_eq!(head_candidates.len(), TOP_K);
            for &token in head_candidates {
                assert!((token as usize) < VOCAB_SIZE);
            }
        }
    }

    #[test]
    fn test_medusa_on_tokens_accepted_noop() {
        let vb = test_vb();
        let mut proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        // Should not panic
        proposer.on_tokens_accepted(42, &[1, 2, 3]);
    }

    #[test]
    fn test_medusa_on_request_finished_noop() {
        let vb = test_vb();
        let mut proposer =
            MedusaProposer::new(NUM_HEADS, HIDDEN_SIZE, VOCAB_SIZE, 1, TOP_K, vb).unwrap();

        // Should not panic
        proposer.on_request_finished(42);
    }
}
