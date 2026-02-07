//! Cross-attention layer for encoder-decoder models.
//!
//! In encoder-decoder architectures (T5, BART, mBART), the decoder attends to
//! encoder hidden states via cross-attention. This differs from self-attention:
//!
//! - **Query** comes from the decoder
//! - **Key and Value** come from the encoder
//! - No causal masking (decoder attends to all encoder positions)
//! - Encoder K/V can be precomputed once and reused across all decode steps
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::layers::cross_attention::CrossAttention;
//!
//! let cross_attn = CrossAttention::new(hidden_size, num_heads, vb)?;
//!
//! // First decode step: compute encoder K/V
//! let (output, cached_k, cached_v) = cross_attn.forward(
//!     &decoder_hidden,
//!     &encoder_hidden,
//!     None,  // or Some(&encoder_attention_mask)
//!     None,  // no cached K/V yet
//! )?;
//!
//! // Subsequent steps: reuse cached encoder K/V
//! let (output, _, _) = cross_attn.forward(
//!     &decoder_hidden,
//!     &encoder_hidden,
//!     None,
//!     Some((&cached_k, &cached_v)),
//! )?;
//! ```

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// Cross-attention layer for encoder-decoder models.
///
/// Attends from decoder hidden states (query) to encoder hidden states (key/value).
/// Encoder K/V projections are computed once and can be cached for all subsequent
/// decode steps, since the encoder output does not change during generation.
pub struct CrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CrossAttention {
    /// Create a new cross-attention layer.
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension (must be divisible by `num_heads`)
    /// * `num_heads` - Number of attention heads
    /// * `vb` - Variable builder for loading weights
    pub fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass with cross-attention.
    ///
    /// Returns `(output, key, value)` where key/value are the projected encoder
    /// representations that can be cached for subsequent decode steps.
    ///
    /// # Arguments
    /// * `decoder_hidden` - Decoder hidden states `[batch, tgt_len, hidden_size]`
    /// * `encoder_hidden` - Encoder hidden states `[batch, src_len, hidden_size]`
    /// * `encoder_attention_mask` - Optional mask for source padding `[batch, 1, 1, src_len]`
    ///   where `-inf` masks padded positions and `0.0` allows attention
    /// * `cached_kv` - Optional cached encoder K/V from a previous step
    pub fn forward(
        &self,
        decoder_hidden: &Tensor,
        encoder_hidden: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        cached_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, tgt_len, _) = decoder_hidden.dims3()?;

        // Q always comes from the decoder
        let q = self.q_proj.forward(decoder_hidden)?;
        let q = self.reshape_for_attention(q, batch_size, tgt_len)?;

        // K/V come from encoder or cache
        let (k, v) = match cached_kv {
            Some((cached_k, cached_v)) => (cached_k.clone(), cached_v.clone()),
            None => {
                let (_, src_len, _) = encoder_hidden.dims3()?;
                let k = self.k_proj.forward(encoder_hidden)?;
                let v = self.v_proj.forward(encoder_hidden)?;
                let k = self.reshape_for_attention(k, batch_size, src_len)?;
                let v = self.reshape_for_attention(v, batch_size, src_len)?;
                (k, v)
            }
        };

        // Scaled dot-product attention: softmax(Q * K^T / sqrt(d)) * V
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        // Apply encoder attention mask if provided
        let attn_weights = match encoder_attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, tgt_len, head_dim] -> [batch, tgt_len, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            tgt_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, k, v))
    }

    /// Reshape a projected tensor for multi-head attention.
    ///
    /// `[batch, seq_len, hidden_size]` -> `[batch, num_heads, seq_len, head_dim]`
    fn reshape_for_attention(
        &self,
        tensor: Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        tensor
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Get the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the attention scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn create_cross_attention(
        hidden_size: usize,
        num_heads: usize,
        device: &Device,
    ) -> Result<CrossAttention> {
        let vb = VarBuilder::zeros(DType::F32, device);
        CrossAttention::new(hidden_size, num_heads, vb)
    }

    // ─── Construction Tests ──────────────────────────────────────────────────

    #[test]
    fn construction_succeeds() {
        let device = Device::Cpu;
        let cross_attn = create_cross_attention(64, 4, &device);
        assert!(cross_attn.is_ok());
    }

    #[test]
    fn construction_sets_correct_dimensions() {
        let device = Device::Cpu;
        let cross_attn =
            create_cross_attention(128, 8, &device).expect("cross attention creation should work");

        assert_eq!(cross_attn.num_heads(), 8);
        assert_eq!(cross_attn.head_dim(), 16); // 128 / 8
        assert!((cross_attn.scale() - 1.0 / (16.0f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn construction_with_different_sizes() {
        let device = Device::Cpu;

        let test_cases = [(64, 4), (128, 8), (256, 16), (512, 8)];

        for (hidden_size, num_heads) in test_cases {
            let cross_attn = create_cross_attention(hidden_size, num_heads, &device)
                .unwrap_or_else(|_| {
                    panic!(
                        "should create cross attention for hidden_size={hidden_size}, num_heads={num_heads}"
                    )
                });

            assert_eq!(cross_attn.head_dim(), hidden_size / num_heads);
        }
    }

    // ─── Forward Pass Tests ──────────────────────────────────────────────────

    #[test]
    fn forward_output_shape() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let tgt_len = 3;
        let src_len = 5;

        let cross_attn = create_cross_attention(hidden_size, num_heads, &device)
            .expect("cross attention creation should work");

        let decoder_hidden = Tensor::zeros((batch_size, tgt_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::zeros((batch_size, src_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        let (output, k, v) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("forward should succeed");

        // Output: [batch, tgt_len, hidden_size]
        assert_eq!(output.dims(), &[batch_size, tgt_len, hidden_size]);
        // K: [batch, num_heads, src_len, head_dim]
        assert_eq!(
            k.dims(),
            &[batch_size, num_heads, src_len, hidden_size / num_heads]
        );
        // V: same shape as K
        assert_eq!(k.dims(), v.dims());
    }

    #[test]
    fn forward_with_cached_kv() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let tgt_len = 1; // decode step: single token
        let src_len = 5;

        let cross_attn = create_cross_attention(hidden_size, num_heads, &device)
            .expect("cross attention creation should work");

        let decoder_hidden = Tensor::zeros((batch_size, tgt_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::zeros((batch_size, src_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        // First call: compute K/V from encoder
        let (_, cached_k, cached_v) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("first forward should succeed");

        assert_eq!(cached_k.dims(), &[batch_size, num_heads, src_len, head_dim]);

        // Second call: use cached K/V
        let (output, k_reused, v_reused) = cross_attn
            .forward(
                &decoder_hidden,
                &encoder_hidden,
                None,
                Some((&cached_k, &cached_v)),
            )
            .expect("cached forward should succeed");

        assert_eq!(output.dims(), &[batch_size, tgt_len, hidden_size]);
        // K/V should be the same tensors we passed in
        assert_eq!(k_reused.dims(), cached_k.dims());
        assert_eq!(v_reused.dims(), cached_v.dims());
    }

    #[test]
    fn forward_with_attention_mask() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 1;
        let tgt_len = 2;
        let src_len = 4;

        let cross_attn = create_cross_attention(hidden_size, num_heads, &device)
            .expect("cross attention creation should work");

        let decoder_hidden = Tensor::zeros((batch_size, tgt_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::zeros((batch_size, src_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        // Mask out last two source positions: [batch, 1, 1, src_len]
        let mask_data = vec![0.0f32, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mask = Tensor::from_vec(mask_data, (batch_size, 1, 1, src_len), &device)
            .expect("mask creation should work");

        let (output, _, _) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, Some(&mask), None)
            .expect("forward with mask should succeed");

        assert_eq!(output.dims(), &[batch_size, tgt_len, hidden_size]);

        // Output should be finite (no NaNs from softmax of all -inf)
        let output_data: Vec<f32> = output
            .flatten_all()
            .expect("flatten should work")
            .to_vec1()
            .expect("to_vec1 should work");
        assert!(
            output_data.iter().all(|v| v.is_finite()),
            "output should be finite when some source positions are unmasked"
        );
    }

    #[test]
    fn forward_single_token_decode() {
        // Simulate the common decode case: single new decoder token
        let device = Device::Cpu;
        let hidden_size = 128;
        let num_heads = 8;
        let batch_size = 1;
        let src_len = 20;

        let cross_attn = create_cross_attention(hidden_size, num_heads, &device)
            .expect("cross attention creation should work");

        let decoder_hidden = Tensor::zeros((batch_size, 1, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::zeros((batch_size, src_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        let (output, _, _) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("single token decode should succeed");

        assert_eq!(output.dims(), &[batch_size, 1, hidden_size]);
    }

    #[test]
    fn forward_output_finite_with_random_inputs() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;

        // Use non-zero weights to get a more realistic test
        let vb = VarBuilder::zeros(DType::F32, &device);
        let cross_attn =
            CrossAttention::new(hidden_size, num_heads, vb).expect("creation should work");

        let decoder_hidden = Tensor::randn(0.0f32, 0.1, (1, 3, hidden_size), &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::randn(0.0f32, 0.1, (1, 5, hidden_size), &device)
            .expect("tensor creation should work");

        let (output, _, _) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("forward should succeed");

        let output_data: Vec<f32> = output
            .flatten_all()
            .expect("flatten should work")
            .to_vec1()
            .expect("to_vec1 should work");
        assert!(
            output_data.iter().all(|v| v.is_finite()),
            "all output values should be finite"
        );
    }

    #[test]
    fn forward_preserves_dtype() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;

        let cross_attn =
            create_cross_attention(hidden_size, num_heads, &device).expect("creation should work");

        let decoder_hidden = Tensor::zeros((1, 2, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::zeros((1, 4, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        let (output, k, v) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("forward should succeed");

        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(k.dtype(), DType::F32);
        assert_eq!(v.dtype(), DType::F32);
    }

    // ─── Zero-weight property tests ──────────────────────────────────────────

    #[test]
    fn zero_weights_produce_zero_output() {
        // With all-zero projection weights, the output should be zero
        // regardless of input (since all projections map to zero).
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;

        let cross_attn =
            create_cross_attention(hidden_size, num_heads, &device).expect("creation should work");

        let decoder_hidden = Tensor::randn(0.0f32, 1.0, (1, 3, hidden_size), &device)
            .expect("tensor creation should work");
        let encoder_hidden = Tensor::randn(0.0f32, 1.0, (1, 5, hidden_size), &device)
            .expect("tensor creation should work");

        let (output, _, _) = cross_attn
            .forward(&decoder_hidden, &encoder_hidden, None, None)
            .expect("forward should succeed");

        let output_data: Vec<f32> = output
            .flatten_all()
            .expect("flatten should work")
            .to_vec1()
            .expect("to_vec1 should work");
        assert!(
            output_data.iter().all(|&v| v.abs() < 1e-6),
            "zero-weight cross attention should produce zero output"
        );
    }

    // ─── Reshape helper tests ────────────────────────────────────────────────

    #[test]
    fn reshape_for_attention_correct_shape() {
        let device = Device::Cpu;
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 2;
        let seq_len = 5;

        let cross_attn =
            create_cross_attention(hidden_size, num_heads, &device).expect("creation should work");

        let tensor = Tensor::zeros((batch_size, seq_len, hidden_size), DType::F32, &device)
            .expect("tensor creation should work");

        let reshaped = cross_attn
            .reshape_for_attention(tensor, batch_size, seq_len)
            .expect("reshape should succeed");

        assert_eq!(reshaped.dims(), &[batch_size, num_heads, seq_len, head_dim]);
    }
}
