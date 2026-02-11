use candle_core::{DType, Device, Result, Tensor};

/// Rotary positional embedding (RoPE).
///
/// Supports both standard RoPE and partial RoPE used by GLM models.
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Actual dimension used for rotation (may be less than head_dim with partial rotary).
    rotary_dim: usize,
    /// Full head dimension.
    head_dim: usize,
    /// Whether to use neox-style interleaved rotation (true) or split rotation (false).
    /// GLM models use is_neox_style = false.
    is_neox_style: bool,
}

impl RotaryEmbedding {
    /// Create a standard rotary embedding (full head_dim rotation, neox style).
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_partial(head_dim, max_seq_len, rope_theta, 1.0, true, dtype, device)
    }

    /// Create a rotary embedding with partial rotation support.
    ///
    /// # Arguments
    /// * `head_dim` - Full head dimension
    /// * `max_seq_len` - Maximum sequence length
    /// * `rope_theta` - RoPE base frequency
    /// * `partial_rotary_factor` - Fraction of head_dim to rotate (0.0-1.0, GLM uses 0.5)
    /// * `is_neox_style` - true for interleaved rotation, false for split (GLM uses false)
    /// * `dtype` - Data type
    /// * `device` - Target device
    pub fn new_partial(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        is_neox_style: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = (head_dim as f64 * partial_rotary_factor) as usize;
        // rotary_dim must be even
        let rotary_dim = rotary_dim - (rotary_dim % 2);

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            rotary_dim,
            head_dim,
            is_neox_style,
        })
    }

    /// Create a su-scaled (LongRoPE) rotary embedding used by Phi-3 long-context models.
    ///
    /// Per-dimension scaling factors are applied to the inverse frequencies. The model
    /// config provides `short_factor` and `long_factor` arrays of length `rotary_dim / 2`.
    /// At construction time, we choose long factors when `max_seq_len > original_max_position_embeddings`,
    /// matching vLLM's approach of deciding once based on the serving context length.
    ///
    /// The mscale (magnitude scaling) compensates for the frequency rescaling:
    ///   `mscale = sqrt(1 + log(scale) / log(original_max_pos))`
    /// where `scale = max_seq_len / original_max_position_embeddings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_su_scaled(
        head_dim: usize,
        max_seq_len: usize,
        original_max_position_embeddings: usize,
        rope_theta: f64,
        short_factor: &[f64],
        long_factor: &[f64],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = head_dim;
        let half_dim = rotary_dim / 2;

        if short_factor.len() != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "short_factor length ({}) must equal rotary_dim/2 ({})",
                short_factor.len(),
                half_dim
            )));
        }
        if long_factor.len() != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "long_factor length ({}) must equal rotary_dim/2 ({})",
                long_factor.len(),
                half_dim
            )));
        }

        let use_long = max_seq_len > original_max_position_embeddings;
        let (factors, cache_len) = if use_long {
            (long_factor, max_seq_len)
        } else {
            (short_factor, original_max_position_embeddings)
        };

        // mscale = sqrt(1 + log(scale) / log(original_max_pos))
        let scale = max_seq_len as f64 / original_max_position_embeddings as f64;
        let mscale = if scale <= 1.0 {
            1.0_f64
        } else {
            (1.0 + scale.ln() / (original_max_position_embeddings as f64).ln()).sqrt()
        };

        // Scaled inverse frequencies: inv_freq[i] = 1.0 / (factors[i] * base^(2i / rotary_dim))
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .enumerate()
            .map(|(idx, i)| {
                1.0 / (factors[idx] as f32 * (rope_theta as f32).powf(i as f32 / rotary_dim as f32))
            })
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;

        let t = Tensor::arange(0u32, cache_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cache_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        Ok(Self {
            sin: freqs.sin()?.affine(mscale, 0.0)?.to_dtype(dtype)?,
            cos: freqs.cos()?.affine(mscale, 0.0)?.to_dtype(dtype)?,
            rotary_dim,
            head_dim,
            is_neox_style: true,
        })
    }

    /// Get the rotary dimension.
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Check if this is a partial rotation (rotary_dim < head_dim).
    pub fn is_partial(&self) -> bool {
        self.rotary_dim < self.head_dim
    }

    /// Access the precomputed cosine table [max_seq_len, rotary_dim/2].
    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    /// Access the precomputed sine table [max_seq_len, rotary_dim/2].
    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    /// Apply RoPE to Q and K tensors.
    ///
    /// # Arguments
    /// * `q` - Query tensor of shape [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    /// * `seqlen_offset` - Position offset (for cached/decode scenarios)
    ///
    /// For full rotary (rotary_dim == head_dim), applies rotation to entire tensor.
    /// For partial rotary, only rotates first rotary_dim dimensions, passes rest through.
    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        if self.is_partial() {
            // Partial rotation: only rotate first rotary_dim dimensions
            self.apply_partial_rope(q, k, &cos, &sin)
        } else if self.is_neox_style {
            // Standard neox-style rotation (candle_nn default)
            let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q, k))
        } else {
            // Non-neox style (split rotation) - used by some models
            self.apply_split_rope(q, k, &cos, &sin)
        }
    }

    /// Apply partial rotation - only rotates first rotary_dim dimensions.
    fn apply_partial_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, h, seq_len, d) = q.dims4()?;
        let (_, kv_h, _, _) = k.dims4()?;

        // Split into rotary and passthrough parts
        let q_rot = q.narrow(3, 0, self.rotary_dim)?.contiguous()?;
        let q_pass = q.narrow(3, self.rotary_dim, d - self.rotary_dim)?;

        let k_rot = k.narrow(3, 0, self.rotary_dim)?.contiguous()?;
        let k_pass = k.narrow(3, self.rotary_dim, d - self.rotary_dim)?;

        // Apply rotation (using split style for GLM compatibility)
        let (q_rotated, k_rotated) = if self.is_neox_style {
            let q_rotated = candle_nn::rotary_emb::rope(&q_rot, cos, sin)?;
            let k_rotated = candle_nn::rotary_emb::rope(&k_rot, cos, sin)?;
            (q_rotated, k_rotated)
        } else {
            self.apply_split_rope_inner(&q_rot, &k_rot, cos, sin, b, h, kv_h, seq_len)?
        };

        // Concatenate rotated and passthrough parts
        let q_out = Tensor::cat(&[q_rotated, q_pass.contiguous()?], 3)?;
        let k_out = Tensor::cat(&[k_rotated, k_pass.contiguous()?], 3)?;

        Ok((q_out, k_out))
    }

    /// Apply split-style (non-neox) rotation to full head_dim.
    fn apply_split_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, h, seq_len, _d) = q.dims4()?;
        let (_, kv_h, _, _) = k.dims4()?;
        self.apply_split_rope_inner(
            &q.contiguous()?,
            &k.contiguous()?,
            cos,
            sin,
            b,
            h,
            kv_h,
            seq_len,
        )
    }

    /// Inner implementation of split-style rotation.
    ///
    /// Split rotation: x = [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    #[allow(clippy::too_many_arguments)]
    fn apply_split_rope_inner(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _b: usize,
        _h: usize,
        _kv_h: usize,
        _seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let rot_dim = q.dim(3)?;
        let half_dim = rot_dim / 2;

        // Split into two halves
        let q1 = q.narrow(3, 0, half_dim)?;
        let q2 = q.narrow(3, half_dim, half_dim)?;
        let k1 = k.narrow(3, 0, half_dim)?;
        let k2 = k.narrow(3, half_dim, half_dim)?;

        // Expand cos/sin to match tensor shapes
        // cos/sin shape: [seq_len, half_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, half_dim]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply rotation: [x1*cos - x2*sin, x2*cos + x1*sin]
        let q1_rot = q1
            .broadcast_mul(&cos)?
            .broadcast_sub(&q2.broadcast_mul(&sin)?)?;
        let q2_rot = q2
            .broadcast_mul(&cos)?
            .broadcast_add(&q1.broadcast_mul(&sin)?)?;
        let k1_rot = k1
            .broadcast_mul(&cos)?
            .broadcast_sub(&k2.broadcast_mul(&sin)?)?;
        let k2_rot = k2
            .broadcast_mul(&cos)?
            .broadcast_add(&k1.broadcast_mul(&sin)?)?;

        let q_out = Tensor::cat(&[q1_rot, q2_rot], 3)?;
        let k_out = Tensor::cat(&[k1_rot, k2_rot], 3)?;

        Ok((q_out, k_out))
    }

    /// Apply RoPE to variable-length batched tokens with per-token positions.
    /// q: [total_tokens, num_heads, head_dim]
    /// k: [total_tokens, num_kv_heads, head_dim]
    /// positions: position offset for each token (length = total_tokens)
    pub fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let total_tokens = positions.len();

        let pos_tensor = Tensor::from_vec(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            (total_tokens,),
            self.sin.device(),
        )?;
        // cos, sin: [total_tokens, rotary_dim/2]
        let cos = self.cos.index_select(&pos_tensor, 0)?;
        let sin = self.sin.index_select(&pos_tensor, 0)?;

        // rope expects 4D input [b, h, t, d]
        // [total_tokens, heads, head_dim] -> [1, heads, total_tokens, head_dim]
        let q = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

        let (q, k) = if self.is_partial() {
            self.apply_partial_rope(&q, &k, &cos, &sin)?
        } else if self.is_neox_style {
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
            (q, k)
        } else {
            self.apply_split_rope(&q, &k, &cos, &sin)?
        };

        // Back to [total_tokens, heads, head_dim]
        let q = q.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let k = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        Ok((q, k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test parameters from vLLM: max_positions [11, 4096, 32768], head_dim [32, 64, 128], rope_theta [10000, 1000000]

    #[test]
    fn test_rotary_embedding_new_shape() {
        let device = Device::Cpu;
        let head_dim = 64;
        let max_seq_len = 128;
        let rope_theta = 10000.0;

        let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let sin_shape = rope.sin.dims();
        let cos_shape = rope.cos.dims();

        assert_eq!(sin_shape, &[max_seq_len, head_dim / 2]);
        assert_eq!(cos_shape, &[max_seq_len, head_dim / 2]);
        assert_eq!(rope.rotary_dim(), head_dim);
        assert!(!rope.is_partial());
    }

    #[test]
    fn test_partial_rotary_embedding() {
        let device = Device::Cpu;
        let head_dim = 64;
        let max_seq_len = 128;
        let rope_theta = 10000.0;
        let partial_factor = 0.5;

        let rope = RotaryEmbedding::new_partial(
            head_dim,
            max_seq_len,
            rope_theta,
            partial_factor,
            false,
            DType::F32,
            &device,
        )
        .expect("Failed to create partial RotaryEmbedding");

        let rotary_dim = (head_dim as f64 * partial_factor) as usize;
        assert_eq!(rope.rotary_dim(), rotary_dim);
        assert!(rope.is_partial());

        let sin_shape = rope.sin.dims();
        let cos_shape = rope.cos.dims();

        assert_eq!(sin_shape, &[max_seq_len, rotary_dim / 2]);
        assert_eq!(cos_shape, &[max_seq_len, rotary_dim / 2]);
    }

    #[test]
    fn test_partial_rotary_apply_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 8;
        let seq_len = 16;
        let head_dim = 64;
        let partial_factor = 0.5;

        let rope = RotaryEmbedding::new_partial(
            head_dim,
            128,
            10000.0,
            partial_factor,
            false,
            DType::F32,
            &device,
        )
        .expect("Failed to create partial RotaryEmbedding");

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("Failed to apply partial RoPE");

        assert_eq!(q_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(k_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_partial_rotary_passthrough() {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 8;
        let partial_factor = 0.5;

        let rope = RotaryEmbedding::new_partial(
            head_dim,
            128,
            10000.0,
            partial_factor,
            false,
            DType::F32,
            &device,
        )
        .expect("Failed to create partial RotaryEmbedding");

        let q = Tensor::ones((batch, num_heads, seq_len, head_dim), DType::F32, &device)
            .expect("Failed to create q");
        let k = Tensor::ones((batch, num_heads, seq_len, head_dim), DType::F32, &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("Failed to apply partial RoPE");

        let rotary_dim = rope.rotary_dim();
        let q_pass: Vec<f32> = q_rot
            .narrow(3, rotary_dim, head_dim - rotary_dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let k_pass: Vec<f32> = k_rot
            .narrow(3, rotary_dim, head_dim - rotary_dim)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        for &v in &q_pass {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "Passthrough q should be unchanged, got {v}"
            );
        }
        for &v in &k_pass {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "Passthrough k should be unchanged, got {v}"
            );
        }
    }

    #[test]
    fn test_rotary_embedding_different_head_dims() {
        let device = Device::Cpu;
        let max_seq_len = 32;
        let rope_theta = 10000.0;

        for head_dim in [32, 64, 128] {
            let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, DType::F32, &device)
                .expect("Failed to create RotaryEmbedding");

            let sin_shape = rope.sin.dims();
            assert_eq!(
                sin_shape,
                &[max_seq_len, head_dim / 2],
                "Failed for head_dim={head_dim}"
            );
        }
    }

    #[test]
    fn test_rotary_embedding_different_max_positions() {
        let device = Device::Cpu;
        let head_dim = 64;
        let rope_theta = 10000.0;

        for max_seq_len in [11, 4096] {
            let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, DType::F32, &device)
                .expect("Failed to create RotaryEmbedding");

            let cos_shape = rope.cos.dims();
            assert_eq!(
                cos_shape,
                &[max_seq_len, head_dim / 2],
                "Failed for max_seq_len={max_seq_len}"
            );
        }
    }

    #[test]
    fn test_rotary_cos_sin_values_are_bounded() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 128, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let cos_data: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();

        for val in &cos_data {
            assert!(*val >= -1.0 && *val <= 1.0, "cos value {val} out of bounds");
        }
        for val in &sin_data {
            assert!(*val >= -1.0 && *val <= 1.0, "sin value {val} out of bounds");
        }
    }

    #[test]
    fn test_rotary_inv_freq_formula() {
        let device = Device::Cpu;
        let head_dim = 64;
        let rope_theta = 10000.0;
        let rope = RotaryEmbedding::new(head_dim, 16, rope_theta, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let cos_row0: Vec<f32> = rope
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let sin_row0: Vec<f32> = rope
            .sin
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        for &c in &cos_row0 {
            assert!(
                (c - 1.0).abs() < 1e-5,
                "cos at position 0 should be 1.0, got {c}"
            );
        }
        for &s in &sin_row0 {
            assert!(s.abs() < 1e-5, "sin at position 0 should be 0.0, got {s}");
        }
    }

    #[test]
    fn test_rotary_apply_basic_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 8;
        let seq_len = 16;
        let head_dim = 64;

        let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("Failed to apply RoPE");

        assert_eq!(q_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(k_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_rotary_apply_with_offset() {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 32;

        let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope
            .apply(&q, &k, 10)
            .expect("Failed to apply RoPE with offset");

        assert_eq!(q_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(k_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_rotary_apply_varlen_shape() {
        let device = Device::Cpu;
        let total_tokens = 20;
        let num_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 64;

        let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let q = Tensor::randn(0.0f32, 1.0, (total_tokens, num_heads, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (total_tokens, num_kv_heads, head_dim), &device)
            .expect("Failed to create k");

        let positions: Vec<usize> = (0..total_tokens).collect();

        let (q_rot, k_rot) = rope
            .apply_varlen(&q, &k, &positions)
            .expect("Failed to apply varlen RoPE");

        assert_eq!(q_rot.dims(), &[total_tokens, num_heads, head_dim]);
        assert_eq!(k_rot.dims(), &[total_tokens, num_kv_heads, head_dim]);
    }

    #[test]
    fn test_rotary_apply_varlen_non_contiguous_positions() {
        let device = Device::Cpu;
        let total_tokens = 5;
        let num_heads = 4;
        let head_dim = 32;

        let rope = RotaryEmbedding::new(head_dim, 128, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let q = Tensor::randn(0.0f32, 1.0, (total_tokens, num_heads, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (total_tokens, num_heads, head_dim), &device)
            .expect("Failed to create k");

        let positions = vec![0, 1, 5, 10, 100];

        let (q_rot, k_rot) = rope
            .apply_varlen(&q, &k, &positions)
            .expect("Failed to apply varlen RoPE with non-contiguous positions");

        assert_eq!(q_rot.dims(), &[total_tokens, num_heads, head_dim]);
        assert_eq!(k_rot.dims(), &[total_tokens, num_heads, head_dim]);
    }

    #[test]
    fn test_rotary_preserves_dtype() {
        let device = Device::Cpu;
        let rope_f32 = RotaryEmbedding::new(64, 32, 10000.0, DType::F32, &device)
            .expect("Failed to create f32 RoPE");

        assert_eq!(rope_f32.sin.dtype(), DType::F32);
        assert_eq!(rope_f32.cos.dtype(), DType::F32);
    }

    #[test]
    fn test_rotary_different_rope_theta() {
        let device = Device::Cpu;
        let head_dim = 64;
        let max_seq_len = 32;

        let rope1 = RotaryEmbedding::new(head_dim, max_seq_len, 10000.0, DType::F32, &device)
            .expect("Failed to create RoPE with theta=10000");

        let rope2 = RotaryEmbedding::new(head_dim, max_seq_len, 1000000.0, DType::F32, &device)
            .expect("Failed to create RoPE with theta=1000000");

        assert_eq!(rope1.cos.dims(), &[max_seq_len, head_dim / 2]);
        assert_eq!(rope2.cos.dims(), &[max_seq_len, head_dim / 2]);

        let cos1_data: Vec<f32> = rope1.cos.flatten_all().unwrap().to_vec1().unwrap();
        let cos2_data: Vec<f32> = rope2.cos.flatten_all().unwrap().to_vec1().unwrap();

        assert!(
            cos1_data[head_dim / 2..]
                .iter()
                .zip(cos2_data[head_dim / 2..].iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "Different rope_theta should produce different cos values"
        );
    }

    #[test]
    fn test_rotary_cos_sin_pythagorean_identity() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 32, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let cos_data: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();

        for (c, s) in cos_data.iter().zip(sin_data.iter()) {
            let sum = c * c + s * s;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Pythagorean identity violated: cos^2+sin^2 = {sum}"
            );
        }
    }

    // ── Su-Scaled (LongRoPE) Tests ──────────────────────────────────────────

    fn uniform_factors(half_dim: usize, value: f64) -> Vec<f64> {
        vec![value; half_dim]
    }

    #[test]
    fn test_su_scaled_rope_construction_short() {
        let device = Device::Cpu;
        let head_dim = 16;
        let half_dim = head_dim / 2;
        let original_max_pos = 4096;
        let max_seq_len = 2048;
        let short_factor = uniform_factors(half_dim, 1.0);
        let long_factor = uniform_factors(half_dim, 4.0);
        let rope = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            original_max_pos,
            10000.0,
            &short_factor,
            &long_factor,
            DType::F32,
            &device,
        )
        .expect("Failed to create su-scaled RoPE");
        assert_eq!(rope.cos.dims(), &[original_max_pos, half_dim]);
        assert_eq!(rope.sin.dims(), &[original_max_pos, half_dim]);
        assert_eq!(rope.rotary_dim(), head_dim);
        assert!(!rope.is_partial());
    }

    #[test]
    fn test_su_scaled_rope_construction_long() {
        let device = Device::Cpu;
        let head_dim = 16;
        let half_dim = head_dim / 2;
        let original_max_pos = 4096;
        let max_seq_len = 131072;
        let short_factor = uniform_factors(half_dim, 1.0);
        let long_factor = uniform_factors(half_dim, 4.0);
        let rope = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            original_max_pos,
            10000.0,
            &short_factor,
            &long_factor,
            DType::F32,
            &device,
        )
        .expect("Failed to create su-scaled RoPE (long)");
        assert_eq!(rope.cos.dims(), &[max_seq_len, half_dim]);
        assert_eq!(rope.sin.dims(), &[max_seq_len, half_dim]);
    }

    #[test]
    fn test_su_scaled_rope_frequency_computation() {
        let device = Device::Cpu;
        let head_dim = 16;
        let half_dim = head_dim / 2;
        let max_seq_len = 64;
        let rope_theta = 10000.0;
        let standard = RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, DType::F32, &device)
            .expect("standard rope");
        let su_factors = uniform_factors(half_dim, 1.0);
        let su_scaled = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            max_seq_len,
            rope_theta,
            &su_factors,
            &su_factors,
            DType::F32,
            &device,
        )
        .expect("su-scaled rope with factor=1.0");
        let std_cos: Vec<f32> = standard.cos.flatten_all().unwrap().to_vec1().unwrap();
        let su_cos: Vec<f32> = su_scaled.cos.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in std_cos.iter().zip(su_cos.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Position {i}: standard cos={a}, su-scaled cos={b} should match"
            );
        }
    }

    #[test]
    fn test_su_scaled_rope_factors_change_frequencies() {
        let device = Device::Cpu;
        let head_dim = 16;
        let half_dim = head_dim / 2;
        let max_seq_len = 64;
        let short_factor = uniform_factors(half_dim, 1.0);
        let long_factor = uniform_factors(half_dim, 4.0);
        let rope_short = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            max_seq_len,
            10000.0,
            &short_factor,
            &long_factor,
            DType::F32,
            &device,
        )
        .expect("short rope");
        let rope_long = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len * 2,
            max_seq_len,
            10000.0,
            &short_factor,
            &long_factor,
            DType::F32,
            &device,
        )
        .expect("long rope");
        let short_cos: Vec<f32> = rope_short
            .cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let long_cos: Vec<f32> = rope_long
            .cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            short_cos
                .iter()
                .zip(long_cos.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "Short and long factors should produce different frequencies"
        );
    }

    #[test]
    fn test_su_scaled_rope_mscale_applied() {
        let device = Device::Cpu;
        let head_dim = 8;
        let half_dim = head_dim / 2;
        let original_max_pos = 4096;
        let max_seq_len = original_max_pos * 32;
        let factors = uniform_factors(half_dim, 1.0);
        let rope = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            original_max_pos,
            10000.0,
            &factors,
            &factors,
            DType::F32,
            &device,
        )
        .expect("su-scaled with mscale");
        let expected_mscale =
            (1.0 + (32.0_f64).ln() / (original_max_pos as f64).ln()).sqrt() as f32;
        let cos_row0: Vec<f32> = rope
            .cos
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &c) in cos_row0.iter().enumerate() {
            assert!(
                (c - expected_mscale).abs() < 1e-4,
                "cos[0,{i}] = {c}, expected mscale = {expected_mscale}"
            );
        }
    }

    #[test]
    fn test_su_scaled_rope_invalid_factor_length() {
        let device = Device::Cpu;
        let head_dim = 16;
        let too_short = vec![1.0; 4];
        let correct = vec![1.0; 8];
        let result = RotaryEmbedding::new_su_scaled(
            head_dim,
            4096,
            4096,
            10000.0,
            &too_short,
            &correct,
            DType::F32,
            &device,
        );
        assert!(
            result.is_err(),
            "Should fail with wrong short_factor length"
        );
        let result = RotaryEmbedding::new_su_scaled(
            head_dim,
            4096,
            4096,
            10000.0,
            &correct,
            &too_short,
            DType::F32,
            &device,
        );
        assert!(result.is_err(), "Should fail with wrong long_factor length");
    }

    #[test]
    fn test_su_scaled_rope_apply_shape() {
        let device = Device::Cpu;
        let batch = 1;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 16;
        let half_dim = head_dim / 2;
        let factors = uniform_factors(half_dim, 1.5);
        let rope = RotaryEmbedding::new_su_scaled(
            head_dim,
            128,
            128,
            10000.0,
            &factors,
            &factors,
            DType::F32,
            &device,
        )
        .expect("su-scaled rope");
        let q =
            Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).expect("q");
        let k =
            Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device).expect("k");
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("apply su-scaled");
        assert_eq!(q_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
        assert_eq!(k_rot.dims(), &[batch, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_su_scaled_rope_pythagorean_with_mscale() {
        let device = Device::Cpu;
        let head_dim = 8;
        let half_dim = head_dim / 2;
        let original_max_pos = 4096;
        let max_seq_len = original_max_pos * 4;
        let factors = uniform_factors(half_dim, 1.0);
        let rope = RotaryEmbedding::new_su_scaled(
            head_dim,
            max_seq_len,
            original_max_pos,
            10000.0,
            &factors,
            &factors,
            DType::F32,
            &device,
        )
        .expect("su-scaled with mscale");
        let scale = max_seq_len as f64 / original_max_pos as f64;
        let expected_mscale_sq = (1.0 + scale.ln() / (original_max_pos as f64).ln()) as f32;
        let cos_data: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();
        for (c, s) in cos_data.iter().zip(sin_data.iter()) {
            let sum = c * c + s * s;
            assert!(
                (sum - expected_mscale_sq).abs() < 1e-4,
                "cos^2+sin^2 = {sum}, expected mscale^2 = {expected_mscale_sq}"
            );
        }
    }
}
