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

    /// Get the rotary dimension.
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Check if this is a partial rotation (rotary_dim < head_dim).
    pub fn is_partial(&self) -> bool {
        self.rotary_dim < self.head_dim
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
        // [total_tokens, heads, head_dim] → [1, heads, total_tokens, head_dim]
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

        // sin and cos should have shape [max_seq_len, rotary_dim/2]
        // For full rotation, rotary_dim == head_dim
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
        let partial_factor = 0.5; // GLM uses 0.5

        let rope = RotaryEmbedding::new_partial(
            head_dim,
            max_seq_len,
            rope_theta,
            partial_factor,
            false, // is_neox_style = false for GLM
            DType::F32,
            &device,
        )
        .expect("Failed to create partial RotaryEmbedding");

        let rotary_dim = (head_dim as f64 * partial_factor) as usize;
        assert_eq!(rope.rotary_dim(), rotary_dim);
        assert!(rope.is_partial());

        // sin and cos should have shape [max_seq_len, rotary_dim/2]
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

        // Create q and k with shape [batch, heads, seq, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("Failed to apply partial RoPE");

        // Output shape should match input shape
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

        // Create q and k with known values
        let q = Tensor::ones((batch, num_heads, seq_len, head_dim), DType::F32, &device)
            .expect("Failed to create q");
        let k = Tensor::ones((batch, num_heads, seq_len, head_dim), DType::F32, &device)
            .expect("Failed to create k");

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).expect("Failed to apply partial RoPE");

        // At position 0, cos=1, sin=0, so first half should be unchanged
        // The second half (passthrough) should be unchanged regardless of position
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

        // Passthrough dimensions should be unchanged (all 1s)
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

        // cos and sin should be in [-1, 1]
        for val in &cos_data {
            assert!(*val >= -1.0 && *val <= 1.0, "cos value {val} out of bounds");
        }
        for val in &sin_data {
            assert!(*val >= -1.0 && *val <= 1.0, "sin value {val} out of bounds");
        }
    }

    #[test]
    fn test_rotary_inv_freq_formula() {
        // Verify inv_freq computation: inv_freq[i] = 1 / (theta ^ (2i / head_dim))
        let device = Device::Cpu;
        let head_dim = 64;
        let rope_theta = 10000.0;
        let rope = RotaryEmbedding::new(head_dim, 16, rope_theta, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        // At position 0, freqs = 0 * inv_freq = 0, so cos=1, sin=0
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

        // Create q and k with shape [batch, heads, seq, head_dim]
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

        // Apply with seqlen_offset = 10 (as if continuing from position 10)
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

        // q: [total_tokens, num_heads, head_dim]
        // k: [total_tokens, num_kv_heads, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (total_tokens, num_heads, head_dim), &device)
            .expect("Failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (total_tokens, num_kv_heads, head_dim), &device)
            .expect("Failed to create k");

        // Variable positions for each token
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

        // Non-contiguous positions (simulating batched sequences)
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

        // Default theta (Llama)
        let rope1 = RotaryEmbedding::new(head_dim, max_seq_len, 10000.0, DType::F32, &device)
            .expect("Failed to create RoPE with theta=10000");

        // Larger theta (used by some models for longer context)
        let rope2 = RotaryEmbedding::new(head_dim, max_seq_len, 1000000.0, DType::F32, &device)
            .expect("Failed to create RoPE with theta=1000000");

        // Both should have valid shape
        assert_eq!(rope1.cos.dims(), &[max_seq_len, head_dim / 2]);
        assert_eq!(rope2.cos.dims(), &[max_seq_len, head_dim / 2]);

        // But different values (larger theta → slower frequency decay)
        let cos1_data: Vec<f32> = rope1.cos.flatten_all().unwrap().to_vec1().unwrap();
        let cos2_data: Vec<f32> = rope2.cos.flatten_all().unwrap().to_vec1().unwrap();

        // Values should differ for non-zero positions
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
        // cos²(x) + sin²(x) = 1
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 32, 10000.0, DType::F32, &device)
            .expect("Failed to create RotaryEmbedding");

        let cos_data: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = rope.sin.flatten_all().unwrap().to_vec1().unwrap();

        for (c, s) in cos_data.iter().zip(sin_data.iter()) {
            let sum = c * c + s * s;
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Pythagorean identity violated: cos²+sin² = {sum}"
            );
        }
    }
}
