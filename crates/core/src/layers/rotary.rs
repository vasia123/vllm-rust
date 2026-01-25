use candle_core::{DType, Device, Result, Tensor};

pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
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
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
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
        // cos, sin: [total_tokens, head_dim/2]
        let cos = self.cos.index_select(&pos_tensor, 0)?;
        let sin = self.sin.index_select(&pos_tensor, 0)?;

        // rope expects 4D input [b, h, t, d]
        // [total_tokens, heads, head_dim] → [1, heads, total_tokens, head_dim]
        let q = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

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

        // sin and cos should have shape [max_seq_len, head_dim/2]
        let sin_shape = rope.sin.dims();
        let cos_shape = rope.cos.dims();

        assert_eq!(sin_shape, &[max_seq_len, head_dim / 2]);
        assert_eq!(cos_shape, &[max_seq_len, head_dim / 2]);
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
