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

    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
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
