use candle_core::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda-kernels")]
use std::sync::OnceLock;

/// Rotary positional embedding (RoPE).
///
/// Supports both standard RoPE and partial RoPE used by GLM models.
/// When the `cuda-kernels` feature is enabled and tensors are on GPU,
/// uses a fused CUDA kernel that replaces ~6 tensor ops per tensor with
/// a single kernel launch (3-8% throughput improvement).
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
    /// Precomputed interleaved cos/sin cache for CUDA kernel: [max_seq_len, rot_dim] f32.
    /// Lazily initialized on first CUDA call.
    #[cfg(feature = "cuda-kernels")]
    cos_sin_cache: OnceLock<Tensor>,
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
            #[cfg(feature = "cuda-kernels")]
            cos_sin_cache: OnceLock::new(),
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
            #[cfg(feature = "cuda-kernels")]
            cos_sin_cache: OnceLock::new(),
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

        // CUDA fast path: fused kernel for bf16 neox-style (most common LLM case)
        #[cfg(feature = "cuda-kernels")]
        if q.dtype() == DType::BF16
            && crate::cuda_kernels::rotary_embedding_cuda_available(q)
            && self.is_neox_style
        {
            let num_heads = q.dim(1)?;
            let num_kv_heads = k.dim(1)?;

            let cos_sin_cache = self.get_or_build_cos_sin_cache()?;

            let pos_tensor = Tensor::from_vec(
                positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
                (total_tokens,),
                q.device(),
            )?;

            return crate::cuda_kernels::rotary_embedding_cuda(
                q,
                k,
                &pos_tensor,
                &cos_sin_cache,
                self.rotary_dim,
                self.head_dim,
                num_heads,
                num_kv_heads,
                true, // is_neox
            );
        }

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

    /// Build the interleaved cos_sin_cache for the CUDA kernel.
    /// Format: [max_seq_len, rot_dim] f32 where each row is
    /// [cos_0, cos_1, ..., cos_{half-1}, sin_0, sin_1, ..., sin_{half-1}]
    #[cfg(feature = "cuda-kernels")]
    fn get_or_build_cos_sin_cache(&self) -> Result<Tensor> {
        if let Some(cache) = self.cos_sin_cache.get() {
            return Ok(cache.clone());
        }

        let cos_f32 = self.cos.to_dtype(DType::F32)?;
        let sin_f32 = self.sin.to_dtype(DType::F32)?;
        let cache = Tensor::cat(&[cos_f32, sin_f32], 1)?.contiguous()?;

        // OnceLock::get_or_init could race, but both threads would compute the same value
        let _ = self.cos_sin_cache.set(cache.clone());
        Ok(self.cos_sin_cache.get().unwrap().clone())
    }
}

// ─── XDRotaryEmbedding ────────────────────────────────────────────────────────

/// Cross-Dimensional RoPE (XDRoPE) for multimodal models.
///
/// Extends `DynamicNTKAlphaRotaryEmbedding` with per-dimension position channels.
/// Each section of the rotary half-dim uses positions from a different channel
/// (P=positional, W=width, H=height, T=temporal), enabling the model to encode
/// 2D/3D spatial structure without separate encoders.
///
/// Reference: `vllm/model_executor/layers/rotary_embedding/xdrope.py`
pub struct XDRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Size of each section in the rotary half-dim. `sum(xdrope_section) == rotary_dim / 2`.
    xdrope_section: Vec<usize>,
    /// Actual rotation dimension (may be less than head_dim).
    rotary_dim: usize,
    /// Full head dimension.
    head_dim: usize,
}

impl XDRotaryEmbedding {
    /// Create an XDRoPE embedding with NTK-alpha scaling.
    ///
    /// # Arguments
    /// * `head_dim` - Full head dimension
    /// * `max_seq_len` - Maximum sequence length (cache size)
    /// * `rope_theta` - RoPE base frequency
    /// * `scaling_alpha` - NTK scaling factor; scaled_base = theta * alpha^(dim/(dim-2))
    /// * `xdrope_section` - Section sizes; must sum to `rotary_dim / 2`
    /// * `rotary_dim` - Effective rotation dimension (≤ head_dim)
    /// * `dtype` - Data type for sin/cos tables
    /// * `device` - Target device
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        scaling_alpha: f64,
        xdrope_section: Vec<usize>,
        rotary_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = rotary_dim / 2;
        let section_sum: usize = xdrope_section.iter().sum();
        if section_sum != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "xdrope_section sum ({section_sum}) must equal rotary_dim/2 ({half_dim})"
            )));
        }

        // NTK alpha scaling: scaled_base = theta * alpha^(dim/(dim-2))
        let scaled_theta = if scaling_alpha > 1.0 && rotary_dim > 2 {
            rope_theta * scaling_alpha.powf(rotary_dim as f64 / (rotary_dim as f64 - 2.0))
        } else {
            rope_theta
        };

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (scaled_theta as f32).powf(i as f32 / rotary_dim as f32))
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
            xdrope_section,
            rotary_dim,
            head_dim,
        })
    }

    /// Apply XDRoPE to variable-length batched tokens with per-dimension position channels.
    ///
    /// # Arguments
    /// * `q` - Query tensor `[num_tokens, num_heads, head_dim]`
    /// * `k` - Key tensor `[num_tokens, num_kv_heads, head_dim]`
    /// * `positions` - Position indices `[xd_sections, num_tokens]` (u32 on CPU)
    ///
    /// For each section `i`, positions from channel `i` are used to index the
    /// cos/sin tables for the corresponding rotary half-dim slice.
    pub fn apply_varlen_xd(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let num_sections = self.xdrope_section.len();
        let (num_tokens, num_heads, _) = q.dims3()?;
        let num_kv_heads = k.dim(1)?;

        // Build per-token cos/sin by gathering each section's positions.
        let mut cos_parts: Vec<Tensor> = Vec::with_capacity(num_sections);
        let mut sin_parts: Vec<Tensor> = Vec::with_capacity(num_sections);

        let mut half_offset = 0usize;
        for (i, &section_size) in self.xdrope_section.iter().enumerate() {
            // positions[i, :] → [num_tokens] position indices for this channel
            let pos_i = positions.narrow(0, i, 1)?.squeeze(0)?;
            // Index the full cos/sin table: [num_tokens, rotary_dim/2]
            let cos_full = self.cos.index_select(&pos_i, 0)?;
            let sin_full = self.sin.index_select(&pos_i, 0)?;
            // Slice the section columns
            cos_parts.push(cos_full.narrow(1, half_offset, section_size)?);
            sin_parts.push(sin_full.narrow(1, half_offset, section_size)?);
            half_offset += section_size;
        }

        // Concatenate all sections → [num_tokens, rotary_dim/2]; must be contiguous for rope().
        let cos = Tensor::cat(&cos_parts, 1)?.contiguous()?;
        let sin = Tensor::cat(&sin_parts, 1)?.contiguous()?;

        // Reshape q/k to [1, heads, num_tokens, head_dim] for the rope utilities.
        let q4 = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k4 = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

        let (q_out4, k_out4) = if self.rotary_dim < self.head_dim {
            // Partial rotation: only rotate first rotary_dim dims.
            let q_rot = q4.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q4.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            let k_rot = k4.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k4.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;
            (
                Tensor::cat(&[q_rot, q_pass.contiguous()?], 3)?,
                Tensor::cat(&[k_rot, k_pass.contiguous()?], 3)?,
            )
        } else {
            let q_rot = candle_nn::rotary_emb::rope(&q4, &cos, &sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k4, &cos, &sin)?;
            (q_rot, k_rot)
        };

        // Back to [num_tokens, heads, head_dim]
        let q_out = q_out4.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let k_out = k_out4.squeeze(0)?.transpose(0, 1)?.contiguous()?;

        // Sanity-check shapes.
        debug_assert_eq!(q_out.dims(), &[num_tokens, num_heads, self.head_dim]);
        debug_assert_eq!(k_out.dims(), &[num_tokens, num_kv_heads, self.head_dim]);

        Ok((q_out, k_out))
    }

    /// Returns the rotary dimension.
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Returns the number of XD sections (position channels).
    pub fn num_sections(&self) -> usize {
        self.xdrope_section.len()
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

    // ── XDRotaryEmbedding Tests ──────────────────────────────────────────────

    fn make_xdrope(head_dim: usize, max_seq: usize, sections: Vec<usize>) -> XDRotaryEmbedding {
        let rotary_dim = sections.iter().sum::<usize>() * 2;
        XDRotaryEmbedding::new(
            head_dim,
            max_seq,
            10000.0,
            1.0, // no NTK scaling
            sections,
            rotary_dim,
            DType::F32,
            &Device::Cpu,
        )
        .expect("XDRoPE construction failed")
    }

    #[test]
    fn test_xdrope_construction() {
        // sections [4, 4] → rotary_dim = 16, head_dim = 32
        let xd = make_xdrope(32, 128, vec![4, 4]);
        assert_eq!(xd.rotary_dim(), 16);
        assert_eq!(xd.num_sections(), 2);
    }

    #[test]
    fn test_xdrope_section_sum_mismatch_error() {
        let result = XDRotaryEmbedding::new(
            32,
            128,
            10000.0,
            1.0,
            vec![3, 4], // sum=7, but rotary_dim/2 must equal sum
            16,         // rotary_dim=16, half=8 ≠ 7
            DType::F32,
            &Device::Cpu,
        );
        assert!(
            result.is_err(),
            "Should fail when section sum ≠ rotary_dim/2"
        );
    }

    #[test]
    fn test_xdrope_apply_varlen_shape() {
        let head_dim = 32;
        let rotary_dim = 16;
        let num_tokens = 8;
        let num_heads = 4;
        let num_kv_heads = 2;

        let xd = make_xdrope(head_dim, 128, vec![4, 4]);
        assert_eq!(xd.rotary_dim(), rotary_dim);

        let q =
            Tensor::randn(0.0f32, 1.0, (num_tokens, num_heads, head_dim), &Device::Cpu).unwrap();
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (num_tokens, num_kv_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();

        // positions: [2, num_tokens] — section 0 uses 0..7, section 1 uses 4..11
        let pos_data: Vec<u32> = (0..num_tokens as u32)
            .chain((4..4 + num_tokens as u32).map(|p| p.min(127)))
            .collect();
        let positions = Tensor::from_vec(pos_data, (2, num_tokens), &Device::Cpu).unwrap();

        let (q_out, k_out) = xd.apply_varlen_xd(&q, &k, &positions).unwrap();
        assert_eq!(q_out.dims(), &[num_tokens, num_heads, head_dim]);
        assert_eq!(k_out.dims(), &[num_tokens, num_kv_heads, head_dim]);
    }

    #[test]
    fn test_xdrope_ntk_alpha_changes_frequencies() {
        let head_dim = 16;
        let rotary_dim = 16;
        let sections = vec![4, 4];

        let xd_no_ntk = XDRotaryEmbedding::new(
            head_dim,
            64,
            10000.0,
            1.0,
            sections.clone(),
            rotary_dim,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let xd_ntk = XDRotaryEmbedding::new(
            head_dim,
            64,
            10000.0,
            2.0, // alpha > 1 → scaled theta
            sections,
            rotary_dim,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let cos_no_ntk: Vec<f32> = xd_no_ntk
            .cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let cos_ntk: Vec<f32> = xd_ntk
            .cos
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            cos_no_ntk
                .iter()
                .zip(cos_ntk.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "NTK alpha scaling should change frequencies"
        );
    }

    #[test]
    fn test_xdrope_uniform_sections_matches_standard_rope() {
        // When all sections use the same positions (same as 1D RoPE), the result
        // should match applying standard neox-style RoPE with those positions.
        let head_dim = 16;
        let rotary_dim = 16;
        let num_tokens = 4;
        let num_heads = 2;
        let max_seq = 32;

        // Standard RoPE (neox, full head_dim)
        let std_rope =
            RotaryEmbedding::new(head_dim, max_seq, 10000.0, DType::F32, &Device::Cpu).unwrap();
        // XDRoPE with 2 equal sections, same alpha=1
        let xd = XDRotaryEmbedding::new(
            head_dim,
            max_seq,
            10000.0,
            1.0,
            vec![4, 4],
            rotary_dim,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let q =
            Tensor::randn(0.0f32, 1.0, (num_tokens, num_heads, head_dim), &Device::Cpu).unwrap();
        let k = Tensor::zeros((num_tokens, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();

        // Standard varlen with positions [0,1,2,3]
        let positions: Vec<usize> = (0..num_tokens).collect();
        let (q_std, _) = std_rope.apply_varlen(&q, &k, &positions).unwrap();

        // XDRoPE with uniform positions (all sections use same 1D positions)
        let pos_data: Vec<u32> = (0..num_tokens as u32).chain(0..num_tokens as u32).collect();
        let pos_tensor = Tensor::from_vec(pos_data, (2, num_tokens), &Device::Cpu).unwrap();
        let (q_xd, _) = xd.apply_varlen_xd(&q, &k, &pos_tensor).unwrap();

        let q_std_v: Vec<f32> = q_std.flatten_all().unwrap().to_vec1().unwrap();
        let q_xd_v: Vec<f32> = q_xd.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in q_std_v.iter().zip(q_xd_v.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "XDRoPE with uniform positions should match standard RoPE: {a} vs {b}"
            );
        }
    }
}

// ─── MRoPEInterleaved ────────────────────────────────────────────────────────

/// Generate an interleaved index sequence for multi-modal rotary embedding.
///
/// Given per-dimension counts `a`, `b`, `c`, produces a sequence of dimension
/// indices `[0..total)` that is balanced (each index appears its count times)
/// and avoids placing the same index consecutively wherever possible.
///
/// Algorithm: greedy minimum-placed selection — at each step, from all
/// dimensions that still have remaining budget and are not equal to `last`,
/// pick the one that has been placed the fewest times relative to its total
/// count. If no non-`last` candidate remains, relax the constraint.
///
/// When `force_last` is true, one unit is reserved from count `a` and
/// appended unconditionally at the end (used for the temporal dimension in
/// 3-section setups to ensure the last position is always temporal).
///
/// Reference: `MRotaryEmbeddingInterleaved.get_mrope_interleaved_id_list`
/// in `vllm/model_executor/layers/rotary_embedding/mrope_interleaved.py`.
pub fn get_mrope_interleaved_id_list(a: usize, b: usize, c: usize, force_last: bool) -> Vec<usize> {
    let mut counts = [a, b, c];
    if force_last {
        if counts[0] == 0 {
            // Cannot force_last when a == 0; just skip.
        } else {
            counts[0] -= 1;
        }
    }

    let totals = counts;
    let total: usize = counts.iter().sum();
    let mut placed = [0usize; 3];
    let mut remaining = counts;
    let mut seq = Vec::with_capacity(total + if force_last { 1 } else { 0 });
    let mut last: Option<usize> = None;

    for _ in 0..total {
        // Prefer candidates that differ from the previous choice.
        let mut cands: Vec<usize> = (0..3)
            .filter(|&k| remaining[k] > 0 && Some(k) != last)
            .collect();
        if cands.is_empty() {
            // Only the last dimension has remaining budget.
            cands = (0..3).filter(|&k| remaining[k] > 0).collect();
        }

        // Greedy: pick the dimension that has been placed the least fraction
        // of its total. Break ties by index (lower = first).
        let best = cands
            .into_iter()
            .min_by(|&x, &y| {
                let fx = if totals[x] == 0 {
                    f64::MAX
                } else {
                    placed[x] as f64 / totals[x] as f64
                };
                let fy = if totals[y] == 0 {
                    f64::MAX
                } else {
                    placed[y] as f64 / totals[y] as f64
                };
                fx.partial_cmp(&fy)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(x.cmp(&y))
            })
            .unwrap_or(0);

        seq.push(best);
        placed[best] += 1;
        remaining[best] -= 1;
        last = Some(best);
    }

    if force_last {
        seq.push(0);
    }

    seq
}

/// Multimodal Rotary Embedding with interleaved frequency assignment.
///
/// Differs from section-based MRoPE (which assigns contiguous frequency blocks
/// to each dimension) by interleaving: each individual frequency `i` is
/// assigned to dimension `mrope_dim[i]` ∈ {0, 1, 2} (temporal/height/width)
/// via the balanced greedy algorithm [`get_mrope_interleaved_id_list`].
///
/// This produces a more even distribution of positional information across the
/// full frequency spectrum, which can improve performance for video/image inputs
/// where all three spatial dimensions are important.
///
/// Reference: `MRotaryEmbeddingInterleaved`
/// in `vllm/model_executor/layers/rotary_embedding/mrope_interleaved.py`.
pub struct MRoPEInterleaved {
    sin: Tensor,
    cos: Tensor,
    /// Per-frequency dimension assignment; length == head_dim / 2.
    /// `mrope_dim[i]` ∈ {0, 1, 2} tells which of the three position channels
    /// (temporal/height/width) supplies the position index for frequency `i`.
    mrope_dim: Vec<usize>,
}

impl MRoPEInterleaved {
    /// Create a new interleaved MRoPE embedding.
    ///
    /// # Arguments
    /// * `head_dim` - Full head dimension; rotation uses all `head_dim` dims
    /// * `max_seq_len` - Maximum sequence length (cache size)
    /// * `rope_theta` - RoPE base frequency
    /// * `mrope_section` - Per-dimension frequency count; length 2 or 3;
    ///   must sum to `head_dim / 2`
    /// * `dtype` - Data type for the sin/cos tables
    /// * `device` - Target device
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        mrope_section: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let section_sum: usize = mrope_section.iter().sum();
        if section_sum != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "mrope_section sum ({section_sum}) must equal head_dim/2 ({half_dim})"
            )));
        }
        if mrope_section.len() < 2 || mrope_section.len() > 3 {
            return Err(candle_core::Error::Msg(
                "mrope_section must have length 2 or 3".to_string(),
            ));
        }

        // Build cos/sin cache [max_seq_len, half_dim].
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

        // Build interleaved dimension-assignment indices of length half_dim.
        let (a, b, c, force_last) = if mrope_section.len() == 2 {
            (mrope_section[0], mrope_section[1], 0, false)
        } else {
            (mrope_section[0], mrope_section[1], mrope_section[2], true)
        };
        let mrope_dim = get_mrope_interleaved_id_list(a, b, c, force_last);
        debug_assert_eq!(mrope_dim.len(), half_dim);

        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            mrope_dim,
        })
    }

    /// Apply interleaved MRoPE to prefill tokens.
    ///
    /// # Arguments
    /// * `q` - `[num_tokens, num_heads, head_dim]`
    /// * `k` - `[num_tokens, num_kv_heads, head_dim]`
    /// * `position_ids` - `[3, num_tokens]` (temporal/height/width positions; u32/i64)
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let half_dim = self.mrope_dim.len();

        // For each of the 3 dimensions pre-fetch the per-token, full-half_dim
        // cos and sin rows:  dim_cos[d] = cos.index_select(position_ids[d], 0)
        // Shape: [num_tokens, half_dim]
        let mut dim_cos = Vec::with_capacity(3);
        let mut dim_sin = Vec::with_capacity(3);
        for d in 0..3 {
            let pos_d = position_ids
                .narrow(0, d, 1)?
                .squeeze(0)?
                .to_dtype(DType::U32)?;
            dim_cos.push(self.cos.index_select(&pos_d, 0)?);
            dim_sin.push(self.sin.index_select(&pos_d, 0)?);
        }

        // Interleave: for each freq i, pick the column from the right dim.
        // Result: [num_tokens, half_dim]
        let mut cos_cols = Vec::with_capacity(half_dim);
        let mut sin_cols = Vec::with_capacity(half_dim);
        for (i, &d) in self.mrope_dim.iter().enumerate() {
            cos_cols.push(dim_cos[d].narrow(1, i, 1)?);
            sin_cols.push(dim_sin[d].narrow(1, i, 1)?);
        }
        let cos = Tensor::cat(&cos_cols, 1)?.contiguous()?;
        let sin = Tensor::cat(&sin_cols, 1)?.contiguous()?;

        // rope() expects [b, h, t, d]; input is [num_tokens, num_heads, head_dim].
        // Reshape: [T, H, D] → [1, H, T, D], call rope, then back to [T, H, D].
        let q4 = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k4 = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let q_rot = candle_nn::rotary_emb::rope(&q4, &cos, &sin)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?;
        let k_rot = candle_nn::rotary_emb::rope(&k4, &cos, &sin)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?;
        Ok((q_rot, k_rot))
    }

    /// Apply interleaved MRoPE at a scalar decode position.
    ///
    /// For decode steps where all 3 position dimensions share the same offset
    /// (text-only tokens), this is equivalent to standard RoPE.
    pub fn apply_scalar(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // All dimensions use the same position → standard RoPE from the cache.
        // q: [num_tokens, num_heads, head_dim]; rope() expects [b, h, t, d].
        let seq_len = q.dim(0)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q4 = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k4 = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let q_rot = candle_nn::rotary_emb::rope(&q4, &cos, &sin)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?;
        let k_rot = candle_nn::rotary_emb::rope(&k4, &cos, &sin)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?;
        Ok((q_rot, k_rot))
    }
}

#[cfg(test)]
mod mrope_interleaved_tests {
    use super::*;
    use candle_core::{DType, Device};

    // ─── get_mrope_interleaved_id_list ────────────────────────────────────────

    #[test]
    fn id_list_2d_balanced() {
        // 2-section [3, 3]: should alternate 0 and 1 as much as possible.
        let ids = get_mrope_interleaved_id_list(3, 3, 0, false);
        assert_eq!(ids.len(), 6);
        assert_eq!(ids.iter().filter(|&&x| x == 0).count(), 3);
        assert_eq!(ids.iter().filter(|&&x| x == 1).count(), 3);
        assert_eq!(ids.iter().filter(|&&x| x == 2).count(), 0);
        // No two adjacent duplicates when counts are equal.
        for w in ids.windows(2) {
            assert_ne!(w[0], w[1], "adjacent duplicate in 2D balanced");
        }
    }

    #[test]
    fn id_list_3d_force_last() {
        // 3-section [4, 3, 3], force_last: last element must be 0.
        let ids = get_mrope_interleaved_id_list(4, 3, 3, true);
        assert_eq!(ids.len(), 10);
        assert_eq!(ids.iter().filter(|&&x| x == 0).count(), 4);
        assert_eq!(ids.iter().filter(|&&x| x == 1).count(), 3);
        assert_eq!(ids.iter().filter(|&&x| x == 2).count(), 3);
        assert_eq!(*ids.last().unwrap(), 0, "force_last violated");
    }

    #[test]
    fn id_list_zero_counts() {
        // c == 0: effectively a 2-section list.
        let ids = get_mrope_interleaved_id_list(4, 4, 0, false);
        assert_eq!(ids.len(), 8);
        assert!(ids.iter().all(|&x| x < 2), "c=0 should only produce 0/1");
    }

    // ─── MRoPEInterleaved ─────────────────────────────────────────────────────

    fn make_interleaved_rope(head_dim: usize, mrope_section: &[usize]) -> MRoPEInterleaved {
        MRoPEInterleaved::new(
            head_dim,
            128,
            10000.0,
            mrope_section,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn construction_2d() {
        let rope = make_interleaved_rope(64, &[16, 16]);
        assert_eq!(rope.mrope_dim.len(), 32);
    }

    #[test]
    fn construction_3d() {
        let rope = make_interleaved_rope(64, &[10, 11, 11]);
        assert_eq!(rope.mrope_dim.len(), 32);
    }

    #[test]
    fn construction_mismatched_section_fails() {
        let result = MRoPEInterleaved::new(64, 128, 10000.0, &[10, 10], DType::F32, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn apply_output_shape_3d() {
        let head_dim = 64;
        let rope = make_interleaved_rope(head_dim, &[10, 11, 11]);
        let num_tokens = 12;
        let num_heads = 4;
        let q =
            Tensor::randn(0.0f32, 1.0, (num_tokens, num_heads, head_dim), &Device::Cpu).unwrap();
        let k = Tensor::zeros((num_tokens, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        // position_ids: [3, num_tokens]
        let pos: Vec<u32> = (0..num_tokens as u32)
            .cycle()
            .take(3 * num_tokens)
            .collect();
        let position_ids = Tensor::from_vec(pos, (3, num_tokens), &Device::Cpu).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, &position_ids).unwrap();
        assert_eq!(q_rot.shape().dims(), &[num_tokens, num_heads, head_dim]);
        assert_eq!(k_rot.shape().dims(), &[num_tokens, num_heads, head_dim]);
    }

    #[test]
    fn apply_scalar_matches_standard_rope_at_zero() {
        // When all position dims use offset 0, scalar apply should match
        // regular rope at position 0.
        let head_dim = 32;
        let rope = make_interleaved_rope(head_dim, &[8, 8]);
        let q = Tensor::ones((1, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((1, 4, head_dim), DType::F32, &Device::Cpu).unwrap();
        let (q_rot, _) = rope.apply_scalar(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 4, head_dim]);
    }
}
