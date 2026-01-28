//! Multi-head Latent Attention (MLA) backend.
//!
//! MLA compresses KV cache using low-rank projections, achieving significant
//! memory reduction compared to standard attention. Used by DeepSeek V2/V3.
//!
//! # Memory Comparison (per token)
//! - Standard KV cache: ~49KB/token
//! - MLA compressed cache: ~1.2KB/token
//! - MLA + FP8 quantization: ~0.6KB/token
//!
//! # How it works
//!
//! Instead of caching full K and V tensors, MLA caches:
//! - `kv_c`: Compressed latent representation [kv_lora_rank]
//! - `k_pe`: RoPE-applied key component [qk_rope_head_dim]
//!
//! During attention:
//! - Prefill: Expand latent to full K/V for compute efficiency
//! - Decode: Can operate in latent space for memory efficiency

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Linear, RmsNorm};

use crate::kv_cache::mla_cache_engine::MLACacheEngine;
use crate::kv_cache::BlockId;
use crate::layers::RotaryEmbedding;

/// MLA Attention layer with compressed KV cache support.
///
/// Uses MLACacheEngine to store compressed latent representations
/// instead of full K/V tensors.
pub struct MLAAttention {
    // Query projections (low-rank path)
    q_a_proj: Option<Linear>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Option<Linear>,
    // Query projection (direct path, for smaller models)
    q_proj: Option<Linear>,
    // KV projections
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    // Output projection
    o_proj: Linear,
    // RoPE
    rotary_emb: RotaryEmbedding,
    // Config
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    head_dim: usize,
    /// Pre-scaling factor for queries (mscale^2 for YaRN).
    q_scale: f64,
}

impl MLAAttention {
    /// Create MLA attention from projections.
    ///
    /// # Arguments
    /// * `q_a_proj`, `q_a_layernorm`, `q_b_proj` - Low-rank query path (for large models)
    /// * `q_proj` - Direct query projection (for smaller models, mutually exclusive with above)
    /// * `kv_a_proj_with_mqa` - Projects hidden state to latent + rope key
    /// * `kv_a_layernorm` - Normalizes the latent before caching
    /// * `kv_b_proj` - Expands latent to K_nope and V
    /// * `o_proj` - Output projection
    /// * `rotary_emb` - Rotary embedding for positional encoding
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_a_proj: Option<Linear>,
        q_a_layernorm: Option<RmsNorm>,
        q_b_proj: Option<Linear>,
        q_proj: Option<Linear>,
        kv_a_proj_with_mqa: Linear,
        kv_a_layernorm: RmsNorm,
        kv_b_proj: Linear,
        o_proj: Linear,
        rotary_emb: RotaryEmbedding,
        num_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        q_scale: f64,
    ) -> Self {
        Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            head_dim: qk_nope_head_dim + qk_rope_head_dim,
            q_scale,
        }
    }

    /// Prefill forward pass with MLA cache.
    ///
    /// Expands the compressed latent to full K/V for attention computation.
    /// This is compute-friendly for longer sequences.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache: &mut MLACacheEngine,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Split Q into nope and rope parts
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection through latent space
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize kv_a (this is the compressed latent to cache)
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;

        // Write compressed latent to MLA cache
        let kv_c_for_cache = kv_a.reshape((batch_size * seq_len, self.kv_lora_rank))?;

        // Apply RoPE to k_pe
        let q_pe_for_rope = q_pe.transpose(1, 2)?;
        let k_pe_for_rope = k_pe_raw.reshape((batch_size, 1, seq_len, self.qk_rope_head_dim))?;
        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_for_rope, &k_pe_for_rope, seqlen_offset)?;
        let q_pe = q_pe_rotated.transpose(1, 2)?;

        // Flatten k_pe for cache
        let k_pe_for_cache = k_pe_rotated
            .squeeze(1)?
            .transpose(0, 1)?
            .reshape((batch_size * seq_len, self.qk_rope_head_dim))?;

        // Write to MLA cache
        cache
            .write(&kv_c_for_cache, &k_pe_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache write: {e}")))?;

        // Read and expand from cache for attention
        let num_tokens = seqlen_offset + seq_len;
        let (k_nope_cached, k_pe_cached, v_cached) = cache
            .read_expand_prefill(block_ids, num_tokens, &self.kv_b_proj)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache read: {e}")))?;

        // Broadcast k_pe to all heads
        let k_pe_expanded = k_pe_cached
            .unsqueeze(1)?
            .broadcast_as((num_tokens, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Concatenate nope and rope parts for K
        let k_full = Tensor::cat(&[&k_nope_cached, &k_pe_expanded], D::Minus1)?;

        // Concatenate Q parts
        let q_full = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;

        // Apply q_scale
        let q_full = (q_full * self.q_scale)?;

        // Compute attention
        let attn_output = self.compute_attention(
            &q_full,
            &k_full,
            &v_cached,
            attention_mask,
            batch_size,
            seq_len,
            num_tokens,
        )?;

        // Output projection
        let attn_output =
            attn_output.reshape((batch_size, seq_len, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    /// Decode forward pass with MLA cache.
    ///
    /// For decode, we still expand to full K/V but only for cached tokens.
    /// Future optimization: operate in latent space for memory efficiency.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        cache: &mut MLACacheEngine,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        debug_assert_eq!(seq_len, 1, "Decode expects single token");

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;

        // Split Q
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection for new token
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw = kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;

        // Apply RoPE
        let q_pe_for_rope = q_pe.transpose(1, 2)?;
        let k_pe_for_rope = k_pe_raw.reshape((batch_size, 1, 1, self.qk_rope_head_dim))?;
        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_for_rope, &k_pe_for_rope, seqlen_offset)?;
        let q_pe = q_pe_rotated.transpose(1, 2)?;

        // Write new token to cache
        let kv_c_for_cache = kv_a.reshape((batch_size, self.kv_lora_rank))?;
        let k_pe_for_cache = k_pe_rotated.reshape((batch_size, self.qk_rope_head_dim))?;

        cache
            .write(&kv_c_for_cache, &k_pe_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache write: {e}")))?;

        // Read and expand from cache for attention
        let num_tokens = seqlen_offset + 1;
        let (k_nope_cached, k_pe_cached, v_cached) = cache
            .read_expand_prefill(block_ids, num_tokens, &self.kv_b_proj)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache read: {e}")))?;

        // Broadcast k_pe to all heads
        let k_pe_expanded = k_pe_cached
            .unsqueeze(1)?
            .broadcast_as((num_tokens, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Full K
        let k_full = Tensor::cat(&[&k_nope_cached, &k_pe_expanded], D::Minus1)?;

        // Full Q
        let q_full = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;
        let q_full = (q_full * self.q_scale)?;

        // Compute attention
        let attn_output =
            self.compute_attention(&q_full, &k_full, &v_cached, None, batch_size, 1, num_tokens)?;

        // Output projection
        let attn_output = attn_output.reshape((batch_size, 1, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    fn project_query(&self, x: &Tensor) -> Result<Tensor> {
        if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_latent = q_a.forward(x)?;
            let q_latent = q_a_ln.forward(&q_latent)?;
            q_b.forward(&q_latent)
        } else if let Some(q_proj) = &self.q_proj {
            q_proj.forward(x)
        } else {
            candle_core::bail!("No query projection configured")
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        _batch_size: usize,
        _q_len: usize,
        _kv_len: usize,
    ) -> Result<Tensor> {
        // q: [batch, q_len, num_heads, head_dim]
        // k: [kv_len, num_heads, head_dim]
        // v: [kv_len, num_heads, v_head_dim]

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Reshape for batch matmul
        // q: [batch, num_heads, q_len, head_dim]
        let q = q.transpose(1, 2)?;
        // k: [1, num_heads, kv_len, head_dim]
        let k = k.unsqueeze(0)?.transpose(1, 2)?;
        // v: [1, num_heads, kv_len, v_head_dim]
        let v = v.unsqueeze(0)?.transpose(1, 2)?;

        // Attention scores: [batch, num_heads, q_len, kv_len]
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Apply causal mask if provided
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Output: [batch, num_heads, q_len, v_head_dim]
        let output = attn_weights.matmul(&v)?;

        // Reshape: [batch, q_len, num_heads, v_head_dim]
        output.transpose(1, 2)
    }

    /// Get the kv_b_proj for external use (e.g., in cache expansion).
    pub fn kv_b_proj(&self) -> &Linear {
        &self.kv_b_proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use crate::kv_cache::mla_cache_config::MLACacheConfig;

    fn create_test_attention(device: &Device) -> MLAAttention {
        let vb = VarBuilder::zeros(DType::F32, device);
        let hidden_size = 64;
        let num_heads = 4;
        let qk_nope_head_dim = 8;
        let qk_rope_head_dim = 4;
        let v_head_dim = 8;
        let kv_lora_rank = 16;
        let head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let q_proj =
            candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q")).unwrap();
        let kv_a =
            candle_nn::linear_no_bias(hidden_size, kv_lora_rank + qk_rope_head_dim, vb.pp("kv_a"))
                .unwrap();
        let kv_a_ln = candle_nn::rms_norm(kv_lora_rank, 1e-5, vb.pp("kv_a_ln")).unwrap();
        let kv_b = candle_nn::linear_no_bias(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            vb.pp("kv_b"),
        )
        .unwrap();
        let o_proj =
            candle_nn::linear_no_bias(num_heads * v_head_dim, hidden_size, vb.pp("o")).unwrap();

        let rotary =
            crate::layers::RotaryEmbedding::new(qk_rope_head_dim, 512, 10000.0, DType::F32, device)
                .unwrap();

        MLAAttention::new(
            None,
            None,
            None,
            Some(q_proj),
            kv_a,
            kv_a_ln,
            kv_b,
            o_proj,
            rotary,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            1.0,
        )
    }

    fn create_test_cache(device: &Device) -> MLACacheEngine {
        let config = MLACacheConfig::new(
            16, // kv_lora_rank
            4,  // qk_rope_head_dim
            8,  // qk_nope_head_dim
            8,  // v_head_dim
            4,  // num_heads
            4,  // block_size
            8,  // num_blocks
            1,  // num_layers
            DType::F32,
            device.clone(),
        );
        MLACacheEngine::new(&config).unwrap()
    }

    #[test]
    fn test_mla_attention_creation() {
        let device = Device::Cpu;
        let _attn = create_test_attention(&device);
    }

    #[test]
    fn test_mla_attention_prefill() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);
        let mut cache = create_test_cache(&device);

        let x = Tensor::randn(0f32, 1f32, (1, 4, 64), &device).unwrap();
        let block_ids = vec![0];
        let slot_mapping: Vec<usize> = (0..4).collect();

        let output = attn
            .forward_prefill(&x, None, 0, &mut cache, &block_ids, &slot_mapping)
            .unwrap();

        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_mla_attention_decode() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);
        let mut cache = create_test_cache(&device);

        // First do prefill with 3 tokens (leave room for decode)
        let x_prefill = Tensor::randn(0f32, 1f32, (1, 3, 64), &device).unwrap();
        let block_ids = vec![0];
        let slot_mapping_prefill: Vec<usize> = (0..3).collect();

        attn.forward_prefill(
            &x_prefill,
            None,
            0,
            &mut cache,
            &block_ids,
            &slot_mapping_prefill,
        )
        .unwrap();

        // Then decode (slot 3 is still within block 0 which has 4 slots)
        let x_decode = Tensor::randn(0f32, 1f32, (1, 1, 64), &device).unwrap();
        let slot_mapping_decode = vec![3]; // Next slot after prefill

        let output = attn
            .forward_decode(
                &x_decode,
                3, // seqlen_offset = prefill length
                &mut cache,
                &block_ids,
                &slot_mapping_decode,
            )
            .unwrap();

        assert_eq!(output.dims(), &[1, 1, 64]);
    }
}
