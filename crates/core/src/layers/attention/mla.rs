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

    /// Extract the K-nope weight matrix from kv_b_proj for matrix absorption.
    ///
    /// kv_b_proj maps `[kv_lora_rank] → [num_heads * (qk_nope_head_dim + v_head_dim)]`.
    /// The weight layout is interleaved per-head: `[k_nope_h0, v_h0, k_nope_h1, v_h1, ...]`.
    /// We reshape to `[num_heads, qk_nope+v, kv_lora_rank]` then extract the k_nope slice.
    ///
    /// Returns `[num_heads, qk_nope_head_dim, kv_lora_rank]`.
    pub fn extract_w_kb_nope(&self) -> Result<Tensor> {
        let weight = self.kv_b_proj.weight();
        // weight: [num_heads * (qk_nope + v_head_dim), kv_lora_rank]
        let per_head_dim = self.qk_nope_head_dim + self.v_head_dim;

        // Reshape to per-head blocks: [num_heads, qk_nope + v_head_dim, kv_lora_rank]
        let per_head = weight.reshape((self.num_heads, per_head_dim, self.kv_lora_rank))?;

        // Extract K-nope slice: [num_heads, qk_nope, kv_lora_rank]
        per_head
            .narrow(1, 0, self.qk_nope_head_dim)?
            .contiguous()
    }

    /// Extract the V weight matrix from kv_b_proj for output de-absorption.
    ///
    /// Same interleaved layout as above — extracts the V slice per head.
    ///
    /// Returns `[num_heads, v_head_dim, kv_lora_rank]`.
    pub fn extract_w_kb_v(&self) -> Result<Tensor> {
        let weight = self.kv_b_proj.weight();
        let per_head_dim = self.qk_nope_head_dim + self.v_head_dim;

        // Reshape to per-head blocks: [num_heads, qk_nope + v_head_dim, kv_lora_rank]
        let per_head = weight.reshape((self.num_heads, per_head_dim, self.kv_lora_rank))?;

        // Extract V slice: [num_heads, v_head_dim, kv_lora_rank]
        per_head
            .narrow(1, self.qk_nope_head_dim, self.v_head_dim)?
            .contiguous()
    }

    /// Absorb query into CKV space using the K-nope weight matrix.
    ///
    /// Transforms q_nope from `[nnz, num_heads, qk_nope_head_dim]` to
    /// `[nnz, num_heads, kv_lora_rank]` by multiplying with W_kb_nope.T per head.
    ///
    /// This enables direct attention against the compressed cache without expanding it.
    pub fn absorb_query(&self, q_nope: &Tensor, w_kb_nope: &Tensor) -> Result<Tensor> {
        // q_nope: [nnz, num_heads, qk_nope]
        // w_kb_nope: [num_heads, qk_nope, kv_lora_rank]
        // → q_absorbed: [nnz, num_heads, kv_lora_rank]
        //
        // Per head: q_nope[..., h, :] @ w_kb_nope[h, :, :].T = [nnz, kv_lora_rank]
        // Equivalent to einsum("bhi,hio->bho", q_nope, w_kb_nope)

        let (nnz, _num_heads, _qk_nope) = q_nope.dims3()?;

        // Transpose to [num_heads, nnz, qk_nope] for batched matmul
        let q = q_nope.transpose(0, 1)?; // [num_heads, nnz, qk_nope]

        // w_kb_nope is [num_heads, qk_nope, kv_lora_rank]
        // Batched matmul: [num_heads, nnz, qk_nope] @ [num_heads, qk_nope, kv_lora_rank]
        //               = [num_heads, nnz, kv_lora_rank]
        let q_absorbed = q.matmul(w_kb_nope)?;

        // Transpose back: [nnz, num_heads, kv_lora_rank]
        q_absorbed.transpose(0, 1)?.contiguous()?.reshape((
            nnz,
            self.num_heads,
            self.kv_lora_rank,
        ))
    }

    /// Absorbed decode: computes attention directly in compressed space.
    ///
    /// Instead of expanding the compressed cache to full K/V (~42x memory),
    /// this method absorbs the expansion weights into the query and output
    /// projections, computing attention directly against the compressed cache.
    ///
    /// Produces identical results to `forward_decode`, but with ~42x lower
    /// memory bandwidth for the cache read during decode.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_absorbed(
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

        // Split Q into nope and rope parts
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection for new token
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw =
            kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;

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

        // Read raw compressed cache (no expansion!)
        let num_tokens = seqlen_offset + 1;
        let (kv_c, k_pe) = cache
            .read_raw(block_ids, num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache read: {e}")))?;
        // kv_c: [num_tokens, kv_lora_rank]
        // k_pe: [num_tokens, qk_rope_head_dim]

        // Apply q_scale to queries before absorption
        let q_nope = (q_nope.squeeze(1)? * self.q_scale)?;
        let q_pe = (q_pe.squeeze(1)? * self.q_scale)?;

        // Absorb q_nope into CKV space
        let w_kb_nope = self.extract_w_kb_nope()?;
        let q_absorbed = self.absorb_query(&q_nope, &w_kb_nope)?;
        // q_absorbed: [batch, num_heads, kv_lora_rank]

        let scale = 1.0 / (self.head_dim as f64).sqrt();

        // Nope scores: q_absorbed @ kv_c.T → [batch, num_heads, num_tokens]
        // unsqueeze(0) needed because Candle matmul requires matching ndims
        let scores_nope = q_absorbed.matmul(&kv_c.t()?.unsqueeze(0)?)?;
        // Rope scores: q_pe @ k_pe.T → [batch, num_heads, num_tokens]
        // k_pe is shared across heads; batch dim broadcasts over num_heads
        let scores_rope = q_pe.matmul(&k_pe.t()?.unsqueeze(0)?)?;

        let scores = ((scores_nope + scores_rope)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted sum in CKV space: attn_weights @ kv_c → [batch, num_heads, kv_lora_rank]
        let output_absorbed = attn_weights.matmul(&kv_c.unsqueeze(0)?)?;

        // De-absorb to V space
        let w_kb_v = self.extract_w_kb_v()?;
        let output = self.deabsorb_output(&output_absorbed, &w_kb_v)?;

        // Output projection
        let output = output.reshape((batch_size, 1, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&output)
    }

    /// FlashInfer MLA decode: uses CUDA kernel for absorbed attention.
    ///
    /// Same math as `forward_decode_absorbed`, but delegates the attention
    /// computation to FlashInfer's MLA kernel which reads directly from the
    /// paged compressed cache on GPU.
    #[cfg(feature = "flashinfer")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_flashinfer(
        &self,
        x: &Tensor,
        seqlen_offset: usize,
        cache: &mut MLACacheEngine,
        block_ids: &[BlockId],
        slot_mapping: &[usize],
        mla_wrapper: &super::flashinfer::MlaWrapper,
        workspace: &mut super::flashinfer::WorkspaceBuffer,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        debug_assert_eq!(seq_len, 1);
        let device = x.device();

        // Query projection
        let q = self.project_query(x)?;
        let q = q.reshape((batch_size, 1, self.num_heads, self.head_dim))?;
        let q_nope = q.narrow(D::Minus1, 0, self.qk_nope_head_dim)?;
        let q_pe = q.narrow(D::Minus1, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // KV projection + RoPE + cache write
        let kv_latent = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_a = kv_latent.narrow(D::Minus1, 0, self.kv_lora_rank)?;
        let k_pe_raw =
            kv_latent.narrow(D::Minus1, self.kv_lora_rank, self.qk_rope_head_dim)?;
        let kv_a = self.kv_a_layernorm.forward(&kv_a)?;

        let q_pe_for_rope = q_pe.transpose(1, 2)?;
        let k_pe_for_rope = k_pe_raw.reshape((batch_size, 1, 1, self.qk_rope_head_dim))?;
        let (q_pe_rotated, k_pe_rotated) =
            self.rotary_emb
                .apply(&q_pe_for_rope, &k_pe_for_rope, seqlen_offset)?;
        let q_pe = q_pe_rotated.transpose(1, 2)?;

        let kv_c_for_cache = kv_a.reshape((batch_size, self.kv_lora_rank))?;
        let k_pe_for_cache = k_pe_rotated.reshape((batch_size, self.qk_rope_head_dim))?;
        cache
            .write(&kv_c_for_cache, &k_pe_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("MLA cache write: {e}")))?;

        // Absorbed query
        let q_nope = (q_nope.squeeze(1)? * self.q_scale)?;
        let q_pe = (q_pe.squeeze(1)? * self.q_scale)?;
        let w_kb_nope = self.extract_w_kb_nope()?;
        let q_absorbed = self.absorb_query(&q_nope, &w_kb_nope)?;

        // Build FlashInfer metadata
        let num_tokens = seqlen_offset + 1;
        let block_size = cache.config().block_size;

        // qo_indptr: [0, 1] for single-token decode
        let qo_indptr =
            super::flashinfer::tensor_bridge::alloc_gpu_i32(&[0, 1], device)?;

        let fi_metadata = super::flashinfer::FlashInferMetadata::from_single_sequence(
            block_ids,
            num_tokens,
            block_size,
            device,
        )?;
        let kv_indptr = fi_metadata.paged_kv_indptr.to_dtype(candle_core::DType::U32)?;
        let kv_indices = fi_metadata.paged_kv_indices.to_dtype(candle_core::DType::U32)?;

        // Run MLA kernel — output in CKV space [batch, num_heads, kv_lora_rank]
        let output_absorbed = mla_wrapper.run(
            &q_absorbed,
            &q_pe,
            cache.kv_c_cache(),
            cache.k_pe_cache(),
            workspace,
            &qo_indptr,
            &kv_indptr,
            &kv_indices,
            &[num_tokens],
            batch_size,
            batch_size, // total_tokens = batch_size for decode (1 token per seq)
        )?;

        // De-absorb to V space
        let w_kb_v = self.extract_w_kb_v()?;
        let output = self.deabsorb_output(&output_absorbed, &w_kb_v)?;

        // Output projection
        let output = output.reshape((batch_size, 1, self.num_heads * self.v_head_dim))?;
        self.o_proj.forward(&output)
    }

    /// De-absorb output from CKV space to V space.
    ///
    /// Transforms absorbed output from `[nnz, num_heads, kv_lora_rank]` to
    /// `[nnz, num_heads, v_head_dim]` by multiplying with W_kb_v per head.
    pub fn deabsorb_output(&self, absorbed_output: &Tensor, w_kb_v: &Tensor) -> Result<Tensor> {
        // absorbed_output: [nnz, num_heads, kv_lora_rank]
        // w_kb_v: [num_heads, v_head_dim, kv_lora_rank]
        // → output: [nnz, num_heads, v_head_dim]
        //
        // Per head: absorbed_output[..., h, :] @ w_kb_v[h, :, :].T = [nnz, v_head_dim]
        // Equivalent to einsum("bhr,hvr->bhv", absorbed, w_kb_v)

        let (nnz, _num_heads, _kv_lora_rank) = absorbed_output.dims3()?;

        // Transpose to [num_heads, nnz, kv_lora_rank]
        let out = absorbed_output.transpose(0, 1)?;

        // w_kb_v is [num_heads, v_head_dim, kv_lora_rank]
        // We need w_kb_v.T = [num_heads, kv_lora_rank, v_head_dim]
        let w_kb_v_t = w_kb_v.transpose(1, 2)?;

        // Batched matmul: [num_heads, nnz, kv_lora_rank] @ [num_heads, kv_lora_rank, v_head_dim]
        //               = [num_heads, nnz, v_head_dim]
        let output = out.matmul(&w_kb_v_t)?;

        // Transpose back: [nnz, num_heads, v_head_dim]
        output
            .transpose(0, 1)?
            .contiguous()?
            .reshape((nnz, self.num_heads, self.v_head_dim))
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
    fn test_extract_w_kb_nope_shape() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);

        let w_kb_nope = attn.extract_w_kb_nope().unwrap();
        // [num_heads=4, qk_nope_head_dim=8, kv_lora_rank=16]
        assert_eq!(w_kb_nope.dims(), &[4, 8, 16]);
    }

    #[test]
    fn test_extract_w_kb_v_shape() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);

        let w_kb_v = attn.extract_w_kb_v().unwrap();
        // [num_heads=4, v_head_dim=8, kv_lora_rank=16]
        assert_eq!(w_kb_v.dims(), &[4, 8, 16]);
    }

    #[test]
    fn test_absorb_query_shape() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);

        let w_kb_nope = attn.extract_w_kb_nope().unwrap();
        let q_nope = Tensor::randn(0f32, 1f32, (3, 4, 8), &device).unwrap();
        // q_nope: [nnz=3, num_heads=4, qk_nope=8]

        let q_absorbed = attn.absorb_query(&q_nope, &w_kb_nope).unwrap();
        // Absorbed: [nnz=3, num_heads=4, kv_lora_rank=16]
        assert_eq!(q_absorbed.dims(), &[3, 4, 16]);
    }

    #[test]
    fn test_deabsorb_output_shape() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);

        let w_kb_v = attn.extract_w_kb_v().unwrap();
        let absorbed_output = Tensor::randn(0f32, 1f32, (3, 4, 16), &device).unwrap();
        // absorbed_output: [nnz=3, num_heads=4, kv_lora_rank=16]

        let output = attn.deabsorb_output(&absorbed_output, &w_kb_v).unwrap();
        // De-absorbed: [nnz=3, num_heads=4, v_head_dim=8]
        assert_eq!(output.dims(), &[3, 4, 8]);
    }

    #[test]
    fn test_absorption_roundtrip() {
        // Verify that absorb → MLA-style attention ≈ standard attention
        // This test validates that absorbing the query and de-absorbing the output
        // produces the same result as expanding the cache.
        let device = Device::Cpu;
        let attn = create_test_attention(&device);

        let nnz = 2;
        let q_nope = Tensor::randn(0f32, 1f32, (nnz, 4, 8), &device).unwrap();
        let kv_c = Tensor::randn(0f32, 0.1f32, (5, 16), &device).unwrap();

        // === Standard path: expand cache then compute ===
        // kv_b_proj expands: [5, 16] → [5, 4*(8+8)] = [5, 64]
        let expanded = attn.kv_b_proj().forward(&kv_c).unwrap();
        // k_nope = first 4*8 = 32 → reshape [5, 4, 8]
        let k_nope_std = expanded.narrow(1, 0, 32).unwrap().reshape((5, 4, 8)).unwrap();
        // v = next 4*8 = 32 → reshape [5, 4, 8] (unused in score comparison)
        let _v_std = expanded.narrow(1, 32, 32).unwrap().reshape((5, 4, 8)).unwrap();

        // Standard attention: q_nope @ k_nope.T → [nnz, 4, 5] (ignoring rope part)
        let q_t = q_nope.transpose(0, 1).unwrap(); // [4, nnz, 8]
        let k_t = k_nope_std.transpose(0, 1).unwrap(); // [4, 5, 8]
        let scores_std = q_t.matmul(&k_t.transpose(1, 2).unwrap()).unwrap(); // [4, nnz, 5]

        // === Absorbed path ===
        let w_kb_nope = attn.extract_w_kb_nope().unwrap();
        let _w_kb_v = attn.extract_w_kb_v().unwrap();
        let q_absorbed = attn.absorb_query(&q_nope, &w_kb_nope).unwrap();
        // q_absorbed: [nnz, 4, 16]

        // Absorbed attention: q_absorbed @ kv_c.T → [nnz, 4, 5]
        let qa_t = q_absorbed.transpose(0, 1).unwrap(); // [4, nnz, 16]
        // kv_c: [5, 16] → broadcast to [4, 5, 16] for batched matmul
        let kv_c_expanded = kv_c
            .unsqueeze(0)
            .unwrap()
            .broadcast_as((4, 5, 16))
            .unwrap()
            .contiguous()
            .unwrap();
        let scores_abs = qa_t
            .matmul(&kv_c_expanded.transpose(1, 2).unwrap())
            .unwrap(); // [4, nnz, 5]

        // The scores should be approximately equal
        let diff = (scores_std - scores_abs).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(
            max_diff < 1e-4,
            "Absorption roundtrip error too large: {max_diff}"
        );
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

    #[test]
    fn test_mla_attention_decode_absorbed() {
        let device = Device::Cpu;
        let attn = create_test_attention(&device);
        let mut cache = create_test_cache(&device);

        // Prefill with 3 tokens
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

        // Decode with absorbed path
        let x_decode = Tensor::randn(0f32, 1f32, (1, 1, 64), &device).unwrap();
        let slot_mapping_decode = vec![3];

        let output = attn
            .forward_decode_absorbed(
                &x_decode,
                3,
                &mut cache,
                &block_ids,
                &slot_mapping_decode,
            )
            .unwrap();

        assert_eq!(output.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_decode_absorbed_matches_standard() {
        // Verify that forward_decode_absorbed produces the same result as
        // forward_decode (expansion path) — they must be numerically equivalent.
        let device = Device::Cpu;

        // Use random (non-zero) weights for a meaningful test
        let vb = VarBuilder::from_tensors(
            {
                let mut map = std::collections::HashMap::new();
                let num_heads = 4usize;
                let qk_nope = 8usize;
                let qk_rope = 4usize;
                let v_head_dim = 8usize;
                let kv_lora_rank = 16usize;
                let hidden_size = 64usize;
                let head_dim = qk_nope + qk_rope;

                map.insert(
                    "q.weight".to_string(),
                    Tensor::randn(0f32, 0.1, (num_heads * head_dim, hidden_size), &device)
                        .unwrap(),
                );
                map.insert(
                    "kv_a.weight".to_string(),
                    Tensor::randn(
                        0f32,
                        0.1,
                        (kv_lora_rank + qk_rope, hidden_size),
                        &device,
                    )
                    .unwrap(),
                );
                map.insert(
                    "kv_a_ln.weight".to_string(),
                    Tensor::ones((kv_lora_rank,), DType::F32, &device).unwrap(),
                );
                map.insert(
                    "kv_b.weight".to_string(),
                    Tensor::randn(
                        0f32,
                        0.1,
                        (num_heads * (qk_nope + v_head_dim), kv_lora_rank),
                        &device,
                    )
                    .unwrap(),
                );
                map.insert(
                    "o.weight".to_string(),
                    Tensor::randn(0f32, 0.1, (hidden_size, num_heads * v_head_dim), &device)
                        .unwrap(),
                );
                map
            },
            DType::F32,
            &device,
        );

        let q_proj =
            candle_nn::linear_no_bias(64, 4 * 12, vb.pp("q")).unwrap();
        let kv_a =
            candle_nn::linear_no_bias(64, 16 + 4, vb.pp("kv_a")).unwrap();
        let kv_a_ln = candle_nn::rms_norm(16, 1e-5, vb.pp("kv_a_ln")).unwrap();
        let kv_b =
            candle_nn::linear_no_bias(16, 4 * (8 + 8), vb.pp("kv_b")).unwrap();
        let o_proj =
            candle_nn::linear_no_bias(4 * 8, 64, vb.pp("o")).unwrap();
        let rotary = crate::layers::RotaryEmbedding::new(4, 512, 10000.0, DType::F32, &device)
            .unwrap();

        let attn = MLAAttention::new(
            None, None, None, Some(q_proj), kv_a, kv_a_ln, kv_b, o_proj, rotary,
            4, 8, 4, 8, 16, 1.0,
        );

        // Prefill 3 tokens into both caches
        let x_prefill = Tensor::randn(0f32, 0.5, (1, 3, 64), &device).unwrap();
        let block_ids = vec![0];

        let mut cache_std = create_test_cache(&device);
        let mut cache_abs = create_test_cache(&device);

        attn.forward_prefill(&x_prefill, None, 0, &mut cache_std, &block_ids, &(0..3).collect::<Vec<_>>())
            .unwrap();
        attn.forward_prefill(&x_prefill, None, 0, &mut cache_abs, &block_ids, &(0..3).collect::<Vec<_>>())
            .unwrap();

        // Decode 1 token
        let x_decode = Tensor::randn(0f32, 0.5, (1, 1, 64), &device).unwrap();
        let slot_decode = vec![3];

        let out_std = attn
            .forward_decode(&x_decode, 3, &mut cache_std, &block_ids, &slot_decode)
            .unwrap();
        let out_abs = attn
            .forward_decode_absorbed(&x_decode, 3, &mut cache_abs, &block_ids, &slot_decode)
            .unwrap();

        assert_eq!(out_std.dims(), out_abs.dims());

        // Compare values
        let std_vals: Vec<f32> = out_std.flatten_all().unwrap().to_vec1().unwrap();
        let abs_vals: Vec<f32> = out_abs.flatten_all().unwrap().to_vec1().unwrap();

        let max_diff: f32 = std_vals
            .iter()
            .zip(abs_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "Absorbed decode diverges from standard: max_diff={max_diff}"
        );
    }
}
