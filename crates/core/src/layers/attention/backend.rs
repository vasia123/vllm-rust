//! Attention backend traits for pluggable attention implementations.
//!
//! This module provides the `AttentionBackend` trait which enables different
//! attention implementations (naive, FlashAttention, FlashInfer, etc.) to be
//! used interchangeably.

use candle_core::{Result, Tensor};

use crate::kv_cache::{BlockId, CacheEngine};

/// Metadata for a single sequence in paged attention.
#[derive(Debug, Clone)]
pub struct PagedAttentionMetadata<'a> {
    /// Block IDs for this sequence's KV cache
    pub block_ids: &'a [BlockId],
    /// Slot mapping for writing new tokens
    pub slot_mapping: &'a [usize],
    /// Total sequence length (including new tokens)
    pub seq_len: usize,
    /// Offset into the sequence (for decode phase)
    pub seqlen_offset: usize,
}

/// Metadata for batched decode attention.
#[derive(Debug, Clone)]
pub struct BatchedDecodeMetadata<'a> {
    /// Block IDs for each sequence
    pub seq_block_ids: &'a [&'a [BlockId]],
    /// Flattened slot mapping for all sequences
    pub all_slot_mapping: &'a [usize],
    /// KV lengths for each sequence
    pub kv_lengths: &'a [usize],
}

/// Attention backend trait for pluggable attention implementations.
///
/// This trait enables different attention kernels to be used:
/// - `NaiveAttention`: Reference implementation using standard matmuls
/// - `FlashAttention`: FlashAttention-2 via candle-flash-attn
/// - Future: FlashInfer, PagedAttention kernels, etc.
pub trait AttentionBackend: Send + Sync {
    /// Returns the name of this backend.
    fn name(&self) -> &'static str;

    /// Execute prefill attention (multiple query tokens).
    ///
    /// # Arguments
    /// * `q` - Query tensor `[batch, num_heads, seq_len, head_dim]`
    /// * `k` - Key tensor `[batch, num_kv_heads, seq_len, head_dim]`
    /// * `v` - Value tensor `[batch, num_kv_heads, seq_len, head_dim]`
    /// * `attention_mask` - Optional causal mask
    /// * `cache_engine` - KV cache engine for reading/writing
    /// * `metadata` - Paged attention metadata
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of KV heads (for GQA)
    /// * `head_dim` - Dimension per head
    ///
    /// Returns attention output `[batch, seq_len, num_heads * head_dim]`
    #[allow(clippy::too_many_arguments)]
    fn prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_engine: &mut CacheEngine,
        metadata: &PagedAttentionMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor>;

    /// Execute batched decode attention (single query token per sequence).
    ///
    /// # Arguments
    /// * `q` - Query tensor `[batch_size, num_heads, head_dim]` (after RoPE)
    /// * `k_new` - New key tensor `[batch_size, num_kv_heads, head_dim]`
    /// * `v_new` - New value tensor `[batch_size, num_kv_heads, head_dim]`
    /// * `cache_engine` - KV cache engine
    /// * `metadata` - Batched decode metadata
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of KV heads
    /// * `head_dim` - Dimension per head
    ///
    /// Returns attention output `[batch_size, num_heads * head_dim]`
    #[allow(clippy::too_many_arguments)]
    fn batched_decode_attention(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &mut CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor>;

    /// Check if this backend supports the given configuration.
    fn supports_config(&self, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> bool {
        // Default: support all configurations
        let _ = (num_heads, num_kv_heads, head_dim);
        true
    }

    /// Check if this backend requires specific dtype.
    fn supported_dtypes(&self) -> &[candle_core::DType] {
        &[
            candle_core::DType::F32,
            candle_core::DType::F16,
            candle_core::DType::BF16,
        ]
    }
}

/// Select the best available attention backend.
///
/// Priority order:
/// 1. FlashInfer (best for paged attention decode)
/// 2. FlashAttention (good for prefill)
/// 3. Naive (fallback)
pub fn select_backend() -> Box<dyn AttentionBackend> {
    #[cfg(feature = "flashinfer")]
    {
        Box::new(super::flashinfer::FlashInferBackend::new())
    }

    #[cfg(all(feature = "flash-attn", not(feature = "flashinfer")))]
    {
        Box::new(super::flash::FlashAttentionBackend::new())
    }

    #[cfg(not(any(feature = "flash-attn", feature = "flashinfer")))]
    {
        Box::new(super::naive::NaiveAttentionBackend::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_backend_returns_valid() {
        let backend = select_backend();
        assert!(!backend.name().is_empty());
    }
}
