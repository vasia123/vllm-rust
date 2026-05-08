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

    /// Execute batched decode attention, also returning log-sum-exp softmax statistics.
    ///
    /// Required for Decode Context Parallelism (DCP): each DCP rank computes attention
    /// over its local KV slice, then ranks exchange LSE to merge partial results correctly.
    ///
    /// # Returns
    /// - `output`: attention result in shape `[batch, num_heads, head_dim]` (3D)
    /// - `lse`: log-sum-exp values `[batch, num_heads]` (natural log), or `None` if
    ///   this backend does not expose LSE (DCP correction will be skipped)
    ///
    /// # Default Implementation
    /// Calls `batched_decode_attention` and reshapes output to 3D. Returns `None` for LSE.
    /// Override in LSE-capable backends (e.g. FlashInfer) for accurate DCP merging.
    #[allow(clippy::too_many_arguments)]
    fn batched_decode_attention_with_lse(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &mut CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let out_2d = self.batched_decode_attention(
            q,
            k_new,
            v_new,
            cache_engine,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )?;
        let batch = out_2d.dim(0)?;
        let out_3d = out_2d.reshape((batch, num_heads, head_dim))?;
        Ok((out_3d, None))
    }

    /// Pre-build a backend-specific prefill plan that can be replayed
    /// across every attention layer of the same forward pass.
    ///
    /// Symmetric to [`Self::prepare_decode_plan`]. Backends that benefit
    /// from caching (FlashInfer's `BatchPrefillPlan`, which otherwise
    /// rebuilds CPU-side work-estimates + a page-locked allocation
    /// ×36 per Qwen3-4B prefill) override this and return `Some(plan)`.
    /// Other backends return `None` and `prefill_attention_with_plan`
    /// falls back to the per-layer eager path.
    ///
    /// Engine usage (Stage 14-A): build once before `model.forward`,
    /// pass to each layer's [`Self::prefill_attention_with_plan`] call
    /// via the metadata struct, drop at end of forward.
    #[allow(clippy::too_many_arguments)]
    fn prepare_prefill_plan(
        &self,
        cache_engine: &CacheEngine,
        metadata: &PagedAttentionMetadata,
        query_dtype: candle_core::DType,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>> {
        let _ = (
            cache_engine,
            metadata,
            query_dtype,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        Ok(None)
    }

    /// Replay a pre-built prefill plan returned by
    /// [`Self::prepare_prefill_plan`]. Default implementation ignores the
    /// plan and falls through to the regular `prefill_attention`, so
    /// callers can pass a cached plan unconditionally without checking
    /// which backend they have.
    #[allow(clippy::too_many_arguments)]
    fn prefill_attention_with_plan(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_engine: &mut CacheEngine,
        metadata: &PagedAttentionMetadata,
        plan: Option<&(dyn std::any::Any + Send + Sync)>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let _ = plan;
        self.prefill_attention(
            q,
            k,
            v,
            attention_mask,
            cache_engine,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    /// Pre-build a backend-specific decode plan that can be replayed across
    /// every attention layer of the same forward pass.
    ///
    /// Backends that benefit from caching (FlashInfer's `BatchDecodePlan`,
    /// which otherwise rebuilds and host-syncs `kv_indptr` ×36 per token on
    /// Qwen3-4B) override this and return `Some(plan)`. Other backends
    /// return `None` and `batched_decode_attention_with_plan` falls back
    /// to the per-layer eager path.
    ///
    /// The plan is opaque to the caller; pass it back unchanged via the
    /// `plan: Option<&dyn Any>` argument of
    /// `batched_decode_attention_with_plan`.
    #[allow(clippy::too_many_arguments)]
    fn prepare_decode_plan(
        &self,
        cache_engine: &CacheEngine,
        metadata: &BatchedDecodeMetadata,
        query_dtype: candle_core::DType,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>> {
        let _ = (
            cache_engine,
            metadata,
            query_dtype,
            num_heads,
            num_kv_heads,
            head_dim,
        );
        Ok(None)
    }

    /// Replay a pre-built decode plan returned by `prepare_decode_plan`.
    /// Default implementation ignores the plan and falls through to the
    /// regular `batched_decode_attention`, so callers can pass a cached
    /// plan unconditionally without checking which backend they have.
    #[allow(clippy::too_many_arguments)]
    fn batched_decode_attention_with_plan(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &mut CacheEngine,
        metadata: &BatchedDecodeMetadata,
        plan: Option<&(dyn std::any::Any + Send + Sync)>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let _ = plan;
        self.batched_decode_attention(
            q,
            k_new,
            v_new,
            cache_engine,
            metadata,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

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
///
/// Returns a reference to a process-wide singleton attention backend.
///
/// Stage 14-A.0c — was `Box<dyn AttentionBackend>` constructed fresh on
/// every call. That ate 36 box allocations per Qwen3 forward AND made
/// the per-backend workspace die together with the box, which leaves
/// any plan returned by `prepare_*_plan` with dangling pointers.
/// The singleton outlives every plan that flows through it; the inner
/// workspace `Mutex` keeps the (single-threaded) engine forward path
/// serial.
///
/// Tests that construct backend instances directly (e.g.
/// `FlashInferBackend::with_block_size(...)`) keep their own per-instance
/// workspace and don't go through this singleton — `cargo test --jobs N`
/// stays race-free.
pub fn select_backend() -> &'static dyn AttentionBackend {
    use std::sync::OnceLock;
    // `Box::leak` returns `&'static mut`; we coerce to `&'static dyn`
    // by binding through `OnceLock<Box<dyn AttentionBackend + Sync>>`.
    static BACKEND: OnceLock<Box<dyn AttentionBackend + Sync>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
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
        })
        .as_ref()
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
