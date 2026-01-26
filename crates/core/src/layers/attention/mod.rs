//! Attention backends and operations for transformer inference.
//!
//! This module provides:
//! - Pluggable attention backends (naive, FlashAttention, future: FlashInfer)
//! - Common attention operations (repeat_kv, apply_per_head_norm)
//! - Paged attention for efficient KV cache management
//!
//! # Architecture
//!
//! The attention system uses the Strategy pattern via `AttentionBackend`:
//!
//! - `NaiveAttentionBackend`: Reference implementation using standard matmuls
//! - `FlashAttentionBackend`: FlashAttention-2 via candle-flash-attn
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::layers::attention::{select_backend, PagedAttentionMetadata};
//!
//! let backend = select_backend();
//! let output = backend.prefill_attention(
//!     &q, &k, &v,
//!     Some(&mask),
//!     &cache_engine,
//!     &metadata,
//!     num_heads, num_kv_heads, head_dim,
//! )?;
//! ```

mod backend;
pub mod flash;
pub mod naive;
mod ops;

// Re-export public types
pub use backend::{
    select_backend, AttentionBackend, BatchedDecodeMetadata, PagedAttentionMetadata,
};
pub use ops::{apply_per_head_norm, repeat_kv};

// Re-export backend implementations
pub use flash::FlashAttentionBackend;
pub use naive::NaiveAttentionBackend;

use candle_core::{Result, Tensor};

use crate::kv_cache::{BlockId, CacheEngine};

// ─── Convenience Wrappers ────────────────────────────────────────────────────

/// Batched paged attention for decode: processes all sequences in one fused call.
///
/// Convenience wrapper around the attention backend. For advanced control
/// (e.g., backend selection, custom metadata), use `AttentionBackend` directly.
///
/// # Arguments
/// * `q`, `k_new`, `v_new` - Tensors `[batch_size, num_kv_or_q_heads, head_dim]` after RoPE
/// * `cache_engine` - KV cache engine
/// * `seq_block_ids` - Block IDs for each sequence
/// * `all_slot_mapping` - Slot mapping for writing new tokens
/// * `kv_lengths` - KV lengths per sequence
/// * `num_heads`, `num_kv_heads`, `head_dim` - Attention configuration
///
/// Returns `[batch_size, num_heads * head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn batched_paged_attention_decode(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    cache_engine: &mut CacheEngine,
    seq_block_ids: &[&[BlockId]],
    all_slot_mapping: &[usize],
    kv_lengths: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let backend = select_backend();
    let metadata = BatchedDecodeMetadata {
        seq_block_ids,
        all_slot_mapping,
        kv_lengths,
    };
    backend.batched_decode_attention(
        q,
        k_new,
        v_new,
        cache_engine,
        &metadata,
        num_heads,
        num_kv_heads,
        head_dim,
    )
}

/// Paged attention for prefill: write to cache, read full history, compute GQA attention.
///
/// Convenience wrapper around the attention backend.
///
/// # Arguments
/// * `q`, `k`, `v` - Tensors `[b, heads, seq, head_dim]` (K/V have num_kv_heads)
/// * `attention_mask` - Optional causal mask
/// * `seqlen_offset` - Offset into sequence
/// * `cache_engine` - KV cache engine
/// * `block_ids` - Block IDs for this sequence
/// * `slot_mapping` - Slot mapping for new tokens
/// * `num_heads`, `num_kv_heads`, `head_dim` - Attention configuration
///
/// Returns `[b, seq, num_heads * head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    seqlen_offset: usize,
    cache_engine: &mut CacheEngine,
    block_ids: &[BlockId],
    slot_mapping: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let backend = select_backend();
    let metadata = PagedAttentionMetadata {
        block_ids,
        slot_mapping,
        seq_len: seqlen_offset + q.dims()[2],
        seqlen_offset,
    };
    backend.prefill_attention(
        q,
        k,
        v,
        attention_mask,
        cache_engine,
        &metadata,
        num_heads,
        num_kv_heads,
        head_dim,
    )
}
