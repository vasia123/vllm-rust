//! Naive attention backend using standard matrix operations.
//!
//! This is the reference implementation that works on any device.
//! It uses standard scaled dot-product attention without optimization.

use candle_core::{Result, Tensor};

use crate::kv_cache::CacheEngine;

use super::backend::{AttentionBackend, BatchedDecodeMetadata, PagedAttentionMetadata};
use super::ops::repeat_kv;

/// Naive attention backend using standard matmul operations.
///
/// This backend serves as:
/// - Reference implementation for correctness testing
/// - Fallback when FlashAttention is not available
/// - CPU-compatible implementation
pub struct NaiveAttentionBackend;

impl NaiveAttentionBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NaiveAttentionBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionBackend for NaiveAttentionBackend {
    fn name(&self) -> &'static str {
        "naive"
    }

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
    ) -> Result<Tensor> {
        let (b_sz, _num_heads, q_len, _head_dim) = q.dims4()?;

        // Write new K, V to paged cache
        let k_for_cache = k.squeeze(0)?;
        let v_for_cache = v.squeeze(0)?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, metadata.slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        // Read full K, V from cache (all tokens including new)
        let num_tokens = metadata.seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(metadata.block_ids, num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        // GQA: repeat KV heads to match Q heads
        let num_kv_groups = num_heads / num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        // Scaled dot-product attention
        let scale = 1.0 / (head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_full.transpose(2, 3)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_full)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, num_heads * head_dim))
    }

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
    ) -> Result<Tensor> {
        let batch_size = metadata.kv_lengths.len();

        // Write new K/V tokens to cache (one per sequence, batched)
        cache_engine
            .write_batch(k_new, v_new, metadata.all_slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write_batch: {e}")))?;

        // Per-sequence attention computation
        per_seq_decode(
            q,
            cache_engine,
            metadata.seq_block_ids,
            metadata.kv_lengths,
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }
}

/// Per-sequence decode: read KV from cache and compute attention individually.
#[allow(clippy::too_many_arguments)]
fn per_seq_decode(
    q: &Tensor,
    cache_engine: &mut CacheEngine,
    seq_block_ids: &[&[crate::kv_cache::BlockId]],
    kv_lengths: &[usize],
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let num_kv_groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let mut outputs = Vec::with_capacity(batch_size);

    for (i, (&kv_len, &block_ids)) in kv_lengths.iter().zip(seq_block_ids.iter()).enumerate() {
        // q_i: [1, num_heads, head_dim] → [1, num_heads, 1, head_dim]
        let q_i = q.narrow(0, i, 1)?.unsqueeze(2)?;

        // Read from cache: [1, kv_heads, kv_len, head_dim]
        let (k_i, v_i) = cache_engine
            .read(block_ids, kv_len)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        // GQA repeat
        let k_i = repeat_kv(k_i, num_kv_groups)?;
        let v_i = repeat_kv(v_i, num_kv_groups)?;

        // Scaled dot-product attention
        let attn_weights = (q_i.matmul(&k_i.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_out = attn_weights.matmul(&v_i)?;
        // [1, num_heads, 1, head_dim] → [1, num_heads * head_dim]
        let attn_out = attn_out.squeeze(2)?.reshape((1, num_heads * head_dim))?;
        outputs.push(attn_out);
    }

    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_naive_backend_name() {
        let backend = NaiveAttentionBackend::new();
        assert_eq!(backend.name(), "naive");
    }

    #[test]
    fn test_naive_backend_supports_all_dtypes() {
        let backend = NaiveAttentionBackend::new();
        let dtypes = backend.supported_dtypes();
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::F16));
        assert!(dtypes.contains(&DType::BF16));
    }

    #[test]
    fn test_naive_backend_supports_config() {
        let backend = NaiveAttentionBackend::new();
        // Should support any configuration
        assert!(backend.supports_config(32, 8, 64));
        assert!(backend.supports_config(16, 16, 128));
        assert!(backend.supports_config(1, 1, 32));
    }
}
