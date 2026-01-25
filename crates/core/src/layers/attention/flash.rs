//! FlashAttention-2 backend using candle-flash-attn.
//!
//! This backend provides optimized attention using FlashAttention-2 kernels.
//! Requires the `flash-attn` feature and CUDA support.

use candle_core::{Result, Tensor};

use crate::kv_cache::CacheEngine;

use super::backend::{AttentionBackend, BatchedDecodeMetadata, PagedAttentionMetadata};

#[cfg(feature = "flash-attn")]
use candle_core::DType;

#[cfg(feature = "flash-attn")]
use super::ops::repeat_kv;

/// FlashAttention-2 backend.
///
/// Uses optimized CUDA kernels for attention computation.
/// Falls back to naive implementation for unsupported configurations.
pub struct FlashAttentionBackend;

impl FlashAttentionBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FlashAttentionBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "flash-attn")]
impl AttentionBackend for FlashAttentionBackend {
    fn name(&self) -> &'static str {
        "flash-attention"
    }

    fn prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _attention_mask: Option<&Tensor>,
        cache_engine: &CacheEngine,
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

        // Read full K, V from cache
        let num_tokens = metadata.seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(metadata.block_ids, num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        // GQA: repeat KV heads
        let num_kv_groups = num_heads / num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        // FlashAttention requires F16/BF16
        let orig_dtype = q.dtype();
        let fa_dtype = if orig_dtype == DType::F16 || orig_dtype == DType::BF16 {
            orig_dtype
        } else {
            DType::BF16
        };

        let q = q.to_dtype(fa_dtype)?;
        let k_full = k_full.to_dtype(fa_dtype)?;
        let v_full = v_full.to_dtype(fa_dtype)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        // FlashAttention expects [batch, seq, heads, head_dim]
        let q = q.transpose(1, 2)?;
        let k_full = k_full.transpose(1, 2)?;
        let v_full = v_full.transpose(1, 2)?;

        let output = candle_flash_attn::flash_attn(&q, &k_full, &v_full, softmax_scale, true)?;

        // Back to original dtype and shape
        let output = output.to_dtype(orig_dtype)?;
        output.reshape((b_sz, q_len, num_heads * head_dim))
    }

    fn batched_decode_attention(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let batch_size = metadata.kv_lengths.len();
        let device = q.device();
        let orig_dtype = q.dtype();

        // Write new K/V tokens to cache
        cache_engine
            .write_batch(k_new, v_new, metadata.all_slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write_batch: {e}")))?;

        // Read all cached K/V concatenated for flash_attn_varlen
        let seq_infos: Vec<(&[crate::kv_cache::BlockId], usize)> = metadata
            .seq_block_ids
            .iter()
            .zip(metadata.kv_lengths.iter())
            .map(|(&blocks, &len)| (blocks, len))
            .collect();

        let (k_full, v_full) = cache_engine
            .read_contiguous_multi(&seq_infos)
            .map_err(|e| candle_core::Error::Msg(format!("cache read_contiguous_multi: {e}")))?;

        // FlashAttention requires F16/BF16
        let fa_dtype = if orig_dtype == DType::F16 || orig_dtype == DType::BF16 {
            orig_dtype
        } else {
            DType::BF16
        };

        let q = q.to_dtype(fa_dtype)?;
        let k_full = k_full.to_dtype(fa_dtype)?;
        let v_full = v_full.to_dtype(fa_dtype)?;

        // Build cumulative sequence lengths
        let cu_seqlens_q: Vec<i32> = (0..=batch_size as i32).collect();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (batch_size + 1,), device)?;

        let mut cu_seqlens_k = Vec::with_capacity(batch_size + 1);
        cu_seqlens_k.push(0i32);
        let mut cumsum = 0i32;
        for &len in metadata.kv_lengths {
            cumsum += len as i32;
            cu_seqlens_k.push(cumsum);
        }
        let max_seqlen_k = *metadata.kv_lengths.iter().max().unwrap_or(&1);
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (batch_size + 1,), device)?;

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();

        let output = candle_flash_attn::flash_attn_varlen(
            &q,
            &k_full,
            &v_full,
            &cu_seqlens_q,
            &cu_seqlens_k,
            1,
            max_seqlen_k,
            softmax_scale,
            true,
        )?;

        let output = output.to_dtype(orig_dtype)?;
        output.reshape((batch_size, num_heads * head_dim))
    }

    fn supported_dtypes(&self) -> &[DType] {
        // FlashAttention natively supports F16/BF16, but we can convert from F32
        &[DType::F32, DType::F16, DType::BF16]
    }
}

#[cfg(not(feature = "flash-attn"))]
use super::naive::NaiveAttentionBackend;

/// When flash-attn feature is disabled, FlashAttentionBackend delegates to
/// NaiveAttentionBackend for compatibility.
#[cfg(not(feature = "flash-attn"))]
impl AttentionBackend for FlashAttentionBackend {
    fn name(&self) -> &'static str {
        "flash-attention-fallback-to-naive"
    }

    fn prefill_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_engine: &CacheEngine,
        metadata: &PagedAttentionMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        NaiveAttentionBackend::new().prefill_attention(
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

    fn batched_decode_attention(
        &self,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        cache_engine: &CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        NaiveAttentionBackend::new().batched_decode_attention(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_backend_name() {
        let backend = FlashAttentionBackend::new();
        #[cfg(feature = "flash-attn")]
        assert_eq!(backend.name(), "flash-attention");
        #[cfg(not(feature = "flash-attn"))]
        assert_eq!(backend.name(), "flash-attention-fallback-to-naive");
    }
}
