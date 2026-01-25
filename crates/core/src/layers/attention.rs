use candle_core::{Result, Tensor};
use candle_nn::RmsNorm;

use crate::kv_cache::{BlockId, CacheEngine};

/// Batched paged attention for decode: processes all sequences in one fused call.
///
/// q, k_new, v_new: [batch_size, num_kv_or_q_heads, head_dim] — after RoPE.
/// Writes new K/V to cache, reads full history, computes attention.
/// Returns [batch_size, num_heads * head_dim].
#[allow(clippy::too_many_arguments)]
pub fn batched_paged_attention_decode(
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    cache_engine: &CacheEngine,
    seq_block_ids: &[&[BlockId]],
    all_slot_mapping: &[usize],
    kv_lengths: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let batch_size = kv_lengths.len();

    // Write new K/V tokens to cache (one per sequence, batched)
    cache_engine
        .write_batch(k_new, v_new, all_slot_mapping)
        .map_err(|e| candle_core::Error::Msg(format!("cache write_batch: {e}")))?;

    #[cfg(feature = "flash-attn")]
    {
        // Read all cached K/V concatenated for flash_attn_varlen
        let seq_infos: Vec<(&[BlockId], usize)> = seq_block_ids
            .iter()
            .zip(kv_lengths.iter())
            .map(|(&blocks, &len)| (blocks, len))
            .collect();
        let (k_full, v_full) = cache_engine
            .read_contiguous_multi(&seq_infos)
            .map_err(|e| candle_core::Error::Msg(format!("cache read_contiguous_multi: {e}")))?;
        flash_decode(
            q, &k_full, &v_full, kv_lengths, batch_size, num_heads, head_dim,
        )
    }

    #[cfg(not(feature = "flash-attn"))]
    {
        per_seq_decode(
            q,
            cache_engine,
            seq_block_ids,
            kv_lengths,
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }
}

/// FlashAttention-2 varlen decode path (requires flash-attn feature).
#[cfg(feature = "flash-attn")]
fn flash_decode(
    q: &Tensor,
    k_full: &Tensor,
    v_full: &Tensor,
    kv_lengths: &[usize],
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    use candle_core::DType;

    let device = q.device();
    let orig_dtype = q.dtype();

    let fa_dtype = if orig_dtype == DType::F16 || orig_dtype == DType::BF16 {
        orig_dtype
    } else {
        DType::BF16
    };

    let q = q.to_dtype(fa_dtype)?;
    let k_full = k_full.to_dtype(fa_dtype)?;
    let v_full = v_full.to_dtype(fa_dtype)?;

    let cu_seqlens_q: Vec<i32> = (0..=batch_size as i32).collect();
    let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (batch_size + 1,), device)?;

    let mut cu_seqlens_k = Vec::with_capacity(batch_size + 1);
    cu_seqlens_k.push(0i32);
    let mut cumsum = 0i32;
    for &len in kv_lengths {
        cumsum += len as i32;
        cu_seqlens_k.push(cumsum);
    }
    let max_seqlen_k = *kv_lengths.iter().max().unwrap_or(&1);
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

/// Per-sequence decode: read KV from cache and compute attention individually.
/// Batched RoPE and cache write are already done; this handles read + attention.
#[cfg(not(feature = "flash-attn"))]
#[allow(clippy::too_many_arguments)]
fn per_seq_decode(
    q: &Tensor,
    cache_engine: &CacheEngine,
    seq_block_ids: &[&[BlockId]],
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

/// Shared paged attention: write to cache, read full history, compute GQA attention.
///
/// Expects Q, K, V already projected and reshaped to `[b, heads, seq, head_dim]`.
/// K and V should have `num_kv_heads` heads; Q should have `num_heads` heads.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    seqlen_offset: usize,
    cache_engine: &CacheEngine,
    block_ids: &[BlockId],
    slot_mapping: &[usize],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (b_sz, _num_heads, q_len, _head_dim) = q.dims4()?;

    // Write new K, V to paged cache
    let k_for_cache = k.squeeze(0)?;
    let v_for_cache = v.squeeze(0)?;
    cache_engine
        .write(&k_for_cache, &v_for_cache, slot_mapping)
        .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

    // Read full K, V from cache (all tokens including new)
    let num_tokens = seqlen_offset + q_len;
    let (k_full, v_full) = cache_engine
        .read(block_ids, num_tokens)
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

/// Repeat KV heads for Grouped Query Attention.
pub fn repeat_kv(x: Tensor, num_kv_groups: usize) -> Result<Tensor> {
    if num_kv_groups == 1 {
        return Ok(x);
    }
    let (b, num_kv_heads, s, d) = x.dims4()?;
    let num_heads = num_kv_heads * num_kv_groups;
    x.unsqueeze(2)?
        .expand((b, num_kv_heads, num_kv_groups, s, d))?
        .reshape((b, num_heads, s, d))
}

/// Apply RMSNorm per attention head (used by Qwen3).
/// Reshapes [b, h, s, d] → [b*h*s, d], applies norm, reshapes back.
pub fn apply_per_head_norm(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let x = x.reshape((b * h * s, d))?;
    let x = candle_nn::Module::forward(norm, &x)?;
    x.reshape((b, h, s, d))
}
