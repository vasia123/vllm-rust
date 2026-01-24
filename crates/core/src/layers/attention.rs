use candle_core::{Result, Tensor};
use candle_nn::RmsNorm;

use crate::kv_cache::{BlockTable, CacheEngine};

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
    block_table: &BlockTable,
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
        .read(block_table.block_ids(), num_tokens)
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
