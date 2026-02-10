//! FlashInfer attention backend.
//!
//! This backend uses FlashInfer kernels for high-performance attention
//! computation with paged KV cache support. FlashInfer provides 2-3x speedup
//! over naive attention for decode operations.
//!
//! On CUDA devices with the `flashinfer` feature enabled, calls into
//! FlashInfer's FFI layer via `wrapper.rs`. On CPU or without the feature,
//! falls back to naive matmul-based attention.
//!
//! # Module Structure
//! - `metadata`: Paged KV cache metadata construction
//! - `workspace`: Workspace buffer management
//! - `wrapper`: Direct FFI wrappers for BatchPrefillPlan/BatchDecodePlan
//! - `tensor_bridge`: Candle tensor ↔ raw CUDA pointer conversion

pub mod metadata;
pub mod tensor_bridge;
pub mod workspace;
pub mod wrapper;

pub use metadata::FlashInferMetadata;
pub use workspace::{WorkspaceBuffer, DEFAULT_WORKSPACE_SIZE};
pub use wrapper::{DecodeWrapper, FlashInferConfig, PrefillWrapper};

#[cfg(feature = "flashinfer")]
use candle_core::DType;
use candle_core::{Device, IndexOp, Result, Tensor};

use crate::kv_cache::CacheEngine;

use super::backend::{AttentionBackend, BatchedDecodeMetadata, PagedAttentionMetadata};

/// FlashInfer attention backend.
///
/// Uses FlashInfer CUDA kernels for optimized attention computation.
/// Supports paged KV cache, GQA, sliding window, and soft-capping.
///
/// Falls back to naive attention on CPU or when FlashInfer is unavailable.
pub struct FlashInferBackend {
    /// Workspace buffer for FlashInfer operations (lazily initialized on first CUDA call)
    #[cfg(feature = "flashinfer")]
    workspace: std::sync::Mutex<Option<WorkspaceBuffer>>,

    /// Block size for paged KV cache
    #[cfg_attr(not(feature = "flashinfer"), allow(dead_code))]
    block_size: usize,
}

impl FlashInferBackend {
    /// Create a new FlashInfer backend with default configuration.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "flashinfer")]
            workspace: std::sync::Mutex::new(None),
            block_size: 16, // Default block size
        }
    }

    /// Create a new FlashInfer backend with specified block size.
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            #[cfg(feature = "flashinfer")]
            workspace: std::sync::Mutex::new(None),
            block_size,
        }
    }

    /// Get or create workspace buffer for the given device.
    ///
    /// Uses interior mutability via Mutex since AttentionBackend trait
    /// methods take `&self`.
    #[cfg(feature = "flashinfer")]
    fn get_or_create_workspace(&self, device: &Device) -> Result<()> {
        let mut ws_guard = self.workspace.lock().map_err(|e| {
            candle_core::Error::Msg(format!("workspace lock poisoned: {e}"))
        })?;

        if let Some(ref ws) = *ws_guard {
            if ws.device().same_device(device) {
                return Ok(());
            }
        }

        *ws_guard = Some(WorkspaceBuffer::new(device)?);
        Ok(())
    }

    /// Run prefill attention using FlashInfer kernels.
    ///
    /// Assumes K/V have already been written to cache before this call.
    #[cfg(feature = "flashinfer")]
    #[allow(clippy::too_many_arguments)]
    fn prefill_flashinfer(
        &self,
        q: &Tensor,
        cache_engine: &CacheEngine,
        metadata: &PagedAttentionMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let device = q.device();
        let (b_sz, _num_heads, q_len, _head_dim) = q.dims4()?;
        let total_tokens = b_sz * q_len;

        // Reshape Q from [b, heads, q_len, head_dim] to [total_tokens, heads, head_dim]
        // FlashInfer expects flattened token dimension
        let q_flat = q
            .transpose(1, 2)? // [b, q_len, heads, head_dim]
            .reshape((total_tokens, num_heads, head_dim))?;

        // Build FlashInfer paged KV metadata
        let fi_metadata = FlashInferMetadata::from_single_sequence(
            metadata.block_ids,
            metadata.seq_len,
            self.block_size,
            device,
        )?;

        // Convert metadata tensors from i64/u32 to i32 for FFI
        let kv_indptr = fi_metadata.paged_kv_indptr.to_dtype(DType::U32)?;
        let kv_indices = fi_metadata.paged_kv_indices.to_dtype(DType::U32)?;
        let kv_last_page_len = fi_metadata.paged_kv_last_page_len.to_dtype(DType::U32)?;

        // Build qo_indptr: cumulative query token counts per sequence
        // For single-sequence prefill: [0, total_tokens]
        let qo_indptr_data: Vec<i32> = vec![0, total_tokens as i32];
        let qo_indptr = tensor_bridge::alloc_gpu_i32(&qo_indptr_data, device)?;

        // Create wrapper and run (pass cache layout for correct KV access pattern)
        let config = FlashInferConfig::new(
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            self.block_size as u32,
        )
        .with_kv_layout(cache_engine.layout());
        let wrapper = PrefillWrapper::new(config, device)?;

        self.get_or_create_workspace(device)?;
        let mut ws_guard = self.workspace.lock().map_err(|e| {
            candle_core::Error::Msg(format!("workspace lock poisoned: {e}"))
        })?;
        let ws = ws_guard.as_mut().ok_or_else(|| {
            candle_core::Error::Msg("workspace not initialized".to_string())
        })?;

        let output = wrapper.run(
            &q_flat,
            cache_engine.k_cache(),
            cache_engine.v_cache(),
            ws,
            &qo_indptr,
            &kv_indptr,
            &kv_indices,
            &kv_last_page_len,
            1, // batch_size = 1 for single-sequence prefill
            total_tokens,
        )?;

        // Reshape output from [total_tokens, num_heads, head_dim]
        // to [b, q_len, num_heads * head_dim]
        output.reshape((b_sz, q_len, num_heads * head_dim))
    }

    /// Run batched decode attention using FlashInfer kernels.
    ///
    /// Assumes K/V have already been written to cache before this call.
    #[cfg(feature = "flashinfer")]
    #[allow(clippy::too_many_arguments)]
    fn decode_flashinfer(
        &self,
        q: &Tensor,
        cache_engine: &CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let device = q.device();
        let batch_size = metadata.kv_lengths.len();

        // q shape: [batch_size, num_heads, head_dim] — already correct for FlashInfer

        // Build FlashInfer paged KV metadata
        let fi_metadata = FlashInferMetadata::from_paged_attention(
            metadata.seq_block_ids,
            metadata.kv_lengths,
            self.block_size,
            device,
        )?;

        // Convert metadata tensors from i64/u32 to i32 for FFI
        let kv_indptr = fi_metadata.paged_kv_indptr.to_dtype(DType::U32)?;
        let kv_indices = fi_metadata.paged_kv_indices.to_dtype(DType::U32)?;
        let kv_last_page_len = fi_metadata.paged_kv_last_page_len.to_dtype(DType::U32)?;

        // Create wrapper and run (pass cache layout for correct KV access pattern)
        let config = FlashInferConfig::new(
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            self.block_size as u32,
        )
        .with_kv_layout(cache_engine.layout());
        let wrapper = DecodeWrapper::new(config, device)?;

        self.get_or_create_workspace(device)?;
        let mut ws_guard = self.workspace.lock().map_err(|e| {
            candle_core::Error::Msg(format!("workspace lock poisoned: {e}"))
        })?;
        let ws = ws_guard.as_mut().ok_or_else(|| {
            candle_core::Error::Msg("workspace not initialized".to_string())
        })?;

        let output = wrapper.run(
            q,
            cache_engine.k_cache(),
            cache_engine.v_cache(),
            ws,
            &kv_indptr,
            &kv_indices,
            &kv_last_page_len,
            batch_size,
        )?;

        // Reshape output from [batch_size, num_heads, head_dim]
        // to [batch_size, num_heads * head_dim]
        output.reshape((batch_size, num_heads * head_dim))
    }

    /// Naive prefill attention implementation (CPU fallback).
    #[allow(clippy::too_many_arguments)]
    fn prefill_naive(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _attention_mask: Option<&Tensor>,
        cache_engine: &mut CacheEngine,
        metadata: &PagedAttentionMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let (b_sz, _num_heads, q_len, _head_dim) = q.dims4()?;
        let device = q.device();

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
        let k_full = super::ops::repeat_kv(k_full, num_kv_groups)?;
        let v_full = super::ops::repeat_kv(v_full, num_kv_groups)?;

        let scale = 1.0 / (head_dim as f64).sqrt();

        // q: [b, heads, q_len, head_dim], k_full: [kv_len, heads, head_dim]
        let k_full = k_full.unsqueeze(0)?; // [1, kv_len, heads, head_dim]
        let k_full = k_full.transpose(1, 2)?; // [1, heads, kv_len, head_dim]
        let v_full = v_full.unsqueeze(0)?;
        let v_full = v_full.transpose(1, 2)?;

        // Attention scores: [b, heads, q_len, kv_len]
        let attn_weights = q.matmul(&k_full.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Causal mask
        let kv_len = k_full.dim(2)?;
        let causal_mask = create_causal_mask(q_len, kv_len, metadata.seqlen_offset, device)?;
        let attn_weights = attn_weights.broadcast_add(&causal_mask)?;

        // Softmax and output
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let output = attn_weights.matmul(&v_full)?;

        // Reshape to [b, q_len, num_heads * head_dim]
        output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, num_heads * head_dim))
    }

    /// Naive batched decode attention implementation (CPU fallback).
    #[allow(clippy::too_many_arguments)]
    fn batched_decode_naive(
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

        // Write new K/V tokens to cache
        cache_engine
            .write_batch(k_new, v_new, metadata.all_slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write_batch: {e}")))?;

        let mut outputs = Vec::with_capacity(batch_size);
        let num_kv_groups = num_heads / num_kv_heads;

        for (seq_idx, (&block_ids, &kv_len)) in metadata
            .seq_block_ids
            .iter()
            .zip(metadata.kv_lengths.iter())
            .enumerate()
        {
            // Read cached KV for this sequence
            let (k_cached, v_cached) = cache_engine
                .read(block_ids, kv_len)
                .map_err(|e| candle_core::Error::Msg(format!("cache read seq {seq_idx}: {e}")))?;

            // GQA expansion
            let k_cached = super::ops::repeat_kv(k_cached, num_kv_groups)?;
            let v_cached = super::ops::repeat_kv(v_cached, num_kv_groups)?;

            // Get query for this sequence: [num_heads, head_dim]
            let q_seq = q.i(seq_idx)?;

            // Reshape for attention
            // q: [num_heads, head_dim] -> [1, num_heads, 1, head_dim]
            let q_seq = q_seq.unsqueeze(0)?.unsqueeze(2)?;
            // k: [kv_len, num_heads, head_dim] -> [1, num_heads, kv_len, head_dim]
            let k_seq = k_cached.unsqueeze(0)?.transpose(1, 2)?;
            let v_seq = v_cached.unsqueeze(0)?.transpose(1, 2)?;

            let scale = 1.0 / (head_dim as f64).sqrt();
            let attn_weights = q_seq.matmul(&k_seq.transpose(2, 3)?)?;
            let attn_weights = (attn_weights * scale)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let output = attn_weights.matmul(&v_seq)?;

            // [1, num_heads, 1, head_dim] -> [num_heads * head_dim]
            let output = output.squeeze(0)?.squeeze(1)?.flatten_all()?;
            outputs.push(output);
        }

        Tensor::stack(&outputs, 0)
    }
}

impl Default for FlashInferBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "flashinfer")]
impl AttentionBackend for FlashInferBackend {
    fn name(&self) -> &'static str {
        "flashinfer"
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
        let device = q.device();

        // CPU fallback — FlashInfer requires CUDA
        if !device.is_cuda() {
            return self.prefill_naive(
                q,
                k,
                v,
                attention_mask,
                cache_engine,
                metadata,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        // Write new K, V to paged cache first
        let k_for_cache = k.squeeze(0)?;
        let v_for_cache = v.squeeze(0)?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, metadata.slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        // Run FlashInfer prefill kernel (reads from paged cache directly)
        self.prefill_flashinfer(q, cache_engine, metadata, num_heads, num_kv_heads, head_dim)
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
        let device = q.device();

        // CPU fallback — FlashInfer requires CUDA
        if !device.is_cuda() {
            return self.batched_decode_naive(
                q,
                k_new,
                v_new,
                cache_engine,
                metadata,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        // Write new K/V tokens to cache first
        cache_engine
            .write_batch(k_new, v_new, metadata.all_slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write_batch: {e}")))?;

        // Run FlashInfer decode kernel (reads from paged cache directly)
        self.decode_flashinfer(q, cache_engine, metadata, num_heads, num_kv_heads, head_dim)
    }

    fn supported_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }
}

/// Fallback implementation when flashinfer feature is disabled.
#[cfg(not(feature = "flashinfer"))]
impl AttentionBackend for FlashInferBackend {
    fn name(&self) -> &'static str {
        "flashinfer-fallback-to-naive"
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
        self.prefill_naive(
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
        cache_engine: &mut CacheEngine,
        metadata: &BatchedDecodeMetadata,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        self.batched_decode_naive(
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

/// Create a causal attention mask (for naive fallback).
fn create_causal_mask(
    q_len: usize,
    kv_len: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; q_len * kv_len];

    for i in 0..q_len {
        let query_pos = offset + i;
        for j in 0..kv_len {
            if j > query_pos {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    Tensor::from_vec(mask_data, (q_len, kv_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flashinfer_backend_name() {
        let backend = FlashInferBackend::new();
        #[cfg(feature = "flashinfer")]
        assert_eq!(backend.name(), "flashinfer");
        #[cfg(not(feature = "flashinfer"))]
        assert_eq!(backend.name(), "flashinfer-fallback-to-naive");
    }

    #[test]
    fn test_flashinfer_with_block_size() {
        let backend = FlashInferBackend::with_block_size(32);
        assert_eq!(backend.block_size, 32);
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(2, 4, 2, &device).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Query pos 2 can attend to positions 0,1,2 but not 3
        assert_eq!(data[0], 0.0); // pos 0
        assert_eq!(data[1], 0.0); // pos 1
        assert_eq!(data[2], 0.0); // pos 2
        assert_eq!(data[3], f32::NEG_INFINITY); // pos 3

        // Query pos 3 can attend to positions 0,1,2,3
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], 0.0);
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
    }
}
