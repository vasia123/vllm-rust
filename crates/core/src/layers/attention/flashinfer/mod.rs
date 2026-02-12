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
pub use wrapper::{DecodeWrapper, FlashInferConfig, MlaConfig, MlaWrapper, PrefillWrapper};

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
        let mut ws_guard = self
            .workspace
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("workspace lock poisoned: {e}")))?;

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
        let mut ws_guard = self
            .workspace
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("workspace lock poisoned: {e}")))?;
        let ws = ws_guard
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("workspace not initialized".to_string()))?;

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
        let mut ws_guard = self
            .workspace
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("workspace lock poisoned: {e}")))?;
        let ws = ws_guard
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("workspace not initialized".to_string()))?;

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

        // After repeat_kv, k_full: [1, num_heads, kv_len, head_dim] — already correct
        // (cache_engine.read returns [1, kv_heads, kv_len, head_dim])

        // Attention scores: [b, heads, q_len, kv_len]
        let attn_weights = q.matmul(&k_full.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * scale)?;

        // Causal mask (cast to match attn_weights dtype for F16)
        let kv_len = k_full.dim(2)?;
        let causal_mask = create_causal_mask(q_len, kv_len, metadata.seqlen_offset, device)?;
        let causal_mask = causal_mask.to_dtype(attn_weights.dtype())?;
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
            // k after repeat_kv: [1, num_heads, kv_len, head_dim] — already correct
            let k_seq = k_cached;
            let v_seq = v_cached;

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

/// GPU integration tests for FlashInfer kernels.
///
/// These tests run actual CUDA kernels and compare FlashInfer output
/// against the naive reference implementation for numerical correctness.
#[cfg(all(test, feature = "flashinfer", feature = "gpu-test-small"))]
mod gpu_tests {
    use super::*;
    use crate::kv_cache::{CacheConfig, KVCacheDtype};
    use candle_core::DType;

    const BLOCK_SIZE: usize = 16;
    const NUM_KV_HEADS: usize = 4;
    const NUM_HEADS: usize = 8; // GQA ratio 2:1
    const HEAD_DIM: usize = 128; // FlashInfer common head dim

    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required")
    }

    #[test]
    fn test_gpu_support_check() {
        let (supported, sm) = flashinfer_rs::ffi::check_gpu_support().unwrap();
        assert!(supported, "GPU not supported for FlashInfer");
        assert!(sm >= 80, "Need SM80+, got SM{sm}");
        eprintln!("FlashInfer GPU support: SM{sm}");
    }

    #[test]
    fn test_workspace_on_gpu() {
        let device = cuda_device();
        let ws = WorkspaceBuffer::with_size(16 * 1024 * 1024, &device).unwrap();
        let ptr = ws.as_ptr().unwrap();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_tensor_bridge_gpu() {
        let device = cuda_device();
        let t = Tensor::zeros((4, 8, 128), DType::F16, &device).unwrap();
        let ptr = tensor_bridge::tensor_to_device_ptr(&t).unwrap();
        assert!(!ptr.is_null());
    }

    fn test_cache_config(device: &Device, num_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: BLOCK_SIZE,
            num_blocks,
            num_layers: 1,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            dtype: DType::F16,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    /// Generate deterministic random-looking F16 tensors on CUDA.
    fn rand_f16(shape: &[usize], device: &Device, seed: u64) -> Tensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n: usize = shape.iter().product();
        let data: Vec<half::f16> = (0..n)
            .map(|_| half::f16::from_f32(rng.gen_range(-0.5..0.5)))
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap()
    }

    /// Compare two tensors element-wise with tolerance.
    fn assert_tensors_close(a: &Tensor, b: &Tensor, atol: f32, name: &str) {
        let a_f32: Vec<f32> = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b_f32: Vec<f32> = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert_eq!(a_f32.len(), b_f32.len(), "{name}: length mismatch");

        let mut max_diff: f32 = 0.0;
        let mut max_idx = 0;
        for (i, (x, y)) in a_f32.iter().zip(b_f32.iter()).enumerate() {
            assert!(x.is_finite(), "{name}[{i}]: FlashInfer output is {x}");
            assert!(y.is_finite(), "{name}[{i}]: naive output is {y}");
            let diff = (x - y).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }

        assert!(
            max_diff <= atol,
            "{name}: max diff {max_diff} at index {max_idx} exceeds tolerance {atol}\n  \
             flashinfer[{max_idx}] = {}, naive[{max_idx}] = {}",
            a_f32[max_idx],
            b_f32[max_idx],
        );
    }

    /// Direct FFI test: call BatchDecodePlan with host kv_indptr + page-locked buffer.
    #[test]
    fn test_ffi_decode_plan_direct_gpu() {
        let device = cuda_device();

        let ws_size: usize = 128 * 1024 * 1024;
        let ws_tensor = Tensor::zeros(ws_size, DType::U8, &device).unwrap();
        let ws_ptr = tensor_bridge::tensor_to_device_ptr(&ws_tensor).unwrap();
        let float_ws = ws_ptr as *mut std::ffi::c_void;
        let int_ws_size = ws_size / 2;
        let int_ws = unsafe { (ws_ptr as *const u8).add(int_ws_size) } as *mut std::ffi::c_void;

        let stream = tensor_bridge::get_cuda_stream_ptr(&device).unwrap();
        let kv_indptr_host: Vec<i32> = vec![0, 1];
        let mut page_locked_buf: Vec<u8> = vec![0u8; int_ws_size];

        let plan = unsafe {
            flashinfer_rs::ffi::BatchDecodePlan::new(
                float_ws,
                int_ws_size,
                int_ws,
                int_ws_size,
                page_locked_buf.as_mut_ptr() as *mut std::ffi::c_void,
                int_ws_size,
                kv_indptr_host.as_ptr(),
                1,
                8,
                4,
                128,
                16,
                flashinfer_rs::ffi::DType::Float16,
                flashinfer_rs::ffi::PosEncoding::None,
                0.0,
                -1,
                false,
                stream,
            )
        }
        .unwrap();

        let batch_size = 1;
        let num_qo_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 128;
        let num_pages = 4;
        let block_size = 16;

        let q = Tensor::zeros((batch_size, num_qo_heads, head_dim), DType::F16, &device).unwrap();
        let k_cache = Tensor::zeros(
            (num_pages, block_size, num_kv_heads, head_dim),
            DType::F16,
            &device,
        )
        .unwrap();
        let v_cache = Tensor::zeros(
            (num_pages, block_size, num_kv_heads, head_dim),
            DType::F16,
            &device,
        )
        .unwrap();
        let output =
            Tensor::zeros((batch_size, num_qo_heads, head_dim), DType::F16, &device).unwrap();

        let q_ptr = tensor_bridge::tensor_to_device_ptr(&q).unwrap();
        let k_ptr = tensor_bridge::tensor_to_device_ptr(&k_cache).unwrap();
        let v_ptr = tensor_bridge::tensor_to_device_ptr(&v_cache).unwrap();
        let out_ptr =
            tensor_bridge::tensor_to_device_ptr(&output).unwrap() as *mut std::ffi::c_void;

        let kv_indptr_gpu = tensor_bridge::alloc_gpu_i32(&[0i32, 1i32], &device).unwrap();
        let kv_indices_gpu = tensor_bridge::alloc_gpu_i32(&[0i32], &device).unwrap();
        let kv_last_page_gpu = tensor_bridge::alloc_gpu_i32(&[1i32], &device).unwrap();
        let indptr_d = tensor_bridge::tensor_to_device_ptr(&kv_indptr_gpu).unwrap() as *const i32;
        let indices_d = tensor_bridge::tensor_to_device_ptr(&kv_indices_gpu).unwrap() as *const i32;
        let last_page_d =
            tensor_bridge::tensor_to_device_ptr(&kv_last_page_gpu).unwrap() as *const i32;

        unsafe {
            plan.run(
                q_ptr,
                k_ptr,
                v_ptr,
                indptr_d,
                indices_d,
                last_page_d,
                out_ptr,
                std::ptr::null_mut(),
                flashinfer_rs::ffi::KVLayout::NHD,
                stream,
            )
        }
        .unwrap();
    }

    /// Minimal test: create plan + run on a tiny batch directly via wrapper.
    #[test]
    fn test_decode_wrapper_direct_gpu() {
        let device = cuda_device();

        let block_size: usize = 16;
        let num_qo_heads: usize = 8;
        let num_kv_heads: usize = 4;
        let head_dim: usize = 128;
        let batch_size: usize = 1;
        let num_pages = 4;

        let k_cache = Tensor::zeros(
            (num_pages, block_size, num_kv_heads, head_dim),
            DType::F16,
            &device,
        )
        .unwrap();
        let v_cache = Tensor::zeros(
            (num_pages, block_size, num_kv_heads, head_dim),
            DType::F16,
            &device,
        )
        .unwrap();
        let q = Tensor::zeros((batch_size, num_qo_heads, head_dim), DType::F16, &device).unwrap();
        let mut ws = WorkspaceBuffer::with_size(128 * 1024 * 1024, &device).unwrap();

        let kv_indptr = tensor_bridge::alloc_gpu_i32(&[0i32, 1i32], &device).unwrap();
        let kv_indices = tensor_bridge::alloc_gpu_i32(&[0i32], &device).unwrap();
        let kv_last_page_len = tensor_bridge::alloc_gpu_i32(&[1i32], &device).unwrap();

        let config = FlashInferConfig::new(
            num_qo_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            block_size as u32,
        );
        let wrapper = DecodeWrapper::new(config, &device).unwrap();

        let output = wrapper
            .run(
                &q,
                &k_cache,
                &v_cache,
                &mut ws,
                &kv_indptr,
                &kv_indices,
                &kv_last_page_len,
                batch_size,
            )
            .unwrap();

        assert_eq!(output.dims(), &[batch_size, num_qo_heads, head_dim]);
    }

    #[test]
    fn test_decode_single_sequence_gpu() {
        let device = cuda_device();
        let num_blocks = 4;
        let config = test_cache_config(&device, num_blocks);
        let mut cache = CacheEngine::new(&config).unwrap();

        // Pre-fill cache with 20 tokens (spans 2 blocks of 16)
        let prefill_len = 20;
        let k_prefill = rand_f16(&[NUM_KV_HEADS, prefill_len, HEAD_DIM], &device, 42);
        let v_prefill = rand_f16(&[NUM_KV_HEADS, prefill_len, HEAD_DIM], &device, 43);
        let slot_mapping: Vec<usize> = (0..prefill_len).collect();
        cache.write(&k_prefill, &v_prefill, &slot_mapping).unwrap();

        // Decode: 1 new token at position 20
        let decode_slot = prefill_len; // slot 20
        let q = rand_f16(&[1, NUM_HEADS, HEAD_DIM], &device, 44);
        let k_new = rand_f16(&[1, NUM_KV_HEADS, HEAD_DIM], &device, 45);
        let v_new = rand_f16(&[1, NUM_KV_HEADS, HEAD_DIM], &device, 46);

        let block_ids: Vec<usize> = vec![0, 1]; // blocks holding 20+1 tokens
        let kv_len = prefill_len + 1;

        let metadata = BatchedDecodeMetadata {
            seq_block_ids: &[&block_ids],
            all_slot_mapping: &[decode_slot],
            kv_lengths: &[kv_len],
        };

        // Run FlashInfer decode
        let backend = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output = backend
            .batched_decode_attention(
                &q,
                &k_new,
                &v_new,
                &mut cache,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        // Verify shape [1, NUM_HEADS * HEAD_DIM]
        assert_eq!(output.dims(), &[1, NUM_HEADS * HEAD_DIM]);

        // Verify output is finite (not NaN/Inf)
        let output_f32: Vec<f32> = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &v) in output_f32.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
        // Verify output is not all zeros
        let sum: f32 = output_f32.iter().map(|x| x.abs()).sum();
        assert!(sum > 1e-6, "output is all zeros");
    }

    #[test]
    fn test_decode_vs_naive_gpu() {
        let device = cuda_device();
        let num_blocks = 4;

        // Create two identical caches
        let config = test_cache_config(&device, num_blocks);
        let mut cache_fi = CacheEngine::new(&config).unwrap();
        let mut cache_naive = CacheEngine::new(&config).unwrap();

        // Pre-fill with same data (20 tokens)
        let prefill_len = 20;
        let k_prefill = rand_f16(&[NUM_KV_HEADS, prefill_len, HEAD_DIM], &device, 100);
        let v_prefill = rand_f16(&[NUM_KV_HEADS, prefill_len, HEAD_DIM], &device, 101);
        let slot_mapping: Vec<usize> = (0..prefill_len).collect();
        cache_fi
            .write(&k_prefill, &v_prefill, &slot_mapping)
            .unwrap();
        cache_naive
            .write(&k_prefill, &v_prefill, &slot_mapping)
            .unwrap();

        // Decode token
        let decode_slot = prefill_len;
        let q = rand_f16(&[1, NUM_HEADS, HEAD_DIM], &device, 102);
        let k_new = rand_f16(&[1, NUM_KV_HEADS, HEAD_DIM], &device, 103);
        let v_new = rand_f16(&[1, NUM_KV_HEADS, HEAD_DIM], &device, 104);

        let block_ids: Vec<usize> = vec![0, 1];
        let kv_len = prefill_len + 1;

        let metadata = BatchedDecodeMetadata {
            seq_block_ids: &[&block_ids],
            all_slot_mapping: &[decode_slot],
            kv_lengths: &[kv_len],
        };

        // FlashInfer path
        let backend_fi = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output_fi = backend_fi
            .batched_decode_attention(
                &q,
                &k_new,
                &v_new,
                &mut cache_fi,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        // Naive path (on same GPU data)
        let backend_naive = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output_naive = backend_naive
            .batched_decode_naive(
                &q,
                &k_new,
                &v_new,
                &mut cache_naive,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        // F16 tolerance: 1e-2 (half precision has ~3 significant digits)
        assert_tensors_close(&output_fi, &output_naive, 0.02, "decode_vs_naive");
    }

    #[test]
    fn test_prefill_vs_naive_gpu() {
        let device = cuda_device();
        let num_blocks = 4;

        let config = test_cache_config(&device, num_blocks);
        let mut cache_fi = CacheEngine::new(&config).unwrap();
        let mut cache_naive = CacheEngine::new(&config).unwrap();

        // Prefill 8 tokens (fits in 1 block)
        let seq_len = 8;
        let q = rand_f16(&[1, NUM_HEADS, seq_len, HEAD_DIM], &device, 200);
        let k = rand_f16(&[1, NUM_KV_HEADS, seq_len, HEAD_DIM], &device, 201);
        let v = rand_f16(&[1, NUM_KV_HEADS, seq_len, HEAD_DIM], &device, 202);

        let block_ids: Vec<usize> = vec![0];
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let metadata = PagedAttentionMetadata {
            block_ids: &block_ids,
            slot_mapping: &slot_mapping,
            seq_len,
            seqlen_offset: 0,
        };

        // FlashInfer path
        let backend_fi = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output_fi = backend_fi
            .prefill_attention(
                &q,
                &k,
                &v,
                None,
                &mut cache_fi,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        // Naive path
        let backend_naive = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output_naive = backend_naive
            .prefill_naive(
                &q,
                &k,
                &v,
                None,
                &mut cache_naive,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        // Output shape: [1, seq_len, NUM_HEADS * HEAD_DIM]
        assert_eq!(output_fi.dims(), &[1, seq_len, NUM_HEADS * HEAD_DIM]);
        assert_eq!(output_naive.dims(), &[1, seq_len, NUM_HEADS * HEAD_DIM]);

        assert_tensors_close(&output_fi, &output_naive, 0.02, "prefill_vs_naive");
    }

    #[test]
    fn test_batched_decode_multi_sequence_gpu() {
        let device = cuda_device();
        let num_blocks = 8;

        let config = test_cache_config(&device, num_blocks);
        let mut cache = CacheEngine::new(&config).unwrap();

        // Seq 0: 10 tokens in block 0 (slots 0..10)
        let k0 = rand_f16(&[NUM_KV_HEADS, 10, HEAD_DIM], &device, 300);
        let v0 = rand_f16(&[NUM_KV_HEADS, 10, HEAD_DIM], &device, 301);
        let slots0: Vec<usize> = (0..10).collect();
        cache.write(&k0, &v0, &slots0).unwrap();

        // Seq 1: 25 tokens in blocks 2,3 (slots 32..57)
        let k1 = rand_f16(&[NUM_KV_HEADS, 25, HEAD_DIM], &device, 302);
        let v1 = rand_f16(&[NUM_KV_HEADS, 25, HEAD_DIM], &device, 303);
        let slots1: Vec<usize> = (32..57).collect();
        cache.write(&k1, &v1, &slots1).unwrap();

        // Decode: 2 new tokens (1 per sequence)
        let batch_size = 2;
        let q = rand_f16(&[batch_size, NUM_HEADS, HEAD_DIM], &device, 304);
        let k_new = rand_f16(&[batch_size, NUM_KV_HEADS, HEAD_DIM], &device, 305);
        let v_new = rand_f16(&[batch_size, NUM_KV_HEADS, HEAD_DIM], &device, 306);

        // Seq 0 decode at slot 10 (still block 0), seq 1 at slot 57 (block 3)
        let block_ids_0: Vec<usize> = vec![0];
        let block_ids_1: Vec<usize> = vec![2, 3];

        let metadata = BatchedDecodeMetadata {
            seq_block_ids: &[&block_ids_0, &block_ids_1],
            all_slot_mapping: &[10, 57],
            kv_lengths: &[11, 26],
        };

        let backend = FlashInferBackend::with_block_size(BLOCK_SIZE);
        let output = backend
            .batched_decode_attention(
                &q,
                &k_new,
                &v_new,
                &mut cache,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        assert_eq!(output.dims(), &[batch_size, NUM_HEADS * HEAD_DIM]);

        let output_f32: Vec<f32> = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &v) in output_f32.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_prefill_long_sequence_gpu() {
        let device = cuda_device();
        let num_blocks = 8;

        let config = test_cache_config(&device, num_blocks);
        let mut cache_fi = CacheEngine::new(&config).unwrap();
        let mut cache_naive = CacheEngine::new(&config).unwrap();

        // Longer prefill: 48 tokens (3 blocks)
        let seq_len = 48;
        let q = rand_f16(&[1, NUM_HEADS, seq_len, HEAD_DIM], &device, 400);
        let k = rand_f16(&[1, NUM_KV_HEADS, seq_len, HEAD_DIM], &device, 401);
        let v = rand_f16(&[1, NUM_KV_HEADS, seq_len, HEAD_DIM], &device, 402);

        let block_ids: Vec<usize> = vec![0, 1, 2];
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let metadata = PagedAttentionMetadata {
            block_ids: &block_ids,
            slot_mapping: &slot_mapping,
            seq_len,
            seqlen_offset: 0,
        };

        let backend = FlashInferBackend::with_block_size(BLOCK_SIZE);

        let output_fi = backend
            .prefill_attention(
                &q,
                &k,
                &v,
                None,
                &mut cache_fi,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        let output_naive = backend
            .prefill_naive(
                &q,
                &k,
                &v,
                None,
                &mut cache_naive,
                &metadata,
                NUM_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
            )
            .unwrap();

        assert_eq!(output_fi.dims(), &[1, seq_len, NUM_HEADS * HEAD_DIM]);
        // Longer sequences accumulate more F16 numerical error in softmax/matmul
        assert_tensors_close(&output_fi, &output_naive, 0.2, "prefill_long");
    }

    // =========================================================================
    // MLA (Multi-head Latent Attention) GPU tests
    // =========================================================================

    /// Test MlaWrapper creation and basic forward pass.
    ///
    /// Verifies the complete MLA pipeline: plan → run → output with
    /// DeepSeek-like configuration (128 heads, ckv=512, kpe=64).
    #[test]
    fn test_mla_wrapper_forward_gpu() {
        let device = cuda_device();

        // DeepSeek MLA config
        let page_size: u32 = 16;
        let config = MlaConfig::deepseek(page_size);
        let wrapper = MlaWrapper::new(config.clone(), &device).unwrap();
        let mut workspace = WorkspaceBuffer::new(&device).unwrap();

        let batch_size = 1;
        let kv_len: usize = 10; // tokens in KV cache
        let num_pages = (kv_len + page_size as usize - 1) / page_size as usize; // 1 page

        // Query tensors: single decode token
        let q_nope = rand_f16(&[1, 128, 512], &device, 42);
        let q_pe = rand_f16(&[1, 128, 64], &device, 43);

        // Paged caches: [num_pages, page_size, dim]
        let ckv_cache = rand_f16(&[num_pages, page_size as usize, 512], &device, 44);
        let kpe_cache = rand_f16(&[num_pages, page_size as usize, 64], &device, 45);

        // Metadata: single-sequence decode
        let qo_indptr = tensor_bridge::alloc_gpu_i32(&[0, 1], &device).unwrap();
        let kv_indptr = tensor_bridge::alloc_gpu_i32(&[0, num_pages as i32], &device).unwrap();
        let kv_indices_vec: Vec<i32> = (0..num_pages as i32).collect();
        let kv_indices = tensor_bridge::alloc_gpu_i32(&kv_indices_vec, &device).unwrap();
        let kv_lengths = vec![kv_len];

        let output = wrapper
            .run(
                &q_nope,
                &q_pe,
                &ckv_cache,
                &kpe_cache,
                &mut workspace,
                &qo_indptr,
                &kv_indptr,
                &kv_indices,
                &kv_lengths,
                batch_size,
                1,
            )
            .unwrap();

        // Verify output shape: [1, 128, 512]
        assert_eq!(output.dims(), &[1, 128, 512]);
        assert_eq!(output.dtype(), DType::F16);

        // Verify all values are finite
        let data: Vec<f32> = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "MLA output[{i}] = {v} is not finite");
        }
    }

    /// Test MLA with batched sequences (2 sequences, different KV lengths).
    #[test]
    fn test_mla_batched_forward_gpu() {
        let device = cuda_device();

        let page_size: u32 = 16;
        let config = MlaConfig::deepseek(page_size);
        let wrapper = MlaWrapper::new(config, &device).unwrap();
        let mut workspace = WorkspaceBuffer::new(&device).unwrap();

        let batch_size = 2;
        let kv_lens = [10_usize, 20];
        let pages_per_seq: Vec<usize> = kv_lens
            .iter()
            .map(|&l| (l + page_size as usize - 1) / page_size as usize)
            .collect();
        let total_pages: usize = pages_per_seq.iter().sum();

        // 2 decode tokens (one per sequence)
        let q_nope = rand_f16(&[2, 128, 512], &device, 50);
        let q_pe = rand_f16(&[2, 128, 64], &device, 51);

        let ckv_cache = rand_f16(&[total_pages, page_size as usize, 512], &device, 52);
        let kpe_cache = rand_f16(&[total_pages, page_size as usize, 64], &device, 53);

        let qo_indptr = tensor_bridge::alloc_gpu_i32(&[0, 1, 2], &device).unwrap();
        let kv_indptr = tensor_bridge::alloc_gpu_i32(
            &[0, pages_per_seq[0] as i32, total_pages as i32],
            &device,
        )
        .unwrap();
        let kv_indices_vec: Vec<i32> = (0..total_pages as i32).collect();
        let kv_indices = tensor_bridge::alloc_gpu_i32(&kv_indices_vec, &device).unwrap();
        let kv_lengths: Vec<usize> = kv_lens.to_vec();

        let output = wrapper
            .run(
                &q_nope,
                &q_pe,
                &ckv_cache,
                &kpe_cache,
                &mut workspace,
                &qo_indptr,
                &kv_indptr,
                &kv_indices,
                &kv_lengths,
                batch_size,
                2,
            )
            .unwrap();

        assert_eq!(output.dims(), &[2, 128, 512]);
        assert_eq!(output.dtype(), DType::F16);

        let data: Vec<f32> = output
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "MLA batched output[{i}] = {v} is not finite");
        }
    }
}
