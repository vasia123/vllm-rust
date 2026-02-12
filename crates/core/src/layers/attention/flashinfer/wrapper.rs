//! High-level wrappers for FlashInfer attention operations.
//!
//! Provides PrefillWrapper and DecodeWrapper that call flashinfer-rs
//! FFI functions directly using raw CUDA pointers extracted from Candle tensors.

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "flashinfer")]
use super::tensor_bridge;
use super::workspace::WorkspaceBuffer;
use crate::kv_cache::KVCacheLayout;

/// Configuration for FlashInfer attention operations.
#[derive(Debug, Clone)]
pub struct FlashInferConfig {
    /// Number of query heads
    pub num_qo_heads: u32,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
    /// Page/block size for paged KV cache
    pub page_size: u32,
    /// Use causal mask
    pub causal: bool,
    /// Soft-capping value (0.0 = disabled)
    pub soft_cap: f32,
    /// Sliding window size (0 = disabled)
    pub window_left: i32,
    /// KV cache memory layout (NHD or HND)
    pub kv_layout: KVCacheLayout,
}

impl FlashInferConfig {
    /// Create a new config with standard settings.
    pub fn new(num_qo_heads: u32, num_kv_heads: u32, head_dim: u32, page_size: u32) -> Self {
        Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal: true,
            soft_cap: 0.0,
            window_left: -1,
            kv_layout: KVCacheLayout::NHD,
        }
    }

    /// Enable soft-capping (for Gemma-2, etc.).
    pub fn with_soft_cap(mut self, soft_cap: f32) -> Self {
        self.soft_cap = soft_cap;
        self
    }

    /// Set sliding window size.
    pub fn with_window(mut self, window: i32) -> Self {
        self.window_left = window;
        self
    }

    /// Set KV cache layout.
    pub fn with_kv_layout(mut self, layout: KVCacheLayout) -> Self {
        self.kv_layout = layout;
        self
    }

    /// Convert our KVCacheLayout to the flashinfer-rs FFI KVLayout.
    #[cfg(feature = "flashinfer")]
    fn ffi_kv_layout(&self) -> flashinfer_rs::ffi::KVLayout {
        match self.kv_layout {
            KVCacheLayout::NHD => flashinfer_rs::ffi::KVLayout::NHD,
            KVCacheLayout::HND => flashinfer_rs::ffi::KVLayout::HND,
        }
    }
}

/// Wrapper for FlashInfer batch prefill attention via direct FFI calls.
///
/// Handles prefill attention where we have multiple query tokens
/// attending to paged KV cache. Uses flashinfer-rs FFI layer directly,
/// bypassing the cudarc-dependent handler layer.
#[cfg(feature = "flashinfer")]
pub struct PrefillWrapper {
    config: FlashInferConfig,
    device: Device,
}

#[cfg(feature = "flashinfer")]
impl PrefillWrapper {
    /// Create a new prefill wrapper.
    pub fn new(config: FlashInferConfig, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(candle_core::Error::Msg(
                "PrefillWrapper requires CUDA device".to_string(),
            ));
        }

        Ok(Self {
            config,
            device: device.clone(),
        })
    }

    /// Run prefill attention via FlashInfer FFI.
    ///
    /// # Arguments
    /// * `q` - Query tensor [total_tokens, num_qo_heads, head_dim]
    /// * `k_cache` - Key cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `v_cache` - Value cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `workspace` - Workspace buffer for FlashInfer temporaries
    /// * `qo_indptr` - Cumulative query lengths [batch_size + 1], i32 GPU tensor
    /// * `kv_indptr` - Cumulative block counts [batch_size + 1], i32 GPU tensor
    /// * `kv_indices` - Flattened block IDs [total_blocks], i32 GPU tensor
    /// * `kv_last_page_len` - Valid tokens in last page [batch_size], i32 GPU tensor
    /// * `kv_lengths` - KV lengths (host, for workspace sizing)
    ///
    /// # Returns
    /// Output tensor [total_tokens, num_qo_heads, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        workspace: &mut WorkspaceBuffer,
        qo_indptr: &Tensor,
        kv_indptr: &Tensor,
        kv_indices: &Tensor,
        kv_last_page_len: &Tensor,
        batch_size: usize,
        total_tokens: usize,
    ) -> Result<Tensor> {
        let q = tensor_bridge::ensure_contiguous(q)?;

        // Ensure workspace is large enough
        let required_size = estimate_prefill_workspace(
            batch_size,
            total_tokens,
            self.config.num_qo_heads as usize,
            self.config.head_dim as usize,
        );
        workspace.ensure_size(required_size)?;

        // Get raw pointers from tensors
        let q_ptr = tensor_bridge::tensor_to_device_ptr(&q)?;
        let k_cache_ptr = tensor_bridge::tensor_to_device_ptr(k_cache)?;
        let v_cache_ptr = tensor_bridge::tensor_to_device_ptr(v_cache)?;
        let qo_indptr_ptr = tensor_bridge::tensor_to_device_ptr(qo_indptr)? as *const i32;
        let kv_indptr_ptr = tensor_bridge::tensor_to_device_ptr(kv_indptr)? as *const i32;
        let kv_indices_ptr = tensor_bridge::tensor_to_device_ptr(kv_indices)? as *const i32;
        let kv_last_page_len_ptr =
            tensor_bridge::tensor_to_device_ptr(kv_last_page_len)? as *const i32;
        let workspace_ptr = workspace.as_ptr()?;
        let stream_ptr = tensor_bridge::get_cuda_stream_ptr(&self.device)?;

        // Allocate output
        let output = Tensor::zeros_like(&q)?;
        let output_ptr = tensor_bridge::tensor_to_device_ptr(&output)? as *mut std::ffi::c_void;

        let dtype = tensor_bridge::candle_to_ffi_dtype(q.dtype())?;

        // Split workspace: first half for float, second half for int
        let ws_size = workspace.size();
        let float_ws_size = ws_size / 2;
        let int_ws_size = ws_size - float_ws_size;
        let float_ws_ptr = workspace_ptr as *mut std::ffi::c_void;
        let int_ws_ptr = unsafe { workspace_ptr.add(float_ws_size) } as *mut std::ffi::c_void;

        // PrefillPlan writes plan metadata to page-locked host memory, then
        // copies it to device via cudaMemcpyAsync. Must match int workspace size.
        let mut page_locked_buf: Vec<u8> = vec![0u8; int_ws_size];

        let plan = unsafe {
            flashinfer_rs::ffi::BatchPrefillPlan::new(
                float_ws_ptr,
                float_ws_size,
                int_ws_ptr,
                int_ws_size,
                page_locked_buf.as_mut_ptr() as *mut std::ffi::c_void,
                int_ws_size,
                qo_indptr_ptr,
                kv_indptr_ptr,
                batch_size as i32,
                self.config.num_qo_heads as i32,
                self.config.num_kv_heads as i32,
                self.config.head_dim as i32,
                self.config.page_size as i32,
                dtype,
                flashinfer_rs::ffi::PosEncoding::None,
                self.config.soft_cap,
                self.config.window_left,
                self.config.causal,
                false, // enable_cuda_graph
                stream_ptr,
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer prefill plan failed: {e}")))?
        };

        // Run the kernel
        unsafe {
            plan.run(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                kv_indptr_ptr,
                kv_indices_ptr,
                kv_last_page_len_ptr,
                qo_indptr_ptr,
                output_ptr,
                std::ptr::null_mut(), // lse
                self.config.ffi_kv_layout(),
                stream_ptr,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("FlashInfer prefill forward failed: {e}"))
            })?;
        }

        Ok(output)
    }
}

/// Wrapper for FlashInfer batch decode attention via direct FFI calls.
///
/// Handles decode attention where we have a single query token per sequence
/// attending to the entire cached KV history.
#[cfg(feature = "flashinfer")]
pub struct DecodeWrapper {
    config: FlashInferConfig,
    device: Device,
}

#[cfg(feature = "flashinfer")]
impl DecodeWrapper {
    /// Create a new decode wrapper.
    pub fn new(config: FlashInferConfig, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(candle_core::Error::Msg(
                "DecodeWrapper requires CUDA device".to_string(),
            ));
        }

        Ok(Self {
            config,
            device: device.clone(),
        })
    }

    /// Run batch decode attention via FlashInfer FFI.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, num_qo_heads, head_dim]
    /// * `k_cache` - Key cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `v_cache` - Value cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `workspace` - Workspace buffer for FlashInfer temporaries
    /// * `kv_indptr` - Cumulative block counts [batch_size + 1], i32 GPU tensor
    /// * `kv_indices` - Flattened block IDs [total_blocks], i32 GPU tensor
    /// * `kv_last_page_len` - Valid tokens in last page [batch_size], i32 GPU tensor
    /// * `batch_size` - Number of sequences
    ///
    /// # Returns
    /// Output tensor [batch_size, num_qo_heads, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        workspace: &mut WorkspaceBuffer,
        kv_indptr: &Tensor,
        kv_indices: &Tensor,
        kv_last_page_len: &Tensor,
        batch_size: usize,
    ) -> Result<Tensor> {
        let q = tensor_bridge::ensure_contiguous(q)?;

        // Ensure workspace is large enough
        let required_size = estimate_decode_workspace(
            batch_size,
            self.config.num_qo_heads as usize,
            self.config.head_dim as usize,
        );
        workspace.ensure_size(required_size)?;

        // Get raw DEVICE pointers
        let q_ptr = tensor_bridge::tensor_to_device_ptr(&q)?;
        let k_cache_ptr = tensor_bridge::tensor_to_device_ptr(k_cache)?;
        let v_cache_ptr = tensor_bridge::tensor_to_device_ptr(v_cache)?;
        let kv_indptr_ptr = tensor_bridge::tensor_to_device_ptr(kv_indptr)? as *const i32;
        let kv_indices_ptr = tensor_bridge::tensor_to_device_ptr(kv_indices)? as *const i32;
        let kv_last_page_len_ptr =
            tensor_bridge::tensor_to_device_ptr(kv_last_page_len)? as *const i32;
        let workspace_ptr = workspace.as_ptr()?;
        let stream_ptr = tensor_bridge::get_cuda_stream_ptr(&self.device)?;

        // Copy kv_indptr to HOST for plan creation.
        // FlashInfer's DecodePlan reads indptr on the CPU for work estimation
        // (the C++ parameter is named `indptr_h` — the `_h` suffix = host).
        let kv_indptr_host_u32: Vec<u32> = kv_indptr.to_vec1::<u32>()?;
        let kv_indptr_host: Vec<i32> = kv_indptr_host_u32.iter().map(|&x| x as i32).collect();

        // Allocate output
        let output = Tensor::zeros_like(&q)?;
        let output_ptr = tensor_bridge::tensor_to_device_ptr(&output)? as *mut std::ffi::c_void;

        let dtype = tensor_bridge::candle_to_ffi_dtype(q.dtype())?;

        // Split workspace: first half for float, second half for int
        let ws_size = workspace.size();
        let float_ws_size = ws_size / 2;
        let int_ws_size = ws_size - float_ws_size;
        let float_ws_ptr = workspace_ptr as *mut std::ffi::c_void;
        let int_ws_ptr = unsafe { workspace_ptr.add(float_ws_size) } as *mut std::ffi::c_void;

        // DecodePlan writes plan metadata to page-locked host memory, then
        // copies it to device via cudaMemcpyAsync. Must match int workspace size.
        let mut page_locked_buf: Vec<u8> = vec![0u8; int_ws_size];

        // Create FFI plan (uses HOST kv_indptr for CPU-side work estimation)
        let plan = unsafe {
            flashinfer_rs::ffi::BatchDecodePlan::new(
                float_ws_ptr,
                float_ws_size,
                int_ws_ptr,
                int_ws_size,
                page_locked_buf.as_mut_ptr() as *mut std::ffi::c_void,
                int_ws_size,
                kv_indptr_host.as_ptr(),
                batch_size as i32,
                self.config.num_qo_heads as i32,
                self.config.num_kv_heads as i32,
                self.config.head_dim as i32,
                self.config.page_size as i32,
                dtype,
                flashinfer_rs::ffi::PosEncoding::None,
                self.config.soft_cap,
                self.config.window_left,
                false, // enable_cuda_graph
                stream_ptr,
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer decode plan failed: {e}")))?
        };

        // Run the kernel (uses DEVICE pointers for GPU execution)
        unsafe {
            plan.run(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                kv_indptr_ptr,
                kv_indices_ptr,
                kv_last_page_len_ptr,
                output_ptr,
                std::ptr::null_mut(), // lse
                self.config.ffi_kv_layout(),
                stream_ptr,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("FlashInfer decode forward failed: {e}"))
            })?;
        }

        Ok(output)
    }
}

/// Configuration for FlashInfer MLA attention operations.
///
/// MLA uses compressed KV cache (ckv + kpe) instead of separate K/V.
/// Queries are split into q_nope (absorbed, head_dim_ckv) and q_pe (head_dim_kpe).
#[derive(Debug, Clone)]
pub struct MlaConfig {
    /// Number of attention heads (128 for DeepSeek)
    pub num_heads: u32,
    /// Compressed KV dimension (512 for DeepSeek)
    pub head_dim_ckv: u32,
    /// K position embedding dimension (64 for DeepSeek)
    pub head_dim_kpe: u32,
    /// Page/block size for paged KV cache
    pub page_size: u32,
    /// Use causal mask
    pub causal: bool,
    /// Softmax scale (1/sqrt(192) for DeepSeek; 0.0 = auto)
    pub sm_scale: f32,
}

impl MlaConfig {
    /// Create DeepSeek v2/v3 MLA configuration.
    pub fn deepseek(page_size: u32) -> Self {
        Self {
            num_heads: 128,
            head_dim_ckv: 512,
            head_dim_kpe: 64,
            page_size,
            causal: true,
            sm_scale: 1.0 / 192.0_f32.sqrt(),
        }
    }

    /// Create MLA configuration with custom parameters.
    pub fn new(
        num_heads: u32,
        head_dim_ckv: u32,
        head_dim_kpe: u32,
        page_size: u32,
        sm_scale: f32,
    ) -> Self {
        Self {
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal: true,
            sm_scale,
        }
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Wrapper for FlashInfer MLA attention via direct FFI calls.
///
/// Handles Multi-head Latent Attention for DeepSeek V2/V3 models.
/// Takes absorbed queries (q_nope in CKV space + q_pe) and reads
/// directly from compressed paged caches (ckv_cache + kpe_cache).
#[cfg(feature = "flashinfer")]
pub struct MlaWrapper {
    config: MlaConfig,
    device: Device,
}

#[cfg(feature = "flashinfer")]
impl MlaWrapper {
    /// Create a new MLA wrapper.
    pub fn new(config: MlaConfig, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            return Err(candle_core::Error::Msg(
                "MlaWrapper requires CUDA device".to_string(),
            ));
        }

        Ok(Self {
            config,
            device: device.clone(),
        })
    }

    /// Run MLA attention via FlashInfer FFI.
    ///
    /// # Arguments
    /// * `q_nope` - Absorbed query nope `[nnz, num_heads, head_dim_ckv]`
    /// * `q_pe` - Query PE `[nnz, num_heads, head_dim_kpe]`
    /// * `ckv_cache` - Compressed KV cache `[num_pages, page_size, head_dim_ckv]`
    /// * `kpe_cache` - K position embedding cache `[num_pages, page_size, head_dim_kpe]`
    /// * `workspace` - Workspace buffer
    /// * `qo_indptr` - Cumulative query lengths `[batch_size + 1]`, i32 GPU
    /// * `kv_indptr` - Cumulative page counts `[batch_size + 1]`, i32 GPU
    /// * `kv_indices` - Flattened page IDs `[total_pages]`, i32 GPU
    /// * `kv_lengths` - KV lengths per sequence (host)
    /// * `batch_size` - Number of sequences
    /// * `total_tokens` - Total query tokens
    ///
    /// # Returns
    /// Output tensor `[nnz, num_heads, head_dim_ckv]`
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        q_nope: &Tensor,
        q_pe: &Tensor,
        ckv_cache: &Tensor,
        kpe_cache: &Tensor,
        workspace: &mut WorkspaceBuffer,
        qo_indptr: &Tensor,
        kv_indptr: &Tensor,
        kv_indices: &Tensor,
        kv_lengths: &[usize],
        batch_size: usize,
        _total_tokens: usize,
    ) -> Result<Tensor> {
        let q_nope = tensor_bridge::ensure_contiguous(q_nope)?;
        let q_pe = tensor_bridge::ensure_contiguous(q_pe)?;

        // Ensure workspace is large enough
        let required_size = estimate_mla_workspace(
            batch_size,
            self.config.num_heads as usize,
            self.config.head_dim_ckv as usize,
        );
        workspace.ensure_size(required_size)?;

        // Get raw pointers for tensors that stay on GPU (queries, caches, indices)
        let qn_ptr = tensor_bridge::tensor_to_device_ptr(&q_nope)?;
        let qp_ptr = tensor_bridge::tensor_to_device_ptr(&q_pe)?;
        let ckv_ptr = tensor_bridge::tensor_to_device_ptr(ckv_cache)?;
        let kpe_ptr = tensor_bridge::tensor_to_device_ptr(kpe_cache)?;
        let kv_indices_ptr = tensor_bridge::tensor_to_device_ptr(kv_indices)? as *const i32;
        let workspace_ptr = workspace.as_ptr()?;
        let stream_ptr = tensor_bridge::get_cuda_stream_ptr(&self.device)?;

        // Copy indptr arrays to HOST for plan creation.
        // FlashInfer's MLAPlan reads qo_indptr_h, kv_indptr_h, kv_len_arr_h on the CPU
        // (the C++ parameter names use `_h` suffix = host).
        let qo_indptr_host: Vec<i32> = qo_indptr
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as i32)
            .collect();
        let kv_indptr_host: Vec<i32> = kv_indptr
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as i32)
            .collect();
        let kv_len_host: Vec<i32> = kv_lengths.iter().map(|&l| l as i32).collect();

        // Allocate output: same shape as q_nope [nnz, num_heads, head_dim_ckv]
        let output = Tensor::zeros_like(&q_nope)?;
        let output_ptr = tensor_bridge::tensor_to_device_ptr(&output)? as *mut std::ffi::c_void;

        let dtype = tensor_bridge::candle_to_ffi_dtype(q_nope.dtype())?;

        // Split workspace: first half float, second half int
        let ws_size = workspace.size();
        let float_ws_size = ws_size / 2;
        let int_ws_size = ws_size - float_ws_size;
        let float_ws_ptr = workspace_ptr as *mut std::ffi::c_void;
        let int_ws_ptr = unsafe { workspace_ptr.add(float_ws_size) } as *mut std::ffi::c_void;

        // MLA plan writes metadata to page-locked host memory, then copies to device.
        let mut page_locked_buf: Vec<u8> = vec![0u8; int_ws_size];

        // Create MLA plan (all indptr/len arrays must be HOST pointers)
        let plan = unsafe {
            flashinfer_rs::ffi::MLAPlan::new(
                float_ws_ptr,
                float_ws_size,
                int_ws_ptr,
                int_ws_size,
                page_locked_buf.as_mut_ptr() as *mut std::ffi::c_void,
                int_ws_size,
                qo_indptr_host.as_ptr(),
                kv_indptr_host.as_ptr(),
                kv_len_host.as_ptr(),
                batch_size as i32,
                self.config.num_heads as i32,
                self.config.head_dim_ckv as i32,
                self.config.head_dim_kpe as i32,
                self.config.page_size as i32,
                self.config.causal,
                stream_ptr,
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer MLA plan failed: {e}")))?
        };

        // Determine mask mode
        let mask_mode = if self.config.causal {
            flashinfer_rs::ffi::MaskMode::Causal
        } else {
            flashinfer_rs::ffi::MaskMode::None
        };

        // Run MLA kernel
        unsafe {
            plan.run(
                qn_ptr,
                qp_ptr,
                ckv_ptr,
                kpe_ptr,
                kv_indices_ptr,
                output_ptr,
                std::ptr::null_mut(), // lse
                mask_mode,
                self.config.sm_scale,
                dtype,
                dtype, // KV cache same dtype as query
                stream_ptr,
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer MLA forward failed: {e}")))?;
        }

        Ok(output)
    }
}

/// Stub MLA wrapper when flashinfer feature is disabled.
#[cfg(not(feature = "flashinfer"))]
pub struct MlaWrapper {
    _config: MlaConfig,
}

#[cfg(not(feature = "flashinfer"))]
impl MlaWrapper {
    pub fn new(config: MlaConfig, _device: &Device) -> Result<Self> {
        Ok(Self { _config: config })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        _q_nope: &Tensor,
        _q_pe: &Tensor,
        _ckv_cache: &Tensor,
        _kpe_cache: &Tensor,
        _workspace: &mut WorkspaceBuffer,
        _qo_indptr: &Tensor,
        _kv_indptr: &Tensor,
        _kv_indices: &Tensor,
        _kv_lengths: &[usize],
        _batch_size: usize,
        _total_tokens: usize,
    ) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "FlashInfer MLA not available - enable 'flashinfer' feature".to_string(),
        ))
    }
}

/// Estimate workspace size for prefill operations.
#[cfg(feature = "flashinfer")]
fn estimate_prefill_workspace(
    batch_size: usize,
    total_tokens: usize,
    num_qo_heads: usize,
    head_dim: usize,
) -> usize {
    // Float workspace: temporary output for split-K
    let tmp_size = total_tokens * num_qo_heads * head_dim * 4; // f32
                                                               // Int workspace: request and tile info
    let request_info_size = batch_size * 4 * 4; // i32
    let tile_info_size = total_tokens * 4; // i32
                                           // Total with generous padding
    let total = tmp_size + request_info_size + tile_info_size + 16384;
    // Ensure at least 16 MB
    total.max(16 * 1024 * 1024)
}

/// Estimate workspace size for decode operations.
#[cfg(feature = "flashinfer")]
fn estimate_decode_workspace(batch_size: usize, num_qo_heads: usize, head_dim: usize) -> usize {
    // Float workspace: tmp_v for split-K, tmp_s for softmax
    let tmp_v_size = batch_size * num_qo_heads * head_dim * 4; // f32
    let tmp_s_size = batch_size * num_qo_heads * 4; // f32
                                                    // Int workspace: partition info
    let partition_info_size = batch_size * num_qo_heads * 2 * 4; // i32
                                                                 // Total with generous padding
    let total = tmp_v_size + tmp_s_size + partition_info_size + 16384;
    // Ensure at least 16 MB
    total.max(16 * 1024 * 1024)
}

/// Estimate workspace size for MLA operations.
#[cfg(feature = "flashinfer")]
fn estimate_mla_workspace(batch_size: usize, num_heads: usize, head_dim_ckv: usize) -> usize {
    // Float workspace: tmp_v for split-K + tmp_s for softmax
    let tmp_v_size = batch_size * num_heads * head_dim_ckv * 4; // f32
    let tmp_s_size = batch_size * num_heads * 4; // f32
                                                 // Int workspace: partition info
    let partition_info_size = batch_size * 2 * 4; // i32
                                                  // Total with generous padding
    let total = tmp_v_size + tmp_s_size + partition_info_size + 16384;
    // Ensure at least 16 MB (MLA has large intermediate state due to 128 heads)
    total.max(16 * 1024 * 1024)
}

/// Stub implementations when flashinfer feature is disabled.
#[cfg(not(feature = "flashinfer"))]
pub struct PrefillWrapper {
    _config: FlashInferConfig,
}

#[cfg(not(feature = "flashinfer"))]
impl PrefillWrapper {
    pub fn new(config: FlashInferConfig, _device: &Device) -> Result<Self> {
        Ok(Self { _config: config })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        _q: &Tensor,
        _k_cache: &Tensor,
        _v_cache: &Tensor,
        _workspace: &mut WorkspaceBuffer,
        _qo_indptr: &Tensor,
        _kv_indptr: &Tensor,
        _kv_indices: &Tensor,
        _kv_last_page_len: &Tensor,
        _batch_size: usize,
        _total_tokens: usize,
    ) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }
}

#[cfg(not(feature = "flashinfer"))]
pub struct DecodeWrapper {
    _config: FlashInferConfig,
}

#[cfg(not(feature = "flashinfer"))]
impl DecodeWrapper {
    pub fn new(config: FlashInferConfig, _device: &Device) -> Result<Self> {
        Ok(Self { _config: config })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        _q: &Tensor,
        _k_cache: &Tensor,
        _v_cache: &Tensor,
        _workspace: &mut WorkspaceBuffer,
        _kv_indptr: &Tensor,
        _kv_indices: &Tensor,
        _kv_last_page_len: &Tensor,
        _batch_size: usize,
    ) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = FlashInferConfig::new(32, 8, 128, 16);
        assert_eq!(config.num_qo_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.page_size, 16);
        assert!(config.causal);
        assert_eq!(config.soft_cap, 0.0);
        assert_eq!(config.kv_layout, KVCacheLayout::NHD);
    }

    #[test]
    fn test_config_with_soft_cap() {
        let config = FlashInferConfig::new(32, 8, 128, 16).with_soft_cap(50.0);
        assert_eq!(config.soft_cap, 50.0);
    }

    #[test]
    fn test_config_with_kv_layout() {
        let config = FlashInferConfig::new(32, 8, 128, 16).with_kv_layout(KVCacheLayout::HND);
        assert_eq!(config.kv_layout, KVCacheLayout::HND);
    }

    #[test]
    #[cfg(feature = "flashinfer")]
    fn test_estimate_prefill_workspace() {
        let size = estimate_prefill_workspace(4, 1024, 32, 128);
        // Should be at least 16 MB
        assert!(size >= 16 * 1024 * 1024);
        // Should be large enough for the actual computation
        assert!(size >= 1024 * 32 * 128 * 4);
    }

    #[test]
    #[cfg(feature = "flashinfer")]
    fn test_estimate_decode_workspace() {
        let size = estimate_decode_workspace(4, 32, 128);
        assert!(size >= 16 * 1024 * 1024);
    }

    #[test]
    fn test_mla_config_deepseek() {
        let config = MlaConfig::deepseek(16);
        assert_eq!(config.num_heads, 128);
        assert_eq!(config.head_dim_ckv, 512);
        assert_eq!(config.head_dim_kpe, 64);
        assert_eq!(config.page_size, 16);
        assert!(config.causal);
        let expected_scale = 1.0 / 192.0_f32.sqrt();
        assert!((config.sm_scale - expected_scale).abs() < 1e-6);
    }

    #[test]
    fn test_mla_config_custom() {
        let config = MlaConfig::new(128, 512, 64, 16, 0.072).with_causal(false);
        assert_eq!(config.num_heads, 128);
        assert!(!config.causal);
    }

    #[test]
    #[cfg(feature = "flashinfer")]
    fn test_estimate_mla_workspace() {
        let size = estimate_mla_workspace(4, 128, 512);
        // Should be at least 16 MB
        assert!(size >= 16 * 1024 * 1024);
        // Should cover tmp_v for split-K: 4 * 128 * 512 * 4 = 1048576
        assert!(size >= 4 * 128 * 512 * 4);
    }
}
