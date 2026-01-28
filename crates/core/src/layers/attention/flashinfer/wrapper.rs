//! High-level wrappers for FlashInfer operations.
//!
//! Provides PrefillWrapper and DecodeWrapper that encapsulate
//! BatchPrefillHandler and BatchDecodeHandler from flashinfer-rs.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

#[allow(unused_imports)]
use candle_core::Device;

use super::metadata::FlashInferMetadata;
use super::workspace::WorkspaceBuffer;

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
    /// Rotary embedding mode (0 = none, 1 = interleaved, 2 = half)
    pub rotary_mode: u32,
    /// Soft-capping value (0.0 = disabled)
    pub soft_cap: f32,
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
            rotary_mode: 0,
            soft_cap: 0.0,
        }
    }

    /// Enable soft-capping (for Gemma-2, etc.).
    pub fn with_soft_cap(mut self, soft_cap: f32) -> Self {
        self.soft_cap = soft_cap;
        self
    }

    /// Set rotary mode.
    pub fn with_rotary_mode(mut self, mode: u32) -> Self {
        self.rotary_mode = mode;
        self
    }
}

/// Wrapper for FlashInfer BatchPrefillHandler.
///
/// Handles prefill attention where we have multiple query tokens
/// attending to cached KV and new KV.
#[cfg(feature = "flashinfer")]
pub struct PrefillWrapper {
    handler: flashinfer_rs::BatchPrefillHandler,
    workspace: Arc<WorkspaceBuffer>,
    config: FlashInferConfig,
    planned: bool,
}

#[cfg(feature = "flashinfer")]
impl PrefillWrapper {
    /// Create a new prefill wrapper.
    pub fn new(workspace: Arc<WorkspaceBuffer>, config: FlashInferConfig) -> Result<Self> {
        let device = workspace.device();
        if !device.is_cuda() {
            return Err(candle_core::Error::Msg(
                "PrefillWrapper requires CUDA device".to_string(),
            ));
        }

        let fi_config = flashinfer_rs::AttentionConfig::new(
            config.num_qo_heads,
            config.num_kv_heads,
            config.head_dim,
        )
        .with_page_size(config.page_size as usize)
        .with_causal(config.causal);

        let handler = flashinfer_rs::BatchPrefillHandler::new(device.ordinal()?, fi_config)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer init failed: {}", e)))?;

        Ok(Self {
            handler,
            workspace,
            config,
            planned: false,
        })
    }

    /// Plan the prefill operation for given batch configuration.
    ///
    /// Must be called before `run()` when batch shape changes.
    pub fn plan<T: half::GpuFloat>(
        &mut self,
        metadata: &FlashInferMetadata,
        q_lens: &[usize],
        dtype: DType,
    ) -> Result<()> {
        // Convert metadata tensors to raw slices
        let indptr: Vec<i32> = metadata.paged_kv_indptr.to_vec1()?;
        let indices: Vec<i32> = metadata.paged_kv_indices.to_vec1()?;
        let last_page_lens: Vec<i32> = metadata.paged_kv_last_page_len.to_vec1()?;

        self.handler
            .plan::<T>(
                q_lens,
                &indptr.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                &indices.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                &last_page_lens
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer plan failed: {}", e)))?;

        self.planned = true;
        Ok(())
    }

    /// Run prefill attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor [total_tokens, num_qo_heads, head_dim]
    /// * `k_cache` - Key cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `v_cache` - Value cache [num_pages, page_size, num_kv_heads, head_dim]
    ///
    /// # Returns
    /// Output tensor [total_tokens, num_qo_heads, head_dim]
    pub fn run<T: half::GpuFloat>(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
    ) -> Result<Tensor> {
        if !self.planned {
            return Err(candle_core::Error::Msg(
                "Must call plan() before run()".to_string(),
            ));
        }

        // Allocate output
        let output = Tensor::zeros_like(q)?;

        // Get raw pointers and call handler
        // Note: This requires flashinfer-rs to expose a tensor-based API
        // or we need to use the raw FFI layer
        self.handler
            .forward_with_tensors::<T>(q, k_cache, v_cache, &output)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer prefill failed: {}", e)))?;

        Ok(output)
    }
}

/// Wrapper for FlashInfer BatchDecodeHandler.
///
/// Handles decode attention where we have a single query token per sequence
/// attending to the entire cached KV history.
#[cfg(feature = "flashinfer")]
pub struct DecodeWrapper {
    handler: flashinfer_rs::BatchDecodeHandler,
    workspace: Arc<WorkspaceBuffer>,
    config: FlashInferConfig,
    planned: bool,
}

#[cfg(feature = "flashinfer")]
impl DecodeWrapper {
    /// Create a new decode wrapper.
    pub fn new(workspace: Arc<WorkspaceBuffer>, config: FlashInferConfig) -> Result<Self> {
        let device = workspace.device();
        if !device.is_cuda() {
            return Err(candle_core::Error::Msg(
                "DecodeWrapper requires CUDA device".to_string(),
            ));
        }

        let fi_config = flashinfer_rs::AttentionConfig::new(
            config.num_qo_heads,
            config.num_kv_heads,
            config.head_dim,
        )
        .with_page_size(config.page_size as usize)
        .with_causal(config.causal);

        let handler = flashinfer_rs::BatchDecodeHandler::new(device.ordinal()?, fi_config)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer init failed: {}", e)))?;

        Ok(Self {
            handler,
            workspace,
            config,
            planned: false,
        })
    }

    /// Plan the decode operation for given batch configuration.
    ///
    /// Must be called before `run()` when batch shape changes.
    pub fn plan<T: half::GpuFloat>(
        &mut self,
        metadata: &FlashInferMetadata,
        kv_lengths: &[usize],
    ) -> Result<()> {
        let indptr: Vec<i32> = metadata.paged_kv_indptr.to_vec1()?;
        let indices: Vec<i32> = metadata.paged_kv_indices.to_vec1()?;
        let last_page_lens: Vec<i32> = metadata.paged_kv_last_page_len.to_vec1()?;

        self.handler
            .plan::<T>(
                kv_lengths,
                &indptr.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                &indices.iter().map(|&x| x as usize).collect::<Vec<_>>(),
                &last_page_lens
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer plan failed: {}", e)))?;

        self.planned = true;
        Ok(())
    }

    /// Run decode attention.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, num_qo_heads, head_dim]
    /// * `k_cache` - Key cache [num_pages, page_size, num_kv_heads, head_dim]
    /// * `v_cache` - Value cache [num_pages, page_size, num_kv_heads, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch_size, num_qo_heads, head_dim]
    pub fn run<T: half::GpuFloat>(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
    ) -> Result<Tensor> {
        if !self.planned {
            return Err(candle_core::Error::Msg(
                "Must call plan() before run()".to_string(),
            ));
        }

        let output = Tensor::zeros_like(q)?;

        self.handler
            .forward_with_tensors::<T>(q, k_cache, v_cache, &output)
            .map_err(|e| candle_core::Error::Msg(format!("FlashInfer decode failed: {}", e)))?;

        Ok(output)
    }

    /// Check if handler is ready (plan has been called).
    pub fn is_planned(&self) -> bool {
        self.planned
    }
}

/// Stub implementations when flashinfer feature is disabled.
#[cfg(not(feature = "flashinfer"))]
pub struct PrefillWrapper {
    _workspace: Arc<WorkspaceBuffer>,
    #[allow(dead_code)]
    config: FlashInferConfig,
}

#[cfg(not(feature = "flashinfer"))]
impl PrefillWrapper {
    pub fn new(workspace: Arc<WorkspaceBuffer>, config: FlashInferConfig) -> Result<Self> {
        Ok(Self {
            _workspace: workspace,
            config,
        })
    }

    pub fn plan(
        &mut self,
        _metadata: &FlashInferMetadata,
        _q_lens: &[usize],
        _dtype: DType,
    ) -> Result<()> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }

    pub fn run(&self, _q: &Tensor, _k_cache: &Tensor, _v_cache: &Tensor) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }
}

#[cfg(not(feature = "flashinfer"))]
pub struct DecodeWrapper {
    _workspace: Arc<WorkspaceBuffer>,
    #[allow(dead_code)]
    config: FlashInferConfig,
}

#[cfg(not(feature = "flashinfer"))]
impl DecodeWrapper {
    pub fn new(workspace: Arc<WorkspaceBuffer>, config: FlashInferConfig) -> Result<Self> {
        Ok(Self {
            _workspace: workspace,
            config,
        })
    }

    pub fn plan(&mut self, _metadata: &FlashInferMetadata, _kv_lengths: &[usize]) -> Result<()> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }

    pub fn run(&self, _q: &Tensor, _k_cache: &Tensor, _v_cache: &Tensor) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "FlashInfer not available - enable 'flashinfer' feature".to_string(),
        ))
    }

    pub fn is_planned(&self) -> bool {
        false
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
    }

    #[test]
    fn test_config_with_soft_cap() {
        let config = FlashInferConfig::new(32, 8, 128, 16).with_soft_cap(50.0);
        assert_eq!(config.soft_cap, 50.0);
    }
}
