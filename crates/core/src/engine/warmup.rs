//! Warmup infrastructure for JIT compilation and CUDA graph capture.
//!
//! This module provides the warmup system that runs before the engine
//! starts accepting requests. Warmup ensures:
//!
//! 1. **JIT Compilation**: All CUDA kernels are compiled for configured batch sizes
//! 2. **CUDA Graph Capture**: Graphs are captured and cached for fast replay
//!
//! # Usage
//!
//! ```ignore
//! let config = EngineConfig {
//!     cuda_graph_config: CudaGraphConfig::enabled(),
//!     ..Default::default()
//! };
//!
//! let (handle, stats) = start_engine_with_warmup(model, tokenizer, kv_cache, config);
//! println!("Warmed up {} batch sizes", stats.jit_warmed_sizes.len());
//! ```

use candle_core::{Device, Tensor};

use super::cuda_graph::{CudaGraphConfig, CudaGraphError};

// ─── Configuration ────────────────────────────────────────────────────────

/// Configuration for warmup behavior.
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Enable JIT warmup (run forward pass for each size to compile kernels).
    pub enable_jit_warmup: bool,
    /// Enable CUDA graph capture during warmup.
    pub enable_graph_capture: bool,
    /// Batch sizes to warm up for decode operations.
    pub decode_batch_sizes: Vec<usize>,
    /// Whether to log progress during warmup.
    pub show_progress: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            enable_jit_warmup: true,
            enable_graph_capture: false,
            decode_batch_sizes: vec![1, 2, 4, 8, 16, 32],
            show_progress: true,
        }
    }
}

impl From<&CudaGraphConfig> for WarmupConfig {
    fn from(config: &CudaGraphConfig) -> Self {
        Self {
            enable_jit_warmup: true, // Always do JIT warmup
            enable_graph_capture: config.enabled,
            decode_batch_sizes: config.capture_sizes.clone(),
            show_progress: true,
        }
    }
}

impl WarmupConfig {
    /// Create a config for JIT warmup only (no graph capture).
    pub fn jit_only() -> Self {
        Self {
            enable_jit_warmup: true,
            enable_graph_capture: false,
            ..Default::default()
        }
    }

    /// Create a config with both JIT warmup and graph capture.
    pub fn with_graph_capture() -> Self {
        Self {
            enable_jit_warmup: true,
            enable_graph_capture: true,
            ..Default::default()
        }
    }

    /// Check if any warmup is needed.
    pub fn needs_warmup(&self) -> bool {
        (self.enable_jit_warmup || self.enable_graph_capture) && !self.decode_batch_sizes.is_empty()
    }

    /// Set custom batch sizes.
    pub fn with_batch_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.decode_batch_sizes = sizes;
        self
    }
}

// ─── Statistics ───────────────────────────────────────────────────────────

/// Statistics from a warmup operation.
#[derive(Debug, Clone, Default)]
pub struct WarmupStats {
    /// Batch sizes that completed JIT warmup successfully.
    pub jit_warmed_sizes: Vec<usize>,
    /// Number of CUDA graphs successfully captured.
    pub graphs_captured: usize,
    /// Number of CUDA graph captures that failed.
    pub graphs_failed: usize,
    /// Total warmup time in milliseconds.
    pub total_time_ms: u64,
    /// Error messages for failed operations.
    pub errors: Vec<String>,
}

impl WarmupStats {
    /// Create stats for when warmup is skipped.
    pub fn skipped() -> Self {
        Self::default()
    }

    /// Check if warmup completed without errors.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get the number of sizes that failed JIT warmup.
    pub fn jit_failed_count(&self, total_sizes: usize) -> usize {
        total_sizes.saturating_sub(self.jit_warmed_sizes.len())
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────

/// Errors that can occur during warmup.
#[derive(Debug, thiserror::Error)]
pub enum WarmupError {
    /// Failed to allocate KV cache blocks for dummy sequences.
    #[error("cache allocation failed: {0}")]
    CacheAllocation(String),

    /// Model forward pass failed during warmup.
    #[error("forward pass failed: {0}")]
    ForwardFailed(#[from] candle_core::Error),

    /// CUDA graph capture failed.
    #[error("CUDA graph capture failed: {0}")]
    GraphCapture(#[from] CudaGraphError),

    /// Device synchronization failed.
    #[error("device sync failed")]
    SyncFailed,

    /// Invalid batch size configuration.
    #[error("invalid batch size {0}: {1}")]
    InvalidBatchSize(usize, String),
}

// ─── Dummy Input Generation ───────────────────────────────────────────────

/// Trait for generating dummy inputs during warmup.
///
/// Different model architectures may require different dummy input patterns.
/// For example, some models may need valid token IDs within vocabulary range.
pub trait DummyInputGenerator: Send + Sync {
    /// Generate dummy input token IDs for the given batch size.
    ///
    /// Returns a tensor of shape `[batch_size, 1]` for decode warmup.
    fn generate_input_ids(&self, batch_size: usize, device: &Device)
        -> candle_core::Result<Tensor>;

    /// Get the vocabulary size for output buffer allocation.
    fn vocab_size(&self) -> usize;
}

/// Default dummy input generator using zero tokens.
///
/// This is suitable for most models where the actual token values
/// don't affect kernel compilation.
pub struct DefaultDummyInputGenerator {
    vocab_size: usize,
}

impl DefaultDummyInputGenerator {
    /// Create a new generator with the given vocabulary size.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl DummyInputGenerator for DefaultDummyInputGenerator {
    fn generate_input_ids(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        // Use zeros - token value doesn't affect kernel compilation
        Tensor::zeros((batch_size, 1), candle_core::DType::U32, device)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Dummy input generator that uses random valid token IDs.
///
/// Use this for models that may have special handling for specific tokens.
pub struct RandomDummyInputGenerator {
    vocab_size: usize,
}

impl RandomDummyInputGenerator {
    /// Create a new generator with the given vocabulary size.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl DummyInputGenerator for RandomDummyInputGenerator {
    fn generate_input_ids(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        // Use token ID 1 (usually a valid token in most vocabularies)
        let token_id = 1u32.min(self.vocab_size.saturating_sub(1) as u32);
        let data = vec![token_id; batch_size];
        Tensor::from_vec(data, (batch_size, 1), device)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ─── Dummy Sequence for Warmup ────────────────────────────────────────────

use crate::kv_cache::{BlockId, BlockTable};

/// A dummy sequence created for warmup purposes.
///
/// This holds the allocated resources that must be cleaned up after warmup.
pub struct DummySequence {
    /// Block table with allocated cache blocks.
    pub block_table: BlockTable,
    /// Block IDs for the sequence.
    pub block_ids: Vec<BlockId>,
    /// Slot mapping for the current position.
    pub slot_mapping: Vec<usize>,
    /// Current sequence length offset.
    pub seqlen_offset: usize,
}

impl DummySequence {
    /// Create a new dummy sequence with the given block table.
    pub fn new(block_table: BlockTable, seqlen_offset: usize) -> Self {
        let block_ids = block_table.block_ids().to_vec();
        let slot_mapping = block_table.slot_mapping(seqlen_offset, 1);
        Self {
            block_table,
            block_ids,
            slot_mapping,
            seqlen_offset,
        }
    }

    /// Release the block table and return it for cleanup.
    pub fn release(self) -> BlockTable {
        self.block_table
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();
        assert!(config.enable_jit_warmup);
        assert!(!config.enable_graph_capture);
        assert_eq!(config.decode_batch_sizes, vec![1, 2, 4, 8, 16, 32]);
        assert!(config.show_progress);
    }

    #[test]
    fn test_warmup_config_from_cuda_graph_config() {
        let cuda_config = CudaGraphConfig::enabled();
        let warmup_config = WarmupConfig::from(&cuda_config);

        assert!(warmup_config.enable_jit_warmup);
        assert!(warmup_config.enable_graph_capture);
        assert_eq!(warmup_config.decode_batch_sizes, cuda_config.capture_sizes);
    }

    #[test]
    fn test_warmup_config_jit_only() {
        let config = WarmupConfig::jit_only();
        assert!(config.enable_jit_warmup);
        assert!(!config.enable_graph_capture);
    }

    #[test]
    fn test_warmup_config_with_graph_capture() {
        let config = WarmupConfig::with_graph_capture();
        assert!(config.enable_jit_warmup);
        assert!(config.enable_graph_capture);
    }

    #[test]
    fn test_warmup_config_needs_warmup() {
        let config = WarmupConfig::default();
        assert!(config.needs_warmup());

        let config = WarmupConfig {
            enable_jit_warmup: false,
            enable_graph_capture: false,
            ..Default::default()
        };
        assert!(!config.needs_warmup());

        let config = WarmupConfig {
            decode_batch_sizes: vec![],
            ..Default::default()
        };
        assert!(!config.needs_warmup());
    }

    #[test]
    fn test_warmup_config_with_batch_sizes() {
        let config = WarmupConfig::default().with_batch_sizes(vec![1, 4, 16]);
        assert_eq!(config.decode_batch_sizes, vec![1, 4, 16]);
    }

    #[test]
    fn test_warmup_stats_default() {
        let stats = WarmupStats::default();
        assert!(stats.jit_warmed_sizes.is_empty());
        assert_eq!(stats.graphs_captured, 0);
        assert_eq!(stats.graphs_failed, 0);
        assert_eq!(stats.total_time_ms, 0);
        assert!(stats.errors.is_empty());
    }

    #[test]
    fn test_warmup_stats_skipped() {
        let stats = WarmupStats::skipped();
        assert!(stats.is_success());
        assert!(stats.jit_warmed_sizes.is_empty());
    }

    #[test]
    fn test_warmup_stats_is_success() {
        let mut stats = WarmupStats::default();
        assert!(stats.is_success());

        stats.errors.push("some error".to_string());
        assert!(!stats.is_success());
    }

    #[test]
    fn test_warmup_stats_jit_failed_count() {
        let mut stats = WarmupStats::default();
        stats.jit_warmed_sizes = vec![1, 2, 4];

        assert_eq!(stats.jit_failed_count(6), 3);
        assert_eq!(stats.jit_failed_count(3), 0);
        assert_eq!(stats.jit_failed_count(2), 0); // saturating
    }

    #[test]
    fn test_default_dummy_input_generator() {
        let gen = DefaultDummyInputGenerator::new(32000);
        assert_eq!(gen.vocab_size(), 32000);

        let input = gen.generate_input_ids(4, &Device::Cpu).unwrap();
        assert_eq!(input.dims(), &[4, 1]);

        // All values should be zeros
        let values: Vec<u32> = input.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_random_dummy_input_generator() {
        let gen = RandomDummyInputGenerator::new(32000);
        assert_eq!(gen.vocab_size(), 32000);

        let input = gen.generate_input_ids(4, &Device::Cpu).unwrap();
        assert_eq!(input.dims(), &[4, 1]);

        // All values should be 1
        let values: Vec<u32> = input.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_random_dummy_input_generator_small_vocab() {
        let gen = RandomDummyInputGenerator::new(1);
        let input = gen.generate_input_ids(2, &Device::Cpu).unwrap();
        let values: Vec<u32> = input.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 0)); // min(1, 0) = 0
    }

    #[test]
    fn test_warmup_error_display() {
        let err = WarmupError::CacheAllocation("out of blocks".to_string());
        assert_eq!(err.to_string(), "cache allocation failed: out of blocks");

        let err = WarmupError::SyncFailed;
        assert_eq!(err.to_string(), "device sync failed");

        let err = WarmupError::InvalidBatchSize(0, "must be positive".to_string());
        assert_eq!(err.to_string(), "invalid batch size 0: must be positive");
    }
}
