//! Error types for distributed operations.

use thiserror::Error;

/// Errors that can occur during distributed operations.
#[derive(Error, Debug)]
pub enum DistributedError {
    /// Rank is out of valid range for the process group.
    #[error("invalid rank {rank}: must be < world_size {world_size}")]
    InvalidRank { rank: usize, world_size: usize },

    /// Operation requires more than one GPU but only one is available.
    #[error("operation requires world_size > 1, but world_size = 1")]
    SingleGpuNotSupported,

    /// Tensor shape mismatch for collective operation.
    #[error("tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Tensor device mismatch.
    #[error("tensor device mismatch: expected {expected}, got {actual}")]
    DeviceMismatch { expected: String, actual: String },

    /// NCCL operation failed.
    #[error("NCCL error: {0}")]
    NcclError(String),

    /// Communication timeout.
    #[error("communication timeout after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// Underlying tensor operation failed.
    #[error("tensor error: {0}")]
    TensorError(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, DistributedError>;
