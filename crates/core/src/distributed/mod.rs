//! Distributed computing abstractions for tensor/pipeline parallelism.
//!
//! This module provides abstractions for multi-GPU inference:
//! - [`ProcessGroup`] - Process group management (ranks, world size)
//! - [`DeviceCommunicator`] - Collective operations (all_reduce, all_gather, etc.)
//!
//! # Architecture
//!
//! Following vLLM's pattern:
//! - Single GPU: All operations are identity/no-op (world_size=1 bypass)
//! - Multi GPU: Uses NCCL for efficient GPU-to-GPU communication
//!
//! # Usage
//!
//! ```ignore
//! use vllm_core::distributed::{ProcessGroup, LocalProcessGroup};
//!
//! // Single GPU setup (world_size = 1)
//! let pg = LocalProcessGroup::new();
//! assert_eq!(pg.world_size(), 1);
//! ```

mod attention;
mod communicator;
mod context_parallel;
mod error;
#[cfg(feature = "cuda")]
mod launcher;
#[cfg(feature = "cuda")]
mod nccl;
mod parallel_layers;
mod pipeline;
mod process_group;

pub use attention::{TensorParallelAttention, TensorParallelMLP};
pub use communicator::{DeviceCommunicator, MockCommunicator, ReduceOp};
pub use context_parallel::{
    get_dcp_local_seq_lens, get_dcp_local_slot_mapping, lse_correct_and_reduce, CpConfig,
    CpContext, DcpAttentionWrapper,
};
pub use error::DistributedError;
#[cfg(feature = "cuda")]
pub use launcher::{DistributedConfig, NcclProcessGroup};
#[cfg(feature = "cuda")]
pub use nccl::{
    is_nccl_available, NcclCommunicator, NcclDataType, NcclDeviceCommunicator, NcclLibrary,
    NcclUniqueId,
};
pub use parallel_layers::{ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding};
pub use pipeline::{
    merge_microbatches, optimal_microbatches, split_microbatches, PipelineCommunicator,
    PipelineSchedule, PipelineStageConfig, SyncPipelineExecutor,
};
pub use process_group::{EPContext, LocalProcessGroup, ParallelConfig, ProcessGroup};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_process_group_defaults() {
        let pg = LocalProcessGroup::new();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);
        assert_eq!(pg.local_rank(), 0);
    }

    #[test]
    fn parallel_config_no_parallelism() {
        let cfg = ParallelConfig::no_parallelism();
        assert_eq!(cfg.tensor_parallel_size, 1);
        assert_eq!(cfg.pipeline_parallel_size, 1);
        assert_eq!(cfg.world_size(), 1);
    }

    #[test]
    fn parallel_config_tensor_parallel() {
        let cfg = ParallelConfig::tensor_parallel(4);
        assert_eq!(cfg.tensor_parallel_size, 4);
        assert_eq!(cfg.pipeline_parallel_size, 1);
        assert_eq!(cfg.world_size(), 4);
    }

    #[test]
    fn parallel_config_combined() {
        let cfg = ParallelConfig::new(2, 4); // TP=2, PP=4
        assert_eq!(cfg.tensor_parallel_size, 2);
        assert_eq!(cfg.pipeline_parallel_size, 4);
        assert_eq!(cfg.world_size(), 8);
    }

    #[test]
    fn is_single_gpu_check() {
        let single = ParallelConfig::no_parallelism();
        let multi = ParallelConfig::tensor_parallel(2);

        assert!(single.is_single_gpu());
        assert!(!multi.is_single_gpu());
    }
}
