//! Process group abstractions for distributed inference.
//!
//! A process group represents a set of processes that participate in
//! collective operations. In tensor parallelism, each GPU is a process.

/// Configuration for parallel execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    /// Number of GPUs for tensor parallelism (splitting layers).
    pub tensor_parallel_size: usize,
    /// Number of GPUs for pipeline parallelism (splitting stages).
    pub pipeline_parallel_size: usize,
    /// Number of GPUs for expert parallelism (MoE expert distribution).
    /// Must divide total_experts evenly. When EP > 1, TP is disabled within experts.
    pub expert_parallel_size: usize,
    /// Number of GPUs for decode context parallelism (splitting KV cache along seq dim).
    /// DCP ranks share the same physical GPUs as TP — each GPU participates in both a
    /// TP group and a DCP group. `world_size()` is still `tp * pp`.
    pub decode_context_parallel_size: usize,
}

impl ParallelConfig {
    /// Create a new parallel configuration.
    ///
    /// # Panics
    /// Panics if any size is 0.
    pub fn new(tensor_parallel_size: usize, pipeline_parallel_size: usize) -> Self {
        assert!(tensor_parallel_size > 0, "tensor_parallel_size must be > 0");
        assert!(
            pipeline_parallel_size > 0,
            "pipeline_parallel_size must be > 0"
        );
        Self {
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size: 1,
            decode_context_parallel_size: 1,
        }
    }

    /// Create with all parallelism dimensions.
    ///
    /// # Panics
    /// Panics if any size is 0.
    pub fn new_with_ep(
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        expert_parallel_size: usize,
    ) -> Self {
        assert!(tensor_parallel_size > 0, "tensor_parallel_size must be > 0");
        assert!(
            pipeline_parallel_size > 0,
            "pipeline_parallel_size must be > 0"
        );
        assert!(expert_parallel_size > 0, "expert_parallel_size must be > 0");
        Self {
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            decode_context_parallel_size: 1,
        }
    }

    /// Create a configuration with decode context parallelism.
    ///
    /// DCP splits the KV cache along the sequence dimension across `dcp_size` ranks.
    /// These ranks share the same physical GPUs as TP ranks.
    ///
    /// # Panics
    /// Panics if any size is 0.
    pub fn with_dcp(
        tensor_parallel_size: usize,
        pipeline_parallel_size: usize,
        decode_context_parallel_size: usize,
    ) -> Self {
        assert!(tensor_parallel_size > 0, "tensor_parallel_size must be > 0");
        assert!(
            pipeline_parallel_size > 0,
            "pipeline_parallel_size must be > 0"
        );
        assert!(
            decode_context_parallel_size > 0,
            "decode_context_parallel_size must be > 0"
        );
        Self {
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size: 1,
            decode_context_parallel_size,
        }
    }

    /// No parallelism (single GPU).
    pub fn no_parallelism() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: 1,
            decode_context_parallel_size: 1,
        }
    }

    /// Tensor parallelism only.
    pub fn tensor_parallel(size: usize) -> Self {
        Self::new(size, 1)
    }

    /// Pipeline parallelism only.
    pub fn pipeline_parallel(size: usize) -> Self {
        Self::new(1, size)
    }

    /// Expert parallelism only (for MoE models).
    pub fn expert_parallel(size: usize) -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            expert_parallel_size: size,
            decode_context_parallel_size: 1,
        }
    }

    /// Whether decode context parallelism is enabled.
    pub fn uses_context_parallelism(&self) -> bool {
        self.decode_context_parallel_size > 1
    }

    /// Total number of GPUs required.
    ///
    /// Note: EP ranks share the same physical GPUs as TP ranks.
    /// EP and TP are mutually exclusive for expert layers.
    pub fn world_size(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }

    /// Whether this is effectively single-GPU execution.
    pub fn is_single_gpu(&self) -> bool {
        self.world_size() == 1 && self.expert_parallel_size == 1
    }

    /// Whether expert parallelism is enabled.
    pub fn uses_expert_parallelism(&self) -> bool {
        self.expert_parallel_size > 1
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::no_parallelism()
    }
}

/// Context for Expert Parallelism within a process group.
///
/// Provides the EP rank and size for expert placement decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EPContext {
    /// This rank's position in the EP group (0..ep_size).
    pub ep_rank: usize,
    /// Total number of EP ranks.
    pub ep_size: usize,
}

impl EPContext {
    /// Create a new EP context.
    pub fn new(ep_rank: usize, ep_size: usize) -> Self {
        assert!(ep_rank < ep_size, "ep_rank must be < ep_size");
        Self { ep_rank, ep_size }
    }

    /// Create context for single-GPU (no EP).
    pub fn single_gpu() -> Self {
        Self {
            ep_rank: 0,
            ep_size: 1,
        }
    }

    /// Whether this is effectively single-GPU EP.
    pub fn is_single(&self) -> bool {
        self.ep_size == 1
    }
}

impl Default for EPContext {
    fn default() -> Self {
        Self::single_gpu()
    }
}

/// Trait for process group operations.
///
/// A process group manages rank assignment and provides the foundation
/// for collective communications.
pub trait ProcessGroup: Send + Sync {
    /// Global rank of this process (0..world_size).
    fn rank(&self) -> usize;

    /// Total number of processes in the group.
    fn world_size(&self) -> usize;

    /// Local rank on this node (for multi-node setups).
    fn local_rank(&self) -> usize;

    /// Whether this is the coordinator (rank 0).
    fn is_coordinator(&self) -> bool {
        self.rank() == 0
    }

    /// Whether this is a single-process group.
    fn is_single(&self) -> bool {
        self.world_size() == 1
    }
}

/// Local process group for single-GPU execution.
///
/// This is the simplest implementation where world_size = 1.
/// All collective operations become identity/no-ops.
#[derive(Debug, Clone)]
pub struct LocalProcessGroup {
    rank: usize,
    world_size: usize,
}

impl LocalProcessGroup {
    /// Create a new local process group (single GPU).
    pub fn new() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    /// Create a local process group with specific rank/size.
    ///
    /// Useful for testing multi-GPU logic on a single GPU.
    pub fn with_rank(rank: usize, world_size: usize) -> Self {
        assert!(rank < world_size, "rank must be < world_size");
        Self { rank, world_size }
    }
}

impl Default for LocalProcessGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessGroup for LocalProcessGroup {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn local_rank(&self) -> usize {
        self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_config_defaults() {
        let cfg = ParallelConfig::default();
        assert_eq!(cfg.tensor_parallel_size, 1);
        assert_eq!(cfg.pipeline_parallel_size, 1);
    }

    #[test]
    #[should_panic(expected = "tensor_parallel_size must be > 0")]
    fn parallel_config_zero_tp_panics() {
        ParallelConfig::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "pipeline_parallel_size must be > 0")]
    fn parallel_config_zero_pp_panics() {
        ParallelConfig::new(1, 0);
    }

    #[test]
    fn local_pg_is_coordinator() {
        let pg = LocalProcessGroup::new();
        assert!(pg.is_coordinator());
    }

    #[test]
    fn local_pg_is_single() {
        let pg = LocalProcessGroup::new();
        assert!(pg.is_single());
    }

    #[test]
    fn local_pg_with_rank() {
        let pg = LocalProcessGroup::with_rank(2, 4);
        assert_eq!(pg.rank(), 2);
        assert_eq!(pg.world_size(), 4);
        assert!(!pg.is_coordinator());
        assert!(!pg.is_single());
    }

    #[test]
    #[should_panic(expected = "rank must be < world_size")]
    fn local_pg_invalid_rank_panics() {
        LocalProcessGroup::with_rank(5, 4);
    }
}
