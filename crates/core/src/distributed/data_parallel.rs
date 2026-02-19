//! Data Parallelism (DP) infrastructure for inference.
//!
//! In inference, data parallelism means running identical model replicas on separate
//! GPU groups, with request routing handled at the server level. Each DP rank is an
//! independent full engine instance — no gradient synchronization occurs.
//!
//! The one coordination requirement is that DP ranks must execute forward passes in
//! lock-step when CUDA graphs or MoE expert layers are active. Before each forward pass
//! all ranks all_reduce the `num_tokens` count so they run the same CUDA graph bucket.
//!
//! Reference: `reference/vllm/vllm/v1/worker/dp_utils.py::coordinate_batch_across_dp()`

use std::sync::Arc;

use super::communicator::{DeviceCommunicator, ReduceOp};
use super::error::Result;

// ─── DpContext ────────────────────────────────────────────────────────────────

/// Context for a single DP rank.
///
/// Each engine instance holds one `DpContext`. For single-GPU deployments (the
/// common case), `DpContext::single_gpu()` is used and all coordination calls
/// are no-ops. For multi-rank DP, the context carries a communicator that talks
/// to a separate NCCL group spanning one GPU per replica.
///
/// `DpContext` is `Clone`; cloning is cheap (Arc clone for the communicator).
#[derive(Clone)]
pub struct DpContext {
    /// This engine's index within the DP group (0-based).
    pub rank: usize,
    /// Total number of DP replicas (1 = single-GPU, no coordination).
    pub world_size: usize,
    /// Communicator for the DP group. `None` when `world_size == 1`.
    comm: Option<Arc<dyn DeviceCommunicator>>,
}

impl DpContext {
    /// Create a single-GPU context with no communicator.
    ///
    /// This is the default for all single-GPU deployments and the default value
    /// stored in `EngineConfig`. All coordination calls become no-ops.
    pub fn single_gpu() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            comm: None,
        }
    }

    /// Create a multi-rank DP context.
    ///
    /// # Panics
    /// Panics if `rank >= world_size` or `world_size == 0`.
    pub fn new(rank: usize, world_size: usize, comm: Arc<dyn DeviceCommunicator>) -> Self {
        assert!(
            world_size > 0,
            "DP world_size must be > 0, got {world_size}"
        );
        assert!(
            rank < world_size,
            "DP rank {rank} out of range (world_size={world_size})"
        );
        Self {
            rank,
            world_size,
            comm: Some(comm),
        }
    }

    /// Returns `true` when this is a single-GPU context (no coordination needed).
    #[inline]
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Get the communicator.
    ///
    /// # Panics
    /// Panics if called on a single-GPU context. Guard with `is_single()` first.
    pub fn comm(&self) -> &dyn DeviceCommunicator {
        self.comm
            .as_ref()
            .expect("comm() called on single-GPU DpContext — guard with is_single()")
            .as_ref()
    }
}

impl std::fmt::Debug for DpContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DpContext")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .field("is_single", &self.is_single())
            .finish()
    }
}

// ─── BatchCoordinationResult ─────────────────────────────────────────────────

/// Result of batch coordination across DP ranks.
///
/// After `coordinate_batch_across_dp`, all DP ranks will have the same
/// `synced_num_tokens` value so they can select the same CUDA graph bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchCoordinationResult {
    /// Maximum `num_tokens` observed across all DP ranks.
    ///
    /// All ranks must execute a forward pass with at least this many tokens
    /// (padding shorter batches as needed) to keep CUDA graph execution in sync.
    pub synced_num_tokens: usize,
    /// `true` if any DP rank has work to do.
    ///
    /// When `false`, all ranks are idle and can skip the forward pass entirely.
    /// When `true`, ranks with empty batches still run a dummy forward pass so
    /// that MoE expert collective operations complete correctly.
    pub has_active_work: bool,
}

// ─── coordinate_batch_across_dp ──────────────────────────────────────────────

/// Synchronize batch size across all DP ranks before each forward pass.
///
/// Ensures all ranks select the same CUDA graph bucket (via `synced_num_tokens`)
/// and that no rank skips a forward pass when others are still active (required
/// for MoE expert all-reduce correctness).
///
/// ## Algorithm
/// 1. Pack `[num_tokens as f32, (num_tokens > 0) as f32]` into a 2-element vector.
/// 2. All-reduce (max) across DP ranks.
/// 3. Unpack: `synced_num_tokens = result[0] as usize`, `has_active_work = result[1] > 0.5`.
///
/// ## Single-GPU fast path
/// If `ctx.is_single()`, returns immediately without any tensor operations.
///
/// Reference: `dp_utils.py::coordinate_batch_across_dp()` lines 173–240.
pub fn coordinate_batch_across_dp(
    num_tokens: usize,
    ctx: &DpContext,
) -> Result<BatchCoordinationResult> {
    if ctx.is_single() {
        return Ok(BatchCoordinationResult {
            synced_num_tokens: num_tokens,
            has_active_work: num_tokens > 0,
        });
    }

    let comm = ctx.comm();

    // Pack into a 2-element f32 tensor on CPU for the all_reduce.
    // Using f32 so both values fit: num_tokens is at most ~64K for any
    // practical batch, well within f32's exact integer range (2^24 ≈ 16M).
    let data = vec![num_tokens as f32, (num_tokens > 0) as u32 as f32];
    let tensor = candle_core::Tensor::from_vec(data, 2, &candle_core::Device::Cpu)?;

    let result = comm.all_reduce(&tensor, ReduceOp::Max)?;

    let vals = result.to_vec1::<f32>()?;
    let synced_num_tokens = vals[0] as usize;
    let has_active_work = vals[1] > 0.5;

    Ok(BatchCoordinationResult {
        synced_num_tokens,
        has_active_work,
    })
}

// ─── request_belongs_to_rank ─────────────────────────────────────────────────

/// Deterministic request-to-rank assignment via modulo routing.
///
/// Returns `true` if the request identified by `request_id` should be handled
/// by the DP rank at index `dp_rank`. Always returns `true` when `dp_size == 1`.
///
/// This helper is used by external request routers or server-level admission
/// filters; the engine itself does not call it.
#[inline]
pub fn request_belongs_to_rank(request_id: u64, dp_rank: usize, dp_size: usize) -> bool {
    dp_size == 1 || (request_id as usize % dp_size) == dp_rank
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator};

    // ── DpContext construction ───────────────────────────────────────────────

    #[test]
    fn test_dp_context_single_gpu() {
        let ctx = DpContext::single_gpu();
        assert!(ctx.is_single());
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }

    #[test]
    #[should_panic(expected = "DP rank 2 out of range")]
    fn test_dp_context_invalid_rank_panics() {
        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        let _ = DpContext::new(2, 2, comm); // rank >= world_size
    }

    #[test]
    fn test_dp_context_comm_panics_on_single() {
        let ctx = DpContext::single_gpu();
        // AssertUnwindSafe needed: Arc<dyn DeviceCommunicator> may not be RefUnwindSafe.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.comm();
        }));
        assert!(result.is_err(), "comm() on single-GPU context should panic");
    }

    #[test]
    fn test_dp_context_clone_is_cheap() {
        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        let ctx = DpContext::new(0, 2, comm);
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.rank, ctx.rank);
        assert_eq!(ctx2.world_size, ctx.world_size);
    }

    // ── request_belongs_to_rank ─────────────────────────────────────────────

    #[test]
    fn test_request_belongs_to_rank_single_gpu() {
        // With dp_size=1, every request belongs to rank 0.
        for id in 0u64..16 {
            assert!(request_belongs_to_rank(id, 0, 1));
        }
    }

    #[test]
    fn test_request_belongs_to_rank_two_ranks() {
        // Even IDs → rank 0, odd IDs → rank 1.
        assert!(request_belongs_to_rank(0, 0, 2));
        assert!(request_belongs_to_rank(2, 0, 2));
        assert!(request_belongs_to_rank(1, 1, 2));
        assert!(request_belongs_to_rank(3, 1, 2));
        assert!(!request_belongs_to_rank(0, 1, 2));
        assert!(!request_belongs_to_rank(1, 0, 2));
    }

    #[test]
    fn test_request_belongs_to_rank_four_ranks() {
        for dp_size in [2usize, 4] {
            for id in 0u64..64 {
                let owner = (id as usize) % dp_size;
                for rank in 0..dp_size {
                    assert_eq!(
                        request_belongs_to_rank(id, rank, dp_size),
                        rank == owner,
                        "id={id} dp_size={dp_size} rank={rank}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_request_belongs_to_rank_invariant() {
        // Exactly one rank owns each request_id.
        let dp_size = 4usize;
        for id in 0u64..32 {
            let owners: usize = (0..dp_size)
                .filter(|&r| request_belongs_to_rank(id, r, dp_size))
                .count();
            assert_eq!(owners, 1, "request {id} should have exactly one owner");
        }
    }

    // ── coordinate_batch_across_dp — single-GPU ─────────────────────────────

    #[test]
    fn test_coordinate_batch_single_gpu_is_noop() {
        let ctx = DpContext::single_gpu();

        let result = coordinate_batch_across_dp(42, &ctx).unwrap();
        assert_eq!(result.synced_num_tokens, 42);
        assert!(result.has_active_work);
    }

    #[test]
    fn test_coordinate_batch_has_active_work_true() {
        let ctx = DpContext::single_gpu();
        let result = coordinate_batch_across_dp(1, &ctx).unwrap();
        assert!(result.has_active_work);
    }

    #[test]
    fn test_coordinate_batch_has_active_work_false() {
        let ctx = DpContext::single_gpu();
        let result = coordinate_batch_across_dp(0, &ctx).unwrap();
        assert!(!result.has_active_work);
        assert_eq!(result.synced_num_tokens, 0);
    }

    #[test]
    fn test_batch_coordination_result_zero_tokens() {
        let ctx = DpContext::single_gpu();
        let result = coordinate_batch_across_dp(0, &ctx).unwrap();
        assert_eq!(result.synced_num_tokens, 0);
        assert!(!result.has_active_work);
    }

    #[test]
    fn test_coordinate_batch_large_num_tokens() {
        let ctx = DpContext::single_gpu();
        // f32 can exactly represent integers up to 2^24 ≈ 16M, well above any
        // practical batch size.
        let large = 65_536usize;
        let result = coordinate_batch_across_dp(large, &ctx).unwrap();
        assert_eq!(result.synced_num_tokens, large);
        assert!(result.has_active_work);
    }

    // ── coordinate_batch_across_dp — mock multi-GPU ─────────────────────────

    #[test]
    fn test_coordinate_batch_mock_two_ranks() {
        // MockCommunicator::all_reduce returns the tensor unchanged (identity).
        // So the result equals what rank 0 put in — this verifies the pack/unpack
        // round-trip works correctly for a multi-rank context.
        let pg = LocalProcessGroup::with_rank(0, 2);
        let comm = Arc::new(MockCommunicator::new(pg));
        let ctx = DpContext::new(0, 2, comm);

        let result = coordinate_batch_across_dp(128, &ctx).unwrap();
        assert_eq!(result.synced_num_tokens, 128);
        assert!(result.has_active_work);
    }

    // ── ParallelConfig integration ───────────────────────────────────────────

    #[test]
    fn test_dp_config_world_size_unchanged() {
        use crate::distributed::ParallelConfig;

        // DP replicas do NOT increase world_size() — that stays tp * pp.
        let cfg = ParallelConfig::with_dp(2, 4, 8);
        assert_eq!(cfg.tensor_parallel_size, 2);
        assert_eq!(cfg.pipeline_parallel_size, 4);
        assert_eq!(cfg.data_parallel_size, 8);
        // world_size is per-replica GPU count: tp * pp
        assert_eq!(cfg.world_size(), 8); // 2 * 4, NOT 2 * 4 * 8
    }
}
