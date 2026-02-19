//! Expert Parallelism (EP) coordinator context.
//!
//! Expert Parallelism partitions MoE experts across GPU ranks. Each rank loads only
//! `num_experts / ep_size` experts. Token routing uses all-to-all communication to
//! send tokens to the rank that owns the selected expert.
//!
//! This module provides [`ExpertParallelContext`]: the runtime coordinator that
//! bundles the EP rank identity with its communicator. It is analogous to
//! [`super::data_parallel::DpContext`] for DP and [`super::context_parallel::CpContext`]
//! for DCP.
//!
//! ## Relation to `EPContext`
//!
//! [`super::process_group::EPContext`] is a plain value type (rank + ep_size) used in
//! model constructor signatures. `ExpertParallelContext` extends that with an optional
//! communicator and is stored in [`crate::engine::EngineConfig`].
//!
//! Model `new_with_ep()` constructors continue to accept `(&EPContext, Arc<dyn DeviceCommunicator>)`.
//! Use [`ExpertParallelContext::to_ep_context`] and [`ExpertParallelContext::comm`] to extract them.

use std::sync::Arc;

use super::communicator::DeviceCommunicator;
use super::process_group::EPContext;

// ─── ExpertParallelContext ────────────────────────────────────────────────────

/// Runtime context for an Expert Parallelism rank.
///
/// Bundles EP rank identity (`rank`, `world_size`) with the optional
/// communicator needed for all-to-all token dispatch.
///
/// For single-GPU deployments (the common case), [`ExpertParallelContext::single_gpu()`]
/// returns a no-op context. All coordination calls in [`crate::moe::TokenDispatcher`]
/// already short-circuit when the communicator is absent.
///
/// `ExpertParallelContext` is `Clone`; cloning is cheap (two `usize` copies + one `Arc` clone).
#[derive(Clone)]
pub struct ExpertParallelContext {
    /// This rank's position in the EP group (0-based).
    pub rank: usize,
    /// Total number of EP ranks (ep_size). 1 = single-GPU, no communication.
    pub world_size: usize,
    /// Communicator for the EP group. `None` when `world_size == 1`.
    comm: Option<Arc<dyn DeviceCommunicator>>,
}

impl ExpertParallelContext {
    /// Create a single-GPU context with no communicator.
    ///
    /// All token dispatch in [`crate::moe::TokenDispatcher`] fast-paths when no
    /// communicator is present, so this is truly zero-overhead.
    pub fn single_gpu() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            comm: None,
        }
    }

    /// Create a multi-rank EP context.
    ///
    /// # Panics
    /// Panics if `rank >= world_size` or `world_size == 0`.
    pub fn new(rank: usize, world_size: usize, comm: Arc<dyn DeviceCommunicator>) -> Self {
        assert!(
            world_size > 0,
            "EP world_size must be > 0, got {world_size}"
        );
        assert!(
            rank < world_size,
            "EP rank {rank} out of range (world_size={world_size})"
        );
        Self {
            rank,
            world_size,
            comm: Some(comm),
        }
    }

    /// Returns `true` when this is a single-GPU context (no EP communication).
    #[inline]
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// Get the EP group communicator.
    ///
    /// # Panics
    /// Panics if called on a single-GPU context. Guard with [`is_single`][Self::is_single] first.
    pub fn comm(&self) -> &dyn DeviceCommunicator {
        self.comm
            .as_ref()
            .expect(
                "comm() called on single-GPU ExpertParallelContext — guard with is_single() first",
            )
            .as_ref()
    }

    /// Convert to the plain [`EPContext`] value type required by model `new_with_ep()` constructors.
    ///
    /// `EPContext` carries only rank + ep_size; the communicator is provided separately via
    /// [`comm()`][Self::comm] or the `Arc<dyn DeviceCommunicator>` extracted from this context.
    pub fn to_ep_context(&self) -> EPContext {
        EPContext::new(self.rank, self.world_size)
    }

    /// Returns a clone of the underlying communicator `Arc`, or `None` for single-GPU.
    ///
    /// Use this when a model constructor needs an owned `Arc<dyn DeviceCommunicator>`.
    pub fn comm_arc(&self) -> Option<Arc<dyn DeviceCommunicator>> {
        self.comm.clone()
    }
}

impl std::fmt::Debug for ExpertParallelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertParallelContext")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .field("is_single", &self.is_single())
            .finish()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::distributed::{LocalProcessGroup, MockCommunicator, ParallelConfig};

    fn make_comm(rank: usize, world_size: usize) -> Arc<dyn DeviceCommunicator> {
        Arc::new(MockCommunicator::new(LocalProcessGroup::with_rank(
            rank, world_size,
        )))
    }

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn test_ep_context_single_gpu() {
        let ctx = ExpertParallelContext::single_gpu();
        assert!(ctx.is_single());
        assert_eq!(ctx.rank, 0);
        assert_eq!(ctx.world_size, 1);
    }

    #[test]
    fn test_ep_context_new_valid() {
        let comm = make_comm(1, 4);
        let ctx = ExpertParallelContext::new(1, 4, comm);
        assert_eq!(ctx.rank, 1);
        assert_eq!(ctx.world_size, 4);
        assert!(!ctx.is_single());
    }

    #[test]
    fn test_ep_context_world_size_one() {
        // Constructing with world_size=1 via new() is valid (ep_size=1 is legal).
        let comm = make_comm(0, 1);
        let ctx = ExpertParallelContext::new(0, 1, comm);
        assert_eq!(ctx.world_size, 1);
        // is_single checks world_size == 1, so this is true even though comm exists.
        assert!(ctx.is_single());
    }

    #[test]
    #[should_panic(expected = "EP rank 4 out of range")]
    fn test_ep_context_invalid_rank_panics() {
        let comm = make_comm(0, 4);
        let _ = ExpertParallelContext::new(4, 4, comm);
    }

    // ── comm() ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ep_context_comm_panics_on_single() {
        let ctx = ExpertParallelContext::single_gpu();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ctx.comm();
        }));
        assert!(result.is_err(), "comm() on single-GPU context should panic");
    }

    #[test]
    fn test_ep_context_comm_returns_communicator() {
        let comm = make_comm(0, 2);
        let ctx = ExpertParallelContext::new(0, 2, comm);
        // Just calling comm() without panic is sufficient
        let _c = ctx.comm();
    }

    // ── to_ep_context ────────────────────────────────────────────────────────

    #[test]
    fn test_ep_context_to_ep_context_single() {
        let ctx = ExpertParallelContext::single_gpu();
        let ep = ctx.to_ep_context();
        assert_eq!(ep.ep_rank, 0);
        assert_eq!(ep.ep_size, 1);
    }

    #[test]
    fn test_ep_context_to_ep_context_multi() {
        let comm = make_comm(2, 8);
        let ctx = ExpertParallelContext::new(2, 8, comm);
        let ep = ctx.to_ep_context();
        assert_eq!(ep.ep_rank, 2);
        assert_eq!(ep.ep_size, 8);
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    #[test]
    fn test_ep_context_clone() {
        let comm = make_comm(0, 4);
        let ctx = ExpertParallelContext::new(0, 4, comm);
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.rank, ctx.rank);
        assert_eq!(ctx2.world_size, ctx.world_size);
    }

    // ── ParallelConfig integration ───────────────────────────────────────────

    #[test]
    fn test_ep_config_uses_expert_parallelism() {
        let cfg = ParallelConfig::expert_parallel(4);
        assert!(cfg.uses_expert_parallelism());
        assert_eq!(cfg.expert_parallel_size, 4);
    }

    #[test]
    fn test_ep_config_world_size_unchanged() {
        // expert_parallel_size does NOT contribute to world_size() — EP shares TP ranks.
        let cfg = ParallelConfig::new_with_ep(2, 1, 8);
        assert_eq!(cfg.tensor_parallel_size, 2);
        assert_eq!(cfg.expert_parallel_size, 8);
        assert_eq!(cfg.world_size(), 2); // tp * pp only
    }

    #[test]
    fn test_parallel_config_ep_field_exists() {
        let cfg = ParallelConfig::no_parallelism();
        assert_eq!(cfg.expert_parallel_size, 1);
        assert!(!cfg.uses_expert_parallelism());
    }
}
