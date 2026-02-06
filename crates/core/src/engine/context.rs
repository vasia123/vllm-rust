//! Execution context and request state for the engine.

use std::collections::HashMap;
use std::sync::Arc;

use crate::kv_cache::BlockTable;
use crate::request::{RequestId, SequenceState};
use crate::scheduler::Scheduler;

use super::cuda_graph::CudaGraphDispatcher;
use super::types::{EngineConfig, ResponseChannel};

/// Per-request state for the draft model's KV cache during speculative decoding.
pub(crate) struct DraftState {
    pub block_table: BlockTable,
    pub seqlen_offset: usize,
}

/// Active request being processed by the engine.
pub(crate) struct ActiveRequest {
    pub state: SequenceState,
    pub response: ResponseChannel,
    pub num_streamed_tokens: usize,
    pub streamed_text_len: usize,
    pub draft_state: Option<DraftState>,
}

/// Owned execution state for strategies.
///
/// Prefix caching is managed by `KVCacheManager` (not stored here) so that
/// all prefix operations (match, register, release, eviction) are coordinated
/// through the manager's block pool.
pub(crate) struct OwnedExecutionState {
    pub scheduler: Scheduler,
    pub requests: HashMap<RequestId, ActiveRequest>,
    pub next_id: RequestId,
    /// CUDA Graph dispatcher for graph capture/replay
    pub cuda_graph_dispatcher: Arc<std::sync::RwLock<CudaGraphDispatcher>>,
}

impl OwnedExecutionState {
    pub fn new(config: &EngineConfig) -> Self {
        let scheduler = Scheduler::new(config.scheduler_config);

        let cuda_graph_dispatcher = Arc::new(std::sync::RwLock::new(CudaGraphDispatcher::new(
            config.cuda_graph_config.clone(),
        )));

        Self {
            scheduler,
            requests: HashMap::new(),
            next_id: 0,
            cuda_graph_dispatcher,
        }
    }

    /// Check if CUDA graphs are enabled.
    #[allow(dead_code)] // Will be used in warmup/stats
    pub fn cuda_graphs_enabled(&self) -> bool {
        self.cuda_graph_dispatcher
            .read()
            .map(|guard| guard.is_enabled())
            .unwrap_or(false)
    }
}
