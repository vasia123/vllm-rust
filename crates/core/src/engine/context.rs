//! Execution context and request state for the engine.

use std::collections::HashMap;
use std::sync::Arc;

use crate::kv_cache::BlockTable;
use crate::request::{RequestId, SequenceState};
use crate::sampling::BeamSearchState;

use super::cuda_graph::CudaGraphDispatcher;
use super::types::{EngineConfig, ResponseChannel};

/// Per-request state for beam search decoding.
///
/// Each beam request internally manages `beam_width` beams. The `search` field
/// tracks hypotheses and performs beam expansion. Each beam has its own block
/// table and sequence length offset for independent KV cache management.
pub(crate) struct BeamState {
    /// Beam search algorithm state (hypotheses, expansion, completion).
    pub search: BeamSearchState,
    /// Per-beam block tables for KV cache management.
    pub beam_block_tables: Vec<BlockTable>,
    /// Per-beam sequence length offsets (prompt_len + generated tokens for that beam).
    pub beam_seqlen_offsets: Vec<usize>,
}

/// Active request being processed by the engine.
pub(crate) struct ActiveRequest {
    pub state: SequenceState,
    pub response: ResponseChannel,
    pub num_streamed_tokens: usize,
    pub streamed_text_len: usize,
    /// Beam search state. When `Some`, this request uses beam search decoding
    /// instead of standard autoregressive sampling.
    pub beam_state: Option<BeamState>,
}

/// Owned execution state for strategies.
///
/// The `Scheduler` is intentionally NOT stored here — it lives in the async
/// engine loop so that optimistic pre-scheduling can run before `spawn_blocking`
/// moves this struct into the blocking thread pool.
///
/// Prefix caching is managed by `KVCacheManager` (not stored here) so that
/// all prefix operations (match, register, release, eviction) are coordinated
/// through the manager's block pool.
pub(crate) struct OwnedExecutionState {
    pub requests: HashMap<RequestId, ActiveRequest>,
    pub next_id: RequestId,
    /// CUDA Graph dispatcher for graph capture/replay
    pub cuda_graph_dispatcher: Arc<std::sync::RwLock<CudaGraphDispatcher>>,
    /// Request IDs that errored during strategy execution and need deferred
    /// `scheduler.remove_request()` after the blocking task returns.
    pub errored_ids: Vec<RequestId>,
}

impl OwnedExecutionState {
    pub fn new(config: &EngineConfig) -> Self {
        let cuda_graph_dispatcher = Arc::new(std::sync::RwLock::new(CudaGraphDispatcher::new(
            config.cuda_graph_config.clone(),
        )));

        Self {
            requests: HashMap::new(),
            next_id: 0,
            cuda_graph_dispatcher,
            errored_ids: Vec::new(),
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
