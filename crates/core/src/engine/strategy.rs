//! ExecutionStrategy trait for different execution modes.

use std::collections::HashMap;

use tokio::sync::mpsc;
use tracing::warn;

use crate::kv_cache::KVCacheManager;
use crate::request::RequestId;
use crate::scheduler::SchedulerOutput;
use crate::tokenizer::TokenizerWrapper;

use super::context::OwnedExecutionState;
use super::cuda_graph::CudaGraphDispatcher;
use super::types::{EngineCommand, EngineConfig};
use super::warmup::{WarmupConfig, WarmupStats};

/// Trait defining execution strategy for different inference modes.
///
/// This trait enables pluggable execution strategies:
/// - StandardExecution: Normal autoregressive decoding
/// - SpeculativeExecution: Speculative decoding with draft model
/// - Future: CUDA Graph execution, disaggregated prefill/decode, etc.
pub(crate) trait ExecutionStrategy: Send {
    /// Execute prefill for scheduled requests.
    fn execute_prefills(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        tokenizer: &TokenizerWrapper,
    );

    /// Execute decode for scheduled requests.
    fn execute_decodes(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        multi_step_count: usize,
        tokenizer: &TokenizerWrapper,
    );

    /// Handle preemption of requests.
    fn handle_preemptions(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
    );

    /// Called when a request is about to be removed as completed.
    /// Allows strategy to clean up any owned resources (e.g., draft cache).
    fn on_request_completed(&mut self, req_id: RequestId, state: &mut OwnedExecutionState) {
        // Default: no-op. Override in strategies that own additional resources.
        let _ = (req_id, state);
    }

    /// Perform warmup for this execution strategy.
    ///
    /// Warmup is called before the engine starts accepting requests to:
    /// 1. JIT compile CUDA kernels for configured batch sizes
    /// 2. Optionally capture CUDA graphs for fast replay
    ///
    /// The default implementation does nothing. Strategies should override
    /// this to perform model-specific warmup.
    fn warmup(
        &mut self,
        config: &WarmupConfig,
        kv_cache_mgr: &mut KVCacheManager,
        dispatcher: &mut CudaGraphDispatcher,
    ) -> WarmupStats {
        // Default: no warmup
        let _ = (config, kv_cache_mgr, dispatcher);
        WarmupStats::default()
    }
}

/// Unified engine loop that works with any ExecutionStrategy.
pub async fn run_engine_loop<S: ExecutionStrategy>(
    mut strategy: S,
    mut state: OwnedExecutionState,
    mut kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
    tokenizer: crate::tokenizer::TokenizerWrapper,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
) {
    use super::helpers::{
        build_generation_result, check_beam_finished, check_finished, finalize_beam_request,
        handle_command, is_beam_request, send_stream_token,
    };
    use super::types::ResponseChannel;
    use crate::request::FinishReason;

    // Initialize prefix cache on the KVCacheManager so all prefix operations
    // (match, register, release, eviction) are coordinated through the manager's
    // block pool. This ensures eviction can reclaim cached blocks when needed.
    if config.enable_prefix_caching && !kv_cache_mgr.has_prefix_cache() {
        kv_cache_mgr.enable_prefix_cache();
    }

    // Pause state: `paused` rejects new requests, `frozen` skips scheduling.
    let mut paused = false;
    let mut frozen = false;

    loop {
        // Phase 1: Drain incoming commands (non-blocking)
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut state.scheduler,
                        &mut state.requests,
                        config.block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                    ) {
                        return; // shutdown
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // Phase 2: If frozen (Keep mode), block-wait for commands only.
        // No scheduling occurs while frozen — requests stay in the queue.
        if frozen {
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut state.scheduler,
                        &mut state.requests,
                        config.block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 2b: If idle, block-wait for next command
        if state.scheduler.is_idle() {
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut state.scheduler,
                        &mut state.requests,
                        config.block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 3: Schedule
        let states_view: HashMap<RequestId, &crate::request::SequenceState> = state
            .requests
            .iter()
            .map(|(&id, r)| (id, &r.state))
            .collect();
        let output = state
            .scheduler
            .schedule(&states_view, kv_cache_mgr.num_free_blocks());

        // Phase 4: Handle preemptions
        strategy.handle_preemptions(&output, &mut state, &mut kv_cache_mgr);

        // Phase 5: Execute prefills
        strategy.execute_prefills(&output, &mut state, &mut kv_cache_mgr, &tokenizer);

        // Send stream tokens after prefill
        for schedule in &output.prefill_requests {
            send_stream_token(schedule.request_id, &tokenizer, &mut state.requests);
        }

        // Phase 6: Execute decodes
        strategy.execute_decodes(
            &output,
            &mut state,
            &mut kv_cache_mgr,
            config.multi_step_count,
            &tokenizer,
        );

        // Send stream tokens after decode
        for &req_id in &output.decode_requests {
            if state.requests.contains_key(&req_id) {
                send_stream_token(req_id, &tokenizer, &mut state.requests);
            }
        }

        // Phase 7: Check completion
        let mut finished = Vec::new();
        let scheduled_ids: Vec<RequestId> = output
            .prefill_requests
            .iter()
            .map(|s| s.request_id)
            .chain(output.decode_requests.iter().copied())
            .collect();
        for &req_id in &scheduled_ids {
            if let Some(req) = state.requests.get(&req_id) {
                let check = if is_beam_request(req) {
                    check_beam_finished(req)
                } else {
                    check_finished(&req.state, &tokenizer)
                };
                if let Some(check) = check {
                    finished.push((req_id, check));
                }
            }
        }

        // Phase 8: Finalize completed requests
        for (req_id, check) in finished {
            // Let strategy clean up owned resources (e.g., draft cache)
            strategy.on_request_completed(req_id, &mut state);

            let Some(mut req) = state.requests.remove(&req_id) else {
                continue;
            };
            state.scheduler.remove_request(req_id);

            // Beam search finalization: select best hypothesis and free beam block tables
            let (mut text, finish_reason) = if req.beam_state.is_some() {
                finalize_beam_request(&mut req, &mut kv_cache_mgr, &tokenizer)
            } else {
                let text = tokenizer
                    .decode(&req.state.generated_token_ids)
                    .unwrap_or_default();
                (text, check.reason)
            };

            let block_ids = req.state.block_table.release();
            if kv_cache_mgr.has_prefix_cache() {
                kv_cache_mgr.register_prefix(&req.state.prompt_token_ids, &block_ids);
                let to_free = kv_cache_mgr.release_prefix(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    if let Err(e) = kv_cache_mgr.free_blocks(&to_free) {
                        warn!(error = %e, blocks = ?to_free, "Failed to free uncached blocks");
                    }
                }

                if let Some(cache) = kv_cache_mgr.prefix_cache_mut() {
                    cache.record_request(
                        req.state.prompt_token_ids.len(),
                        req.state
                            .num_computed_tokens
                            .min(req.state.prompt_token_ids.len()),
                    );
                }
            } else if !block_ids.is_empty() {
                if let Err(e) = kv_cache_mgr.free_blocks(&block_ids) {
                    warn!(error = %e, blocks = ?block_ids, "Failed to free cache blocks");
                }
            }

            if check.trim_bytes > 0 {
                let new_len = text.len().saturating_sub(check.trim_bytes);
                text.truncate(new_len);
            }

            if finish_reason == FinishReason::Stop && !req.state.stop_token_ids.is_empty() {
                if let Some(&last) = req.state.generated_token_ids.last() {
                    if req.state.stop_token_ids.contains(&last) {
                        req.state.generated_token_ids.pop();
                    }
                }
            }

            let stop_reason = check.stop_reason;
            match req.response {
                ResponseChannel::Complete(tx) => {
                    let result = build_generation_result(
                        req_id,
                        text,
                        &mut req.state,
                        finish_reason,
                        stop_reason,
                    );
                    let _ = tx.send(Ok(result));
                }
                ResponseChannel::Stream(tx) => {
                    let _ = tx.try_send(super::types::StreamEvent::Done {
                        finish_reason,
                        generated_text: text,
                        stop_reason,
                    });
                }
            }
        }

        tokio::task::yield_now().await;
    }
}
