//! ExecutionStrategy trait for different execution modes.

use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;
use tracing::warn;

use crate::kv_cache::KVCacheManager;
use crate::request::RequestId;
use crate::scheduler::{ScheduleDecision, Scheduler, SchedulerOutput};
use crate::tokenizer::TokenizerWrapper;

use super::context::OwnedExecutionState;
use super::cuda_graph::CudaGraphDispatcher;
use super::types::{EngineCommand, EngineConfig, SpecDecodingStats};
use super::warmup::{WarmupConfig, WarmupStats};

/// Result of a single engine step: IDs of completed and errored requests.
///
/// Returned by `execute_engine_step` so the async loop can perform
/// deferred `scheduler.remove_request()` calls after reclaiming ownership.
pub(crate) struct StepResult {
    pub completed: Vec<RequestId>,
    pub errored: Vec<RequestId>,
}

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

    /// Returns accumulated speculative decoding statistics, if applicable.
    fn spec_decode_stats(&self) -> Option<SpecDecodingStats> {
        None
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

// ─── Engine Step Execution ───────────────────────────────────────────────

/// Execute one complete engine step: preemptions, prefills, decodes,
/// stream token delivery, completion checking, and request finalization.
///
/// Returns a [`StepResult`] containing the IDs of completed and errored
/// requests. The caller (async loop) is responsible for calling
/// `scheduler.remove_request()` for each ID after reclaiming ownership.
///
/// This function runs inside `spawn_blocking` so the tokio async runtime
/// stays free during GPU work. All channel sends within are non-blocking
/// (`send` for oneshot, `try_send` for mpsc).
fn execute_engine_step<S: ExecutionStrategy>(
    strategy: &mut S,
    state: &mut OwnedExecutionState,
    kv_cache_mgr: &mut KVCacheManager,
    tokenizer: &TokenizerWrapper,
    output: &SchedulerOutput,
    multi_step_count: usize,
) -> StepResult {
    use super::helpers::{
        build_generation_result, check_beam_finished, check_finished, finalize_beam_request,
        is_beam_request, send_stream_token,
    };
    use super::types::ResponseChannel;
    use crate::request::FinishReason;

    // Clear any stale errored IDs from a previous step
    state.errored_ids.clear();

    let mut step_result = StepResult {
        completed: Vec::new(),
        errored: Vec::new(),
    };

    // Handle preemptions
    strategy.handle_preemptions(output, state, kv_cache_mgr);

    // Execute prefills
    strategy.execute_prefills(output, state, kv_cache_mgr, tokenizer);

    // Send stream tokens after prefill
    for schedule in &output.prefill_requests {
        send_stream_token(schedule.request_id, tokenizer, &mut state.requests);
    }

    // Execute decodes — collect errored request IDs
    strategy.execute_decodes(output, state, kv_cache_mgr, multi_step_count, tokenizer);

    // Send stream tokens after decode
    for &req_id in &output.decode_requests {
        if state.requests.contains_key(&req_id) {
            send_stream_token(req_id, tokenizer, &mut state.requests);
        }
    }

    // Check completion
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
                check_finished(&req.state, tokenizer)
            };
            if let Some(check) = check {
                finished.push((req_id, check));
            }
        }
    }

    // Finalize completed requests
    for (req_id, check) in finished {
        strategy.on_request_completed(req_id, state);

        let Some(mut req) = state.requests.remove(&req_id) else {
            continue;
        };
        step_result.completed.push(req_id);

        let (mut text, finish_reason) = if req.beam_state.is_some() {
            finalize_beam_request(&mut req, kv_cache_mgr, tokenizer)
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

    // Drain errored IDs collected by strategy execution into the result
    step_result.errored.append(&mut state.errored_ids);

    step_result
}

// ─── Optimistic Scheduling Helpers ───────────────────────────────────────

/// Check whether a pre-computed schedule decision is still valid after
/// GPU execution completes.
///
/// A pre-schedule is invalidated if:
/// - Any new commands arrived while the GPU was running (could have added,
///   removed, or aborted requests that affect scheduling)
/// - Any request that the pre-schedule references was completed or errored
///   during execution
fn is_optimistic_schedule_valid(
    decision: &ScheduleDecision,
    result: &StepResult,
    num_new_cmds: usize,
) -> bool {
    // New commands may have changed scheduler state (adds, aborts, pauses)
    if num_new_cmds > 0 {
        return false;
    }

    // If nothing finished, the pre-schedule is still valid
    if result.completed.is_empty() && result.errored.is_empty() {
        return true;
    }

    // Check that no scheduled request was completed or errored
    let finished: HashSet<RequestId> = result
        .completed
        .iter()
        .chain(&result.errored)
        .copied()
        .collect();

    let scheduled: HashSet<RequestId> = decision
        .output
        .decode_requests
        .iter()
        .copied()
        .chain(
            decision
                .output
                .prefill_requests
                .iter()
                .map(|p| p.request_id),
        )
        .collect();

    scheduled.is_disjoint(&finished)
}

/// Estimate blocks that will be allocated during the current step.
///
/// Sums `blocks_needed_for_step` for all scheduled requests (prefill and
/// decode). This is conservative: the actual allocation may be lower if
/// some requests complete mid-step, and completions will free additional
/// blocks. Using this as a deduction from `num_free_blocks` for the
/// next-step pre-schedule guarantees we never overcommit.
fn compute_blocks_allocated(
    output: &SchedulerOutput,
    requests: &HashMap<RequestId, super::context::ActiveRequest>,
) -> usize {
    let mut total = 0;
    for schedule in &output.prefill_requests {
        if let Some(req) = requests.get(&schedule.request_id) {
            total += req.state.block_table.blocks_needed(schedule.chunk_size);
        }
    }
    for &req_id in &output.decode_requests {
        if let Some(req) = requests.get(&req_id) {
            total += req.state.blocks_needed_for_step();
        }
    }
    total
}

// ─── Engine Loop ─────────────────────────────────────────────────────────

/// Pipelined engine loop that works with any ExecutionStrategy.
///
/// Model execution (prefills + decodes) runs in `tokio::task::spawn_blocking`
/// to free the async runtime during GPU work. Incoming commands are buffered
/// via `tokio::select!` while the blocking task runs and processed once
/// execution completes. See ADR-0006 for the design rationale.
///
/// The `Scheduler` lives here in the async loop rather than inside
/// `OwnedExecutionState`. This separation enables optimistic pre-scheduling:
/// while `state` moves into `spawn_blocking`, the scheduler stays behind for
/// pre-computing the next step's schedule.
pub async fn run_engine_loop<S: ExecutionStrategy + 'static>(
    mut strategy: S,
    mut state: OwnedExecutionState,
    mut kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
    mut tokenizer: crate::tokenizer::TokenizerWrapper,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
) {
    use super::helpers::handle_command;

    if config.enable_prefix_caching && !kv_cache_mgr.has_prefix_cache() {
        kv_cache_mgr.enable_prefix_cache();
    }

    let multi_step_count = config.multi_step_count;
    let block_size = config.block_size;
    let optimistic_scheduling = config.enable_optimistic_scheduling;

    // Scheduler lives in the async loop, separate from OwnedExecutionState.
    let mut scheduler = Scheduler::new(config.scheduler_config);

    // Pause state: `paused` rejects new requests, `frozen` skips scheduling.
    let mut paused = false;
    let mut frozen = false;

    // Optimistic pre-scheduling: holds the pre-computed schedule for the
    // next step. Validated after GPU execution returns; discarded if
    // invalidated by completions or new commands.
    let mut pending_decision: Option<ScheduleDecision> = None;

    loop {
        // Phase 1: Drain incoming commands (non-blocking)
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    // Any command invalidates the pre-schedule
                    pending_decision = None;
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut state.requests,
                        block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                        strategy.spec_decode_stats(),
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
            pending_decision = None;
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut state.requests,
                        block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                        strategy.spec_decode_stats(),
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 2b: If idle, block-wait for next command
        if scheduler.is_idle() {
            pending_decision = None;
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut state.next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut state.requests,
                        block_size,
                        &mut kv_cache_mgr,
                        &mut paused,
                        &mut frozen,
                        strategy.spec_decode_stats(),
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 3: Use cached pre-schedule if valid, else compute fresh.
        // The `pending_decision` was validated at the end of the previous
        // iteration; if it survived, we skip scheduling entirely.
        let decision = match pending_decision.take() {
            Some(d) => d,
            None => {
                let states_view: HashMap<RequestId, &crate::request::SequenceState> = state
                    .requests
                    .iter()
                    .map(|(&id, r)| (id, &r.state))
                    .collect();
                scheduler.compute_schedule(&states_view, kv_cache_mgr.num_free_blocks())
            }
        };
        scheduler.apply_schedule(&decision);
        let output = decision.output;

        // Phase 3b: Pre-schedule N+1 with optimistic free blocks.
        // After apply_schedule, the scheduler's running/waiting sets reflect
        // the current step. We estimate blocks allocated this step and
        // subtract from free blocks to get a conservative lower bound.
        let next_decision = if optimistic_scheduling {
            let blocks_allocated = compute_blocks_allocated(&output, &state.requests);
            let optimistic_free = kv_cache_mgr
                .num_free_blocks()
                .saturating_sub(blocks_allocated);
            let next_states_view: HashMap<RequestId, &crate::request::SequenceState> = state
                .requests
                .iter()
                .map(|(&id, r)| (id, &r.state))
                .collect();
            Some(scheduler.compute_schedule(&next_states_view, optimistic_free))
        } else {
            None
        };

        // Phase 4: Execute in spawn_blocking, buffer commands concurrently.
        //
        // GPU-facing state (strategy, execution state, KV cache, tokenizer)
        // moves into the blocking thread pool. The scheduler stays in the async
        // loop. When execution returns, we reclaim ownership and apply deferred
        // scheduler removals from the StepResult.
        let exec_future = tokio::task::spawn_blocking(move || {
            let step_result = execute_engine_step(
                &mut strategy,
                &mut state,
                &mut kv_cache_mgr,
                &tokenizer,
                &output,
                multi_step_count,
            );
            (strategy, state, kv_cache_mgr, tokenizer, step_result)
        });

        let mut pending_cmds: Vec<EngineCommand> = Vec::new();
        tokio::pin!(exec_future);

        loop {
            tokio::select! {
                biased;

                result = &mut exec_future => {
                    let (s, st, kv, tok, step_result) = result.expect("engine execution task panicked");
                    strategy = s;
                    state = st;
                    kv_cache_mgr = kv;
                    tokenizer = tok;

                    // Validate the pre-schedule before processing completions
                    // and buffered commands (which will invalidate it).
                    if let Some(next) = next_decision {
                        if is_optimistic_schedule_valid(
                            &next,
                            &step_result,
                            pending_cmds.len(),
                        ) {
                            pending_decision = Some(next);
                        }
                    }

                    // Deferred scheduler removal: apply completions and errors
                    for req_id in step_result.completed {
                        scheduler.remove_request(req_id);
                    }
                    for req_id in step_result.errored {
                        scheduler.remove_request(req_id);
                    }

                    break;
                }

                // Buffer incoming commands while GPU runs. If the channel
                // closes (all senders dropped), the pattern fails and this
                // branch is disabled — select then waits solely on exec_future.
                Some(cmd) = cmd_rx.recv() => {
                    pending_cmds.push(cmd);
                }
            }
        }

        // Phase 5: Process commands that arrived during execution.
        // Any processed command invalidates the pre-schedule (already handled
        // by the validation above — pending_cmds.len() > 0 → invalid).
        for cmd in pending_cmds {
            if handle_command(
                cmd,
                &mut state.next_id,
                &tokenizer,
                &mut scheduler,
                &mut state.requests,
                block_size,
                &mut kv_cache_mgr,
                &mut paused,
                &mut frozen,
                strategy.spec_decode_stats(),
            ) {
                return; // shutdown
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::{PrefillSchedule, SchedulerOutput};

    fn empty_step_result() -> StepResult {
        StepResult {
            completed: Vec::new(),
            errored: Vec::new(),
        }
    }

    fn make_decision(
        decode_ids: Vec<RequestId>,
        prefill_ids: Vec<(RequestId, usize)>,
    ) -> ScheduleDecision {
        ScheduleDecision {
            output: SchedulerOutput {
                decode_requests: decode_ids,
                prefill_requests: prefill_ids
                    .into_iter()
                    .map(|(id, chunk)| PrefillSchedule {
                        request_id: id,
                        chunk_size: chunk,
                    })
                    .collect(),
                preempted_requests: Vec::new(),
            },
            newly_admitted: Vec::new(),
        }
    }

    #[test]
    fn validation_accepts_when_nothing_changed() {
        let decision = make_decision(vec![1, 2, 3], vec![]);
        let result = empty_step_result();

        assert!(is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn validation_rejects_when_new_commands_arrived() {
        let decision = make_decision(vec![1, 2, 3], vec![]);
        let result = empty_step_result();

        assert!(!is_optimistic_schedule_valid(&decision, &result, 1));
    }

    #[test]
    fn validation_rejects_when_scheduled_decode_completed() {
        let decision = make_decision(vec![1, 2, 3], vec![]);
        let result = StepResult {
            completed: vec![2],
            errored: Vec::new(),
        };

        assert!(!is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn validation_rejects_when_scheduled_prefill_completed() {
        let decision = make_decision(vec![], vec![(5, 128)]);
        let result = StepResult {
            completed: vec![5],
            errored: Vec::new(),
        };

        assert!(!is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn validation_rejects_when_scheduled_request_errored() {
        let decision = make_decision(vec![1, 2], vec![]);
        let result = StepResult {
            completed: Vec::new(),
            errored: vec![1],
        };

        assert!(!is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn validation_accepts_when_unrelated_request_completed() {
        // Request 99 completed but was not in the pre-schedule
        let decision = make_decision(vec![1, 2], vec![(3, 64)]);
        let result = StepResult {
            completed: vec![99],
            errored: Vec::new(),
        };

        assert!(is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn validation_accepts_empty_schedule() {
        // Pre-schedule has nothing (idle step) — always valid if no cmds
        let decision = make_decision(vec![], vec![]);
        let result = StepResult {
            completed: vec![1, 2, 3],
            errored: Vec::new(),
        };

        assert!(is_optimistic_schedule_valid(&decision, &result, 0));
    }

    #[test]
    fn compute_blocks_allocated_decode_only() {
        use crate::request::SequenceState;

        let mut requests = HashMap::new();

        // Request 0: 16 tokens in block of 16 — decode needs 1 new block
        let mut state = SequenceState::new(0, vec![0; 16], 64, 0, 16, 0);
        state.status = crate::request::RequestStatus::Decoding;
        state.block_table.append_blocks(&[0]);
        state.block_table.advance(16);
        state.seqlen_offset = 16;
        let req = super::super::context::ActiveRequest {
            state,
            response: super::super::types::ResponseChannel::Complete(
                tokio::sync::oneshot::channel().0,
            ),
            num_streamed_tokens: 0,
            streamed_text_len: 0,
            beam_state: None,
        };
        requests.insert(0, req);

        // Request 1: 5 tokens in block of 16 — decode fits, 0 new blocks
        let mut state = SequenceState::new(1, vec![0; 5], 64, 0, 16, 1);
        state.status = crate::request::RequestStatus::Decoding;
        state.block_table.append_blocks(&[1]);
        state.block_table.advance(5);
        state.seqlen_offset = 5;
        let req = super::super::context::ActiveRequest {
            state,
            response: super::super::types::ResponseChannel::Complete(
                tokio::sync::oneshot::channel().0,
            ),
            num_streamed_tokens: 0,
            streamed_text_len: 0,
            beam_state: None,
        };
        requests.insert(1, req);

        let output = SchedulerOutput {
            decode_requests: vec![0, 1],
            prefill_requests: Vec::new(),
            preempted_requests: Vec::new(),
        };

        // req 0 needs 1 block, req 1 needs 0 → total 1
        assert_eq!(compute_blocks_allocated(&output, &requests), 1);
    }

    #[test]
    fn compute_blocks_allocated_prefill() {
        use crate::request::SequenceState;

        let mut requests = HashMap::new();

        // Request 0: new prefill, 20 tokens, block_size=16 → needs 2 blocks
        let state = SequenceState::new(0, vec![0; 20], 64, 0, 16, 0);
        let req = super::super::context::ActiveRequest {
            state,
            response: super::super::types::ResponseChannel::Complete(
                tokio::sync::oneshot::channel().0,
            ),
            num_streamed_tokens: 0,
            streamed_text_len: 0,
            beam_state: None,
        };
        requests.insert(0, req);

        let output = SchedulerOutput {
            decode_requests: Vec::new(),
            prefill_requests: vec![PrefillSchedule {
                request_id: 0,
                chunk_size: 20,
            }],
            preempted_requests: Vec::new(),
        };

        assert_eq!(compute_blocks_allocated(&output, &requests), 2);
    }
}
