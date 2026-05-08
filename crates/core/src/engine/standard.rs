//! Standard (non-speculative) execution strategy.

use std::sync::Arc;

use tracing::{info, warn};

use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::request::{RequestId, RequestStatus};
use crate::scheduler::SchedulerOutput;
use crate::tokenizer::TokenizerWrapper;

use super::context::OwnedExecutionState;
use super::cuda_graph::CudaGraphDispatcher;
use super::cuda_graph_runner::CudaGraphRunner;
use super::helpers::{
    execute_batched_decode_with_graph, execute_beam_decode, execute_prefill,
    finish_request_with_error_deferred, is_beam_request, reclaim_sliding_window_blocks,
};
use super::model_forward::{DecodeSequenceMetadata, ModelForward};
use super::strategy::ExecutionStrategy;
use super::types::EngineError;
use super::warmup::{DummySequence, WarmupConfig, WarmupError, WarmupStats};

/// Standard autoregressive execution strategy.
///
/// This strategy executes prefill and decode phases in the standard way,
/// without speculative decoding. It supports multi-step decode and batched
/// decode for improved throughput.
pub struct StandardExecution<M: ModelForward> {
    model: M,
    /// CUDA graph runner for optimized decode execution.
    /// Wrapped in Arc for shared access during decode operations.
    graph_runner: Option<Arc<CudaGraphRunner>>,
    /// Sliding window size for KV cache reclamation.
    sliding_window: Option<usize>,
    /// Block size for sliding window reclamation math.
    block_size: usize,
}

impl<M: ModelForward> StandardExecution<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            graph_runner: None,
            sliding_window: None,
            block_size: 16,
        }
    }

    pub fn with_sliding_window(mut self, sliding_window: Option<usize>, block_size: usize) -> Self {
        self.sliding_window = sliding_window;
        self.block_size = block_size;
        self
    }

    // ─── Warmup Methods ───────────────────────────────────────────────────

    /// Create dummy sequences for warmup with allocated cache blocks.
    fn create_dummy_sequences(
        &self,
        batch_size: usize,
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Vec<DummySequence>, WarmupError> {
        let mut sequences = Vec::with_capacity(batch_size);

        // Allocate minimal blocks for each sequence (simulate decode with 1 token)
        for _ in 0..batch_size {
            let mut block_table = BlockTable::new(kv_cache_mgr.block_size());
            // Allocate 1 block for each sequence (enough for decode warmup)
            kv_cache_mgr
                .allocate_for_request(&mut block_table, kv_cache_mgr.block_size())
                .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;

            // Simulate that we've filled the first position
            block_table.advance(1);
            let seqlen_offset = 1;

            sequences.push(DummySequence::new(block_table, seqlen_offset));
        }

        Ok(sequences)
    }

    /// Run a dummy decode pass for JIT warmup.
    fn run_dummy_decode(
        &self,
        batch_size: usize,
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<(), WarmupError> {
        // Create dummy sequences with allocated blocks
        let mut dummy_seqs = self.create_dummy_sequences(batch_size, kv_cache_mgr)?;

        // Allocate one more slot for the decode token
        for seq in &mut dummy_seqs {
            kv_cache_mgr
                .allocate_for_request(&mut seq.block_table, 1)
                .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;
            // Update slot mapping for the new position
            seq.slot_mapping = seq.block_table.slot_mapping(seq.seqlen_offset, 1);
        }

        // Generate dummy input (zeros)
        let input = candle_core::Tensor::zeros(
            (batch_size, 1),
            candle_core::DType::U32,
            self.model.device(),
        )?;

        // Build sequence metadata
        let sequences: Vec<DecodeSequenceMetadata> = dummy_seqs
            .iter()
            .enumerate()
            .map(|(i, seq)| DecodeSequenceMetadata {
                request_id: i as u64,
                seqlen_offset: seq.seqlen_offset,
                block_ids: seq.block_ids.clone(),
                slot_mapping: seq.slot_mapping.clone(),
            })
            .collect();

        // Run forward pass (this triggers JIT compilation)
        let _logits = self
            .model
            .forward_decode_batch(&input, &sequences, kv_cache_mgr)?;

        // Clean up
        self.cleanup_dummy_sequences(dummy_seqs, kv_cache_mgr)?;

        Ok(())
    }

    /// Capture a CUDA graph for the given decode batch size.
    ///
    /// If a `CudaGraphRunner` is attached, this builds dummy sequences plus
    /// KV slots, calls `try_capture_decode_step` with a closure that runs
    /// the real `model.forward_decode_batch`, and registers the captured
    /// size with the dispatcher on success. If no runner is attached,
    /// falls back to plain JIT warmup so cold-shape kernels still compile.
    /// Returns `true` iff a CUDA graph was actually captured for this
    /// batch size; `false` means the call ran a plain JIT warmup forward
    /// (capture was unavailable, skipped, or failed cleanly).
    fn capture_decode_graph(
        &mut self,
        batch_size: usize,
        kv_cache_mgr: &mut KVCacheManager,
        dispatcher: &mut CudaGraphDispatcher,
    ) -> Result<bool, WarmupError> {
        let descriptor = super::cuda_graph::BatchDescriptor::for_decode(batch_size);

        // Without an attached runner there's nothing to capture — fall
        // back to JIT warmup and register the descriptor so the
        // dispatcher's mode logic stays consistent.
        if self.graph_runner.is_none() {
            self.run_dummy_decode(batch_size, kv_cache_mgr)?;
            dispatcher.register_valid_key(descriptor);
            return Ok(false);
        }

        // Build dummy sequences + slot for the decode token. Mirrors
        // run_dummy_decode but keeps the metadata around so the capture
        // closure can read it on every call.
        let mut dummy_seqs = self.create_dummy_sequences(batch_size, kv_cache_mgr)?;
        for seq in &mut dummy_seqs {
            kv_cache_mgr
                .allocate_for_request(&mut seq.block_table, 1)
                .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;
            seq.slot_mapping = seq.block_table.slot_mapping(seq.seqlen_offset, 1);
        }
        let sequences: Vec<DecodeSequenceMetadata> = dummy_seqs
            .iter()
            .enumerate()
            .map(|(i, seq)| DecodeSequenceMetadata {
                request_id: i as u64,
                seqlen_offset: seq.seqlen_offset,
                block_ids: seq.block_ids.clone(),
                slot_mapping: seq.slot_mapping.clone(),
            })
            .collect();

        // Borrow self's fields disjointly: the runner needs &mut, the
        // model needs only &, and kv_cache_mgr is already &mut from the
        // caller. The runner is wrapped in `Arc`; capture only succeeds
        // when nothing else holds a clone yet (which is guaranteed during
        // single-threaded warmup).
        let model = &self.model;
        let runner_arc = self.graph_runner.as_mut().unwrap();
        let captured = match Arc::get_mut(runner_arc) {
            Some(runner) => runner.try_capture_decode_step(batch_size, |input| {
                model.forward_decode_batch(input, &sequences, kv_cache_mgr)
            }),
            None => Ok(false),
        };

        // Always release the dummy sequences, regardless of whether
        // capture succeeded — captured graphs hold their own
        // input/output buffers, not these blocks.
        self.cleanup_dummy_sequences(dummy_seqs, kv_cache_mgr)?;

        match captured {
            Ok(true) => {
                dispatcher.register_valid_key(descriptor);
                Ok(true)
            }
            Ok(false) => {
                // Capture skipped or failed cleanly — fall back to JIT
                // warmup so the eager path is at least primed for this
                // batch size.
                self.run_dummy_decode(batch_size, kv_cache_mgr)?;
                Ok(false)
            }
            Err(e) => Err(WarmupError::CacheAllocation(format!(
                "graph capture for batch {batch_size}: {e}"
            ))),
        }
    }

    /// Initialize the CUDA graph runner for capture and replay.
    fn initialize_graph_runner(
        &mut self,
        config: &WarmupConfig,
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<(), WarmupError> {
        use super::cuda_graph::CudaGraphConfig;
        use candle_core::DType;

        // Get device from model
        let device = self.model.device().clone();

        // Skip if not CUDA device
        if !device.is_cuda() {
            return Ok(());
        }

        // Determine vocab size by running a single forward pass
        let vocab_size = {
            let mut dummy_seqs = self.create_dummy_sequences(1, kv_cache_mgr)?;

            for seq in &mut dummy_seqs {
                kv_cache_mgr
                    .allocate_for_request(&mut seq.block_table, 1)
                    .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;
                seq.slot_mapping = seq.block_table.slot_mapping(seq.seqlen_offset, 1);
            }

            let input = candle_core::Tensor::zeros((1, 1), DType::U32, &device)
                .map_err(WarmupError::from)?;

            let sequences: Vec<DecodeSequenceMetadata> = dummy_seqs
                .iter()
                .enumerate()
                .map(|(i, seq)| DecodeSequenceMetadata {
                    request_id: i as u64,
                    seqlen_offset: seq.seqlen_offset,
                    block_ids: seq.block_ids.clone(),
                    slot_mapping: seq.slot_mapping.clone(),
                })
                .collect();

            let logits = self
                .model
                .forward_decode_batch(&input, &sequences, kv_cache_mgr)
                .map_err(|e| WarmupError::CacheAllocation(format!("forward failed: {e}")))?;

            self.cleanup_dummy_sequences(dummy_seqs, kv_cache_mgr)?;

            // Vocab size is the last dimension of logits
            logits.dims().last().copied().unwrap_or(32000)
        };

        // Build CUDA graph config
        let graph_config = CudaGraphConfig {
            enabled: config.enable_graph_capture,
            capture_sizes: config.decode_batch_sizes.clone(),
            ..Default::default()
        };

        // Create runner
        let runner = CudaGraphRunner::new(graph_config, device, vocab_size, DType::F32);
        self.graph_runner = Some(Arc::new(runner));

        Ok(())
    }

    /// Clean up dummy sequences and free allocated blocks.
    /// Run a dummy prefill pass for JIT warmup at a given prompt length.
    ///
    /// Decode warmup compiles only the M=batch decode-attention path; the
    /// first real prefill request still pays JIT cost on the prefill
    /// attention path (causal mask, distinct kernel from paged-attn decode).
    /// This pre-warms the prefill path at typical prompt lengths.
    fn run_dummy_prefill(
        &self,
        prompt_len: usize,
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<(), WarmupError> {
        let mut block_table = BlockTable::new(kv_cache_mgr.block_size());
        kv_cache_mgr
            .allocate_for_request(&mut block_table, prompt_len)
            .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;

        let slot_mapping = block_table.slot_mapping(0, prompt_len);

        let input = candle_core::Tensor::zeros(
            (1, prompt_len),
            candle_core::DType::U32,
            self.model.device(),
        )?;

        let forward_result =
            self.model
                .forward(&input, 0, kv_cache_mgr, &block_table, &slot_mapping);

        // Free regardless of forward result so a single failure doesn't
        // leak blocks across the warmup loop.
        let block_ids = block_table.release();
        if !block_ids.is_empty() {
            if let Err(e) = kv_cache_mgr.free_blocks(&block_ids) {
                tracing::warn!(error = %e, prompt_len, "Failed to free dummy prefill blocks");
            }
        }

        forward_result
            .map(|_| ())
            .map_err(|e| WarmupError::CacheAllocation(e.to_string()))
    }

    fn cleanup_dummy_sequences(
        &self,
        sequences: Vec<DummySequence>,
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<(), WarmupError> {
        for seq in sequences {
            let mut block_table = seq.release();
            let block_ids = block_table.release();
            if !block_ids.is_empty() {
                kv_cache_mgr
                    .free_blocks(&block_ids)
                    .map_err(|e| WarmupError::CacheAllocation(e.to_string()))?;
            }
        }
        Ok(())
    }
}

impl<M: ModelForward> ExecutionStrategy for StandardExecution<M> {
    fn execute_prefills(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        tokenizer: &TokenizerWrapper,
    ) {
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;
            if let Err(e) = execute_prefill(
                req_id,
                schedule.chunk_size,
                &self.model,
                kv_cache_mgr,
                &mut state.requests,
                tokenizer,
            ) {
                if let Some(id) =
                    finish_request_with_error_deferred(req_id, e, &mut state.requests, kv_cache_mgr)
                {
                    state.errored_ids.push(id);
                }
            }
        }
    }

    fn execute_decodes(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        multi_step_count: usize,
        _tokenizer: &TokenizerWrapper,
    ) {
        if output.decode_requests.is_empty() {
            return;
        }

        // Partition into regular and beam search requests.
        // Beam requests decode individually (each internally batches its own beams).
        let mut regular_ids = Vec::new();
        let mut beam_ids = Vec::new();

        for &req_id in &output.decode_requests {
            if let Some(req) = state.requests.get(&req_id) {
                if is_beam_request(req) {
                    beam_ids.push(req_id);
                } else {
                    regular_ids.push(req_id);
                }
            }
        }

        // Execute beam search decode for each beam request
        for req_id in beam_ids {
            if let Err(e) = execute_beam_decode(
                req_id,
                &self.model,
                kv_cache_mgr,
                &mut state.requests,
                _tokenizer,
            ) {
                if let Some(id) =
                    finish_request_with_error_deferred(req_id, e, &mut state.requests, kv_cache_mgr)
                {
                    state.errored_ids.push(id);
                }
            }
        }

        // Execute regular batched decode
        if !regular_ids.is_empty() {
            // Reclaim blocks outside sliding window before allocation
            reclaim_sliding_window_blocks(
                &regular_ids,
                self.sliding_window,
                self.block_size,
                kv_cache_mgr,
                &mut state.requests,
            );

            let num_steps = multi_step_count.max(1);
            let mut active_decode_ids = regular_ids;
            // Substep 2.1 / ADR 0017: GPU tensor pass-through across
            // multi-step iterations. Carries the previous step's
            // sampler-output tensor so the next step can use it as
            // `input_ids` directly (no host vec → tensor round-trip).
            // Only valid when the active set didn't shrink between
            // steps; on any size change we drop it and the helper
            // falls back to the host path.
            let mut prev_sampled: Option<candle_core::Tensor> = None;

            for _step in 0..num_steps {
                if active_decode_ids.is_empty() {
                    break;
                }
                let active_size_before = active_decode_ids.len();

                let mut step_preempted: Vec<RequestId> = Vec::new();
                let (failed, next_sampled) = execute_batched_decode_with_graph(
                    &active_decode_ids,
                    prev_sampled.as_ref(),
                    &self.model,
                    kv_cache_mgr,
                    &mut state.requests,
                    Some(&state.cuda_graph_dispatcher),
                    self.graph_runner.as_ref(),
                    &mut step_preempted,
                );

                for (req_id, err_msg) in failed {
                    active_decode_ids.retain(|&id| id != req_id);
                    if let Some(id) = finish_request_with_error_deferred(
                        req_id,
                        EngineError::Model(format!("batched decode failed: {err_msg}")),
                        &mut state.requests,
                        kv_cache_mgr,
                    ) {
                        state.errored_ids.push(id);
                    }
                }

                // Preempted requests leave the active multi-step set this
                // step; the engine async loop will re-admit them via
                // `Scheduler::move_to_waiting` after the blocking task
                // returns. Keeping them here would have the next
                // multi-step iteration try to decode a Preempted state.
                for req_id in &step_preempted {
                    active_decode_ids.retain(|&id| id != *req_id);
                }
                state.preempted_ids.append(&mut step_preempted);

                // Remove sequences that finished mid-step
                active_decode_ids.retain(|&id| {
                    state
                        .requests
                        .get(&id)
                        .map(|r| {
                            let s = &r.state;
                            let last = s.generated_token_ids.last().copied();
                            let eos = last.map(|t| t == s.eos_token_id).unwrap_or(false);
                            let stop_token =
                                last.map(|t| s.stop_token_ids.contains(&t)).unwrap_or(false);
                            let max_len = s.num_generated() >= s.max_new_tokens;
                            !eos && !stop_token && !max_len
                        })
                        .unwrap_or(false)
                });

                // Carry the GPU tensor forward iff the active set is
                // unchanged AND the helper produced one (GPU sampler
                // path; CPU fallback returns None). On any divergence
                // we drop it — next step rebuilds input_ids from host.
                prev_sampled = if active_decode_ids.len() == active_size_before {
                    next_sampled
                } else {
                    None
                };
            }
        }
    }

    fn handle_preemptions(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
    ) {
        for &req_id in &output.preempted_requests {
            let Some(req) = state.requests.get_mut(&req_id) else {
                // Request already removed, skip
                continue;
            };
            if let Err(e) = kv_cache_mgr.free_request(&mut req.state.block_table) {
                warn!(error = %e, request_id = req_id, "Failed to free request cache blocks during preemption");
            }
            req.state.status = RequestStatus::Preempted;
            req.state.generated_token_ids.clear();
            req.state.num_computed_tokens = 0;
            req.state.seqlen_offset = 0;
        }
    }

    fn warmup(
        &mut self,
        config: &WarmupConfig,
        kv_cache_mgr: &mut KVCacheManager,
        dispatcher: &mut CudaGraphDispatcher,
    ) -> WarmupStats {
        if !config.needs_warmup() {
            return WarmupStats::skipped();
        }

        let start = std::time::Instant::now();
        let mut stats = WarmupStats::default();

        if config.show_progress {
            info!(
                batch_sizes = ?config.decode_batch_sizes,
                jit = config.enable_jit_warmup,
                graphs = config.enable_graph_capture,
                "Starting warmup"
            );
        }

        // Initialize CUDA graph runner if graph capture is enabled
        if config.enable_graph_capture && dispatcher.is_enabled() {
            if let Err(e) = self.initialize_graph_runner(config, kv_cache_mgr) {
                warn!(error = %e, "Failed to initialize graph runner");
                stats.errors.push(format!("graph runner init: {e}"));
            }
        }

        // Process batch sizes in descending order (fail fast on OOM for large batches)
        let mut sorted_sizes = config.decode_batch_sizes.clone();
        sorted_sizes.sort_by(|a, b| b.cmp(a));

        for batch_size in sorted_sizes {
            if batch_size == 0 {
                stats
                    .errors
                    .push("batch_size=0: skipped (invalid)".to_string());
                continue;
            }

            // Phase 1: JIT warmup
            if config.enable_jit_warmup {
                match self.run_dummy_decode(batch_size, kv_cache_mgr) {
                    Ok(()) => {
                        stats.jit_warmed_sizes.push(batch_size);
                        if config.show_progress {
                            info!(batch_size, "JIT warmup complete");
                        }
                    }
                    Err(e) => {
                        warn!(batch_size, error = %e, "JIT warmup failed");
                        stats
                            .errors
                            .push(format!("jit batch_size={batch_size}: {e}"));
                        // Continue with other sizes
                    }
                }
            }

            // Phase 2: CUDA graph registration
            if config.enable_graph_capture && dispatcher.is_enabled() {
                match self.capture_decode_graph(batch_size, kv_cache_mgr, dispatcher) {
                    Ok(true) => {
                        stats.graphs_captured += 1;
                        if config.show_progress {
                            info!(batch_size, "CUDA graph registered");
                        }
                    }
                    Ok(false) => {
                        stats.graphs_failed += 1;
                        if config.show_progress {
                            info!(
                                batch_size,
                                "CUDA graph capture skipped (JIT-only fallback) — \
                                 a kernel inside the forward pass is not capturable; \
                                 this size will run eager"
                            );
                        }
                    }
                    Err(e) => {
                        stats.graphs_failed += 1;
                        warn!(batch_size, error = %e, "CUDA graph registration failed");
                        stats
                            .errors
                            .push(format!("graph batch_size={batch_size}: {e}"));
                    }
                }
            }
        }

        // Phase 3: prefill prompt-length warmup. Compiles the prefill
        // attention path (causal mask, separate kernel from paged-attn
        // decode) for typical prompt lengths so the first real request
        // doesn't pay full JIT cost. Empty by default; opt-in via
        // `WarmupConfig::with_prefill_lens(...)`.
        if config.enable_jit_warmup && !config.prefill_prompt_lens.is_empty() {
            // Process in ascending order so a smaller prompt JIT-compiles the
            // attention shape before larger prompts that need more VRAM.
            let mut prefill_lens = config.prefill_prompt_lens.clone();
            prefill_lens.sort();
            for prompt_len in prefill_lens {
                if prompt_len == 0 {
                    stats
                        .errors
                        .push("prefill prompt_len=0: skipped (invalid)".to_string());
                    continue;
                }
                match self.run_dummy_prefill(prompt_len, kv_cache_mgr) {
                    Ok(()) => {
                        stats.prefill_warmed_lens.push(prompt_len);
                        if config.show_progress {
                            info!(prompt_len, "Prefill JIT warmup complete");
                        }
                    }
                    Err(e) => {
                        warn!(prompt_len, error = %e, "Prefill JIT warmup failed");
                        stats
                            .errors
                            .push(format!("prefill prompt_len={prompt_len}: {e}"));
                    }
                }
            }
        }

        // Once every batch size has been processed, mark the runner as
        // warmed up so dispatch can route to captured graphs.
        if let Some(runner_arc) = self.graph_runner.as_mut() {
            if let Some(runner) = Arc::get_mut(runner_arc) {
                runner.mark_warmed_up();
                tracing::info!(
                    target: "vllm_core::cuda_graph",
                    "graph runner marked warmed_up — replay path now active"
                );
            } else {
                tracing::warn!(
                    target: "vllm_core::cuda_graph",
                    "graph runner Arc has > 1 strong refs — mark_warmed_up SKIPPED. \
                     Replay will never fire; eager fallback in effect."
                );
            }
        }

        stats.total_time_ms = start.elapsed().as_millis() as u64;

        // Sort jit_warmed_sizes for consistent output
        stats.jit_warmed_sizes.sort();

        if config.show_progress {
            info!(
                jit_sizes = ?stats.jit_warmed_sizes,
                graphs_captured = stats.graphs_captured,
                graphs_failed = stats.graphs_failed,
                time_ms = stats.total_time_ms,
                runner_enabled = self.graph_runner.is_some(),
                "Warmup complete"
            );
        }

        stats
    }
}
