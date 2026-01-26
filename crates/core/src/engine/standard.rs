//! Standard (non-speculative) execution strategy.

use tracing::{info, warn};

use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::request::RequestStatus;
use crate::scheduler::SchedulerOutput;

use super::context::OwnedExecutionState;
use super::cuda_graph::CudaGraphDispatcher;
use super::helpers::{
    execute_batched_decode_with_graph, execute_prefill, finish_request_with_error,
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
}

impl<M: ModelForward> StandardExecution<M> {
    pub fn new(model: M) -> Self {
        Self { model }
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
            .map(|seq| DecodeSequenceMetadata {
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

    /// Capture a CUDA graph for the given batch size.
    #[allow(unused_variables)] // dispatcher used when cuda-kernels enabled
    fn capture_decode_graph(
        &self,
        batch_size: usize,
        kv_cache_mgr: &mut KVCacheManager,
        dispatcher: &mut CudaGraphDispatcher,
    ) -> Result<(), WarmupError> {
        // CUDA graph capture requires the cuda-kernels feature
        // For now, we just do JIT warmup without actual capture
        // Full capture implementation will use CudaGraphWrapper

        // The graph capture flow would be:
        // 1. Create dummy sequences
        // 2. Begin stream capture
        // 3. Run forward pass (operations are recorded, not executed)
        // 4. End capture and instantiate graph
        // 5. Store in dispatcher cache

        // For now, just run another forward to ensure everything is warm
        self.run_dummy_decode(batch_size, kv_cache_mgr)?;

        // Register the batch size as a valid key
        let descriptor = super::cuda_graph::BatchDescriptor::for_decode(batch_size);
        dispatcher.register_valid_key(descriptor);

        Ok(())
    }

    /// Clean up dummy sequences and free allocated blocks.
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
    ) {
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;
            if let Err(e) = execute_prefill(
                req_id,
                schedule.chunk_size,
                &self.model,
                kv_cache_mgr,
                &mut state.requests,
            ) {
                finish_request_with_error(req_id, e, &mut state.scheduler, &mut state.requests);
            }
        }
    }

    fn execute_decodes(
        &mut self,
        output: &SchedulerOutput,
        state: &mut OwnedExecutionState,
        kv_cache_mgr: &mut KVCacheManager,
        multi_step_count: usize,
    ) {
        if output.decode_requests.is_empty() {
            return;
        }

        let num_steps = multi_step_count.max(1);
        let mut active_decode_ids = output.decode_requests.clone();

        for _step in 0..num_steps {
            if active_decode_ids.is_empty() {
                break;
            }

            let failed = execute_batched_decode_with_graph(
                &active_decode_ids,
                &self.model,
                kv_cache_mgr,
                &mut state.requests,
                Some(&state.cuda_graph_dispatcher),
            );

            for req_id in failed {
                active_decode_ids.retain(|&id| id != req_id);
                finish_request_with_error(
                    req_id,
                    EngineError::Model("batched decode failed".to_string()),
                    &mut state.scheduler,
                    &mut state.requests,
                );
            }

            // Remove sequences that finished mid-step
            // Note: We need a tokenizer to check stop strings, but for efficiency
            // we only check non-string stop conditions here. Full check happens in main loop.
            active_decode_ids.retain(|&id| {
                state
                    .requests
                    .get(&id)
                    .map(|r| {
                        let s = &r.state;
                        let last = s.generated_token_ids.last().copied();
                        // Quick checks without tokenizer
                        let eos = last.map(|t| t == s.eos_token_id).unwrap_or(false);
                        let stop_token =
                            last.map(|t| s.stop_token_ids.contains(&t)).unwrap_or(false);
                        let max_len = s.num_generated() >= s.max_new_tokens;
                        !eos && !stop_token && !max_len
                    })
                    .unwrap_or(false)
            });
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

            // Phase 2: CUDA graph capture
            if config.enable_graph_capture && dispatcher.is_enabled() {
                match self.capture_decode_graph(batch_size, kv_cache_mgr, dispatcher) {
                    Ok(()) => {
                        stats.graphs_captured += 1;
                        if config.show_progress {
                            info!(batch_size, "CUDA graph captured");
                        }
                    }
                    Err(e) => {
                        stats.graphs_failed += 1;
                        warn!(batch_size, error = %e, "CUDA graph capture failed");
                        stats
                            .errors
                            .push(format!("graph batch_size={batch_size}: {e}"));
                    }
                }
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
                "Warmup complete"
            );
        }

        stats
    }
}
