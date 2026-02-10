//! Helper functions for engine operations.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Tensor;

use crate::kv_cache::KVCacheManager;
use crate::request::{FinishReason, RequestId, RequestStatus, SequenceState};
use crate::sampling::{self, SamplerState};
use crate::scheduler::Scheduler;
use crate::tokenizer::TokenizerWrapper;

use super::context::ActiveRequest;
use super::cuda_graph::{BatchDescriptor, CudaGraphDispatcher, ForwardContext};
use super::cuda_graph_runner::CudaGraphRunner;
use super::model_forward::{DecodeSequenceMetadata, ModelForward};
use super::types::{
    EngineCommand, EngineError, EngineStats, GenerationRequest, GenerationResult, PauseMode,
    ResponseChannel, StreamEvent,
};
use crate::lora::LoraContext;

/// Result of checking if generation should stop.
pub(crate) struct FinishCheck {
    pub reason: FinishReason,
    pub trim_bytes: usize,
    /// For stop token matches: the specific token ID that triggered the stop.
    pub stop_reason: Option<u32>,
}

/// Handle an incoming command. Returns true if shutdown was requested.
///
/// `paused` and `frozen` track the engine pause state:
/// - `paused`: new generation requests are rejected.
/// - `frozen`: scheduling is completely skipped (Keep mode).
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_command(
    cmd: EngineCommand,
    next_id: &mut RequestId,
    tokenizer: &TokenizerWrapper,
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    block_size: usize,
    kv_cache_mgr: &mut KVCacheManager,
    paused: &mut bool,
    frozen: &mut bool,
    spec_decode_stats: Option<super::types::SpecDecodingStats>,
) -> bool {
    match cmd {
        EngineCommand::Generate {
            request,
            response_tx,
        } => {
            if *paused {
                let _ = response_tx.send(Err(EngineError::Paused));
                return false;
            }
            admit_request(
                request,
                ResponseChannel::Complete(response_tx),
                next_id,
                tokenizer,
                scheduler,
                requests,
                block_size,
                kv_cache_mgr,
            );
            false
        }
        EngineCommand::GenerateStream { request, stream_tx } => {
            if *paused {
                let _ = stream_tx.try_send(StreamEvent::Error {
                    error: EngineError::Paused.to_string(),
                });
                return false;
            }
            admit_request(
                request,
                ResponseChannel::Stream(stream_tx),
                next_id,
                tokenizer,
                scheduler,
                requests,
                block_size,
                kv_cache_mgr,
            );
            false
        }
        EngineCommand::GetStats { response_tx } => {
            // Get detailed prefix cache stats from the KVCacheManager
            let (prefix_cache_detailed_stats, prefix_cache_recent_stats) =
                match kv_cache_mgr.prefix_cache() {
                    Some(cache) => (
                        Some(cache.get_stats()),
                        Some(cache.get_sliding_window_stats()),
                    ),
                    None => (None, None),
                };

            let stats = EngineStats {
                num_running_requests: scheduler.num_running(),
                num_waiting_requests: scheduler.num_waiting(),
                num_free_blocks: kv_cache_mgr.num_free_blocks(),
                num_total_blocks: kv_cache_mgr.num_total_blocks(),
                block_size,
                is_paused: *paused,
                kv_cache_metrics: kv_cache_mgr.metrics().snapshot(),
                prefix_cache_stats: kv_cache_mgr.prefix_cache_stats(),
                prefix_cache_detailed_stats,
                prefix_cache_recent_stats,
                spec_decode_stats,
            };
            let _ = response_tx.send(stats);
            false
        }
        EngineCommand::Abort { request_id } => {
            if let Some(mut active) = requests.remove(&request_id) {
                // Free KV cache blocks
                let _ = kv_cache_mgr.free_request(&mut active.state.block_table);
                scheduler.remove_request(request_id);
                tracing::debug!(request_id, "Request aborted, resources freed");
            }
            false
        }
        EngineCommand::Pause { mode, response_tx } => {
            if *paused {
                let _ = response_tx.send(Ok(()));
                return false;
            }
            match mode {
                PauseMode::Abort => {
                    // Abort all active requests
                    let request_ids: Vec<RequestId> = requests.keys().copied().collect();
                    for req_id in request_ids {
                        if let Some(mut active) = requests.remove(&req_id) {
                            let _ = kv_cache_mgr.free_request(&mut active.state.block_table);
                            scheduler.remove_request(req_id);
                            send_error(active.response, EngineError::Paused);
                        }
                    }
                    *paused = true;
                    *frozen = false;
                    tracing::info!("Engine paused (abort mode): all requests aborted");
                }
                PauseMode::Wait => {
                    // Reject new requests but let existing ones finish
                    *paused = true;
                    *frozen = false;
                    tracing::info!(
                        active_requests = requests.len(),
                        "Engine paused (wait mode): draining active requests"
                    );
                }
                PauseMode::Keep => {
                    // Freeze scheduling entirely
                    *paused = true;
                    *frozen = true;
                    tracing::info!(
                        active_requests = requests.len(),
                        "Engine paused (keep mode): requests frozen"
                    );
                }
            }
            let _ = response_tx.send(Ok(()));
            false
        }
        EngineCommand::Resume { response_tx } => {
            if *paused {
                *paused = false;
                *frozen = false;
                tracing::info!("Engine resumed");
            }
            let _ = response_tx.send(Ok(()));
            false
        }
        EngineCommand::IsPaused { response_tx } => {
            let _ = response_tx.send(*paused);
            false
        }
        EngineCommand::Shutdown => true,
    }
}

#[allow(clippy::too_many_arguments)]
fn admit_request(
    request: GenerationRequest,
    response: ResponseChannel,
    next_id: &mut RequestId,
    tokenizer: &TokenizerWrapper,
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    block_size: usize,
    kv_cache_mgr: &mut KVCacheManager,
) {
    let prompt_ids = match tokenizer.encode(&request.prompt) {
        Ok(ids) => ids,
        Err(e) => {
            send_error(response, EngineError::Tokenization(e.to_string()));
            return;
        }
    };

    let id = *next_id;
    *next_id += 1;

    let mut state = SequenceState::new(
        id,
        prompt_ids,
        request.max_new_tokens,
        request.eos_token_id,
        block_size,
        id,
    );
    state.sampler_state = SamplerState::new(request.sampling_params.seed);
    state.sampling_params = request.sampling_params;
    state.stop_token_ids = request.stop_token_ids;
    state.stop_strings = request.stop_strings;
    state.include_stop_str_in_output = request.include_stop_str_in_output;
    state.ignore_eos = request.ignore_eos;
    state.num_top_logprobs = request.logprobs.map(|n| n as usize);
    state.echo = request.echo;
    state.lora_request = request.lora_request;
    state.constraint = request.constraint;

    // Check prefix cache for reusable blocks via the KVCacheManager
    if kv_cache_mgr.has_prefix_cache() {
        let (cached_blocks, _) = kv_cache_mgr.match_prefix(&state.prompt_token_ids);
        if !cached_blocks.is_empty() {
            // Don't cache the entire prompt - leave at least 1 token for prefill
            // to ensure the model produces logits for the final position.
            let max_cached = state.prompt_token_ids.len().saturating_sub(1);
            let blocks_to_use = (max_cached / block_size).min(cached_blocks.len());
            if blocks_to_use > 0 {
                let tokens_covered = blocks_to_use * block_size;
                state
                    .block_table
                    .append_blocks(&cached_blocks[..blocks_to_use]);
                state.block_table.advance(tokens_covered);
                state.num_computed_tokens = tokens_covered;
                state.seqlen_offset = tokens_covered;
            }
        }
    }

    scheduler.add_request(id);
    // Initialize beam state if beam search is configured
    let beam_state = state.sampling_params.beam_search.as_ref().map(|config| {
        use crate::sampling::BeamSearchState;
        super::context::BeamState {
            search: BeamSearchState::new(config.clone(), state.eos_token_id),
            beam_block_tables: Vec::new(),
            beam_seqlen_offsets: Vec::new(),
        }
    });

    requests.insert(
        id,
        ActiveRequest {
            state,
            response,
            num_streamed_tokens: 0,
            streamed_text_len: 0,

            beam_state,
        },
    );
}

pub(crate) fn send_error(response: ResponseChannel, error: EngineError) {
    match response {
        ResponseChannel::Complete(tx) => {
            let _ = tx.send(Err(error));
        }
        ResponseChannel::Stream(tx) => {
            let _ = tx.try_send(StreamEvent::Error {
                error: error.to_string(),
            });
        }
    }
}

pub(crate) fn execute_prefill<M: ModelForward>(
    req_id: RequestId,
    chunk_size: usize,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    tokenizer: &TokenizerWrapper,
) -> Result<(), EngineError> {
    let req = requests
        .get_mut(&req_id)
        .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
    let prompt_len = req.state.prompt_token_ids.len();
    let offset = req.state.num_computed_tokens;
    let chunk_end = (offset + chunk_size).min(prompt_len);
    let actual_chunk = chunk_end - offset;
    let is_final_chunk = chunk_end == prompt_len;

    // Use eviction-aware allocation when prefix caching is enabled.
    // This allows evicting unreferenced cached blocks to make room for new requests.
    if kv_cache_mgr.has_prefix_cache() {
        kv_cache_mgr
            .allocate_with_eviction(&mut req.state.block_table, actual_chunk)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
    } else {
        kv_cache_mgr
            .allocate_for_request(&mut req.state.block_table, actual_chunk)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
    }
    let slot_mapping = req.state.block_table.slot_mapping(offset, actual_chunk);

    let chunk_tokens = &req.state.prompt_token_ids[offset..chunk_end];
    let input = Tensor::from_vec(chunk_tokens.to_vec(), (1, actual_chunk), model.device())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    // Build LoRA context from request's lora_request if present
    let lora_ctx = req
        .state
        .lora_request
        .as_ref()
        .map(|lr| LoraContext::with_adapter(&lr.name))
        .unwrap_or_else(LoraContext::none);

    let logits = model
        .forward_with_lora(
            &input,
            offset,
            kv_cache_mgr,
            &req.state.block_table,
            &slot_mapping,
            &lora_ctx,
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

    req.state.block_table.advance(actual_chunk);
    req.state.num_computed_tokens = chunk_end;
    req.state.seqlen_offset = chunk_end;

    // Compute prompt logprobs if echo mode is enabled with logprobs
    let compute_prompt_logprobs = req.state.echo && req.state.num_top_logprobs.is_some();
    if compute_prompt_logprobs {
        if offset == 0 {
            req.state.prompt_logprobs.push(None);
        }

        let prompt_token_ids = &req.state.prompt_token_ids;
        let last_pos = if is_final_chunk {
            actual_chunk - 1
        } else {
            actual_chunk
        };

        for i in 0..last_pos {
            let target_token_idx = offset + i + 1;
            if target_token_idx < prompt_len {
                let target_token = prompt_token_ids[target_token_idx];
                let pos_logits = logits
                    .narrow(1, i, 1)
                    .and_then(|t| t.squeeze(0))
                    .and_then(|t| t.squeeze(0))
                    .and_then(|t| t.to_dtype(candle_core::DType::F32))
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| EngineError::Model(e.to_string()))?;

                let log_probs = sampling::log_softmax(&pos_logits);
                let logprob = log_probs.get(target_token as usize).copied();
                req.state.prompt_logprobs.push(logprob);
            }
        }
    }

    if is_final_chunk {
        req.state.status = RequestStatus::Decoding;

        let seq_dim = logits.dims()[1];
        let logits = logits
            .narrow(1, seq_dim - 1, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;

        if let Some(beam_state) = req.beam_state.as_mut() {
            // Beam search: compute log_softmax over full vocab and run initial beam step
            let logits_vec: Vec<f32> = logits
                .squeeze(0)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1())
                .map_err(|e| EngineError::Model(e.to_string()))?;

            let log_probs = sampling::log_softmax(&logits_vec);
            let beam_width = beam_state.search.config.beam_width;

            // Initial step: all beams see the same log_probs (from the single prompt)
            let initial_log_probs: Vec<Vec<f32>> = vec![log_probs; beam_width];
            let transitions = beam_state.search.step(&initial_log_probs);

            // Clone the request's block table into per-beam block tables
            let prompt_len = req.state.seqlen_offset;
            let block_ids = req.state.block_table.block_ids().to_vec();
            let block_size = kv_cache_mgr.block_size();

            beam_state.beam_block_tables.clear();
            beam_state.beam_seqlen_offsets.clear();

            for _ in 0..beam_width {
                let mut bt = crate::kv_cache::BlockTable::new(block_size);
                bt.append_blocks(&block_ids);
                bt.advance(prompt_len);
                beam_state.beam_block_tables.push(bt);
                beam_state.beam_seqlen_offsets.push(prompt_len);
            }

            // Set generated_token_ids to the best beam's first token for streaming
            if let Some(&(_, token_id)) = transitions.first() {
                req.state.generated_token_ids.push(token_id);
            }
        } else {
            let next_token = sample_token(&logits, &mut req.state, tokenizer)?;
            req.state.generated_token_ids.push(next_token);
        }
    } else {
        req.state.status = RequestStatus::Prefilling;
    }

    Ok(())
}

pub(crate) fn execute_decode<M: ModelForward>(
    req_id: RequestId,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    tokenizer: &TokenizerWrapper,
) -> Result<(), EngineError> {
    let req = requests
        .get_mut(&req_id)
        .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;

    kv_cache_mgr
        .allocate_for_request(&mut req.state.block_table, 1)
        .map_err(|e| EngineError::Cache(e.to_string()))?;
    let slot_mapping = req
        .state
        .block_table
        .slot_mapping(req.state.seqlen_offset, 1);

    let last_token = *req
        .state
        .generated_token_ids
        .last()
        .ok_or_else(|| EngineError::Model("no generated tokens for decode".to_string()))?;
    let input = Tensor::new(&[[last_token]], model.device())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    // Build LoRA context from request's lora_request if present
    let lora_ctx = req
        .state
        .lora_request
        .as_ref()
        .map(|lr| LoraContext::with_adapter(&lr.name))
        .unwrap_or_else(LoraContext::none);

    let logits = model
        .forward_with_lora(
            &input,
            req.state.seqlen_offset,
            kv_cache_mgr,
            &req.state.block_table,
            &slot_mapping,
            &lora_ctx,
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

    req.state.block_table.advance(1);
    req.state.seqlen_offset += 1;

    let seq_dim = logits.dims()[1];
    let logits = logits
        .narrow(1, seq_dim - 1, 1)
        .map_err(|e| EngineError::Model(e.to_string()))?;
    let next_token = sample_token(&logits, &mut req.state, tokenizer)?;
    req.state.generated_token_ids.push(next_token);

    Ok(())
}

/// Execute batched decode for multiple sequences in a single forward pass.
/// Returns IDs of requests that failed.
///
/// Convenience wrapper around `execute_batched_decode_with_graph` that
/// disables CUDA graph dispatch. Used primarily in tests.
#[cfg(test)]
pub(crate) fn execute_batched_decode<M: ModelForward>(
    decode_request_ids: &[RequestId],
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> Vec<RequestId> {
    execute_batched_decode_with_graph(
        decode_request_ids,
        model,
        kv_cache_mgr,
        requests,
        None,
        None,
    )
}

/// Execute batched decode with optional CUDA graph support.
/// When a dispatcher is provided, dispatches to cached graphs when available.
/// When a graph runner is provided, uses it for optimized execution.
pub(crate) fn execute_batched_decode_with_graph<M: ModelForward>(
    decode_request_ids: &[RequestId],
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    dispatcher: Option<&Arc<std::sync::RwLock<CudaGraphDispatcher>>>,
    graph_runner: Option<&Arc<CudaGraphRunner>>,
) -> Vec<RequestId> {
    let mut failed = Vec::new();
    let mut batch_ids: Vec<RequestId> = Vec::with_capacity(decode_request_ids.len());

    // Step 1: Allocate blocks for each sequence
    for &req_id in decode_request_ids {
        let Some(req) = requests.get_mut(&req_id) else {
            failed.push(req_id);
            continue;
        };
        if kv_cache_mgr
            .allocate_for_request(&mut req.state.block_table, 1)
            .is_err()
        {
            failed.push(req_id);
        } else {
            batch_ids.push(req_id);
        }
    }

    if batch_ids.is_empty() {
        return failed;
    }

    // Step 2: Collect input tokens, per-sequence metadata, and LoRA context
    let mut token_ids: Vec<u32> = Vec::with_capacity(batch_ids.len());
    let mut sequences: Vec<DecodeSequenceMetadata> = Vec::with_capacity(batch_ids.len());

    // Determine batch LoRA context from first request with a LoRA adapter
    // Note: In production, you'd want to batch requests by LoRA adapter for efficiency
    let batch_lora_ctx = batch_ids
        .iter()
        .find_map(|&req_id| {
            requests.get(&req_id).and_then(|req| {
                req.state
                    .lora_request
                    .as_ref()
                    .map(|lr| LoraContext::with_adapter(&lr.name))
            })
        })
        .unwrap_or_else(LoraContext::none);

    for &req_id in &batch_ids {
        let Some(req) = requests.get(&req_id) else {
            // Request was removed between allocation and metadata collection
            continue;
        };
        let Some(&last_token) = req.state.generated_token_ids.last() else {
            // No generated tokens yet - skip this request
            continue;
        };
        let slot_mapping = req
            .state
            .block_table
            .slot_mapping(req.state.seqlen_offset, 1);
        token_ids.push(last_token);
        sequences.push(DecodeSequenceMetadata {
            request_id: req_id,
            seqlen_offset: req.state.seqlen_offset,
            block_ids: req.state.block_table.block_ids().to_vec(),
            slot_mapping,
        });
    }

    // Step 3: Batched forward pass
    let batch_size = batch_ids.len();

    // CUDA Graph dispatch: determine execution mode
    let forward_ctx = if let Some(disp) = dispatcher {
        let descriptor = BatchDescriptor::for_decode(batch_size);
        match disp.read() {
            Ok(disp_guard) => {
                let result = disp_guard.dispatch(descriptor);
                ForwardContext::from_dispatch(result)
            }
            Err(_poisoned) => {
                // Lock was poisoned, fall back to eager execution
                ForwardContext::eager()
            }
        }
    } else {
        ForwardContext::eager()
    };

    let input = match Tensor::from_vec(token_ids, (batch_size, 1), model.device()) {
        Ok(t) => t,
        Err(_) => {
            failed.extend(&batch_ids);
            return failed;
        }
    };

    // Execute forward pass - prioritize: LoRA > CUDA Graph Runner > Model Context > Eager
    // Note: CUDA graphs and LoRA are currently mutually exclusive for simplicity
    let logits = if batch_lora_ctx.has_adapter() {
        // LoRA path - no graph support
        match model.forward_decode_batch_with_lora(
            &input,
            &sequences,
            kv_cache_mgr,
            &batch_lora_ctx,
        ) {
            Ok(l) => l,
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        }
    } else if let Some(runner) = graph_runner {
        // Try CUDA graph runner path for optimized execution
        match runner.execute(&input, |inp| {
            model.forward_decode_batch(inp, &sequences, kv_cache_mgr)
        }) {
            Ok(l) => l,
            Err(_) => {
                // Fall back to eager execution on runner error
                match model.forward_decode_batch(&input, &sequences, kv_cache_mgr) {
                    Ok(l) => l,
                    Err(_) => {
                        failed.extend(&batch_ids);
                        return failed;
                    }
                }
            }
        }
    } else {
        // Execute forward pass with CUDA graph context
        // The context indicates whether we can replay a cached graph
        // or should execute eagerly. Graph capture happens during warmup.
        match model.forward_decode_batch_with_ctx(&input, &sequences, kv_cache_mgr, &forward_ctx) {
            Ok(l) => l,
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        }
    };

    // Step 4: Extract logits and sample per-sequence
    let seq_dim = logits.dims()[1];
    let last_logits = match logits.narrow(1, seq_dim - 1, 1) {
        Ok(l) => match l.squeeze(1) {
            Ok(l) => l,
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        },
        Err(_) => {
            failed.extend(&batch_ids);
            return failed;
        }
    };

    let all_greedy = batch_ids.iter().all(|&id| {
        requests
            .get(&id)
            .map(|r| r.state.sampling_params.is_greedy())
            .unwrap_or(true)
    });
    let any_need_logprobs = batch_ids.iter().any(|&id| {
        requests
            .get(&id)
            .map(|r| r.state.num_top_logprobs.is_some())
            .unwrap_or(false)
    });

    let logits_cpu: Vec<f32> = match last_logits.to_vec2::<f32>() {
        Ok(v) => v.into_iter().flatten().collect(),
        Err(_) => {
            failed.extend(&batch_ids);
            return failed;
        }
    };
    let vocab_size = last_logits.dims()[1];

    // Step 5: Sample tokens and update state
    if all_greedy && !any_need_logprobs {
        for (i, &req_id) in batch_ids.iter().enumerate() {
            let Some(req) = requests.get_mut(&req_id) else {
                failed.push(req_id);
                continue;
            };
            let logit_slice = &logits_cpu[i * vocab_size..(i + 1) * vocab_size];
            let token_id = logit_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            req.state.block_table.advance(1);
            req.state.seqlen_offset += 1;
            req.state.generated_token_ids.push(token_id);
        }
    } else {
        for (i, &req_id) in batch_ids.iter().enumerate() {
            let Some(req) = requests.get_mut(&req_id) else {
                failed.push(req_id);
                continue;
            };
            let logit_slice = &logits_cpu[i * vocab_size..(i + 1) * vocab_size];
            let result = sampling::sample(
                logit_slice,
                &req.state.sampling_params,
                &req.state.generated_token_ids,
                &mut req.state.sampler_state,
                req.state.num_top_logprobs,
            );
            req.state.block_table.advance(1);
            req.state.seqlen_offset += 1;
            req.state.generated_token_ids.push(result.token_id);

            if req.state.num_top_logprobs.is_some() {
                req.state.token_logprobs.push(result.logprob);
                if let Some(top) = result.top_logprobs {
                    req.state.top_logprobs.push(top);
                }
            }
        }
    }

    failed
}

pub(crate) fn check_finished(
    state: &SequenceState,
    tokenizer: &TokenizerWrapper,
) -> Option<FinishCheck> {
    let last_token = *state.generated_token_ids.last()?;

    // min_tokens guard: don't stop before minimum generated tokens.
    // Length limit still overrides min_tokens.
    if state.generated_token_ids.len() < state.sampling_params.min_tokens {
        if state.num_generated() >= state.max_new_tokens {
            return Some(FinishCheck {
                reason: FinishReason::Length,
                trim_bytes: 0,
                stop_reason: None,
            });
        }
        return None;
    }

    // EOS check — respect ignore_eos
    if !state.ignore_eos && last_token == state.eos_token_id {
        return Some(FinishCheck {
            reason: FinishReason::Eos,
            trim_bytes: 0,
            stop_reason: None,
        });
    }

    if state.stop_token_ids.contains(&last_token) {
        return Some(FinishCheck {
            reason: FinishReason::Stop,
            trim_bytes: 0,
            stop_reason: Some(last_token),
        });
    }

    if !state.stop_strings.is_empty() {
        if let Ok(text) = tokenizer.decode(&state.generated_token_ids) {
            for stop_str in &state.stop_strings {
                if text.ends_with(stop_str.as_str()) {
                    let trim = if state.include_stop_str_in_output {
                        0
                    } else {
                        stop_str.len()
                    };
                    return Some(FinishCheck {
                        reason: FinishReason::Stop,
                        trim_bytes: trim,
                        stop_reason: None,
                    });
                }
            }
        }
    }

    // Check if constraint is satisfied (structured output complete)
    if let Some(ref constraint) = state.constraint {
        if let Ok(text) = tokenizer.decode(&state.generated_token_ids) {
            if constraint.is_complete(&state.generated_token_ids, &text) {
                return Some(FinishCheck {
                    reason: FinishReason::Stop,
                    trim_bytes: 0,
                    stop_reason: None,
                });
            }
        }
    }

    if state.num_generated() >= state.max_new_tokens {
        return Some(FinishCheck {
            reason: FinishReason::Length,
            trim_bytes: 0,
            stop_reason: None,
        });
    }

    None
}

pub(crate) fn sample_token(
    logits: &Tensor,
    state: &mut SequenceState,
    tokenizer: &TokenizerWrapper,
) -> Result<u32, EngineError> {
    let mut logits_vec: Vec<f32> = logits
        .squeeze(0)
        .and_then(|t| t.squeeze(0))
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    // Apply constraint masking if present
    if let Some(ref mut constraint) = state.constraint {
        let generated_text = tokenizer
            .decode(&state.generated_token_ids)
            .unwrap_or_default();
        constraint
            .mask_logits(&mut logits_vec, &generated_text)
            .map_err(|e| EngineError::Model(format!("constraint error: {}", e)))?;
    }

    let result = sampling::sample(
        &logits_vec,
        &state.sampling_params,
        &state.generated_token_ids,
        &mut state.sampler_state,
        state.num_top_logprobs,
    );

    if state.num_top_logprobs.is_some() {
        state.token_logprobs.push(result.logprob);
        if let Some(top) = result.top_logprobs {
            state.top_logprobs.push(top);
        }
    }

    Ok(result.token_id)
}

pub(crate) fn finish_request_with_error(
    req_id: RequestId,
    error: EngineError,
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) {
    if let Some(req) = requests.remove(&req_id) {
        scheduler.remove_request(req_id);
        send_error(req.response, error);
    }
}

/// Send streaming token events for a request.
pub(crate) fn send_stream_token(
    req_id: RequestId,
    tokenizer: &TokenizerWrapper,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) {
    let Some(req) = requests.get_mut(&req_id) else {
        return;
    };
    let ResponseChannel::Stream(ref tx) = req.response else {
        return;
    };
    let start = req.num_streamed_tokens;
    let end = req.state.generated_token_ids.len();
    if start >= end {
        return;
    }
    for i in start..end {
        let token_id = req.state.generated_token_ids[i];
        let text_so_far = tokenizer
            .decode(&req.state.generated_token_ids[..=i])
            .unwrap_or_default();
        let token_text = text_so_far[req.streamed_text_len..].to_string();
        req.streamed_text_len = text_so_far.len();

        // Include logprob data when the request has logprobs enabled
        let logprob = if req.state.num_top_logprobs.is_some() {
            req.state.token_logprobs.get(i).copied()
        } else {
            None
        };
        let top_logprobs = if req.state.num_top_logprobs.is_some() {
            req.state.top_logprobs.get(i).cloned()
        } else {
            None
        };

        let _ = tx.try_send(StreamEvent::Token {
            token_id,
            token_text,
            logprob,
            top_logprobs,
        });
    }
    req.num_streamed_tokens = end;
}

/// Build a GenerationResult from sequence state.
pub(crate) fn build_generation_result(
    request_id: RequestId,
    generated_text: String,
    state: &mut SequenceState,
    finish_reason: FinishReason,
    stop_reason: Option<u32>,
) -> GenerationResult {
    let has_logprobs = state.num_top_logprobs.is_some();

    let token_logprobs = if has_logprobs && !state.token_logprobs.is_empty() {
        Some(std::mem::take(&mut state.token_logprobs))
    } else {
        None
    };

    let top_logprobs = if has_logprobs && !state.top_logprobs.is_empty() {
        Some(std::mem::take(&mut state.top_logprobs))
    } else {
        None
    };

    let prompt_token_ids = if has_logprobs {
        Some(state.prompt_token_ids.clone())
    } else {
        None
    };

    let prompt_logprobs = if has_logprobs && !state.prompt_logprobs.is_empty() {
        Some(std::mem::take(&mut state.prompt_logprobs))
    } else {
        None
    };

    GenerationResult {
        request_id,
        generated_text,
        generated_token_ids: std::mem::take(&mut state.generated_token_ids),
        finish_reason,
        stop_reason,
        token_logprobs,
        top_logprobs,
        prompt_token_ids,
        prompt_logprobs,
    }
}

/// Greedy sampling from logits tensor.
pub fn greedy_sample(logits: &Tensor) -> anyhow::Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(token_id)
}

// ─── Beam Search Helpers ───────────────────────────────────────────────────

/// Execute a single beam search decode step for a request.
///
/// For each active beam: allocate 1 slot, run forward pass, compute log_softmax,
/// then call `BeamSearchState::step()` to select the best candidates. After
/// expansion, reassign block tables so each new beam inherits from its parent.
pub(crate) fn execute_beam_decode<M: ModelForward>(
    req_id: RequestId,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    tokenizer: &TokenizerWrapper,
) -> Result<(), EngineError> {
    let req = requests
        .get_mut(&req_id)
        .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;

    let beam_state = req
        .beam_state
        .as_mut()
        .ok_or_else(|| EngineError::Model("beam_state is None for beam decode".to_string()))?;

    let num_active = beam_state.search.num_active_beams();
    if num_active == 0 {
        return Ok(());
    }

    let beam_width = beam_state.search.config.beam_width;

    // 1. Allocate slots and build per-beam metadata for batched forward
    let mut beam_input_tokens: Vec<u32> = Vec::with_capacity(beam_width);
    let mut sequences: Vec<DecodeSequenceMetadata> = Vec::with_capacity(beam_width);

    for beam_idx in 0..beam_width {
        if beam_state.search.beams[beam_idx].is_finished {
            continue;
        }

        // Allocate a slot for this beam
        kv_cache_mgr
            .allocate_for_request(&mut beam_state.beam_block_tables[beam_idx], 1)
            .map_err(|e| EngineError::Cache(e.to_string()))?;

        let seqlen_offset = beam_state.beam_seqlen_offsets[beam_idx];
        let slot_mapping = beam_state.beam_block_tables[beam_idx].slot_mapping(seqlen_offset, 1);
        let block_ids = beam_state.beam_block_tables[beam_idx].block_ids().to_vec();

        // The last token for this beam
        let last_token = beam_state.search.beams[beam_idx]
            .token_ids
            .last()
            .copied()
            .unwrap_or_else(|| {
                // If no tokens yet (first decode step), use the last generated token
                req.state.generated_token_ids.last().copied().unwrap_or(0)
            });

        beam_input_tokens.push(last_token);
        sequences.push(DecodeSequenceMetadata {
            request_id: req_id,
            seqlen_offset,
            block_ids,
            slot_mapping,
        });
    }

    if sequences.is_empty() {
        return Ok(());
    }

    // 2. Batched forward pass for all active beams
    let input = Tensor::new(
        beam_input_tokens
            .iter()
            .map(|&t| vec![t])
            .collect::<Vec<_>>(),
        model.device(),
    )
    .map_err(|e| EngineError::Model(e.to_string()))?;

    let logits = model
        .forward_decode_batch(&input, &sequences, kv_cache_mgr)
        .map_err(|e| EngineError::Model(e.to_string()))?;

    // 3. Advance block tables and seqlen offsets for active beams
    let mut active_beam_indices: Vec<usize> = Vec::new();
    for beam_idx in 0..beam_width {
        if !beam_state.search.beams[beam_idx].is_finished {
            beam_state.beam_block_tables[beam_idx].advance(1);
            beam_state.beam_seqlen_offsets[beam_idx] += 1;
            active_beam_indices.push(beam_idx);
        }
    }

    // 4. Extract per-beam logits and compute log_softmax
    let mut per_beam_log_probs: Vec<Vec<f32>> = Vec::with_capacity(beam_width);
    let mut active_idx = 0;

    for beam_idx in 0..beam_width {
        if beam_state.search.beams[beam_idx].is_finished {
            // Dead beams get -inf log probs so they won't be selected
            let vocab_size = logits.dims().last().copied().unwrap_or(0);
            per_beam_log_probs.push(vec![f32::NEG_INFINITY; vocab_size]);
        } else {
            let beam_logits: Vec<f32> = logits
                .narrow(0, active_idx, 1)
                .and_then(|t| t.squeeze(0))
                .and_then(|t| {
                    // Handle both [1, vocab] and [vocab] shapes
                    if t.dims().len() > 1 {
                        t.squeeze(0)
                    } else {
                        Ok(t)
                    }
                })
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1())
                .map_err(|e| EngineError::Model(e.to_string()))?;

            per_beam_log_probs.push(sampling::log_softmax(&beam_logits));
            active_idx += 1;
        }
    }

    // 5. Beam expansion step
    let transitions = beam_state.search.step(&per_beam_log_probs);

    // 6. Reassign block tables: new beam i gets a copy of parent beam's block table
    let old_block_tables = beam_state.beam_block_tables.clone();
    let old_offsets = beam_state.beam_seqlen_offsets.clone();

    for (new_beam_idx, &(old_beam_idx, _token_id)) in transitions.iter().enumerate() {
        beam_state.beam_block_tables[new_beam_idx] = old_block_tables[old_beam_idx].clone();
        beam_state.beam_seqlen_offsets[new_beam_idx] = old_offsets[old_beam_idx];
    }

    // 7. Update generated_token_ids to best beam's sequence for streaming
    let best = beam_state.search.get_best_hypotheses();
    if let Some(best_hyp) = best.first() {
        req.state.generated_token_ids.clear();
        req.state.generated_token_ids.extend(&best_hyp.token_ids);
    }

    // For non-streaming requests, also update generated text eagerly
    let _ = tokenizer; // Satisfies the signature; text is built at completion

    Ok(())
}

/// Check if a beam search request has finished.
///
/// Returns `Some(FinishCheck)` when either:
/// - All beams have completed (hit EOS or been pruned)
/// - The max_new_tokens limit has been reached
/// - Early stopping criteria are met
pub(crate) fn check_beam_finished(req: &ActiveRequest) -> Option<FinishCheck> {
    let beam_state = req.beam_state.as_ref()?;

    if beam_state.search.is_done() {
        return Some(FinishCheck {
            reason: FinishReason::Eos,
            trim_bytes: 0,
            stop_reason: None,
        });
    }

    // Check max_new_tokens against the longest active beam
    let max_beam_len = beam_state
        .search
        .beams
        .iter()
        .map(|b| b.token_ids.len())
        .max()
        .unwrap_or(0);

    if max_beam_len >= req.state.max_new_tokens {
        return Some(FinishCheck {
            reason: FinishReason::Length,
            trim_bytes: 0,
            stop_reason: None,
        });
    }

    None
}

/// Finalize a beam search request: select the best hypothesis, update the
/// sequence state, and free all per-beam block tables.
pub(crate) fn finalize_beam_request(
    req: &mut ActiveRequest,
    kv_cache_mgr: &mut KVCacheManager,
    tokenizer: &TokenizerWrapper,
) -> (String, FinishReason) {
    let beam_state = req
        .beam_state
        .as_mut()
        .expect("finalize_beam_request called without beam_state");

    let best = beam_state.search.get_best_hypotheses();
    let (token_ids, finish_reason) = if let Some(best_hyp) = best.first() {
        let reason = if best_hyp.is_finished {
            FinishReason::Eos
        } else {
            FinishReason::Length
        };
        (best_hyp.token_ids.clone(), reason)
    } else {
        (Vec::new(), FinishReason::Length)
    };

    // Update sequence state with the best hypothesis
    req.state.generated_token_ids = token_ids;

    // Free all per-beam block tables
    for bt in &mut beam_state.beam_block_tables {
        let freed_ids = bt.release();
        if !freed_ids.is_empty() {
            let _ = kv_cache_mgr.free_blocks(&freed_ids);
        }
    }

    // Clear beam state
    req.beam_state = None;

    let generated_text = tokenizer
        .decode(&req.state.generated_token_ids)
        .unwrap_or_default();

    (generated_text, finish_reason)
}

/// Returns `true` if the given request is a beam search request.
pub(crate) fn is_beam_request(req: &ActiveRequest) -> bool {
    req.beam_state.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(eos: u32, max_new: usize) -> SequenceState {
        SequenceState::new(0, vec![1, 2, 3], max_new, eos, 16, 0)
    }

    fn dummy_tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(100)
    }

    #[test]
    fn test_check_finished_eos() {
        let mut state = make_state(99, 100);
        state.generated_token_ids.push(99);
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Eos);
        assert!(check.stop_reason.is_none());
    }

    #[test]
    fn test_check_finished_ignore_eos() {
        let mut state = make_state(99, 100);
        state.ignore_eos = true;
        state.generated_token_ids.push(99);
        let tok = dummy_tokenizer();
        // EOS should be ignored
        assert!(check_finished(&state, &tok).is_none());
    }

    #[test]
    fn test_check_finished_stop_token_with_reason() {
        let mut state = make_state(99, 100);
        state.stop_token_ids = vec![42, 55];
        state.generated_token_ids.push(42);
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Stop);
        assert_eq!(check.stop_reason, Some(42));
    }

    #[test]
    fn test_check_finished_length_limit() {
        let mut state = make_state(99, 2);
        state.generated_token_ids.push(10);
        state.generated_token_ids.push(20);
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Length);
        assert!(check.stop_reason.is_none());
    }

    #[test]
    fn test_check_finished_min_tokens_blocks_eos() {
        let mut state = make_state(99, 100);
        state.sampling_params.min_tokens = 5;
        // Only 1 token generated, below min_tokens
        state.generated_token_ids.push(99);
        let tok = dummy_tokenizer();
        // EOS should NOT trigger stop because min_tokens not reached
        assert!(check_finished(&state, &tok).is_none());
    }

    #[test]
    fn test_check_finished_min_tokens_blocks_stop_token() {
        let mut state = make_state(99, 100);
        state.sampling_params.min_tokens = 3;
        state.stop_token_ids = vec![42];
        state.generated_token_ids.push(42);
        let tok = dummy_tokenizer();
        // Stop token should NOT trigger because min_tokens not reached
        assert!(check_finished(&state, &tok).is_none());
    }

    #[test]
    fn test_check_finished_min_tokens_allows_length_limit() {
        let mut state = make_state(99, 1);
        state.sampling_params.min_tokens = 10;
        // 1 token generated, hits max_new_tokens (1) even though < min_tokens
        state.generated_token_ids.push(99);
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Length);
    }

    #[test]
    fn test_check_finished_min_tokens_met_allows_eos() {
        let mut state = make_state(99, 100);
        state.sampling_params.min_tokens = 2;
        state.generated_token_ids.push(10);
        state.generated_token_ids.push(99); // EOS at position 2, min_tokens=2 met
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Eos);
    }

    #[test]
    fn test_check_finished_no_tokens() {
        let state = make_state(99, 100);
        let tok = dummy_tokenizer();
        assert!(check_finished(&state, &tok).is_none());
    }

    #[test]
    fn test_check_finished_non_stop_token() {
        let mut state = make_state(99, 100);
        state.generated_token_ids.push(50);
        let tok = dummy_tokenizer();
        assert!(check_finished(&state, &tok).is_none());
    }

    #[test]
    fn test_check_finished_ignore_eos_still_respects_stop_tokens() {
        let mut state = make_state(99, 100);
        state.ignore_eos = true;
        state.stop_token_ids = vec![42];
        state.generated_token_ids.push(42);
        let tok = dummy_tokenizer();
        let check = check_finished(&state, &tok).unwrap();
        assert_eq!(check.reason, FinishReason::Stop);
        assert_eq!(check.stop_reason, Some(42));
    }
}
