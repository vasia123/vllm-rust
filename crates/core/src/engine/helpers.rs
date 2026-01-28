//! Helper functions for engine operations.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::Tensor;

use crate::kv_cache::{prefix_cache::PrefixCache, KVCacheManager};
use crate::request::{FinishReason, RequestId, RequestStatus, SequenceState};
use crate::sampling::{self, SamplerState};
use crate::scheduler::Scheduler;
use crate::tokenizer::TokenizerWrapper;

use super::context::ActiveRequest;
use super::cuda_graph::{BatchDescriptor, CudaGraphDispatcher, ForwardContext};
use super::cuda_graph_runner::CudaGraphRunner;
use super::model_forward::{DecodeSequenceMetadata, ModelForward};
use super::types::{
    EngineCommand, EngineError, EngineStats, GenerationRequest, GenerationResult, ResponseChannel,
    StreamEvent,
};
use crate::lora::LoraContext;

/// Result of checking if generation should stop.
pub(crate) struct FinishCheck {
    pub reason: FinishReason,
    pub trim_bytes: usize,
}

/// Handle an incoming command. Returns true if shutdown was requested.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_command(
    cmd: EngineCommand,
    next_id: &mut RequestId,
    tokenizer: &TokenizerWrapper,
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    block_size: usize,
    prefix_cache: &mut Option<PrefixCache>,
    kv_cache_mgr: &mut KVCacheManager,
) -> bool {
    match cmd {
        EngineCommand::Generate {
            request,
            response_tx,
        } => {
            admit_request(
                request,
                ResponseChannel::Complete(response_tx),
                next_id,
                tokenizer,
                scheduler,
                requests,
                block_size,
                prefix_cache,
            );
            false
        }
        EngineCommand::GenerateStream { request, stream_tx } => {
            admit_request(
                request,
                ResponseChannel::Stream(stream_tx),
                next_id,
                tokenizer,
                scheduler,
                requests,
                block_size,
                prefix_cache,
            );
            false
        }
        EngineCommand::GetStats { response_tx } => {
            let stats = EngineStats {
                num_running_requests: scheduler.num_running(),
                num_waiting_requests: scheduler.num_waiting(),
                num_free_blocks: kv_cache_mgr.num_free_blocks(),
                num_total_blocks: kv_cache_mgr.num_total_blocks(),
                block_size,
                kv_cache_metrics: kv_cache_mgr.metrics().snapshot(),
                prefix_cache_stats: kv_cache_mgr.prefix_cache_stats(),
            };
            let _ = response_tx.send(stats);
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
    prefix_cache: &mut Option<PrefixCache>,
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
    state.num_top_logprobs = request.logprobs.map(|n| n as usize);
    state.echo = request.echo;
    state.lora_request = request.lora_request;
    state.constraint = request.constraint;

    // Check prefix cache for reusable blocks
    if let Some(cache) = prefix_cache.as_mut() {
        let (cached_blocks, _) = cache.match_prefix(&state.prompt_token_ids);
        if !cached_blocks.is_empty() {
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
    requests.insert(
        id,
        ActiveRequest {
            state,
            response,
            num_streamed_tokens: 0,
            streamed_text_len: 0,
            draft_state: None,
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

    kv_cache_mgr
        .allocate_for_request(&mut req.state.block_table, actual_chunk)
        .map_err(|e| EngineError::Cache(e.to_string()))?;
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
        let next_token = sample_token(&logits, &mut req.state, tokenizer)?;
        req.state.generated_token_ids.push(next_token);
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

    if last_token == state.eos_token_id {
        return Some(FinishCheck {
            reason: FinishReason::Eos,
            trim_bytes: 0,
        });
    }

    if state.stop_token_ids.contains(&last_token) {
        return Some(FinishCheck {
            reason: FinishReason::Stop,
            trim_bytes: 0,
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
                });
            }
        }
    }

    if state.num_generated() >= state.max_new_tokens {
        return Some(FinishCheck {
            reason: FinishReason::Length,
            trim_bytes: 0,
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
        let _ = tx.try_send(StreamEvent::Token {
            token_id,
            token_text,
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
