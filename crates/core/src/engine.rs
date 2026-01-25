use std::collections::HashMap;

use candle_core::{Device, Tensor};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::kv_cache::prefix_cache::PrefixCache;
use crate::kv_cache::{BlockId, BlockTable, KVCacheManager};
use crate::request::{FinishReason, RequestId, RequestStatus, SequenceState};
use crate::sampling::{self, SamplerState, SamplingParams};
use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::tokenizer::TokenizerWrapper;

// ─── Streaming types ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token {
        token_id: u32,
        token_text: String,
    },
    Done {
        finish_reason: FinishReason,
        generated_text: String,
    },
    Error {
        error: String,
    },
}

// ─── ModelForward trait ────────────────────────────────────────────────────

/// Per-sequence metadata for batched decode (one token per sequence).
pub struct DecodeSequenceMetadata {
    pub seqlen_offset: usize,
    pub block_ids: Vec<BlockId>,
    pub slot_mapping: Vec<usize>,
}

pub trait ModelForward: Send + 'static {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor>;

    /// Batched decode: process multiple sequences each generating one token.
    /// Default implementation falls back to sequential forward calls.
    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        let batch_size = sequences.len();
        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let token = input_ids.narrow(0, i, 1)?;
            let block_table = BlockTable::from_block_ids(seq.block_ids.clone(), seq.seqlen_offset);
            let logits = self.forward(
                &token,
                seq.seqlen_offset,
                kv_cache_mgr,
                &block_table,
                &seq.slot_mapping,
            )?;
            outputs.push(logits);
        }
        Tensor::cat(&outputs, 0)
    }

    fn device(&self) -> &Device;
}

impl ModelForward for Box<dyn ModelForward> {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        (**self).forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        (**self).forward_decode_batch(input_ids, sequences, kv_cache_mgr)
    }

    fn device(&self) -> &Device {
        (**self).device()
    }
}

// ─── Engine API types ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine has shut down")]
    Shutdown,
    #[error("tokenization error: {0}")]
    Tokenization(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("cache error: {0}")]
    Cache(String),
}

pub struct GenerationRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub sampling_params: SamplingParams,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub include_stop_str_in_output: bool,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 128,
            eos_token_id: 0,
            sampling_params: SamplingParams::greedy(),
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            include_stop_str_in_output: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub request_id: RequestId,
    pub generated_text: String,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
}

pub struct SpeculativeConfig {
    pub num_speculative_tokens: usize,
}

pub struct EngineConfig {
    pub scheduler_config: SchedulerConfig,
    pub block_size: usize,
    pub speculative_config: Option<SpeculativeConfig>,
    /// Run N decode steps per scheduler invocation to amortize scheduling overhead.
    /// Sequences that finish mid-step break early. Default: 1 (no multi-step).
    pub multi_step_count: usize,
    /// Enable automatic prefix caching for KV cache block reuse.
    pub enable_prefix_caching: bool,
}

// ─── Engine commands (internal) ────────────────────────────────────────────

enum ResponseChannel {
    Complete(oneshot::Sender<Result<GenerationResult, EngineError>>),
    Stream(mpsc::Sender<StreamEvent>),
}

enum EngineCommand {
    Generate {
        request: GenerationRequest,
        response_tx: oneshot::Sender<Result<GenerationResult, EngineError>>,
    },
    GenerateStream {
        request: GenerationRequest,
        stream_tx: mpsc::Sender<StreamEvent>,
    },
    Shutdown,
}

// ─── EngineHandle (public, cloneable) ──────────────────────────────────────

#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    pub async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Generate {
                request,
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)?
    }

    pub async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>, EngineError> {
        let (stream_tx, stream_rx) = mpsc::channel(64);
        self.cmd_tx
            .send(EngineCommand::GenerateStream { request, stream_tx })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        Ok(stream_rx)
    }

    pub async fn shutdown(&self) -> Result<(), EngineError> {
        self.cmd_tx
            .send(EngineCommand::Shutdown)
            .await
            .map_err(|_| EngineError::Shutdown)
    }
}

// ─── Engine start ──────────────────────────────────────────────────────────

pub fn start_engine<M: ModelForward>(
    model: M,
    tokenizer: TokenizerWrapper,
    kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);
    tokio::spawn(engine_loop(model, tokenizer, kv_cache_mgr, config, cmd_rx));
    EngineHandle { cmd_tx }
}

pub fn start_engine_with_draft<M: ModelForward, D: ModelForward>(
    target_model: M,
    draft_model: D,
    tokenizer: TokenizerWrapper,
    target_kv_cache: KVCacheManager,
    draft_kv_cache: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);
    tokio::spawn(engine_loop_speculative(
        target_model,
        draft_model,
        tokenizer,
        target_kv_cache,
        draft_kv_cache,
        config,
        cmd_rx,
    ));
    EngineHandle { cmd_tx }
}

// ─── Engine loop ───────────────────────────────────────────────────────────

/// Per-request state for the draft model's KV cache during speculative decoding.
struct DraftState {
    block_table: BlockTable,
    seqlen_offset: usize,
}

struct ActiveRequest {
    state: SequenceState,
    response: ResponseChannel,
    /// Number of tokens already sent via streaming
    num_streamed_tokens: usize,
    /// Length of text already sent via streaming (for incremental decode)
    streamed_text_len: usize,
    /// Draft model state (only present when using speculative decoding)
    draft_state: Option<DraftState>,
}

async fn engine_loop<M: ModelForward>(
    model: M,
    tokenizer: TokenizerWrapper,
    mut kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
) {
    let mut scheduler = Scheduler::new(config.scheduler_config);
    let mut requests: HashMap<RequestId, ActiveRequest> = HashMap::new();
    let mut next_id: RequestId = 0;
    let mut prefix_cache = if config.enable_prefix_caching {
        Some(PrefixCache::new(config.block_size))
    } else {
        None
    };

    loop {
        // Phase 1: Drain incoming commands (non-blocking)
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    if handle_command(
                        cmd,
                        &mut next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut requests,
                        config.block_size,
                        &mut prefix_cache,
                    ) {
                        return; // shutdown
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // Phase 2: If idle, block-wait for next command
        if scheduler.is_idle() {
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut requests,
                        config.block_size,
                        &mut prefix_cache,
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 3: Schedule
        let states_view: HashMap<RequestId, &SequenceState> =
            requests.iter().map(|(&id, r)| (id, &r.state)).collect();
        let output = scheduler.schedule(&states_view, kv_cache_mgr.num_free_blocks());

        // Phase 4: Handle preemptions
        for &req_id in &output.preempted_requests {
            let req = requests.get_mut(&req_id).unwrap();
            let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
            req.state.status = RequestStatus::Preempted;
            req.state.generated_token_ids.clear();
            req.state.seqlen_offset = 0;
        }

        // Phase 5: Execute prefills
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;
            if let Err(e) = execute_prefill(
                req_id,
                schedule.chunk_size,
                &model,
                &mut kv_cache_mgr,
                &mut requests,
            ) {
                finish_request_with_error(req_id, e, &mut scheduler, &mut requests);
            } else {
                send_stream_token(req_id, &tokenizer, &mut requests);
            }
        }

        // Phase 6: Execute batched decode (multi-step)
        if !output.decode_requests.is_empty() {
            let num_steps = config.multi_step_count.max(1);
            let mut active_decode_ids = output.decode_requests.clone();

            for _step in 0..num_steps {
                if active_decode_ids.is_empty() {
                    break;
                }

                let failed = execute_batched_decode(
                    &active_decode_ids,
                    &model,
                    &mut kv_cache_mgr,
                    &mut requests,
                );
                for req_id in failed {
                    active_decode_ids.retain(|&id| id != req_id);
                    finish_request_with_error(
                        req_id,
                        EngineError::Model("batched decode failed".to_string()),
                        &mut scheduler,
                        &mut requests,
                    );
                }

                // Remove sequences that finished (EOS, stop, or max_tokens)
                active_decode_ids.retain(|&id| {
                    requests
                        .get(&id)
                        .map(|r| check_finished(&r.state, &tokenizer).is_none())
                        .unwrap_or(false)
                });
            }

            // Send all accumulated stream tokens
            for &req_id in &output.decode_requests {
                if requests.contains_key(&req_id) {
                    send_stream_token(req_id, &tokenizer, &mut requests);
                }
            }
        }

        // Phase 7: Check completion and notify
        let mut finished = Vec::new();
        let scheduled_ids: Vec<RequestId> = output
            .prefill_requests
            .iter()
            .map(|s| s.request_id)
            .chain(output.decode_requests.iter().copied())
            .collect();
        for &req_id in &scheduled_ids {
            if let Some(req) = requests.get(&req_id) {
                if let Some(check) = check_finished(&req.state, &tokenizer) {
                    finished.push((req_id, check));
                }
            }
        }

        for (req_id, check) in finished {
            let mut req = requests.remove(&req_id).unwrap();
            scheduler.remove_request(req_id);
            let block_ids = req.state.block_table.release();
            if let Some(cache) = prefix_cache.as_mut() {
                cache.register_blocks(&req.state.prompt_token_ids, &block_ids);
                let to_free = cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = kv_cache_mgr.free_blocks(&to_free);
                }
            } else if !block_ids.is_empty() {
                let _ = kv_cache_mgr.free_blocks(&block_ids);
            }

            let mut text = tokenizer
                .decode(&req.state.generated_token_ids)
                .unwrap_or_default();
            if check.trim_bytes > 0 {
                let new_len = text.len().saturating_sub(check.trim_bytes);
                text.truncate(new_len);
            }

            if check.reason == FinishReason::Stop && !req.state.stop_token_ids.is_empty() {
                if let Some(&last) = req.state.generated_token_ids.last() {
                    if req.state.stop_token_ids.contains(&last) {
                        req.state.generated_token_ids.pop();
                    }
                }
            }

            match req.response {
                ResponseChannel::Complete(tx) => {
                    let result = GenerationResult {
                        request_id: req_id,
                        generated_text: text,
                        generated_token_ids: req.state.generated_token_ids,
                        finish_reason: check.reason,
                    };
                    let _ = tx.send(Ok(result));
                }
                ResponseChannel::Stream(tx) => {
                    let _ = tx.try_send(StreamEvent::Done {
                        finish_reason: check.reason,
                        generated_text: text,
                    });
                }
            }
        }

        tokio::task::yield_now().await;
    }
}

// ─── Speculative engine loop ──────────────────────────────────────────────

async fn engine_loop_speculative<M: ModelForward, D: ModelForward>(
    target_model: M,
    draft_model: D,
    tokenizer: TokenizerWrapper,
    mut target_kv_cache: KVCacheManager,
    mut draft_kv_cache: KVCacheManager,
    config: EngineConfig,
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
) {
    let num_speculative_tokens = config
        .speculative_config
        .as_ref()
        .map(|c| c.num_speculative_tokens)
        .unwrap_or(3);
    let mut scheduler = Scheduler::new(config.scheduler_config);
    let mut requests: HashMap<RequestId, ActiveRequest> = HashMap::new();
    let mut next_id: RequestId = 0;
    let mut prefix_cache = if config.enable_prefix_caching {
        Some(PrefixCache::new(config.block_size))
    } else {
        None
    };

    loop {
        // Phase 1: Drain incoming commands
        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => {
                    if handle_command(
                        cmd,
                        &mut next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut requests,
                        config.block_size,
                        &mut prefix_cache,
                    ) {
                        return;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        // Phase 2: Block-wait if idle
        if scheduler.is_idle() {
            match cmd_rx.recv().await {
                Some(cmd) => {
                    if handle_command(
                        cmd,
                        &mut next_id,
                        &tokenizer,
                        &mut scheduler,
                        &mut requests,
                        config.block_size,
                        &mut prefix_cache,
                    ) {
                        return;
                    }
                    continue;
                }
                None => return,
            }
        }

        // Phase 3: Schedule
        let states_view: HashMap<RequestId, &SequenceState> =
            requests.iter().map(|(&id, r)| (id, &r.state)).collect();
        let output = scheduler.schedule(&states_view, target_kv_cache.num_free_blocks());

        // Phase 4: Preemptions (free both target and draft caches)
        for &req_id in &output.preempted_requests {
            let req = requests.get_mut(&req_id).unwrap();
            let _ = target_kv_cache.free_request(&mut req.state.block_table);
            if let Some(ref mut ds) = req.draft_state {
                let freed = ds.block_table.release();
                let _ = draft_kv_cache.free_blocks(&freed);
            }
            req.draft_state = None;
            req.state.status = RequestStatus::Preempted;
            req.state.generated_token_ids.clear();
            req.state.seqlen_offset = 0;
        }

        // Phase 5: Prefills (target + draft)
        for schedule in &output.prefill_requests {
            let req_id = schedule.request_id;
            if let Err(e) = execute_prefill(
                req_id,
                schedule.chunk_size,
                &target_model,
                &mut target_kv_cache,
                &mut requests,
            ) {
                finish_request_with_error(req_id, e, &mut scheduler, &mut requests);
                continue;
            }
            if let Err(e) =
                execute_draft_prefill(req_id, &draft_model, &mut draft_kv_cache, &mut requests)
            {
                finish_request_with_error(req_id, e, &mut scheduler, &mut requests);
                continue;
            }
            send_stream_token(req_id, &tokenizer, &mut requests);
        }

        // Phase 6: Decode (speculative)
        for &req_id in &output.decode_requests {
            let result = execute_speculative_decode(
                req_id,
                &target_model,
                &draft_model,
                &mut target_kv_cache,
                &mut draft_kv_cache,
                &mut requests,
                num_speculative_tokens,
            );
            match result {
                Err(e) => finish_request_with_error(req_id, e, &mut scheduler, &mut requests),
                Ok(_) => send_stream_token(req_id, &tokenizer, &mut requests),
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
            if let Some(req) = requests.get(&req_id) {
                if let Some(check) = check_finished(&req.state, &tokenizer) {
                    finished.push((req_id, check));
                }
            }
        }

        for (req_id, check) in finished {
            let mut req = requests.remove(&req_id).unwrap();
            scheduler.remove_request(req_id);
            let block_ids = req.state.block_table.release();
            if let Some(cache) = prefix_cache.as_mut() {
                cache.register_blocks(&req.state.prompt_token_ids, &block_ids);
                let to_free = cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = target_kv_cache.free_blocks(&to_free);
                }
            } else if !block_ids.is_empty() {
                let _ = target_kv_cache.free_blocks(&block_ids);
            }
            if let Some(mut ds) = req.draft_state {
                let freed = ds.block_table.release();
                let _ = draft_kv_cache.free_blocks(&freed);
            }

            let mut text = tokenizer
                .decode(&req.state.generated_token_ids)
                .unwrap_or_default();
            if check.trim_bytes > 0 {
                let new_len = text.len().saturating_sub(check.trim_bytes);
                text.truncate(new_len);
            }

            if check.reason == FinishReason::Stop && !req.state.stop_token_ids.is_empty() {
                if let Some(&last) = req.state.generated_token_ids.last() {
                    if req.state.stop_token_ids.contains(&last) {
                        req.state.generated_token_ids.pop();
                    }
                }
            }

            match req.response {
                ResponseChannel::Complete(tx) => {
                    let result = GenerationResult {
                        request_id: req_id,
                        generated_text: text,
                        generated_token_ids: req.state.generated_token_ids,
                        finish_reason: check.reason,
                    };
                    let _ = tx.send(Ok(result));
                }
                ResponseChannel::Stream(tx) => {
                    let _ = tx.try_send(StreamEvent::Done {
                        finish_reason: check.reason,
                        generated_text: text,
                    });
                }
            }
        }

        tokio::task::yield_now().await;
    }
}

fn execute_draft_prefill<D: ModelForward>(
    req_id: RequestId,
    draft_model: &D,
    draft_kv_cache: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> Result<(), EngineError> {
    let req = requests.get_mut(&req_id).unwrap();
    let prompt_len = req.state.prompt_token_ids.len();

    let mut draft_block_table = BlockTable::new(draft_kv_cache.block_size());
    draft_kv_cache
        .allocate_for_request(&mut draft_block_table, prompt_len)
        .map_err(|e| EngineError::Cache(e.to_string()))?;
    let slot_mapping = draft_block_table.slot_mapping(0, prompt_len);

    let input = Tensor::from_vec(
        req.state.prompt_token_ids.clone(),
        (1, prompt_len),
        draft_model.device(),
    )
    .map_err(|e| EngineError::Model(e.to_string()))?;

    // Run draft model forward to populate its KV cache
    let _logits = draft_model
        .forward(&input, 0, draft_kv_cache, &draft_block_table, &slot_mapping)
        .map_err(|e| EngineError::Model(e.to_string()))?;

    draft_block_table.advance(prompt_len);

    req.draft_state = Some(DraftState {
        block_table: draft_block_table,
        seqlen_offset: prompt_len,
    });

    Ok(())
}

fn execute_speculative_decode<M: ModelForward, D: ModelForward>(
    req_id: RequestId,
    target_model: &M,
    draft_model: &D,
    target_kv_cache: &mut KVCacheManager,
    draft_kv_cache: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    num_speculative_tokens: usize,
) -> Result<(), EngineError> {
    let k = {
        let req = requests.get(&req_id).unwrap();
        let tokens_remaining = req
            .state
            .max_new_tokens
            .saturating_sub(req.state.num_generated());
        num_speculative_tokens.min(tokens_remaining.saturating_sub(1))
    };

    if k == 0 {
        return execute_decode(req_id, target_model, target_kv_cache, requests);
    }

    let req = requests.get_mut(&req_id).unwrap();
    let draft_state = req.draft_state.as_mut().unwrap();

    // --- Draft phase: generate K tokens ---
    let mut draft_tokens = Vec::with_capacity(k);
    let last_target_token = *req.state.generated_token_ids.last().unwrap();
    let mut draft_input_token = last_target_token;

    for _ in 0..k {
        draft_kv_cache
            .allocate_for_request(&mut draft_state.block_table, 1)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = draft_state
            .block_table
            .slot_mapping(draft_state.seqlen_offset, 1);

        let input = Tensor::new(&[[draft_input_token]], draft_model.device())
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let logits = draft_model
            .forward(
                &input,
                draft_state.seqlen_offset,
                draft_kv_cache,
                &draft_state.block_table,
                &slot_mapping,
            )
            .map_err(|e| EngineError::Model(e.to_string()))?;

        draft_state.block_table.advance(1);
        draft_state.seqlen_offset += 1;

        let seq_dim = logits.dims()[1];
        let logits = logits
            .narrow(1, seq_dim - 1, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let token = greedy_sample(&logits).map_err(|e| EngineError::Model(e.to_string()))?;
        draft_tokens.push(token);
        draft_input_token = token;
    }

    // --- Verify phase: target model forward on K+1 tokens ---
    let k_plus_1 = k + 1;
    target_kv_cache
        .allocate_for_request(&mut req.state.block_table, k_plus_1)
        .map_err(|e| EngineError::Cache(e.to_string()))?;
    let slot_mapping = req
        .state
        .block_table
        .slot_mapping(req.state.seqlen_offset, k_plus_1);

    let mut verify_input = vec![last_target_token];
    verify_input.extend_from_slice(&draft_tokens);
    let input = Tensor::from_vec(verify_input, (1, k_plus_1), target_model.device())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    let logits = target_model
        .forward(
            &input,
            req.state.seqlen_offset,
            target_kv_cache,
            &req.state.block_table,
            &slot_mapping,
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

    req.state.block_table.advance(k_plus_1);

    // --- Greedy verification ---
    let mut accepted = 0;
    for i in 0..k {
        let pos_logits = logits
            .narrow(1, i, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let target_token =
            greedy_sample(&pos_logits).map_err(|e| EngineError::Model(e.to_string()))?;
        if target_token == draft_tokens[i] {
            accepted += 1;
        } else {
            // First mismatch: target_token is the bonus token
            req.state.generated_token_ids.extend(&draft_tokens[..i]);
            req.state.generated_token_ids.push(target_token);
            break;
        }
    }

    if accepted == k {
        // All draft tokens accepted; bonus token from position K
        let bonus_logits = logits
            .narrow(1, k, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let bonus_token =
            greedy_sample(&bonus_logits).map_err(|e| EngineError::Model(e.to_string()))?;
        req.state.generated_token_ids.extend(&draft_tokens);
        req.state.generated_token_ids.push(bonus_token);
    }

    // --- Rollback: trim caches to actual accepted length ---
    let original_offset = req.state.seqlen_offset;
    let new_tokens = accepted + 1; // accepted drafts + bonus/correction
    let target_total = original_offset + new_tokens;
    let draft_total = original_offset + accepted; // draft doesn't have bonus token K,V

    // Target: allocated k+1, keep only accepted+1
    let target_freed = req.state.block_table.trim_to(target_total);
    if !target_freed.is_empty() {
        target_kv_cache
            .free_blocks(&target_freed)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
    }
    req.state.seqlen_offset = target_total;

    // Draft: advanced K, trim to accepted (bonus not in draft cache)
    let draft_state = req.draft_state.as_mut().unwrap();
    let draft_freed = draft_state.block_table.trim_to(draft_total);
    if !draft_freed.is_empty() {
        draft_kv_cache
            .free_blocks(&draft_freed)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
    }
    draft_state.seqlen_offset = draft_total;

    Ok(())
}

// ─── Helper functions ──────────────────────────────────────────────────────

/// Returns true if shutdown was requested.
fn handle_command(
    cmd: EngineCommand,
    next_id: &mut RequestId,
    tokenizer: &TokenizerWrapper,
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    block_size: usize,
    prefix_cache: &mut Option<PrefixCache>,
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

    // Check prefix cache for reusable blocks
    if let Some(cache) = prefix_cache.as_mut() {
        let (cached_blocks, _) = cache.match_prefix(&state.prompt_token_ids);
        if !cached_blocks.is_empty() {
            // Always leave at least 1 token for prefill to produce first decode logits.
            // Only use full blocks that fit within (prompt_len - 1) tokens.
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

fn send_error(response: ResponseChannel, error: EngineError) {
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

fn execute_prefill<M: ModelForward>(
    req_id: RequestId,
    chunk_size: usize,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> Result<(), EngineError> {
    let req = requests.get_mut(&req_id).unwrap();
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

    let logits = model
        .forward(
            &input,
            offset,
            kv_cache_mgr,
            &req.state.block_table,
            &slot_mapping,
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

    req.state.block_table.advance(actual_chunk);
    req.state.num_computed_tokens = chunk_end;
    req.state.seqlen_offset = chunk_end;

    if is_final_chunk {
        req.state.status = RequestStatus::Decoding;

        let seq_dim = logits.dims()[1];
        let logits = logits
            .narrow(1, seq_dim - 1, 1)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let next_token = sample_token(&logits, &mut req.state)?;
        req.state.generated_token_ids.push(next_token);
    } else {
        req.state.status = RequestStatus::Prefilling;
    }

    Ok(())
}

fn execute_decode<M: ModelForward>(
    req_id: RequestId,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> Result<(), EngineError> {
    let req = requests.get_mut(&req_id).unwrap();

    kv_cache_mgr
        .allocate_for_request(&mut req.state.block_table, 1)
        .map_err(|e| EngineError::Cache(e.to_string()))?;
    let slot_mapping = req
        .state
        .block_table
        .slot_mapping(req.state.seqlen_offset, 1);

    let last_token = *req.state.generated_token_ids.last().unwrap();
    let input = Tensor::new(&[[last_token]], model.device())
        .map_err(|e| EngineError::Model(e.to_string()))?;

    let logits = model
        .forward(
            &input,
            req.state.seqlen_offset,
            kv_cache_mgr,
            &req.state.block_table,
            &slot_mapping,
        )
        .map_err(|e| EngineError::Model(e.to_string()))?;

    req.state.block_table.advance(1);
    req.state.seqlen_offset += 1;

    let seq_dim = logits.dims()[1];
    let logits = logits
        .narrow(1, seq_dim - 1, 1)
        .map_err(|e| EngineError::Model(e.to_string()))?;
    let next_token = sample_token(&logits, &mut req.state)?;
    req.state.generated_token_ids.push(next_token);

    Ok(())
}

/// Execute batched decode for multiple sequences in a single forward pass.
/// Returns IDs of requests that failed (caller should remove them).
fn execute_batched_decode<M: ModelForward>(
    decode_request_ids: &[RequestId],
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> Vec<RequestId> {
    let mut failed = Vec::new();
    let mut batch_ids: Vec<RequestId> = Vec::with_capacity(decode_request_ids.len());

    // Step 1: Allocate blocks for each sequence
    for &req_id in decode_request_ids {
        let req = requests.get_mut(&req_id).unwrap();
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

    // Step 2: Collect input tokens and per-sequence metadata
    let mut token_ids: Vec<u32> = Vec::with_capacity(batch_ids.len());
    let mut sequences: Vec<DecodeSequenceMetadata> = Vec::with_capacity(batch_ids.len());

    for &req_id in &batch_ids {
        let req = requests.get(&req_id).unwrap();
        let last_token = *req.state.generated_token_ids.last().unwrap();
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
    let input = match Tensor::from_vec(token_ids, (batch_size, 1), model.device()) {
        Ok(t) => t,
        Err(_) => {
            failed.extend(&batch_ids);
            return failed;
        }
    };

    let logits = match model.forward_decode_batch(&input, &sequences, kv_cache_mgr) {
        Ok(l) => l,
        Err(_) => {
            failed.extend(&batch_ids);
            return failed;
        }
    };

    // Step 4: Extract logits and sample per-sequence
    let seq_dim = logits.dims()[1];
    let last_logits = match logits.narrow(1, seq_dim - 1, 1) {
        Ok(l) => match l.squeeze(1) {
            Ok(l) => l, // [batch, vocab]
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

    // Check if all sequences use greedy sampling for fast-path
    let all_greedy = batch_ids.iter().all(|&id| {
        requests
            .get(&id)
            .map(|r| r.state.sampling_params.is_greedy())
            .unwrap_or(true)
    });

    let sampled_tokens: Vec<u32> = if all_greedy {
        // Fast path: single argmax kernel
        let token_ids_tensor = match last_logits.argmax(1) {
            Ok(t) => t,
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        };
        match token_ids_tensor.to_vec1() {
            Ok(v) => v,
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        }
    } else {
        // Per-sequence sampling: transfer logits to CPU, sample individually
        let logits_cpu: Vec<f32> = match last_logits.to_vec2::<f32>() {
            Ok(v) => v.into_iter().flatten().collect(),
            Err(_) => {
                failed.extend(&batch_ids);
                return failed;
            }
        };
        let vocab_size = last_logits.dims()[1];
        let mut tokens = Vec::with_capacity(batch_ids.len());
        for (i, &req_id) in batch_ids.iter().enumerate() {
            let req = requests.get_mut(&req_id).unwrap();
            let logit_slice = &logits_cpu[i * vocab_size..(i + 1) * vocab_size];
            let token = sampling::sample(
                logit_slice,
                &req.state.sampling_params,
                &req.state.generated_token_ids,
                &mut req.state.sampler_state,
            );
            tokens.push(token);
        }
        tokens
    };

    // Step 5: Update state with sampled tokens
    for (i, &req_id) in batch_ids.iter().enumerate() {
        let req = requests.get_mut(&req_id).unwrap();
        req.state.block_table.advance(1);
        req.state.seqlen_offset += 1;
        req.state.generated_token_ids.push(sampled_tokens[i]);
    }

    failed
}

/// Result of checking if generation should stop, including optional text trim length.
struct FinishCheck {
    reason: FinishReason,
    /// Number of bytes to trim from generated text (for stop string not included in output).
    trim_bytes: usize,
}

fn check_finished(state: &SequenceState, tokenizer: &TokenizerWrapper) -> Option<FinishCheck> {
    let last_token = *state.generated_token_ids.last()?;

    // Check EOS
    if last_token == state.eos_token_id {
        return Some(FinishCheck {
            reason: FinishReason::Eos,
            trim_bytes: 0,
        });
    }

    // Check stop token IDs
    if state.stop_token_ids.contains(&last_token) {
        return Some(FinishCheck {
            reason: FinishReason::Stop,
            trim_bytes: 0,
        });
    }

    // Check stop strings
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

    // Check max length
    if state.num_generated() >= state.max_new_tokens {
        return Some(FinishCheck {
            reason: FinishReason::Length,
            trim_bytes: 0,
        });
    }

    None
}

/// Sample a single token from logits tensor using the sequence's sampling params.
fn sample_token(logits: &Tensor, state: &mut SequenceState) -> Result<u32, EngineError> {
    if state.sampling_params.is_greedy() {
        greedy_sample(logits).map_err(|e| EngineError::Model(e.to_string()))
    } else {
        let logits_vec: Vec<f32> = logits
            .squeeze(0)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_vec1())
            .map_err(|e| EngineError::Model(e.to_string()))?;
        Ok(sampling::sample(
            &logits_vec,
            &state.sampling_params,
            &state.generated_token_ids,
            &mut state.sampler_state,
        ))
    }
}

fn finish_request_with_error(
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

/// Send streaming token events for a request (no-op for non-streaming requests).
/// Sends one StreamEvent::Token per un-streamed token, using incremental decode
/// to compute text deltas correctly even when multiple tokens are generated at once
/// (e.g. speculative decoding).
fn send_stream_token(
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

pub fn greedy_sample(logits: &Tensor) -> anyhow::Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?;
    let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
    Ok(token_id)
}

// ─── Legacy API (kept for backward compatibility) ──────────────────────────

pub struct GenerationParams {
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            eos_token_id: 151645,
        }
    }
}

pub fn generate<M: ModelForward>(
    model: &M,
    tokenizer: &TokenizerWrapper,
    prompt: &str,
    _config: &crate::config::ModelConfig,
    params: &GenerationParams,
    kv_cache_mgr: &mut KVCacheManager,
    device: &Device,
) -> anyhow::Result<String> {
    let prompt_ids = tokenizer.encode(prompt)?;
    let generated_ids = generate_tokens(model, &prompt_ids, params, kv_cache_mgr, device)?;
    let output = tokenizer.decode(&generated_ids)?;
    Ok(output)
}

pub fn generate_tokens<M: ModelForward>(
    model: &M,
    prompt_ids: &[u32],
    params: &GenerationParams,
    kv_cache_mgr: &mut KVCacheManager,
    device: &Device,
) -> anyhow::Result<Vec<u32>> {
    let mut block_table = BlockTable::new(kv_cache_mgr.block_size());

    kv_cache_mgr.allocate_for_request(&mut block_table, prompt_ids.len())?;
    let slot_mapping = block_table.slot_mapping(0, prompt_ids.len());

    let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;
    let logits = model.forward(&input, 0, kv_cache_mgr, &block_table, &slot_mapping)?;
    block_table.advance(prompt_ids.len());

    let seq_dim = logits.dims()[1];
    let logits = logits.narrow(1, seq_dim - 1, 1)?;
    let mut next_token = greedy_sample(&logits)?;
    let mut generated = vec![next_token];
    let mut seqlen_offset = prompt_ids.len();

    for _ in 1..params.max_new_tokens {
        if next_token == params.eos_token_id {
            break;
        }
        kv_cache_mgr.allocate_for_request(&mut block_table, 1)?;
        let slot_mapping = block_table.slot_mapping(seqlen_offset, 1);

        let input = Tensor::new(&[[next_token]], device)?;
        let logits = model.forward(
            &input,
            seqlen_offset,
            kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )?;
        block_table.advance(1);

        let seq_dim = logits.dims()[1];
        let logits = logits.narrow(1, seq_dim - 1, 1)?;
        next_token = greedy_sample(&logits)?;
        generated.push(next_token);
        seqlen_offset += 1;
    }

    kv_cache_mgr.free_request(&mut block_table)?;
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use candle_core::DType;

    // ─── MockModel ─────────────────────────────────────────────────────────

    struct MockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
    }

    impl MockModel {
        fn new(output_token: u32, vocab_size: usize) -> Self {
            Self {
                output_token,
                vocab_size,
                device: Device::Cpu,
            }
        }
    }

    impl ModelForward for MockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    /// Mock that emits EOS after N calls
    struct CountingMockModel {
        output_token: u32,
        eos_token: u32,
        vocab_size: usize,
        device: Device,
        eos_after: usize,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl CountingMockModel {
        fn new(output_token: u32, eos_token: u32, vocab_size: usize, eos_after: usize) -> Self {
            Self {
                output_token,
                eos_token,
                vocab_size,
                device: Device::Cpu,
                eos_after,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl ModelForward for CountingMockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let count = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let token = if count >= self.eos_after {
                self.eos_token
            } else {
                self.output_token
            };
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + token as usize] = 100.0;
            }
            Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    /// Mock model that outputs a predefined token at each absolute position.
    /// Position = seqlen_offset + i (for the i-th token in the input batch).
    struct SequenceMockModel {
        token_sequence: Vec<u32>,
        vocab_size: usize,
        device: Device,
    }

    impl SequenceMockModel {
        fn new(token_sequence: Vec<u32>, vocab_size: usize) -> Self {
            Self {
                token_sequence,
                vocab_size,
                device: Device::Cpu,
            }
        }
    }

    impl ModelForward for SequenceMockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            seqlen_offset: usize,
            _kv_cache_mgr: &KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for i in 0..seq_len {
                let pos = seqlen_offset + i;
                let token = if pos < self.token_sequence.len() {
                    self.token_sequence[pos]
                } else {
                    *self.token_sequence.last().unwrap_or(&0)
                };
                logits_vec[i * self.vocab_size + token as usize] = 100.0;
            }
            Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn test_cache_config() -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    fn test_engine_config() -> EngineConfig {
        EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
        }
    }

    // Helper to run engine tests without real tokenizer:
    // Directly submit pre-tokenized requests via the internal channel
    async fn run_engine_with_pretokenized<M: ModelForward>(
        model: M,
        kv_cache_mgr: KVCacheManager,
        config: EngineConfig,
        requests: Vec<(Vec<u32>, usize, u32)>, // (prompt_ids, max_tokens, eos_token)
    ) -> Vec<Result<GenerationResult, EngineError>> {
        let block_size = config.block_size;
        let num_requests = requests.len();

        let initial_free_blocks = kv_cache_mgr.num_free_blocks();
        let handle = tokio::spawn(async move {
            let tokenizer = TokenizerWrapper::for_testing(1000);
            let mut scheduler = Scheduler::new(config.scheduler_config);
            let mut active: HashMap<RequestId, ActiveRequest> = HashMap::new();
            let mut kv_cache_mgr = kv_cache_mgr;

            for (id, (prompt_ids, max_tokens, eos_token)) in (0u64..).zip(requests.iter()) {
                let state = SequenceState::new(
                    id,
                    prompt_ids.clone(),
                    *max_tokens,
                    *eos_token,
                    block_size,
                    id,
                );
                scheduler.add_request(id);
                let (tx, _rx) = oneshot::channel();
                active.insert(
                    id,
                    ActiveRequest {
                        state,
                        response: ResponseChannel::Complete(tx),
                        num_streamed_tokens: 0,
                        streamed_text_len: 0,
                        draft_state: None,
                    },
                );
            }

            let mut results: HashMap<RequestId, GenerationResult> = HashMap::new();

            for _ in 0..10000 {
                if scheduler.is_idle() {
                    break;
                }

                let states: HashMap<RequestId, &SequenceState> =
                    active.iter().map(|(&id, r)| (id, &r.state)).collect();
                let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

                for &req_id in &output.preempted_requests {
                    let req = active.get_mut(&req_id).unwrap();
                    let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
                    req.state.status = RequestStatus::Preempted;
                    req.state.generated_token_ids.clear();
                    req.state.seqlen_offset = 0;
                }

                for schedule in &output.prefill_requests {
                    let _ = execute_prefill(
                        schedule.request_id,
                        schedule.chunk_size,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                    );
                }

                if !output.decode_requests.is_empty() {
                    let _failed = execute_batched_decode(
                        &output.decode_requests,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                    );
                }

                let mut finished = Vec::new();
                let scheduled_ids: Vec<RequestId> = output
                    .prefill_requests
                    .iter()
                    .map(|s| s.request_id)
                    .chain(output.decode_requests.iter().copied())
                    .collect();
                for &req_id in &scheduled_ids {
                    if let Some(req) = active.get(&req_id) {
                        if let Some(check) = check_finished(&req.state, &tokenizer) {
                            finished.push((req_id, check));
                        }
                    }
                }

                for (req_id, check) in finished {
                    let req = active.remove(&req_id).unwrap();
                    scheduler.remove_request(req_id);
                    let mut bt = req.state.block_table;
                    let _ = kv_cache_mgr.free_request(&mut bt);
                    results.insert(
                        req_id,
                        GenerationResult {
                            request_id: req_id,
                            generated_text: String::new(),
                            generated_token_ids: req.state.generated_token_ids,
                            finish_reason: check.reason,
                        },
                    );
                }
            }

            // Verify all blocks freed
            assert_eq!(kv_cache_mgr.num_free_blocks(), initial_free_blocks);
            results
        });

        let results = handle.await.unwrap();
        let mut output: Vec<Result<GenerationResult, EngineError>> = Vec::new();
        for i in 0..num_requests {
            if let Some(r) = results.get(&(i as u64)) {
                output.push(Ok(r.clone()));
            } else {
                output.push(Err(EngineError::Shutdown));
            }
        }
        output
    }

    #[tokio::test]
    async fn single_request_generates_tokens() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![1, 2, 3], 5, 999)],
        )
        .await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42, 42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn eos_stops_generation() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        // EOS when call_count >= 2 (prefill + 1 decode = 2 normal, 3rd call is EOS)
        let model = CountingMockModel::new(42, 999, 1000, 2);
        let config = test_engine_config();

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![1, 2, 3], 10, 999)],
        )
        .await;

        let result = results[0].as_ref().unwrap();
        // 3 calls: tokens 42, 42, 999(EOS)
        assert_eq!(result.generated_token_ids, vec![42, 42, 999]);
        assert_eq!(result.finish_reason, FinishReason::Eos);
    }

    #[tokio::test]
    async fn multiple_concurrent_requests() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![
                (vec![1, 2, 3], 3, 999),
                (vec![4, 5], 4, 999),
                (vec![6, 7, 8, 9], 2, 999),
            ],
        )
        .await;

        assert_eq!(results[0].as_ref().unwrap().generated_token_ids.len(), 3);
        assert_eq!(results[1].as_ref().unwrap().generated_token_ids.len(), 4);
        assert_eq!(results[2].as_ref().unwrap().generated_token_ids.len(), 2);
    }

    #[tokio::test]
    async fn blocks_freed_after_completion() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();

        // The run_engine_with_pretokenized helper asserts num_free_blocks == 64 at the end
        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 20, 999)],
        )
        .await;

        assert_eq!(results[0].as_ref().unwrap().generated_token_ids.len(), 20);
    }

    #[tokio::test]
    async fn preemption_under_memory_pressure() {
        // Small cache: only 4 blocks of size 4 = 16 token slots
        let cache_config = CacheConfig {
            block_size: 4,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
        };
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 4,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        // 3 requests each needing 2 tokens prompt → 1 block each
        // With 4 blocks total, all 3 fit initially (3 blocks for prompts)
        // As they decode, they'll need more blocks → preemption
        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![
                (vec![1, 2], 3, 999),
                (vec![3, 4], 3, 999),
                (vec![5, 6], 3, 999),
            ],
        )
        .await;

        // All should eventually complete despite memory pressure
        for result in &results {
            let r = result.as_ref().unwrap();
            assert_eq!(r.generated_token_ids.len(), 3);
            assert_eq!(r.finish_reason, FinishReason::Length);
        }
    }

    // ─── Streaming tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn stream_receives_all_tokens() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 5,
            eos_token_id: 999,
            ..Default::default()
        };
        let mut rx = handle.generate_stream(req).await.unwrap();

        let mut token_events = Vec::new();
        let mut done_event = None;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token { token_id, .. } => token_events.push(token_id),
                StreamEvent::Done {
                    finish_reason,
                    generated_text,
                } => {
                    done_event = Some((finish_reason, generated_text));
                    break;
                }
                StreamEvent::Error { error } => panic!("unexpected error: {error}"),
            }
        }

        assert_eq!(token_events, vec![42, 42, 42, 42, 42]);
        let (reason, _text) = done_event.expect("should receive Done event");
        assert_eq!(reason, FinishReason::Length);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn stream_eos_sends_done() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        // EOS after 2 calls: prefill → 42, decode → 999(EOS)
        let model = CountingMockModel::new(42, 999, 1000, 2);
        let config = test_engine_config();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 10,
            eos_token_id: 999,
            ..Default::default()
        };
        let mut rx = handle.generate_stream(req).await.unwrap();

        let mut token_ids = Vec::new();
        let mut done_reason = None;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token { token_id, .. } => token_ids.push(token_id),
                StreamEvent::Done { finish_reason, .. } => {
                    done_reason = Some(finish_reason);
                    break;
                }
                StreamEvent::Error { error } => panic!("unexpected error: {error}"),
            }
        }

        assert_eq!(token_ids, vec![42, 42, 999]);
        assert_eq!(done_reason, Some(FinishReason::Eos));

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn stream_client_disconnect_no_crash() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 100,
            eos_token_id: 999,
            ..Default::default()
        };
        let rx = handle.generate_stream(req).await.unwrap();
        // Drop the receiver immediately — engine should not crash
        drop(rx);

        // Submit another request to verify engine is still alive
        let req2 = GenerationRequest {
            prompt: "t4 t5".to_string(),
            max_new_tokens: 3,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(req2).await.unwrap();
        assert_eq!(result.generated_token_ids.len(), 3);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn stream_and_nonstream_concurrent() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Start a streaming request
        let h1 = handle.clone();
        let stream_task = tokio::spawn(async move {
            let req = GenerationRequest {
                prompt: "t1 t2".to_string(),
                max_new_tokens: 4,
                eos_token_id: 999,
                ..Default::default()
            };
            let mut rx = h1.generate_stream(req).await.unwrap();
            let mut count = 0;
            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token { .. } => count += 1,
                    StreamEvent::Done { .. } => break,
                    StreamEvent::Error { error } => panic!("error: {error}"),
                }
            }
            count
        });

        // Start a non-streaming request concurrently
        let h2 = handle.clone();
        let nonstream_task = tokio::spawn(async move {
            let req = GenerationRequest {
                prompt: "t3 t4 t5".to_string(),
                max_new_tokens: 3,
                eos_token_id: 999,
                ..Default::default()
            };
            h2.generate(req).await.unwrap()
        });

        let stream_count = stream_task.await.unwrap();
        let nonstream_result = nonstream_task.await.unwrap();

        assert_eq!(stream_count, 4);
        assert_eq!(nonstream_result.generated_token_ids.len(), 3);

        handle.shutdown().await.unwrap();
    }

    // ─── Speculative decode tests ───────────────────────────────────────────

    #[tokio::test]
    async fn speculative_all_accepted() {
        // Both models produce the same tokens → all drafts accepted, K+1 tokens per round
        // Prompt = [1, 2, 3] (3 tokens), positions 0,1,2 are prompt
        // Position 2 determines first generated token (from prefill narrow to last)
        // Positions 3,4,5,6 determine subsequent tokens
        let seq = vec![0, 0, 10, 20, 30, 40, 50];
        let target = SequenceMockModel::new(seq.clone(), 1000);
        let draft = SequenceMockModel::new(seq, 1000);

        let cache_config = test_cache_config();
        let target_cache = KVCacheManager::new(&cache_config).unwrap();
        let draft_cache = KVCacheManager::new(&cache_config).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens: 3,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        let handle =
            start_engine_with_draft(target, draft, tokenizer, target_cache, draft_cache, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 5,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(req).await.unwrap();

        // All 5 tokens should be generated: prefill→10, then speculative→20,30,40,50
        assert_eq!(result.generated_token_ids, vec![10, 20, 30, 40, 50]);
        assert_eq!(result.finish_reason, FinishReason::Length);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn speculative_all_rejected() {
        // Draft produces different tokens than target → 0 accepted each round
        // Target: position 2→10, 3→20, 4→30, 5→40, 6→50, 7→60, 8→70
        // Draft: position 2→10 (same for prefill), but 3→100, 4→100, ... (all wrong)
        let target_seq = vec![0, 0, 10, 20, 30, 40, 50, 60, 70];
        let draft_seq = vec![0, 0, 10, 100, 100, 100, 100, 100, 100];
        let target = SequenceMockModel::new(target_seq, 1000);
        let draft = SequenceMockModel::new(draft_seq, 1000);

        let cache_config = test_cache_config();
        let target_cache = KVCacheManager::new(&cache_config).unwrap();
        let draft_cache = KVCacheManager::new(&cache_config).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens: 3,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        let handle =
            start_engine_with_draft(target, draft, tokenizer, target_cache, draft_cache, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 5,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(req).await.unwrap();

        // Even with all rejections, should still produce correct target output
        assert_eq!(result.generated_token_ids, vec![10, 20, 30, 40, 50]);
        assert_eq!(result.finish_reason, FinishReason::Length);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn speculative_partial_accept() {
        // Draft matches target at positions 3,4 but not 5
        // Target: pos 2→10, 3→20, 4→30, 5→40, 6→50
        // Draft: pos 2→10, 3→20, 4→30, 5→99 (mismatch at position 5)
        let target_seq = vec![0, 0, 10, 20, 30, 40, 50, 60];
        let draft_seq = vec![0, 0, 10, 20, 30, 99, 99, 99];
        let target = SequenceMockModel::new(target_seq, 1000);
        let draft = SequenceMockModel::new(draft_seq, 1000);

        let cache_config = test_cache_config();
        let target_cache = KVCacheManager::new(&cache_config).unwrap();
        let draft_cache = KVCacheManager::new(&cache_config).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens: 3,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        let handle =
            start_engine_with_draft(target, draft, tokenizer, target_cache, draft_cache, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 5,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(req).await.unwrap();

        // Should produce target's output regardless of partial acceptance
        assert_eq!(result.generated_token_ids, vec![10, 20, 30, 40, 50]);
        assert_eq!(result.finish_reason, FinishReason::Length);

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn speculative_streaming() {
        // Test that speculative decode works with streaming (multi-token batches)
        let seq = vec![0, 0, 10, 20, 30, 40, 50];
        let target = SequenceMockModel::new(seq.clone(), 1000);
        let draft = SequenceMockModel::new(seq, 1000);

        let cache_config = test_cache_config();
        let target_cache = KVCacheManager::new(&cache_config).unwrap();
        let draft_cache = KVCacheManager::new(&cache_config).unwrap();
        let tokenizer = TokenizerWrapper::for_testing(1000);

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens: 3,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        let handle =
            start_engine_with_draft(target, draft, tokenizer, target_cache, draft_cache, config);

        let req = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 5,
            eos_token_id: 999,
            ..Default::default()
        };
        let mut rx = handle.generate_stream(req).await.unwrap();

        let mut token_ids = Vec::new();
        let mut done_reason = None;
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token { token_id, .. } => token_ids.push(token_id),
                StreamEvent::Done { finish_reason, .. } => {
                    done_reason = Some(finish_reason);
                    break;
                }
                StreamEvent::Error { error } => panic!("unexpected error: {error}"),
            }
        }

        assert_eq!(token_ids, vec![10, 20, 30, 40, 50]);
        assert_eq!(done_reason, Some(FinishReason::Length));

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn chunked_prefill_generates_output() {
        // A 25-token prompt with budget 10 requires multiple prefill chunks
        let seq = vec![0u32; 25]
            .into_iter()
            .chain(vec![10, 20, 30])
            .collect::<Vec<_>>();
        let model = SequenceMockModel::new(seq, 1000);
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 10,
                enable_chunked_prefill: true,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![0u32; 25], 3, 999)],
        )
        .await;

        assert_eq!(results.len(), 1);
        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids.len(), 3);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_with_concurrent_decode() {
        // Submit two requests: one already decoding, one new with long prompt
        let seq = vec![0u32; 30]
            .into_iter()
            .chain(vec![10, 20, 30, 40, 50])
            .collect::<Vec<_>>();
        let model = SequenceMockModel::new(seq, 1000);
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();

        let config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 15,
                enable_chunked_prefill: true,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

        // Two requests: one short (completes prefill in one step) and one long
        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![
                (vec![0u32; 5], 3, 999),  // short: fits in one chunk
                (vec![0u32; 25], 3, 999), // long: needs multiple chunks
            ],
        )
        .await;

        assert_eq!(results.len(), 2);
        for r in &results {
            let result = r.as_ref().unwrap();
            assert_eq!(result.generated_token_ids.len(), 3);
            assert_eq!(result.finish_reason, FinishReason::Length);
        }
    }

    // ─── Prefix caching tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn prefix_cache_skips_cached_prefix() {
        use crate::kv_cache::prefix_cache::PrefixCache;

        let cache_config = test_cache_config(); // block_size=16
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let block_size = cache_config.block_size;

        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
            enable_chunked_prefill: false,
        });
        let mut active: HashMap<RequestId, ActiveRequest> = HashMap::new();
        let mut prefix_cache = PrefixCache::new(block_size);
        let tokenizer = TokenizerWrapper::for_testing(1000);

        // Request 1: prompt of 32 tokens (2 full blocks)
        let prompt = vec![1u32; 32];
        let state = SequenceState::new(0, prompt.clone(), 3, 999, block_size, 0);
        scheduler.add_request(0);
        let (tx, _rx) = oneshot::channel();
        active.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                draft_state: None,
            },
        );

        // Run request 1 to completion
        for _ in 0..100 {
            if scheduler.is_idle() {
                break;
            }
            let states: HashMap<RequestId, &SequenceState> =
                active.iter().map(|(&id, r)| (id, &r.state)).collect();
            let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

            for schedule in &output.prefill_requests {
                let _ = execute_prefill(
                    schedule.request_id,
                    schedule.chunk_size,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }
            if !output.decode_requests.is_empty() {
                let _ = execute_batched_decode(
                    &output.decode_requests,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }

            let scheduled_ids: Vec<RequestId> = output
                .prefill_requests
                .iter()
                .map(|s| s.request_id)
                .chain(output.decode_requests.iter().copied())
                .collect();
            let mut finished = Vec::new();
            for &req_id in &scheduled_ids {
                if let Some(req) = active.get(&req_id) {
                    if let Some(check) = check_finished(&req.state, &tokenizer) {
                        finished.push((req_id, check));
                    }
                }
            }
            for (req_id, _check) in finished {
                let mut req = active.remove(&req_id).unwrap();
                scheduler.remove_request(req_id);
                // Register blocks in prefix cache, then free uncached
                let block_ids = req.state.block_table.release();
                prefix_cache.register_blocks(&req.state.prompt_token_ids, &block_ids);
                let to_free = prefix_cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = kv_cache_mgr.free_blocks(&to_free);
                }
            }
        }

        // Verify prefix cache has entries
        assert_eq!(prefix_cache.num_cached_blocks(), 2); // 32 tokens / 16 block_size = 2 full blocks

        // Request 2: same prompt prefix → should reuse 1 cached block (leave last block for prefill)
        let mut state2 = SequenceState::new(1, prompt.clone(), 3, 999, block_size, 1);

        // Simulate what admit_request does with prefix cache
        let (cached_blocks, num_cached) = prefix_cache.match_prefix(&state2.prompt_token_ids);
        assert_eq!(cached_blocks.len(), 2);
        assert_eq!(num_cached, 32);

        // Apply the same logic as admit_request: leave at least 1 token for prefill
        let max_cached = state2.prompt_token_ids.len().saturating_sub(1);
        let blocks_to_use = (max_cached / block_size).min(cached_blocks.len());
        assert_eq!(blocks_to_use, 1); // 31/16 = 1 block

        let tokens_covered = blocks_to_use * block_size;
        state2
            .block_table
            .append_blocks(&cached_blocks[..blocks_to_use]);
        state2.block_table.advance(tokens_covered);
        state2.num_computed_tokens = tokens_covered;
        state2.seqlen_offset = tokens_covered;

        // 16 tokens cached, 16 remaining for prefill
        assert_eq!(state2.num_computed_tokens, 16);

        scheduler.add_request(1);
        let (tx2, _rx2) = oneshot::channel();
        active.insert(
            1,
            ActiveRequest {
                state: state2,
                response: ResponseChannel::Complete(tx2),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                draft_state: None,
            },
        );

        // Run request 2 — needs prefill for remaining 16 tokens, then decode
        let mut saw_prefill = false;
        let mut saw_decode = false;
        for _ in 0..100 {
            if scheduler.is_idle() {
                break;
            }
            let states: HashMap<RequestId, &SequenceState> =
                active.iter().map(|(&id, r)| (id, &r.state)).collect();
            let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

            if !output.prefill_requests.is_empty() {
                saw_prefill = true;
                for schedule in &output.prefill_requests {
                    let _ = execute_prefill(
                        schedule.request_id,
                        schedule.chunk_size,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                    );
                }
            }
            if !output.decode_requests.is_empty() {
                saw_decode = true;
                let _ = execute_batched_decode(
                    &output.decode_requests,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }

            let scheduled_ids: Vec<RequestId> = output
                .prefill_requests
                .iter()
                .map(|s| s.request_id)
                .chain(output.decode_requests.iter().copied())
                .collect();
            let mut finished = Vec::new();
            for &req_id in &scheduled_ids {
                if let Some(req) = active.get(&req_id) {
                    if let Some(check) = check_finished(&req.state, &tokenizer) {
                        finished.push((req_id, check));
                    }
                }
            }
            for (req_id, _check) in finished {
                let mut req = active.remove(&req_id).unwrap();
                scheduler.remove_request(req_id);
                let block_ids = req.state.block_table.release();
                let to_free = prefix_cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = kv_cache_mgr.free_blocks(&to_free);
                }
            }
        }

        // Partial prefix means some prefill was still needed
        assert!(saw_prefill);
        assert!(saw_decode);
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn prefix_cache_partial_prefix_match() {
        use crate::kv_cache::prefix_cache::PrefixCache;

        let cache_config = test_cache_config(); // block_size=16
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let block_size = cache_config.block_size;

        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
            enable_chunked_prefill: false,
        });
        let mut active: HashMap<RequestId, ActiveRequest> = HashMap::new();
        let mut prefix_cache = PrefixCache::new(block_size);
        let tokenizer = TokenizerWrapper::for_testing(1000);

        // Request 1: 32 token prompt
        let prompt1: Vec<u32> = (0..32).collect();
        let state = SequenceState::new(0, prompt1.clone(), 2, 999, block_size, 0);
        scheduler.add_request(0);
        let (tx, _rx) = oneshot::channel();
        active.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                draft_state: None,
            },
        );

        // Run to completion
        for _ in 0..100 {
            if scheduler.is_idle() {
                break;
            }
            let states: HashMap<RequestId, &SequenceState> =
                active.iter().map(|(&id, r)| (id, &r.state)).collect();
            let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());
            for schedule in &output.prefill_requests {
                let _ = execute_prefill(
                    schedule.request_id,
                    schedule.chunk_size,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }
            if !output.decode_requests.is_empty() {
                let _ = execute_batched_decode(
                    &output.decode_requests,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }
            let scheduled_ids: Vec<RequestId> = output
                .prefill_requests
                .iter()
                .map(|s| s.request_id)
                .chain(output.decode_requests.iter().copied())
                .collect();
            let mut finished = Vec::new();
            for &req_id in &scheduled_ids {
                if let Some(req) = active.get(&req_id) {
                    if let Some(check) = check_finished(&req.state, &tokenizer) {
                        finished.push((req_id, check));
                    }
                }
            }
            for (req_id, _check) in finished {
                let mut req = active.remove(&req_id).unwrap();
                scheduler.remove_request(req_id);
                let block_ids = req.state.block_table.release();
                prefix_cache.register_blocks(&req.state.prompt_token_ids, &block_ids);
                let to_free = prefix_cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = kv_cache_mgr.free_blocks(&to_free);
                }
            }
        }

        assert_eq!(prefix_cache.num_cached_blocks(), 2);

        // Request 2: same first 16 tokens, different second block
        let mut prompt2: Vec<u32> = (0..16).collect(); // same first block
        prompt2.extend(100..116u32); // different second block
        prompt2.extend(200..210u32); // extra tokens (partial third block)

        let mut state2 = SequenceState::new(1, prompt2.clone(), 2, 999, block_size, 1);
        let (cached_blocks, num_cached) = prefix_cache.match_prefix(&state2.prompt_token_ids);

        // Only first block matches (second block has different tokens)
        assert_eq!(cached_blocks.len(), 1);
        assert_eq!(num_cached, 16);

        state2.block_table.append_blocks(&cached_blocks);
        state2.block_table.advance(num_cached);
        state2.num_computed_tokens = num_cached;
        state2.seqlen_offset = num_cached;

        // Still needs prefill for remaining 26 tokens (16 + 10 uncached)
        assert_eq!(state2.num_computed_tokens, 16);
        assert_eq!(state2.prompt_token_ids.len(), 42);

        scheduler.add_request(1);
        let (tx2, _rx2) = oneshot::channel();
        active.insert(
            1,
            ActiveRequest {
                state: state2,
                response: ResponseChannel::Complete(tx2),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                draft_state: None,
            },
        );

        // Run request 2 — should prefill remaining tokens then decode
        let mut saw_prefill = false;
        for _ in 0..100 {
            if scheduler.is_idle() {
                break;
            }
            let states: HashMap<RequestId, &SequenceState> =
                active.iter().map(|(&id, r)| (id, &r.state)).collect();
            let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

            if !output.prefill_requests.is_empty() {
                saw_prefill = true;
                for schedule in &output.prefill_requests {
                    let _ = execute_prefill(
                        schedule.request_id,
                        schedule.chunk_size,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                    );
                }
            }
            if !output.decode_requests.is_empty() {
                let _ = execute_batched_decode(
                    &output.decode_requests,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                );
            }
            let scheduled_ids: Vec<RequestId> = output
                .prefill_requests
                .iter()
                .map(|s| s.request_id)
                .chain(output.decode_requests.iter().copied())
                .collect();
            let mut finished = Vec::new();
            for &req_id in &scheduled_ids {
                if let Some(req) = active.get(&req_id) {
                    if let Some(check) = check_finished(&req.state, &tokenizer) {
                        finished.push((req_id, check));
                    }
                }
            }
            for (req_id, _check) in finished {
                let mut req = active.remove(&req_id).unwrap();
                scheduler.remove_request(req_id);
                let block_ids = req.state.block_table.release();
                let to_free = prefix_cache.release_blocks(&req.state.prompt_token_ids, &block_ids);
                if !to_free.is_empty() {
                    let _ = kv_cache_mgr.free_blocks(&to_free);
                }
            }
        }

        // With partial match, remaining tokens needed prefill
        assert!(saw_prefill);
        assert!(active.is_empty());
    }

    // ─── Legacy API test (requires model, ignored) ─────────────────────────

    #[test]
    #[ignore]
    fn greedy_generation_produces_output() {
        use crate::loader;

        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let device = Device::Cpu;
        let vb = loader::load_weights(&files.weights, DType::F32, &device).expect("load weights");
        let model = crate::models::Qwen3ForCausalLM::new(&files.config, vb).expect("build model");
        let tokenizer = TokenizerWrapper::from_file(&files.tokenizer).expect("load tokenizer");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: files.config.num_hidden_layers,
            num_kv_heads: files.config.num_key_value_heads,
            head_dim: files.config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");

        let params = GenerationParams {
            max_new_tokens: 10,
            eos_token_id: files.config.eos_token_id,
        };

        let output = generate(
            &model,
            &tokenizer,
            "Hello",
            &files.config,
            &params,
            &mut kv_cache_mgr,
            &device,
        )
        .expect("generation");

        assert!(!output.is_empty(), "generation produced empty output");
    }
}
