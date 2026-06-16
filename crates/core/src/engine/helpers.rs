//! Helper functions for engine operations.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{IndexOp, Tensor};

use crate::kv_cache::{BlockTable, KVCacheManager};
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

// ─── Engine-step profiler ────────────────────────────────────────────────────
//
// `VLLM_PROFILE_STEP=1` enables per-stage timing inside
// `execute_batched_decode_with_graph` and dumps a μs/step breakdown via
// `tracing::info!` once per `STEP_PROFILE_EVERY` steps, broken down by
// batch_size so c=1 vs c=4 vs c=8 cohabit a single log without losing
// resolution. Each timed block ends with a CUDA stream sync so the GPU
// stages (forward, sampling) carry the actual kernel time, not the
// launch latency.
pub(crate) mod step_profile {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub const STEP_PROFILE_EVERY: usize = 200;

    static ENABLED: OnceLock<bool> = OnceLock::new();

    pub fn enabled() -> bool {
        *ENABLED.get_or_init(|| std::env::var("VLLM_PROFILE_STEP").is_ok())
    }

    // 8 stages × 8 batch-bucket slots = 64 atomics. Bucket index =
    // batch_size.next_power_of_two().trailing_zeros() clamped to 0..7,
    // i.e. 1, 2, 4, 8, 16, 32, 64, ≥128.
    pub const N_BUCKETS: usize = 8;

    pub struct Stage {
        pub ns: [AtomicU64; N_BUCKETS],
        pub steps: [AtomicUsize; N_BUCKETS],
    }

    impl Stage {
        // `[AtomicU64::new(0); N]` requires Copy (atomics intentionally
        // don't implement it), and `clippy::declare-interior-mutable-const`
        // forbids the const-then-array trick. Spell each slot out so the
        // compiler emits one independent atomic per bucket — which is what
        // we want anyway, since each bucket is bumped from a different
        // batch_size cohort.
        const fn new() -> Self {
            Self {
                ns: [
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                    AtomicU64::new(0),
                ],
                steps: [
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                    AtomicUsize::new(0),
                ],
            }
        }
    }

    pub static ALLOC: Stage = Stage::new();
    pub static METADATA: Stage = Stage::new();
    pub static SHARED: Stage = Stage::new();
    pub static INPUT_TENSOR: Stage = Stage::new();
    pub static FORWARD: Stage = Stage::new();
    pub static SAMPLING: Stage = Stage::new();
    pub static DISPATCH: Stage = Stage::new();
    pub static TOTAL: Stage = Stage::new();

    pub fn bucket(batch_size: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        let idx = batch_size.next_power_of_two().trailing_zeros() as usize;
        idx.min(N_BUCKETS - 1)
    }

    /// Untyped variant for closures that return a plain value (no Result).
    pub fn time_plain<T>(
        dev: Option<&candle_core::Device>,
        stage: &Stage,
        bucket_idx: usize,
        f: impl FnOnce() -> T,
    ) -> T {
        if !enabled() {
            return f();
        }
        let t0 = Instant::now();
        let out = f();
        #[cfg(feature = "cuda")]
        if let Some(candle_core::Device::Cuda(cd)) = dev {
            let _ = cd.cuda_stream().synchronize();
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        stage.ns[bucket_idx].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    pub fn step_done(batch_size: usize, total_ns: u64) {
        if !enabled() {
            return;
        }
        let b = bucket(batch_size);
        TOTAL.ns[b].fetch_add(total_ns, Ordering::Relaxed);
        let n = TOTAL.steps[b].fetch_add(1, Ordering::Relaxed) + 1;
        if !n.is_multiple_of(STEP_PROFILE_EVERY) {
            return;
        }
        // Snapshot all stages for this bucket and reset.
        let snap = |s: &Stage| (s.ns[b].swap(0, Ordering::Relaxed) as f64) / 1_000.0;
        let s_a = snap(&ALLOC);
        let s_m = snap(&METADATA);
        let s_sh = snap(&SHARED);
        let s_i = snap(&INPUT_TENSOR);
        let s_f = snap(&FORWARD);
        let s_s = snap(&SAMPLING);
        let s_d = snap(&DISPATCH);
        let s_t = (TOTAL.ns[b].swap(0, Ordering::Relaxed) as f64) / 1_000.0;
        let _ = TOTAL.steps[b].swap(0, Ordering::Relaxed);
        let n = STEP_PROFILE_EVERY as f64;
        let bucket_label = match b {
            0 => "≤1",
            1 => "≤2",
            2 => "≤4",
            3 => "≤8",
            4 => "≤16",
            5 => "≤32",
            6 => "≤64",
            _ => "≥128",
        };
        tracing::info!(
            target: "vllm_core::step_profile",
            "decode step (bucket={bucket_label}, {STEP_PROFILE_EVERY} steps, μs/step): \
             alloc={a:.1} metadata={m:.1} shared={sh:.1} input={i:.1} forward={f:.1} \
             sampling={s:.1} dispatch={d:.1} | total={t:.1} sum={sum:.1}",
            a = s_a / n,
            m = s_m / n,
            sh = s_sh / n,
            i = s_i / n,
            f = s_f / n,
            s = s_s / n,
            d = s_d / n,
            t = s_t / n,
            sum = (s_a + s_m + s_sh + s_i + s_f + s_s + s_d) / n,
        );
    }
}

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
        EngineCommand::GenerateStream {
            request,
            stream_tx,
            request_id_tx,
        } => {
            if *paused {
                let _ = stream_tx.try_send(StreamEvent::Error {
                    error: EngineError::Paused.to_string(),
                });
                return false;
            }
            // Peek at the ID that will be assigned before admit_request increments it
            let assigned_id = *next_id;
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
            // Report the assigned ID back so the server can abort on disconnect
            let _ = request_id_tx.send(assigned_id);
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
        EngineCommand::ResetPrefixCache { response_tx } => {
            // Clear the prefix cache and collect evicted block IDs.
            // Two-phase to avoid overlapping &mut borrows on kv_cache_mgr.
            let evicted = kv_cache_mgr
                .prefix_cache_mut()
                .map(|cache| cache.clear())
                .unwrap_or_default();
            let num_evicted = evicted.len();
            if !evicted.is_empty() {
                let _ = kv_cache_mgr.free_blocks(&evicted);
            }
            tracing::info!(num_evicted, "Prefix cache reset");
            let _ = response_tx.send(Ok(num_evicted));
            false
        }
        EngineCommand::Embed { response_tx, .. } => {
            // Embeddings are peeled off by the engine loop's `process_command!`
            // macro before reaching here (they need direct strategy/KV access).
            // Reaching this arm means the macro was bypassed — fail cleanly
            // rather than silently dropping the request.
            let _ = response_tx.send(Err(EngineError::Model(
                "embed command must be handled by the engine loop".into(),
            )));
            false
        }
        EngineCommand::Shutdown => true,
    }
}

/// Reclaim every request whose client has disconnected — the HTTP
/// handler future was dropped (curl timeout / client gone), closing the
/// receiving half of its [`ResponseChannel`].
///
/// This is the authoritative cancellation path. The per-step
/// `send_stream_token` disconnect check only inspects requests that are
/// SCHEDULED that step (and only the streaming variant), so a request
/// abandoned while sitting in the scheduler's WAITING queue — or a
/// non-streaming request whose client gave up — was never noticed:
/// `num_waiting` climbed across retries and never drained. Scanning all
/// active requests once per engine iteration closes both gaps.
///
/// Returns the number reaped so the caller can invalidate any
/// optimistic pre-schedule (the running/waiting sets changed).
pub(crate) fn reap_disconnected_requests(
    scheduler: &mut Scheduler,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    kv_cache_mgr: &mut KVCacheManager,
) -> usize {
    let dead: Vec<RequestId> = requests
        .iter()
        .filter(|(_, req)| req.response.is_disconnected())
        .map(|(&id, _)| id)
        .collect();
    for &id in &dead {
        if let Some(mut req) = requests.remove(&id) {
            // Free KV blocks the request may already hold (it could have
            // been mid-decode when the client vanished).
            let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
        }
        scheduler.remove_request(id);
        tracing::debug!(request_id = id, "client disconnected; request reclaimed");
    }
    dead.len()
}

/// True if a candle / cudarc error string denotes a CUDA out-of-memory.
/// The error is stringified at the model→engine boundary (the structured
/// `candle_core::Error` variant is lost), so we match on the message —
/// both the cudarc form (`CUDA_ERROR_OUT_OF_MEMORY`) and the generic
/// `out of memory` wording.
pub(crate) fn is_cuda_oom(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    e.contains("out_of_memory") || e.contains("out of memory")
}

/// Pick the victim to fail on a CUDA OOM during batched decode, and
/// reclaim retained mem-pool memory.
///
/// The engine must not OOM-KILL a whole concurrent batch for one
/// allocation failure. Instead we trim the pool and fail only the NEWEST
/// (highest arrival_order) sequence — least progress lost, fairest to
/// shed under FCFS. The survivors are left untouched and retry next step
/// in a smaller batch.
///
/// We deliberately do NOT preempt-for-recompute here: recompute folds
/// generated tokens back into the prompt and resets the generation
/// budget, which under heavy preemption regrows the sequence past
/// `max_model_len` (and loses already-streamed tokens) — a pre-existing
/// limitation of the recompute path (see ADR / known-gap). Failing the
/// newest bounds the blast radius to one request without that hazard.
///
/// Returns the victim id (`Some` for any non-empty batch). The caller
/// adds it to `failed` and returns the survivors for retry.
fn select_decode_oom_victim(
    batch_ids: &[RequestId],
    requests: &HashMap<RequestId, ActiveRequest>,
) -> Option<RequestId> {
    crate::engine::cuda_mem::trim(crate::engine::cuda_mem::keep_bytes());
    let victim = batch_ids
        .iter()
        .copied()
        .max_by_key(|id| requests.get(id).map(|r| r.state.arrival_order).unwrap_or(0));
    if let Some(v) = victim {
        tracing::warn!(
            request_id = v,
            batch = batch_ids.len(),
            "decode OOM — failing newest of batch (survivors retry next step)"
        );
    }
    victim
}

/// Max times a request may be preempted-and-requeued after a prefill OOM
/// before we give up and fail it. Sequential serving on a tight GPU
/// succeeds, so a prefill that OOMs under concurrent load almost always
/// fits once peers finish; this bound only catches the genuinely-too-big
/// case (no concurrent load and still OOM).
const MAX_PREFILL_OOM_RETRIES: u32 = 8;

/// Handle a prefill error as backpressure when possible. If it is a CUDA
/// OOM and the request still has retries left, reclaim pool memory and
/// preempt-requeue it (free KV, reset to re-prefill, mark Preempted) so it
/// retries on a later step once concurrent decodes free their working set
/// — the engine queues/serializes under memory pressure instead of
/// OOM-killing. Returns `Ok(id)` to push into `preempted_ids`, or
/// `Err(finalized)` (a real error, or OOM-retry budget exhausted) where
/// `finalized` is the `finish_request_with_error_deferred` result the
/// caller pushes into `errored_ids`.
pub(crate) fn handle_prefill_error(
    req_id: RequestId,
    err: EngineError,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    kv_cache_mgr: &mut KVCacheManager,
) -> Result<RequestId, Option<RequestId>> {
    let is_oom = is_cuda_oom(&err.to_string());
    let retries = requests
        .get(&req_id)
        .map(|r| r.state.oom_retries)
        .unwrap_or(u32::MAX);
    if is_oom && retries < MAX_PREFILL_OOM_RETRIES {
        crate::engine::cuda_mem::trim(crate::engine::cuda_mem::keep_bytes());
        if let Some(req) = requests.get_mut(&req_id) {
            let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
            let state = &mut req.state;
            state.oom_retries += 1;
            state.seqlen_offset = 0;
            state.num_computed_tokens = 0;
            state.status = RequestStatus::Preempted;
        }
        tracing::warn!(
            request_id = req_id,
            retry = retries + 1,
            "prefill OOM — preempting request for retry (backpressure)"
        );
        return Ok(req_id);
    }
    Err(finish_request_with_error_deferred(
        req_id,
        err,
        requests,
        kv_cache_mgr,
    ))
}

/// Why a prompt can NEVER be scheduled by this engine configuration, or
/// `None` if it is schedulable. Pure so admission policy is unit-testable.
///
/// Two permanent-starvation cases (`compute_schedule` re-evaluates every
/// step, but neither condition can ever change for the request):
/// 1. The prompt (+1 headroom token for the first generated token) needs
///    more KV blocks than the pool has IN TOTAL — `blocks_needed <=
///    free_blocks <= total` can never hold. Mirrors the decode-time
///    recompute guard and the startup max_model_len clamp.
/// 2. Chunked prefill is disabled (explicit opt-out) and the prompt exceeds
///    `max_tokens_per_step`: the scheduler then requires the WHOLE remaining
///    prompt to fit one step's budget.
pub(crate) fn prompt_unschedulable_reason(
    prompt_tokens: usize,
    block_size: usize,
    total_blocks: usize,
    sched_cfg: &crate::scheduler::SchedulerConfig,
) -> Option<String> {
    let blocks_for_prompt = prompt_tokens.saturating_add(1).div_ceil(block_size);
    if blocks_for_prompt > total_blocks {
        return Some(format!(
            "prompt of {} tokens needs {} KV cache blocks but the pool has only {} \
             in total ({} tokens) — it can never be scheduled. Shorten the prompt, \
             raise --gpu-memory-utilization / --num-blocks, or use --kv-cache-dtype fp8.",
            prompt_tokens,
            blocks_for_prompt,
            total_blocks,
            total_blocks * block_size,
        ));
    }
    if !sched_cfg.enable_chunked_prefill && prompt_tokens > sched_cfg.max_tokens_per_step {
        return Some(format!(
            "prompt of {} tokens exceeds the per-step token budget of {} \
             (--max-num-batched-tokens) and chunked prefill is disabled — it can \
             never be scheduled. Remove --disable-chunked-prefill, raise \
             --max-num-batched-tokens, or shorten the prompt.",
            prompt_tokens, sched_cfg.max_tokens_per_step,
        ));
    }
    None
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
    let prompt_ids = match tokenizer.encode_prompt(&request.prompt) {
        Ok(ids) => ids,
        Err(e) => {
            send_error(response, EngineError::Tokenization(e.to_string()));
            return;
        }
    };

    // Admission guards: fail immediately (with an actionable error) any
    // prompt this engine configuration could NEVER schedule — otherwise it
    // sits in the waiting queue until the client times out.
    if let Some(reason) = prompt_unschedulable_reason(
        prompt_ids.len(),
        block_size,
        kv_cache_mgr.num_total_blocks(),
        scheduler.config(),
    ) {
        send_error(response, EngineError::Cache(reason));
        return;
    }

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
    state.prompt_adapter_request = request.prompt_adapter_request;
    state.constraint = request.constraint;

    // Check prefix cache for reusable blocks via the KVCacheManager
    if kv_cache_mgr.has_prefix_cache() && !request.skip_prefix_cache {
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

/// Compute pooled embeddings for pre-tokenized inputs on the loaded model.
///
/// Stateless one-shot: each input gets a temporary block table, a single
/// `forward_hidden` prefill, pooling over all positions + optional L2
/// normalization, then the blocks are freed **unconditionally** (even when the
/// forward errors). No scheduler entry, no generation. Bypasses the output
/// pool via `ModeGuard::prefill` (transients drop on scope exit; `forward_hidden`
/// never allocates the `lm_head` buffer, so peak memory is below a generation
/// prefill). Inputs are processed sequentially.
pub(crate) fn execute_embed<M: ModelForward>(
    model: &M,
    inputs: &[Vec<u32>],
    pooling: Option<super::embedding_forward::PoolingStrategy>,
    normalize: bool,
    kv_cache_mgr: &mut KVCacheManager,
) -> Result<Vec<Vec<f32>>, EngineError> {
    use super::embedding_forward::PoolingStrategy;
    let _mode_guard = crate::engine::output_pool::ModeGuard::prefill();

    if !model.supports_embeddings() {
        return Err(EngineError::Model(
            "loaded model does not support embeddings".into(),
        ));
    }

    // Resolve pooling: explicit request override → model's native pooling
    // (e.g. mean for an encoder embedder) → last-token default (causal LMs).
    let pooling = pooling
        .or_else(|| model.embedding_pooling())
        .unwrap_or(PoolingStrategy::LastToken);

    let device = model.device();
    let block_size = kv_cache_mgr.block_size();
    let mut out = Vec::with_capacity(inputs.len());

    for tokens in inputs {
        let len = tokens.len();
        if len == 0 {
            return Err(EngineError::Model("empty embedding input".into()));
        }

        // Temporary block table — not tied to any ActiveRequest. Allocate,
        // run, then free regardless of outcome so a forward error cannot leak
        // blocks out of the shared pool.
        let mut block_table = BlockTable::new(block_size);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, len)
            .map_err(|e| EngineError::Cache(e.to_string()))?;
        let slot_mapping = block_table.slot_mapping(0, len);

        let input = match Tensor::from_vec(tokens.clone(), (1, len), device) {
            Ok(t) => t,
            Err(e) => {
                let _ = kv_cache_mgr.free_request(&mut block_table);
                return Err(EngineError::Model(e.to_string()));
            }
        };

        // seqlen_offset is always 0: each embed input is a fresh sequence with
        // a fresh block table (a non-zero offset would silently corrupt RoPE).
        let hidden = model.forward_hidden(&input, 0, kv_cache_mgr, &block_table, &slot_mapping);
        let _ = kv_cache_mgr.free_request(&mut block_table);
        let hidden = hidden.map_err(|e| EngineError::Model(e.to_string()))?;

        // forward_hidden must return one row per real token; a mismatch means
        // bucketing/padding leaked in and would poison the pooled vector.
        let hidden_positions = hidden
            .dim(0)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        if hidden_positions != len {
            return Err(EngineError::Model(format!(
                "forward_hidden returned {hidden_positions} positions for {len} tokens"
            )));
        }

        // Pool over all positions: [1, seq, hidden] with an all-ones mask.
        let token_embeddings = hidden
            .unsqueeze(0)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let mask = Tensor::ones((1, len), candle_core::DType::F32, device)
            .map_err(|e| EngineError::Model(e.to_string()))?;
        let pooled = super::embedding_forward::pool_embeddings(&token_embeddings, &mask, pooling)
            .map_err(|e| EngineError::Model(e.to_string()))?;

        // Model-specific projection between pooling and normalization
        // (EmbeddingGemma's sentence-transformers Dense head); identity otherwise.
        let mut pooled = model
            .project_embedding(&pooled)
            .map_err(|e| EngineError::Model(e.to_string()))?;

        if normalize {
            let norm = pooled
                .sqr()
                .and_then(|t| t.sum_keepdim(candle_core::D::Minus1))
                .and_then(|t| t.sqrt())
                .map_err(|e| EngineError::Model(e.to_string()))?;
            pooled = pooled
                .broadcast_div(&norm)
                .map_err(|e| EngineError::Model(e.to_string()))?;
        }

        let vector = pooled
            .squeeze(0)
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| EngineError::Model(e.to_string()))?;
        out.push(vector);
    }

    Ok(out)
}

pub(crate) fn execute_prefill<M: ModelForward>(
    req_id: RequestId,
    chunk_size: usize,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    tokenizer: &TokenizerWrapper,
) -> Result<(), EngineError> {
    // Mark this forward as prefill so any large pool reserve reached
    // through it (lm_head outputs, per-layer activations, etc.)
    // auto-bypasses the pool and drops on scope exit. See
    // `crate::engine::output_pool::ExecutionMode` for the contract.
    // Captured CUDA Graph replay never enters this function (no
    // `graph_runner` parameter), so bypassing the pool is safe.
    let _mode_guard = crate::engine::output_pool::ModeGuard::prefill();

    let req = requests
        .get_mut(&req_id)
        .ok_or_else(|| EngineError::Model(format!("request {req_id} not found")))?;
    // Prefill operates over the FULL sequence (prompt + already-generated).
    // For a fresh request `generated` is empty so this is the prompt; for a
    // request resumed after recompute-preemption it re-prefills prompt +
    // generated and then continues decoding. See `SequenceState::total_len`.
    let prompt_len = req.state.prompt_token_ids.len();
    let total_len = req.state.total_len();
    let is_resumed = !req.state.generated_token_ids.is_empty();
    let offset = req.state.num_computed_tokens;
    let chunk_end = (offset + chunk_size).min(total_len);
    let actual_chunk = chunk_end - offset;
    let is_final_chunk = chunk_end == total_len;

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

    // Tokens for this chunk come from the full sequence (prompt ++ generated),
    // so a resumed request's chunk can span the prompt→generated boundary.
    let chunk_tokens = req.state.token_window(offset, chunk_end);
    let input = Tensor::from_vec(chunk_tokens, (1, actual_chunk), model.device())
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
        .map_err(|e| {
            // On a prefill OOM, reclaim retained CUDA mem-pool memory so
            // the next admitted request isn't starved by buffers this
            // failed forward left pooled. Prefill is single-request, so we
            // surface a clean error rather than shrinking a batch.
            let s = e.to_string();
            if is_cuda_oom(&s) {
                crate::engine::cuda_mem::trim(crate::engine::cuda_mem::keep_bytes());
            }
            EngineError::Model(s)
        })?;

    req.state.block_table.advance(actual_chunk);
    req.state.num_computed_tokens = chunk_end;
    req.state.seqlen_offset = chunk_end;

    // Compute prompt logprobs if echo mode is enabled with logprobs — but
    // only on the ORIGINAL prefill. A resumed (recompute) re-prefill must not
    // re-append prompt logprobs (they were computed the first time); doing so
    // would duplicate them.
    let compute_prompt_logprobs =
        req.state.echo && req.state.num_top_logprobs.is_some() && !is_resumed;
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

        // Models with very large vocabularies (Gemma 4: 262k) return only
        // the final position's logits from prefill — full-sequence logits
        // would cost hundreds of MB per cached shape. Echo prompt-logprobs
        // need every position, so they are unavailable for such models;
        // degrade explicitly instead of indexing out of bounds.
        let logits_positions = logits.dims()[1];
        let last_pos = if logits_positions < last_pos {
            tracing::warn!(
                "prompt_logprobs (echo) unavailable: model returned logits for \
                 {logits_positions} of {last_pos} prompt positions"
            );
            0
        } else {
            last_pos
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

/// Reclaim KV cache blocks that are entirely outside the sliding window.
///
/// For each request in the batch, computes how many leading blocks are
/// no longer needed based on the sequence's current token count and the
/// sliding window size. Replaces those blocks with `NULL_BLOCK` in the
/// block table and returns them to the free pool.
///
/// This is called before block allocation in the decode step to maximize
/// available blocks. No-op if `sliding_window` is `None`.
pub(crate) fn reclaim_sliding_window_blocks(
    request_ids: &[RequestId],
    sliding_window: Option<usize>,
    block_size: usize,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) {
    let window = match sliding_window {
        Some(w) => w,
        None => return,
    };

    for &req_id in request_ids {
        let Some(req) = requests.get_mut(&req_id) else {
            continue;
        };
        let total_tokens = req.state.block_table.num_tokens();
        if total_tokens <= window {
            continue;
        }
        // Tokens [0, num_skipped) are entirely outside the window
        let num_skipped_tokens = total_tokens - window;
        let num_skipped_blocks = num_skipped_tokens / block_size;
        let already_null = req.state.block_table.num_null_blocks();
        if num_skipped_blocks <= already_null {
            continue;
        }
        let freed = req
            .state
            .block_table
            .reclaim_leading_blocks(num_skipped_blocks);
        if !freed.is_empty() {
            // Ignore errors — blocks should always be freeable
            let _ = kv_cache_mgr.free_blocks(&freed);
        }
    }
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
) -> Vec<(RequestId, String)> {
    let mut preempted: Vec<RequestId> = Vec::new();
    let (failed, _next) = execute_batched_decode_with_graph(
        decode_request_ids,
        None,
        false, // defer_dtoh
        model,
        kv_cache_mgr,
        requests,
        None,
        None,
        &mut preempted,
    );
    failed
}

/// Group sequences by LoRA adapter name and execute forward passes per group.
///
/// For a batch with mixed adapters (e.g., base + adapter_A + adapter_B),
/// this creates sub-batches per adapter, calls `forward_decode_batch_with_lora`
/// for each, then reassembles logits in original batch order.
fn forward_grouped_by_adapter<M: ModelForward>(
    model: &M,
    token_ids: &[u32],
    sequences: &[DecodeSequenceMetadata],
    adapter_names: &[Option<String>],
    kv_cache_mgr: &mut KVCacheManager,
    batch_size: usize,
) -> candle_core::Result<Tensor> {
    // Build per-adapter groups: maps adapter_name → Vec<original_index>
    let mut groups: Vec<(Option<&str>, Vec<usize>)> = Vec::new();
    for (i, name) in adapter_names.iter().enumerate() {
        let key = name.as_deref();
        if let Some(group) = groups.iter_mut().find(|(k, _)| *k == key) {
            group.1.push(i);
        } else {
            groups.push((key, vec![i]));
        }
    }

    // Fast path: single adapter (or all base) — no regrouping needed
    if groups.len() == 1 {
        let (adapter, indices) = &groups[0];
        let input = Tensor::from_vec(token_ids.to_vec(), (batch_size, 1), model.device())?;
        let lora_ctx = match adapter {
            Some(name) => LoraContext::with_adapter(*name),
            None => LoraContext::none(),
        };
        // indices covers entire batch in order, use directly
        let _ = indices;
        return model.forward_decode_batch_with_lora(&input, sequences, kv_cache_mgr, &lora_ctx);
    }

    // Multi-adapter path: forward each group, then reassemble
    // We'll store (original_index, logits_row) pairs to reassemble in order
    let mut all_logits: Vec<(usize, Tensor)> = Vec::with_capacity(batch_size);

    for (adapter, indices) in &groups {
        let group_tokens: Vec<u32> = indices.iter().map(|&i| token_ids[i]).collect();
        let group_sequences: Vec<DecodeSequenceMetadata> =
            indices.iter().map(|&i| sequences[i].clone()).collect();
        let group_size = indices.len();

        let group_input = Tensor::from_vec(group_tokens, (group_size, 1), model.device())?;
        let lora_ctx = match adapter {
            Some(name) => LoraContext::with_adapter(*name),
            None => LoraContext::none(),
        };

        let group_logits = model.forward_decode_batch_with_lora(
            &group_input,
            &group_sequences,
            kv_cache_mgr,
            &lora_ctx,
        )?;

        // Extract per-sequence logit rows from the group output
        for (local_idx, &orig_idx) in indices.iter().enumerate() {
            let row = group_logits.i(local_idx)?;
            all_logits.push((orig_idx, row));
        }
    }

    // Reassemble in original batch order
    all_logits.sort_by_key(|(idx, _)| *idx);
    let ordered_rows: Vec<Tensor> = all_logits.into_iter().map(|(_, t)| t).collect();
    Tensor::stack(&ordered_rows, 0)
}

/// v1.9: build a packed-i32 bitmask covering every row in the active
/// batch and apply it in-place to `last_logits` via the native CUDA
/// `apply_grammar_bitmask` kernel.
///
/// Rows for requests without a constraint get an all-1s mask (no-op
/// on logits — preserves their values). Rows for requests with a
/// constraint are filled by `fill_cpu_bitmask_for_gpu`; the tail past
/// the grammar's `vocab_size` stays zero so padded lm_head tokens are
/// implicitly forbidden.
///
/// One CPU vec + one HtoD copy per step; cheaper than the legacy
/// `to_vec2::<f32>` of the entire logits tensor (≈ batch × vocab × 4
/// bytes). Holds the engine's mutable `requests` map because each
/// constraint advances no state here — only `fill_cpu_bitmask_for_gpu`
/// is called, which is a query.
fn apply_grammar_bitmask_to_logits(
    last_logits: &Tensor,
    batch_ids: &[RequestId],
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> anyhow::Result<()> {
    let dims = last_logits.dims();
    if dims.len() != 2 {
        anyhow::bail!("apply_grammar_bitmask_to_logits: logits must be 2D, got {dims:?}",);
    }
    let (batch, logits_vocab) = (dims[0], dims[1]);
    if batch != batch_ids.len() {
        anyhow::bail!(
            "apply_grammar_bitmask_to_logits: logits batch {batch} ≠ batch_ids {}",
            batch_ids.len(),
        );
    }
    let words_per_row = logits_vocab.div_ceil(32);
    let mut cpu_bitmask = vec![0i32; batch * words_per_row];

    let prof = std::env::var("VLLM_PROFILE_GRAMMAR").is_ok();
    let t_fill_start = prof.then(std::time::Instant::now);

    // First, mark unconstrained rows all-allowed (no-op on the apply
    // kernel) so only constrained rows need a real fill.
    for (i, &req_id) in batch_ids.iter().enumerate() {
        let unconstrained = requests
            .get(&req_id)
            .map(|r| r.state.constraint.is_none())
            .unwrap_or(false);
        if unconstrained {
            let row = &mut cpu_bitmask[i * words_per_row..(i + 1) * words_per_row];
            for w in row.iter_mut() {
                *w = !0;
            }
        }
    }

    // Batched parallel fill for xgrammar-backed constrained rows.
    // `xgrammar_rs::BatchMatcher` dispatches one `fill_next_token_bitmask`
    // per matcher across a thread pool — replacing the O(batch) sequential
    // CPU fill that dominates constrained decode at c≥2 (profiled at
    // ~4 ms/matcher on Qwen3's 151k vocab). Rows whose constraint has no
    // xgrammar matcher fall through to the per-request path below.
    #[cfg(feature = "xgrammar")]
    let mut handled: std::collections::HashSet<usize> = std::collections::HashSet::new();
    #[cfg(feature = "xgrammar")]
    let batch_enabled = std::env::var("VLLM_DISABLE_GRAMMAR_BATCH").is_err();
    #[cfg(feature = "xgrammar")]
    if batch_enabled {
        // Collect (row_idx, matcher, grammar_words) for xgrammar rows.
        // Scoped immutable borrow of `requests` — released before the
        // per-request mutable fallback below.
        let mut row_idx: Vec<usize> = Vec::new();
        let mut guards: Vec<std::sync::MutexGuard<'_, xgrammar_rs::GrammarMatcher>> = Vec::new();
        let mut grammar_words: Option<usize> = None;
        for (i, &req_id) in batch_ids.iter().enumerate() {
            if let Some(req) = requests.get(&req_id) {
                if let Some(c) = req.state.constraint.as_ref() {
                    if let (Some(m), Some(gw)) = (c.xgrammar_matcher(), c.grammar_bitmask_words()) {
                        if let Ok(guard) = m.lock() {
                            // xgrammar aborts (LogFatalError) if asked to
                            // fill a bitmask after the stop token was
                            // accepted. Terminated matchers are left for
                            // the per-request path, which emits an
                            // all-forbidden row without touching xgrammar.
                            if guard.is_terminated() {
                                continue;
                            }
                            row_idx.push(i);
                            guards.push(guard);
                            grammar_words.get_or_insert(gw);
                        }
                    }
                }
            }
        }
        if let Some(gw) = grammar_words {
            if guards.len() >= 2 && gw <= words_per_row {
                let matchers: Vec<&xgrammar_rs::GrammarMatcher> =
                    guards.iter().map(|g| &**g).collect();
                let mut tmp = vec![0i32; matchers.len() * gw];
                match crate::sampling::grammar::xgrammar_backend::batch_matcher_handle()
                    .fill_bitmasks(&matchers, &mut tmp, gw)
                {
                    Ok(()) => {
                        // Scatter each grammar-sized row into the
                        // logits-sized cpu_bitmask slot; the tail
                        // [gw..words_per_row] stays zero (padded vocab
                        // forbidden).
                        for (k, &i) in row_idx.iter().enumerate() {
                            let dst = &mut cpu_bitmask[i * words_per_row..i * words_per_row + gw];
                            dst.copy_from_slice(&tmp[k * gw..k * gw + gw]);
                            handled.insert(i);
                        }
                    }
                    Err(e) => {
                        // Fall through to per-request fill on batch error.
                        tracing::warn!(
                            error = %e,
                            "BatchMatcher fill failed; using per-request fill",
                        );
                    }
                }
            }
        }
        // `guards` drop here → matcher locks released, immutable borrow ends.
    }

    // Per-request fill for any constrained row not handled by the batch
    // path (single constrained request, non-xgrammar constraint, or
    // batch-fill error fallback).
    for (i, &req_id) in batch_ids.iter().enumerate() {
        #[cfg(feature = "xgrammar")]
        if handled.contains(&i) {
            continue;
        }
        let Some(req) = requests.get_mut(&req_id) else {
            continue;
        };
        if let Some(ref mut constraint) = req.state.constraint {
            let row = &mut cpu_bitmask[i * words_per_row..(i + 1) * words_per_row];
            match constraint.fill_cpu_bitmask_for_gpu(row) {
                Some(Ok(())) => {}
                Some(Err(e)) => return Err(e),
                None => anyhow::bail!(
                    "constraint claims supports_gpu=true but \
                     fill_cpu_bitmask_for_gpu returned None",
                ),
            }
        }
    }

    let t_fill_ns = t_fill_start.map(|t| t.elapsed().as_nanos() as u64);

    let dev = last_logits.device();
    let t_up_start = prof.then(std::time::Instant::now);
    let bitmask_t = Tensor::from_vec(cpu_bitmask, (batch, words_per_row), dev)
        .map_err(|e| anyhow::anyhow!("bitmask Tensor::from_vec: {e}"))?;
    crate::sampling::gpu::gpu_apply_grammar_bitmask(last_logits, &bitmask_t)
        .map_err(|e| anyhow::anyhow!("gpu_apply_grammar_bitmask: {e}"))?;

    if prof {
        // Force a device sync so the apply-kernel time is attributed
        // here rather than leaking into the next sampler op.
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cd) = dev {
            let _ = cd.cuda_stream().synchronize();
        }
        let t_up_ns = t_up_start
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        // Accumulate into process-global counters; dump every 64 calls.
        use std::sync::atomic::{AtomicU64, Ordering};
        static FILL_NS: AtomicU64 = AtomicU64::new(0);
        static UP_NS: AtomicU64 = AtomicU64::new(0);
        static N: AtomicU64 = AtomicU64::new(0);
        FILL_NS.fetch_add(t_fill_ns.unwrap_or(0), Ordering::Relaxed);
        UP_NS.fetch_add(t_up_ns, Ordering::Relaxed);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(64) {
            tracing::info!(
                target: "vllm_core::engine::helpers",
                batch,
                calls = n,
                fill_us_avg = FILL_NS.load(Ordering::Relaxed) / n / 1000,
                upload_apply_us_avg = UP_NS.load(Ordering::Relaxed) / n / 1000,
                "grammar bitmask profile",
            );
        }
    }
    Ok(())
}

/// Execute batched decode with optional CUDA graph support.
/// When a dispatcher is provided, dispatches to cached graphs when available.
/// When a graph runner is provided, uses it for optimized execution.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_batched_decode_with_graph<M: ModelForward>(
    decode_request_ids: &[RequestId],
    // `prev_sampled_ids`: the U32 [B] tensor returned from the previous
    // decode step's GPU sampler (Phase 1 foundation, ADR 0017). When
    // `Some` AND its leading dim equals the current batch size AND no
    // preemption occurred (`batch_ids.len() == decode_request_ids.len()`),
    // it is reshaped to [B,1] and used directly as `input_ids`,
    // bypassing the host `Tensor::from_vec` round-trip. When shape /
    // batch composition mismatch, fallback to host path. This is
    // perf-neutral on its own (Substep 2.1) but is the prerequisite for
    // async DtoH overlap in Substep 2.2.
    prev_sampled_ids: Option<&Tensor>,
    // `defer_dtoh`: ADR 0017 / Substep 2.2. When `true` AND the GPU
    // sampler fast path succeeds, skip the blocking `to_vec1()` and
    // skip `generated_token_ids.push`. Block-table / seqlen counters
    // are still advanced (they don't depend on the token value). The
    // caller is responsible for syncing the returned tensor on a
    // dedicated DtoH stream and pushing the tokens once the event
    // completes. Has no effect when the GPU fast path is unavailable
    // (mixed-mode batches, CPU sampling fallback, etc.) — those go
    // through the existing host-blocking path regardless.
    defer_dtoh: bool,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    dispatcher: Option<&Arc<std::sync::RwLock<CudaGraphDispatcher>>>,
    graph_runner: Option<&Arc<CudaGraphRunner>>,
    // `preempted_out`: IDs to push when KV-cache allocation fails at
    // decode-step time. The caller (engine async loop) drains this and
    // calls `Scheduler::move_to_waiting(id)` so the request gets
    // re-admitted on the next compute_schedule pass instead of being
    // errored out.
    preempted_out: &mut Vec<RequestId>,
) -> (Vec<(RequestId, String)>, Option<Tensor>) {
    let mut failed: Vec<(RequestId, String)> = Vec::new();
    let mut batch_ids: Vec<RequestId> = Vec::with_capacity(decode_request_ids.len());

    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(
            target: "vllm_core::step_profile",
            "execute_batched_decode_with_graph entry, profiler enabled={}",
            step_profile::enabled()
        );
    });

    let step_start = std::time::Instant::now();
    let step_dev = if step_profile::enabled() {
        Some(model.device().clone())
    } else {
        None
    };
    let step_dev_ref = step_dev.as_ref();

    // Step 1: Allocate blocks for each sequence.
    //
    // Allocation can fail at decode-step time if the scheduler's free-block
    // estimate drifted from the live cache state (e.g. the prefill in this
    // same step consumed more blocks than estimated, or chunked prefill
    // landed at a boundary we didn't account for). Instead of failing the
    // request fatally we preempt it: free its existing KV blocks, fold any
    // generated tokens back into the prompt, and reset for recompute. The
    // engine's async loop calls `Scheduler::move_to_waiting(id)` so the
    // request gets re-admitted on the next `compute_schedule` pass once
    // free blocks become available again. See ADR / docs/perf for the
    // recompute-vs-swap trade-off (we use recompute by default; sampler
    // RNG state is preserved so the resumed stream stays coherent).
    let bucket_in = step_profile::bucket(decode_request_ids.len());
    step_profile::time_plain(step_dev_ref, &step_profile::ALLOC, bucket_in, || {
        for &req_id in decode_request_ids {
            let Some(req) = requests.get_mut(&req_id) else {
                failed.push((req_id, "request no longer active".to_string()));
                continue;
            };
            match kv_cache_mgr.allocate_for_request(&mut req.state.block_table, 1) {
                Ok(()) => batch_ids.push(req_id),
                Err(e) => {
                    tracing::warn!(
                        request_id = req_id,
                        error = %e,
                        "kv cache exhausted at decode step — preempting request for recompute"
                    );
                    // Free whatever blocks the request currently holds.
                    if let Err(free_err) = kv_cache_mgr.free_request(&mut req.state.block_table) {
                        // Failing to free blocks is unrecoverable — the cache
                        // accounting is now out of sync with reality. Surface
                        // as a fatal error instead of leaking blocks silently.
                        failed.push((req_id, format!("kv cache preempt-free failed: {free_err}")));
                        continue;
                    }
                    // Recompute preemption (vLLM model): keep prompt and
                    // generated tokens SEPARATE — do NOT fold generated into
                    // the prompt. Reset only `num_computed_tokens` (and the
                    // RoPE offset) to 0 so the next schedule re-prefills the
                    // full sequence (prompt + generated, via `total_len()`)
                    // and resumes generation. generated_token_ids and its
                    // logprobs are preserved, so the generation budget
                    // (`num_generated()` vs max_new_tokens) stays correct and
                    // the final output keeps every token. The earlier fold
                    // reset the budget (regrowth past max_model_len) and lost
                    // tokens for non-streaming requests.
                    let state = &mut req.state;
                    state.seqlen_offset = 0;
                    state.num_computed_tokens = 0;
                    // Guard: if the full sequence can never fit the pool (more
                    // blocks than EXIST, not merely more than are free),
                    // re-admission is impossible — compute_schedule would
                    // refuse it forever and, under FCFS, head-of-line starve
                    // every request behind it. Fail it explicitly. With the
                    // budget no longer reset, this is reachable only when a
                    // single request's prompt+max_new_tokens genuinely exceeds
                    // the pool (and should be caught at admission), not via
                    // preemption regrowth.
                    let blocks_for_recompute = state.block_table.blocks_needed(state.total_len());
                    let total_blocks = kv_cache_mgr.num_total_blocks();
                    if blocks_for_recompute > total_blocks {
                        failed.push((
                            req_id,
                            format!(
                                "sequence of {} tokens needs {} KV cache blocks but the pool \
                                 has only {} in total — recompute after preemption can never \
                                 be scheduled (lower --max-model-len or raise \
                                 --gpu-memory-utilization)",
                                state.total_len(),
                                blocks_for_recompute,
                                total_blocks,
                            ),
                        ));
                        continue;
                    }
                    state.status = RequestStatus::Preempted;
                    preempted_out.push(req_id);
                }
            }
        }
    });

    if batch_ids.is_empty() {
        return (failed, None);
    }

    // Step 2: Collect input tokens, per-sequence metadata, and adapter grouping
    let mut token_ids: Vec<u32> = Vec::with_capacity(batch_ids.len());
    let mut sequences: Vec<DecodeSequenceMetadata> = Vec::with_capacity(batch_ids.len());
    // Per-sequence adapter name (None = base model), in same order as token_ids/sequences
    let mut adapter_names: Vec<Option<String>> = Vec::with_capacity(batch_ids.len());
    // Authoritative processed set, built in lockstep with token_ids /
    // sequences. A sequence allocated in Step 1 but lacking a last token
    // (e.g. just-(re)prefilled and not yet sampled) is excluded here; it
    // MUST also drop out of `batch_ids` so that `batch_size`, the input
    // tensor, and the attention metadata all describe the same number of
    // rows. Otherwise the `prev_sampled` GPU pass-through (sized to the
    // pre-filter `batch_ids`) feeds an N-row input into an (N-1)-row
    // attention build → `shape mismatch in reshape [N,1,H] vs [N-1,…]`.
    let mut processed_ids: Vec<RequestId> = Vec::with_capacity(batch_ids.len());

    let bucket = step_profile::bucket(batch_ids.len());
    step_profile::time_plain(step_dev_ref, &step_profile::METADATA, bucket, || {
        for &req_id in &batch_ids {
            let Some(req) = requests.get(&req_id) else {
                continue;
            };
            let Some(&last_token) = req.state.generated_token_ids.last() else {
                // Normal operation never hits this (a decode-set sequence
                // has at least its prefill-sampled token). It appears only
                // under heavy KV-exhaustion preemption thrash, where a
                // just-preempted sequence can momentarily reappear with its
                // generated tokens already folded back into the prompt. We
                // exclude it for this step (it re-prefills next), but
                // rate-limit the log so a thrash storm can't flood it.
                use std::sync::atomic::{AtomicU64, Ordering};
                static SKIP_COUNT: AtomicU64 = AtomicU64::new(0);
                let n = SKIP_COUNT.fetch_add(1, Ordering::Relaxed);
                if n == 0 || n.is_power_of_two() {
                    tracing::warn!(
                        request_id = req_id,
                        total_skips = n + 1,
                        "decode-batch sequence has no generated token; \
                         excluding from this step to keep batch dims \
                         consistent (rate-limited log)"
                    );
                }
                continue;
            };
            let slot_mapping = req
                .state
                .block_table
                .slot_mapping(req.state.seqlen_offset, 1);
            processed_ids.push(req_id);
            token_ids.push(last_token);
            sequences.push(DecodeSequenceMetadata {
                request_id: req_id,
                seqlen_offset: req.state.seqlen_offset,
                block_ids: req.state.block_table.block_ids().to_vec(),
                slot_mapping,
            });
            adapter_names.push(req.state.lora_request.as_ref().map(|lr| lr.name.clone()));
        }
    });

    // Adopt the filtered set as the batch for the rest of this step.
    batch_ids = processed_ids;
    if batch_ids.is_empty() {
        return (failed, None);
    }

    // Step 3: Batched forward pass with multi-adapter grouping
    let batch_size = batch_ids.len();
    let bucket = step_profile::bucket(batch_size);
    let t_forward_start = step_profile::enabled().then(std::time::Instant::now);
    let has_any_adapter = adapter_names.iter().any(|a| a.is_some());

    // Helper: tag every request in the current batch with the same
    // error message. Used when a whole-batch op (forward, tensor build,
    // sampling) fails — every in-flight request in that batch bubbles
    // up the same root cause to the caller.
    let failed_batch = |msg: String, batch: &[RequestId]| -> Vec<(RequestId, String)> {
        batch.iter().map(|id| (*id, msg.clone())).collect()
    };

    let logits = if has_any_adapter {
        // Multi-adapter path: group sequences by adapter, forward each group separately,
        // then reassemble logits in original batch order.
        match forward_grouped_by_adapter(
            model,
            &token_ids,
            &sequences,
            &adapter_names,
            kv_cache_mgr,
            batch_size,
        ) {
            Ok(l) => l,
            Err(e) => {
                if is_cuda_oom(&e.to_string()) {
                    if let Some(v) = select_decode_oom_victim(&batch_ids, requests) {
                        failed.push((v, format!("decode out of memory: {e}")));
                        return (failed, None);
                    }
                }
                let msg = format!("forward_grouped_by_adapter: {e}");
                tracing::error!(error = %e, "batched decode forward failed");
                failed.extend(failed_batch(msg, &batch_ids));
                return (failed, None);
            }
        }
    } else {
        // No adapters — use standard path with CUDA graph support
        let mut forward_ctx = if let Some(disp) = dispatcher {
            let descriptor = BatchDescriptor::for_decode(batch_size);
            match disp.read() {
                Ok(disp_guard) => {
                    let result = disp_guard.dispatch(descriptor);
                    ForwardContext::from_dispatch(result)
                }
                Err(_poisoned) => ForwardContext::eager(),
            }
        } else {
            ForwardContext::eager()
        };

        // Rewind the global GEMV output-buffer pool. Each shape's
        // round-robin cursor restarts at 0 so subsequent `reserve()` calls
        // walk the same allocation order this forward as last forward —
        // the pool stops growing once a steady decode workload has been
        // observed.
        crate::engine::output_pool::OutputPool::global().reset_cursors();

        // Build decode-batch shared tensors once per forward. Models that
        // route through `forward_decode_batch_with_ctx` and propagate
        // `ctx.decode_shared` to their attention layers will reuse this
        // bundle 36× per Qwen3 token instead of rebuilding it per layer.
        // Phase D.3: signal pool-backed attention only when capture is
        // engaged. With a graph_runner attached, replay reads from
        // pool-backed device pointers and the worst-case partition
        // sizing is required for shape stability. Without a runner
        // (pure eager mode), `paged_attention_auto` with actual
        // max_seq_len partitioning is significantly faster — pool
        // sizing iterates over up to 3× more empty partitions per call.
        // D10 diagnostic: force the pool-V2 path even in pure eager mode
        // so eager and capture-replay both exercise the same typed forward.
        // Without this override an eager run uses non-pool kernels, making
        // its dump unsuitable as a baseline for comparison.
        let prefer_pooled = graph_runner.is_some() || std::env::var("VLLM_FORCE_POOLED").is_ok();
        match crate::layers::attention::build_decode_batch_shared_with_options(
            &sequences,
            model.device(),
            prefer_pooled,
        ) {
            Ok(shared) => {
                // D10: one-shot print of positions_device value AFTER
                // the build wrote real positions into the pool slot.
                // Confirms (or refutes) that production updates the
                // slot the captured graph reads from.
                if crate::engine::layer_dump::is_enabled() {
                    static ONCE_POS: std::sync::Once = std::sync::Once::new();
                    ONCE_POS.call_once(|| {
                        if let Some(pt) = shared.positions_device.as_ref() {
                            #[cfg(feature = "cuda")]
                            if let candle_core::Device::Cuda(c) = model.device() {
                                let _ = c.cuda_stream().synchronize();
                            }
                            match pt.as_tensor().flatten_all() {
                                Ok(flat) => match flat.to_vec1::<u32>() {
                                    Ok(v) => eprintln!(
                                        "D10: after build_shared, positions_device={:?}",
                                        v
                                    ),
                                    Err(e) => eprintln!("D10: pos to_vec1 failed: {e}"),
                                },
                                Err(e) => eprintln!("D10: pos flatten failed: {e}"),
                            }
                        }
                    });
                }
                forward_ctx = forward_ctx.with_decode_shared(std::sync::Arc::new(shared));
            }
            Err(e) => {
                // Non-fatal: fall back to per-layer build at the
                // attention site. Log so unexpected build failures
                // don't go silent.
                tracing::debug!(error = %e, "build_decode_batch_shared failed; per-layer build will run");
            }
        }

        // Substep 2.1 / ADR 0017: prefer GPU tensor pass-through from
        // the previous decode step's sampler output. Valid only when no
        // sequences were preempted this step (batch_ids covers all
        // decode_request_ids) AND prev tensor's leading dim matches
        // current batch size. Otherwise fallback to host vec → tensor.
        let prev_input = prev_sampled_ids
            .filter(|_| batch_ids.len() == decode_request_ids.len())
            .filter(|t| t.dims().first().copied() == Some(batch_size))
            .and_then(|t| t.unsqueeze(1).ok());

        let input = match prev_input {
            Some(t) => {
                // token_ids is unused on this path; dropped at end of scope.
                drop(token_ids);
                t
            }
            None => match Tensor::from_vec(token_ids, (batch_size, 1), model.device()) {
                Ok(t) => t,
                Err(e) => {
                    let msg = format!("Tensor::from_vec(decode input): {e}");
                    tracing::error!(error = %e, "batched decode tensor build failed");
                    failed.extend(failed_batch(msg, &batch_ids));
                    return (failed, None);
                }
            },
        };

        if let Some(runner) = graph_runner {
            match runner.execute(&input, |inp| {
                model.forward_decode_batch(inp, &sequences, kv_cache_mgr)
            }) {
                Ok(l) => l,
                Err(cuda_err) => {
                    tracing::warn!(error = %cuda_err, "CUDA graph decode failed; falling back to eager");
                    match model.forward_decode_batch(&input, &sequences, kv_cache_mgr) {
                        Ok(l) => l,
                        Err(e) => {
                            if is_cuda_oom(&e.to_string()) {
                                if let Some(v) = select_decode_oom_victim(&batch_ids, requests) {
                                    failed.push((v, format!("decode out of memory: {e}")));
                                    return (failed, None);
                                }
                            }
                            let msg = format!("forward_decode_batch (eager fallback): {e}");
                            tracing::error!(error = %e, "eager decode fallback failed");
                            failed.extend(failed_batch(msg, &batch_ids));
                            return (failed, None);
                        }
                    }
                }
            }
        } else {
            match model.forward_decode_batch_with_ctx(
                &input,
                &sequences,
                kv_cache_mgr,
                &forward_ctx,
            ) {
                Ok(l) => l,
                Err(e) => {
                    if is_cuda_oom(&e.to_string()) {
                        if let Some(v) = select_decode_oom_victim(&batch_ids, requests) {
                            failed.push((v, format!("decode out of memory: {e}")));
                            return (failed, None);
                        }
                    }
                    let msg = format!("forward_decode_batch_with_ctx: {e}");
                    tracing::error!(
                        error = %e,
                        input_dims = ?input.dims(),
                        batch_ids_len = batch_ids.len(),
                        sequences_len = sequences.len(),
                        decode_request_ids_len = decode_request_ids.len(),
                        seqlen_offsets = ?sequences.iter().map(|s| s.seqlen_offset).collect::<Vec<_>>(),
                        block_id_counts = ?sequences.iter().map(|s| s.block_ids.len()).collect::<Vec<_>>(),
                        "batched decode forward failed"
                    );
                    failed.extend(failed_batch(msg, &batch_ids));
                    return (failed, None);
                }
            }
        }
    };

    // Stamp the forward stage. We sync the device once here so the elapsed
    // time captures actual GPU work rather than just kernel launches.
    if let (Some(t0), Some(dev)) = (t_forward_start, step_dev_ref) {
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cd) = dev {
            let _ = cd.cuda_stream().synchronize();
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        step_profile::FORWARD.ns[bucket].fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    // Phase 11.2.D diagnostic: dump per-forward decode logits when
    // `VLLM_EXL3_DUMP_DIR` is set. Counters monotonically increment so
    // forward N's logits land at `decode_logits.<NNNN>.bin`. Used to
    // localize the first divergent forward between eager and capture.
    // Phase 11.2.D one-shot diagnostic: capture forward 1's logits +
    // pool-slot snapshot, then disable subsequent dumps. The
    // intermediate `tensor_to_bytes` GPU→CPU copies provoke
    // CUDA_ERROR_ILLEGAL_ADDRESS on the NEXT captured replay (most
    // likely via cudarc 0.16's auto event-tracking on the dtoh stream
    // interacting with captured graph state). For named-slot diff
    // diagnosis we only need forward 1, so dump-and-disable is the
    // path of least invasiveness.
    if crate::engine::debug_dump::is_enabled() {
        // D10 follow-up (task #222): dump logits EVERY decode forward so
        // we can diff per-step and locate which decode first diverges.
        // The earlier `ONCE` gating was a workaround for an event-tracking
        // interaction with captured replay; with `disable_event_tracking()`
        // active in production (server/src/main.rs::create_cuda_device)
        // the dtoh post-replay is safe. The first-forward pool-slot snapshot
        // stays one-shot since it captures setup state.
        crate::engine::debug_dump::dump_tensor("decode_logits", &logits);
        static ONCE_POOL: std::sync::Once = std::sync::Once::new();
        ONCE_POOL.call_once(|| {
            if let Ok(dir) = std::env::var("VLLM_EXL3_DUMP_DIR") {
                let subdir = std::path::PathBuf::from(dir).join("pool_forward_0000");
                crate::engine::output_pool::OutputPool::global().dump_used_slots(&subdir);
            }
        });
    }
    // D10: layer-dump flush is independent of debug_dump — it has its
    // own env (`VLLM_LAYER_DUMP_DIR`) and its own one-shot semantics.
    // `VLLM_LAYER_DUMP_AFTER_N` (default 0) defers the flush until that
    // many forwards have completed, so per-layer dumps reflect e.g. the
    // first divergent decode step rather than only decode 0.
    if crate::engine::layer_dump::is_enabled() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FLUSH_COUNTDOWN: AtomicUsize = AtomicUsize::new(usize::MAX);
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let n: usize = std::env::var("VLLM_LAYER_DUMP_AFTER_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            FLUSH_COUNTDOWN.store(n + 1, Ordering::Relaxed);
        });
        let remaining = FLUSH_COUNTDOWN.fetch_sub(1, Ordering::Relaxed);
        if remaining == 1 {
            crate::engine::layer_dump::flush();
        }
    }

    let t_sampling_start = step_profile::enabled().then(std::time::Instant::now);

    // Step 4: Extract logits and sample per-sequence
    let seq_dim = logits.dims()[1];
    let last_logits = match logits.narrow(1, seq_dim - 1, 1) {
        Ok(l) => match l.squeeze(1) {
            Ok(l) => l,
            Err(e) => {
                let msg = format!("last-logits squeeze: {e}");
                tracing::error!(error = %e, "batched decode last-logits squeeze failed");
                failed.extend(failed_batch(msg, &batch_ids));
                return (failed, None);
            }
        },
        Err(e) => {
            let msg = format!("last-logits narrow: {e}");
            tracing::error!(error = %e, "batched decode last-logits narrow failed");
            failed.extend(failed_batch(msg, &batch_ids));
            return (failed, None);
        }
    };

    // Step 5: Sample tokens and update state.
    //
    // GPU fast path: every sequence in the batch must be `gpu_eligible_strict`
    // (no multiplicative penalties, no min_p/typical_p, no allowed-list
    // whitelist, no beam search), no constraints (FSM-based guided
    // decoding), and no logprobs requested.  Additive modifiers
    // (logit_bias, freq+pres penalties, banned tokens, bad words) get
    // folded into a single `index_add` on the GPU before temperature
    // scaling, so they no longer force a full logits → CPU transfer.
    // v1.9: constraint is no longer a hard blocker for the GPU
    // sampler — `SamplingConstraint::supports_gpu()` returning `true`
    // means the constraint can produce a packed-i32 bitmask that the
    // engine uploads to GPU and applies via `gpu_apply_grammar_bitmask`
    // before sampling, keeping the rest of the pipeline (softmax,
    // top-k/top-p, multinomial) on-device.
    let all_gpu_eligible = batch_ids.iter().all(|&id| {
        requests.get(&id).is_some_and(|r| {
            r.state.sampling_params.gpu_eligible_strict()
                && r.state.num_top_logprobs.is_none()
                && r.state.constraint.as_ref().is_none_or(|c| c.supports_gpu())
        })
    });

    if all_gpu_eligible && sampling::gpu::gpu_sampling_available(last_logits.device()) {
        // Pre-generate random values (needs mutable access to sampler_state)
        let rand_vals: Vec<f32> = batch_ids
            .iter()
            .map(|&id| {
                let req = requests.get_mut(&id).unwrap();
                if req.state.sampling_params.is_greedy() {
                    0.0
                } else {
                    req.state.sampler_state.next_rand_f32()
                }
            })
            .collect();

        // Build per-sequence GPU sampling configs and additive diffs.
        let mut diffs: Vec<sampling::gpu::LogitsDiff> = Vec::new();
        let configs: Vec<sampling::gpu::GpuSamplingConfig> = batch_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let req = &requests[&id];
                let params = &req.state.sampling_params;
                let seq_idx = i as u32;

                // Frequency + presence penalties — counted over generated
                // tokens, applied once per unique token.
                if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
                    let mut counts: ahash::AHashMap<u32, u32> = ahash::AHashMap::new();
                    for &t in &req.state.generated_token_ids {
                        *counts.entry(t).or_insert(0) += 1;
                    }
                    for (&token_id, &count) in &counts {
                        let delta = -(params.frequency_penalty * count as f32
                            + if count > 0 {
                                params.presence_penalty
                            } else {
                                0.0
                            });
                        if delta != 0.0 {
                            diffs.push(sampling::gpu::LogitsDiff {
                                seq_idx,
                                token_id,
                                delta,
                            });
                        }
                    }
                }
                // Logit bias.
                if let Some(ref bias) = params.logit_bias {
                    for &(token_id, delta) in bias {
                        diffs.push(sampling::gpu::LogitsDiff {
                            seq_idx,
                            token_id,
                            delta,
                        });
                    }
                }
                // Banned tokens — set to -inf via additive sentinel.
                if let Some(ref banned) = params.banned_token_ids {
                    for &token_id in banned {
                        diffs.push(sampling::gpu::LogitsDiff {
                            seq_idx,
                            token_id,
                            delta: f32::NEG_INFINITY,
                        });
                    }
                }
                // Bad words — only the final token of a matched prefix is
                // disabled this step.  Collect those that match into the
                // diff list.
                if let Some(ref bad_words) = params.bad_words_token_ids {
                    for word in bad_words {
                        if word.is_empty() {
                            continue;
                        }
                        if word.len() == 1 {
                            diffs.push(sampling::gpu::LogitsDiff {
                                seq_idx,
                                token_id: word[0],
                                delta: f32::NEG_INFINITY,
                            });
                            continue;
                        }
                        let prefix_len = word.len() - 1;
                        let gen = &req.state.generated_token_ids;
                        if gen.len() < prefix_len {
                            continue;
                        }
                        if gen[gen.len() - prefix_len..] == word[..prefix_len] {
                            diffs.push(sampling::gpu::LogitsDiff {
                                seq_idx,
                                token_id: word[word.len() - 1],
                                delta: f32::NEG_INFINITY,
                            });
                        }
                    }
                }

                if params.is_greedy() {
                    sampling::gpu::GpuSamplingConfig {
                        inv_temperature: 0.0,
                        top_k: 0,
                        top_p: 1.0,
                        rand_val: 0.0,
                    }
                } else {
                    sampling::gpu::GpuSamplingConfig {
                        inv_temperature: 1.0 / params.temperature.max(1e-6),
                        top_k: params.top_k as i32,
                        top_p: params.top_p,
                        rand_val: rand_vals[i],
                    }
                }
            })
            .collect();

        // v1.9 constrained GPU path: any sequence in the batch
        // carrying a grammar constraint gets its per-step bitmask
        // built on CPU (one call per request — sequential for now;
        // `xgrammar_rs::BatchMatcher` is wired in the FFI but not
        // dispatched here yet, pending a Phase-7 perf measurement),
        // uploaded as a single `[batch, words_per_row]` int32 tensor,
        // and applied in-place to `last_logits` via the native
        // `apply_grammar_bitmask` CUDA kernel before sampling.
        let any_constrained = batch_ids.iter().any(|&id| {
            requests
                .get(&id)
                .is_some_and(|r| r.state.constraint.is_some())
        });
        let mut constrained_apply_failed = false;
        if any_constrained {
            if let Err(e) = apply_grammar_bitmask_to_logits(&last_logits, &batch_ids, requests) {
                tracing::warn!(
                    error = %e,
                    "GPU grammar bitmask apply failed; falling back to CPU sampler",
                );
                constrained_apply_failed = true;
            }
        }
        if !constrained_apply_failed {
            match sampling::gpu::gpu_sample_batch_with_diffs_to_tensor(
                &last_logits,
                &configs,
                &diffs,
            ) {
                Ok(token_tensor) => {
                    if defer_dtoh {
                        // ADR 0017 / Substep 2.2: skip the host-blocking
                        // `to_vec1` and `generated_token_ids.push`. Caller
                        // will sync a side-stream DtoH event and push the
                        // tokens once they land in pinned host memory. We
                        // still advance block-table and seqlen_offset
                        // because those are pure host counters that don't
                        // depend on the token value, and the next decode
                        // step's metadata reads them.
                        for &req_id in &batch_ids {
                            if let Some(req) = requests.get_mut(&req_id) {
                                req.state.block_table.advance(1);
                                req.state.seqlen_offset += 1;
                            }
                        }
                        // No SAMPLING profile sync — that would block on
                        // the very stream we're trying to keep busy.
                        if step_profile::enabled() {
                            step_profile::step_done(
                                batch_size,
                                step_start.elapsed().as_nanos() as u64,
                            );
                        }
                        return (failed, Some(token_tensor));
                    }
                    match token_tensor.to_vec1::<u32>() {
                        Ok(token_ids_out) => {
                            for (i, &req_id) in batch_ids.iter().enumerate() {
                                let Some(req) = requests.get_mut(&req_id) else {
                                    failed.push((req_id, "request no longer active".to_string()));
                                    continue;
                                };
                                // v1.9: advance grammar constraint state
                                // post-sample so the next step's bitmask
                                // reflects the freshly-committed token —
                                // mirrors the CPU path at lines 1499 / 1538.
                                if let Some(ref mut constraint) = req.state.constraint {
                                    constraint.accept_token(token_ids_out[i]);
                                }
                                req.state.block_table.advance(1);
                                req.state.seqlen_offset += 1;
                                req.state.generated_token_ids.push(token_ids_out[i]);
                            }
                            if let (Some(t0), Some(dev)) = (t_sampling_start, step_dev_ref) {
                                #[cfg(feature = "cuda")]
                                if let candle_core::Device::Cuda(cd) = dev {
                                    let _ = cd.cuda_stream().synchronize();
                                }
                                #[cfg(not(feature = "cuda"))]
                                let _ = dev;
                                step_profile::SAMPLING.ns[bucket].fetch_add(
                                    t0.elapsed().as_nanos() as u64,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            if step_profile::enabled() {
                                step_profile::step_done(
                                    batch_size,
                                    step_start.elapsed().as_nanos() as u64,
                                );
                            }
                            return (failed, Some(token_tensor));
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "GPU sampler tensor->vec failed; falling back to CPU");
                        }
                    }
                }
                Err(e) => {
                    // Fall through to CPU path on GPU sampling failure
                    tracing::warn!(error = %e, "GPU batched sampling failed; falling back to CPU");
                }
            }
        } // end `if !constrained_apply_failed`
    }

    // CPU path: transfer logits and sample per-sequence.
    // Cast to F32 first — quantized models (AWQ, GPTQ, FP8, ...) can return
    // BF16/F16 logits depending on activation dtype; sampler contract is F32.
    let logits_cpu: Vec<f32> = match last_logits
        .to_dtype(candle_core::DType::F32)
        .and_then(|t| t.to_vec2::<f32>())
    {
        Ok(v) => v.into_iter().flatten().collect(),
        Err(e) => {
            let msg = format!("last-logits to F32 vec2: {e}");
            tracing::error!(error = %e, "batched decode logits->cpu transfer failed");
            failed.extend(failed_batch(msg, &batch_ids));
            return (failed, None);
        }
    };
    let vocab_size = last_logits.dims()[1];

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

    if all_greedy && !any_need_logprobs {
        for (i, &req_id) in batch_ids.iter().enumerate() {
            let Some(req) = requests.get_mut(&req_id) else {
                failed.push((req_id, "request no longer active".to_string()));
                continue;
            };
            // Copy the logit row so we can mutate it (apply constraint mask).
            // Cheap: vocab_size × 4 bytes ≈ 600 KB for Qwen3.
            let mut logits: Vec<f32> = logits_cpu[i * vocab_size..(i + 1) * vocab_size].to_vec();
            if let Some(ref mut constraint) = req.state.constraint {
                // Pass empty `generated_text` — the constraints that
                // care about per-step state advance via `accept_token`
                // (grammar adapter / future per-token-state ones), not
                // via re-decoding the full sequence. Legacy text-only
                // constraints (`JsonSchemaConstraint`) only validate
                // the final output and don't mask intra-stream, so
                // they don't need the running text either. If a future
                // constraint needs the live text it should plumb the
                // tokenizer in through `ActiveRequest`.
                if let Err(e) = constraint.mask_logits(&mut logits, "") {
                    failed.push((req_id, format!("constraint error: {e}")));
                    continue;
                }
            }
            let token_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            if let Some(ref mut constraint) = req.state.constraint {
                constraint.accept_token(token_id);
            }
            req.state.block_table.advance(1);
            req.state.seqlen_offset += 1;
            req.state.generated_token_ids.push(token_id);
        }
    } else {
        for (i, &req_id) in batch_ids.iter().enumerate() {
            let Some(req) = requests.get_mut(&req_id) else {
                failed.push((req_id, "request no longer active".to_string()));
                continue;
            };
            let mut logits: Vec<f32> = logits_cpu[i * vocab_size..(i + 1) * vocab_size].to_vec();
            if let Some(ref mut constraint) = req.state.constraint {
                // Pass empty `generated_text` — the constraints that
                // care about per-step state advance via `accept_token`
                // (grammar adapter / future per-token-state ones), not
                // via re-decoding the full sequence. Legacy text-only
                // constraints (`JsonSchemaConstraint`) only validate
                // the final output and don't mask intra-stream, so
                // they don't need the running text either. If a future
                // constraint needs the live text it should plumb the
                // tokenizer in through `ActiveRequest`.
                if let Err(e) = constraint.mask_logits(&mut logits, "") {
                    failed.push((req_id, format!("constraint error: {e}")));
                    continue;
                }
            }
            let result = sampling::sample(
                &logits,
                &req.state.sampling_params,
                &req.state.generated_token_ids,
                &mut req.state.sampler_state,
                req.state.num_top_logprobs,
                &req.state.stop_token_ids,
            );
            if let Some(ref mut constraint) = req.state.constraint {
                constraint.accept_token(result.token_id);
            }
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

    if let (Some(t0), Some(dev)) = (t_sampling_start, step_dev_ref) {
        #[cfg(feature = "cuda")]
        if let candle_core::Device::Cuda(cd) = dev {
            let _ = cd.cuda_stream().synchronize();
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        step_profile::SAMPLING.ns[bucket].fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if step_profile::enabled() {
        step_profile::step_done(batch_size, step_start.elapsed().as_nanos() as u64);
    }

    // CPU sampler path: no GPU tensor to carry forward; next step uses host vec.
    (failed, None)
}

/// Pipelined multi-step decode (ADR 0017 / Substep 2.2).
///
/// Replaces the serial multi-step loop's "forward → sampler →
/// to_vec1 (blocks ~50 ms) → next forward" pattern with "forward →
/// sampler → async DtoH on side stream → next forward immediately".
/// The host syncs the side-stream events only at the END of the
/// multi-step block — one host stall instead of N.
///
/// Invariants:
/// - `infra.slots.len() >= num_steps` AND `infra.max_batch >=
///   decode_request_ids.len()` — caller (StandardExecution) guarantees
///   this; if violated we bail to safe sync DtoH for that step.
/// - The model device is CUDA (env gate already verified).
/// - Tensors held in `pending` keep their underlying `CudaSlice`
///   alive until the corresponding pinned slot has been read.
/// - Push of `generated_token_ids` is deferred until the drain phase;
///   block-table / seqlen counters advance per step (they don't
///   depend on the token value).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_pipelined_multi_step_decode<M: ModelForward>(
    decode_request_ids: &[RequestId],
    num_steps: usize,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    dispatcher: Option<&Arc<std::sync::RwLock<CudaGraphDispatcher>>>,
    graph_runner: Option<&Arc<CudaGraphRunner>>,
    preempted_out: &mut Vec<RequestId>,
    infra: &mut super::async_sampler::AsyncSamplerInfra,
) -> Vec<(RequestId, String)> {
    let prof_enabled = std::env::var("VLLM_PROFILE_PIPELINED").is_ok();
    let prof_total_start = prof_enabled.then(std::time::Instant::now);

    // v1.9: any request with a grammar constraint MUST observe its
    // freshly-sampled token before the next step's bitmask is built —
    // otherwise the xgrammar matcher never advances and the GPU path
    // emits broken output (initial-state bitmask reused every step).
    // The pipelined async path defers the sampler-tensor DtoH past
    // the next forward, breaking that invariant. Fall back to the
    // synchronous serial decoder, which runs accept_token inline.
    let any_constrained = decode_request_ids.iter().any(|id| {
        requests
            .get(id)
            .map(|r| r.state.constraint.is_some())
            .unwrap_or(false)
    });
    if any_constrained {
        return run_serial_fallback_decode(
            decode_request_ids,
            num_steps,
            model,
            kv_cache_mgr,
            requests,
            dispatcher,
            graph_runner,
            preempted_out,
        );
    }

    let mut failed: Vec<(RequestId, String)> = Vec::new();
    let mut active: Vec<RequestId> = decode_request_ids.to_vec();
    let mut prev_sampled: Option<Tensor> = None;

    // Get main CUDA stream for cross-stream event synchronisation.
    let main_stream = match model.device() {
        candle_core::Device::Cuda(d) => d.cuda_stream(),
        _ => {
            tracing::warn!("pipelined decode invoked on non-CUDA device; std fallback");
            // Safe fallback: run serial sync path
            return run_serial_fallback_decode(
                decode_request_ids,
                num_steps,
                model,
                kv_cache_mgr,
                requests,
                dispatcher,
                graph_runner,
                preempted_out,
            );
        }
    };

    // Per-step pending: (batch_ids, slot_idx, tensor_keep_alive). Tensor
    // is kept alive until the pinned slot is drained — its underlying
    // CudaSlice is what dtoh_stream's memcpy reads from.
    let mut pending: Vec<(Vec<RequestId>, usize, Tensor)> = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        if active.is_empty() {
            break;
        }
        // Capacity guards. If violated, bail this step into sync path.
        if step >= infra.slots.len() || active.len() > infra.max_batch {
            tracing::warn!(
                step,
                slots = infra.slots.len(),
                batch = active.len(),
                max_batch = infra.max_batch,
                "pipelined decode capacity exceeded; bailing remainder of multi-step block"
            );
            break;
        }

        let active_size_before = active.len();
        let active_at_call = active.clone();
        let mut step_preempted: Vec<RequestId> = Vec::new();

        let (step_failed, sampler_tensor) = execute_batched_decode_with_graph(
            &active,
            prev_sampled.as_ref(),
            true, // defer_dtoh — that's the whole point
            model,
            kv_cache_mgr,
            requests,
            dispatcher,
            graph_runner,
            &mut step_preempted,
        );

        // Compute batch_ids in active order (subset of active that
        // actually went through Step 1's KV-alloc and was processed
        // by the sampler).
        let failed_set: std::collections::HashSet<RequestId> =
            step_failed.iter().map(|(id, _)| *id).collect();
        let preempted_set: std::collections::HashSet<RequestId> =
            step_preempted.iter().copied().collect();
        let batch_ids: Vec<RequestId> = active_at_call
            .iter()
            .copied()
            .filter(|id| !failed_set.contains(id) && !preempted_set.contains(id))
            .collect();

        // Apply step's failed/preempted to outer state.
        for (req_id, msg) in step_failed {
            failed.push((req_id, msg));
            active.retain(|&id| id != req_id);
        }
        for &p in &step_preempted {
            active.retain(|&id| id != p);
        }
        preempted_out.extend(step_preempted);

        // Dispatch async DtoH if helper produced a sampler tensor.
        // (None = CPU fallback path inside helper which already pushed
        // tokens synchronously; next step rebuilds input from host.)
        if let Some(tensor) = sampler_tensor {
            let actual_size = tensor.dims().first().copied().unwrap_or(0);
            if actual_size != batch_ids.len() {
                tracing::warn!(
                    actual = actual_size,
                    batch_ids_len = batch_ids.len(),
                    "sampler tensor / batch_ids size mismatch in pipelined decode; sync fallback"
                );
                if let Ok(ids) = tensor.to_vec1::<u32>() {
                    for (i, &id) in batch_ids.iter().enumerate().take(ids.len()) {
                        if let Some(req) = requests.get_mut(&id) {
                            req.state.generated_token_ids.push(ids[i]);
                        }
                    }
                }
                prev_sampled = None;
                continue;
            }

            match dispatch_async_dtoh_for_sampler(
                &tensor,
                actual_size,
                &main_stream,
                &infra.dtoh_stream,
                &mut infra.slots[step],
            ) {
                Ok(()) => {
                    pending.push((batch_ids.clone(), step, tensor.clone()));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "async DtoH dispatch failed; sync fallback this step");
                    if let Ok(ids) = tensor.to_vec1::<u32>() {
                        for (i, &id) in batch_ids.iter().enumerate().take(ids.len()) {
                            if let Some(req) = requests.get_mut(&id) {
                                req.state.generated_token_ids.push(ids[i]);
                            }
                        }
                    }
                }
            }

            prev_sampled = if active.len() == active_size_before {
                Some(tensor)
            } else {
                None
            };
        } else {
            prev_sampled = None;
        }
    }

    // Profile: time the dispatch-loop wallclock vs the drain.
    let prof_enqueue_us = prof_total_start.map(|t| t.elapsed().as_micros());
    let prof_drain_start = prof_enabled.then(std::time::Instant::now);

    // Drain pending DtoHs and push tokens. `as_slice()` syncs each
    // slot's pinned event (= sampler kernel + memcpy_dtoh complete);
    // total wallclock is dominated by the LAST step's drain.
    for (batch_ids, slot_idx, _tensor_alive) in &pending {
        if let Err(e) = infra.slots[*slot_idx].event_synchronize() {
            tracing::warn!(error = %e, "pinned slot event_synchronize failed");
            continue;
        }
        let buf = infra.slots[*slot_idx].as_slice();
        for (i, &id) in batch_ids.iter().enumerate() {
            if i >= buf.len() {
                break;
            }
            if let Some(req) = requests.get_mut(&id) {
                req.state.generated_token_ids.push(buf[i]);
            }
        }
    }
    if let (Some(t0), Some(enq_us)) = (prof_drain_start, prof_enqueue_us) {
        let drain_us = t0.elapsed().as_micros();
        let total_us = prof_total_start.unwrap().elapsed().as_micros();
        tracing::info!(
            target: "vllm_core::async_sampling",
            enqueue_us = enq_us as u64,
            drain_us = drain_us as u64,
            total_us = total_us as u64,
            steps = pending.len(),
            "pipelined timings"
        );
    }
    drop(pending); // tensors drop here, freeing their CUDA storage

    // Diagnostic counter — confirms env-gated path is exercised in
    // production benches (memory rule `feedback_shape_gate_diag.md`).
    let n = infra
        .call_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    if n == 1 || n.is_power_of_two() || n.is_multiple_of(200) {
        tracing::info!(
            target: "vllm_core::async_sampling",
            calls = n,
            "pipelined multi-step decode active"
        );
    }

    failed
}

/// Issue an async `DtoH` on `dtoh_stream` for the first `n_elems`
/// elements of `tensor`'s underlying `CudaSlice<u32>` into `pinned`.
/// Cross-stream synchronisation is established via a `record_event` on
/// `main_stream` and `wait` on `dtoh_stream` — non-blocking on the host.
/// On `Ok(())` the pinned slot's internal event tracks the memcpy
/// completion; the caller reads via `pinned.as_slice()` later.
#[cfg(feature = "cuda")]
fn dispatch_async_dtoh_for_sampler(
    tensor: &Tensor,
    n_elems: usize,
    main_stream: &Arc<candle_core::cuda::cudarc::driver::CudaStream>,
    dtoh_stream: &Arc<candle_core::cuda::cudarc::driver::CudaStream>,
    pinned: &mut super::async_sampler::SamplerPinnedSlot,
) -> candle_core::Result<()> {
    use candle_core::Storage;

    // Ensure tensor is contiguous before extracting CudaSlice.
    // sampler returns a 1D [B] tensor that should already be contiguous
    // (allocated fresh by the kernel) — this is a defensive guard.
    let owned;
    let t = if tensor.is_contiguous() {
        tensor
    } else {
        owned = tensor.contiguous()?;
        &owned
    };

    let (storage, _layout) = t.storage_and_layout();
    let cuda_storage = match &*storage {
        Storage::Cuda(s) => s,
        _ => {
            return Err(candle_core::Error::Msg(
                "dispatch_async_dtoh: non-CUDA storage".into(),
            ));
        }
    };
    let cuda_slice = cuda_storage.as_cuda_slice::<u32>()?;

    // Cross-stream sync: dtoh waits for main's sampler kernel.
    let main_event = main_stream
        .record_event(None)
        .map_err(|e| candle_core::Error::Msg(format!("main record_event: {e}")))?;
    dtoh_stream
        .wait(&main_event)
        .map_err(|e| candle_core::Error::Msg(format!("dtoh wait: {e}")))?;

    // Issue the DtoH on the side stream. We pass a `&mut [u32]` of
    // length `n_elems` as the destination — cudarc's `[T]` HostSlice
    // impl uses `SyncOnDrop::Sync(None)` (no post-op sync), and the
    // driver auto-detects the pointer as pinned (we allocated via
    // `cuMemHostAlloc`), so the copy stays asynchronous.
    let src_view = cuda_slice.slice(0..n_elems);
    // SAFETY: pinned slot owns the memory for max_batch u32s; we
    // sub-slice to n_elems which fits. We record the dtoh_stream
    // event onto the slot's internal event right after queueing so
    // the caller can synchronise on it before reading.
    let dst_slice: &mut [u32] = unsafe { pinned.as_mut_slice_first(n_elems) };
    dtoh_stream
        .memcpy_dtoh(&src_view, dst_slice)
        .map_err(|e| candle_core::Error::Msg(format!("memcpy_dtoh: {e}")))?;
    // Record dtoh_stream completion onto pinned slot's event so we
    // can synchronise on it during the drain phase.
    pinned.record_event_on(dtoh_stream)?;

    Ok(())
}

/// Defensive serial fallback used by the pipelined helper only when
/// the device unexpectedly turns out non-CUDA (env gate should
/// prevent this; here as a last-resort safety net).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_serial_fallback_decode<M: ModelForward>(
    decode_request_ids: &[RequestId],
    num_steps: usize,
    model: &M,
    kv_cache_mgr: &mut KVCacheManager,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    dispatcher: Option<&Arc<std::sync::RwLock<CudaGraphDispatcher>>>,
    graph_runner: Option<&Arc<CudaGraphRunner>>,
    preempted_out: &mut Vec<RequestId>,
) -> Vec<(RequestId, String)> {
    let mut failed: Vec<(RequestId, String)> = Vec::new();
    let mut active: Vec<RequestId> = decode_request_ids.to_vec();
    let mut prev: Option<Tensor> = None;
    for _ in 0..num_steps {
        if active.is_empty() {
            break;
        }
        let size_before = active.len();
        let mut p: Vec<RequestId> = Vec::new();
        let (f, next) = execute_batched_decode_with_graph(
            &active,
            prev.as_ref(),
            false,
            model,
            kv_cache_mgr,
            requests,
            dispatcher,
            graph_runner,
            &mut p,
        );
        for (id, m) in f {
            failed.push((id, m));
            active.retain(|&x| x != id);
        }
        for &x in &p {
            active.retain(|&y| y != x);
        }
        preempted_out.extend(p);
        active.retain(|&id| {
            requests
                .get(&id)
                .map(|r| {
                    let s = &r.state;
                    let last = s.generated_token_ids.last().copied();
                    let eos = last.map(|t| t == s.eos_token_id).unwrap_or(false);
                    let stop_token = last.map(|t| s.stop_token_ids.contains(&t)).unwrap_or(false);
                    let max_len = s.num_generated() >= s.max_new_tokens;
                    !eos && !stop_token && !max_len
                })
                .unwrap_or(false)
        });
        prev = if active.len() == size_before {
            next
        } else {
            None
        };
    }
    failed
}

/// Whether a sequence has reached the effective context limit. Total context
/// (prompt + generated) `>=` the engine's `max_model_len` (clamped to the KV
/// pool capacity at startup) means the next decode step cannot allocate a slot
/// — the sequence must finish with `Length` here rather than overrun the pool
/// and wedge the FCFS engine in a preempt-for-recompute loop it can never exit.
/// `>=` (not `>`) is load-bearing: a sequence that *exactly* fills the pool
/// still cannot generate the next token. `None` (limit unpublished, e.g. unit
/// tests that never call `set_max_model_len`) disables the check.
pub(crate) fn context_capacity_reached(total_len: usize, max_model_len: Option<usize>) -> bool {
    matches!(max_model_len, Some(limit) if total_len >= limit)
}

pub(crate) fn check_finished(
    state: &SequenceState,
    tokenizer: &TokenizerWrapper,
) -> Option<FinishCheck> {
    let last_token = *state.generated_token_ids.last()?;

    // Context-capacity stop (hard limit, overrides min_tokens). Total context
    // = prompt + generated cannot exceed the engine's effective `max_model_len`,
    // which is clamped to the KV-pool capacity at startup
    // (`clamp_max_model_len_to_capacity`). Without this the sequence keeps
    // generating until `max_new_tokens`, then exhausts the pool mid-decode →
    // the decode-step preempt-for-recompute can never make progress (the full
    // context no longer fits) → the request wedges the FCFS engine and stalls
    // every request behind it. Stopping here at the boundary returns the
    // generated tokens with `Length` instead. Inactive (None) in unit tests
    // that never call `set_max_model_len`.
    if context_capacity_reached(
        state.total_len(),
        crate::engine::engine_limits::max_model_len(),
    ) {
        return Some(FinishCheck {
            reason: FinishReason::Length,
            trim_bytes: 0,
            stop_reason: None,
        });
    }

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
        // Detokenizing every accumulated token each decode step is O(N)
        // and dominates check_finished for long generations. We only need
        // enough of the tail to decide whether the suffix matches any
        // stop string. Budget = max(stop_string byte length) tokens × 2,
        // floored at 16 — large enough to absorb multi-byte UTF-8 splits
        // and BPE continuations in practice, far smaller than full
        // history for any non-trivial generation.
        let max_stop_bytes = state
            .stop_strings
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        let budget = max_stop_bytes.saturating_mul(2).max(16);
        let start = state.generated_token_ids.len().saturating_sub(budget);
        let tail = &state.generated_token_ids[start..];
        if let Ok(text) = tokenizer.decode(tail) {
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
        &state.stop_token_ids,
    );

    // Advance the constraint's state machine with the freshly-sampled
    // token so the next step's `mask_logits` reflects the grammar's
    // post-token position. Default `accept_token` impl is a no-op for
    // text-based constraints; stateful grammar adapters override.
    if let Some(ref mut constraint) = state.constraint {
        constraint.accept_token(result.token_id);
    }

    if state.num_top_logprobs.is_some() {
        state.token_logprobs.push(result.logprob);
        if let Some(top) = result.top_logprobs {
            state.top_logprobs.push(top);
        }
    }

    Ok(result.token_id)
}

/// Finish a request with an error, deferring scheduler cleanup.
///
/// Removes the request from the map and sends the error to the caller.
/// Returns `Some(req_id)` if the request was found (so the caller can
/// collect errored IDs for deferred `scheduler.remove_request()`).
pub(crate) fn finish_request_with_error_deferred(
    req_id: RequestId,
    error: EngineError,
    requests: &mut HashMap<RequestId, ActiveRequest>,
    kv_cache_mgr: &mut KVCacheManager,
) -> Option<RequestId> {
    if let Some(mut req) = requests.remove(&req_id) {
        // Free KV cache blocks BEFORE dropping the request — without this
        // the blocks stay allocated in the pool's free-list accounting
        // even though no request owns them, eventually hard-deadlocking
        // the scheduler (cache=100%, num_running=0, num_waiting>0).
        // Diagnosed 2026-05-08 under c=16 bench: prefill OOM cascade
        // leaked all 384 blocks within seconds.
        if let Err(e) = kv_cache_mgr.free_request(&mut req.state.block_table) {
            tracing::warn!(error = %e, request_id = req_id, "Failed to free request KV blocks during error finalisation");
        }
        send_error(req.response, error);
        Some(req_id)
    } else {
        None
    }
}

/// Send streaming token events for a request.
///
/// Returns `true` if the stream receiver is still alive, `false` if
/// it has been dropped (client disconnected). When `false`, the caller
/// should abort the request to reclaim GPU resources.
pub(crate) fn send_stream_token(
    req_id: RequestId,
    tokenizer: &TokenizerWrapper,
    requests: &mut HashMap<RequestId, ActiveRequest>,
) -> bool {
    let Some(req) = requests.get_mut(&req_id) else {
        return true;
    };
    let ResponseChannel::Stream(ref tx) = req.response else {
        return true;
    };

    // Early check: if the receiver has been dropped, signal disconnect
    if tx.is_closed() {
        return false;
    }

    let start = req.num_streamed_tokens;
    let end = req.state.generated_token_ids.len();
    if start >= end {
        return true;
    }
    for i in start..end {
        let token_id = req.state.generated_token_ids[i];
        let text_so_far = tokenizer
            .decode(&req.state.generated_token_ids[..=i])
            .unwrap_or_default();
        // SAFETY: streamed_text_len from a previous iteration may not land
        // on a char boundary if cumulative tokenizer decode is non-monotonic
        // (rare but observed with CJK / multi-byte BPE merges) — `.get()`
        // returns None instead of panicking; in that case emit nothing this
        // round and DON'T advance streamed_text_len, so the next iteration's
        // longer prefix has a chance to land on a valid boundary.
        let (token_text, advance_to) = match text_so_far.get(req.streamed_text_len..) {
            Some(s) => (s.to_string(), text_so_far.len()),
            None => (String::new(), req.streamed_text_len),
        };
        req.streamed_text_len = advance_to;

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

        if let Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) =
            tx.try_send(StreamEvent::Token {
                token_id,
                token_text,
                logprob,
                top_logprobs,
            })
        {
            return false;
        }
    }
    req.num_streamed_tokens = end;
    true
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
    use crate::kv_cache::BlockTable;
    use candle_core::Device;

    fn make_state(eos: u32, max_new: usize) -> SequenceState {
        SequenceState::new(0, vec![1, 2, 3], max_new, eos, 16, 0)
    }

    fn dummy_tokenizer() -> TokenizerWrapper {
        TokenizerWrapper::for_testing(100)
    }

    /// Build a request that has exhausted a KV pool of `num_blocks`
    /// 16-token blocks exactly at a block boundary: `prompt_len`
    /// prompt + 2 generated tokens, all pool blocks held and full, so
    /// the next decode step must allocate one more block.
    fn boundary_exhausted_request(
        prompt_len: usize,
        num_blocks: usize,
    ) -> (KVCacheManager, HashMap<RequestId, ActiveRequest>, RequestId) {
        use crate::engine::types::ResponseChannel;
        use crate::kv_cache::{CacheConfig, KVCacheDtype};

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_config).unwrap();

        let req_id: RequestId = 0;
        let mut state = SequenceState::new(req_id, (0..prompt_len as u32).collect(), 64, 99, 16, 0);
        state.generated_token_ids = vec![7, 8];
        state.status = RequestStatus::Decoding;
        // Occupy `num_blocks` blocks, all full: stored = num_blocks*16
        // tokens (prompt + first generated token's KV), so writing the
        // second generated token's KV needs a new block.
        kv.allocate_for_request(&mut state.block_table, num_blocks * 16)
            .unwrap();
        state.block_table.advance(num_blocks * 16);
        state.seqlen_offset = num_blocks * 16;
        state.num_computed_tokens = prompt_len;

        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut requests = HashMap::new();
        requests.insert(
            req_id,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        (kv, requests, req_id)
    }

    /// Regression: live wedge on gemma-4 (2026-06-07). A sole running
    /// request exhausted the 22-block KV pool exactly at a block
    /// boundary; preempt-for-recompute folded the generated tokens into
    /// the prompt, after which the re-prefill needed MORE blocks than
    /// the whole pool. The request sat in the waiting queue forever and
    /// (FCFS head-of-line) starved every request queued behind it.
    /// A request whose folded context can never fit must FAIL with an
    /// explicit error instead of being preempted.
    #[test]
    fn preempt_at_capacity_boundary_fails_request_instead_of_wedging() {
        struct NoopModel(Device);
        impl ModelForward for NoopModel {
            fn forward(
                &self,
                _input_ids: &Tensor,
                _seqlen_offset: usize,
                _kv_cache_mgr: &mut KVCacheManager,
                _block_table: &BlockTable,
                _slot_mapping: &[usize],
            ) -> candle_core::Result<Tensor> {
                unreachable!("allocation must fail before forward")
            }
            fn device(&self) -> &Device {
                &self.0
            }
        }

        // Pool: 2 blocks × 16. Request: 31-token prompt + 2 generated;
        // 32 KV entries stored (prompt + first generated token — both
        // blocks full), the second generated token's KV needs a third
        // block. Folded recompute prompt = 33 tokens ⇒ 3 blocks > 2
        // total ⇒ impossible.
        let (mut kv, mut requests, req_id) = boundary_exhausted_request(31, 2);
        let mut preempted: Vec<RequestId> = Vec::new();
        let (failed, _next) = execute_batched_decode_with_graph(
            &[req_id],
            None,
            false,
            &NoopModel(Device::Cpu),
            &mut kv,
            &mut requests,
            None,
            None,
            &mut preempted,
        );

        assert!(
            preempted.is_empty(),
            "impossible request must not be preempted (it would wedge the queue): {preempted:?}"
        );
        assert_eq!(failed.len(), 1, "expected exactly one failure: {failed:?}");
        assert_eq!(failed[0].0, req_id);
        assert!(
            failed[0].1.contains("KV cache"),
            "error should explain capacity: {}",
            failed[0].1
        );
        // All pool blocks must be back in the free pool.
        assert_eq!(kv.num_free_blocks(), 2);
    }

    /// Companion: when the folded context still fits the pool (other
    /// requests' blocks will free up), preemption-for-recompute must
    /// keep working as before.
    #[test]
    fn preempt_when_refit_possible_still_preempts() {
        struct NoopModel(Device);
        impl ModelForward for NoopModel {
            fn forward(
                &self,
                _input_ids: &Tensor,
                _seqlen_offset: usize,
                _kv_cache_mgr: &mut KVCacheManager,
                _block_table: &BlockTable,
                _slot_mapping: &[usize],
            ) -> candle_core::Result<Tensor> {
                unreachable!("allocation must fail before forward")
            }
            fn device(&self) -> &Device {
                &self.0
            }
        }

        // Pool: 3 blocks × 16, but one block is held by another request
        // (simulated by allocating it out of the pool) → the running
        // request holds 2 full blocks and allocation of a 3rd fails.
        // Folded prompt = 32 tokens ⇒ 2 blocks ≤ 3 total ⇒ recompute is
        // possible once the other block frees: must preempt, not fail.
        // The cache from this helper is discarded — we rebuild with a
        // 3-block pool below (same geometry, but hold one block externally).
        let (_, mut requests, req_id) = boundary_exhausted_request(30, 2);
        use crate::kv_cache::{CacheConfig, KVCacheDtype};
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 3,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv = KVCacheManager::new(&cache_config).unwrap();
        let mut other_table = BlockTable::new(16);
        kv.allocate_for_request(&mut other_table, 16).unwrap();
        {
            let state = &mut requests.get_mut(&req_id).unwrap().state;
            state.block_table = BlockTable::new(16);
            kv.allocate_for_request(&mut state.block_table, 32).unwrap();
            state.block_table.advance(32);
        }

        let mut preempted: Vec<RequestId> = Vec::new();
        let (failed, _next) = execute_batched_decode_with_graph(
            &[req_id],
            None,
            false,
            &NoopModel(Device::Cpu),
            &mut kv,
            &mut requests,
            None,
            None,
            &mut preempted,
        );

        assert!(
            failed.is_empty(),
            "refittable request must not fail: {failed:?}"
        );
        assert_eq!(preempted, vec![req_id]);
        let state = &requests[&req_id].state;
        assert_eq!(state.status, RequestStatus::Preempted);
        // vLLM-model recompute: prompt and generated stay SEPARATE (no fold);
        // only num_computed_tokens / seqlen_offset reset. The re-prefill will
        // recompute the full sequence (prompt + generated) via total_len.
        assert_eq!(state.prompt_token_ids.len(), 30, "prompt unchanged");
        assert_eq!(state.generated_token_ids, vec![7, 8], "output preserved");
        assert_eq!(state.total_len(), 32);
        assert_eq!(state.num_computed_tokens, 0);
        assert_eq!(state.seqlen_offset, 0);
    }

    // ─── reap_disconnected_requests (client-disconnect cancellation) ──────

    use crate::scheduler::{Scheduler, SchedulerConfig};

    /// Minimal 8-block pool + scheduler for disconnect-scan tests.
    fn reap_fixture() -> (Scheduler, KVCacheManager) {
        use crate::kv_cache::{CacheConfig, KVCacheDtype};
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        (
            Scheduler::new(SchedulerConfig::default()),
            KVCacheManager::new(&cache_config).unwrap(),
        )
    }

    /// Insert a waiting request holding `held_blocks` KV blocks, with the
    /// given response channel, and register it with the scheduler.
    fn insert_waiting(
        scheduler: &mut Scheduler,
        kv: &mut KVCacheManager,
        requests: &mut HashMap<RequestId, ActiveRequest>,
        id: RequestId,
        response: ResponseChannel,
        held_blocks: usize,
    ) {
        let mut state = SequenceState::new(id, vec![1, 2, 3], 64, 99, 16, id);
        if held_blocks > 0 {
            kv.allocate_for_request(&mut state.block_table, held_blocks * 16)
                .unwrap();
            state.block_table.advance(held_blocks * 16);
        }
        scheduler.add_request(id);
        requests.insert(
            id,
            ActiveRequest {
                state,
                response,
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
    }

    /// A non-streaming (Complete) request whose client gave up — receiver
    /// dropped — must be reaped from BOTH the requests map and the
    /// scheduler's waiting queue, and its KV blocks freed. This is the
    /// num_waiting-climbs-forever leak.
    #[test]
    fn reap_drops_disconnected_nonstreaming_waiting_request() {
        let (mut scheduler, mut kv) = reap_fixture();
        let mut requests = HashMap::new();

        let (tx, rx) = tokio::sync::oneshot::channel::<Result<GenerationResult, EngineError>>();
        insert_waiting(
            &mut scheduler,
            &mut kv,
            &mut requests,
            7,
            ResponseChannel::Complete(tx),
            2,
        );
        let free_before = kv.num_free_blocks();
        drop(rx); // client disconnects while the request is still waiting

        let reaped = reap_disconnected_requests(&mut scheduler, &mut requests, &mut kv);

        assert_eq!(reaped, 1);
        assert!(!requests.contains_key(&7));
        assert!(scheduler.is_idle(), "waiting queue must be drained");
        assert_eq!(kv.num_free_blocks(), free_before + 2, "KV blocks freed");
    }

    /// Same for a streaming (Stream) request — the engine scan is the
    /// authoritative backstop even though streaming also has an
    /// AbortGuard on the HTTP side.
    #[test]
    fn reap_drops_disconnected_streaming_request() {
        let (mut scheduler, mut kv) = reap_fixture();
        let mut requests = HashMap::new();

        let (tx, rx) = tokio::sync::mpsc::channel::<super::super::types::StreamEvent>(4);
        insert_waiting(
            &mut scheduler,
            &mut kv,
            &mut requests,
            9,
            ResponseChannel::Stream(tx),
            1,
        );
        drop(rx);

        let reaped = reap_disconnected_requests(&mut scheduler, &mut requests, &mut kv);

        assert_eq!(reaped, 1);
        assert!(!requests.contains_key(&9));
        assert!(scheduler.is_idle());
    }

    /// A live client (receiver still held) must NOT be reaped.
    #[test]
    fn reap_keeps_connected_requests() {
        let (mut scheduler, mut kv) = reap_fixture();
        let mut requests = HashMap::new();

        let (tx, _rx_alive) =
            tokio::sync::oneshot::channel::<Result<GenerationResult, EngineError>>();
        insert_waiting(
            &mut scheduler,
            &mut kv,
            &mut requests,
            3,
            ResponseChannel::Complete(tx),
            1,
        );

        let reaped = reap_disconnected_requests(&mut scheduler, &mut requests, &mut kv);

        assert_eq!(reaped, 0);
        assert!(requests.contains_key(&3));
        assert!(!scheduler.is_idle());
        // _rx_alive holds the channel open until end of scope.
    }

    /// Mixed batch: only the disconnected ones go, the live one stays.
    #[test]
    fn reap_only_removes_disconnected_in_mixed_batch() {
        let (mut scheduler, mut kv) = reap_fixture();
        let mut requests = HashMap::new();

        let (tx_dead, rx_dead) =
            tokio::sync::oneshot::channel::<Result<GenerationResult, EngineError>>();
        let (tx_live, _rx_live) =
            tokio::sync::oneshot::channel::<Result<GenerationResult, EngineError>>();
        insert_waiting(
            &mut scheduler,
            &mut kv,
            &mut requests,
            1,
            ResponseChannel::Complete(tx_dead),
            1,
        );
        insert_waiting(
            &mut scheduler,
            &mut kv,
            &mut requests,
            2,
            ResponseChannel::Complete(tx_live),
            1,
        );
        drop(rx_dead);

        let reaped = reap_disconnected_requests(&mut scheduler, &mut requests, &mut kv);

        assert_eq!(reaped, 1);
        assert!(!requests.contains_key(&1));
        assert!(requests.contains_key(&2));
    }

    // ─── is_cuda_oom + OOM backpressure ───────────────────────────────────

    #[test]
    fn is_cuda_oom_matches_driver_and_generic_wording() {
        assert!(is_cuda_oom(
            "forward_decode_batch_with_ctx: cache read: candle error: \
             DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(is_cuda_oom(
            "model error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, ...)"
        ));
        assert!(is_cuda_oom("Out Of Memory"));
        assert!(!is_cuda_oom("shape mismatch in reshape"));
        assert!(!is_cuda_oom("cache read: invalid block id"));
    }

    /// OOM on a multi-sequence batch selects the NEWEST request (highest
    /// arrival_order) as the single victim to fail — survivors are left
    /// untouched and retry next step. The whole batch is never killed.
    #[test]
    fn decode_oom_victim_is_newest_of_batch() {
        let (_sched, mut kv) = reap_fixture();
        let mut requests = HashMap::new();
        // Three running requests, arrival_order 10/20/30; id=3 is newest.
        for (id, arrival) in [(1u64, 10u64), (2, 20), (3, 30)] {
            let mut state = SequenceState::new(id, vec![1, 2, 3], 64, 99, 16, arrival);
            state.generated_token_ids = vec![7];
            kv.allocate_for_request(&mut state.block_table, 16).unwrap();
            state.block_table.advance(16);
            let (tx, _rx) = tokio::sync::oneshot::channel();
            requests.insert(
                id,
                ActiveRequest {
                    state,
                    response: ResponseChannel::Complete(tx),
                    num_streamed_tokens: 0,
                    streamed_text_len: 0,
                    beam_state: None,
                },
            );
        }
        let victim = select_decode_oom_victim(&[1u64, 2, 3], &requests);
        assert_eq!(victim, Some(3), "newest (arrival_order 30) is the victim");
        // Selection does not mutate the requests map (caller fails the victim).
        assert!(requests.contains_key(&1));
        assert!(requests.contains_key(&2));
        assert!(requests.contains_key(&3));
    }

    /// Single-sequence batch: the lone request is the victim.
    #[test]
    fn decode_oom_victim_single_sequence() {
        let (_sched, mut kv) = reap_fixture();
        let mut requests = HashMap::new();
        let mut state = SequenceState::new(1, vec![1, 2, 3], 64, 99, 16, 0);
        kv.allocate_for_request(&mut state.block_table, 16).unwrap();
        state.block_table.advance(16);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        requests.insert(
            1,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        assert_eq!(select_decode_oom_victim(&[1u64], &requests), Some(1));
        assert_eq!(select_decode_oom_victim(&[], &requests), None);
    }

    /// Prefill OOM with retries left → preempt-requeue (Ok), KV freed,
    /// status Preempted, oom_retries incremented. So concurrent prefills
    /// serialize under memory pressure instead of OOM-killing.
    #[test]
    fn prefill_oom_preempts_and_requeues_within_budget() {
        let (_sched, mut kv) = reap_fixture();
        let mut requests = HashMap::new();
        let mut state = SequenceState::new(1, vec![1, 2, 3, 4], 64, 99, 16, 0);
        kv.allocate_for_request(&mut state.block_table, 16).unwrap();
        state.block_table.advance(16);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        requests.insert(
            1,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let free_before = kv.num_free_blocks();
        let oom = EngineError::Model(
            "cache write: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")".into(),
        );
        let r = handle_prefill_error(1, oom, &mut requests, &mut kv);
        assert_eq!(r, Ok(1), "OOM within budget must requeue");
        let s = &requests[&1].state;
        assert_eq!(s.status, RequestStatus::Preempted);
        assert_eq!(s.oom_retries, 1);
        assert_eq!(s.num_computed_tokens, 0);
        assert_eq!(kv.num_free_blocks(), free_before + 1, "KV freed for retry");
        assert!(requests.contains_key(&1), "kept for retry, not removed");
    }

    /// A non-OOM prefill error fails the request immediately (no requeue).
    #[test]
    fn prefill_non_oom_error_fails_immediately() {
        let (_sched, mut kv) = reap_fixture();
        let mut requests = HashMap::new();
        let state = SequenceState::new(1, vec![1, 2, 3], 64, 99, 16, 0);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        requests.insert(
            1,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let r = handle_prefill_error(
            1,
            EngineError::Model("shape mismatch in reshape".into()),
            &mut requests,
            &mut kv,
        );
        assert_eq!(r, Err(Some(1)), "non-OOM error must fail (finalize)");
        assert!(!requests.contains_key(&1), "failed request removed");
    }

    /// Once the OOM retry budget is exhausted, a further OOM fails cleanly
    /// instead of requeueing forever.
    #[test]
    fn prefill_oom_fails_after_retry_budget() {
        let (_sched, mut kv) = reap_fixture();
        let mut requests = HashMap::new();
        let mut state = SequenceState::new(1, vec![1, 2, 3], 64, 99, 16, 0);
        state.oom_retries = MAX_PREFILL_OOM_RETRIES;
        let (tx, _rx) = tokio::sync::oneshot::channel();
        requests.insert(
            1,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let oom = EngineError::Model("DriverError(CUDA_ERROR_OUT_OF_MEMORY, ...)".into());
        let r = handle_prefill_error(1, oom, &mut requests, &mut kv);
        assert_eq!(r, Err(Some(1)), "exhausted budget must fail");
        assert!(!requests.contains_key(&1));
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

    // ---- Stream disconnect detection tests ----

    #[test]
    fn send_stream_token_returns_true_for_complete_channel() {
        // Complete (oneshot) requests are not stream requests — always returns true
        let mut requests = HashMap::new();
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut state = make_state(99, 100);
        state.generated_token_ids.push(42);
        requests.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let tok = dummy_tokenizer();
        assert!(send_stream_token(0, &tok, &mut requests));
    }

    #[test]
    fn send_stream_token_returns_true_when_stream_alive() {
        let mut requests = HashMap::new();
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let mut state = make_state(99, 100);
        state.generated_token_ids.push(42);
        requests.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Stream(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let tok = dummy_tokenizer();
        assert!(send_stream_token(0, &tok, &mut requests));
    }

    #[test]
    fn send_stream_token_returns_false_when_receiver_dropped() {
        let mut requests = HashMap::new();
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        // Drop the receiver to simulate client disconnect
        drop(rx);

        let mut state = make_state(99, 100);
        state.generated_token_ids.push(42);
        requests.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Stream(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let tok = dummy_tokenizer();
        // Should detect the dropped receiver
        assert!(!send_stream_token(0, &tok, &mut requests));
    }

    #[test]
    fn send_stream_token_returns_true_for_unknown_request() {
        let mut requests: HashMap<RequestId, ActiveRequest> = HashMap::new();
        let tok = dummy_tokenizer();
        // Request 99 doesn't exist — should return true (not disconnected)
        assert!(send_stream_token(99, &tok, &mut requests));
    }

    #[test]
    fn send_stream_token_returns_true_when_no_new_tokens() {
        let mut requests = HashMap::new();
        let (tx, _rx) = tokio::sync::mpsc::channel(16);
        let state = make_state(99, 100);
        // No generated tokens — nothing to send
        requests.insert(
            0,
            ActiveRequest {
                state,
                response: ResponseChannel::Stream(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,
                beam_state: None,
            },
        );
        let tok = dummy_tokenizer();
        assert!(send_stream_token(0, &tok, &mut requests));
    }

    // ---- Sliding window reclamation tests ----

    fn make_active_request_with_blocks(
        block_ids: &[usize],
        num_tokens: usize,
        block_size: usize,
    ) -> ActiveRequest {
        use crate::engine::types::ResponseChannel;
        use crate::kv_cache::BlockTable;
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut state =
            crate::request::SequenceState::new(1, vec![0; num_tokens], 100, 0, block_size, 0);
        state.block_table = {
            let mut bt = BlockTable::new(block_size);
            bt.append_blocks(block_ids);
            bt.advance(num_tokens);
            bt
        };
        state.seqlen_offset = num_tokens;
        ActiveRequest {
            state,
            response: ResponseChannel::Complete(tx),
            num_streamed_tokens: 0,
            streamed_text_len: 0,
            beam_state: None,
        }
    }

    #[test]
    fn test_reclaim_sliding_window_none_is_noop() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device};

        let config = CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut requests = HashMap::new();
        let req = make_active_request_with_blocks(&[0, 1, 2, 3], 16, 4);
        requests.insert(1, req);

        reclaim_sliding_window_blocks(&[1], None, 4, &mut mgr, &mut requests);
        // No change — sliding_window is None
        assert_eq!(requests[&1].state.block_table.num_null_blocks(), 0);
    }

    #[test]
    fn test_reclaim_sliding_window_reclaims_blocks() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device};

        let config = CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();

        // Allocate 4 blocks (blocks 0-3) for a 16-token sequence
        let mut requests = HashMap::new();
        let mut table = crate::kv_cache::BlockTable::new(4);
        mgr.allocate_for_request(&mut table, 16).unwrap();
        table.advance(16);
        let block_ids: Vec<usize> = table.block_ids().to_vec();
        assert_eq!(block_ids.len(), 4);

        let mut req = make_active_request_with_blocks(&block_ids, 16, 4);
        req.state.block_table = table;
        requests.insert(1, req);

        assert_eq!(mgr.num_free_blocks(), 12); // 16 - 4

        // sliding_window = 8 → 16 - 8 = 8 skipped tokens → 8/4 = 2 blocks to reclaim
        reclaim_sliding_window_blocks(&[1], Some(8), 4, &mut mgr, &mut requests);

        assert_eq!(requests[&1].state.block_table.num_null_blocks(), 2);
        assert_eq!(mgr.num_free_blocks(), 14); // 12 + 2 reclaimed
    }

    #[test]
    fn test_reclaim_sliding_window_within_window_is_noop() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device};

        let config = CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut requests = HashMap::new();
        // 8 tokens, window = 10 → nothing to reclaim
        let req = make_active_request_with_blocks(&[0, 1], 8, 4);
        requests.insert(1, req);

        reclaim_sliding_window_blocks(&[1], Some(10), 4, &mut mgr, &mut requests);
        assert_eq!(requests[&1].state.block_table.num_null_blocks(), 0);
    }

    #[test]
    fn test_reclaim_sliding_window_partial_block() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device};

        let config = CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut requests = HashMap::new();
        // 10 tokens, window = 8 → 2 skipped tokens → 2/4 = 0 blocks (partial, not full)
        let req = make_active_request_with_blocks(&[0, 1, 2], 10, 4);
        requests.insert(1, req);

        reclaim_sliding_window_blocks(&[1], Some(8), 4, &mut mgr, &mut requests);
        assert_eq!(requests[&1].state.block_table.num_null_blocks(), 0);
    }

    // ---- Multi-LoRA grouping tests ----

    /// Mock model that returns different logits based on LoRA adapter name.
    /// Base (no adapter): all logits = 1.0
    /// Adapter "A": logits[0] = 10.0 (rest 1.0)
    /// Adapter "B": logits[1] = 10.0 (rest 1.0)
    struct LoraAwareMockModel {
        vocab_size: usize,
        device: candle_core::Device,
    }

    impl ModelForward for LoraAwareMockModel {
        fn forward(
            &self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            _kv_cache: &mut KVCacheManager,
            _block_table: &crate::kv_cache::BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<Tensor> {
            let batch = input_ids.dim(0)?;
            Tensor::ones(
                (batch, 1, self.vocab_size),
                candle_core::DType::F32,
                &self.device,
            )
        }

        fn device(&self) -> &candle_core::Device {
            &self.device
        }

        fn forward_decode_batch(
            &self,
            input_ids: &Tensor,
            _sequences: &[DecodeSequenceMetadata],
            _kv_cache_mgr: &mut KVCacheManager,
        ) -> candle_core::Result<Tensor> {
            let batch = input_ids.dim(0)?;
            Tensor::ones(
                (batch, 1, self.vocab_size),
                candle_core::DType::F32,
                &self.device,
            )
        }

        fn forward_decode_batch_with_lora(
            &self,
            input_ids: &Tensor,
            _sequences: &[DecodeSequenceMetadata],
            _kv_cache_mgr: &mut KVCacheManager,
            lora_ctx: &LoraContext,
        ) -> candle_core::Result<Tensor> {
            let batch = input_ids.dim(0)?;
            let mut logits = vec![1.0f32; batch * self.vocab_size];

            // Set a marker logit based on adapter name
            match lora_ctx.adapter_name() {
                Some("A") => {
                    for b in 0..batch {
                        logits[b * self.vocab_size] = 10.0; // index 0
                    }
                }
                Some("B") => {
                    for b in 0..batch {
                        logits[b * self.vocab_size + 1] = 10.0; // index 1
                    }
                }
                _ => {} // base: all 1.0
            }

            let t = Tensor::from_vec(logits, (batch, 1, self.vocab_size), &self.device)?;
            Ok(t)
        }
    }

    fn make_decode_metadata(req_id: RequestId) -> DecodeSequenceMetadata {
        DecodeSequenceMetadata {
            request_id: req_id,
            seqlen_offset: 5,
            block_ids: vec![0],
            slot_mapping: vec![5],
        }
    }

    #[test]
    fn test_forward_grouped_single_adapter() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device, IndexOp};

        let model = LoraAwareMockModel {
            vocab_size: 4,
            device: Device::Cpu,
        };
        let config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();

        let tokens = vec![1u32, 2, 3];
        let sequences = vec![
            make_decode_metadata(1),
            make_decode_metadata(2),
            make_decode_metadata(3),
        ];
        let adapters = vec![
            Some("A".to_string()),
            Some("A".to_string()),
            Some("A".to_string()),
        ];

        let logits =
            forward_grouped_by_adapter(&model, &tokens, &sequences, &adapters, &mut mgr, 3)
                .unwrap();

        // All adapter "A" → logits[0] should be 10.0 for each row
        let row0: Vec<f32> = logits.i(0).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(row0[0], 10.0);
        assert_eq!(row0[1], 1.0);
    }

    #[test]
    fn test_forward_grouped_multi_adapter() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device, IndexOp};

        let model = LoraAwareMockModel {
            vocab_size: 4,
            device: Device::Cpu,
        };
        let config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();

        let tokens = vec![1u32, 2, 3, 4];
        let sequences = vec![
            make_decode_metadata(1),
            make_decode_metadata(2),
            make_decode_metadata(3),
            make_decode_metadata(4),
        ];
        // Mixed: 2 adapter "A", 1 adapter "B", 1 base (no adapter)
        let adapters = vec![
            Some("A".to_string()),
            Some("B".to_string()),
            None,
            Some("A".to_string()),
        ];

        let logits =
            forward_grouped_by_adapter(&model, &tokens, &sequences, &adapters, &mut mgr, 4)
                .unwrap();

        // Verify logits are in original batch order with correct adapter markers
        // Row 0: adapter A → logits[0]=10, rest=1
        let row0: Vec<f32> = logits.i(0).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(row0[0], 10.0);
        assert_eq!(row0[1], 1.0);

        // Row 1: adapter B → logits[1]=10, rest=1
        let row1: Vec<f32> = logits.i(1).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(row1[0], 1.0);
        assert_eq!(row1[1], 10.0);

        // Row 2: base → all 1.0
        let row2: Vec<f32> = logits.i(2).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(row2[0], 1.0);
        assert_eq!(row2[1], 1.0);

        // Row 3: adapter A → logits[0]=10
        let row3: Vec<f32> = logits.i(3).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert_eq!(row3[0], 10.0);
        assert_eq!(row3[1], 1.0);
    }

    #[test]
    fn test_forward_grouped_all_base() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device, IndexOp};

        let model = LoraAwareMockModel {
            vocab_size: 4,
            device: Device::Cpu,
        };
        let config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();

        let tokens = vec![1u32, 2];
        let sequences = vec![make_decode_metadata(1), make_decode_metadata(2)];
        let adapters = vec![None, None];

        let logits =
            forward_grouped_by_adapter(&model, &tokens, &sequences, &adapters, &mut mgr, 2)
                .unwrap();

        // All base → all 1.0
        let row0: Vec<f32> = logits.i(0).unwrap().to_vec2::<f32>().unwrap()[0].clone();
        assert!(row0.iter().all(|&v| v == 1.0));
    }

    // ---- ResetPrefixCache command tests ----

    #[test]
    fn handle_reset_prefix_cache_no_cache() {
        use crate::kv_cache::config::CacheConfig;
        use crate::kv_cache::KVCacheDtype;
        use candle_core::{DType, Device};

        let config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let tokenizer = dummy_tokenizer();
        let mut scheduler = Scheduler::new(crate::scheduler::SchedulerConfig {
            max_running_requests: 4,
            max_tokens_per_step: 512,
            enable_chunked_prefill: false,
            scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
            ..crate::scheduler::SchedulerConfig::default()
        });
        let mut requests = HashMap::new();
        let mut next_id = 0;
        let mut paused = false;
        let mut frozen = false;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let shutdown = handle_command(
            EngineCommand::ResetPrefixCache { response_tx: tx },
            &mut next_id,
            &tokenizer,
            &mut scheduler,
            &mut requests,
            16,
            &mut mgr,
            &mut paused,
            &mut frozen,
            None,
        );
        assert!(!shutdown);
        // No prefix cache → 0 blocks evicted
        let result = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(result, 0);
    }

    // ---- skip_prefix_cache tests ----

    #[test]
    fn generation_request_default_skip_prefix_cache_false() {
        let req = GenerationRequest::default();
        assert!(!req.skip_prefix_cache);
    }

    #[test]
    fn context_capacity_reached_boundary() {
        // Unpublished limit (unit-test default) never trips.
        assert!(!context_capacity_reached(100_000, None));
        // Below the limit: keep generating.
        assert!(!context_capacity_reached(1023, Some(1024)));
        // Exactly at capacity MUST trip (off-by-one guard): a sequence that
        // fills the pool can't allocate the next token. Was the wedge bug.
        assert!(context_capacity_reached(1024, Some(1024)));
        // Past the limit also trips.
        assert!(context_capacity_reached(1025, Some(1024)));
    }

    #[test]
    fn prompt_unschedulable_admission_guards() {
        use crate::scheduler::SchedulerConfig;
        let chunked = SchedulerConfig {
            max_tokens_per_step: 2048,
            enable_chunked_prefill: true,
            ..SchedulerConfig::default()
        };
        let unchunked = SchedulerConfig {
            enable_chunked_prefill: false,
            ..chunked
        };
        // Pool: 64 blocks × 16 = 1024 tokens.
        let (bs, total) = (16usize, 64usize);

        // Fits pool, chunked: schedulable even when prompt > per-step budget
        // (1000 < 1024 pool; budget irrelevant with chunking).
        assert!(prompt_unschedulable_reason(1000, bs, total, &chunked).is_none());

        // Pool overflow trips regardless of chunking (+1 headroom: 1024
        // prompt tokens need 1025 slots → 65 blocks > 64).
        let r = prompt_unschedulable_reason(1024, bs, total, &chunked);
        assert!(r.is_some_and(|m| m.contains("KV cache blocks")));

        // Chunking disabled: prompt > max_tokens_per_step can never be
        // scheduled (the 2026-06-10 E4B starvation) → explicit rejection.
        let big_pool = 1024usize; // 16k tokens — pool is not the limiter
        let r = prompt_unschedulable_reason(2900, bs, big_pool, &unchunked);
        assert!(r.is_some_and(|m| m.contains("chunked prefill is disabled")));
        // Same prompt with chunking: fine.
        assert!(prompt_unschedulable_reason(2900, bs, big_pool, &chunked).is_none());
        // Exactly at the budget: schedulable even unchunked.
        assert!(prompt_unschedulable_reason(2048, bs, big_pool, &unchunked).is_none());
    }
}
