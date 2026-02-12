//! Inference engine with pluggable execution strategies.
//!
//! This module provides the core engine for running LLM inference with support
//! for different execution modes through the ExecutionStrategy trait.
//!
//! # Architecture
//!
//! The engine is designed around the Strategy pattern:
//!
//! - `ExecutionStrategy` trait defines the interface for execution modes
//! - `StandardExecution` implements standard autoregressive decoding
//! - `SpeculativeExecution` implements speculative decoding with draft model
//!
//! # Example
//!
//! ```ignore
//! let handle = start_engine(model, tokenizer, kv_cache_mgr, config);
//! let result = handle.generate(request).await?;
//! ```

mod context;
pub mod cuda_graph;
pub mod cuda_graph_runner;
mod embedding_forward;
mod encoder_decoder;
mod handle;
mod helpers;
mod model_forward;
pub mod spec_decode;
mod speculative;
mod standard;
mod strategy;
mod types;
pub mod warmup;

// Re-export public types
pub use cuda_graph::{
    BatchDescriptor, CudaGraphConfig, CudaGraphDispatcher, CudaGraphError, CudaGraphStats,
    ForwardContext, RuntimeMode, WarmupManager, WarmupResult,
};
pub use cuda_graph_runner::{
    cuda_memcpy_inplace, CudaGraphRunner, CudaGraphRunnerBuilder, CudaGraphRunnerError,
    CudaGraphRunnerStats,
};
pub use embedding_forward::{pool_embeddings, EmbeddingOutput, ModelForEmbedding, PoolingStrategy};
pub use encoder_decoder::{EncoderOutput, ModelForEncoderDecoder};
pub use handle::EngineHandle;
pub use model_forward::{DecodeSequenceMetadata, ModelForward};
pub use spec_decode::{
    DraftModelDraftProposer, DraftModelProposer, DraftProposer, EagleConfig, EagleProposer,
    MedusaHead, MedusaProposer, NGramConfig, NGramProposer, SpeculationTree, SpeculativeProposer,
    SuffixArrayConfig, SuffixArrayProposer,
};
pub use types::{
    EngineConfig, EngineError, EngineStats, GenerationParams, GenerationRequest, GenerationResult,
    PauseMode, SpecDecodingStats, SpeculativeConfig, StreamEvent,
};
pub use warmup::{
    DefaultDummyInputGenerator, DummyInputGenerator, DummySequence, RandomDummyInputGenerator,
    WarmupConfig, WarmupError, WarmupStats,
};

// Re-export helpers for legacy API
pub use helpers::greedy_sample;

// Internal re-exports for strategy types
pub(crate) use speculative::SpeculativeExecution;
pub(crate) use standard::StandardExecution;
use strategy::ExecutionStrategy;

use tokio::sync::mpsc;

use crate::kv_cache::KVCacheManager;
use crate::tokenizer::TokenizerWrapper;

use context::OwnedExecutionState;

// ─── Engine start functions ───────────────────────────────────────────────

/// Start the inference engine with standard execution.
pub fn start_engine<M: ModelForward>(
    model: M,
    tokenizer: TokenizerWrapper,
    kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);

    let state = OwnedExecutionState::new(&config);
    let strategy =
        StandardExecution::new(model).with_sliding_window(config.sliding_window, config.block_size);

    tokio::spawn(strategy::run_engine_loop(
        strategy,
        state,
        kv_cache_mgr,
        config,
        tokenizer,
        cmd_rx,
    ));

    EngineHandle { cmd_tx }
}

/// Start the inference engine with speculative decoding using a draft model.
pub fn start_engine_with_draft<M: ModelForward, D: ModelForward>(
    target_model: M,
    draft_model: D,
    tokenizer: TokenizerWrapper,
    target_kv_cache: KVCacheManager,
    draft_kv_cache: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let num_speculative_tokens = config
        .speculative_config
        .as_ref()
        .map(|c| c.num_speculative_tokens)
        .unwrap_or(3);

    let proposer = Box::new(DraftModelDraftProposer::new(
        draft_model,
        draft_kv_cache,
        num_speculative_tokens,
    ));

    start_engine_with_proposer(target_model, proposer, tokenizer, target_kv_cache, config)
}

/// Start the inference engine with speculative decoding using any [`DraftProposer`].
///
/// This is the general entry point. Use `start_engine_with_draft` as a convenience
/// when you have a draft model and its KV cache.
pub fn start_engine_with_proposer<M: ModelForward>(
    target_model: M,
    proposer: Box<dyn DraftProposer>,
    tokenizer: TokenizerWrapper,
    target_kv_cache: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);

    let state = OwnedExecutionState::new(&config);
    let strategy = SpeculativeExecution::new(target_model, proposer);

    tokio::spawn(strategy::run_engine_loop(
        strategy,
        state,
        target_kv_cache,
        config,
        tokenizer,
        cmd_rx,
    ));

    EngineHandle { cmd_tx }
}

// ─── Engine start with warmup ─────────────────────────────────────────────

/// Start the inference engine with warmup phase.
///
/// Warmup is performed synchronously before the engine starts accepting requests.
/// This ensures:
/// 1. All CUDA kernels are JIT-compiled for configured batch sizes
/// 2. CUDA graphs are captured (if enabled) for fast replay
///
/// # Returns
/// A tuple of (EngineHandle, WarmupStats) where WarmupStats contains
/// information about which batch sizes were warmed up and any errors.
///
/// # Example
///
/// ```ignore
/// let config = EngineConfig {
///     cuda_graph_config: CudaGraphConfig::enabled(),
///     ..Default::default()
/// };
///
/// let (handle, stats) = start_engine_with_warmup(model, tokenizer, kv_cache, config);
/// println!("Warmed up {} batch sizes", stats.jit_warmed_sizes.len());
/// ```
pub fn start_engine_with_warmup<M: ModelForward>(
    model: M,
    tokenizer: TokenizerWrapper,
    mut kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
) -> (EngineHandle, WarmupStats) {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);

    let state = OwnedExecutionState::new(&config);
    let mut strategy =
        StandardExecution::new(model).with_sliding_window(config.sliding_window, config.block_size);

    // Synchronous warmup before starting loop
    let warmup_config = WarmupConfig::from(&config.cuda_graph_config);
    let warmup_stats = {
        let mut dispatcher = state
            .cuda_graph_dispatcher
            .write()
            .expect("dispatcher lock poisoned during startup");
        strategy.warmup(&warmup_config, &mut kv_cache_mgr, &mut dispatcher)
    };

    tokio::spawn(strategy::run_engine_loop(
        strategy,
        state,
        kv_cache_mgr,
        config,
        tokenizer,
        cmd_rx,
    ));

    (EngineHandle { cmd_tx }, warmup_stats)
}

/// Start the inference engine with warmup phase (async version).
///
/// This variant doesn't block the calling async task during warmup.
/// Instead, warmup runs in a blocking task and returns when complete.
pub async fn start_engine_with_warmup_async<M: ModelForward>(
    model: M,
    tokenizer: TokenizerWrapper,
    kv_cache_mgr: KVCacheManager,
    config: EngineConfig,
) -> (EngineHandle, WarmupStats) {
    tokio::task::spawn_blocking(move || {
        start_engine_with_warmup(model, tokenizer, kv_cache_mgr, config)
    })
    .await
    .expect("warmup task panicked")
}

// ─── Legacy API ───────────────────────────────────────────────────────────

use crate::kv_cache::BlockTable;
use candle_core::Device;

/// Generate text from a prompt (legacy API).
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

/// Generate tokens from prompt IDs (legacy API).
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

    let input = candle_core::Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;
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

        let input = candle_core::Tensor::new(&[[next_token]], device)?;
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

// ─── Test utilities ───────────────────────────────────────────────────────

#[cfg(any(test, feature = "test-utils"))]
pub mod testing {
    use super::types::EngineCommand;
    use super::*;
    use tokio::sync::oneshot;

    /// Create an EngineHandle from an mpsc sender for testing.
    pub fn engine_handle_from_sender(tx: mpsc::Sender<TestEngineCommand>) -> EngineHandle {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<EngineCommand>(16);

        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    EngineCommand::GetStats { response_tx } => {
                        let (test_tx, test_rx) = oneshot::channel();
                        if tx
                            .send(TestEngineCommand::GetStats {
                                response_tx: test_tx,
                            })
                            .await
                            .is_ok()
                        {
                            if let Ok(stats) = test_rx.await {
                                let _ = response_tx.send(stats);
                            }
                        }
                    }
                    EngineCommand::Pause { response_tx, .. } => {
                        let _ = response_tx.send(Ok(()));
                    }
                    EngineCommand::Resume { response_tx } => {
                        let _ = response_tx.send(Ok(()));
                    }
                    EngineCommand::IsPaused { response_tx } => {
                        let _ = response_tx.send(false);
                    }
                    EngineCommand::Shutdown => {
                        let (test_tx, test_rx) = oneshot::channel();
                        if tx
                            .send(TestEngineCommand::Shutdown {
                                response_tx: test_tx,
                            })
                            .await
                            .is_ok()
                        {
                            let _ = test_rx.await;
                        }
                        break;
                    }
                    _ => {}
                }
            }
        });

        EngineHandle { cmd_tx }
    }

    /// Test-friendly engine command enum.
    pub enum TestEngineCommand {
        GetStats {
            response_tx: oneshot::Sender<EngineStats>,
        },
        Shutdown {
            response_tx: oneshot::Sender<Result<(), EngineError>>,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::request::{FinishReason, RequestStatus, SequenceState};
    use crate::scheduler::SchedulerConfig;
    use candle_core::DType;
    use std::collections::HashMap;
    use tokio::sync::oneshot;

    // Internal imports for test utilities
    use super::context::ActiveRequest;
    use super::helpers::{check_finished, execute_batched_decode, execute_prefill};
    use super::types::ResponseChannel;

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
            input_ids: &candle_core::Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<candle_core::Tensor> {
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            candle_core::Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
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
            input_ids: &candle_core::Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<candle_core::Tensor> {
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
            candle_core::Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &self.device)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    fn test_cache_config() -> CacheConfig {
        use crate::kv_cache::KVCacheDtype;
        CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn test_engine_config() -> EngineConfig {
        EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .build()
    }

    async fn run_engine_with_pretokenized<M: ModelForward>(
        model: M,
        kv_cache_mgr: KVCacheManager,
        config: EngineConfig,
        requests: Vec<(Vec<u32>, usize, u32)>,
    ) -> Vec<Result<GenerationResult, EngineError>> {
        use crate::scheduler::Scheduler;

        let block_size = config.block_size;
        let num_requests = requests.len();

        let initial_free_blocks = kv_cache_mgr.num_free_blocks();
        let handle = tokio::spawn(async move {
            let tokenizer = TokenizerWrapper::for_testing(1000);
            let mut scheduler = Scheduler::new(config.scheduler_config);
            let mut active: HashMap<crate::request::RequestId, ActiveRequest> = HashMap::new();
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

                        beam_state: None,
                    },
                );
            }

            let mut results: HashMap<crate::request::RequestId, GenerationResult> = HashMap::new();

            for _ in 0..10000 {
                if scheduler.is_idle() {
                    break;
                }

                let states: HashMap<crate::request::RequestId, &SequenceState> =
                    active.iter().map(|(&id, r)| (id, &r.state)).collect();
                let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

                for &req_id in &output.preempted_requests {
                    let req = active.get_mut(&req_id).unwrap();
                    let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
                    req.state.status = RequestStatus::Preempted;
                    req.state.generated_token_ids.clear();
                    req.state.num_computed_tokens = 0;
                    req.state.seqlen_offset = 0;
                }

                for schedule in &output.prefill_requests {
                    let _ = execute_prefill(
                        schedule.request_id,
                        schedule.chunk_size,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                        &tokenizer,
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
                let scheduled_ids: Vec<crate::request::RequestId> = output
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
                            stop_reason: check.stop_reason,
                            token_logprobs: None,
                            top_logprobs: None,
                            prompt_token_ids: None,
                            prompt_logprobs: None,
                        },
                    );
                }
            }

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

    fn chunked_engine_config(max_tokens_per_step: usize) -> EngineConfig {
        EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step,
                enable_chunked_prefill: true,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .build()
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

        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().unwrap().generated_token_ids,
            vec![42, 42, 42]
        );
        assert_eq!(
            results[1].as_ref().unwrap().generated_token_ids,
            vec![42, 42, 42, 42]
        );
        assert_eq!(
            results[2].as_ref().unwrap().generated_token_ids,
            vec![42, 42]
        );
    }

    // ─── Chunked Prefill Tests ────────────────────────────────────────────

    #[tokio::test]
    async fn chunked_prefill_long_prompt_split_into_chunks() {
        // 25-token prompt with max_tokens_per_step=10 should be split
        // into chunks of 10, 10, 5 tokens across 3 engine steps.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(10);

        let prompt: Vec<u32> = (1..=25).collect();
        let results =
            run_engine_with_pretokenized(model, kv_cache_mgr, config, vec![(prompt, 3, 999)]).await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_matches_non_chunked_output() {
        // Same prompt and model should produce identical tokens whether
        // chunked prefill is enabled or not.
        let prompt: Vec<u32> = (1..=30).collect();
        let max_new_tokens = 5;
        let eos_token = 999u32;

        // Non-chunked
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();
        let results_no_chunk = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(prompt.clone(), max_new_tokens, eos_token)],
        )
        .await;

        // Chunked (budget=8, so 30 tokens -> chunks of 8,8,8,6)
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(8);
        let results_chunked = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(prompt, max_new_tokens, eos_token)],
        )
        .await;

        let r1 = results_no_chunk[0].as_ref().unwrap();
        let r2 = results_chunked[0].as_ref().unwrap();
        assert_eq!(r1.generated_token_ids, r2.generated_token_ids);
        assert_eq!(r1.finish_reason, r2.finish_reason);
    }

    #[tokio::test]
    async fn chunked_prefill_eos_after_chunked_prompt() {
        // The model should still stop at EOS even when prefill was chunked.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        // CountingMockModel: emits output_token for first N calls, then eos.
        // Call 0 = prefill chunk 1 (10 tokens), call 1 = chunk 2 (10 tokens),
        // call 2 = final chunk (5 tokens, transitions to decode + samples token 42),
        // call 3 = decode step (token 42),
        // call 4 = decode step (eos=999).
        let model = CountingMockModel::new(42, 999, 1000, 4);
        let config = chunked_engine_config(10);

        let prompt: Vec<u32> = (1..=25).collect();
        let results =
            run_engine_with_pretokenized(model, kv_cache_mgr, config, vec![(prompt, 10, 999)])
                .await;

        let result = results[0].as_ref().unwrap();
        // Prefill chunks: 10, 10, 5 (3 forward calls). Then decode produces tokens.
        // call 0,1,2 = prefill (token 42 sampled from call 2 = final chunk)
        // call 3 = decode (token 42)
        // call 4 = decode (eos 999)
        assert_eq!(result.generated_token_ids, vec![42, 42, 999]);
        assert_eq!(result.finish_reason, FinishReason::Eos);
    }

    #[tokio::test]
    async fn chunked_prefill_decode_interleaves_with_prefill() {
        // One short request that enters decode quickly, and a long request
        // that takes multiple prefill chunks. Both should complete correctly.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        // Budget of 10 tokens per step. The short request (3 tokens) enters
        // decode in step 1. The long request (25 tokens) takes multiple steps
        // to prefill. Decode tokens for the short request interleave with
        // prefill chunks for the long request.
        let config = chunked_engine_config(10);

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![
                (vec![1, 2, 3], 3, 999),      // Short: prefilled in 1 chunk
                ((1..=25).collect(), 3, 999), // Long: chunked prefill
            ],
        )
        .await;

        assert_eq!(results.len(), 2);
        let r0 = results[0].as_ref().unwrap();
        let r1 = results[1].as_ref().unwrap();
        assert_eq!(r0.generated_token_ids, vec![42, 42, 42]);
        assert_eq!(r0.finish_reason, FinishReason::Length);
        assert_eq!(r1.generated_token_ids, vec![42, 42, 42]);
        assert_eq!(r1.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_prompt_fits_in_single_chunk() {
        // A short prompt that fits entirely within the token budget
        // should work exactly like a non-chunked prefill.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(100); // budget >> prompt length

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![1, 2, 3, 4, 5], 4, 999)],
        )
        .await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_multiple_long_requests() {
        // Two long requests that both require chunking.
        // With budget=10 and max_running=8, both should be admitted but
        // each will take multiple steps to prefill.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(15);

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![
                ((1..=20).collect(), 2, 999),
                ((100..=120).collect(), 2, 999),
            ],
        )
        .await;

        assert_eq!(results.len(), 2);
        let r0 = results[0].as_ref().unwrap();
        let r1 = results[1].as_ref().unwrap();
        assert_eq!(r0.generated_token_ids, vec![42, 42]);
        assert_eq!(r0.finish_reason, FinishReason::Length);
        assert_eq!(r1.generated_token_ids, vec![42, 42]);
        assert_eq!(r1.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_exact_boundary() {
        // Prompt length is an exact multiple of the budget.
        // 20 tokens with budget=10 -> 2 exact chunks of 10.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(10);

        let prompt: Vec<u32> = (1..=20).collect();
        let results =
            run_engine_with_pretokenized(model, kv_cache_mgr, config, vec![(prompt, 3, 999)]).await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn chunked_prefill_single_token_budget() {
        // Extreme case: budget=1 means each step processes only 1 prefill token.
        // A 5-token prompt takes 5 steps for prefill.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = chunked_engine_config(1);

        let results = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(vec![1, 2, 3, 4, 5], 2, 999)],
        )
        .await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    // ─── Prefix Cache Integration Tests ──────────────────────────────────

    /// Helper: run pretokenized requests with prefix caching enabled.
    /// Handles register/release of prefix blocks on completion.
    async fn run_engine_with_prefix_caching<M: ModelForward>(
        model: M,
        mut kv_cache_mgr: KVCacheManager,
        config: EngineConfig,
        requests: Vec<(Vec<u32>, usize, u32)>,
    ) -> (Vec<Result<GenerationResult, EngineError>>, KVCacheManager) {
        use crate::scheduler::Scheduler;

        let block_size = config.block_size;
        let num_requests = requests.len();

        // Enable prefix cache on the manager (same as run_engine_loop does)
        if config.enable_prefix_caching && !kv_cache_mgr.has_prefix_cache() {
            kv_cache_mgr.enable_prefix_cache();
        }

        let handle = tokio::spawn(async move {
            let tokenizer = TokenizerWrapper::for_testing(1000);
            let mut scheduler = Scheduler::new(config.scheduler_config);
            let mut active: HashMap<crate::request::RequestId, ActiveRequest> = HashMap::new();
            let mut kv_cache_mgr = kv_cache_mgr;

            for (id, (prompt_ids, max_tokens, eos_token)) in (0u64..).zip(requests.iter()) {
                let mut state = SequenceState::new(
                    id,
                    prompt_ids.clone(),
                    *max_tokens,
                    *eos_token,
                    block_size,
                    id,
                );

                // Prefix cache lookup on admission (mirrors admit_request)
                if kv_cache_mgr.has_prefix_cache() {
                    let (cached_blocks, _) = kv_cache_mgr.match_prefix(&state.prompt_token_ids);
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
                let (tx, _rx) = oneshot::channel();
                active.insert(
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

            let mut results: HashMap<crate::request::RequestId, GenerationResult> = HashMap::new();

            for _ in 0..10000 {
                if scheduler.is_idle() {
                    break;
                }

                let states: HashMap<crate::request::RequestId, &SequenceState> =
                    active.iter().map(|(&id, r)| (id, &r.state)).collect();
                let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

                for &req_id in &output.preempted_requests {
                    let req = active.get_mut(&req_id).expect("preempted request missing");
                    let _ = kv_cache_mgr.free_request(&mut req.state.block_table);
                    req.state.status = RequestStatus::Preempted;
                    req.state.generated_token_ids.clear();
                    req.state.num_computed_tokens = 0;
                    req.state.seqlen_offset = 0;
                }

                for schedule in &output.prefill_requests {
                    let _ = execute_prefill(
                        schedule.request_id,
                        schedule.chunk_size,
                        &model,
                        &mut kv_cache_mgr,
                        &mut active,
                        &tokenizer,
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
                let scheduled_ids: Vec<crate::request::RequestId> = output
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
                    let req = active.remove(&req_id).expect("finished request missing");
                    scheduler.remove_request(req_id);
                    let prompt_tokens = req.state.prompt_token_ids.clone();
                    let mut bt = req.state.block_table;
                    let block_ids = bt.release();

                    // Register and release prefix blocks (mirrors run_engine_loop)
                    if kv_cache_mgr.has_prefix_cache() {
                        kv_cache_mgr.register_prefix(&prompt_tokens, &block_ids);
                        let to_free = kv_cache_mgr.release_prefix(&prompt_tokens, &block_ids);
                        if !to_free.is_empty() {
                            let _ = kv_cache_mgr.free_blocks(&to_free);
                        }
                    } else if !block_ids.is_empty() {
                        let _ = kv_cache_mgr.free_blocks(&block_ids);
                    }

                    results.insert(
                        req_id,
                        GenerationResult {
                            request_id: req_id,
                            generated_text: String::new(),
                            generated_token_ids: req.state.generated_token_ids,
                            finish_reason: check.reason,
                            stop_reason: check.stop_reason,
                            token_logprobs: None,
                            top_logprobs: None,
                            prompt_token_ids: None,
                            prompt_logprobs: None,
                        },
                    );
                }
            }

            (results, kv_cache_mgr)
        });

        let (results, kv_cache_mgr) = handle.await.expect("engine task panicked");
        let mut output: Vec<Result<GenerationResult, EngineError>> = Vec::new();
        for i in 0..num_requests {
            if let Some(r) = results.get(&(i as u64)) {
                output.push(Ok(r.clone()));
            } else {
                output.push(Err(EngineError::Shutdown));
            }
        }
        (output, kv_cache_mgr)
    }

    fn prefix_cache_engine_config() -> EngineConfig {
        EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .enable_prefix_caching(true)
        .build()
    }

    #[tokio::test]
    async fn prefix_cache_match_skips_tokens_during_prefill() {
        // Run first request to populate the prefix cache, then run a second
        // request with the same prefix. The second request should reuse
        // cached blocks and skip those tokens during prefill.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = prefix_cache_engine_config();

        // block_size=16, prompt=32 tokens -> 2 full blocks
        let prompt: Vec<u32> = (1..=32).collect();

        // First request: populates the prefix cache
        let (results, kv_cache_mgr) = run_engine_with_prefix_caching(
            model,
            kv_cache_mgr,
            config,
            vec![(prompt.clone(), 2, 999)],
        )
        .await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42]);

        // Verify prefix cache has entries
        let (cached, _evictable) = kv_cache_mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 2); // 2 blocks of 16 tokens each

        // Second request: same prompt, should hit prefix cache
        let model = MockModel::new(42, 1000);
        let config = prefix_cache_engine_config();

        let (results, _kv_cache_mgr) =
            run_engine_with_prefix_caching(model, kv_cache_mgr, config, vec![(prompt, 3, 999)])
                .await;

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42]);
        assert_eq!(result.finish_reason, FinishReason::Length);
    }

    #[tokio::test]
    async fn prefix_cache_registered_prefixes_found_on_subsequent_requests() {
        // Verify that after one request completes, the next request with
        // the same prefix actually finds blocks in the cache.
        use std::sync::Arc;
        let cache_config = test_cache_config();
        let metrics = Arc::new(crate::kv_cache::KVCacheMetrics::new());
        let kv_cache_mgr = KVCacheManager::with_metrics(&cache_config, metrics).unwrap();
        let model = MockModel::new(42, 1000);
        let config = prefix_cache_engine_config();

        // Run first request with a 32-token prompt (2 full blocks of 16)
        let prompt: Vec<u32> = (1..=32).collect();
        let (results, kv_cache_mgr) = run_engine_with_prefix_caching(
            model,
            kv_cache_mgr,
            config,
            vec![(prompt.clone(), 1, 999)],
        )
        .await;
        assert!(results[0].is_ok());

        // Verify blocks are in the cache
        let (cached, evictable) = kv_cache_mgr.prefix_cache_stats().unwrap();
        assert!(cached > 0, "prefix cache should have entries");
        assert_eq!(
            evictable, cached,
            "all blocks should be evictable after owner releases"
        );

        // Directly test match_prefix on the manager
        let mut kv_cache_mgr = kv_cache_mgr;
        let (matched, num_cached) = kv_cache_mgr.match_prefix(&prompt);
        assert_eq!(matched.len(), 2, "should match 2 cached blocks");
        assert_eq!(num_cached, 32, "should cover 32 tokens");
    }

    #[tokio::test]
    async fn prefix_cache_eviction_when_blocks_run_out() {
        // With limited blocks and prefix caching enabled, eviction should
        // reclaim unreferenced cached blocks to make room for new requests.
        use crate::kv_cache::KVCacheDtype;
        let cache_config = CacheConfig {
            block_size: 4,
            num_blocks: 8, // Very limited
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .block_size(4)
        .enable_prefix_caching(true)
        .build();

        // First request: 8 tokens (2 blocks of 4) + 1 generated token (needs 1 more block) = 3 blocks
        let prompt1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (results, kv_cache_mgr) =
            run_engine_with_prefix_caching(model, kv_cache_mgr, config, vec![(prompt1, 1, 999)])
                .await;
        assert!(results[0].is_ok());

        // After completion: 2 blocks cached (prompt), 1 decode block freed
        // Cached: 2 blocks, Free: 8 - 2 = 6 blocks
        let (cached, evictable) = kv_cache_mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 2);
        assert_eq!(evictable, 2);

        // Second request: different prompt, needs 6 blocks for prompt + 1 for decode = 7
        // We have 6 free + 2 evictable = 8 total available. Should succeed via eviction.
        let model = MockModel::new(42, 1000);
        let config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .block_size(4)
        .enable_prefix_caching(true)
        .build();
        let prompt2: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let (results, _kv_cache_mgr) =
            run_engine_with_prefix_caching(model, kv_cache_mgr, config, vec![(prompt2, 1, 999)])
                .await;
        // Should succeed because eviction freed cached blocks
        assert!(results[0].is_ok());
        let result = results[0].as_ref().unwrap();
        assert_eq!(result.generated_token_ids, vec![42]);
    }

    #[tokio::test]
    async fn prefix_cache_partial_prefix_match() {
        // Verify partial prefix matching: first half matches, second half differs.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = prefix_cache_engine_config();

        // First request: 32 tokens in 2 blocks
        let prompt1: Vec<u32> = (1..=32).collect();
        let (results, kv_cache_mgr) =
            run_engine_with_prefix_caching(model, kv_cache_mgr, config, vec![(prompt1, 1, 999)])
                .await;
        assert!(results[0].is_ok());

        // Second request: same first 16 tokens, different second 16
        let mut prompt2: Vec<u32> = (1..=16).collect();
        prompt2.extend(100..=115); // Different second block

        let mut kv_cache_mgr = kv_cache_mgr;
        let (matched, num_cached) = kv_cache_mgr.match_prefix(&prompt2);
        assert_eq!(matched.len(), 1, "only first block should match");
        assert_eq!(num_cached, 16, "first block covers 16 tokens");
    }

    #[tokio::test]
    async fn prefix_cache_generation_produces_correct_tokens() {
        // End-to-end: prefix caching should not affect correctness.
        // Same prompt should produce identical tokens with and without prefix caching.
        let prompt: Vec<u32> = (1..=32).collect();

        // Without prefix caching
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = test_engine_config();
        let results_no_cache = run_engine_with_pretokenized(
            model,
            kv_cache_mgr,
            config,
            vec![(prompt.clone(), 5, 999)],
        )
        .await;

        // With prefix caching
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let config = prefix_cache_engine_config();
        let (results_with_cache, _) =
            run_engine_with_prefix_caching(model, kv_cache_mgr, config, vec![(prompt, 5, 999)])
                .await;

        let r1 = results_no_cache[0].as_ref().unwrap();
        let r2 = results_with_cache[0].as_ref().unwrap();
        assert_eq!(r1.generated_token_ids, r2.generated_token_ids);
        assert_eq!(r1.finish_reason, r2.finish_reason);
    }

    #[tokio::test]
    async fn prefix_cache_enable_on_kv_cache_manager() {
        // Verify that enable_prefix_cache() works and the manager reports
        // correct state before and after enabling.
        let cache_config = test_cache_config();
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();

        assert!(!kv_cache_mgr.has_prefix_cache());
        assert!(kv_cache_mgr.prefix_cache_stats().is_none());

        kv_cache_mgr.enable_prefix_cache();

        assert!(kv_cache_mgr.has_prefix_cache());
        let (cached, evictable) = kv_cache_mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 0);
        assert_eq!(evictable, 0);

        // Double-enable is a no-op
        kv_cache_mgr.enable_prefix_cache();
        assert!(kv_cache_mgr.has_prefix_cache());
    }

    #[tokio::test]
    async fn prefix_cache_stats_available_via_manager() {
        // Verify that prefix cache statistics are accessible through the
        // KVCacheManager after enabling prefix caching.
        let cache_config = test_cache_config();
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        kv_cache_mgr.enable_prefix_cache();

        // Register a prefix manually
        let prompt: Vec<u32> = (1..=16).collect();
        let mut block_table = BlockTable::new(cache_config.block_size);
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 16)
            .unwrap();
        block_table.advance(16);
        let block_ids = block_table.block_ids().to_vec();
        kv_cache_mgr.register_prefix(&prompt, &block_ids);

        // Check stats
        let (cached, _) = kv_cache_mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 1);

        // Check detailed stats
        let cache = kv_cache_mgr.prefix_cache().unwrap();
        assert_eq!(cache.num_cached_blocks(), 1);
    }

    // ─── Beam Search Tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn beam_search_produces_tokens() {
        // Beam search with width=2 should produce the same tokens as greedy
        // when the model always outputs the same token with highest logit.
        let model = MockModel::new(42, 100);
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let config = test_engine_config();

        let prompt = vec![1u32, 2, 3];
        let max_tokens = 5;
        let eos_token = 99;
        let block_size = config.block_size;

        let tokenizer = crate::tokenizer::TokenizerWrapper::for_testing(100);
        let mut scheduler = crate::scheduler::Scheduler::new(config.scheduler_config);
        let mut active: HashMap<crate::request::RequestId, ActiveRequest> = HashMap::new();
        let mut kv_cache_mgr = kv_cache_mgr;

        let beam_config = crate::sampling::BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };

        let id = 0u64;
        let mut state =
            SequenceState::new(id, prompt.clone(), max_tokens, eos_token, block_size, id);
        state.sampling_params.beam_search = Some(beam_config);

        scheduler.add_request(id);
        let (tx, rx) = oneshot::channel();
        active.insert(
            id,
            ActiveRequest {
                state,
                response: ResponseChannel::Complete(tx),
                num_streamed_tokens: 0,
                streamed_text_len: 0,

                beam_state: None, // Beam state is initialized during handle_command/prefill
            },
        );

        // Re-initialize beam_state on the request (simulating what handle_command does)
        {
            let req = active.get_mut(&id).unwrap();
            let config = req
                .state
                .sampling_params
                .beam_search
                .as_ref()
                .unwrap()
                .clone();
            req.beam_state = Some(super::context::BeamState {
                search: crate::sampling::BeamSearchState::new(config, eos_token),
                beam_block_tables: Vec::new(),
                beam_seqlen_offsets: Vec::new(),
            });
        }

        // Run engine loop manually
        for _ in 0..100 {
            if scheduler.is_idle() {
                break;
            }

            let states: HashMap<crate::request::RequestId, &SequenceState> =
                active.iter().map(|(&id, r)| (id, &r.state)).collect();
            let output = scheduler.schedule(&states, kv_cache_mgr.num_free_blocks());

            for schedule in &output.prefill_requests {
                let _ = execute_prefill(
                    schedule.request_id,
                    schedule.chunk_size,
                    &model,
                    &mut kv_cache_mgr,
                    &mut active,
                    &tokenizer,
                );
            }

            // Execute beam decode for beam requests
            for &req_id in &output.decode_requests {
                if let Some(req) = active.get(&req_id) {
                    if req.beam_state.is_some() {
                        let _ = super::helpers::execute_beam_decode(
                            req_id,
                            &model,
                            &mut kv_cache_mgr,
                            &mut active,
                            &tokenizer,
                        );
                    }
                }
            }

            // Check completion
            let mut finished = Vec::new();
            let scheduled_ids: Vec<crate::request::RequestId> = output
                .prefill_requests
                .iter()
                .map(|s| s.request_id)
                .chain(output.decode_requests.iter().copied())
                .collect();
            for &req_id in &scheduled_ids {
                if let Some(req) = active.get(&req_id) {
                    let check = if req.beam_state.is_some() {
                        super::helpers::check_beam_finished(req)
                    } else {
                        check_finished(&req.state, &tokenizer)
                    };
                    if let Some(check) = check {
                        finished.push((req_id, check));
                    }
                }
            }

            for (req_id, check) in finished {
                let mut req = active.remove(&req_id).unwrap();
                scheduler.remove_request(req_id);

                if req.beam_state.is_some() {
                    let (text, reason) = super::helpers::finalize_beam_request(
                        &mut req,
                        &mut kv_cache_mgr,
                        &tokenizer,
                    );
                    let _ = text;
                    let _ = reason;
                }

                let mut bt = req.state.block_table;
                let _ = kv_cache_mgr.free_request(&mut bt);

                let result = GenerationResult {
                    request_id: req_id,
                    generated_text: String::new(),
                    generated_token_ids: req.state.generated_token_ids,
                    finish_reason: check.reason,
                    stop_reason: check.stop_reason,
                    token_logprobs: None,
                    top_logprobs: None,
                    prompt_token_ids: None,
                    prompt_logprobs: None,
                };
                let _ = match req.response {
                    ResponseChannel::Complete(tx) => tx.send(Ok(result)),
                    _ => Ok(()),
                };
            }
        }

        let result = rx.await.unwrap().unwrap();
        assert!(
            !result.generated_token_ids.is_empty(),
            "beam search should produce tokens"
        );
        assert!(
            result.generated_token_ids.len() <= max_tokens,
            "should not exceed max_tokens"
        );
    }

    #[test]
    fn beam_search_width_1_equivalent_to_greedy() {
        // With beam_width=1, beam search degenerates to greedy search.
        // The BeamSearchState should produce the same result.
        let config = crate::sampling::BeamSearchConfig {
            beam_width: 1,
            length_penalty: 0.0,
            ..Default::default()
        };
        let eos_token = 99u32;
        let mut beam_state = crate::sampling::BeamSearchState::new(config, eos_token);

        // Simulate a vocab of 5 tokens, token 3 has highest prob
        let log_probs = vec![vec![-5.0, -3.0, -10.0, -0.5, -4.0]];
        let transitions = beam_state.step(&log_probs);

        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].1, 3); // Should pick token 3 (highest log prob)
    }

    #[test]
    fn beam_search_eos_completes_beams() {
        let config = crate::sampling::BeamSearchConfig {
            beam_width: 2,
            early_stopping: true,
            num_return_beams: 1,
            length_penalty: 0.0,
            ..Default::default()
        };
        let eos_token = 2u32;
        let mut beam_state = crate::sampling::BeamSearchState::new(config, eos_token);

        // First step: tokens 0 and 1 are best (not EOS)
        let log_probs1 = vec![vec![-0.5, -1.0, -10.0, -3.0], vec![-1.0, -0.5, -10.0, -3.0]];
        beam_state.step(&log_probs1);
        assert!(beam_state.completed.is_empty());

        // Second step: make EOS (token 2) the best for both beams
        let log_probs2 = vec![
            vec![-10.0, -10.0, -0.1, -10.0],
            vec![-10.0, -10.0, -0.2, -10.0],
        ];
        beam_state.step(&log_probs2);

        assert!(
            !beam_state.completed.is_empty(),
            "at least one beam should complete via EOS"
        );
    }

    #[test]
    fn check_beam_finished_max_tokens() {
        let block_size = 16;
        let eos_token = 99u32;
        let max_tokens = 3;

        let state = SequenceState::new(0, vec![1, 2, 3], max_tokens, eos_token, block_size, 0);

        let beam_config = crate::sampling::BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };

        let mut beam_state = super::context::BeamState {
            search: crate::sampling::BeamSearchState::new(beam_config, eos_token),
            beam_block_tables: Vec::new(),
            beam_seqlen_offsets: Vec::new(),
        };

        // Simulate 3 tokens generated (reaching max_tokens)
        beam_state.search.beams[0].token_ids = vec![1, 2, 3];
        beam_state.search.beams[1].token_ids = vec![4, 5, 6];

        let (tx, _rx) = oneshot::channel();
        let req = ActiveRequest {
            state,
            response: ResponseChannel::Complete(tx),
            num_streamed_tokens: 0,
            streamed_text_len: 0,

            beam_state: Some(beam_state),
        };

        let check = super::helpers::check_beam_finished(&req);
        assert!(check.is_some(), "should detect max_tokens reached");
        assert_eq!(check.unwrap().reason, FinishReason::Length);
    }

    #[test]
    fn check_beam_finished_not_done() {
        let block_size = 16;
        let eos_token = 99u32;
        let max_tokens = 10;

        let state = SequenceState::new(0, vec![1, 2, 3], max_tokens, eos_token, block_size, 0);

        let beam_config = crate::sampling::BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };

        let beam_state = super::context::BeamState {
            search: crate::sampling::BeamSearchState::new(beam_config, eos_token),
            beam_block_tables: Vec::new(),
            beam_seqlen_offsets: Vec::new(),
        };

        let (tx, _rx) = oneshot::channel();
        let req = ActiveRequest {
            state,
            response: ResponseChannel::Complete(tx),
            num_streamed_tokens: 0,
            streamed_text_len: 0,

            beam_state: Some(beam_state),
        };

        let check = super::helpers::check_beam_finished(&req);
        assert!(check.is_none(), "should not be done yet");
    }

    #[test]
    fn is_beam_request_detects_beam_state() {
        let block_size = 16;
        let (tx, _rx) = oneshot::channel();

        let non_beam_req = ActiveRequest {
            state: SequenceState::new(0, vec![1, 2], 10, 99, block_size, 0),
            response: ResponseChannel::Complete(tx),
            num_streamed_tokens: 0,
            streamed_text_len: 0,

            beam_state: None,
        };
        assert!(!super::helpers::is_beam_request(&non_beam_req));

        let beam_config = crate::sampling::BeamSearchConfig {
            beam_width: 2,
            ..Default::default()
        };
        let beam_state = super::context::BeamState {
            search: crate::sampling::BeamSearchState::new(beam_config, 99),
            beam_block_tables: Vec::new(),
            beam_seqlen_offsets: Vec::new(),
        };

        let (tx2, _rx2) = oneshot::channel();
        let beam_req = ActiveRequest {
            state: SequenceState::new(1, vec![1, 2], 10, 99, block_size, 1),
            response: ResponseChannel::Complete(tx2),
            num_streamed_tokens: 0,
            streamed_text_len: 0,

            beam_state: Some(beam_state),
        };
        assert!(super::helpers::is_beam_request(&beam_req));
    }

    // ─── Pause / Resume Tests ─────────────────────────────────────────

    #[tokio::test]
    async fn pause_keep_mode_freezes_scheduling() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Engine should not be paused initially
        assert!(!handle.is_paused().await.unwrap());

        // Pause with Keep mode
        handle.pause(PauseMode::Keep).await.unwrap();
        assert!(handle.is_paused().await.unwrap());

        // Stats should report paused
        let stats = handle.get_stats().await.unwrap();
        assert!(stats.is_paused);

        // Resume
        handle.resume().await.unwrap();
        assert!(!handle.is_paused().await.unwrap());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pause_rejects_new_requests() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        handle.pause(PauseMode::Keep).await.unwrap();

        // New generate request should be rejected
        let request = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 3,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(request).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), EngineError::Paused));

        // Resume and try again — should succeed
        handle.resume().await.unwrap();
        let request = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 3,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(request).await;
        assert!(result.is_ok());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pause_abort_clears_all_requests() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Submit a streaming request that will keep generating
        let request = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 1000, // Long generation
            eos_token_id: 999,
            ..Default::default()
        };
        let mut stream_rx = handle.generate_stream(request).await.unwrap();

        // Give it a moment to start processing
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Pause with Abort mode
        handle.pause(PauseMode::Abort).await.unwrap();
        assert!(handle.is_paused().await.unwrap());

        // The stream should eventually receive an error event
        let mut got_error = false;
        while let Some(event) = stream_rx.recv().await {
            if matches!(event, StreamEvent::Error { .. }) {
                got_error = true;
                break;
            }
            // Token events from before the abort are OK
            if matches!(event, StreamEvent::Done { .. }) {
                // Request completed before abort — also acceptable
                break;
            }
        }
        // Either got an error (abort succeeded) or request completed naturally
        assert!(got_error || stream_rx.recv().await.is_none());

        // After abort, no requests should be active
        let stats = handle.get_stats().await.unwrap();
        assert_eq!(stats.num_running_requests, 0);
        assert_eq!(stats.num_waiting_requests, 0);

        handle.resume().await.unwrap();
        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pause_wait_lets_existing_requests_finish() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Submit a short request
        let request = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 3,
            eos_token_id: 999,
            ..Default::default()
        };
        let gen_handle = handle.clone();
        let gen_task = tokio::spawn(async move { gen_handle.generate(request).await });

        // Give it a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Pause with Wait mode — existing request should still complete
        handle.pause(PauseMode::Wait).await.unwrap();

        // Wait for the generation to complete
        let result = gen_task.await.unwrap();
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42, 42]);

        handle.resume().await.unwrap();
        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pause_idempotent() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Double pause should be OK
        handle.pause(PauseMode::Keep).await.unwrap();
        handle.pause(PauseMode::Keep).await.unwrap();
        assert!(handle.is_paused().await.unwrap());

        // Double resume should be OK
        handle.resume().await.unwrap();
        handle.resume().await.unwrap();
        assert!(!handle.is_paused().await.unwrap());

        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn pause_keep_then_resume_processes_requests() {
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = test_engine_config();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Pause, then resume and submit a request — should work
        handle.pause(PauseMode::Keep).await.unwrap();
        handle.resume().await.unwrap();

        let request = GenerationRequest {
            prompt: "t1 t2 t3".to_string(),
            max_new_tokens: 2,
            eos_token_id: 999,
            ..Default::default()
        };
        let result = handle.generate(request).await.unwrap();
        assert_eq!(result.generated_token_ids, vec![42, 42]);

        handle.shutdown().await.unwrap();
    }

    // ─── Optimistic Scheduling Tests ─────────────────────────────────

    #[tokio::test]
    async fn optimistic_scheduling_produces_same_output_as_disabled() {
        // Run identical requests through two real engine loops: one with
        // optimistic scheduling enabled, one disabled. Outputs must match.
        let prompts = vec![
            ("t1 t2 t3", 5),
            ("t1 t2", 3),
            ("t1 t2 t3 t4 t5", 4),
        ];

        let mut results_on = Vec::new();
        let mut results_off = Vec::new();

        for enable in [true, false] {
            let cache_config = test_cache_config();
            let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
            let model = MockModel::new(42, 1000);
            let tokenizer = TokenizerWrapper::for_testing(1000);
            let config = EngineConfig::builder(
                SchedulerConfig {
                    max_running_requests: 4,
                    max_tokens_per_step: 512,
                    enable_chunked_prefill: false,
                    scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
                },
                None,
            )
            .enable_optimistic_scheduling(enable)
            .build();

            let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

            let mut handles = Vec::new();
            for &(prompt, max_tokens) in &prompts {
                let req = GenerationRequest {
                    prompt: prompt.to_string(),
                    max_new_tokens: max_tokens,
                    eos_token_id: 999,
                    ..Default::default()
                };
                let h = handle.clone();
                handles.push(tokio::spawn(async move { h.generate(req).await }));
            }

            let mut batch_results = Vec::new();
            for jh in handles {
                batch_results.push(jh.await.unwrap().unwrap());
            }

            handle.shutdown().await.unwrap();

            if enable {
                results_on = batch_results;
            } else {
                results_off = batch_results;
            }
        }

        // Sort by request_id for deterministic comparison
        results_on.sort_by_key(|r| r.request_id);
        results_off.sort_by_key(|r| r.request_id);

        assert_eq!(results_on.len(), results_off.len());
        for (on, off) in results_on.iter().zip(results_off.iter()) {
            assert_eq!(on.generated_token_ids, off.generated_token_ids);
            assert_eq!(on.finish_reason, off.finish_reason);
        }
    }

    #[tokio::test]
    async fn optimistic_scheduling_short_and_long_requests_concurrent() {
        // Short requests complete during execution of longer ones, invalidating
        // pre-schedules. The engine must handle this gracefully.
        let cache_config = test_cache_config();
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel::new(42, 1000);
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            None,
        )
        .enable_optimistic_scheduling(true)
        .build();

        let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

        // Mix of short (1-2 tokens) and long (10-20 tokens) requests.
        // Short ones will complete quickly, invalidating pre-schedules.
        let requests: Vec<(&str, usize)> = vec![
            ("t1 t2 t3", 1),       // short
            ("t1 t2 t3 t4 t5", 15), // long
            ("t1 t2", 2),          // short
            ("t1 t2 t3", 20),      // long
            ("t1 t2 t3 t4", 1),    // short
            ("t1 t2 t3", 10),      // long
        ];

        let mut join_handles = Vec::new();
        for (prompt, max_tokens) in &requests {
            let req = GenerationRequest {
                prompt: prompt.to_string(),
                max_new_tokens: *max_tokens,
                eos_token_id: 999,
                ..Default::default()
            };
            let h = handle.clone();
            join_handles.push(tokio::spawn(async move { h.generate(req).await }));
        }

        for (i, jh) in join_handles.into_iter().enumerate() {
            let result = jh.await.unwrap().unwrap();
            let expected_len = requests[i].1;
            assert_eq!(
                result.generated_token_ids.len(),
                expected_len,
                "request {i} should generate {expected_len} tokens"
            );
            assert!(
                result.generated_token_ids.iter().all(|&t| t == 42),
                "all tokens should be 42 from MockModel"
            );
        }

        handle.shutdown().await.unwrap();
    }

    // ─── Scheduling Overlap Benchmark ────────────────────────────────

    /// Mock model that performs real GPU matmuls to simulate forward pass cost.
    /// Each forward call runs `num_matmuls` random matrix multiplications of
    /// shape `[matmul_dim, matmul_dim]` on the GPU device, producing realistic
    /// compute load for benchmarking async scheduling overlap.
    struct GpuMockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
        /// Number of matmul ops per forward call (controls GPU time).
        num_matmuls: usize,
        /// Size of each square matmul (NxN).
        matmul_dim: usize,
    }

    impl GpuMockModel {
        fn new(
            output_token: u32,
            vocab_size: usize,
            device: Device,
            num_matmuls: usize,
            matmul_dim: usize,
        ) -> Self {
            Self {
                output_token,
                vocab_size,
                device,
                num_matmuls,
                matmul_dim,
            }
        }
    }

    impl ModelForward for GpuMockModel {
        fn forward(
            &self,
            input_ids: &candle_core::Tensor,
            _seqlen_offset: usize,
            _kv_cache_mgr: &mut KVCacheManager,
            _block_table: &BlockTable,
            _slot_mapping: &[usize],
        ) -> candle_core::Result<candle_core::Tensor> {
            // Burn GPU cycles with matmuls
            let n = self.matmul_dim;
            let mut acc =
                candle_core::Tensor::randn(0f32, 1.0, (n, n), &self.device)?;
            for _ in 0..self.num_matmuls {
                let b = candle_core::Tensor::randn(0f32, 1.0, (n, n), &self.device)?;
                acc = acc.matmul(&b)?;
            }
            // Force sync so GPU work completes before returning
            let _sync = acc.to_vec2::<f32>()?;

            // Build logits on CPU (like real decode output)
            let seq_len = input_ids.dims()[1];
            let mut logits_vec = vec![-100.0f32; seq_len * self.vocab_size];
            for pos in 0..seq_len {
                logits_vec[pos * self.vocab_size + self.output_token as usize] = 100.0;
            }
            candle_core::Tensor::from_vec(logits_vec, (1, seq_len, self.vocab_size), &Device::Cpu)
        }

        fn device(&self) -> &Device {
            &self.device
        }
    }

    #[tokio::test]
    /// Benchmark: optimistic scheduling overlap with GPU matmuls.
    ///
    /// Run with CUDA:
    /// ```sh
    /// cargo test -p vllm-core --lib --features cuda bench_scheduling_overlap -- --ignored --nocapture
    /// ```
    ///
    /// Baseline results (RTX 4060 Laptop, 8GB, debug build, 2026-02-12):
    ///   8 requests × 20 tokens, 4 matmuls 512×512 per forward
    ///   optimistic=ON:  114ms  (1400 tok/s)
    ///   optimistic=OFF: 123ms  (1303 tok/s)
    ///   Speedup: 7.0%
    #[ignore]
    async fn bench_scheduling_overlap() {
        use std::time::Instant;

        let gpu_device = Device::cuda_if_available(0).expect("GPU required for this benchmark");
        let is_gpu = gpu_device.is_cuda();
        eprintln!(
            "Device: {} ({})",
            if is_gpu { "CUDA" } else { "CPU" },
            if is_gpu { "GPU matmuls" } else { "CPU fallback" }
        );

        let num_requests = 8;
        let tokens_per_request = 20;
        // Tuned for ~5-15ms per forward on RTX 4060: 4 matmuls of 512x512
        let num_matmuls = 4;
        let matmul_dim = 512;

        let mut timings = Vec::new();

        for (label, enable) in [("optimistic=ON", true), ("optimistic=OFF", false)] {
            let cache_config = test_cache_config();
            let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
            let model = GpuMockModel::new(
                42,
                1000,
                gpu_device.clone(),
                num_matmuls,
                matmul_dim,
            );
            let tokenizer = TokenizerWrapper::for_testing(1000);
            let config = EngineConfig::builder(
                SchedulerConfig {
                    max_running_requests: num_requests,
                    max_tokens_per_step: 512,
                    enable_chunked_prefill: false,
                    scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
                },
                None,
            )
            .enable_optimistic_scheduling(enable)
            .build();

            let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

            // Warmup: 2 requests to JIT-compile CUDA kernels
            for _ in 0..2 {
                let req = GenerationRequest {
                    prompt: "warmup".to_string(),
                    max_new_tokens: 3,
                    eos_token_id: 999,
                    ..Default::default()
                };
                let _ = handle.generate(req).await;
            }

            let start = Instant::now();

            let mut join_handles = Vec::new();
            for i in 0..num_requests {
                let prompt = format!("t{i}");
                let req = GenerationRequest {
                    prompt,
                    max_new_tokens: tokens_per_request,
                    eos_token_id: 999,
                    ..Default::default()
                };
                let h = handle.clone();
                join_handles.push(tokio::spawn(async move { h.generate(req).await }));
            }

            for jh in join_handles {
                let result = jh.await.unwrap().unwrap();
                assert_eq!(result.generated_token_ids.len(), tokens_per_request);
            }

            let elapsed = start.elapsed();
            handle.shutdown().await.unwrap();

            let total_tokens = num_requests * tokens_per_request;
            eprintln!(
                "{label}: {elapsed:?} ({total_tokens} tokens, {:.0} tok/s)",
                total_tokens as f64 / elapsed.as_secs_f64()
            );
            timings.push((label, elapsed));
        }

        let on = timings[0].1;
        let off = timings[1].1;
        let speedup = (1.0 - on.as_secs_f64() / off.as_secs_f64()) * 100.0;
        eprintln!("Speedup: {speedup:.1}% ({:?} → {:?})", off, on);
    }
}
