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
mod handle;
mod helpers;
mod model_forward;
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
    CudaGraphRunner, CudaGraphRunnerBuilder, CudaGraphRunnerError, CudaGraphRunnerStats,
};
pub use handle::EngineHandle;
pub use model_forward::{DecodeSequenceMetadata, ModelForward};
pub use types::{
    EngineConfig, EngineError, EngineStats, GenerationParams, GenerationRequest, GenerationResult,
    SpeculativeConfig, StreamEvent,
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
    let strategy = StandardExecution::new(model);

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

/// Start the inference engine with speculative decoding.
pub fn start_engine_with_draft<M: ModelForward, D: ModelForward>(
    target_model: M,
    draft_model: D,
    tokenizer: TokenizerWrapper,
    target_kv_cache: KVCacheManager,
    draft_kv_cache: KVCacheManager,
    config: EngineConfig,
) -> EngineHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(256);

    let num_speculative_tokens = config
        .speculative_config
        .as_ref()
        .map(|c| c.num_speculative_tokens)
        .unwrap_or(3);

    let state = OwnedExecutionState::new(&config);
    let strategy = SpeculativeExecution::new(
        target_model,
        draft_model,
        draft_kv_cache,
        num_speculative_tokens,
    );

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
    let mut strategy = StandardExecution::new(model);

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
        }
    }

    fn test_engine_config() -> EngineConfig {
        EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count: 1,
            enable_prefix_caching: false,
            cuda_graph_config: cuda_graph::CudaGraphConfig::default(),
        }
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
                        draft_state: None,
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
}
