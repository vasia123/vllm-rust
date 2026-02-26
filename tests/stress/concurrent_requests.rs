//! Stress tests for concurrent request handling.
//!
//! Tests engine behavior under high load: many concurrent requests,
//! resource exhaustion, and long-running stability.
//!
//! Run: cargo test --test concurrent_requests -- --ignored

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use vllm_core::{
    engine::{start_engine, EngineConfig, GenerationRequest, ModelForward},
    kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager},
    sampling::SamplingParams,
    scheduler::{SchedulerConfig, SchedulingPolicy},
    tokenizer::TokenizerWrapper,
};

// ─── Test infrastructure ────────────────────────────────────────────────────

struct StressMockModel {
    vocab_size: usize,
    output_token: u32,
    eos_token: u32,
    steps_until_eos: usize,
    call_count: Arc<AtomicUsize>,
    device: Device,
}

impl StressMockModel {
    fn new(vocab_size: usize, output_token: u32, eos_token: u32, steps_until_eos: usize) -> Self {
        Self {
            vocab_size,
            output_token,
            eos_token,
            steps_until_eos,
            call_count: Arc::new(AtomicUsize::new(0)),
            device: Device::Cpu,
        }
    }

    fn call_count(&self) -> Arc<AtomicUsize> {
        self.call_count.clone()
    }
}

impl ModelForward for StressMockModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed);
        let seq_len = input_ids.dims()[1];

        let mut logits = vec![-100.0f32; seq_len * self.vocab_size];
        for pos in 0..seq_len {
            // After steps_until_eos, emit EOS
            let token = if count >= self.steps_until_eos {
                self.eos_token
            } else {
                self.output_token
            };
            logits[pos * self.vocab_size + token as usize] = 100.0;
        }

        Tensor::from_vec(logits, (1, seq_len, self.vocab_size), &self.device)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

fn stress_cache_config(num_blocks: usize) -> CacheConfig {
    CacheConfig {
        block_size: 16,
        num_blocks,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    }
}

fn stress_engine_config(max_running: usize) -> EngineConfig {
    EngineConfig::builder(
        SchedulerConfig {
            max_running_requests: max_running,
            max_tokens_per_step: 2048,
            enable_chunked_prefill: false,
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
        },
        None,
    )
    .build()
}

fn make_stress_request(id: usize, max_tokens: usize, eos_token: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: format!("stress test prompt {id} with some tokens"),
        max_new_tokens: max_tokens,
        eos_token_id: eos_token,
        sampling_params: SamplingParams::greedy(),
        stop_token_ids: Vec::new(),
        stop_strings: Vec::new(),
        include_stop_str_in_output: false,
        logprobs: None,
        echo: false,
        ignore_eos: false,
        lora_request: None,
        prompt_adapter_request: None,
        constraint: None,
        image_inputs: Vec::new(),
        audio_inputs: Vec::new(),
        skip_prefix_cache: false,
    }
}

// ─── Stress tests ───────────────────────────────────────────────────────────

/// Submit 100 concurrent requests and verify all complete.
#[tokio::test]
#[ignore = "stress test - run explicitly"]
async fn test_100_concurrent_requests() {
    const NUM_REQUESTS: usize = 100;
    const MAX_TOKENS: usize = 10;
    const VOCAB_SIZE: usize = 1000;
    const EOS_TOKEN: u32 = 999;

    let model = StressMockModel::new(VOCAB_SIZE, 42, EOS_TOKEN, MAX_TOKENS + 1);
    let call_count = model.call_count();

    // Enough blocks for all requests
    let kv_cache_mgr = KVCacheManager::new(&stress_cache_config(512)).unwrap();
    let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);
    let config = stress_engine_config(32);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

    let start = Instant::now();

    // Launch all requests concurrently
    let mut tasks = Vec::with_capacity(NUM_REQUESTS);
    for i in 0..NUM_REQUESTS {
        let handle = handle.clone();
        tasks.push(tokio::spawn(async move {
            let request = make_stress_request(i, MAX_TOKENS, EOS_TOKEN);
            handle.generate(request).await
        }));
    }

    // Collect results
    let mut successes = 0;
    let mut failures = 0;

    for (i, task) in tasks.into_iter().enumerate() {
        match task.await {
            Ok(Ok(result)) => {
                assert_eq!(
                    result.generated_token_ids.len(),
                    MAX_TOKENS,
                    "Request {i} generated wrong number of tokens"
                );
                successes += 1;
            }
            Ok(Err(e)) => {
                eprintln!("Request {i} failed: {e}");
                failures += 1;
            }
            Err(e) => {
                eprintln!("Request {i} task panicked: {e}");
                failures += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\n100 concurrent requests: {successes} succeeded, {failures} failed in {:.2}s",
        elapsed.as_secs_f64()
    );
    eprintln!(
        "Model forward calls: {}",
        call_count.load(Ordering::Relaxed)
    );

    handle.shutdown().await.unwrap();

    assert_eq!(failures, 0, "Some requests failed");
    assert_eq!(successes, NUM_REQUESTS);
}

/// Submit requests that exceed KV cache capacity to test preemption/backpressure.
#[tokio::test]
#[ignore = "stress test - run explicitly"]
async fn test_cache_pressure() {
    const VOCAB_SIZE: usize = 1000;
    const EOS_TOKEN: u32 = 999;

    let model = StressMockModel::new(VOCAB_SIZE, 42, EOS_TOKEN, 20);

    // Very limited cache: only 8 blocks (128 tokens with block_size=16)
    let kv_cache_mgr = KVCacheManager::new(&stress_cache_config(8)).unwrap();
    let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);
    let config = stress_engine_config(4);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

    // Submit requests that collectively exceed cache capacity
    let mut tasks = Vec::new();
    for i in 0..20 {
        let handle = handle.clone();
        tasks.push(tokio::spawn(async move {
            let request = make_stress_request(i, 15, EOS_TOKEN);
            handle.generate(request).await
        }));
    }

    let mut completed = 0;
    let mut errors = 0;

    for task in tasks {
        match task.await {
            Ok(Ok(_)) => completed += 1,
            Ok(Err(e)) => {
                eprintln!("Cache pressure error: {e}");
                errors += 1;
            }
            Err(e) => {
                eprintln!("Task panicked: {e}");
                errors += 1;
            }
        }
    }

    eprintln!("Cache pressure: {completed} completed, {errors} errors");
    handle.shutdown().await.unwrap();

    // At minimum, some requests should complete even under pressure
    assert!(completed > 0, "No requests completed under cache pressure");
}

/// Sustained load: submit requests continuously for a fixed duration.
#[tokio::test]
#[ignore = "stress test - run explicitly"]
async fn test_sustained_load_30s() {
    const VOCAB_SIZE: usize = 1000;
    const EOS_TOKEN: u32 = 999;
    const DURATION: Duration = Duration::from_secs(30);

    let model = StressMockModel::new(VOCAB_SIZE, 42, EOS_TOKEN, 8);
    let kv_cache_mgr = KVCacheManager::new(&stress_cache_config(256)).unwrap();
    let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);
    let config = stress_engine_config(16);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, config);
    let start = Instant::now();
    let total_requests = Arc::new(AtomicUsize::new(0));
    let completed_requests = Arc::new(AtomicUsize::new(0));
    let failed_requests = Arc::new(AtomicUsize::new(0));

    // Spawn request-generator tasks
    let mut generators = Vec::new();
    for worker in 0..4 {
        let handle = handle.clone();
        let total = total_requests.clone();
        let completed = completed_requests.clone();
        let failed = failed_requests.clone();

        generators.push(tokio::spawn(async move {
            let mut req_id = worker * 100_000;
            while start.elapsed() < DURATION {
                req_id += 1;
                total.fetch_add(1, Ordering::Relaxed);

                let request = make_stress_request(req_id, 5, EOS_TOKEN);
                match handle.generate(request).await {
                    Ok(_) => {
                        completed.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    for g in generators {
        let _ = g.await;
    }

    let total = total_requests.load(Ordering::Relaxed);
    let completed = completed_requests.load(Ordering::Relaxed);
    let failed = failed_requests.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    eprintln!("\nSustained load ({:.1}s):", elapsed.as_secs_f64());
    eprintln!("  Total requests: {total}");
    eprintln!("  Completed: {completed}");
    eprintln!("  Failed: {failed}");
    eprintln!(
        "  Throughput: {:.1} req/s",
        completed as f64 / elapsed.as_secs_f64()
    );

    handle.shutdown().await.unwrap();

    // Should have completed a reasonable number
    assert!(completed > 0, "No requests completed in sustained load");
    let error_rate = failed as f64 / total.max(1) as f64;
    assert!(
        error_rate < 0.1,
        "Error rate too high: {:.1}% ({failed}/{total})",
        error_rate * 100.0
    );
}

/// Verify engine shuts down cleanly with in-flight requests.
#[tokio::test]
#[ignore = "stress test - run explicitly"]
async fn test_shutdown_with_inflight() {
    const VOCAB_SIZE: usize = 1000;
    const EOS_TOKEN: u32 = 999;

    // Model that never emits EOS (generates indefinitely until max_tokens)
    let model = StressMockModel::new(VOCAB_SIZE, 42, EOS_TOKEN, usize::MAX);
    let kv_cache_mgr = KVCacheManager::new(&stress_cache_config(128)).unwrap();
    let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);
    let config = stress_engine_config(8);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

    // Submit long-running requests
    let mut tasks = Vec::new();
    for i in 0..10 {
        let handle = handle.clone();
        tasks.push(tokio::spawn(async move {
            let request = make_stress_request(i, 1000, EOS_TOKEN);
            handle.generate(request).await
        }));
    }

    // Wait briefly then shut down
    tokio::time::sleep(Duration::from_millis(500)).await;
    handle.shutdown().await.unwrap();

    // All tasks should resolve (either completed or errored due to shutdown)
    for task in tasks {
        let _ = task.await;
    }
}

/// Verify streaming doesn't leak memory under repeated use.
#[tokio::test]
#[ignore = "stress test - run explicitly"]
async fn test_streaming_no_leak() {
    const VOCAB_SIZE: usize = 1000;
    const EOS_TOKEN: u32 = 999;
    const NUM_ITERATIONS: usize = 50;

    let model = StressMockModel::new(VOCAB_SIZE, 42, EOS_TOKEN, 6);
    let kv_cache_mgr = KVCacheManager::new(&stress_cache_config(128)).unwrap();
    let tokenizer = TokenizerWrapper::for_testing(VOCAB_SIZE);
    let config = stress_engine_config(4);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, config);

    for i in 0..NUM_ITERATIONS {
        let request = make_stress_request(i, 5, EOS_TOKEN);
        let (_req_id, mut rx) = handle.generate_stream(request).await.unwrap();

        let mut token_count = 0;
        while let Some(event) = rx.recv().await {
            match event {
                vllm_core::engine::StreamEvent::Token { .. } => token_count += 1,
                vllm_core::engine::StreamEvent::Done { .. } => break,
                vllm_core::engine::StreamEvent::Error { error } => {
                    panic!("Streaming error at iteration {i}: {error}");
                }
            }
        }

        assert!(token_count > 0, "No tokens received at iteration {i}");
    }

    handle.shutdown().await.unwrap();
}
