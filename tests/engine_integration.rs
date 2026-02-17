//! Integration tests for the full engine pipeline.
//!
//! These tests exercise the engine from request submission through to output,
//! using a MockModel that returns deterministic logits. All tests are CPU-only
//! and use tiny configurations.

use candle_core::{Device, Tensor};
use vllm_core::{
    engine::{start_engine, EngineConfig, GenerationRequest, ModelForward, StreamEvent},
    kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager},
    sampling::SamplingParams,
    scheduler::{SchedulerConfig, SchedulingPolicy},
    tokenizer::TokenizerWrapper,
};

// ─── Mock Models ─────────────────────────────────────────────────────────────

/// Mock model that always produces a fixed output token.
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
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dims()[1];
        let mut logits = vec![-100.0f32; seq_len * self.vocab_size];
        for pos in 0..seq_len {
            logits[pos * self.vocab_size + self.output_token as usize] = 100.0;
        }
        Tensor::from_vec(logits, (1, seq_len, self.vocab_size), &self.device)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// Mock model that emits EOS after a fixed number of decode steps.
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
        _kv_cache_mgr: &mut KVCacheManager,
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
        let mut logits = vec![-100.0f32; seq_len * self.vocab_size];
        for pos in 0..seq_len {
            logits[pos * self.vocab_size + token as usize] = 100.0;
        }
        Tensor::from_vec(logits, (1, seq_len, self.vocab_size), &self.device)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn test_cache_config() -> CacheConfig {
    CacheConfig {
        block_size: 16,
        num_blocks: 32,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: candle_core::DType::F32,
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
            scheduling_policy: SchedulingPolicy::Fcfs,
            max_loras_per_batch: 0,
        },
        None,
    )
    .build()
}

fn make_request(prompt: &str, max_new_tokens: usize, eos_token_id: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_string(),
        max_new_tokens,
        eos_token_id,
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
        skip_prefix_cache: false,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_submit_and_complete() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());
    let request = make_request("t1 t2 t3", 5, 999);

    let result = handle.generate(request).await.unwrap();

    assert_eq!(result.generated_token_ids.len(), 5);
    assert!(result.generated_token_ids.iter().all(|&t| t == 42));
    assert_eq!(
        result.finish_reason,
        vllm_core::request::FinishReason::Length
    );

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_max_tokens_one_completes_immediately() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());
    let request = make_request("t1 t2", 1, 999);

    let result = handle.generate(request).await.unwrap();

    assert_eq!(result.generated_token_ids.len(), 1);
    assert_eq!(result.generated_token_ids[0], 42);
    assert_eq!(
        result.finish_reason,
        vllm_core::request::FinishReason::Length
    );

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_multiple_concurrent_requests() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());

    let req1 = make_request("t1 t2 t3", 3, 999);
    let req2 = make_request("t4 t5", 4, 999);
    let req3 = make_request("t6 t7 t8 t9", 2, 999);

    let (r1, r2, r3) = tokio::join!(
        handle.generate(req1),
        handle.generate(req2),
        handle.generate(req3),
    );

    let r1 = r1.unwrap();
    let r2 = r2.unwrap();
    let r3 = r3.unwrap();

    assert_eq!(r1.generated_token_ids.len(), 3);
    assert_eq!(r2.generated_token_ids.len(), 4);
    assert_eq!(r3.generated_token_ids.len(), 2);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_handles_eos_token() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    // CountingMockModel emits token 42 twice, then EOS (999) on third decode
    let model = CountingMockModel::new(42, 999, 1000, 2);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());
    let request = make_request("t1 t2", 10, 999);

    let result = handle.generate(request).await.unwrap();

    // Should have generated 2 normal tokens + the EOS token, then stopped
    assert_eq!(result.generated_token_ids.len(), 3);
    assert_eq!(result.generated_token_ids[0], 42);
    assert_eq!(result.generated_token_ids[1], 42);
    assert_eq!(result.generated_token_ids[2], 999);
    assert_eq!(result.finish_reason, vllm_core::request::FinishReason::Eos);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_streaming_tokens_arrive_in_order() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());
    let request = make_request("t1 t2", 3, 999);

    let (_req_id, mut rx) = handle.generate_stream(request).await.unwrap();

    let mut token_events = Vec::new();
    let mut done_event = None;

    while let Some(event) = rx.recv().await {
        match &event {
            StreamEvent::Token { .. } => token_events.push(event),
            StreamEvent::Done { .. } => {
                done_event = Some(event);
                break;
            }
            StreamEvent::Error { error } => panic!("unexpected error: {error}"),
        }
    }

    // All token events should have arrived before the done event
    assert!(
        !token_events.is_empty(),
        "should have received token events"
    );
    assert!(done_event.is_some(), "should have received done event");

    // Verify done event has correct finish reason
    if let Some(StreamEvent::Done { finish_reason, .. }) = &done_event {
        assert_eq!(*finish_reason, vllm_core::request::FinishReason::Length);
    }

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_generation_stops_at_stop_token() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    // The model always produces token 42. We set 42 as a stop token.
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());

    let request = GenerationRequest {
        prompt: "t1 t2 t3".to_string(),
        max_new_tokens: 10,
        eos_token_id: 999,
        sampling_params: SamplingParams::greedy(),
        stop_token_ids: vec![42],
        stop_strings: Vec::new(),
        include_stop_str_in_output: false,
        logprobs: None,
        echo: false,
        ignore_eos: false,
        lora_request: None,
        prompt_adapter_request: None,
        constraint: None,
        image_inputs: Vec::new(),
        skip_prefix_cache: false,
    };

    let result = handle.generate(request).await.unwrap();

    // Should stop after producing the stop token
    assert!(result.generated_token_ids.len() <= 2);
    assert_eq!(result.finish_reason, vllm_core::request::FinishReason::Stop);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_get_stats() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());

    let stats = handle.get_stats().await.unwrap();

    assert_eq!(stats.block_size, 16);
    assert_eq!(stats.num_total_blocks, 32);
    // No requests running, all blocks should be free
    assert_eq!(stats.num_running_requests, 0);
    assert_eq!(stats.num_waiting_requests, 0);

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_generation_stops_at_stop_string() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());

    // The mock tokenizer maps token 42 -> "t42". We set "t42" as stop string.
    let request = GenerationRequest {
        prompt: "t1 t2 t3".to_string(),
        max_new_tokens: 10,
        eos_token_id: 999,
        sampling_params: SamplingParams::greedy(),
        stop_token_ids: Vec::new(),
        stop_strings: vec!["t42".to_string()],
        include_stop_str_in_output: false,
        logprobs: None,
        echo: false,
        ignore_eos: false,
        lora_request: None,
        prompt_adapter_request: None,
        constraint: None,
        image_inputs: Vec::new(),
        skip_prefix_cache: false,
    };

    let result = handle.generate(request).await.unwrap();

    // Should stop after the model produces text containing the stop string
    assert_eq!(result.finish_reason, vllm_core::request::FinishReason::Stop);
    // The stop string should NOT be in the output text
    assert!(
        !result.generated_text.contains("t42"),
        "stop string should be excluded from output"
    );

    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_engine_cancel_request_via_drop() {
    let kv_cache_mgr = KVCacheManager::new(&test_cache_config()).unwrap();
    let model = MockModel::new(42, 1000);
    let tokenizer = TokenizerWrapper::for_testing(1000);

    let handle = start_engine(model, tokenizer, kv_cache_mgr, test_engine_config());

    // Start a streaming request then drop the receiver to cancel
    let request = make_request("t1 t2", 100, 999);
    let rx = handle.generate_stream(request).await.unwrap();
    // Drop the receiver immediately to signal cancellation
    drop(rx);

    // Engine should still be functional after cancellation
    let request2 = make_request("t1 t2", 2, 999);
    let result = handle.generate(request2).await.unwrap();
    assert_eq!(result.generated_token_ids.len(), 2);

    handle.shutdown().await.unwrap();
}
