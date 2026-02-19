//! Core types for the inference engine.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::distributed::{DpContext, ExpertParallelContext};
use crate::kv_cache::{MetricsSnapshot, PrefixCacheStatsSnapshot, SlidingWindowSnapshot};
use crate::lora::LoraRequest;
use crate::multimodal::ImageData;
use crate::prompt_adapter::PromptAdapterRequest;
use crate::request::{FinishReason, RequestId};
use crate::sampling::{SamplingConstraint, SamplingParams};

// ─── Streaming types ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token {
        token_id: u32,
        token_text: String,
        /// Log probability of the sampled token (present when logprobs requested).
        logprob: Option<f32>,
        /// Top-k alternative tokens with their log probabilities.
        top_logprobs: Option<Vec<(u32, f32)>>,
    },
    Done {
        finish_reason: FinishReason,
        generated_text: String,
        /// For stop token matches: the specific token ID that triggered the stop.
        stop_reason: Option<u32>,
    },
    Error {
        error: String,
    },
}

// ─── Engine errors ────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine has shut down")]
    Shutdown,
    #[error("engine is paused")]
    Paused,
    #[error("tokenization error: {0}")]
    Tokenization(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("cache error: {0}")]
    Cache(String),
}

// ─── Request/Response types ───────────────────────────────────────────────

pub struct GenerationRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub sampling_params: SamplingParams,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
    pub include_stop_str_in_output: bool,
    /// When true, EOS token does not trigger generation stop.
    pub ignore_eos: bool,
    pub logprobs: Option<u32>,
    pub echo: bool,
    /// LoRA adapter to use for this request (None = no adapter).
    pub lora_request: Option<LoraRequest>,
    /// Prompt adapter (soft prompt tuning) for this request (None = no adapter).
    pub prompt_adapter_request: Option<PromptAdapterRequest>,
    /// Sampling constraint for structured output (JSON schema, regex, etc.).
    pub constraint: Option<Box<dyn SamplingConstraint>>,
    /// Image inputs for multimodal models (e.g. LLaVA).
    /// Extracted from chat message content parts and decoded from data URIs.
    pub image_inputs: Vec<ImageData>,
    /// When true, skip reading from the prefix cache for this request.
    /// Useful when prompt_logprobs are requested (cached blocks have no logits).
    pub skip_prefix_cache: bool,
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
            ignore_eos: false,
            logprobs: None,
            echo: false,
            lora_request: None,
            prompt_adapter_request: None,
            constraint: None,
            image_inputs: Vec::new(),
            skip_prefix_cache: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub request_id: RequestId,
    pub generated_text: String,
    pub generated_token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    /// For stop token matches: the specific token ID that triggered the stop.
    pub stop_reason: Option<u32>,
    pub token_logprobs: Option<Vec<f32>>,
    pub top_logprobs: Option<Vec<Vec<(u32, f32)>>>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub prompt_logprobs: Option<Vec<Option<f32>>>,
}

// ─── Engine configuration ─────────────────────────────────────────────────

pub struct SpeculativeConfig {
    pub num_speculative_tokens: usize,
}

pub struct EngineConfig {
    pub scheduler_config: crate::scheduler::SchedulerConfig,
    pub block_size: usize,
    pub speculative_config: Option<SpeculativeConfig>,
    pub multi_step_count: usize,
    pub enable_prefix_caching: bool,
    pub cuda_graph_config: super::cuda_graph::CudaGraphConfig,
    /// Sliding window size for attention. When set, blocks entirely outside
    /// the window are reclaimed during decode to reduce memory usage.
    pub sliding_window: Option<usize>,
    /// When true, the engine pre-computes the next step's schedule before
    /// GPU execution finishes. The pre-schedule is validated on return and
    /// discarded if invalidated by completions or new commands.
    /// Cost of invalidation: one extra `compute_schedule` call (same as
    /// without the optimization). Never worse than disabled.
    pub enable_optimistic_scheduling: bool,
    /// Data parallelism context for this engine instance.
    ///
    /// Default is `DpContext::single_gpu()` (no coordination). Set to a
    /// multi-rank context when running DP inference to enable pre-forward-pass
    /// batch-size synchronization across all DP ranks.
    pub dp_context: DpContext,
    /// Expert parallelism context for MoE models.
    ///
    /// Default is `ExpertParallelContext::single_gpu()` (no EP overhead). Set to a
    /// multi-rank context when loading MoE models with EP to provide the rank identity
    /// and communicator required by `new_with_ep()` model constructors.
    pub ep_context: ExpertParallelContext,
}

impl EngineConfig {
    /// Start building an `EngineConfig` with the two required fields.
    /// All other fields use sensible defaults and can be overridden via
    /// builder methods.
    pub fn builder(
        scheduler_config: crate::scheduler::SchedulerConfig,
        speculative_config: Option<SpeculativeConfig>,
    ) -> EngineConfigBuilder {
        EngineConfigBuilder {
            scheduler_config,
            speculative_config,
            block_size: 16,
            multi_step_count: 1,
            enable_prefix_caching: false,
            cuda_graph_config: super::cuda_graph::CudaGraphConfig::default(),
            sliding_window: None,
            enable_optimistic_scheduling: true,
            dp_context: DpContext::single_gpu(),
            ep_context: ExpertParallelContext::single_gpu(),
        }
    }
}

pub struct EngineConfigBuilder {
    scheduler_config: crate::scheduler::SchedulerConfig,
    speculative_config: Option<SpeculativeConfig>,
    block_size: usize,
    multi_step_count: usize,
    enable_prefix_caching: bool,
    cuda_graph_config: super::cuda_graph::CudaGraphConfig,
    sliding_window: Option<usize>,
    enable_optimistic_scheduling: bool,
    dp_context: DpContext,
    ep_context: ExpertParallelContext,
}

impl EngineConfigBuilder {
    pub fn block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn multi_step_count(mut self, count: usize) -> Self {
        self.multi_step_count = count;
        self
    }

    pub fn enable_prefix_caching(mut self, enabled: bool) -> Self {
        self.enable_prefix_caching = enabled;
        self
    }

    pub fn cuda_graph_config(mut self, config: super::cuda_graph::CudaGraphConfig) -> Self {
        self.cuda_graph_config = config;
        self
    }

    pub fn sliding_window(mut self, window: Option<usize>) -> Self {
        self.sliding_window = window;
        self
    }

    pub fn enable_optimistic_scheduling(mut self, enabled: bool) -> Self {
        self.enable_optimistic_scheduling = enabled;
        self
    }

    /// Set the data parallelism context for this engine instance.
    ///
    /// Only needed when running multiple engine replicas in DP mode.
    /// The default is `DpContext::single_gpu()` (no coordination overhead).
    pub fn dp_context(mut self, ctx: DpContext) -> Self {
        self.dp_context = ctx;
        self
    }

    /// Set the expert parallelism context for MoE models.
    ///
    /// Only needed when loading MoE models with EP enabled.
    /// The default is `ExpertParallelContext::single_gpu()` (no EP overhead).
    pub fn ep_context(mut self, ctx: ExpertParallelContext) -> Self {
        self.ep_context = ctx;
        self
    }

    pub fn build(self) -> EngineConfig {
        EngineConfig {
            scheduler_config: self.scheduler_config,
            block_size: self.block_size,
            speculative_config: self.speculative_config,
            multi_step_count: self.multi_step_count,
            enable_prefix_caching: self.enable_prefix_caching,
            cuda_graph_config: self.cuda_graph_config,
            sliding_window: self.sliding_window,
            enable_optimistic_scheduling: self.enable_optimistic_scheduling,
            dp_context: self.dp_context,
            ep_context: self.ep_context,
        }
    }
}

// ─── Pause mode ──────────────────────────────────────────────────────────

/// How to handle in-flight requests when pausing the engine.
///
/// Mirrors vLLM's `PauseMode` (PR #32351) for reinforcement learning
/// workflows where model weights are updated between batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseMode {
    /// Abort all in-flight requests immediately. Default.
    #[default]
    Abort,
    /// Wait for in-flight requests to complete, then pause.
    /// New generation requests are rejected while draining.
    Wait,
    /// Freeze requests in the scheduler queue; they resume on `resume()`.
    /// No scheduling occurs while frozen.
    Keep,
}

// ─── Speculative Decoding Stats ───────────────────────────────────────────

/// Accumulated speculative decoding statistics.
///
/// Tracks draft proposal counts, acceptance rates, and per-position
/// acceptance to measure speculative decoding efficiency.
#[derive(Debug, Clone, Serialize)]
pub struct SpecDecodingStats {
    /// Maximum number of draft tokens per proposal (K).
    pub num_spec_tokens: usize,
    /// Number of draft rounds (one per request that went through spec decode).
    pub num_drafts: u64,
    /// Total draft tokens generated.
    pub num_draft_tokens: u64,
    /// Total draft tokens accepted by the target model.
    pub num_accepted_tokens: u64,
    /// Per-position acceptance counts, length = num_spec_tokens.
    /// Position i counts how many drafts had at least i+1 tokens accepted.
    pub num_accepted_tokens_per_pos: Vec<u64>,
}

impl SpecDecodingStats {
    pub fn new(num_spec_tokens: usize) -> Self {
        Self {
            num_spec_tokens,
            num_drafts: 0,
            num_draft_tokens: 0,
            num_accepted_tokens: 0,
            num_accepted_tokens_per_pos: vec![0; num_spec_tokens],
        }
    }

    /// Record a single draft proposal's verification results.
    pub fn observe_draft(&mut self, num_draft_tokens: usize, num_accepted_tokens: usize) {
        self.num_drafts += 1;
        self.num_draft_tokens += num_draft_tokens as u64;
        self.num_accepted_tokens += num_accepted_tokens as u64;
        for pos in self
            .num_accepted_tokens_per_pos
            .iter_mut()
            .take(num_accepted_tokens)
        {
            *pos += 1;
        }
    }

    /// Overall draft acceptance rate (0.0–1.0).
    pub fn acceptance_rate(&self) -> f64 {
        if self.num_draft_tokens == 0 {
            return 0.0;
        }
        self.num_accepted_tokens as f64 / self.num_draft_tokens as f64
    }

    /// Mean number of tokens accepted per draft round (includes bonus token).
    pub fn mean_acceptance_length(&self) -> f64 {
        if self.num_drafts == 0 {
            return 0.0;
        }
        1.0 + self.num_accepted_tokens as f64 / self.num_drafts as f64
    }

    /// Per-position acceptance rates (monotonically decreasing).
    pub fn per_position_acceptance_rates(&self) -> Vec<f64> {
        if self.num_drafts == 0 {
            return vec![0.0; self.num_spec_tokens];
        }
        self.num_accepted_tokens_per_pos
            .iter()
            .map(|&count| count as f64 / self.num_drafts as f64)
            .collect()
    }
}

// ─── Engine Stats ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub num_running_requests: usize,
    pub num_waiting_requests: usize,
    pub num_free_blocks: usize,
    pub num_total_blocks: usize,
    pub block_size: usize,
    /// Whether the engine is currently paused.
    pub is_paused: bool,
    pub kv_cache_metrics: MetricsSnapshot,
    /// Legacy prefix cache stats: (num_cached_blocks, num_evictable_blocks)
    pub prefix_cache_stats: Option<(usize, usize)>,
    /// Detailed prefix cache statistics (lifetime counters).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_detailed_stats: Option<PrefixCacheStatsSnapshot>,
    /// Sliding window prefix cache statistics (recent hit rate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_recent_stats: Option<SlidingWindowSnapshot>,
    /// Speculative decoding statistics (present only when spec decode is active).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_decode_stats: Option<SpecDecodingStats>,
}

// ─── Internal command types ───────────────────────────────────────────────

pub(crate) enum ResponseChannel {
    Complete(oneshot::Sender<Result<GenerationResult, EngineError>>),
    Stream(mpsc::Sender<StreamEvent>),
}

pub(crate) enum EngineCommand {
    Generate {
        request: GenerationRequest,
        response_tx: oneshot::Sender<Result<GenerationResult, EngineError>>,
    },
    GenerateStream {
        request: GenerationRequest,
        stream_tx: mpsc::Sender<StreamEvent>,
        /// Channel to report the assigned engine-internal request ID back
        /// to the caller. Used for abort-on-disconnect: the server can call
        /// `EngineHandle::abort(request_id)` when the client disconnects.
        request_id_tx: oneshot::Sender<RequestId>,
    },
    /// Abort a running request, freeing its GPU resources.
    Abort {
        request_id: crate::request::RequestId,
    },
    GetStats {
        response_tx: oneshot::Sender<EngineStats>,
    },
    /// Pause the engine, controlling how in-flight requests are handled.
    Pause {
        mode: PauseMode,
        response_tx: oneshot::Sender<Result<(), EngineError>>,
    },
    /// Resume a paused engine.
    Resume {
        response_tx: oneshot::Sender<Result<(), EngineError>>,
    },
    /// Query whether the engine is currently paused.
    IsPaused {
        response_tx: oneshot::Sender<bool>,
    },
    /// Reset (clear) the prefix cache, returning the number of evicted blocks.
    ResetPrefixCache {
        response_tx: oneshot::Sender<Result<usize, EngineError>>,
    },
    Shutdown,
}

// ─── Legacy types ─────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_mode_default_is_abort() {
        assert_eq!(PauseMode::default(), PauseMode::Abort);
    }

    #[test]
    fn pause_mode_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&PauseMode::Abort).unwrap(),
            "\"abort\""
        );
        assert_eq!(serde_json::to_string(&PauseMode::Wait).unwrap(), "\"wait\"");
        assert_eq!(serde_json::to_string(&PauseMode::Keep).unwrap(), "\"keep\"");
    }

    #[test]
    fn pause_mode_deserializes_snake_case() {
        assert_eq!(
            serde_json::from_str::<PauseMode>("\"abort\"").unwrap(),
            PauseMode::Abort
        );
        assert_eq!(
            serde_json::from_str::<PauseMode>("\"wait\"").unwrap(),
            PauseMode::Wait
        );
        assert_eq!(
            serde_json::from_str::<PauseMode>("\"keep\"").unwrap(),
            PauseMode::Keep
        );
    }

    #[test]
    fn engine_error_paused_display() {
        let err = EngineError::Paused;
        assert_eq!(err.to_string(), "engine is paused");
    }

    #[test]
    fn spec_decoding_stats_new() {
        let stats = SpecDecodingStats::new(5);
        assert_eq!(stats.num_spec_tokens, 5);
        assert_eq!(stats.num_drafts, 0);
        assert_eq!(stats.num_draft_tokens, 0);
        assert_eq!(stats.num_accepted_tokens, 0);
        assert_eq!(stats.num_accepted_tokens_per_pos, vec![0; 5]);
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.mean_acceptance_length(), 0.0);
    }

    #[test]
    fn spec_decoding_stats_observe_draft() {
        let mut stats = SpecDecodingStats::new(5);

        // Draft 1: proposed 5 tokens, 3 accepted
        stats.observe_draft(5, 3);
        assert_eq!(stats.num_drafts, 1);
        assert_eq!(stats.num_draft_tokens, 5);
        assert_eq!(stats.num_accepted_tokens, 3);
        assert_eq!(stats.num_accepted_tokens_per_pos, vec![1, 1, 1, 0, 0]);

        // Draft 2: proposed 4 tokens, all 4 accepted
        stats.observe_draft(4, 4);
        assert_eq!(stats.num_drafts, 2);
        assert_eq!(stats.num_draft_tokens, 9);
        assert_eq!(stats.num_accepted_tokens, 7);
        assert_eq!(stats.num_accepted_tokens_per_pos, vec![2, 2, 2, 1, 0]);
    }

    #[test]
    fn spec_decoding_stats_acceptance_rate() {
        let mut stats = SpecDecodingStats::new(3);
        stats.observe_draft(3, 2);
        stats.observe_draft(3, 1);
        // Total: 6 drafted, 3 accepted
        assert!((stats.acceptance_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn spec_decoding_stats_mean_acceptance_length() {
        let mut stats = SpecDecodingStats::new(3);
        stats.observe_draft(3, 2);
        stats.observe_draft(3, 0);
        // 2 drafts, 2 total accepted → mean = 1 + 2/2 = 2.0
        assert!((stats.mean_acceptance_length() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn spec_decoding_stats_per_position_rates() {
        let mut stats = SpecDecodingStats::new(3);
        // Draft 1: 3 accepted → [1,1,1]
        stats.observe_draft(3, 3);
        // Draft 2: 1 accepted → [2,1,1]
        stats.observe_draft(3, 1);
        // Draft 3: 0 accepted → [2,1,1]
        stats.observe_draft(3, 0);
        // 3 drafts: rates = [2/3, 1/3, 1/3]
        let rates = stats.per_position_acceptance_rates();
        assert!((rates[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((rates[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((rates[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn spec_decoding_stats_serializes() {
        let mut stats = SpecDecodingStats::new(2);
        stats.observe_draft(2, 1);
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["num_spec_tokens"], 2);
        assert_eq!(json["num_drafts"], 1);
        assert_eq!(json["num_draft_tokens"], 2);
        assert_eq!(json["num_accepted_tokens"], 1);
        assert_eq!(
            json["num_accepted_tokens_per_pos"],
            serde_json::json!([1, 0])
        );
    }

    #[test]
    fn engine_stats_omits_spec_decode_when_none() {
        let stats = EngineStats {
            num_running_requests: 0,
            num_waiting_requests: 0,
            num_free_blocks: 10,
            num_total_blocks: 10,
            block_size: 16,
            is_paused: false,
            kv_cache_metrics: Default::default(),
            prefix_cache_stats: None,
            prefix_cache_detailed_stats: None,
            prefix_cache_recent_stats: None,
            spec_decode_stats: None,
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert!(json.get("spec_decode_stats").is_none());
    }

    #[test]
    fn engine_stats_includes_spec_decode_when_present() {
        let mut sd_stats = SpecDecodingStats::new(3);
        sd_stats.observe_draft(3, 2);
        let stats = EngineStats {
            num_running_requests: 1,
            num_waiting_requests: 0,
            num_free_blocks: 8,
            num_total_blocks: 10,
            block_size: 16,
            is_paused: false,
            kv_cache_metrics: Default::default(),
            prefix_cache_stats: None,
            prefix_cache_detailed_stats: None,
            prefix_cache_recent_stats: None,
            spec_decode_stats: Some(sd_stats),
        };
        let json = serde_json::to_value(&stats).unwrap();
        let sd = json.get("spec_decode_stats").unwrap();
        assert_eq!(sd["num_drafts"], 1);
        assert_eq!(sd["num_accepted_tokens"], 2);
    }

    #[test]
    fn engine_config_builder_defaults() {
        let config = EngineConfig::builder(
            crate::scheduler::SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
                max_loras_per_batch: 0,
            },
            None,
        )
        .build();

        assert_eq!(config.block_size, 16);
        assert_eq!(config.multi_step_count, 1);
        assert!(!config.enable_prefix_caching);
        assert!(config.sliding_window.is_none());
        assert!(config.speculative_config.is_none());
        assert!(config.enable_optimistic_scheduling);
    }

    #[test]
    fn engine_config_builder_overrides() {
        let config = EngineConfig::builder(
            crate::scheduler::SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: true,
                scheduling_policy: crate::scheduler::SchedulingPolicy::Fcfs,
                max_loras_per_batch: 0,
            },
            Some(SpeculativeConfig {
                num_speculative_tokens: 5,
            }),
        )
        .block_size(4)
        .multi_step_count(3)
        .enable_prefix_caching(true)
        .sliding_window(Some(256))
        .enable_optimistic_scheduling(false)
        .build();

        assert_eq!(config.block_size, 4);
        assert_eq!(config.multi_step_count, 3);
        assert!(config.enable_prefix_caching);
        assert_eq!(config.sliding_window, Some(256));
        assert_eq!(config.speculative_config.unwrap().num_speculative_tokens, 5);
        assert!(!config.enable_optimistic_scheduling);
    }

    #[test]
    fn engine_stats_includes_is_paused() {
        let stats = EngineStats {
            num_running_requests: 0,
            num_waiting_requests: 0,
            num_free_blocks: 10,
            num_total_blocks: 10,
            block_size: 16,
            is_paused: true,
            kv_cache_metrics: Default::default(),
            prefix_cache_stats: None,
            prefix_cache_detailed_stats: None,
            prefix_cache_recent_stats: None,
            spec_decode_stats: None,
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["is_paused"], true);
    }
}
