//! Core types for the inference engine.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};

use crate::kv_cache::{MetricsSnapshot, PrefixCacheStatsSnapshot, SlidingWindowSnapshot};
use crate::lora::LoraRequest;
use crate::multimodal::ImageData;
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
    /// Sampling constraint for structured output (JSON schema, regex, etc.).
    pub constraint: Option<Box<dyn SamplingConstraint>>,
    /// Image inputs for multimodal models (e.g. LLaVA).
    /// Extracted from chat message content parts and decoded from data URIs.
    pub image_inputs: Vec<ImageData>,
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
            constraint: None,
            image_inputs: Vec::new(),
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
        };
        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["is_paused"], true);
    }
}
