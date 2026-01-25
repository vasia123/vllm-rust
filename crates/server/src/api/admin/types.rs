//! Admin API types for serialization.

use serde::{Deserialize, Serialize};
use vllm_core::kv_cache::MetricsSnapshot;

/// Complete metrics snapshot for admin dashboard.
#[derive(Debug, Clone, Serialize)]
pub struct AdminMetrics {
    /// KV cache metrics snapshot.
    pub kv_cache: MetricsSnapshot,
    /// Number of requests currently running.
    pub running_requests: usize,
    /// Number of requests waiting in queue.
    pub waiting_requests: usize,
    /// Model identifier.
    pub model_id: String,
    /// Server uptime in seconds.
    pub uptime_seconds: u64,
    /// Whether the server is accepting new requests.
    pub accepting_requests: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Number of free KV cache blocks.
    pub num_free_blocks: usize,
    /// Total number of KV cache blocks.
    pub num_total_blocks: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Prefix cache statistics: (cached_blocks, evictable_blocks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_stats: Option<PrefixCacheStats>,
}

/// Prefix cache statistics.
#[derive(Debug, Clone, Serialize)]
pub struct PrefixCacheStats {
    pub cached_blocks: usize,
    pub evictable_blocks: usize,
}

/// Runtime configuration (read-only for now).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Model identifier.
    pub model: String,
    /// Draft model for speculative decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// Number of speculative tokens.
    pub num_speculative_tokens: usize,
    /// Number of KV cache blocks.
    pub num_blocks: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Maximum concurrent requests.
    pub max_requests: usize,
    /// Maximum tokens per scheduling step.
    pub max_tokens_per_step: usize,
    /// Enable chunked prefill.
    pub enable_chunked_prefill: bool,
    /// Multi-step decode count.
    pub multi_step_count: usize,
    /// Enable prefix caching.
    pub enable_prefix_caching: bool,
    /// Data type (e.g., "bf16").
    pub dtype: String,
    /// Device (e.g., "cuda:0").
    pub device: String,
}

/// Health check response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub model_id: String,
    pub uptime_seconds: u64,
}

/// Health status.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Restart request body.
#[derive(Debug, Clone, Deserialize)]
pub struct RestartRequest {
    /// New configuration to apply (optional).
    #[serde(default)]
    pub config: Option<RuntimeConfig>,
}

/// Restart status for SSE streaming.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RestartStatus {
    Idle,
    Draining { active_requests: usize },
    ShuttingDown,
    Loading { model: String },
    Ready,
    Failed { error: String },
}

/// Request body for saving configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ConfigSaveRequest {
    /// Configuration to save.
    pub config: RuntimeConfig,
}

/// Response for config save operation.
#[derive(Debug, Clone, Serialize)]
pub struct ConfigSaveResponse {
    /// Whether the save was successful.
    pub success: bool,
    /// Path where config was saved.
    pub path: String,
    /// Human-readable message.
    pub message: String,
}
