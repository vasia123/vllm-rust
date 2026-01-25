//! Admin API for monitoring and management.

pub mod metrics;
pub mod static_files;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use vllm_core::engine::EngineHandle;

use self::types::{ConfigSaveRequest, ConfigSaveResponse, HealthResponse, HealthStatus, RuntimeConfig};
use crate::api::error::ApiError;
use crate::config::ServerConfig;

/// Shared state for admin endpoints.
#[derive(Clone)]
pub struct AdminState {
    pub engine: EngineHandle,
    pub model_id: String,
    /// Server start time as duration since UNIX_EPOCH.
    pub start_time: Duration,
    /// Configuration snapshot.
    pub config: Arc<RuntimeConfig>,
    /// Whether the server is accepting new requests.
    accepting: Arc<AtomicBool>,
}

impl AdminState {
    pub fn new(
        engine: EngineHandle,
        model_id: String,
        start_time: Duration,
        config: RuntimeConfig,
    ) -> Self {
        Self {
            engine,
            model_id,
            start_time,
            config: Arc::new(config),
            accepting: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn accepting_requests(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }

    pub fn set_accepting(&self, accepting: bool) {
        self.accepting.store(accepting, Ordering::SeqCst);
    }
}

/// Create admin router with all admin endpoints.
pub fn create_admin_router(state: AdminState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics::get_metrics))
        .route("/metrics/stream", get(metrics::metrics_stream))
        .route("/config", get(get_config).post(save_config))
        .route("/", get(static_files::index_handler))
        .route("/*path", get(static_files::static_handler))
        .with_state(state)
}

/// GET /admin/health - Health check endpoint.
async fn health(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    let stats = state
        .engine
        .get_stats()
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let uptime = now
        .checked_sub(state.start_time)
        .unwrap_or_default()
        .as_secs();

    let status = if !state.accepting_requests() || stats.num_free_blocks == 0 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    };

    Ok(Json(HealthResponse {
        status,
        model_id: state.model_id.clone(),
        uptime_seconds: uptime,
    }))
}

/// GET /admin/config - Get current runtime configuration.
async fn get_config(State(state): State<AdminState>) -> impl IntoResponse {
    Json(state.config.as_ref().clone())
}

/// POST /admin/config - Save configuration to file.
async fn save_config(
    State(_state): State<AdminState>,
    Json(request): Json<ConfigSaveRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let config = ServerConfig {
        model: Some(request.config.model),
        draft_model: request.config.draft_model,
        num_speculative_tokens: Some(request.config.num_speculative_tokens),
        port: None, // Don't save port to avoid conflicts
        host: None, // Don't save host to avoid conflicts
        num_blocks: Some(request.config.num_blocks),
        max_requests: Some(request.config.max_requests),
        multi_step_count: Some(request.config.multi_step_count),
        max_tokens_per_step: Some(request.config.max_tokens_per_step),
        enable_chunked_prefill: Some(request.config.enable_chunked_prefill),
        enable_prefix_caching: Some(request.config.enable_prefix_caching),
    };

    let path = config
        .save()
        .map_err(|e| ApiError::InternalError(format!("Failed to save config: {}", e)))?;

    Ok(Json(ConfigSaveResponse {
        success: true,
        path: path.display().to_string(),
        message: "Configuration saved. Restart the server to apply changes.".to_string(),
    }))
}
