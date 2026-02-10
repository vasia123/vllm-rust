//! Admin API for monitoring and management.

pub mod metrics;
pub mod prometheus;
pub mod restart;
pub mod static_files;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::RwLock;

use self::restart::{
    restart_handler, restart_status_stream, AtomicEngineHandle, EngineBuilder, EngineController,
    RestartCoordinator, RestartState,
};
use self::types::{
    ConfigSaveRequest, ConfigSaveResponse, HealthResponse, HealthStatus, IsPausedResponse,
    PauseRequest, PauseResponse, RuntimeConfig,
};
use crate::api::error::ApiError;
use crate::config::ServerConfig;

/// Shared state for admin endpoints.
#[derive(Clone)]
pub struct AdminState {
    pub engine: AtomicEngineHandle,
    pub model_id: Arc<RwLock<String>>,
    /// Server start time as duration since UNIX_EPOCH.
    pub start_time: Duration,
    /// Configuration snapshot (mutable via restart).
    pub config: Arc<RwLock<RuntimeConfig>>,
    /// Whether the server is accepting new requests.
    accepting: Arc<AtomicBool>,
    /// Restart coordinator for graceful restarts.
    restart_coordinator: Arc<RestartCoordinator>,
    /// Engine builder for creating new engines.
    engine_builder: Arc<dyn EngineBuilder>,
}

impl AdminState {
    /// Create a new AdminState with atomic engine handle and restart coordinator.
    pub fn new(
        engine: AtomicEngineHandle,
        engine_controller: EngineController,
        model_id: String,
        start_time: Duration,
        config: RuntimeConfig,
        accepting: Arc<AtomicBool>,
        engine_builder: Arc<dyn EngineBuilder>,
    ) -> Self {
        let restart_coordinator = Arc::new(RestartCoordinator::new(
            engine_controller,
            engine.clone(),
            config.clone(),
            accepting.clone(),
        ));

        Self {
            engine,
            model_id: Arc::new(RwLock::new(model_id)),
            start_time,
            config: Arc::new(RwLock::new(config)),
            accepting,
            restart_coordinator,
            engine_builder,
        }
    }

    pub fn accepting_requests(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }

    pub fn set_accepting(&self, accepting: bool) {
        self.accepting.store(accepting, Ordering::SeqCst);
    }

    /// Get the accepting flag for sharing with other components.
    pub fn accepting_flag(&self) -> Arc<AtomicBool> {
        self.accepting.clone()
    }

    fn restart_state(&self) -> RestartState {
        RestartState {
            coordinator: self.restart_coordinator.clone(),
            engine_builder: self.engine_builder.clone(),
        }
    }
}

/// Create admin router with all admin endpoints.
pub fn create_admin_router(state: AdminState) -> Router {
    let restart_state = state.restart_state();

    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/live", get(live))
        .route("/metrics", get(metrics::get_metrics))
        .route("/metrics/stream", get(metrics::metrics_stream))
        .route("/metrics/prometheus", get(prometheus::prometheus_metrics))
        .route("/config", get(get_config).post(save_config))
        .route("/pause", post(pause_engine))
        .route("/resume", post(resume_engine))
        .route("/is_paused", get(is_engine_paused))
        .with_state(state)
        .route("/restart", post(restart_handler))
        .route("/restart/status", get(restart_status_stream))
        .with_state(restart_state)
        .route("/", get(static_files::index_handler))
        .route("/{*path}", get(static_files::static_handler))
}

/// GET /admin/health - Health check endpoint (detailed status).
async fn health(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    let stats = state
        .engine
        .get()
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
        model_id: state.model_id.read().await.clone(),
        uptime_seconds: uptime,
    }))
}

/// GET /admin/ready - Kubernetes readiness probe.
/// Returns 200 if server is ready to accept traffic, 503 otherwise.
async fn ready(State(state): State<AdminState>) -> impl IntoResponse {
    // Check if server is accepting requests
    if !state.accepting_requests() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }

    // Check if engine is responsive
    match state.engine.get().get_stats().await {
        Ok(stats) => {
            // Consider not ready if no free blocks (can't accept new requests)
            if stats.num_free_blocks == 0 {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::OK
            }
        }
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// GET /admin/live - Kubernetes liveness probe.
/// Returns 200 if server is alive (process is running), 503 if unhealthy.
async fn live(State(state): State<AdminState>) -> impl IntoResponse {
    // Liveness just checks if the engine can respond at all
    match state.engine.get().get_stats().await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

/// GET /admin/config - Get current runtime configuration.
async fn get_config(State(state): State<AdminState>) -> impl IntoResponse {
    Json(state.config.read().await.clone())
}

/// POST /admin/pause - Pause the engine.
async fn pause_engine(
    State(state): State<AdminState>,
    Json(request): Json<PauseRequest>,
) -> Result<impl IntoResponse, ApiError> {
    state
        .engine
        .get()
        .pause(request.mode)
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    let message = match request.mode {
        vllm_core::engine::PauseMode::Abort => {
            "Engine paused: all in-flight requests aborted".to_string()
        }
        vllm_core::engine::PauseMode::Wait => {
            "Engine paused: draining in-flight requests".to_string()
        }
        vllm_core::engine::PauseMode::Keep => "Engine paused: requests frozen in queue".to_string(),
    };

    Ok(Json(PauseResponse {
        paused: true,
        message,
    }))
}

/// POST /admin/resume - Resume a paused engine.
async fn resume_engine(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    state
        .engine
        .get()
        .resume()
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    Ok(Json(PauseResponse {
        paused: false,
        message: "Engine resumed".to_string(),
    }))
}

/// GET /admin/is_paused - Check if engine is paused.
async fn is_engine_paused(State(state): State<AdminState>) -> Result<impl IntoResponse, ApiError> {
    let paused = state
        .engine
        .get()
        .is_paused()
        .await
        .map_err(|e| ApiError::EngineError(e.to_string()))?;

    Ok(Json(IsPausedResponse { paused }))
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
        max_requests_per_second: None, // Rate limiting configured via CLI
        max_queue_depth: None,         // Queue depth configured via CLI
        allowed_origins: None,         // CORS configured via CLI
        allowed_methods: None,
        allowed_headers: None,
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
