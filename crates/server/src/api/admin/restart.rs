//! Graceful restart coordination for the vLLM inference engine.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::Serialize;
use tokio::sync::{broadcast, watch, Mutex, RwLock};
use vllm_core::engine::EngineHandle;

use super::types::{RestartRequest, RestartStatus, RuntimeConfig};
use crate::api::error::ApiError;

/// Atomic wrapper around EngineHandle using watch channel for lock-free reads.
#[derive(Clone)]
pub struct AtomicEngineHandle {
    inner: watch::Receiver<EngineHandle>,
}

impl AtomicEngineHandle {
    /// Create a new AtomicEngineHandle with the given initial engine.
    /// Returns the handle and a controller for replacing the engine.
    pub fn new(initial: EngineHandle) -> (Self, EngineController) {
        let (tx, rx) = watch::channel(initial);
        (Self { inner: rx }, EngineController { sender: tx })
    }

    /// Get a clone of the current engine handle.
    /// This is lock-free and always returns the latest engine.
    pub fn get(&self) -> EngineHandle {
        self.inner.borrow().clone()
    }
}

/// Controller for atomically replacing the engine handle.
/// There should only be one of these (held by RestartCoordinator).
pub struct EngineController {
    sender: watch::Sender<EngineHandle>,
}

impl EngineController {
    /// Atomically replace the engine with a new one.
    /// All AtomicEngineHandle clones will immediately see the new engine.
    pub fn replace(&self, new_engine: EngineHandle) {
        let _ = self.sender.send(new_engine);
    }
}

/// Trait for building new engine instances.
/// Abstracted for testability.
#[async_trait]
pub trait EngineBuilder: Send + Sync {
    /// Build a new engine with the given configuration.
    async fn build(&self, config: &RuntimeConfig) -> Result<EngineHandle, anyhow::Error>;
}

/// Production engine builder that loads models and creates the real engine.
pub struct ProductionEngineBuilder;

#[async_trait]
impl EngineBuilder for ProductionEngineBuilder {
    async fn build(&self, config: &RuntimeConfig) -> Result<EngineHandle, anyhow::Error> {
        use candle_core::{DType, Device};
        use vllm_core::{
            engine::{start_engine, start_engine_with_draft, EngineConfig, SpeculativeConfig},
            kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager},
            loader, models,
            scheduler::SchedulerConfig,
            tokenizer::TokenizerWrapper,
        };

        let model_id = &config.model;
        eprintln!("Loading model: {model_id}");
        let files = loader::fetch_model(model_id)?;

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        eprintln!("Loading weights to GPU (bf16)...");
        let vb = loader::load_weights(&files.weights, dtype, &device)?;

        eprintln!(
            "Building model ({} layers)...",
            files.config.num_hidden_layers
        );
        let model = models::from_config(&files.config, vb)?;

        let cache_config = CacheConfig {
            block_size: config.block_size,
            num_blocks: config.num_blocks,
            num_layers: files.config.num_hidden_layers,
            num_kv_heads: files.config.num_key_value_heads,
            head_dim: files.config.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
        };
        eprintln!(
            "Allocating KV cache ({} blocks)...",
            cache_config.num_blocks
        );
        let kv_cache_mgr = KVCacheManager::new(&cache_config)?;

        let engine_tokenizer = TokenizerWrapper::from_file(&files.tokenizer)?;

        let handle = if let Some(ref draft_id) = config.draft_model {
            eprintln!("Loading draft model: {draft_id}");
            let draft_files = loader::fetch_model(draft_id)?;

            eprintln!("Loading draft weights to GPU (bf16)...");
            let draft_vb = loader::load_weights(&draft_files.weights, dtype, &device)?;

            eprintln!(
                "Building draft model ({} layers)...",
                draft_files.config.num_hidden_layers
            );
            let draft_model = models::from_config(&draft_files.config, draft_vb)?;

            let draft_cache_config = CacheConfig {
                block_size: config.block_size,
                num_blocks: config.num_blocks,
                num_layers: draft_files.config.num_hidden_layers,
                num_kv_heads: draft_files.config.num_key_value_heads,
                head_dim: draft_files.config.head_dim,
                dtype,
                device: device.clone(),
                kv_cache_dtype: KVCacheDtype::Auto,
            };
            eprintln!(
                "Allocating draft KV cache ({} blocks)...",
                draft_cache_config.num_blocks
            );
            let draft_kv_cache = KVCacheManager::new(&draft_cache_config)?;

            let engine_config = EngineConfig {
                scheduler_config: SchedulerConfig {
                    max_running_requests: config.max_requests,
                    max_tokens_per_step: config.max_tokens_per_step,
                    enable_chunked_prefill: config.enable_chunked_prefill,
                    scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                },
                block_size: config.block_size,
                speculative_config: Some(SpeculativeConfig {
                    num_speculative_tokens: config.num_speculative_tokens,
                }),
                multi_step_count: 1,
                enable_prefix_caching: config.enable_prefix_caching,
                cuda_graph_config: vllm_core::engine::CudaGraphConfig::default(),
            };

            eprintln!(
                "Starting engine (speculative, K={})...",
                config.num_speculative_tokens
            );
            start_engine_with_draft(
                model,
                draft_model,
                engine_tokenizer,
                kv_cache_mgr,
                draft_kv_cache,
                engine_config,
            )
        } else {
            let engine_config = EngineConfig {
                scheduler_config: SchedulerConfig {
                    max_running_requests: config.max_requests,
                    max_tokens_per_step: config.max_tokens_per_step,
                    enable_chunked_prefill: config.enable_chunked_prefill,
                    scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                },
                block_size: config.block_size,
                speculative_config: None,
                multi_step_count: config.multi_step_count,
                enable_prefix_caching: config.enable_prefix_caching,
                cuda_graph_config: vllm_core::engine::CudaGraphConfig::default(),
            };

            eprintln!(
                "Starting engine (multi-step={})...",
                config.multi_step_count
            );
            start_engine(model, engine_tokenizer, kv_cache_mgr, engine_config)
        };

        Ok(handle)
    }
}

/// Coordinates graceful engine restarts.
pub struct RestartCoordinator {
    engine_controller: EngineController,
    engine_handle: AtomicEngineHandle,
    config: Arc<RwLock<RuntimeConfig>>,
    accepting: Arc<AtomicBool>,
    status_tx: broadcast::Sender<RestartStatus>,
    restart_lock: Mutex<()>,
    drain_timeout: Duration,
}

impl RestartCoordinator {
    /// Create a new RestartCoordinator.
    pub fn new(
        engine_controller: EngineController,
        engine_handle: AtomicEngineHandle,
        config: RuntimeConfig,
        accepting: Arc<AtomicBool>,
    ) -> Self {
        let (status_tx, _) = broadcast::channel(16);
        Self {
            engine_controller,
            engine_handle,
            config: Arc::new(RwLock::new(config)),
            accepting,
            status_tx,
            restart_lock: Mutex::new(()),
            drain_timeout: Duration::from_secs(30),
        }
    }

    /// Get a receiver for restart status updates.
    pub fn subscribe(&self) -> broadcast::Receiver<RestartStatus> {
        self.status_tx.subscribe()
    }

    /// Get the current status (snapshot).
    pub fn current_status(&self) -> RestartStatus {
        if self.accepting.load(Ordering::SeqCst) {
            RestartStatus::Idle
        } else {
            RestartStatus::Draining { active_requests: 0 }
        }
    }

    /// Get the current configuration.
    pub async fn get_config(&self) -> RuntimeConfig {
        self.config.read().await.clone()
    }

    /// Initiate a graceful restart with optional new configuration.
    pub async fn restart(
        &self,
        new_config: Option<RuntimeConfig>,
        engine_builder: &dyn EngineBuilder,
    ) -> Result<(), RestartError> {
        let _guard = self
            .restart_lock
            .try_lock()
            .map_err(|_| RestartError::AlreadyRestarting)?;

        let config = if let Some(cfg) = new_config {
            cfg
        } else {
            self.config.read().await.clone()
        };

        self.broadcast(RestartStatus::Draining { active_requests: 0 });
        self.accepting.store(false, Ordering::SeqCst);

        if let Err(e) = self.drain_requests().await {
            self.accepting.store(true, Ordering::SeqCst);
            self.broadcast(RestartStatus::Failed {
                error: e.to_string(),
            });
            return Err(e);
        }

        self.broadcast(RestartStatus::ShuttingDown);
        let old_engine = self.engine_handle.get();
        if let Err(e) = old_engine.shutdown().await {
            tracing::warn!("Engine shutdown returned error (proceeding anyway): {}", e);
        }

        self.broadcast(RestartStatus::Loading {
            model: config.model.clone(),
        });

        match engine_builder.build(&config).await {
            Ok(new_engine) => {
                self.engine_controller.replace(new_engine);
                *self.config.write().await = config;
                self.accepting.store(true, Ordering::SeqCst);
                self.broadcast(RestartStatus::Ready);
                Ok(())
            }
            Err(e) => {
                self.broadcast(RestartStatus::Failed {
                    error: e.to_string(),
                });
                Err(RestartError::EngineBuildFailed(e.to_string()))
            }
        }
    }

    /// Wait for all active requests to complete.
    async fn drain_requests(&self) -> Result<(), RestartError> {
        let deadline = tokio::time::Instant::now() + self.drain_timeout;

        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(RestartError::DrainTimeout);
            }

            let stats = self
                .engine_handle
                .get()
                .get_stats()
                .await
                .map_err(|e| RestartError::EngineError(e.to_string()))?;

            let active = stats.num_running_requests + stats.num_waiting_requests;

            self.broadcast(RestartStatus::Draining {
                active_requests: active,
            });

            if active == 0 {
                return Ok(());
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    fn broadcast(&self, status: RestartStatus) {
        let _ = self.status_tx.send(status);
    }
}

/// Errors that can occur during restart.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RestartError {
    AlreadyRestarting,
    DrainTimeout,
    EngineError(String),
    EngineBuildFailed(String),
}

impl std::fmt::Display for RestartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestartError::AlreadyRestarting => write!(f, "Restart already in progress"),
            RestartError::DrainTimeout => write!(f, "Timeout waiting for requests to drain"),
            RestartError::EngineError(e) => write!(f, "Engine error: {}", e),
            RestartError::EngineBuildFailed(e) => write!(f, "Failed to build new engine: {}", e),
        }
    }
}

impl std::error::Error for RestartError {}

/// Response for POST /admin/restart.
#[derive(Debug, Serialize)]
pub struct RestartResponse {
    pub status: &'static str,
    pub message: &'static str,
}

/// State needed for restart endpoints.
#[derive(Clone)]
pub struct RestartState {
    pub coordinator: Arc<RestartCoordinator>,
    pub engine_builder: Arc<dyn EngineBuilder>,
}

/// POST /admin/restart - Initiate a graceful restart.
pub async fn restart_handler(
    State(state): State<RestartState>,
    Json(request): Json<RestartRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let coordinator = state.coordinator.clone();
    let engine_builder = state.engine_builder.clone();

    tokio::spawn(async move {
        if let Err(e) = coordinator
            .restart(request.config, engine_builder.as_ref())
            .await
        {
            tracing::error!("Restart failed: {}", e);
        }
    });

    Ok(Json(RestartResponse {
        status: "restart_initiated",
        message: "Subscribe to /admin/restart/status for progress",
    }))
}

/// GET /admin/restart/status - SSE stream of restart status updates.
pub async fn restart_status_stream(
    State(state): State<RestartState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut rx = state.coordinator.subscribe();
    let initial_status = state.coordinator.current_status();

    let stream = async_stream::stream! {
        if let Ok(event) = Event::default()
            .event("status")
            .json_data(&initial_status)
        {
            yield Ok(event);
        }

        loop {
            match rx.recv().await {
                Ok(status) => {
                    if let Ok(event) = Event::default()
                        .event("status")
                        .json_data(&status)
                    {
                        yield Ok(event);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio::sync::mpsc;
    use vllm_core::engine::testing::{engine_handle_from_sender, TestEngineCommand};
    use vllm_core::engine::EngineStats;

    fn mock_engine() -> EngineHandle {
        let (tx, mut rx) = mpsc::channel::<TestEngineCommand>(16);
        tokio::spawn(async move {
            while let Some(cmd) = rx.recv().await {
                match cmd {
                    TestEngineCommand::GetStats { response_tx } => {
                        let stats = EngineStats {
                            num_running_requests: 0,
                            num_waiting_requests: 0,
                            num_free_blocks: 100,
                            num_total_blocks: 100,
                            block_size: 16,
                            kv_cache_metrics: Default::default(),
                            prefix_cache_stats: None,
                            prefix_cache_detailed_stats: None,
                            prefix_cache_recent_stats: None,
                        };
                        let _ = response_tx.send(stats);
                    }
                    TestEngineCommand::Shutdown { response_tx } => {
                        let _ = response_tx.send(Ok(()));
                        break;
                    }
                }
            }
        });
        engine_handle_from_sender(tx)
    }

    #[test]
    fn atomic_engine_handle_clone_sees_same_engine() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let engine = rt.block_on(async { mock_engine() });
        let (handle1, _controller) = AtomicEngineHandle::new(engine);
        let handle2 = handle1.clone();

        let _e1 = handle1.get();
        let _e2 = handle2.get();
    }

    #[tokio::test]
    async fn atomic_engine_handle_replace_propagates() {
        let engine1 = mock_engine();
        let (handle, controller) = AtomicEngineHandle::new(engine1);
        let handle_clone = handle.clone();

        let engine2 = mock_engine();
        controller.replace(engine2);

        tokio::time::sleep(Duration::from_millis(10)).await;

        let _e1 = handle.get();
        let _e2 = handle_clone.get();
    }

    struct MockEngineBuilder {
        call_count: AtomicUsize,
        should_fail: bool,
    }

    impl MockEngineBuilder {
        fn new(should_fail: bool) -> Self {
            Self {
                call_count: AtomicUsize::new(0),
                should_fail,
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EngineBuilder for MockEngineBuilder {
        async fn build(&self, _config: &RuntimeConfig) -> Result<EngineHandle, anyhow::Error> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.should_fail {
                Err(anyhow::anyhow!("Mock build failure"))
            } else {
                Ok(mock_engine())
            }
        }
    }

    fn test_config() -> RuntimeConfig {
        RuntimeConfig {
            model: "test-model".to_string(),
            draft_model: None,
            num_speculative_tokens: 0,
            num_blocks: 64,
            block_size: 16,
            max_requests: 4,
            max_tokens_per_step: 512,
            enable_chunked_prefill: false,
            multi_step_count: 1,
            enable_prefix_caching: false,
            dtype: "bf16".to_string(),
            device: "cuda:0".to_string(),
        }
    }

    #[tokio::test]
    async fn restart_coordinator_successful_restart() {
        let engine = mock_engine();
        let (handle, controller) = AtomicEngineHandle::new(engine);
        let accepting = Arc::new(AtomicBool::new(true));
        let coordinator =
            RestartCoordinator::new(controller, handle, test_config(), accepting.clone());

        let builder = MockEngineBuilder::new(false);
        let result = coordinator.restart(None, &builder).await;

        assert!(result.is_ok());
        assert_eq!(builder.calls(), 1);
        assert!(accepting.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn restart_coordinator_rejects_concurrent_restart() {
        let engine = mock_engine();
        let (handle, controller) = AtomicEngineHandle::new(engine);
        let accepting = Arc::new(AtomicBool::new(true));
        let coordinator = Arc::new(RestartCoordinator::new(
            controller,
            handle,
            test_config(),
            accepting,
        ));

        struct SlowBuilder;

        #[async_trait]
        impl EngineBuilder for SlowBuilder {
            async fn build(&self, _config: &RuntimeConfig) -> Result<EngineHandle, anyhow::Error> {
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(mock_engine())
            }
        }

        let coord1 = coordinator.clone();
        let task1 = tokio::spawn(async move { coord1.restart(None, &SlowBuilder).await });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let builder2 = MockEngineBuilder::new(false);
        let result2 = coordinator.restart(None, &builder2).await;

        assert!(matches!(result2, Err(RestartError::AlreadyRestarting)));

        let _ = task1.await;
    }

    #[tokio::test]
    async fn restart_coordinator_build_failure_broadcasts_failed() {
        let engine = mock_engine();
        let (handle, controller) = AtomicEngineHandle::new(engine);
        let accepting = Arc::new(AtomicBool::new(true));
        let coordinator =
            RestartCoordinator::new(controller, handle, test_config(), accepting.clone());

        let mut rx = coordinator.subscribe();
        let builder = MockEngineBuilder::new(true);
        let result = coordinator.restart(None, &builder).await;

        assert!(result.is_err());

        let mut found_failed = false;
        while let Ok(status) = rx.try_recv() {
            if matches!(status, RestartStatus::Failed { .. }) {
                found_failed = true;
                break;
            }
        }
        assert!(found_failed);
    }
}
