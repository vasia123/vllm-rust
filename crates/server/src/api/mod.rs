pub mod admin;
pub mod audio;
pub mod batch;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod middleware;
pub mod models;
pub mod realtime;
pub mod response_format;
pub mod responses;
pub mod responses_types;
pub mod streaming;
pub mod tokenize;
pub mod types;
pub mod validation;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use vllm_core::lora::LoraRequest;
use vllm_core::reasoning::ReasoningParser;
use vllm_core::tokenizer::{ChatTemplateEngine, TokenizerWrapper};
use vllm_core::tool_parser::ToolCallParser;

use responses_types::ResponsesResponse;

/// Default maximum request body size: 32 MiB.
const DEFAULT_MAX_BODY_SIZE: usize = 32 * 1024 * 1024;

pub use admin::restart::{
    AtomicEngineHandle, EngineBuilder, EngineController, ProductionEngineBuilder,
};
pub use admin::{create_admin_router, AdminState};

#[derive(Clone)]
pub struct AppState {
    pub engine: AtomicEngineHandle,
    pub model_id: String,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub chat_template: Option<Arc<ChatTemplateEngine>>,
    pub eos_token_id: u32,
    pub max_model_len: usize,
    /// Tool call parser for extracting function calls from model output.
    pub tool_call_parser: Arc<dyn ToolCallParser>,
    /// Reasoning parser for extracting chain-of-thought content from model output.
    pub reasoning_parser: Option<Arc<dyn ReasoningParser>>,
    /// Whether the server is accepting new requests.
    accepting: Arc<AtomicBool>,
    /// Count of in-flight GPU requests (for /load endpoint).
    server_load: Arc<AtomicUsize>,
    /// Registry of dynamically loaded LoRA adapters (name → request).
    lora_adapters: Arc<RwLock<HashMap<String, LoraRequest>>>,
    /// Monotonically increasing ID counter for LoRA adapters.
    next_lora_id: Arc<AtomicUsize>,
    /// In-memory store for completed Responses API objects (response_id → response).
    pub response_store: Arc<RwLock<HashMap<String, ResponsesResponse>>>,
    /// In-memory store for batch jobs (batch_id → job).
    pub batch_store: batch::BatchStore,
    /// Role name used in chat completion responses (default: "assistant").
    pub response_role: String,
    /// If true, automatically attempt tool-call parsing when tools are provided but
    /// `tool_choice` is not set. Mirrors vLLM `--enable-auto-tool-choice`.
    pub enable_auto_tool_choice: bool,
    /// If true, include raw `token_ids` in non-streaming choices and format logprob
    /// token strings as `"token_{id}"`. Mirrors vLLM `--return-tokens-as-token-ids`.
    pub return_tokens_as_token_ids: bool,
    /// Maximum number of log probabilities per token the server will return.
    /// Caps both `top_logprobs` (chat) and `logprobs` (completions).
    pub max_logprobs: usize,
    /// Per-modality count limits for multimodal inputs in a single request.
    /// Keys: "image", "video", "audio". Empty map means unlimited.
    pub mm_limits: HashMap<String, usize>,
    /// Minimum number of tokens to buffer before sending a streaming chunk.
    /// Value of 1 (default) sends every token immediately; higher values batch
    /// tokens to reduce SSE/HTTP overhead. Mirrors vLLM `--stream-interval`.
    pub stream_interval: usize,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: AtomicEngineHandle,
        model_id: String,
        tokenizer: Arc<TokenizerWrapper>,
        chat_template: Option<Arc<ChatTemplateEngine>>,
        eos_token_id: u32,
        max_model_len: usize,
        tool_call_parser: Arc<dyn ToolCallParser>,
        reasoning_parser: Option<Arc<dyn ReasoningParser>>,
        accepting: Arc<AtomicBool>,
        response_role: String,
        enable_auto_tool_choice: bool,
        return_tokens_as_token_ids: bool,
        max_logprobs: usize,
        mm_limits: HashMap<String, usize>,
        stream_interval: usize,
    ) -> Self {
        Self {
            engine,
            model_id,
            tokenizer,
            chat_template,
            eos_token_id,
            max_model_len,
            tool_call_parser,
            reasoning_parser,
            accepting,
            server_load: Arc::new(AtomicUsize::new(0)),
            lora_adapters: Arc::new(RwLock::new(HashMap::new())),
            next_lora_id: Arc::new(AtomicUsize::new(1)),
            response_store: Arc::new(RwLock::new(HashMap::new())),
            batch_store: batch::new_batch_store(),
            response_role,
            enable_auto_tool_choice,
            return_tokens_as_token_ids,
            max_logprobs,
            mm_limits,
            stream_interval: stream_interval.max(1),
        }
    }

    pub fn accepting_requests(&self) -> bool {
        self.accepting.load(Ordering::SeqCst)
    }

    /// Increment the in-flight GPU request counter.
    pub fn increment_load(&self) {
        self.server_load.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the in-flight GPU request counter.
    pub fn decrement_load(&self) {
        self.server_load.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current number of in-flight GPU requests.
    pub fn current_load(&self) -> usize {
        self.server_load.load(Ordering::Relaxed)
    }
}

/// Create a tool call parser by name.
///
/// Supported names: `hermes`, `glm4`, `json`, `llama`, `mistral`, `deepseek_v3`,
/// `deepseek_v31`, `internlm2`, `jamba`, `pythonic`, `granite`, `granite-20b-fc`,
/// `kimi_k2`, `phi4mini`, `longcat`, `xlam`.
/// Defaults to `hermes` for unknown names.
pub fn create_tool_call_parser(name: &str) -> Arc<dyn ToolCallParser> {
    use vllm_core::tool_parser::*;

    match name {
        "hermes" => Arc::new(HermesToolParser::new()),
        "glm4" | "glm45" | "glm47" | "glm4_moe" => Arc::new(Glm4ToolParser::new()),
        "json" | "openai" => Arc::new(JsonToolParser::new()),
        "llama" | "llama3_json" | "llama4_json" => Arc::new(LlamaToolParser::new()),
        "llama4_pythonic" => Arc::new(Llama4PythonicToolParser::new()),
        "mistral" => Arc::new(MistralToolParser::new()),
        "deepseek_v3" => Arc::new(DeepSeekV3ToolParser::new()),
        "deepseek_v31" => Arc::new(DeepSeekV31ToolParser::new()),
        "internlm" | "internlm2" => Arc::new(InternLm2ToolParser::new()),
        "jamba" => Arc::new(JambaToolParser::new()),
        "pythonic" => Arc::new(PythonicToolParser::new()),
        "olmo3" => Arc::new(Olmo3PythonicToolParser::new()),
        "granite" => Arc::new(GraniteToolParser::new()),
        "granite-20b-fc" => Arc::new(Granite20bFCToolParser::new()),
        "kimi_k2" | "kimi-k2" => Arc::new(KimiK2ToolParser::new()),
        "phi4mini" | "phi4_mini_json" => Arc::new(Phi4MiniToolParser::new()),
        "longcat" => Arc::new(LongcatToolParser::new()),
        "xlam" => Arc::new(XLamToolParser::new()),
        "gigachat3" | "gigachat" => Arc::new(GigaChat3ToolParser::new()),
        "functiongemma" | "function_gemma" => Arc::new(FunctionGemmaToolParser::new()),
        "hunyuan" | "hunyuan_a13b" => Arc::new(HunyuanToolParser::new()),
        "ernie45" | "ernie_45" | "ernie-4.5" => Arc::new(Ernie45ToolParser::new()),
        "seed_oss" | "seed-oss" => Arc::new(SeedOssToolParser::new()),
        "minimax" => Arc::new(MinimaxToolParser::new()),
        "minimax_m2" => Arc::new(MinimaxM2ToolParser::new()),
        "deepseek_v32" => Arc::new(DeepSeekV32ToolParser::new()),
        "step3" | "step-3" => Arc::new(Step3ToolParser::new()),
        "step3p5" | "step-3.5" | "qwen3_xml" | "qwen3xml" => Arc::new(Qwen3CoderToolParser::new()),
        "qwen3coder" | "qwen3_coder" => Arc::new(Qwen3CoderToolParser::new()),
        unknown => {
            tracing::warn!("Unknown tool call parser '{unknown}', defaulting to hermes");
            Arc::new(HermesToolParser::new())
        }
    }
}

/// Create a reasoning parser wrapped in `Arc`, or `None` if disabled.
///
/// An empty name or `"identity"` returns `None` (no reasoning extraction).
/// Otherwise, creates the named parser via `vllm_core::reasoning::create_reasoning_parser`.
pub fn create_reasoning_parser_arc(name: &str) -> Option<Arc<dyn ReasoningParser>> {
    if name.is_empty() || name == "identity" {
        None
    } else {
        Some(Arc::from(vllm_core::reasoning::create_reasoning_parser(
            name,
        )))
    }
}

/// Configuration for CORS middleware.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Comma-separated allowed origins, or "*" for all.
    pub allowed_origins: String,
    /// Comma-separated allowed methods.
    pub allowed_methods: String,
    /// Comma-separated allowed headers, or "*" for all.
    pub allowed_headers: String,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: "*".to_string(),
            allowed_methods: "GET,POST,OPTIONS".to_string(),
            allowed_headers: "*".to_string(),
        }
    }
}

/// Build a `CorsLayer` from a `CorsConfig`.
///
/// When all three fields use their wildcard defaults ("*" for origins/headers,
/// "GET,POST,OPTIONS" for methods), this returns `CorsLayer::very_permissive()`.
/// Otherwise it parses each field into the corresponding typed values.
pub fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    if config.allowed_origins == "*"
        && config.allowed_headers == "*"
        && config.allowed_methods == "GET,POST,OPTIONS"
    {
        return CorsLayer::very_permissive();
    }

    let mut layer = CorsLayer::new();

    // Origins
    if config.allowed_origins == "*" {
        layer = layer.allow_origin(AllowOrigin::any());
    } else {
        let origins: Vec<HeaderValue> = config
            .allowed_origins
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return None;
                }
                HeaderValue::from_str(trimmed).ok()
            })
            .collect();
        layer = layer.allow_origin(origins);
    }

    // Methods
    let methods: Vec<Method> = config
        .allowed_methods
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            trimmed.parse::<Method>().ok()
        })
        .collect();
    layer = layer.allow_methods(AllowMethods::list(methods));

    // Headers
    if config.allowed_headers == "*" {
        layer = layer.allow_headers(AllowHeaders::any());
    } else {
        let headers: Vec<HeaderName> = config
            .allowed_headers
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    return None;
                }
                trimmed.parse::<HeaderName>().ok()
            })
            .collect();
        layer = layer.allow_headers(AllowHeaders::list(headers));
    }

    layer
}

// ─── Root-level health and version endpoints ───────────────────────────────

/// Simple health check for load balancers and orchestration systems.
/// Returns 200 if the server is accepting requests, 503 otherwise.
async fn health_check(axum::extract::State(state): axum::extract::State<AppState>) -> StatusCode {
    if state.accepting_requests() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Server version information.
#[derive(Serialize)]
struct VersionInfo {
    version: &'static str,
}

async fn version() -> Json<VersionInfo> {
    Json(VersionInfo {
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// GET /server_info — Server configuration and environment information.
#[derive(Serialize)]
struct ServerInfo {
    version: &'static str,
    model_id: String,
    max_model_len: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template: Option<String>,
    accepting_requests: bool,
    server_load: usize,
}

async fn server_info(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<ServerInfo> {
    let chat_template = state
        .chat_template
        .as_ref()
        .map(|ct| ct.raw_template().to_string());

    Json(ServerInfo {
        version: env!("CARGO_PKG_VERSION"),
        model_id: state.model_id.clone(),
        max_model_len: state.max_model_len,
        chat_template,
        accepting_requests: state.accepting_requests(),
        server_load: state.current_load(),
    })
}

/// Server load metrics (count of in-flight GPU requests).
#[derive(Serialize)]
struct ServerLoadMetrics {
    server_load: usize,
}

async fn get_server_load(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<ServerLoadMetrics> {
    Json(ServerLoadMetrics {
        server_load: state.current_load(),
    })
}

/// POST /start_profile — Start performance profiling.
///
/// Stub endpoint for API compatibility with Python vLLM.
/// Returns 200 OK immediately. Actual profiling can be added via
/// external tools (perf, flamegraph, tokio-console).
async fn start_profile() -> StatusCode {
    tracing::info!("Profile start requested (stub)");
    StatusCode::OK
}

/// POST /stop_profile — Stop performance profiling.
///
/// Stub endpoint for API compatibility with Python vLLM.
async fn stop_profile() -> StatusCode {
    tracing::info!("Profile stop requested (stub)");
    StatusCode::OK
}

/// GET/POST /ping — Lightweight liveness probe for load balancers.
///
/// Always returns 200 OK regardless of server state.
async fn ping() -> StatusCode {
    StatusCode::OK
}

/// POST /reset_prefix_cache — Reset the prefix cache, evicting all cached blocks.
///
/// Compatible with vLLM's `/reset_prefix_cache` endpoint.
/// Returns a JSON object with `{ "success": true }` on success.
async fn reset_prefix_cache(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<impl axum::response::IntoResponse, error::ApiError> {
    let num_evicted = state
        .engine
        .get()
        .reset_prefix_cache()
        .await
        .map_err(|e| error::ApiError::EngineError(e.to_string()))?;

    tracing::info!("Prefix cache reset: {num_evicted} blocks evicted");

    Ok(Json(serde_json::json!({ "success": true })))
}

/// POST /sleep — Put the engine to sleep (pause with Keep mode).
///
/// Compatible with vLLM's `/sleep` endpoint.
/// Paused engine keeps existing requests frozen but rejects new ones.
async fn sleep_engine(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<impl axum::response::IntoResponse, error::ApiError> {
    state
        .engine
        .get()
        .pause(vllm_core::engine::PauseMode::Keep)
        .await
        .map_err(|e| error::ApiError::EngineError(e.to_string()))?;

    tracing::info!("Engine put to sleep");
    Ok(Json(serde_json::json!({ "success": true })))
}

/// POST /wake_up — Wake the engine from sleep (resume).
///
/// Compatible with vLLM's `/wake_up` endpoint.
async fn wake_up_engine(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<impl axum::response::IntoResponse, error::ApiError> {
    state
        .engine
        .get()
        .resume()
        .await
        .map_err(|e| error::ApiError::EngineError(e.to_string()))?;

    tracing::info!("Engine woken up");
    Ok(Json(serde_json::json!({ "success": true })))
}

/// GET /is_sleeping — Check if the engine is sleeping (paused).
///
/// Compatible with vLLM's `/is_sleeping` endpoint.
async fn is_sleeping(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Result<impl axum::response::IntoResponse, error::ApiError> {
    let paused = state
        .engine
        .get()
        .is_paused()
        .await
        .map_err(|e| error::ApiError::EngineError(e.to_string()))?;

    Ok(Json(serde_json::json!({ "is_sleeping": paused })))
}

/// Request to load a LoRA adapter dynamically.
#[derive(Debug, Deserialize)]
pub struct LoadLoRAAdapterRequest {
    pub lora_name: String,
    pub lora_path: String,
    #[serde(default)]
    pub load_inplace: bool,
}

/// Request to unload a LoRA adapter.
#[derive(Debug, Deserialize)]
pub struct UnloadLoRAAdapterRequest {
    pub lora_name: String,
    #[serde(default)]
    pub lora_int_id: Option<u32>,
}

/// POST /v1/load_lora_adapter — Load a LoRA adapter at runtime.
async fn load_lora_adapter(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<LoadLoRAAdapterRequest>,
) -> Result<String, error::ApiError> {
    if req.lora_name.is_empty() {
        return Err(error::ApiError::InvalidRequest(
            "lora_name is required".to_string(),
        ));
    }
    if req.lora_path.is_empty() {
        return Err(error::ApiError::InvalidRequest(
            "lora_path is required".to_string(),
        ));
    }

    let mut adapters = state.lora_adapters.write().await;
    if !req.load_inplace && adapters.contains_key(&req.lora_name) {
        return Err(error::ApiError::InvalidRequest(format!(
            "LoRA adapter '{}' is already loaded. Use load_inplace=true to reload.",
            req.lora_name
        )));
    }

    let lora_id = state.next_lora_id.fetch_add(1, Ordering::Relaxed) as u32;
    let lora_request = LoraRequest::new(req.lora_name.clone(), lora_id, req.lora_path);
    adapters.insert(req.lora_name.clone(), lora_request);

    Ok(format!(
        "Success: LoRA adapter '{}' added successfully.",
        req.lora_name
    ))
}

/// GET /v1/lora_adapters — List all loaded LoRA adapters.
async fn list_lora_adapters(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> impl axum::response::IntoResponse {
    let adapters = state.lora_adapters.read().await;
    let list: Vec<serde_json::Value> = adapters
        .values()
        .map(|lr| {
            serde_json::json!({
                "lora_name": lr.name,
                "lora_int_id": lr.id,
                "lora_path": lr.path,
            })
        })
        .collect();
    Json(serde_json::json!({ "lora_adapters": list }))
}

/// POST /v1/unload_lora_adapter — Unload a LoRA adapter.
async fn unload_lora_adapter(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<UnloadLoRAAdapterRequest>,
) -> Result<String, error::ApiError> {
    if req.lora_name.is_empty() {
        return Err(error::ApiError::InvalidRequest(
            "lora_name is required".to_string(),
        ));
    }

    let mut adapters = state.lora_adapters.write().await;
    if adapters.remove(&req.lora_name).is_none() {
        return Err(error::ApiError::InvalidRequest(format!(
            "LoRA adapter '{}' is not loaded.",
            req.lora_name
        )));
    }

    Ok(format!(
        "Success: LoRA adapter '{}' removed successfully.",
        req.lora_name
    ))
}

/// GET /v1/responses/{response_id} — Retrieve a stored response object.
async fn get_response(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Result<Json<ResponsesResponse>, error::ApiError> {
    let store = state.response_store.read().await;
    match store.get(&response_id) {
        Some(resp) => Ok(Json(resp.clone())),
        None => Err(error::ApiError::InvalidRequest(format!(
            "Response '{}' not found",
            response_id,
        ))),
    }
}

/// POST /v1/responses/{response_id}/cancel — Cancel a response.
///
/// For completed/failed responses, this returns the response unchanged.
/// For in-progress responses, this would cancel generation (currently a no-op
/// since we store responses only after completion).
async fn cancel_response(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Result<Json<ResponsesResponse>, error::ApiError> {
    let mut store = state.response_store.write().await;
    match store.get_mut(&response_id) {
        Some(resp) => {
            // Only cancel if still in progress
            if resp.status == responses_types::ResponseStatus::InProgress {
                resp.status = responses_types::ResponseStatus::Cancelled;
            }
            Ok(Json(resp.clone()))
        }
        None => Err(error::ApiError::InvalidRequest(format!(
            "Response '{}' not found",
            response_id,
        ))),
    }
}

/// GET /metrics — Prometheus-format metrics scraping endpoint.
async fn prometheus_metrics() -> Result<axum::response::Response, error::ApiError> {
    use axum::http::header::CONTENT_TYPE;
    use axum::response::IntoResponse;

    use ::prometheus::Encoder;
    let encoder = ::prometheus::TextEncoder::new();
    let metric_families = ::prometheus::gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .map_err(|e| error::ApiError::InternalError(format!("Failed to encode metrics: {}", e)))?;

    Ok((
        [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        buffer,
    )
        .into_response())
}

/// Tokenize bad_words strings into token ID sequences for the `NoBadWordsProcessor`.
///
/// Each bad word is tokenized both with and without a leading space (to handle
/// prefix-space tokenization variants). Multi-token sequences only ban the last
/// token when the prefix matches generated output.
pub fn tokenize_bad_words(bad_words: &[String], tokenizer: &TokenizerWrapper) -> Vec<Vec<u32>> {
    let mut result: Vec<Vec<u32>> = Vec::new();

    for word in bad_words {
        let stripped = word.trim_start();
        if stripped.is_empty() {
            continue;
        }

        // Tokenize without prefix space
        if let Ok(ids) = tokenizer.encode(stripped) {
            if !ids.is_empty() {
                result.push(ids);
            }
        }

        // Tokenize with prefix space (captures tokenizers that use add_prefix_space)
        let with_space = format!(" {stripped}");
        if let Ok(ids) = tokenizer.encode(&with_space) {
            if !ids.is_empty() {
                // Only add if it differs from the no-space variant
                let dominated = result
                    .last()
                    .is_none_or(|prev| ids[0] != prev[0] && ids.len() == prev.len());
                if dominated {
                    result.push(ids);
                }
            }
        }
    }

    result
}

pub fn create_router(state: AppState) -> Router {
    create_router_with_cors(state, CorsLayer::very_permissive())
}

pub fn create_router_with_cors(state: AppState, cors: CorsLayer) -> Router {
    let accepting = state.accepting.clone();
    Router::new()
        .route("/health", get(health_check))
        .route("/version", get(version))
        .route("/load", get(get_server_load))
        .route("/metrics", get(prometheus_metrics))
        .route("/server_info", get(server_info))
        .route("/ping", get(ping).post(ping))
        .route("/reset_prefix_cache", post(reset_prefix_cache))
        .route("/sleep", post(sleep_engine))
        .route("/wake_up", post(wake_up_engine))
        .route("/is_sleeping", get(is_sleeping))
        .route("/start_profile", post(start_profile))
        .route("/stop_profile", post(stop_profile))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/{model_id}", get(models::retrieve_model))
        .route("/v1/completions", post(completions::create_completion))
        .route(
            "/v1/completions/render",
            post(completions::render_completion),
        )
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route(
            "/v1/chat/completions/render",
            post(chat::render_chat_completion),
        )
        .route("/v1/responses", post(responses::create_response))
        .route("/v1/responses/{response_id}", get(get_response))
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
        .route("/v1/embeddings", post(embeddings::create_embedding))
        .route("/score", post(embeddings::score))
        .route("/v1/score", post(embeddings::score))
        .route("/rerank", post(embeddings::rerank))
        .route("/v1/rerank", post(embeddings::rerank))
        .route("/v2/rerank", post(embeddings::rerank))
        .route("/pooling", post(embeddings::pooling))
        .route("/v1/pooling", post(embeddings::pooling))
        .route("/classify", post(embeddings::classify))
        .route("/v1/classify", post(embeddings::classify))
        .route("/v1/load_lora_adapter", post(load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(unload_lora_adapter))
        .route("/v1/lora_adapters", get(list_lora_adapters))
        .route("/v1/tokenize", post(tokenize::tokenize))
        .route("/v1/detokenize", post(tokenize::detokenize))
        .route("/tokenizer_info", get(tokenize::get_tokenizer_info))
        .route("/v1/batches", post(batch::create_batch))
        .route("/v1/batches/{batch_id}", get(batch::get_batch))
        .route(
            "/v1/batches/{batch_id}/output",
            get(batch::get_batch_output),
        )
        .route("/v1/batches/{batch_id}/cancel", post(batch::cancel_batch))
        .route(
            "/v1/audio/transcriptions",
            post(audio::create_transcription),
        )
        .route("/v1/audio/translations", post(audio::create_translation))
        .route("/v1/realtime", get(realtime::realtime_ws))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
        .layer(cors)
        .with_state(state)
}

/// Create the full router including admin endpoints.
pub fn create_full_router(app_state: AppState, admin_state: AdminState) -> Router {
    create_full_router_with_cors_and_rate_limit(
        app_state,
        admin_state,
        CorsLayer::very_permissive(),
        middleware::RateLimitState::unlimited(),
    )
}

/// Create the full router with CORS and rate limiting.
pub fn create_full_router_with_cors_and_rate_limit(
    app_state: AppState,
    admin_state: AdminState,
    cors: CorsLayer,
    rate_limit_state: middleware::RateLimitState,
) -> Router {
    create_full_router_with_options(
        app_state,
        admin_state,
        cors,
        rate_limit_state,
        DEFAULT_MAX_BODY_SIZE,
        true,
    )
}

/// Create the full router with all configurable options.
pub fn create_full_router_with_options(
    app_state: AppState,
    admin_state: AdminState,
    cors: CorsLayer,
    rate_limit_state: middleware::RateLimitState,
    max_body_size: usize,
    enable_request_logging: bool,
) -> Router {
    create_full_router_with_all_options(
        app_state,
        admin_state,
        cors,
        rate_limit_state,
        max_body_size,
        middleware::ApiKeyState::disabled(),
        enable_request_logging,
    )
}

/// Create the full router with all configurable options including API key auth.
pub fn create_full_router_with_all_options(
    app_state: AppState,
    admin_state: AdminState,
    cors: CorsLayer,
    rate_limit_state: middleware::RateLimitState,
    max_body_size: usize,
    api_key_state: middleware::ApiKeyState,
    enable_request_logging: bool,
) -> Router {
    let accepting = app_state.accepting.clone();
    let router = Router::new()
        .route("/health", get(health_check))
        .route("/version", get(version))
        .route("/load", get(get_server_load))
        .route("/metrics", get(prometheus_metrics))
        .route("/server_info", get(server_info))
        .route("/ping", get(ping).post(ping))
        .route("/reset_prefix_cache", post(reset_prefix_cache))
        .route("/sleep", post(sleep_engine))
        .route("/wake_up", post(wake_up_engine))
        .route("/is_sleeping", get(is_sleeping))
        .route("/start_profile", post(start_profile))
        .route("/stop_profile", post(stop_profile))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/v1/models", get(models::list_models))
        .route("/v1/models/{model_id}", get(models::retrieve_model))
        .route("/v1/completions", post(completions::create_completion))
        .route(
            "/v1/completions/render",
            post(completions::render_completion),
        )
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route(
            "/v1/chat/completions/render",
            post(chat::render_chat_completion),
        )
        .route("/v1/responses", post(responses::create_response))
        .route("/v1/responses/{response_id}", get(get_response))
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
        .route("/v1/embeddings", post(embeddings::create_embedding))
        .route("/score", post(embeddings::score))
        .route("/v1/score", post(embeddings::score))
        .route("/rerank", post(embeddings::rerank))
        .route("/v1/rerank", post(embeddings::rerank))
        .route("/v2/rerank", post(embeddings::rerank))
        .route("/pooling", post(embeddings::pooling))
        .route("/v1/pooling", post(embeddings::pooling))
        .route("/classify", post(embeddings::classify))
        .route("/v1/classify", post(embeddings::classify))
        .route("/v1/load_lora_adapter", post(load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(unload_lora_adapter))
        .route("/v1/lora_adapters", get(list_lora_adapters))
        .route("/v1/tokenize", post(tokenize::tokenize))
        .route("/v1/detokenize", post(tokenize::detokenize))
        .route("/tokenizer_info", get(tokenize::get_tokenizer_info))
        .route("/v1/batches", post(batch::create_batch))
        .route("/v1/batches/{batch_id}", get(batch::get_batch))
        .route(
            "/v1/batches/{batch_id}/output",
            get(batch::get_batch_output),
        )
        .route("/v1/batches/{batch_id}/cancel", post(batch::cancel_batch))
        .route("/v1/realtime", get(realtime::realtime_ws))
        .layer(DefaultBodyLimit::max(max_body_size))
        .layer(axum::middleware::from_fn_with_state(
            api_key_state,
            middleware::api_key_auth,
        ))
        .layer(axum::middleware::from_fn_with_state(
            rate_limit_state,
            middleware::rate_limit,
        ))
        .layer(axum::middleware::from_fn_with_state(
            accepting,
            middleware::reject_during_restart,
        ))
        .layer(cors)
        .with_state(app_state)
        .nest("/admin", create_admin_router(admin_state));

    // http_logging is the outermost layer so it captures the final status code
    // (set by inner layers). Applied last so it wraps everything.
    if enable_request_logging {
        router.layer(axum::middleware::from_fn(middleware::http_logging))
    } else {
        router
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use candle_core::{Device, Tensor};
    use tower::ServiceExt;
    use vllm_core::{
        engine::{start_engine, EngineConfig, ModelForward},
        kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager},
        scheduler::SchedulerConfig,
        tokenizer::{ChatTemplateEngine, TokenizerWrapper},
    };

    struct MockModel {
        output_token: u32,
        vocab_size: usize,
        device: Device,
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

    fn test_app_state() -> AppState {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: candle_core::DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let model = MockModel {
            output_token: 42,
            vocab_size: 1000,
            device: Device::Cpu,
        };
        let tokenizer = TokenizerWrapper::for_testing(1000);
        let engine_config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 4,
                max_tokens_per_step: 512,
                enable_chunked_prefill: false,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                max_loras_per_batch: 0,
            },
            None,
        )
        .build();
        let handle = start_engine(model, tokenizer, kv_cache_mgr, engine_config);
        let (atomic_handle, _controller) = AtomicEngineHandle::new(handle);

        let api_tokenizer = TokenizerWrapper::for_testing(1000);
        let chat_template = ChatTemplateEngine::new(
            r#"{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#
                .to_string(),
            "".to_string(),
            "".to_string(),
        );

        let accepting = Arc::new(AtomicBool::new(true));
        // 64 blocks * 16 tokens/block = 1024 max model length
        let max_model_len = 64 * 16;
        AppState::new(
            atomic_handle,
            "test-model".to_string(),
            Arc::new(api_tokenizer),
            Some(Arc::new(chat_template)),
            999,
            max_model_len,
            create_tool_call_parser("hermes"),
            None,
            accepting,
            "assistant".to_string(),
            false,
            false,
            20,
            HashMap::new(),
            1,
        )
    }

    #[tokio::test]
    async fn models_endpoint() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "test-model");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["owned_by"], "vllm-rust");
        assert!(json["data"][0]["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn models_endpoint_full_fields() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &json["data"][0];

        // root matches id
        assert_eq!(model["root"], "test-model");
        assert_eq!(model["root"], model["id"]);

        // parent is omitted (None with skip_serializing_if)
        assert!(model.get("parent").is_none());

        // permission array
        let perms = model["permission"].as_array().unwrap();
        assert_eq!(perms.len(), 1);
        let perm = &perms[0];
        assert_eq!(perm["object"], "model_permission");
        assert!(perm["id"].as_str().unwrap().starts_with("modelperm-"));
        assert!(perm["allow_sampling"].as_bool().unwrap());
        assert!(perm["allow_logprobs"].as_bool().unwrap());
        assert!(perm["allow_view"].as_bool().unwrap());
        assert!(perm["allow_fine_tuning"].as_bool().unwrap());
        assert!(perm["allow_create_engine"].as_bool().unwrap());
        assert!(perm["allow_search_indices"].as_bool().unwrap());
        assert_eq!(perm["organization"], "*");
        assert!(perm["group"].is_null());
        assert!(!perm["is_blocking"].as_bool().unwrap());
        assert!(perm["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn completions_non_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert!(json["choices"][0]["text"].as_str().is_some());
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        assert_eq!(json["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completions_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "prompt": "t1 t2",
            "max_tokens": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_completions_non_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn completions_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream"));

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));
    }

    #[tokio::test]
    async fn chat_completions_streaming() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream"));

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));
    }

    #[tokio::test]
    async fn chat_completions_multi_turn() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn chat_completions_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn completions_batch_prompts() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": ["t1 t2", "t3 t4 t5"],
            "max_tokens": 2
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["choices"].as_array().unwrap().len(), 2);
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][1]["index"], 1);
    }

    #[tokio::test]
    async fn completions_with_sampling_params() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2",
            "max_tokens": 3,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completions_token_ids_input() {
        let state = test_app_state();
        let app = create_router(state);

        // Token IDs as prompt input
        let body = serde_json::json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 2
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn completions_usage_tokens_counted() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3 t4",
            "max_tokens": 5
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap();

        assert!(prompt_tokens > 0);
        assert_eq!(completion_tokens, 5);
        assert_eq!(total_tokens, prompt_tokens + completion_tokens);
    }

    #[tokio::test]
    async fn completions_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1",
            "max_tokens": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify OpenAI-compatible response format
        assert!(json["id"].as_str().unwrap().starts_with("cmpl-"));
        assert_eq!(json["object"], "text_completion");
        assert!(json["created"].as_u64().is_some());
        assert_eq!(json["model"], "test-model");
        assert!(json["choices"].is_array());
        assert!(json["usage"].is_object());
    }

    #[tokio::test]
    async fn chat_completions_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify OpenAI-compatible response format
        assert!(json["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(json["object"], "chat.completion");
        assert!(json["created"].as_u64().is_some());
        assert_eq!(json["model"], "test-model");
        assert!(json["choices"].is_array());
        assert!(json["usage"].is_object());
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test]
    async fn chat_completions_with_system_message() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
    }

    #[tokio::test]
    async fn error_response_format() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "nonexistent-model",
            "prompt": "test",
            "max_tokens": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // OpenAI-compatible error format
        assert!(json["error"].is_object());
        assert!(json["error"]["message"].is_string());
        assert!(json["error"]["type"].is_string());
    }

    #[tokio::test]
    async fn chat_completions_streaming_with_stream_options_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        // Should contain data chunks and [DONE]
        assert!(body_str.contains("data: "));
        assert!(body_str.contains("[DONE]"));

        // Should contain a usage chunk with prompt_tokens, completion_tokens, total_tokens
        assert!(body_str.contains("\"usage\""));
        assert!(body_str.contains("\"prompt_tokens\""));
        assert!(body_str.contains("\"completion_tokens\""));
        assert!(body_str.contains("\"total_tokens\""));
    }

    #[tokio::test]
    async fn chat_completions_streaming_without_stream_options_no_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("[DONE]"));
        // Without stream_options, no usage chunk should appear
        // Parse each SSE data line and check none have "usage" key
        let has_usage_chunk = body_str.lines().any(|line| {
            if let Some(data) = line.strip_prefix("data: ") {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                    return json.get("usage").is_some();
                }
            }
            false
        });
        assert!(!has_usage_chunk);
    }

    #[tokio::test]
    async fn completions_streaming_with_stream_options_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("[DONE]"));
        assert!(body_str.contains("\"usage\""));
        assert!(body_str.contains("\"prompt_tokens\""));
    }

    #[tokio::test]
    async fn chat_completions_non_streaming_has_usage() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Non-streaming responses must always include usage
        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap();
        assert!(prompt_tokens > 0);
        assert_eq!(completion_tokens, 3);
        assert_eq!(total_tokens, prompt_tokens + completion_tokens);
    }

    #[tokio::test]
    async fn chat_completions_with_tools() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
            "max_tokens": 3,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "The city name"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify response format is correct
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        // The mock model doesn't produce tool calls, so content should be present
        // In a real scenario with a tool-calling model, tool_calls would be populated
        assert!(json["choices"][0]["message"].is_object());
    }

    // ─── Tokenize / Detokenize integration tests ─────────────────────────

    #[tokio::test]
    async fn tokenize_success() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3"
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let tokens = json["tokens"].as_array().unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(json["count"], 3);
        assert_eq!(json["max_model_len"], 1024);
    }

    #[tokio::test]
    async fn tokenize_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "prompt": "t1 t2"
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn tokenize_empty_prompt() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": ""
        });
        let req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["count"], 0);
        assert!(json["tokens"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn detokenize_success() {
        let state = test_app_state();
        let app = create_router(state.clone());

        // First tokenize to get valid token IDs
        let tok_body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3"
        });
        let tok_req = Request::post("/v1/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&tok_body).unwrap()))
            .unwrap();
        let tok_resp = app.oneshot(tok_req).await.unwrap();
        let tok_body = axum::body::to_bytes(tok_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let tok_json: serde_json::Value = serde_json::from_slice(&tok_body).unwrap();
        let tokens: Vec<u32> = tok_json["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();

        // Now detokenize those IDs
        let app2 = create_router(state);
        let body = serde_json::json!({
            "model": "test-model",
            "tokens": tokens
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app2.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["prompt"], "t1 t2 t3");
    }

    #[tokio::test]
    async fn detokenize_wrong_model() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "wrong-model",
            "tokens": [1, 2, 3]
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn detokenize_empty_tokens() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "tokens": []
        });
        let req = Request::post("/v1/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["prompt"].as_str().is_some());
    }

    #[tokio::test]
    async fn tokenizer_info_returns_metadata() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/tokenizer_info").body(Body::empty()).unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["tokenizer_class"], "PreTrainedTokenizerFast");
        assert!(json["max_chars_per_token"].as_u64().unwrap() > 0);
        assert!(json["model_max_length"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn completions_invalid_temperature() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 3.0
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("temperature"));
    }

    #[tokio::test]
    async fn chat_completions_invalid_frequency_penalty() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "frequency_penalty": 5.0
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("frequency_penalty"));
    }

    // ─── n > 1 for chat completions ─────────────────────────────────

    #[tokio::test]
    async fn chat_completions_n_greater_than_one() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "n": 2
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(choices[0]["message"]["role"], "assistant");
        assert_eq!(choices[1]["message"]["role"], "assistant");

        // Usage should sum completion tokens across all choices
        let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
        assert_eq!(completion_tokens, 6); // 3 tokens * 2 choices
    }

    #[tokio::test]
    async fn chat_completions_n_streaming_error() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "t1 t2"}],
            "max_tokens": 3,
            "n": 2,
            "stream": true
        });
        let req = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"].as_str().unwrap().contains("n > 1"));
    }

    // ─── best_of for completions ────────────────────────────────────

    #[tokio::test]
    async fn completions_best_of_returns_single_choice() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "best_of": 3
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // best_of generates multiple candidates but returns only the best one
        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0]["index"], 0);

        // Logprobs should not be present since user didn't request them
        assert!(choices[0]["logprobs"].is_null());
    }

    #[tokio::test]
    async fn completions_best_of_with_logprobs() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "t1 t2 t3",
            "max_tokens": 3,
            "best_of": 2,
            "logprobs": 1
        });
        let req = Request::post("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);

        // Logprobs should be present since user requested them
        assert!(choices[0]["logprobs"].is_object());
    }

    // ─── CORS tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn cors_preflight_returns_headers() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/models")
            .header("origin", "http://example.com")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let headers = resp.headers();
        assert!(headers.get("access-control-allow-origin").is_some());
        assert!(headers.get("access-control-allow-methods").is_some());
    }

    #[tokio::test]
    async fn cors_regular_request_includes_headers() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/v1/models")
            .header("origin", "http://example.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header")
            .to_str()
            .expect("header should be valid UTF-8");
        // very_permissive() mirrors the request origin or returns "*"
        assert!(
            allow_origin == "*" || allow_origin == "http://example.com",
            "expected wildcard or mirrored origin, got: {allow_origin}"
        );
    }

    #[tokio::test]
    async fn cors_custom_origin_restricts_access() {
        let state = test_app_state();
        let cors = build_cors_layer(&CorsConfig {
            allowed_origins: "http://allowed.example.com".to_string(),
            allowed_methods: "GET,POST".to_string(),
            allowed_headers: "content-type".to_string(),
        });
        let app = create_router_with_cors(state, cors);

        // Preflight from an allowed origin
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/models")
            .header("origin", "http://allowed.example.com")
            .header("access-control-request-method", "GET")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header");
        assert_eq!(allow_origin, "http://allowed.example.com");
    }

    #[tokio::test]
    async fn cors_default_config_is_very_permissive() {
        let config = CorsConfig::default();
        assert_eq!(config.allowed_origins, "*");
        assert_eq!(config.allowed_methods, "GET,POST,OPTIONS");
        assert_eq!(config.allowed_headers, "*");

        // Should produce a layer that allows any origin
        let state = test_app_state();
        let cors = build_cors_layer(&config);
        let app = create_router_with_cors(state, cors);

        let req = Request::get("/v1/models")
            .header("origin", "http://any-origin.example.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("access-control-allow-origin").is_some());
    }

    #[tokio::test]
    async fn cors_build_layer_with_multiple_origins() {
        let config = CorsConfig {
            allowed_origins: "http://a.com, http://b.com".to_string(),
            allowed_methods: "GET,POST".to_string(),
            allowed_headers: "*".to_string(),
        };

        let state = test_app_state();
        let cors = build_cors_layer(&config);
        let app = create_router_with_cors(state, cors);

        // Request from first allowed origin
        let req = Request::get("/v1/models")
            .header("origin", "http://a.com")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("missing access-control-allow-origin header");
        assert_eq!(allow_origin, "http://a.com");
    }

    #[test]
    fn create_tool_call_parser_known_names() {
        let known = [
            "hermes",
            "glm4",
            "glm45",
            "glm47",
            "glm4_moe",
            "json",
            "openai",
            "llama",
            "llama3_json",
            "llama4_json",
            "llama4_pythonic",
            "mistral",
            "deepseek_v3",
            "deepseek_v31",
            "internlm",
            "internlm2",
            "jamba",
            "pythonic",
            "olmo3",
            "granite",
            "granite-20b-fc",
            "kimi_k2",
            "kimi-k2",
            "phi4mini",
            "phi4_mini_json",
            "longcat",
            "xlam",
            "gigachat3",
            "gigachat",
            "functiongemma",
            "function_gemma",
            "hunyuan",
            "hunyuan_a13b",
            "ernie45",
            "ernie_45",
            "ernie-4.5",
            "seed_oss",
            "seed-oss",
            "minimax",
            "minimax_m2",
            "deepseek_v32",
            "step3",
            "step3p5",
            "step-3",
            "step-3.5",
            "qwen3_xml",
            "qwen3xml",
            "qwen3coder",
            "qwen3_coder",
        ];
        for name in &known {
            let parser = create_tool_call_parser(name);
            // Parser should be able to handle empty input without panicking
            let result = parser.parse("");
            assert!(result.is_ok(), "parser '{name}' failed on empty input");
        }
    }

    #[test]
    fn create_tool_call_parser_unknown_defaults_to_hermes() {
        let parser = create_tool_call_parser("nonexistent");
        // Should still work (defaults to hermes)
        let result = parser.parse(r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    // ─── Health and version endpoints ────────────────────────────────

    #[tokio::test]
    async fn health_returns_200_when_accepting() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_returns_503_when_not_accepting() {
        let state = test_app_state();
        state.accepting.store(false, Ordering::SeqCst);
        let router = create_router(state);
        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn version_returns_json() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .uri("/version")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["version"].as_str().is_some());
    }

    #[tokio::test]
    async fn server_info_returns_model_metadata() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .uri("/server_info")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["version"].as_str().is_some());
        assert!(json["model_id"].as_str().is_some());
        assert!(json["max_model_len"].as_u64().is_some());
        assert_eq!(json["accepting_requests"], true);
    }

    #[tokio::test]
    async fn load_returns_zero_initially() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder().uri("/load").body(Body::empty()).unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["server_load"], 0);
    }

    #[tokio::test]
    async fn server_load_increment_decrement() {
        let state = test_app_state();
        assert_eq!(state.current_load(), 0);
        state.increment_load();
        state.increment_load();
        assert_eq!(state.current_load(), 2);
        state.decrement_load();
        assert_eq!(state.current_load(), 1);
    }

    #[tokio::test]
    async fn load_lora_adapter_succeeds() {
        let state = test_app_state();
        let router = create_router(state);
        let body = serde_json::json!({
            "lora_name": "my-adapter",
            "lora_path": "/path/to/adapter"
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/load_lora_adapter")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("my-adapter"));
        assert!(text.contains("added successfully"));
    }

    #[tokio::test]
    async fn load_lora_adapter_rejects_duplicate() {
        let state = test_app_state();
        // Pre-populate an adapter
        state.lora_adapters.write().await.insert(
            "existing".into(),
            LoraRequest::new("existing".to_string(), 1, "/p".to_string()),
        );
        let router = create_router(state);
        let body = serde_json::json!({
            "lora_name": "existing",
            "lora_path": "/other"
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/load_lora_adapter")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unload_lora_adapter_succeeds() {
        let state = test_app_state();
        state.lora_adapters.write().await.insert(
            "my-adapter".into(),
            LoraRequest::new("my-adapter".to_string(), 1, "/p".to_string()),
        );
        let router = create_router(state);
        let body = serde_json::json!({ "lora_name": "my-adapter" });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/unload_lora_adapter")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unload_lora_adapter_not_found() {
        let state = test_app_state();
        let router = create_router(state);
        let body = serde_json::json!({ "lora_name": "nonexistent" });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/unload_lora_adapter")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn list_lora_adapters_empty() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .method("GET")
            .uri("/v1/lora_adapters")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["lora_adapters"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn list_lora_adapters_returns_loaded() {
        let state = test_app_state();
        state.lora_adapters.write().await.insert(
            "adapter-a".into(),
            LoraRequest::new("adapter-a", 1, "/path/a"),
        );
        state.lora_adapters.write().await.insert(
            "adapter-b".into(),
            LoraRequest::new("adapter-b", 2, "/path/b"),
        );
        let router = create_router(state);
        let request = Request::builder()
            .method("GET")
            .uri("/v1/lora_adapters")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let adapters = json["lora_adapters"].as_array().unwrap();
        assert_eq!(adapters.len(), 2);
        // Verify adapter data is present (order may vary due to HashMap)
        let names: Vec<&str> = adapters
            .iter()
            .map(|a| a["lora_name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"adapter-a"));
        assert!(names.contains(&"adapter-b"));
    }

    #[tokio::test]
    async fn get_response_retrieves_stored_response() {
        let state = test_app_state();
        // Pre-populate the response store
        let resp = responses_types::ResponsesResponse {
            id: "resp_test123456789012345678".to_string(),
            object: "response",
            created_at: 100,
            model: "test-model".to_string(),
            status: responses_types::ResponseStatus::Completed,
            output: vec![],
            usage: Some(responses_types::ResponseUsage {
                input_tokens: 5,
                output_tokens: 10,
                total_tokens: 15,
            }),
            incomplete_details: None,
            metadata: None,
        };
        state
            .response_store
            .write()
            .await
            .insert(resp.id.clone(), resp);

        let router = create_router(state);
        let request = Request::builder()
            .uri("/v1/responses/resp_test123456789012345678")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "resp_test123456789012345678");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[tokio::test]
    async fn get_response_not_found() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .uri("/v1/responses/resp_nonexistent")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn cancel_response_completed_stays_completed() {
        let state = test_app_state();
        let resp = responses_types::ResponsesResponse {
            id: "resp_cancel_test_completed1".to_string(),
            object: "response",
            created_at: 100,
            model: "test-model".to_string(),
            status: responses_types::ResponseStatus::Completed,
            output: vec![],
            usage: None,
            incomplete_details: None,
            metadata: None,
        };
        state
            .response_store
            .write()
            .await
            .insert(resp.id.clone(), resp);

        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses/resp_cancel_test_completed1/cancel")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Already completed — status should remain "completed"
        assert_eq!(json["status"], "completed");
    }

    #[tokio::test]
    async fn cancel_response_in_progress_becomes_cancelled() {
        let state = test_app_state();
        let resp = responses_types::ResponsesResponse {
            id: "resp_cancel_inprogress_01".to_string(),
            object: "response",
            created_at: 100,
            model: "test-model".to_string(),
            status: responses_types::ResponseStatus::InProgress,
            output: vec![],
            usage: None,
            incomplete_details: None,
            metadata: None,
        };
        state
            .response_store
            .write()
            .await
            .insert(resp.id.clone(), resp);

        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses/resp_cancel_inprogress_01/cancel")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "cancelled");
    }

    #[tokio::test]
    async fn cancel_response_not_found() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .method("POST")
            .uri("/v1/responses/resp_nonexistent/cancel")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn metrics_returns_prometheus_format() {
        let state = test_app_state();
        let router = create_router(state);
        let request = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/plain"));
    }

    #[tokio::test]
    async fn ping_get_returns_ok() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/ping").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn ping_post_returns_ok() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::post("/ping").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tokenize_root_alias_exists() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "prompt": "hello world"
        });
        let req = Request::post("/tokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn detokenize_root_alias_exists() {
        let state = test_app_state();
        let app = create_router(state);

        let body = serde_json::json!({
            "model": "test-model",
            "tokens": [1, 2, 3]
        });
        let req = Request::post("/detokenize")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn reset_prefix_cache_returns_success() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::post("/reset_prefix_cache")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["success"], true);
    }

    #[tokio::test]
    async fn sleep_and_wake_endpoints() {
        let state = test_app_state();
        let app = create_router(state);

        // Sleep the engine
        let req = Request::post("/sleep").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Check it's sleeping
        let req = Request::get("/is_sleeping").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["is_sleeping"], true);

        // Wake up
        let req = Request::post("/wake_up").body(Body::empty()).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Check it's awake
        let req = Request::get("/is_sleeping").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["is_sleeping"], false);
    }

    #[tokio::test]
    async fn is_sleeping_default_false() {
        let state = test_app_state();
        let app = create_router(state);

        let req = Request::get("/is_sleeping").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["is_sleeping"], false);
    }
}
