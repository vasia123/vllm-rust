//! Model estimation and GPU info endpoints.
//!
//! These endpoints do NOT require a running engine — they work with
//! GPU hardware specs and model config.json only.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};

use vllm_core::perf_estimate::gpu_profile::{detect_gpu_profile, GpuHardwareProfile};
use vllm_core::perf_estimate::model_profile::ModelProfile;
use vllm_core::perf_estimate::roofline::{self, EstimationConfig};
use vllm_core::perf_estimate::vram_fitness::{self, VramFitness};

use crate::api::error::ApiError;

/// State for estimation endpoints (no running engine required).
#[derive(Clone)]
pub struct EstimateState {
    /// Cached GPU profile (detected once on startup).
    gpu_profile: Arc<Mutex<Option<GpuHardwareProfile>>>,
    /// Download progress broadcast channel.
    download_tx: broadcast::Sender<DownloadProgressEvent>,
    /// Currently downloading model (if any).
    downloading: Arc<Mutex<Option<String>>>,
}

impl Default for EstimateState {
    fn default() -> Self {
        Self::new()
    }
}

impl EstimateState {
    pub fn new() -> Self {
        let (download_tx, _) = broadcast::channel(64);
        Self {
            gpu_profile: Arc::new(Mutex::new(None)),
            download_tx,
            downloading: Arc::new(Mutex::new(None)),
        }
    }

    async fn get_gpu_profile(&self) -> Result<GpuHardwareProfile, ApiError> {
        let mut cached = self.gpu_profile.lock().await;
        if let Some(ref profile) = *cached {
            return Ok(profile.clone());
        }
        let profile = detect_gpu_profile()
            .map_err(|e| ApiError::InternalError(format!("GPU detection failed: {e}")))?;
        *cached = Some(profile.clone());
        Ok(profile)
    }
}

// ─── GET /admin/gpu/info ─────────────────────────────────────────────────────

pub async fn gpu_info(State(state): State<EstimateState>) -> Result<impl IntoResponse, ApiError> {
    let profile = state.get_gpu_profile().await?;
    Ok(Json(profile))
}

// ─── POST /admin/models/estimate ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct EstimateRequest {
    pub model_id: String,
    #[serde(default = "default_revision")]
    pub revision: String,
    #[serde(flatten)]
    pub config: EstimationConfig,
}

fn default_revision() -> String {
    "main".to_string()
}

pub async fn estimate_performance(
    State(state): State<EstimateState>,
    Json(req): Json<EstimateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let gpu = state.get_gpu_profile().await?;

    // Fetch config.json (blocking I/O — run in spawn_blocking)
    let model_id = req.model_id.clone();
    let revision = req.revision.clone();
    let model_config = tokio::task::spawn_blocking(move || {
        vllm_core::loader::fetch_model_config_only(&model_id, &revision)
    })
    .await
    .map_err(|e| ApiError::InternalError(format!("Task join error: {e}")))?
    .map_err(|e| ApiError::InvalidRequest(format!("Failed to fetch model config: {e}")))?;

    let profile = ModelProfile::from_config(&req.model_id, &model_config, req.config.weight_dtype);
    let estimate = roofline::estimate(&gpu, &profile, &req.config);

    Ok(Json(estimate))
}

// ─── GET /admin/models/search ────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_fits_only")]
    pub fits_only: bool,
}

fn default_limit() -> usize {
    50
}
fn default_fits_only() -> bool {
    true
}

/// HuggingFace model search result. Uses `deny_unknown_fields` = false (default)
/// to tolerate extra fields from the HF API.
#[derive(Serialize, Deserialize, Debug)]
pub struct HfModelSearchResult {
    pub id: String,
    #[serde(default, alias = "modelId")]
    pub model_id: Option<String>,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub safetensors: Option<SafetensorsInfo>,
    /// Ignore any other fields from HF API.
    #[serde(flatten)]
    pub _extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SafetensorsInfo {
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub total: Option<u64>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub total_params: Option<u64>,
    pub vram_fitness: Option<VramFitness>,
}

fn extract_total_params(model: &HfModelSearchResult) -> Option<u64> {
    if let Some(ref st) = model.safetensors {
        if let Some(total) = st.total {
            return Some(total);
        }
        if let Some(ref params) = st.parameters {
            // Sometimes it's { "F32": 123456, "BF16": 789012, ... }
            // Sum all values or take the largest
            if let Some(obj) = params.as_object() {
                return Some(obj.values().filter_map(|v| v.as_u64()).sum());
            }
            if let Some(n) = params.as_u64() {
                return Some(n);
            }
        }
    }
    None
}

pub async fn search_models(
    State(state): State<EstimateState>,
    Query(query): Query<SearchQuery>,
) -> Result<impl IntoResponse, ApiError> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=text-generation&sort=downloads&limit={}&expand[]=safetensors&expand[]=likes&expand[]=tags",
        urlencoding::encode(&query.q),
        query.limit
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| ApiError::InternalError(format!("HTTP client error: {e}")))?;

    let response = client
        .get(&url)
        .header("User-Agent", "vllm-rust/0.1")
        .send()
        .await
        .map_err(|e| ApiError::InternalError(format!("HuggingFace API error: {e}")))?;

    if !response.status().is_success() {
        return Err(ApiError::InternalError(format!(
            "HuggingFace API returned {}",
            response.status()
        )));
    }

    let models: Vec<HfModelSearchResult> = response
        .json()
        .await
        .map_err(|e| ApiError::InternalError(format!("Failed to parse HF response: {e}")))?;

    // Get GPU profile for VRAM fitness check
    let gpu_profile = state.get_gpu_profile().await.ok();
    let gpu_vram = gpu_profile
        .as_ref()
        .map(|g| g.total_vram_bytes)
        .unwrap_or(0);

    let results: Vec<SearchResult> = models
        .into_iter()
        .filter_map(|model| {
            let total_params = extract_total_params(&model);

            let vram_fitness = if gpu_vram > 0 {
                total_params
                    .map(|params| vram_fitness::quick_vram_check(&model.id, params, gpu_vram, 0.9))
            } else {
                None
            };

            // Filter out models that don't fit if fits_only=true
            if query.fits_only {
                if let Some(ref fitness) = vram_fitness {
                    if !fitness.any_fits {
                        return None;
                    }
                }
            }

            Some(SearchResult {
                id: model.id,
                downloads: model.downloads,
                likes: model.likes,
                pipeline_tag: model.pipeline_tag,
                tags: model.tags,
                total_params,
                vram_fitness,
            })
        })
        .collect();

    Ok(Json(results))
}

// ─── POST /admin/models/download ─────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DownloadRequest {
    pub model_id: String,
    #[serde(default = "default_revision")]
    pub revision: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct DownloadProgressEvent {
    pub status: String,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Serialize)]
pub struct DownloadResponse {
    pub status: String,
    pub model_id: String,
}

pub async fn start_download(
    State(state): State<EstimateState>,
    Json(req): Json<DownloadRequest>,
) -> Result<impl IntoResponse, ApiError> {
    {
        let mut dl = state.downloading.lock().await;
        if dl.is_some() {
            return Err(ApiError::InvalidRequest(
                "Another download is already in progress".to_string(),
            ));
        }
        *dl = Some(req.model_id.clone());
    }

    let model_id = req.model_id.clone();
    let revision = req.revision.clone();
    let tx = state.download_tx.clone();
    let downloading = state.downloading.clone();

    tokio::task::spawn_blocking(move || {
        let _ = tx.send(DownloadProgressEvent {
            status: "downloading".to_string(),
            model_id: model_id.clone(),
            file: None,
            message: Some("Starting download...".to_string()),
        });

        let result = vllm_core::loader::fetch_model_with_options(
            &model_id,
            &revision,
            None,
            None,
            vllm_core::loader::LoadFormat::Auto,
            4,
        );

        match result {
            Ok(_) => {
                let _ = tx.send(DownloadProgressEvent {
                    status: "complete".to_string(),
                    model_id: model_id.clone(),
                    file: None,
                    message: Some("Download complete".to_string()),
                });
            }
            Err(e) => {
                let _ = tx.send(DownloadProgressEvent {
                    status: "error".to_string(),
                    model_id: model_id.clone(),
                    file: None,
                    message: Some(format!("Download failed: {e}")),
                });
            }
        }

        // Clear downloading state
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let mut dl = downloading.lock().await;
            *dl = None;
        });
    });

    Ok(Json(DownloadResponse {
        status: "started".to_string(),
        model_id: req.model_id,
    }))
}

// ─── GET /admin/models/download/progress ─────────────────────────────────────

pub async fn download_progress(
    State(state): State<EstimateState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut rx = state.download_tx.subscribe();

    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if let Ok(data) = serde_json::to_string(&event) {
                        yield Ok(Event::default().event("progress").data(data));
                    }
                    if event.status == "complete" || event.status == "error" {
                        break;
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

    #[test]
    fn test_extract_total_params_from_total() {
        let model = HfModelSearchResult {
            id: "test/model".to_string(),
            model_id: None,
            downloads: 100,
            likes: 10,
            pipeline_tag: Some("text-generation".to_string()),
            tags: vec![],
            safetensors: Some(SafetensorsInfo {
                parameters: None,
                total: Some(8_000_000_000),
            }),
            _extra: Default::default(),
        };
        assert_eq!(extract_total_params(&model), Some(8_000_000_000));
    }

    #[test]
    fn test_extract_total_params_from_parameters_map() {
        let model = HfModelSearchResult {
            id: "test/model".to_string(),
            model_id: None,
            downloads: 100,
            likes: 10,
            pipeline_tag: None,
            tags: vec![],
            safetensors: Some(SafetensorsInfo {
                parameters: Some(serde_json::json!({"BF16": 8000000000_u64})),
                total: None,
            }),
            _extra: Default::default(),
        };
        assert_eq!(extract_total_params(&model), Some(8_000_000_000));
    }

    #[test]
    fn test_extract_total_params_none() {
        let model = HfModelSearchResult {
            id: "test/model".to_string(),
            model_id: None,
            downloads: 100,
            likes: 10,
            pipeline_tag: None,
            tags: vec![],
            safetensors: None,
            _extra: Default::default(),
        };
        assert_eq!(extract_total_params(&model), None);
    }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            id: "meta-llama/Llama-3.1-8B".to_string(),
            downloads: 1_000_000,
            likes: 5000,
            pipeline_tag: Some("text-generation".to_string()),
            tags: vec!["llama".to_string()],
            total_params: Some(8_000_000_000),
            vram_fitness: Some(vram_fitness::quick_vram_check(
                "test",
                8_000_000_000,
                24 * 1024 * 1024 * 1024,
                0.9,
            )),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("vram_fitness"));
        assert!(json.contains("recommended_dtype"));
    }

    #[test]
    fn test_estimate_state_creation() {
        let state = EstimateState::new();
        // Should create without panicking
        let _ = state.download_tx.subscribe();
    }

    #[test]
    fn test_download_progress_serialization() {
        let event = DownloadProgressEvent {
            status: "downloading".to_string(),
            model_id: "test/model".to_string(),
            file: Some("model-00001.safetensors".to_string()),
            message: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("downloading"));
        assert!(!json.contains("message")); // None should be skipped
    }
}
