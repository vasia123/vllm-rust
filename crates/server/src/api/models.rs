use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use super::types::{timestamp_now, ModelObject, ModelPermission, ModelsResponse};
use super::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let created = timestamp_now();
    let model_id = state.model_id.clone();
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: model_id.clone(),
            object: "model",
            created,
            owned_by: "vllm-rust".to_string(),
            root: model_id,
            parent: None,
            permission: vec![ModelPermission {
                id: format!("modelperm-{}", uuid::Uuid::new_v4()),
                object: "model_permission",
                created,
                allow_create_engine: true,
                allow_sampling: true,
                allow_logprobs: true,
                allow_search_indices: true,
                allow_view: true,
                allow_fine_tuning: true,
                organization: "*".to_string(),
                group: None,
                is_blocking: false,
            }],
        }],
    })
}

/// GET /v1/models/{model_id} — retrieve a single model by ID.
///
/// Returns the model info if the requested ID matches the loaded model,
/// otherwise returns 404.
pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    if model_id == state.model_id {
        let created = timestamp_now();
        let resp = ModelObject {
            id: model_id.clone(),
            object: "model",
            created,
            owned_by: "vllm-rust".to_string(),
            root: model_id,
            parent: None,
            permission: vec![ModelPermission {
                id: format!("modelperm-{}", uuid::Uuid::new_v4()),
                object: "model_permission",
                created,
                allow_create_engine: true,
                allow_sampling: true,
                allow_logprobs: true,
                allow_search_indices: true,
                allow_view: true,
                allow_fine_tuning: true,
                organization: "*".to_string(),
                group: None,
                is_blocking: false,
            }],
        };
        (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
    } else {
        // Check LoRA adapters
        let adapters = state.lora_adapters.read().await;
        if adapters.contains_key(&model_id) {
            let created = timestamp_now();
            let resp = ModelObject {
                id: model_id.clone(),
                object: "model",
                created,
                owned_by: "vllm-rust".to_string(),
                root: state.model_id.clone(),
                parent: Some(state.model_id.clone()),
                permission: vec![ModelPermission {
                    id: format!("modelperm-{}", uuid::Uuid::new_v4()),
                    object: "model_permission",
                    created,
                    allow_create_engine: true,
                    allow_sampling: true,
                    allow_logprobs: true,
                    allow_search_indices: true,
                    allow_view: true,
                    allow_fine_tuning: true,
                    organization: "*".to_string(),
                    group: None,
                    is_blocking: false,
                }],
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        } else {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("The model '{model_id}' does not exist"),
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::types::{ModelObject, ModelPermission};

    fn make_test_model(model_id: &str) -> ModelObject {
        let created = 1_700_000_000;
        ModelObject {
            id: model_id.to_string(),
            object: "model",
            created,
            owned_by: "vllm-rust".to_string(),
            root: model_id.to_string(),
            parent: None,
            permission: vec![ModelPermission {
                id: format!("modelperm-{}", uuid::Uuid::new_v4()),
                object: "model_permission",
                created,
                allow_create_engine: true,
                allow_sampling: true,
                allow_logprobs: true,
                allow_search_indices: true,
                allow_view: true,
                allow_fine_tuning: true,
                organization: "*".to_string(),
                group: None,
                is_blocking: false,
            }],
        }
    }

    #[test]
    fn list_models_response_includes_permission_array() {
        let model = make_test_model("meta-llama/Llama-3-8B");
        let json = serde_json::to_value(&model).unwrap();

        let perms = json["permission"].as_array().unwrap();
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
    }

    #[test]
    fn root_field_matches_model_id() {
        let model = make_test_model("meta-llama/Llama-3-8B");
        let json = serde_json::to_value(&model).unwrap();

        assert_eq!(json["root"], "meta-llama/Llama-3-8B");
        assert_eq!(json["root"], json["id"]);
    }

    #[test]
    fn parent_is_null_for_base_model() {
        let model = make_test_model("meta-llama/Llama-3-8B");
        let json = serde_json::to_value(&model).unwrap();

        // parent is None, so skip_serializing_if omits it
        assert!(json.get("parent").is_none());
    }

    #[test]
    fn lora_model_has_parent() {
        let model = ModelObject {
            id: "my-lora-adapter".to_string(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "vllm-rust".to_string(),
            root: "meta-llama/Llama-3-8B".to_string(),
            parent: Some("meta-llama/Llama-3-8B".to_string()),
            permission: vec![],
        };
        let json = serde_json::to_value(&model).unwrap();

        assert_eq!(json["parent"], "meta-llama/Llama-3-8B");
        assert_eq!(json["root"], "meta-llama/Llama-3-8B");
        assert_ne!(json["id"], json["root"]);
    }

    #[test]
    fn model_not_found_error_format() {
        let error = serde_json::json!({
            "error": {
                "message": "The model 'nonexistent' does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        });
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(error["error"]["code"], "model_not_found");
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nonexistent"));
    }
}
