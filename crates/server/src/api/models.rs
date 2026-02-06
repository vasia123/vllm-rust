use axum::extract::State;
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
}
