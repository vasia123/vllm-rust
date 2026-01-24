use axum::extract::State;
use axum::Json;

use super::types::{ModelObject, ModelsResponse};
use super::AppState;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model",
            created: 0,
            owned_by: "local".to_string(),
        }],
    })
}
