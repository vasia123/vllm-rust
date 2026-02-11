//! Prompt adapter types and data structures.

use candle_core::Tensor;
use serde::Deserialize;

/// Request for a specific prompt adapter.
///
/// Passed with generation requests to specify which adapter to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PromptAdapterRequest {
    /// Human-readable name for the adapter.
    pub name: String,
    /// Globally unique integer ID (must be > 0).
    pub id: u32,
    /// Path to the adapter files.
    pub path: String,
    /// Number of virtual tokens this adapter prepends.
    pub num_virtual_tokens: usize,
}

impl PromptAdapterRequest {
    pub fn new(
        name: impl Into<String>,
        id: u32,
        path: impl Into<String>,
        num_virtual_tokens: usize,
    ) -> Self {
        Self {
            name: name.into(),
            id,
            path: path.into(),
            num_virtual_tokens,
        }
    }

    /// Reference an already-loaded adapter by name.
    pub fn by_name(name: impl Into<String>, num_virtual_tokens: usize) -> Self {
        Self {
            name: name.into(),
            id: 0,
            path: String::new(),
            num_virtual_tokens,
        }
    }
}

/// Configuration from adapter_config.json (PEFT prompt tuning format).
#[derive(Debug, Clone, Deserialize)]
pub struct PromptAdapterConfig {
    /// Number of virtual tokens to prepend.
    pub num_virtual_tokens: usize,
    /// PEFT method type (should be "PROMPT_TUNING" or "P_TUNING").
    #[serde(default)]
    pub peft_type: Option<String>,
    /// Task type this adapter was trained for.
    #[serde(default)]
    pub task_type: Option<String>,
    /// Base model name/path for validation.
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
    /// Token dimension (embedding_dim). May be absent if inferred from weights.
    #[serde(default)]
    pub token_dim: Option<usize>,
}

/// A loaded prompt adapter with its embedding tensor.
///
/// The `prompt_embeddings` tensor has shape `[num_virtual_tokens, embedding_dim]`.
/// During prefill, position `i` (for `i < num_virtual_tokens`) uses
/// `prompt_embeddings[i]` instead of the base model's token embedding.
#[derive(Debug, Clone)]
pub struct PromptAdapter {
    /// Adapter name.
    pub name: String,
    /// Unique ID.
    pub id: u32,
    /// Number of virtual tokens.
    pub num_virtual_tokens: usize,
    /// Learned prompt embeddings: `[num_virtual_tokens, embedding_dim]`.
    pub prompt_embeddings: Tensor,
}

impl PromptAdapter {
    pub fn new(
        name: impl Into<String>,
        id: u32,
        num_virtual_tokens: usize,
        prompt_embeddings: Tensor,
    ) -> Self {
        Self {
            name: name.into(),
            id,
            num_virtual_tokens,
            prompt_embeddings,
        }
    }

    /// Embedding dimension (from tensor shape).
    pub fn embedding_dim(&self) -> usize {
        self.prompt_embeddings.dims().last().copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn prompt_adapter_request_new() {
        let req = PromptAdapterRequest::new("task-adapter", 1, "/path/to/adapter", 20);
        assert_eq!(req.name, "task-adapter");
        assert_eq!(req.id, 1);
        assert_eq!(req.path, "/path/to/adapter");
        assert_eq!(req.num_virtual_tokens, 20);
    }

    #[test]
    fn prompt_adapter_request_by_name() {
        let req = PromptAdapterRequest::by_name("task-adapter", 20);
        assert_eq!(req.name, "task-adapter");
        assert_eq!(req.id, 0);
        assert_eq!(req.num_virtual_tokens, 20);
    }

    #[test]
    fn prompt_adapter_dimensions() {
        let device = Device::Cpu;
        let num_virtual_tokens = 20;
        let embed_dim = 768;

        let embeddings =
            Tensor::zeros((num_virtual_tokens, embed_dim), DType::F32, &device).unwrap();
        let adapter = PromptAdapter::new("test", 1, num_virtual_tokens, embeddings);

        assert_eq!(adapter.num_virtual_tokens, 20);
        assert_eq!(adapter.embedding_dim(), 768);
    }

    #[test]
    fn prompt_adapter_config_deserialize() {
        let json = r#"{
            "num_virtual_tokens": 20,
            "peft_type": "PROMPT_TUNING",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "meta-llama/Llama-2-7b-hf"
        }"#;

        let config: PromptAdapterConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_virtual_tokens, 20);
        assert_eq!(config.peft_type.as_deref(), Some("PROMPT_TUNING"));
        assert_eq!(config.task_type.as_deref(), Some("CAUSAL_LM"));
    }

    #[test]
    fn prompt_adapter_config_minimal() {
        let json = r#"{ "num_virtual_tokens": 10 }"#;
        let config: PromptAdapterConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_virtual_tokens, 10);
        assert!(config.peft_type.is_none());
    }
}
