use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

use crate::config::ModelConfig;

pub struct ModelFiles {
    pub config: ModelConfig,
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
}

/// Downloads model files from HuggingFace Hub (or uses cache).
pub fn fetch_model(model_id: &str) -> anyhow::Result<ModelFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

    let weights = load_safetensor_paths(&repo)?;

    Ok(ModelFiles {
        config,
        weights,
        tokenizer: tokenizer_path,
        tokenizer_config: tokenizer_config_path,
    })
}

/// Creates a VarBuilder from safetensor weight files.
pub fn load_weights(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    Ok(vb)
}

fn load_safetensor_paths(repo: &hf_hub::api::sync::ApiRepo) -> anyhow::Result<Vec<PathBuf>> {
    // Try model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    // Multi-file: read model.safetensors.index.json
    let index_path = repo.get("model.safetensors.index.json")?;
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;

    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

    let mut filenames: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    filenames.sort();
    filenames.dedup();

    let mut paths = Vec::new();
    for filename in &filenames {
        paths.push(repo.get(filename)?);
    }

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires network + disk space
    fn fetch_qwen3_06b() {
        let files = fetch_model("Qwen/Qwen3-0.6B").expect("failed to fetch model");

        assert_eq!(files.config.hidden_size, 1024);
        assert_eq!(files.config.num_hidden_layers, 28);
        assert!(!files.weights.is_empty());
        assert!(files.tokenizer.exists());
    }

    #[test]
    #[ignore] // requires network + disk space
    fn load_qwen3_06b_weights() {
        let files = fetch_model("Qwen/Qwen3-0.6B").expect("failed to fetch model");
        let device = Device::Cpu;
        let vb = load_weights(&files.weights, DType::F32, &device).expect("failed to load weights");

        // Verify embedding tensor shape
        let embed = vb
            .get(
                (files.config.vocab_size, files.config.hidden_size),
                "model.embed_tokens.weight",
            )
            .expect("missing embed_tokens");
        assert_eq!(
            embed.dims(),
            &[files.config.vocab_size, files.config.hidden_size]
        );

        // Verify first layer attention Q projection shape
        let q_weight = vb
            .get(
                (
                    files.config.num_attention_heads * files.config.head_dim,
                    files.config.hidden_size,
                ),
                "model.layers.0.self_attn.q_proj.weight",
            )
            .expect("missing q_proj weight");
        assert_eq!(
            q_weight.dims(),
            &[
                files.config.num_attention_heads * files.config.head_dim,
                files.config.hidden_size
            ]
        );

        // Verify Qwen3-specific: Q norm weight exists
        let q_norm = vb
            .get(
                (files.config.head_dim,),
                "model.layers.0.self_attn.q_norm.weight",
            )
            .expect("missing q_norm (Qwen3-specific)");
        assert_eq!(q_norm.dims(), &[files.config.head_dim]);
    }
}
