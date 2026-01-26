use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

use crate::config::ModelConfig;
use crate::quantization::{
    create_weight_loader_with_params, detect_from_directory, detect_from_json, DetectedQuantConfig,
    QuantizationMethod, QuantizedWeightLoader,
};

pub struct ModelFiles {
    pub config: ModelConfig,
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    /// Detected quantization configuration (if any).
    pub quantization: DetectedQuantConfig,
}

/// Downloads model files from HuggingFace Hub (or uses cache).
pub fn fetch_model(model_id: &str) -> anyhow::Result<ModelFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_content)?;

    // Detect quantization from config.json
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
    let mut quantization = detect_from_json(&config_json);

    // If no quantization detected in config.json, try quantize_config.json
    if quantization.method == QuantizationMethod::None {
        if let Ok(quant_config_path) = repo.get("quantize_config.json") {
            if let Ok(model_dir) = quant_config_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))
            {
                quantization = detect_from_directory(model_dir);
            }
        }
    }

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

    let weights = load_safetensor_paths(&repo)?;

    Ok(ModelFiles {
        config,
        weights,
        tokenizer: tokenizer_path,
        tokenizer_config: tokenizer_config_path,
        quantization,
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

/// Creates a quantized weight loader from model files.
///
/// This function creates the appropriate weight loader based on the detected
/// quantization method.
///
/// # Arguments
/// * `files` - Model files from `fetch_model`
/// * `dtype` - Data type for compute (activations)
/// * `device` - Device to load weights to
///
/// # Returns
/// A boxed `QuantizedWeightLoader` implementation
pub fn create_quantized_loader(
    files: &ModelFiles,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Box<dyn QuantizedWeightLoader>> {
    let vb = load_weights(&files.weights, dtype, device)?;
    Ok(create_weight_loader_with_params(vb, &files.quantization))
}

/// Returns true if the model uses quantization.
pub fn is_quantized(files: &ModelFiles) -> bool {
    files.quantization.method != QuantizationMethod::None
}

/// Get quantization info string for logging.
pub fn quantization_info(files: &ModelFiles) -> String {
    match files.quantization.method {
        QuantizationMethod::None => "None (full precision)".to_string(),
        QuantizationMethod::Fp8 => {
            let scheme = files
                .quantization
                .activation_scheme
                .as_deref()
                .unwrap_or("dynamic");
            format!("FP8 (activation scheme: {})", scheme)
        }
        QuantizationMethod::Gptq => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("GPTQ (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::Awq => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("AWQ (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::Gguf => "GGUF".to_string(),
        QuantizationMethod::BitsAndBytes => "BitsAndBytes".to_string(),
        QuantizationMethod::SqueezeLlm => "SqueezeLLM".to_string(),
        QuantizationMethod::Marlin => "Marlin".to_string(),
    }
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
        // Qwen3-0.6B is not quantized
        assert_eq!(files.quantization.method, QuantizationMethod::None);
    }

    #[test]
    fn test_quantization_info_none() {
        let files = ModelFiles {
            config: ModelConfig::default(),
            weights: vec![],
            tokenizer: PathBuf::new(),
            tokenizer_config: None,
            quantization: DetectedQuantConfig::default(),
        };
        assert_eq!(quantization_info(&files), "None (full precision)");
        assert!(!is_quantized(&files));
    }

    #[test]
    fn test_quantization_info_gptq() {
        let files = ModelFiles {
            config: ModelConfig::default(),
            weights: vec![],
            tokenizer: PathBuf::new(),
            tokenizer_config: None,
            quantization: DetectedQuantConfig {
                method: QuantizationMethod::Gptq,
                bits: Some(4),
                group_size: Some(128),
                desc_act: None,
                activation_scheme: None,
                raw_config: Default::default(),
            },
        };
        assert_eq!(quantization_info(&files), "GPTQ (bits: 4, group_size: 128)");
        assert!(is_quantized(&files));
    }

    #[test]
    fn test_quantization_info_fp8() {
        let files = ModelFiles {
            config: ModelConfig::default(),
            weights: vec![],
            tokenizer: PathBuf::new(),
            tokenizer_config: None,
            quantization: DetectedQuantConfig {
                method: QuantizationMethod::Fp8,
                bits: Some(8),
                group_size: None,
                desc_act: None,
                activation_scheme: Some("static".to_string()),
                raw_config: Default::default(),
            },
        };
        assert_eq!(quantization_info(&files), "FP8 (activation scheme: static)");
        assert!(is_quantized(&files));
    }

    #[test]
    fn test_quantization_info_awq() {
        let files = ModelFiles {
            config: ModelConfig::default(),
            weights: vec![],
            tokenizer: PathBuf::new(),
            tokenizer_config: None,
            quantization: DetectedQuantConfig {
                method: QuantizationMethod::Awq,
                bits: Some(4),
                group_size: Some(128),
                desc_act: None,
                activation_scheme: None,
                raw_config: Default::default(),
            },
        };
        assert_eq!(quantization_info(&files), "AWQ (bits: 4, group_size: 128)");
        assert!(is_quantized(&files));
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
