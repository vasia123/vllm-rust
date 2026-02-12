//! Prompt adapter loading from PEFT checkpoints.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use thiserror::Error;

use super::types::{PromptAdapter, PromptAdapterConfig};

/// Errors from loading prompt adapters.
#[derive(Debug, Error)]
pub enum PromptAdapterLoadError {
    #[error("config error: {0}")]
    Config(String),
    #[error("weights error: {0}")]
    Weights(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shape mismatch: expected [{expected_vt}, {expected_dim}], got {actual:?}")]
    ShapeMismatch {
        expected_vt: usize,
        expected_dim: usize,
        actual: Vec<usize>,
    },
}

/// Loads prompt adapters from PEFT checkpoint directories.
///
/// Expected directory structure:
/// ```text
/// adapter_dir/
///   adapter_config.json     — PromptAdapterConfig
///   adapter_model.safetensors  — contains "prompt_embeddings" tensor
///   (or adapter_model.bin)
/// ```
pub struct PromptAdapterLoader {
    device: Device,
    dtype: DType,
}

impl PromptAdapterLoader {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }

    /// Load a prompt adapter from a directory path.
    pub fn load(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        id: u32,
    ) -> Result<PromptAdapter, PromptAdapterLoadError> {
        let path = path.as_ref();

        // Load config
        let config_path = path.join("adapter_config.json");
        let config: PromptAdapterConfig = {
            let content = std::fs::read_to_string(&config_path).map_err(|e| {
                PromptAdapterLoadError::Config(format!(
                    "failed to read {}: {e}",
                    config_path.display()
                ))
            })?;
            serde_json::from_str(&content).map_err(|e| {
                PromptAdapterLoadError::Config(format!(
                    "failed to parse {}: {e}",
                    config_path.display()
                ))
            })?
        };

        // Load weights — try safetensors first, then bin
        let embeddings = self.load_embeddings(path, &config)?;

        // Validate shape
        let dims = embeddings.dims().to_vec();
        if dims.len() != 2 || dims[0] != config.num_virtual_tokens {
            return Err(PromptAdapterLoadError::ShapeMismatch {
                expected_vt: config.num_virtual_tokens,
                expected_dim: config.token_dim.unwrap_or(0),
                actual: dims,
            });
        }

        // Cast to target dtype if needed
        let embeddings = if embeddings.dtype() != self.dtype {
            embeddings
                .to_dtype(self.dtype)
                .map_err(|e| PromptAdapterLoadError::Weights(format!("dtype cast failed: {e}")))?
        } else {
            embeddings
        };

        // Move to target device if needed
        let embeddings = embeddings
            .to_device(&self.device)
            .map_err(|e| PromptAdapterLoadError::Weights(format!("device transfer failed: {e}")))?;

        Ok(PromptAdapter::new(
            name,
            id,
            config.num_virtual_tokens,
            embeddings,
        ))
    }

    /// Load prompt_embeddings tensor from safetensors or pickle format.
    fn load_embeddings(
        &self,
        dir: &Path,
        _config: &PromptAdapterConfig,
    ) -> Result<Tensor, PromptAdapterLoadError> {
        // Try safetensors first
        let safetensors_path = dir.join("adapter_model.safetensors");
        if safetensors_path.exists() {
            return self.load_from_safetensors(&safetensors_path);
        }

        // Try .bin (PyTorch pickle) — we only support safetensors in Rust
        let bin_path = dir.join("adapter_model.bin");
        if bin_path.exists() {
            return Err(PromptAdapterLoadError::Weights(
                "PyTorch .bin format not supported; convert to safetensors first".to_string(),
            ));
        }

        Err(PromptAdapterLoadError::Weights(format!(
            "no adapter weights found in {}",
            dir.display()
        )))
    }

    /// Load from safetensors file.
    fn load_from_safetensors(&self, path: &Path) -> Result<Tensor, PromptAdapterLoadError> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .map_err(|e| PromptAdapterLoadError::Weights(format!("safetensors load: {e}")))?;

        // PEFT stores the embeddings under "prompt_embeddings"
        tensors.get("prompt_embeddings").cloned().ok_or_else(|| {
            let keys: Vec<&String> = tensors.keys().collect();
            PromptAdapterLoadError::Weights(format!(
                "tensor 'prompt_embeddings' not found; available keys: {keys:?}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_missing_dir() {
        let loader = PromptAdapterLoader::new(Device::Cpu, DType::F32);
        let result = loader.load("/nonexistent/path", "test", 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            PromptAdapterLoadError::Config(msg) => {
                assert!(msg.contains("failed to read"));
            }
            other => panic!("expected Config error, got: {other}"),
        }
    }

    #[test]
    fn loader_creates() {
        let loader = PromptAdapterLoader::new(Device::Cpu, DType::F32);
        assert_eq!(loader.dtype, DType::F32);
    }

    #[test]
    fn shape_mismatch_error_display() {
        let err = PromptAdapterLoadError::ShapeMismatch {
            expected_vt: 20,
            expected_dim: 768,
            actual: vec![10, 768],
        };
        let msg = err.to_string();
        assert!(msg.contains("expected [20, 768]"));
        assert!(msg.contains("[10, 768]"));
    }
}
