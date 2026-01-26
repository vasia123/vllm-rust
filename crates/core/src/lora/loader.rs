//! LoRA adapter loading from HuggingFace format.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use thiserror::Error;

use super::types::{LoraAdapter, LoraConfig, LoraModel};

/// Errors that can occur during LoRA loading.
#[derive(Debug, Error)]
pub enum LoraLoadError {
    #[error("adapter config not found at {0}")]
    ConfigNotFound(String),
    #[error("adapter weights not found at {0}")]
    WeightsNotFound(String),
    #[error("failed to parse adapter config: {0}")]
    ConfigParse(String),
    #[error("failed to load weights: {0}")]
    WeightsLoad(String),
    #[error("mismatched lora_a and lora_b shapes for {module}: a={a_shape:?}, b={b_shape:?}")]
    ShapeMismatch {
        module: String,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    #[error("missing lora_a or lora_b for module {0}")]
    IncompleteAdapter(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// LoRA adapter loader for HuggingFace format.
///
/// Loads adapters from directories containing:
/// - `adapter_config.json`: PEFT configuration
/// - `adapter_model.safetensors` or `adapter_model.bin`: Weights
pub struct LoraLoader {
    device: Device,
    dtype: DType,
}

impl LoraLoader {
    /// Create a new loader for the specified device and dtype.
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }

    /// Load a LoRA adapter from a directory.
    ///
    /// # Arguments
    /// * `adapter_path` - Path to adapter directory
    /// * `name` - Name to assign to the adapter
    /// * `id` - Unique ID for the adapter
    pub fn load(
        &self,
        adapter_path: impl AsRef<Path>,
        name: impl Into<String>,
        id: u32,
    ) -> Result<LoraModel, LoraLoadError> {
        let path = adapter_path.as_ref();
        let name = name.into();

        // Load config
        let config = self.load_config(path)?;

        // Load weights
        let weights = self.load_weights(path)?;

        // Build LoRA model
        self.build_model(name, id, config, weights)
    }

    /// Load adapter_config.json.
    fn load_config(&self, path: &Path) -> Result<LoraConfig, LoraLoadError> {
        let config_path = path.join("adapter_config.json");
        if !config_path.exists() {
            return Err(LoraLoadError::ConfigNotFound(
                config_path.display().to_string(),
            ));
        }

        let config_str = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&config_str).map_err(|e| LoraLoadError::ConfigParse(e.to_string()))
    }

    /// Load weights from safetensors or bin format.
    fn load_weights(&self, path: &Path) -> Result<HashMap<String, Tensor>, LoraLoadError> {
        // Try safetensors first
        let safetensors_path = path.join("adapter_model.safetensors");
        if safetensors_path.exists() {
            return self.load_safetensors(&safetensors_path);
        }

        // Fall back to bin format
        let bin_path = path.join("adapter_model.bin");
        if bin_path.exists() {
            return self.load_pickle(&bin_path);
        }

        Err(LoraLoadError::WeightsNotFound(path.display().to_string()))
    }

    /// Load weights from safetensors file.
    fn load_safetensors(&self, path: &Path) -> Result<HashMap<String, Tensor>, LoraLoadError> {
        let tensors = candle_core::safetensors::load(path, &self.device)
            .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?;

        // Convert to target dtype if needed
        let mut result = HashMap::new();
        for (name, tensor) in tensors {
            let tensor = if tensor.dtype() != self.dtype {
                tensor
                    .to_dtype(self.dtype)
                    .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?
            } else {
                tensor
            };
            result.insert(name, tensor);
        }

        Ok(result)
    }

    /// Load weights from pickle (.bin) file.
    fn load_pickle(&self, path: &Path) -> Result<HashMap<String, Tensor>, LoraLoadError> {
        let tensors = candle_core::pickle::read_all(path)
            .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?;

        let mut result = HashMap::new();
        for (name, tensor) in tensors {
            let tensor = tensor
                .to_device(&self.device)
                .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?;
            let tensor = if tensor.dtype() != self.dtype {
                tensor
                    .to_dtype(self.dtype)
                    .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?
            } else {
                tensor
            };
            result.insert(name, tensor);
        }

        Ok(result)
    }

    /// Build LoraModel from config and weights.
    fn build_model(
        &self,
        name: String,
        id: u32,
        config: LoraConfig,
        weights: HashMap<String, Tensor>,
    ) -> Result<LoraModel, LoraLoadError> {
        let mut model = LoraModel::new(&name, id, config.r, config.lora_alpha);
        model.target_modules = config.target_modules.clone();

        // Group weights by module
        let module_weights = self.group_weights_by_module(weights);

        // Build adapters for each module
        for (module_name, tensors) in module_weights {
            let lora_a = tensors
                .get("lora_A")
                .or_else(|| tensors.get("lora_a"))
                .ok_or_else(|| LoraLoadError::IncompleteAdapter(module_name.clone()))?;

            let lora_b = tensors
                .get("lora_B")
                .or_else(|| tensors.get("lora_b"))
                .ok_or_else(|| LoraLoadError::IncompleteAdapter(module_name.clone()))?;

            // Validate shapes
            let a_dims = lora_a.dims();
            let b_dims = lora_b.dims();

            // lora_a: [rank, input_dim] or [input_dim, rank] depending on convention
            // lora_b: [output_dim, rank] or [rank, output_dim]
            // HF PEFT uses: lora_A: [rank, in], lora_B: [out, rank]
            if a_dims.len() != 2 || b_dims.len() != 2 {
                return Err(LoraLoadError::ShapeMismatch {
                    module: module_name,
                    a_shape: a_dims.to_vec(),
                    b_shape: b_dims.to_vec(),
                });
            }

            // Rank should match: a_dims[0] == b_dims[1]
            if a_dims[0] != b_dims[1] {
                return Err(LoraLoadError::ShapeMismatch {
                    module: module_name,
                    a_shape: a_dims.to_vec(),
                    b_shape: b_dims.to_vec(),
                });
            }

            let adapter = if config.use_rslora {
                LoraAdapter::new_with_rslora(
                    lora_a.clone(),
                    lora_b.clone(),
                    config.r,
                    config.lora_alpha,
                )
            } else {
                LoraAdapter::new(lora_a.clone(), lora_b.clone(), config.r, config.lora_alpha)
            };

            model.add_adapter(module_name, adapter);
        }

        Ok(model)
    }

    /// Group weights by module name.
    ///
    /// HuggingFace PEFT weight names follow the pattern:
    /// `base_model.model.layers.{i}.self_attn.{proj}.lora_{A|B}.weight`
    ///
    /// This function extracts the module path and groups lora_A/lora_B together.
    fn group_weights_by_module(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> HashMap<String, HashMap<String, Tensor>> {
        let mut grouped: HashMap<String, HashMap<String, Tensor>> = HashMap::new();

        for (full_name, tensor) in weights {
            // Parse the weight name to extract module path and lora type
            if let Some((module_path, lora_type)) = self.parse_weight_name(&full_name) {
                grouped
                    .entry(module_path)
                    .or_default()
                    .insert(lora_type, tensor);
            }
        }

        grouped
    }

    /// Parse a weight name to extract module path and LoRA type.
    ///
    /// Examples:
    /// - `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`
    ///   -> ("layers.0.self_attn.q_proj", "lora_A")
    /// - `model.layers.0.self_attn.q_proj.lora_B.weight`
    ///   -> ("layers.0.self_attn.q_proj", "lora_B")
    fn parse_weight_name(&self, name: &str) -> Option<(String, String)> {
        // Remove common prefixes
        let name = name
            .strip_prefix("base_model.model.")
            .or_else(|| name.strip_prefix("base_model."))
            .or_else(|| name.strip_prefix("model."))
            .unwrap_or(name);

        // Remove .weight suffix
        let name = name.strip_suffix(".weight").unwrap_or(name);

        // Find lora_A or lora_B
        if let Some(pos) = name.rfind(".lora_A") {
            let module_path = &name[..pos];
            return Some((module_path.to_string(), "lora_A".to_string()));
        }
        if let Some(pos) = name.rfind(".lora_B") {
            let module_path = &name[..pos];
            return Some((module_path.to_string(), "lora_B".to_string()));
        }
        if let Some(pos) = name.rfind(".lora_a") {
            let module_path = &name[..pos];
            return Some((module_path.to_string(), "lora_a".to_string()));
        }
        if let Some(pos) = name.rfind(".lora_b") {
            let module_path = &name[..pos];
            return Some((module_path.to_string(), "lora_b".to_string()));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_weight_name_hf_format() {
        let loader = LoraLoader::new(Device::Cpu, DType::F32);

        // Standard HuggingFace PEFT format
        let (module, lora_type) = loader
            .parse_weight_name("base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
            .unwrap();
        assert_eq!(module, "layers.0.self_attn.q_proj");
        assert_eq!(lora_type, "lora_A");

        let (module, lora_type) = loader
            .parse_weight_name("base_model.model.layers.5.self_attn.v_proj.lora_B.weight")
            .unwrap();
        assert_eq!(module, "layers.5.self_attn.v_proj");
        assert_eq!(lora_type, "lora_B");
    }

    #[test]
    fn test_parse_weight_name_variants() {
        let loader = LoraLoader::new(Device::Cpu, DType::F32);

        // Without base_model prefix
        let (module, lora_type) = loader
            .parse_weight_name("model.layers.0.mlp.gate_proj.lora_A.weight")
            .unwrap();
        assert_eq!(module, "layers.0.mlp.gate_proj");
        assert_eq!(lora_type, "lora_A");

        // Lowercase lora_a/lora_b
        let (module, lora_type) = loader
            .parse_weight_name("layers.0.self_attn.o_proj.lora_b.weight")
            .unwrap();
        assert_eq!(module, "layers.0.self_attn.o_proj");
        assert_eq!(lora_type, "lora_b");
    }

    #[test]
    fn test_parse_weight_name_invalid() {
        let loader = LoraLoader::new(Device::Cpu, DType::F32);

        // No lora suffix
        assert!(loader
            .parse_weight_name("layers.0.self_attn.q_proj.weight")
            .is_none());

        // Random tensor
        assert!(loader.parse_weight_name("some_other_tensor").is_none());
    }

    #[test]
    fn test_group_weights_by_module() {
        let loader = LoraLoader::new(Device::Cpu, DType::F32);

        let mut weights = HashMap::new();
        weights.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            Tensor::zeros((8, 512), DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            Tensor::zeros((512, 8), DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "base_model.model.layers.0.self_attn.k_proj.lora_A.weight".to_string(),
            Tensor::zeros((8, 512), DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "base_model.model.layers.0.self_attn.k_proj.lora_B.weight".to_string(),
            Tensor::zeros((512, 8), DType::F32, &Device::Cpu).unwrap(),
        );

        let grouped = loader.group_weights_by_module(weights);

        assert_eq!(grouped.len(), 2);
        assert!(grouped.contains_key("layers.0.self_attn.q_proj"));
        assert!(grouped.contains_key("layers.0.self_attn.k_proj"));

        let q_proj = grouped.get("layers.0.self_attn.q_proj").unwrap();
        assert!(q_proj.contains_key("lora_A"));
        assert!(q_proj.contains_key("lora_B"));
    }

    #[test]
    fn test_lora_config_from_json() {
        let json = r#"{
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "use_rslora": false,
            "base_model_name_or_path": "meta-llama/Llama-2-7b-hf"
        }"#;

        let config: LoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.r, 16);
        assert!((config.lora_alpha - 32.0).abs() < f32::EPSILON);
        assert_eq!(config.target_modules.len(), 4);
        assert!(!config.use_rslora);
    }

    #[test]
    fn test_lora_config_minimal() {
        let json = r#"{
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"]
        }"#;

        let config: LoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.r, 8);
        assert_eq!(config.bias, "none"); // default
        assert!(!config.use_rslora); // default
    }
}
