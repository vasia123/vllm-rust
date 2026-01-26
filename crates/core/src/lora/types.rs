//! LoRA types and data structures.

use std::collections::HashMap;

use candle_core::Tensor;
use serde::Deserialize;

/// Request for a specific LoRA adapter.
///
/// Passed with generation requests to specify which adapter to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraRequest {
    /// Human-readable name for the adapter.
    pub name: String,
    /// Globally unique integer ID (must be > 0).
    pub id: u32,
    /// Path to the adapter files.
    pub path: String,
}

impl LoraRequest {
    /// Create a new LoRA request.
    pub fn new(name: impl Into<String>, id: u32, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id,
            path: path.into(),
        }
    }

    /// Create a LoRA request that references an adapter by name only.
    ///
    /// This is used when the adapter is already loaded and only the name
    /// is needed to look it up. The id and path are set to defaults.
    pub fn by_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: 0,
            path: String::new(),
        }
    }
}

/// Configuration from adapter_config.json (PEFT format).
#[derive(Debug, Clone, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank (r parameter).
    pub r: usize,
    /// Scaling parameter (alpha).
    pub lora_alpha: f32,
    /// Which modules to apply LoRA to.
    pub target_modules: Vec<String>,
    /// Dropout probability (optional, not used at inference).
    #[serde(default)]
    pub lora_dropout: f32,
    /// Bias handling: "none", "all", "lora_only".
    #[serde(default = "default_bias")]
    pub bias: String,
    /// Use rank-stabilized LoRA scaling.
    #[serde(default)]
    pub use_rslora: bool,
    /// Base model name/path for validation.
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
}

fn default_bias() -> String {
    "none".to_string()
}

impl LoraConfig {
    /// Compute the scaling factor for LoRA.
    ///
    /// For standard LoRA: scale = alpha / rank
    /// For rsLoRA: scale = alpha / sqrt(rank)
    pub fn scaling(&self) -> f32 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f32).sqrt()
        } else {
            self.lora_alpha / self.r as f32
        }
    }
}

/// LoRA weights for a single layer (low-rank matrices A and B).
///
/// The LoRA computation is:
/// ```text
/// output = base_output + scale * (x @ lora_a.T @ lora_b.T)
/// ```
///
/// Where:
/// - lora_a: [rank, input_dim]
/// - lora_b: [output_dim, rank]
/// - scale: alpha / rank
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Low-rank matrix A: [rank, input_dim].
    pub lora_a: Tensor,
    /// Low-rank matrix B: [output_dim, rank].
    pub lora_b: Tensor,
    /// LoRA rank.
    pub rank: usize,
    /// Alpha parameter.
    pub alpha: f32,
    /// Pre-computed scale (alpha / rank).
    pub scale: f32,
}

impl LoraAdapter {
    /// Create a new LoRA adapter.
    ///
    /// # Arguments
    /// * `lora_a` - Low-rank matrix A [rank, input_dim]
    /// * `lora_b` - Low-rank matrix B [output_dim, rank]
    /// * `rank` - LoRA rank
    /// * `alpha` - Alpha scaling parameter
    pub fn new(lora_a: Tensor, lora_b: Tensor, rank: usize, alpha: f32) -> Self {
        let scale = alpha / rank as f32;
        Self {
            lora_a,
            lora_b,
            rank,
            alpha,
            scale,
        }
    }

    /// Create with rank-stabilized scaling (rsLoRA).
    pub fn new_with_rslora(lora_a: Tensor, lora_b: Tensor, rank: usize, alpha: f32) -> Self {
        let scale = alpha / (rank as f32).sqrt();
        Self {
            lora_a,
            lora_b,
            rank,
            alpha,
            scale,
        }
    }

    /// Create with custom scaling factor.
    pub fn with_scale(lora_a: Tensor, lora_b: Tensor, rank: usize, alpha: f32, scale: f32) -> Self {
        Self {
            lora_a,
            lora_b,
            rank,
            alpha,
            scale,
        }
    }

    /// Input dimension (from lora_a shape).
    pub fn input_dim(&self) -> usize {
        self.lora_a.dims()[1]
    }

    /// Output dimension (from lora_b shape).
    pub fn output_dim(&self) -> usize {
        self.lora_b.dims()[0]
    }

    /// Pre-merge scaling into lora_b for faster inference.
    ///
    /// After calling this, scale becomes 1.0 and the scaling is baked into lora_b.
    pub fn optimize(&mut self) -> candle_core::Result<()> {
        if (self.scale - 1.0).abs() > f32::EPSILON {
            self.lora_b = (&self.lora_b * self.scale as f64)?;
            self.scale = 1.0;
        }
        Ok(())
    }
}

/// A complete LoRA model containing adapters for multiple layers.
#[derive(Debug)]
pub struct LoraModel {
    /// Adapter name.
    pub name: String,
    /// Unique ID.
    pub id: u32,
    /// LoRA rank.
    pub rank: usize,
    /// Alpha parameter.
    pub alpha: f32,
    /// Adapters by module name (e.g., "layers.0.self_attn.q_proj").
    pub adapters: HashMap<String, LoraAdapter>,
    /// Target modules this LoRA was trained for.
    pub target_modules: Vec<String>,
}

impl LoraModel {
    /// Create a new empty LoRA model.
    pub fn new(name: impl Into<String>, id: u32, rank: usize, alpha: f32) -> Self {
        Self {
            name: name.into(),
            id,
            rank,
            alpha,
            adapters: HashMap::new(),
            target_modules: Vec::new(),
        }
    }

    /// Get adapter for a specific module path.
    pub fn get_adapter(&self, module_name: &str) -> Option<&LoraAdapter> {
        self.adapters.get(module_name)
    }

    /// Add an adapter for a module.
    pub fn add_adapter(&mut self, module_name: impl Into<String>, adapter: LoraAdapter) {
        self.adapters.insert(module_name.into(), adapter);
    }

    /// Number of layers with adapters.
    pub fn num_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Optimize all adapters by pre-merging scale into lora_b.
    pub fn optimize(&mut self) -> candle_core::Result<()> {
        for adapter in self.adapters.values_mut() {
            adapter.optimize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_lora_request() {
        let req = LoraRequest::new("sql-adapter", 1, "/path/to/adapter");
        assert_eq!(req.name, "sql-adapter");
        assert_eq!(req.id, 1);
        assert_eq!(req.path, "/path/to/adapter");
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoraConfig {
            r: 16,
            lora_alpha: 32.0,
            target_modules: vec!["q_proj".to_string()],
            lora_dropout: 0.0,
            bias: "none".to_string(),
            use_rslora: false,
            base_model_name_or_path: None,
        };

        // Standard scaling: alpha / rank = 32 / 16 = 2.0
        assert!((config.scaling() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lora_config_rslora_scaling() {
        let config = LoraConfig {
            r: 16,
            lora_alpha: 32.0,
            target_modules: vec!["q_proj".to_string()],
            lora_dropout: 0.0,
            bias: "none".to_string(),
            use_rslora: true,
            base_model_name_or_path: None,
        };

        // rsLoRA scaling: alpha / sqrt(rank) = 32 / 4 = 8.0
        assert!((config.scaling() - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lora_adapter_dimensions() {
        let device = Device::Cpu;
        let rank = 8;
        let input_dim = 512;
        let output_dim = 256;

        let lora_a = Tensor::zeros((rank, input_dim), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((output_dim, rank), DType::F32, &device).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, rank, 16.0);

        assert_eq!(adapter.rank, 8);
        assert_eq!(adapter.input_dim(), 512);
        assert_eq!(adapter.output_dim(), 256);
        assert!((adapter.scale - 2.0).abs() < f32::EPSILON); // 16.0 / 8
    }

    #[test]
    fn test_lora_adapter_optimize() {
        let device = Device::Cpu;
        let rank = 8;
        let scale = 2.0;

        let lora_a = Tensor::ones((rank, 16), DType::F32, &device).unwrap();
        let lora_b = Tensor::ones((32, rank), DType::F32, &device).unwrap();

        let mut adapter = LoraAdapter::new(lora_a, lora_b, rank, 16.0);
        assert!((adapter.scale - scale).abs() < f32::EPSILON);

        adapter.optimize().unwrap();
        assert!((adapter.scale - 1.0).abs() < f32::EPSILON);

        // lora_b should now be scaled by 2.0
        let lora_b_values: Vec<f32> = adapter.lora_b.flatten_all().unwrap().to_vec1().unwrap();
        assert!(lora_b_values
            .iter()
            .all(|&v| (v - scale).abs() < f32::EPSILON));
    }

    #[test]
    fn test_lora_model() {
        let device = Device::Cpu;
        let mut model = LoraModel::new("test-adapter", 1, 8, 16.0);

        let lora_a = Tensor::zeros((8, 512), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((256, 8), DType::F32, &device).unwrap();

        let adapter = LoraAdapter::new(lora_a, lora_b, 8, 16.0);
        model.add_adapter("layers.0.self_attn.q_proj", adapter);

        assert_eq!(model.num_adapters(), 1);
        assert!(model.get_adapter("layers.0.self_attn.q_proj").is_some());
        assert!(model.get_adapter("layers.0.self_attn.k_proj").is_none());
    }

    #[test]
    fn test_lora_config_deserialize() {
        let json = r#"{
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.05,
            "bias": "none"
        }"#;

        let config: LoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.r, 16);
        assert!((config.lora_alpha - 32.0).abs() < f32::EPSILON);
        assert_eq!(config.target_modules.len(), 4);
        assert!((config.lora_dropout - 0.05).abs() < f32::EPSILON);
        assert_eq!(config.bias, "none");
    }
}
