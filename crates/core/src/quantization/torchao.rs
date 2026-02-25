//! TorchAO quantization — detection and weight-loading skeleton.
//!
//! TorchAO (`quant_method = "torchao"`) stores weights as standard F32/BF16
//! tensors and applies quantization transformations dynamically via the
//! `torchao` Python library at inference time.  Because no stable Rust
//! `torchao` bindings exist, this implementation:
//!
//! - **Detects** `torchao` checkpoints and reports [`QuantizationMethod::Torchao`].
//! - **Loads weights** as standard tensors (no packing / bit-unpacking needed).
//! - **Runs inference** with standard F32/BF16 matmul — functionally correct
//!   but without the quantization speedups that the Python library provides.
//!
//! Future work: integrate torchao Rust bindings when available, or port the
//! relevant INT4/FP8 kernels from torchao to CUDA/cudarc directly.
//!
//! Reference: `vllm/model_executor/layers/quantization/torchao.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result};

use super::config::{
    NoQuantizationConfig, QuantizationConfig, QuantizationMethod, QuantizedLinear,
};

// ─── TorchaoConfig ───────────────────────────────────────────────────────────

/// TorchAO quantization configuration.
///
/// Weights are stored as standard BF16/F32 tensors; no bit-unpacking is needed
/// at load time.  Inference falls back to standard matmul until native TorchAO
/// Rust kernels are integrated.
#[derive(Debug, Clone)]
pub struct TorchaoConfig {
    inner: NoQuantizationConfig,
}

impl TorchaoConfig {
    /// Create from a detected quantization config.
    ///
    /// Reads `params_dtype` to infer the weight dtype (default BF16).
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let dtype = match raw_config
            .get("quant_type")
            .and_then(|v| {
                v.get("default")
                    .and_then(|d| d.get("_data"))
                    .and_then(|d| d.get("params_dtype"))
                    .and_then(|d| d.as_str())
            })
            .or_else(|| raw_config.get("params_dtype").and_then(|v| v.as_str()))
        {
            Some("float32") | Some("fp32") => DType::F32,
            Some("float16") | Some("fp16") => DType::F16,
            // BF16 is the TorchAO default for most models
            _ => DType::BF16,
        };

        Self {
            inner: NoQuantizationConfig::new(dtype),
        }
    }
}

impl QuantizationConfig for TorchaoConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Torchao
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        self.inner.supported_act_dtypes()
    }

    fn min_capability(&self) -> u32 {
        // Python says 75; standard matmul works on any CUDA capability.
        75
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        // All layers use standard matmul — none are skipped.
        self.inner.is_layer_skipped(layer_name)
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Standard matmul — no weight packing or transformation needed.
        // NOTE: Replace with torchao kernel dispatch when Rust bindings exist.
        self.inner
            .create_linear(in_features, out_features, bias, device)
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn default_config() -> TorchaoConfig {
        TorchaoConfig::from_detected(&HashMap::new())
    }

    #[test]
    fn test_torchao_method() {
        assert_eq!(default_config().method(), QuantizationMethod::Torchao);
    }

    #[test]
    fn test_torchao_min_capability() {
        assert_eq!(default_config().min_capability(), 75);
    }

    #[test]
    fn test_torchao_default_dtype_is_bf16() {
        let cfg = default_config();
        let l = cfg.create_linear(64, 64, false, &Device::Cpu).unwrap();
        // Default weight dtype is BF16 (TorchAO default).
        assert_eq!(l.weight_dtype(), DType::BF16);
    }

    #[test]
    fn test_torchao_create_linear() {
        let cfg = default_config();
        let linear = cfg.create_linear(64, 128, false, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
        assert!(!l.has_bias());
    }

    #[test]
    fn test_torchao_create_linear_with_bias() {
        let cfg = default_config();
        let linear = cfg.create_linear(32, 64, true, &Device::Cpu);
        assert!(linear.is_ok());
        assert!(linear.unwrap().has_bias());
    }
}
