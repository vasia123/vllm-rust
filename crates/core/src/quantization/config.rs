//! Core quantization configuration and traits.
//!
//! This module provides the base traits and types for quantization:
//! - `QuantizationConfig` - Configuration for quantization methods
//! - `QuantizedLinear` - Trait for quantized linear layer operations
//! - `QuantizationMethod` - Enum of supported quantization methods

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported quantization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    /// No quantization (full precision)
    #[default]
    None,
    /// FP8 quantization (E4M3 or E5M2)
    Fp8,
    /// GPTQ quantization (INT4/INT8 with grouping)
    Gptq,
    /// AWQ quantization (Activation-aware Weight Quantization)
    Awq,
    /// GGUF/GGML quantization formats
    Gguf,
    /// BitsAndBytes quantization
    BitsAndBytes,
    /// SqueezeLLM quantization
    SqueezeLlm,
    /// Marlin optimized format (for GPTQ/AWQ)
    Marlin,
    /// Compressed-Tensors quantization (NVIDIA/Neural Magic)
    CompressedTensors,
    /// TorchAO quantization (PyTorch native)
    Torchao,
    /// ModelOpt quantization (NVIDIA MXFP8)
    ModelOpt,
    /// ExpertsInt8: online INT8 quantization for MoE expert weights (W8A16)
    ExpertsInt8,
    /// MoeWNA16: GPTQ/AWQ weight-only INT4/INT8 for MoE experts
    MoeWNA16,
    /// AWQ weights with Marlin inference kernel (AWQ-Marlin)
    AwqMarlin,
    /// FBGEMM FP8: per-channel FP8 weights with dynamic activation quantization (Meta)
    FbgemmFp8,
    /// PTPC FP8: Per-Token Per-Channel dynamic FP8 quantization (ROCm/AMD MI300+)
    PtpcFp8,
    /// MXFP4 (OCP MX FP4 E2M1): 4-bit microscaling quantization
    Mxfp4,
}

impl std::fmt::Display for QuantizationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Gptq => write!(f, "gptq"),
            Self::Awq => write!(f, "awq"),
            Self::Gguf => write!(f, "gguf"),
            Self::BitsAndBytes => write!(f, "bitsandbytes"),
            Self::SqueezeLlm => write!(f, "squeezellm"),
            Self::Marlin => write!(f, "marlin"),
            Self::CompressedTensors => write!(f, "compressed-tensors"),
            Self::Torchao => write!(f, "torchao"),
            Self::ModelOpt => write!(f, "modelopt"),
            Self::ExpertsInt8 => write!(f, "experts_int8"),
            Self::MoeWNA16 => write!(f, "moe_wna16"),
            Self::AwqMarlin => write!(f, "awq_marlin"),
            Self::FbgemmFp8 => write!(f, "fbgemm_fp8"),
            Self::PtpcFp8 => write!(f, "ptpc_fp8"),
            Self::Mxfp4 => write!(f, "mxfp4"),
        }
    }
}

/// Activation quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationScheme {
    /// Static quantization with pre-computed scales
    Static,
    /// Dynamic quantization computing scales at runtime
    #[default]
    Dynamic,
}

/// Base trait for quantization configurations.
///
/// This trait defines the interface for quantization method configurations.
/// Each quantization method (FP8, GPTQ, AWQ, etc.) implements this trait.
pub trait QuantizationConfig: Send + Sync + std::fmt::Debug {
    /// Returns the quantization method name.
    fn method(&self) -> QuantizationMethod;

    /// Returns supported activation dtypes.
    fn supported_act_dtypes(&self) -> &[DType];

    /// Returns minimum GPU compute capability required.
    /// E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
    fn min_capability(&self) -> u32;

    /// Check if a layer should be skipped (not quantized).
    fn is_layer_skipped(&self, layer_name: &str) -> bool;

    /// Create a quantized linear layer for the given configuration.
    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>>;

    /// Clone the configuration into a boxed trait object.
    fn clone_box(&self) -> Box<dyn QuantizationConfig>;
}

impl Clone for Box<dyn QuantizationConfig> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Trait for quantized linear layer operations.
///
/// This trait defines the interface for quantized linear layers.
/// Different quantization methods implement this trait differently.
pub trait QuantizedLinear: Send + Sync {
    /// Forward pass through the quantized linear layer.
    fn forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Load weights from a state dict.
    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()>;

    /// Get the weight dtype.
    fn weight_dtype(&self) -> DType;

    /// Get input features.
    fn in_features(&self) -> usize;

    /// Get output features.
    fn out_features(&self) -> usize;

    /// Check if bias is present.
    fn has_bias(&self) -> bool;
}

/// No quantization (full precision) configuration.
#[derive(Debug, Clone)]
pub struct NoQuantizationConfig {
    dtype: DType,
}

impl NoQuantizationConfig {
    pub fn new(dtype: DType) -> Self {
        Self { dtype }
    }
}

impl Default for NoQuantizationConfig {
    fn default() -> Self {
        Self { dtype: DType::BF16 }
    }
}

impl QuantizationConfig for NoQuantizationConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::None
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        0 // No minimum requirement
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(UnquantizedLinear::new(
            in_features,
            out_features,
            bias,
            self.dtype,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// Unquantized (full precision) linear layer.
#[derive(Debug)]
pub struct UnquantizedLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl UnquantizedLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Initialize with zeros (weights will be loaded later)
        let weight = Tensor::zeros((out_features, in_features), dtype, device)?;
        let bias = if has_bias {
            Some(Tensor::zeros(out_features, dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }
}

impl QuantizedLinear for UnquantizedLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_method_display() {
        assert_eq!(QuantizationMethod::None.to_string(), "none");
        assert_eq!(QuantizationMethod::Fp8.to_string(), "fp8");
        assert_eq!(QuantizationMethod::Gptq.to_string(), "gptq");
        assert_eq!(QuantizationMethod::Awq.to_string(), "awq");
    }

    #[test]
    fn test_no_quantization_config() {
        let config = NoQuantizationConfig::default();
        assert_eq!(config.method(), QuantizationMethod::None);
        assert_eq!(config.min_capability(), 0);
        assert!(!config.is_layer_skipped("layer.0.weight"));
    }

    #[test]
    fn test_unquantized_linear_creation() {
        let config = NoQuantizationConfig::new(DType::F32);
        let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
        assert_eq!(linear.weight_dtype(), DType::F32);
    }

    #[test]
    fn test_unquantized_linear_forward() {
        let config = NoQuantizationConfig::new(DType::F32);
        let linear = config.create_linear(4, 8, false, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
    }
}
