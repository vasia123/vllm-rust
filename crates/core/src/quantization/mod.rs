//! Quantization infrastructure for efficient LLM inference.
//!
//! This module provides support for various quantization methods:
//! - **FP8**: 8-bit floating point (Hopper GPUs)
//! - **GPTQ**: INT4/INT8 with grouping (Marlin kernels)
//! - **AWQ**: Activation-aware weight quantization (future)
//!
//! # Architecture
//!
//! The quantization system uses a trait-based design:
//!
//! - `QuantizationConfig`: Configuration for each method
//! - `QuantizedLinear`: Trait for quantized layer operations
//! - Detection utilities for HuggingFace model configs
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::quantization::{detect_from_directory, create_config};
//!
//! let detected = detect_from_directory(model_path);
//! let config = create_config(&detected)?;
//! let linear = config.create_linear(4096, 4096, false, &device)?;
//! ```

mod config;
mod detection;
pub mod fp8;
#[cfg(feature = "cuda-kernels")]
pub mod fp8_cuda;
pub mod gptq;
#[cfg(feature = "cuda-kernels")]
pub mod gptq_cuda;

// Re-export public types
pub use config::{
    ActivationScheme, NoQuantizationConfig, QuantizationConfig, QuantizationMethod,
    QuantizedLinear, UnquantizedLinear,
};
pub use detection::{detect_from_directory, detect_from_json, DetectedQuantConfig};
pub use fp8::Fp8Config;
#[cfg(feature = "cuda-kernels")]
pub use fp8_cuda::{fp8_dequantize, fp8_gemm, fp8_quantize_dynamic_per_token, fp8_quantize_static};
pub use gptq::GptqConfig;
#[cfg(feature = "cuda-kernels")]
pub use gptq_cuda::{gptq_dequantize, gptq_gemm};

use std::path::Path;

/// Create a quantization config from detected configuration.
///
/// # Arguments
/// * `detected` - Detected quantization configuration from model files
///
/// # Returns
/// Boxed quantization config trait object
pub fn create_config(detected: &DetectedQuantConfig) -> Box<dyn QuantizationConfig> {
    match detected.method {
        QuantizationMethod::Fp8 => Box::new(Fp8Config::from_detected(
            detected.bits,
            detected.activation_scheme.as_deref(),
            &detected.raw_config,
        )),
        QuantizationMethod::Gptq => Box::new(GptqConfig::from_detected(
            detected.bits,
            detected.group_size,
            detected.desc_act,
            &detected.raw_config,
        )),
        QuantizationMethod::Awq => {
            // AWQ uses similar config to GPTQ
            Box::new(GptqConfig::from_detected(
                detected.bits,
                detected.group_size,
                None,
                &detected.raw_config,
            ))
        }
        _ => Box::new(NoQuantizationConfig::default()),
    }
}

/// Create a quantization config from a model directory.
///
/// Convenience function that combines detection and config creation.
///
/// # Arguments
/// * `model_dir` - Path to the model directory
///
/// # Returns
/// Boxed quantization config trait object
pub fn create_config_from_directory(model_dir: &Path) -> Box<dyn QuantizationConfig> {
    let detected = detect_from_directory(model_dir);
    create_config(&detected)
}

/// Check if a GPU compute capability supports the given quantization method.
///
/// # Arguments
/// * `capability` - GPU compute capability (e.g., 80 for Ampere)
/// * `method` - Quantization method to check
///
/// # Returns
/// true if the GPU supports the method
pub fn is_supported(capability: u32, method: QuantizationMethod) -> bool {
    let min_cap = match method {
        QuantizationMethod::None => 0,
        QuantizationMethod::Fp8 => 89,  // Hopper
        QuantizationMethod::Gptq => 70, // Volta (Marlin needs 80)
        QuantizationMethod::Awq => 70,  // Volta
        QuantizationMethod::Gguf => 0,  // CPU supported
        QuantizationMethod::BitsAndBytes => 70,
        QuantizationMethod::SqueezeLlm => 70,
        QuantizationMethod::Marlin => 80, // Ampere
    };
    capability >= min_cap
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_create_config_fp8() {
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Fp8,
            bits: Some(8),
            group_size: None,
            desc_act: None,
            activation_scheme: Some("dynamic".to_string()),
            raw_config: Default::default(),
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_create_config_gptq() {
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(true),
            activation_scheme: None,
            raw_config: Default::default(),
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_create_config_none() {
        let detected = DetectedQuantConfig::default();
        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::None);
    }

    #[test]
    fn test_is_supported() {
        // Hopper (H100) supports everything
        assert!(is_supported(90, QuantizationMethod::Fp8));
        assert!(is_supported(90, QuantizationMethod::Gptq));
        assert!(is_supported(90, QuantizationMethod::Marlin));

        // Ampere (A100) supports most
        assert!(!is_supported(80, QuantizationMethod::Fp8));
        assert!(is_supported(80, QuantizationMethod::Gptq));
        assert!(is_supported(80, QuantizationMethod::Marlin));

        // Volta supports basic
        assert!(is_supported(70, QuantizationMethod::Gptq));
        assert!(!is_supported(70, QuantizationMethod::Marlin));
    }

    #[test]
    fn test_quantization_config_creates_linear() {
        let config: Box<dyn QuantizationConfig> = Box::new(NoQuantizationConfig::default());
        let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
    }
}
