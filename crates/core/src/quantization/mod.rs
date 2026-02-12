//! Quantization infrastructure for efficient LLM inference.
//!
//! This module provides support for various quantization methods:
//! - **FP8**: 8-bit floating point (Hopper GPUs)
//! - **GPTQ**: INT4/INT8 with grouping (Marlin kernels)
//! - **AWQ**: Activation-aware weight quantization
//! - **GGUF**: GGML universal format (llama.cpp compatible)
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

pub mod awq;
pub mod bitsandbytes;
#[cfg(feature = "cuda-kernels")]
pub mod bnb_cuda;
pub mod compressed_tensors;
mod config;
mod detection;
pub mod experts_int8;
pub mod fp8;
#[cfg(feature = "cuda-kernels")]
pub mod fp8_cuda;
pub mod gguf;
pub mod gptq;
#[cfg(feature = "cuda-kernels")]
pub mod gptq_cuda;
pub mod marlin;
#[cfg(feature = "marlin")]
pub mod marlin_cuda;
pub mod moe_wna16;
pub mod mxfp8;
pub mod weight_loader;

// Re-export public types
pub use awq::{AwqConfig, AwqLinear, AwqVersion};
pub use bitsandbytes::{
    quantize_int8, quantize_nf4, unpack_nf4, BitsAndBytesConfig, BitsAndBytesLinear, BnbQuantType,
};
#[cfg(feature = "cuda-kernels")]
pub use bnb_cuda::{bnb_int8_gemm, bnb_nf4_gemm};
pub use compressed_tensors::CompressedTensorsConfig;
pub use config::{
    ActivationScheme, NoQuantizationConfig, QuantizationConfig, QuantizationMethod,
    QuantizedLinear, UnquantizedLinear,
};
pub use detection::{detect_from_directory, detect_from_json, DetectedQuantConfig};
pub use experts_int8::{ExpertsInt8Config, ExpertsInt8Linear};
pub use fp8::Fp8Config;
#[cfg(feature = "cuda-kernels")]
pub use fp8_cuda::{fp8_dequantize, fp8_gemm, fp8_quantize_dynamic_per_token, fp8_quantize_static};
pub use gguf::{
    dequantize as gguf_dequantize, GgmlType, GgufConfig, GgufFile, GgufLinear, GgufMetadata,
    GgufTensorInfo, GgufValue, GgufWeightLoader,
};
pub use gptq::GptqConfig;
#[cfg(feature = "cuda-kernels")]
pub use gptq_cuda::{gptq_dequantize, gptq_gemm};
pub use marlin::{
    check_marlin_supported, check_marlin_supports_shape, marlin_make_workspace,
    marlin_permute_scales, repack_gptq_to_marlin, MarlinConfig, MarlinLinear, MarlinScalarType,
    GPTQ_MARLIN_MIN_THREAD_K, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_TILE,
    MARLIN_SUPPORTED_GROUP_SIZES,
};
#[cfg(feature = "marlin")]
pub use marlin_cuda::marlin_gemm;
pub use moe_wna16::{MoeWNA16Config, MoeWNA16Format};
pub use mxfp8::{MxFp8Config, MxFp8Linear, MXFP8_BLOCK_SIZE};
pub use weight_loader::{
    create_weight_loader, create_weight_loader_from_detected, create_weight_loader_with_params,
    AwqWeightLoader, BitsAndBytesWeightLoader, CompressedTensorsWeightLoader,
    ExpertsInt8WeightLoader, Fp8WeightLoader, GptqWeightLoader, MoeWNA16WeightLoader,
    MxFp8WeightLoader, QuantizedWeightLoader, UnquantizedWeightLoader,
};

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
        QuantizationMethod::Awq => Box::new(AwqConfig::from_detected(
            detected.bits,
            detected.group_size,
            &detected.raw_config,
        )),
        QuantizationMethod::Gguf => Box::new(GgufConfig::default()),
        QuantizationMethod::Marlin => {
            let is_sym = detected
                .raw_config
                .get("sym")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Box::new(MarlinConfig::from_detected(
                detected.bits,
                detected.group_size,
                detected.desc_act,
                Some(is_sym),
                &detected.raw_config,
            ))
        }
        QuantizationMethod::BitsAndBytes => {
            Box::new(BitsAndBytesConfig::from_detected(&detected.raw_config))
        }
        QuantizationMethod::ModelOpt => Box::new(MxFp8Config::from_detected(&detected.raw_config)),
        QuantizationMethod::CompressedTensors => {
            Box::new(CompressedTensorsConfig::from_detected(&detected.raw_config))
        }
        QuantizationMethod::ExpertsInt8 => {
            Box::new(ExpertsInt8Config::from_detected(&detected.raw_config))
        }
        QuantizationMethod::MoeWNA16 => {
            Box::new(MoeWNA16Config::from_detected(&detected.raw_config))
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
        QuantizationMethod::Marlin => 80,            // Ampere
        QuantizationMethod::CompressedTensors => 70, // Volta
        QuantizationMethod::Torchao => 70,           // Volta
        QuantizationMethod::ModelOpt => 0,           // Emulation on any GPU
        QuantizationMethod::ExpertsInt8 => 0,        // CPU supported (online quantization)
        QuantizationMethod::MoeWNA16 => 70,          // Volta (GPTQ/AWQ based)
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

    #[test]
    fn test_create_config_gguf() {
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gguf,
            bits: None,
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config: Default::default(),
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::Gguf);
        assert_eq!(config.min_capability(), 0); // CPU supported
    }

    #[test]
    fn test_is_supported_gguf() {
        // GGUF works on any device including CPU
        assert!(is_supported(0, QuantizationMethod::Gguf));
        assert!(is_supported(70, QuantizationMethod::Gguf));
        assert!(is_supported(90, QuantizationMethod::Gguf));
    }

    #[test]
    fn test_create_config_modelopt() {
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::ModelOpt,
            bits: None,
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config: Default::default(),
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::ModelOpt);
        assert_eq!(config.min_capability(), 0);
    }

    #[test]
    fn test_is_supported_modelopt() {
        // MXFP8 emulation works on any GPU
        assert!(is_supported(0, QuantizationMethod::ModelOpt));
        assert!(is_supported(70, QuantizationMethod::ModelOpt));
        assert!(is_supported(90, QuantizationMethod::ModelOpt));
    }

    #[test]
    fn test_create_config_compressed_tensors() {
        let mut raw_config = std::collections::HashMap::new();
        raw_config.insert(
            "format".to_string(),
            serde_json::Value::String("float-quantized".to_string()),
        );
        raw_config.insert(
            "config_groups".to_string(),
            serde_json::json!({
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": true,
                        "strategy": "tensor"
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": true,
                        "dynamic": true
                    }
                }
            }),
        );

        let detected = DetectedQuantConfig {
            method: QuantizationMethod::CompressedTensors,
            bits: Some(8),
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config,
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::CompressedTensors);
        assert_eq!(config.min_capability(), 89); // W8A8Fp8
    }

    #[test]
    fn test_create_config_experts_int8() {
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::ExpertsInt8,
            bits: None,
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config: Default::default(),
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::ExpertsInt8);
        assert_eq!(config.min_capability(), 0);
    }

    #[test]
    fn test_create_config_moe_wna16() {
        let mut raw_config = std::collections::HashMap::new();
        raw_config.insert(
            "weight_bits".to_string(),
            serde_json::Value::Number(4.into()),
        );
        raw_config.insert(
            "group_size".to_string(),
            serde_json::Value::Number(128.into()),
        );

        let detected = DetectedQuantConfig {
            method: QuantizationMethod::MoeWNA16,
            bits: Some(4),
            group_size: Some(128),
            desc_act: None,
            activation_scheme: None,
            raw_config,
        };

        let config = create_config(&detected);
        assert_eq!(config.method(), QuantizationMethod::MoeWNA16);
        assert_eq!(config.min_capability(), 70);
    }

    #[test]
    fn test_is_supported_experts_int8() {
        assert!(is_supported(0, QuantizationMethod::ExpertsInt8));
        assert!(is_supported(70, QuantizationMethod::ExpertsInt8));
        assert!(is_supported(90, QuantizationMethod::ExpertsInt8));
    }

    #[test]
    fn test_is_supported_moe_wna16() {
        assert!(!is_supported(0, QuantizationMethod::MoeWNA16));
        assert!(is_supported(70, QuantizationMethod::MoeWNA16));
        assert!(is_supported(90, QuantizationMethod::MoeWNA16));
    }
}
