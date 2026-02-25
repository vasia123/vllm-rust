//! Quantization method detection from HuggingFace model configs.
//!
//! This module provides utilities to detect the quantization method used
//! by a model from its configuration files (config.json, quantize_config.json).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::config::QuantizationMethod;

/// Detected quantization configuration from model files.
#[derive(Debug, Clone)]
pub struct DetectedQuantConfig {
    pub method: QuantizationMethod,
    pub bits: Option<u32>,
    pub group_size: Option<usize>,
    pub desc_act: Option<bool>,
    pub activation_scheme: Option<String>,
    pub raw_config: HashMap<String, Value>,
}

impl Default for DetectedQuantConfig {
    fn default() -> Self {
        Self {
            method: QuantizationMethod::None,
            bits: None,
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        }
    }
}

/// HuggingFace quantization config in config.json.
#[derive(Debug, Deserialize)]
struct HfQuantizationConfig {
    quant_method: Option<String>,
    bits: Option<u32>,
    group_size: Option<i64>,
    desc_act: Option<bool>,
    activation_scheme: Option<String>,
}

/// GPTQ-specific config (quantize_config.json).
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields used for deserialization
struct GptqConfig {
    bits: Option<u32>,
    group_size: Option<i64>,
    desc_act: Option<bool>,
    sym: Option<bool>,
    damp_percent: Option<f64>,
    model_name_or_path: Option<String>,
    model_file_base_name: Option<String>,
}

/// AWQ-specific config (quantize_config.json).
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Fields used for deserialization
struct AwqConfig {
    zero_point: Option<bool>,
    q_group_size: Option<i64>,
    w_bit: Option<u32>,
    version: Option<String>,
}

/// Detect quantization method from a model directory.
///
/// Checks the following files in order:
/// 1. config.json - for HF native quantization config
/// 2. quantize_config.json - for GPTQ/AWQ configs
///
/// # Arguments
/// * `model_dir` - Path to the model directory
///
/// # Returns
/// Detected quantization configuration
pub fn detect_from_directory(model_dir: &Path) -> DetectedQuantConfig {
    // Try config.json first
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        if let Some(config) = detect_from_config_json(&config_path) {
            return config;
        }
    }

    // Try quantize_config.json
    let quant_config_path = model_dir.join("quantize_config.json");
    if quant_config_path.exists() {
        if let Some(config) = detect_from_quantize_config(&quant_config_path) {
            return config;
        }
    }

    // Check for GGUF files
    if has_gguf_files(model_dir) {
        return DetectedQuantConfig {
            method: QuantizationMethod::Gguf,
            ..Default::default()
        };
    }

    DetectedQuantConfig::default()
}

/// Detect quantization from config.json.
fn detect_from_config_json(path: &Path) -> Option<DetectedQuantConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;

    // Check for quantization_config field
    let quant_config = json.get("quantization_config")?;

    let hf_config: HfQuantizationConfig = serde_json::from_value(quant_config.clone()).ok()?;

    let method = match hf_config.quant_method.as_deref() {
        Some("fp8") => QuantizationMethod::Fp8,
        Some("gptq") => QuantizationMethod::Gptq,
        Some("awq") => QuantizationMethod::Awq,
        Some("bitsandbytes") => QuantizationMethod::BitsAndBytes,
        Some("squeezellm") => QuantizationMethod::SqueezeLlm,
        Some("compressed-tensors") => QuantizationMethod::CompressedTensors,
        Some("torchao") => QuantizationMethod::Torchao,
        Some("experts_int8") => QuantizationMethod::ExpertsInt8,
        Some("moe_wna16") => QuantizationMethod::MoeWNA16,
        Some("awq_marlin") => QuantizationMethod::AwqMarlin,
        Some("fbgemm_fp8") => QuantizationMethod::FbgemmFp8,
        Some("ptpc_fp8") => QuantizationMethod::PtpcFp8,
        Some("mxfp4") => QuantizationMethod::Mxfp4,
        // gptq_marlin uses GPTQ-format weights with Marlin kernels — route to Marlin
        Some("gptq_marlin") => QuantizationMethod::Marlin,
        Some("modelopt") => {
            // ModelOpt supports both nested {"quantization": {"quant_algo": ...}} and
            // flat {"quant_algo": ...} layouts depending on the config file source.
            let quant_algo = quant_config
                .get("quantization")
                .and_then(|q| q.get("quant_algo"))
                .or_else(|| quant_config.get("quant_algo"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_uppercase());
            match quant_algo.as_deref() {
                Some("MXFP8") => QuantizationMethod::ModelOpt,
                Some("FP8")
                | Some("FP8_PER_CHANNEL_PER_TOKEN")
                | Some("FP8_PB_WO")
                | Some("NVFP4") => QuantizationMethod::ModelOptFull,
                _ => return None,
            }
        }
        _ => return None,
    };

    let raw_config: HashMap<String, Value> = serde_json::from_value(quant_config.clone()).ok()?;

    Some(DetectedQuantConfig {
        method,
        bits: hf_config.bits,
        group_size: hf_config.group_size.map(|g| g as usize),
        desc_act: hf_config.desc_act,
        activation_scheme: hf_config.activation_scheme,
        raw_config,
    })
}

/// Detect quantization from quantize_config.json (GPTQ/AWQ format).
fn detect_from_quantize_config(path: &Path) -> Option<DetectedQuantConfig> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;

    // Try GPTQ format first
    if let Ok(gptq) = serde_json::from_value::<GptqConfig>(json.clone()) {
        if gptq.bits.is_some() && gptq.group_size.is_some() {
            let raw_config: HashMap<String, Value> = serde_json::from_value(json).ok()?;
            // "gptq_marlin" in quantize_config.json → route to Marlin (same weight format)
            let method =
                if raw_config.get("quant_method").and_then(|v| v.as_str()) == Some("gptq_marlin") {
                    QuantizationMethod::Marlin
                } else {
                    QuantizationMethod::Gptq
                };
            return Some(DetectedQuantConfig {
                method,
                bits: gptq.bits,
                group_size: gptq.group_size.map(|g| g as usize),
                desc_act: gptq.desc_act,
                activation_scheme: None,
                raw_config,
            });
        }
    }

    // Try AWQ format
    if let Ok(awq) = serde_json::from_value::<AwqConfig>(json.clone()) {
        if awq.w_bit.is_some() {
            let raw_config: HashMap<String, Value> = serde_json::from_value(json).ok()?;
            return Some(DetectedQuantConfig {
                method: QuantizationMethod::Awq,
                bits: awq.w_bit,
                group_size: awq.q_group_size.map(|g| g as usize),
                desc_act: None,
                activation_scheme: None,
                raw_config,
            });
        }
    }

    None
}

/// Check if directory contains GGUF files.
fn has_gguf_files(dir: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".gguf") {
                    return true;
                }
            }
        }
    }
    false
}

/// Detect quantization method from a JSON config value.
///
/// This is useful when the config is already loaded.
pub fn detect_from_json(config: &Value) -> DetectedQuantConfig {
    if let Some(quant_config) = config.get("quantization_config") {
        if let Some(method_str) = quant_config.get("quant_method").and_then(|v| v.as_str()) {
            let method = match method_str {
                "fp8" => QuantizationMethod::Fp8,
                "gptq" => QuantizationMethod::Gptq,
                "awq" => QuantizationMethod::Awq,
                "bitsandbytes" => QuantizationMethod::BitsAndBytes,
                "squeezellm" => QuantizationMethod::SqueezeLlm,
                "compressed-tensors" => QuantizationMethod::CompressedTensors,
                "torchao" => QuantizationMethod::Torchao,
                "experts_int8" => QuantizationMethod::ExpertsInt8,
                "moe_wna16" => QuantizationMethod::MoeWNA16,
                "awq_marlin" => QuantizationMethod::AwqMarlin,
                "fbgemm_fp8" => QuantizationMethod::FbgemmFp8,
                "ptpc_fp8" => QuantizationMethod::PtpcFp8,
                "mxfp4" => QuantizationMethod::Mxfp4,
                // gptq_marlin uses GPTQ-format weights with Marlin kernels — route to Marlin
                "gptq_marlin" => QuantizationMethod::Marlin,
                "modelopt" => {
                    let quant_algo = quant_config
                        .get("quantization")
                        .and_then(|q| q.get("quant_algo"))
                        .or_else(|| quant_config.get("quant_algo"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_uppercase());
                    match quant_algo.as_deref() {
                        Some("MXFP8") => QuantizationMethod::ModelOpt,
                        Some("FP8")
                        | Some("FP8_PER_CHANNEL_PER_TOKEN")
                        | Some("FP8_PB_WO")
                        | Some("NVFP4") => QuantizationMethod::ModelOptFull,
                        _ => QuantizationMethod::None,
                    }
                }
                _ => QuantizationMethod::None,
            };

            let bits = quant_config
                .get("bits")
                .and_then(|v| v.as_u64())
                .map(|b| b as u32);
            let group_size = quant_config
                .get("group_size")
                .and_then(|v| v.as_i64())
                .map(|g| g as usize);
            let desc_act = quant_config.get("desc_act").and_then(|v| v.as_bool());
            let activation_scheme = quant_config
                .get("activation_scheme")
                .and_then(|v| v.as_str())
                .map(String::from);

            let raw_config: HashMap<String, Value> =
                serde_json::from_value(quant_config.clone()).unwrap_or_default();

            return DetectedQuantConfig {
                method,
                bits,
                group_size,
                desc_act,
                activation_scheme,
                raw_config,
            };
        }
    }

    DetectedQuantConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_detect_fp8_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic"
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::Fp8);
        assert_eq!(detected.activation_scheme, Some("dynamic".to_string()));
    }

    #[test]
    fn test_detect_gptq_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": true
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::Gptq);
        assert_eq!(detected.bits, Some(4));
        assert_eq!(detected.group_size, Some(128));
        assert_eq!(detected.desc_act, Some(true));
    }

    #[test]
    fn test_detect_awq_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::Awq);
        assert_eq!(detected.bits, Some(4));
        assert_eq!(detected.group_size, Some(128));
    }

    #[test]
    fn test_detect_no_quantization() {
        let config = json!({
            "hidden_size": 4096,
            "num_attention_heads": 32
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::None);
    }

    #[test]
    fn test_detected_config_default() {
        let config = DetectedQuantConfig::default();
        assert_eq!(config.method, QuantizationMethod::None);
        assert!(config.bits.is_none());
        assert!(config.group_size.is_none());
    }

    #[test]
    fn test_detect_modelopt_mxfp8_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "MXFP8",
                    "kv_cache_quant_algo": "FP8"
                }
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ModelOpt);
    }

    #[test]
    fn test_detect_modelopt_unknown_algo() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "INT8_SQ"
                }
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::None);
    }

    #[test]
    fn test_detect_modelopt_missing_quantization() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt"
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::None);
    }

    #[test]
    fn test_detect_experts_int8_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "experts_int8"
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ExpertsInt8);
    }

    #[test]
    fn test_detect_moe_wna16_from_json() {
        let config = json!({
            "quantization_config": {
                "quant_method": "moe_wna16",
                "bits": 4,
                "group_size": 128
            }
        });

        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::MoeWNA16);
        assert_eq!(detected.bits, Some(4));
        assert_eq!(detected.group_size, Some(128));
    }

    #[test]
    fn test_detect_modelopt_fp8_nested() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": { "quant_algo": "FP8" }
            }
        });
        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ModelOptFull);
    }

    #[test]
    fn test_detect_modelopt_fp8_per_channel() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": { "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN" }
            }
        });
        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ModelOptFull);
    }

    #[test]
    fn test_detect_modelopt_nvfp4_flat_layout() {
        // Flat layout: quant_algo at top level of quantization_config
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4"
            }
        });
        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ModelOptFull);
    }

    #[test]
    fn test_detect_modelopt_fp8_pb_wo() {
        let config = json!({
            "quantization_config": {
                "quant_method": "modelopt",
                "quantization": { "quant_algo": "FP8_PB_WO" }
            }
        });
        let detected = detect_from_json(&config);
        assert_eq!(detected.method, QuantizationMethod::ModelOptFull);
    }
}
