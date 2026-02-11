//! MoeWNA16: Weight-only INT4/INT8 quantization for MoE experts.
//!
//! Uses GPTQ or AWQ packed format for expert weights. Non-MoE layers
//! remain at full precision. The underlying weight format is the same
//! as GPTQ/AWQ — this config routes MoE expert layers to the
//! appropriate packed-weight linear implementation and keeps all other
//! layers unquantized.

use candle_core::{DType, Device, Result};
use std::collections::HashMap;

use super::config::{
    QuantizationConfig, QuantizationMethod, QuantizedLinear, UnquantizedLinear,
};
use super::gptq::GptqConfig;

/// Which packed format the MoE expert weights use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeWNA16Format {
    /// GPTQ packed format
    Gptq,
    /// AWQ packed format
    Awq,
}

/// Configuration for MoeWNA16 quantization.
///
/// Only MoE expert weights are quantized (INT4/INT8 packed).
/// Non-MoE layers remain at full precision.
#[derive(Debug, Clone)]
pub struct MoeWNA16Config {
    /// Weight bit width (4 or 8).
    weight_bits: u32,
    /// Group size for weight quantization.
    group_size: usize,
    /// Whether zero points are present.
    has_zp: bool,
    /// Underlying packed format.
    format: MoeWNA16Format,
    /// Model dtype for non-quantized layers.
    model_dtype: DType,
}

impl MoeWNA16Config {
    /// Create a new MoeWNA16 config.
    pub fn new(
        weight_bits: u32,
        group_size: usize,
        has_zp: bool,
        format: MoeWNA16Format,
        model_dtype: DType,
    ) -> Self {
        Self {
            weight_bits,
            group_size,
            has_zp,
            format,
            model_dtype,
        }
    }

    /// Create from detected config.
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let weight_bits = raw_config
            .get("weight_bits")
            .or_else(|| raw_config.get("bits"))
            .and_then(|v| v.as_u64())
            .map(|b| b as u32)
            .unwrap_or(4);

        let group_size = raw_config
            .get("group_size")
            .and_then(|v| v.as_i64())
            .map(|g| if g < 0 { usize::MAX } else { g as usize })
            .unwrap_or(128);

        let has_zp = raw_config
            .get("has_zp")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let format = raw_config
            .get("linear_quant_method")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "awq" => MoeWNA16Format::Awq,
                _ => MoeWNA16Format::Gptq,
            })
            .unwrap_or(MoeWNA16Format::Gptq);

        let model_dtype = raw_config
            .get("model_dtype")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "float16" => DType::F16,
                "bfloat16" => DType::BF16,
                _ => DType::BF16,
            })
            .unwrap_or(DType::BF16);

        Self {
            weight_bits,
            group_size,
            has_zp,
            format,
            model_dtype,
        }
    }

    /// Get weight bits.
    pub fn weight_bits(&self) -> u32 {
        self.weight_bits
    }

    /// Get group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Whether zero points are present.
    pub fn has_zero_points(&self) -> bool {
        self.has_zp
    }

    /// Get the underlying format.
    pub fn format(&self) -> MoeWNA16Format {
        self.format
    }

    /// Create a GPTQ linear for MoE expert weights.
    ///
    /// Uses the GPTQ infrastructure with the configured bit width and group size.
    pub fn create_expert_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Delegate to GPTQ config for the packed weight linear
        let gptq = GptqConfig::from_detected(
            Some(self.weight_bits),
            Some(self.group_size),
            Some(false), // desc_act
            &HashMap::new(),
        );
        gptq.create_linear(in_features, out_features, bias, device)
    }
}

impl QuantizationConfig for MoeWNA16Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::MoeWNA16
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        70 // Volta (same as GPTQ/AWQ)
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false
    }

    /// For MoeWNA16, non-MoE layers return unquantized linear.
    /// MoE expert layers should use `create_expert_linear()` instead.
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
            self.model_dtype,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_wna16_config_basic() {
        let config =
            MoeWNA16Config::new(4, 128, false, MoeWNA16Format::Gptq, DType::BF16);
        assert_eq!(config.method(), QuantizationMethod::MoeWNA16);
        assert_eq!(config.weight_bits(), 4);
        assert_eq!(config.group_size(), 128);
        assert!(!config.has_zero_points());
        assert_eq!(config.format(), MoeWNA16Format::Gptq);
        assert_eq!(config.min_capability(), 70);
    }

    #[test]
    fn test_moe_wna16_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert(
            "weight_bits".to_string(),
            serde_json::Value::Number(4.into()),
        );
        raw.insert(
            "group_size".to_string(),
            serde_json::Value::Number(64.into()),
        );
        raw.insert(
            "has_zp".to_string(),
            serde_json::Value::Bool(true),
        );
        raw.insert(
            "linear_quant_method".to_string(),
            serde_json::Value::String("awq".to_string()),
        );

        let config = MoeWNA16Config::from_detected(&raw);
        assert_eq!(config.weight_bits(), 4);
        assert_eq!(config.group_size(), 64);
        assert!(config.has_zero_points());
        assert_eq!(config.format(), MoeWNA16Format::Awq);
    }

    #[test]
    fn test_moe_wna16_config_from_detected_defaults() {
        let raw = HashMap::new();
        let config = MoeWNA16Config::from_detected(&raw);
        assert_eq!(config.weight_bits(), 4);
        assert_eq!(config.group_size(), 128);
        assert!(!config.has_zero_points());
        assert_eq!(config.format(), MoeWNA16Format::Gptq);
    }

    #[test]
    fn test_moe_wna16_non_moe_layer_unquantized() {
        let config =
            MoeWNA16Config::new(4, 128, false, MoeWNA16Format::Gptq, DType::F32);
        let linear = config
            .create_linear(64, 128, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        // Non-MoE layers should be full precision
        assert_eq!(linear.weight_dtype(), DType::F32);
    }

    #[test]
    fn test_moe_wna16_expert_linear_creation() {
        let config =
            MoeWNA16Config::new(4, 128, false, MoeWNA16Format::Gptq, DType::F32);
        let linear = config
            .create_expert_linear(128, 256, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 128);
        assert_eq!(linear.out_features(), 256);
    }

    #[test]
    fn test_moe_wna16_8bit() {
        let config =
            MoeWNA16Config::new(8, 128, false, MoeWNA16Format::Gptq, DType::BF16);
        assert_eq!(config.weight_bits(), 8);
    }

    #[test]
    fn test_moe_wna16_negative_group_size() {
        // group_size=-1 means channelwise
        let mut raw = HashMap::new();
        raw.insert(
            "group_size".to_string(),
            serde_json::Value::Number((-1).into()),
        );
        let config = MoeWNA16Config::from_detected(&raw);
        assert_eq!(config.group_size(), usize::MAX);
    }

    #[test]
    fn test_moe_wna16_clone_box() {
        let config =
            MoeWNA16Config::new(4, 128, false, MoeWNA16Format::Gptq, DType::BF16);
        let boxed: Box<dyn QuantizationConfig> = config.clone_box();
        assert_eq!(boxed.method(), QuantizationMethod::MoeWNA16);
    }
}
