//! Compressed-Tensors quantization (Neural Magic / NVIDIA).
//!
//! Compressed-Tensors is a meta-format that describes quantization via
//! `config_groups` in the model config. Each group specifies weight and
//! optional activation quantization parameters. The actual computation
//! is delegated to the appropriate backend (FP8, GPTQ/Marlin, INT8).
//!
//! Supported schemes:
//! - W8A8Fp8: 8-bit float weights + activations → delegates to Fp8Linear
//! - W8A16Fp8: 8-bit float weights, FP16 activations → delegates to Fp8Linear
//! - WNA16: Packed INT4/INT8 weights, FP16 activations → delegates to GptqLinear/MarlinLinear
//! - W8A8Int8: 8-bit integer weights + activations (channel/tensor symmetric)

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use serde_json::Value;

use super::config::{
    ActivationScheme, QuantizationConfig, QuantizationMethod, QuantizedLinear, UnquantizedLinear,
};
use super::fp8::Fp8Config;
use super::gptq::GptqConfig;

// ─── Compressed-Tensors Type System ─────────────────────────────────────────

/// Quantization value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtQuantType {
    Int,
    Float,
}

impl CtQuantType {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "int" => Self::Int,
            _ => Self::Float,
        }
    }
}

/// How quantization scales are structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtStrategy {
    /// Single scale per tensor
    Tensor,
    /// Per-output-channel scale
    Channel,
    /// Per-group scale (e.g., every 128 elements)
    Group,
    /// 2D block scales [block_n, block_k]
    Block,
    /// Per-token scale (for activations)
    Token,
}

impl CtStrategy {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "tensor" => Self::Tensor,
            "channel" => Self::Channel,
            "group" => Self::Group,
            "block" => Self::Block,
            "token" => Self::Token,
            _ => Self::Tensor,
        }
    }
}

/// Per-tensor quantization arguments.
#[derive(Debug, Clone)]
pub struct CtQuantArgs {
    pub num_bits: u32,
    pub quant_type: CtQuantType,
    pub strategy: CtStrategy,
    pub symmetric: bool,
    pub dynamic: bool,
    pub group_size: Option<i32>,
    pub block_structure: Option<[usize; 2]>,
}

impl CtQuantArgs {
    fn from_json(val: &Value) -> Option<Self> {
        let obj = val.as_object()?;
        let num_bits = obj.get("num_bits").and_then(|v| v.as_u64()).unwrap_or(8) as u32;
        let quant_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .map(CtQuantType::from_str)
            .unwrap_or(CtQuantType::Float);
        let strategy = obj
            .get("strategy")
            .and_then(|v| v.as_str())
            .map(CtStrategy::from_str)
            .unwrap_or(CtStrategy::Tensor);
        let symmetric = obj
            .get("symmetric")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let dynamic = obj
            .get("dynamic")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let group_size = obj
            .get("group_size")
            .and_then(|v| v.as_i64())
            .map(|g| g as i32);
        let block_structure = obj
            .get("block_structure")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 2 {
                    let r = arr[0].as_u64()? as usize;
                    let c = arr[1].as_u64()? as usize;
                    Some([r, c])
                } else {
                    None
                }
            });

        Some(Self {
            num_bits,
            quant_type,
            strategy,
            symmetric,
            dynamic,
            group_size,
            block_structure,
        })
    }
}

/// Compression format from the model config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionFormat {
    Dense,
    PackQuantized,
    IntQuantized,
    FloatQuantized,
    NaiveQuantized,
}

impl CompressionFormat {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().replace('-', "_").as_str() {
            "pack_quantized" | "pack-quantized" => Self::PackQuantized,
            "int_quantized" | "int-quantized" => Self::IntQuantized,
            "float_quantized" | "float-quantized" => Self::FloatQuantized,
            "naive_quantized" | "naive-quantized" => Self::NaiveQuantized,
            "dense" => Self::Dense,
            _ => Self::Dense,
        }
    }
}

/// A config group entry mapping targets to quantization schemes.
#[derive(Debug, Clone)]
struct CtConfigGroup {
    #[allow(dead_code)] // Used by scheme_for_layer() and tests
    targets: Vec<String>,
    weights: CtQuantArgs,
    input_activations: Option<CtQuantArgs>,
}

// ─── Scheme Selection ───────────────────────────────────────────────────────

/// The resolved quantization scheme for a layer.
#[derive(Debug, Clone)]
enum CtScheme {
    /// FP8 weights and activations
    W8A8Fp8 {
        is_static_input: bool,
        weight_block_size: Option<[usize; 2]>,
    },
    /// FP8 weights, unquantized activations
    W8A16Fp8 {
        weight_block_size: Option<[usize; 2]>,
    },
    /// INT8 weights and activations
    W8A8Int8 {
        #[allow(dead_code)] // Used in tests and future INT8 GEMM kernel dispatch
        strategy: CtStrategy,
        #[allow(dead_code)]
        is_static_input: bool,
    },
    /// Packed N-bit integer weights (4 or 8), FP16 activations
    WNA16 {
        num_bits: u32,
        group_size: i32,
        symmetric: bool,
        desc_act: bool,
    },
    /// Unquantized (skip this layer)
    Unquantized,
}

fn select_scheme(
    weights: &CtQuantArgs,
    input: Option<&CtQuantArgs>,
    format: CompressionFormat,
) -> CtScheme {
    // WNA16: packed quantized INT4/INT8, no activation quantization
    if format == CompressionFormat::PackQuantized
        && weights.quant_type == CtQuantType::Int
        && (weights.num_bits == 4 || weights.num_bits == 8)
        && input.is_none()
        && (weights.strategy == CtStrategy::Group || weights.strategy == CtStrategy::Channel)
    {
        let group_size = match weights.strategy {
            CtStrategy::Group => weights.group_size.unwrap_or(128),
            CtStrategy::Channel => -1,
            _ => 128,
        };
        return CtScheme::WNA16 {
            num_bits: weights.num_bits,
            group_size,
            symmetric: weights.symmetric,
            desc_act: false,
        };
    }

    // FP8 schemes
    if weights.quant_type == CtQuantType::Float && weights.num_bits == 8 && weights.symmetric {
        if let Some(inp) = input {
            if inp.quant_type == CtQuantType::Float && inp.num_bits == 8 {
                // W8A8Fp8
                return CtScheme::W8A8Fp8 {
                    is_static_input: !inp.dynamic,
                    weight_block_size: weights.block_structure,
                };
            }
        } else {
            // W8A16Fp8 (no activation quantization)
            return CtScheme::W8A16Fp8 {
                weight_block_size: weights.block_structure,
            };
        }
    }

    // INT8 schemes
    if weights.quant_type == CtQuantType::Int && weights.num_bits == 8 {
        if let Some(inp) = input {
            if inp.quant_type == CtQuantType::Int && inp.num_bits == 8 {
                // W8A8Int8
                return CtScheme::W8A8Int8 {
                    strategy: weights.strategy,
                    is_static_input: !inp.dynamic,
                };
            }
        }
    }

    CtScheme::Unquantized
}

// ─── CompressedTensorsConfig ────────────────────────────────────────────────

/// Compressed-Tensors quantization configuration.
///
/// Parses the `quantization_config.config_groups` from HuggingFace model
/// config and resolves each layer to the appropriate quantization scheme.
#[derive(Debug, Clone)]
pub struct CompressedTensorsConfig {
    /// Parsed config groups with their targets and quantization args
    #[allow(dead_code)] // Used in scheme_for_layer() and tests
    groups: Vec<CtConfigGroup>,
    /// Global compression format
    #[allow(dead_code)] // Used in tests
    format: CompressionFormat,
    /// Layer names/patterns to skip
    ignore: Vec<String>,
    /// Resolved default scheme (for layers matching the default group)
    default_scheme: CtScheme,
}

impl CompressedTensorsConfig {
    /// Parse from the raw quantization_config JSON.
    pub fn from_detected(raw_config: &HashMap<String, Value>) -> Self {
        let format = raw_config
            .get("format")
            .and_then(|v| v.as_str())
            .map(CompressionFormat::from_str)
            .unwrap_or(CompressionFormat::Dense);

        let ignore = raw_config
            .get("ignore")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let mut groups = Vec::new();
        if let Some(config_groups) = raw_config.get("config_groups").and_then(|v| v.as_object()) {
            for (_name, group_val) in config_groups {
                if let Some(group) = Self::parse_group(group_val) {
                    groups.push(group);
                }
            }
        }

        // Determine default scheme from first group (most configs have one group)
        let default_scheme = groups
            .first()
            .map(|g| select_scheme(&g.weights, g.input_activations.as_ref(), format))
            .unwrap_or(CtScheme::Unquantized);

        Self {
            groups,
            format,
            ignore,
            default_scheme,
        }
    }

    fn parse_group(val: &Value) -> Option<CtConfigGroup> {
        let obj = val.as_object()?;

        let targets = obj
            .get("targets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec!["Linear".to_string()]);

        let weights = obj.get("weights").and_then(CtQuantArgs::from_json)?;

        let input_activations = obj
            .get("input_activations")
            .and_then(CtQuantArgs::from_json);

        Some(CtConfigGroup {
            targets,
            weights,
            input_activations,
        })
    }

    /// Check if a layer name should be ignored (not quantized).
    fn should_ignore(&self, layer_name: &str) -> bool {
        self.ignore
            .iter()
            .any(|pattern| layer_name.contains(pattern))
    }

    /// Get the scheme for a specific layer.
    #[allow(dead_code)] // Will be used for per-layer scheme dispatch
    fn scheme_for_layer(&self, layer_name: &str) -> CtScheme {
        if self.should_ignore(layer_name) {
            return CtScheme::Unquantized;
        }

        // Check each config group for a matching target.
        // In practice most models have a single group targeting "Linear".
        for group in &self.groups {
            let matches = group.targets.iter().any(|t| {
                t == "Linear"
                    || t == "nn.Linear"
                    || layer_name.contains(t)
                    || layer_name.ends_with("_proj")
                    || layer_name.ends_with("_proj.weight")
            });
            if matches {
                return select_scheme(
                    &group.weights,
                    group.input_activations.as_ref(),
                    self.format,
                );
            }
        }

        self.default_scheme.clone()
    }
}

impl QuantizationConfig for CompressedTensorsConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::CompressedTensors
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        match &self.default_scheme {
            CtScheme::W8A8Fp8 { .. } => 89,  // FP8 needs Ada/Hopper
            CtScheme::W8A16Fp8 { .. } => 75, // Marlin can handle this on Turing+
            CtScheme::W8A8Int8 { .. } => 75, // INT8 on Turing+
            CtScheme::WNA16 { .. } => 75,    // Marlin on Turing+
            CtScheme::Unquantized => 0,
        }
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.should_ignore(layer_name)
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        match &self.default_scheme {
            CtScheme::W8A8Fp8 {
                is_static_input,
                weight_block_size,
            } => {
                let fp8_config = Fp8Config {
                    is_checkpoint_fp8_serialized: true,
                    activation_scheme: if *is_static_input {
                        ActivationScheme::Static
                    } else {
                        ActivationScheme::Dynamic
                    },
                    ignored_layers: self.ignore.clone(),
                    weight_block_size: *weight_block_size,
                };
                fp8_config.create_linear(in_features, out_features, bias, device)
            }
            CtScheme::W8A16Fp8 { weight_block_size } => {
                // Weight-only FP8: use Fp8 with dynamic activation (no input quant)
                let fp8_config = Fp8Config {
                    is_checkpoint_fp8_serialized: true,
                    activation_scheme: ActivationScheme::Dynamic,
                    ignored_layers: self.ignore.clone(),
                    weight_block_size: *weight_block_size,
                };
                fp8_config.create_linear(in_features, out_features, bias, device)
            }
            CtScheme::W8A8Int8 { .. } => {
                // INT8 symmetric: use INT8 linear with per-channel scales
                Ok(Box::new(Int8Linear::new(
                    in_features,
                    out_features,
                    bias,
                    device,
                )?))
            }
            CtScheme::WNA16 {
                num_bits,
                group_size,
                symmetric,
                desc_act,
            } => {
                let gptq_config = GptqConfig {
                    bits: *num_bits,
                    group_size: *group_size as i64,
                    desc_act: *desc_act,
                    sym: *symmetric,
                    damp_percent: 0.01,
                    use_marlin: *num_bits == 4,
                };
                gptq_config.create_linear(in_features, out_features, bias, device)
            }
            CtScheme::Unquantized => Ok(Box::new(UnquantizedLinear::new(
                in_features,
                out_features,
                bias,
                DType::BF16,
                device,
            )?)),
        }
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── INT8 Linear (Symmetric Channel Quantization) ──────────────────────────

/// INT8 symmetric quantized linear layer.
///
/// Weights stored as i8, per-channel scale as f32.
/// Forward: dequantize weight → matmul.
/// TODO: Use INT8 GEMM CUDA kernel for actual hardware acceleration.
#[derive(Debug)]
pub struct Int8Linear {
    weight: Tensor,
    weight_scale: Option<Tensor>,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl Int8Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::zeros((out_features, in_features), DType::F32, device)?;
        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F32, device)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_scale: None,
            bias,
            in_features,
            out_features,
        })
    }
}

impl QuantizedLinear for Int8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantize: weight_f32 = weight_i8 * scale
        let weight_f32 = if let Some(scale) = &self.weight_scale {
            // scale is [out_features, 1] or [out_features]
            let scale = if scale.dims().len() == 1 {
                scale.unsqueeze(1)?
            } else {
                scale.clone()
            };
            self.weight.broadcast_mul(&scale)?
        } else {
            self.weight.clone()
        };

        let y = x.matmul(&weight_f32.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            // INT8 weights stored as i8, convert to f32 for computation
            self.weight = w.to_dtype(DType::F32)?;
        }
        if let Some(s) = weights.get("weight_scale") {
            self.weight_scale = Some(s.to_dtype(DType::F32)?);
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.to_dtype(DType::F32)?);
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::U8 // INT8 stored as U8
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp8_w8a8_config() -> HashMap<String, Value> {
        serde_json::from_str(
            r#"{
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "symmetric": true,
                            "strategy": "tensor",
                            "dynamic": false
                        },
                        "input_activations": {
                            "num_bits": 8,
                            "type": "float",
                            "symmetric": true,
                            "strategy": "tensor",
                            "dynamic": true
                        }
                    }
                },
                "ignore": ["lm_head"]
            }"#,
        )
        .unwrap()
    }

    fn make_wna16_config() -> HashMap<String, Value> {
        serde_json::from_str(
            r#"{
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": true,
                            "strategy": "group",
                            "group_size": 128
                        }
                    }
                },
                "ignore": ["lm_head"]
            }"#,
        )
        .unwrap()
    }

    fn make_int8_config() -> HashMap<String, Value> {
        serde_json::from_str(
            r#"{
                "quant_method": "compressed-tensors",
                "format": "int-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "int",
                            "symmetric": true,
                            "strategy": "channel"
                        },
                        "input_activations": {
                            "num_bits": 8,
                            "type": "int",
                            "symmetric": true,
                            "strategy": "token",
                            "dynamic": true
                        }
                    }
                },
                "ignore": []
            }"#,
        )
        .unwrap()
    }

    fn make_w8a16_fp8_config() -> HashMap<String, Value> {
        serde_json::from_str(
            r#"{
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "symmetric": true,
                            "strategy": "channel"
                        }
                    }
                },
                "ignore": ["lm_head"]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn test_parse_fp8_w8a8_config() {
        let raw = make_fp8_w8a8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        assert_eq!(config.method(), QuantizationMethod::CompressedTensors);
        assert_eq!(config.format, CompressionFormat::FloatQuantized);
        assert_eq!(config.ignore, vec!["lm_head".to_string()]);
        assert_eq!(config.groups.len(), 1);

        // Should resolve to W8A8Fp8
        match &config.default_scheme {
            CtScheme::W8A8Fp8 {
                is_static_input, ..
            } => {
                assert!(!is_static_input, "Input should be dynamic");
            }
            other => panic!("Expected W8A8Fp8, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_wna16_config() {
        let raw = make_wna16_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        match &config.default_scheme {
            CtScheme::WNA16 {
                num_bits,
                group_size,
                symmetric,
                ..
            } => {
                assert_eq!(*num_bits, 4);
                assert_eq!(*group_size, 128);
                assert!(*symmetric);
            }
            other => panic!("Expected WNA16, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_int8_config() {
        let raw = make_int8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        match &config.default_scheme {
            CtScheme::W8A8Int8 {
                strategy,
                is_static_input,
            } => {
                assert_eq!(*strategy, CtStrategy::Channel);
                assert!(!is_static_input, "Should be dynamic");
            }
            other => panic!("Expected W8A8Int8, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_w8a16_fp8_config() {
        let raw = make_w8a16_fp8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        match &config.default_scheme {
            CtScheme::W8A16Fp8 { .. } => {}
            other => panic!("Expected W8A16Fp8, got {other:?}"),
        }
    }

    #[test]
    fn test_ignore_layer() {
        let raw = make_fp8_w8a8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn test_min_capability() {
        let fp8_raw = make_fp8_w8a8_config();
        let fp8_config = CompressedTensorsConfig::from_detected(&fp8_raw);
        assert_eq!(fp8_config.min_capability(), 89);

        let wna16_raw = make_wna16_config();
        let wna16_config = CompressedTensorsConfig::from_detected(&wna16_raw);
        assert_eq!(wna16_config.min_capability(), 75);

        let int8_raw = make_int8_config();
        let int8_config = CompressedTensorsConfig::from_detected(&int8_raw);
        assert_eq!(int8_config.min_capability(), 75);
    }

    #[test]
    fn test_create_linear_unquantized_fallback() {
        let raw: HashMap<String, Value> = serde_json::from_str(
            r#"{
                "quant_method": "compressed-tensors",
                "format": "dense",
                "config_groups": {}
            }"#,
        )
        .unwrap();
        let config = CompressedTensorsConfig::from_detected(&raw);

        let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_create_int8_linear() {
        let raw = make_int8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);

        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_int8_linear_forward() {
        let mut linear = Int8Linear::new(4, 8, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::ones(&[8, 4], DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::from_vec(vec![0.5f32; 8], 8, &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);

        // Each output: sum of 4 weights (1.0) * scale (0.5) * input (1.0) = 4 * 0.5 = 2.0
        let vals: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (vals[0] - 2.0).abs() < 1e-5,
            "Expected 2.0, got {}",
            vals[0]
        );
    }

    #[test]
    fn test_quant_args_from_json() {
        let json: Value = serde_json::from_str(
            r#"{
                "num_bits": 4,
                "type": "int",
                "symmetric": false,
                "strategy": "group",
                "dynamic": false,
                "group_size": 128
            }"#,
        )
        .unwrap();

        let args = CtQuantArgs::from_json(&json).unwrap();
        assert_eq!(args.num_bits, 4);
        assert_eq!(args.quant_type, CtQuantType::Int);
        assert!(!args.symmetric);
        assert_eq!(args.strategy, CtStrategy::Group);
        assert_eq!(args.group_size, Some(128));
    }

    #[test]
    fn test_compression_format_parsing() {
        assert_eq!(
            CompressionFormat::from_str("float-quantized"),
            CompressionFormat::FloatQuantized
        );
        assert_eq!(
            CompressionFormat::from_str("pack-quantized"),
            CompressionFormat::PackQuantized
        );
        assert_eq!(
            CompressionFormat::from_str("int_quantized"),
            CompressionFormat::IntQuantized
        );
        assert_eq!(
            CompressionFormat::from_str("dense"),
            CompressionFormat::Dense
        );
        assert_eq!(
            CompressionFormat::from_str("unknown"),
            CompressionFormat::Dense
        );
    }

    #[test]
    fn test_scheme_selection_fp8_with_block() {
        let weights = CtQuantArgs {
            num_bits: 8,
            quant_type: CtQuantType::Float,
            strategy: CtStrategy::Block,
            symmetric: true,
            dynamic: false,
            group_size: None,
            block_structure: Some([128, 128]),
        };
        let input = CtQuantArgs {
            num_bits: 8,
            quant_type: CtQuantType::Float,
            strategy: CtStrategy::Token,
            symmetric: true,
            dynamic: true,
            group_size: None,
            block_structure: None,
        };

        let scheme = select_scheme(&weights, Some(&input), CompressionFormat::FloatQuantized);
        match scheme {
            CtScheme::W8A8Fp8 {
                weight_block_size, ..
            } => {
                assert_eq!(weight_block_size, Some([128, 128]));
            }
            other => panic!("Expected W8A8Fp8 with block, got {other:?}"),
        }
    }

    #[test]
    fn test_scheme_selection_wna16_channel() {
        let weights = CtQuantArgs {
            num_bits: 8,
            quant_type: CtQuantType::Int,
            strategy: CtStrategy::Channel,
            symmetric: true,
            dynamic: false,
            group_size: None,
            block_structure: None,
        };

        let scheme = select_scheme(&weights, None, CompressionFormat::PackQuantized);
        match scheme {
            CtScheme::WNA16 { group_size, .. } => {
                assert_eq!(group_size, -1, "Channel strategy → per-channel");
            }
            other => panic!("Expected WNA16 channel, got {other:?}"),
        }
    }

    #[test]
    fn test_clone_config() {
        let raw = make_fp8_w8a8_config();
        let config = CompressedTensorsConfig::from_detected(&raw);
        let cloned = config.clone_box();
        assert_eq!(cloned.method(), QuantizationMethod::CompressedTensors);
        assert_eq!(cloned.min_capability(), 89);
    }

    #[test]
    fn test_empty_config_groups() {
        let raw: HashMap<String, Value> = serde_json::from_str(r#"{"format": "dense"}"#).unwrap();
        let config = CompressedTensorsConfig::from_detected(&raw);
        assert!(config.groups.is_empty());
        assert!(matches!(config.default_scheme, CtScheme::Unquantized));
    }

    #[test]
    fn test_multiple_config_groups() {
        let raw: HashMap<String, Value> = serde_json::from_str(
            r#"{
                "format": "float-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["q_proj", "k_proj"],
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
                    },
                    "group_1": {
                        "targets": ["gate_proj"],
                        "weights": {
                            "num_bits": 8,
                            "type": "float",
                            "symmetric": true,
                            "strategy": "channel"
                        }
                    }
                },
                "ignore": []
            }"#,
        )
        .unwrap();
        let config = CompressedTensorsConfig::from_detected(&raw);
        assert_eq!(config.groups.len(), 2);
    }
}
