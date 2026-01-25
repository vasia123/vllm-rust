//! Weight loaders for different quantization methods.
//!
//! This module provides traits and implementations for loading quantized
//! weights from HuggingFace model checkpoints (safetensors format).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::awq::{AwqConfig, AwqLinear};
use super::config::{QuantizationConfig, QuantizedLinear};
use super::fp8::{Fp8Config, Fp8Linear};
use super::gptq::{GptqConfig, GptqLinear};
use super::{create_config_from_directory, DetectedQuantConfig, QuantizationMethod};

/// Trait for loading quantized weights from safetensors.
///
/// Each quantization method has a specific weight naming convention:
/// - Unquantized: `{prefix}.weight`, `{prefix}.bias`
/// - GPTQ: `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`, `{prefix}.g_idx`
/// - FP8: `{prefix}.weight` (U8), `{prefix}.weight_scale`, `{prefix}.input_scale`
/// - AWQ: `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`
pub trait QuantizedWeightLoader: Send + Sync {
    /// Load a quantized linear layer from the checkpoint.
    ///
    /// # Arguments
    /// * `prefix` - Weight name prefix (e.g., "model.layers.0.self_attn.q_proj")
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `bias` - Whether the layer has bias
    ///
    /// # Returns
    /// A boxed `QuantizedLinear` implementation with weights loaded
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>>;

    /// Get the quantization method this loader handles.
    fn method(&self) -> QuantizationMethod;

    /// Get the device for this loader.
    fn device(&self) -> &Device;

    /// Get the compute dtype for activations.
    fn dtype(&self) -> DType;
}

/// Weight loader for unquantized (full precision) models.
pub struct UnquantizedWeightLoader {
    vb: VarBuilder<'static>,
    device: Device,
    dtype: DType,
}

impl UnquantizedWeightLoader {
    /// Create a new unquantized weight loader.
    pub fn new(vb: VarBuilder<'static>) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self { vb, device, dtype }
    }
}

impl QuantizedWeightLoader for UnquantizedWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // Load weight tensor
        let weight = vb.get((out_features, in_features), "weight")?;

        // Optionally load bias
        let bias_tensor = if bias {
            Some(vb.get(out_features, "bias")?)
        } else {
            None
        };

        Ok(Box::new(LoadedUnquantizedLinear {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        }))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::None
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Unquantized linear layer with pre-loaded weights.
#[derive(Debug)]
struct LoadedUnquantizedLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear for LoadedUnquantizedLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        // Weights already loaded during construction
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

/// Weight loader for GPTQ quantized models.
pub struct GptqWeightLoader {
    vb: VarBuilder<'static>,
    config: GptqConfig,
    device: Device,
    dtype: DType,
}

impl GptqWeightLoader {
    /// Create a new GPTQ weight loader.
    pub fn new(vb: VarBuilder<'static>, config: GptqConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for GptqWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // Create GPTQ linear layer
        let mut linear = GptqLinear::new(
            in_features,
            out_features,
            bias,
            self.config.bits,
            self.config.group_size,
            &self.device,
        )?;

        // Calculate packed dimensions for GPTQ
        let pack_factor = 32 / self.config.bits as usize;
        let packed_in = in_features.div_ceil(pack_factor);
        let num_groups = if self.config.group_size <= 0 {
            1
        } else {
            in_features.div_ceil(self.config.group_size as usize)
        };

        // Load quantized weights
        let mut weights = HashMap::new();

        // Try to load qweight - GPTQ stores as (in_features/pack_factor, out_features)
        if let Ok(qweight) = vb.get((packed_in, out_features), "qweight") {
            weights.insert("qweight".to_string(), qweight);
        }

        // Load scales
        if let Ok(scales) = vb.get((num_groups, out_features), "scales") {
            weights.insert("scales".to_string(), scales);
        }

        // Load qzeros - packed format
        let packed_out = out_features.div_ceil(pack_factor);
        if let Ok(qzeros) = vb.get((num_groups, packed_out), "qzeros") {
            weights.insert("qzeros".to_string(), qzeros);
        }

        // Optional g_idx for desc_act
        if self.config.desc_act {
            if let Ok(g_idx) = vb.get(in_features, "g_idx") {
                weights.insert("g_idx".to_string(), g_idx);
            }
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gptq
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for FP8 quantized models.
pub struct Fp8WeightLoader {
    vb: VarBuilder<'static>,
    config: Fp8Config,
    device: Device,
    dtype: DType,
}

impl Fp8WeightLoader {
    /// Create a new FP8 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: Fp8Config) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for Fp8WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Check if layer should be skipped (not quantized)
        if self.config.is_layer_skipped(prefix) {
            // Fall back to unquantized
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);

        // Create FP8 linear layer
        let mut linear = Fp8Linear::new(
            in_features,
            out_features,
            bias,
            self.config.activation_scheme,
            &self.device,
        )?;

        let mut weights = HashMap::new();

        // FP8 weights are stored as U8 (E4M3 format) - (out_features, in_features)
        if let Ok(weight) = vb.get((out_features, in_features), "weight") {
            weights.insert("weight".to_string(), weight);
        }

        // Weight scale for dequantization
        if let Ok(scale) = vb.get((), "weight_scale") {
            weights.insert("weight_scale".to_string(), scale);
        } else if let Ok(scale) = vb.get(1, "weight_scale") {
            // Per-tensor scale stored as 1D
            weights.insert("weight_scale".to_string(), scale.squeeze(0)?);
        }

        // Input scale for static quantization
        if let Ok(scale) = vb.get((), "input_scale") {
            weights.insert("input_scale".to_string(), scale);
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Fp8
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for AWQ quantized models.
pub struct AwqWeightLoader {
    vb: VarBuilder<'static>,
    config: AwqConfig,
    device: Device,
    dtype: DType,
}

impl AwqWeightLoader {
    /// Create a new AWQ weight loader.
    pub fn new(vb: VarBuilder<'static>, config: AwqConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for AwqWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // Create AWQ linear layer
        let mut linear = AwqLinear::new(
            in_features,
            out_features,
            bias,
            self.config.bits,
            self.config.group_size,
            &self.device,
        )?;

        // Calculate packed dimensions for AWQ (same as GPTQ)
        let pack_factor = 32 / self.config.bits as usize;
        let packed_in = in_features.div_ceil(pack_factor);
        let num_groups = if self.config.group_size <= 0 {
            1
        } else {
            in_features.div_ceil(self.config.group_size as usize)
        };

        let mut weights = HashMap::new();

        // Load qweight
        if let Ok(qweight) = vb.get((packed_in, out_features), "qweight") {
            weights.insert("qweight".to_string(), qweight);
        }

        // Load scales
        if let Ok(scales) = vb.get((num_groups, out_features), "scales") {
            weights.insert("scales".to_string(), scales);
        }

        // Load qzeros - packed format
        let packed_out = out_features.div_ceil(pack_factor);
        if let Ok(qzeros) = vb.get((num_groups, packed_out), "qzeros") {
            weights.insert("qzeros".to_string(), qzeros);
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Awq
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Create an appropriate weight loader for a model directory.
///
/// This function detects the quantization method from the model config
/// and returns the corresponding weight loader.
pub fn create_weight_loader(
    model_dir: &Path,
    vb: VarBuilder<'static>,
) -> Box<dyn QuantizedWeightLoader> {
    let config = create_config_from_directory(model_dir);
    create_weight_loader_from_config(vb, config)
}

/// Create a weight loader from a detected quantization config.
pub fn create_weight_loader_from_detected(
    vb: VarBuilder<'static>,
    detected: &DetectedQuantConfig,
) -> Box<dyn QuantizedWeightLoader> {
    let config = super::create_config(detected);
    create_weight_loader_from_config(vb, config)
}

/// Create a weight loader from a quantization config.
pub fn create_weight_loader_from_config(
    vb: VarBuilder<'static>,
    config: Box<dyn QuantizationConfig>,
) -> Box<dyn QuantizedWeightLoader> {
    match config.method() {
        QuantizationMethod::Gptq => {
            // Downcast to GptqConfig
            let gptq_config = GptqConfig::from_detected(None, None, None, &HashMap::new());
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        QuantizationMethod::Fp8 => {
            let fp8_config = Fp8Config::default();
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        QuantizationMethod::Awq => {
            let awq_config = AwqConfig::default();
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        _ => Box::new(UnquantizedWeightLoader::new(vb)),
    }
}

/// Create a weight loader with explicit config parameters.
pub fn create_weight_loader_with_params(
    vb: VarBuilder<'static>,
    detected: &DetectedQuantConfig,
) -> Box<dyn QuantizedWeightLoader> {
    match detected.method {
        QuantizationMethod::Gptq => {
            let gptq_config = GptqConfig::from_detected(
                detected.bits,
                detected.group_size,
                detected.desc_act,
                &detected.raw_config,
            );
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        QuantizationMethod::Fp8 => {
            let fp8_config = Fp8Config::from_detected(
                detected.bits,
                detected.activation_scheme.as_deref(),
                &detected.raw_config,
            );
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        QuantizationMethod::Awq => {
            let awq_config = AwqConfig::from_detected(
                detected.bits,
                detected.group_size,
                &detected.raw_config,
            );
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        _ => Box::new(UnquantizedWeightLoader::new(vb)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unquantized_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = UnquantizedWeightLoader::new(vb);
        assert_eq!(loader.method(), QuantizationMethod::None);
    }

    #[test]
    fn test_gptq_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = GptqConfig::int4(128);
        let loader = GptqWeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_fp8_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = Fp8Config::dynamic();
        let loader = Fp8WeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_create_weight_loader_default() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_from_detected(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::None);
    }

    #[test]
    fn test_create_weight_loader_gptq() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_create_weight_loader_fp8() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Fp8,
            bits: Some(8),
            group_size: None,
            desc_act: None,
            activation_scheme: Some("dynamic".to_string()),
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_awq_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = AwqConfig::int4(128);
        let loader = AwqWeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_create_weight_loader_awq() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Awq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_loaded_unquantized_linear_forward() {
        let device = Device::Cpu;
        let weight = Tensor::ones((8, 4), DType::F32, &device).unwrap();
        let linear = LoadedUnquantizedLinear {
            weight,
            bias: None,
            in_features: 4,
            out_features: 8,
        };

        let x = Tensor::ones((2, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
        assert_eq!(linear.in_features(), 4);
        assert_eq!(linear.out_features(), 8);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_loaded_unquantized_linear_with_bias() {
        let device = Device::Cpu;
        let weight = Tensor::ones((8, 4), DType::F32, &device).unwrap();
        let bias = Tensor::ones(8, DType::F32, &device).unwrap();
        let linear = LoadedUnquantizedLinear {
            weight,
            bias: Some(bias),
            in_features: 4,
            out_features: 8,
        };

        let x = Tensor::ones((2, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
        assert!(linear.has_bias());
    }
}
