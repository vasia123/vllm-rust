//! GGUF format support for loading quantized models.
//!
//! GGUF (GGML Universal Format) is a file format for storing quantized LLMs.
//! It is used by llama.cpp and other inference engines.
//!
//! # Supported quantization types
//!
//! - F32, F16, BF16 - Unquantized
//! - Q4_0, Q4_1 - 4-bit quantization
//! - Q5_0, Q5_1 - 5-bit quantization
//! - Q8_0, Q8_1 - 8-bit quantization
//! - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K - K-quant formats
//!
//! # References
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

mod dequant;
mod parser;

pub use dequant::{dequantize, GgmlType};
pub use parser::{GgufFile, GgufMetadata, GgufTensorInfo, GgufValue};

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::weight_loader::QuantizedWeightLoader;

/// GGUF quantization configuration.
#[derive(Debug, Clone, Default)]
pub struct GgufConfig {
    /// List of layers to skip quantization (use full precision)
    pub ignored_layers: Vec<String>,
}

impl GgufConfig {
    /// Create a new GGUF config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add layers to skip quantization.
    pub fn with_ignored_layers(mut self, layers: Vec<String>) -> Self {
        self.ignored_layers = layers;
        self
    }
}

impl QuantizationConfig for GgufConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        // GGUF dequantization produces F16 internally
        &[DType::F16, DType::BF16, DType::F32]
    }

    fn min_capability(&self) -> u32 {
        // GGUF can run on CPU (capability 0)
        0
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.ignored_layers
            .iter()
            .any(|ignored| layer_name.contains(ignored))
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(GgufLinear::new(
            in_features,
            out_features,
            bias,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// GGUF quantized linear layer.
///
/// Stores quantized weights and dequantizes them on-the-fly during forward pass.
#[derive(Debug)]
pub struct GgufLinear {
    /// Quantized weight data (raw bytes)
    qweight: Option<Tensor>,
    /// Weight quantization type
    qtype: GgmlType,
    /// Bias tensor (always unquantized)
    bias: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Device (stored for potential future use in dequantization)
    #[allow(dead_code)]
    device: Device,
}

impl GgufLinear {
    /// Create a new GGUF linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F16, device)?)
        } else {
            None
        };

        Ok(Self {
            qweight: None,
            qtype: GgmlType::F16,
            bias,
            in_features,
            out_features,
            device: device.clone(),
        })
    }

    /// Set the quantized weight data.
    pub fn set_qweight(&mut self, qweight: Tensor, qtype: GgmlType) {
        self.qweight = Some(qweight);
        self.qtype = qtype;
    }

    /// Get the quantization type.
    pub fn qtype(&self) -> GgmlType {
        self.qtype
    }

    /// Check if weights are loaded.
    pub fn has_weights(&self) -> bool {
        self.qweight.is_some()
    }
}

impl QuantizedLinear for GgufLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let qweight = self.qweight.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "GGUF layer has no weights loaded - call load_weights() first".to_string(),
            )
        })?;

        // Dequantize weights to F16
        let weight = dequantize(qweight, self.qtype, self.out_features, self.in_features)?;

        // Compute matmul: y = x @ W^T
        let y = x.to_dtype(DType::F16)?.matmul(&weight.t()?)?;

        // Add bias if present
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(qweight) = weights.get("qweight") {
            self.qweight = Some(qweight.clone());
        }
        if let Some(qtype_tensor) = weights.get("qtype") {
            // qtype is stored as a scalar u8
            let qtype_val = qtype_tensor.to_vec0::<u8>()?;
            self.qtype = GgmlType::from_u32(qtype_val as u32);
        }
        if let Some(bias) = weights.get("bias") {
            self.bias = Some(bias.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        // GGUF stores quantized weights as raw bytes (U8)
        DType::U8
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

/// Weight loader for GGUF files.
pub struct GgufWeightLoader {
    /// Parsed GGUF file
    gguf: GgufFile,
    /// Device to load tensors to
    device: Device,
    /// Compute dtype for activations
    dtype: DType,
    /// Config
    config: GgufConfig,
}

impl GgufWeightLoader {
    /// Create a new GGUF weight loader from a file path.
    pub fn from_path(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {e}")))?;

        Ok(Self {
            gguf,
            device,
            dtype,
            config: GgufConfig::default(),
        })
    }

    /// Create with a specific config.
    pub fn with_config(mut self, config: GgufConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the GGUF file metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        self.gguf.metadata()
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Result<(Tensor, GgmlType)> {
        self.gguf.load_tensor(name, &self.device)
    }

    /// Get model architecture from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.gguf
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Get number of layers from metadata.
    pub fn num_layers(&self) -> Option<u32> {
        // Try common metadata keys
        let arch = self.architecture()?;
        let key = format!("{arch}.block_count");
        self.gguf.metadata().get(&key).and_then(|v| v.as_u32())
    }

    /// Get hidden size from metadata.
    pub fn hidden_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.embedding_length");
        self.gguf.metadata().get(&key).and_then(|v| v.as_u32())
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.gguf.tensor_names()
    }
}

impl QuantizedWeightLoader for GgufWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let mut linear = GgufLinear::new(in_features, out_features, bias, &self.device)?;

        // GGUF tensor naming follows llama.cpp conventions
        // Try different naming patterns
        let weight_name = format!("{prefix}.weight");

        if let Ok((qweight, qtype)) = self.get_tensor(&weight_name) {
            linear.set_qweight(qweight, qtype);
        } else {
            // Try without .weight suffix (some GGUF files omit it)
            if let Ok((qweight, qtype)) = self.get_tensor(prefix) {
                linear.set_qweight(qweight, qtype);
            }
        }

        // Load bias if present
        if bias {
            let bias_name = format!("{prefix}.bias");
            if let Ok((bias_tensor, _)) = self.get_tensor(&bias_name) {
                // Bias is usually unquantized, convert to F16
                let bias_f16 = bias_tensor.to_dtype(DType::F16)?;
                linear.bias = Some(bias_f16);
            }
        }

        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_config_default() {
        let config = GgufConfig::default();
        assert_eq!(config.method(), QuantizationMethod::Gguf);
        assert_eq!(config.min_capability(), 0);
        assert!(!config.is_layer_skipped("some_layer"));
    }

    #[test]
    fn test_gguf_config_ignored_layers() {
        let config = GgufConfig::new().with_ignored_layers(vec!["lm_head".to_string()]);
        assert!(config.is_layer_skipped("lm_head"));
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("self_attn"));
    }

    #[test]
    fn test_gguf_linear_creation() {
        let config = GgufConfig::default();
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_gguf_linear_validation() {
        let result = GgufLinear::new(0, 128, false, &Device::Cpu);
        assert!(result.is_err());

        let result = GgufLinear::new(64, 0, false, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_linear_forward_requires_weights() {
        let linear = GgufLinear::new(64, 128, false, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[2, 64], DType::F16, &Device::Cpu).unwrap();

        let result = linear.forward(&x);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no weights loaded"));
    }

    #[test]
    fn test_ggml_type_roundtrip() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8), GgmlType::Q8_0);
    }
}
