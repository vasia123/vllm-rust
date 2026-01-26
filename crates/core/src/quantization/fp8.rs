//! FP8 quantization configuration and layers.
//!
//! FP8 (8-bit floating point) quantization uses either E4M3 or E5M2 formats
//! for efficient inference on modern GPUs (Hopper and later).
//!
//! Supports:
//! - Static quantization with pre-computed scales
//! - Dynamic quantization with runtime scale computation

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{ActivationScheme, QuantizationConfig, QuantizationMethod, QuantizedLinear};

#[cfg(feature = "cuda-kernels")]
use super::fp8_cuda;

/// FP8 quantization configuration.
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// Whether the checkpoint was serialized in FP8 format
    pub is_checkpoint_fp8_serialized: bool,
    /// Activation quantization scheme (static or dynamic)
    pub activation_scheme: ActivationScheme,
    /// Layers to skip (not quantize)
    pub ignored_layers: Vec<String>,
    /// Block size for block-wise quantization [rows, cols]
    pub weight_block_size: Option<[usize; 2]>,
}

impl Fp8Config {
    /// Create a new FP8 config with dynamic activation scheme.
    pub fn dynamic() -> Self {
        Self {
            is_checkpoint_fp8_serialized: false,
            activation_scheme: ActivationScheme::Dynamic,
            ignored_layers: Vec::new(),
            weight_block_size: None,
        }
    }

    /// Create a new FP8 config with static activation scheme.
    pub fn static_scheme() -> Self {
        Self {
            is_checkpoint_fp8_serialized: true,
            activation_scheme: ActivationScheme::Static,
            ignored_layers: Vec::new(),
            weight_block_size: None,
        }
    }

    /// Create from detected config.
    pub fn from_detected(
        bits: Option<u32>,
        activation_scheme: Option<&str>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let _ = bits; // FP8 is always 8 bits

        let scheme = match activation_scheme {
            Some("static") => ActivationScheme::Static,
            _ => ActivationScheme::Dynamic,
        };

        let is_serialized = raw_config
            .get("is_checkpoint_fp8_serialized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let ignored = raw_config
            .get("ignored_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let block_size = raw_config
            .get("weight_block_size")
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

        Self {
            is_checkpoint_fp8_serialized: is_serialized,
            activation_scheme: scheme,
            ignored_layers: ignored,
            weight_block_size: block_size,
        }
    }

    /// Set ignored layers.
    pub fn with_ignored_layers(mut self, layers: Vec<String>) -> Self {
        self.ignored_layers = layers;
        self
    }

    /// Set block size for block-wise quantization.
    pub fn with_block_size(mut self, rows: usize, cols: usize) -> Self {
        self.weight_block_size = Some([rows, cols]);
        self
    }
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self::dynamic()
    }
}

impl QuantizationConfig for Fp8Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Fp8
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        89 // Hopper (H100) required for native FP8
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
        Ok(Box::new(Fp8Linear::new(
            in_features,
            out_features,
            bias,
            self.activation_scheme,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// FP8 quantized linear layer.
///
/// When the `cuda-kernels` feature is enabled and running on CUDA,
/// this uses fused FP8 dequantization + GEMM kernels for efficient inference.
/// Otherwise, it falls back to standard BF16/F16 matmul.
#[derive(Debug)]
pub struct Fp8Linear {
    /// Weight tensor - stored in FP8 (U8) format on CUDA, or compute dtype on CPU
    weight: Tensor,
    /// Optional bias
    bias: Option<Tensor>,
    /// Weight scale for dequantization [1] for per-tensor, [N] for per-channel
    weight_scale: Option<Tensor>,
    /// Input scale (for static quantization)
    input_scale: Option<Tensor>,
    /// Activation scheme (static or dynamic quantization)
    #[allow(dead_code)] // Used for future W8A8 support
    activation_scheme: ActivationScheme,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Whether weights are stored in FP8 format
    is_fp8_weights: bool,
}

impl Fp8Linear {
    /// Create a new FP8 linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        activation_scheme: ActivationScheme,
        device: &Device,
    ) -> Result<Self> {
        // Initialize with BF16 zeros (actual FP8 weights loaded later)
        let weight = Tensor::zeros((out_features, in_features), DType::BF16, device)?;
        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::BF16, device)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            weight_scale: None,
            input_scale: None,
            activation_scheme,
            in_features,
            out_features,
            is_fp8_weights: false,
        })
    }

    /// Check if this layer can use FP8 CUDA kernels.
    #[cfg(feature = "cuda-kernels")]
    fn can_use_fp8_kernel(&self) -> bool {
        // Requires: CUDA device, FP8 weights, and weight scale
        self.is_fp8_weights && self.weight_scale.is_some() && self.weight.device().is_cuda()
    }

    /// Perform FP8 forward pass using CUDA kernels.
    #[cfg(feature = "cuda-kernels")]
    fn forward_fp8(&self, x: &Tensor) -> Result<Tensor> {
        let weight_scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("FP8 forward requires weight scale".to_string())
        })?;

        // Use FP8 GEMM: output = x @ (weight_fp8 * scale).T
        fp8_cuda::fp8_gemm(x, &self.weight, weight_scale, self.bias.as_ref())
    }

    /// Quantize BF16/F16 weights to FP8 format.
    ///
    /// This converts the current weights to FP8 E4M3 format with per-tensor
    /// scaling. Requires CUDA device.
    #[cfg(feature = "cuda-kernels")]
    pub fn quantize_weights_to_fp8(&mut self) -> Result<()> {
        if self.is_fp8_weights {
            return Ok(()); // Already FP8
        }

        if !self.weight.device().is_cuda() {
            candle_core::bail!("FP8 quantization requires CUDA device");
        }

        // Convert to BF16 if not already
        let weight_bf16 = if self.weight.dtype() == DType::BF16 {
            self.weight.clone()
        } else {
            self.weight.to_dtype(DType::BF16)?
        };

        // Compute per-tensor scale: max(abs(weight)) / 448.0 (FP8 E4M3 max)
        let abs_weight = weight_bf16.abs()?.to_dtype(DType::F32)?;
        let max_val = abs_weight.max(0)?.max(0)?; // Scalar
        let scale = (max_val / 448.0f64)?.maximum(1e-12)?;

        // Quantize using CUDA kernel
        let fp8_weight = fp8_cuda::fp8_quantize_static(&weight_bf16, &scale)?;

        self.weight = fp8_weight;
        self.weight_scale = Some(scale);
        self.is_fp8_weights = true;

        Ok(())
    }

    /// Check if weights are in FP8 format.
    pub fn is_fp8(&self) -> bool {
        self.is_fp8_weights
    }

    /// Get the weight scale (if quantized).
    pub fn weight_scale(&self) -> Option<&Tensor> {
        self.weight_scale.as_ref()
    }
}

impl QuantizedLinear for Fp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use FP8 CUDA kernels when available
        #[cfg(feature = "cuda-kernels")]
        if self.can_use_fp8_kernel() {
            return self.forward_fp8(x);
        }

        // Fallback: standard matmul (BF16/F16/F32)
        // This is used when:
        // - Running on CPU
        // - FP8 weights not loaded
        // - cuda-kernels feature not enabled
        let y = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            // FP8 weights are stored as U8 (E4M3 format)
            self.is_fp8_weights = w.dtype() == DType::U8;
            self.weight = w.clone();
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        if let Some(s) = weights.get("weight_scale") {
            self.weight_scale = Some(s.clone());
        }
        if let Some(s) = weights.get("input_scale") {
            self.input_scale = Some(s.clone());
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
    fn test_fp8_config_dynamic() {
        let config = Fp8Config::dynamic();
        assert_eq!(config.activation_scheme, ActivationScheme::Dynamic);
        assert!(!config.is_checkpoint_fp8_serialized);
        assert_eq!(config.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_fp8_config_static() {
        let config = Fp8Config::static_scheme();
        assert_eq!(config.activation_scheme, ActivationScheme::Static);
        assert!(config.is_checkpoint_fp8_serialized);
    }

    #[test]
    fn test_fp8_config_min_capability() {
        let config = Fp8Config::default();
        assert_eq!(config.min_capability(), 89); // Hopper
    }

    #[test]
    fn test_fp8_config_ignored_layers() {
        let config = Fp8Config::default().with_ignored_layers(vec!["lm_head".to_string()]);
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn test_fp8_linear_creation() {
        let config = Fp8Config::default();
        let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_fp8_linear_forward() {
        // Use F32 for CPU testing (BF16 matmul not supported on CPU)
        let mut linear =
            Fp8Linear::new(4, 8, false, ActivationScheme::Dynamic, &Device::Cpu).unwrap();
        // Override with F32 weights for CPU compatibility
        linear.weight = Tensor::zeros((8, 4), DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
    }

    #[test]
    fn test_fp8_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert(
            "is_checkpoint_fp8_serialized".to_string(),
            serde_json::json!(true),
        );
        raw.insert(
            "ignored_layers".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );

        let config = Fp8Config::from_detected(Some(8), Some("static"), &raw);

        assert_eq!(config.activation_scheme, ActivationScheme::Static);
        assert!(config.is_checkpoint_fp8_serialized);
        assert_eq!(config.ignored_layers.len(), 2);
    }

    // GPU tests - only run when cuda-kernels feature is enabled
    #[cfg(feature = "cuda-kernels")]
    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_fp8_linear_quantize_weights() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let mut linear =
                Fp8Linear::new(64, 128, false, ActivationScheme::Dynamic, &device).unwrap();

            // Set random weights
            let weight = Tensor::randn(0.0f32, 0.1, (128, 64), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            linear.weight = weight;

            // Quantize to FP8
            assert!(!linear.is_fp8());
            linear.quantize_weights_to_fp8().unwrap();
            assert!(linear.is_fp8());
            assert!(linear.weight_scale().is_some());
            assert_eq!(linear.weight.dtype(), DType::U8);
        }

        #[test]
        fn test_fp8_linear_forward_gpu() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let mut linear =
                Fp8Linear::new(64, 128, false, ActivationScheme::Dynamic, &device).unwrap();

            // Initialize with random BF16 weights
            let weight = Tensor::randn(0.0f32, 0.1, (128, 64), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            linear.weight = weight.clone();

            // Get reference output before quantization
            let x = Tensor::randn(0.0f32, 1.0, (4, 64), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let ref_output = linear.forward(&x).unwrap();

            // Quantize and run FP8 forward
            linear.quantize_weights_to_fp8().unwrap();
            let fp8_output = linear.forward(&x).unwrap();

            // Verify output shape
            assert_eq!(fp8_output.dims(), ref_output.dims());
            assert_eq!(fp8_output.dims(), &[4, 128]);

            // FP8 output should be close to reference (within quantization error)
            let diff = (&fp8_output.to_dtype(DType::F32).unwrap()
                - &ref_output.to_dtype(DType::F32).unwrap())
                .unwrap()
                .abs()
                .unwrap();
            let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();

            // Allow some quantization error (FP8 has limited precision)
            assert!(
                max_diff < 0.5,
                "FP8 output differs too much from reference: max_diff = {max_diff}"
            );
        }

        #[test]
        fn test_fp8_linear_forward_with_bias() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let mut linear =
                Fp8Linear::new(32, 64, true, ActivationScheme::Dynamic, &device).unwrap();

            // Set weights and bias
            let weight = Tensor::randn(0.0f32, 0.1, (64, 32), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let bias = Tensor::randn(0.0f32, 0.1, 64, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            linear.weight = weight;
            linear.bias = Some(bias);

            // Quantize and forward
            linear.quantize_weights_to_fp8().unwrap();

            let x = Tensor::randn(0.0f32, 1.0, (2, 32), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = linear.forward(&x).unwrap();

            assert_eq!(y.dims(), &[2, 64]);
        }
    }
}
