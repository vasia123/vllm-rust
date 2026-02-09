//! GPTQ quantization configuration and layers.
//!
//! GPTQ (Gradient-based Post-Training Quantization) is a popular method
//! for quantizing LLM weights to INT4/INT8 with grouping.
//!
//! Supports:
//! - 4-bit and 8-bit quantization
//! - Group-wise quantization with configurable group size
//! - Desc_act (descending activation order) optimization
//! - Fused INT4 GEMM kernel acceleration
//! - Automatic Marlin kernel selection when conditions are met

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::marlin::{
    check_marlin_supports_shape, MarlinConfig, MarlinLinear, MarlinScalarType,
    MARLIN_SUPPORTED_GROUP_SIZES,
};

#[cfg(feature = "cuda-kernels")]
use super::gptq_cuda;

/// GPTQ quantization configuration.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    /// Quantization bits (4 or 8)
    pub bits: u32,
    /// Group size for quantization (-1 for per-channel)
    pub group_size: i64,
    /// Whether to use descending activation order
    pub desc_act: bool,
    /// Whether quantization is symmetric
    pub sym: bool,
    /// Dampening percentage for GPTQ algorithm
    pub damp_percent: f64,
    /// Use Marlin optimized kernels when available
    pub use_marlin: bool,
}

impl GptqConfig {
    /// Create a new GPTQ config with 4-bit quantization.
    pub fn int4(group_size: i64) -> Self {
        Self {
            bits: 4,
            group_size,
            desc_act: false,
            sym: true,
            damp_percent: 0.01,
            use_marlin: true,
        }
    }

    /// Create a new GPTQ config with 8-bit quantization.
    pub fn int8(group_size: i64) -> Self {
        Self {
            bits: 8,
            group_size,
            desc_act: false,
            sym: true,
            damp_percent: 0.01,
            use_marlin: false,
        }
    }

    /// Create from detected config.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        desc_act: Option<bool>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let bits = bits.unwrap_or(4);
        let group_size = group_size.map(|g| g as i64).unwrap_or(128);
        let desc_act = desc_act.unwrap_or(false);

        let sym = raw_config
            .get("sym")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let damp_percent = raw_config
            .get("damp_percent")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.01);

        // Check if checkpoint is in Marlin format
        let is_marlin = raw_config
            .get("is_marlin_format")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            bits,
            group_size,
            desc_act,
            sym,
            damp_percent,
            use_marlin: is_marlin || bits == 4, // Marlin only supports 4-bit
        }
    }

    /// Set desc_act optimization.
    pub fn with_desc_act(mut self, enabled: bool) -> Self {
        self.desc_act = enabled;
        self
    }

    /// Enable/disable Marlin kernels.
    pub fn with_marlin(mut self, enabled: bool) -> Self {
        self.use_marlin = enabled;
        self
    }

    /// Calculate number of groups for a given input size.
    pub fn num_groups(&self, in_features: usize) -> usize {
        if self.group_size <= 0 {
            1 // Per-channel quantization
        } else {
            in_features.div_ceil(self.group_size as usize)
        }
    }

    /// Check if Marlin kernel can be used for this config.
    ///
    /// Marlin supports:
    /// - 4-bit quantization (uint4b8)
    /// - 8-bit quantization (uint8b128)
    /// - Group sizes: -1, 32, 64, 128
    /// - GPU compute capability >= 8.0 (Ampere)
    pub fn can_use_marlin(&self) -> bool {
        // Check bits
        if self.bits != 4 && self.bits != 8 {
            return false;
        }

        // Check group size
        if !MARLIN_SUPPORTED_GROUP_SIZES.contains(&(self.group_size as i32)) {
            return false;
        }

        // Only symmetric quantization supported by Marlin GPTQ path
        if !self.sym {
            return false;
        }

        true
    }

    /// Check if Marlin supports a specific layer shape.
    pub fn can_use_marlin_for_shape(&self, in_features: usize, out_features: usize) -> bool {
        if !self.can_use_marlin() {
            return false;
        }

        check_marlin_supports_shape(out_features, in_features, self.group_size as i32).is_ok()
    }

    /// Convert to MarlinConfig if Marlin is supported.
    pub fn to_marlin_config(&self) -> Option<MarlinConfig> {
        if !self.can_use_marlin() {
            return None;
        }

        let scalar_type = match (self.bits, self.sym) {
            (4, true) => MarlinScalarType::Uint4b8,
            (8, true) => MarlinScalarType::Uint8b128,
            _ => return None,
        };

        Some(MarlinConfig {
            bits: self.bits,
            group_size: self.group_size as i32,
            desc_act: self.desc_act,
            is_sym: self.sym,
            scalar_type,
            use_fp32_reduce: true,
            is_act_and_mul: true,
        })
    }
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self::int4(128)
    }
}

impl QuantizationConfig for GptqConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gptq
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        if self.use_marlin {
            80 // Ampere required for Marlin
        } else {
            70 // Volta for basic GPTQ
        }
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false // GPTQ quantizes all linear layers
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Try to use Marlin if conditions are met
        if self.use_marlin && self.can_use_marlin_for_shape(in_features, out_features) {
            if let Some(marlin_config) = self.to_marlin_config() {
                return Ok(Box::new(MarlinLinear::new(
                    in_features,
                    out_features,
                    bias,
                    marlin_config,
                    device,
                )?));
            }
        }

        // Fallback to standard GPTQ
        Ok(Box::new(GptqLinear::new(
            in_features,
            out_features,
            bias,
            self.bits,
            self.group_size,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// GPTQ quantized linear layer.
///
/// Stores quantized weights in packed INT32 format with scales and zeros.
/// When the `cuda-kernels` feature is enabled and running on CUDA, uses
/// fused INT4 GEMM kernels for efficient inference.
#[derive(Debug)]
pub struct GptqLinear {
    /// Quantized weights packed into INT32 [K/pack_factor, N]
    qweight: Tensor,
    /// Quantization scales per group [num_groups, N]
    scales: Tensor,
    /// Zero points per group packed [num_groups, N/pack_factor]
    qzeros: Tensor,
    /// Optional bias [N]
    bias: Option<Tensor>,
    /// G_idx for desc_act (optional)
    g_idx: Option<Tensor>,
    /// Bits (4 or 8)
    #[allow(dead_code)] // Used for kernel dispatch in future CUDA implementation
    bits: u32,
    /// Group size
    #[allow(dead_code)] // Used for kernel dispatch in future CUDA implementation
    group_size: i64,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Whether weights are loaded in quantized format
    is_quantized: bool,
}

impl GptqLinear {
    /// Create a new GPTQ linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        bits: u32,
        group_size: i64,
        device: &Device,
    ) -> Result<Self> {
        // Validate inputs
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }
        if bits != 4 && bits != 8 {
            candle_core::bail!("GPTQ only supports 4-bit or 8-bit quantization, got {bits}");
        }

        // Calculate packed dimensions
        let pack_factor = 32 / bits as usize;
        let packed_in = in_features.div_ceil(pack_factor);

        let num_groups = if group_size <= 0 {
            1
        } else {
            in_features.div_ceil(group_size as usize)
        };

        // Initialize with zeros (weights loaded later)
        let qweight = Tensor::zeros((packed_in, out_features), DType::U32, device)?;
        let scales = Tensor::zeros((num_groups, out_features), DType::F16, device)?;
        let qzeros = Tensor::zeros(
            (num_groups, out_features.div_ceil(pack_factor)),
            DType::U32,
            device,
        )?;

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F16, device)?)
        } else {
            None
        };

        Ok(Self {
            qweight,
            scales,
            qzeros,
            bias,
            g_idx: None,
            bits,
            group_size,
            in_features,
            out_features,
            is_quantized: false,
        })
    }

    /// Check if this layer can use GPTQ CUDA kernels.
    #[cfg(feature = "cuda-kernels")]
    fn can_use_gptq_kernel(&self) -> bool {
        self.is_quantized
            && self.bits == 4 // Currently only INT4 GEMM is implemented
            && self.qweight.device().is_cuda()
    }

    /// Perform GPTQ forward pass using CUDA kernels.
    #[cfg(feature = "cuda-kernels")]
    fn forward_gptq(&self, x: &Tensor) -> Result<Tensor> {
        // Convert scales to BF16 if needed
        let scales_bf16 = if self.scales.dtype() == DType::BF16 {
            self.scales.clone()
        } else {
            self.scales.to_dtype(DType::BF16)?
        };

        // Use fused GPTQ GEMM
        gptq_cuda::gptq_gemm(
            x,
            &self.qweight,
            &scales_bf16,
            &self.qzeros,
            self.bias.as_ref(),
            self.group_size as i32,
        )
    }

    /// Dequantize weights to full precision for computation.
    /// Requires CUDA device with cuda-kernels feature for quantized weights.
    /// Returns weights in shape [out_features, in_features] for matmul.
    fn dequantize(&self) -> Result<Tensor> {
        if !self.is_quantized {
            // Layer not yet initialized with quantized weights - weights must be
            // loaded via load_weights() before inference
            candle_core::bail!(
                "GPTQ layer has no quantized weights loaded - call load_weights() first"
            );
        }

        #[cfg(feature = "cuda-kernels")]
        if self.qweight.device().is_cuda() {
            let scales_bf16 = if self.scales.dtype() == DType::BF16 {
                self.scales.clone()
            } else {
                self.scales.to_dtype(DType::BF16)?
            };
            // CUDA kernel returns [in_features, out_features], transpose for matmul
            let weight = gptq_cuda::gptq_dequantize(
                &self.qweight,
                &scales_bf16,
                &self.qzeros,
                self.in_features,
                self.out_features,
                self.group_size as i32,
                self.bits,
            )?;
            return weight.t()?.contiguous();
        }

        // GPTQ dequantization requires CUDA - CPU inference not supported for quantized models
        candle_core::bail!(
            "GPTQ dequantization requires CUDA device with cuda-kernels feature enabled"
        )
    }

    /// Check if weights are quantized.
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }
}

impl QuantizedLinear for GptqLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use GPTQ CUDA kernels when available
        #[cfg(feature = "cuda-kernels")]
        if self.can_use_gptq_kernel() {
            // Convert input to BF16 if needed
            let x_bf16 = if x.dtype() == DType::BF16 {
                x.clone()
            } else {
                x.to_dtype(DType::BF16)?
            };
            return self.forward_gptq(&x_bf16);
        }

        // Fallback: dequantize and compute
        let weight = self.dequantize()?;
        let y = x.to_dtype(DType::F16)?.matmul(&weight.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("qweight") {
            // Check if weights are in quantized format (U32 packed)
            self.is_quantized = w.dtype() == DType::U32;
            self.qweight = w.clone();
        }
        if let Some(s) = weights.get("scales") {
            self.scales = s.clone();
        }
        if let Some(z) = weights.get("qzeros") {
            self.qzeros = z.clone();
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        if let Some(g) = weights.get("g_idx") {
            self.g_idx = Some(g.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::U32 // Packed format
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
    fn test_gptq_config_int4() {
        let config = GptqConfig::int4(128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_gptq_config_int8() {
        let config = GptqConfig::int8(-1);
        assert_eq!(config.bits, 8);
        assert_eq!(config.group_size, -1);
    }

    #[test]
    fn test_gptq_config_min_capability() {
        let config_marlin = GptqConfig::int4(128).with_marlin(true);
        assert_eq!(config_marlin.min_capability(), 80);

        let config_no_marlin = GptqConfig::int4(128).with_marlin(false);
        assert_eq!(config_no_marlin.min_capability(), 70);
    }

    #[test]
    fn test_gptq_num_groups() {
        let config = GptqConfig::int4(128);
        assert_eq!(config.num_groups(4096), 32);
        assert_eq!(config.num_groups(128), 1);
        assert_eq!(config.num_groups(200), 2);

        let config_perchannel = GptqConfig::int4(-1);
        assert_eq!(config_perchannel.num_groups(4096), 1);
    }

    #[test]
    fn test_gptq_linear_creation() {
        let config = GptqConfig::int4(128);
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_gptq_linear_validation_zero_in_features() {
        let result = GptqLinear::new(0, 128, false, 4, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("in_features"));
    }

    #[test]
    fn test_gptq_linear_validation_zero_out_features() {
        let result = GptqLinear::new(64, 0, false, 4, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out_features"));
    }

    #[test]
    fn test_gptq_linear_validation_invalid_bits() {
        let result = GptqLinear::new(64, 128, false, 2, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("4-bit or 8-bit"));
    }

    #[test]
    fn test_gptq_linear_forward_requires_loaded_weights() {
        // GPTQ layers require weights to be loaded before forward() can be called
        let linear = GptqLinear::new(64, 128, false, 4, 32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 64], DType::F16, &Device::Cpu).unwrap();
        let result = linear.forward(&x);

        // Should error because no weights are loaded
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no quantized weights loaded"));
    }

    #[test]
    fn test_gptq_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert("sym".to_string(), serde_json::json!(false));
        raw.insert("damp_percent".to_string(), serde_json::json!(0.02));

        let config = GptqConfig::from_detected(Some(4), Some(128), Some(true), &raw);

        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.desc_act);
        assert!(!config.sym);
        assert!((config.damp_percent - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_gptq_can_use_marlin() {
        // INT4 with supported group sizes should work
        let config = GptqConfig::int4(128);
        assert!(config.can_use_marlin());

        let config = GptqConfig::int4(64);
        assert!(config.can_use_marlin());

        let config = GptqConfig::int4(32);
        assert!(config.can_use_marlin());

        let config = GptqConfig::int4(-1); // per-channel
        assert!(config.can_use_marlin());

        // Unsupported group size
        let mut config = GptqConfig::int4(100);
        config.group_size = 100;
        assert!(!config.can_use_marlin());

        // Asymmetric quantization not supported
        let mut config = GptqConfig::int4(128);
        config.sym = false;
        assert!(!config.can_use_marlin());
    }

    #[test]
    fn test_gptq_can_use_marlin_for_shape() {
        let config = GptqConfig::int4(128);

        // Valid Marlin shapes (divisible by MIN_THREAD_N=64 and MIN_THREAD_K=128)
        assert!(config.can_use_marlin_for_shape(4096, 4096));
        assert!(config.can_use_marlin_for_shape(8192, 1024));

        // Invalid shapes
        assert!(!config.can_use_marlin_for_shape(100, 4096)); // K not divisible by 128
        assert!(!config.can_use_marlin_for_shape(4096, 100)); // N not divisible by 64
    }

    #[test]
    fn test_gptq_to_marlin_config() {
        let config = GptqConfig::int4(128);
        let marlin_config = config.to_marlin_config();

        assert!(marlin_config.is_some());
        let mc = marlin_config.unwrap();
        assert_eq!(mc.bits, 4);
        assert_eq!(mc.group_size, 128);
        assert_eq!(mc.scalar_type, MarlinScalarType::Uint4b8);

        // Asymmetric returns None
        let mut config = GptqConfig::int4(128);
        config.sym = false;
        assert!(config.to_marlin_config().is_none());
    }

    #[test]
    fn test_gptq_creates_marlin_for_valid_shape() {
        // With Marlin enabled and valid shape, should create MarlinLinear
        let config = GptqConfig::int4(128);
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        // Verify shape requirements are checked
        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
    }

    #[test]
    fn test_gptq_falls_back_to_gptq_linear() {
        // When Marlin is disabled, should create GptqLinear
        let config = GptqConfig::int4(128).with_marlin(false);
        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
    }

    #[cfg(feature = "cuda-kernels")]
    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_gptq_linear_forward_gpu() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let in_features: usize = 64;
            let out_features: usize = 128;
            let group_size: i64 = 32;
            let bits: u32 = 4;
            let pack_factor = 8usize; // 32 / 4
            let num_groups = in_features.div_ceil(group_size as usize);

            let mut linear =
                GptqLinear::new(in_features, out_features, false, bits, group_size, &device)
                    .unwrap();

            // Create quantized weights
            let qweight = Tensor::zeros(
                &[in_features / pack_factor, out_features],
                DType::U32,
                &device,
            )
            .unwrap();
            let scales = Tensor::ones(&[num_groups, out_features], DType::BF16, &device).unwrap();
            let qzeros = Tensor::zeros(
                &[num_groups, out_features / pack_factor],
                DType::U32,
                &device,
            )
            .unwrap();

            // Load weights
            let mut weights = HashMap::new();
            weights.insert("qweight".to_string(), qweight);
            weights.insert("scales".to_string(), scales);
            weights.insert("qzeros".to_string(), qzeros);
            linear.load_weights(&weights).unwrap();

            assert!(linear.is_quantized());

            // Forward pass
            let x = Tensor::randn(0.0f32, 1.0, (4, in_features), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = linear.forward(&x).unwrap();

            assert_eq!(y.dims(), &[4, out_features]);
        }

        #[test]
        fn test_gptq_linear_forward_with_bias() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let in_features: usize = 32;
            let out_features: usize = 64;
            let group_size: i64 = 16;
            let bits: u32 = 4;
            let pack_factor = 8usize;
            let num_groups = in_features.div_ceil(group_size as usize);

            let mut linear =
                GptqLinear::new(in_features, out_features, true, bits, group_size, &device)
                    .unwrap();

            let qweight = Tensor::zeros(
                &[in_features / pack_factor, out_features],
                DType::U32,
                &device,
            )
            .unwrap();
            let scales = Tensor::ones(&[num_groups, out_features], DType::BF16, &device).unwrap();
            let qzeros = Tensor::zeros(
                &[num_groups, out_features / pack_factor],
                DType::U32,
                &device,
            )
            .unwrap();
            let bias = Tensor::randn(0.0f32, 0.1, out_features, &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            let mut weights = HashMap::new();
            weights.insert("qweight".to_string(), qweight);
            weights.insert("scales".to_string(), scales);
            weights.insert("qzeros".to_string(), qzeros);
            weights.insert("bias".to_string(), bias);
            linear.load_weights(&weights).unwrap();

            let x = Tensor::randn(0.0f32, 1.0, (2, in_features), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = linear.forward(&x).unwrap();

            assert_eq!(y.dims(), &[2, out_features]);
        }
    }
}
