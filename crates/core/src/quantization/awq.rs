//! AWQ (Activation-aware Weight Quantization) configuration and layers.
//!
//! AWQ is a quantization method that uses activation-aware scaling to
//! maintain model accuracy while reducing memory footprint.
//!
//! Supports:
//! - 4-bit quantization (most common)
//! - Group-wise quantization with configurable group size
//! - Zero-point quantization
//! - GEMM and GEMV kernel variants

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::marlin::{
    check_marlin_supports_shape, MarlinConfig, MarlinLinear,
    MARLIN_SUPPORTED_GROUP_SIZES,
};

#[cfg(feature = "cuda-kernels")]
use super::gptq_cuda;

/// AWQ quantization configuration.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Quantization bits (typically 4)
    pub bits: u32,
    /// Group size for quantization (typically 128)
    pub group_size: i64,
    /// Whether to use zero-point quantization
    pub zero_point: bool,
    /// AWQ kernel version ("GEMM" or "GEMV")
    pub version: AwqVersion,
    /// Whether to use Marlin kernels (2-4x faster on Ampere+)
    pub use_marlin: bool,
}

/// AWQ kernel version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AwqVersion {
    /// GEMM version (batch matrix multiplication)
    #[default]
    Gemm,
    /// GEMV version (matrix-vector multiplication, optimized for single-token decode)
    Gemv,
}

impl AwqConfig {
    /// Create a new AWQ config with standard 4-bit quantization.
    pub fn int4(group_size: i64) -> Self {
        Self {
            bits: 4,
            group_size,
            zero_point: true,
            version: AwqVersion::Gemm,
            use_marlin: true, // Auto-enable Marlin for 4-bit
        }
    }

    /// Create from detected config.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let bits = bits.unwrap_or(4);
        let group_size = group_size.map(|g| g as i64).unwrap_or(128);

        let zero_point = raw_config
            .get("zero_point")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let version = match raw_config.get("version").and_then(|v| v.as_str()) {
            Some("GEMV") | Some("gemv") => AwqVersion::Gemv,
            _ => AwqVersion::Gemm,
        };

        // Auto-enable Marlin for 4-bit AWQ
        let is_marlin = raw_config
            .get("is_marlin_format")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            bits,
            group_size,
            zero_point,
            version,
            use_marlin: is_marlin || bits == 4,
        }
    }

    /// Set the kernel version.
    pub fn with_version(mut self, version: AwqVersion) -> Self {
        self.version = version;
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
            1
        } else {
            in_features.div_ceil(self.group_size as usize)
        }
    }

    /// Check if Marlin kernel can be used for this config.
    ///
    /// AWQ uses Marlin's Uint4 path (4-bit with runtime zero points).
    /// Requires:
    /// - 4-bit quantization
    /// - Group size in [-1, 32, 64, 128]
    /// - GPU compute capability >= 8.0 (Ampere)
    pub fn can_use_marlin(&self) -> bool {
        if self.bits != 4 {
            return false;
        }
        MARLIN_SUPPORTED_GROUP_SIZES.contains(&(self.group_size as i32))
    }

    /// Check if Marlin supports a specific layer shape.
    pub fn can_use_marlin_for_shape(&self, in_features: usize, out_features: usize) -> bool {
        if !self.can_use_marlin() {
            return false;
        }
        check_marlin_supports_shape(out_features, in_features, self.group_size as i32).is_ok()
    }

    /// Convert to MarlinConfig for AWQ (asymmetric with zero points).
    pub fn to_marlin_config(&self) -> Option<MarlinConfig> {
        if !self.can_use_marlin() {
            return None;
        }
        Some(MarlinConfig::awq_int4(self.group_size as i32))
    }
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self::int4(128)
    }
}

impl QuantizationConfig for AwqConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Awq
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        if self.use_marlin {
            80 // Ampere required for Marlin
        } else {
            70 // Volta for basic AWQ
        }
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false // AWQ quantizes all linear layers
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

        // Fallback to standard AWQ
        Ok(Box::new(AwqLinear::new(
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

/// AWQ quantized linear layer.
///
/// AWQ stores weights in a format similar to GPTQ but with different
/// scale/zero-point semantics based on activation-aware quantization.
#[derive(Debug)]
pub struct AwqLinear {
    /// Quantized weights packed into INT32 [K/pack_factor, N]
    qweight: Tensor,
    /// Quantization scales per group [num_groups, N]
    scales: Tensor,
    /// Zero points per group packed [num_groups, N/pack_factor]
    qzeros: Tensor,
    /// Optional bias [N]
    bias: Option<Tensor>,
    /// Bits (typically 4)
    #[allow(dead_code)] // Used for kernel dispatch
    bits: u32,
    /// Group size
    #[allow(dead_code)] // Used for kernel dispatch
    group_size: i64,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Whether weights are loaded in quantized format
    is_quantized: bool,
}

impl AwqLinear {
    /// Create a new AWQ linear layer.
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
        if bits != 4 {
            candle_core::bail!("AWQ only supports 4-bit quantization, got {bits}");
        }

        // Calculate packed dimensions (same packing as GPTQ)
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
            bits,
            group_size,
            in_features,
            out_features,
            is_quantized: false,
        })
    }

    /// Check if this layer can use AWQ CUDA kernels.
    /// AWQ uses the same kernel infrastructure as GPTQ for 4-bit.
    #[cfg(feature = "cuda-kernels")]
    fn can_use_awq_kernel(&self) -> bool {
        self.is_quantized && self.bits == 4 && self.qweight.device().is_cuda()
    }

    /// Perform AWQ forward pass using CUDA kernels.
    /// AWQ uses GPTQ-compatible kernels for the actual computation.
    #[cfg(feature = "cuda-kernels")]
    fn forward_awq(&self, x: &Tensor) -> Result<Tensor> {
        // Convert scales to BF16 if needed
        let scales_bf16 = if self.scales.dtype() == DType::BF16 {
            self.scales.clone()
        } else {
            self.scales.to_dtype(DType::BF16)?
        };

        // Use GPTQ GEMM kernel (AWQ uses compatible weight format)
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
    fn dequantize(&self) -> Result<Tensor> {
        if !self.is_quantized {
            candle_core::bail!(
                "AWQ layer has no quantized weights loaded - call load_weights() first"
            );
        }

        #[cfg(feature = "cuda-kernels")]
        if self.qweight.device().is_cuda() {
            let scales_bf16 = if self.scales.dtype() == DType::BF16 {
                self.scales.clone()
            } else {
                self.scales.to_dtype(DType::BF16)?
            };
            // Use GPTQ dequantize (AWQ uses compatible format)
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

        candle_core::bail!(
            "AWQ dequantization requires CUDA device with cuda-kernels feature enabled"
        )
    }

    /// Check if weights are quantized.
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }
}

impl QuantizedLinear for AwqLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use AWQ CUDA kernels when available
        #[cfg(feature = "cuda-kernels")]
        if self.can_use_awq_kernel() {
            let x_bf16 = if x.dtype() == DType::BF16 {
                x.clone()
            } else {
                x.to_dtype(DType::BF16)?
            };
            return self.forward_awq(&x_bf16);
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
    use super::super::marlin::MarlinScalarType;

    #[test]
    fn test_awq_config_int4() {
        let config = AwqConfig::int4(128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.zero_point);
        assert_eq!(config.version, AwqVersion::Gemm);
        assert_eq!(config.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_awq_config_min_capability() {
        // Default 4-bit AWQ auto-enables Marlin → requires Ampere
        let config = AwqConfig::default();
        assert_eq!(config.min_capability(), 80);

        // Disabling Marlin falls back to Volta requirement
        let config_no_marlin = AwqConfig::int4(128).with_marlin(false);
        assert_eq!(config_no_marlin.min_capability(), 70);
    }

    #[test]
    fn test_awq_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert("zero_point".to_string(), serde_json::json!(false));
        raw.insert("version".to_string(), serde_json::json!("GEMV"));

        let config = AwqConfig::from_detected(Some(4), Some(64), &raw);

        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert!(!config.zero_point);
        assert_eq!(config.version, AwqVersion::Gemv);
    }

    #[test]
    fn test_awq_num_groups() {
        let config = AwqConfig::int4(128);
        assert_eq!(config.num_groups(4096), 32);
        assert_eq!(config.num_groups(128), 1);
        assert_eq!(config.num_groups(200), 2);
    }

    #[test]
    fn test_awq_linear_creation() {
        let config = AwqConfig::int4(128);
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_awq_linear_validation_zero_in_features() {
        let result = AwqLinear::new(0, 128, false, 4, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("in_features"));
    }

    #[test]
    fn test_awq_linear_validation_zero_out_features() {
        let result = AwqLinear::new(64, 0, false, 4, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out_features"));
    }

    #[test]
    fn test_awq_linear_validation_invalid_bits() {
        let result = AwqLinear::new(64, 128, false, 8, 128, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("4-bit"));
    }

    #[test]
    fn test_awq_linear_forward_requires_loaded_weights() {
        let linear = AwqLinear::new(64, 128, false, 4, 32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 64], DType::F16, &Device::Cpu).unwrap();
        let result = linear.forward(&x);

        // Should error because no weights are loaded
        assert!(result.is_err());
    }

    #[test]
    fn test_awq_version_default() {
        assert_eq!(AwqVersion::default(), AwqVersion::Gemm);
    }

    #[test]
    fn test_awq_config_with_version() {
        let config = AwqConfig::int4(128).with_version(AwqVersion::Gemv);
        assert_eq!(config.version, AwqVersion::Gemv);
    }

    #[test]
    fn test_awq_can_use_marlin() {
        // 4-bit with supported group size → can use Marlin
        let config = AwqConfig::int4(128);
        assert!(config.can_use_marlin());
        assert!(config.use_marlin);

        // 4-bit with per-channel (group_size=-1) → can use Marlin
        let config = AwqConfig::int4(-1);
        assert!(config.can_use_marlin());

        // Unsupported group size
        let mut config = AwqConfig::int4(100);
        config.group_size = 100;
        assert!(!config.can_use_marlin());
    }

    #[test]
    fn test_awq_can_use_marlin_for_shape() {
        let config = AwqConfig::int4(128);

        // Valid shapes (divisible by 64/128)
        assert!(config.can_use_marlin_for_shape(4096, 4096));

        // Invalid N dimension (not divisible by 64)
        assert!(!config.can_use_marlin_for_shape(100, 4096));

        // Invalid K dimension (not divisible by 128)
        assert!(!config.can_use_marlin_for_shape(4096, 100));
    }

    #[test]
    fn test_awq_to_marlin_config() {
        let config = AwqConfig::int4(128);
        let marlin_config = config.to_marlin_config();
        assert!(marlin_config.is_some());

        let mc = marlin_config.unwrap();
        assert_eq!(mc.bits, 4);
        assert_eq!(mc.group_size, 128);
        assert_eq!(mc.scalar_type, MarlinScalarType::Uint4);
        assert!(!mc.is_sym); // AWQ is asymmetric
    }

    #[test]
    fn test_awq_marlin_routing_in_create_linear() {
        // With valid Marlin shape, should create MarlinLinear
        let config = AwqConfig::int4(128);
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);

        // With invalid shape, should fall back to AwqLinear
        let config = AwqConfig::int4(128).with_marlin(false);
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 4096);
    }

    #[test]
    fn test_awq_from_detected_marlin_auto_enable() {
        // 4-bit auto-enables Marlin
        let raw = HashMap::new();
        let config = AwqConfig::from_detected(Some(4), Some(128), &raw);
        assert!(config.use_marlin);

        // Explicit Marlin format flag
        let mut raw = HashMap::new();
        raw.insert(
            "is_marlin_format".to_string(),
            serde_json::json!(true),
        );
        let config = AwqConfig::from_detected(Some(4), Some(128), &raw);
        assert!(config.use_marlin);
    }

    #[test]
    fn test_awq_with_marlin_toggle() {
        let config = AwqConfig::int4(128);
        assert!(config.use_marlin);

        let config = config.with_marlin(false);
        assert!(!config.use_marlin);
    }

    #[cfg(feature = "cuda-kernels")]
    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_awq_linear_forward_gpu() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let in_features: usize = 64;
            let out_features: usize = 128;
            let group_size: i64 = 32;
            let bits: u32 = 4;
            let pack_factor = 8usize;
            let num_groups = in_features.div_ceil(group_size as usize);

            let mut linear =
                AwqLinear::new(in_features, out_features, false, bits, group_size, &device)
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

            let mut weights = HashMap::new();
            weights.insert("qweight".to_string(), qweight);
            weights.insert("scales".to_string(), scales);
            weights.insert("qzeros".to_string(), qzeros);
            linear.load_weights(&weights).unwrap();

            assert!(linear.is_quantized());

            let x = Tensor::randn(0.0f32, 1.0, (4, in_features), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let y = linear.forward(&x).unwrap();

            assert_eq!(y.dims(), &[4, out_features]);
        }
    }
}
