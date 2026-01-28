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
}

/// AWQ kernel version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwqVersion {
    /// GEMM version (batch matrix multiplication)
    Gemm,
    /// GEMV version (matrix-vector multiplication, optimized for single-token decode)
    Gemv,
}

impl Default for AwqVersion {
    fn default() -> Self {
        Self::Gemm
    }
}

impl AwqConfig {
    /// Create a new AWQ config with standard 4-bit quantization.
    pub fn int4(group_size: i64) -> Self {
        Self {
            bits: 4,
            group_size,
            zero_point: true,
            version: AwqVersion::Gemm,
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

        Self {
            bits,
            group_size,
            zero_point,
            version,
        }
    }

    /// Set the kernel version.
    pub fn with_version(mut self, version: AwqVersion) -> Self {
        self.version = version;
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
        70 // Volta
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
        let config = AwqConfig::default();
        assert_eq!(config.min_capability(), 70);
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
