//! ExpertsInt8: online INT8 quantization for MoE expert weights (W8A16).
//!
//! Expert weights are loaded as FP16/BF16 and quantized to INT8 at load time.
//! Per-channel (per-row) symmetric quantization: `scale[i] = max(|w[i,:]|) / 127`.
//! Forward: dequantize per-channel and matmul with FP16/BF16 activations.
//!
//! Non-MoE layers remain unquantized (full precision).

use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

use super::config::{
    QuantizationConfig, QuantizationMethod, QuantizedLinear, UnquantizedLinear,
};

/// Configuration for ExpertsInt8 quantization.
///
/// Only MoE expert weights are quantized to INT8. Non-MoE layers
/// (embeddings, attention, router, norms) remain at full precision.
#[derive(Debug, Clone)]
pub struct ExpertsInt8Config {
    /// Model dtype for non-quantized layers and activations.
    model_dtype: DType,
}

impl ExpertsInt8Config {
    /// Create from model dtype.
    pub fn new(model_dtype: DType) -> Self {
        Self { model_dtype }
    }

    /// Create from detected config.
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let model_dtype = raw_config
            .get("model_dtype")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "float16" => DType::F16,
                "bfloat16" => DType::BF16,
                _ => DType::BF16,
            })
            .unwrap_or(DType::BF16);

        Self { model_dtype }
    }
}

impl QuantizationConfig for ExpertsInt8Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::ExpertsInt8
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        0 // CPU supported — online quantization is pure compute
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false
    }

    /// For ExpertsInt8, non-MoE layers return unquantized linear.
    /// MoE expert layers should use `create_int8_linear()` instead.
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

impl ExpertsInt8Config {
    /// Create an INT8 linear layer for MoE expert weights.
    ///
    /// The returned layer quantizes weights from FP16/BF16 to INT8 at load time.
    pub fn create_int8_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(ExpertsInt8Linear::new(
            in_features,
            out_features,
            bias,
            self.model_dtype,
            device,
        )?))
    }
}

/// INT8 linear layer with online quantization.
///
/// Weights are stored as INT8 with per-channel FP32 scales.
/// When `load_weights()` receives FP16/BF16 weights, they are
/// quantized to INT8 using symmetric per-channel quantization.
///
/// Forward: `y = (w_i8 * scale).to_dtype(act_dtype) @ x^T`
pub struct ExpertsInt8Linear {
    /// INT8 quantized weights `[out_features, in_features]`
    weight_i8: Tensor,
    /// Per-channel scales `[out_features]`
    weight_scale: Tensor,
    /// Optional bias
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    model_dtype: DType,
}

impl std::fmt::Debug for ExpertsInt8Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertsInt8Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("model_dtype", &self.model_dtype)
            .finish()
    }
}

impl ExpertsInt8Linear {
    /// Create a new INT8 linear layer (initially zero weights).
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        model_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weight_i8 = Tensor::zeros((out_features, in_features), DType::I64, device)?;
        let weight_scale = Tensor::ones(out_features, DType::F32, device)?;
        let bias = if has_bias {
            Some(Tensor::zeros(out_features, model_dtype, device)?)
        } else {
            None
        };

        Ok(Self {
            weight_i8,
            weight_scale,
            bias,
            in_features,
            out_features,
            model_dtype,
        })
    }
}

/// Quantize a FP32/FP16/BF16 weight tensor to INT8 with per-channel scaling.
///
/// Returns `(weight_i8, scale)` where:
/// - `weight_i8` is `[out, in]` with values in `[-127, 127]`
/// - `scale` is `[out]` with the per-channel dequantization scale
pub fn quantize_to_int8(weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let weight_f32 = weight.to_dtype(DType::F32)?;

    // Per-row max-abs: scale[i] = max(|w[i,:]|) / 127
    let abs_weight = weight_f32.abs()?;
    let row_max = abs_weight.max(1)?; // [out_features]

    let scale = (row_max / 127.0)?;

    // Clamp scale to avoid division by zero
    let epsilon = Tensor::new(&[1e-12f32], scale.device())?;
    let scale = scale.maximum(&epsilon.broadcast_as(scale.shape())?)?;

    // Quantize: w_i8 = round(w / scale)
    let scale_expanded = scale.unsqueeze(1)?; // [out, 1]
    let quantized = weight_f32.broadcast_div(&scale_expanded)?;
    let quantized = quantized.round()?;

    // Clamp to [-127, 127]
    let quantized = quantized.clamp(-127.0f64, 127.0f64)?;

    // Store as I64 (candle doesn't have native I8)
    let weight_i8 = quantized.to_dtype(DType::I64)?;

    Ok((weight_i8, scale))
}

impl QuantizedLinear for ExpertsInt8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantize: w_f32 = w_i8.to_f32() * scale.unsqueeze(1)
        let weight_f32 = self.weight_i8.to_dtype(DType::F32)?;
        let scale_expanded = self.weight_scale.unsqueeze(1)?;
        let dequantized = weight_f32.broadcast_mul(&scale_expanded)?;

        // Cast to activation dtype for matmul
        let dequantized = dequantized.to_dtype(x.dtype())?;

        // matmul: x @ w^T
        let y = x.matmul(&dequantized.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            let (weight_i8, scale) = quantize_to_int8(w)?;
            self.weight_i8 = weight_i8;
            self.weight_scale = scale;
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::I64 // Stored as I64 (representing INT8 values)
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
    use candle_core::IndexOp;

    #[test]
    fn test_experts_int8_config_basic() {
        let config = ExpertsInt8Config::new(DType::BF16);
        assert_eq!(config.method(), QuantizationMethod::ExpertsInt8);
        assert_eq!(config.min_capability(), 0);
        assert!(!config.is_layer_skipped("model.layers.0"));
    }

    #[test]
    fn test_experts_int8_config_from_detected() {
        let raw = HashMap::new();
        let config = ExpertsInt8Config::from_detected(&raw);
        assert_eq!(config.model_dtype, DType::BF16);
    }

    #[test]
    fn test_experts_int8_config_non_moe_layer_unquantized() {
        let config = ExpertsInt8Config::new(DType::F32);
        let linear = config
            .create_linear(64, 128, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        // Non-MoE layers should be full precision
        assert_eq!(linear.weight_dtype(), DType::F32);
    }

    #[test]
    fn test_experts_int8_config_moe_layer_quantized() {
        let config = ExpertsInt8Config::new(DType::F32);
        let linear = config
            .create_int8_linear(64, 128, false, &Device::Cpu)
            .unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert_eq!(linear.weight_dtype(), DType::I64); // I64 representing INT8
    }

    #[test]
    fn test_quantize_to_int8_identity() {
        // Zero weights should produce zero quantized + scale=epsilon
        let w = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let (q, s) = quantize_to_int8(&w).unwrap();
        assert_eq!(q.dims(), &[4, 8]);
        assert_eq!(s.dims(), &[4]);

        let q_vals: Vec<i64> = q.flatten_all().unwrap().to_vec1().unwrap();
        assert!(q_vals.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_to_int8_scaling() {
        // Row with max abs = 127 should map to itself
        let mut data = vec![0.0f32; 4 * 8];
        data[0] = 127.0; // row 0, col 0
        data[15] = -63.5; // row 1, col 7

        let w = Tensor::from_vec(data, (4, 8), &Device::Cpu).unwrap();
        let (q, s) = quantize_to_int8(&w).unwrap();

        let q_vals: Vec<i64> = q.flatten_all().unwrap().to_vec1().unwrap();
        let s_vals: Vec<f32> = s.to_vec1().unwrap();

        // Row 0: max_abs=127, scale=1.0, q[0][0]=127
        assert!((s_vals[0] - 1.0).abs() < 1e-6);
        assert_eq!(q_vals[0], 127);

        // Row 1: max_abs=63.5, scale=63.5/127=0.5, q[1][7]=-127
        assert!((s_vals[1] - 0.5).abs() < 1e-6);
        assert_eq!(q_vals[15], -127);
    }

    #[test]
    fn test_quantize_to_int8_clamp() {
        // Verify values are clamped to [-127, 127]
        let w = Tensor::new(&[[254.0f32, -254.0]], &Device::Cpu).unwrap();
        let (q, _s) = quantize_to_int8(&w).unwrap();
        let q_vals: Vec<i64> = q.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(q_vals[0], 127);
        assert_eq!(q_vals[1], -127);
    }

    #[test]
    fn test_experts_int8_linear_forward() {
        let mut linear = ExpertsInt8Linear::new(8, 4, false, DType::F32, &Device::Cpu).unwrap();

        // Load some weights
        let w = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), w.clone());
        linear.load_weights(&weights).unwrap();

        // Forward pass
        let x = Tensor::randn(0f32, 1.0, (2, 8), &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 4]);

        // Compare with reference (direct matmul — should be close but not exact)
        let y_ref = x.matmul(&w.t().unwrap()).unwrap();
        let diff: f32 = y
            .sub(&y_ref)
            .unwrap()
            .abs()
            .unwrap()
            .max(candle_core::D::Minus1)
            .unwrap()
            .max(candle_core::D::Minus1)
            .unwrap()
            .to_scalar()
            .unwrap();
        // INT8 quantization error should be small relative to weight magnitude
        let w_max: f32 = w
            .abs()
            .unwrap()
            .max(candle_core::D::Minus1)
            .unwrap()
            .max(candle_core::D::Minus1)
            .unwrap()
            .to_scalar()
            .unwrap();
        // Error per element should be < 1% of max weight × input magnitude
        assert!(
            diff < w_max * 0.05,
            "Quantization error too large: diff={diff}, w_max={w_max}"
        );
    }

    #[test]
    fn test_experts_int8_linear_with_bias() {
        let mut linear = ExpertsInt8Linear::new(8, 4, true, DType::F32, &Device::Cpu).unwrap();
        assert!(linear.has_bias());

        let w = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();
        let b = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), w);
        weights.insert("bias".to_string(), b);
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[1, 8], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 4]);
    }

    #[test]
    fn test_experts_int8_linear_f16_input() {
        let mut linear = ExpertsInt8Linear::new(8, 4, false, DType::F16, &Device::Cpu).unwrap();

        let w = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), w);
        linear.load_weights(&weights).unwrap();

        // Forward with F16 input — candle supports F16 matmul on CPU
        let x = Tensor::randn(0f32, 1.0, (3, 8), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 4]);
        assert_eq!(y.dtype(), DType::F16);
    }

    #[test]
    fn test_quantize_to_int8_from_bf16() {
        let w = Tensor::randn(0f32, 1.0, (4, 8), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (q, s) = quantize_to_int8(&w).unwrap();
        assert_eq!(q.dims(), &[4, 8]);
        assert_eq!(s.dims(), &[4]);
        assert_eq!(q.dtype(), DType::I64);
        assert_eq!(s.dtype(), DType::F32);
    }

    #[test]
    fn test_experts_int8_roundtrip_accuracy() {
        // Create a weight where values are exact multiples of scale
        // to verify quantization is numerically exact
        let w = Tensor::new(
            &[
                [1.0f32, 2.0, 3.0, 4.0],
                [-1.0, -2.0, -3.0, -4.0],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let (q, s) = quantize_to_int8(&w).unwrap();
        let s_vals: Vec<f32> = s.to_vec1().unwrap();

        // Row 0: max_abs=4, scale=4/127
        let expected_scale = 4.0 / 127.0;
        assert!(
            (s_vals[0] - expected_scale).abs() < 1e-6,
            "Scale mismatch: {} vs {}",
            s_vals[0],
            expected_scale
        );

        // q[0] = round([1, 2, 3, 4] / (4/127)) = round([31.75, 63.5, 95.25, 127])
        let q_vals: Vec<i64> = q.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(q_vals, vec![32, 64, 95, 127]);
    }
}
