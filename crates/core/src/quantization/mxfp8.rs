//! MXFP8 (Microscaling FP8) quantization from NVIDIA ModelOpt.
//!
//! MXFP8 differs from standard FP8:
//! - Weights: FP8 E4M3 stored as U8
//! - Scales: uint8 in E8M0 format (exponent-only, 0 bits mantissa)
//! - Block size: 32 elements along K dimension
//! - Decoding: `scale_f32 = exp2(scale_u8 as f32 - 127.0)`
//! - Emulation: dequantize → BF16 matmul (no native CUDA kernel)

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

/// Block size along K dimension for MXFP8 quantization.
pub const MXFP8_BLOCK_SIZE: usize = 32;

/// E8M0 exponent bias (IEEE-like bias for 8-bit exponent).
const E8M0_BIAS: f32 = 127.0;

/// MXFP8 quantization configuration.
#[derive(Debug, Clone, Default)]
pub struct MxFp8Config {
    /// Layers to skip (not quantize).
    pub ignored_layers: Vec<String>,
    /// Optional KV cache quantization algorithm.
    pub kv_cache_quant_algo: Option<String>,
}

impl MxFp8Config {
    /// Create from detected raw config.
    ///
    /// Expected JSON structure:
    /// ```json
    /// {
    ///   "quant_method": "modelopt",
    ///   "quantization": {
    ///     "quant_algo": "MXFP8",
    ///     "kv_cache_quant_algo": "FP8"
    ///   },
    ///   "exclude_modules": ["lm_head"]
    /// }
    /// ```
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let kv_cache_quant_algo = raw_config
            .get("quantization")
            .and_then(|q| q.get("kv_cache_quant_algo"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let ignored_layers = raw_config
            .get("exclude_modules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            ignored_layers,
            kv_cache_quant_algo,
        }
    }
}

impl QuantizationConfig for MxFp8Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::ModelOpt
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        0 // Emulation works on any GPU
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
        Ok(Box::new(MxFp8Linear::new(
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

/// Decode E8M0 scales (uint8 exponent-only) to F32.
///
/// E8M0 format: 8-bit unsigned exponent with bias 127.
/// `decoded = 2^(raw - 127)`
fn decode_e8m0_scales(scales: &Tensor) -> Result<Tensor> {
    let float_scales = scales.to_dtype(DType::F32)?;
    let biased = (float_scales - E8M0_BIAS as f64)?;
    // exp2(x) = e^(x * ln(2))
    (biased * std::f64::consts::LN_2)?.exp()
}

/// MXFP8 quantized linear layer.
///
/// Weights are FP8 E4M3 (stored as U8), scales are E8M0 (stored as U8).
/// Forward pass: dequantize blocks → BF16 matmul.
#[derive(Debug)]
pub struct MxFp8Linear {
    /// Weight tensor — U8 (FP8 E4M3) or BF16 (after dequant/init).
    weight: Tensor,
    /// Optional bias.
    bias: Option<Tensor>,
    /// E8M0 block scales — U8, shape `[N, K/32]`.
    weight_scale: Option<Tensor>,
    /// Input features (K dimension).
    in_features: usize,
    /// Output features (N dimension).
    out_features: usize,
    /// Whether weights are in FP8 (U8) format.
    is_fp8_weights: bool,
}

impl MxFp8Linear {
    /// Create a new MXFP8 linear layer.
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
            in_features,
            out_features,
            is_fp8_weights: false,
        })
    }

    /// Dequantize MXFP8 weights using E8M0 block scales.
    ///
    /// 1. Decode E8M0 scales to F32
    /// 2. Cast U8 weights to F32
    /// 3. Reshape weights into blocks of 32 along K: `[N, K] → [N, K/32, 32]`
    /// 4. Multiply each block by its scale
    /// 5. Reshape back to `[N, K]`
    /// 6. Convert to BF16
    fn dequantize(&self) -> Result<Tensor> {
        let scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("MXFP8 dequantization requires weight_scale".to_string())
        })?;

        let decoded_scales = decode_e8m0_scales(scale)?;

        let weight_f32 = self.weight.to_dtype(DType::F32)?;

        let num_blocks = self.in_features / MXFP8_BLOCK_SIZE;

        // Reshape: [N, K] → [N, K/32, 32]
        let blocked = weight_f32.reshape((self.out_features, num_blocks, MXFP8_BLOCK_SIZE))?;

        // Expand scales: [N, K/32] → [N, K/32, 1] → [N, K/32, 32] for element-wise mul
        let scales_expanded = decoded_scales.unsqueeze(2)?.expand(&[
            self.out_features,
            num_blocks,
            MXFP8_BLOCK_SIZE,
        ])?;

        // Multiply and reshape back: [N, K/32, 32] → [N, K]
        let dequantized = (blocked * scales_expanded)?;
        let flat = dequantized.reshape((self.out_features, self.in_features))?;
        flat.to_dtype(DType::BF16)
    }
}

impl QuantizedLinear for MxFp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = if self.is_fp8_weights && self.weight_scale.is_some() {
            self.dequantize()?.to_dtype(x.dtype())?
        } else {
            self.weight.to_dtype(x.dtype())?
        };

        let y = x.matmul(&weight.t()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            self.is_fp8_weights = w.dtype() == DType::U8;
            self.weight = w.clone();
        }
        if let Some(s) = weights.get("weight_scale") {
            // Validate K divisibility by block size
            if !self.in_features.is_multiple_of(MXFP8_BLOCK_SIZE) {
                candle_core::bail!(
                    "MXFP8 requires in_features ({}) divisible by block_size ({})",
                    self.in_features,
                    MXFP8_BLOCK_SIZE,
                );
            }
            // Validate scale shape
            let scale_dims = s.dims();
            if scale_dims.len() == 2 {
                let expected_blocks = self.in_features / MXFP8_BLOCK_SIZE;
                if scale_dims[1] != expected_blocks {
                    candle_core::bail!(
                        "MXFP8 weight_scale K-dim {} doesn't match expected {} \
                         (in_features={} / block_size={})",
                        scale_dims[1],
                        expected_blocks,
                        self.in_features,
                        MXFP8_BLOCK_SIZE,
                    );
                }
                if scale_dims[0] != self.out_features {
                    candle_core::bail!(
                        "MXFP8 weight_scale N-dim {} doesn't match out_features {}",
                        scale_dims[0],
                        self.out_features,
                    );
                }
            }
            self.weight_scale = Some(s.clone());
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
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

    // ─── E8M0 decode tests ─────────────────────────────────────────────

    #[test]
    fn test_e8m0_decode_zero() {
        // exp2(0 - 127) = 2^(-127) ≈ 5.88e-39
        let scales = Tensor::new(&[0u8], &Device::Cpu).unwrap();
        let decoded = decode_e8m0_scales(&scales).unwrap();
        let val: f32 = decoded.to_vec1().unwrap()[0];
        let expected = 2.0f32.powi(-127);
        assert!(
            (val - expected).abs() < expected * 1e-5,
            "got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_e8m0_decode_127() {
        // exp2(127 - 127) = 2^0 = 1.0
        let scales = Tensor::new(&[127u8], &Device::Cpu).unwrap();
        let decoded = decode_e8m0_scales(&scales).unwrap();
        let val: f32 = decoded.to_vec1().unwrap()[0];
        assert!((val - 1.0).abs() < 1e-6, "got {val}, expected 1.0");
    }

    #[test]
    fn test_e8m0_decode_128() {
        // exp2(128 - 127) = 2^1 = 2.0
        let scales = Tensor::new(&[128u8], &Device::Cpu).unwrap();
        let decoded = decode_e8m0_scales(&scales).unwrap();
        let val: f32 = decoded.to_vec1().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-6, "got {val}, expected 2.0");
    }

    #[test]
    fn test_e8m0_decode_254() {
        // exp2(254 - 127) = 2^127
        let scales = Tensor::new(&[254u8], &Device::Cpu).unwrap();
        let decoded = decode_e8m0_scales(&scales).unwrap();
        let val: f32 = decoded.to_vec1().unwrap()[0];
        let expected = 2.0f32.powi(127);
        assert!(
            (val - expected).abs() / expected < 1e-5,
            "got {val}, expected {expected}"
        );
    }

    // ─── Config tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = MxFp8Config::default();
        assert_eq!(config.method(), QuantizationMethod::ModelOpt);
        assert_eq!(config.supported_act_dtypes(), &[DType::BF16]);
        assert_eq!(config.min_capability(), 0);
        assert!(config.ignored_layers.is_empty());
        assert!(config.kv_cache_quant_algo.is_none());
    }

    #[test]
    fn test_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert(
            "quantization".to_string(),
            serde_json::json!({
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": "FP8"
            }),
        );
        raw.insert(
            "exclude_modules".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );

        let config = MxFp8Config::from_detected(&raw);
        assert_eq!(config.kv_cache_quant_algo.as_deref(), Some("FP8"));
        assert_eq!(config.ignored_layers.len(), 2);
        assert!(config.ignored_layers.contains(&"lm_head".to_string()));
    }

    #[test]
    fn test_config_ignored_layers() {
        let config = MxFp8Config {
            ignored_layers: vec!["lm_head".to_string()],
            kv_cache_quant_algo: None,
        };
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    // ─── Linear construction tests ──────────────────────────────────────

    #[test]
    fn test_linear_construction() {
        let linear = MxFp8Linear::new(64, 128, true, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
        assert!(!linear.is_fp8_weights);
    }

    #[test]
    fn test_linear_construction_zero_in_features() {
        let result = MxFp8Linear::new(0, 128, false, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_construction_zero_out_features() {
        let result = MxFp8Linear::new(64, 0, false, &Device::Cpu);
        assert!(result.is_err());
    }

    // ─── Forward tests ──────────────────────────────────────────────────

    #[test]
    fn test_linear_forward_shape() {
        let mut linear = MxFp8Linear::new(64, 128, false, &Device::Cpu).unwrap();
        // Override with F32 for CPU testing
        linear.weight = Tensor::zeros((128, 64), DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 128]);
    }

    #[test]
    fn test_dequantize_identity_scales() {
        // Scale = 127 → decoded = 2^0 = 1.0 → weights unchanged
        let mut linear = MxFp8Linear::new(32, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        // All-ones FP8 weights
        weights.insert(
            "weight".to_string(),
            Tensor::ones((4, 32), DType::U8, &Device::Cpu).unwrap(),
        );
        // Scale = 127 → 1.0 (identity), shape [4, 1]
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (4, 1), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        // Weight U8 value 1 cast to F32 = 1.0, scale = 1.0 → result = 1.0
        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
            }
        }
    }

    #[test]
    fn test_dequantize_double_scales() {
        // Scale = 128 → decoded = 2^1 = 2.0 → weights * 2
        let mut linear = MxFp8Linear::new(32, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::ones((4, 32), DType::U8, &Device::Cpu).unwrap(),
        );
        // Scale = 128 → 2.0, shape [4, 1]
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(128u8, (4, 1), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        // Weight U8 value 1 * scale 2.0 = 2.0
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 0.01, "expected ~2.0, got {v}");
            }
        }
    }

    #[test]
    fn test_block_size_validation() {
        // K=33 is not divisible by 32
        let mut linear = MxFp8Linear::new(33, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::ones((4, 33), DType::U8, &Device::Cpu).unwrap(),
        );
        // Scale shape [4, 1] but K/32 = 1 while K=33 → mismatch
        weights.insert(
            "weight_scale".to_string(),
            Tensor::ones(&[4, 1], DType::U8, &Device::Cpu).unwrap(),
        );

        let result = linear.load_weights(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_forward_with_dequant() {
        let mut linear = MxFp8Linear::new(32, 8, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::ones((8, 32), DType::U8, &Device::Cpu).unwrap(),
        );
        // Scale = 127 → 1.0
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (8, 1), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[2, 32], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let mut linear = MxFp8Linear::new(32, 8, true, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::zeros((8, 32), DType::U8, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "bias".to_string(),
            Tensor::ones(8, DType::F32, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (8, 1), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[3, 32], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 8]);

        // Weight=0 → matmul=0, bias=1 → output=1
        let vals: Vec<Vec<f32>> = y.to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < 0.01, "expected ~1.0 (bias only), got {v}");
            }
        }
    }

    #[test]
    fn test_create_linear_from_config() {
        let config = MxFp8Config::default();
        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(!linear.has_bias());
    }
}
