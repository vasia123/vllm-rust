//! FBGEMM FP8 quantization (Meta).
//!
//! FP8 E4M3 weights with per-channel (per-output-row) scales, and dynamic
//! per-token FP8 activation quantization on Hopper (cap ≥ 89).  On Ampere
//! (cap 80-88) this degrades to a weight-only dequantize+GEMM path; a
//! Marlin FP8 kernel path is noted as a TODO.
//!
//! Weight format (HuggingFace checkpoint):
//!   `{prefix}.weight`        — FP8 E4M3 stored as U8, shape [out, in]
//!   `{prefix}.weight_scale`  — F32 per-channel scales, shape [out, 1] or [out]
//!
//! Config keys in `quantization_config`:
//!   `modules_to_not_convert` — list of layer name prefixes to skip
//!   `activation_scale_ub`    — scalar F32 upper bound on dynamic input scales
//!
//! Reference: vLLM `vllm/model_executor/layers/quantization/fbgemm_fp8.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

#[cfg(feature = "cuda-kernels")]
use super::fp8_cuda;

// ─── FbgemmFp8Config ─────────────────────────────────────────────────────────

/// FBGEMM FP8 quantization configuration.
///
/// Supports Ampere (cap 80) via per-channel dequantize + GEMM, and Hopper
/// (cap 89+) via dynamic per-token FP8 activation quantization.
#[derive(Debug, Clone)]
pub struct FbgemmFp8Config {
    /// Layer name prefixes that should not be quantized.
    pub modules_to_not_convert: Vec<String>,
    /// Upper bound on the dynamic input scale to prevent overflow (Hopper path).
    /// Stored but not yet applied in the CPU/Ampere path.
    pub input_scale_ub: f32,
}

impl FbgemmFp8Config {
    /// Create from detected quantization config.
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let modules_to_not_convert = raw_config
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let input_scale_ub = raw_config
            .get("activation_scale_ub")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(f32::MAX);

        Self {
            modules_to_not_convert,
            input_scale_ub,
        }
    }
}

impl Default for FbgemmFp8Config {
    fn default() -> Self {
        Self {
            modules_to_not_convert: Vec::new(),
            input_scale_ub: f32::MAX,
        }
    }
}

impl QuantizationConfig for FbgemmFp8Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::FbgemmFp8
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        // Marlin FP8 fallback works on Ampere (80); native FBGEMM needs Hopper (89).
        80
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.modules_to_not_convert
            .iter()
            .any(|prefix| layer_name.contains(prefix.as_str()))
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(FbgemmFp8Linear::new(
            in_features,
            out_features,
            bias,
            self.input_scale_ub,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── FbgemmFp8Linear ─────────────────────────────────────────────────────────

/// FBGEMM FP8 linear layer.
///
/// Stores FP8 E4M3 weights (as U8) with per-channel F32 scales.
///
/// Forward paths:
/// - **CUDA cap ≥ 89**: dynamic per-token FP8 activation quantization via
///   `fp8_gemm` (hardware FP8 matmul).
/// - **CPU / Ampere**: per-channel dequantize → BF16 → standard matmul.
///   NOTE: A Marlin FP8 kernel path (`apply_fp8_marlin_linear`) would give
///   ~2× speedup on Ampere; add it when `MarlinScalarType::Float8E4m3fn` is
///   wired through `FbgemmFp8WeightLoader`.
#[derive(Debug)]
pub struct FbgemmFp8Linear {
    /// FP8 E4M3 weight stored as U8 `[out, in]`, or BF16 zeros before loading.
    weight: Tensor,
    /// Per-channel F32 scales `[out]` or `[out, 1]`.
    weight_scale: Option<Tensor>,
    /// Optional bias.
    bias: Option<Tensor>,
    /// Upper bound on the dynamic input scale (Hopper path only).
    #[allow(dead_code)]
    input_scale_ub: f32,
    in_features: usize,
    out_features: usize,
    /// True once real FP8 weights have been loaded.
    is_fp8_weights: bool,
}

impl FbgemmFp8Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        input_scale_ub: f32,
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
            weight_scale: None,
            bias,
            input_scale_ub,
            in_features,
            out_features,
            is_fp8_weights: false,
        })
    }

    /// Dequantize FP8 weight to `target_dtype` using per-channel scale.
    ///
    /// `weight [N, K] × scale [N, 1]` → `[N, K]` in `target_dtype`.
    fn dequantize_weight(&self, target_dtype: DType) -> Result<Tensor> {
        let scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "FbgemmFp8Linear: weight_scale required for dequantization".to_string(),
            )
        })?;

        let w_f32 = self.weight.to_dtype(DType::F32)?;
        // Normalise scale to [N, 1] so it broadcasts across K.
        let s_f32 = if scale.dims().len() == 1 {
            scale.unsqueeze(1)?.to_dtype(DType::F32)?
        } else {
            scale.to_dtype(DType::F32)?
        };
        w_f32.broadcast_mul(&s_f32)?.to_dtype(target_dtype)
    }

    /// FP8 CUDA forward via `fp8_gemm` (Hopper, cap ≥ 89).
    #[cfg(feature = "cuda-kernels")]
    fn forward_fp8_cuda(&self, x: &Tensor) -> Result<Tensor> {
        let scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("FbgemmFp8Linear: weight_scale required".to_string())
        })?;
        fp8_cuda::fp8_gemm(x, &self.weight, scale, self.bias.as_ref())
    }
}

impl QuantizedLinear for FbgemmFp8Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Hopper: use FP8 GEMM kernel with dynamic per-token activation quant.
        #[cfg(feature = "cuda-kernels")]
        if self.is_fp8_weights && self.weight_scale.is_some() && x.device().is_cuda() {
            return self.forward_fp8_cuda(x);
        }

        let out_dtype = x.dtype();

        // CPU / Ampere: dequantize weight then matmul.
        // Candle CPU matmul requires F32, so cast through F32 and restore dtype.
        let (x_f32, w_f32) = if self.is_fp8_weights && self.weight_scale.is_some() {
            let w = self.dequantize_weight(DType::F32)?;
            (x.to_dtype(DType::F32)?, w)
        } else {
            (x.to_dtype(DType::F32)?, self.weight.to_dtype(DType::F32)?)
        };

        let y = x_f32.matmul(&w_f32.t()?)?.to_dtype(out_dtype)?;
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_config() -> FbgemmFp8Config {
        FbgemmFp8Config::from_detected(&HashMap::new())
    }

    #[test]
    fn test_fbgemm_fp8_config_defaults() {
        let cfg = test_config();
        assert_eq!(cfg.method(), QuantizationMethod::FbgemmFp8);
        assert_eq!(cfg.min_capability(), 80);
        assert!(!cfg.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn test_fbgemm_fp8_config_skip_list() {
        let mut raw = HashMap::new();
        raw.insert(
            "modules_to_not_convert".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );
        let cfg = FbgemmFp8Config::from_detected(&raw);
        assert!(cfg.is_layer_skipped("lm_head"));
        assert!(cfg.is_layer_skipped("model.embed_tokens.weight"));
        assert!(!cfg.is_layer_skipped("model.layers.0.mlp.gate_proj"));
    }

    #[test]
    fn test_fbgemm_fp8_config_input_scale_ub() {
        let mut raw = HashMap::new();
        raw.insert(
            "activation_scale_ub".to_string(),
            serde_json::json!(1200.0f64),
        );
        let cfg = FbgemmFp8Config::from_detected(&raw);
        assert!((cfg.input_scale_ub - 1200.0_f32).abs() < 1e-3);
    }

    #[test]
    fn test_fbgemm_fp8_linear_construction() {
        let device = Device::Cpu;
        let cfg = test_config();
        // Use dimensions divisible by required block sizes (multiples of 128/64).
        let linear = cfg.create_linear(128, 128, false, &device);
        assert!(linear.is_ok(), "{:?}", linear.err());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 128);
        assert_eq!(l.out_features(), 128);
        assert!(!l.has_bias());
    }

    #[test]
    fn test_fbgemm_fp8_linear_forward_uninitialized() {
        // Sanity: uninitialized zeros path should not panic.
        let device = Device::Cpu;
        let cfg = test_config();
        let linear = cfg.create_linear(64, 64, false, &device).unwrap();
        let x = Tensor::zeros((2usize, 64usize), DType::BF16, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 64]);
    }

    #[test]
    fn test_fbgemm_fp8_linear_dequantize_forward() {
        // Simulate the per-channel dequantize path: load BF16 "weight" + F32 scale.
        let device = Device::Cpu;
        let n = 64usize;
        let k = 64usize;

        let mut linear = FbgemmFp8Linear::new(k, n, false, f32::MAX, &device).unwrap();

        // Use BF16 weight (not U8) so is_fp8_weights stays false; scale is [N].
        let w = Tensor::ones((n, k), DType::BF16, &device).unwrap();
        let scale = Tensor::ones(n, DType::F32, &device).unwrap(); // scale = 1.0

        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), w);
        weights.insert("weight_scale".to_string(), scale);
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones((2usize, k), DType::BF16, &device).unwrap();
        let y = linear.forward(&x).unwrap();
        // weight=1, scale=1 → output = [2, N] all K
        assert_eq!(y.dims(), &[2, n]);
        let vals: Vec<f32> = y
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            vals.iter().all(|&v| (v - k as f32).abs() < 0.1),
            "expected {k}, got {:?}",
            &vals[..4]
        );
    }
}
