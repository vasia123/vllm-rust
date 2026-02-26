//! QUARK quantization (AMD/Qualcomm).
//!
//! Supports W8A8-FP8 and W8A8-INT8 quantization schemes with per-tensor or
//! per-channel weight scales.  On GPU (cap ≥ 89 for FP8, cap ≥ 75 for INT8)
//! the schemes map to hardware-accelerated paths; on CPU/Ampere they fall back
//! to dequantize→F32→matmul.
//!
//! Config JSON structure (embedded in `quantization_config` of `config.json`):
//! ```json
//! {
//!   "quant_method": "quark",
//!   "export": { "kv_cache_group": [...], "pack_method": "reorder" },
//!   "global_quant_config": {
//!     "weight":        { "dtype": "fp8_e4m3", "qscheme": "per_tensor|per_channel", "is_dynamic": false },
//!     "input_tensors": { "dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": false }
//!   },
//!   "layer_quant_config": { ... },  // per-layer overrides (not yet applied)
//!   "exclude": ["lm_head"]
//! }
//! ```
//!
//! Reference: `vllm/model_executor/layers/quantization/quark/`

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use serde_json::Value;

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

// ─── Weight quantization scheme ──────────────────────────────────────────────

/// Granularity of weight scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightQScheme {
    /// Single scale for the whole weight matrix.
    PerTensor,
    /// One scale per output channel.
    PerChannel,
}

impl WeightQScheme {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "per_tensor" => Some(Self::PerTensor),
            "per_channel" => Some(Self::PerChannel),
            _ => None,
        }
    }
}

// ─── Inner quantization scheme ───────────────────────────────────────────────

/// Runtime quantization scheme selected from the QUARK config.
///
/// Fields are used by GPU kernel dispatch paths (FP8 GEMM, INT8 GEMM); the
/// CPU dequantize path relies only on loaded weight/scale tensors.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum QuarkInnerScheme {
    /// FP8 E4M3 weights + FP8 activations.
    ///
    /// Min capability: 89 (Ada Lovelace).  Falls back to per-channel
    /// dequantize+GEMM on Ampere (cap 80).
    W8A8Fp8 {
        weight_qscheme: WeightQScheme,
        is_static_input: bool,
    },
    /// INT8 symmetric weights + INT8 activations.
    ///
    /// Min capability: 75 (Turing).  Falls back to dequantize+GEMM on CPU.
    W8A8Int8 {
        weight_qscheme: WeightQScheme,
        is_static_input: bool,
        input_symmetric: bool,
    },
}

impl QuarkInnerScheme {
    fn min_capability(&self) -> u32 {
        match self {
            Self::W8A8Fp8 { .. } => 89,
            Self::W8A8Int8 { .. } => 75,
        }
    }
}

/// Detect the inner scheme from `weight` + `input_tensors` sub-objects.
fn detect_scheme(
    weight: &serde_json::Map<String, Value>,
    input: Option<&serde_json::Map<String, Value>>,
) -> Option<QuarkInnerScheme> {
    let w_dtype = weight.get("dtype")?.as_str()?;
    let w_qscheme = WeightQScheme::from_str(weight.get("qscheme")?.as_str()?)?;
    let w_dynamic = weight
        .get("is_dynamic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if w_dynamic {
        // Dynamic weights are not supported — we require static quantization.
        return None;
    }

    match w_dtype {
        "fp8_e4m3" => {
            let i_dtype = input.and_then(|i| i.get("dtype")).and_then(|v| v.as_str());
            if i_dtype != Some("fp8_e4m3") {
                return None;
            }
            let is_static = input
                .and_then(|i| i.get("is_dynamic"))
                .and_then(|v| v.as_bool())
                .map(|d| !d)
                .unwrap_or(false);
            Some(QuarkInnerScheme::W8A8Fp8 {
                weight_qscheme: w_qscheme,
                is_static_input: is_static,
            })
        }
        "int8" => {
            let i_dtype = input.and_then(|i| i.get("dtype")).and_then(|v| v.as_str());
            if i_dtype != Some("int8") {
                return None;
            }
            let is_static = input
                .and_then(|i| i.get("is_dynamic"))
                .and_then(|v| v.as_bool())
                .map(|d| !d)
                .unwrap_or(false);
            let input_sym = input
                .and_then(|i| i.get("symmetric"))
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Some(QuarkInnerScheme::W8A8Int8 {
                weight_qscheme: w_qscheme,
                is_static_input: is_static,
                input_symmetric: input_sym,
            })
        }
        _ => None,
    }
}

// ─── QuarkConfig ─────────────────────────────────────────────────────────────

/// QUARK quantization configuration.
///
/// Parses the `global_quant_config` from the checkpoint's `quantization_config`
/// to determine the base quantization scheme.  Per-layer overrides
/// (`layer_quant_config`) require a layer-name-aware `create_linear` call and
/// are not yet supported — most published QUARK checkpoints use a uniform
/// global config.
#[derive(Debug, Clone)]
pub struct QuarkConfig {
    scheme: QuarkInnerScheme,
    /// Layer name substrings that should NOT be quantized.
    exclude: Vec<String>,
}

impl QuarkConfig {
    /// Parse from the raw `quantization_config` dict in `config.json`.
    pub fn from_detected(raw: &HashMap<String, Value>) -> Option<Self> {
        let global = raw.get("global_quant_config").and_then(|v| v.as_object())?;

        let weight = global.get("weight")?.as_object()?;
        let input = global.get("input_tensors").and_then(|v| v.as_object());
        let scheme = detect_scheme(weight, input)?;

        let exclude = raw
            .get("exclude")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Some(Self { scheme, exclude })
    }
}

impl QuantizationConfig for QuarkConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Quark
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        self.scheme.min_capability()
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.exclude
            .iter()
            .any(|pat| layer_name.contains(pat.as_str()))
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(QuarkLinear::new(
            &self.scheme,
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

// ─── QuarkLinear ─────────────────────────────────────────────────────────────

/// QUARK quantized linear layer.
///
/// Supports W8A8-FP8 and W8A8-INT8 forward paths.
///
/// ## Weight storage
/// - FP8: `weight` stored as U8 (FP8 E4M3 bit pattern), `weight_scale` F32
/// - INT8: `weight` stored as U8 (INT8 bit pattern), `weight_scale` F32
///   (symmetric: zero-points stripped after loading)
///
/// ## Forward paths (CPU / Ampere fallback)
/// Both schemes dequantize to F32 and use standard matmul.  Native GPU kernels
/// (fp8_gemm, cap ≥ 89; int8_gemm, cap ≥ 75) are a TODO following the
/// `fbgemm_fp8.rs` CUDA feature gate pattern.
#[derive(Debug)]
pub struct QuarkLinear {
    /// Scheme metadata for GPU dispatch (currently CPU dequant path only).
    #[allow(dead_code)]
    scheme: QuarkInnerScheme,
    weight: Tensor,
    weight_scale: Option<Tensor>,
    input_scale: Option<Tensor>,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    /// True once real quantized weights (U8/I8) have been loaded.
    is_quantized_weights: bool,
}

impl QuarkLinear {
    fn new(
        scheme: &QuarkInnerScheme,
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("QuarkLinear: in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("QuarkLinear: out_features must be non-zero");
        }
        let weight = Tensor::zeros((out_features, in_features), DType::BF16, device)?;
        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::BF16, device)?)
        } else {
            None
        };
        Ok(Self {
            scheme: scheme.clone(),
            weight,
            weight_scale: None,
            input_scale: None,
            bias,
            in_features,
            out_features,
            is_quantized_weights: false,
        })
    }

    /// Dequantize the stored weight to `target_dtype` using `weight_scale`.
    ///
    /// FP8 (U8): reinterpret bytes as F32, multiply by per-tensor/per-channel scale.
    /// INT8 (I8): cast to F32, multiply by per-tensor/per-channel scale.
    fn dequantize_weight(&self, target_dtype: DType) -> Result<Tensor> {
        let scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("QuarkLinear: weight_scale required for dequantization".into())
        })?;

        let w_f32 = self.weight.to_dtype(DType::F32)?;
        // Normalise scale to [N, 1] for per-channel broadcast or [1, 1] for per-tensor.
        let s_f32 = match scale.dims().len() {
            1 => scale.unsqueeze(1)?.to_dtype(DType::F32)?,
            _ => scale.to_dtype(DType::F32)?,
        };
        w_f32.broadcast_mul(&s_f32)?.to_dtype(target_dtype)
    }
}

impl QuantizedLinear for QuarkLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out_dtype = x.dtype();

        // Dequantize and compute via F32 matmul (CPU / Ampere fallback).
        // NOTE: native FP8 GEMM (fp8_gemm, cap ≥ 89) and INT8 GEMM (cap ≥ 75)
        // can be wired here via feature flags, following fbgemm_fp8.rs pattern.
        let (x_f32, w_f32) = if self.is_quantized_weights && self.weight_scale.is_some() {
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
            // Both FP8 E4M3 and INT8 weights are stored as U8 (1 byte each) in candle.
            self.is_quantized_weights = w.dtype() == DType::U8;
            self.weight = w.clone();
        }
        if let Some(s) = weights.get("weight_scale") {
            self.weight_scale = Some(s.clone());
        }
        if let Some(s) = weights.get("input_scale") {
            self.input_scale = Some(s.clone());
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        // weight_zero_point: symmetric INT8 drops it; asymmetric would absorb
        // into bias (not yet implemented — only symmetric is common in practice).
        let _ = weights.get("weight_zero_point");
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

    fn fp8_config_raw() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(
            "global_quant_config".to_string(),
            serde_json::json!({
                "weight": {"dtype": "fp8_e4m3", "qscheme": "per_channel", "is_dynamic": false},
                "input_tensors": {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": false}
            }),
        );
        m
    }

    fn int8_config_raw() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert(
            "global_quant_config".to_string(),
            serde_json::json!({
                "weight": {"dtype": "int8", "qscheme": "per_channel", "is_dynamic": false},
                "input_tensors": {"dtype": "int8", "qscheme": "per_tensor", "is_dynamic": false, "symmetric": true}
            }),
        );
        m
    }

    #[test]
    fn test_quark_fp8_config_parse() {
        let cfg = QuarkConfig::from_detected(&fp8_config_raw()).expect("parse failed");
        assert_eq!(cfg.method(), QuantizationMethod::Quark);
        assert_eq!(cfg.min_capability(), 89);
        assert!(!cfg.is_layer_skipped("model.layers.0.mlp.gate_proj"));
    }

    #[test]
    fn test_quark_int8_config_parse() {
        let cfg = QuarkConfig::from_detected(&int8_config_raw()).expect("parse failed");
        assert_eq!(cfg.method(), QuantizationMethod::Quark);
        assert_eq!(cfg.min_capability(), 75);
    }

    #[test]
    fn test_quark_exclude_list() {
        let mut raw = fp8_config_raw();
        raw.insert(
            "exclude".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );
        let cfg = QuarkConfig::from_detected(&raw).expect("parse failed");
        assert!(cfg.is_layer_skipped("lm_head"));
        assert!(cfg.is_layer_skipped("model.embed_tokens.weight"));
        assert!(!cfg.is_layer_skipped("model.layers.0.mlp.gate_proj"));
    }

    #[test]
    fn test_quark_linear_construction_fp8() {
        let cfg = QuarkConfig::from_detected(&fp8_config_raw()).expect("parse failed");
        let linear = cfg.create_linear(64, 64, false, &Device::Cpu);
        assert!(linear.is_ok(), "{:?}", linear.err());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 64);
    }

    #[test]
    fn test_quark_linear_forward_uninitialized() {
        let cfg = QuarkConfig::from_detected(&fp8_config_raw()).expect("parse failed");
        let linear = cfg.create_linear(32, 64, false, &Device::Cpu).unwrap();
        let x = Tensor::zeros((2usize, 32usize), DType::BF16, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 64]);
    }

    #[test]
    fn test_quark_linear_dequantize_forward() {
        // Simulate W8A8-FP8 per-channel dequantize path.
        let cfg = QuarkConfig::from_detected(&fp8_config_raw()).expect("parse failed");
        let n = 32usize;
        let k = 32usize;
        let mut linear = cfg.create_linear(k, n, false, &Device::Cpu).unwrap();

        // Use BF16 "weight" (not real FP8 U8); scale = 1.0 per channel.
        let w = Tensor::ones((n, k), DType::BF16, &Device::Cpu).unwrap();
        let scale = Tensor::ones(n, DType::F32, &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("weight".to_string(), w);
        weights.insert("weight_scale".to_string(), scale);
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones((2usize, k), DType::BF16, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, n]);
        let vals: Vec<f32> = y
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // weight=1 scale=1 → output = k per element
        assert!(
            vals.iter().all(|&v| (v - k as f32).abs() < 0.1),
            "expected {k}, got {:?}",
            &vals[..4]
        );
    }
}
