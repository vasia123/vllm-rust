//! FP-Quant quantization (https://arxiv.org/abs/2509.23202)
//!
//! Weights are stored as packed FP4 E2M1 nibbles with per-group scales and an
//! optional global F32 scale.  Two `forward_dtype` variants:
//!
//! | `forward_dtype` | `group_size` | scale dtype | GPU GEMM |
//! |---|---|---|---|
//! | `"mxfp4"` | 32 | E8M0 (`uint8`) | `matmul_mxf4_bf16_tn` (Blackwell) |
//! | `"nvfp4"` | 16 | FP8 E4M3 (`uint8`) | `cutlass_scaled_fp4_mm` (Blackwell) |
//!
//! At GPU-inference time the method also applies a Hadamard rotation to
//! activations before quantizing them to FP4.  Our CPU path dequantizes only
//! the **weights** (offline) and then does a standard BF16 matmul — no
//! activation quantization and no Hadamard transform are applied.
//!
//! Weight tensors in the HF checkpoint:
//! - `qweight`           — `[N, K/2]` uint8 (packed FP4 nibbles, lower nibble first)
//! - `scales`            — `[N, K/group_size]` uint8 (E8M0 or E4M3 per-group scales)
//! - `weight_global_scale` — `[1]` F32 (secondary global scale)
//! - `act_global_scale`  — `[1]` F32 (activation-side; ignored on CPU path)
//! - `forward_hadamard_matrix` / `backward_hadamard_matrix` — skipped on CPU
//!
//! Detection: `quant_type = "fp_quant"` in `quantize_config.json`.
//! Min capability: 100 (Blackwell H200/B200) for the GPU FP4-GEMM path.
//!
//! Reference: `vllm/model_executor/layers/quantization/fp_quant.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

// ─── Forward-dtype ────────────────────────────────────────────────────────────

/// Which FP4 kernel variant the checkpoint targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpQuantForwardDtype {
    /// OCP Microscaling FP4 (group_size=32, E8M0 per-group scales).
    MxFp4,
    /// NVIDIA FP4 (group_size=16, FP8 E4M3 per-group scales).
    NvFp4,
}

impl FpQuantForwardDtype {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "nvfp4" => Self::NvFp4,
            _ => Self::MxFp4, // "mxfp4" and unknown → MxFp4
        }
    }

    fn group_size(self) -> usize {
        match self {
            Self::MxFp4 => 32,
            Self::NvFp4 => 16,
        }
    }
}

// ─── FpQuantConfig ────────────────────────────────────────────────────────────

/// Configuration for FP-Quant quantization.
#[derive(Debug, Clone)]
pub struct FpQuantConfig {
    pub forward_dtype: FpQuantForwardDtype,
    pub hadamard_group_size: usize,
    pub modules_to_not_convert: Vec<String>,
}

impl FpQuantConfig {
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let forward_dtype_str = raw_config
            .get("forward_dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("mxfp4");

        let hadamard_group_size = raw_config
            .get("hadamard_group_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(32);

        let modules_to_not_convert = raw_config
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            forward_dtype: FpQuantForwardDtype::from_str(forward_dtype_str),
            hadamard_group_size,
            modules_to_not_convert,
        }
    }
}

impl QuantizationConfig for FpQuantConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::FpQuant
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::BF16]
    }

    /// GPU FP4-GEMM requires Blackwell (100). CPU dequant path works anywhere.
    fn min_capability(&self) -> u32 {
        0
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.modules_to_not_convert.iter().any(|pattern| {
            layer_name.contains(pattern.trim_end_matches('*').trim_end_matches('.'))
        })
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(FpQuantLinear::new(
            in_features,
            out_features,
            bias,
            self.forward_dtype,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── Scale decoding ───────────────────────────────────────────────────────────

/// FP4 E2M1 lookup table: index = 4-bit nibble value → F32.
///
/// Bit layout: `[sign(3) | exp(2:1) | man(0)]`; exponent bias = 1.
const FP4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // negative
];

/// Decode E8M0 scales: `2^(byte − 127)`.
fn decode_e8m0(scales: &Tensor) -> Result<Tensor> {
    let float_scales = scales.to_dtype(DType::F32)?;
    let biased = (float_scales - 127.0_f64)?;
    (biased * std::f64::consts::LN_2)?.exp()
}

/// Decode FP8 E4M3 (bias=7) stored as uint8 → F32.
fn decode_fp8_e4m3(t: &Tensor) -> Result<Tensor> {
    let bytes: Vec<u8> = t.flatten_all()?.to_vec1()?;
    let f32_vals: Vec<f32> = bytes
        .iter()
        .map(|&v| {
            let sign: f32 = if v & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = (v >> 3) & 0x0F;
            let man = v & 0x07;
            if exp == 15 && man == 7 {
                return 0.0; // NaN → 0 for safe matmul
            }
            if exp == 0 {
                sign * 2.0f32.powi(-6) * (man as f32 / 8.0)
            } else {
                sign * 2.0f32.powi(exp as i32 - 7) * (1.0 + man as f32 / 8.0)
            }
        })
        .collect();
    Tensor::from_vec(f32_vals, t.dims().to_vec(), t.device())
}

// ─── Dequantization ───────────────────────────────────────────────────────────

/// Unpack packed FP4 nibbles and apply per-group + global scales → BF16.
///
/// * `weight`        – `[N, K/2]` uint8 (lower nibble = first element)
/// * `scale`         – `[N, K/group_size]` uint8 (E8M0 for mxfp4, E4M3 for nvfp4)
/// * `global_scale`  – `[1]` F32 (optional secondary scale)
fn dequant_fp4(
    weight: &Tensor,
    scale: Option<&Tensor>,
    global_scale: Option<&Tensor>,
    out_features: usize,
    in_features: usize,
    dtype: FpQuantForwardDtype,
) -> Result<Tensor> {
    let group_size = dtype.group_size();

    // 1. Unpack FP4 nibbles → [N, K] F32
    let packed: Vec<u8> = weight.flatten_all()?.to_vec1()?;
    let mut f32_vals = Vec::with_capacity(packed.len() * 2);
    for byte in &packed {
        f32_vals.push(FP4_LUT[(byte & 0x0F) as usize]);
        f32_vals.push(FP4_LUT[((byte >> 4) & 0x0F) as usize]);
    }
    let w_f32 = Tensor::from_vec(f32_vals, (out_features, in_features), weight.device())?;

    // 2. Apply per-group scales if present
    let w_scaled = if let Some(s) = scale {
        let num_groups = in_features / group_size;
        let s_f32 = match dtype {
            FpQuantForwardDtype::MxFp4 => decode_e8m0(s)?,
            FpQuantForwardDtype::NvFp4 => decode_fp8_e4m3(s)?,
        };
        let s_f32 = s_f32.reshape((out_features, num_groups))?;
        let blocked = w_f32.reshape((out_features, num_groups, group_size))?;
        let scales_3d = s_f32
            .unsqueeze(2)?
            .expand(&[out_features, num_groups, group_size])?;
        (blocked * scales_3d)?.reshape((out_features, in_features))?
    } else {
        w_f32
    };

    // 3. Apply global F32 scale if present
    let w_final = if let Some(gs) = global_scale {
        let scalar = gs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0] as f64;
        (w_scaled * scalar)?
    } else {
        w_scaled
    };

    w_final.to_dtype(DType::BF16)
}

// ─── Linear layer ─────────────────────────────────────────────────────────────

/// FP-Quant quantized linear layer.
///
/// Dequantizes FP4 weights to BF16 at forward time (CPU emulation path).
/// GPU path (Blackwell FP4-GEMM) requires `min_capability=100` and CUDA custom ops.
#[derive(Debug)]
pub struct FpQuantLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    /// Per-group scales: `[N, K/group_size]` uint8 (E8M0 or E4M3).
    weight_scale: Option<Tensor>,
    /// Global F32 scale scalar `[1]`.
    weight_scale_2: Option<Tensor>,
    forward_dtype: FpQuantForwardDtype,
    in_features: usize,
    out_features: usize,
    is_quantized: bool,
}

impl FpQuantLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        forward_dtype: FpQuantForwardDtype,
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
            weight_scale_2: None,
            forward_dtype,
            in_features,
            out_features,
            is_quantized: false,
        })
    }
}

impl QuantizedLinear for FpQuantLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight_bf16 = if self.is_quantized {
            dequant_fp4(
                &self.weight,
                self.weight_scale.as_ref(),
                self.weight_scale_2.as_ref(),
                self.out_features,
                self.in_features,
                self.forward_dtype,
            )?
        } else {
            self.weight.to_dtype(DType::BF16)?
        };

        // BF16 matmul is not available on CPU; use F32 for portability.
        let (x_f32, weight_f32) = (x.to_dtype(DType::F32)?, weight_bf16.to_dtype(DType::F32)?);
        let y = x_f32
            .matmul(&weight_f32.t()?)?
            .to_dtype(x.dtype())?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        // FP-Quant checkpoint keys differ from standard "weight"/"weight_scale".
        if let Some(w) = weights.get("qweight").or_else(|| weights.get("weight")) {
            self.is_quantized = w.dtype() == DType::U8;
            self.weight = w.clone();
        }

        if let Some(s) = weights.get("scales").or_else(|| weights.get("weight_scale")) {
            self.weight_scale = Some(s.clone());
        }

        // Weight global scale (secondary): key is "weight_global_scale" in HF format.
        if let Some(s2) = weights
            .get("weight_global_scale")
            .or_else(|| weights.get("weight_scale_2"))
        {
            self.weight_scale_2 = Some(s2.clone());
        }

        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }

        // act_global_scale, forward_hadamard_matrix, backward_hadamard_matrix:
        // These are activation-side tensors used by the GPU FP4 kernel.
        // Silently ignore on the CPU dequant path.

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

    fn default_cfg() -> FpQuantConfig {
        FpQuantConfig::from_detected(&HashMap::new())
    }

    #[test]
    fn test_fp_quant_method() {
        assert_eq!(default_cfg().method(), QuantizationMethod::FpQuant);
    }

    #[test]
    fn test_fp_quant_min_capability() {
        assert_eq!(default_cfg().min_capability(), 0);
    }

    #[test]
    fn test_fp_quant_default_dtype_is_mxfp4() {
        let cfg = default_cfg();
        assert_eq!(cfg.forward_dtype, FpQuantForwardDtype::MxFp4);
        assert_eq!(cfg.hadamard_group_size, 32);
    }

    #[test]
    fn test_fp_quant_nvfp4_config() {
        let mut raw = HashMap::new();
        raw.insert(
            "forward_dtype".to_string(),
            serde_json::json!("nvfp4"),
        );
        raw.insert("hadamard_group_size".to_string(), serde_json::json!(16));
        let cfg = FpQuantConfig::from_detected(&raw);
        assert_eq!(cfg.forward_dtype, FpQuantForwardDtype::NvFp4);
        assert_eq!(cfg.hadamard_group_size, 16);
        assert_eq!(cfg.forward_dtype.group_size(), 16);
    }

    #[test]
    fn test_fp_quant_modules_to_not_convert() {
        let mut raw = HashMap::new();
        raw.insert(
            "modules_to_not_convert".to_string(),
            serde_json::json!(["lm_head"]),
        );
        let cfg = FpQuantConfig::from_detected(&raw);
        assert!(cfg.is_layer_skipped("model.lm_head"));
        assert!(!cfg.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn test_fp_quant_mxfp4_dequant() {
        // Pack a simple mxfp4 weight: all-zero nibbles → all 0.0
        let device = Device::Cpu;
        let n: usize = 4;
        let k: usize = 32;

        let mut cfg = FpQuantConfig::from_detected(&HashMap::new());
        cfg.forward_dtype = FpQuantForwardDtype::MxFp4;
        let mut linear = cfg.create_linear(k, n, false, &device).unwrap();

        // qweight: [4, 16] uint8 (all zeros → nibbles = 0 → FP4_LUT[0] = 0.0)
        let qweight = Tensor::zeros((n, k / 2), DType::U8, &device).unwrap();
        // E8M0 scale 127 → 2^0 = 1.0; [4, 1] (one group per row since K/32=1)
        let scales =
            Tensor::from_vec(vec![127u8; n * (k / 32)], (n, k / 32), &device).unwrap();
        // global scale = 1.0 (F32)
        let global = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();

        let mut weights = HashMap::new();
        weights.insert("qweight".to_string(), qweight);
        weights.insert("scales".to_string(), scales);
        weights.insert("weight_global_scale".to_string(), global);
        linear.load_weights(&weights).unwrap();

        // x: [1, 32] BF16
        let x = Tensor::zeros((1, k), DType::BF16, &device).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, n], "output shape should be [1, N]");
    }

    #[test]
    fn test_fp_quant_nvfp4_dequant() {
        // NvFp4: group_size=16, FP8 E4M3 scales
        let device = Device::Cpu;
        let n: usize = 4;
        let k: usize = 32;

        let mut raw = HashMap::new();
        raw.insert("forward_dtype".to_string(), serde_json::json!("nvfp4"));
        let cfg = FpQuantConfig::from_detected(&raw);
        let mut linear = cfg.create_linear(k, n, false, &device).unwrap();

        // qweight: [4, 16] uint8 all zeros
        let qweight = Tensor::zeros((n, k / 2), DType::U8, &device).unwrap();
        // FP8 E4M3 scale 0x38 = 1.0; [4, 2] (k/16=2 groups per row)
        let scales =
            Tensor::from_vec(vec![0x38u8; n * (k / 16)], (n, k / 16), &device).unwrap();
        let global = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();

        let mut weights = HashMap::new();
        weights.insert("qweight".to_string(), qweight);
        weights.insert("scales".to_string(), scales);
        weights.insert("weight_global_scale".to_string(), global);
        linear.load_weights(&weights).unwrap();

        let x = Tensor::zeros((1, k), DType::BF16, &device).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, n]);
    }
}
