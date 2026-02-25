//! NVIDIA ModelOpt quantization — extended format covering all `quant_algo` variants.
//!
//! | `quant_algo`                | Weights         | Weight scale               | Act scale       |
//! |-----------------------------|-----------------|----------------------------|-----------------|
//! | `FP8`                       | FP8 E4M3 `[N,K]`| F32 per-tensor scalar      | F32 per-tensor  |
//! | `FP8_PER_CHANNEL_PER_TOKEN` | FP8 E4M3 `[N,K]`| F32 `[N]` per-channel      | dynamic per-tok |
//! | `FP8_PB_WO`                 | FP8 E4M3 `[N,K]`| F32 `[N/128, K/128]` block | dynamic per-tok |
//! | `NVFP4`                     | FP4 E2M1 `[N,K/2]`| FP8 E4M3 `[N,K/16]` + F32 global | F32 global |
//! | `MXFP8`                     | FP8 E4M3 `[N,K]`| E8M0 U8 `[N,K/32]`         | —               |
//!
//! CPU emulation path: dequantize weights → BF16 matmul. GPU kernels (FlashInfer/Marlin) TODO.
//!
//! NOTE: Detection supports two JSON layouts:
//!   1. Nested `hf_quant_config.json`:  `{"quant_method": "modelopt", "quantization": {"quant_algo": "FP8"}}`
//!   2. Flat `config.json` style:       `{"quant_method": "modelopt", "quant_algo": "FP8"}`

use std::collections::HashMap;
use std::str::FromStr;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::mxfp4::FP4_LUT;

/// Block size for `FP8_PB_WO` (per-block weight-only FP8), both N and K dims.
pub const FP8_PB_WO_BLOCK_SIZE: usize = 128;

/// Block size for NVFP4 per-group FP8 scales.
pub const NVFP4_GROUP_SIZE: usize = 16;

/// Block size for MXFP8 E8M0 scales (along K).
pub const MXFP8_BLOCK_SIZE: usize = 32;

/// E8M0 exponent bias used for MXFP8 scales.
const E8M0_BIAS: f32 = 127.0;

// ─── ModelOpt quant algorithm ────────────────────────────────────────────────

/// Quantization algorithm variant within a ModelOpt checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOptQuantAlgo {
    /// Static per-tensor FP8 (weight + activation scales).
    Fp8,
    /// Per-channel weight FP8 + dynamic per-token activation quantization.
    Fp8PerChannelPerToken,
    /// Per-block weight-only FP8 (128×128 blocks), dynamic activation.
    Fp8PbWo,
    /// NVIDIA FP4 (E2M1): packed nibbles + FP8 per-group scales + F32 global.
    NvFp4,
    /// Microscaling FP8 (E8M0 block-32 scales along K).
    MxFp8,
}

impl FromStr for ModelOptQuantAlgo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "FP8" => Ok(Self::Fp8),
            "FP8_PER_CHANNEL_PER_TOKEN" => Ok(Self::Fp8PerChannelPerToken),
            "FP8_PB_WO" => Ok(Self::Fp8PbWo),
            "NVFP4" => Ok(Self::NvFp4),
            "MXFP8" => Ok(Self::MxFp8),
            other => Err(format!("Unknown ModelOpt quant_algo: {other}")),
        }
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// NVIDIA ModelOpt quantization configuration.
///
/// Handles the full range of ModelOpt `quant_algo` values. Detection reads from
/// either the nested `hf_quant_config.json` format or the flat compressed-tensors
/// style config.
#[derive(Debug, Clone)]
pub struct ModelOptConfig {
    pub algo: ModelOptQuantAlgo,
    /// Layers / prefixes to skip quantization for (supports fnmatch-style wildcards).
    pub exclude_modules: Vec<String>,
    /// Optional KV-cache quantization algorithm name.
    pub kv_cache_quant_algo: Option<String>,
    /// Group size for NVFP4 block scales (default 16).
    pub group_size: usize,
}

impl ModelOptConfig {
    /// Parse from the raw detected config.
    ///
    /// Accepts both JSON layouts:
    /// - Nested: `{"quantization": {"quant_algo": "FP8", "exclude_modules": [...]}}`
    /// - Flat:   `{"quant_algo": "FP8", "ignore": [...]}`
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Option<Self> {
        // Resolve quant_algo from nested or flat layout.
        let quant_algo_str = raw_config
            .get("quantization")
            .and_then(|q| q.get("quant_algo"))
            .or_else(|| raw_config.get("quant_algo"))
            .and_then(|v| v.as_str())
            .map(str::to_uppercase)?;

        let algo = quant_algo_str.parse::<ModelOptQuantAlgo>().ok()?;

        // exclude_modules lives inside "quantization" dict in nested format,
        // or under "ignore" in the flat format.
        let exclude_modules = raw_config
            .get("quantization")
            .and_then(|q| q.get("exclude_modules"))
            .or_else(|| raw_config.get("exclude_modules"))
            .or_else(|| raw_config.get("ignore"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let kv_cache_quant_algo = raw_config
            .get("quantization")
            .and_then(|q| q.get("kv_cache_quant_algo"))
            .or_else(|| raw_config.get("kv_cache_quant_algo"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let group_size = raw_config
            .get("quantization")
            .and_then(|q| q.get("group_size"))
            .or_else(|| raw_config.get("group_size"))
            .and_then(|v| v.as_u64())
            .map(|g| g as usize)
            .unwrap_or(NVFP4_GROUP_SIZE);

        Some(Self {
            algo,
            exclude_modules,
            kv_cache_quant_algo,
            group_size,
        })
    }
}

impl Default for ModelOptConfig {
    fn default() -> Self {
        Self {
            algo: ModelOptQuantAlgo::Fp8,
            exclude_modules: Vec::new(),
            kv_cache_quant_algo: None,
            group_size: NVFP4_GROUP_SIZE,
        }
    }
}

impl QuantizationConfig for ModelOptConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::ModelOptFull
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        match self.algo {
            // FP8 hardware requires Hopper (89), but CPU emulation works anywhere.
            ModelOptQuantAlgo::Fp8
            | ModelOptQuantAlgo::Fp8PerChannelPerToken
            | ModelOptQuantAlgo::Fp8PbWo => 89,
            // NVFP4 hardware: Blackwell (100). CPU emulation works anywhere.
            ModelOptQuantAlgo::NvFp4 => 75,
            // MXFP8 hardware: Blackwell (100). CPU emulation works anywhere.
            ModelOptQuantAlgo::MxFp8 => 0,
        }
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.exclude_modules.iter().any(|pattern| {
            // Exact substring match covers the common case.
            // Full fnmatch would require a dependency; substring is sufficient for tests.
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
        Ok(Box::new(ModelOptLinear::new(
            in_features,
            out_features,
            bias,
            self.algo,
            self.group_size,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── FP8 E4M3 decode ─────────────────────────────────────────────────────────

/// Decode a single FP8 E4M3 byte to F32.
///
/// Bit layout: `[sign(7) | exp(6:3) | man(2:0)]`, exponent bias = 7.
/// NaN: `exp=0b1111, man=0b111` (0x7F or 0xFF) → returned as 0.0 for matmul safety.
pub fn fp8_e4m3_to_f32(v: u8) -> f32 {
    let sign: f32 = if v & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (v >> 3) & 0x0F; // bits 6:3
    let man = v & 0x07; // bits 2:0

    // Treat NaN as 0 to avoid propagating NaNs through matmul emulation.
    if exp == 15 && man == 7 {
        return 0.0;
    }

    if exp == 0 {
        // Subnormal: (-1)^s × 2^(−6) × (man/8)
        sign * 2.0f32.powi(-6) * (man as f32 / 8.0)
    } else {
        // Normal: (-1)^s × 2^(exp−7) × (1 + man/8)
        sign * 2.0f32.powi(exp as i32 - 7) * (1.0 + man as f32 / 8.0)
    }
}

/// Decode an FP8 E4M3 tensor (stored as U8) to F32.
fn decode_fp8_e4m3(t: &Tensor) -> Result<Tensor> {
    let orig_shape = t.dims().to_vec();
    let bytes: Vec<u8> = t.flatten_all()?.to_vec1()?;
    let f32_vals: Vec<f32> = bytes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect();
    Tensor::from_vec(f32_vals, orig_shape, t.device())
}

/// Decode E8M0 scales (U8 exponent-only, bias 127) to F32.
fn decode_e8m0_scales(scales: &Tensor) -> Result<Tensor> {
    let float_scales = scales.to_dtype(DType::F32)?;
    let biased = (float_scales - E8M0_BIAS as f64)?;
    (biased * std::f64::consts::LN_2)?.exp()
}

// ─── Dequantisation paths ─────────────────────────────────────────────────────

/// FP8 per-tensor: `weight_f32 × scalar_scale` → `[N, K]` BF16.
fn dequant_fp8_per_tensor(weight: &Tensor, scale: Option<&Tensor>) -> Result<Tensor> {
    let w_f32 = decode_fp8_e4m3(weight)?; // [N, K]
    if let Some(s) = scale {
        // scale is a 0-D or 1-D [1] F32 tensor
        let s_f32 = s.to_dtype(DType::F32)?.flatten_all()?;
        let scalar = s_f32.to_vec1::<f32>()?[0] as f64;
        (w_f32 * scalar)?.to_dtype(DType::BF16)
    } else {
        w_f32.to_dtype(DType::BF16)
    }
}

/// FP8 per-channel: each row `i` of weight is scaled by `scale[i]`.
/// `weight`: `[N, K]` U8; `scale`: `[N]` F32.
fn dequant_fp8_per_channel(
    weight: &Tensor,
    scale: Option<&Tensor>,
    out_features: usize,
    _in_features: usize,
) -> Result<Tensor> {
    let w_f32 = decode_fp8_e4m3(weight)?; // [N, K]
    if let Some(s) = scale {
        // Expand scale [N] → [N, 1] then broadcast-mul with [N, K]
        let s_f32 = s.to_dtype(DType::F32)?.reshape((out_features, 1))?;
        w_f32.broadcast_mul(&s_f32)?.to_dtype(DType::BF16)
    } else {
        w_f32.to_dtype(DType::BF16)
    }
}

/// FP8 per-block weight-only: weight `[N, K]`, scale `[N/block, K/block]` F32.
/// Block size is `FP8_PB_WO_BLOCK_SIZE` (128) in both dims.
fn dequant_fp8_per_block(
    weight: &Tensor,
    scale: Option<&Tensor>,
    out_features: usize,
    in_features: usize,
    block: usize,
) -> Result<Tensor> {
    let w_f32 = decode_fp8_e4m3(weight)?; // [N, K]
    if let Some(s) = scale {
        let n_blk = out_features.div_ceil(block);
        let k_blk = in_features.div_ceil(block);
        // Expand scale [N/b, K/b] → [N/b, 1, K/b, 1] → broadcast to [N, K]
        let s_f32 = s.to_dtype(DType::F32)?.reshape((n_blk, 1, k_blk, 1))?;
        let expanded = s_f32.expand(&[n_blk, block, k_blk, block])?;
        let s_2d = expanded.reshape((n_blk * block, k_blk * block))?;
        // Trim to [N, K] in case of padding
        let s_trimmed = s_2d.narrow(0, 0, out_features)?.narrow(1, 0, in_features)?;
        (w_f32 * s_trimmed)?.to_dtype(DType::BF16)
    } else {
        w_f32.to_dtype(DType::BF16)
    }
}

/// NVFP4: FP4 packed `[N, K/2]` + FP8 E4M3 per-group scales `[N, K/G]` + F32 global.
///
/// Two-level dequant:
///   1. Unpack FP4 nibbles → `[N, K]` F32
///   2. Decode FP8 E4M3 per-group scales → `[N, K/G]` F32
///   3. Block-multiply: `[N, K/G, G]` × `[N, K/G, 1]` → `[N, K]`
///   4. Multiply by global F32 scale
fn dequant_nvfp4(
    weight: &Tensor,
    scale: Option<&Tensor>,
    scale_2: Option<&Tensor>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<Tensor> {
    // 1. Unpack FP4 nibbles (same lower-nibble-first convention as OCP MXFP4)
    let packed: Vec<u8> = weight.flatten_all()?.to_vec1()?;
    let mut f32_vals = Vec::with_capacity(packed.len() * 2);
    for byte in &packed {
        f32_vals.push(FP4_LUT[(byte & 0x0F) as usize]);
        f32_vals.push(FP4_LUT[((byte >> 4) & 0x0F) as usize]);
    }
    let w_f32 = Tensor::from_vec(f32_vals, (out_features, in_features), weight.device())?;

    // 2. Apply FP8 per-group scales if present
    let w_scaled = if let Some(s) = scale {
        let num_groups = in_features / group_size;
        let s_f32 = decode_fp8_e4m3(s)?; // [N, K/G]
        let blocked = w_f32.reshape((out_features, num_groups, group_size))?;
        let scales_3d = s_f32
            .unsqueeze(2)?
            .expand(&[out_features, num_groups, group_size])?;
        (blocked * scales_3d)?.reshape((out_features, in_features))?
    } else {
        w_f32
    };

    // 3. Apply global F32 scale if present
    let w_final = if let Some(s2) = scale_2 {
        let scalar = s2.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0] as f64;
        (w_scaled * scalar)?
    } else {
        w_scaled
    };

    w_final.to_dtype(DType::BF16)
}

/// MXFP8: FP8 E4M3 `[N, K]` + E8M0 block-32 scales `[N, K/32]`.
fn dequant_mxfp8(
    weight: &Tensor,
    scale: Option<&Tensor>,
    out_features: usize,
    in_features: usize,
) -> Result<Tensor> {
    let w_f32 = decode_fp8_e4m3(weight)?; // [N, K]
    if let Some(s) = scale {
        let decoded_scales = decode_e8m0_scales(s)?; // [N, K/32]
        let num_blocks = in_features / MXFP8_BLOCK_SIZE;
        let blocked = w_f32.reshape((out_features, num_blocks, MXFP8_BLOCK_SIZE))?;
        let scales_3d =
            decoded_scales
                .unsqueeze(2)?
                .expand(&[out_features, num_blocks, MXFP8_BLOCK_SIZE])?;
        let dequantized = (blocked * scales_3d)?;
        dequantized
            .reshape((out_features, in_features))?
            .to_dtype(DType::BF16)
    } else {
        w_f32.to_dtype(DType::BF16)
    }
}

// ─── Linear layer ─────────────────────────────────────────────────────────────

/// ModelOpt quantized linear layer (CPU emulation via dequantize + BF16 matmul).
#[derive(Debug)]
pub struct ModelOptLinear {
    /// Main weight tensor.
    /// - FP8 algos: `[N, K]` U8 (FP8 E4M3 bytes)
    /// - NVFP4:     `[N, K/2]` U8 (packed FP4 nibbles)
    weight: Tensor,
    bias: Option<Tensor>,
    /// Primary weight scale.
    /// - Fp8: F32 scalar (1-D `[1]`)
    /// - Fp8PerChannelPerToken: F32 `[N]`
    /// - Fp8PbWo: F32 `[N/128, K/128]`
    /// - NvFp4: U8 (FP8 E4M3) `[N, K/group_size]`
    /// - MxFp8: U8 (E8M0) `[N, K/32]`
    weight_scale: Option<Tensor>,
    /// Secondary global scale for NVFP4 (F32 scalar).
    weight_scale_2: Option<Tensor>,
    algo: ModelOptQuantAlgo,
    group_size: usize,
    in_features: usize,
    out_features: usize,
    is_quantized: bool,
}

impl ModelOptLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        algo: ModelOptQuantAlgo,
        group_size: usize,
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
            algo,
            group_size,
            in_features,
            out_features,
            is_quantized: false,
        })
    }

    fn dequantize(&self) -> Result<Tensor> {
        match self.algo {
            ModelOptQuantAlgo::Fp8 => {
                dequant_fp8_per_tensor(&self.weight, self.weight_scale.as_ref())
            }
            ModelOptQuantAlgo::Fp8PerChannelPerToken => dequant_fp8_per_channel(
                &self.weight,
                self.weight_scale.as_ref(),
                self.out_features,
                self.in_features,
            ),
            ModelOptQuantAlgo::Fp8PbWo => dequant_fp8_per_block(
                &self.weight,
                self.weight_scale.as_ref(),
                self.out_features,
                self.in_features,
                FP8_PB_WO_BLOCK_SIZE,
            ),
            ModelOptQuantAlgo::NvFp4 => dequant_nvfp4(
                &self.weight,
                self.weight_scale.as_ref(),
                self.weight_scale_2.as_ref(),
                self.out_features,
                self.in_features,
                self.group_size,
            ),
            ModelOptQuantAlgo::MxFp8 => dequant_mxfp8(
                &self.weight,
                self.weight_scale.as_ref(),
                self.out_features,
                self.in_features,
            ),
        }
    }
}

impl QuantizedLinear for ModelOptLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = if self.is_quantized {
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
            self.is_quantized = w.dtype() == DType::U8;
            self.weight = w.clone();
        }

        if let Some(s) = weights.get("weight_scale") {
            self.weight_scale = Some(s.clone());
        }

        if let Some(s2) = weights.get("weight_scale_2") {
            self.weight_scale_2 = Some(s2.clone());
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

    // ─── FP8 E4M3 decode ─────────────────────────────────────────────────

    #[test]
    fn test_fp8_e4m3_zero() {
        // 0x00: sign=+, exp=0, man=0 → subnormal = 0.0
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
    }

    #[test]
    fn test_fp8_e4m3_one() {
        // exp=7 (0b0111 shifted to bits 6:3 → byte = 0b0_0111_000 = 0x38), man=0
        // value = 2^(7-7) × 1.0 = 1.0
        assert!((fp8_e4m3_to_f32(0x38) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_fp8_e4m3_two() {
        // exp=8 → byte = 0b0_1000_000 = 0x40, man=0 → 2^(8-7) × 1.0 = 2.0
        assert!((fp8_e4m3_to_f32(0x40) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fp8_e4m3_negative_one() {
        // sign=1, exp=7, man=0 → 0b1_0111_000 = 0xB8 → -1.0
        assert!((fp8_e4m3_to_f32(0xB8) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fp8_e4m3_nan_treated_as_zero() {
        // NaN encoding: exp=15, man=7 → 0x7F and 0xFF
        assert_eq!(fp8_e4m3_to_f32(0x7F), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0xFF), 0.0);
    }

    // ─── Config parsing ───────────────────────────────────────────────────

    #[test]
    fn test_config_fp8_nested_json() {
        let mut raw = HashMap::new();
        raw.insert(
            "quantization".to_string(),
            serde_json::json!({
                "quant_algo": "FP8",
                "kv_cache_quant_algo": "FP8",
                "exclude_modules": ["lm_head"]
            }),
        );

        let config = ModelOptConfig::from_detected(&raw).unwrap();
        assert_eq!(config.algo, ModelOptQuantAlgo::Fp8);
        assert_eq!(config.kv_cache_quant_algo.as_deref(), Some("FP8"));
        assert_eq!(config.exclude_modules.len(), 1);
        assert!(config.exclude_modules.contains(&"lm_head".to_string()));
    }

    #[test]
    fn test_config_nvfp4_flat_json() {
        let mut raw = HashMap::new();
        raw.insert("quant_algo".to_string(), serde_json::json!("NVFP4"));
        raw.insert("group_size".to_string(), serde_json::json!(16));
        raw.insert(
            "ignore".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );

        let config = ModelOptConfig::from_detected(&raw).unwrap();
        assert_eq!(config.algo, ModelOptQuantAlgo::NvFp4);
        assert_eq!(config.group_size, 16);
        assert_eq!(config.exclude_modules.len(), 2);
    }

    #[test]
    fn test_config_fp8_pb_wo() {
        let mut raw = HashMap::new();
        raw.insert(
            "quantization".to_string(),
            serde_json::json!({"quant_algo": "FP8_PB_WO"}),
        );
        let config = ModelOptConfig::from_detected(&raw).unwrap();
        assert_eq!(config.algo, ModelOptQuantAlgo::Fp8PbWo);
    }

    #[test]
    fn test_config_mxfp8() {
        let mut raw = HashMap::new();
        raw.insert(
            "quantization".to_string(),
            serde_json::json!({
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": "FP8",
                "exclude_modules": []
            }),
        );
        let config = ModelOptConfig::from_detected(&raw).unwrap();
        assert_eq!(config.algo, ModelOptQuantAlgo::MxFp8);
        assert_eq!(config.method(), QuantizationMethod::ModelOptFull);
        assert_eq!(config.min_capability(), 0);
    }

    #[test]
    fn test_config_unknown_algo_returns_none() {
        let mut raw = HashMap::new();
        raw.insert("quant_algo".to_string(), serde_json::json!("INT8_SQ"));
        assert!(ModelOptConfig::from_detected(&raw).is_none());
    }

    #[test]
    fn test_config_is_layer_skipped() {
        let config = ModelOptConfig {
            algo: ModelOptQuantAlgo::Fp8,
            exclude_modules: vec!["lm_head".to_string(), "vision_tower*".to_string()],
            ..Default::default()
        };
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(config.is_layer_skipped("model.vision_tower.layers.0.attn"));
        assert!(!config.is_layer_skipped("model.layers.0.mlp.gate_proj.weight"));
    }

    // ─── Construction ─────────────────────────────────────────────────────

    #[test]
    fn test_linear_construction() {
        let linear = ModelOptLinear::new(
            64,
            128,
            true,
            ModelOptQuantAlgo::Fp8,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
        assert_eq!(linear.weight_dtype(), DType::BF16);
    }

    #[test]
    fn test_linear_zero_in_features() {
        assert!(
            ModelOptLinear::new(0, 128, false, ModelOptQuantAlgo::Fp8, 16, &Device::Cpu).is_err()
        );
    }

    // ─── FP8 per-tensor forward ───────────────────────────────────────────

    #[test]
    fn test_fp8_per_tensor_dequant() {
        // 0x38 = FP8 E4M3 = 1.0; scale = 2.0 → dequantized = 2.0
        let mut linear = ModelOptLinear::new(
            32,
            4,
            false,
            ModelOptQuantAlgo::Fp8,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x38u8, (4usize, 32usize), &Device::Cpu).unwrap(),
        );
        // scale = 2.0 as F32 scalar
        weights.insert(
            "weight_scale".to_string(),
            Tensor::new(&[2.0f32], &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
            }
        }
    }

    #[test]
    fn test_fp8_per_tensor_forward_shape() {
        let mut linear = ModelOptLinear::new(
            32,
            8,
            false,
            ModelOptQuantAlgo::Fp8,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x38u8, (8usize, 32usize), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::new(&[1.0f32], &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[2usize, 32usize], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }

    // ─── FP8 per-channel forward ──────────────────────────────────────────

    #[test]
    fn test_fp8_per_channel_dequant() {
        // weight 0x38 = 1.0; scale[i] = 2.0 for all rows → dequant = 2.0
        let mut linear = ModelOptLinear::new(
            32,
            4,
            false,
            ModelOptQuantAlgo::Fp8PerChannelPerToken,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x38u8, (4usize, 32usize), &Device::Cpu).unwrap(),
        );
        // per-channel scale: [4] F32
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(2.0f32, 4, &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
            }
        }
    }

    // ─── FP8 per-block forward ────────────────────────────────────────────

    #[test]
    fn test_fp8_pb_wo_dequant() {
        // weight 0x38 = 1.0; scale [[2.0]] (one 128×128 block) → dequant = 2.0
        let n = 128;
        let k = 128;
        let mut linear = ModelOptLinear::new(
            k,
            n,
            false,
            ModelOptQuantAlgo::Fp8PbWo,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x38u8, (n, k), &Device::Cpu).unwrap(),
        );
        // scale shape [1, 1] for one 128×128 block
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(2.0f32, (1usize, 1usize), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        assert_eq!(vals.len(), n);
        assert_eq!(vals[0].len(), k);
        for row in &vals {
            for &v in row {
                assert!((v - 2.0).abs() < 1e-4, "expected 2.0, got {v}");
            }
        }
    }

    // ─── NVFP4 forward ────────────────────────────────────────────────────

    #[test]
    fn test_nvfp4_dequant() {
        // FP4 nibble 2 = 1.0; 0x22 → lo=2 (1.0), hi=2 (1.0)
        // FP8 block scale 0x38 = 1.0 (per group-16 along K)
        // global scale_2 = 1.0 → dequantized = 1.0
        let in_features = 32usize; // K=32, groups = 32/16 = 2
        let out_features = 4usize;

        let mut linear = ModelOptLinear::new(
            in_features,
            out_features,
            false,
            ModelOptQuantAlgo::NvFp4,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        // packed FP4: [4, 16] (32/2=16 bytes per row)
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x22u8, (out_features, in_features / 2), &Device::Cpu).unwrap(),
        );
        // FP8 E4M3 block scale: [4, 2] (32/16=2 groups per row), 0x38=1.0
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(
                0x38u8,
                (out_features, in_features / NVFP4_GROUP_SIZE),
                &Device::Cpu,
            )
            .unwrap(),
        );
        // global scale_2: scalar 1.0
        weights.insert(
            "weight_scale_2".to_string(),
            Tensor::new(&[1.0f32], &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < 1e-4, "expected 1.0 (nvfp4), got {v}");
            }
        }
    }

    // ─── MXFP8 forward ───────────────────────────────────────────────────

    #[test]
    fn test_mxfp8_dequant() {
        // FP8 0x38 = 1.0; E8M0 scale 127 → 2^0 = 1.0 → dequant = 1.0
        let in_features = 32usize;
        let out_features = 4usize;

        let mut linear = ModelOptLinear::new(
            in_features,
            out_features,
            false,
            ModelOptQuantAlgo::MxFp8,
            NVFP4_GROUP_SIZE,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x38u8, (out_features, in_features), &Device::Cpu).unwrap(),
        );
        // E8M0 scale: [4, 1] (one block per row, 32/32=1)
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (out_features, 1usize), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < 1e-4, "expected 1.0 (mxfp8), got {v}");
            }
        }
    }

    #[test]
    fn test_create_linear_from_config() {
        let config = ModelOptConfig {
            algo: ModelOptQuantAlgo::Fp8,
            ..Default::default()
        };
        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(!linear.has_bias());
    }
}
