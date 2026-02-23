//! MXFP4 (OCP Microscaling FP4 E2M1) quantization.
//!
//! Format:
//! - Weights: FP4 E2M1 (4-bit), packed 2 per U8 byte (lower nibble first)
//! - Scales:  uint8 E8M0 (exponent-only, bias 127), one scale per 32-element block
//! - Block size: 32 elements along K dimension
//!
//! FP4 E2M1 value encoding (bit layout: [sign | exp1 | exp0 | mant]):
//!   Normal (e≠0): `(-1)^s × 2^(e-1) × (1 + m/2)`
//!   Subnormal (e=0): `(-1)^s × m/2`
//!   → values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
//!
//! Decoding (CPU emulation path):
//!   1. Unpack nibbles: `[N, K/2]` U8 → `[N, K]` F32 via 16-entry lookup table
//!   2. Decode E8M0 scales: `2^(u8 − 127)`, expand to `[N, K/32, 1]`
//!   3. Reshape weights to `[N, K/32, 32]`, multiply by scales, flatten
//!   4. Cast to BF16 and matmul
//!
//! NOTE: Linear layer dequantization is the CPU emulation path. MoE FP4 dispatch
//! via FlashInfer/Marlin/Triton kernels is not yet wired (separate TODO).

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

/// Block size along K dimension for MXFP4 quantization.
pub const MXFP4_BLOCK_SIZE: usize = 32;

/// E8M0 exponent bias (bias-127 encoding).
const E8M0_BIAS: f32 = 127.0;

/// FP4 E2M1 lookup table, indexed by the 4-bit nibble value.
///
/// Bit layout: [s(3) | e1(2) | e0(1) | m(0)]
///   Normal  (e≠0): `(-1)^s × 2^(e−1) × (1 + m×0.5)`
///   Subnorm (e=0): `(-1)^s × m×0.5`
pub const FP4_LUT: [f32; 16] = [
    0.0,  // 0b0000  s=0 e=0 m=0  → zero
    0.5,  // 0b0001  s=0 e=0 m=1  → subnormal 0.5
    1.0,  // 0b0010  s=0 e=1 m=0  → 2^0 × 1.0
    1.5,  // 0b0011  s=0 e=1 m=1  → 2^0 × 1.5
    2.0,  // 0b0100  s=0 e=2 m=0  → 2^1 × 1.0
    3.0,  // 0b0101  s=0 e=2 m=1  → 2^1 × 1.5
    4.0,  // 0b0110  s=0 e=3 m=0  → 2^2 × 1.0
    6.0,  // 0b0111  s=0 e=3 m=1  → 2^2 × 1.5
    0.0,  // 0b1000  s=1 e=0 m=0  → negative zero (= 0)
    -0.5, // 0b1001  s=1 e=0 m=1  → −0.5
    -1.0, // 0b1010  s=1 e=1 m=0  → −1.0
    -1.5, // 0b1011  s=1 e=1 m=1  → −1.5
    -2.0, // 0b1100  s=1 e=2 m=0  → −2.0
    -3.0, // 0b1101  s=1 e=2 m=1  → −3.0
    -4.0, // 0b1110  s=1 e=3 m=0  → −4.0
    -6.0, // 0b1111  s=1 e=3 m=1  → −6.0
];

/// MXFP4 quantization configuration.
#[derive(Debug, Clone, Default)]
pub struct MxFp4Config {
    /// Layers to skip (not quantize).
    pub ignored_layers: Vec<String>,
}

impl MxFp4Config {
    /// Create from detected raw config.
    ///
    /// Expected JSON structure:
    /// ```json
    /// { "quant_method": "mxfp4", "ignored_layers": ["lm_head"] }
    /// ```
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let ignored_layers = raw_config
            .get("ignored_layers")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self { ignored_layers }
    }
}

impl QuantizationConfig for MxFp4Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Mxfp4
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        // Hardware FP4 kernels require Ampere or newer.
        // CPU emulation path (this module) works on any device.
        80
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
        Ok(Box::new(MxFp4Linear::new(
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

/// Decode E8M0 scales (uint8 exponent-only, bias 127) to F32.
///
/// `decoded = 2^(raw_u8 - 127)`
fn decode_e8m0_scales(scales: &Tensor) -> Result<Tensor> {
    let float_scales = scales.to_dtype(DType::F32)?;
    let biased = (float_scales - E8M0_BIAS as f64)?;
    (biased * std::f64::consts::LN_2)?.exp()
}

/// Unpack FP4 E2M1 nibbles from packed U8 bytes to a flat F32 vector.
///
/// Packing convention: lower nibble (bits 3:0) = first element,
/// upper nibble (bits 7:4) = second element.
///
/// Input shape: any — treated as flat. Output: 1-D F32 of length `2 × n_bytes`.
fn unpack_fp4_to_f32(packed: &Tensor) -> Result<Tensor> {
    let packed_data: Vec<u8> = packed.flatten_all()?.to_vec1()?;
    let mut f32_vals = Vec::with_capacity(packed_data.len() * 2);

    for byte in &packed_data {
        let lo = (byte & 0x0F) as usize;
        let hi = ((byte >> 4) & 0x0F) as usize;
        f32_vals.push(FP4_LUT[lo]);
        f32_vals.push(FP4_LUT[hi]);
    }

    Tensor::from_vec(f32_vals, packed_data.len() * 2, packed.device())
}

/// MXFP4 quantized linear layer (CPU emulation via dequantize + BF16 matmul).
///
/// Weights are stored as packed FP4 E2M1 (`[N, K/2]` U8) and block scales
/// are E8M0 (`[N, K/32]` U8).  On `forward()` the layer dequantizes to BF16
/// and performs a standard matmul, matching vLLM's Python fallback behaviour.
#[derive(Debug)]
pub struct MxFp4Linear {
    /// Packed FP4 weight: `[N, K/2]` U8 when loaded from checkpoint,
    /// or `[N, K]` BF16 for the zero-initialized placeholder.
    weight: Tensor,
    bias: Option<Tensor>,
    /// E8M0 block scales: `[N, K/32]` U8.
    weight_scale: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    is_fp4_weights: bool,
}

impl MxFp4Linear {
    /// Create a new zero-initialised MXFP4 linear layer.
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
            is_fp4_weights: false,
        })
    }

    /// Dequantize packed FP4 weights to BF16.
    ///
    /// Steps:
    /// 1. Unpack nibbles `[N, K/2]` U8 → `[N, K]` F32
    /// 2. Decode E8M0 scales → `[N, K/32]` F32
    /// 3. Reshape to `[N, K/32, 32]`, broadcast-multiply by scale, flatten
    /// 4. Cast to BF16
    fn dequantize(&self) -> Result<Tensor> {
        let scale = self.weight_scale.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("MXFP4 dequantization requires weight_scale".to_string())
        })?;

        let decoded_scales = decode_e8m0_scales(scale)?; // [N, K/32]

        // Unpack nibbles from [N, K/2] U8 → flat [N*K] F32
        let flat_f32 = unpack_fp4_to_f32(&self.weight)?;
        let weights_f32 = flat_f32.reshape((self.out_features, self.in_features))?;

        let num_blocks = self.in_features / MXFP4_BLOCK_SIZE;

        // [N, K] → [N, K/32, 32]
        let blocked = weights_f32.reshape((self.out_features, num_blocks, MXFP4_BLOCK_SIZE))?;

        // Expand scales [N, K/32] → [N, K/32, 1] for broadcasting
        let scales_3d = decoded_scales.unsqueeze(2)?.expand(&[
            self.out_features,
            num_blocks,
            MXFP4_BLOCK_SIZE,
        ])?;

        let dequantized = (blocked * scales_3d)?;
        let flat = dequantized.reshape((self.out_features, self.in_features))?;
        flat.to_dtype(DType::BF16)
    }
}

impl QuantizedLinear for MxFp4Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = if self.is_fp4_weights && self.weight_scale.is_some() {
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
            // Packed FP4: shape [N, K/2] stored as U8 (2 nibbles per byte)
            self.is_fp4_weights = w.dtype() == DType::U8;
            self.weight = w.clone();
        }

        if let Some(s) = weights.get("weight_scale") {
            if !self.in_features.is_multiple_of(MXFP4_BLOCK_SIZE) {
                candle_core::bail!(
                    "MXFP4 requires in_features ({}) divisible by block_size ({})",
                    self.in_features,
                    MXFP4_BLOCK_SIZE,
                );
            }

            let scale_dims = s.dims();
            if scale_dims.len() == 2 {
                let expected_blocks = self.in_features / MXFP4_BLOCK_SIZE;
                if scale_dims[1] != expected_blocks {
                    candle_core::bail!(
                        "MXFP4 weight_scale K-dim {} doesn't match expected {} \
                         (in_features={} / block_size={})",
                        scale_dims[1],
                        expected_blocks,
                        self.in_features,
                        MXFP4_BLOCK_SIZE,
                    );
                }
                if scale_dims[0] != self.out_features {
                    candle_core::bail!(
                        "MXFP4 weight_scale N-dim {} doesn't match out_features {}",
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

    // ─── FP4 LUT tests ──────────────────────────────────────────────────

    #[test]
    fn test_fp4_lut_positive() {
        assert_eq!(FP4_LUT[0], 0.0); // 0b0000
        assert_eq!(FP4_LUT[1], 0.5); // 0b0001  subnormal
        assert_eq!(FP4_LUT[2], 1.0); // 0b0010
        assert_eq!(FP4_LUT[3], 1.5); // 0b0011
        assert_eq!(FP4_LUT[4], 2.0); // 0b0100
        assert_eq!(FP4_LUT[5], 3.0); // 0b0101
        assert_eq!(FP4_LUT[6], 4.0); // 0b0110
        assert_eq!(FP4_LUT[7], 6.0); // 0b0111
    }

    #[test]
    fn test_fp4_lut_negative() {
        assert_eq!(FP4_LUT[8], 0.0); // 0b1000  negative zero
        assert_eq!(FP4_LUT[9], -0.5); // 0b1001
        assert_eq!(FP4_LUT[10], -1.0); // 0b1010
        assert_eq!(FP4_LUT[11], -1.5); // 0b1011
        assert_eq!(FP4_LUT[12], -2.0); // 0b1100
        assert_eq!(FP4_LUT[13], -3.0); // 0b1101
        assert_eq!(FP4_LUT[14], -4.0); // 0b1110
        assert_eq!(FP4_LUT[15], -6.0); // 0b1111
    }

    // ─── Unpack tests ───────────────────────────────────────────────────

    #[test]
    fn test_unpack_fp4_nibbles() {
        // 0x23 = 0b0010_0011 → lo=3 (1.5), hi=2 (1.0)
        let packed = Tensor::new(&[0x23u8], &Device::Cpu).unwrap();
        let unpacked: Vec<f32> = unpack_fp4_to_f32(&packed).unwrap().to_vec1().unwrap();
        assert_eq!(unpacked.len(), 2);
        assert!(
            (unpacked[0] - 1.5).abs() < 1e-6,
            "lo nibble: {}",
            unpacked[0]
        );
        assert!(
            (unpacked[1] - 1.0).abs() < 1e-6,
            "hi nibble: {}",
            unpacked[1]
        );
    }

    #[test]
    fn test_unpack_fp4_all_zeros() {
        // 0x00 → lo=0 (0.0), hi=0 (0.0)
        let packed = Tensor::new(&[0x00u8, 0x00u8], &Device::Cpu).unwrap();
        let unpacked: Vec<f32> = unpack_fp4_to_f32(&packed).unwrap().to_vec1().unwrap();
        assert_eq!(unpacked.len(), 4);
        assert!(unpacked.iter().all(|&v| v == 0.0));
    }

    // ─── E8M0 decode tests ───────────────────────────────────────────────

    #[test]
    fn test_e8m0_decode_identity() {
        // scale = 127 → 2^(127-127) = 2^0 = 1.0
        let scales = Tensor::new(&[127u8], &Device::Cpu).unwrap();
        let decoded: Vec<f32> = decode_e8m0_scales(&scales).unwrap().to_vec1().unwrap();
        assert!((decoded[0] - 1.0).abs() < 1e-6, "got {}", decoded[0]);
    }

    #[test]
    fn test_e8m0_decode_double() {
        // scale = 128 → 2^(128-127) = 2^1 = 2.0
        let scales = Tensor::new(&[128u8], &Device::Cpu).unwrap();
        let decoded: Vec<f32> = decode_e8m0_scales(&scales).unwrap().to_vec1().unwrap();
        assert!((decoded[0] - 2.0).abs() < 1e-6, "got {}", decoded[0]);
    }

    // ─── Config tests ────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = MxFp4Config::default();
        assert_eq!(config.method(), QuantizationMethod::Mxfp4);
        assert_eq!(config.supported_act_dtypes(), &[DType::BF16]);
        assert_eq!(config.min_capability(), 80);
        assert!(config.ignored_layers.is_empty());
    }

    #[test]
    fn test_config_from_detected() {
        let mut raw = HashMap::new();
        raw.insert(
            "ignored_layers".to_string(),
            serde_json::json!(["lm_head", "embed_tokens"]),
        );

        let config = MxFp4Config::from_detected(&raw);
        assert_eq!(config.ignored_layers.len(), 2);
        assert!(config.ignored_layers.contains(&"lm_head".to_string()));
    }

    #[test]
    fn test_config_is_layer_skipped() {
        let config = MxFp4Config {
            ignored_layers: vec!["lm_head".to_string()],
        };
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("model.layers.0.mlp.gate_proj.weight"));
    }

    // ─── Linear construction tests ────────────────────────────────────────

    #[test]
    fn test_linear_construction() {
        let linear = MxFp4Linear::new(64, 128, true, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
        assert!(!linear.is_fp4_weights);
        assert_eq!(linear.weight_dtype(), DType::BF16);
    }

    #[test]
    fn test_linear_construction_zero_in_features() {
        assert!(MxFp4Linear::new(0, 128, false, &Device::Cpu).is_err());
    }

    #[test]
    fn test_linear_construction_zero_out_features() {
        assert!(MxFp4Linear::new(64, 0, false, &Device::Cpu).is_err());
    }

    // ─── Dequantize tests ─────────────────────────────────────────────────

    #[test]
    fn test_dequantize_identity_scales() {
        // FP4 nibble 2 = 1.0; pack as 0x22 (both lo and hi nibble = 2)
        // in_features=32 (one block), out_features=4, packed shape [4, 16]
        let mut linear = MxFp4Linear::new(32, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        // Each byte 0x22 → lo=2 (1.0), hi=2 (1.0)
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x22u8, (4usize, 16usize), &Device::Cpu).unwrap(),
        );
        // Scale = 127 → 1.0 (identity), shape [4, 1]
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (4usize, 1usize), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        let vals: Vec<Vec<f32>> = dequantized.to_dtype(DType::F32).unwrap().to_vec2().unwrap();

        for row in &vals {
            for &v in row {
                assert!((v - 1.0).abs() < 1e-4, "expected 1.0, got {v}");
            }
        }
    }

    #[test]
    fn test_dequantize_double_scales() {
        // FP4 nibble 2 = 1.0; scale 128 → 2.0; dequantized = 2.0
        let mut linear = MxFp4Linear::new(32, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x22u8, (4usize, 16usize), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(128u8, (4usize, 1usize), &Device::Cpu).unwrap(),
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
    fn test_block_size_validation() {
        // K=33 is not divisible by 32
        let mut linear = MxFp4Linear::new(33, 4, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            // Shape [4, 17] (ceil(33/2) bytes, approximating packed FP4)
            Tensor::zeros((4usize, 17usize), DType::U8, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::zeros((4usize, 1usize), DType::U8, &Device::Cpu).unwrap(),
        );

        // Loading scale should fail: 33 not divisible by 32
        let result = linear.load_weights(&weights);
        assert!(
            result.is_err(),
            "expected error for K=33 not divisible by 32"
        );
    }

    // ─── Forward tests ────────────────────────────────────────────────────

    #[test]
    fn test_linear_forward_with_dequant() {
        // Weight = all 1.0 (nibble 2, scale 127=1.0), input = all 1.0
        // Output[i,j] = sum_{k=0}^{31} 1.0 * 1.0 = 32.0
        let mut linear = MxFp4Linear::new(32, 8, false, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::full(0x22u8, (8usize, 16usize), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (8usize, 1usize), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[2usize, 32usize], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);

        let vals: Vec<Vec<f32>> = y.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                // 32 elements × weight=1.0 × scale=1.0 × input=1.0
                assert!((v - 32.0).abs() < 0.5, "expected ~32.0, got {v}");
            }
        }
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let mut linear = MxFp4Linear::new(32, 8, true, &Device::Cpu).unwrap();

        let mut weights = HashMap::new();
        // Zero weight → matmul = 0; bias = 1 → output = 1
        weights.insert(
            "weight".to_string(),
            Tensor::zeros((8usize, 16usize), DType::U8, &Device::Cpu).unwrap(),
        );
        weights.insert(
            "weight_scale".to_string(),
            Tensor::full(127u8, (8usize, 1usize), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "bias".to_string(),
            Tensor::ones(8, DType::F32, &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let x = Tensor::ones(&[3usize, 32usize], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[3, 8]);

        let vals: Vec<Vec<f32>> = y.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
        for row in &vals {
            for &v in row {
                // nibble 0 = 0.0, scale 1.0, weight=0 → matmul=0, bias=1 → out=1
                assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
            }
        }
    }

    #[test]
    fn test_create_linear_from_config() {
        let config = MxFp4Config::default();
        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_linear_forward_unquantized_fallback() {
        // Without loading FP4 weights, should use BF16 zero weights
        let linear = MxFp4Linear::new(32, 8, false, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[2usize, 32usize], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }
}
