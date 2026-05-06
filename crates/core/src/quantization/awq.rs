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
    check_marlin_supports_shape, MarlinConfig, MarlinLinear, MARLIN_SUPPORTED_GROUP_SIZES,
};

/// AWQ-gemm nibble interleaving within a u32 word.
///
/// In HuggingFace AWQ gemm checkpoints, the 8 int4 values stored in
/// a single u32 encode the 8 adjacent **output** dimensions at the nibble
/// positions given by this table: output `i` of the block lives at nibble
/// `AWQ_UNDO_PACK[i]`, so
///   output 0 ← bits 0..3,
///   output 1 ← bits 16..19,
///   output 2 ← bits 4..7,
///   output 3 ← bits 20..23,
///   output 4 ← bits 8..11,
///   output 5 ← bits 24..27,
///   output 6 ← bits 12..15,
///   output 7 ← bits 28..31.
///
/// This matches `undo_pack = [0, 4, 1, 5, 2, 6, 3, 7]` used by vLLM's
/// `awq_marlin_repack` kernel and by our own `repack_awq_nibbles`
/// helper in `awq_marlin.rs`.
const AWQ_UNDO_PACK: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Dequantize AWQ-gemm weights into a dense `[in_features, out_features]`
/// F16 tensor on the qweight's device.
///
/// Input shapes (matching HuggingFace AWQ gemm format):
/// - `qweight`: `[in_features, out_features / 8]`, I32 or U32 (packed)
/// - `qzeros`:  `[num_groups, out_features / 8]`, I32 or U32 (packed)
/// - `scales`:  `[num_groups, out_features]`, F16 / F32 / BF16
///
/// Formula: `weight[i, o] = (int4(w) - int4(z)) * scale[group(i), o]`
/// where `group(i) = i / group_size`.
///
/// The computation runs entirely on the CPU (scalar Rust) because the
/// packing is bit-level and the tensor sizes for a single linear layer
/// are small enough that a well-vectorised loop is fast. The result is
/// uploaded back to the original device before returning so downstream
/// matmuls stay where the caller put them.
pub fn awq_dequantize_cpu(
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    group_size: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    const PACK_FACTOR: usize = 8;

    if !out_features.is_multiple_of(PACK_FACTOR) {
        candle_core::bail!(
            "awq_dequantize_cpu: out_features ({out_features}) must be divisible by 8"
        );
    }
    let packed_out = out_features / PACK_FACTOR;

    // Candle does not expose an I32 DType — HF safetensors I32 tensors
    // arrive as U32 (bit-reinterpret) because the packed AWQ nibbles are
    // unsigned anyway. Accept U32 only.
    if qweight.dtype() != DType::U32 {
        candle_core::bail!(
            "awq_dequantize_cpu: qweight must be U32, got {:?}",
            qweight.dtype()
        );
    }
    if qzeros.dtype() != DType::U32 {
        candle_core::bail!(
            "awq_dequantize_cpu: qzeros must be U32, got {:?}",
            qzeros.dtype()
        );
    }

    let expected_qw_shape = &[in_features, packed_out];
    if qweight.dims() != expected_qw_shape {
        candle_core::bail!(
            "awq_dequantize_cpu: qweight shape {:?} != expected {:?}",
            qweight.dims(),
            expected_qw_shape
        );
    }

    let num_groups = if group_size == 0 {
        1
    } else {
        in_features.div_ceil(group_size)
    };
    let effective_group_size = if group_size == 0 {
        in_features.max(1)
    } else {
        group_size
    };

    if qzeros.dims() != [num_groups, packed_out] {
        candle_core::bail!(
            "awq_dequantize_cpu: qzeros shape {:?} != expected {:?}",
            qzeros.dims(),
            [num_groups, packed_out]
        );
    }
    if scales.dims() != [num_groups, out_features] {
        candle_core::bail!(
            "awq_dequantize_cpu: scales shape {:?} != expected {:?}",
            scales.dims(),
            [num_groups, out_features]
        );
    }

    let original_device = qweight.device().clone();

    // Pull everything to the CPU as plain Vecs. `to_dtype` on I32→U32 in
    // Candle is a numeric cast (not a bit reinterpret), so we go via I32
    // and then `as u32` which IS a bit-reinterpret in Rust. The packed
    // values are effectively unsigned nibbles; negative i32 values only
    // appear because the high nibble's top bit can be set.
    let qw_words: Vec<u32> = qweight.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let qz_words: Vec<u32> = qzeros.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;

    let scales_f32: Vec<f32> = scales
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?;

    let mut weight = vec![0.0f32; in_features * out_features];

    for in_idx in 0..in_features {
        let group = (in_idx / effective_group_size).min(num_groups - 1);
        let qz_row = group * packed_out;
        let sc_row = group * out_features;
        let qw_row = in_idx * packed_out;
        let w_row = in_idx * out_features;
        for out_packed in 0..packed_out {
            let w_word = qw_words[qw_row + out_packed];
            let z_word = qz_words[qz_row + out_packed];
            let out_base = out_packed * PACK_FACTOR;
            for (lane, &undo) in AWQ_UNDO_PACK.iter().enumerate() {
                let nibble_shift = undo * 4;
                let w_val = ((w_word >> nibble_shift) & 0xF) as i32;
                let z_val = ((z_word >> nibble_shift) & 0xF) as i32;
                let out_idx = out_base + lane;
                let scale = scales_f32[sc_row + out_idx];
                weight[w_row + out_idx] = (w_val - z_val) as f32 * scale;
            }
        }
    }

    Tensor::from_vec(weight, (in_features, out_features), &Device::Cpu)?
        .to_dtype(DType::F16)?
        .to_device(&original_device)
}

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
    /// Explicit list of module name substrings whose layers must NOT be
    /// quantized (loaded as plain fp16/bf16 weights instead). This mirrors
    /// the `modules_to_not_convert` field HuggingFace AWQ configs use to
    /// exclude `lm_head`, specific MoE blocks, etc.
    pub modules_to_not_convert: Vec<String>,
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
            // `lm_head` is excluded by default — the HF AWQ convention is
            // to leave the vocabulary projection in fp16 unless the
            // checkpoint specifically lists it in `modules_to_not_convert`.
            modules_to_not_convert: vec!["lm_head".to_string()],
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

        // HF AWQ uses `modules_to_not_convert` (list of strings) — every
        // layer whose full prefix contains one of these substrings is
        // loaded as a plain fp16 weight. Default to excluding `lm_head`
        // when the config doesn't say otherwise (matches `autoawq`).
        let modules_to_not_convert = raw_config
            .get("modules_to_not_convert")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec!["lm_head".to_string()]);

        Self {
            bits,
            group_size,
            zero_point,
            version,
            use_marlin: is_marlin || bits == 4,
            modules_to_not_convert,
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

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.modules_to_not_convert
            .iter()
            .any(|pat| layer_name.contains(pat))
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

        // AWQ GEMM layout: qweight is packed along the OUTPUT axis.
        // HuggingFace stores shape `[in_features, out_features / pack_factor]`
        // with each int32 word encoding 8 int4 values for 8 neighbouring
        // output dims (not input dims — that's GPTQ). This is the
        // opposite packing orientation to GPTQ so we must not reuse
        // GPTQ's `[in_features/pack, out_features]` shape here.
        let pack_factor = 32 / bits as usize;
        let packed_out = out_features.div_ceil(pack_factor);

        let num_groups = if group_size <= 0 {
            1
        } else {
            in_features.div_ceil(group_size as usize)
        };

        // Initialize with zeros (weights loaded later). Stored as I32 to
        // match HuggingFace safetensors dtype; the bit pattern is the
        // same as U32 but Candle reads it without a numeric conversion.
        let qweight = Tensor::zeros((in_features, packed_out), DType::U32, device)?;
        let scales = Tensor::zeros((num_groups, out_features), DType::F16, device)?;
        let qzeros = Tensor::zeros((num_groups, packed_out), DType::U32, device)?;

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

    /// Dequantize AWQ weights to a dense `[in_features, out_features]`
    /// F16 tensor on the layer's device.
    ///
    /// Shared between CPU and GPU paths: the math is fast enough to run
    /// entirely in scalar Rust for typical layer sizes, and the result
    /// gets uploaded back to the original device before matmul.
    /// A dedicated AWQ CUDA kernel can later replace this for throughput —
    /// it is NOT the GPTQ kernel (AWQ packs along the output axis, GPTQ
    /// packs along the input axis; they are not interchangeable).
    fn dequantize(&self) -> Result<Tensor> {
        if !self.is_quantized {
            candle_core::bail!(
                "AWQ layer has no quantized weights loaded - call load_weights() first"
            );
        }
        let group_size = if self.group_size <= 0 {
            self.in_features
        } else {
            self.group_size as usize
        };
        awq_dequantize_cpu(
            &self.qweight,
            &self.scales,
            &self.qzeros,
            group_size,
            self.in_features,
            self.out_features,
        )
    }

    /// Check if weights are quantized.
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }
}

impl QuantizedLinear for AwqLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantized weight lives in F16 but the caller's activation
        // dtype may be F32/BF16 — preserve the input dtype on the way
        // out so downstream ops (norms, RoPE) don't see a mixed-dtype
        // tensor. Candle's matmul does not auto-broadcast a 2D weight
        // over a 3D batched input, so we flatten leading dims first.
        let orig_dtype = x.dtype();
        let weight_f16 = self.dequantize()?;
        let compute_dtype = DType::F32;
        let weight = weight_f16.to_dtype(compute_dtype)?;
        let x_compute = x.to_dtype(compute_dtype)?;

        let dims = x_compute.dims().to_vec();
        if dims.is_empty() {
            candle_core::bail!("AwqLinear forward: scalar input is not supported");
        }
        let in_features = *dims.last().unwrap();
        if in_features != self.in_features {
            candle_core::bail!(
                "AwqLinear forward: last dim {in_features} != in_features {}",
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2d = x_compute.reshape((leading, in_features))?;
        let y2d = x2d.matmul(&weight)?;

        let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        let y = y2d.reshape(out_dims)?;

        let y = match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(compute_dtype)?)?,
            None => y,
        };
        y.to_dtype(orig_dtype)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("qweight") {
            // HuggingFace stores AWQ qweight as int32; Candle reads that
            // as U32 (bit-reinterpret, safe because the packed nibbles
            // are semantically unsigned).
            if w.dtype() != DType::U32 {
                candle_core::bail!("AWQ qweight must be U32, got {:?}", w.dtype());
            }
            self.is_quantized = true;
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
    use super::super::marlin::MarlinScalarType;
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
        raw.insert("is_marlin_format".to_string(), serde_json::json!(true));
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

            // Create quantized weights using the canonical AWQ layout:
            //   qweight: [in_features, out_features / pack_factor]
            //   scales:  [num_groups,  out_features]
            //   qzeros:  [num_groups,  out_features / pack_factor]
            let qweight = Tensor::zeros(
                &[in_features, out_features / pack_factor],
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
