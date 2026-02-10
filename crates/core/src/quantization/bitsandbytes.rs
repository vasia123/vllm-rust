//! BitsAndBytes NF4/INT8 quantization configuration and layers.
//!
//! BitsAndBytes provides two quantization modes:
//! - NF4 (4-bit NormalFloat): Uses a fixed lookup table of 16 values optimized
//!   for normally-distributed weights (QLoRA paper).
//! - INT8: Standard int8 quantization with absmax scaling.
//!
//! Reference: https://arxiv.org/abs/2305.14314

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

// Canonical reference values from the QLoRA paper — keep full f64 precision
// so the f32 rounding is identical across platforms.
#[allow(clippy::excessive_precision)]
const NF4_LOOKUP: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// BitsAndBytes quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbQuantType {
    /// 4-bit NormalFloat quantization
    NF4,
    /// 8-bit integer quantization
    INT8,
}

impl std::fmt::Display for BnbQuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NF4 => write!(f, "nf4"),
            Self::INT8 => write!(f, "int8"),
        }
    }
}

/// BitsAndBytes quantization configuration.
#[derive(Debug, Clone)]
pub struct BitsAndBytesConfig {
    /// Quantization type (NF4 or INT8)
    pub quant_type: BnbQuantType,
    /// Block size for quantization (default 64)
    pub block_size: usize,
    /// Whether to use double quantization (quantize the scales themselves)
    pub double_quant: bool,
    /// Layers to skip (not quantize)
    pub ignored_layers: Vec<String>,
}

impl BitsAndBytesConfig {
    /// Create a NF4 (4-bit) configuration with default settings.
    pub fn nf4() -> Self {
        Self {
            quant_type: BnbQuantType::NF4,
            block_size: 64,
            double_quant: false,
            ignored_layers: Vec::new(),
        }
    }

    /// Create an INT8 configuration with default settings.
    pub fn int8() -> Self {
        Self {
            quant_type: BnbQuantType::INT8,
            block_size: 64,
            double_quant: false,
            ignored_layers: Vec::new(),
        }
    }

    /// Create from detected config in HuggingFace checkpoint.
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let load_in_8bit = raw_config
            .get("load_in_8bit")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let quant_type = if load_in_8bit {
            BnbQuantType::INT8
        } else {
            BnbQuantType::NF4
        };

        let block_size = raw_config
            .get("bnb_4bit_blocksize")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(64);

        let double_quant = raw_config
            .get("bnb_4bit_use_double_quant")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let ignored_layers = raw_config
            .get("llm_int8_skip_modules")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Self {
            quant_type,
            block_size,
            double_quant,
            ignored_layers,
        }
    }

    /// Set ignored layers.
    pub fn with_ignored_layers(mut self, layers: Vec<String>) -> Self {
        self.ignored_layers = layers;
        self
    }

    /// Set block size.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set double quantization.
    pub fn with_double_quant(mut self, double_quant: bool) -> Self {
        self.double_quant = double_quant;
        self
    }
}

impl Default for BitsAndBytesConfig {
    fn default() -> Self {
        Self::nf4()
    }
}

impl QuantizationConfig for BitsAndBytesConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::BitsAndBytes
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F32, DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        70 // Volta
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        // Match component-level like the Python implementation
        let components: Vec<&str> = layer_name.split('.').collect();
        self.ignored_layers.iter().any(|ignored| {
            components.contains(&ignored.as_str())
                || components.iter().enumerate().any(|(i, _)| {
                    let prefix = components[..=i].join(".");
                    prefix == *ignored
                })
        })
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(BitsAndBytesLinear::new(
            in_features,
            out_features,
            bias,
            self.quant_type,
            self.block_size,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// BitsAndBytes quantized linear layer.
///
/// Supports NF4 and INT8 modes. NF4 packs two 4-bit values per byte.
/// On CUDA with the `cuda-kernels` feature, uses fused dequantize+GEMM
/// kernels that avoid materializing full-precision weight tensors.
/// Falls back to CPU dequantize-then-matmul when fused path is unavailable.
#[derive(Debug)]
pub struct BitsAndBytesLinear {
    /// NF4: packed u8 tensor (2 values per byte), shape [out_features * in_features / 2]
    /// INT8: i8 weights stored as U8, shape [out_features, in_features]
    quantized_weight: Tensor,
    /// Per-block absmax scales for dequantization
    absmax: Tensor,
    /// Optional bias
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    quant_type: BnbQuantType,
    block_size: usize,
    device: Device,
}

impl BitsAndBytesLinear {
    /// Create a new BitsAndBytes linear layer with placeholder weights.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        quant_type: BnbQuantType,
        block_size: usize,
        device: &Device,
    ) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }
        if block_size == 0 {
            candle_core::bail!("block_size must be non-zero");
        }

        let total_elements = out_features * in_features;
        let num_blocks = total_elements.div_ceil(block_size);

        let quantized_weight = match quant_type {
            BnbQuantType::NF4 => {
                let packed_len = total_elements.div_ceil(2);
                Tensor::zeros(packed_len, DType::U8, device)?
            }
            BnbQuantType::INT8 => Tensor::zeros((out_features, in_features), DType::U8, device)?,
        };

        let absmax = Tensor::zeros(num_blocks, DType::F32, device)?;

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F32, device)?)
        } else {
            None
        };

        Ok(Self {
            quantized_weight,
            absmax,
            bias,
            in_features,
            out_features,
            quant_type,
            block_size,
            device: device.clone(),
        })
    }

    /// Dequantize NF4 packed weights to f32.
    ///
    /// Each byte holds two 4-bit indices: low nibble first, high nibble second.
    /// Each index maps to a value in the NF4_LOOKUP table, then scaled by the
    /// per-block absmax factor.
    fn dequantize_nf4(&self) -> Result<Tensor> {
        let packed_data: Vec<u8> = self.quantized_weight.flatten_all()?.to_vec1()?;
        let absmax_data: Vec<f32> = self.absmax.flatten_all()?.to_vec1()?;

        let total_elements = self.out_features * self.in_features;
        let mut output = vec![0.0f32; total_elements];

        for (i, &packed_byte) in packed_data.iter().enumerate() {
            let low_idx = (packed_byte & 0x0F) as usize;
            let high_idx = ((packed_byte >> 4) & 0x0F) as usize;

            let elem_low = i * 2;
            let elem_high = i * 2 + 1;

            if elem_low < total_elements {
                let block_idx = elem_low / self.block_size;
                let scale = if block_idx < absmax_data.len() {
                    absmax_data[block_idx]
                } else {
                    1.0
                };
                output[elem_low] = NF4_LOOKUP[low_idx] * scale;
            }

            if elem_high < total_elements {
                let block_idx = elem_high / self.block_size;
                let scale = if block_idx < absmax_data.len() {
                    absmax_data[block_idx]
                } else {
                    1.0
                };
                output[elem_high] = NF4_LOOKUP[high_idx] * scale;
            }
        }

        Tensor::from_vec(output, (self.out_features, self.in_features), &self.device)
    }

    /// Dequantize INT8 weights to f32.
    ///
    /// Each i8 value is stored as u8 (reinterpreted). Scaled by per-block absmax
    /// divided by 127 (the max absolute value for signed int8).
    fn dequantize_int8(&self) -> Result<Tensor> {
        let int8_data: Vec<u8> = self.quantized_weight.flatten_all()?.to_vec1()?;
        let absmax_data: Vec<f32> = self.absmax.flatten_all()?.to_vec1()?;

        let total_elements = self.out_features * self.in_features;
        let mut output = vec![0.0f32; total_elements];

        for (i, &raw_byte) in int8_data.iter().enumerate() {
            if i >= total_elements {
                break;
            }
            // Reinterpret u8 as i8
            let val = raw_byte as i8;
            let block_idx = i / self.block_size;
            let scale = if block_idx < absmax_data.len() {
                absmax_data[block_idx] / 127.0
            } else {
                1.0 / 127.0
            };
            output[i] = val as f32 * scale;
        }

        Tensor::from_vec(output, (self.out_features, self.in_features), &self.device)
    }

    /// Dequantize weights to a full-precision tensor.
    pub fn dequantize(&self) -> Result<Tensor> {
        match self.quant_type {
            BnbQuantType::NF4 => self.dequantize_nf4(),
            BnbQuantType::INT8 => self.dequantize_int8(),
        }
    }

    /// Attempt fused dequantize+GEMM on CUDA via custom kernels.
    ///
    /// Returns the output tensor if the fused kernel launched successfully,
    /// or an error if the kernel is unavailable (caller should fall back to
    /// the dequantize-then-matmul path).
    #[cfg(feature = "cuda-kernels")]
    fn forward_fused(&self, x: &Tensor) -> Result<Tensor> {
        use super::bnb_cuda;

        match self.quant_type {
            BnbQuantType::NF4 => bnb_cuda::bnb_nf4_gemm(
                x,
                &self.quantized_weight,
                &self.absmax,
                self.bias.as_ref(),
                self.out_features,
                self.block_size as i32,
            ),
            BnbQuantType::INT8 => {
                let weight_2d = self
                    .quantized_weight
                    .reshape((self.out_features, self.in_features))?;
                bnb_cuda::bnb_int8_gemm(
                    x,
                    &weight_2d,
                    &self.absmax,
                    self.bias.as_ref(),
                    self.block_size as i32,
                )
            }
        }
    }
}

impl QuantizedLinear for BitsAndBytesLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = if x.dtype() != DType::F32 {
            x.to_dtype(DType::F32)?
        } else {
            x.clone()
        };

        // Reshape >2D inputs to 2D for matmul, then restore shape
        let original_shape = x_f32.dims().to_vec();
        let x_2d = if original_shape.len() > 2 {
            let batch: usize = original_shape[..original_shape.len() - 1].iter().product();
            let k = *original_shape.last().unwrap_or(&0);
            x_f32.reshape((batch, k))?
        } else {
            x_f32.clone()
        };

        // On CUDA, try the fused dequantize+GEMM kernel first
        #[cfg(feature = "cuda-kernels")]
        if x_2d.device().is_cuda() {
            match self.forward_fused(&x_2d) {
                Ok(output) => {
                    if original_shape.len() > 2 {
                        let mut out_shape = original_shape[..original_shape.len() - 1].to_vec();
                        out_shape.push(self.out_features);
                        return output.reshape(out_shape);
                    }
                    return Ok(output);
                }
                Err(_) => {
                    // Fused kernel failed (e.g. PTX not loaded); fall through
                    // to the dequantize-then-matmul path below.
                }
            }
        }

        // Fallback: dequantize to full precision, then matmul
        let weight = self.dequantize()?;
        let y = x_2d.matmul(&weight.t()?)?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };

        if original_shape.len() > 2 {
            let mut out_shape = original_shape[..original_shape.len() - 1].to_vec();
            out_shape.push(self.out_features);
            y.reshape(out_shape)
        } else {
            Ok(y)
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("weight") {
            self.quantized_weight = w.clone();
        }
        if let Some(a) = weights.get("absmax") {
            self.absmax = a.clone();
        }
        // INT8 uses SCB (Scale Column-wise/Block-wise) naming
        if let Some(s) = weights.get("SCB") {
            self.absmax = s.clone();
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::U8
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

// ─── Quantization helpers (for testing and offline quantization) ──────────────

/// Quantize a f32 tensor to NF4 packed format.
///
/// Returns (packed_bytes, absmax) where packed_bytes contains two 4-bit
/// NF4 indices per byte and absmax contains the per-block scale factors.
pub fn quantize_nf4(data: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let num_blocks = data.len().div_ceil(block_size);
    let mut absmax = vec![0.0f32; num_blocks];

    // Compute per-block absmax
    for (block_idx, chunk) in data.chunks(block_size).enumerate() {
        let max_val = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        absmax[block_idx] = max_val;
    }

    // Quantize: find nearest NF4 lookup index for each normalized value
    let mut indices = Vec::with_capacity(data.len());
    for (i, &val) in data.iter().enumerate() {
        let block_idx = i / block_size;
        let scale = absmax[block_idx];

        let normalized = if scale > 0.0 { val / scale } else { 0.0 };

        // Find nearest NF4 lookup value
        let mut best_idx = 0u8;
        let mut best_dist = f32::MAX;
        for (j, &lut_val) in NF4_LOOKUP.iter().enumerate() {
            let dist = (normalized - lut_val).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = j as u8;
            }
        }
        indices.push(best_idx);
    }

    // Pack two 4-bit values per byte (low nibble first)
    let packed_len = data.len().div_ceil(2);
    let mut packed = vec![0u8; packed_len];
    for (i, chunk) in indices.chunks(2).enumerate() {
        let low = chunk[0] & 0x0F;
        let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        packed[i] = low | (high << 4);
    }

    (packed, absmax)
}

/// Quantize a f32 tensor to INT8 format with per-block absmax scaling.
///
/// Returns (int8_as_u8, absmax) where the i8 values are stored as u8
/// via bit reinterpretation.
pub fn quantize_int8(data: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let num_blocks = data.len().div_ceil(block_size);
    let mut absmax = vec![0.0f32; num_blocks];

    // Compute per-block absmax
    for (block_idx, chunk) in data.chunks(block_size).enumerate() {
        let max_val = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        absmax[block_idx] = max_val;
    }

    // Quantize to int8 range [-127, 127]
    let mut quantized = Vec::with_capacity(data.len());
    for (i, &val) in data.iter().enumerate() {
        let block_idx = i / block_size;
        let scale = absmax[block_idx];

        let normalized = if scale > 0.0 {
            val / scale * 127.0
        } else {
            0.0
        };

        let clamped = normalized.round().clamp(-127.0, 127.0) as i8;
        // Store i8 as u8 via reinterpretation
        quantized.push(clamped as u8);
    }

    (quantized, absmax)
}

/// Unpack NF4 packed bytes into 4-bit indices.
///
/// Each byte yields two indices: low nibble first, high nibble second.
pub fn unpack_nf4(packed: &[u8], total_elements: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(total_elements);
    for &byte in packed {
        let low = byte & 0x0F;
        let high = (byte >> 4) & 0x0F;
        indices.push(low);
        if indices.len() < total_elements {
            indices.push(high);
        }
    }
    indices.truncate(total_elements);
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── NF4 Lookup Table Tests ──────────────────────────────────────────

    #[test]
    fn test_nf4_lookup_table_length() {
        assert_eq!(NF4_LOOKUP.len(), 16);
    }

    #[test]
    fn test_nf4_lookup_table_sorted() {
        for i in 0..NF4_LOOKUP.len() - 1 {
            assert!(
                NF4_LOOKUP[i] < NF4_LOOKUP[i + 1],
                "NF4 lookup table must be sorted: index {} ({}) >= index {} ({})",
                i,
                NF4_LOOKUP[i],
                i + 1,
                NF4_LOOKUP[i + 1]
            );
        }
    }

    #[test]
    fn test_nf4_lookup_table_range() {
        assert_eq!(NF4_LOOKUP[0], -1.0);
        assert_eq!(NF4_LOOKUP[15], 1.0);
        assert_eq!(NF4_LOOKUP[7], 0.0);
    }

    #[test]
    fn test_nf4_lookup_table_symmetry() {
        // NF4 is approximately symmetric around 0 but not exactly,
        // because it optimizes for normal distribution quantiles.
        // Verify the zero point is exactly at index 7.
        assert_eq!(NF4_LOOKUP[7], 0.0);
        // First 7 are negative, last 8 are non-negative
        for &v in &NF4_LOOKUP[..7] {
            assert!(v < 0.0, "expected negative value, got {v}");
        }
        for &v in &NF4_LOOKUP[8..] {
            assert!(v > 0.0, "expected positive value, got {v}");
        }
    }

    // ─── NF4 Quantize/Dequantize Roundtrip ──────────────────────────────

    #[test]
    fn test_nf4_quantize_dequantize_roundtrip() {
        let block_size = 64;
        // Generate test data in [-1, 1] range
        let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();

        let (packed, absmax) = quantize_nf4(&data, block_size);

        // Dequantize manually
        let indices = unpack_nf4(&packed, data.len());
        let mut recovered = Vec::with_capacity(data.len());
        for (i, &idx) in indices.iter().enumerate() {
            let block_idx = i / block_size;
            let scale = absmax[block_idx];
            recovered.push(NF4_LOOKUP[idx as usize] * scale);
        }

        // NF4 is lossy, but the error should be bounded
        for (i, (&original, &dequantized)) in data.iter().zip(recovered.iter()).enumerate() {
            let error = (original - dequantized).abs();
            // NF4 quantization error per element should be bounded by the
            // maximum distance between adjacent NF4 levels scaled by absmax
            let block_idx = i / block_size;
            let max_level_gap = 0.35; // approximate max gap between NF4 levels
            let max_error = max_level_gap * absmax[block_idx];
            assert!(
                error <= max_error + 1e-6,
                "NF4 roundtrip error too large at index {i}: original={original}, recovered={dequantized}, error={error}, max_allowed={max_error}"
            );
        }
    }

    #[test]
    fn test_nf4_quantize_zeros() {
        let data = vec![0.0f32; 64];
        let (packed, absmax) = quantize_nf4(&data, 64);

        // absmax of all-zero block is 0
        assert_eq!(absmax[0], 0.0);

        // All indices should map to 0.0 (index 7)
        let indices = unpack_nf4(&packed, 64);
        for &idx in &indices {
            assert_eq!(
                NF4_LOOKUP[idx as usize], 0.0,
                "zero input should quantize to zero NF4 value"
            );
        }
    }

    #[test]
    fn test_nf4_quantize_single_block() {
        let data = vec![0.5f32; 8];
        let block_size = 8;
        let (packed, absmax) = quantize_nf4(&data, block_size);

        assert_eq!(absmax.len(), 1);
        assert!((absmax[0] - 0.5).abs() < 1e-6);
        assert_eq!(packed.len(), 4);
    }

    // ─── INT8 Quantize/Dequantize Roundtrip ─────────────────────────────

    #[test]
    fn test_int8_quantize_dequantize_roundtrip() {
        let block_size = 64;
        let data: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();

        let (quantized, absmax) = quantize_int8(&data, block_size);

        // Dequantize
        let mut recovered = Vec::with_capacity(data.len());
        for (i, &raw) in quantized.iter().enumerate() {
            let val = raw as i8;
            let block_idx = i / block_size;
            let scale = absmax[block_idx] / 127.0;
            recovered.push(val as f32 * scale);
        }

        // INT8 has better precision than NF4
        for (i, (&original, &dequantized)) in data.iter().zip(recovered.iter()).enumerate() {
            let error = (original - dequantized).abs();
            let block_idx = i / block_size;
            // Max quantization error for int8 is absmax/127
            let max_error = absmax[block_idx] / 127.0 + 1e-6;
            assert!(
                error <= max_error,
                "INT8 roundtrip error too large at index {i}: original={original}, recovered={dequantized}, error={error}"
            );
        }
    }

    #[test]
    fn test_int8_quantize_zeros() {
        let data = vec![0.0f32; 64];
        let (quantized, absmax) = quantize_int8(&data, 64);

        assert_eq!(absmax[0], 0.0);
        for &v in &quantized {
            assert_eq!(v as i8, 0);
        }
    }

    #[test]
    fn test_int8_quantize_max_range() {
        let data = vec![1.0f32; 64];
        let (quantized, absmax) = quantize_int8(&data, 64);

        assert!((absmax[0] - 1.0).abs() < 1e-6);
        // All values should quantize to 127
        for &v in &quantized {
            assert_eq!(v as i8, 127);
        }
    }

    // ─── Weight Packing/Unpacking ────────────────────────────────────────

    #[test]
    fn test_nf4_pack_unpack_roundtrip() {
        let indices: Vec<u8> = (0..16).collect();
        // Pack indices
        let packed_len = indices.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        for (i, chunk) in indices.chunks(2).enumerate() {
            let low = chunk[0] & 0x0F;
            let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed[i] = low | (high << 4);
        }

        // Unpack
        let unpacked = unpack_nf4(&packed, indices.len());
        assert_eq!(unpacked, indices);
    }

    #[test]
    fn test_nf4_pack_unpack_odd_length() {
        let indices: Vec<u8> = vec![3, 7, 11, 15, 0];
        let packed_len = indices.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        for (i, chunk) in indices.chunks(2).enumerate() {
            let low = chunk[0] & 0x0F;
            let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed[i] = low | (high << 4);
        }

        let unpacked = unpack_nf4(&packed, indices.len());
        assert_eq!(unpacked, indices);
    }

    #[test]
    fn test_nf4_pack_boundary_values() {
        // Test with min (0) and max (15) indices
        let packed = vec![0xF0u8]; // low=0, high=15
        let unpacked = unpack_nf4(&packed, 2);
        assert_eq!(unpacked[0], 0);
        assert_eq!(unpacked[1], 15);

        let packed = vec![0x0Fu8]; // low=15, high=0
        let unpacked = unpack_nf4(&packed, 2);
        assert_eq!(unpacked[0], 15);
        assert_eq!(unpacked[1], 0);
    }

    // ─── BitsAndBytesConfig Tests ────────────────────────────────────────

    #[test]
    fn test_config_nf4_defaults() {
        let config = BitsAndBytesConfig::nf4();
        assert_eq!(config.quant_type, BnbQuantType::NF4);
        assert_eq!(config.block_size, 64);
        assert!(!config.double_quant);
        assert!(config.ignored_layers.is_empty());
        assert_eq!(config.method(), QuantizationMethod::BitsAndBytes);
    }

    #[test]
    fn test_config_int8_defaults() {
        let config = BitsAndBytesConfig::int8();
        assert_eq!(config.quant_type, BnbQuantType::INT8);
        assert_eq!(config.block_size, 64);
    }

    #[test]
    fn test_config_min_capability() {
        let config = BitsAndBytesConfig::default();
        assert_eq!(config.min_capability(), 70);
    }

    #[test]
    fn test_config_supported_dtypes() {
        let config = BitsAndBytesConfig::default();
        let dtypes = config.supported_act_dtypes();
        assert!(dtypes.contains(&DType::F32));
        assert!(dtypes.contains(&DType::F16));
        assert!(dtypes.contains(&DType::BF16));
    }

    #[test]
    fn test_config_builders() {
        let config = BitsAndBytesConfig::nf4()
            .with_block_size(128)
            .with_double_quant(true)
            .with_ignored_layers(vec!["lm_head".to_string()]);

        assert_eq!(config.block_size, 128);
        assert!(config.double_quant);
        assert_eq!(config.ignored_layers, vec!["lm_head"]);
    }

    #[test]
    fn test_config_from_detected_nf4() {
        let mut raw = HashMap::new();
        raw.insert("load_in_4bit".to_string(), serde_json::json!(true));
        raw.insert(
            "bnb_4bit_use_double_quant".to_string(),
            serde_json::json!(true),
        );
        raw.insert(
            "llm_int8_skip_modules".to_string(),
            serde_json::json!(["lm_head"]),
        );

        let config = BitsAndBytesConfig::from_detected(&raw);
        assert_eq!(config.quant_type, BnbQuantType::NF4);
        assert!(config.double_quant);
        assert_eq!(config.ignored_layers, vec!["lm_head"]);
    }

    #[test]
    fn test_config_from_detected_int8() {
        let mut raw = HashMap::new();
        raw.insert("load_in_8bit".to_string(), serde_json::json!(true));

        let config = BitsAndBytesConfig::from_detected(&raw);
        assert_eq!(config.quant_type, BnbQuantType::INT8);
    }

    #[test]
    fn test_config_layer_skipping() {
        let config = BitsAndBytesConfig::nf4()
            .with_ignored_layers(vec!["lm_head".to_string(), "embed_tokens".to_string()]);

        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(config.is_layer_skipped("model.embed_tokens.weight"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj.weight"));
    }

    // ─── BitsAndBytesLinear Tests ────────────────────────────────────────

    #[test]
    fn test_linear_creation_nf4() {
        let config = BitsAndBytesConfig::nf4();
        let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(linear.has_bias());
        assert_eq!(linear.weight_dtype(), DType::U8);
    }

    #[test]
    fn test_linear_creation_int8() {
        let config = BitsAndBytesConfig::int8();
        let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();

        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 128);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_linear_validation_zero_in_features() {
        let result = BitsAndBytesLinear::new(0, 128, false, BnbQuantType::NF4, 64, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("in_features"));
    }

    #[test]
    fn test_linear_validation_zero_out_features() {
        let result = BitsAndBytesLinear::new(64, 0, false, BnbQuantType::NF4, 64, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("out_features"));
    }

    #[test]
    fn test_linear_validation_zero_block_size() {
        let result = BitsAndBytesLinear::new(64, 128, false, BnbQuantType::NF4, 0, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("block_size"));
    }

    #[test]
    fn test_nf4_linear_forward_shape() {
        let in_features = 32;
        let out_features = 64;
        let block_size = 16;

        // Generate random weights and quantize them
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        let (packed, absmax_data) = quantize_nf4(&weight_data, block_size);

        let mut linear = BitsAndBytesLinear::new(
            in_features,
            out_features,
            false,
            BnbQuantType::NF4,
            block_size,
            &Device::Cpu,
        )
        .unwrap();

        // Load quantized weights
        let packed_len = packed.len();
        let absmax_len = absmax_data.len();
        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::from_vec(packed, (packed_len,), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "absmax".to_string(),
            Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        // Forward pass
        let batch_size = 4;
        let x = Tensor::ones(&[batch_size, in_features], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[batch_size, out_features]);
    }

    #[test]
    fn test_int8_linear_forward_shape() {
        let in_features = 32;
        let out_features = 64;
        let block_size = 32;

        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.01).cos() * 0.3)
            .collect();

        let (quantized, absmax_data) = quantize_int8(&weight_data, block_size);

        let mut linear = BitsAndBytesLinear::new(
            in_features,
            out_features,
            true,
            BnbQuantType::INT8,
            block_size,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::from_vec(quantized, (out_features, in_features), &Device::Cpu).unwrap(),
        );
        let absmax_len = absmax_data.len();
        weights.insert(
            "SCB".to_string(),
            Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).unwrap(),
        );
        let bias_data = vec![0.1f32; out_features];
        weights.insert(
            "bias".to_string(),
            Tensor::from_vec(bias_data, (out_features,), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let batch_size = 2;
        let x = Tensor::ones(&[batch_size, in_features], DType::F32, &Device::Cpu).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[batch_size, out_features]);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_nf4_linear_forward_correctness() {
        // Small matrix to verify computation
        let in_features = 4;
        let out_features = 2;
        let block_size = 8;

        // Use known weight values
        let weight_data = vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4, -0.6, 0.7];

        let (packed, absmax_data) = quantize_nf4(&weight_data, block_size);

        let mut linear = BitsAndBytesLinear::new(
            in_features,
            out_features,
            false,
            BnbQuantType::NF4,
            block_size,
            &Device::Cpu,
        )
        .unwrap();

        let packed_len = packed.len();
        let absmax_len = absmax_data.len();
        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::from_vec(packed, (packed_len,), &Device::Cpu).unwrap(),
        );
        weights.insert(
            "absmax".to_string(),
            Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        // Verify dequantization produces reasonable values
        let dequantized = linear.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[out_features, in_features]);

        let deq_data: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, deq) in weight_data.iter().zip(deq_data.iter()) {
            let error = (orig - deq).abs();
            // NF4 has limited precision: error bounded by ~0.35 * absmax
            assert!(
                error < 0.5,
                "Dequantized value {deq} too far from original {orig}"
            );
        }
    }

    #[test]
    fn test_int8_linear_forward_correctness() {
        let in_features = 4;
        let out_features = 2;
        let block_size = 8;

        let weight_data = vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4, -0.6, 0.7];

        let (quantized, absmax_data) = quantize_int8(&weight_data, block_size);

        let mut linear = BitsAndBytesLinear::new(
            in_features,
            out_features,
            false,
            BnbQuantType::INT8,
            block_size,
            &Device::Cpu,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "weight".to_string(),
            Tensor::from_vec(quantized, (out_features, in_features), &Device::Cpu).unwrap(),
        );
        let absmax_len = absmax_data.len();
        weights.insert(
            "absmax".to_string(),
            Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).unwrap(),
        );
        linear.load_weights(&weights).unwrap();

        let dequantized = linear.dequantize().unwrap();
        assert_eq!(dequantized.dims(), &[out_features, in_features]);

        let deq_data: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, deq) in weight_data.iter().zip(deq_data.iter()) {
            let error = (orig - deq).abs();
            // INT8 has better precision than NF4
            assert!(
                error < 0.02,
                "INT8 dequantized value {deq} too far from original {orig}, error={error}"
            );
        }
    }

    #[test]
    fn test_bnb_quant_type_display() {
        assert_eq!(BnbQuantType::NF4.to_string(), "nf4");
        assert_eq!(BnbQuantType::INT8.to_string(), "int8");
    }

    #[test]
    fn test_config_clone_box() {
        let config = BitsAndBytesConfig::nf4().with_block_size(128);
        let cloned = config.clone_box();
        assert_eq!(cloned.method(), QuantizationMethod::BitsAndBytes);
        assert_eq!(cloned.min_capability(), 70);
    }

    // ─── GPU Tests ──────────────────────────────────────────────────────

    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        /// Create a NF4 linear layer on the given device with known weights.
        fn create_nf4_linear(
            in_features: usize,
            out_features: usize,
            block_size: usize,
            has_bias: bool,
            device: &Device,
        ) -> BitsAndBytesLinear {
            let weight_data: Vec<f32> = (0..in_features * out_features)
                .map(|i| (i as f32 * 0.037).sin() * 0.6)
                .collect();
            let (packed, absmax_data) = quantize_nf4(&weight_data, block_size);

            let mut linear = BitsAndBytesLinear::new(
                in_features,
                out_features,
                has_bias,
                BnbQuantType::NF4,
                block_size,
                device,
            )
            .expect("failed to create NF4 linear");

            let packed_len = packed.len();
            let absmax_len = absmax_data.len();
            let mut weights = HashMap::new();
            weights.insert(
                "weight".to_string(),
                Tensor::from_vec(packed, (packed_len,), device).expect("packed tensor"),
            );
            weights.insert(
                "absmax".to_string(),
                Tensor::from_vec(absmax_data, (absmax_len,), device).expect("absmax tensor"),
            );
            if has_bias {
                let bias_data: Vec<f32> =
                    (0..out_features).map(|i| (i as f32 * 0.01) - 0.5).collect();
                weights.insert(
                    "bias".to_string(),
                    Tensor::from_vec(bias_data, (out_features,), device).expect("bias tensor"),
                );
            }
            linear.load_weights(&weights).expect("load weights");
            linear
        }

        /// Create an INT8 linear layer on the given device with known weights.
        fn create_int8_linear(
            in_features: usize,
            out_features: usize,
            block_size: usize,
            device: &Device,
        ) -> BitsAndBytesLinear {
            let weight_data: Vec<f32> = (0..in_features * out_features)
                .map(|i| (i as f32 * 0.023).cos() * 0.4)
                .collect();
            let (quantized, absmax_data) = quantize_int8(&weight_data, block_size);

            let mut linear = BitsAndBytesLinear::new(
                in_features,
                out_features,
                false,
                BnbQuantType::INT8,
                block_size,
                device,
            )
            .expect("failed to create INT8 linear");

            let absmax_len = absmax_data.len();
            let mut weights = HashMap::new();
            weights.insert(
                "weight".to_string(),
                Tensor::from_vec(quantized, (out_features, in_features), device)
                    .expect("weight tensor"),
            );
            weights.insert(
                "absmax".to_string(),
                Tensor::from_vec(absmax_data, (absmax_len,), device).expect("absmax tensor"),
            );
            linear.load_weights(&weights).expect("load weights");
            linear
        }

        #[test]
        fn test_nf4_fused_vs_unfused_correctness() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            };

            let in_f = 64;
            let out_f = 32;
            let block_size = 64;

            let linear = create_nf4_linear(in_f, out_f, block_size, false, &device);

            let x = Tensor::ones(&[4, in_f], DType::F32, &device).expect("input tensor");

            // Use forward() which will attempt fused path on CUDA
            let result = linear.forward(&x).expect("forward pass");
            assert_eq!(result.dims(), &[4, out_f]);

            // Compare with explicit dequantize-then-matmul
            let weight = linear.dequantize().expect("dequantize");
            let expected = x.matmul(&weight.t().expect("transpose")).expect("matmul");

            let result_data: Vec<f32> = result
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to_vec");
            let expected_data: Vec<f32> = expected
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to_vec");

            for (i, (r, e)) in result_data.iter().zip(expected_data.iter()).enumerate() {
                let err = (r - e).abs();
                assert!(
                    err < 0.01,
                    "NF4 fused vs unfused mismatch at {i}: fused={r}, expected={e}, err={err}"
                );
            }
        }

        #[test]
        fn test_int8_fused_vs_unfused_correctness() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            };

            let in_f = 64;
            let out_f = 32;
            let block_size = 64;

            let linear = create_int8_linear(in_f, out_f, block_size, &device);

            let x = Tensor::ones(&[4, in_f], DType::F32, &device).expect("input tensor");

            let result = linear.forward(&x).expect("forward pass");
            assert_eq!(result.dims(), &[4, out_f]);

            let weight = linear.dequantize().expect("dequantize");
            let expected = x.matmul(&weight.t().expect("transpose")).expect("matmul");

            let result_data: Vec<f32> = result
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to_vec");
            let expected_data: Vec<f32> = expected
                .flatten_all()
                .expect("flatten")
                .to_vec1()
                .expect("to_vec");

            for (i, (r, e)) in result_data.iter().zip(expected_data.iter()).enumerate() {
                let err = (r - e).abs();
                assert!(
                    err < 0.01,
                    "INT8 fused vs unfused mismatch at {i}: fused={r}, expected={e}, err={err}"
                );
            }
        }

        #[test]
        fn test_nf4_fused_with_bias() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            };

            let in_f = 32;
            let out_f = 16;
            let block_size = 32;

            let linear = create_nf4_linear(in_f, out_f, block_size, true, &device);

            let x = Tensor::ones(&[2, in_f], DType::F32, &device).expect("input tensor");
            let result = linear.forward(&x).expect("forward pass");
            assert_eq!(result.dims(), &[2, out_f]);
        }

        #[test]
        fn test_nf4_fused_various_sizes() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping GPU test: no CUDA device");
                return;
            };

            let sizes: Vec<(usize, usize, usize, usize)> = vec![
                (1, 32, 16, 32),   // single token, small
                (8, 64, 32, 64),   // batch, medium
                (16, 128, 64, 64), // larger batch
                (4, 256, 128, 64), // wider
            ];

            for (batch, in_f, out_f, block_size) in sizes {
                let linear = create_nf4_linear(in_f, out_f, block_size, false, &device);
                let x = Tensor::ones(&[batch, in_f], DType::F32, &device).expect("input tensor");
                let result = linear.forward(&x).expect("forward pass");
                assert_eq!(
                    result.dims(),
                    &[batch, out_f],
                    "Shape mismatch for ({batch}, {in_f}, {out_f})"
                );
            }
        }
    }
}
