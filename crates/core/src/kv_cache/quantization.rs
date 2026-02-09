//! KV Cache Quantization for memory-efficient inference.
//!
//! Reduces KV cache memory by 2x using FP8 or INT8 quantization,
//! enabling longer context windows on memory-limited GPUs.
//!
//! Quantization strategies:
//! - **FP8 E4M3** (Hopper+): 1 sign, 4 exponent, 3 mantissa bits. Range ~1e-9 to 448.0
//! - **INT8** (Ampere+): Symmetric quantization with per-tensor scale
//!
//! Per-tensor scaling is used for simplicity and sufficient accuracy.

use candle_core::{DType, Device, Result, Tensor};

/// KV cache storage data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVCacheDtype {
    /// Automatic: use model's compute dtype (BF16/F16), no quantization
    #[default]
    Auto,
    /// FP8 E4M3 format: 2x compression, requires Hopper+ (sm_89)
    Fp8E4m3,
    /// INT8 symmetric: 2x compression, wider hardware support (Ampere+)
    Int8,
}

impl KVCacheDtype {
    /// Returns the storage element size in bytes.
    pub fn element_size(&self, compute_dtype: DType) -> usize {
        match self {
            KVCacheDtype::Auto => compute_dtype.size_in_bytes(),
            KVCacheDtype::Fp8E4m3 | KVCacheDtype::Int8 => 1,
        }
    }

    /// Returns the storage DType for cache tensors.
    pub fn storage_dtype(&self, compute_dtype: DType) -> DType {
        match self {
            KVCacheDtype::Auto => compute_dtype,
            KVCacheDtype::Fp8E4m3 | KVCacheDtype::Int8 => DType::U8,
        }
    }

    /// Returns true if this dtype requires quantization/dequantization.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, KVCacheDtype::Auto)
    }

    /// Minimum CUDA compute capability required.
    pub fn min_capability(&self) -> u32 {
        match self {
            KVCacheDtype::Auto => 70,    // Volta for BF16
            KVCacheDtype::Fp8E4m3 => 89, // Hopper (H100)
            KVCacheDtype::Int8 => 80,    // Ampere (A100)
        }
    }
}

/// Scales for quantized KV cache.
///
/// Per-tensor scales are used for simplicity and sufficient accuracy.
/// K and V scales are separate as they may have different value distributions.
#[derive(Debug, Clone)]
pub struct KVScales {
    /// Scale for K cache: `k_fp8 = k / k_scale`
    pub k_scale: Tensor,
    /// Scale for V cache: `v_fp8 = v / v_scale`
    pub v_scale: Tensor,
}

impl KVScales {
    /// Create new scales initialized to 1.0 (identity scaling).
    pub fn new(device: &Device) -> Result<Self> {
        let k_scale = Tensor::ones(1, DType::F32, device)?;
        let v_scale = Tensor::ones(1, DType::F32, device)?;
        Ok(Self { k_scale, v_scale })
    }

    /// Create scales from explicit values.
    pub fn from_values(k_scale: f32, v_scale: f32, device: &Device) -> Result<Self> {
        let k_scale = Tensor::from_slice(&[k_scale], 1, device)?;
        let v_scale = Tensor::from_slice(&[v_scale], 1, device)?;
        Ok(Self { k_scale, v_scale })
    }

    /// Update K scale based on observed data.
    pub fn calibrate_k(&mut self, k: &Tensor) -> Result<()> {
        self.k_scale = compute_scale(k)?;
        Ok(())
    }

    /// Update V scale based on observed data.
    pub fn calibrate_v(&mut self, v: &Tensor) -> Result<()> {
        self.v_scale = compute_scale(v)?;
        Ok(())
    }

    /// Update both scales based on observed data.
    pub fn calibrate(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.calibrate_k(k)?;
        self.calibrate_v(v)?;
        Ok(())
    }

    /// Reset scales back to identity (1.0).
    ///
    /// Called during cache reset to ensure stale calibration data
    /// from previous requests doesn't affect new requests.
    pub fn reset(&mut self) -> Result<()> {
        let device = self.k_scale.device().clone();
        self.k_scale = Tensor::ones(1, DType::F32, &device)?;
        self.v_scale = Tensor::ones(1, DType::F32, &device)?;
        Ok(())
    }
}

// ============================================================================
// FP8 E4M3 Constants
// ============================================================================

/// Maximum representable value in FP8 E4M3 format.
const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to avoid division by zero.
const SCALE_MIN: f32 = 1e-12;

// ============================================================================
// Quantization Functions
// ============================================================================

/// Compute per-tensor scale for quantization.
///
/// scale = max(abs(tensor)) / FP8_MAX
fn compute_scale(tensor: &Tensor) -> Result<Tensor> {
    let abs_tensor = tensor.abs()?.to_dtype(DType::F32)?;
    let max_val = abs_tensor.flatten_all()?.max(0)?;
    let scale = (max_val / FP8_E4M3_MAX as f64)?.maximum(SCALE_MIN as f64)?;
    Ok(scale)
}

/// Quantize a tensor to FP8 E4M3 format.
///
/// # Arguments
/// * `tensor` - Input tensor (BF16/F16/F32)
/// * `scale` - Per-tensor scale (F32, shape [1])
///
/// # Returns
/// Quantized tensor in U8 format
pub fn quantize_fp8(tensor: &Tensor, scale: &Tensor) -> Result<Tensor> {
    // Convert to F32 for computation
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let scale_f32 = scale.to_dtype(DType::F32)?;

    // Scale and clamp: x_scaled = clamp(x / scale, -FP8_MAX, FP8_MAX)
    let scaled = tensor_f32.broadcast_div(&scale_f32)?;
    let clamped = scaled.clamp(-FP8_E4M3_MAX as f64, FP8_E4M3_MAX as f64)?;

    // Encode to FP8 E4M3 (stored as U8)
    // For CPU: use software encoding; for CUDA: use hardware
    if tensor.device().is_cuda() {
        encode_fp8_cuda(&clamped)
    } else {
        encode_fp8_cpu(&clamped)
    }
}

/// Dequantize FP8 E4M3 tensor back to compute dtype.
///
/// # Arguments
/// * `tensor` - Quantized tensor (U8)
/// * `scale` - Per-tensor scale (F32)
/// * `target_dtype` - Target dtype (BF16/F16)
///
/// # Returns
/// Dequantized tensor
pub fn dequantize_fp8(tensor: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    if tensor.device().is_cuda() {
        decode_fp8_cuda(tensor, scale, target_dtype)
    } else {
        decode_fp8_cpu(tensor, scale, target_dtype)
    }
}

/// Quantize a tensor to INT8 symmetric format.
///
/// # Arguments
/// * `tensor` - Input tensor (BF16/F16/F32)
/// * `scale` - Per-tensor scale (F32, shape [1])
///
/// # Returns
/// Quantized tensor in U8 format (shifted to unsigned: x + 128)
pub fn quantize_int8(tensor: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let scale_f32 = scale.to_dtype(DType::F32)?;

    // Scale and clamp to INT8 range
    let scaled = tensor_f32.broadcast_div(&scale_f32)?;
    let clamped = scaled.clamp(-127.0, 127.0)?;

    // Round and shift to unsigned (0-255)
    let rounded = clamped.round()?;
    let shifted = (rounded + 128.0)?;

    shifted.to_dtype(DType::U8)
}

/// Dequantize INT8 tensor back to compute dtype.
///
/// # Arguments
/// * `tensor` - Quantized tensor (U8)
/// * `scale` - Per-tensor scale (F32)
/// * `target_dtype` - Target dtype (BF16/F16)
///
/// # Returns
/// Dequantized tensor
pub fn dequantize_int8(tensor: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    // Convert back to signed range
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let shifted = (tensor_f32 - 128.0)?;

    // Apply scale
    let scale_f32 = scale.to_dtype(DType::F32)?;
    let dequantized = shifted.broadcast_mul(&scale_f32)?;

    dequantized.to_dtype(target_dtype)
}

/// Compute INT8 scale for a tensor.
pub fn compute_int8_scale(tensor: &Tensor) -> Result<Tensor> {
    let abs_tensor = tensor.abs()?.to_dtype(DType::F32)?;
    let max_val = abs_tensor.flatten_all()?.max(0)?;
    let scale = (max_val / 127.0)?.maximum(SCALE_MIN as f64)?;
    Ok(scale)
}

// ============================================================================
// CPU FP8 Encoding/Decoding
// ============================================================================

/// CPU implementation of FP8 E4M3 encoding.
fn encode_fp8_cpu(tensor: &Tensor) -> Result<Tensor> {
    let shape = tensor.dims();
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let encoded: Vec<u8> = data.iter().map(|&v| fp8_e4m3_encode(v)).collect();

    Tensor::from_vec(encoded, shape, tensor.device())
}

/// CPU implementation of FP8 E4M3 decoding.
fn decode_fp8_cpu(tensor: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let shape = tensor.dims();
    let data: Vec<u8> = tensor.flatten_all()?.to_vec1()?;
    let scale_val: f32 = scale
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];

    let decoded: Vec<f32> = data
        .iter()
        .map(|&b| fp8_e4m3_decode(b) * scale_val)
        .collect();

    let result = Tensor::from_vec(decoded, shape, tensor.device())?;
    result.to_dtype(target_dtype)
}

/// Encode a float value to FP8 E4M3 format.
#[inline]
fn fp8_e4m3_encode(val: f32) -> u8 {
    if val == 0.0 {
        return 0;
    }

    let sign = if val < 0.0 { 0x80u8 } else { 0u8 };
    let abs_val = val.abs().min(FP8_E4M3_MAX);

    if abs_val < 1.0 / 512.0 {
        // Underflow to zero
        return sign;
    }

    let bits = abs_val.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_mant = bits & 0x7FFFFF;

    // FP8 E4M3 bias is 7
    let fp8_exp = (fp32_exp + 7).clamp(0, 15) as u8;
    let fp8_mant = (fp32_mant >> 20) as u8; // Top 3 bits of mantissa

    sign | (fp8_exp << 3) | fp8_mant
}

/// Decode FP8 E4M3 byte to float.
#[inline]
fn fp8_e4m3_decode(byte: u8) -> f32 {
    if byte == 0 || byte == 0x80 {
        return 0.0;
    }

    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as u32;

    // Convert to float
    // FP8 E4M3: bias = 7, so actual exponent = exp - 7
    // For normalized numbers: value = 2^(exp-7) * (1 + mant/8)
    if exp == 0 {
        // Subnormal
        sign * (mant as f32 / 8.0) * 2.0f32.powi(-6)
    } else {
        sign * 2.0f32.powi(exp - 7) * (1.0 + mant as f32 / 8.0)
    }
}

// ============================================================================
// CUDA FP8 Encoding/Decoding (use existing kernels when available)
// ============================================================================

#[cfg(feature = "cuda-kernels")]
fn encode_fp8_cuda(tensor: &Tensor) -> Result<Tensor> {
    use crate::quantization::fp8_cuda;

    // Reshape to 2D for the kernel
    let original_shape = tensor.dims();
    let total_elements = tensor.elem_count();
    let tensor_2d = tensor.reshape((1, total_elements))?;
    let tensor_bf16 = tensor_2d.to_dtype(DType::BF16)?;

    // Use identity scale (data already scaled)
    let scale = Tensor::ones(1, DType::F32, tensor.device())?;
    let quantized = fp8_cuda::fp8_quantize_static(&tensor_bf16, &scale)?;

    quantized.reshape(original_shape)
}

#[cfg(not(feature = "cuda-kernels"))]
fn encode_fp8_cuda(tensor: &Tensor) -> Result<Tensor> {
    // Fallback to CPU implementation for CUDA without kernels
    encode_fp8_cpu(tensor)
}

#[cfg(feature = "cuda-kernels")]
fn decode_fp8_cuda(tensor: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    use crate::quantization::fp8_cuda;

    let original_shape = tensor.dims();
    let total_elements = tensor.elem_count();
    let tensor_2d = tensor.reshape((1, total_elements))?;

    let dequantized = fp8_cuda::fp8_dequantize(&tensor_2d, scale)?;
    let reshaped = dequantized.reshape(original_shape)?;

    reshaped.to_dtype(target_dtype)
}

#[cfg(not(feature = "cuda-kernels"))]
fn decode_fp8_cuda(tensor: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    // Fallback to CPU implementation
    decode_fp8_cpu(tensor, scale, target_dtype)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_dtype_element_size() {
        assert_eq!(KVCacheDtype::Auto.element_size(DType::BF16), 2);
        assert_eq!(KVCacheDtype::Auto.element_size(DType::F16), 2);
        assert_eq!(KVCacheDtype::Fp8E4m3.element_size(DType::BF16), 1);
        assert_eq!(KVCacheDtype::Int8.element_size(DType::BF16), 1);
    }

    #[test]
    fn test_kv_cache_dtype_storage_dtype() {
        assert_eq!(KVCacheDtype::Auto.storage_dtype(DType::BF16), DType::BF16);
        assert_eq!(KVCacheDtype::Fp8E4m3.storage_dtype(DType::BF16), DType::U8);
        assert_eq!(KVCacheDtype::Int8.storage_dtype(DType::BF16), DType::U8);
    }

    #[test]
    fn test_kv_cache_dtype_is_quantized() {
        assert!(!KVCacheDtype::Auto.is_quantized());
        assert!(KVCacheDtype::Fp8E4m3.is_quantized());
        assert!(KVCacheDtype::Int8.is_quantized());
    }

    #[test]
    fn test_kv_scales_new() {
        let scales = KVScales::new(&Device::Cpu).unwrap();
        let k_val: Vec<f32> = scales.k_scale.to_vec1().unwrap();
        let v_val: Vec<f32> = scales.v_scale.to_vec1().unwrap();
        assert!((k_val[0] - 1.0).abs() < 1e-6);
        assert!((v_val[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_scales_from_values() {
        let scales = KVScales::from_values(0.5, 0.25, &Device::Cpu).unwrap();
        let k_val: Vec<f32> = scales.k_scale.to_vec1().unwrap();
        let v_val: Vec<f32> = scales.v_scale.to_vec1().unwrap();
        assert!((k_val[0] - 0.5).abs() < 1e-6);
        assert!((v_val[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_fp8_encode_decode_zero() {
        assert_eq!(fp8_e4m3_encode(0.0), 0);
        assert_eq!(fp8_e4m3_decode(0), 0.0);
    }

    #[test]
    fn test_fp8_encode_decode_one() {
        let encoded = fp8_e4m3_encode(1.0);
        let decoded = fp8_e4m3_decode(encoded);
        assert!(
            (decoded - 1.0).abs() < 0.1,
            "decoded={}, expected=1.0",
            decoded
        );
    }

    #[test]
    fn test_fp8_encode_decode_negative() {
        let encoded = fp8_e4m3_encode(-2.0);
        let decoded = fp8_e4m3_decode(encoded);
        assert!(
            (decoded - (-2.0)).abs() < 0.2,
            "decoded={}, expected=-2.0",
            decoded
        );
    }

    #[test]
    fn test_fp8_clamps_to_max() {
        let encoded = fp8_e4m3_encode(1000.0);
        let decoded = fp8_e4m3_decode(encoded);
        assert!(
            decoded <= FP8_E4M3_MAX,
            "decoded={} should be <= {}",
            decoded,
            FP8_E4M3_MAX
        );
    }

    #[test]
    fn test_compute_scale() {
        let tensor = Tensor::from_slice(&[1.0f32, -4.0, 2.0, -1.0], (2, 2), &Device::Cpu).unwrap();
        let scale = compute_scale(&tensor).unwrap();
        let scale_val: f32 = scale.to_scalar().unwrap();
        // max(abs) = 4.0, scale = 4.0 / 448.0 ≈ 0.00893
        let expected = 4.0 / FP8_E4M3_MAX;
        assert!(
            (scale_val - expected).abs() < 1e-6,
            "scale={}, expected={}",
            scale_val,
            expected
        );
    }

    #[test]
    fn test_quantize_dequantize_fp8_roundtrip() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.5, -3.0, 100.0];
        let tensor = Tensor::from_vec(data.clone(), (2, 3), &Device::Cpu).unwrap();

        let scale = compute_scale(&tensor).unwrap();
        let quantized = quantize_fp8(&tensor, &scale).unwrap();
        let dequantized = dequantize_fp8(&quantized, &scale, DType::F32).unwrap();

        let result: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();

        // FP8 has limited precision, allow some error
        for (orig, result) in data.iter().zip(result.iter()) {
            let abs_error = (orig - result).abs();
            let rel_error = if orig.abs() > 1e-6 {
                abs_error / orig.abs()
            } else {
                abs_error
            };
            assert!(
                rel_error < 0.15 || abs_error < 0.5,
                "orig={}, result={}, rel_error={}",
                orig,
                result,
                rel_error
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_int8_roundtrip() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.5, -3.0, 10.0];
        let tensor = Tensor::from_vec(data.clone(), (2, 3), &Device::Cpu).unwrap();

        let scale = compute_int8_scale(&tensor).unwrap();
        let quantized = quantize_int8(&tensor, &scale).unwrap();
        let dequantized = dequantize_int8(&quantized, &scale, DType::F32).unwrap();

        let result: Vec<f32> = dequantized.flatten_all().unwrap().to_vec1().unwrap();

        // INT8 also has limited precision
        for (orig, result) in data.iter().zip(result.iter()) {
            let abs_error = (orig - result).abs();
            let rel_error = if orig.abs() > 1e-6 {
                abs_error / orig.abs()
            } else {
                abs_error
            };
            assert!(
                rel_error < 0.1 || abs_error < 0.2,
                "orig={}, result={}, rel_error={}",
                orig,
                result,
                rel_error
            );
        }
    }

    #[test]
    fn test_kv_scales_calibrate() {
        let k = Tensor::from_slice(&[1.0f32, 4.0, -2.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let v = Tensor::from_slice(&[0.5f32, 1.0, -0.5, 2.0], (2, 2), &Device::Cpu).unwrap();

        let mut scales = KVScales::new(&Device::Cpu).unwrap();
        scales.calibrate(&k, &v).unwrap();

        let k_scale: f32 = scales.k_scale.to_scalar().unwrap();
        let v_scale: f32 = scales.v_scale.to_scalar().unwrap();

        // K max = 4.0, V max = 2.0
        assert!((k_scale - 4.0 / FP8_E4M3_MAX).abs() < 1e-6);
        assert!((v_scale - 2.0 / FP8_E4M3_MAX).abs() < 1e-6);
    }

    #[test]
    fn test_min_capability() {
        assert_eq!(KVCacheDtype::Auto.min_capability(), 70);
        assert_eq!(KVCacheDtype::Fp8E4m3.min_capability(), 89);
        assert_eq!(KVCacheDtype::Int8.min_capability(), 80);
    }
}
