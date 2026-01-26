//! FP8 CUDA kernel bindings for quantization operations.
//!
//! Provides GPU-accelerated FP8 quantization for Ada Lovelace+ GPUs (sm_89+).

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Storage, Tensor,
};

const FP8_QUANT_PTX: &str = include_str!("../../kernels/fp8_quant.ptx");
const FP8_GEMM_PTX: &str = include_str!("../../kernels/fp8_gemm.ptx");
const BLOCK_SIZE: u32 = 256;

/// Quantization mode for FP8 conversion.
#[derive(Debug, Clone, Copy)]
pub enum Fp8QuantMode {
    /// Static quantization with pre-computed scale
    Static,
    /// Dynamic per-tensor quantization
    DynamicTensor,
    /// Dynamic per-token quantization
    DynamicPerToken,
}

/// FP8 static quantization operation.
struct Fp8StaticQuantOp {
    scale: Tensor,
    hidden_size: usize,
    per_token_scale: bool,
}

impl CustomOp1 for Fp8StaticQuantOp {
    fn name(&self) -> &'static str {
        "fp8_static_quant"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fp8_static_quant requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let dims = layout.dims();
        let num_tokens = dims[0];

        if layout.start_offset() != 0 {
            candle_core::bail!("fp8_static_quant: input must be contiguous from offset 0");
        }

        let in_slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("fp8_static_quant expects bf16 input"),
        };

        let (scale_guard, _) = self.scale.storage_and_layout();
        let scale_slice = match &*scale_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("scale must be f32"),
            },
            _ => candle_core::bail!("scale must be on CUDA"),
        };

        // Allocate output as U8 (FP8 stored as bytes)
        let elem_count = num_tokens * self.hidden_size;
        let output_slice = dev.alloc_zeros::<u8>(elem_count)?;

        let func =
            dev.get_or_load_custom_func("fp8_static_quant_bf16", "fp8_quant", FP8_QUANT_PTX)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = self.hidden_size as i32;
        let per_token_i32: i32 = if self.per_token_scale { 1 } else { 0 };

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(in_slice);
        builder.arg(scale_slice);
        builder.arg(&hidden_size_i32);
        builder.arg(&per_token_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fp8_static_quant launch: {e}")))?;

        drop(scale_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::U8(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(dims);
        Ok((output_storage, output_shape))
    }
}

/// FP8 dynamic per-token quantization - Step 1: compute per-token scales.
/// Returns scales tensor [num_tokens] in F32.
struct Fp8ComputeScalesOp {
    hidden_size: usize,
}

impl CustomOp1 for Fp8ComputeScalesOp {
    fn name(&self) -> &'static str {
        "fp8_compute_scales"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // CPU fallback: compute max absolute value per token
        let dims = layout.dims();
        let num_tokens = dims[0];

        let data = match storage {
            CpuStorage::BF16(data) => data,
            _ => candle_core::bail!("fp8_compute_scales expects bf16 input"),
        };

        const FP8_E4M3_MAX: f32 = 448.0;
        let mut scales = Vec::with_capacity(num_tokens);

        for token_idx in 0..num_tokens {
            let offset = layout.start_offset() + token_idx * self.hidden_size;
            let mut max_abs: f32 = 0.0;
            for i in 0..self.hidden_size {
                let val = data[offset + i].to_f32().abs();
                if val > max_abs {
                    max_abs = val;
                }
            }
            // Scale = max_abs / FP8_MAX, with minimum to avoid division by zero
            let scale = (max_abs / FP8_E4M3_MAX).max(1e-12);
            scales.push(scale);
        }

        let output_shape = Shape::from_dims(&[num_tokens]);
        Ok((CpuStorage::F32(scales), output_shape))
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let dims = layout.dims();
        let num_tokens = dims[0];

        if layout.start_offset() != 0 {
            candle_core::bail!("fp8_compute_scales: input must be contiguous from offset 0");
        }

        let in_slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("fp8_compute_scales expects bf16 input"),
        };

        // Allocate scales output
        let scales_slice = dev.alloc_zeros::<f32>(num_tokens)?;

        let func =
            dev.get_or_load_custom_func("fp8_compute_scales_bf16", "fp8_quant", FP8_QUANT_PTX)?;

        // Shared memory for block reduction
        let num_warps = BLOCK_SIZE / 32;
        let shared_mem_bytes = num_warps * 4;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes,
        };

        let hidden_size_i32 = self.hidden_size as i32;

        let mut builder = func.builder();
        builder.arg(&scales_slice);
        builder.arg(in_slice);
        builder.arg(&hidden_size_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fp8_compute_scales launch: {e}")))?;

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::F32(scales_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_tokens]);
        Ok((output_storage, output_shape))
    }
}

/// FP8 dynamic per-token quantization - Step 2: quantize with pre-computed scales.
/// This op takes pre-computed scales and applies them during quantization.
struct Fp8QuantWithScalesOp {
    scale: Tensor,
    hidden_size: usize,
}

impl CustomOp1 for Fp8QuantWithScalesOp {
    fn name(&self) -> &'static str {
        "fp8_quant_with_scales"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let dims = layout.dims();
        let num_tokens = dims[0];

        let data = match storage {
            CpuStorage::BF16(data) => data,
            _ => candle_core::bail!("fp8_quant_with_scales expects bf16 input"),
        };

        let (scale_guard, scale_layout) = self.scale.storage_and_layout();
        let scales = match &*scale_guard {
            Storage::Cpu(CpuStorage::F32(s)) => s,
            _ => candle_core::bail!("scales must be F32 on CPU"),
        };

        const FP8_E4M3_MAX: f32 = 448.0;
        let mut output = Vec::with_capacity(num_tokens * self.hidden_size);

        for token_idx in 0..num_tokens {
            let offset = layout.start_offset() + token_idx * self.hidden_size;
            let scale_offset = scale_layout.start_offset() + token_idx;
            let scale = scales[scale_offset];
            let inv_scale = 1.0 / scale;

            for i in 0..self.hidden_size {
                let val = data[offset + i].to_f32();
                // Quantize: clamp(val / scale, -FP8_MAX, FP8_MAX) and convert to FP8
                let scaled = (val * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX);
                // Simple FP8 E4M3 encoding (approximate for CPU fallback)
                let fp8_byte = fp8_e4m3_encode(scaled);
                output.push(fp8_byte);
            }
        }

        let output_shape = Shape::from_dims(dims);
        Ok((CpuStorage::U8(output), output_shape))
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let dims = layout.dims();
        let num_tokens = dims[0];

        if layout.start_offset() != 0 {
            candle_core::bail!("fp8_quant_with_scales: input must be contiguous from offset 0");
        }

        let in_slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("fp8_quant_with_scales expects bf16 input"),
        };

        let (scale_guard, _) = self.scale.storage_and_layout();
        let scale_slice = match &*scale_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("scale must be f32"),
            },
            _ => candle_core::bail!("scale must be on CUDA"),
        };

        let elem_count = num_tokens * self.hidden_size;
        let output_slice = dev.alloc_zeros::<u8>(elem_count)?;

        // Use the static quant kernel with per-token scale
        let func =
            dev.get_or_load_custom_func("fp8_static_quant_bf16", "fp8_quant", FP8_QUANT_PTX)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = self.hidden_size as i32;
        let per_token_i32: i32 = 1; // Always per-token for dynamic quant

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(in_slice);
        builder.arg(scale_slice);
        builder.arg(&hidden_size_i32);
        builder.arg(&per_token_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fp8_quant_with_scales launch: {e}")))?;

        drop(scale_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::U8(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(dims);
        Ok((output_storage, output_shape))
    }
}

/// Encode a float value to FP8 E4M3 format (CPU fallback).
#[inline]
fn fp8_e4m3_encode(val: f32) -> u8 {
    // Simplified FP8 E4M3 encoding for CPU fallback
    // E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits
    // Max value: 448.0, Min subnormal: 2^-9
    if val == 0.0 {
        return 0;
    }

    let sign = if val < 0.0 { 0x80u8 } else { 0u8 };
    let abs_val = val.abs();

    // Clamp to FP8 range
    let clamped = abs_val.min(448.0);

    if clamped < 1.0 / 512.0 {
        // Underflow to zero
        return sign;
    }

    // Convert to FP8 representation
    let bits = clamped.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_mant = bits & 0x7FFFFF;

    // FP8 E4M3 bias is 7
    let fp8_exp = (fp32_exp + 7).clamp(0, 15) as u8;
    let fp8_mant = (fp32_mant >> 20) as u8; // Take top 3 bits of mantissa

    sign | (fp8_exp << 3) | fp8_mant
}

/// FP8 dequantization operation.
struct Fp8DequantOp {
    scale: Tensor,
    hidden_size: usize,
    per_token_scale: bool,
}

impl CustomOp1 for Fp8DequantOp {
    fn name(&self) -> &'static str {
        "fp8_dequant"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fp8_dequant requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &storage.device;
        let dims = layout.dims();
        let num_tokens = dims[0];

        if layout.start_offset() != 0 {
            candle_core::bail!("fp8_dequant: input must be contiguous from offset 0");
        }

        let in_slice = match &storage.slice {
            CudaStorageSlice::U8(s) => s,
            _ => candle_core::bail!("fp8_dequant expects u8 (fp8) input"),
        };

        let (scale_guard, _) = self.scale.storage_and_layout();
        let scale_slice = match &*scale_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("scale must be f32"),
            },
            _ => candle_core::bail!("scale must be on CUDA"),
        };

        // Allocate output as BF16
        let elem_count = num_tokens * self.hidden_size;
        let output_slice = dev.alloc_zeros::<bf16>(elem_count)?;

        let func = dev.get_or_load_custom_func("fp8_dequant_bf16", "fp8_quant", FP8_QUANT_PTX)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = self.hidden_size as i32;
        let per_token_i32: i32 = if self.per_token_scale { 1 } else { 0 };

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(in_slice);
        builder.arg(scale_slice);
        builder.arg(&hidden_size_i32);
        builder.arg(&per_token_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("fp8_dequant launch: {e}")))?;

        drop(scale_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(dims);
        Ok((output_storage, output_shape))
    }
}

/// Quantize a BF16 tensor to FP8 with a static scale.
///
/// # Arguments
/// * `input` - Input tensor [num_tokens, hidden_size] in BF16
/// * `scale` - Scale tensor [1] or [num_tokens] in F32
///
/// # Returns
/// Quantized tensor in U8 (FP8 E4M3 format)
pub fn fp8_quantize_static(input: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("fp8_quantize_static expects 2D input [num_tokens, hidden_size]");
    }

    let num_tokens = dims[0];
    let hidden_size = dims[1];

    let per_token_scale = scale.elem_count() == num_tokens;

    let op = Fp8StaticQuantOp {
        scale: scale.clone(),
        hidden_size,
        per_token_scale,
    };

    input.apply_op1_no_bwd(&op)
}

/// Quantize a BF16 tensor to FP8 with dynamic per-token scaling.
///
/// Uses a two-pass approach:
/// 1. Compute per-token scales based on max absolute values
/// 2. Quantize using the computed scales
///
/// # Arguments
/// * `input` - Input tensor [num_tokens, hidden_size] in BF16
///
/// # Returns
/// Tuple of (quantized tensor in U8, scales tensor [num_tokens] in F32)
pub fn fp8_quantize_dynamic_per_token(input: &Tensor) -> Result<(Tensor, Tensor)> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("fp8_quantize_dynamic_per_token expects 2D input");
    }

    let hidden_size = dims[1];

    // Pass 1: Compute per-token scales
    let compute_scales_op = Fp8ComputeScalesOp { hidden_size };
    let scales = input.apply_op1_no_bwd(&compute_scales_op)?;

    // Pass 2: Quantize with the computed scales
    let quant_op = Fp8QuantWithScalesOp {
        scale: scales.clone(),
        hidden_size,
    };
    let quantized = input.apply_op1_no_bwd(&quant_op)?;

    Ok((quantized, scales))
}

/// Dequantize an FP8 tensor to BF16.
///
/// # Arguments
/// * `input` - Input tensor [num_tokens, hidden_size] in U8 (FP8)
/// * `scale` - Scale tensor [1] or [num_tokens] in F32
///
/// # Returns
/// Dequantized tensor in BF16
pub fn fp8_dequantize(input: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("fp8_dequantize expects 2D input");
    }

    let num_tokens = dims[0];
    let hidden_size = dims[1];

    let per_token_scale = scale.elem_count() == num_tokens;

    let op = Fp8DequantOp {
        scale: scale.clone(),
        hidden_size,
        per_token_scale,
    };

    input.apply_op1_no_bwd(&op)
}

// ============================================================================
// FP8 GEMM Operations
// ============================================================================

/// FP8 GEMM operation: output = input @ (weight * scale).T
struct Fp8GemmOp {
    weight: Tensor,       // [N, K] in U8 (FP8)
    scale: Tensor,        // [1] or [N] in F32
    bias: Option<Tensor>, // [N] in BF16, optional
    per_channel_scale: bool,
}

impl CustomOp1 for Fp8GemmOp {
    fn name(&self) -> &'static str {
        "fp8_gemm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fp8_gemm requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &storage.device;
        let dims = layout.dims();

        // Input shape: [M, K] (or [batch..., M, K] flattened to [M, K])
        let m = if dims.len() == 2 {
            dims[0]
        } else {
            dims[..dims.len() - 1].iter().product()
        };
        let k = dims[dims.len() - 1];

        // Weight shape: [N, K]
        let weight_dims = self.weight.dims();
        let n = weight_dims[0];
        let weight_k = weight_dims[1];

        if k != weight_k {
            candle_core::bail!(
                "fp8_gemm: input K ({}) doesn't match weight K ({})",
                k,
                weight_k
            );
        }

        if layout.start_offset() != 0 {
            candle_core::bail!("fp8_gemm: input must be contiguous from offset 0");
        }

        let in_slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("fp8_gemm expects bf16 input"),
        };

        let (weight_guard, _) = self.weight.storage_and_layout();
        let weight_slice = match &*weight_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("weight must be u8 (fp8)"),
            },
            _ => candle_core::bail!("weight must be on CUDA"),
        };

        let (scale_guard, _) = self.scale.storage_and_layout();
        let scale_slice = match &*scale_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("scale must be f32"),
            },
            _ => candle_core::bail!("scale must be on CUDA"),
        };

        // Allocate output: [M, N]
        let output_slice = dev.alloc_zeros::<bf16>(m * n)?;

        // Choose kernel based on whether we have bias
        let has_bias = self.bias.is_some();
        let func_name = if has_bias {
            "fp8_gemm_bf16_with_bias"
        } else {
            "fp8_gemm_bf16_tiled"
        };

        let func = dev.get_or_load_custom_func(func_name, "fp8_gemm", FP8_GEMM_PTX)?;

        // Grid: (ceil(N/16), ceil(M/16))
        // Block: (16, 16)
        let grid_x = (n as u32).div_ceil(16);
        let grid_y = (m as u32).div_ceil(16);

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let k_i32 = k as i32;
        let per_channel_i32: i32 = if self.per_channel_scale { 1 } else { 0 };

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(in_slice);
        builder.arg(weight_slice);
        builder.arg(scale_slice);

        if has_bias {
            let bias_tensor = self.bias.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("bias tensor missing despite has_bias=true".to_string())
            })?;
            let (bias_guard, _) = bias_tensor.storage_and_layout();
            let bias_slice = match &*bias_guard {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::BF16(s) => s,
                    _ => candle_core::bail!("bias must be bf16"),
                },
                _ => candle_core::bail!("bias must be on CUDA"),
            };
            builder.arg(bias_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&per_channel_i32);
            let has_bias_i32: i32 = 1;
            builder.arg(&has_bias_i32);

            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("fp8_gemm launch: {e}")))?;

            drop(bias_guard);
        } else {
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&per_channel_i32);

            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("fp8_gemm launch: {e}")))?;
        }

        drop(weight_guard);
        drop(scale_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output_slice),
            device: dev.clone(),
        };

        // Output shape: [M, N] or original batch dims with N
        let output_shape = if dims.len() == 2 {
            Shape::from_dims(&[m, n])
        } else {
            let mut out_dims = dims[..dims.len() - 1].to_vec();
            out_dims.push(n);
            Shape::from_dims(&out_dims)
        };

        Ok((output_storage, output_shape))
    }
}

/// FP8 GEMM: output = input @ (weight * scale).T
///
/// # Arguments
/// * `input` - Input tensor [M, K] or [batch..., K] in BF16
/// * `weight` - Weight tensor [N, K] in U8 (FP8 E4M3)
/// * `scale` - Scale tensor [1] or [N] in F32
/// * `bias` - Optional bias tensor [N] in BF16
///
/// # Returns
/// Output tensor [M, N] or [batch..., N] in BF16
pub fn fp8_gemm(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let weight_dims = weight.dims();

    if weight_dims.len() != 2 {
        candle_core::bail!("fp8_gemm: weight must be 2D [N, K]");
    }

    let n = weight_dims[0];
    let per_channel_scale = scale.elem_count() == n;

    let op = Fp8GemmOp {
        weight: weight.clone(),
        scale: scale.clone(),
        bias: bias.cloned(),
        per_channel_scale,
    };

    input.apply_op1_no_bwd(&op)
}

/// Dequantize FP8 weights to BF16 for use with standard GEMM.
///
/// This is useful when you need to use cuBLAS or other optimized GEMM
/// implementations that don't support FP8 directly.
///
/// # Arguments
/// * `weight` - Weight tensor [N, K] in U8 (FP8)
/// * `scale` - Scale tensor [1] or [N] in F32
///
/// # Returns
/// Dequantized weight tensor [N, K] in BF16
pub fn fp8_dequant_weights(weight: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let weight = weight.contiguous()?;
    let dims = weight.dims();

    if dims.len() != 2 {
        candle_core::bail!("fp8_dequant_weights expects 2D weight [N, K]");
    }

    let _n = dims[0];
    // Note: per_channel detection is handled inside fp8_dequantize

    // Use the dequant function from fp8_cuda
    fp8_dequantize(&weight, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fp8_quant_requires_cuda() {
        // This test verifies that FP8 operations fail gracefully on CPU
        let input = Tensor::ones(&[4, 128], DType::BF16, &Device::Cpu).unwrap();
        let scale = Tensor::ones(1, DType::F32, &Device::Cpu).unwrap();

        let result = fp8_quantize_static(&input, &scale);
        assert!(result.is_err());
    }

    #[test]
    fn test_fp8_gemm_requires_cuda() {
        let input = Tensor::ones(&[4, 64], DType::BF16, &Device::Cpu).unwrap();
        let weight = Tensor::ones(&[128, 64], DType::U8, &Device::Cpu).unwrap();
        let scale = Tensor::ones(1, DType::F32, &Device::Cpu).unwrap();

        let result = fp8_gemm(&input, &weight, &scale, None);
        assert!(result.is_err());
    }
}
