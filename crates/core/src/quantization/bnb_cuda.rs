//! CUDA kernels for BitsAndBytes fused dequantization + GEMM.
//!
//! This module provides GPU-accelerated fused operations for BitsAndBytes
//! quantized models, avoiding the intermediate materialization of full-precision
//! weight tensors:
//!
//! - `bnb_nf4_gemm`: Fused NF4 dequantization + matrix multiplication
//! - `bnb_int8_gemm`: Fused INT8 dequantization + matrix multiplication

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

const BNB_FUSED_MATMUL_PTX: &str = include_str!("../../kernels/bnb_fused_matmul.ptx");

// ============================================================================
// NF4 Fused GEMM
// ============================================================================

/// NF4 fused dequantize + GEMM operation.
///
/// Computes output = input @ dequant_nf4(packed_weight).T [+ bias]
/// without materializing the dequantized weight matrix.
struct BnbNf4GemmOp {
    packed_weight: Tensor,
    absmax: Tensor,
    bias: Option<Tensor>,
    m: usize,
    n: usize,
    k: usize,
    block_size: i32,
}

impl CustomOp1 for BnbNf4GemmOp {
    fn name(&self) -> &'static str {
        "bnb_nf4_gemm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "BnB NF4 fused GEMM requires CUDA; use the CPU dequantize-then-matmul fallback path"
        )
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        // Choose kernel: with or without bias
        let kernel_name = if self.bias.is_some() {
            "bnb_nf4_gemm_f32_bias"
        } else {
            "bnb_nf4_gemm_f32_tiled"
        };

        let func =
            dev.get_or_load_custom_func(kernel_name, "bnb_fused_matmul", BNB_FUSED_MATMUL_PTX)?;

        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let elem_count = self.m * self.n;
        let output = dev.alloc_zeros::<f32>(elem_count)?;

        // Input activations (f32)
        let input_slice = match &storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("BnB NF4 fused GEMM: input must be F32"),
        };

        // Packed NF4 weights (u8)
        let (weight_guard, _) = self.packed_weight.storage_and_layout();
        let weight_slice = match &*weight_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("BnB NF4 fused GEMM: packed_weight must be U8"),
            },
            _ => candle_core::bail!("BnB NF4 fused GEMM: packed_weight must be on CUDA"),
        };

        // Absmax scales (f32)
        let (absmax_guard, _) = self.absmax.storage_and_layout();
        let absmax_slice = match &*absmax_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("BnB NF4 fused GEMM: absmax must be F32"),
            },
            _ => candle_core::bail!("BnB NF4 fused GEMM: absmax must be on CUDA"),
        };

        let block_dim = (16u32, 16u32, 1u32);
        let grid_x = (self.n as u32).div_ceil(16);
        let grid_y = (self.m as u32).div_ceil(16);

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim,
            shared_mem_bytes: 0,
        };

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;

        if let Some(ref bias_tensor) = self.bias {
            let (bias_guard, _) = bias_tensor.storage_and_layout();
            let bias_slice = match &*bias_guard {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::F32(s) => s,
                    _ => candle_core::bail!("BnB NF4 fused GEMM: bias must be F32"),
                },
                _ => candle_core::bail!("BnB NF4 fused GEMM: bias must be on CUDA"),
            };

            let has_bias_i32: i32 = 1;
            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input_slice);
            builder.arg(weight_slice);
            builder.arg(absmax_slice);
            builder.arg(bias_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.block_size);
            builder.arg(&has_bias_i32);

            // SAFETY: All device pointers are valid for the duration of the kernel launch.
            // The input, weight, absmax, and bias slices are borrowed from Tensor storage
            // guards that remain alive until after the launch completes. Output is freshly
            // allocated on the same device. Grid and block dimensions are validated above
            // to cover the full output matrix.
            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("bnb_nf4_gemm_f32_bias launch: {e}"))
            })?;

            drop(bias_guard);
        } else {
            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input_slice);
            builder.arg(weight_slice);
            builder.arg(absmax_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.block_size);

            // SAFETY: All device pointers are valid for the duration of the kernel launch.
            // The input, weight, and absmax slices are borrowed from Tensor storage guards
            // that remain alive until after the launch completes. Output is freshly
            // allocated on the same device.
            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("bnb_nf4_gemm_f32_tiled launch: {e}"))
            })?;
        }

        drop(weight_guard);
        drop(absmax_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::F32(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Fused NF4 dequantization + GEMM: output = input @ dequant_nf4(weight).T [+ bias]
///
/// This avoids materializing the full-precision weight tensor, reducing memory
/// bandwidth and allocation overhead compared to separate dequantize + matmul.
///
/// # Arguments
/// * `input` - Input activations [M, K] in F32
/// * `packed_weight` - Packed NF4 weights [N * K / 2] in U8
/// * `absmax` - Per-block absmax scales [num_blocks] in F32
/// * `bias` - Optional bias [N] in F32
/// * `out_features` - Output features (N dimension)
/// * `block_size` - Quantization block size
///
/// # Returns
/// Output [M, N] in F32
pub fn bnb_nf4_gemm(
    input: &Tensor,
    packed_weight: &Tensor,
    absmax: &Tensor,
    bias: Option<&Tensor>,
    out_features: usize,
    block_size: i32,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("bnb_nf4_gemm: input must be 2D [M, K], got {:?}", dims);
    }

    let m = dims[0];
    let k = dims[1];

    let op = BnbNf4GemmOp {
        packed_weight: packed_weight.clone(),
        absmax: absmax.clone(),
        bias: bias.cloned(),
        m,
        n: out_features,
        k,
        block_size,
    };

    input.apply_op1(op)
}

// ============================================================================
// INT8 Fused GEMM
// ============================================================================

/// INT8 fused dequantize + GEMM operation.
///
/// Computes output = input @ dequant_int8(weight).T [+ bias]
/// without materializing the dequantized weight matrix.
struct BnbInt8GemmOp {
    weight: Tensor,
    absmax: Tensor,
    bias: Option<Tensor>,
    m: usize,
    n: usize,
    k: usize,
    block_size: i32,
}

impl CustomOp1 for BnbInt8GemmOp {
    fn name(&self) -> &'static str {
        "bnb_int8_gemm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "BnB INT8 fused GEMM requires CUDA; use the CPU dequantize-then-matmul fallback path"
        )
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        let kernel_name = if self.bias.is_some() {
            "bnb_int8_gemm_f32_bias"
        } else {
            "bnb_int8_gemm_f32_tiled"
        };

        let func =
            dev.get_or_load_custom_func(kernel_name, "bnb_fused_matmul", BNB_FUSED_MATMUL_PTX)?;

        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let elem_count = self.m * self.n;
        let output = dev.alloc_zeros::<f32>(elem_count)?;

        // Input activations (f32)
        let input_slice = match &storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("BnB INT8 fused GEMM: input must be F32"),
        };

        // INT8 weights stored as u8
        let (weight_guard, _) = self.weight.storage_and_layout();
        let weight_slice = match &*weight_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("BnB INT8 fused GEMM: weight must be U8"),
            },
            _ => candle_core::bail!("BnB INT8 fused GEMM: weight must be on CUDA"),
        };

        // Absmax scales (f32)
        let (absmax_guard, _) = self.absmax.storage_and_layout();
        let absmax_slice = match &*absmax_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("BnB INT8 fused GEMM: absmax must be F32"),
            },
            _ => candle_core::bail!("BnB INT8 fused GEMM: absmax must be on CUDA"),
        };

        let block_dim = (16u32, 16u32, 1u32);
        let grid_x = (self.n as u32).div_ceil(16);
        let grid_y = (self.m as u32).div_ceil(16);

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim,
            shared_mem_bytes: 0,
        };

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;

        if let Some(ref bias_tensor) = self.bias {
            let (bias_guard, _) = bias_tensor.storage_and_layout();
            let bias_slice = match &*bias_guard {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::F32(s) => s,
                    _ => candle_core::bail!("BnB INT8 fused GEMM: bias must be F32"),
                },
                _ => candle_core::bail!("BnB INT8 fused GEMM: bias must be on CUDA"),
            };

            let has_bias_i32: i32 = 1;
            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input_slice);
            builder.arg(weight_slice);
            builder.arg(absmax_slice);
            builder.arg(bias_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.block_size);
            builder.arg(&has_bias_i32);

            // SAFETY: All device pointers are valid for the duration of the kernel launch.
            // See BnbNf4GemmOp::cuda_fwd for detailed safety justification.
            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("bnb_int8_gemm_f32_bias launch: {e}"))
            })?;

            drop(bias_guard);
        } else {
            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input_slice);
            builder.arg(weight_slice);
            builder.arg(absmax_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.block_size);

            // SAFETY: All device pointers are valid for the duration of the kernel launch.
            // See BnbNf4GemmOp::cuda_fwd for detailed safety justification.
            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("bnb_int8_gemm_f32_tiled launch: {e}"))
            })?;
        }

        drop(weight_guard);
        drop(absmax_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::F32(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Fused INT8 dequantization + GEMM: output = input @ dequant_int8(weight).T [+ bias]
///
/// # Arguments
/// * `input` - Input activations [M, K] in F32
/// * `weight` - INT8 weights stored as U8 [N, K]
/// * `absmax` - Per-block absmax scales [num_blocks] in F32
/// * `bias` - Optional bias [N] in F32
/// * `block_size` - Quantization block size
///
/// # Returns
/// Output [M, N] in F32
pub fn bnb_int8_gemm(
    input: &Tensor,
    weight: &Tensor,
    absmax: &Tensor,
    bias: Option<&Tensor>,
    block_size: i32,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("bnb_int8_gemm: input must be 2D [M, K], got {:?}", dims);
    }

    let weight_dims = weight.dims();
    if weight_dims.len() != 2 {
        candle_core::bail!(
            "bnb_int8_gemm: weight must be 2D [N, K], got {:?}",
            weight_dims
        );
    }

    let m = dims[0];
    let k = dims[1];
    let n = weight_dims[0];

    if weight_dims[1] != k {
        candle_core::bail!(
            "bnb_int8_gemm: input K ({}) doesn't match weight K ({})",
            k,
            weight_dims[1]
        );
    }

    let op = BnbInt8GemmOp {
        weight: weight.clone(),
        absmax: absmax.clone(),
        bias: bias.cloned(),
        m,
        n,
        k,
        block_size,
    };

    input.apply_op1(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_bnb_nf4_gemm_requires_cuda() {
        let input = Tensor::ones(&[4, 64], DType::F32, &Device::Cpu).unwrap();
        let packed_weight = Tensor::zeros(64 * 128 / 2, DType::U8, &Device::Cpu).unwrap();
        let absmax = Tensor::ones(64 * 128 / 64, DType::F32, &Device::Cpu).unwrap();

        let result = bnb_nf4_gemm(&input, &packed_weight, &absmax, None, 128, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_bnb_int8_gemm_requires_cuda() {
        let input = Tensor::ones(&[4, 64], DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::zeros(&[128, 64], DType::U8, &Device::Cpu).unwrap();
        let absmax = Tensor::ones(128 * 64 / 64, DType::F32, &Device::Cpu).unwrap();

        let result = bnb_int8_gemm(&input, &weight, &absmax, None, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_bnb_nf4_gemm_input_validation() {
        // 3D input should be rejected
        let input = Tensor::ones(&[2, 4, 64], DType::F32, &Device::Cpu).unwrap();
        let packed_weight = Tensor::zeros(64 * 128 / 2, DType::U8, &Device::Cpu).unwrap();
        let absmax = Tensor::ones(64 * 128 / 64, DType::F32, &Device::Cpu).unwrap();

        let result = bnb_nf4_gemm(&input, &packed_weight, &absmax, None, 128, 64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("2D"),
            "Expected error about 2D input, got: {err_msg}"
        );
    }

    #[test]
    fn test_bnb_int8_gemm_input_validation() {
        // 3D input should be rejected
        let input = Tensor::ones(&[2, 4, 64], DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::zeros(&[128, 64], DType::U8, &Device::Cpu).unwrap();
        let absmax = Tensor::ones(128 * 64 / 64, DType::F32, &Device::Cpu).unwrap();

        let result = bnb_int8_gemm(&input, &weight, &absmax, None, 64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("2D"),
            "Expected error about 2D input, got: {err_msg}"
        );
    }

    #[test]
    fn test_bnb_int8_gemm_k_mismatch() {
        let input = Tensor::ones(&[4, 64], DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::zeros(&[128, 32], DType::U8, &Device::Cpu).unwrap(); // K=32 != 64
        let absmax = Tensor::ones(128 * 32 / 64, DType::F32, &Device::Cpu).unwrap();

        let result = bnb_int8_gemm(&input, &weight, &absmax, None, 64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("doesn't match"),
            "Expected K mismatch error, got: {err_msg}"
        );
    }

    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_bnb_nf4_gemm_basic() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 4;
            let k: usize = 64;
            let n: usize = 32;
            let block_size: usize = 64;

            // Create test data
            let input = Tensor::ones(&[m, k], DType::F32, &device).unwrap();
            let packed_len = n * k / 2;
            let num_blocks = (n * k).div_ceil(block_size);

            // All zeros packed weight => all codebook[0] = -1.0
            let packed_weight = Tensor::zeros(packed_len, DType::U8, &device).unwrap();
            let absmax = Tensor::ones(num_blocks, DType::F32, &device).unwrap();

            let result = bnb_nf4_gemm(&input, &packed_weight, &absmax, None, n, block_size as i32);

            assert!(result.is_ok(), "NF4 GEMM failed: {:?}", result.err());
            let output = result.unwrap();
            assert_eq!(output.dims(), &[m, n]);
        }

        #[test]
        fn test_bnb_int8_gemm_basic() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 4;
            let k: usize = 64;
            let n: usize = 32;
            let block_size: usize = 64;

            let input = Tensor::ones(&[m, k], DType::F32, &device).unwrap();
            let num_blocks = (n * k).div_ceil(block_size);

            // All-zero INT8 weights
            let weight = Tensor::zeros(&[n, k], DType::U8, &device).unwrap();
            let absmax = Tensor::ones(num_blocks, DType::F32, &device).unwrap();

            let result = bnb_int8_gemm(&input, &weight, &absmax, None, block_size as i32);

            assert!(result.is_ok(), "INT8 GEMM failed: {:?}", result.err());
            let output = result.unwrap();
            assert_eq!(output.dims(), &[m, n]);
        }

        #[test]
        fn test_bnb_nf4_gemm_with_bias() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 2;
            let k: usize = 32;
            let n: usize = 16;
            let block_size: usize = 32;

            let input = Tensor::ones(&[m, k], DType::F32, &device).unwrap();
            let packed_len = n * k / 2;
            let num_blocks = (n * k).div_ceil(block_size);

            let packed_weight = Tensor::zeros(packed_len, DType::U8, &device).unwrap();
            let absmax = Tensor::ones(num_blocks, DType::F32, &device).unwrap();
            let bias = Tensor::ones(n, DType::F32, &device).unwrap();

            let result = bnb_nf4_gemm(
                &input,
                &packed_weight,
                &absmax,
                Some(&bias),
                n,
                block_size as i32,
            );

            assert!(
                result.is_ok(),
                "NF4 GEMM with bias failed: {:?}",
                result.err()
            );
            let output = result.unwrap();
            assert_eq!(output.dims(), &[m, n]);
        }

        #[test]
        fn test_bnb_int8_gemm_with_bias() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 2;
            let k: usize = 32;
            let n: usize = 16;
            let block_size: usize = 32;

            let input = Tensor::ones(&[m, k], DType::F32, &device).unwrap();
            let num_blocks = (n * k).div_ceil(block_size);

            let weight = Tensor::zeros(&[n, k], DType::U8, &device).unwrap();
            let absmax = Tensor::ones(num_blocks, DType::F32, &device).unwrap();
            let bias = Tensor::ones(n, DType::F32, &device).unwrap();

            let result = bnb_int8_gemm(&input, &weight, &absmax, Some(&bias), block_size as i32);

            assert!(
                result.is_ok(),
                "INT8 GEMM with bias failed: {:?}",
                result.err()
            );
            let output = result.unwrap();
            assert_eq!(output.dims(), &[m, n]);
        }
    }
}
