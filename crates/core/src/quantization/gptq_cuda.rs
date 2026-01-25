//! CUDA kernels for GPTQ INT4/INT8 dequantization and fused GEMM.
//!
//! This module provides GPU-accelerated operations for GPTQ quantized models:
//! - `gptq_dequantize`: Unpack and dequantize weights to BF16/F16
//! - `gptq_gemm`: Fused dequantization + matrix multiplication

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

// PTX module for GPTQ kernels
const GPTQ_DEQUANT_PTX: &str = include_str!("../../kernels/gptq_dequant.ptx");

/// GPTQ dequantization operation.
struct GptqDequantOp {
    scales: Tensor,
    qzeros: Tensor,
    in_features: usize,
    out_features: usize,
    group_size: i32,
    num_groups: usize,
    bits: u32,
}

impl CustomOp1 for GptqDequantOp {
    fn name(&self) -> &'static str {
        "gptq_dequant"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GPTQ dequantization requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        // Get kernel function based on bits
        let kernel_name = if self.bits == 4 {
            "gptq_dequant_int4_bf16"
        } else {
            "gptq_dequant_int8_bf16"
        };
        let func = dev.get_or_load_custom_func(kernel_name, "gptq_dequant", GPTQ_DEQUANT_PTX)?;

        // Allocate output: [in_features, out_features] in BF16
        let output_shape = Shape::from_dims(&[self.in_features, self.out_features]);
        let elem_count = self.in_features * self.out_features;
        let output = dev.alloc_zeros::<half::bf16>(elem_count)?;

        // Get input (qweight) slice
        let qweight = match &storage.slice {
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("qweight must be U32"),
        };

        // Get scales slice
        let (scales_guard, _) = self.scales.storage_and_layout();
        let scales = match &*scales_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("scales must be BF16"),
            },
            _ => candle_core::bail!("scales must be on CUDA"),
        };

        // Get qzeros slice
        let (qzeros_guard, _) = self.qzeros.storage_and_layout();
        let qzeros = match &*qzeros_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("qzeros must be U32"),
            },
            _ => candle_core::bail!("qzeros must be on CUDA"),
        };

        // Launch kernel
        let block_dim = (16u32, 16u32, 1u32);
        let grid_x = (self.out_features as u32).div_ceil(16);
        let grid_y = (self.in_features as u32).div_ceil(16);

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim,
            shared_mem_bytes: 0,
        };

        let in_features_i32 = self.in_features as i32;
        let out_features_i32 = self.out_features as i32;
        let num_groups_i32 = self.num_groups as i32;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(qweight);
        builder.arg(scales);
        builder.arg(qzeros);
        builder.arg(&in_features_i32);
        builder.arg(&out_features_i32);
        builder.arg(&self.group_size);
        builder.arg(&num_groups_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("gptq_dequant launch: {e}")))?;

        drop(scales_guard);
        drop(qzeros_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Dequantize GPTQ weights from packed INT4/INT8 to BF16.
///
/// # Arguments
/// * `qweight` - Packed quantized weights [K/pack_factor, N] in U32
/// * `scales` - Quantization scales [num_groups, N] in BF16
/// * `qzeros` - Packed zero points [num_groups, N/pack_factor] in U32
/// * `in_features` - Input feature dimension (K)
/// * `out_features` - Output feature dimension (N)
/// * `group_size` - Group size for quantization (-1 for per-channel)
/// * `bits` - Quantization bits (4 or 8)
///
/// # Returns
/// Dequantized weights [K, N] in BF16
pub fn gptq_dequantize(
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    in_features: usize,
    out_features: usize,
    group_size: i32,
    bits: u32,
) -> Result<Tensor> {
    let num_groups = if group_size <= 0 {
        1
    } else {
        in_features.div_ceil(group_size as usize)
    };

    let op = GptqDequantOp {
        scales: scales.clone(),
        qzeros: qzeros.clone(),
        in_features,
        out_features,
        group_size,
        num_groups,
        bits,
    };

    qweight.apply_op1(op)
}

/// GPTQ fused GEMM operation: input @ dequant(weight).T + bias
struct GptqGemmOp {
    qweight: Tensor,
    scales: Tensor,
    qzeros: Tensor,
    bias: Option<Tensor>,
    m: usize,
    n: usize,
    k: usize,
    group_size: i32,
    num_groups: usize,
}

impl CustomOp1 for GptqGemmOp {
    fn name(&self) -> &'static str {
        "gptq_gemm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("GPTQ GEMM requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        let func = dev.get_or_load_custom_func(
            "gptq_gemm_int4_bf16_tiled",
            "gptq_dequant",
            GPTQ_DEQUANT_PTX,
        )?;

        // Output shape: [M, N]
        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let elem_count = self.m * self.n;
        let output = dev.alloc_zeros::<half::bf16>(elem_count)?;

        // Get input (activations) slice
        let input = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("input must be BF16"),
        };

        // Get weight tensors
        let (qweight_guard, _) = self.qweight.storage_and_layout();
        let qweight = match &*qweight_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("qweight must be U32"),
            },
            _ => candle_core::bail!("qweight must be on CUDA"),
        };

        let (scales_guard, _) = self.scales.storage_and_layout();
        let scales = match &*scales_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("scales must be BF16"),
            },
            _ => candle_core::bail!("scales must be on CUDA"),
        };

        let (qzeros_guard, _) = self.qzeros.storage_and_layout();
        let qzeros = match &*qzeros_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("qzeros must be U32"),
            },
            _ => candle_core::bail!("qzeros must be on CUDA"),
        };

        // Launch kernel
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
        let num_groups_i32 = self.num_groups as i32;
        let has_bias_i32: i32 = if self.bias.is_some() { 1 } else { 0 };

        // Build kernel launch based on whether bias is present
        if let Some(ref bias) = self.bias {
            let (bias_guard, _) = bias.storage_and_layout();
            let bias_slice = match &*bias_guard {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::BF16(s) => s,
                    _ => candle_core::bail!("bias must be BF16"),
                },
                _ => candle_core::bail!("bias must be on CUDA"),
            };

            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input);
            builder.arg(qweight);
            builder.arg(scales);
            builder.arg(qzeros);
            builder.arg(bias_slice);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.group_size);
            builder.arg(&num_groups_i32);
            builder.arg(&has_bias_i32);

            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("gptq_gemm launch: {e}")))?;

            drop(bias_guard);
        } else {
            // No bias - push a null pointer (0u64)
            let null_ptr: u64 = 0;
            let mut builder = func.builder();
            builder.arg(&output);
            builder.arg(input);
            builder.arg(qweight);
            builder.arg(scales);
            builder.arg(qzeros);
            builder.arg(&null_ptr);
            builder.arg(&m_i32);
            builder.arg(&n_i32);
            builder.arg(&k_i32);
            builder.arg(&self.group_size);
            builder.arg(&num_groups_i32);
            builder.arg(&has_bias_i32);

            unsafe { builder.launch(cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("gptq_gemm launch: {e}")))?;
        }

        drop(qweight_guard);
        drop(scales_guard);
        drop(qzeros_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Perform fused GPTQ GEMM: output = input @ dequant(weight).T + bias
///
/// # Arguments
/// * `input` - Input activations [M, K] in BF16
/// * `qweight` - Packed quantized weights [K/8, N] in U32
/// * `scales` - Quantization scales [num_groups, N] in BF16
/// * `qzeros` - Packed zero points [num_groups, N/8] in U32
/// * `bias` - Optional bias [N] in BF16
/// * `group_size` - Group size for quantization
///
/// # Returns
/// Output [M, N] in BF16
pub fn gptq_gemm(
    input: &Tensor,
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    bias: Option<&Tensor>,
    group_size: i32,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    if dims.len() != 2 {
        candle_core::bail!("gptq_gemm expects 2D input [M, K]");
    }

    let m = dims[0];
    let k = dims[1];

    // Get N from scales shape
    let scales_dims = scales.dims();
    let n = scales_dims[scales_dims.len() - 1];
    let num_groups = if group_size <= 0 {
        1
    } else {
        k.div_ceil(group_size as usize)
    };

    let op = GptqGemmOp {
        qweight: qweight.clone(),
        scales: scales.clone(),
        qzeros: qzeros.clone(),
        bias: bias.cloned(),
        m,
        n,
        k,
        group_size,
        num_groups,
    };

    input.apply_op1(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_gptq_requires_cuda() {
        // Verify GPTQ operations fail gracefully on CPU
        let qweight = Tensor::zeros(&[8, 64], DType::U32, &Device::Cpu).unwrap();
        let scales = Tensor::ones(&[4, 64], DType::BF16, &Device::Cpu).unwrap();
        let qzeros = Tensor::zeros(&[4, 8], DType::U32, &Device::Cpu).unwrap();

        let result = gptq_dequantize(&qweight, &scales, &qzeros, 64, 64, 16, 4);
        assert!(result.is_err());
    }

    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_gptq_dequantize() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let in_features: usize = 64;
            let out_features: usize = 128;
            let group_size: usize = 32;
            let bits = 4;
            let pack_factor = 32 / bits as usize; // 8 for 4-bit
            let num_groups = in_features.div_ceil(group_size);

            // Create packed weights (all zeros for simplicity)
            let qweight = Tensor::zeros(
                &[in_features / pack_factor, out_features],
                DType::U32,
                &device,
            )
            .unwrap();

            // Create scales (all ones)
            let scales = Tensor::ones(&[num_groups, out_features], DType::BF16, &device).unwrap();

            // Create zero points (all zeros packed)
            let qzeros = Tensor::zeros(
                &[num_groups, out_features / pack_factor],
                DType::U32,
                &device,
            )
            .unwrap();

            let result = gptq_dequantize(
                &qweight,
                &scales,
                &qzeros,
                in_features,
                out_features,
                group_size as i32,
                bits,
            );

            assert!(result.is_ok());
            let output = result.unwrap();
            assert_eq!(output.dims(), &[in_features, out_features]);
        }

        #[test]
        fn test_gptq_gemm() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 4; // batch size
            let k: usize = 64; // in_features
            let n: usize = 128; // out_features
            let group_size: usize = 32;
            let pack_factor: usize = 8; // 4-bit packing
            let num_groups = k.div_ceil(group_size);

            // Input activations
            let input = Tensor::randn(0.0f32, 1.0, (m, k), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            // Quantized weights
            let qweight = Tensor::zeros(&[k / pack_factor, n], DType::U32, &device).unwrap();
            let scales = Tensor::ones(&[num_groups, n], DType::BF16, &device).unwrap();
            let qzeros =
                Tensor::zeros(&[num_groups, n / pack_factor], DType::U32, &device).unwrap();

            let result = gptq_gemm(&input, &qweight, &scales, &qzeros, None, group_size as i32);

            assert!(result.is_ok());
            let output = result.unwrap();
            assert_eq!(output.dims(), &[m, n]);
        }
    }
}
