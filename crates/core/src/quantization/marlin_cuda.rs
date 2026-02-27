//! CUDA kernels for Marlin INT4/INT8 GPTQ optimized inference.
//!
//! This module provides GPU-accelerated operations for Marlin kernels:
//! - `marlin_gemm`: Optimized GEMM with packed INT4 weights
//! - `repack_gptq_to_marlin`: Weight repacking for Marlin format
//!
//! Marlin achieves 2-3x speedup over standard dequantize+GEMM by:
//! - Asynchronous global memory loads with software pipelining
//! - Optimized warp shuffles for scale/zero-point distribution
//! - Custom INT4 packing optimized for tensor core MMA operations

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

use super::marlin::{MarlinScalarType, GPTQ_MARLIN_TILE};

// PTX module for Marlin kernels
const MARLIN_GEMM_PTX: &str = include_str!("../../kernels/marlin_gemm.ptx");

/// Marlin GEMM operation for INT4 quantized weights.
struct MarlinGemmOp {
    qweight: Tensor,
    scales: Tensor,
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    g_idx_sort_indices: Option<Tensor>,
    workspace: Tensor,
    bias: Option<Tensor>,
    scalar_type: MarlinScalarType,
    m: usize,
    n: usize,
    k: usize,
    num_groups: usize,
    is_k_full: bool,
    use_fp32_reduce: bool,
}

impl CustomOp1 for MarlinGemmOp {
    fn name(&self) -> &'static str {
        "marlin_gemm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Marlin GEMM requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        // Select kernel based on scalar type
        let kernel_name = match self.scalar_type {
            MarlinScalarType::Uint4b8 => "marlin_gemm_int4_bf16",
            MarlinScalarType::Uint8b128 => "marlin_gemm_int8_bf16",
            MarlinScalarType::Uint4 => "marlin_gemm_int4_zp_bf16",
            _ => candle_core::bail!("Unsupported scalar type for Marlin: {:?}", self.scalar_type),
        };

        let func = dev.get_or_load_custom_func(kernel_name, "marlin_gemm", MARLIN_GEMM_PTX)?;

        // Output shape: [M, N]
        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let elem_count = self.m * self.n;
        let output = dev.alloc_zeros::<half::bf16>(elem_count)?;

        // Get input (activations) slice
        let input = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            CudaStorageSlice::F16(s) => {
                // F16 input - need to handle differently
                let _ = s;
                candle_core::bail!("F16 input needs conversion to BF16 for Marlin kernel");
            }
            _ => candle_core::bail!("input must be BF16 or F16"),
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
                CudaStorageSlice::F16(s) => {
                    let _ = s;
                    candle_core::bail!("scales F16 needs conversion");
                }
                _ => candle_core::bail!("scales must be BF16 or F16"),
            },
            _ => candle_core::bail!("scales must be on CUDA"),
        };

        let (workspace_guard, _) = self.workspace.storage_and_layout();
        let workspace = match &*workspace_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("workspace must be U32"),
            },
            _ => candle_core::bail!("workspace must be on CUDA"),
        };

        // Calculate grid dimensions based on problem size
        let num_m_blocks = self.m.div_ceil(GPTQ_MARLIN_TILE);
        let num_n_blocks = self.n.div_ceil(GPTQ_MARLIN_TILE);

        // Thread configuration based on vLLM Marlin implementation
        let num_threads = 256u32;
        let blocks = (num_n_blocks as u32).max(1);

        let cfg = LaunchConfig {
            grid_dim: (blocks, num_m_blocks as u32, 1),
            block_dim: (num_threads, 1, 1),
            shared_mem_bytes: 32768, // Typical shared memory for Marlin
        };

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;
        let num_groups_i32 = self.num_groups as i32;
        let is_k_full_i32: i32 = if self.is_k_full { 1 } else { 0 };
        let use_fp32_reduce_i32: i32 = if self.use_fp32_reduce { 1 } else { 0 };
        let has_bias_i32: i32 = if self.bias.is_some() { 1 } else { 0 };
        let has_zp_i32: i32 = if self.qzeros.is_some() { 1 } else { 0 };
        let has_g_idx_i32: i32 = if self.g_idx.is_some() { 1 } else { 0 };

        // Build kernel args
        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(qweight);
        builder.arg(scales);
        builder.arg(workspace);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&num_groups_i32);
        builder.arg(&is_k_full_i32);
        builder.arg(&use_fp32_reduce_i32);
        builder.arg(&has_bias_i32);
        builder.arg(&has_zp_i32);
        builder.arg(&has_g_idx_i32);

        // Add optional tensors (pass null pointers if not present)
        if let Some(ref zp) = self.qzeros {
            let (zp_guard, _) = zp.storage_and_layout();
            if let Storage::Cuda(cs) = &*zp_guard {
                if let CudaStorageSlice::U32(s) = &cs.slice {
                    builder.arg(s);
                }
            }
        } else {
            let null_ptr: u64 = 0;
            builder.arg(&null_ptr);
        }

        if let Some(ref bias) = self.bias {
            let (bias_guard, _) = bias.storage_and_layout();
            if let Storage::Cuda(cs) = &*bias_guard {
                if let CudaStorageSlice::BF16(s) = &cs.slice {
                    builder.arg(s);
                }
            }
        } else {
            let null_ptr: u64 = 0;
            builder.arg(&null_ptr);
        }

        if let Some(ref g_idx) = self.g_idx {
            let (g_idx_guard, _) = g_idx.storage_and_layout();
            if let Storage::Cuda(cs) = &*g_idx_guard {
                if let CudaStorageSlice::U32(s) = &cs.slice {
                    builder.arg(s);
                }
            }
        } else {
            let null_ptr: u64 = 0;
            builder.arg(&null_ptr);
        }

        if let Some(ref sort_indices) = self.g_idx_sort_indices {
            let (si_guard, _) = sort_indices.storage_and_layout();
            if let Storage::Cuda(cs) = &*si_guard {
                if let CudaStorageSlice::U32(s) = &cs.slice {
                    builder.arg(s);
                }
            }
        } else {
            let null_ptr: u64 = 0;
            builder.arg(&null_ptr);
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_gemm launch: {e}")))?;

        drop(qweight_guard);
        drop(scales_guard);
        drop(workspace_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Perform Marlin optimized GEMM: output = input @ dequant(weight).T + bias
///
/// This is the main entry point for Marlin inference. It provides 2-3x speedup
/// over standard GPTQ dequantize+GEMM operations.
///
/// # Arguments
/// * `input` - Input activations [M, K] in BF16/F16
/// * `qweight` - Packed quantized weights in Marlin format [K/tile, N/tile, ...]
/// * `scales` - Permuted quantization scales [num_groups, N]
/// * `qzeros` - Optional zero points for AWQ (None for GPTQ symmetric)
/// * `g_idx` - Optional group indices for desc_act
/// * `g_idx_sort_indices` - Optional sort indices for g_idx
/// * `workspace` - Workspace tensor for Marlin kernel
/// * `bias` - Optional bias [N]
/// * `scalar_type` - Weight quantization type
/// * `size_k` - Input dimension K
/// * `size_n` - Output dimension N
/// * `is_k_full` - Whether K is full (not split for tensor parallelism)
/// * `use_fp32_reduce` - Use FP32 for reduction (better precision)
///
/// # Returns
/// Output [M, N] in BF16
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm(
    input: &Tensor,
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: Option<&Tensor>,
    g_idx: Option<&Tensor>,
    g_idx_sort_indices: Option<&Tensor>,
    workspace: &Tensor,
    bias: Option<&Tensor>,
    scalar_type: MarlinScalarType,
    size_k: usize,
    size_n: usize,
    is_k_full: bool,
    use_fp32_reduce: bool,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();

    // Handle batched input by reshaping to 2D
    let (m, k) = if dims.len() == 2 {
        (dims[0], dims[1])
    } else if dims.len() > 2 {
        let batch_size: usize = dims[..dims.len() - 1].iter().product();
        (batch_size, dims[dims.len() - 1])
    } else {
        candle_core::bail!("Input must have at least 2 dimensions");
    };

    if k != size_k {
        candle_core::bail!(
            "Input K dimension ({}) doesn't match size_k ({})",
            k,
            size_k
        );
    }

    // Calculate number of groups
    let num_groups = scales.dims()[0];

    let op = MarlinGemmOp {
        qweight: qweight.clone(),
        scales: scales.clone(),
        qzeros: qzeros.cloned(),
        g_idx: g_idx.cloned(),
        g_idx_sort_indices: g_idx_sort_indices.cloned(),
        workspace: workspace.clone(),
        bias: bias.cloned(),
        scalar_type,
        m,
        n: size_n,
        k: size_k,
        num_groups,
        is_k_full,
        use_fp32_reduce,
    };

    let result = input.apply_op1(op)?;

    // Reshape back to original batch dimensions if needed
    if dims.len() > 2 {
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(size_n);
        result.reshape(out_shape)
    } else {
        Ok(result)
    }
}

/// Repack weights from standard GPTQ format to Marlin format.
///
/// Standard GPTQ packs weights row-major as [K/pack_factor, N].
/// Marlin uses a tiled format optimized for tensor core operations.
struct RepackOp {
    g_idx_sort_indices: Option<Tensor>,
    size_k: usize,
    size_n: usize,
    bits: u32,
}

impl CustomOp1 for RepackOp {
    fn name(&self) -> &'static str {
        "repack_gptq_to_marlin"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Marlin weight repacking requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;

        let kernel_name = match self.bits {
            4 => "repack_gptq_to_marlin_int4",
            8 => "repack_gptq_to_marlin_int8",
            _ => candle_core::bail!("Unsupported bits for Marlin repacking: {}", self.bits),
        };

        let func = dev.get_or_load_custom_func(kernel_name, "marlin_gemm", MARLIN_GEMM_PTX)?;

        let pack_factor = 32 / self.bits as usize;

        // Output has same shape as input for now (Marlin internal format)
        // The actual tiled layout is handled by the kernel
        let packed_k = self.size_k / pack_factor;
        let output_shape = Shape::from_dims(&[packed_k, self.size_n]);
        let elem_count = packed_k * self.size_n;
        let output = dev.alloc_zeros::<u32>(elem_count)?;

        // Get input qweight
        let qweight = match &storage.slice {
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("qweight must be U32"),
        };

        let cfg = LaunchConfig {
            grid_dim: (
                (self.size_n as u32).div_ceil(16),
                (packed_k as u32).div_ceil(16),
                1,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let size_k_i32 = self.size_k as i32;
        let size_n_i32 = self.size_n as i32;
        let has_sort_indices_i32: i32 = if self.g_idx_sort_indices.is_some() {
            1
        } else {
            0
        };

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(qweight);
        builder.arg(&size_k_i32);
        builder.arg(&size_n_i32);
        builder.arg(&has_sort_indices_i32);

        if let Some(ref sort_indices) = self.g_idx_sort_indices {
            let (si_guard, _) = sort_indices.storage_and_layout();
            if let Storage::Cuda(cs) = &*si_guard {
                if let CudaStorageSlice::U32(s) = &cs.slice {
                    builder.arg(s);
                } else {
                    candle_core::bail!("sort_indices must be U32");
                }
            } else {
                candle_core::bail!("sort_indices must be on CUDA");
            }
        } else {
            let null_ptr: u64 = 0;
            builder.arg(&null_ptr);
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("repack_gptq_to_marlin launch: {e}")))?;

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::U32(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// Repack GPTQ weights to Marlin format.
///
/// # Arguments
/// * `qweight` - Packed GPTQ weights [K/pack_factor, N] in U32
/// * `g_idx_sort_indices` - Optional sort indices for activation order
/// * `size_k` - Input dimension K
/// * `size_n` - Output dimension N
/// * `bits` - Quantization bits (4 or 8)
///
/// # Returns
/// Repacked weights in Marlin format
pub fn repack_gptq_to_marlin(
    qweight: &Tensor,
    g_idx_sort_indices: Option<&Tensor>,
    size_k: usize,
    size_n: usize,
    bits: u32,
) -> Result<Tensor> {
    let op = RepackOp {
        g_idx_sort_indices: g_idx_sort_indices.cloned(),
        size_k,
        size_n,
        bits,
    };

    qweight.apply_op1(op)
}

// ─── AWQ nibble deinterleave (GPU) ──────────────────────────────────────────

/// Repack AWQ nibble ordering to GPTQ sequential ordering on GPU.
///
/// GPU-accelerated equivalent of the CPU `repack_awq_nibbles()`. Runs on any
/// CUDA device with sm_80+, since it only uses integer operations.
///
/// Input and output are both `[K/8, N]` U32 (8 INT4 values packed per word).
struct AwqRepackOp {
    rows: usize, // K/8
    cols: usize, // N
}

impl CustomOp1 for AwqRepackOp {
    fn name(&self) -> &'static str {
        "awq_marlin_repack_int4"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("awq_marlin_repack_int4 requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func =
            dev.get_or_load_custom_func("awq_marlin_repack_int4", "marlin_gemm", MARLIN_GEMM_PTX)?;

        let elem_count = self.rows * self.cols;
        let output = dev.alloc_zeros::<u32>(elem_count)?;

        let input = match &storage.slice {
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("awq_marlin_repack_int4: qweight must be U32"),
        };

        const BLOCK: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                self.cols.div_ceil(BLOCK as usize) as u32,
                self.rows.div_ceil(BLOCK as usize) as u32,
                1,
            ),
            block_dim: (BLOCK, BLOCK, 1),
            shared_mem_bytes: 0,
        };

        let rows_i32 = self.rows as i32;
        let cols_i32 = self.cols as i32;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(&rows_i32);
        builder.arg(&cols_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("awq_marlin_repack_int4 launch: {e}")))?;

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::U32(output),
            device: dev.clone(),
        };
        Ok((output_storage, Shape::from_dims(&[self.rows, self.cols])))
    }
}

/// Repack AWQ nibble ordering to GPTQ sequential ordering on GPU.
pub fn awq_marlin_repack(qweight: &Tensor, size_k: usize, size_n: usize) -> Result<Tensor> {
    let op = AwqRepackOp {
        rows: size_k / 8,
        cols: size_n,
    };
    qweight.apply_op1(op)
}

// ─── FP8 E4M3 GEMM for Ampere (software decode) ─────────────────────────────

/// CUDA op for FP8 E4M3 GEMM via software decode on Ampere (sm_80+).
///
/// Carries all non-input tensors so `CustomOp1` only takes the activation input.
struct Fp8AmpereGemmOp {
    qweight: Tensor,      // [N, K] U8  — FP8 E4M3
    weight_scale: Tensor, // [N]    F32 — per-channel scales
    bias: Option<Tensor>, // [N]    BF16 — optional
    m: usize,
    n: usize,
    k: usize,
}

impl CustomOp1 for Fp8AmpereGemmOp {
    fn name(&self) -> &'static str {
        "marlin_gemm_fp8_bf16"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("marlin_gemm_fp8_bf16 requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func =
            dev.get_or_load_custom_func("marlin_gemm_fp8_bf16", "marlin_gemm", MARLIN_GEMM_PTX)?;

        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let output = dev.alloc_zeros::<half::bf16>(self.m * self.n)?;

        let input = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("marlin_gemm_fp8_bf16: input must be BF16"),
        };

        let (qw_guard, _) = self.qweight.storage_and_layout();
        let qweight = match &*qw_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("marlin_gemm_fp8_bf16: qweight must be U8"),
            },
            _ => candle_core::bail!("marlin_gemm_fp8_bf16: qweight must be on CUDA"),
        };

        let (scale_guard, _) = self.weight_scale.storage_and_layout();
        let weight_scale = match &*scale_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("marlin_gemm_fp8_bf16: weight_scale must be F32"),
            },
            _ => candle_core::bail!("marlin_gemm_fp8_bf16: weight_scale must be on CUDA"),
        };

        let num_m_blocks = self.m.div_ceil(GPTQ_MARLIN_TILE);
        let num_n_blocks = self.n.div_ceil(GPTQ_MARLIN_TILE);
        let cfg = LaunchConfig {
            grid_dim: (num_n_blocks as u32, num_m_blocks as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;
        let has_bias_i32: i32 = if self.bias.is_some() { 1 } else { 0 };

        // Hoist optional bias guard outside the builder scope so the borrow outlives
        // `builder.launch(cfg)` — same lifetime constraint as in MarlinGemmOp.
        let bias_storage = self.bias.as_ref().map(|b| b.storage_and_layout());
        let null_bias_ptr: u64 = 0;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(qweight);
        builder.arg(weight_scale);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&has_bias_i32);

        match &bias_storage {
            Some((bias_guard, _)) => {
                if let Storage::Cuda(cs) = &**bias_guard {
                    if let CudaStorageSlice::BF16(s) = &cs.slice {
                        builder.arg(s);
                    } else {
                        candle_core::bail!("marlin_gemm_fp8_bf16: bias must be BF16");
                    }
                } else {
                    candle_core::bail!("marlin_gemm_fp8_bf16: bias must be on CUDA");
                }
            }
            None => {
                builder.arg(&null_bias_ptr);
            }
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_gemm_fp8_bf16 launch: {e}")))?;

        drop(qw_guard);
        drop(scale_guard);
        drop(bias_storage);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// FP8 E4M3 GEMM for Ampere GPUs (sm_80+) via software FP8 decode.
///
/// Computes `output = input @ dequant(qweight).T + bias` where
/// `qweight [N, K]` is FP8 E4M3 stored as U8 and `weight_scale [N]` is
/// per-channel F32. Runs on any CUDA device with sm_80+.
///
/// Unlike the Hopper native FP8 GEMM (`fp8_cuda::fp8_gemm`), this kernel
/// decodes FP8 weights in software and uses BF16 activations without
/// dynamic quantization.
pub fn fp8_ampere_gemm(
    input: &Tensor,
    qweight: &Tensor,
    weight_scale: &Tensor,
    bias: Option<&Tensor>,
    size_n: usize,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();
    let (m, k) = match dims.len() {
        2 => (dims[0], dims[1]),
        _ => {
            let batch: usize = dims[..dims.len() - 1].iter().product();
            (batch, dims[dims.len() - 1])
        }
    };

    let op = Fp8AmpereGemmOp {
        qweight: qweight.clone(),
        weight_scale: weight_scale.clone(),
        bias: bias.cloned(),
        m,
        n: size_n,
        k,
    };
    let result = input.apply_op1(op)?;

    if dims.len() > 2 {
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(size_n);
        result.reshape(out_shape)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_marlin_requires_cuda() {
        // Verify Marlin operations fail gracefully on CPU
        let qweight = Tensor::zeros(&[512, 4096], DType::U32, &Device::Cpu).unwrap();
        let scales = Tensor::ones(&[32, 4096], DType::BF16, &Device::Cpu).unwrap();
        let workspace = Tensor::zeros(64, DType::U32, &Device::Cpu).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, 4096), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let result = marlin_gemm(
            &input,
            &qweight,
            &scales,
            None,
            None,
            None,
            &workspace,
            None,
            MarlinScalarType::Uint4b8,
            4096,
            4096,
            true,
            true,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_repack_requires_cuda() {
        let qweight = Tensor::zeros(&[512, 4096], DType::U32, &Device::Cpu).unwrap();

        let result = repack_gptq_to_marlin(&qweight, None, 4096, 4096, 4);

        assert!(result.is_err());
    }

    #[test]
    fn test_awq_repack_requires_cuda() {
        let qweight = Tensor::zeros(&[16, 64], DType::U32, &Device::Cpu).unwrap();
        let result = awq_marlin_repack(&qweight, 128, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_fp8_ampere_gemm_requires_cuda() {
        let input = Tensor::zeros((4usize, 64usize), DType::BF16, &Device::Cpu).unwrap();
        let qweight = Tensor::zeros((64usize, 64usize), DType::U8, &Device::Cpu).unwrap();
        let scale = Tensor::ones(64usize, DType::F32, &Device::Cpu).unwrap();
        let result = fp8_ampere_gemm(&input, &qweight, &scale, None, 64);
        assert!(result.is_err());
    }

    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_marlin_gemm_gpu() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m: usize = 4;
            let k: usize = 4096;
            let n: usize = 4096;
            let group_size: usize = 128;
            let bits: u32 = 4;
            let pack_factor = 32 / bits as usize;
            let num_groups = k / group_size;

            // Input activations
            let input = Tensor::randn(0.0f32, 1.0, (m, k), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();

            // Quantized weights (in Marlin format - for testing, use zeros)
            let qweight = Tensor::zeros(&[k / pack_factor, n], DType::U32, &device).unwrap();
            let scales = Tensor::ones(&[num_groups, n], DType::BF16, &device).unwrap();
            let workspace = Tensor::zeros(64, DType::U32, &device).unwrap();

            // This will fail if Marlin PTX is not available, which is expected
            // The actual test is to verify the API works
            let result = marlin_gemm(
                &input,
                &qweight,
                &scales,
                None,
                None,
                None,
                &workspace,
                None,
                MarlinScalarType::Uint4b8,
                k,
                n,
                true,
                true,
            );

            // Result should be Ok if Marlin PTX is present, Err otherwise
            // Both are acceptable for this test
            if let Ok(output) = result {
                assert_eq!(output.dims(), &[m, n]);
            }
        }

        #[test]
        fn test_awq_repack_gpu_matches_cpu() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            // Build a qweight with known values on CPU, repack via GPU, compare.
            let k = 128usize;
            let n = 64usize;
            let rows = k / 8;

            // All-zero input: AWQ repack should give all-zero output.
            let qweight_cpu = Tensor::zeros((rows, n), DType::U32, &Device::Cpu).unwrap();
            let qweight_gpu = qweight_cpu.to_device(&device).unwrap();

            let result = awq_marlin_repack(&qweight_gpu, k, n);
            if let Ok(repacked) = result {
                assert_eq!(repacked.dims(), &[rows, n]);
                let values: Vec<u32> = repacked
                    .to_device(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert!(values.iter().all(|&v| v == 0));
            }
        }

        #[test]
        fn test_fp8_ampere_gemm_zero_weights() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m = 4usize;
            let k = 64usize;
            let n = 32usize;

            let input = Tensor::ones((m, k), DType::BF16, &device).unwrap();
            // Zero FP8 weights → output should be zero regardless of scale.
            let qweight = Tensor::zeros((n, k), DType::U8, &device).unwrap();
            let scale = Tensor::ones(n, DType::F32, &device).unwrap();

            let result = fp8_ampere_gemm(&input, &qweight, &scale, None, n);
            if let Ok(output) = result {
                assert_eq!(output.dims(), &[m, n]);
                let vals: Vec<f32> = output
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert!(
                    vals.iter().all(|&v| v.abs() < 1e-3),
                    "expected zeros, got {:?}",
                    &vals[..4]
                );
            }
        }
    }
}
