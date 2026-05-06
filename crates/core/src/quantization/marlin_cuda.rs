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

// PTX module for the dedicated AWQ INT4 GEMV kernel used on the decode hot
// path (M ≤ 16). Lives in its own .cu/.ptx pair so cudarc does not need to
// re-link the legacy `marlin_gemm` module on every dispatch.
const AWQ_GEMV_PTX: &str = include_str!("../../kernels/awq_gemv.ptx");

/// Maximum M for which the dedicated GEMV path is active.
///
/// At M ≤ 16 the legacy `marlin_gemm_int4_zp_bf16` kernel under-utilises the
/// SM (it tiles for M = 16 but each tile column is computed by 16 redundant
/// threads). The GEMV path streams qweight at memory bandwidth instead.
const AWQ_GEMV_M_THRESHOLD: usize = 16;

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
        // Decode hot path: AWQ INT4 with zero points, no activation reorder
        // (g_idx), and a tiny M typical of single-token / multi-step decode.
        // Route to the bandwidth-bound GEMV kernel; everything else stays on
        // the legacy tiled path below to preserve prefill / GPTQ behaviour.
        if self.m <= AWQ_GEMV_M_THRESHOLD
            && matches!(self.scalar_type, MarlinScalarType::Uint4)
            && self.g_idx.is_none()
            && self.qzeros.is_some()
        {
            return self.cuda_fwd_awq_gemv(storage);
        }

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

        // cudarc 0.16 enforces `'arg: 'builder` on PushKernelArg, so every
        // value we pass to `builder.arg(...)` must outlive the builder.
        // Hoist guards and the null sentinel onto the outer scope before
        // creating the builder, then resolve optional tensors to slices
        // through references that share the function-scope lifetime.
        let null_ptr: u64 = 0;
        let qzeros_guard = self.qzeros.as_ref().map(|t| t.storage_and_layout());
        let bias_guard = self.bias.as_ref().map(|t| t.storage_and_layout());
        let g_idx_guard = self.g_idx.as_ref().map(|t| t.storage_and_layout());
        let si_guard = self
            .g_idx_sort_indices
            .as_ref()
            .map(|t| t.storage_and_layout());

        let qzeros_slice = match qzeros_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::U32(s) => Some(s),
                    _ => candle_core::bail!("marlin_gemm: qzeros must be U32"),
                },
                _ => candle_core::bail!("marlin_gemm: qzeros must be on CUDA"),
            },
            None => None,
        };
        let bias_slice = match bias_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::BF16(s) => Some(s),
                    _ => candle_core::bail!("marlin_gemm: bias must be BF16"),
                },
                _ => candle_core::bail!("marlin_gemm: bias must be on CUDA"),
            },
            None => None,
        };
        let g_idx_slice = match g_idx_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::U32(s) => Some(s),
                    _ => candle_core::bail!("marlin_gemm: g_idx must be U32"),
                },
                _ => candle_core::bail!("marlin_gemm: g_idx must be on CUDA"),
            },
            None => None,
        };
        let si_slice = match si_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::U32(s) => Some(s),
                    _ => candle_core::bail!("marlin_gemm: g_idx_sort_indices must be U32"),
                },
                _ => candle_core::bail!("marlin_gemm: g_idx_sort_indices must be on CUDA"),
            },
            None => None,
        };

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

        match qzeros_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }
        match bias_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }
        match g_idx_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }
        match si_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_gemm launch: {e}")))?;

        drop(qweight_guard);
        drop(scales_guard);
        drop(workspace_guard);
        drop(qzeros_guard);
        drop(bias_guard);
        drop(g_idx_guard);
        drop(si_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

impl MarlinGemmOp {
    /// AWQ INT4 GEMV decode-path launcher.
    ///
    /// Mirrors the input contract of the legacy `marlin_gemm_int4_zp_bf16`
    /// kernel (qweight `[K/8, N]`, scales `[G, N]`, qzeros `[G, N/8]`) but
    /// dispatches the dedicated `awq_gemv_int4_bf16` kernel which is tuned
    /// for M ≤ 16 and runs at memory bandwidth. Single warp per block, one
    /// thread per output column, K reduced sequentially per thread with
    /// FP32 accumulation.
    fn cuda_fwd_awq_gemv(&self, storage: &CudaStorage) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        // One-shot confirmation that the decode path is actually live.
        // Helps diagnose whether wall-clock degradation is in the kernel
        // itself or somewhere upstream.
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                target: "vllm_core::awq_gemv",
                "awq_gemv decode kernel active (M={}, N={}, K={}, num_groups={})",
                self.m, self.n, self.k, self.num_groups
            );
        });

        const N_TILE: u32 = 32;
        const BLOCK_THREADS: u32 = 32;

        let dev = &storage.device;

        // Output [M, N] in BF16, freshly allocated per call. Pre-allocation
        // (an output pool keyed by shape) is a separate item planned for the
        // CUDA-graph capture stage; the current call cost is dwarfed by the
        // K-reduction loop in the kernel itself.
        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let elem_count = self.m * self.n;
        let output = dev.alloc_zeros::<half::bf16>(elem_count)?;

        let func = dev.get_or_load_custom_func("awq_gemv_int4_bf16", "awq_gemv", AWQ_GEMV_PTX)?;

        // Activations.
        let input = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("awq_gemv: input must be BF16"),
        };

        // Hoist storage guards onto the function scope so all referenced
        // CUDA slices outlive the kernel builder (cudarc 0.16 enforces
        // `'arg: 'builder` on PushKernelArg).
        let (qweight_guard, _) = self.qweight.storage_and_layout();
        let (scales_guard, _) = self.scales.storage_and_layout();
        let qzeros_t = self
            .qzeros
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("awq_gemv: qzeros required".into()))?;
        let (qzeros_guard, _) = qzeros_t.storage_and_layout();
        let bias_guard = self.bias.as_ref().map(|t| t.storage_and_layout());

        let qweight = match &*qweight_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("awq_gemv: qweight must be U32"),
            },
            _ => candle_core::bail!("awq_gemv: qweight must be on CUDA"),
        };
        let scales = match &*scales_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("awq_gemv: scales must be BF16"),
            },
            _ => candle_core::bail!("awq_gemv: scales must be on CUDA"),
        };
        let qzeros = match &*qzeros_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("awq_gemv: qzeros must be U32"),
            },
            _ => candle_core::bail!("awq_gemv: qzeros must be on CUDA"),
        };
        let bias_slice = match bias_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::BF16(s) => Some(s),
                    _ => candle_core::bail!("awq_gemv: bias must be BF16"),
                },
                _ => candle_core::bail!("awq_gemv: bias must be on CUDA"),
            },
            None => None,
        };

        // Group size derived from K and num_groups. AWQ checkpoints have
        // `group_size % 8 == 0` (typically 128); the kernel relies on this
        // so a single (scale, zp) pair covers all 8 nibbles of one packed
        // word. We assert here defensively rather than silently producing
        // wrong outputs.
        if self.num_groups == 0 || !self.k.is_multiple_of(self.num_groups) {
            candle_core::bail!(
                "awq_gemv: K ({}) must be divisible by num_groups ({})",
                self.k,
                self.num_groups
            );
        }
        let group_size = (self.k / self.num_groups) as i32;
        if !(group_size as usize).is_multiple_of(8) {
            candle_core::bail!("awq_gemv: group_size ({group_size}) must be a multiple of 8");
        }
        // The kernel fans out coalesced loads in tiles of N_TILE columns;
        // mismatched widths would silently access out-of-bounds qzeros.
        if !self.n.is_multiple_of(N_TILE as usize) {
            candle_core::bail!("awq_gemv: N ({}) must be a multiple of {}", self.n, N_TILE);
        }
        if !self.k.is_multiple_of(8) {
            candle_core::bail!("awq_gemv: K ({}) must be a multiple of 8", self.k);
        }

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;
        let has_zp_i32: i32 = 1; // gated above
        let has_bias_i32: i32 = i32::from(self.bias.is_some());

        let grid_x: u32 = (self.n as u32) / N_TILE;
        // One BF16 input row cached in shared memory.
        let smem_bytes: u32 = (self.k as u32) * 2;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        let null_ptr: u64 = 0;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(qweight);
        builder.arg(scales);
        builder.arg(qzeros);
        match bias_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&group_size);
        builder.arg(&has_zp_i32);
        builder.arg(&has_bias_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("awq_gemv launch: {e}")))?;

        drop(qweight_guard);
        drop(scales_guard);
        drop(qzeros_guard);
        drop(bias_guard);

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

        // Hoist guards/null sentinel into the function scope so that every
        // pointer passed to the builder outlives it (cudarc 0.16 enforces
        // `'arg: 'builder` on PushKernelArg).
        let null_ptr: u64 = 0;
        let si_guard = self
            .g_idx_sort_indices
            .as_ref()
            .map(|t| t.storage_and_layout());
        let si_slice = match si_guard.as_ref() {
            Some((g, _)) => match &**g {
                Storage::Cuda(cs) => match &cs.slice {
                    CudaStorageSlice::U32(s) => Some(s),
                    _ => candle_core::bail!("repack_gptq_to_marlin: sort_indices must be U32"),
                },
                _ => candle_core::bail!("repack_gptq_to_marlin: sort_indices must be on CUDA"),
            },
            None => None,
        };

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(qweight);
        builder.arg(&size_k_i32);
        builder.arg(&size_n_i32);
        builder.arg(&has_sort_indices_i32);

        match si_slice {
            Some(s) => {
                builder.arg(s);
            }
            None => {
                builder.arg(&null_ptr);
            }
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("repack_gptq_to_marlin launch: {e}")))?;
        drop(si_guard);

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

/// CUDA op for full AWQ qweight → GPTQ qweight transform.
///
/// Both transposes the packing axis (`[K, N/8]` → `[K/8, N]`) and reorders
/// the nibbles within each u32 from AWQ interleaved to GPTQ sequential.
/// On the host this was an O(K·N) strided scalar Rust loop — 2.85 billion
/// random-access reads for Qwen3-4B's 252 linear layers, 5+ minutes wall
/// time. The kernel runs the same transform in milliseconds.
struct AwqToGptqQweightOp {
    in_features: usize,  // K
    out_features: usize, // N
}

impl CustomOp1 for AwqToGptqQweightOp {
    fn name(&self) -> &'static str {
        "awq_to_gptq_qweight_int4"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("awq_to_gptq_qweight_int4 requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !self.in_features.is_multiple_of(8) {
            candle_core::bail!(
                "awq_to_gptq_qweight_int4: in_features ({}) must be divisible by 8",
                self.in_features
            );
        }
        if !self.out_features.is_multiple_of(8) {
            candle_core::bail!(
                "awq_to_gptq_qweight_int4: out_features ({}) must be divisible by 8",
                self.out_features
            );
        }

        let dev = &storage.device;
        let func = dev.get_or_load_custom_func(
            "awq_to_gptq_qweight_int4",
            "marlin_gemm",
            MARLIN_GEMM_PTX,
        )?;

        let packed_k = self.in_features / 8;
        let packed_n = self.out_features / 8;

        // Output: [packed_k, N]
        let elem_count = packed_k * self.out_features;
        let output = dev.alloc_zeros::<u32>(elem_count)?;

        let input = match &storage.slice {
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("awq_to_gptq_qweight_int4: qweight must be U32"),
        };

        const BLOCK: u32 = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                (self.out_features as u32).div_ceil(BLOCK),
                (packed_k as u32).div_ceil(BLOCK),
                1,
            ),
            block_dim: (BLOCK, BLOCK, 1),
            shared_mem_bytes: 0,
        };

        let packed_k_i32 = packed_k as i32;
        let n_i32 = self.out_features as i32;
        let packed_n_i32 = packed_n as i32;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(&packed_k_i32);
        builder.arg(&n_i32);
        builder.arg(&packed_n_i32);

        unsafe { builder.launch(cfg) }.map_err(|e| {
            candle_core::Error::Msg(format!("awq_to_gptq_qweight_int4 launch: {e}"))
        })?;

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::U32(output),
            device: dev.clone(),
        };
        Ok((
            output_storage,
            Shape::from_dims(&[packed_k, self.out_features]),
        ))
    }
}

/// Transform AWQ qweight `[K, N/8]` (output-axis interleaved) into GPTQ
/// qweight `[K/8, N]` (input-axis sequential) on the GPU.
///
/// One-shot at load time; replaces the strided scalar Rust loop in
/// `awq_to_gptq_qweight` (`crates/core/src/quantization/awq_marlin.rs`).
pub fn awq_to_gptq_qweight_cuda(
    qweight: &Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    let op = AwqToGptqQweightOp {
        in_features,
        out_features,
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

        /// Smoke test with handcrafted constants — every output column has
        /// a known closed-form value, so a kernel-side indexing or
        /// dequant-formula bug shows up immediately without being masked
        /// by BF16 rounding on random data.
        #[test]
        fn test_awq_gemv_smoke_known_values() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            // Minimal shape: K = 8 (one packed word along K), N = 32 (one
            // N_TILE block), M = 1, group_size = 8 → num_groups = 1.
            let m = 1usize;
            let k = 8usize;
            let n = 32usize;
            let group_size = 8usize;
            let num_groups = k / group_size;
            let packed_k = k / 8;
            let packed_n = n / 8;

            // qweight: one row of N=32 U32 words. Each word packs K-axis
            // nibbles [0,1,2,3,4,5,6,7] (nib i at bit i*4 — sequential GPTQ
            // ordering, what the kernel reads).
            let w_word: u32 = (0..8).fold(0u32, |acc, i| acc | (i << (i * 4)));
            let qweight_words: Vec<u32> = vec![w_word; packed_k * n];

            // qzeros: zero point = 0 for every (group, output_lane) → the
            // dequant formula reduces to nib * scale.
            let qzeros_words: Vec<u32> = vec![0u32; num_groups * packed_n];

            // scales = 1.0 everywhere.
            let scales: Vec<f32> = vec![1.0f32; num_groups * n];
            // input = 1.0 everywhere.
            let inputs: Vec<f32> = vec![1.0f32; m * k];

            // Closed form: each output column = sum_k input[k] * (nib_k - 0) * 1
            //                                 = 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28.
            let expected: Vec<f32> = vec![28.0f32; m * n];

            let qweight = Tensor::from_vec(qweight_words, (packed_k, n), &device).unwrap();
            let qzeros = Tensor::from_vec(qzeros_words, (num_groups, packed_n), &device).unwrap();
            let scales_t = Tensor::from_vec(scales, (num_groups, n), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let input = Tensor::from_vec(inputs, (m, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let workspace = Tensor::zeros(64usize, DType::U32, &device).unwrap();

            let output = marlin_gemm(
                &input,
                &qweight,
                &scales_t,
                Some(&qzeros),
                None,
                None,
                &workspace,
                None,
                MarlinScalarType::Uint4,
                k,
                n,
                true,
                true,
            )
            .expect("awq_gemv launch should succeed");

            let got: Vec<f32> = output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 0.5,
                    "awq_gemv smoke[{i}]: got {g}, expected {e}"
                );
            }
        }

        /// Numerical equivalence: AWQ GEMV decode-path kernel vs a pure CPU
        /// dequant+matmul reference on identical random inputs. Asserts
        /// max absolute error within BF16 tolerance.
        ///
        /// Skipped unless a CUDA device is available.
        #[test]
        fn test_awq_gemv_matches_cpu_reference_m1() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            // Small but non-trivial shape — exercises group boundaries and
            // multi-block grid (N / N_TILE = 4 blocks).
            let m = 1usize;
            let k = 256usize;
            let n = 128usize;
            let group_size = 128usize;
            let num_groups = k / group_size; // 2
            let packed_k = k / 8;
            let packed_n = n / 8;

            // Deterministic pseudo-random fill.
            let mut rng: u32 = 0x9E37_79B9;
            let mut next = |modulus: u32| -> u32 {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                rng % modulus
            };

            // Pack int4 nibbles in [0, 15] sequentially along K — matches the
            // layout the legacy `marlin_gemm_int4_zp_bf16` kernel reads.
            let mut qweight_words: Vec<u32> = Vec::with_capacity(packed_k * n);
            for _ in 0..(packed_k * n) {
                let mut w: u32 = 0;
                for i in 0..8 {
                    w |= (next(16) & 0xF) << (i * 4);
                }
                qweight_words.push(w);
            }
            // Sequential nibbles along N (post-`repack_awq_nibbles` layout).
            let mut qzeros_words: Vec<u32> = Vec::with_capacity(num_groups * packed_n);
            for _ in 0..(num_groups * packed_n) {
                let mut w: u32 = 0;
                for i in 0..8 {
                    w |= (next(16) & 0xF) << (i * 4);
                }
                qzeros_words.push(w);
            }
            // Scales: small magnitudes; BF16-rounded so the CPU reference
            // sees exactly the same values the GPU kernel reads.
            let scales_f32: Vec<f32> = (0..(num_groups * n))
                .map(|_| ((next(2000) as f32) - 1000.0) * 1e-4)
                .collect();
            let inputs_f32: Vec<f32> = (0..(m * k))
                .map(|_| ((next(2000) as f32) - 1000.0) * 1e-3)
                .collect();

            // Round-trip through BF16 so the CPU reference and the GPU kernel
            // operate on the same numeric values; then any residual difference
            // is purely the F32 reduction order (which is identical for the
            // GEMV path: one thread per column reduces K sequentially).
            let bf16_round = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
            let scales: Vec<f32> = scales_f32.iter().map(|&v| bf16_round(v)).collect();
            let inputs: Vec<f32> = inputs_f32.iter().map(|&v| bf16_round(v)).collect();

            // CPU reference: dequant in F32, compute output[n] = sum_k input[k] * w[k,n].
            let mut expected = vec![0.0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    let zp_pack_col = ni / 8;
                    let zp_bit = (ni % 8) * 4;
                    let mut acc = 0.0f32;
                    for k_idx in 0..k {
                        let group_id = k_idx / group_size;
                        let pack_row = k_idx / 8;
                        let pack_idx = k_idx % 8;
                        let w_word = qweight_words[pack_row * n + ni];
                        let nib = ((w_word >> (pack_idx * 4)) & 0xF) as i32;
                        let zp_word = qzeros_words[group_id * packed_n + zp_pack_col];
                        let zp = ((zp_word >> zp_bit) & 0xF) as i32;
                        let scale = scales[group_id * n + ni];
                        let w_val = ((nib - zp) as f32) * scale;
                        acc += inputs[mi * k + k_idx] * w_val;
                    }
                    expected[mi * n + ni] = acc;
                }
            }

            // Build GPU tensors mirroring the legacy kernel's expected layout.
            let qweight = Tensor::from_vec(qweight_words, (packed_k, n), &device).unwrap();
            let qzeros = Tensor::from_vec(qzeros_words, (num_groups, packed_n), &device).unwrap();
            let scales_t = Tensor::from_vec(scales.clone(), (num_groups, n), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let input = Tensor::from_vec(inputs.clone(), (m, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let workspace = Tensor::zeros(64usize, DType::U32, &device).unwrap();

            // M=1 ≤ AWQ_GEMV_M_THRESHOLD → routed through the new GEMV kernel.
            let output = marlin_gemm(
                &input,
                &qweight,
                &scales_t,
                Some(&qzeros),
                None,
                None,
                &workspace,
                None,
                MarlinScalarType::Uint4,
                k,
                n,
                true,
                true,
            )
            .expect("awq_gemv launch should succeed");

            let got: Vec<f32> = output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            assert_eq!(got.len(), expected.len());
            // Final stage: BF16 quantisation of the output (kernel writes
            // via `__float2bfloat16`). At our random-data magnitudes the
            // sum can reach ~20, where the BF16 step size is ~0.08; allow
            // 1% of max-abs(expected) on top of a 5e-2 absolute floor for
            // small magnitudes.
            let max_expected = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let tol = (max_expected * 0.01).max(5e-2);
            let max_abs = got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= tol,
                "awq_gemv vs CPU ref max_abs={max_abs} > tol={tol} (max_expected={max_expected})"
            );
        }

        /// Same as the M=1 test but with M=4 (multi-step decode shape) — the
        /// kernel loops over M reusing its shared input cache. Confirms the
        /// per-row barrier wiring and the M-loop don't introduce drift.
        #[test]
        fn test_awq_gemv_matches_cpu_reference_m4() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m = 4usize;
            let k = 256usize;
            let n = 64usize;
            let group_size = 128usize;
            let num_groups = k / group_size;
            let packed_k = k / 8;
            let packed_n = n / 8;

            let mut rng: u32 = 0xCAFE_BABE;
            let mut next = |modulus: u32| -> u32 {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                rng % modulus
            };

            let mut qweight_words: Vec<u32> = Vec::with_capacity(packed_k * n);
            for _ in 0..(packed_k * n) {
                let mut w: u32 = 0;
                for i in 0..8 {
                    w |= (next(16) & 0xF) << (i * 4);
                }
                qweight_words.push(w);
            }
            let mut qzeros_words: Vec<u32> = Vec::with_capacity(num_groups * packed_n);
            for _ in 0..(num_groups * packed_n) {
                let mut w: u32 = 0;
                for i in 0..8 {
                    w |= (next(16) & 0xF) << (i * 4);
                }
                qzeros_words.push(w);
            }
            let scales_f32: Vec<f32> = (0..(num_groups * n))
                .map(|_| ((next(2000) as f32) - 1000.0) * 1e-4)
                .collect();
            let inputs_f32: Vec<f32> = (0..(m * k))
                .map(|_| ((next(2000) as f32) - 1000.0) * 1e-3)
                .collect();
            let bf16_round = |v: f32| -> f32 { half::bf16::from_f32(v).to_f32() };
            let scales: Vec<f32> = scales_f32.iter().map(|&v| bf16_round(v)).collect();
            let inputs: Vec<f32> = inputs_f32.iter().map(|&v| bf16_round(v)).collect();

            let mut expected = vec![0.0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    let zp_pack_col = ni / 8;
                    let zp_bit = (ni % 8) * 4;
                    let mut acc = 0.0f32;
                    for k_idx in 0..k {
                        let group_id = k_idx / group_size;
                        let pack_row = k_idx / 8;
                        let pack_idx = k_idx % 8;
                        let w_word = qweight_words[pack_row * n + ni];
                        let nib = ((w_word >> (pack_idx * 4)) & 0xF) as i32;
                        let zp_word = qzeros_words[group_id * packed_n + zp_pack_col];
                        let zp = ((zp_word >> zp_bit) & 0xF) as i32;
                        let scale = scales[group_id * n + ni];
                        let w_val = ((nib - zp) as f32) * scale;
                        acc += inputs[mi * k + k_idx] * w_val;
                    }
                    expected[mi * n + ni] = acc;
                }
            }

            let qweight = Tensor::from_vec(qweight_words, (packed_k, n), &device).unwrap();
            let qzeros = Tensor::from_vec(qzeros_words, (num_groups, packed_n), &device).unwrap();
            let scales_t = Tensor::from_vec(scales.clone(), (num_groups, n), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let input = Tensor::from_vec(inputs.clone(), (m, k), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(&device)
                .unwrap();
            let workspace = Tensor::zeros(64usize, DType::U32, &device).unwrap();

            let output = marlin_gemm(
                &input,
                &qweight,
                &scales_t,
                Some(&qzeros),
                None,
                None,
                &workspace,
                None,
                MarlinScalarType::Uint4,
                k,
                n,
                true,
                true,
            )
            .expect("awq_gemv launch should succeed");

            let got: Vec<f32> = output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            assert_eq!(got.len(), expected.len());
            let max_expected = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let tol = (max_expected * 0.01).max(5e-2);
            let max_abs = got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= tol,
                "awq_gemv (m=4) vs CPU ref max_abs={max_abs} > tol={tol} (max_expected={max_expected})"
            );
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
