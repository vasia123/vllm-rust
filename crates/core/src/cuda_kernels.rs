use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, CustomOp2, DType, Layout, Result,
    Shape, Storage, Tensor,
};

const PTX: &str = include_str!("../kernels/paged_attention.ptx");

#[cfg(feature = "cuda-fused-activations")]
const SWIGLU_PTX: &str = include_str!("../kernels/swiglu.ptx");
const HEAD_DIM: usize = 128;
const NUM_WARPS: usize = 4;

struct PagedAttnOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
}

impl CustomOp1 for PagedAttnOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1 requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &q_storage.device;
        let num_seqs = q_layout.dims()[0];

        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }

        let q_slice = match &q_storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("paged_attention expects bf16 Q tensor"),
        };

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("paged_attention expects bf16 K cache"),
            },
            _ => candle_core::bail!("paged_attention: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("paged_attention expects bf16 V cache"),
            },
            _ => candle_core::bail!("paged_attention: V cache must be on CUDA"),
        };

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        let elem_count = num_seqs * self.num_heads * HEAD_DIM;
        let output_slice = dev.alloc_zeros::<bf16>(elem_count)?;

        let func =
            dev.get_or_load_custom_func("paged_attention_v1_bf16", "paged_attention", PTX)?;

        // Shared memory: q_smem[128] + reduce_smem[4] + logits[max_seq_len], all f32
        let shared_mem_bytes =
            ((HEAD_DIM + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(q_slice);
        builder.arg(k_slice);
        builder.arg(v_slice);
        builder.arg(bt_slice);
        builder.arg(sl_slice);
        builder.arg(&self.scale);
        builder.arg(&num_heads_i32);
        builder.arg(&num_kv_heads_i32);
        builder.arg(&max_blocks_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("paged_attention launch: {e}")))?;

        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, HEAD_DIM]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v1 decode kernel.
///
/// Replaces per-sequence cache read + GQA repeat + matmul + softmax + matmul
/// with a single CUDA kernel launch.
///
/// # Arguments
/// - `q`: Query tensor `[num_seqs, num_heads, 128]` bf16, contiguous
/// - `k_cache`: K cache `[num_blocks, 16, num_kv_heads, 128]` bf16
/// - `v_cache`: V cache `[num_blocks, 16, num_kv_heads, 128]` bf16
/// - `block_tables`: Physical block IDs `[num_seqs, max_blocks_per_seq]` u32
/// - `seq_lens`: KV length per sequence `[num_seqs]` u32
/// - `scale`: Attention scale factor (1/sqrt(head_dim))
/// - `num_heads`: Number of query heads
/// - `num_kv_heads`: Number of KV heads (for GQA)
/// - `max_blocks_per_seq`: Second dim of block_tables
/// - `max_seq_len`: Maximum sequence length in this batch (for shared memory sizing)
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * 128]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;

    let op = PagedAttnOp {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens.contiguous()?,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * HEAD_DIM))
}

// ============================================================================
// PagedAttention with ALiBi
// ============================================================================

/// Paged attention op with ALiBi positional bias support.
///
/// The CUDA kernel adds `alibi_slopes[head] * (key_pos - query_pos)` to each
/// attention logit, implementing ALiBi (Attention with Linear Biases) for
/// models like Bloom and MPT.
struct PagedAttnAlibiOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    alibi_slopes: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
}

impl CustomOp1 for PagedAttnAlibiOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1_alibi"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1_alibi requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &q_storage.device;
        let num_seqs = q_layout.dims()[0];

        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }

        let q_slice = match &q_storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("paged_attention expects bf16 Q tensor"),
        };

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("paged_attention expects bf16 K cache"),
            },
            _ => candle_core::bail!("paged_attention: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("paged_attention expects bf16 V cache"),
            },
            _ => candle_core::bail!("paged_attention: V cache must be on CUDA"),
        };

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        let (alibi_guard, _) = self.alibi_slopes.storage_and_layout();
        let alibi_slice = match &*alibi_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("alibi_slopes must be F32"),
            },
            _ => candle_core::bail!("alibi_slopes must be on CUDA"),
        };

        let elem_count = num_seqs * self.num_heads * HEAD_DIM;
        let output_slice = dev.alloc_zeros::<bf16>(elem_count)?;

        let func =
            dev.get_or_load_custom_func("paged_attention_v1_bf16_alibi", "paged_attention", PTX)?;

        let shared_mem_bytes =
            ((HEAD_DIM + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(q_slice);
        builder.arg(k_slice);
        builder.arg(v_slice);
        builder.arg(bt_slice);
        builder.arg(sl_slice);
        builder.arg(&self.scale);
        builder.arg(&num_heads_i32);
        builder.arg(&num_kv_heads_i32);
        builder.arg(&max_blocks_i32);
        builder.arg(alibi_slice);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("paged_attention alibi launch: {e}")))?;

        drop(alibi_guard);
        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, HEAD_DIM]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v1 decode kernel with ALiBi positional bias.
///
/// Identical to [`paged_attention_cuda`] but adds ALiBi (Attention with Linear
/// Biases) to the attention logits inside the kernel. This is used by models
/// like Bloom and MPT that use ALiBi instead of RoPE for positional encoding.
///
/// The kernel applies: `logit[h, pos] += alibi_slopes[h] * (pos - (seq_len - 1))`
/// which gives non-positive bias to past tokens (closer = less negative).
///
/// # Arguments
/// Same as [`paged_attention_cuda`], plus:
/// - `alibi_slopes`: Per-head ALiBi slopes `[num_heads]` f32, on CUDA
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * 128]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda_alibi(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    alibi_slopes: &Tensor,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;

    let alibi_slopes = alibi_slopes
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?;

    let op = PagedAttnAlibiOp {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens.contiguous()?,
        alibi_slopes,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * HEAD_DIM))
}

// ============================================================================
// Fused SwiGLU Activation
// ============================================================================

/// Fused SwiGLU operation: output = silu(gate) * up
///
/// Computes the SwiGLU activation in a single kernel pass, avoiding
/// materialization of intermediate results and saving memory bandwidth.
#[cfg(feature = "cuda-fused-activations")]
pub struct FusedSwiGluOp;

#[cfg(feature = "cuda-fused-activations")]
impl CustomOp2 for FusedSwiGluOp {
    fn name(&self) -> &'static str {
        "fused_swiglu"
    }

    fn cpu_fwd(
        &self,
        gate_storage: &CpuStorage,
        gate_layout: &Layout,
        up_storage: &CpuStorage,
        up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // CPU fallback implementation using standard operations
        use candle_core::cpu_backend::unary_map;

        let gate_shape = gate_layout.shape();
        let up_shape = up_layout.shape();

        if gate_shape != up_shape {
            candle_core::bail!(
                "fused_swiglu: gate and up shapes must match, got {:?} and {:?}",
                gate_shape,
                up_shape
            );
        }

        // For CPU, we do element-wise: silu(gate) * up
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        match (gate_storage, up_storage) {
            (CpuStorage::F32(gate_data), CpuStorage::F32(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<f32> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let silu_g = g / (1.0 + (-g).exp());
                        silu_g * u
                    })
                    .collect();

                Ok((CpuStorage::F32(result), gate_shape.clone()))
            }
            (CpuStorage::BF16(gate_data), CpuStorage::BF16(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<half::bf16> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let g_f32 = g.to_f32();
                        let u_f32 = u.to_f32();
                        let silu_g = g_f32 / (1.0 + (-g_f32).exp());
                        half::bf16::from_f32(silu_g * u_f32)
                    })
                    .collect();

                Ok((CpuStorage::BF16(result), gate_shape.clone()))
            }
            (CpuStorage::F16(gate_data), CpuStorage::F16(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<half::f16> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let g_f32 = g.to_f32();
                        let u_f32 = u.to_f32();
                        let silu_g = g_f32 / (1.0 + (-g_f32).exp());
                        half::f16::from_f32(silu_g * u_f32)
                    })
                    .collect();

                Ok((CpuStorage::F16(result), gate_shape.clone()))
            }
            _ => candle_core::bail!(
                "fused_swiglu: unsupported dtype combination or mismatched dtypes"
            ),
        }
    }

    fn cuda_fwd(
        &self,
        gate_storage: &CudaStorage,
        gate_layout: &Layout,
        up_storage: &CudaStorage,
        up_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &gate_storage.device;
        let gate_shape = gate_layout.shape();
        let up_shape = up_layout.shape();

        if gate_shape != up_shape {
            candle_core::bail!(
                "fused_swiglu: gate and up shapes must match, got {:?} and {:?}",
                gate_shape,
                up_shape
            );
        }

        if gate_layout.start_offset() != 0 {
            candle_core::bail!("fused_swiglu: gate must be contiguous from offset 0");
        }
        if up_layout.start_offset() != 0 {
            candle_core::bail!("fused_swiglu: up must be contiguous from offset 0");
        }

        // Calculate dimensions
        // Shape is [..., hidden_size], we need num_tokens and hidden_size
        let dims = gate_shape.dims();
        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();
        let num_tokens = if num_tokens == 0 { 1 } else { num_tokens };

        let elem_count = num_tokens * hidden_size;
        let block_size = std::cmp::min(hidden_size, 1024);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = hidden_size as i32;

        match (&gate_storage.slice, &up_storage.slice) {
            (CudaStorageSlice::BF16(gate_slice), CudaStorageSlice::BF16(up_slice)) => {
                let output_slice = dev.alloc_zeros::<half::bf16>(elem_count)?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_bf16", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::BF16(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            (CudaStorageSlice::F16(gate_slice), CudaStorageSlice::F16(up_slice)) => {
                let output_slice = dev.alloc_zeros::<half::f16>(elem_count)?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_fp16", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::F16(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            (CudaStorageSlice::F32(gate_slice), CudaStorageSlice::F32(up_slice)) => {
                let output_slice = dev.alloc_zeros::<f32>(elem_count)?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_fp32", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::F32(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            _ => candle_core::bail!("fused_swiglu: unsupported dtype, expected BF16, F16, or F32"),
        }
    }
}

/// Fused SwiGLU activation: output = silu(gate) * up
///
/// Computes the SwiGLU activation in a single CUDA kernel pass when the
/// `cuda-fused-activations` feature is enabled. Falls back to CPU implementation
/// for CPU tensors.
///
/// This saves memory bandwidth by avoiding materialization of the intermediate
/// silu(gate) result.
///
/// # Arguments
/// - `gate`: Gate tensor from gate_proj linear layer `[..., hidden_size]`
/// - `up`: Up tensor from up_proj linear layer `[..., hidden_size]`
///
/// # Returns
/// Output tensor `[..., hidden_size]` with same dtype as inputs
///
/// # Panics
/// - If gate and up have different shapes
/// - If gate and up have different dtypes
/// - If tensors are not contiguous
#[cfg(feature = "cuda-fused-activations")]
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;

    gate.apply_op2_no_bwd(&up, &FusedSwiGluOp)
}

/// Check if fused SwiGLU CUDA kernel is available.
///
/// Returns true if:
/// - The `cuda-fused-activations` feature is enabled
/// - The tensor is on a CUDA device
#[cfg(feature = "cuda-fused-activations")]
pub fn fused_swiglu_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-fused-activations"))]
pub fn fused_swiglu_available(_tensor: &Tensor) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_silu_formula() {
        // Verify our SiLU implementation matches expected values
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let test_cases = [
            (0.0f32, 0.0f32),         // silu(0) = 0
            (1.0f32, 0.7310586f32),   // silu(1) ≈ 0.731
            (-1.0f32, -0.2689414f32), // silu(-1) ≈ -0.269
            (2.0f32, 1.7615942f32),   // silu(2) ≈ 1.762
        ];

        for (input, expected) in test_cases {
            let computed = input / (1.0 + (-input).exp());
            assert!(
                (computed - expected).abs() < 1e-5,
                "silu({}) = {}, expected {}",
                input,
                computed,
                expected
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_cpu_f32() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], (2, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (2, 2), &device).unwrap();

        let result = fused_swiglu(&gate, &up).unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Expected: silu(gate) * up = silu(gate) * 1 = silu(gate)
        let expected: Vec<f32> = vec![1.0f32, 2.0, -1.0, 0.5]
            .iter()
            .map(|&g| g / (1.0 + (-g).exp()))
            .collect();

        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-5,
                "fused_swiglu mismatch: got {}, expected {}",
                r,
                e
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_cpu_bf16() {
        let device = Device::Cpu;
        let gate_f32 = vec![1.0f32, 2.0, -1.0, 0.5];
        let up_f32 = vec![2.0f32, 0.5, 1.0, 3.0];

        let gate = Tensor::from_vec(gate_f32.clone(), (2, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let up = Tensor::from_vec(up_f32.clone(), (2, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let result = fused_swiglu(&gate, &up).unwrap();
        let result_f32 = result.to_dtype(DType::F32).unwrap();
        let result_data: Vec<f32> = result_f32.flatten_all().unwrap().to_vec1().unwrap();

        // Expected: silu(gate) * up
        let expected: Vec<f32> = gate_f32
            .iter()
            .zip(up_f32.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        for (r, e) in result_data.iter().zip(expected.iter()) {
            // BF16 has lower precision, use larger tolerance
            assert!(
                (r - e).abs() < 0.05,
                "fused_swiglu bf16 mismatch: got {}, expected {}",
                r,
                e
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_shape_mismatch() {
        let device = Device::Cpu;
        let gate = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let up = Tensor::zeros((2, 8), DType::F32, &device).unwrap();

        let result = fused_swiglu(&gate, &up);
        assert!(result.is_err());
    }
}
