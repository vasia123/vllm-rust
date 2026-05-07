#[cfg(feature = "cuda-fused-activations")]
use candle_core::CustomOp2;
use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

/// Clone a CudaStorageSlice by pattern-matching each variant.
/// CudaStorageSlice does not derive Clone, but inner CudaSlice<T> does.
#[cfg(feature = "cuda")]
fn clone_cuda_storage_slice(slice: &CudaStorageSlice) -> CudaStorageSlice {
    match slice {
        CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.clone()),
        CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.clone()),
        CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.clone()),
        CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.clone()),
        CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.clone()),
        CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.clone()),
        CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.clone()),
    }
}

const PTX: &str = include_str!("../kernels/paged_attention.ptx");

#[cfg(feature = "cuda-fused-activations")]
const SWIGLU_PTX: &str = include_str!("../kernels/swiglu.ptx");

/// SM version this binary was compiled against (set by build.rs).
/// Returns 0 when cuda-kernels feature was not enabled at build time.
pub fn cuda_target_sm() -> u32 {
    option_env!("CUDA_TARGET_SM")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Check if FP8 CUDA kernels are available (requires sm_89+).
pub fn fp8_kernels_available() -> bool {
    cuda_target_sm() >= 89
}

const NUM_WARPS: usize = 4;

/// Validate the V2 paged-attention contract `max(seq_lens) <= max_seq_len`.
///
/// `max_seq_len` controls how many partition slots the V2 path allocates and
/// how many `tmp_out` partitions Stage-1 actually writes. The Stage-2 reduce
/// kernel iterates over `ceil(seq_len / PARTITION_SIZE)` partitions per
/// sequence — if any `seq_lens[i] > max_seq_len`, reduce reads beyond the
/// initialised range and silently produces wrong outputs. The check is gated
/// behind `debug_assertions` so release builds avoid the device→host sync.
fn debug_assert_seq_lens_within_bound(seq_lens: &Tensor, max_seq_len: usize) -> Result<()> {
    if cfg!(debug_assertions) {
        let lens: Vec<u32> = seq_lens.to_vec1()?;
        if let Some(&actual_max) = lens.iter().max() {
            if (actual_max as usize) > max_seq_len {
                candle_core::bail!(
                    "paged_attention_v2: contract violation — seq_lens.max()={} exceeds max_seq_len={}",
                    actual_max,
                    max_seq_len
                );
            }
        }
    }
    Ok(())
}

/// Storage dtype for PagedAttention kernels.
///
/// Implementations name the kernel symbols and provide CudaStorageSlice
/// projections for their concrete float type. Adding a new dtype (e.g. fp8
/// activations) is one impl block — no kernel-launch code changes.
trait PagedAttnDtype:
    candle_core::cuda::cudarc::driver::DeviceRepr
    + candle_core::cuda::cudarc::driver::ValidAsZeroBits
    + Default
    + Copy
    + 'static
{
    const KERNEL_V1: &'static str;
    const KERNEL_V1_ALIBI: &'static str;
    const KERNEL_V2: &'static str;
    const KERNEL_V2_ALIBI: &'static str;
    const KERNEL_V2_REDUCE: &'static str;
    const NAME: &'static str;

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>>;
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice;
}

impl PagedAttnDtype for half::bf16 {
    const KERNEL_V1: &'static str = "paged_attention_v1_bf16";
    const KERNEL_V1_ALIBI: &'static str = "paged_attention_v1_bf16_alibi";
    const KERNEL_V2: &'static str = "paged_attention_v2_bf16";
    const KERNEL_V2_ALIBI: &'static str = "paged_attention_v2_bf16_alibi";
    const KERNEL_V2_REDUCE: &'static str = "paged_attention_v2_reduce_bf16";
    const NAME: &'static str = "bf16";

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>> {
        match s {
            CudaStorageSlice::BF16(s) => Some(s),
            _ => None,
        }
    }
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice {
        CudaStorageSlice::BF16(s)
    }
}

impl PagedAttnDtype for half::f16 {
    const KERNEL_V1: &'static str = "paged_attention_v1_f16";
    const KERNEL_V1_ALIBI: &'static str = "paged_attention_v1_f16_alibi";
    const KERNEL_V2: &'static str = "paged_attention_v2_f16";
    const KERNEL_V2_ALIBI: &'static str = "paged_attention_v2_f16_alibi";
    const KERNEL_V2_REDUCE: &'static str = "paged_attention_v2_reduce_f16";
    const NAME: &'static str = "f16";

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>> {
        match s {
            CudaStorageSlice::F16(s) => Some(s),
            _ => None,
        }
    }
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice {
        CudaStorageSlice::F16(s)
    }
}

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
    head_dim: usize,
    block_size: usize,
}

impl CustomOp1 for PagedAttnOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1 requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v1::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v1::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnOp {
    fn run_v1<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention: Q dtype mismatch with selected kernel ({})",
                T::NAME
            ))
        })?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention: K cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention: V cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
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

        let head_dim = self.head_dim;
        let elem_count = num_seqs * self.num_heads * head_dim;
        // Each (seq, head) block writes a full `head_dim` slice of output;
        // every byte is covered before any read. SAFETY: kernel never
        // reads `output_slice` before writing it.
        let output_slice = unsafe { dev.alloc::<T>(elem_count) }?;

        let func = dev.get_or_load_custom_func(T::KERNEL_V1, "paged_attention", PTX)?;

        // Shared memory: q_smem[head_dim] + reduce_smem[NUM_WARPS] + logits[max_seq_len]
        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;

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
        builder.arg(&head_dim_i32);
        builder.arg(&block_size_i32);

        // SAFETY: kernel launch with validated parameters and contiguous buffers
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("paged_attention launch: {e}")))?;

        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v1 decode kernel.
///
/// Replaces per-sequence cache read + GQA repeat + matmul + softmax + matmul
/// with a single CUDA kernel launch. Supports configurable head dimensions
/// and block sizes for compatibility with different model architectures.
///
/// # Arguments
/// - `q`: Query tensor `[num_seqs, num_heads, head_dim]` bf16, contiguous
/// - `k_cache`: K cache `[num_blocks, block_size, num_kv_heads, head_dim]` bf16
/// - `v_cache`: V cache `[num_blocks, block_size, num_kv_heads, head_dim]` bf16
/// - `block_tables`: Physical block IDs `[num_seqs, max_blocks_per_seq]` u32
/// - `seq_lens`: KV length per sequence `[num_seqs]` u32
/// - `scale`: Attention scale factor (1/sqrt(head_dim))
/// - `num_heads`: Number of query heads
/// - `num_kv_heads`: Number of KV heads (for GQA)
/// - `max_blocks_per_seq`: Second dim of block_tables
/// - `max_seq_len`: Maximum sequence length in this batch (for shared memory sizing)
/// - `head_dim`: Head dimension (e.g. 64, 96, 128, 256)
/// - `block_size`: KV cache block size (e.g. 8, 16, 32)
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
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
    head_dim: usize,
    block_size: usize,
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
        head_dim,
        block_size,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

// ============================================================================
// PagedAttention V2: Split-K for long sequences
// ============================================================================

/// Partition size for V2 split-K attention (must match PARTITION_SIZE in paged_attention.cu).
///
/// Smaller partitions give more grid blocks per (q_head, seq) pair which
/// improves SM occupancy at batch=1 decode. With PARTITION_SIZE=128 and
/// seq_len≈480, the V2 grid is (num_heads, 1, 4) = 128 blocks for Qwen3-4B,
/// vs 32 blocks under V1 — ~4× more parallelism.
const PARTITION_SIZE: usize = 128;

/// Threshold: use V2 when max_seq_len exceeds this value.
///
/// At batch=1 V2's split-K parallelism beats V1's single-block-per-head
/// design even for short sequences — the per-token `__syncthreads` count
/// inside V1 dominates the kernel runtime. V1 is kept only for very short
/// sequences where the V2 reduce kernel's small launch overhead would
/// noticeably show up.
const V2_SEQ_LEN_THRESHOLD: usize = 64;

/// Stage 1: compute partitioned attention outputs.
struct PagedAttnV2Op {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    /// Caller's stated upper bound on `seq_lens[i]`. The kernel itself only
    /// reads `max_num_partitions` (derived from this), but we keep the field
    /// to validate the contract `max(seq_lens) <= max_seq_len` at the public
    /// API entry — violating it would make the reduce kernel read
    /// uninitialised partitions.
    #[allow(dead_code)]
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    max_num_partitions: usize,
}

impl CustomOp1 for PagedAttnV2Op {
    fn name(&self) -> &'static str {
        "paged_attention_v2"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v2 requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention_v2: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v2::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v2::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v2 expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnV2Op {
    fn run_v2<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v2: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2: K cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention_v2: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2: V cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention_v2: V cache must be on CUDA"),
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

        let head_dim = self.head_dim;
        let max_num_partitions = self.max_num_partitions;

        // Allocate intermediate buffers
        let tmp_out_size = num_seqs * self.num_heads * max_num_partitions * head_dim;
        let meta_size = num_seqs * self.num_heads * max_num_partitions;
        let tmp_out_slice = dev.alloc_zeros::<f32>(tmp_out_size)?;
        let exp_sums_slice = dev.alloc_zeros::<f32>(meta_size)?;
        let max_logits_slice = dev.alloc_zeros::<f32>(meta_size)?;

        // Final output
        let out_size = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(out_size)?;

        // Stage 1: partitioned attention kernel
        let v2_func = dev.get_or_load_custom_func(T::KERNEL_V2, "paged_attention", PTX)?;

        // Shared memory: q[head_dim] + reduce[NUM_WARPS] + logits[PARTITION_SIZE]
        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + PARTITION_SIZE) * std::mem::size_of::<f32>()) as u32;

        let v2_cfg = LaunchConfig {
            grid_dim: (
                self.num_heads as u32,
                num_seqs as u32,
                max_num_partitions as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;
        let max_partitions_i32 = max_num_partitions as i32;

        {
            let mut builder = v2_func.builder();
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(q_slice);
            builder.arg(k_slice);
            builder.arg(v_slice);
            builder.arg(bt_slice);
            builder.arg(sl_slice);
            builder.arg(&self.scale);
            builder.arg(&num_heads_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&max_blocks_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&block_size_i32);
            builder.arg(&max_partitions_i32);

            // SAFETY: kernel launch with validated parameters and contiguous buffers
            unsafe { builder.launch(v2_cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("paged_attention_v2 launch: {e}")))?;
        }

        // Stage 2: reduce kernel
        let reduce_func =
            dev.get_or_load_custom_func(T::KERNEL_V2_REDUCE, "paged_attention", PTX)?;

        // Reduce shared memory: max_logits[max_num_partitions] + exp_sums[max_num_partitions] + warp_reduce[NUM_WARPS]
        let reduce_shared_bytes =
            ((2 * max_num_partitions + NUM_WARPS) * std::mem::size_of::<f32>()) as u32;

        let reduce_cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: reduce_shared_bytes,
        };

        {
            let mut builder = reduce_func.builder();
            builder.arg(&output_slice);
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(sl_slice);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&max_partitions_i32);

            // SAFETY: reduce kernel launch with validated partition outputs
            unsafe { builder.launch(reduce_cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("paged_attention_v2 reduce: {e}")))?;
        }

        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v2 decode kernel for long sequences.
///
/// Uses split-K partitioning: the sequence is divided into partitions of 512
/// tokens, each processed by an independent thread block. A reduce kernel
/// merges partitions using numerically stable log-sum-exp.
///
/// Preferred over V1 when `max_seq_len > 512` to avoid shared memory pressure
/// and improve occupancy.
///
/// # Arguments
/// Same as [`paged_attention_cuda`].
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda(
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
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;
    let max_num_partitions = max_seq_len.div_ceil(PARTITION_SIZE);

    let seq_lens_c = seq_lens.contiguous()?;
    debug_assert_seq_lens_within_bound(&seq_lens_c, max_seq_len)?;

    let op = PagedAttnV2Op {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens_c,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        max_num_partitions,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// Auto-selecting paged attention: uses V1 for short sequences, V2 for long.
///
/// The threshold is 512 tokens (one partition). Below this, V1 has lower
/// overhead. Above, V2 benefits from split-K parallelism and bounded shared
/// memory.
///
/// This is the recommended entry point for model code.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto(
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
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    if max_seq_len > V2_SEQ_LEN_THRESHOLD {
        paged_attention_v2_cuda(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
        )
    } else {
        paged_attention_cuda(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
        )
    }
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
    head_dim: usize,
    block_size: usize,
}

impl CustomOp1 for PagedAttnAlibiOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1_alibi"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1_alibi requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v1_alibi::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v1_alibi::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v1_alibi expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnAlibiOp {
    fn run_v1_alibi<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v1_alibi: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v1_alibi: K cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v1_alibi: V cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
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

        let head_dim = self.head_dim;
        let elem_count = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(elem_count)?;

        let func = dev.get_or_load_custom_func(T::KERNEL_V1_ALIBI, "paged_attention", PTX)?;

        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;

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
        builder.arg(&head_dim_i32);
        builder.arg(&block_size_i32);
        builder.arg(alibi_slice);

        // SAFETY: kernel launch with validated parameters and contiguous buffers
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("paged_attention alibi launch: {e}")))?;

        drop(alibi_guard);
        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
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
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
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
    head_dim: usize,
    block_size: usize,
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
        head_dim,
        block_size,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// V2 paged attention with ALiBi support.
struct PagedAttnV2AlibiOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    alibi_slopes: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    /// Same contract as in `PagedAttnV2Op::max_seq_len`.
    #[allow(dead_code)]
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    max_num_partitions: usize,
}

impl CustomOp1 for PagedAttnV2AlibiOp {
    fn name(&self) -> &'static str {
        "paged_attention_v2_alibi"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v2_alibi requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention_v2_alibi: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v2_alibi::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v2_alibi::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v2_alibi expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnV2AlibiOp {
    fn run_v2_alibi<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v2_alibi: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2_alibi: K cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2_alibi: V cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("V cache must be on CUDA"),
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

        let head_dim = self.head_dim;
        let max_num_partitions = self.max_num_partitions;

        let tmp_out_size = num_seqs * self.num_heads * max_num_partitions * head_dim;
        let meta_size = num_seqs * self.num_heads * max_num_partitions;
        let tmp_out_slice = dev.alloc_zeros::<f32>(tmp_out_size)?;
        let exp_sums_slice = dev.alloc_zeros::<f32>(meta_size)?;
        let max_logits_slice = dev.alloc_zeros::<f32>(meta_size)?;

        let out_size = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(out_size)?;

        // Stage 1: partitioned attention with ALiBi
        let v2_func = dev.get_or_load_custom_func(T::KERNEL_V2_ALIBI, "paged_attention", PTX)?;

        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + PARTITION_SIZE) * std::mem::size_of::<f32>()) as u32;

        let v2_cfg = LaunchConfig {
            grid_dim: (
                self.num_heads as u32,
                num_seqs as u32,
                max_num_partitions as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;
        let max_partitions_i32 = max_num_partitions as i32;

        {
            let mut builder = v2_func.builder();
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(q_slice);
            builder.arg(k_slice);
            builder.arg(v_slice);
            builder.arg(bt_slice);
            builder.arg(sl_slice);
            builder.arg(&self.scale);
            builder.arg(&num_heads_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&max_blocks_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&block_size_i32);
            builder.arg(&max_partitions_i32);
            builder.arg(alibi_slice);

            // SAFETY: kernel launch with validated parameters
            unsafe { builder.launch(v2_cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("paged_attention_v2_alibi: {e}")))?;
        }

        // Stage 2: reduce (same kernel family as non-ALiBi — reduction is ALiBi-agnostic).
        let reduce_func =
            dev.get_or_load_custom_func(T::KERNEL_V2_REDUCE, "paged_attention", PTX)?;

        let reduce_shared_bytes =
            ((2 * max_num_partitions + NUM_WARPS) * std::mem::size_of::<f32>()) as u32;

        let reduce_cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: reduce_shared_bytes,
        };

        {
            let mut builder = reduce_func.builder();
            builder.arg(&output_slice);
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(sl_slice);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&max_partitions_i32);

            // SAFETY: reduce kernel launch
            unsafe { builder.launch(reduce_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("paged_attention_v2_alibi reduce: {e}"))
            })?;
        }

        drop(alibi_guard);
        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Auto-selecting paged attention with ALiBi: V1 for short, V2 for long.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto_alibi(
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
    head_dim: usize,
    block_size: usize,
    alibi_slopes: &Tensor,
) -> Result<Tensor> {
    if max_seq_len > V2_SEQ_LEN_THRESHOLD {
        let q = q.contiguous()?;
        let num_seqs = q.dim(0)?;
        let max_num_partitions = max_seq_len.div_ceil(PARTITION_SIZE);

        let alibi_slopes = alibi_slopes
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;

        let seq_lens_c = seq_lens.contiguous()?;
        debug_assert_seq_lens_within_bound(&seq_lens_c, max_seq_len)?;

        let op = PagedAttnV2AlibiOp {
            k_cache: k_cache.clone(),
            v_cache: v_cache.clone(),
            block_tables: block_tables.contiguous()?,
            seq_lens: seq_lens_c,
            alibi_slopes,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            max_num_partitions,
        };

        let output = q.apply_op1_no_bwd(&op)?;
        output.reshape((num_seqs, num_heads * head_dim))
    } else {
        paged_attention_cuda_alibi(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            alibi_slopes,
        )
    }
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
                // SAFETY: SwiGLU writes every output element; uninit safe.
                let output_slice = unsafe { dev.alloc::<half::bf16>(elem_count) }?;

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
                // SAFETY: SwiGLU writes every output element; uninit safe.
                let output_slice = unsafe { dev.alloc::<half::f16>(elem_count) }?;

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

// ============================================================================
// RMSNorm CUDA Kernel
// ============================================================================

#[cfg(feature = "cuda-layernorm")]
const LAYERNORM_PTX: &str = include_str!("../kernels/layernorm.ptx");

/// Fused RMSNorm operation: output = (input / sqrt(mean(input^2) + eps)) * weight
///
/// Uses vectorized loads for BF16/FP16 (4 elements per thread per iteration)
/// and warp+block-level reduction for the variance computation.
#[cfg(feature = "cuda-layernorm")]
struct RmsNormOp {
    weight: Tensor,
    epsilon: f32,
}

#[cfg(feature = "cuda-layernorm")]
impl CustomOp1 for RmsNormOp {
    fn name(&self) -> &'static str {
        "rms_norm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("rms_norm CUDA kernel does not support CPU")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::{bf16, f16};

        let dev = &input_storage.device;
        let input_shape = input_layout.shape();
        let dims = input_shape.dims();

        if dims.is_empty() {
            candle_core::bail!("rms_norm: input must have at least 1 dimension");
        }

        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        if num_tokens == 0 {
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                input_shape.clone(),
            ));
        }

        let (weight_storage, _weight_layout) = self.weight.storage_and_layout();
        let weight_storage = match &*weight_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rms_norm: weight must be on CUDA device"),
        };

        // Block size: threads per block. One block per token.
        // For large num_tokens, smaller blocks give better SM utilization.
        let max_block_size: u32 = if num_tokens < 256 { 1024 } else { 256 };
        let block_size = std::cmp::min(hidden_size as u32, max_block_size);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = hidden_size as i32;

        match (&input_storage.slice, &weight_storage.slice) {
            (CudaStorageSlice::BF16(input_slice), CudaStorageSlice::BF16(weight_slice)) => {
                // SAFETY: RMSNorm fills every output element via the
                // per-token block; uninit safe.
                let output_slice = unsafe { dev.alloc::<bf16>(num_tokens * hidden_size) }?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_bf16", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            (CudaStorageSlice::F16(input_slice), CudaStorageSlice::F16(weight_slice)) => {
                let output_slice = dev.alloc_zeros::<f16>(num_tokens * hidden_size)?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_fp16", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F16(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            (CudaStorageSlice::F32(input_slice), CudaStorageSlice::F32(weight_slice)) => {
                let output_slice = dev.alloc_zeros::<f32>(num_tokens * hidden_size)?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_f32", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F32(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            _ => candle_core::bail!("rms_norm: unsupported dtype combination"),
        }
    }
}

/// CUDA-accelerated RMSNorm.
///
/// Applies RMSNorm using a vectorized CUDA kernel that fuses the variance
/// computation, normalization, and weight scaling into a single pass.
///
/// # Arguments
/// - `input`: Input tensor `[..., hidden_size]`, contiguous in last dim
/// - `weight`: Weight tensor `[hidden_size]`
/// - `epsilon`: Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Returns
/// Normalized tensor with same shape and dtype as input
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    let input = input.contiguous()?;
    let op = RmsNormOp {
        weight: weight.clone(),
        epsilon,
    };
    input.apply_op1_no_bwd(&op)
}

/// Check if CUDA RMSNorm kernel is available.
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-layernorm"))]
pub fn rms_norm_cuda_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// Fused RoPE CUDA Kernel
// ============================================================================

#[cfg(feature = "cuda-kernels")]
const ROPE_PTX: &str = include_str!("../kernels/rope.ptx");

/// Apply fused RoPE to a single tensor (Q or K) using a CUDA kernel.
///
/// Uses CustomOp2: primary operand is the tensor to transform, secondary is
/// a packed {positions, cos_sin_cache} tensor. The kernel modifies data in-place
/// on a freshly allocated copy, avoiding mutation of the input tensor.
///
/// # Arguments
/// - `tensor`: Query or Key tensor `[num_tokens, num_heads * head_size]`
/// - `positions`: Position indices `[num_tokens]` i32
/// - `cos_sin_cache`: Precomputed `[max_position, rot_dim]` f32 where each row is
///   `[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`
/// - `rot_dim`: Rotary dimension (= 2 * half_dim)
/// - `head_size`: Full head dimension
/// - `num_heads`: Number of heads for this tensor
/// - `is_neox`: true for NeoX-style (split halves), false for GPT-J (interleaved)
#[cfg(feature = "cuda-kernels")]
struct RopeOp {
    positions: Tensor,
    cos_sin_cache: Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    is_neox: bool,
}

#[cfg(feature = "cuda-kernels")]
impl CustomOp1 for RopeOp {
    fn name(&self) -> &'static str {
        "rope_fused"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("rope_fused CUDA kernel does not support CPU")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &input_storage.device;
        let shape = input_layout.shape();
        let num_tokens = shape.dims()[0];

        if num_tokens == 0 {
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                shape.clone(),
            ));
        }

        let total_elems: usize = shape.elem_count();
        let query_stride = (self.num_heads * self.head_size) as i32;
        let rot_dim_i32 = self.rot_dim as i32;
        let head_size_i32 = self.head_size as i32;
        let num_heads_i32 = self.num_heads as i32;
        // key is null — we process one tensor at a time
        let num_kv_heads_i32 = 0i32;
        let key_stride = 0i32;

        let block_dim = std::cmp::min(self.num_heads * self.rot_dim / 2, 512) as u32;
        let block_dim = std::cmp::max(block_dim, 1);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel_name = if self.is_neox {
            "rotary_embedding_neox_bf16"
        } else {
            "rotary_embedding_gptj_bf16"
        };

        // Get positions storage. The kernel reads each entry as `int`
        // (4 bytes); for non-negative position indices ≤ 2^31 the bit
        // pattern of u32 and i32 is identical, so the U32 path hands the
        // device pointer to the kernel directly — avoiding a synchronous
        // device→host→device round-trip on every RoPE call (which also
        // blocked CUDA-graph capture).
        //
        // I64 positions still need a one-pass conversion; we materialise
        // an i32 buffer on host. This path is currently unreachable from
        // `apply_varlen` (which always builds U32) but kept as a defensive
        // fallback. A one-shot warn tags any future regression.
        let (pos_storage, _pos_layout) = self.positions.storage_and_layout();
        let pos_storage = match &*pos_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope: positions must be on CUDA"),
        };
        let pos_i32_owned: Option<candle_core::cuda::cudarc::driver::CudaSlice<i32>> =
            match &pos_storage.slice {
                CudaStorageSlice::U32(_) => None,
                CudaStorageSlice::I64(_) => {
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::warn!(
                            target: "vllm_core::rope",
                            "RoPE positions arrived as I64 — host-pull conversion path active. \
                             Caller should pass U32 to avoid the per-call host sync."
                        );
                    });
                    let pos_vec: Vec<i32> = self
                        .positions
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&v| v as i32)
                        .collect();
                    Some(dev.memcpy_stod(&pos_vec)?)
                }
                _ => candle_core::bail!(
                    "rope: positions must be i64 or u32, got {:?}",
                    self.positions.dtype()
                ),
            };

        // Get cos_sin_cache storage
        let (cache_storage, _cache_layout) = self.cos_sin_cache.storage_and_layout();
        let cache_storage = match &*cache_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope: cos_sin_cache must be on CUDA"),
        };
        let cache_slice = match &cache_storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("rope: cos_sin_cache must be f32"),
        };

        match &input_storage.slice {
            CudaStorageSlice::BF16(input_slice) => {
                // Copy input to output buffer (kernel modifies in-place).
                // The buffer is fully overwritten by `memcpy_dtod` before
                // the kernel reads it; the implicit zero-init in
                // `alloc_zeros` was dead work — switch to uninit alloc.
                // SAFETY: every byte is initialised by `memcpy_dtod` prior
                // to any read.
                let mut output_slice = unsafe { dev.alloc::<bf16>(total_elems) }?;
                dev.memcpy_dtod(input_slice, &mut output_slice)?;

                let func = dev.get_or_load_custom_func(kernel_name, "rope", ROPE_PTX)?;

                let mut builder = func.builder();
                // Positions arg: U32 source goes directly (kernel reads as `int`,
                // bit pattern matches for non-negative indices); I64 source uses
                // the host-converted I32 slice.
                if let CudaStorageSlice::U32(s) = &pos_storage.slice {
                    builder.arg(s);
                } else if let Some(s) = pos_i32_owned.as_ref() {
                    builder.arg(s);
                } else {
                    candle_core::bail!("rope: positions dtype unsupported (must be U32 or I64)");
                }
                builder.arg(&output_slice); // query (modified in-place)
                builder.arg(&0u64); // key = nullptr
                builder.arg(cache_slice); // cos_sin_cache
                builder.arg(&rot_dim_i32);
                builder.arg(&query_stride);
                builder.arg(&key_stride); // unused since key=null
                builder.arg(&head_size_i32);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32); // 0

                // SAFETY: kernel launch with validated params, output_slice is freshly allocated
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    shape.clone(),
                ))
            }
            _ => candle_core::bail!("rope_fused: only bf16 supported"),
        }
    }
}

/// Apply fused RoPE CUDA kernel to Q and K tensors.
///
/// Each tensor is processed with a single kernel launch (2 launches total).
/// The kernel replaces ~6 candle ops (split, mul, sub, mul, add, cat) per tensor.
///
/// # Arguments
/// - `q`: Query `[num_tokens, num_heads, head_size]` bf16
/// - `k`: Key `[num_tokens, num_kv_heads, head_size]` bf16
/// - `positions`: Position for each token `[num_tokens]`
/// - `cos_sin_cache`: `[max_position, rot_dim]` f32 — interleaved cos/sin
/// - `rot_dim`: Rotary dimension (= 2 * half_dim)
/// - `head_size`: Full head dimension
/// - `num_heads`: Number of Q heads
/// - `num_kv_heads`: Number of KV heads
/// - `is_neox`: NeoX vs GPT-J style
#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn rotary_embedding_cuda(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    cos_sin_cache: &Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    let num_tokens = positions.dim(0)?;
    if num_tokens == 0 {
        return Ok((q.clone(), k.clone()));
    }

    // Flatten [num_tokens, heads, head_size] → [num_tokens, heads * head_size]
    let q_flat = q.reshape((num_tokens, num_heads * head_size))?;
    let k_flat = k.reshape((num_tokens, num_kv_heads * head_size))?;

    let q_op = RopeOp {
        positions: positions.clone(),
        cos_sin_cache: cos_sin_cache.clone(),
        rot_dim,
        head_size,
        num_heads,
        is_neox,
    };
    let q_out = q_flat.contiguous()?.apply_op1_no_bwd(&q_op)?;

    let k_op = RopeOp {
        positions: positions.clone(),
        cos_sin_cache: cos_sin_cache.clone(),
        rot_dim,
        head_size,
        num_heads: num_kv_heads,
        is_neox,
    };
    let k_out = k_flat.contiguous()?.apply_op1_no_bwd(&k_op)?;

    // Reshape back to [num_tokens, heads, head_size]
    let q_out = q_out.reshape(q.shape())?;
    let k_out = k_out.reshape(k.shape())?;

    Ok((q_out, k_out))
}

/// Check if CUDA RoPE kernel is available.
#[cfg(feature = "cuda-kernels")]
pub fn rotary_embedding_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn rotary_embedding_cuda_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// GELU/GeGLU Activation CUDA Kernels
// ============================================================================

#[cfg(feature = "cuda-fused-activations")]
const ACTIVATIONS_PTX: &str = include_str!("../kernels/activations.ptx");

/// GELU activation variant.
#[cfg(feature = "cuda-fused-activations")]
#[derive(Debug, Clone, Copy)]
pub enum GeluVariant {
    /// Exact GELU using erf
    Exact,
    /// Tanh approximation (faster)
    Tanh,
}

/// Fused gated activation: output = act(gate) * up
///
/// Input is `[..., 2 * d]` where first half is gate and second half is up.
/// Output is `[..., d]`.
///
/// # Arguments
/// - `input`: Concatenated gate+up tensor `[..., 2 * d]`
/// - `variant`: Which GELU variant to use
///
/// # Returns
/// Activated tensor `[..., d]`
#[cfg(feature = "cuda-fused-activations")]
pub struct GeluAndMulOp {
    variant: GeluVariant,
}

#[cfg(feature = "cuda-fused-activations")]
impl CustomOp1 for GeluAndMulOp {
    fn name(&self) -> &'static str {
        "gelu_and_mul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let shape = layout.shape();
        let dims = shape.dims();
        if dims.is_empty() {
            candle_core::bail!("gelu_and_mul: input must have at least 1 dimension");
        }
        let last_dim = dims[dims.len() - 1];
        if !last_dim.is_multiple_of(2) {
            candle_core::bail!("gelu_and_mul: last dimension must be even, got {last_dim}");
        }
        let d = last_dim / 2;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = d;
        let out_shape = Shape::from_dims(&out_dims);

        match storage {
            CpuStorage::F32(data) => {
                let offsets = layout.contiguous_offsets();
                let (start, _end) = offsets.ok_or_else(|| {
                    candle_core::Error::Msg("gelu_and_mul: input must be contiguous".to_string())
                })?;
                let data = &data[start..];
                let mut result = Vec::with_capacity(num_tokens * d);
                for t in 0..num_tokens {
                    let gate = &data[t * last_dim..t * last_dim + d];
                    let up = &data[t * last_dim + d..t * last_dim + last_dim];
                    for i in 0..d {
                        let g = gate[i];
                        let u = up[i];
                        let act = match self.variant {
                            GeluVariant::Exact => {
                                g * 0.5 * (1.0 + libm::erff(g * std::f32::consts::FRAC_1_SQRT_2))
                            }
                            GeluVariant::Tanh => {
                                let beta = (2.0f32 / std::f32::consts::PI).sqrt();
                                let inner = beta * (g + 0.044715 * g * g * g);
                                0.5 * g * (1.0 + inner.tanh())
                            }
                        };
                        result.push(act * u);
                    }
                }
                Ok((CpuStorage::F32(result), out_shape))
            }
            _ => candle_core::bail!("gelu_and_mul: CPU fallback only supports F32"),
        }
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::{bf16, f16};

        let dev = &input_storage.device;
        let shape = input_layout.shape();
        let dims = shape.dims();

        let last_dim = dims[dims.len() - 1];
        let d = last_dim / 2;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        if num_tokens == 0 {
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = d;
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                Shape::from_dims(&out_dims),
            ));
        }

        let d_i32 = d as i32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (std::cmp::min(d as u32, 1024), 1, 1),
            shared_mem_bytes: 0,
        };

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = d;
        let out_shape = Shape::from_dims(&out_dims);

        let suffix = match self.variant {
            GeluVariant::Exact => "gelu_and_mul",
            GeluVariant::Tanh => "gelu_tanh_and_mul",
        };

        match &input_storage.slice {
            CudaStorageSlice::BF16(input_slice) => {
                // SAFETY: gelu/silu kernel writes every output element.
                let output_slice = unsafe { dev.alloc::<bf16>(num_tokens * d) }?;
                let func_name = format!("{suffix}_bf16");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            CudaStorageSlice::F16(input_slice) => {
                let output_slice = dev.alloc_zeros::<f16>(num_tokens * d)?;
                let func_name = format!("{suffix}_fp16");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F16(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            CudaStorageSlice::F32(input_slice) => {
                let output_slice = dev.alloc_zeros::<f32>(num_tokens * d)?;
                let func_name = format!("{suffix}_f32");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F32(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            _ => candle_core::bail!("gelu_and_mul: unsupported dtype"),
        }
    }
}

/// Fused GELU-gated activation: output = gelu(gate) * up
///
/// Input is concatenated `[gate, up]` along the last dimension.
///
/// # Arguments
/// - `input`: Tensor `[..., 2 * d]` where first half is gate, second is up
/// - `variant`: GELU variant (exact erf or tanh approximation)
///
/// # Returns
/// Tensor `[..., d]`
#[cfg(feature = "cuda-fused-activations")]
pub fn gelu_and_mul(input: &Tensor, variant: GeluVariant) -> Result<Tensor> {
    let input = input.contiguous()?;
    input.apply_op1_no_bwd(&GeluAndMulOp { variant })
}

/// Check if CUDA GELU activation kernels are available.
#[cfg(feature = "cuda-fused-activations")]
pub fn gelu_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-fused-activations"))]
pub fn gelu_cuda_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// reshape_and_cache CUDA Kernels
// ============================================================================

#[cfg(feature = "cuda-kernels")]
const CACHE_OPS_PTX: &str = include_str!("../kernels/cache_ops.ptx");

/// Cache layout for CUDA kernel dispatch.
#[cfg(feature = "cuda-kernels")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaCacheLayout {
    /// `[num_blocks, block_size, num_kv_heads, head_dim]`
    NHD,
    /// `[num_blocks, num_kv_heads, block_size, head_dim]`
    HND,
}

/// Write K/V from model output into paged KV cache using a CUDA kernel.
///
/// This replaces the Candle scatter path with a direct kernel launch,
/// avoiding intermediate tensor allocations for index expansion.
///
/// # Arguments
/// - `key`: Input key tensor `[num_tokens, num_kv_heads, head_dim]`, contiguous
/// - `value`: Input value tensor `[num_tokens, num_kv_heads, head_dim]`, contiguous
/// - `key_cache`: KV cache key tensor (modified in-place)
/// - `value_cache`: KV cache value tensor (modified in-place)
/// - `slot_mapping`: Physical slot per token `[num_tokens]` (i32, on CUDA)
/// - `num_kv_heads`: Number of KV heads
/// - `head_dim`: Head dimension
/// - `block_size`: Cache block size
/// - `layout`: Cache layout (NHD or HND)
#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_cuda(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    layout: CudaCacheLayout,
) -> Result<()> {
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

    let num_tokens = key.dim(0)?;
    if num_tokens == 0 {
        return Ok(());
    }

    let (key_guard, _) = key.storage_and_layout();
    let (value_guard, _) = value.storage_and_layout();
    let (kc_guard, _) = key_cache.storage_and_layout();
    let (vc_guard, _) = value_cache.storage_and_layout();
    let (sm_guard, _) = slot_mapping.storage_and_layout();

    let dev = match &*key_guard {
        Storage::Cuda(cs) => &cs.device,
        _ => candle_core::bail!("reshape_and_cache: key must be on CUDA"),
    };

    let sm_slice = match &*sm_guard {
        Storage::Cuda(cs) => match &cs.slice {
            CudaStorageSlice::I64(_) => {
                candle_core::bail!("reshape_and_cache: slot_mapping must be U32, got I64")
            }
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("reshape_and_cache: slot_mapping must be U32"),
        },
        _ => candle_core::bail!("reshape_and_cache: slot_mapping must be on CUDA"),
    };

    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let block_size_i32 = block_size as i32;

    // Select kernel name and grid based on layout and dtype
    match layout {
        CudaCacheLayout::NHD => {
            let kv_stride = num_kv_heads * head_dim;
            let block_dim = std::cmp::min(kv_stride, 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            match &*key_guard {
                Storage::Cuda(key_cs) => match (&key_cs.slice, &*value_guard) {
                    (CudaStorageSlice::BF16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::BF16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_bf16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        // SAFETY: all pointers validated, contiguous, same device
                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache NHD launch: {e}"))
                        })?;
                    }
                    (CudaStorageSlice::F16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::F16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_fp16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache NHD launch: {e}"))
                        })?;
                    }
                    _ => candle_core::bail!(
                        "reshape_and_cache: unsupported dtype, expected BF16 or F16"
                    ),
                },
                _ => candle_core::bail!("reshape_and_cache: key not on CUDA"),
            }
        }
        CudaCacheLayout::HND => {
            let block_dim = std::cmp::min(head_dim, 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, num_kv_heads as u32, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            match &*key_guard {
                Storage::Cuda(key_cs) => match (&key_cs.slice, &*value_guard) {
                    (CudaStorageSlice::BF16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::BF16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_hnd_bf16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        // SAFETY: all pointers validated, contiguous, same device
                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache HND launch: {e}"))
                        })?;
                    }
                    (CudaStorageSlice::F16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::F16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_hnd_fp16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache HND launch: {e}"))
                        })?;
                    }
                    _ => candle_core::bail!(
                        "reshape_and_cache: unsupported dtype, expected BF16 or F16"
                    ),
                },
                _ => candle_core::bail!("reshape_and_cache: key not on CUDA"),
            }
        }
    }

    drop(key_guard);
    drop(value_guard);
    drop(kc_guard);
    drop(vc_guard);
    drop(sm_guard);

    Ok(())
}

/// Check if reshape_and_cache CUDA kernel is available.
#[cfg(feature = "cuda-kernels")]
pub fn reshape_and_cache_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn reshape_and_cache_cuda_available(_tensor: &Tensor) -> bool {
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

    /// Paged-attention F16 vs BF16 parity: identical inputs cast to either
    /// dtype must produce outputs that match within F16 mantissa tolerance.
    /// Covers V1 (short seq), V2 (long seq, partitioned), and the auto path.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attention_f16_bf16_parity() {
        let Ok(dev) = Device::new_cuda(0) else { return };

        let num_seqs = 1usize;
        let num_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 128usize;
        let block_size = 16usize;
        let seq_len = 16usize; // one full block, fits V1
        let max_blocks_per_seq = 1usize;

        // Random-ish but deterministic input data.
        let q_f32: Vec<f32> = (0..num_seqs * num_heads * head_dim)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5)
            .collect();
        let k_f32: Vec<f32> = (0..max_blocks_per_seq * block_size * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..max_blocks_per_seq * block_size * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_heads, head_dim), &dev).unwrap();
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let block_tables =
            Tensor::from_vec(vec![0u32], (num_seqs, max_blocks_per_seq), &dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, &dev).unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // BF16 path
        let q_bf16 = q.to_dtype(DType::BF16).unwrap();
        let k_bf16 = k_cache.to_dtype(DType::BF16).unwrap();
        let v_bf16 = v_cache.to_dtype(DType::BF16).unwrap();
        let out_bf16 = paged_attention_cuda(
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &block_tables,
            &seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        // F16 path
        let q_f16 = q.to_dtype(DType::F16).unwrap();
        let k_f16 = k_cache.to_dtype(DType::F16).unwrap();
        let v_f16 = v_cache.to_dtype(DType::F16).unwrap();
        let out_f16 = paged_attention_cuda(
            &q_f16,
            &k_f16,
            &v_f16,
            &block_tables,
            &seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        let bf16_vec: Vec<f32> = out_bf16.flatten_all().unwrap().to_vec1().unwrap();
        let f16_vec: Vec<f32> = out_f16.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(bf16_vec.len(), f16_vec.len());

        // F16 has 10 mantissa bits, BF16 has 7 — different rounding paths.
        // Tolerance: 5e-2 absolute, matches values <0.5 in our synthetic input.
        let max_diff = bf16_vec
            .iter()
            .zip(f16_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 5e-2,
            "paged_attention v1 f16/bf16 max diff {max_diff} exceeds 5e-2"
        );
    }

    /// V2 split-K parity: long sequence forces multi-partition reduce path.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attention_v2_f16_bf16_parity() {
        let Ok(dev) = Device::new_cuda(0) else { return };

        let num_seqs = 1usize;
        let num_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 128usize;
        let block_size = 16usize;
        // > V2_SEQ_LEN_THRESHOLD (512) and > PARTITION_SIZE (512) → exercises reduce.
        let seq_len = 1024usize;
        let max_blocks_per_seq = seq_len / block_size; // 64 blocks

        let q_f32: Vec<f32> = (0..num_seqs * num_heads * head_dim)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5)
            .collect();
        let kv_elements = max_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_heads, head_dim), &dev).unwrap();
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let bt: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
        let block_tables = Tensor::from_vec(bt, (num_seqs, max_blocks_per_seq), &dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, &dev).unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        let run = |dtype: DType| {
            paged_attention_auto(
                &q.to_dtype(dtype).unwrap(),
                &k_cache.to_dtype(dtype).unwrap(),
                &v_cache.to_dtype(dtype).unwrap(),
                &block_tables,
                &seq_lens,
                scale,
                num_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };

        let bf16_vec = run(DType::BF16);
        let f16_vec = run(DType::F16);
        assert_eq!(bf16_vec.len(), f16_vec.len());

        let max_diff = bf16_vec
            .iter()
            .zip(f16_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // V2 sums across partitions accumulate more rounding drift than V1.
        assert!(
            max_diff < 1e-1,
            "paged_attention v2 f16/bf16 max diff {max_diff} exceeds 1e-1"
        );
    }
}
