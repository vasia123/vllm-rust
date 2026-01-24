use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

const PTX: &str = include_str!("../kernels/paged_attention.ptx");
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

        let func = dev.get_or_load_custom_func(
            "paged_attention_v1_bf16",
            "paged_attention",
            PTX,
        )?;

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
