//! CUDA kernel for per-row gather + dequantize of quantized embedding
//! tables (the GGUF `QuantizedEmbedding` GPU path).
//!
//! Lets a huge quantized embedding table — Gemma 4's Per-Layer Embedding,
//! [262144, 42*256] Q6_K (~5.6 GB dense) — stay quantized-resident on the
//! GPU while a forward materializes only the rows selected by the batch's
//! token ids. The whole table is never dequantized. See
//! `kernels/gather_dequant.cu`.

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Storage, Tensor,
};

const GATHER_DEQUANT_PTX: &str = include_str!("../../kernels/gather_dequant.ptx");

/// `CustomOp1` over the quantized table bytes (U8) producing the gathered
/// dense rows `[num_ids, embedding_dim]` (F32).
struct GatherDequantQ6kOp {
    /// Row indices to gather, U32 on the same CUDA device, contiguous.
    ids: Tensor,
    num_ids: usize,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl CustomOp1 for GatherDequantQ6kOp {
    fn name(&self) -> &'static str {
        "gather_dequant_q6k"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gather_dequant_q6k requires CUDA (CPU path uses QTensor gather)")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        const QK_K: usize = 256;
        let dev = &storage.device;

        let table = match &storage.slice {
            CudaStorageSlice::U8(s) => s,
            _ => candle_core::bail!("gather_dequant_q6k: table must be U8 block bytes"),
        };

        if !self.embedding_dim.is_multiple_of(QK_K) {
            candle_core::bail!(
                "gather_dequant_q6k: embedding_dim {} not a multiple of {QK_K}",
                self.embedding_dim
            );
        }
        let blocks_per_row = self.embedding_dim / QK_K;

        let (ids_guard, _) = self.ids.storage_and_layout();
        let ids_slice = match &*ids_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("gather_dequant_q6k: ids must be U32"),
            },
            _ => candle_core::bail!("gather_dequant_q6k: ids must be on CUDA"),
        };

        let elem_count = self.num_ids * self.embedding_dim;
        let output = dev.alloc_zeros::<f32>(elem_count)?;

        let func = dev.get_or_load_custom_func(
            "gather_dequant_q6k_f32",
            "gather_dequant",
            GATHER_DEQUANT_PTX,
        )?;

        // One CUDA block per (gathered row × block-in-row); 64 threads each.
        let total_blocks = (self.num_ids * blocks_per_row) as u32;
        let cfg = LaunchConfig {
            grid_dim: (total_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        let num_ids_i32 = self.num_ids as i32;
        let blocks_per_row_i32 = blocks_per_row as i32;
        let num_embeddings_i32 = self.num_embeddings as i32;

        let mut builder = func.builder();
        builder.arg(table);
        builder.arg(ids_slice);
        builder.arg(&output);
        builder.arg(&num_ids_i32);
        builder.arg(&blocks_per_row_i32);
        builder.arg(&num_embeddings_i32);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("gather_dequant_q6k launch: {e}")))?;

        drop(ids_guard);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::F32(output),
            device: dev.clone(),
        };
        Ok((
            output_storage,
            Shape::from_dims(&[self.num_ids, self.embedding_dim]),
        ))
    }
}

/// Gather + dequantize Q6_K embedding rows on the GPU.
///
/// * `table` — raw Q6_K block bytes of the table, U8 on CUDA, row-major
///   over `[num_embeddings, embedding_dim]`.
/// * `ids` — row indices (any integer dtype) on CUDA.
///
/// Returns the gathered rows `[num_ids, embedding_dim]` in F32. The
/// caller casts to the compute dtype.
pub fn gather_dequant_q6k(
    table: &Tensor,
    ids: &Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
) -> Result<Tensor> {
    let ids = ids.flatten_all()?.to_dtype(DType::U32)?.contiguous()?;
    let num_ids = ids.elem_count();
    table.apply_op1(GatherDequantQ6kOp {
        ids,
        num_ids,
        num_embeddings,
        embedding_dim,
    })
}

#[cfg(test)]
mod tests {
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};
    use std::sync::Arc;

    /// The GPU Q6_K gather kernel must match the bit-exact CPU QTensor
    /// gather (which is itself validated against full dequant). This is the
    /// correctness gate for `kernels/gather_dequant.cu`.
    #[test]
    fn gpu_gather_q6k_matches_cpu() {
        let cuda = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            _ => {
                eprintln!("skip gpu_gather_q6k_matches_cpu: no CUDA device");
                return;
            }
        };
        let cpu = Device::Cpu;
        let (num, dim) = (512usize, 2560usize); // dim % 256 == 0
        let table = Tensor::randn(0.0f32, 1.0, (num, dim), &cpu).unwrap();
        let qt = Arc::new(QTensor::quantize(&table, GgmlDType::Q6K).unwrap());

        // CPU reference (bit-exact block-slice gather + dequant).
        let cpu_emb =
            crate::quantization::QuantizedEmbedding::new(Arc::clone(&qt), DType::F32, cpu.clone())
                .unwrap();

        // GPU: upload the raw Q6_K block bytes and gather via the kernel.
        let bytes = qt.data().unwrap().to_vec();
        let nb = bytes.len();
        let table_gpu = Tensor::from_vec(bytes, (nb,), &cuda).unwrap();
        let gpu_emb = crate::quantization::QuantizedEmbedding::new_gpu_q6k(
            table_gpu,
            num,
            dim,
            DType::F32,
            cuda.clone(),
        )
        .unwrap();

        let ids = Tensor::from_vec(vec![0u32, 5, 100, 511, 5, 1], (6,), &cpu).unwrap();
        let r_cpu = cpu_emb.forward(&ids).unwrap();
        let r_gpu = gpu_emb
            .forward(&ids.to_device(&cuda).unwrap())
            .unwrap()
            .to_device(&cpu)
            .unwrap();

        let diff = (r_cpu - r_gpu)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "GPU Q6_K gather diverged from CPU reference: {diff}"
        );
    }
}
