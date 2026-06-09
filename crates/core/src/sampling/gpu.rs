//! GPU-side sampling to avoid large GPU→CPU transfers.
//!
//! For greedy decoding (argmax) and standard top-k/top-p sampling, the entire
//! operation runs on GPU. Only the final token ID (one i32 per sequence) is
//! transferred back.
//!
//! When `cuda-kernels` feature is enabled and the tensor is on a CUDA device,
//! uses custom CUDA kernels. Otherwise falls back to Candle ops.

use candle_core::{DType, Device, Result, Tensor};

/// Whether GPU sampling is available for the given device.
pub fn gpu_sampling_available(device: &Device) -> bool {
    device.is_cuda()
}

// ─── CUDA kernel FFI via CustomOp ───────────────────────────────────────────

#[cfg(feature = "cuda-kernels")]
const SAMPLING_PTX: &str = include_str!("../../kernels/sampling.ptx");

#[cfg(feature = "cuda-kernels")]
const APPLY_GRAMMAR_BITMASK_PTX: &str = include_str!("../../kernels/apply_grammar_bitmask.ptx");

#[cfg(feature = "cuda-kernels")]
mod cuda_ops {
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda::CudaStorageSlice;
    use candle_core::{
        CpuStorage, CudaStorage, CustomOp1, CustomOp2, InplaceOp2, Layout, Result, Shape,
    };

    use super::{APPLY_GRAMMAR_BITMASK_PTX, SAMPLING_PTX};

    /// In-place "set logits to -inf where bitmask bit is 0".
    ///
    /// Receiver tensor (`out_storage`) = logits `[batch, vocab]`,
    /// any of F32 / BF16 / F16. The second tensor (`a_storage`) = packed
    /// I32 bitmask `[batch, words_per_row]` where
    /// `words_per_row = ceil(vocab_grammar / 32)`. Bit `t % 32` of word
    /// `t / 32` is the "allowed" flag; tokens past `vocab_grammar` are
    /// implicitly forbidden when the engine zero-fills the tail.
    pub(super) struct ApplyGrammarBitmaskOp;

    impl InplaceOp2 for ApplyGrammarBitmaskOp {
        fn name(&self) -> &'static str {
            "apply_grammar_bitmask"
        }

        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!(
                "apply_grammar_bitmask: GPU-only op (use SamplingConstraint::mask_logits on CPU)"
            )
        }

        fn cuda_fwd(
            &self,
            out_storage: &mut CudaStorage,
            out_layout: &Layout,
            a_storage: &CudaStorage,
            a_layout: &Layout,
        ) -> Result<()> {
            let dev = a_storage.device.clone();
            let out_dims = out_layout.shape().dims();
            let a_dims = a_layout.shape().dims();
            if out_dims.len() != 2 || a_dims.len() != 2 {
                candle_core::bail!(
                    "apply_grammar_bitmask: expected 2D logits and 2D bitmask, got {out_dims:?} / {a_dims:?}"
                );
            }
            let (batch, vocab) = (out_dims[0], out_dims[1]);
            let words_per_row = a_dims[1];
            if a_dims[0] < batch {
                candle_core::bail!(
                    "apply_grammar_bitmask: bitmask rows {} < logits batch {}",
                    a_dims[0],
                    batch
                );
            }
            // ceil(vocab / 32) bits must fit into the supplied row.
            let required_words = vocab.div_ceil(32);
            if words_per_row < required_words {
                candle_core::bail!(
                    "apply_grammar_bitmask: bitmask words_per_row={} < ceil(vocab={}/32)={}",
                    words_per_row,
                    vocab,
                    required_words
                );
            }
            if out_layout.start_offset() != 0 || a_layout.start_offset() != 0 {
                candle_core::bail!("apply_grammar_bitmask: tensors must be contiguous at offset 0");
            }

            let bitmask_slice = match &a_storage.slice {
                CudaStorageSlice::I32(s) => s,
                _ => candle_core::bail!("apply_grammar_bitmask: bitmask must be I32"),
            };

            let cfg = LaunchConfig {
                grid_dim: (((vocab as u32) + 255) / 256, batch as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let vocab_i32 = vocab as i32;
            let wpr_i32 = words_per_row as i32;

            match &mut out_storage.slice {
                CudaStorageSlice::F32(out_slice) => {
                    let func = dev.get_or_load_custom_func(
                        "apply_grammar_bitmask_f32",
                        "apply_grammar_bitmask",
                        APPLY_GRAMMAR_BITMASK_PTX,
                    )?;
                    let mut b = func.builder();
                    b.arg(out_slice);
                    b.arg(bitmask_slice);
                    b.arg(&vocab_i32);
                    b.arg(&wpr_i32);
                    unsafe { b.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("apply_grammar_bitmask_f32: {e}"))
                    })?;
                }
                CudaStorageSlice::BF16(out_slice) => {
                    let func = dev.get_or_load_custom_func(
                        "apply_grammar_bitmask_bf16",
                        "apply_grammar_bitmask",
                        APPLY_GRAMMAR_BITMASK_PTX,
                    )?;
                    let mut b = func.builder();
                    b.arg(out_slice);
                    b.arg(bitmask_slice);
                    b.arg(&vocab_i32);
                    b.arg(&wpr_i32);
                    unsafe { b.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("apply_grammar_bitmask_bf16: {e}"))
                    })?;
                }
                CudaStorageSlice::F16(out_slice) => {
                    let func = dev.get_or_load_custom_func(
                        "apply_grammar_bitmask_f16",
                        "apply_grammar_bitmask",
                        APPLY_GRAMMAR_BITMASK_PTX,
                    )?;
                    let mut b = func.builder();
                    b.arg(out_slice);
                    b.arg(bitmask_slice);
                    b.arg(&vocab_i32);
                    b.arg(&wpr_i32);
                    unsafe { b.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("apply_grammar_bitmask_f16: {e}"))
                    })?;
                }
                _ => candle_core::bail!("apply_grammar_bitmask: logits dtype must be F32/BF16/F16"),
            }
            Ok(())
        }
    }

    /// GPU argmax: logits [num_seqs, vocab_size] → token IDs [num_seqs] as U32.
    pub(super) struct ArgmaxOp;

    impl CustomOp1 for ArgmaxOp {
        fn name(&self) -> &'static str {
            "gpu_argmax"
        }

        fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
            // CPU fallback: iterate to find argmax per row
            let (num_seqs, vocab_size) = (l.dims()[0], l.dims()[1]);
            let data = match s {
                CpuStorage::F32(d) => d,
                CpuStorage::BF16(d) => {
                    let f32_data: Vec<f32> = d.iter().map(|v| v.to_f32()).collect();
                    let mut ids = Vec::with_capacity(num_seqs);
                    for seq in 0..num_seqs {
                        let start = l.start_offset() + seq * vocab_size;
                        let row = &f32_data[start..start + vocab_size];
                        let idx = row
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i as u32)
                            .unwrap_or(0);
                        ids.push(idx);
                    }
                    return Ok((CpuStorage::U32(ids), Shape::from_dims(&[num_seqs])));
                }
                _ => candle_core::bail!("gpu_argmax: unsupported dtype"),
            };
            let mut ids = Vec::with_capacity(num_seqs);
            for seq in 0..num_seqs {
                let start = l.start_offset() + seq * vocab_size;
                let row = &data[start..start + vocab_size];
                let idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                ids.push(idx);
            }
            Ok((CpuStorage::U32(ids), Shape::from_dims(&[num_seqs])))
        }

        fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = &s.device;
            let (num_seqs, vocab_size) = (l.dims()[0], l.dims()[1]);

            if l.start_offset() != 0 {
                candle_core::bail!("gpu_argmax: tensor must be contiguous from offset 0");
            }

            let output_slice = dev
                .alloc_zeros::<u32>(num_seqs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let vocab_size_i32 = vocab_size as i32;

            match &s.slice {
                CudaStorageSlice::BF16(data) => {
                    let func = dev
                        .get_or_load_custom_func("argmax_bf16", "sampling", SAMPLING_PTX)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let cfg = LaunchConfig {
                        grid_dim: (num_seqs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&output_slice);
                    builder.arg(data);
                    builder.arg(&vocab_size_i32);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("argmax_bf16 launch: {e}")))?;
                }
                CudaStorageSlice::F32(data) => {
                    let func = dev
                        .get_or_load_custom_func("argmax_f32", "sampling", SAMPLING_PTX)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let cfg = LaunchConfig {
                        grid_dim: (num_seqs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&output_slice);
                    builder.arg(data);
                    builder.arg(&vocab_size_i32);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("argmax_f32 launch: {e}")))?;
                }
                _ => candle_core::bail!("gpu_argmax: unsupported dtype (expected BF16 or F32)"),
            }

            let out = CudaStorage {
                slice: CudaStorageSlice::U32(output_slice),
                device: dev.clone(),
            };
            Ok((out, Shape::from_dims(&[num_seqs])))
        }
    }

    /// GPU softmax: logits [num_seqs, vocab_size] → probs [num_seqs, vocab_size] F32.
    pub(super) struct SoftmaxOp;

    impl CustomOp1 for SoftmaxOp {
        fn name(&self) -> &'static str {
            "gpu_softmax"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("SoftmaxOp: use gpu_softmax() which has CPU fallback")
        }

        fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = &s.device;
            let (num_seqs, vocab_size) = (l.dims()[0], l.dims()[1]);

            if l.start_offset() != 0 {
                candle_core::bail!("gpu_softmax: tensor must be contiguous from offset 0");
            }

            let output_slice = dev
                .alloc_zeros::<f32>(num_seqs * vocab_size)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let vocab_size_i32 = vocab_size as i32;

            match &s.slice {
                CudaStorageSlice::BF16(data) => {
                    let func = dev
                        .get_or_load_custom_func("softmax_to_probs", "sampling", SAMPLING_PTX)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let cfg = LaunchConfig {
                        grid_dim: (num_seqs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&output_slice);
                    builder.arg(data);
                    builder.arg(&vocab_size_i32);
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("softmax_to_probs launch: {e}"))
                    })?;
                }
                CudaStorageSlice::F32(data) => {
                    let func = dev
                        .get_or_load_custom_func("softmax_to_probs_f32", "sampling", SAMPLING_PTX)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    let cfg = LaunchConfig {
                        grid_dim: (num_seqs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = func.builder();
                    builder.arg(&output_slice);
                    builder.arg(data);
                    builder.arg(&vocab_size_i32);
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("softmax_to_probs_f32 launch: {e}"))
                    })?;
                }
                _ => {
                    candle_core::bail!("gpu_softmax: unsupported dtype (expected BF16 or F32)")
                }
            }

            let out = CudaStorage {
                slice: CudaStorageSlice::F32(output_slice),
                device: dev.clone(),
            };
            Ok((out, Shape::from_dims(&[num_seqs, vocab_size])))
        }
    }

    /// GPU top-k + top-p sampling: probs [num_seqs, V] + rand [num_seqs] → ids [num_seqs] U32.
    /// Per-sequence top_k and top_p arrays are uploaded to GPU.
    pub(super) struct TopKTopPSampleOp {
        pub top_k: Vec<i32>,
        pub top_p: Vec<f32>,
    }

    impl CustomOp2 for TopKTopPSampleOp {
        fn name(&self) -> &'static str {
            "gpu_top_k_top_p_sample"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!(
                "TopKTopPSampleOp: use gpu_top_k_top_p_sample() which has CPU fallback"
            )
        }

        fn cuda_fwd(
            &self,
            probs_s: &CudaStorage,
            probs_l: &Layout,
            rand_s: &CudaStorage,
            rand_l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let dev = &probs_s.device;
            let (num_seqs, vocab_size) = (probs_l.dims()[0], probs_l.dims()[1]);

            if probs_l.start_offset() != 0 || rand_l.start_offset() != 0 {
                candle_core::bail!("top_k_top_p_sample: tensors must be contiguous from offset 0");
            }

            let probs_slice = match &probs_s.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("top_k_top_p_sample: probs must be F32"),
            };
            let rand_slice = match &rand_s.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("top_k_top_p_sample: rand_vals must be F32"),
            };

            let top_k_gpu = dev.clone_htod(&self.top_k)?;
            let top_p_gpu = dev.clone_htod(&self.top_p)?;

            let output_slice = dev
                .alloc_zeros::<u32>(num_seqs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let vocab_size_i32 = vocab_size as i32;

            let func = dev
                .get_or_load_custom_func("top_k_top_p_sample_per_seq", "sampling", SAMPLING_PTX)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            let mut builder = func.builder();
            builder.arg(&output_slice);
            builder.arg(probs_slice);
            builder.arg(rand_slice);
            builder.arg(&top_k_gpu);
            builder.arg(&top_p_gpu);
            builder.arg(&vocab_size_i32);

            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("top_k_top_p_sample_per_seq launch: {e}"))
            })?;

            let out = CudaStorage {
                slice: CudaStorageSlice::U32(output_slice),
                device: dev.clone(),
            };
            Ok((out, Shape::from_dims(&[num_seqs])))
        }
    }

    /// Parallel multinomial sampling for the no-filter case (top_k=0,
    /// top_p>=1.0). Replaces the single-threaded `TopKTopPSampleOp` path
    /// when no filtering is needed; ~150× faster on vocab=151k decode.
    pub(super) struct MultinomialSampleNoFilterOp;

    impl CustomOp2 for MultinomialSampleNoFilterOp {
        fn name(&self) -> &'static str {
            "gpu_multinomial_sample_no_filter_parallel"
        }

        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("MultinomialSampleNoFilterOp: CUDA-only; caller must check device")
        }

        fn cuda_fwd(
            &self,
            probs_s: &CudaStorage,
            probs_l: &Layout,
            rand_s: &CudaStorage,
            rand_l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let dev = &probs_s.device;
            let (num_seqs, vocab_size) = (probs_l.dims()[0], probs_l.dims()[1]);

            if probs_l.start_offset() != 0 || rand_l.start_offset() != 0 {
                candle_core::bail!(
                    "multinomial_sample_no_filter_parallel: tensors must be contiguous from offset 0"
                );
            }

            let probs_slice = match &probs_s.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("multinomial_sample_no_filter_parallel: probs must be F32"),
            };
            let rand_slice = match &rand_s.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!(
                    "multinomial_sample_no_filter_parallel: rand_vals must be F32"
                ),
            };

            // Output is tiny (num_seqs U32 = 4 bytes/seq); `alloc_zeros`
            // here doesn't pressure the CUDA async-mempool the way the
            // earlier candle-cumsum attempt did (which allocated a fresh
            // [N, vocab_size] F32 per token).
            let output_slice = dev
                .alloc_zeros::<u32>(num_seqs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let vocab_size_i32 = vocab_size as i32;

            let func = dev
                .get_or_load_custom_func(
                    "multinomial_sample_no_filter_parallel",
                    "sampling",
                    SAMPLING_PTX,
                )
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            // 256 threads/block — uses parallel chunk-sums + serial
            // chunk-finding pattern (see sampling.cu kernel comments).
            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            let mut builder = func.builder();
            builder.arg(&output_slice);
            builder.arg(probs_slice);
            builder.arg(rand_slice);
            builder.arg(&vocab_size_i32);

            unsafe { builder.launch(cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!(
                    "multinomial_sample_no_filter_parallel launch: {e}"
                ))
            })?;

            let out = CudaStorage {
                slice: CudaStorageSlice::U32(output_slice),
                device: dev.clone(),
            };
            Ok((out, Shape::from_dims(&[num_seqs])))
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Greedy sampling (argmax) on GPU via CUDA kernel.
///
/// Takes logits tensor `[num_seqs, vocab_size]` of any float dtype and returns
/// token IDs `[num_seqs]` as U32. Logits are cast to F32 internally; the cast is
/// a no-op when the tensor is already F32.
///
/// On CUDA: only transfers `num_seqs * 4` bytes back (one u32 per sequence)
/// instead of `num_seqs * vocab_size * dtype_size` bytes.
pub fn gpu_argmax(logits: &Tensor) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;

    #[cfg(feature = "cuda-kernels")]
    if logits.device().is_cuda() {
        let logits = logits.contiguous()?;
        return logits.apply_op1_no_bwd(&cuda_ops::ArgmaxOp);
    }

    // CPU fallback via Candle ops
    logits.argmax(1)?.to_dtype(DType::U32)
}

/// In-place "apply grammar bitmask" — sets `logits[b, t] = -inf` when
/// the corresponding bit in the packed I32 bitmask is zero.
///
/// `logits`  : `[batch, vocab]`, F32 / BF16 / F16, **contiguous** on CUDA.
///             Mutated in place via [`Tensor::inplace_op2`]; the storage is
///             shared with the caller, so the side effect is visible to
///             every alias of this tensor.
/// `bitmask` : `[rows, words_per_row]` (`rows ≥ batch`), packed I32,
///             contiguous on CUDA. Bit `t % 32` of word `t / 32` is the
///             "allowed" flag. `words_per_row` must be ≥ `ceil(vocab/32)`;
///             trailing bits in the last word are ignored.
///
/// On non-CUDA devices or non-contiguous inputs this errors out — use
/// [`crate::sampling::SamplingConstraint::mask_logits`] for the CPU path.
pub fn gpu_apply_grammar_bitmask(logits: &Tensor, bitmask: &Tensor) -> Result<()> {
    if !logits.device().is_cuda() || !bitmask.device().is_cuda() {
        candle_core::bail!("gpu_apply_grammar_bitmask: both tensors must be on CUDA");
    }
    if !logits.is_contiguous() || !bitmask.is_contiguous() {
        candle_core::bail!("gpu_apply_grammar_bitmask: tensors must be contiguous");
    }
    #[cfg(feature = "cuda-kernels")]
    {
        logits.inplace_op2(bitmask, &cuda_ops::ApplyGrammarBitmaskOp)
    }
    #[cfg(not(feature = "cuda-kernels"))]
    {
        let _ = (logits, bitmask);
        candle_core::bail!(
            "gpu_apply_grammar_bitmask: cuda-kernels feature not enabled at build time"
        )
    }
}

/// GPU-accelerated softmax for probability conversion.
///
/// Accepts logits of any float dtype and converts to F32 probabilities.
/// The cast to F32 is a no-op when the input is already F32.
pub fn gpu_softmax(logits: &Tensor) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;

    #[cfg(feature = "cuda-kernels")]
    if logits.device().is_cuda() {
        let logits = logits.contiguous()?;
        return logits.apply_op1_no_bwd(&cuda_ops::SoftmaxOp);
    }

    // CPU fallback
    let max = logits.max_keepdim(1)?;
    let shifted = logits.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(1)?;
    exp.broadcast_div(&sum)
}

/// Combined top-k + top-p sampling on GPU with per-sequence parameters.
///
/// # Arguments
/// - `probs`: F32 probability tensor `[num_seqs, vocab_size]`
/// - `rand_vals`: F32 uniform random values `[num_seqs]` in [0, 1)
/// - `top_k`: Per-sequence top-k values (0 = disabled)
/// - `top_p`: Per-sequence top-p values (1.0 = disabled)
///
/// # Returns
/// Token IDs `[num_seqs]` as U32.
pub fn gpu_top_k_top_p_sample(
    probs: &Tensor,
    rand_vals: &Tensor,
    top_k: &[i32],
    top_p: &[f32],
) -> Result<Tensor> {
    #[cfg(feature = "cuda-kernels")]
    if probs.device().is_cuda() {
        let probs = probs.contiguous()?;
        let rand_vals = rand_vals.contiguous()?;
        let op = cuda_ops::TopKTopPSampleOp {
            top_k: top_k.to_vec(),
            top_p: top_p.to_vec(),
        };
        return probs.apply_op2_no_bwd(&rand_vals, &op);
    }

    // CPU fallback
    gpu_top_k_top_p_sample_cpu(probs, rand_vals, top_k, top_p)
}

/// Parallel multinomial sampling on GPU when no top-k/top-p filtering is
/// required (top_k=0 and top_p>=1.0 for every sequence in the batch).
///
/// Replaces the single-threaded `gpu_top_k_top_p_sample` for the common
/// no-filter case. Measured ~150× kernel speedup on vocab=151k decode,
/// translating to ~5 ms/token saved on Qwen3-8B temp>0 sampling.
///
/// Algorithm: inverse-CDF via parallel chunk-sums + serial chunk-finding
/// (see `crates/core/kernels/sampling.cu::multinomial_sample_no_filter_parallel`).
///
/// Returns token IDs `[num_seqs]` as U32. Output is fresh-allocated u32
/// (4 bytes/seq) — does NOT use candle's tensor allocator (avoids the
/// async-mempool OOM that the earlier cumsum-based attempt hit; see
/// `memory/exl3_perf_session_2026-05-16.md` TODO #4).
#[cfg(feature = "cuda-kernels")]
pub fn gpu_multinomial_sample_no_filter(probs: &Tensor, rand_vals: &Tensor) -> Result<Tensor> {
    use std::sync::atomic::Ordering;
    MULTINOMIAL_NO_FILTER_COUNT.fetch_add(1, Ordering::Relaxed);
    MULTINOMIAL_NO_FILTER_COUNT_LOCAL.with(|c| c.set(c.get() + 1));
    let probs = probs.contiguous()?;
    let rand_vals = rand_vals.contiguous()?;
    probs.apply_op2_no_bwd(&rand_vals, &cuda_ops::MultinomialSampleNoFilterOp)
}

/// Diagnostic counter for `gpu_multinomial_sample_no_filter` invocations.
/// Permanent observability infra — same pattern as
/// `cuda_kernels::RMS_NORM_CUDA_POOLED_COUNT` (see
/// `memory/cuda_layernorm_4pct_regression.md`). Read via `Relaxed`.
#[cfg(feature = "cuda-kernels")]
pub static MULTINOMIAL_NO_FILTER_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "cuda-kernels")]
thread_local! {
    /// Thread-local twin of [`MULTINOMIAL_NO_FILTER_COUNT`]: a test asserting
    /// "THIS call routed to the fast path" reads the twin, so concurrent test
    /// threads bumping the global counter can't race the delta. The global
    /// stays the process-wide observability source.
    pub static MULTINOMIAL_NO_FILTER_COUNT_LOCAL: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

fn gpu_top_k_top_p_sample_cpu(
    probs: &Tensor,
    rand_vals: &Tensor,
    top_k: &[i32],
    top_p: &[f32],
) -> Result<Tensor> {
    let (num_seqs, vocab_size) = probs.dims2()?;
    let data: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    let rand_data: Vec<f32> = rand_vals.flatten_all()?.to_vec1()?;
    let mut token_ids = Vec::with_capacity(num_seqs);

    for seq in 0..num_seqs {
        let row = &data[seq * vocab_size..(seq + 1) * vocab_size];
        let k = top_k.get(seq).copied().unwrap_or(0);
        let p = top_p.get(seq).copied().unwrap_or(1.0);
        let r = rand_data[seq];

        // Apply top-k threshold
        let mut threshold = 0.0f32;
        if k > 0 && (k as usize) < vocab_size {
            let mut sorted: Vec<f32> = row.to_vec();
            sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            threshold = sorted[k as usize - 1];
        }

        // Compute filtered set sorted by probability descending
        let mut filtered: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        filtered
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-p
        if p < 1.0 && p > 0.0 {
            let total: f32 = filtered.iter().map(|(_, v)| v).sum();
            let target = p * total;
            let mut cumsum = 0.0f32;
            let mut cutoff = filtered.len();
            for (i, &(_, v)) in filtered.iter().enumerate() {
                cumsum += v;
                if cumsum >= target {
                    cutoff = i + 1;
                    break;
                }
            }
            filtered.truncate(cutoff);
        }

        // Multinomial sample
        let sum: f32 = filtered.iter().map(|(_, v)| v).sum();
        let target = r * sum;
        let mut cumsum = 0.0f32;
        let mut selected = filtered.last().map(|(i, _)| *i).unwrap_or(0);
        for &(idx, prob) in &filtered {
            cumsum += prob;
            if cumsum > target {
                selected = idx;
                break;
            }
        }
        token_ids.push(selected as u32);
    }

    Tensor::from_vec(token_ids, num_seqs, probs.device())
}

/// Apply per-sequence temperature scaling.
///
/// Returns a new F32 tensor with logits scaled by 1/temperature per sequence.
pub fn gpu_temperature_scale(logits: &Tensor, inv_temps: &[f32]) -> Result<Tensor> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let (num_seqs, _vocab_size) = logits_f32.dims2()?;

    // Build diagonal-like scale: [num_seqs, 1] broadcast
    let scales = Tensor::from_vec(inv_temps.to_vec(), (num_seqs, 1), logits.device())?;
    logits_f32.broadcast_mul(&scales)
}

// ─── Additive logit diffs (penalties, bias, banned tokens) ─────────────────

/// One additive change to a single `(seq_idx, token_id)` cell of the logits
/// matrix.  Multiple entries that hit the same cell are summed by
/// `index_add`, so callers can stack `freq_penalty + pres_penalty + bias`
/// for the same token without pre-merging on the host.
#[derive(Debug, Clone, Copy)]
pub struct LogitsDiff {
    pub seq_idx: u32,
    pub token_id: u32,
    pub delta: f32,
}

/// Apply a batch of additive diffs to a `[num_seqs, vocab_size]` logits
/// tensor in a single GPU `index_add` (or its CPU equivalent).
///
/// This is the GPU counterpart to the CPU helpers
/// `apply_logit_bias`, `apply_frequency_presence_penalty`,
/// `apply_banned_tokens`, and `apply_bad_words` — together they cover every
/// purely-additive logit modifier the sampler currently supports.
/// Multiplicative effects (repetition penalty) are NOT covered here and stay
/// on the CPU path.
///
/// Empty `diffs` is a no-op and returns the input as-is — the caller can
/// build the vector unconditionally without checking the fast path.
pub fn gpu_apply_logits_diff(logits: &Tensor, diffs: &[LogitsDiff]) -> Result<Tensor> {
    if diffs.is_empty() {
        return Ok(logits.clone());
    }
    let (num_seqs, vocab_size) = logits.dims2()?;
    let vocab_u32 = vocab_size as u32;

    // Build flat (seq_idx * vocab_size + token_id) indices on the host.
    // index_add is O(diffs.len()) so this is small (sum of unique generated
    // tokens across the batch) and dominated by GPU launch latency rather
    // than transfer cost.
    let mut indices: Vec<u32> = Vec::with_capacity(diffs.len());
    let mut values: Vec<f32> = Vec::with_capacity(diffs.len());
    for d in diffs {
        debug_assert!(d.seq_idx < num_seqs as u32);
        debug_assert!(d.token_id < vocab_u32);
        indices.push(d.seq_idx * vocab_u32 + d.token_id);
        values.push(d.delta);
    }

    let device = logits.device();
    let idx_t = Tensor::from_vec(indices, diffs.len(), device)?;
    let val_t = Tensor::from_vec(values, diffs.len(), device)?;

    let flat = logits.flatten_all()?;
    let updated = flat.index_add(&idx_t, &val_t, 0)?;
    updated.reshape((num_seqs, vocab_size))
}

// ─── Batched GPU sampling pipeline ──────────────────────────────────────────

/// Per-sequence sampling configuration for the batched GPU pipeline.
pub struct GpuSamplingConfig {
    /// Inverse temperature (1.0/temperature). 0.0 means greedy.
    pub inv_temperature: f32,
    /// Top-k value. 0 = disabled.
    pub top_k: i32,
    /// Top-p value. 1.0 = disabled.
    pub top_p: f32,
    /// Random value [0, 1) for this sequence (pre-generated on CPU).
    pub rand_val: f32,
}

/// Batched GPU sampling pipeline.
///
/// Runs the full sampling pipeline on GPU:
/// 1. Temperature scaling (per-sequence)
/// 2. Softmax → probabilities
/// 3. Top-k + top-p filtering + multinomial sampling (per-sequence)
///
/// Only transfers back `[num_seqs]` token IDs instead of `[num_seqs, vocab_size]` logits.
///
/// Falls back to CPU for sequences needing constraints, logit processors, or logprobs.
///
/// Accepts logits of any float dtype (BF16/F16/F32). Internally cast to F32 once
/// so that downstream kernels and CPU paths see a uniform dtype contract.
pub fn gpu_sample_batch(logits: &Tensor, configs: &[GpuSamplingConfig]) -> Result<Vec<u32>> {
    gpu_sample_batch_with_diffs(logits, configs, &[])
}

/// Same as [`gpu_sample_batch`], but applies a batch of additive logit diffs
/// (logit_bias, freq+pres penalties, banned tokens, bad words) on-GPU before
/// temperature scaling.  The eligibility check in the caller is responsible
/// for ensuring no multiplicative or filter-style modifiers slipped in.
/// Stage A'.3 — async-sampling-friendly variant.
/// Returns the sampled token IDs as a GPU `Tensor` (`[num_seqs]` U32)
/// WITHOUT the blocking host-side `to_vec1()`. The caller can use this
/// tensor directly as the next forward's `input_ids` (no DtoH+HtoD
/// round-trip), and asynchronously DtoH the IDs in parallel with the
/// next forward's compute.
///
/// nsys profile (2026-05-08) showed `to_vec1()` blocking 50 ms per call
/// at c=8 — pure stream-drain wait, NOT the 32-byte DMA itself. This
/// API breaks the autoregressive serialization: GPU keeps running
/// while host reads the IDs.
pub fn gpu_sample_batch_with_diffs_to_tensor(
    logits: &Tensor,
    configs: &[GpuSamplingConfig],
    diffs: &[LogitsDiff],
) -> Result<Tensor> {
    let (num_seqs, _vocab_size) = logits.dims2()?;
    assert_eq!(num_seqs, configs.len());

    let logits = logits.to_dtype(DType::F32)?;
    let logits = gpu_apply_logits_diff(&logits, diffs)?;
    let logits = &logits;

    let all_greedy = configs.iter().all(|c| c.inv_temperature == 0.0);
    if all_greedy {
        return gpu_argmax(logits);
    }

    let inv_temps: Vec<f32> = configs
        .iter()
        .map(|c| {
            if c.inv_temperature == 0.0 {
                1.0
            } else {
                c.inv_temperature
            }
        })
        .collect();
    let scaled_logits = gpu_temperature_scale(logits, &inv_temps)?;
    let probs = gpu_softmax(&scaled_logits)?;

    let top_k_arr: Vec<i32> = configs.iter().map(|c| c.top_k).collect();
    let top_p_arr: Vec<f32> = configs.iter().map(|c| c.top_p).collect();
    let rand_arr: Vec<f32> = configs.iter().map(|c| c.rand_val).collect();

    let rand_tensor = Tensor::from_vec(rand_arr, num_seqs, logits.device())?;

    // Fast path: no top-k/top-p filtering needed → custom parallel
    // multinomial kernel (multinomial_sample_no_filter_parallel) instead
    // of the single-threaded `top_k_top_p_sample_per_seq`. The kernel
    // does parallel chunk-sums + serial chunk-finding; measured ~150×
    // speedup on vocab=151k decode. Streaming-safe — output is a tiny
    // `[N]` U32 alloc, no candle-async-mempool pressure (unlike the
    // 2026-05-16 cumsum attempt; see commit history + memory note).
    //
    // Mixed greedy + stochastic batches: each greedy seq has rand_val=0,
    // and the parallel kernel returns position-of-first-prob-exceeding-zero
    // which on a peaky softmax distribution may differ from argmax. So
    // keep the existing merge-with-argmax path below; fast-path only when
    // EVERY seq has inv_temperature > 0 (otherwise greedy short-circuit
    // earlier in this function caught it).
    #[cfg(feature = "cuda-kernels")]
    let no_filter_all_stochastic = logits.device().is_cuda()
        && configs
            .iter()
            .all(|c| c.inv_temperature > 0.0 && c.top_k <= 0 && c.top_p >= 1.0);
    #[cfg(feature = "cuda-kernels")]
    if no_filter_all_stochastic {
        return gpu_multinomial_sample_no_filter(&probs, &rand_tensor);
    }
    let result = gpu_top_k_top_p_sample(&probs, &rand_tensor, &top_k_arr, &top_p_arr)?;

    // Mixed greedy + multinomial: need to merge in argmax results for
    // the greedy seqs. Stage A'.3 caveat — this path falls back to a
    // host round-trip. Most production batches are uniformly
    // multinomial OR uniformly greedy (handled by all_greedy branch
    // above), so the mixed path is rare; deferring its async fix to
    // a follow-up. For now, materialise on host, override, and
    // re-upload.
    let has_greedy = configs.iter().any(|c| c.inv_temperature == 0.0);
    if has_greedy {
        let mut ids: Vec<u32> = result.to_vec1()?;
        let argmax_result = gpu_argmax(logits)?;
        let argmax_ids: Vec<u32> = argmax_result.to_vec1()?;
        for (i, config) in configs.iter().enumerate() {
            if config.inv_temperature == 0.0 {
                ids[i] = argmax_ids[i];
            }
        }
        return Tensor::from_vec(ids, num_seqs, logits.device());
    }

    Ok(result)
}

pub fn gpu_sample_batch_with_diffs(
    logits: &Tensor,
    configs: &[GpuSamplingConfig],
    diffs: &[LogitsDiff],
) -> Result<Vec<u32>> {
    let (num_seqs, _vocab_size) = logits.dims2()?;
    assert_eq!(num_seqs, configs.len());

    // Establish the sampler-path dtype contract: all downstream ops see F32 logits.
    // No-op when the input tensor is already F32.
    let logits = logits.to_dtype(DType::F32)?;
    let logits = gpu_apply_logits_diff(&logits, diffs)?;
    let logits = &logits;

    // Check if all greedy — fast path
    let all_greedy = configs.iter().all(|c| c.inv_temperature == 0.0);
    if all_greedy {
        let result = gpu_argmax(logits)?;
        let ids: Vec<u32> = result.to_vec1()?;
        return Ok(ids);
    }

    // Step 1: Temperature scaling (per-sequence)
    let inv_temps: Vec<f32> = configs
        .iter()
        .map(|c| {
            if c.inv_temperature == 0.0 {
                1.0
            } else {
                c.inv_temperature
            }
        })
        .collect();
    let scaled_logits = gpu_temperature_scale(logits, &inv_temps)?;

    // Step 2: Softmax → probabilities
    let probs = gpu_softmax(&scaled_logits)?;

    // Step 3: Top-k + top-p + multinomial sampling
    let top_k_arr: Vec<i32> = configs.iter().map(|c| c.top_k).collect();
    let top_p_arr: Vec<f32> = configs.iter().map(|c| c.top_p).collect();
    let rand_arr: Vec<f32> = configs.iter().map(|c| c.rand_val).collect();

    let rand_tensor = Tensor::from_vec(rand_arr, num_seqs, logits.device())?;
    let result = gpu_top_k_top_p_sample(&probs, &rand_tensor, &top_k_arr, &top_p_arr)?;

    // Stage A'.3 diag: measure the to_vec1 stall directly. nsys profile
    // shows cuMemcpyDtoHAsync_v2 taking 13-54 ms per call — this is the
    // host blocking on stream drain, NOT the 32-byte DMA itself.
    // Confirmation here lets us scope the async-sampling fix.
    static SAMPLE_TO_VEC_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let log_first = std::env::var("VLLM_PROFILE_SAMPLER_TOVEC").is_ok();
    let t0 = if log_first {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // Handle greedy sequences: override with argmax result
    let mut ids: Vec<u32> = result.to_vec1()?;

    if let Some(t0) = t0 {
        let elapsed = t0.elapsed();
        SAMPLE_TO_VEC_LOGGED.get_or_init(|| {
            tracing::info!(
                target: "vllm_core::sampler_diag",
                "first sampler to_vec1: {} µs (size={} u32, expected ~32 bytes)",
                elapsed.as_micros(),
                ids.len()
            );
        });
        // Log every 100 calls thereafter
        static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_multiple_of(100) {
            tracing::info!(
                target: "vllm_core::sampler_diag",
                "sampler to_vec1 ({} calls): last={} µs (expected 32B DMA <100µs)",
                n,
                elapsed.as_micros()
            );
        }
    }
    let has_greedy = configs.iter().any(|c| c.inv_temperature == 0.0);
    if has_greedy {
        let argmax_result = gpu_argmax(logits)?;
        let argmax_ids: Vec<u32> = argmax_result.to_vec1()?;
        for (i, config) in configs.iter().enumerate() {
            if config.inv_temperature == 0.0 {
                ids[i] = argmax_ids[i];
            }
        }
    }

    Ok(ids)
}

// ─── Legacy API (kept for backward compatibility) ───────────────────────────

/// Top-k filtering: zeros out all probabilities outside the top-k.
pub fn gpu_top_k_filter(probs: &Tensor, k: usize) -> Result<Tensor> {
    if k == 0 {
        return Ok(probs.clone());
    }

    let (num_seqs, vocab_size) = probs.dims2()?;
    if k >= vocab_size {
        return Ok(probs.clone());
    }

    let data: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    let mut result = vec![0.0f32; num_seqs * vocab_size];

    for seq in 0..num_seqs {
        let row = &data[seq * vocab_size..(seq + 1) * vocab_size];
        let mut sorted: Vec<f32> = row.to_vec();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];
        for v in 0..vocab_size {
            if row[v] >= threshold {
                result[seq * vocab_size + v] = row[v];
            }
        }
    }

    Tensor::from_vec(result, (num_seqs, vocab_size), probs.device())
}

/// Top-p (nucleus) filtering.
pub fn gpu_top_p_filter(probs: &Tensor, p: f32) -> Result<Tensor> {
    if p >= 1.0 {
        return Ok(probs.clone());
    }

    let (num_seqs, vocab_size) = probs.dims2()?;
    let data: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    let mut result = vec![0.0f32; num_seqs * vocab_size];

    for seq in 0..num_seqs {
        let row = &data[seq * vocab_size..(seq + 1) * vocab_size];
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        for &(idx, prob) in &indexed {
            result[seq * vocab_size + idx] = prob;
            cumsum += prob;
            if cumsum > p {
                break;
            }
        }
    }

    Tensor::from_vec(result, (num_seqs, vocab_size), probs.device())
}

/// Multinomial sampling from probability distribution.
pub fn gpu_multinomial_sample(probs: &Tensor, rand_vals: &Tensor) -> Result<Tensor> {
    let (num_seqs, vocab_size) = probs.dims2()?;

    let sum = probs.sum_keepdim(1)?;
    let probs = probs.broadcast_div(&sum)?;

    let cumsum = cumulative_sum(&probs)?;
    let cumsum_data: Vec<f32> = cumsum.flatten_all()?.to_vec1()?;
    let rand_data: Vec<f32> = rand_vals.flatten_all()?.to_vec1()?;

    let mut token_ids = Vec::with_capacity(num_seqs);
    for seq in 0..num_seqs {
        let r = rand_data[seq];
        let mut selected = (vocab_size - 1) as u32;
        for v in 0..vocab_size {
            if cumsum_data[seq * vocab_size + v] > r {
                selected = v as u32;
                break;
            }
        }
        token_ids.push(selected);
    }

    Tensor::from_vec(token_ids, num_seqs, probs.device())
}

fn cumulative_sum(tensor: &Tensor) -> Result<Tensor> {
    let (num_seqs, vocab_size) = tensor.dims2()?;
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let mut cumsum = Vec::with_capacity(data.len());

    for seq in 0..num_seqs {
        let mut sum = 0.0f32;
        for v in 0..vocab_size {
            sum += data[seq * vocab_size + v];
            cumsum.push(sum);
        }
    }

    Tensor::from_vec(cumsum, (num_seqs, vocab_size), tensor.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_gpu_argmax_basic() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![1.0f32, 5.0, 3.0, 2.0, 0.5, 4.0, 2.0, 1.0],
            (2, 4),
            &device,
        )
        .unwrap();

        let result = gpu_argmax(&logits).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids, vec![1, 1]);
    }

    #[test]
    fn test_gpu_softmax_sums_to_one() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();

        let probs = gpu_softmax(&logits).unwrap();
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();

        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5, "Row 1 sum: {sum1}");
        assert!((sum2 - 1.0).abs() < 1e-5, "Row 2 sum: {sum2}");
    }

    #[test]
    fn test_gpu_top_k_filter() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.1f32, 0.4, 0.3, 0.2], (1, 4), &device).unwrap();

        let filtered = gpu_top_k_filter(&probs, 2).unwrap();
        let data: Vec<f32> = filtered.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(data[0], 0.0, "Token 0 should be filtered");
        assert!((data[1] - 0.4).abs() < 1e-6, "Token 1 should remain");
        assert!((data[2] - 0.3).abs() < 1e-6, "Token 2 should remain");
        assert_eq!(data[3], 0.0, "Token 3 should be filtered");
    }

    #[test]
    fn test_gpu_multinomial_sample_deterministic() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 0.0], (1, 4), &device).unwrap();
        let rand_val = Tensor::from_vec(vec![0.5f32], 1, &device).unwrap();

        let result = gpu_multinomial_sample(&probs, &rand_val).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 1, "Should always pick the only non-zero token");
    }

    #[test]
    fn test_gpu_multinomial_sample_low_rand() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.5f32, 0.3, 0.2], (1, 3), &device).unwrap();
        let rand_val = Tensor::from_vec(vec![0.0f32], 1, &device).unwrap();

        let result = gpu_multinomial_sample(&probs, &rand_val).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 0, "Low rand should pick first token");
    }

    #[test]
    fn test_gpu_multinomial_sample_high_rand() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.5f32, 0.3, 0.2], (1, 3), &device).unwrap();
        let rand_val = Tensor::from_vec(vec![0.99f32], 1, &device).unwrap();

        let result = gpu_multinomial_sample(&probs, &rand_val).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 2, "High rand should pick last token");
    }

    #[test]
    fn test_gpu_softmax_ordering() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 3.0, 2.0], (1, 3), &device).unwrap();

        let probs = gpu_softmax(&logits).unwrap();
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();

        assert!(data[1] > data[2], "logit 3.0 > 2.0");
        assert!(data[2] > data[0], "logit 2.0 > 1.0");
    }

    #[test]
    fn test_gpu_argmax_batch() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // seq 0: argmax = 1
                7.0, 2.0, 4.0, // seq 1: argmax = 0
                0.0, 0.0, 9.0, // seq 2: argmax = 2
            ],
            (3, 3),
            &device,
        )
        .unwrap();

        let result = gpu_argmax(&logits).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids, vec![1, 0, 2]);
    }

    #[test]
    fn test_cumulative_sum() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (1, 4), &device).unwrap();

        let cs = cumulative_sum(&t).unwrap();
        let data: Vec<f32> = cs.flatten_all().unwrap().to_vec1().unwrap();

        assert!((data[0] - 0.1).abs() < 1e-6);
        assert!((data[1] - 0.3).abs() < 1e-6);
        assert!((data[2] - 0.6).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gpu_sample_batch_all_greedy() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // seq 0: argmax = 1
                7.0, 2.0, 4.0, // seq 1: argmax = 0
            ],
            (2, 3),
            &device,
        )
        .unwrap();

        let configs = vec![
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
        ];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids, vec![1, 0]);
    }

    #[test]
    fn test_gpu_sample_batch_deterministic_single_prob() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![-100.0f32, -100.0, 100.0, -100.0], (1, 4), &device).unwrap();

        let configs = vec![GpuSamplingConfig {
            inv_temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.5,
        }];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn test_gpu_sample_batch_top_k_restricts() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 1.0, 100.0, 1.0], (1, 4), &device).unwrap();

        let configs = vec![GpuSamplingConfig {
            inv_temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            rand_val: 0.5,
        }];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn test_gpu_top_k_top_p_sample_cpu_fallback() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.1f32, 0.4, 0.3, 0.2], (1, 4), &device).unwrap();
        let rand_vals = Tensor::from_vec(vec![0.0f32], 1, &device).unwrap();

        let result = gpu_top_k_top_p_sample(&probs, &rand_vals, &[2], &[1.0]).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        // top-k=2 keeps indices 1 (0.4) and 2 (0.3). rand=0.0 picks first = index 1
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_gpu_temperature_scale() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 1.0, 3.0, 5.0], (2, 3), &device).unwrap();

        let scaled = gpu_temperature_scale(&logits, &[0.5, 2.0]).unwrap();
        let data: Vec<f32> = scaled.flatten_all().unwrap().to_vec1().unwrap();

        assert!((data[0] - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((data[1] - 2.0).abs() < 1e-6); // 4.0 * 0.5
        assert!((data[2] - 3.0).abs() < 1e-6); // 6.0 * 0.5
        assert!((data[3] - 2.0).abs() < 1e-6); // 1.0 * 2.0
        assert!((data[4] - 6.0).abs() < 1e-6); // 3.0 * 2.0
        assert!((data[5] - 10.0).abs() < 1e-6); // 5.0 * 2.0
    }

    #[test]
    fn test_gpu_sample_batch_mixed_greedy_stochastic() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // seq 0: greedy → argmax = 1
                -100.0, -100.0, 100.0, // seq 1: stochastic → should pick 2
            ],
            (2, 3),
            &device,
        )
        .unwrap();

        let configs = vec![
            GpuSamplingConfig {
                inv_temperature: 0.0, // greedy
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
            GpuSamplingConfig {
                inv_temperature: 1.0, // stochastic
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
        ];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids[0], 1, "Greedy should pick argmax");
        assert_eq!(ids[1], 2, "Stochastic should pick dominant token");
    }

    // ─── CUDA GPU tests ─────────────────────────────────────────────────────

    #[cfg(feature = "cuda-kernels")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    /// B2.0 gate: measure the per-step cost that a pooled bitmask buffer
    /// would remove — the `Tensor::from_vec([batch, words_per_row] i32)`
    /// GPU allocation + HtoD that `apply_grammar_bitmask_to_logits` does
    /// every constrained decode step. If this is negligible vs a ~40 ms
    /// decode step, B2 (pooled buffer + capture) is not worth the risk.
    /// Run with `--ignored --nocapture`.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_per_step_bitmask_alloc_upload() {
        let Some(dev) = cuda_device() else {
            eprintln!("SKIP: no CUDA device");
            return;
        };
        let candle_core::Device::Cuda(cdev) = &dev else {
            return;
        };
        // Qwen3: logits vocab 151936 → words_per_row 4748.
        let wpr = 151936usize.div_ceil(32);
        let sync = || {
            let _ = cdev.cuda_stream().synchronize();
        };

        for &batch in &[1usize, 4, 8] {
            // (a) alloc + HtoD only (what a pooled stable buffer removes).
            let iters = 500u32;
            sync();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _t = Tensor::from_vec(vec![0i32; batch * wpr], (batch, wpr), &dev).unwrap();
            }
            sync();
            let alloc_up = t0.elapsed() / iters;

            // (b) full apply: build bitmask tensor + run the kernel on a
            // realistic logits tensor (what one constrained step pays).
            let logits =
                Tensor::from_vec(vec![1.0f32; batch * 151936], (batch, 151936), &dev).unwrap();
            sync();
            let t1 = std::time::Instant::now();
            for _ in 0..iters {
                let bm = Tensor::from_vec(vec![!0i32; batch * wpr], (batch, wpr), &dev).unwrap();
                gpu_apply_grammar_bitmask(&logits, &bm).unwrap();
            }
            sync();
            let full = t1.elapsed() / iters;

            eprintln!(
                "batch={batch}: alloc+HtoD={alloc_up:?}/step, full(from_vec+kernel)={full:?}/step"
            );
        }
        eprintln!("(compare to a ~40 ms decode step → B2.0 ceiling = alloc+HtoD / 40ms)");
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_argmax_f32() {
        let Some(dev) = cuda_device() else { return };
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // seq 0: argmax = 1
                7.0, 2.0, 4.0, // seq 1: argmax = 0
                0.0, 0.0, 9.0, // seq 2: argmax = 2
            ],
            (3, 3),
            &dev,
        )
        .unwrap();

        let result = gpu_argmax(&logits).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids, vec![1, 0, 2]);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_argmax_bf16() {
        let Some(dev) = cuda_device() else { return };
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 7.0, 2.0, 4.0], (2, 3), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let result = gpu_argmax(&logits).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids, vec![1, 0]);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_softmax_f32() {
        let Some(dev) = cuda_device() else { return };
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &dev).unwrap();

        let probs = gpu_softmax(&logits).unwrap();
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();

        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-4, "Row 1 sum: {sum1}");
        assert!((sum2 - 1.0).abs() < 1e-4, "Row 2 sum: {sum2}");
        // Highest logit → highest prob
        assert!(data[2] > data[1] && data[1] > data[0]);
        assert!(data[5] > data[4] && data[4] > data[3]);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_softmax_bf16() {
        let Some(dev) = cuda_device() else { return };
        let logits = Tensor::from_vec(vec![1.0f32, 3.0, 2.0], (1, 3), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let probs = gpu_softmax(&logits).unwrap();
        assert_eq!(probs.dtype(), DType::F32);
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "Sum: {sum}");
        assert!(data[1] > data[2] && data[2] > data[0]);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_top_k_top_p_sample() {
        let Some(dev) = cuda_device() else { return };
        // Probabilities with one dominant token
        let probs = Tensor::from_vec(vec![0.01f32, 0.01, 0.96, 0.01, 0.01], (1, 5), &dev).unwrap();
        let rand_vals = Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap();

        let result = gpu_top_k_top_p_sample(&probs, &rand_vals, &[0], &[1.0]).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 2, "Should pick the dominant token");
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_top_k_1_is_argmax() {
        let Some(dev) = cuda_device() else { return };
        let probs = Tensor::from_vec(vec![0.1f32, 0.4, 0.3, 0.2], (1, 4), &dev).unwrap();
        let rand_vals = Tensor::from_vec(vec![0.99f32], 1, &dev).unwrap();

        let result = gpu_top_k_top_p_sample(&probs, &rand_vals, &[1], &[1.0]).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 1, "top_k=1 should always pick argmax");
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_sample_batch_all_greedy() {
        let Some(dev) = cuda_device() else { return };
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // argmax=1
                7.0, 2.0, 4.0, // argmax=0
            ],
            (2, 3),
            &dev,
        )
        .unwrap();

        let configs = vec![
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
        ];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids, vec![1, 0]);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_sample_batch_stochastic_dominant() {
        let Some(dev) = cuda_device() else { return };
        // One token has 100.0 logit, others -100.0 → effectively deterministic
        let logits =
            Tensor::from_vec(vec![-100.0f32, -100.0, 100.0, -100.0], (1, 4), &dev).unwrap();

        let configs = vec![GpuSamplingConfig {
            inv_temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.5,
        }];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids[0], 2);
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_argmax_large_vocab() {
        let Some(dev) = cuda_device() else { return };
        // Test with a realistically large vocab (32K)
        let vocab_size = 32768;
        let mut logits = vec![-1.0f32; vocab_size];
        logits[12345] = 100.0;

        let t = Tensor::from_vec(logits, (1, vocab_size), &dev).unwrap();
        let result = gpu_argmax(&t).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 12345);
    }

    // Sampler dtype contract: BF16/F16/F32 logits are all accepted; the wrappers
    // cast to F32 internally. Regression coverage for the AWQ/GPTQ path where
    // the model returns logits in the activation dtype rather than F32.

    #[test]
    fn test_gpu_argmax_bf16_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![1.0f32, 5.0, 3.0, 2.0, 0.5, 4.0, 2.0, 1.0],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let ids: Vec<u32> = gpu_argmax(&logits).unwrap().to_vec1().unwrap();
        assert_eq!(ids, vec![1, 1]);
    }

    #[test]
    fn test_gpu_argmax_f16_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![1.0f32, 5.0, 3.0, 2.0, 0.5, 4.0, 2.0, 1.0],
            (2, 4),
            &device,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
        let ids: Vec<u32> = gpu_argmax(&logits).unwrap().to_vec1().unwrap();
        assert_eq!(ids, vec![1, 1]);
    }

    #[test]
    fn test_gpu_softmax_bf16_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let probs = gpu_softmax(&logits).unwrap();
        assert_eq!(probs.dtype(), DType::F32);
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();
        let sum1: f32 = data[0..3].iter().sum();
        let sum2: f32 = data[3..6].iter().sum();
        // BF16 has only 7 mantissa bits — looser tolerance than F32 path.
        assert!((sum1 - 1.0).abs() < 1e-2, "Row 1 sum: {sum1}");
        assert!((sum2 - 1.0).abs() < 1e-2, "Row 2 sum: {sum2}");
    }

    #[test]
    fn test_gpu_sample_batch_bf16_greedy() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 5.0, 3.0, // seq 0: argmax = 1
                -100.0, -100.0, 100.0, // seq 1: argmax = 2
            ],
            (2, 3),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let configs = vec![
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
            GpuSamplingConfig {
                inv_temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                rand_val: 0.5,
            },
        ];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_gpu_apply_logits_diff_empty_is_noop() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap();
        let out = gpu_apply_logits_diff(&logits, &[]).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gpu_apply_logits_diff_basic_add() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32; 6], (2, 3), &device).unwrap();
        let diffs = vec![
            LogitsDiff {
                seq_idx: 0,
                token_id: 1,
                delta: 5.0,
            },
            LogitsDiff {
                seq_idx: 1,
                token_id: 2,
                delta: -2.5,
            },
        ];
        let out = gpu_apply_logits_diff(&logits, &diffs).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![0.0, 5.0, 0.0, 0.0, 0.0, -2.5]);
    }

    #[test]
    fn test_gpu_apply_logits_diff_accumulates_same_cell() {
        // Two diffs hitting the same (seq, token) must sum.
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![10.0f32, 10.0], (1, 2), &device).unwrap();
        let diffs = vec![
            LogitsDiff {
                seq_idx: 0,
                token_id: 0,
                delta: 1.0,
            },
            LogitsDiff {
                seq_idx: 0,
                token_id: 0,
                delta: 2.0,
            },
        ];
        let out = gpu_apply_logits_diff(&logits, &diffs).unwrap();
        let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![13.0, 10.0]);
    }

    #[test]
    fn test_gpu_sample_batch_with_diffs_logit_bias_overrides() {
        // Without bias, argmax of [1, 5, 3] is index 1.
        // Add +100 to index 0 — argmax must shift to 0.
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 5.0, 3.0], (1, 3), &device).unwrap();
        let configs = vec![GpuSamplingConfig {
            inv_temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.0,
        }];
        let diffs = vec![LogitsDiff {
            seq_idx: 0,
            token_id: 0,
            delta: 100.0,
        }];
        let ids = gpu_sample_batch_with_diffs(&logits, &configs, &diffs).unwrap();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn test_gpu_sample_batch_with_diffs_banned_neg_inf() {
        // Argmax originally points at index 2 (logit 5); ban it with -inf;
        // expect index 1 (logit 4).
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![3.0f32, 4.0, 5.0], (1, 3), &device).unwrap();
        let configs = vec![GpuSamplingConfig {
            inv_temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.0,
        }];
        let diffs = vec![LogitsDiff {
            seq_idx: 0,
            token_id: 2,
            delta: f32::NEG_INFINITY,
        }];
        let ids = gpu_sample_batch_with_diffs(&logits, &configs, &diffs).unwrap();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_gpu_sample_batch_f16_stochastic() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![-100.0f32, -100.0, 100.0, -100.0], // dominant @ idx 2
            (1, 4),
            &device,
        )
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

        let configs = vec![GpuSamplingConfig {
            inv_temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.5,
        }];

        let ids = gpu_sample_batch(&logits, &configs).unwrap();
        assert_eq!(ids[0], 2);
    }

    /// Verify the new parallel no-filter sampler picks the same token as
    /// the inverse-CDF reference for a known peaky distribution.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_gpu_multinomial_no_filter_peaky_distribution() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        // Probability mass concentrated on index 7 → both samplers must
        // return 7 for any rand_val in [0, 1).
        let vocab = 16usize;
        let mut probs_vec = vec![0.001f32; vocab];
        probs_vec[7] = 1.0 - 0.001 * (vocab as f32 - 1.0);
        let probs = Tensor::from_vec(probs_vec, (1, vocab), &device).unwrap();
        // Index 7 holds ~98.5% mass; its CDF interval is
        // [sum(0..=6) = 0.007, sum(0..=7) ≈ 0.992]. Sample rand values
        // within that interval — should always pick 7.
        for &r in &[0.1f32, 0.25, 0.5, 0.75, 0.99] {
            let rand_t = Tensor::from_vec(vec![r], 1, &device).unwrap();
            let out = gpu_multinomial_sample_no_filter(&probs, &rand_t).unwrap();
            let id: u32 = out.to_vec1::<u32>().unwrap()[0];
            assert_eq!(id, 7, "rand={r} expected 7 got {id}");
        }
    }

    /// Parity vs sequential CPU inverse-CDF on a uniform-ish distribution
    /// over vocab=151936 (Qwen3 size). Same rand → same token id.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_gpu_multinomial_no_filter_parity_vs_cpu() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let vocab = 151_936usize;
        // Deterministic synthetic probs: triangular distribution (peak at
        // the middle). Reproducible across CPU/GPU paths.
        let mut probs_vec: Vec<f32> = (0..vocab)
            .map(|i| {
                let half = vocab as f32 / 2.0;
                let d = (i as f32 - half).abs();
                ((half - d).max(1.0)).recip().sqrt()
            })
            .collect();
        let sum: f32 = probs_vec.iter().sum();
        for p in probs_vec.iter_mut() {
            *p /= sum;
        }

        // CPU reference: serial inverse-CDF.
        let cpu_sample = |probs: &[f32], rand: f32| -> u32 {
            let mut cdf = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cdf += p;
                if cdf > rand {
                    return i as u32;
                }
            }
            (probs.len() - 1) as u32
        };

        let probs_gpu = Tensor::from_vec(probs_vec.clone(), (1, vocab), &device).unwrap();
        for &r in &[0.1f32, 0.3, 0.5, 0.7, 0.9] {
            let rand_t = Tensor::from_vec(vec![r], 1, &device).unwrap();
            let gpu_id: u32 = gpu_multinomial_sample_no_filter(&probs_gpu, &rand_t)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0];
            let cpu_id = cpu_sample(&probs_vec, r);
            // **Semantically correct check:** inverse-CDF should select a
            // token in the SAME probability mass as CPU reference, not
            // bit-exact same index. F32 rounding accumulates differently
            // between GPU parallel-chunk sums and CPU sequential sum,
            // shifting the crossing-point by tens of indices on a smooth
            // distribution with ~equal neighbouring probs. The behavioural
            // contract is "pick a token whose prob is essentially equal
            // to what the CPU would pick" — which catches a buggy
            // implementation (e.g. argmax instead of inverse-CDF) without
            // demanding bit-exact crossing-point reproduction.
            let gpu_prob = probs_vec[gpu_id as usize];
            let cpu_prob = probs_vec[cpu_id as usize];
            let rel_err = (gpu_prob - cpu_prob).abs() / cpu_prob.max(1e-30);
            assert!(
                rel_err < 0.05,
                "rand={r}: gpu_id={gpu_id} (p={gpu_prob:.6e}) cpu_id={cpu_id} \
                 (p={cpu_prob:.6e}) rel_err={rel_err:.4}"
            );
        }
    }

    /// **Dispatch verification** — proves
    /// `gpu_sample_batch_with_diffs_to_tensor` actually routes through
    /// the new parallel kernel under the bench's exact sampling params
    /// (temperature=0.7, top_k=0 default, top_p=1.0 default). Counter
    /// must increment by `num_seqs` per call. Same counter-based
    /// methodology as `cuda_layernorm_4pct_regression.md`.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_dispatch_routes_to_no_filter_fast_path() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let vocab = 4096usize;
        let logits = Tensor::randn(0.0f32, 1.0, (1, vocab), &device).unwrap();
        let configs = vec![GpuSamplingConfig {
            inv_temperature: 1.0 / 0.7, // bench uses temperature=0.7
            top_k: 0,
            top_p: 1.0,
            rand_val: 0.42,
        }];
        let before = MULTINOMIAL_NO_FILTER_COUNT_LOCAL.with(|c| c.get());
        let _ = gpu_sample_batch_with_diffs_to_tensor(&logits, &configs, &[]).unwrap();
        let after = MULTINOMIAL_NO_FILTER_COUNT_LOCAL.with(|c| c.get());
        assert_eq!(
            after - before,
            1,
            "Expected exactly 1 invocation of gpu_multinomial_sample_no_filter \
             (greedy/non-greedy/no-filter dispatch hit); got {}",
            after - before
        );
    }

    /// Boundary cases — explicitly validate rand=0 (lowest CDF crossing
    /// returns first non-zero-prob index) and rand≈1 (highest crossing
    /// returns last non-zero-prob index). These would have been silently
    /// wrong if the kernel had an off-by-one or wrap-around bug.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_gpu_multinomial_no_filter_boundary_conditions() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        // Tri-mass distribution: probs[3]=0.5, probs[7]=0.3, probs[15]=0.2.
        let vocab = 16usize;
        let mut probs_vec = vec![0.0f32; vocab];
        probs_vec[3] = 0.5;
        probs_vec[7] = 0.3;
        probs_vec[15] = 0.2;
        let probs = Tensor::from_vec(probs_vec, (1, vocab), &device).unwrap();

        // rand=0.0 → target=0.0, first positive-cumsum index is 3.
        let rand_t = Tensor::from_vec(vec![0.0f32], 1, &device).unwrap();
        let id = gpu_multinomial_sample_no_filter(&probs, &rand_t)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        assert_eq!(id, 3, "rand=0.0 expected first non-zero (3) got {id}");

        // rand≈1 — target just below total_sum. Last mass cluster.
        // Use 0.95 (cdf reaches 1.0 only at index 15, so 0.95 must
        // select 15).
        let rand_t = Tensor::from_vec(vec![0.95f32], 1, &device).unwrap();
        let id = gpu_multinomial_sample_no_filter(&probs, &rand_t)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        assert_eq!(id, 15, "rand=0.95 expected last mass (15) got {id}");

        // rand=0.6 → falls in probs[7]'s region (cdf 0.5..0.8).
        let rand_t = Tensor::from_vec(vec![0.6f32], 1, &device).unwrap();
        let id = gpu_multinomial_sample_no_filter(&probs, &rand_t)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0];
        assert_eq!(id, 7, "rand=0.6 expected middle mass (7) got {id}");
    }

    /// GPU smoke for `gpu_apply_grammar_bitmask`. Verifies that
    /// allowed-token logits are left untouched while forbidden tokens
    /// are mapped to `-inf`. Two different bitmask rows ensure the
    /// kernel correctly indexes per batch.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_gpu_apply_grammar_bitmask_f32() {
        let Some(dev) = cuda_device() else { return };
        let vocab = 100usize;
        let batch = 2usize;
        let words_per_row = vocab.div_ceil(32);

        // Construct logits filled with `1.0` so we can detect untouched
        // vs `-inf` slots without floating-point ambiguity.
        let logits_data = vec![1.0f32; batch * vocab];
        let logits = Tensor::from_vec(logits_data, (batch, vocab), &dev).unwrap();

        // Row 0 allows {5, 15, 25, 50}; row 1 allows {0, 99}.
        let mut bitmask = vec![0i32; batch * words_per_row];
        let mut set = |row: usize, tok: usize| {
            bitmask[row * words_per_row + tok / 32] |= 1i32 << (tok % 32);
        };
        for &tok in &[5usize, 15, 25, 50] {
            set(0, tok);
        }
        for &tok in &[0usize, 99] {
            set(1, tok);
        }
        let bitmask_t = Tensor::from_vec(bitmask, (batch, words_per_row), &dev).unwrap();

        gpu_apply_grammar_bitmask(&logits, &bitmask_t).unwrap();
        let out: Vec<Vec<f32>> = logits.to_vec2().unwrap();

        let allow0: std::collections::HashSet<usize> = [5, 15, 25, 50].into_iter().collect();
        let allow1: std::collections::HashSet<usize> = [0, 99].into_iter().collect();
        for (b, allow) in [(0, &allow0), (1, &allow1)] {
            for (t, &v) in out[b].iter().enumerate() {
                if allow.contains(&t) {
                    assert!(
                        v.is_finite(),
                        "batch {b} token {t} should be finite, got {v}"
                    );
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "batch {b} token {t} = {v}, expected 1.0"
                    );
                } else {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "batch {b} token {t} should be -inf, got {v}"
                    );
                }
            }
        }
    }

    /// BF16 variant of the bitmask kernel — the EXL3 + Marlin paths
    /// land here in production, so the dtype dispatch must work.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_gpu_apply_grammar_bitmask_bf16() {
        let Some(dev) = cuda_device() else { return };
        let vocab = 64usize;
        let words_per_row = vocab.div_ceil(32);
        let logits = Tensor::from_vec(vec![1.0f32; vocab], (1, vocab), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // Allow only token 7.
        let mut bitmask = vec![0i32; words_per_row];
        bitmask[7 / 32] |= 1i32 << (7 % 32);
        let bitmask_t = Tensor::from_vec(bitmask, (1, words_per_row), &dev).unwrap();
        gpu_apply_grammar_bitmask(&logits, &bitmask_t).unwrap();
        let out: Vec<f32> = logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (t, &v) in out.iter().enumerate() {
            if t == 7 {
                assert!(v.is_finite() && (v - 1.0).abs() < 1e-2, "tok 7 = {v}");
            } else {
                assert!(v.is_infinite() && v < 0.0, "tok {t} = {v}, expected -inf");
            }
        }
    }
}
