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
mod cuda_ops {
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda::CudaStorageSlice;
    use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, Layout, Result, Shape};

    use super::SAMPLING_PTX;

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
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                ids.push(idx);
            }
            Ok((CpuStorage::U32(ids), Shape::from_dims(&[num_seqs])))
        }

        fn cuda_fwd(
            &self,
            s: &CudaStorage,
            l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
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
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("argmax_bf16 launch: {e}"))
                    })?;
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
                    unsafe { builder.launch(cfg) }.map_err(|e| {
                        candle_core::Error::Msg(format!("argmax_f32 launch: {e}"))
                    })?;
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

        fn cuda_fwd(
            &self,
            s: &CudaStorage,
            l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
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
                        .get_or_load_custom_func(
                            "softmax_to_probs_f32",
                            "sampling",
                            SAMPLING_PTX,
                        )
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

            let top_k_gpu = dev.memcpy_stod(&self.top_k)?;
            let top_p_gpu = dev.memcpy_stod(&self.top_p)?;

            let output_slice = dev
                .alloc_zeros::<u32>(num_seqs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let vocab_size_i32 = vocab_size as i32;

            let func = dev
                .get_or_load_custom_func(
                    "top_k_top_p_sample_per_seq",
                    "sampling",
                    SAMPLING_PTX,
                )
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
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Greedy sampling (argmax) on GPU via CUDA kernel.
///
/// Takes logits tensor `[num_seqs, vocab_size]` and returns
/// token IDs `[num_seqs]` as U32.
///
/// On CUDA: only transfers `num_seqs * 4` bytes back (one u32 per sequence)
/// instead of `num_seqs * vocab_size * dtype_size` bytes.
pub fn gpu_argmax(logits: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda-kernels")]
    if logits.device().is_cuda() {
        let logits = logits.contiguous()?;
        return logits.apply_op1_no_bwd(&cuda_ops::ArgmaxOp);
    }

    // CPU fallback via Candle ops
    let logits_f32 = logits.to_dtype(DType::F32)?;
    logits_f32.argmax(1)?.to_dtype(DType::U32)
}

/// GPU-accelerated softmax for probability conversion.
///
/// Converts logits to F32 probabilities on GPU.
pub fn gpu_softmax(logits: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda-kernels")]
    if logits.device().is_cuda() {
        let logits = logits.contiguous()?;
        return logits.apply_op1_no_bwd(&cuda_ops::SoftmaxOp);
    }

    // CPU fallback
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let max = logits_f32.max_keepdim(1)?;
    let shifted = logits_f32.broadcast_sub(&max)?;
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
            sorted.sort_unstable_by(|a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            threshold = sorted[k as usize - 1];
        }

        // Compute filtered set sorted by probability descending
        let mut filtered: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v >= threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        filtered.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

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
pub fn gpu_sample_batch(logits: &Tensor, configs: &[GpuSamplingConfig]) -> Result<Vec<u32>> {
    let (num_seqs, _vocab_size) = logits.dims2()?;
    assert_eq!(num_seqs, configs.len());

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

    // Handle greedy sequences: override with argmax result
    let mut ids: Vec<u32> = result.to_vec1()?;
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
        indexed
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
        let logits =
            Tensor::from_vec(vec![1.0f32, 1.0, 100.0, 1.0], (1, 4), &device).unwrap();

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
        let logits = Tensor::from_vec(
            vec![1.0f32, 5.0, 3.0, 7.0, 2.0, 4.0],
            (2, 3),
            &dev,
        )
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
        let logits = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            &dev,
        )
        .unwrap();

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
        let probs = Tensor::from_vec(
            vec![0.01f32, 0.01, 0.96, 0.01, 0.01],
            (1, 5),
            &dev,
        )
        .unwrap();
        let rand_vals = Tensor::from_vec(vec![0.5f32], 1, &dev).unwrap();

        let result =
            gpu_top_k_top_p_sample(&probs, &rand_vals, &[0], &[1.0]).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 2, "Should pick the dominant token");
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_cuda_top_k_1_is_argmax() {
        let Some(dev) = cuda_device() else { return };
        let probs = Tensor::from_vec(
            vec![0.1f32, 0.4, 0.3, 0.2],
            (1, 4),
            &dev,
        )
        .unwrap();
        let rand_vals = Tensor::from_vec(vec![0.99f32], 1, &dev).unwrap();

        let result =
            gpu_top_k_top_p_sample(&probs, &rand_vals, &[1], &[1.0]).unwrap();
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
        let logits = Tensor::from_vec(
            vec![-100.0f32, -100.0, 100.0, -100.0],
            (1, 4),
            &dev,
        )
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
}
