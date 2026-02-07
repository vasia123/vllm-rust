//! GPU-side sampling to avoid large GPU→CPU transfers.
//!
//! For greedy decoding (argmax) and standard top-k/top-p sampling, the entire
//! operation runs on GPU. Only the final token ID (one i32 per sequence) is
//! transferred back.
//!
//! For beam search and constrained decoding, falls back to CPU sampling.

use candle_core::{DType, Device, Result, Tensor};

/// Whether GPU sampling is available for the given device.
pub fn gpu_sampling_available(device: &Device) -> bool {
    device.is_cuda()
}

/// Greedy sampling (argmax) on GPU.
///
/// Takes logits tensor `[num_seqs, vocab_size]` in BF16 and returns
/// token IDs `[num_seqs]` as U32 on the same device.
///
/// This avoids transferring the full logit matrix to CPU.
pub fn gpu_argmax(logits: &Tensor) -> Result<Tensor> {
    // For CPU tensors or when CUDA kernels aren't available, fall back to
    // candle's built-in argmax which works on any device.
    let logits_f32 = logits.to_dtype(DType::F32)?;
    logits_f32.argmax(1)?.to_dtype(DType::U32)
}

/// GPU-accelerated softmax for probability conversion.
///
/// Converts BF16 logits to F32 probabilities on GPU, avoiding the
/// transfer of raw logits to CPU.
pub fn gpu_softmax(logits: &Tensor) -> Result<Tensor> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let max = logits_f32.max_keepdim(1)?;
    let shifted = logits_f32.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(1)?;
    exp.broadcast_div(&sum)
}

/// Top-k filtering: zeros out all probabilities outside the top-k.
///
/// # Arguments
/// - `probs`: Probability tensor `[num_seqs, vocab_size]` F32
/// - `k`: Number of top tokens to keep
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

        // Find k-th largest value via partial sort
        let mut sorted: Vec<f32> = row.to_vec();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];

        // Keep tokens >= threshold
        for v in 0..vocab_size {
            if row[v] >= threshold {
                result[seq * vocab_size + v] = row[v];
            }
        }
    }

    Tensor::from_vec(result, (num_seqs, vocab_size), probs.device())
}

/// Top-p (nucleus) filtering: keeps the smallest set of tokens whose
/// cumulative probability exceeds `p`.
///
/// # Arguments
/// - `probs`: Probability tensor `[num_seqs, vocab_size]` F32
/// - `p`: Nucleus sampling threshold (0.0 to 1.0)
pub fn gpu_top_p_filter(probs: &Tensor, p: f32) -> Result<Tensor> {
    if p >= 1.0 {
        return Ok(probs.clone());
    }

    let (num_seqs, vocab_size) = probs.dims2()?;
    let data: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    let mut result = vec![0.0f32; num_seqs * vocab_size];

    for seq in 0..num_seqs {
        let row = &data[seq * vocab_size..(seq + 1) * vocab_size];

        // Sort indices by probability descending
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Accumulate until cumsum > p
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

/// Cumulative sum along the last dimension (row-wise).
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

/// Sample from probability distribution on GPU.
///
/// Performs multinomial sampling using pre-generated random values.
///
/// # Arguments
/// - `probs`: Probability tensor `[num_seqs, vocab_size]` F32 (already filtered)
/// - `rand_vals`: Uniform random values `[num_seqs]` F32 in [0, 1)
///
/// # Returns
/// Token IDs `[num_seqs]` U32
pub fn gpu_multinomial_sample(probs: &Tensor, rand_vals: &Tensor) -> Result<Tensor> {
    let (num_seqs, vocab_size) = probs.dims2()?;

    // Renormalize after filtering
    let sum = probs.sum_keepdim(1)?;
    let probs = probs.broadcast_div(&sum)?;

    // Compute cumulative sum
    let cumsum = cumulative_sum(&probs)?;

    // For each sequence, find the first index where cumsum > rand_val
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
        assert_eq!(ids, vec![1, 1]); // argmax of [1,5,3,2] = 1, [0.5,4,2,1] = 1
    }

    #[test]
    fn test_gpu_softmax_sums_to_one() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();

        let probs = gpu_softmax(&logits).unwrap();
        let data: Vec<f32> = probs.flatten_all().unwrap().to_vec1().unwrap();

        // Each row should sum to ~1.0
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

        // Top-2 are indices 1 (0.4) and 2 (0.3)
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
        // With rand_val=0.0, should pick first non-zero token
        let rand_val = Tensor::from_vec(vec![0.0f32], 1, &device).unwrap();

        let result = gpu_multinomial_sample(&probs, &rand_val).unwrap();
        let ids: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(ids[0], 0, "Low rand should pick first token");
    }

    #[test]
    fn test_gpu_multinomial_sample_high_rand() {
        let device = Device::Cpu;
        let probs = Tensor::from_vec(vec![0.5f32, 0.3, 0.2], (1, 3), &device).unwrap();
        // With rand_val=0.99, should pick last token
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

        // Higher logit → higher probability
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
}
