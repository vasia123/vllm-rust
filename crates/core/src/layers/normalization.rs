use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// RMSNorm layer with optional fused CUDA kernel acceleration.
///
/// Drop-in replacement for `candle_nn::RmsNorm`. When the `cuda-layernorm`
/// feature is enabled and the input tensor is on GPU, dispatches to a fused
/// CUDA kernel that computes variance, normalization, and weight scaling in
/// a single pass (5-15% throughput improvement over candle's multi-kernel path).
/// Falls back to `candle_nn::ops::rms_norm` otherwise.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda-layernorm")]
        {
            if crate::cuda_kernels::rms_norm_cuda_available(xs) {
                let xs = xs.contiguous()?;
                return crate::cuda_kernels::rms_norm_cuda(&xs, &self.weight, self.eps as f32);
            }
        }

        // Fallback: candle_nn ops (fused CUDA via candle for GPU, pure Rust for CPU)
        candle_nn::ops::rms_norm(&xs.contiguous()?, &self.weight, self.eps as f32)
    }
}

/// Create an RMSNorm layer, loading the weight from a VarBuilder.
///
/// Drop-in replacement for `candle_nn::rms_norm()`.
pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_rms_norm_basic_shape() {
        let device = Device::Cpu;
        let hidden = 64;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let input = Tensor::randn(0.0f32, 1.0, (4, hidden), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, hidden]);
    }

    #[test]
    fn test_rms_norm_unit_weight_is_normalized() {
        let device = Device::Cpu;
        let hidden = 32;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-6);

        let input = Tensor::randn(0.0f32, 1.0, (2, hidden), &device).unwrap();
        let output = norm.forward(&input).unwrap();

        // RMSNorm output should have RMS close to 1.0
        let output_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        for row in output_data.chunks(hidden) {
            let rms: f32 = (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.1,
                "RMS should be close to 1.0, got {rms}"
            );
        }
    }

    #[test]
    fn test_rms_norm_3d_input() {
        let device = Device::Cpu;
        let hidden = 16;
        let weight = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let norm = RmsNorm::new(weight, 1e-5);

        let input = Tensor::randn(0.0f32, 1.0, (2, 8, hidden), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 8, hidden]);
    }

    #[test]
    fn test_rms_norm_matches_candle_nn() {
        let device = Device::Cpu;
        let hidden = 64;
        let eps = 1e-6;

        let weight_data: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.01 * i as f32).collect();
        let weight = Tensor::from_vec(weight_data, hidden, &device).unwrap();

        let our_norm = RmsNorm::new(weight.clone(), eps);
        let candle_norm = candle_nn::RmsNorm::new(weight, eps);

        let input = Tensor::randn(0.0f32, 1.0, (4, hidden), &device).unwrap();
        let our_output = our_norm.forward(&input).unwrap();
        let candle_output = candle_norm.forward(&input).unwrap();

        let our_data: Vec<f32> = our_output.flatten_all().unwrap().to_vec1().unwrap();
        let candle_data: Vec<f32> = candle_output.flatten_all().unwrap().to_vec1().unwrap();

        for (i, (a, b)) in our_data.iter().zip(candle_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Mismatch at index {i}: ours={a}, candle={b}"
            );
        }
    }

    #[test]
    fn test_rms_norm_varbuilder() {
        let device = Device::Cpu;
        let hidden = 32;
        let eps = 1e-5;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let norm = rms_norm(hidden, eps, vb);
        assert!(norm.is_ok());

        let norm = norm.unwrap();
        assert_eq!(norm.weight().dims(), &[hidden]);
        assert_eq!(norm.eps(), eps);
    }

    #[test]
    fn test_rms_norm_weight_scaling() {
        let device = Device::Cpu;
        let hidden = 8;

        // Weight of 2.0 should double the output relative to unit weight
        let weight_1 = Tensor::ones(hidden, DType::F32, &device).unwrap();
        let weight_2 = (Tensor::ones(hidden, DType::F32, &device).unwrap() * 2.0).unwrap();

        let norm_1 = RmsNorm::new(weight_1, 1e-6);
        let norm_2 = RmsNorm::new(weight_2, 1e-6);

        let input = Tensor::from_vec(vec![1.0f32; hidden], hidden, &device).unwrap();
        let out_1 = norm_1.forward(&input).unwrap();
        let out_2 = norm_2.forward(&input).unwrap();

        let data_1: Vec<f32> = out_1.to_vec1().unwrap();
        let data_2: Vec<f32> = out_2.to_vec1().unwrap();

        for (a, b) in data_1.iter().zip(data_2.iter()) {
            assert!(
                (b - 2.0 * a).abs() < 1e-5,
                "weight=2 output should be 2x weight=1 output: {b} vs 2*{a}"
            );
        }
    }
}
