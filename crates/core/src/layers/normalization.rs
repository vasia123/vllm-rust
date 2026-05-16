use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Three RMSNorm flavours used across the model zoo.
///
/// - `Standard`: classic `xs / sqrt(var + eps) * weight` (Llama, Qwen,
///   Mistral, Gemma 4, …).
/// - `ScalePlusOne`: `xs / sqrt(var + eps) * (1 + weight)` — the
///   "Gemma trick" used by Gemma 1/2/3 to keep RMSNorm initial weights
///   centred around 0 instead of 1. Order-of-ops sensitive in bf16,
///   so this is implemented manually rather than through candle's
///   fused kernel.
/// - `Unweighted`: `xs / sqrt(var + eps)` with no scale at all
///   (Gemma 4 `v_norm`, router norms in MoE attention). No weight
///   tensor is loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsNormVariant {
    Standard,
    ScalePlusOne,
    Unweighted,
}

/// RMSNorm layer with optional fused CUDA kernel acceleration.
///
/// Drop-in replacement for `candle_nn::RmsNorm`. When the `cuda-layernorm`
/// feature is enabled, the variant is `Standard`, and the input tensor is
/// on GPU, dispatches to a fused CUDA kernel that computes variance,
/// normalization, and weight scaling in a single pass (5-15% throughput
/// improvement over candle's multi-kernel path). Other variants take a
/// manual F32-converted path that matches the per-model implementations
/// the Gemma family historically shipped.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Option<Tensor>,
    eps: f64,
    variant: RmsNormVariant,
}

impl RmsNorm {
    /// Construct a `Standard` RMSNorm from a weight tensor.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight: Some(weight),
            eps,
            variant: RmsNormVariant::Standard,
        }
    }

    /// Construct an RMSNorm with an explicit variant. `Unweighted`
    /// ignores any weight argument (pass `None`).
    pub fn with_variant(weight: Option<Tensor>, eps: f64, variant: RmsNormVariant) -> Self {
        Self {
            weight,
            eps,
            variant,
        }
    }

    /// Pool-backed forward for the captured decode hot path. Only
    /// supports the `Standard` variant — `ScalePlusOne` / `Unweighted`
    /// allocate F32 intermediates via candle ops, which are not pool-
    /// backed; callers on those variants must fall back to
    /// [`Self::forward`] (eager paths only).
    ///
    /// **Capture-state-aware dispatch:** if currently *not* capturing a
    /// CUDA Graph (i.e. eager decode), routes through
    /// `candle_nn::ops::rms_norm` which is ~1.7× faster per call than
    /// the pool-backed custom kernel (`rms_norm_bench`: 10.5 vs 6.2 µs).
    /// Inside `begin_capture`/`end_capture` brackets, falls back to the
    /// pool-backed custom kernel so the captured graph encodes a stable
    /// pool-slot pointer (see ADR 0019 / `pool_tensor_type_safety`).
    /// Wrapping the candle output via `from_pool_unchecked` is sound in
    /// the eager branch because no captured graph ever references this
    /// storage — eager output is dropped after the layer.
    #[cfg(feature = "cuda-layernorm")]
    pub fn forward_pooled(
        &self,
        xs: &crate::engine::output_pool::PooledTensor,
    ) -> Result<crate::engine::output_pool::PooledTensor> {
        match self.variant {
            RmsNormVariant::Standard => {
                let weight = self.weight.as_ref().expect("Standard RmsNorm needs weight");
                if !crate::cuda_kernels::rms_norm_cuda_available(xs.as_tensor()) {
                    candle_core::bail!(
                        "RmsNorm::forward_pooled: cuda kernel unavailable for {:?} on {:?}; \
                         captured forward requires CUDA",
                        xs.dtype(),
                        xs.device()
                    );
                }
                let xs_c = xs.contiguous()?;
                let capturing = crate::engine::cuda_graph::IN_CUDA_GRAPH_CAPTURE
                    .load(std::sync::atomic::Ordering::Relaxed);
                if capturing {
                    return crate::cuda_kernels::rms_norm_cuda_pooled_typed(
                        &xs_c,
                        weight,
                        self.eps as f32,
                    );
                }
                // Eager path: candle is ~4.3 µs/call faster; safe to wrap
                // because no captured graph will reference this tensor.
                let y = candle_nn::ops::rms_norm(xs_c.as_tensor(), weight, self.eps as f32)?;
                // SAFETY: not in capture; no graph node will encode a
                // pointer into this storage. Storage's lifetime is the
                // caller's eager forward — sufficient for the layer
                // pipeline.
                Ok(unsafe { crate::engine::output_pool::PooledTensor::from_pool_unchecked(y) })
            }
            _ => candle_core::bail!(
                "RmsNorm::forward_pooled: only Standard variant is pool-backed; \
                 got {:?} — caller must use eager Self::forward",
                self.variant
            ),
        }
    }

    pub fn weight(&self) -> Option<&Tensor> {
        self.weight.as_ref()
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn variant(&self) -> RmsNormVariant {
        self.variant
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self.variant {
            RmsNormVariant::Standard => {
                let weight = self.weight.as_ref().expect("Standard RmsNorm needs weight");

                // Eager dispatch uses `candle_nn::ops::rms_norm` — empirically
                // ~1.7× faster per call than `rms_norm_cuda_pooled` at decode
                // shape (h=4096 F16, c=1..8: candle 6.2 µs vs custom 10.5 µs
                // per `rms_norm_bench`; perfectly additive across 4 sequential
                // calls per `rms_norm_chain_x4`). Validated at layer level via
                // `quantized_layer_bench` with **explicit GPU sync** in
                // `b.iter` body: cuda-layernorm enabled → +4.3% c=4 layer
                // forward (p=0.00 vs synced fused-only baseline). Without
                // sync, criterion measures CPU enqueue rate where the GPU
                // delta overlaps with adjacent ops and is hidden — but
                // production decode is GPU-completion-bound via the per-token
                // sampler DtoH, so the regression manifests in real TTFT/TPS.
                //
                // The custom pool-backed kernel is still selected for CUDA
                // Graph capture via `forward_pooled` (preserves stable
                // pool-slot addresses across captures); eager `forward` has
                // no such requirement.
                candle_nn::ops::rms_norm(&xs.contiguous()?, weight, self.eps as f32)
            }
            RmsNormVariant::ScalePlusOne => {
                let weight = self
                    .weight
                    .as_ref()
                    .expect("ScalePlusOne RmsNorm needs weight");
                // Manual `(1 + weight) * normed` path — order-of-ops
                // sensitive in bf16, so we explicitly upcast to F32.
                let dtype = xs.dtype();
                let xs_f32 = xs.to_dtype(DType::F32)?;
                let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
                let xs_normed = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
                let scale = (weight.to_dtype(DType::F32)? + 1.0)?;
                xs_normed.broadcast_mul(&scale)?.to_dtype(dtype)
            }
            RmsNormVariant::Unweighted => {
                // No `weight` tensor — pure normalisation.
                let dtype = xs.dtype();
                let xs_f32 = xs.to_dtype(DType::F32)?;
                let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
                let xs_normed = xs_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
                xs_normed.to_dtype(dtype)
            }
        }
    }
}

/// Create a `Standard` RMSNorm layer, loading the weight from a VarBuilder.
///
/// Drop-in replacement for `candle_nn::rms_norm()`.
pub fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::new(weight, eps))
}

/// Create a `ScalePlusOne` RMSNorm — Gemma 1/2/3 style.
pub fn rms_norm_gemma(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    Ok(RmsNorm::with_variant(
        Some(weight),
        eps,
        RmsNormVariant::ScalePlusOne,
    ))
}

/// Create an `Unweighted` RMSNorm — no weight tensor is read from
/// `vb` (the variant takes none).
pub fn rms_norm_unweighted(eps: f64) -> RmsNorm {
    RmsNorm::with_variant(None, eps, RmsNormVariant::Unweighted)
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
        assert_eq!(
            norm.weight().expect("Standard variant has weight").dims(),
            &[hidden]
        );
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

    #[test]
    fn test_scale_plus_one_matches_legacy_gemma_path() {
        // ScalePlusOne should be byte-equal to the per-model `(1 +
        // weight) * normed` implementation that the Gemma family
        // historically shipped (see e.g. `models/gemma3.rs::Gemma3RmsNorm`).
        let device = Device::Cpu;
        let hidden = 32;
        let eps = 1e-6;

        // Pick non-trivial weights so weight ≠ 0 path is exercised.
        let weight_data: Vec<f32> = (0..hidden).map(|i| -0.4 + 0.05 * i as f32).collect();
        let weight = Tensor::from_vec(weight_data, hidden, &device).unwrap();
        let norm = RmsNorm::with_variant(Some(weight.clone()), eps, RmsNormVariant::ScalePlusOne);

        let input = Tensor::randn(0.0f32, 1.0, (4, hidden), &device).unwrap();
        let actual = norm.forward(&input).unwrap();

        // Reference: (1 + w) * (xs / sqrt(var + eps)).
        let xs_f32 = input.to_dtype(DType::F32).unwrap();
        let variance = xs_f32.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let normed = xs_f32
            .broadcast_div(&(variance + eps).unwrap().sqrt().unwrap())
            .unwrap();
        let scale = (weight.to_dtype(DType::F32).unwrap() + 1.0).unwrap();
        let expected = normed.broadcast_mul(&scale).unwrap();

        let a: Vec<f32> = actual.flatten_all().unwrap().to_vec1().unwrap();
        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (x, y)) in a.iter().zip(e.iter()).enumerate() {
            assert!((x - y).abs() < 1e-6, "Mismatch at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn test_unweighted_drops_scale() {
        // Unweighted RMSNorm must be the pure normalisation step —
        // identical to `Standard` with weight=ones except no weight
        // tensor is required.
        let device = Device::Cpu;
        let hidden = 16;
        let eps = 1e-5;

        let norm = rms_norm_unweighted(eps);
        let input = Tensor::randn(0.0f32, 1.0, (2, hidden), &device).unwrap();
        let actual = norm.forward(&input).unwrap();

        let xs_f32 = input.to_dtype(DType::F32).unwrap();
        let variance = xs_f32.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let expected = xs_f32
            .broadcast_div(&(variance + eps).unwrap().sqrt().unwrap())
            .unwrap();

        let a: Vec<f32> = actual.flatten_all().unwrap().to_vec1().unwrap();
        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (x, y)) in a.iter().zip(e.iter()).enumerate() {
            assert!((x - y).abs() < 1e-6, "Mismatch at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn test_rms_norm_gemma_helper_loads_from_vb() {
        // Smoke test: `rms_norm_gemma` constructs a `ScalePlusOne` variant.
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let norm = rms_norm_gemma(64, 1e-6, vb).unwrap();
        assert_eq!(norm.variant(), RmsNormVariant::ScalePlusOne);
        assert!(norm.weight().is_some());
    }
}
