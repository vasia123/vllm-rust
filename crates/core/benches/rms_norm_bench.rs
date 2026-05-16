//! Tier-1 per-kernel bench for `rms_norm_cuda_pooled`.
//!
//! Three norm sites per Qwen3 layer (input_norm, post_attention_norm,
//! plus a final_norm at the top of the decode forward) × 36 layers =
//! 73 calls/token. Each shape `[c, 1, hidden]`. Catches regressions in
//! fused RMSNorm path (variance + scale + weight in one kernel) vs
//! candle's multi-kernel fallback.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
extern crate candle_nn;

const CONCURRENCIES: &[usize] = &[1, 4, 8];
const DTYPES: &[DType] = &[DType::BF16, DType::F16];
// Hidden-size sweep covers Qwen3-1.5B/4B/8B/14B and Llama-3 model families.
const HIDDENS: &[usize] = &[1536, 2560, 4096, 5120];
const EPS: f32 = 1e-6;

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_rms_norm(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping rms_norm_bench: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("rms_norm_pooled");
    group.sample_size(30);

    for &dtype in DTYPES {
        for &hidden in HIDDENS {
            let weight = Tensor::ones(hidden, dtype, &device).unwrap();
            for &m in CONCURRENCIES {
                let xs = Tensor::randn(0.0f32, 1.0, (m, 1, hidden), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                for _ in 0..3 {
                    let _ =
                        vllm_core::cuda_kernels::rms_norm_cuda_pooled(&xs, &weight, EPS).unwrap();
                }
                sync(&device);

                let dtype_label = match dtype {
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    _ => unreachable!(),
                };
                let label = format!("c{m}/h{hidden}/{dtype_label}");
                group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                    b.iter(|| {
                        let y = vllm_core::cuda_kernels::rms_norm_cuda_pooled(
                            black_box(&xs),
                            black_box(&weight),
                            EPS,
                        )
                        .unwrap();
                        let _ = y.dim(0).unwrap();
                    });
                });
            }
        }
    }
    group.finish();
}

/// Comparison bench: `candle_nn::ops::rms_norm` (alternate path used
/// when `cuda-layernorm` is OFF). Diff between this group and
/// `rms_norm_pooled` localises the documented +4% regression to either
/// (a) pool reservation overhead, or (b) custom kernel cost.
fn bench_candle_rms_norm(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("rms_norm_candle_nn");
    group.sample_size(30);

    for &dtype in DTYPES {
        for &hidden in HIDDENS {
            let weight = Tensor::ones(hidden, dtype, &device).unwrap();
            for &m in CONCURRENCIES {
                let xs = Tensor::randn(0.0f32, 1.0, (m, 1, hidden), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                for _ in 0..3 {
                    let _ =
                        candle_nn::ops::rms_norm(&xs.contiguous().unwrap(), &weight, EPS).unwrap();
                }
                sync(&device);

                let dtype_label = match dtype {
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    _ => unreachable!(),
                };
                let label = format!("c{m}/h{hidden}/{dtype_label}");
                group.bench_with_input(BenchmarkId::new("candle", label), &m, |b, _| {
                    b.iter(|| {
                        let y = candle_nn::ops::rms_norm(
                            &black_box(&xs).contiguous().unwrap(),
                            black_box(&weight),
                            EPS,
                        )
                        .unwrap();
                        let _ = y.dim(0).unwrap();
                    });
                });
            }
        }
    }
    group.finish();
}

/// Sequential **4-call** pattern mimicking one Qwen3 layer's RmsNorm
/// chain (input_norm, q_norm, k_norm, post_attn_norm). Goal:
/// distinguish two hypotheses for the rms_norm_bench (1-call) micro
/// delta vs quantized_layer_bench neutral discrepancy:
///   (a) per-call CPU enqueue overhead — additive across 4 calls;
///       4-call diff ≈ 4 × 1-call diff.
///   (b) GPU stream concurrency / overlap — 4-call diff < 4×; some
///       per-call cost overlaps with adjacent kernel.
fn bench_rms_norm_chain_x4(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("rms_norm_chain_x4");
    group.sample_size(30);

    // Hot shape only (Qwen3-8B hidden=4096 F16, c=4 mid bucket).
    let dtype = DType::F16;
    let hidden = 4096;
    let m = 4;
    let weight = Tensor::ones(hidden, dtype, &device).unwrap();
    let xs = Tensor::randn(0.0f32, 1.0, (m, 1, hidden), &device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    // Custom path: 4× rms_norm_cuda_pooled (pool-backed).
    for _ in 0..3 {
        let mut y = xs.clone();
        for _ in 0..4 {
            y = vllm_core::cuda_kernels::rms_norm_cuda_pooled(&y, &weight, EPS).unwrap();
        }
        let _ = y.dim(0).unwrap();
    }
    sync(&device);

    group.bench_function("custom_4x", |b| {
        b.iter(|| {
            let mut y = xs.clone();
            for _ in 0..4 {
                y = vllm_core::cuda_kernels::rms_norm_cuda_pooled(
                    black_box(&y),
                    black_box(&weight),
                    EPS,
                )
                .unwrap();
            }
            let _ = y.dim(0).unwrap();
        });
    });

    // Candle path: 4× candle_nn::ops::rms_norm.
    for _ in 0..3 {
        let mut y = xs.contiguous().unwrap();
        for _ in 0..4 {
            y = candle_nn::ops::rms_norm(&y, &weight, EPS).unwrap();
            y = y.contiguous().unwrap();
        }
        let _ = y.dim(0).unwrap();
    }
    sync(&device);

    group.bench_function("candle_4x", |b| {
        b.iter(|| {
            let mut y = xs.contiguous().unwrap();
            for _ in 0..4 {
                y = candle_nn::ops::rms_norm(black_box(&y), black_box(&weight), EPS).unwrap();
                y = y.contiguous().unwrap();
            }
            let _ = y.dim(0).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_candle_rms_norm,
    bench_rms_norm_chain_x4,
);
criterion_main!(benches);
