//! Tier-1 per-kernel benches for the `cuda-fused-activations` family.
//!
//! Each kernel sits on the Qwen3 / Llama decode hot path. Shapes mirror
//! Qwen3-8B (hidden=4096, kv_heads=8, num_heads=32, head_dim=128,
//! intermediate=12288, vocab=151936). Concurrency sweep `c ∈ {1, 4, 8}`
//! matches the bench_decode.py harness — same buckets where regressions
//! surface end-to-end.
//!
//! Sample sizes are kept low (20-30) because the kernels are short
//! (≤100 μs/call) and run on GPU; criterion's default of 100 wastes
//! wall-clock without reducing variance further.
//!
//! Catches: regressions inside one fused kernel — alloc strategy,
//! dispatch predicate, pool slot stability. If only Tier 2 / 3 lights
//! up, the regression is at composition / dispatch-side-effect level.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12_288;
const VOCAB: usize = 151_936;
const CONCURRENCIES: &[usize] = &[1, 4, 8];
const DTYPES: &[DType] = &[DType::BF16, DType::F16];

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

/// Generate a random tensor on `dev` cast to `dtype`. Allocates F32 then
/// down-casts — fine because called once per shape per dtype.
fn randn(shape: &[usize], dtype: DType, dev: &Device) -> Tensor {
    Tensor::randn(0.0f32, 1.0, shape, dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn bench_silu_and_mul(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping fused_activations_bench::silu_and_mul: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("fused_silu_and_mul");
    group.sample_size(30);

    for &dtype in DTYPES {
        for &m in CONCURRENCIES {
            let gate = randn(&[m, 1, INTERMEDIATE], dtype, &device);
            let up = randn(&[m, 1, INTERMEDIATE], dtype, &device);
            // Warm GPU clocks + kernel JIT.
            for _ in 0..3 {
                let _ = vllm_core::cuda_kernels::silu_and_mul_separate_pooled(&gate, &up).unwrap();
            }
            sync(&device);

            let dtype_label = match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            let label = format!("c{m}/{dtype_label}");
            group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                b.iter(|| {
                    let y = vllm_core::cuda_kernels::silu_and_mul_separate_pooled(
                        black_box(&gate),
                        black_box(&up),
                    )
                    .unwrap();
                    let _ = y.dim(0).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_half_add(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping fused_activations_bench::half_add: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("fused_half_add");
    group.sample_size(30);

    for &dtype in DTYPES {
        for &m in CONCURRENCIES {
            let a = randn(&[m, 1, HIDDEN], dtype, &device);
            let b_t = randn(&[m, 1, HIDDEN], dtype, &device);
            for _ in 0..3 {
                let _ = vllm_core::cuda_kernels::half_add_pooled(&a, &b_t).unwrap();
            }
            sync(&device);

            let dtype_label = match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            let label = format!("c{m}/{dtype_label}");
            group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                b.iter(|| {
                    let y =
                        vllm_core::cuda_kernels::half_add_pooled(black_box(&a), black_box(&b_t))
                            .unwrap();
                    let _ = y.dim(0).unwrap();
                });
            });
        }
    }
    group.finish();
}

fn bench_embedding(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping fused_activations_bench::embedding: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("fused_embedding");
    // Embedding weight is 151936 × 4096 × 2B = ~1.2 GB BF16. Allocate
    // once per dtype, share across all concurrencies.
    group.sample_size(20);

    for &dtype in DTYPES {
        let weight = randn(&[VOCAB, HIDDEN], dtype, &device);
        for &m in CONCURRENCIES {
            let ids: Vec<u32> = (0..m as u32).map(|i| i % VOCAB as u32).collect();
            let input_ids = Tensor::from_vec(ids, (m, 1), &device).unwrap();
            for _ in 0..3 {
                let _ = vllm_core::cuda_kernels::embedding_pooled(&input_ids, &weight).unwrap();
            }
            sync(&device);

            let dtype_label = match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            let label = format!("c{m}/{dtype_label}");
            group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                b.iter(|| {
                    let y = vllm_core::cuda_kernels::embedding_pooled(
                        black_box(&input_ids),
                        black_box(&weight),
                    )
                    .unwrap();
                    let _ = y.dim(0).unwrap();
                });
            });
        }
    }
    group.finish();
}

/// `half_matmul_pooled` at decode shape `[c, 1, hidden] @ [hidden, hidden_or_intermediate]`.
/// The existing `lm_head_matmul_bench` only covers the vocab-output
/// matmul (M, K=hidden, N=vocab); QKV / O / gate / up / down projections
/// have very different shapes (square or near-square) and a different
/// cuBLAS path inside candle.
fn bench_half_matmul_decode(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping fused_activations_bench::half_matmul_decode: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("fused_half_matmul_decode");
    group.sample_size(30);

    // (label, weight_out_dim) — covers Qwen3 layer linears at decode-time.
    let shapes: &[(&str, usize)] = &[
        ("qkv_like", HIDDEN),
        ("gate_or_up", INTERMEDIATE),
        ("down", HIDDEN), // input dim is intermediate; matched below
    ];

    for &dtype in DTYPES {
        for &(weight_label, out_dim) in shapes {
            for &m in CONCURRENCIES {
                let in_dim = if weight_label == "down" {
                    INTERMEDIATE
                } else {
                    HIDDEN
                };
                let x = randn(&[m, 1, in_dim], dtype, &device);
                let w = randn(&[out_dim, in_dim], dtype, &device);
                for _ in 0..3 {
                    let _ = vllm_core::cuda_kernels::half_matmul_pooled(&x, &w).unwrap();
                }
                sync(&device);

                let dtype_label = match dtype {
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    _ => unreachable!(),
                };
                let label = format!("c{m}/{weight_label}/{dtype_label}");
                group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                    b.iter(|| {
                        let y = vllm_core::cuda_kernels::half_matmul_pooled(
                            black_box(&x),
                            black_box(&w),
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

criterion_group!(
    benches,
    bench_silu_and_mul,
    bench_half_add,
    bench_embedding,
    bench_half_matmul_decode,
);
criterion_main!(benches);
