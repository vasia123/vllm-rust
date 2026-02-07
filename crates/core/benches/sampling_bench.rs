//! Criterion benchmarks for the sampling module.
//!
//! Covers both the CPU scalar sampling path (`sampling::sample`, `sampling::log_softmax`)
//! and the tensor-based GPU sampling functions (`sampling::gpu::*`) running on CPU device.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{Device, Tensor};
use vllm_core::sampling::gpu::{
    gpu_argmax, gpu_multinomial_sample, gpu_softmax, gpu_top_k_filter, gpu_top_p_filter,
};
use vllm_core::sampling::{log_softmax, sample, SamplerState, SamplingParams};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a deterministic logits vector of the given size.
fn make_logits(vocab_size: usize) -> Vec<f32> {
    (0..vocab_size)
        .map(|i| ((i as f32 * 0.017).sin() * 5.0))
        .collect()
}

/// Build a 2-D logits tensor [batch_size, vocab_size] on CPU.
fn make_logits_tensor(batch_size: usize, vocab_size: usize) -> Tensor {
    let data: Vec<f32> = (0..batch_size * vocab_size)
        .map(|i| ((i as f32 * 0.013).sin() * 5.0))
        .collect();
    Tensor::from_vec(data, (batch_size, vocab_size), &Device::Cpu)
        .expect("failed to create logits tensor")
}

/// Build a probability tensor [batch_size, vocab_size] on CPU (softmax-normalized).
fn make_probs_tensor(batch_size: usize, vocab_size: usize) -> Tensor {
    let logits = make_logits_tensor(batch_size, vocab_size);
    gpu_softmax(&logits).expect("failed to compute softmax")
}

/// Build a random-values tensor [batch_size] on CPU in [0, 1).
fn make_rand_vals(batch_size: usize) -> Tensor {
    let data: Vec<f32> = (0..batch_size)
        .map(|i| ((i as f32 * 0.7 + 0.1) % 1.0))
        .collect();
    Tensor::from_vec(data, batch_size, &Device::Cpu).expect("failed to create rand_vals tensor")
}

// ---------------------------------------------------------------------------
// CPU scalar sampling benchmarks
// ---------------------------------------------------------------------------

fn bench_sample_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_greedy");
    let params = SamplingParams::greedy();

    for &vocab_size in &[32_000, 128_000] {
        let logits = make_logits(vocab_size);
        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut state = SamplerState::new(Some(42));
                b.iter(|| sample(black_box(&logits), &params, &[], &mut state, None));
            },
        );
    }
    group.finish();
}

fn bench_sample_top_k_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_top_k_top_p");
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        ..Default::default()
    };

    for &vocab_size in &[32_000, 128_000] {
        let logits = make_logits(vocab_size);
        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut state = SamplerState::new(Some(42));
                b.iter(|| sample(black_box(&logits), &params, &[], &mut state, None));
            },
        );
    }
    group.finish();
}

fn bench_log_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_softmax");

    for &vocab_size in &[32_000, 128_000] {
        let logits = make_logits(vocab_size);
        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| log_softmax(black_box(&logits)));
            },
        );
    }
    group.finish();
}

fn bench_sample_with_logprobs(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_with_logprobs");
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        ..Default::default()
    };

    for &vocab_size in &[32_000, 128_000] {
        let logits = make_logits(vocab_size);
        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut state = SamplerState::new(Some(42));
                b.iter(|| sample(black_box(&logits), &params, &[], &mut state, Some(5)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor-based (gpu module, running on CPU device) benchmarks
// ---------------------------------------------------------------------------

fn bench_gpu_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_argmax");

    for &batch_size in &[1, 8, 32, 64] {
        for &vocab_size in &[32_000, 128_000] {
            let logits = make_logits_tensor(batch_size, vocab_size);
            let label = format!("b{batch_size}_v{vocab_size}");
            group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
                b.iter(|| gpu_argmax(black_box(&logits)).expect("gpu_argmax failed"));
            });
        }
    }
    group.finish();
}

fn bench_gpu_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_softmax");

    for &batch_size in &[1, 8, 32, 64] {
        for &vocab_size in &[32_000, 128_000] {
            let logits = make_logits_tensor(batch_size, vocab_size);
            let label = format!("b{batch_size}_v{vocab_size}");
            group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
                b.iter(|| gpu_softmax(black_box(&logits)).expect("gpu_softmax failed"));
            });
        }
    }
    group.finish();
}

fn bench_gpu_top_k_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_top_k_filter");

    for &batch_size in &[1, 8, 32, 64] {
        for &vocab_size in &[32_000, 128_000] {
            let probs = make_probs_tensor(batch_size, vocab_size);
            let label = format!("b{batch_size}_v{vocab_size}_k50");
            group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
                b.iter(|| {
                    gpu_top_k_filter(black_box(&probs), 50).expect("gpu_top_k_filter failed")
                });
            });
        }
    }
    group.finish();
}

fn bench_gpu_top_p_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_top_p_filter");

    for &batch_size in &[1, 8, 32, 64] {
        for &vocab_size in &[32_000, 128_000] {
            let probs = make_probs_tensor(batch_size, vocab_size);
            let label = format!("b{batch_size}_v{vocab_size}_p0.9");
            group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
                b.iter(|| {
                    gpu_top_p_filter(black_box(&probs), 0.9).expect("gpu_top_p_filter failed")
                });
            });
        }
    }
    group.finish();
}

fn bench_gpu_multinomial_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_multinomial_sample");

    for &batch_size in &[1, 8, 32, 64] {
        for &vocab_size in &[32_000, 128_000] {
            let probs = make_probs_tensor(batch_size, vocab_size);
            let rand_vals = make_rand_vals(batch_size);
            let label = format!("b{batch_size}_v{vocab_size}");
            group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
                b.iter(|| {
                    gpu_multinomial_sample(black_box(&probs), black_box(&rand_vals))
                        .expect("gpu_multinomial_sample failed")
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    cpu_sampling,
    bench_sample_greedy,
    bench_sample_top_k_top_p,
    bench_log_softmax,
    bench_sample_with_logprobs,
);

criterion_group!(
    tensor_sampling,
    bench_gpu_argmax,
    bench_gpu_softmax,
    bench_gpu_top_k_filter,
    bench_gpu_top_p_filter,
    bench_gpu_multinomial_sample,
);

criterion_main!(cpu_sampling, tensor_sampling);
