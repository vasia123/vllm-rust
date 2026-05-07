//! Sampling pipeline (CPU per-seq path) bench.
//!
//! Mirrors the per-sequence sampling loop at
//! `crates/core/src/engine/helpers.rs:1029-1063` — the CPU fallback that
//! handles every batched-decode request when the GPU `gpu_sample_batch`
//! call is unavailable or rejected. This is the suspected hot loop behind
//! the c=4 throughput cliff (Stage 13-E.1 hypothesis).
//!
//! Three flavours measured against vocab_size=152_064 (Qwen3-4B):
//!
//! - `greedy_per_seq` — per-seq argmax (current "fast" CPU path).
//! - `full_per_seq`   — per-seq `sampling::sample` with top-k/top-p.
//! - `with_penalties` — adds frequency / presence / repetition penalties
//!   on top of full sampling, which is what production traffic uses.
//!
//! Pure CPU. Numbers are relative; the gate is "≤ 2% slower than the
//! prior commit at the same batch size".

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::sampling::{sample, SamplerState, SamplingParams};

const VOCAB: usize = 152_064;

fn make_logits_batch(batch: usize) -> Vec<f32> {
    let total = batch * VOCAB;
    let mut v = Vec::with_capacity(total);
    for i in 0..total {
        v.push((i as f32 * 0.0007).sin() * 4.0);
    }
    v
}

fn bench_greedy_per_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_greedy_per_seq");
    for &batch in &[1usize, 4, 8, 16, 32, 64] {
        let logits = make_logits_batch(batch);
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter(|| {
                for i in 0..batch {
                    let slice = &logits[i * VOCAB..(i + 1) * VOCAB];
                    let token = slice
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0);
                    black_box(token);
                }
                black_box(&logits);
            });
        });
    }
    group.finish();
}

fn bench_full_per_seq(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_full_per_seq");
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        ..Default::default()
    };
    for &batch in &[1usize, 4, 8, 16, 32, 64] {
        let logits = make_logits_batch(batch);
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter(|| {
                for i in 0..batch {
                    let mut state = SamplerState::new(Some(42));
                    let slice = &logits[i * VOCAB..(i + 1) * VOCAB];
                    let r = sample(slice, &params, &[], &mut state, None, &[]);
                    black_box(r);
                }
            });
        });
    }
    group.finish();
}

fn bench_with_penalties(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_with_penalties");
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.1,
        frequency_penalty: 0.5,
        presence_penalty: 0.5,
        ..Default::default()
    };
    let history: Vec<u32> = (0..64).map(|i| (i * 17) as u32 % 32_000).collect();
    for &batch in &[1usize, 8, 32] {
        let logits = make_logits_batch(batch);
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter(|| {
                for i in 0..batch {
                    let mut state = SamplerState::new(Some(42));
                    let slice = &logits[i * VOCAB..(i + 1) * VOCAB];
                    let r = sample(slice, &params, &history, &mut state, None, &[]);
                    black_box(r);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_greedy_per_seq,
    bench_full_per_seq,
    bench_with_penalties,
);
criterion_main!(benches);
