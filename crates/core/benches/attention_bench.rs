//! Criterion benchmarks for attention-related operations.
//!
//! Covers causal mask generation, RoPE (rotary position embedding) apply,
//! and ALiBi bias computation -- all running on CPU device.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device, Tensor};
use vllm_core::layers::{
    apply_alibi_bias, causal_mask, compute_alibi_slopes, AlibiAttentionBias, RotaryEmbedding,
};

// ---------------------------------------------------------------------------
// Causal mask generation
// ---------------------------------------------------------------------------

fn bench_causal_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_mask");

    for &seq_len in &[128, 512, 2048, 8192] {
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    causal_mask(black_box(seq_len), 0, DType::F32, &Device::Cpu)
                        .expect("causal_mask failed")
                });
            },
        );
    }
    group.finish();
}

fn bench_causal_mask_with_offset(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_mask_with_offset");

    // Decode scenario: seq_len=1 with varying KV cache offsets
    for &offset in &[128, 512, 2048, 8192] {
        group.bench_with_input(BenchmarkId::new("offset", offset), &offset, |b, &offset| {
            b.iter(|| {
                causal_mask(1, black_box(offset), DType::F32, &Device::Cpu)
                    .expect("causal_mask failed")
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Rotary position embedding (RoPE)
// ---------------------------------------------------------------------------

fn bench_rope_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_creation");

    for &head_dim in &[64, 128] {
        for &max_seq_len in &[2048, 8192] {
            let label = format!("d{head_dim}_s{max_seq_len}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| {
                    RotaryEmbedding::new(
                        black_box(head_dim),
                        black_box(max_seq_len),
                        10000.0,
                        DType::F32,
                        &Device::Cpu,
                    )
                    .expect("RotaryEmbedding::new failed")
                });
            });
        }
    }
    group.finish();
}

fn bench_rope_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_apply");
    let device = Device::Cpu;

    for &seq_len in &[128, 512, 2048] {
        let head_dim = 128;
        let num_heads = 32;
        let num_kv_heads = 8;
        let batch = 1;

        let rope = RotaryEmbedding::new(head_dim, 8192, 10000.0, DType::F32, &device)
            .expect("RotaryEmbedding::new failed");

        let q = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, head_dim), &device)
            .expect("failed to create q");
        let k = Tensor::randn(
            0.0f32,
            1.0,
            (batch, num_kv_heads, seq_len, head_dim),
            &device,
        )
        .expect("failed to create k");

        let label = format!("seq{seq_len}_h{num_heads}_d{head_dim}");
        group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
            b.iter(|| {
                rope.apply(black_box(&q), black_box(&k), 0)
                    .expect("rope apply failed")
            });
        });
    }
    group.finish();
}

fn bench_rope_apply_varlen(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope_apply_varlen");
    let device = Device::Cpu;

    for &total_tokens in &[128, 512, 2048] {
        let head_dim = 128;
        let num_heads = 32;
        let num_kv_heads = 8;

        let rope = RotaryEmbedding::new(head_dim, 8192, 10000.0, DType::F32, &device)
            .expect("RotaryEmbedding::new failed");

        let q = Tensor::randn(0.0f32, 1.0, (total_tokens, num_heads, head_dim), &device)
            .expect("failed to create q");
        let k = Tensor::randn(0.0f32, 1.0, (total_tokens, num_kv_heads, head_dim), &device)
            .expect("failed to create k");
        let positions: Vec<usize> = (0..total_tokens).collect();

        let label = format!("tokens{total_tokens}_h{num_heads}_d{head_dim}");
        group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
            b.iter(|| {
                rope.apply_varlen(black_box(&q), black_box(&k), black_box(&positions))
                    .expect("rope apply_varlen failed")
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// ALiBi (Attention with Linear Biases)
// ---------------------------------------------------------------------------

fn bench_compute_alibi_slopes(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_alibi_slopes");

    for &num_heads in &[8, 32, 64, 112] {
        group.bench_with_input(
            BenchmarkId::new("heads", num_heads),
            &num_heads,
            |b, &num_heads| {
                b.iter(|| compute_alibi_slopes(black_box(num_heads)));
            },
        );
    }
    group.finish();
}

fn bench_alibi_build_bias_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("alibi_build_bias_matrix");
    let device = Device::Cpu;

    for &num_heads in &[8, 32] {
        for &seq_len in &[128, 512, 2048] {
            let alibi = AlibiAttentionBias::new(num_heads, DType::F32, &device)
                .expect("AlibiAttentionBias::new failed");

            let label = format!("h{num_heads}_s{seq_len}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| {
                    alibi
                        .build_bias_matrix(black_box(seq_len), black_box(seq_len))
                        .expect("build_bias_matrix failed")
                });
            });
        }
    }
    group.finish();
}

fn bench_apply_alibi_bias(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_alibi_bias");
    let device = Device::Cpu;

    for &num_heads in &[8, 32] {
        for &seq_len in &[128, 512, 2048] {
            let slopes_vec = compute_alibi_slopes(num_heads);
            let slopes =
                Tensor::from_vec(slopes_vec, (num_heads,), &device).expect("slopes tensor");

            let attn_scores = Tensor::randn(0.0f32, 1.0, (1, num_heads, seq_len, seq_len), &device)
                .expect("attn_scores tensor");

            let label = format!("h{num_heads}_s{seq_len}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| {
                    apply_alibi_bias(black_box(&attn_scores), black_box(&slopes))
                        .expect("apply_alibi_bias failed")
                });
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Softmax on attention scores (simulated score computation)
// ---------------------------------------------------------------------------

fn bench_attention_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_softmax");
    let device = Device::Cpu;
    let num_heads = 32;
    let batch = 1;

    for &seq_len in &[128, 512, 2048] {
        // Simulate attention scores as [batch, heads, seq, seq]
        let scores = Tensor::randn(0.0f32, 1.0, (batch, num_heads, seq_len, seq_len), &device)
            .expect("scores tensor");

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                // Softmax along last dim (candle_nn softmax)
                candle_nn::ops::softmax_last_dim(black_box(&scores))
                    .expect("softmax_last_dim failed")
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    mask_benches,
    bench_causal_mask,
    bench_causal_mask_with_offset,
);

criterion_group!(
    rope_benches,
    bench_rope_creation,
    bench_rope_apply,
    bench_rope_apply_varlen,
);

criterion_group!(
    alibi_benches,
    bench_compute_alibi_slopes,
    bench_alibi_build_bias_matrix,
    bench_apply_alibi_bias,
);

criterion_group!(attn_benches, bench_attention_softmax,);

criterion_main!(mask_benches, rope_benches, alibi_benches, attn_benches);
