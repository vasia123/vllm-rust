//! Criterion benchmarks for quantization operations.
//!
//! Covers BitsAndBytes NF4/INT8 quantize, dequantize, and forward pass,
//! plus GGUF dequantization -- all running on CPU.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use vllm_core::quantization::{
    quantize_int8, quantize_nf4, unpack_nf4, BitsAndBytesLinear, BnbQuantType, QuantizedLinear,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate deterministic weight data of the given size, values in [-1, 1].
fn make_weight_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 0.017).sin())).collect()
}

/// Create and load an NF4 linear layer with quantized weights.
fn make_nf4_linear(
    in_features: usize,
    out_features: usize,
    block_size: usize,
) -> BitsAndBytesLinear {
    let weight_data = make_weight_data(in_features * out_features);
    let (packed, absmax_data) = quantize_nf4(&weight_data, block_size);

    let mut linear = BitsAndBytesLinear::new(
        in_features,
        out_features,
        false,
        BnbQuantType::NF4,
        block_size,
        &Device::Cpu,
    )
    .expect("failed to create NF4 linear");

    let packed_len = packed.len();
    let absmax_len = absmax_data.len();
    let mut weights = HashMap::new();
    weights.insert(
        "weight".to_string(),
        Tensor::from_vec(packed, (packed_len,), &Device::Cpu).expect("packed tensor"),
    );
    weights.insert(
        "absmax".to_string(),
        Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).expect("absmax tensor"),
    );
    linear.load_weights(&weights).expect("load weights");
    linear
}

/// Create and load an INT8 linear layer with quantized weights.
fn make_int8_linear(
    in_features: usize,
    out_features: usize,
    block_size: usize,
) -> BitsAndBytesLinear {
    let weight_data = make_weight_data(in_features * out_features);
    let (quantized, absmax_data) = quantize_int8(&weight_data, block_size);

    let mut linear = BitsAndBytesLinear::new(
        in_features,
        out_features,
        false,
        BnbQuantType::INT8,
        block_size,
        &Device::Cpu,
    )
    .expect("failed to create INT8 linear");

    let absmax_len = absmax_data.len();
    let mut weights = HashMap::new();
    weights.insert(
        "weight".to_string(),
        Tensor::from_vec(quantized, (out_features, in_features), &Device::Cpu)
            .expect("weight tensor"),
    );
    weights.insert(
        "absmax".to_string(),
        Tensor::from_vec(absmax_data, (absmax_len,), &Device::Cpu).expect("absmax tensor"),
    );
    linear.load_weights(&weights).expect("load weights");
    linear
}

// ---------------------------------------------------------------------------
// NF4 quantize/dequantize benchmarks
// ---------------------------------------------------------------------------

fn bench_quantize_nf4(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_nf4");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let n = in_f * out_f;
        let data = make_weight_data(n);
        let label = format!("{in_f}x{out_f}");
        group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
            b.iter(|| quantize_nf4(black_box(&data), 64));
        });
    }
    group.finish();
}

fn bench_unpack_nf4(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpack_nf4");

    for &n in &[65_536, 1_048_576, 16_777_216] {
        let data = make_weight_data(n);
        let (packed, _) = quantize_nf4(&data, 64);
        let label = format!("{n}");
        group.bench_with_input(BenchmarkId::new("elements", &label), &label, |b, _| {
            b.iter(|| unpack_nf4(black_box(&packed), n));
        });
    }
    group.finish();
}

fn bench_nf4_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("nf4_dequantize");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let linear = make_nf4_linear(in_f, out_f, 64);
        let label = format!("{in_f}x{out_f}");
        group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
            b.iter(|| linear.dequantize().expect("dequantize failed"));
        });
    }
    group.finish();
}

fn bench_nf4_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("nf4_forward");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let linear = make_nf4_linear(in_f, out_f, 64);

        for &batch_size in &[1, 8] {
            let x =
                Tensor::ones(&[batch_size, in_f], DType::F32, &Device::Cpu).expect("input tensor");
            let label = format!("{in_f}x{out_f}_b{batch_size}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| linear.forward(black_box(&x)).expect("forward failed"));
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// INT8 quantize/dequantize benchmarks
// ---------------------------------------------------------------------------

fn bench_quantize_int8(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_int8");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let n = in_f * out_f;
        let data = make_weight_data(n);
        let label = format!("{in_f}x{out_f}");
        group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
            b.iter(|| quantize_int8(black_box(&data), 64));
        });
    }
    group.finish();
}

fn bench_int8_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_dequantize");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let linear = make_int8_linear(in_f, out_f, 64);
        let label = format!("{in_f}x{out_f}");
        group.bench_with_input(BenchmarkId::new("size", &label), &label, |b, _| {
            b.iter(|| linear.dequantize().expect("dequantize failed"));
        });
    }
    group.finish();
}

fn bench_int8_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_forward");

    for &(in_f, out_f) in &[(256, 256), (1024, 1024), (4096, 4096)] {
        let linear = make_int8_linear(in_f, out_f, 64);

        for &batch_size in &[1, 8] {
            let x =
                Tensor::ones(&[batch_size, in_f], DType::F32, &Device::Cpu).expect("input tensor");
            let label = format!("{in_f}x{out_f}_b{batch_size}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| linear.forward(black_box(&x)).expect("forward failed"));
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Block size variation
// ---------------------------------------------------------------------------

fn bench_nf4_block_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("nf4_block_size");
    let n = 1024 * 1024;
    let data = make_weight_data(n);

    for &block_size in &[32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, &bs| {
                b.iter(|| quantize_nf4(black_box(&data), bs));
            },
        );
    }
    group.finish();
}

fn bench_int8_block_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_block_size");
    let n = 1024 * 1024;
    let data = make_weight_data(n);

    for &block_size in &[32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::new("block_size", block_size),
            &block_size,
            |b, &bs| {
                b.iter(|| quantize_int8(black_box(&data), bs));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    nf4_benches,
    bench_quantize_nf4,
    bench_unpack_nf4,
    bench_nf4_dequantize,
    bench_nf4_forward,
    bench_nf4_block_size,
);

criterion_group!(
    int8_benches,
    bench_quantize_int8,
    bench_int8_dequantize,
    bench_int8_forward,
    bench_int8_block_size,
);

criterion_main!(nf4_benches, int8_benches);
