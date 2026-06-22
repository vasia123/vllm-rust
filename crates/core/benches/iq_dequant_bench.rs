//! Tier 1 — native I-quant (IQ) matmul: dequant-then-matmul vs the q8_1 MMVQ.
//!
//! candle has no I-quant path, so the Unsloth "UD" GGUFs (IQ2_XS / IQ2_S /
//! IQ3_XXS / IQ3_S / IQ4_XS) run through our own `IqLinear`. This bench is the
//! BEFORE/AFTER the CLAUDE.md perf rule requires for the fused decode kernel:
//!
//! - `decode_dequant_matmul` (BEFORE): dequantize the whole weight to dense f32
//!   each forward, then matmul — O(out·in) work per token (the old path).
//! - `decode_fused_mmvq` (AFTER): `iq_matmul` — quantize the activation to q8_1
//!   once, then integer `__dp4a` dots over the I-quant bytes, no dense weight
//!   materialized (the path llama.cpp itself runs for I-quant decode).
//! - `prefill_dequant_matmul` (M=32): the dequant + GEMM path both decode and
//!   prefill share for large M (amortized over the prompt).
//!
//! Byte VALUES don't affect cost (per-block work is data-independent bar cheap
//! sign flips), so a zero-filled block buffer is a faithful perf proxy without
//! a real IQ encoder.
//!
//! IQ3_XXS is the dominant type in the Gemma-4-12B UD checkpoint (187 of 372
//! quantized tensors). Shapes mirror q/o_proj (square) and a wide FFN proj.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vllm_core::quantization::gguf::iq::{dequantize_iq, iq_matmul, IqType};

// (out_features, in_features) — `[out, in]` row layout; `in % 256 == 0`.
const SHAPES: &[(usize, usize, &str)] = &[(4096, 4096, "4096x4096"), (14336, 4096, "14336x4096")];

const IQ: IqType = IqType::Iq3Xxs;

/// BEFORE path: dequantize the whole weight to dense f32, then matmul.
fn dequant_then_matmul(bytes: &Tensor, x: &Tensor, out: usize, inn: usize) -> Tensor {
    let w = dequantize_iq(bytes, IQ, out * inn)
        .unwrap()
        .reshape((out, inn))
        .unwrap();
    x.matmul(&w.t().unwrap().contiguous().unwrap()).unwrap()
}

/// Zero-filled IQ block bytes for an `[out, in]` weight (perf-faithful).
fn make_iq_bytes(out: usize, inn: usize, device: &Device) -> Tensor {
    let n_blocks = out * inn / IQ.block_size();
    let bytes = vec![0u8; n_blocks * IQ.type_size()];
    let n = bytes.len();
    Tensor::from_vec(bytes, (n,), device).unwrap()
}

fn make_input(m: usize, inn: usize, device: &Device) -> Tensor {
    Tensor::randn(0.0f32, 1.0, (m, inn), device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
}

#[cfg(feature = "cuda")]
fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

#[cfg(not(feature = "cuda"))]
fn sync(_dev: &Device) {}

fn run_suite(c: &mut Criterion, device: &Device, tag: &str) {
    let mut group = c.benchmark_group(format!("iq3_xxs_{tag}"));
    group.sample_size(20);

    for &(out, inn, shape_name) in SHAPES {
        let bytes = make_iq_bytes(out, inn, device);

        // ── Decode (M=1): BEFORE (dequant+matmul) vs AFTER (fused GEMV) ──
        let x1 = make_input(1, inn, device);
        for _ in 0..3 {
            let _ = dequant_then_matmul(&bytes, &x1, out, inn);
            let _ = iq_matmul(&bytes, &x1, IQ, out, inn, 1).unwrap();
        }
        sync(device);
        group.bench_with_input(
            BenchmarkId::new("decode_dequant_matmul", shape_name),
            &out,
            |bn, _| {
                bn.iter(|| {
                    let y = dequant_then_matmul(black_box(&bytes), black_box(&x1), out, inn);
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("decode_fused_mmvq", shape_name),
            &out,
            |bn, _| {
                bn.iter(|| {
                    let y = iq_matmul(black_box(&bytes), black_box(&x1), IQ, out, inn, 1).unwrap();
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            },
        );

        // ── Prefill (M=32): the shared dequant+GEMM path ──
        let x32 = make_input(32, inn, device);
        for _ in 0..3 {
            let _ = dequant_then_matmul(&bytes, &x32, out, inn);
        }
        sync(device);
        group.bench_with_input(
            BenchmarkId::new("prefill_dequant_matmul", shape_name),
            &out,
            |bn, _| {
                bn.iter(|| {
                    let y = dequant_then_matmul(black_box(&bytes), black_box(&x32), out, inn);
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            },
        );
    }
    group.finish();
}

fn bench_iq(c: &mut Criterion) {
    run_suite(c, &Device::Cpu, "cpu");

    #[cfg(feature = "cuda")]
    if let Ok(dev) = Device::cuda_if_available(0) {
        if dev.is_cuda() {
            run_suite(c, &dev, "cuda");
        }
    }
}

criterion_group!(benches, bench_iq);
criterion_main!(benches);
