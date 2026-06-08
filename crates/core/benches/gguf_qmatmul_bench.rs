//! Tier 1 — GGUF quantized linear forward: dense-dequant baseline vs
//! candle fused `QMatMul` (MMVQ).
//!
//! The GGUF path used to dequantize the WHOLE weight matrix to a dense
//! f32 tensor on every forward and then run a plain `matmul` — O(out·in)
//! dequant work per token, dwarfing the matmul itself at decode (M=1).
//! The rework backs `GgufLinear` with candle's `QMatMul`, which keeps the
//! weight quantized-resident and fuses dequant into the matmul kernel
//! (CUDA MMVQ for M≤8, q8_1 GEMM otherwise; an equivalent fused CPU path
//! via `matmul_t` on the block storage).
//!
//! This bench records the BEFORE/AFTER pair the CLAUDE.md perf rule
//! requires:
//!   - `dense_dequant_matmul` — dequantize the QTensor to dense f32 each
//!     call, then `x @ wᵀ`. Faithful to the old `GgufLinear::forward`
//!     semantics (full dequant per forward).
//!   - `qmatmul` — `QMatMul::forward` on the same quantized weight.
//!
//! Shapes: a square projection (4096×4096, e.g. q/o_proj) and a wide FFN
//! projection (4096→14336, e.g. gate/up_proj), at decode (M=1) and
//! prefill (M=32). Q4_K — the most common K-quant (Q4_K_M).
//!
//! CPU always runs (candle quantized has a CPU path); the CUDA half is
//! gated on the `cuda` feature so `cargo bench` on a CPU-only checkout
//! still builds.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// (out_features, in_features) — llama.cpp / QMatMul row layout `[out, in]`.
const SHAPES: &[(usize, usize, &str)] = &[
    (4096, 4096, "4096x4096"),   // q_proj / o_proj
    (14336, 4096, "14336x4096"), // gate_proj / up_proj (wide FFN)
];
// M = number of rows in the activation (tokens). 1 = decode, 32 = prefill.
const TOKENS: &[usize] = &[1, 32];

fn make_qtensor(out: usize, inn: usize, device: &Device) -> QTensor {
    // Random dense weight in `[out, in]`, quantized to Q4_K. QK_K = 256,
    // so `in` must be a multiple of 256 — both shapes here satisfy that.
    let w = Tensor::randn(0.0f32, 1.0, (out, inn), device).unwrap();
    QTensor::quantize(&w, GgmlDType::Q4K).unwrap()
}

fn make_input(m: usize, inn: usize, device: &Device) -> Tensor {
    // [1, M, in] — leading batch dim mirrors a real decode/prefill call.
    Tensor::randn(0.0f32, 1.0, (1, m, inn), device)
        .unwrap()
        .to_dtype(candle_core::DType::F32)
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

/// Baseline: full dequant to dense f32 each call, then `x @ wᵀ`.
fn dense_dequant_matmul(x: &Tensor, q: &QTensor, device: &Device) -> Tensor {
    let w = q.dequantize(device).unwrap(); // [out, in], dense f32
    let wt = w.t().unwrap(); // [in, out]
    let (b, m, inn) = x.dims3().unwrap();
    let x2d = x.reshape((b * m, inn)).unwrap();
    let y = x2d.matmul(&wt).unwrap();
    y.reshape((b, m, w.dim(0).unwrap())).unwrap()
}

fn run_suite(c: &mut Criterion, device: &Device, tag: &str) {
    let mut group = c.benchmark_group(format!("gguf_linear_forward_{tag}"));
    group.sample_size(30);

    for &(out, inn, shape_name) in SHAPES {
        let q = make_qtensor(out, inn, device);
        let qmm = QMatMul::from_qtensor(make_qtensor(out, inn, device)).unwrap();

        for &m in TOKENS {
            let x = make_input(m, inn, device);
            let mode = if m == 1 { "decode" } else { "prefill" };
            let id = format!("{shape_name}/{mode}_m{m}");

            // Warm: prime CUBLAS / MMVQ kernel caches and any lazy alloc.
            for _ in 0..3 {
                let _ = dense_dequant_matmul(&x, &q, device);
                let _ = qmm.forward(&x).unwrap();
            }
            sync(device);

            group.bench_with_input(
                BenchmarkId::new("dense_dequant_matmul", &id),
                &m,
                |bn, _| {
                    bn.iter(|| {
                        let y = dense_dequant_matmul(black_box(&x), black_box(&q), device);
                        let _ = y.dim(0).unwrap();
                        sync(device);
                    });
                },
            );

            group.bench_with_input(BenchmarkId::new("qmatmul", &id), &m, |bn, _| {
                bn.iter(|| {
                    let y = qmm.forward(black_box(&x)).unwrap();
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            });
        }
    }
    group.finish();
}

fn bench_gguf_linear(c: &mut Criterion) {
    // CPU always — candle quantized has a CPU matmul_t path.
    run_suite(c, &Device::Cpu, "cpu");

    // CUDA when built with the `cuda` feature and a device is present.
    #[cfg(feature = "cuda")]
    if let Ok(dev) = Device::cuda_if_available(0) {
        if dev.is_cuda() {
            run_suite(c, &dev, "cuda");
        }
    }
}

criterion_group!(benches, bench_gguf_linear);
criterion_main!(benches);
