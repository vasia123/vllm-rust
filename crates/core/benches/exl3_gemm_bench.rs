//! Tier-1 microbench for `exl3_gemm` at decode shapes.
//!
//! EXL3 dispatch goes through `Exl3Linear::forward → exl3_cuda::exl3_gemm`
//! (`crates/core/src/quantization/exl3.rs:239`,
//! `crates/core/src/quantization/exl3_cuda.rs:259`). This bench isolates
//! the kernel from the trait-object dispatch + reshape wrapping so a
//! per-call regression can be distinguished from a dispatch-side effect.
//!
//! Shapes mirror Qwen3-8B-exl3 decode (`hidden=4096`, `intermediate=12288`,
//! `bpw=4` for the 4.0bpw revision):
//! - qkv-like: `[c, 4096] @ [4096 -> 4096]`
//! - gate/up: `[c, 4096] @ [4096 -> 12288]`
//! - down:    `[c, 12288] @ [12288 -> 4096]`
//!
//! Trellis weights are random I16 — valid bit-patterns aren't required
//! since we measure kernel wall-clock, not output correctness. The same
//! random buffer is reused across `b.iter` so we measure steady-state
//! GEMM cost without first-call cuBLAS/PTX init.

#![cfg(feature = "cuda-kernels")]

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::quantization::exl3_cuda::{exl3_gemm, Exl3Codebook};

const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12_288;
const BPW: u32 = 4;
const CONCURRENCIES: &[usize] = &[1, 4, 8];

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

/// `(label, K, N)` — `K` is input feature dim, `N` is output.
const SHAPES: &[(&str, usize, usize)] = &[
    ("qkv_like", HIDDEN, HIDDEN),
    ("gate_or_up", HIDDEN, INTERMEDIATE),
    ("down", INTERMEDIATE, HIDDEN),
];

fn build_trellis(k: usize, n: usize, dev: &Device) -> Tensor {
    let k_blocks = k / 16;
    let n_blocks = n / 16;
    let last = 16 * BPW as usize;
    // Random I16 — bit patterns don't need to decode to valid codebook
    // indices for a wall-clock measurement; the kernel does identical work
    // either way.
    let elems = k_blocks * n_blocks * last;
    let data: Vec<i16> = (0..elems).map(|i| (i as i16).wrapping_mul(7919)).collect();
    Tensor::from_vec(data, (k_blocks, n_blocks, last), dev).unwrap()
}

fn bench_exl3_gemm(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping exl3_gemm_bench: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("exl3_gemm_decode");
    group.sample_size(30);

    let codebook = Exl3Codebook::from_flags(false, false);

    for &(shape_label, k, n) in SHAPES {
        let trellis = build_trellis(k, n, &device);
        let suh = Tensor::randn(0.0f32, 1.0, k, &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let svh = Tensor::randn(0.0f32, 1.0, n, &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        for &m in CONCURRENCIES {
            let a = Tensor::randn(0.0f32, 1.0, (m, k), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();

            for _ in 0..3 {
                let _ = exl3_gemm(&a, &trellis, Some(&suh), Some(&svh), BPW, codebook).unwrap();
            }
            sync(&device);

            let label = format!("c{m}/{shape_label}");
            group.bench_with_input(BenchmarkId::new("f16", label), &m, |b, _| {
                b.iter(|| {
                    let y = exl3_gemm(
                        black_box(&a),
                        black_box(&trellis),
                        Some(black_box(&suh)),
                        Some(black_box(&svh)),
                        BPW,
                        codebook,
                    )
                    .unwrap();
                    let _ = y.dim(0).unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_exl3_gemm);
criterion_main!(benches);
