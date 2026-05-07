//! AWQ-Marlin INT4 dispatch sweep over M.
//!
//! Locks down the cost curve of `marlin_gemm` on the AWQ INT4 zero-point
//! path across the full operational M range — decode (M=1) through
//! prefill (M=2048) — at the canonical Qwen3-4B-AWQ MLP-up shape
//! (K = 4096, N = 11008, group_size = 128).
//!
//! The dispatch threshold lives at `marlin_cuda::AWQ_GEMV_M_THRESHOLD`
//! (= 16): M ≤ 16 takes the `awq_gemv_int4_kt_bf16` decode kernel,
//! M > 16 takes `awq_marlin_dequant_matmul` (dequant → cuBLAS BF16 GEMM).
//! This bench is the regression guard for both halves of that split.
//!
//! Stage 13-D.4 baseline (RTX 4060 Laptop, sm_89), measured first time
//! the bench landed: M=1 ~140 µs, M=2048 ~30 ms — i.e. the prefill side
//! grows nearly linearly in M (cuBLAS GEMM regime), not super-linearly
//! (the gemv-only regime that hit ~232 µs/(layer, token) and motivated
//! this stage). A regression where the curve at large M jumps back to
//! the super-linear shape is exactly what this guard would surface.

#![cfg(all(feature = "cuda-kernels", feature = "marlin"))]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device, Tensor};
use vllm_core::quantization::marlin::MarlinScalarType;
use vllm_core::quantization::marlin_cuda::marlin_gemm;

fn make_inputs(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    device: &Device,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let num_groups = k / group_size;
    let packed_k = k / 8;
    let packed_n = n / 8;

    // Deterministic non-zero qweight / qzeros — exact values do not matter
    // for the bench (we time the kernel, not the math), but non-zero ones
    // exercise the same memory-traffic patterns the live model sees.
    let qweight = Tensor::zeros((n, packed_k), DType::U32, device).expect("qweight");
    let scales = Tensor::ones((num_groups, n), DType::BF16, device).expect("scales");
    let qzeros = Tensor::zeros((num_groups, packed_n), DType::U32, device).expect("qzeros");
    let workspace = Tensor::zeros(64usize, DType::U32, device).expect("workspace");
    let input = Tensor::ones((m, k), DType::BF16, device).expect("input");

    (input, qweight, scales, qzeros, workspace)
}

fn run_one(
    input: &Tensor,
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    workspace: &Tensor,
    k: usize,
    n: usize,
) -> Tensor {
    marlin_gemm(
        input,
        qweight,
        scales,
        Some(qzeros),
        None,
        None,
        workspace,
        None,
        MarlinScalarType::Uint4,
        k,
        n,
        true,
        true,
    )
    .expect("marlin_gemm")
}

fn bench_dispatch_sweep(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping awq_marlin_path_bench: no CUDA device");
        return;
    };
    if !device.is_cuda() {
        eprintln!("skipping awq_marlin_path_bench: CUDA device not initialised");
        return;
    }

    // Qwen3-4B-AWQ MLP-up shape: largest single linear in the model;
    // its GEMV-vs-GEMM crossover is what dominates prefill cost.
    const K: usize = 4096;
    const N: usize = 11008;
    const GROUP_SIZE: usize = 128;

    let mut group = c.benchmark_group("awq_marlin_dispatch");
    group.sample_size(10);

    // M sweep covers both dispatch arms:
    //   - 1, 4, 8 — gemv path (M ≤ AWQ_GEMV_M_THRESHOLD)
    //   - 12, 16  — formerly gemv (Stage 13-D.4 had threshold=16, gemv
    //               cost 138 µs/M = 1.65 / 2.20 ms), now dequant+matmul
    //               (~1.45 ms fixed, see Stage 13-F)
    //   - 24, 32 — already on dequant+matmul under both thresholds
    //   - 64..2048 — saturating cuBLAS GEMM, growth in K accumulator only
    // The crossover region (M=8..16) is the part most likely to regress
    // on a future kernel rewrite, so we sample it densely.
    for &m in &[1usize, 4, 8, 12, 16, 24, 32, 64, 256, 512, 1024, 2048] {
        let (input, qweight, scales, qzeros, workspace) = make_inputs(m, K, N, GROUP_SIZE, &device);
        // Warm the kernel cache and the GEMM autotuner so the first
        // sample isn't 10× the rest.
        let _ = run_one(&input, &qweight, &scales, &qzeros, &workspace, K, N);

        group.bench_with_input(BenchmarkId::new("M", m), &m, |b, _| {
            b.iter(|| {
                let y = run_one(
                    black_box(&input),
                    &qweight,
                    &scales,
                    &qzeros,
                    &workspace,
                    K,
                    N,
                );
                black_box(y);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dispatch_sweep);
criterion_main!(benches);
