//! Stage 15.D — `marlin_tile_mma` (v1 software scaffold) cost curve.
//!
//! Sweeps the v1 dispatcher across the full Qwen3-4B-AWQ MLP-up shape
//! (K = 4096, N = 11008, group_size = 128) with the same M values the
//! `awq_marlin_path_bench` covers. Two purposes:
//!
//! 1. **Regression guard for the v1 software path**: any future change
//!    that accidentally regresses the (already slow) software dot-product
//!    surfaces here.
//! 2. **Anchor point for the tensor-core MMA work**: when Stage 15.D-body
//!    replaces the per-thread software dot-product with the
//!    `mma.m16n8k16` tensor-core path, this bench records the lift. The
//!    "before" numbers from the software scaffold establish the baseline
//!    that the tensor-core variant needs to beat — and the gap to the
//!    existing `awq_marlin_path_bench` numbers (the production kernel)
//!    quantifies how much of the gap tensor cores need to close.
//!
//! The kernel under test is dormant in production — `MarlinLinear`
//! still routes through `marlin_gemm` (Stage 13-J kt_mloop family). This
//! bench imports the kernel directly via `dispatch_marlin_tile_mma_v1`.

#![cfg(all(feature = "cuda-kernels", feature = "marlin"))]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device, Tensor};
use vllm_core::quantization::awq_marlin::awq_to_marlin_tile_repack_cpu;
use vllm_core::quantization::marlin_tile_cuda::dispatch_marlin_tile_mma_v1;

fn make_inputs(
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    device: &Device,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let num_groups = k / group_size;
    let packed_n = n / 8;

    // Empty / zero-valued AWQ qweight (the kernel still walks the full K
    // reduction, so memory-traffic shape matches the live model).
    let awq_zero = Tensor::zeros((k, packed_n), DType::U32, &Device::Cpu).expect("awq qweight");
    let tile_full = awq_to_marlin_tile_repack_cpu(&awq_zero, k, n).expect("repack");
    let b_tile = tile_full
        .flatten_all()
        .expect("flatten")
        .to_device(device)
        .expect("upload b");

    let scales = Tensor::ones((num_groups, n), DType::BF16, device).expect("scales");
    let qzeros = Tensor::zeros((num_groups, packed_n), DType::U32, device).expect("qzeros");
    let input = Tensor::ones((m, k), DType::BF16, device).expect("input");

    (input, b_tile, scales, qzeros)
}

fn run_one(input: &Tensor, b_tile: &Tensor, scales: &Tensor, qzeros: &Tensor, n: usize, g: usize) {
    let y = dispatch_marlin_tile_mma_v1(input, b_tile, scales, qzeros, n, g)
        .expect("tile_mma dispatch");
    black_box(y);
}

fn bench_dispatch_sweep(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping marlin_tile_mma_path_bench: no CUDA device");
        return;
    };
    if !device.is_cuda() {
        eprintln!("skipping marlin_tile_mma_path_bench: CUDA device not initialised");
        return;
    }

    const GROUP_SIZE: usize = 128;

    // All Qwen3-4B (Qwen3 4B Base) AWQ-quantized linear shapes — the model
    // fires each of these ~36 times (one per layer) per decode token.
    // Stage 15.E.3 negative result hypothesised that the microbench's
    // single-shape sweep (K=4096 N=11008 MLP-up) didn't represent the real
    // workload; this bench surfaces whether software wins on the smaller
    // q/k/v_proj and o_proj shapes too, or only on the largest MLP layers.
    //
    // Shapes (hidden=2560, intermediate=11008, num_q_heads=20, num_kv_heads=8,
    // head_dim=128):
    //
    //   q_proj         K=2560  N=2560   (q_heads * head_dim)
    //   k_proj/v_proj  K=2560  N=1024   (kv_heads * head_dim)  — ALSO N % 64 == 0
    //   o_proj         K=2560  N=2560
    //   gate/up_proj   K=2560  N=11008
    //   down_proj      K=11008 N=2560
    //   lm_head        K=2560  N=151936 — non-quantized in Qwen3-4B-AWQ
    //
    // Skip k/v_proj (N=1024 still mult of 64 but small enough that
    // q_proj = k_proj×2.5 is informative on its own). Skip lm_head (not
    // quantized).
    let shapes: &[(usize, usize, &str)] = &[
        (2560, 2560, "q_o"),         // q_proj / o_proj
        (2560, 11008, "gate_up"),    // gate_proj / up_proj
        (11008, 2560, "down"),       // down_proj
        (4096, 11008, "mlp_up_old"), // 15.D microbench shape (kept for trend reference)
    ];

    let mut group = c.benchmark_group("marlin_tile_mma_dispatch_qwen3");
    group.sample_size(10);

    for &(k, n, label) in shapes {
        // Decode-side M values that hit the hybrid threshold.
        for &m in &[1usize, 4, 8] {
            let (input, b_tile, scales, qzeros) = make_inputs(m, k, n, GROUP_SIZE, &device);
            run_one(&input, &b_tile, &scales, &qzeros, n, GROUP_SIZE);
            let id = format!("{label}_K{k}_N{n}_M{m}");
            group.bench_with_input(BenchmarkId::new("shape", id), &m, |b, _| {
                b.iter(|| {
                    run_one(
                        black_box(&input),
                        black_box(&b_tile),
                        black_box(&scales),
                        black_box(&qzeros),
                        n,
                        GROUP_SIZE,
                    );
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_dispatch_sweep);
criterion_main!(benches);
