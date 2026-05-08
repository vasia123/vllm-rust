//! Stage A'.2 — isolated lm_head matmul perf bench.
//!
//! Production profile (post-15.D, c=8) shows lm_head taking 16-24 ms per
//! forward — ~5-8× the ~3-4 ms theoretical memory-bound minimum for
//! `[M=8, K=2560] @ [K=2560, N=151,936]` BF16 dense. This bench measures
//! candle's matmul at that shape across three layout permutations to
//! find which (if any) hit a fast cuBLAS path:
//!
//! 1. **Tied path** (current production): weight stored `[N, K]` (embedding
//!    layout), forward does `x.matmul(&w.t()?)`. This is what
//!    `TiedEmbeddingHead::forward` does in `qwen3_quantized.rs`.
//! 2. **Pre-contig transposed**: weight pre-transposed once at init
//!    `[K, N]` contiguous, forward `x.matmul(&w)`. Tested 2026-05-08 in
//!    e2e and **showed no improvement** (lm_head time unchanged); kept
//!    here so a future candle/cuBLAS upgrade can reveal if/when this
//!    path becomes faster.
//! 3. **Linear-style with bias**: candle_nn::Linear which internally
//!    does `.t()?` per forward but might invoke a different code path.
//!
//! Ground truth in microbench → guides whether the fix lives in:
//!   - candle config (e.g. `cublaslt` feature);
//!   - direct cuBLAS FFI bypassing candle;
//!   - or full custom kernel (worst case).
//!
//! Uses Qwen3-4B-AWQ shape: M ∈ {1, 4, 8}, K=2560, N=151936.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const K: usize = 2560;
const N: usize = 151_936;

fn make_inputs(m: usize, device: &Device) -> (Tensor, Tensor, Tensor) {
    // Activations [M, K] BF16
    let x = Tensor::randn(0.0f32, 1.0, (m, K), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    // Embedding-shaped weight [N, K] BF16 (this is how candle_nn::Embedding
    // and the tied_word_embeddings path stores it).
    let w_emb = Tensor::randn(0.0f32, 0.02, (N, K), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    // Pre-transposed weight [K, N] BF16, contiguous.
    let w_kn = w_emb.t().unwrap().contiguous().unwrap();
    (x, w_emb, w_kn)
}

fn bench_lm_head_paths(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping lm_head_matmul_bench: no CUDA device");
        return;
    };
    if !device.is_cuda() {
        eprintln!("skipping lm_head_matmul_bench: CUDA device not initialised");
        return;
    }

    let mut group = c.benchmark_group("lm_head_matmul");
    group.sample_size(20);

    for &m in &[1usize, 4, 8] {
        let (x, w_emb, w_kn) = make_inputs(m, &device);
        // Pre-warm GPU clocks
        for _ in 0..3 {
            let _ = x.matmul(&w_emb.t().unwrap()).unwrap();
            let _ = x.matmul(&w_kn).unwrap();
        }

        // Path 1: production (tied) — `x @ w_emb.t()?` per forward.
        group.bench_with_input(BenchmarkId::new("tied_view_t", m), &m, |b, _| {
            b.iter(|| {
                let y = black_box(&x).matmul(&w_emb.t().unwrap()).unwrap();
                // Force sync: collect first scalar (tiny but real DtoH).
                let _ = y.dim(0).unwrap();
            });
        });

        // Path 2: pre-transposed contig.
        group.bench_with_input(BenchmarkId::new("contig_kn", m), &m, |b, _| {
            b.iter(|| {
                let y = black_box(&x).matmul(&w_kn).unwrap();
                let _ = y.dim(0).unwrap();
            });
        });

        // Path 3: candle_nn::Linear (which does its own .t() per forward
        // but may invoke a slightly different code path; included for
        // confirmation).
        let linear = candle_nn::Linear::new(w_emb.clone(), None);
        group.bench_with_input(BenchmarkId::new("candle_nn_linear", m), &m, |b, _| {
            b.iter(|| {
                let y = candle_core::Module::forward(&linear, black_box(&x)).unwrap();
                let _ = y.dim(0).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_lm_head_paths);
criterion_main!(benches);
