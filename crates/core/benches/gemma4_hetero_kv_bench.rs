//! Tier-1 bench for the Gemma 4 heterogeneous-KV cache padding path.
//!
//! The released Gemma 4 12B/31B use a different head_dim AND a different
//! KV-head count on full-attention layers vs sliding layers (12B: full =
//! 1 kv head @ head_dim 512, sliding = 8 kv heads @ head_dim 256). All
//! layers share one paged KV cache, so `QuantizedGemma4Attention` /
//! `Gemma4Attention` zero-pad K/V up to the shared cache stride
//! (`cache_num_kv_heads`, `cache_head_dim`) on write and slice back on
//! read. Those `Tensor::cat` (pad) and `narrow` (slice) ops sit on the
//! per-layer decode path, so they need coverage.
//!
//! This bench measures the write-side pad (head_dim then KV-head) and the
//! read-side narrow for a representative 12B full-attention decode step
//! (1 token), and contrasts it with the homogeneous case where both pads
//! collapse to a no-op early return — confirming non-Gemma-4 / sliding
//! layers pay nothing.
//!
//! Run on CPU: WSL2 blocks GPU profilers (nsys/ncu), and these are pure
//! memory-shape ops whose relative cost is stable on CPU.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Gemma 4 12B full-attention layer geometry (toy seq len for decode).
const Q_LEN: usize = 1;
const LAYER_KV_HEADS: usize = 1; // full-attention layer
const LAYER_HEAD_DIM: usize = 512; // global_head_dim
const CACHE_KV_HEADS: usize = 8; // max across layers (sliding count)
const CACHE_HEAD_DIM: usize = 512; // max head_dim

fn pad_last_dim(x: &Tensor, from: usize, to: usize) -> Tensor {
    if from == to {
        return x.clone();
    }
    let (b, h, s, _) = x.dims4().unwrap();
    let zeros = Tensor::zeros((b, h, s, to - from), x.dtype(), x.device()).unwrap();
    Tensor::cat(&[x, &zeros], 3).unwrap()
}

fn pad_kv_heads(x: &Tensor, from: usize, to: usize) -> Tensor {
    if from == to {
        return x.clone();
    }
    let (b, _, s, d) = x.dims4().unwrap();
    let zeros = Tensor::zeros((b, to - from, s, d), x.dtype(), x.device()).unwrap();
    Tensor::cat(&[x, &zeros], 1).unwrap()
}

fn bench_hetero_kv(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("gemma4_hetero_kv");

    for &dtype in &[DType::F32, DType::BF16] {
        // Full-attention layer K/V before cache write: [b, kv_heads, q, head_dim].
        let k = Tensor::zeros((1, LAYER_KV_HEADS, Q_LEN, LAYER_HEAD_DIM), dtype, &device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("write_pad_full_layer", format!("{dtype:?}")),
            &k,
            |bn, k| {
                bn.iter(|| {
                    let kp = pad_last_dim(black_box(k), LAYER_HEAD_DIM, CACHE_HEAD_DIM);
                    let kp = pad_kv_heads(&kp, LAYER_KV_HEADS, CACHE_KV_HEADS);
                    black_box(kp)
                })
            },
        );

        // Read-side: cache returns [b, cache_kv_heads, kv_len, cache_head_dim];
        // slice back to the layer's real KV-head count and head_dim.
        let cached = Tensor::zeros((1, CACHE_KV_HEADS, 4, CACHE_HEAD_DIM), dtype, &device).unwrap();
        group.bench_with_input(
            BenchmarkId::new("read_narrow_full_layer", format!("{dtype:?}")),
            &cached,
            |bn, cached| {
                bn.iter(|| {
                    let k = cached.narrow(1, 0, LAYER_KV_HEADS).unwrap();
                    let k = k.narrow(3, 0, LAYER_HEAD_DIM).unwrap();
                    black_box(k)
                })
            },
        );

        // Homogeneous (sliding / non-Gemma-4) layer: pads are no-ops.
        let k_homo =
            Tensor::zeros((1, CACHE_KV_HEADS, Q_LEN, CACHE_HEAD_DIM), dtype, &device).unwrap();
        group.bench_with_input(
            BenchmarkId::new("write_pad_homogeneous_noop", format!("{dtype:?}")),
            &k_homo,
            |bn, k| {
                bn.iter(|| {
                    let kp = pad_last_dim(black_box(k), CACHE_HEAD_DIM, CACHE_HEAD_DIM);
                    let kp = pad_kv_heads(&kp, CACHE_KV_HEADS, CACHE_KV_HEADS);
                    black_box(kp)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_hetero_kv);
criterion_main!(benches);
