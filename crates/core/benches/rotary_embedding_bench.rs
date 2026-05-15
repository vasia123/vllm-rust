//! Tier-1 per-kernel bench for `rotary_embedding_cuda_pooled`.
//!
//! Called once per layer in the decode forward (after Q/K projections,
//! before paged_attention). Qwen3 shape: Q `[c, 32, 128]`, K `[c, 8, 128]`,
//! head_dim=128. Pool-backed path (POOL_MAX_NUM_TOKENS=64) covers c ≤ 64.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const ROT_DIM: usize = 128;
const MAX_POS: usize = 4096;
const CONCURRENCIES: &[usize] = &[1, 4, 8];
const DTYPES: &[DType] = &[DType::BF16, DType::F16];

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn build_cos_sin(dev: &Device) -> Tensor {
    // Standard interleaved cos/sin cache: shape [max_pos, rot_dim].
    // The kernel requires F32 regardless of Q/K dtype (see
    // `cuda_kernels::rope_inplace` precondition).
    Tensor::randn(0.0f32, 1.0, (MAX_POS, ROT_DIM), dev).unwrap()
}

fn bench_rotary(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping rotary_embedding_bench: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("rotary_embedding_pooled");
    group.sample_size(30);

    let cos_sin = build_cos_sin(&device);
    for &dtype in DTYPES {
        for &m in CONCURRENCIES {
            let q = Tensor::randn(0.0f32, 1.0, (m, NUM_HEADS, HEAD_DIM), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let k = Tensor::randn(0.0f32, 1.0, (m, NUM_KV_HEADS, HEAD_DIM), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            // Positions are i64 in candle / U32 for our kernel. Use U32.
            let positions =
                Tensor::from_vec((0..m as u32).collect::<Vec<_>>(), m, &device).unwrap();

            for _ in 0..3 {
                let _ = vllm_core::cuda_kernels::rotary_embedding_cuda_pooled(
                    &q,
                    &k,
                    &positions,
                    &cos_sin,
                    ROT_DIM,
                    HEAD_DIM,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    true,
                )
                .unwrap();
            }
            sync(&device);

            let dtype_label = match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => unreachable!(),
            };
            let label = format!("c{m}/{dtype_label}");
            group.bench_with_input(BenchmarkId::new("neox", label), &m, |b, _| {
                b.iter(|| {
                    let (q_out, k_out) = vllm_core::cuda_kernels::rotary_embedding_cuda_pooled(
                        black_box(&q),
                        black_box(&k),
                        black_box(&positions),
                        black_box(&cos_sin),
                        ROT_DIM,
                        HEAD_DIM,
                        NUM_HEADS,
                        NUM_KV_HEADS,
                        true,
                    )
                    .unwrap();
                    let _ = q_out.dim(0).unwrap();
                    let _ = k_out.dim(0).unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rotary);
criterion_main!(benches);
