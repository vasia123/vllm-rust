//! Tier-1 bench for KV cache scatter (`reshape_and_cache_cuda`).
//!
//! Decode-time KV write is called once per layer per forward — 36 layers
//! × `c` tokens means it ends up on the wall-clock per-token budget even
//! though the per-call cost is small. Before this bench landed there was
//! **no coverage** of the kernel, so regressions in the slot-mapping
//! gather path could go unnoticed until they showed up at e2e level.
//!
//! Shape: Qwen3-8B-exl3 (num_kv_heads=8, head_dim=128, block_size=16).
//! Layout: NHD (production default for Qwen3 paged_attention V2). Sweep
//! c ∈ {1, 4, 8} × dtype ∈ {BF16, F16}.

#![cfg(feature = "cuda-kernels")]

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::cuda_kernels::{reshape_and_cache_cuda, CudaCacheLayout};

const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 64;
const CONCURRENCIES: &[usize] = &[1, 4, 8];
const DTYPES: &[DType] = &[DType::BF16, DType::F16];

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_reshape_and_cache(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping kv_cache_write_bench: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("reshape_and_cache_nhd");
    group.sample_size(30);

    for &dtype in DTYPES {
        // Allocate cache buffers once per dtype, reuse across c sweep.
        // Cache size: 64 blocks × 16 × 8 × 128 × 2B = 2 MiB BF16.
        let key_cache = Tensor::randn(
            0.0f32,
            1.0,
            (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let value_cache = Tensor::randn(
            0.0f32,
            1.0,
            (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            &device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

        for &m in CONCURRENCIES {
            let key = Tensor::randn(0.0f32, 1.0, (m, NUM_KV_HEADS, HEAD_DIM), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let value = Tensor::randn(0.0f32, 1.0, (m, NUM_KV_HEADS, HEAD_DIM), &device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            // Distinct slots per token, distributed across blocks so the
            // kernel walks the same stride pattern production sees.
            let slot_ids: Vec<u32> = (0..m).map(|i| (i * BLOCK_SIZE) as u32).collect();
            let slot_mapping = Tensor::from_vec(slot_ids, (m,), &device).unwrap();

            for _ in 0..3 {
                reshape_and_cache_cuda(
                    &key,
                    &value,
                    &key_cache,
                    &value_cache,
                    &slot_mapping,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    BLOCK_SIZE,
                    CudaCacheLayout::NHD,
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
            group.bench_with_input(BenchmarkId::new("nhd", label), &m, |b, _| {
                b.iter(|| {
                    reshape_and_cache_cuda(
                        black_box(&key),
                        black_box(&value),
                        black_box(&key_cache),
                        black_box(&value_cache),
                        black_box(&slot_mapping),
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        BLOCK_SIZE,
                        CudaCacheLayout::NHD,
                    )
                    .unwrap();
                    // No DtoH copy is needed — the kernel returns Result<()>
                    // and writes inplace; the next iter's launch syncs the
                    // stream via dependency.
                });
            });
            sync(&device);
        }
    }
    group.finish();
}

criterion_group!(benches, bench_reshape_and_cache);
criterion_main!(benches);
