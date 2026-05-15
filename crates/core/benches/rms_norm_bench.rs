//! Tier-1 per-kernel bench for `rms_norm_cuda_pooled`.
//!
//! Three norm sites per Qwen3 layer (input_norm, post_attention_norm,
//! plus a final_norm at the top of the decode forward) × 36 layers =
//! 73 calls/token. Each shape `[c, 1, hidden]`. Catches regressions in
//! fused RMSNorm path (variance + scale + weight in one kernel) vs
//! candle's multi-kernel fallback.

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const CONCURRENCIES: &[usize] = &[1, 4, 8];
const DTYPES: &[DType] = &[DType::BF16, DType::F16];
// Hidden-size sweep covers Qwen3-1.5B/4B/8B/14B and Llama-3 model families.
const HIDDENS: &[usize] = &[1536, 2560, 4096, 5120];
const EPS: f32 = 1e-6;

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_rms_norm(c: &mut Criterion) {
    let Ok(device) = Device::cuda_if_available(0) else {
        eprintln!("skipping rms_norm_bench: no CUDA");
        return;
    };
    if !device.is_cuda() {
        return;
    }

    let mut group = c.benchmark_group("rms_norm_pooled");
    group.sample_size(30);

    for &dtype in DTYPES {
        for &hidden in HIDDENS {
            let weight = Tensor::ones(hidden, dtype, &device).unwrap();
            for &m in CONCURRENCIES {
                let xs = Tensor::randn(0.0f32, 1.0, (m, 1, hidden), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                for _ in 0..3 {
                    let _ =
                        vllm_core::cuda_kernels::rms_norm_cuda_pooled(&xs, &weight, EPS).unwrap();
                }
                sync(&device);

                let dtype_label = match dtype {
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    _ => unreachable!(),
                };
                let label = format!("c{m}/h{hidden}/{dtype_label}");
                group.bench_with_input(BenchmarkId::new("pooled", label), &m, |b, _| {
                    b.iter(|| {
                        let y = vllm_core::cuda_kernels::rms_norm_cuda_pooled(
                            black_box(&xs),
                            black_box(&weight),
                            EPS,
                        )
                        .unwrap();
                        let _ = y.dim(0).unwrap();
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rms_norm);
criterion_main!(benches);
