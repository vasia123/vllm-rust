//! Tier 1 — native I-quant (IQ) dequant + linear forward baseline.
//!
//! candle has no I-quant path, so the Unsloth "UD" GGUFs (IQ2_XS / IQ2_S /
//! IQ3_XXS / IQ3_S / IQ4_XS) run through our own `IqLinear`: the weight stays
//! I-quant-resident on the device and the forward dequantizes it to a dense
//! f32 matrix (`kernels/iq_dequant.cu` on CUDA, the scalar port on CPU) then
//! matmuls — O(out·in) dequant work per forward, like the OLD GGUF dense path.
//!
//! This records the BEFORE the CLAUDE.md perf rule requires: the current
//! dequant-then-matmul `IqLinear`. The planned fused-MMVQ kernel (skip
//! materializing the dense weight; see the DEFERRED task) adds its AFTER
//! variant here for an honest diff.
//!
//! Byte VALUES don't affect dequant cost (per-block work is data-independent
//! bar cheap sign flips), so a zero-filled block buffer is a faithful perf
//! proxy without a real IQ encoder.
//!
//! IQ3_XXS is the dominant type in the Gemma-4-12B UD checkpoint (187 of 372
//! quantized tensors). Shapes mirror q/o_proj (square) and a wide FFN
//! projection; decode (M=1) and prefill (M=32).

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vllm_core::quantization::gguf::iq::{dequantize_iq, IqType};
use vllm_core::quantization::gguf::IqLinear;
use vllm_core::quantization::QuantizedLinear;

// (out_features, in_features) — `[out, in]` row layout; `in % 256 == 0`.
const SHAPES: &[(usize, usize, &str)] = &[(4096, 4096, "4096x4096"), (14336, 4096, "14336x4096")];
const TOKENS: &[usize] = &[1, 32];

const IQ: IqType = IqType::Iq3Xxs;

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
        let n_elem = out * inn;
        let lin = IqLinear::new(bytes.clone(), IQ, out, inn, None);

        // Dequant-only cost (the per-forward weight expansion).
        for _ in 0..3 {
            let _ = dequantize_iq(&bytes, IQ, n_elem).unwrap();
        }
        sync(device);
        group.bench_with_input(
            BenchmarkId::new("dequant_only", shape_name),
            &out,
            |bn, _| {
                bn.iter(|| {
                    let y = dequantize_iq(black_box(&bytes), IQ, n_elem).unwrap();
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            },
        );

        // Full IqLinear forward (dequant + matmul) at decode/prefill.
        for &m in TOKENS {
            let x = make_input(m, inn, device);
            let mode = if m == 1 { "decode" } else { "prefill" };
            let id = format!("{shape_name}/{mode}_m{m}");
            for _ in 0..3 {
                let _ = lin.forward(&x).unwrap();
            }
            sync(device);
            group.bench_with_input(BenchmarkId::new("iqlinear_forward", &id), &m, |bn, _| {
                bn.iter(|| {
                    let y = lin.forward(black_box(&x)).unwrap();
                    let _ = y.dim(0).unwrap();
                    sync(device);
                });
            });
        }
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
