//! H2 probe for the marlin-feature regression on EXL3.
//!
//! Hypothesis: enabling `feature = "marlin"` changes how LLVM lays out
//! the `Box<dyn QuantizedLinear>` vtable / inline-call sites — adding the
//! `MarlinLinear` concrete type to the reachable set could perturb
//! branch prediction or indirect-jump costs on the EXL3 hot path. To
//! test, this bench measures the **pure trait-call overhead** of
//! `Box<dyn QuantizedLinear>::forward()` on a no-op mock that does
//! nothing but return its input — no GPU work, no allocation.
//!
//! Run the same bench under two feature sets:
//!   cargo bench --features cuda-kernels        -- exl3_dispatch
//!   cargo bench --features cuda-kernels,marlin -- exl3_dispatch
//!
//! If the second is measurably slower, H2 is supported. If both match,
//! the marlin regression lives elsewhere (PTX loading, scratch
//! contention, …) — falsifying H2 is just as useful as confirming it.
//!
//! CPU-runnable on purpose: the mock returns the input tensor unchanged,
//! so no CUDA device is required. This makes the bench cheap to run
//! across feature combinations.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use vllm_core::quantization::QuantizedLinear;

/// Minimal `QuantizedLinear` impl that clones its input. The clone is
/// cheap (refcount bump on the storage Arc — no data copy) and keeps the
/// optimiser from constant-folding the entire loop. We don't return a
/// fresh allocation because allocator noise would dominate the measure.
struct NopLinear {
    in_f: usize,
    out_f: usize,
}

impl QuantizedLinear for NopLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::F16
    }

    fn in_features(&self) -> usize {
        self.in_f
    }

    fn out_features(&self) -> usize {
        self.out_f
    }

    fn has_bias(&self) -> bool {
        false
    }
}

fn bench_trait_call(c: &mut Criterion) {
    let device = Device::Cpu;
    let x = Tensor::ones((1, 1, 4096), DType::F32, &device).unwrap();

    let linear: Box<dyn QuantizedLinear> = Box::new(NopLinear {
        in_f: 4096,
        out_f: 4096,
    });

    c.bench_function("trait_forward_call", |b| {
        b.iter(|| {
            let y = linear.forward(black_box(&x)).unwrap();
            black_box(y);
        });
    });
}

criterion_group!(benches, bench_trait_call);
criterion_main!(benches);
