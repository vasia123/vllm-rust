//! Phase 7 baseline: throughput bench for MoE routing and expert
//! dispatch primitives.
//!
//! Phase 7 of the architecture refactor migrates 25+ bespoke MoE
//! implementations onto [`vllm_core::moe::MoELayer`] and
//! [`vllm_core::moe::TopKRouter`]. Per-PR rule: throughput delta vs.
//! the baseline recorded here must be ≤ 2%.
//!
//! What this measures:
//!
//! - `topk_router_softmax` / `topk_router_sigmoid`: gating-only
//!   throughput at typical batch sizes.
//! - `moe_layer_naive`: end-to-end `MoELayer::forward` (per-token
//!   loop, used as the correctness fallback in the migration target).
//!
//! All benches run on CPU with f32 weights and no CUDA. Numbers are
//! relative — the gate is "≤ 2% slower than the previous commit",
//! not an absolute target. The HTML report under
//! `target/criterion/` is the source of truth.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use vllm_core::moe::{MoELayer, MoELayerConfig, MoERouter, RouterConfig, ScoringFunc, TopKRouter};

// Mixtral-8x7B is the canonical MoE shape the plan benchmarks against.
// Hidden / intermediate are scaled down so the bench finishes in
// reasonable time on CPU; ratios match production.
const HIDDEN: usize = 1024;
const INTERMEDIATE: usize = 3584;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;

fn build_router(scoring: ScoringFunc) -> TopKRouter {
    let cfg = RouterConfig {
        hidden_size: HIDDEN,
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        renormalize: true,
        scoring_func: scoring,
        use_grouped_topk: false,
        num_expert_groups: None,
        topk_per_group: None,
        routed_scaling_factor: 1.0,
    };
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    TopKRouter::new(cfg, vb).expect("router construction")
}

fn build_moe_layer() -> MoELayer {
    let cfg = MoELayerConfig {
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        renormalize: true,
        inplace: false,
        is_act_and_mul: true,
    };
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    MoELayer::new(cfg, vb).expect("MoELayer construction")
}

fn random_hidden(num_tokens: usize) -> Tensor {
    // A deterministic non-zero input so softmax/sigmoid don't degenerate.
    let data: Vec<f32> = (0..num_tokens * HIDDEN)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    Tensor::from_vec(data, (num_tokens, HIDDEN), &Device::Cpu).expect("input tensor")
}

fn bench_router(c: &mut Criterion) {
    for (label, scoring) in [
        ("softmax", ScoringFunc::Softmax),
        ("sigmoid", ScoringFunc::Sigmoid),
    ] {
        let router = build_router(scoring);
        let mut group = c.benchmark_group(format!("topk_router_{label}"));
        for &num_tokens in &[16usize, 64, 256, 1024] {
            let xs = random_hidden(num_tokens);
            group.bench_with_input(
                BenchmarkId::new("tokens", num_tokens),
                &num_tokens,
                |b, _| {
                    b.iter(|| router.route(black_box(&xs)).expect("route"));
                },
            );
        }
        group.finish();
    }
}

fn bench_moe_layer_naive(c: &mut Criterion) {
    let layer = build_moe_layer();
    let mut group = c.benchmark_group("moe_layer_naive");
    // Naive path is per-token; keep token counts modest so the bench
    // finishes in a reasonable time on CPU.
    for &num_tokens in &[16usize, 64, 256] {
        let xs = random_hidden(num_tokens);
        group.bench_with_input(
            BenchmarkId::new("tokens", num_tokens),
            &num_tokens,
            |b, _| {
                b.iter(|| layer.forward_naive(black_box(&xs)).expect("forward_naive"));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_router, bench_moe_layer_naive);
criterion_main!(benches);
