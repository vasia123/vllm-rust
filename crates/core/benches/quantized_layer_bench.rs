//! Tier 2.b — production-realistic single-decoder-layer bench.
//!
//! Wraps `QuantizedQwen3DecoderLayer::forward_decode_batch_with_shared`
//! on a **real** Qwen3-8B-exl3 layer (loaded once at setup time and
//! shared across all concurrency variants). Sweep c ∈ {1, 4, 8} matches
//! `bench_decode.py`.
//!
//! Why this is needed alongside Tier 1 kernel micros + Tier 2.a synthetic
//! chain:
//!
//! The current c=4 regression (cuda-default → -17 % vs
//! cuda-kernels,flashinfer) is invisible to micro-benches because the
//! affected dispatch lives inside the quantised linear layer
//! (`Exl3GemmOp` / `MarlinLinear` selector / AWQ fallback). A synthetic
//! `mini_decoder_layer_forward` substitutes `half_matmul_pooled` and
//! cannot see those side-effects. This bench loads the **real** EXL3
//! weights so each Q/K/V/O/Gate/Up/Down projection exercises the same
//! kernel selector path that production decode runs.
//!
//! Setup cost: model load ≈ 30 s, ~5 GB GPU memory. `b.iter` measures
//! one decoder layer forward (~700 µs at c=1, ~1500 µs at c=8 — easily
//! within criterion's default sample budget).

use std::sync::OnceLock;

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::config::ModelConfig;
use vllm_core::engine::DecodeSequenceMetadata;
use vllm_core::kv_cache::config::CacheConfig;
use vllm_core::kv_cache::{KVCacheDtype, KVCacheManager};
use vllm_core::layers::attention::build_decode_batch_shared;
use vllm_core::loader;
use vllm_core::models::qwen3_quantized::QuantizedQwen3ForCausalLM;

const MODEL_ID: &str = "turboderp/Qwen3-8B-exl3";
const REVISION: &str = "4.0bpw";
const CONCURRENCIES: &[usize] = &[1, 4, 8];
const NUM_BLOCKS: usize = 64;
const BLOCK_SIZE: usize = 16;
// Each sequence gets a few prefilled tokens so paged_attention has real
// data to read — not just an empty cache lookup.
const SEQLEN_OFFSET: usize = 32;

/// Holder for the loaded model + a cache manager prebuilt for the
/// widest c we benchmark.
struct BenchSetup {
    model: QuantizedQwen3ForCausalLM,
    cfg: ModelConfig,
    device: Device,
}

/// Load once per process — criterion replays `b.iter` many times so
/// amortising the ~30 s load across all benches in this file matters.
fn setup() -> Option<&'static BenchSetup> {
    static CACHE: OnceLock<Option<BenchSetup>> = OnceLock::new();
    CACHE
        .get_or_init(|| match try_setup() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("skipping quantized_layer_bench: setup failed: {e}");
                None
            }
        })
        .as_ref()
}

fn try_setup() -> anyhow::Result<BenchSetup> {
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        anyhow::bail!("CUDA device required");
    }

    // Publish engine limits BEFORE building KVCacheManager / running
    // any forward. Production server does the same in main.rs:1921
    // — without this, `pool_worst_case_seq_len()` falls back to 1024
    // while pool slots may get reserved against a different size, and
    // paged_attention V2 reads a stride that doesn't match the actual
    // block_tables layout → CUDA_ERROR_UNKNOWN on first launch. See
    // ADR 0019 "Bug B" for the production-path version of the same
    // failure mode.
    const MAX_MODEL_LEN: usize = NUM_BLOCKS * BLOCK_SIZE; // 1024 here
    vllm_core::engine::engine_limits::set_max_model_len(MAX_MODEL_LEN);
    vllm_core::engine::engine_limits::set_max_seq_len_to_capture(MAX_MODEL_LEN);

    eprintln!("quantized_layer_bench: fetching {MODEL_ID} @ {REVISION}…");
    let files = loader::fetch_model_with_revision(MODEL_ID, REVISION)?;
    // EXL3 forces F16 — see the activation-dtype override in the
    // production loader. We mirror that here so the layer's linears
    // route through the exact same `Exl3GemmOp` instantiation as the
    // production decode forward.
    let dtype = DType::F16;
    // Single VarBuilder shared between (a) non-quant weights pulled by
    // `QuantizedQwen3ForCausalLM::new` (embeddings, norms) and (b) the
    // quantised-weight loader. This mirrors how server `main.rs` does
    // it — one mmap, one DAG of references into it. Going through
    // `create_quantized_loader(&files, ...)` would mmap a second time.
    let vb = loader::load_weights(&files.weights, dtype, &device)?;
    let weight_loader =
        vllm_core::quantization::create_weight_loader_with_params(vb.clone(), &files.quantization);

    eprintln!("quantized_layer_bench: building QuantizedQwen3ForCausalLM…");
    let model = QuantizedQwen3ForCausalLM::new(&files.config, vb, weight_loader.as_ref())?;

    Ok(BenchSetup {
        model,
        cfg: files.config,
        device,
    })
}

fn make_kv_cache_mgr(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
    let cache_cfg = CacheConfig {
        block_size: BLOCK_SIZE,
        num_blocks: NUM_BLOCKS,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        dtype: DType::F16,
        device: device.clone(),
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    };
    KVCacheManager::new(&cache_cfg).expect("kv cache manager")
}

fn make_sequences(c: usize) -> Vec<DecodeSequenceMetadata> {
    // Each sequence gets a unique block range so paged_attention reads
    // from non-overlapping cache regions (mirrors production).
    let mut out = Vec::with_capacity(c);
    let blocks_per_seq = SEQLEN_OFFSET.div_ceil(BLOCK_SIZE);
    for i in 0..c {
        let start = (i * blocks_per_seq) % NUM_BLOCKS;
        // `DecodeSequenceMetadata::block_ids` is `Vec<BlockId>` where
        // `BlockId = usize` — see `crates/core/src/kv_cache/block_pool.rs`.
        let block_ids: Vec<usize> = (0..blocks_per_seq)
            .map(|b| (start + b) % NUM_BLOCKS)
            .collect();
        // Slot for the *new* decode token: continues from seqlen_offset
        // in the last allocated block.
        let new_slot = (start + blocks_per_seq - 1) * BLOCK_SIZE + (SEQLEN_OFFSET % BLOCK_SIZE);
        out.push(DecodeSequenceMetadata {
            request_id: i as u64,
            seqlen_offset: SEQLEN_OFFSET,
            block_ids,
            slot_mapping: vec![new_slot],
        });
    }
    out
}

fn make_input_xs(c: usize, hidden: usize, dtype: DType, device: &Device) -> Tensor {
    Tensor::randn(0.0f32, 1.0, (c, 1, hidden), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_decoder_layer(c: &mut Criterion) {
    let Some(setup) = setup() else {
        return;
    };
    let BenchSetup { model, cfg, device } = setup;

    let mut group = c.benchmark_group("quantized_qwen3_layer");
    // Single-layer forward is short (≤1.5 ms) but allocates pool slots —
    // 20 samples is enough to land a stable mean without paying
    // criterion's default 100-sample overhead.
    group.sample_size(20);

    let mut kv = make_kv_cache_mgr(cfg, device);
    let dtype = DType::F16;

    for &m in CONCURRENCIES {
        let xs = make_input_xs(m, cfg.hidden_size, dtype, device);
        let sequences = make_sequences(m);
        let shared =
            build_decode_batch_shared(&sequences, device).expect("build_decode_batch_shared");

        // Warm: 5 forwards prime CUBLAS handles, EXL3 module cache,
        // OutputPool slot allocations.
        for _ in 0..5 {
            let _ = model
                .layer(0)
                .forward_decode_batch_with_shared(&xs, &sequences, &mut kv, 0, Some(&shared))
                .unwrap();
        }
        sync(device);

        group.bench_with_input(BenchmarkId::new("layer0_forward_decode", m), &m, |b, _| {
            b.iter(|| {
                let y = model
                    .layer(0)
                    .forward_decode_batch_with_shared(
                        black_box(&xs),
                        black_box(&sequences),
                        &mut kv,
                        0,
                        Some(black_box(&shared)),
                    )
                    .unwrap();
                // Tiny DtoH to force kernel completion before next iter.
                let _ = y.dim(0).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decoder_layer);
criterion_main!(benches);
