//! Microbench for paged_attention V1 / V2 across seq_len × partition_size.
//!
//! Purpose: derive the seq_len → partition_size adaptive selector empirically
//! instead of guessing.  Each row of the resulting Criterion report gives
//! μs/call for one (variant, seq_len) point on the operating box.
//!
//! Shape pinned to Qwen3-4B-AWQ (num_q_heads=32, num_kv_heads=8, head_dim=128,
//! block_size=16).  batch=1 — the only shape that matters for decode tok/s on
//! the kind of laptop hardware Stage 13-C targets.
//!
//! Run on the test box:
//!   cargo bench --features cuda-default -p vllm-core
//!     --bench paged_attention_bench
//!
//! Skips silently when no CUDA device is available so CI CPU runs do not
//! fail the bench gate.

#![cfg(feature = "cuda-kernels")]

use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::cuda_kernels::{
    paged_attention_cuda, paged_attention_v2_cuda_pooled,
    paged_attention_v2_cuda_with_partition_size, select_v2_partition_size,
};

// ─── Shape and grid pinned to Qwen3-4B-AWQ ──────────────────────────────────
// Single-sequence batch=1 (decode hot path).  Vary only `seq_len` and the V2
// `partition_size`.

const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;

/// Sequence lengths that span the operating range we care about.  Includes
/// values just above and below partition boundaries so the bench captures
/// reduce-kernel overhead transitions.  V2 covers all of these; V1 stops
/// before its `logits[max_seq_len]` shared-memory allocation overflows the
/// per-block 100 KB Ada limit (≈ 12 K floats).
const SEQ_LENS_V2: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
const SEQ_LENS_V1: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// Partition sizes to sweep for V2.  The current dispatcher uses 128.  Smaller
/// values increase grid parallelism but add reduce-kernel overhead per call;
/// larger values reduce occupancy at short seq_len but cut tmp_out write
/// traffic at long seq_len.  These four cover that trade-off cleanly.
const PARTITION_SIZES: &[usize] = &[64, 128, 256, 512];

struct CudaInputs {
    dev: Device,
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
}

fn build_inputs(seq_len: usize) -> Option<CudaInputs> {
    let dev = Device::new_cuda(0).ok()?;
    let max_blocks_per_seq = seq_len.div_ceil(BLOCK_SIZE);
    let q_f32: Vec<f32> = (0..NUM_Q_HEADS * HEAD_DIM)
        .map(|i| (i as f32 * 0.0123).sin() * 0.5)
        .collect();
    let kv_elems = max_blocks_per_seq * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k_f32: Vec<f32> = (0..kv_elems)
        .map(|i| (i as f32 * 0.0271).cos() * 0.5)
        .collect();
    let v_f32: Vec<f32> = (0..kv_elems)
        .map(|i| (i as f32 * 0.0411).sin() * 0.5)
        .collect();

    let q = Tensor::from_vec(q_f32, (1, NUM_Q_HEADS, HEAD_DIM), &dev)
        .ok()?
        .to_dtype(DType::BF16)
        .ok()?;
    let k_cache = Tensor::from_vec(
        k_f32,
        (max_blocks_per_seq, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        &dev,
    )
    .ok()?
    .to_dtype(DType::BF16)
    .ok()?;
    let v_cache = Tensor::from_vec(
        v_f32,
        (max_blocks_per_seq, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        &dev,
    )
    .ok()?
    .to_dtype(DType::BF16)
    .ok()?;

    let bt: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
    let block_tables = Tensor::from_vec(bt, (1, max_blocks_per_seq), &dev).ok()?;
    let seq_lens = Tensor::from_vec(vec![seq_len as u32], 1, &dev).ok()?;

    Some(CudaInputs {
        dev,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
    })
}

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_v1(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention_v1");
    for &seq_len in SEQ_LENS_V1 {
        let Some(inputs) = build_inputs(seq_len) else {
            return; // CUDA unavailable
        };
        let max_blocks_per_seq = seq_len.div_ceil(BLOCK_SIZE);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        // Warmup once outside the timed loop.
        let _ = paged_attention_cuda(
            &inputs.q,
            &inputs.k_cache,
            &inputs.v_cache,
            &inputs.block_tables,
            &inputs.seq_lens,
            scale,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            max_blocks_per_seq,
            seq_len,
            HEAD_DIM,
            BLOCK_SIZE,
        );
        sync(&inputs.dev);

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &seq_len,
            |b, &seq_len| {
                b.iter(|| {
                    let out = paged_attention_cuda(
                        &inputs.q,
                        &inputs.k_cache,
                        &inputs.v_cache,
                        &inputs.block_tables,
                        &inputs.seq_lens,
                        scale,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        max_blocks_per_seq,
                        seq_len,
                        HEAD_DIM,
                        BLOCK_SIZE,
                    )
                    .expect("paged_attention_v1");
                    sync(&inputs.dev);
                    out
                });
            },
        );
    }
    group.finish();
}

fn bench_v2_partition_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention_v2");
    for &partition_size in PARTITION_SIZES {
        for &seq_len in SEQ_LENS_V2 {
            let Some(inputs) = build_inputs(seq_len) else {
                return; // CUDA unavailable
            };
            let max_blocks_per_seq = seq_len.div_ceil(BLOCK_SIZE);
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();

            // Warmup
            let _ = paged_attention_v2_cuda_with_partition_size(
                &inputs.q,
                &inputs.k_cache,
                &inputs.v_cache,
                &inputs.block_tables,
                &inputs.seq_lens,
                scale,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                max_blocks_per_seq,
                seq_len,
                HEAD_DIM,
                BLOCK_SIZE,
                partition_size,
            );
            sync(&inputs.dev);

            let label = format!("p{partition_size}_s{seq_len}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| {
                    let out = paged_attention_v2_cuda_with_partition_size(
                        &inputs.q,
                        &inputs.k_cache,
                        &inputs.v_cache,
                        &inputs.block_tables,
                        &inputs.seq_lens,
                        scale,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        max_blocks_per_seq,
                        seq_len,
                        HEAD_DIM,
                        BLOCK_SIZE,
                        partition_size,
                    )
                    .expect("paged_attention_v2");
                    sync(&inputs.dev);
                    out
                });
            });
        }
    }
    group.finish();
}

/// Reproduces the **documented pool worst-case regression** described in
/// `crates/core/src/models/qwen3_quantized.rs:820-823`:
///
/// > Worst-case sizing iterates over ~3× more empty partitions for short
/// > decode positions — a 5-10× per-call slowdown at c=8 with prompt=256,
/// > observed empirically in the eager-bench head-to-head.
///
/// The pool path uses `worst_case_max_seq_len` to size both the pooled
/// scratch buffers AND the kernel grid Z dimension
/// (`PagedAttnV2InplaceOp::cuda_fwd`, `cuda_kernels.rs:1322-1327`). For a
/// 32-token decode with `worst_case=131_072` and partition_size=64 we
/// launch 2048 partition blocks while only 1 has work; the other 2047
/// pay block-launch + scratch-zero cost.
///
/// Sweep:
/// - `actual_max_seq_len ∈ {32, 256, 1024}` (decode buckets)
/// - `worst_case_max_seq_len ∈ {1024, 8192, 32_768, 131_072}` (typical
///   engine_limits values; max-model-len=131k matches the production
///   Qwen3-8B server config)
///
/// Baseline reading: `worst_case == actual` is the no-overhead case;
/// growing `worst_case` while holding `actual` fixed measures the
/// pure regression caused by empty-partition launches.
fn bench_v2_pooled_worst_case_sweep(c: &mut Criterion) {
    const ACTUAL_SEQ_LENS: &[usize] = &[32, 256, 1024];
    const WORST_CASE_SEQ_LENS: &[usize] = &[1024, 8192, 32_768, 131_072];

    let mut group = c.benchmark_group("paged_attention_v2_pooled_worst_case");
    // Each sample triggers a pool `reserve` + scratch zero + two kernel
    // launches; 30 samples keeps wall-clock under a minute even on the
    // 4-axis sweep, and noise is dominated by GPU clock-up so larger
    // sample counts don't tighten the mean.
    group.sample_size(30);

    for &actual_seq_len in ACTUAL_SEQ_LENS {
        let Some(inputs) = build_inputs(actual_seq_len) else {
            return;
        };
        let max_blocks_per_seq = actual_seq_len.div_ceil(BLOCK_SIZE);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        for &worst_case_seq_len in WORST_CASE_SEQ_LENS {
            // worst_case must be ≥ actual or the kernel reads OOB
            // partitions in the reduce stage.
            if worst_case_seq_len < actual_seq_len {
                continue;
            }
            let partition_size = select_v2_partition_size(worst_case_seq_len);

            // Warmup: 3 calls prime CUBLAS handles, pool slot reservations,
            // PTX module load. Without this the first criterion sample
            // includes one-time setup cost.
            for _ in 0..3 {
                let _ = paged_attention_v2_cuda_pooled(
                    &inputs.q,
                    &inputs.k_cache,
                    &inputs.v_cache,
                    &inputs.block_tables,
                    &inputs.seq_lens,
                    scale,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    max_blocks_per_seq,
                    actual_seq_len,
                    worst_case_seq_len,
                    HEAD_DIM,
                    BLOCK_SIZE,
                    partition_size,
                );
            }
            sync(&inputs.dev);

            let label = format!("act{actual_seq_len}_worst{worst_case_seq_len}");
            group.bench_with_input(BenchmarkId::new("config", &label), &label, |b, _| {
                b.iter(|| {
                    let out = paged_attention_v2_cuda_pooled(
                        &inputs.q,
                        &inputs.k_cache,
                        &inputs.v_cache,
                        &inputs.block_tables,
                        &inputs.seq_lens,
                        scale,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        max_blocks_per_seq,
                        actual_seq_len,
                        worst_case_seq_len,
                        HEAD_DIM,
                        BLOCK_SIZE,
                        partition_size,
                    )
                    .expect("paged_attention_v2_pooled");
                    sync(&inputs.dev);
                    out
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_v1,
    bench_v2_partition_sweep,
    bench_v2_pooled_worst_case_sweep,
);
criterion_main!(benches);
