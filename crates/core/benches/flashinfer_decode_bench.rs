//! Microbench: FlashInfer DecodeWrapper vs paged_attention_auto (V2) on
//! Qwen3-4B-AWQ realistic decode shape.
//!
//! Direction A of the 2026-05-09 research roadmap:
//! `docs/perf/closing-the-vllm-gap-status.md` (TBD). Decision gate:
//! FlashInfer must beat paged_attn V2 by ≥ 1.3× per-replay on at least one
//! seq_len for direction A to proceed to integration.
//!
//! Shape pinned to Qwen3-4B-AWQ:
//!   batch_size  = 8        (c=8 from bench_decode head-to-head)
//!   num_q_heads = 32
//!   num_kv_heads= 8        (GQA 4:1)
//!   head_dim    = 128
//!   block_size  = 16
//!   seq_len     = {64, 256, 384, 512, 1024}
//!
//! Run on the test box:
//!   cargo bench --features cuda-default -p vllm-core
//!     --bench flashinfer_decode_bench
//!
//! Reports three groups:
//! - `paged_attn_v2`: per-call cost of the current production path.
//! - `flashinfer_decode_full`: build_plan + run_with_plan (one-shot cost).
//! - `flashinfer_decode_replay`: run_with_plan only — the relevant cost
//!   when a plan is amortised across all 36 attention layers of one
//!   forward (per-layer decode cost).
//!
//! Skips silently when no CUDA device is available so CI CPU runs do not
//! fail the bench gate.

#![cfg(all(feature = "cuda-kernels", feature = "flashinfer"))]

use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::cuda_kernels::paged_attention_auto;
use vllm_core::kv_cache::KVCacheLayout;
use vllm_core::layers::attention::flashinfer::{DecodeWrapper, FlashInferConfig, WorkspaceBuffer};

const BATCH_SIZE: usize = 8;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK_SIZE: usize = 16;

const SEQ_LENS: &[usize] = &[64, 256, 384, 512, 1024];

struct Inputs {
    dev: Device,
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    // paged_attention_auto inputs:
    block_tables: Tensor, // u32 [batch, max_blocks_per_seq]
    seq_lens: Tensor,     // u32 [batch]
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    // FlashInfer DecodeWrapper inputs:
    kv_indptr: Tensor,        // u32 [batch+1] — cast to i32 inside FFI
    kv_indices: Tensor,       // u32 [total_blocks]
    kv_last_page_len: Tensor, // u32 [batch]
}

fn build_inputs(seq_len: usize) -> Option<Inputs> {
    let dev = Device::new_cuda(0).ok()?;
    let max_blocks_per_seq = seq_len.div_ceil(BLOCK_SIZE);
    let total_blocks = BATCH_SIZE * max_blocks_per_seq;

    // Q: [batch, num_q_heads, head_dim]
    let q_f32: Vec<f32> = (0..BATCH_SIZE * NUM_Q_HEADS * HEAD_DIM)
        .map(|i| (i as f32 * 0.0123).sin() * 0.5)
        .collect();
    let q = Tensor::from_vec(q_f32, (BATCH_SIZE, NUM_Q_HEADS, HEAD_DIM), &dev)
        .ok()?
        .to_dtype(DType::BF16)
        .ok()?;

    // KV cache: [num_pages, page_size, num_kv_heads, head_dim] (NHD).
    // Allocate exactly total_blocks pages, one per (batch_i, block_j) pair.
    let kv_elems = total_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    let k_f32: Vec<f32> = (0..kv_elems)
        .map(|i| (i as f32 * 0.0271).cos() * 0.5)
        .collect();
    let v_f32: Vec<f32> = (0..kv_elems)
        .map(|i| (i as f32 * 0.0411).sin() * 0.5)
        .collect();
    let k_cache = Tensor::from_vec(
        k_f32,
        (total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        &dev,
    )
    .ok()?
    .to_dtype(DType::BF16)
    .ok()?;
    let v_cache = Tensor::from_vec(
        v_f32,
        (total_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        &dev,
    )
    .ok()?
    .to_dtype(DType::BF16)
    .ok()?;

    // block_tables: [batch, max_blocks_per_seq] dense, sequence i owns blocks
    // [i*M, (i+1)*M).
    let mut bt_data = vec![0u32; BATCH_SIZE * max_blocks_per_seq];
    for i in 0..BATCH_SIZE {
        for j in 0..max_blocks_per_seq {
            bt_data[i * max_blocks_per_seq + j] = (i * max_blocks_per_seq + j) as u32;
        }
    }
    let block_tables = Tensor::from_vec(bt_data, (BATCH_SIZE, max_blocks_per_seq), &dev).ok()?;
    let seq_lens = Tensor::from_vec(vec![seq_len as u32; BATCH_SIZE], (BATCH_SIZE,), &dev).ok()?;

    // FlashInfer indptr/indices/last_page_len.
    let mut indptr_data = Vec::with_capacity(BATCH_SIZE + 1);
    indptr_data.push(0u32);
    let mut acc = 0u32;
    for _ in 0..BATCH_SIZE {
        acc += max_blocks_per_seq as u32;
        indptr_data.push(acc);
    }
    let kv_indptr = Tensor::from_vec(indptr_data, (BATCH_SIZE + 1,), &dev).ok()?;

    let indices_data: Vec<u32> = (0..total_blocks as u32).collect();
    let kv_indices = Tensor::from_vec(indices_data, (total_blocks,), &dev).ok()?;

    let last_in_block = seq_len % BLOCK_SIZE;
    let last = if last_in_block == 0 && seq_len > 0 {
        BLOCK_SIZE
    } else {
        last_in_block
    };
    let kv_last_page_len =
        Tensor::from_vec(vec![last as u32; BATCH_SIZE], (BATCH_SIZE,), &dev).ok()?;

    Some(Inputs {
        dev,
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_blocks_per_seq,
        max_seq_len: seq_len,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
    })
}

fn sync(dev: &Device) {
    if let Device::Cuda(cd) = dev {
        let _ = cd.cuda_stream().synchronize();
    }
}

fn bench_paged_attn_v2(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attn_v2");
    for &seq_len in SEQ_LENS {
        let Some(inp) = build_inputs(seq_len) else {
            return;
        };
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        // Warmup
        let _ = paged_attention_auto(
            &inp.q,
            &inp.k_cache,
            &inp.v_cache,
            &inp.block_tables,
            &inp.seq_lens,
            scale,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            inp.max_blocks_per_seq,
            inp.max_seq_len,
            HEAD_DIM,
            BLOCK_SIZE,
        );
        sync(&inp.dev);

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let out = paged_attention_auto(
                    &inp.q,
                    &inp.k_cache,
                    &inp.v_cache,
                    &inp.block_tables,
                    &inp.seq_lens,
                    scale,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    inp.max_blocks_per_seq,
                    inp.max_seq_len,
                    HEAD_DIM,
                    BLOCK_SIZE,
                )
                .expect("paged_attention_auto");
                sync(&inp.dev);
                out
            });
        });
    }
    group.finish();
}

fn bench_flashinfer_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("flashinfer_decode_full");
    for &seq_len in SEQ_LENS {
        let Some(inp) = build_inputs(seq_len) else {
            return;
        };
        let cfg = FlashInferConfig::new(
            NUM_Q_HEADS as u32,
            NUM_KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_SIZE as u32,
        )
        .with_kv_layout(KVCacheLayout::NHD);
        let wrapper = DecodeWrapper::new(cfg, &inp.dev).expect("wrapper");
        let mut workspace = WorkspaceBuffer::new(&inp.dev).expect("ws");

        // Warmup
        let _ = wrapper.run(
            &inp.q,
            &inp.k_cache,
            &inp.v_cache,
            &mut workspace,
            &inp.kv_indptr,
            &inp.kv_indices,
            &inp.kv_last_page_len,
            BATCH_SIZE,
        );
        sync(&inp.dev);

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let out = wrapper
                    .run(
                        &inp.q,
                        &inp.k_cache,
                        &inp.v_cache,
                        &mut workspace,
                        &inp.kv_indptr,
                        &inp.kv_indices,
                        &inp.kv_last_page_len,
                        BATCH_SIZE,
                    )
                    .expect("flashinfer decode");
                sync(&inp.dev);
                out
            });
        });
    }
    group.finish();
}

fn bench_flashinfer_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("flashinfer_decode_replay");
    for &seq_len in SEQ_LENS {
        let Some(inp) = build_inputs(seq_len) else {
            return;
        };
        let cfg = FlashInferConfig::new(
            NUM_Q_HEADS as u32,
            NUM_KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_SIZE as u32,
        )
        .with_kv_layout(KVCacheLayout::NHD);
        let wrapper = DecodeWrapper::new(cfg, &inp.dev).expect("wrapper");
        let mut workspace = WorkspaceBuffer::new(&inp.dev).expect("ws");

        // Build plan once — reused across all replays.
        let plan = wrapper
            .build_plan(
                &mut workspace,
                inp.kv_indptr.clone(),
                inp.kv_indices.clone(),
                inp.kv_last_page_len.clone(),
                inp.q.dtype(),
                BATCH_SIZE,
            )
            .expect("build_plan");

        // Warmup
        let _ = wrapper.run_with_plan(&inp.q, &inp.k_cache, &inp.v_cache, &plan);
        sync(&inp.dev);

        group.bench_with_input(BenchmarkId::new("seq_len", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                let out = wrapper
                    .run_with_plan(&inp.q, &inp.k_cache, &inp.v_cache, &plan)
                    .expect("flashinfer replay");
                sync(&inp.dev);
                out
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_paged_attn_v2,
    bench_flashinfer_full,
    bench_flashinfer_replay
);
criterion_main!(benches);
