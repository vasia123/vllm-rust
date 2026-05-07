//! KV cache allocator hot-path bench.
//!
//! `KVCacheManager::allocate_for_request` runs once per scheduled request
//! per step (and on initial admission). For decode batches it is invoked
//! `batch_size` times per step. The block pool's allocate/free path holds
//! a Mutex around the free list, so this microbench is the cheapest way
//! to spot regressions before they show up as decode-tps loss at high
//! concurrency.
//!
//! Three patterns measured:
//!
//! - `allocate_burst` — N fresh requests each grab `prompt_len` tokens.
//! - `decode_step` — `batch_size` running requests, each appending one
//!   token (the common steady-state allocation).
//! - `alloc_then_free` — full lifecycle, isolates allocator + free-list.
//!
//! Pure CPU; the cache manager is created on `Device::Cpu` to avoid CUDA
//! init overhead in CI.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use candle_core::{DType, Device};
use vllm_core::kv_cache::config::CacheConfig;
use vllm_core::kv_cache::{BlockTable, KVCacheDtype, KVCacheManager};

const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 8192;
const NUM_LAYERS: usize = 4;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const PROMPT_LEN: usize = 256;

fn make_cache_config() -> CacheConfig {
    CacheConfig {
        block_size: BLOCK_SIZE,
        num_blocks: NUM_BLOCKS,
        num_layers: NUM_LAYERS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    }
}

fn bench_allocate_burst(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_allocate_burst");
    for &batch in &[1usize, 4, 16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter_with_setup(
                || {
                    let mgr = KVCacheManager::new(&make_cache_config()).expect("manager");
                    let bts: Vec<BlockTable> =
                        (0..batch).map(|_| BlockTable::new(BLOCK_SIZE)).collect();
                    (mgr, bts)
                },
                |(mut mgr, mut bts)| {
                    for bt in bts.iter_mut() {
                        mgr.allocate_for_request(bt, PROMPT_LEN).expect("allocate");
                    }
                    black_box(&mut bts);
                    black_box(&mut mgr);
                },
            );
        });
    }
    group.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_decode_step");
    for &batch in &[1usize, 4, 16, 64, 256] {
        // Pre-warm batch with their prompts so each iteration is just the
        // single-token append (steady-state decode).
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter_with_setup(
                || {
                    let mut mgr = KVCacheManager::new(&make_cache_config()).expect("manager");
                    let mut bts: Vec<BlockTable> =
                        (0..batch).map(|_| BlockTable::new(BLOCK_SIZE)).collect();
                    for bt in bts.iter_mut() {
                        mgr.allocate_for_request(bt, PROMPT_LEN).expect("prefill");
                    }
                    (mgr, bts)
                },
                |(mut mgr, mut bts)| {
                    for bt in bts.iter_mut() {
                        mgr.allocate_for_request(bt, 1).expect("decode tok");
                    }
                    black_box(&mut bts);
                    black_box(&mut mgr);
                },
            );
        });
    }
    group.finish();
}

fn bench_alloc_then_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_alloc_then_free");
    for &batch in &[1usize, 16, 64] {
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, _| {
            b.iter_with_setup(
                || {
                    let mgr = KVCacheManager::new(&make_cache_config()).expect("manager");
                    let bts: Vec<BlockTable> =
                        (0..batch).map(|_| BlockTable::new(BLOCK_SIZE)).collect();
                    (mgr, bts)
                },
                |(mut mgr, mut bts)| {
                    for bt in bts.iter_mut() {
                        mgr.allocate_for_request(bt, PROMPT_LEN).expect("alloc");
                    }
                    for bt in bts.iter_mut() {
                        mgr.free_request(bt).expect("free");
                    }
                    black_box(&mut mgr);
                },
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_allocate_burst,
    bench_decode_step,
    bench_alloc_then_free,
);
criterion_main!(benches);
