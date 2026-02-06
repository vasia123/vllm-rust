//! Integration tests for the KV cache lifecycle.
//!
//! These tests exercise the full KV cache path: allocation, write, read,
//! block table slot mapping, prefix caching, and eviction. All CPU-only.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use vllm_core::kv_cache::{
    config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager, KVCacheMetrics,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn small_cache_config() -> CacheConfig {
    CacheConfig {
        block_size: 16,
        num_blocks: 32,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    }
}

fn multi_layer_config() -> CacheConfig {
    CacheConfig {
        block_size: 4,
        num_blocks: 16,
        num_layers: 3,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    }
}

// ─── Allocate, write, read lifecycle ─────────────────────────────────────────

#[test]
fn test_allocate_write_read_kv() {
    let config = small_cache_config();
    let mut mgr = KVCacheManager::new(&config).unwrap();

    // Allocate blocks for a short prompt
    let mut block_table = BlockTable::new(config.block_size);
    let prompt_len = 5;
    mgr.allocate_for_request(&mut block_table, prompt_len)
        .unwrap();

    let slot_mapping = block_table.slot_mapping(0, prompt_len);
    assert_eq!(slot_mapping.len(), prompt_len);

    // Create KV data: [num_kv_heads, num_tokens, head_dim]
    let k_data: Vec<f32> = (0..(2 * prompt_len * 8)).map(|i| i as f32).collect();
    let k = Tensor::from_vec(k_data.clone(), (2, prompt_len, 8), &Device::Cpu).unwrap();
    let v_data: Vec<f32> = (0..(2 * prompt_len * 8))
        .map(|i| (i as f32) * 0.5)
        .collect();
    let v = Tensor::from_vec(v_data.clone(), (2, prompt_len, 8), &Device::Cpu).unwrap();

    // Write to cache at layer 0
    mgr.engine_mut(0).write(&k, &v, &slot_mapping).unwrap();

    // Read back from the block(s)
    let block_ids = block_table.block_ids();
    let (k_out, v_out) = mgr.engine(0).read(block_ids, prompt_len).unwrap();

    // Verify shapes
    // CacheEngine.read returns [num_blocks, num_kv_heads, tokens_read, head_dim]
    let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
    let v_flat: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();

    // The first prompt_len * 2 * 8 elements should match our written data
    // (reading may return extra zeros for the remaining block slots)
    for (i, (&written, &read)) in k_data.iter().zip(k_flat.iter()).enumerate() {
        assert!(
            (written - read).abs() < 1e-6,
            "K mismatch at index {i}: expected {written}, got {read}"
        );
    }
    for (i, (&written, &read)) in v_data.iter().zip(v_flat.iter()).enumerate() {
        assert!(
            (written - read).abs() < 1e-6,
            "V mismatch at index {i}: expected {written}, got {read}"
        );
    }

    mgr.free_request(&mut block_table).unwrap();
}

// ─── Block table slot mapping correctness ────────────────────────────────────

#[test]
fn test_slot_mapping_correctness() {
    let block_size = 4;
    let mut block_table = BlockTable::new(block_size);
    // Simulate allocating blocks with IDs [2, 5, 8]
    block_table.append_blocks(&[2, 5, 8]);

    // Tokens 0-3 map to block 2, tokens 4-7 map to block 5, etc.
    let mapping = block_table.slot_mapping(0, 10);
    assert_eq!(mapping.len(), 10);

    // Token 0 -> block 2, offset 0 -> slot 2*4 = 8
    assert_eq!(mapping[0], 2 * block_size);
    // Token 1 -> block 2, offset 1 -> slot 2*4+1 = 9
    assert_eq!(mapping[1], 2 * block_size + 1);
    // Token 4 -> block 5, offset 0 -> slot 5*4 = 20
    assert_eq!(mapping[4], 5 * block_size);
    // Token 7 -> block 5, offset 3 -> slot 5*4+3 = 23
    assert_eq!(mapping[7], 5 * block_size + 3);
    // Token 8 -> block 8, offset 0 -> slot 8*4 = 32
    assert_eq!(mapping[8], 8 * block_size);
    // Token 9 -> block 8, offset 1 -> slot 8*4+1 = 33
    assert_eq!(mapping[9], 8 * block_size + 1);
}

#[test]
fn test_slot_mapping_with_offset() {
    let block_size = 4;
    let mut block_table = BlockTable::new(block_size);
    block_table.append_blocks(&[0, 1]);

    // Request mapping starting from position 3 for 3 tokens
    let mapping = block_table.slot_mapping(3, 3);
    assert_eq!(mapping.len(), 3);

    // Token 3 -> block 0, offset 3 -> slot 3
    assert_eq!(mapping[0], 3);
    // Token 4 -> block 1, offset 0 -> slot 4
    assert_eq!(mapping[1], block_size);
    // Token 5 -> block 1, offset 1 -> slot 5
    assert_eq!(mapping[2], block_size + 1);
}

// ─── Block allocation and exhaustion ─────────────────────────────────────────

#[test]
fn test_cache_allocation_exhaustion() {
    let mut config = small_cache_config();
    config.num_blocks = 4;
    let mut mgr = KVCacheManager::new(&config).unwrap();

    assert_eq!(mgr.num_free_blocks(), 4);

    // Allocate all blocks
    let mut table = BlockTable::new(config.block_size);
    mgr.allocate_for_request(&mut table, 4 * config.block_size)
        .unwrap();
    assert_eq!(mgr.num_free_blocks(), 0);

    // Trying to allocate more should fail
    let mut table2 = BlockTable::new(config.block_size);
    let result = mgr.allocate_for_request(&mut table2, config.block_size);
    assert!(
        result.is_err(),
        "allocation should fail when blocks exhausted"
    );

    // Free the first request
    mgr.free_request(&mut table).unwrap();
    assert_eq!(mgr.num_free_blocks(), 4);

    // Now allocation should succeed again
    mgr.allocate_for_request(&mut table2, config.block_size)
        .unwrap();
    assert!(!table2.block_ids().is_empty());

    mgr.free_request(&mut table2).unwrap();
}

// ─── Multi-layer cache isolation ─────────────────────────────────────────────

#[test]
fn test_multi_layer_cache_isolation() {
    let config = multi_layer_config();
    let mut mgr = KVCacheManager::new(&config).unwrap();

    // Write different data to each layer
    let k0 = Tensor::ones((2, 2, 8), DType::F32, &Device::Cpu).unwrap();
    let v0 = Tensor::ones((2, 2, 8), DType::F32, &Device::Cpu).unwrap();
    mgr.engine_mut(0).write(&k0, &v0, &[0, 1]).unwrap();

    let k1_data: Vec<f32> = (0..32).map(|i| i as f32 * 2.0).collect();
    let k1 = Tensor::from_vec(k1_data, (2, 2, 8), &Device::Cpu).unwrap();
    let v1_data: Vec<f32> = (0..32).map(|i| i as f32 * 3.0).collect();
    let v1 = Tensor::from_vec(v1_data, (2, 2, 8), &Device::Cpu).unwrap();
    mgr.engine_mut(1).write(&k1, &v1, &[0, 1]).unwrap();

    // Layer 2 should be all zeros (untouched)
    let (k2_out, v2_out) = mgr.engine(2).read(&[0], 2).unwrap();
    let k2_flat: Vec<f32> = k2_out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        k2_flat.iter().all(|&x| x == 0.0),
        "layer 2 should be all zeros"
    );
    let v2_flat: Vec<f32> = v2_out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        v2_flat.iter().all(|&x| x == 0.0),
        "layer 2 V cache should be all zeros"
    );

    // Layer 0 should have ones
    let (k0_out, _) = mgr.engine(0).read(&[0], 2).unwrap();
    let k0_flat: Vec<f32> = k0_out.flatten_all().unwrap().to_vec1().unwrap();
    assert!(
        k0_flat.iter().all(|&x| x == 1.0),
        "layer 0 should have ones"
    );
}

// ─── Prefix cache matching ───────────────────────────────────────────────────

#[test]
fn test_prefix_cache_matching() {
    let mut config = multi_layer_config();
    config.num_blocks = 16;
    let metrics = Arc::new(KVCacheMetrics::new());
    let mut mgr = KVCacheManager::with_prefix_cache(&config, metrics).unwrap();

    // Initially no match
    let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks (block_size=4)
    let (matched, _) = mgr.match_prefix(&prompt);
    assert!(matched.is_empty(), "no match expected for fresh cache");

    // Allocate and register a prefix
    let mut table = BlockTable::new(config.block_size);
    mgr.allocate_for_request(&mut table, 8).unwrap();
    table.advance(8);
    let block_ids = table.block_ids().to_vec();
    mgr.register_prefix(&prompt, &block_ids);

    // Now matching should find the prefix
    let (matched, num_cached) = mgr.match_prefix(&prompt);
    assert_eq!(matched.len(), 2);
    assert_eq!(num_cached, 8);

    // Partial prefix should also match the common part
    let shorter_prompt = vec![1, 2, 3, 4]; // Just 1 full block
    let (partial_matched, partial_cached) = mgr.match_prefix(&shorter_prompt);
    assert_eq!(partial_matched.len(), 1);
    assert_eq!(partial_cached, 4);
}

// ─── Cache eviction when blocks exhausted ────────────────────────────────────

#[test]
fn test_cache_eviction_when_blocks_exhausted() {
    let mut config = multi_layer_config();
    config.num_blocks = 4; // Very limited
    let metrics = Arc::new(KVCacheMetrics::new());
    let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

    // Fill cache: 1 block for prompt
    let prompt = vec![1, 2, 3, 4]; // 1 full block
    let mut table = BlockTable::new(config.block_size);
    mgr.allocate_for_request(&mut table, 4).unwrap();
    table.advance(4);
    let ids = table.block_ids().to_vec();
    mgr.register_prefix(&prompt, &ids);
    // Release reference so the block becomes evictable
    let to_free = mgr.release_prefix(&prompt, &ids);
    assert!(to_free.is_empty()); // Stays in cache

    // 3 free blocks + 1 cached evictable
    assert_eq!(mgr.num_free_blocks(), 3);

    // Try to allocate all 4 blocks: should evict the cached block
    let mut table2 = BlockTable::new(config.block_size);
    mgr.allocate_with_eviction(&mut table2, 4 * config.block_size)
        .unwrap();
    assert_eq!(table2.block_ids().len(), 4);
    assert_eq!(metrics.blocks_evicted(), 1);

    mgr.free_request(&mut table2).unwrap();
}

// ─── CPU offload allocate/evict cycle ────────────────────────────────────────

#[test]
fn test_cpu_offload_store_and_load() {
    let config = CacheConfig {
        block_size: 4,
        num_blocks: 8,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: Some(vllm_core::kv_cache::CpuOffloadConfig {
            max_cpu_blocks: 4,
            use_pinned_memory: false,
            prefetch_count: 2,
        }),
    };
    let mut mgr = KVCacheManager::new(&config).unwrap();
    assert!(mgr.has_cpu_offload());

    // Write data to GPU block 0
    let k = Tensor::ones((2, 3, 8), DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::ones((2, 3, 8), DType::F32, &Device::Cpu).unwrap();
    for layer in 0..2 {
        mgr.engine_mut(layer).write(&k, &v, &[0, 1, 2]).unwrap();
    }

    // Offload block 0 with hash 42
    mgr.offload_block(42, 0).unwrap();

    // Load it back
    let loaded = mgr.try_load_from_cpu(42).unwrap();
    assert!(loaded.is_some(), "block should load from CPU offload");

    let gpu_block = loaded.unwrap();

    // Verify data survived the roundtrip
    for layer in 0..2 {
        let (k_out, v_out) = mgr.engine(layer).read(&[gpu_block], 3).unwrap();
        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            k_flat.iter().all(|&x| x == 1.0),
            "layer {layer} K data should be ones after roundtrip"
        );
        let v_flat: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            v_flat.iter().all(|&x| x == 1.0),
            "layer {layer} V data should be ones after roundtrip"
        );
    }
}

#[test]
fn test_cpu_offload_miss_returns_none() {
    let config = CacheConfig {
        block_size: 4,
        num_blocks: 8,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: Some(vllm_core::kv_cache::CpuOffloadConfig {
            max_cpu_blocks: 4,
            use_pinned_memory: false,
            prefetch_count: 2,
        }),
    };
    let mut mgr = KVCacheManager::new(&config).unwrap();

    let result = mgr.try_load_from_cpu(999).unwrap();
    assert!(result.is_none(), "missing hash should return None");
}

#[test]
fn test_cpu_offload_disabled_is_noop() {
    let config = small_cache_config();
    let mut mgr = KVCacheManager::new(&config).unwrap();
    assert!(!mgr.has_cpu_offload());

    // These should be no-ops, not errors
    mgr.offload_block(42, 0).unwrap();
    let result = mgr.try_load_from_cpu(42).unwrap();
    assert!(result.is_none());
}

// ─── Incremental allocation ──────────────────────────────────────────────────

#[test]
fn test_incremental_allocation_across_blocks() {
    let block_size = 4;
    let config = CacheConfig {
        block_size,
        num_blocks: 16,
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 8,
        dtype: DType::F32,
        device: Device::Cpu,
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    };
    let mut mgr = KVCacheManager::new(&config).unwrap();
    let mut table = BlockTable::new(block_size);

    // Prefill: 5 tokens -> 2 blocks
    mgr.allocate_for_request(&mut table, 5).unwrap();
    assert_eq!(table.block_ids().len(), 2);
    table.advance(5);

    // Decode tokens one at a time
    // Token 6: fits in block 2 (5 < 8)
    mgr.allocate_for_request(&mut table, 1).unwrap();
    assert_eq!(table.block_ids().len(), 2);
    table.advance(1);

    // Token 7: still fits in block 2
    mgr.allocate_for_request(&mut table, 1).unwrap();
    assert_eq!(table.block_ids().len(), 2);
    table.advance(1);

    // Token 8: fills block 2
    mgr.allocate_for_request(&mut table, 1).unwrap();
    assert_eq!(table.block_ids().len(), 2);
    table.advance(1);

    // Token 9: needs block 3
    mgr.allocate_for_request(&mut table, 1).unwrap();
    assert_eq!(table.block_ids().len(), 3);
    table.advance(1);

    assert_eq!(table.num_tokens(), 9);
    mgr.free_request(&mut table).unwrap();
}

// ─── Metrics tracking ────────────────────────────────────────────────────────

#[test]
fn test_metrics_track_full_lifecycle() {
    let config = small_cache_config();
    let metrics = Arc::new(KVCacheMetrics::new());
    let mut mgr = KVCacheManager::with_metrics(&config, Arc::clone(&metrics)).unwrap();

    let mut table1 = BlockTable::new(config.block_size);
    let mut table2 = BlockTable::new(config.block_size);

    // Allocate for two requests
    mgr.allocate_for_request(&mut table1, 20).unwrap(); // ceil(20/16) = 2 blocks
    mgr.allocate_for_request(&mut table2, 40).unwrap(); // ceil(40/16) = 3 blocks

    assert_eq!(metrics.allocations(), 2);
    assert_eq!(metrics.blocks_allocated(), 5);

    // Free both
    mgr.free_request(&mut table1).unwrap();
    mgr.free_request(&mut table2).unwrap();

    assert_eq!(metrics.blocks_freed(), 5);

    // Snapshot should reflect all operations
    let snap = metrics.snapshot();
    assert_eq!(snap.allocations, 2);
    assert_eq!(snap.blocks_allocated, 5);
    assert_eq!(snap.blocks_freed, 5);
}

// ─── Block table trim ────────────────────────────────────────────────────────

#[test]
fn test_block_table_trim_frees_excess_blocks() {
    let config = small_cache_config();
    let mut mgr = KVCacheManager::new(&config).unwrap();
    let initial_free = mgr.num_free_blocks();

    let mut table = BlockTable::new(config.block_size);
    // Allocate 3 blocks worth of tokens (48 tokens with block_size=16)
    mgr.allocate_for_request(&mut table, 48).unwrap();
    table.advance(48);
    assert_eq!(table.block_ids().len(), 3);
    assert_eq!(mgr.num_free_blocks(), initial_free - 3);

    // Trim to 20 tokens (keeps 2 blocks, frees 1)
    let freed = table.trim_to(20);
    assert_eq!(freed.len(), 1);
    mgr.free_blocks(&freed).unwrap();
    assert_eq!(mgr.num_free_blocks(), initial_free - 2);

    // Clean up
    mgr.free_request(&mut table).unwrap();
    assert_eq!(mgr.num_free_blocks(), initial_free);
}

// ─── create_cache_manager from ModelConfig ───────────────────────────────────

#[test]
fn test_create_cache_manager_standard() {
    let cfg = vllm_core::config::ModelConfig {
        architectures: vec!["LlamaForCausalLM".to_string()],
        hidden_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_hidden_layers: 2,
        intermediate_size: 128,
        vocab_size: 256,
        max_position_embeddings: 512,
        head_dim: 16,
        hidden_act: "silu".to_string(),
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        bos_token_id: 1,
        eos_token_id: 2,
        sliding_window: None,
        attention_bias: None,
        extra: serde_json::Map::new(),
    };

    let mgr = vllm_core::models::create_cache_manager(&cfg, 16, 32, DType::F32, &Device::Cpu);
    assert!(mgr.is_ok());
    let mgr = mgr.unwrap();
    assert!(!mgr.is_mla());
    assert_eq!(mgr.num_free_blocks(), 32);
    assert_eq!(mgr.block_size(), 16);
}
