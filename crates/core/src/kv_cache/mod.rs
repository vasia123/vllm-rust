mod block_metrics;
mod block_pool;
mod block_table;
mod cache_engine;
pub mod config;
mod error;
mod free_block_queue;
pub mod metrics;
pub mod prefix_cache;

pub use block_metrics::{BlockEvictionEvent, BlockMetricsCollector, BlockMetricsState};
pub use block_pool::BlockId;
pub use block_table::BlockTable;
pub use cache_engine::CacheEngine;
pub use config::CacheConfig;
pub use error::CacheError;
pub use free_block_queue::FreeBlockQueue;
pub use metrics::{KVCacheMetrics, MetricsSnapshot};

use block_pool::BlockPool;
use prefix_cache::PrefixCache;
use std::sync::Arc;

/// Top-level coordinator: BlockPool + per-layer CacheEngines.
pub struct KVCacheManager {
    block_pool: BlockPool,
    engines: Vec<CacheEngine>,
    block_size: usize,
    metrics: Arc<KVCacheMetrics>,
    /// Optional prefix cache for block reuse and eviction coordination
    prefix_cache: Option<PrefixCache>,
}

impl KVCacheManager {
    pub fn new(config: &CacheConfig) -> Result<Self, CacheError> {
        Self::with_metrics(config, Arc::new(KVCacheMetrics::new()))
    }

    /// Create a KVCacheManager with custom metrics instance.
    pub fn with_metrics(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
    ) -> Result<Self, CacheError> {
        let mut engines = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            engines.push(CacheEngine::new(config)?);
        }
        Ok(Self {
            block_pool: BlockPool::new(config.num_blocks),
            engines,
            block_size: config.block_size,
            metrics,
            prefix_cache: None,
        })
    }

    /// Create a KVCacheManager with prefix caching enabled.
    pub fn with_prefix_cache(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
    ) -> Result<Self, CacheError> {
        let mut mgr = Self::with_metrics(config, Arc::clone(&metrics))?;
        mgr.prefix_cache = Some(PrefixCache::with_metrics(config.block_size, metrics));
        Ok(mgr)
    }

    /// Allocate blocks needed for `new_tokens` additional tokens.
    pub fn allocate_for_request(
        &mut self,
        block_table: &mut BlockTable,
        new_tokens: usize,
    ) -> Result<(), CacheError> {
        let needed = block_table.blocks_needed(new_tokens);
        if needed > 0 {
            let ids = self.block_pool.allocate(needed)?;
            self.metrics.record_allocation(ids.len());
            block_table.append_blocks(&ids);
        }
        Ok(())
    }

    /// Free all blocks owned by a request.
    pub fn free_request(&mut self, block_table: &mut BlockTable) -> Result<(), CacheError> {
        let ids = block_table.release();
        if !ids.is_empty() {
            self.metrics.record_free(ids.len());
            self.block_pool.free(&ids)?;
        }
        Ok(())
    }

    /// Get the CacheEngine for a specific layer.
    pub fn engine(&self, layer_idx: usize) -> &CacheEngine {
        &self.engines[layer_idx]
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free()
    }

    /// Free specific blocks back to the pool (used during speculative decode rollback).
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), CacheError> {
        if !block_ids.is_empty() {
            self.metrics.record_free(block_ids.len());
            self.block_pool.free(block_ids)?;
        }
        Ok(())
    }

    /// Get the metrics instance for monitoring.
    pub fn metrics(&self) -> &Arc<KVCacheMetrics> {
        &self.metrics
    }

    /// Check if prefix caching is enabled.
    pub fn has_prefix_cache(&self) -> bool {
        self.prefix_cache.is_some()
    }

    /// Match prefix from cache (if enabled).
    ///
    /// Returns `(block_ids, num_cached_tokens)` for the matched prefix.
    /// Block IDs should be prepended to the request's block table.
    pub fn match_prefix(&mut self, prompt_tokens: &[u32]) -> (Vec<BlockId>, usize) {
        match &mut self.prefix_cache {
            Some(cache) => cache.match_prefix(prompt_tokens),
            None => (Vec::new(), 0),
        }
    }

    /// Register blocks from a completed prefill into the prefix cache.
    pub fn register_prefix(&mut self, prompt_tokens: &[u32], block_ids: &[BlockId]) {
        if let Some(cache) = &mut self.prefix_cache {
            cache.register_blocks(prompt_tokens, block_ids);
        }
    }

    /// Release prefix cache references when a request completes.
    ///
    /// Returns block IDs that should be freed (not in cache).
    pub fn release_prefix(&mut self, prompt_tokens: &[u32], block_ids: &[BlockId]) -> Vec<BlockId> {
        match &mut self.prefix_cache {
            Some(cache) => cache.release_blocks(prompt_tokens, block_ids),
            None => block_ids.to_vec(),
        }
    }

    /// Try to allocate blocks, evicting from prefix cache if needed.
    ///
    /// This is the main allocation method when prefix caching is enabled.
    /// If allocation fails due to OOM, it will evict unreferenced blocks
    /// from the prefix cache and retry.
    pub fn allocate_with_eviction(
        &mut self,
        block_table: &mut BlockTable,
        new_tokens: usize,
    ) -> Result<(), CacheError> {
        let needed = block_table.blocks_needed(new_tokens);
        if needed == 0 {
            return Ok(());
        }

        // Try direct allocation first
        let available = self.block_pool.num_free();
        if available >= needed {
            let ids = self.block_pool.allocate(needed)?;
            self.metrics.record_allocation(ids.len());
            block_table.append_blocks(&ids);
            return Ok(());
        }

        // Not enough blocks - try to evict from prefix cache
        if let Some(cache) = &mut self.prefix_cache {
            let deficit = needed - available;
            let evictable = cache.num_evictable_blocks();

            if evictable >= deficit {
                // Evict enough blocks
                let evicted = cache.evict(deficit);
                self.metrics.record_eviction(evicted.len());
                // Return evicted blocks to pool
                self.block_pool.free(&evicted)?;
            }
        }

        // Retry allocation (may still fail if not enough evictable)
        let ids = self.block_pool.allocate(needed)?;
        self.metrics.record_allocation(ids.len());
        block_table.append_blocks(&ids);
        Ok(())
    }

    /// Evict blocks from prefix cache to free memory.
    ///
    /// Returns the number of blocks actually evicted.
    pub fn evict_from_cache(&mut self, count: usize) -> Result<usize, CacheError> {
        match &mut self.prefix_cache {
            Some(cache) => {
                let evicted = cache.evict(count);
                let num_evicted = evicted.len();
                if !evicted.is_empty() {
                    self.metrics.record_eviction(num_evicted);
                    self.block_pool.free(&evicted)?;
                }
                Ok(num_evicted)
            }
            None => Ok(0),
        }
    }

    /// Get prefix cache statistics.
    pub fn prefix_cache_stats(&self) -> Option<(usize, usize)> {
        self.prefix_cache
            .as_ref()
            .map(|c| (c.num_cached_blocks(), c.num_evictable_blocks()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn test_config() -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    #[test]
    fn allocate_and_free_lifecycle() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        assert_eq!(mgr.num_free_blocks(), 16);

        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 10).unwrap(); // needs 3 blocks (ceil(10/4))
        assert_eq!(mgr.num_free_blocks(), 13);

        mgr.free_request(&mut table).unwrap();
        assert_eq!(mgr.num_free_blocks(), 16);
    }

    #[test]
    fn write_read_through_manager() {
        let config = test_config();
        let mgr = KVCacheManager::new(&config).unwrap();

        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();

        // Write to layer 0, slots 0,1,2 (block 0)
        mgr.engine(0).write(&k, &v, &[0, 1, 2]).unwrap();

        // Read back
        let (k_out, _) = mgr.engine(0).read(&[0], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        // Layer 1 should be independent (all zeros still)
        let (k_layer1, _) = mgr.engine(1).read(&[0], 3).unwrap();
        let flat: Vec<f32> = k_layer1.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn free_blocks_returns_to_pool() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table, 10).unwrap(); // 3 blocks
        table.advance(10);
        assert_eq!(mgr.num_free_blocks(), 13);

        let freed = table.trim_to(5); // keep 2 blocks, free 1
        assert_eq!(freed.len(), 1);
        mgr.free_blocks(&freed).unwrap();
        assert_eq!(mgr.num_free_blocks(), 14);
    }

    #[test]
    fn incremental_allocation() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        // Prefill: 5 tokens → need 2 blocks
        mgr.allocate_for_request(&mut table, 5).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(5);

        // Decode: 1 token → fits in existing block (5 < 8)
        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2); // no new block
        table.advance(1);

        // Decode more: fill up to 8, then 9th needs new block
        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(1);

        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(1); // 8 tokens, last block full

        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 3); // new block!
        table.advance(1);

        assert_eq!(table.num_tokens(), 9);
    }

    // ─── Metrics integration tests ────────────────────────────────────────────

    #[test]
    fn metrics_track_allocations() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        assert_eq!(mgr.metrics().allocations(), 0);
        assert_eq!(mgr.metrics().blocks_allocated(), 0);

        mgr.allocate_for_request(&mut table, 10).unwrap(); // 3 blocks
        assert_eq!(mgr.metrics().allocations(), 1);
        assert_eq!(mgr.metrics().blocks_allocated(), 3);
    }

    #[test]
    fn metrics_track_frees() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table, 10).unwrap();
        assert_eq!(mgr.metrics().blocks_freed(), 0);

        mgr.free_request(&mut table).unwrap();
        assert_eq!(mgr.metrics().blocks_freed(), 3);
    }

    #[test]
    fn metrics_track_partial_frees() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table, 10).unwrap(); // 3 blocks
        table.advance(10);

        let freed = table.trim_to(5); // keep 2 blocks, free 1
        mgr.free_blocks(&freed).unwrap();
        assert_eq!(mgr.metrics().blocks_freed(), 1);

        mgr.free_request(&mut table).unwrap();
        assert_eq!(mgr.metrics().blocks_freed(), 3); // 1 + 2
    }

    #[test]
    fn metrics_shared_across_operations() {
        use std::sync::Arc;

        let config = test_config();
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_metrics(&config, Arc::clone(&metrics)).unwrap();

        let mut table1 = BlockTable::new(config.block_size);
        let mut table2 = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table1, 8).unwrap(); // 2 blocks
        mgr.allocate_for_request(&mut table2, 4).unwrap(); // 1 block

        assert_eq!(metrics.allocations(), 2);
        assert_eq!(metrics.blocks_allocated(), 3);

        mgr.free_request(&mut table1).unwrap();
        mgr.free_request(&mut table2).unwrap();

        assert_eq!(metrics.blocks_freed(), 3);
    }

    #[test]
    fn metrics_snapshot() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table, 12).unwrap(); // 3 blocks
        mgr.free_request(&mut table).unwrap();

        let snap = mgr.metrics().snapshot();
        assert_eq!(snap.allocations, 1);
        assert_eq!(snap.blocks_allocated, 3);
        assert_eq!(snap.blocks_freed, 3);
    }

    // ─── Prefix cache integration tests ───────────────────────────────────────

    #[test]
    fn prefix_cache_disabled_by_default() {
        let config = test_config();
        let mgr = KVCacheManager::new(&config).unwrap();
        assert!(!mgr.has_prefix_cache());
    }

    #[test]
    fn prefix_cache_can_be_enabled() {
        let config = test_config();
        let metrics = Arc::new(KVCacheMetrics::new());
        let mgr = KVCacheManager::with_prefix_cache(&config, metrics).unwrap();
        assert!(mgr.has_prefix_cache());
    }

    #[test]
    fn prefix_cache_match_and_register() {
        let config = test_config();
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, metrics).unwrap();

        // Initially no match
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks (block_size=4)
        let (matched, _) = mgr.match_prefix(&prompt);
        assert!(matched.is_empty());

        // Allocate and register
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 8).unwrap();
        table.advance(8);
        let block_ids = table.block_ids().to_vec();
        mgr.register_prefix(&prompt, &block_ids);

        // Now should match
        let (matched, num_cached) = mgr.match_prefix(&prompt);
        assert_eq!(matched.len(), 2);
        assert_eq!(num_cached, 8);
    }

    #[test]
    fn evict_from_cache_frees_blocks() {
        let mut config = test_config();
        config.num_blocks = 8; // Limited blocks
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Allocate 1 block for a prompt (4 tokens exactly)
        let prompt = vec![1, 2, 3, 4]; // 1 full block
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 4).unwrap();
        table.advance(4);
        let block_ids = table.block_ids().to_vec();
        assert_eq!(block_ids.len(), 1);

        // Register to cache
        mgr.register_prefix(&prompt, &block_ids);

        // Release the request (but blocks stay in cache)
        let to_free = mgr.release_prefix(&prompt, &block_ids);
        assert!(to_free.is_empty()); // All blocks cached

        // Verify cache has blocks
        let (cached, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 1);
        assert_eq!(evictable, 1);

        // Now we have 7 free blocks (8 - 1 cached)
        assert_eq!(mgr.num_free_blocks(), 7);

        // Evict 1 block from cache
        let evicted = mgr.evict_from_cache(1).unwrap();
        assert_eq!(evicted, 1);
        assert_eq!(mgr.num_free_blocks(), 8); // All free now
        assert_eq!(metrics.blocks_evicted(), 1);
    }

    #[test]
    fn allocate_with_eviction_success() {
        let mut config = test_config();
        config.num_blocks = 4; // Very limited
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Fill cache with 1 block (4 tokens)
        let prompt1 = vec![1, 2, 3, 4];
        let mut table1 = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table1, 4).unwrap();
        table1.advance(4);
        let ids1 = table1.block_ids().to_vec();
        assert_eq!(ids1.len(), 1);
        mgr.register_prefix(&prompt1, &ids1);
        mgr.release_prefix(&prompt1, &ids1);

        // Now 1 block in cache (evictable), 3 free
        assert_eq!(mgr.num_free_blocks(), 3);

        // Try to allocate 4 blocks - need to evict 1 from cache
        let mut table2 = BlockTable::new(config.block_size);
        mgr.allocate_with_eviction(&mut table2, 16).unwrap(); // needs 4 blocks
        assert_eq!(table2.block_ids().len(), 4);

        // Should have evicted 1 block
        assert_eq!(metrics.blocks_evicted(), 1);
    }

    #[test]
    fn allocate_with_eviction_fails_if_not_enough() {
        let mut config = test_config();
        config.num_blocks = 4;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Allocate all 4 blocks (no cache, just in use)
        let mut table1 = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table1, 16).unwrap(); // 4 blocks
        assert_eq!(mgr.num_free_blocks(), 0);

        // Try to allocate more - should fail
        let mut table2 = BlockTable::new(config.block_size);
        let result = mgr.allocate_with_eviction(&mut table2, 4);
        assert!(result.is_err());
    }

    #[test]
    fn allocate_with_eviction_partial_eviction() {
        let mut config = test_config();
        config.num_blocks = 6;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Create 2 cached prefixes (1 block each)
        let prompt1 = vec![1, 2, 3, 4]; // 1 block
        let prompt2 = vec![5, 6, 7, 8]; // 1 block (different hash chain)

        let mut table1 = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table1, 4).unwrap();
        table1.advance(4);
        let ids1 = table1.block_ids().to_vec();
        assert_eq!(ids1.len(), 1);
        mgr.register_prefix(&prompt1, &ids1);
        mgr.release_prefix(&prompt1, &ids1);

        let mut table2 = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table2, 4).unwrap();
        table2.advance(4);
        let ids2 = table2.block_ids().to_vec();
        assert_eq!(ids2.len(), 1);
        mgr.register_prefix(&prompt2, &ids2);
        mgr.release_prefix(&prompt2, &ids2);

        // 2 blocks in cache (evictable), 4 free
        let (cached, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 2);
        assert_eq!(evictable, 2);
        assert_eq!(mgr.num_free_blocks(), 4);

        // Allocate 5 blocks - need to evict 1 from cache
        let mut table3 = BlockTable::new(config.block_size);
        mgr.allocate_with_eviction(&mut table3, 20).unwrap(); // needs 5 blocks
        assert_eq!(table3.block_ids().len(), 5);
        assert_eq!(metrics.blocks_evicted(), 1);
    }

    // ─── Copy-on-Write (COW) integration tests ────────────────────────────────

    #[test]
    fn cow_two_requests_share_prefix() {
        let mut config = test_config();
        config.num_blocks = 10;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Request A allocates blocks for a prompt and registers to cache
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks
        let mut table_a = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table_a, 8).unwrap();
        table_a.advance(8);
        let blocks_a = table_a.block_ids().to_vec();
        assert_eq!(blocks_a.len(), 2);
        mgr.register_prefix(&prompt, &blocks_a);

        // After registration, ref_count = 1 (owner), not evictable
        let (cached, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(cached, 2);
        assert_eq!(evictable, 0); // Owner has reference

        // Request B matches the prefix - ref_count = 2 (owner + B)
        let (blocks_b, num_cached) = mgr.match_prefix(&prompt);
        assert_eq!(blocks_b, blocks_a); // Same physical blocks - COW sharing!
        assert_eq!(num_cached, 8);

        // Request C also matches - ref_count = 3 (owner + B + C)
        let (blocks_c, _) = mgr.match_prefix(&prompt);
        assert_eq!(blocks_c, blocks_a);
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 0); // Not evictable

        // Request A finishes (owner releases) - ref_count = 2 (B + C)
        let to_free_a = mgr.release_prefix(&prompt, &blocks_a);
        assert!(to_free_a.is_empty()); // Blocks stay in cache
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 0); // B and C still have references

        // Request B releases - ref_count = 1 (only C)
        let to_free_b = mgr.release_prefix(&prompt, &blocks_b);
        assert!(to_free_b.is_empty());
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 0); // C still has reference

        // Request C releases - ref_count = 0, now evictable
        let to_free_c = mgr.release_prefix(&prompt, &blocks_c);
        assert!(to_free_c.is_empty());
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 2); // Now can be evicted
    }

    #[test]
    fn cow_eviction_respects_active_references() {
        let mut config = test_config();
        config.num_blocks = 6;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Request 1 creates a cached prefix (ref_count = 1)
        let prompt = vec![1, 2, 3, 4]; // 1 block
        let mut table1 = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table1, 4).unwrap();
        table1.advance(4);
        mgr.register_prefix(&prompt, table1.block_ids());

        // Request 2 matches the prefix (ref_count = 2)
        let (matched, _) = mgr.match_prefix(&prompt);
        assert_eq!(matched.len(), 1);

        // Request 1 finishes (ref_count = 1, still Request 2's match)
        let to_free = mgr.release_prefix(&prompt, table1.block_ids());
        assert!(to_free.is_empty()); // Cached block stays

        // Try to evict - should fail because Request 2 has reference
        let evicted = mgr.evict_from_cache(1).unwrap();
        assert_eq!(evicted, 0); // Cannot evict referenced block

        // Request 2 finishes (ref_count = 0)
        mgr.release_prefix(&prompt, &matched);

        // Now eviction should succeed
        let evicted = mgr.evict_from_cache(1).unwrap();
        assert_eq!(evicted, 1);
    }

    #[test]
    fn cow_diverging_requests() {
        let mut config = test_config();
        config.num_blocks = 10;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        // Request A: "Hello world how"
        let prompt_a = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks
        let mut table_a = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table_a, 8).unwrap();
        table_a.advance(8);
        mgr.register_prefix(&prompt_a, table_a.block_ids());

        // Request B: "Hello world why" - same first block, different second
        let prompt_b = vec![1, 2, 3, 4, 9, 9, 9, 9]; // Different second block
        let (matched, num_cached) = mgr.match_prefix(&prompt_b);
        assert_eq!(matched.len(), 1); // Only first block matches
        assert_eq!(num_cached, 4);

        // Request B needs to allocate its own second block
        // (this is the "write" part of copy-on-write)
        let mut table_b = BlockTable::new(config.block_size);
        table_b.append_blocks(&matched); // Reuse matched block
        table_b.advance(4);
        // Allocate the divergent second block
        mgr.allocate_for_request(&mut table_b, 4).unwrap();
        assert_eq!(table_b.block_ids().len(), 2);
        // Second block should be different from A's
        assert_ne!(table_b.block_ids()[1], table_a.block_ids()[1]);
    }
}
