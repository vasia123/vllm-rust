mod block_metrics;
mod block_pool;
mod block_table;
mod cache_engine;
pub mod config;
mod error;
mod free_block_queue;
pub mod metrics;
pub mod mla_cache_config;
pub mod mla_cache_engine;
pub mod offload;
pub mod offload_metrics;
pub mod prefix_cache;
pub mod prefix_cache_stats;
pub mod quantization;

pub use block_metrics::{BlockEvictionEvent, BlockMetricsCollector, BlockMetricsState};
pub use block_pool::{BlockId, NULL_BLOCK};
pub use block_table::BlockTable;
pub use cache_engine::CacheEngine;
pub use config::{CacheConfig, KVCacheLayout};
pub use error::CacheError;
pub use free_block_queue::FreeBlockQueue;
pub use metrics::{KVCacheMetrics, MetricsSnapshot};
pub use mla_cache_config::{MLACacheConfig, MLADims};
pub use mla_cache_engine::MLACacheEngine;
pub use offload::{CpuOffloadConfig, CpuOffloadManager};
pub use offload_metrics::{CpuOffloadMetrics, CpuOffloadMetricsSnapshot};
pub use prefix_cache_stats::{
    export_prometheus_metrics, format_prometheus_output, PrefixCacheStats,
    PrefixCacheStatsSnapshot, PrometheusMetric, PrometheusMetricType, SlidingWindowMetrics,
    SlidingWindowSnapshot,
};
pub use quantization::{
    dequantize_fp8, dequantize_int8, quantize_fp8, quantize_int8, KVCacheDtype, KVScales,
};

use block_pool::BlockPool;
use prefix_cache::PrefixCache;
use std::sync::Arc;

/// Cache variant for supporting different KV storage formats.
///
/// Standard models (Llama, Qwen, Mistral) use `Standard` with full K/V tensors.
/// DeepSeek V2/V3 uses `MLA` with compressed latent representations (42x memory savings).
pub enum CacheVariant {
    /// Standard paged KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
    Standard(CacheEngine),
    /// MLA compressed cache: kv_c [kv_lora_rank] + k_pe [qk_rope_head_dim]
    MLA(MLACacheEngine),
}

impl CacheVariant {
    /// Get as standard cache engine, panics if MLA.
    ///
    /// For a non-panicking alternative, use [`try_as_standard`](Self::try_as_standard).
    pub fn as_standard(&self) -> &CacheEngine {
        match self {
            CacheVariant::Standard(e) => e,
            CacheVariant::MLA(_) => panic!("Expected Standard cache, got MLA"),
        }
    }

    /// Get as standard cache engine (mutable), panics if MLA.
    ///
    /// For a non-panicking alternative, use [`try_as_standard_mut`](Self::try_as_standard_mut).
    pub fn as_standard_mut(&mut self) -> &mut CacheEngine {
        match self {
            CacheVariant::Standard(e) => e,
            CacheVariant::MLA(_) => panic!("Expected Standard cache, got MLA"),
        }
    }

    /// Get as MLA cache engine, panics if Standard.
    ///
    /// For a non-panicking alternative, use [`try_as_mla`](Self::try_as_mla).
    pub fn as_mla(&self) -> &MLACacheEngine {
        match self {
            CacheVariant::MLA(e) => e,
            CacheVariant::Standard(_) => panic!("Expected MLA cache, got Standard"),
        }
    }

    /// Get as MLA cache engine (mutable), panics if Standard.
    ///
    /// For a non-panicking alternative, use [`try_as_mla_mut`](Self::try_as_mla_mut).
    pub fn as_mla_mut(&mut self) -> &mut MLACacheEngine {
        match self {
            CacheVariant::MLA(e) => e,
            CacheVariant::Standard(_) => panic!("Expected MLA cache, got Standard"),
        }
    }

    /// Try to get as standard cache engine, returns error if MLA.
    pub fn try_as_standard(&self) -> Result<&CacheEngine, CacheError> {
        match self {
            CacheVariant::Standard(e) => Ok(e),
            CacheVariant::MLA(_) => Err(CacheError::CacheTypeMismatch {
                expected: "Standard",
                found: "MLA",
            }),
        }
    }

    /// Try to get as standard cache engine (mutable), returns error if MLA.
    pub fn try_as_standard_mut(&mut self) -> Result<&mut CacheEngine, CacheError> {
        match self {
            CacheVariant::Standard(e) => Ok(e),
            CacheVariant::MLA(_) => Err(CacheError::CacheTypeMismatch {
                expected: "Standard",
                found: "MLA",
            }),
        }
    }

    /// Try to get as MLA cache engine, returns error if Standard.
    pub fn try_as_mla(&self) -> Result<&MLACacheEngine, CacheError> {
        match self {
            CacheVariant::MLA(e) => Ok(e),
            CacheVariant::Standard(_) => Err(CacheError::CacheTypeMismatch {
                expected: "MLA",
                found: "Standard",
            }),
        }
    }

    /// Try to get as MLA cache engine (mutable), returns error if Standard.
    pub fn try_as_mla_mut(&mut self) -> Result<&mut MLACacheEngine, CacheError> {
        match self {
            CacheVariant::MLA(e) => Ok(e),
            CacheVariant::Standard(_) => Err(CacheError::CacheTypeMismatch {
                expected: "MLA",
                found: "Standard",
            }),
        }
    }

    /// Check if this is an MLA cache variant.
    pub fn is_mla(&self) -> bool {
        matches!(self, CacheVariant::MLA(_))
    }
}

/// Top-level coordinator: BlockPool + per-layer CacheEngines.
///
/// Supports both standard KV cache and MLA compressed cache through `CacheVariant`.
pub struct KVCacheManager {
    block_pool: BlockPool,
    engines: Vec<CacheVariant>,
    block_size: usize,
    metrics: Arc<KVCacheMetrics>,
    /// Optional prefix cache for block reuse and eviction coordination
    prefix_cache: Option<PrefixCache>,
    /// Optional CPU offload manager for storing evicted blocks in host memory
    cpu_offload: Option<CpuOffloadManager>,
    /// Memory layout of standard KV cache tensors
    layout: KVCacheLayout,
}

impl KVCacheManager {
    /// Create a KVCacheManager with standard KV cache (for Llama, Qwen, Mistral, etc.)
    pub fn new(config: &CacheConfig) -> Result<Self, CacheError> {
        Self::with_metrics(config, Arc::new(KVCacheMetrics::new()))
    }

    /// Create a KVCacheManager with explicit cache layout.
    pub fn with_layout(config: &CacheConfig, layout: KVCacheLayout) -> Result<Self, CacheError> {
        Self::with_metrics_and_layout(config, Arc::new(KVCacheMetrics::new()), layout)
    }

    /// Create a KVCacheManager with custom metrics instance.
    pub fn with_metrics(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
    ) -> Result<Self, CacheError> {
        Self::with_metrics_and_layout(config, metrics, KVCacheLayout::NHD)
    }

    /// Create a KVCacheManager with custom metrics and explicit layout.
    pub fn with_metrics_and_layout(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
        layout: KVCacheLayout,
    ) -> Result<Self, CacheError> {
        let mut engines = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            engines.push(CacheVariant::Standard(CacheEngine::with_layout(
                config, layout,
            )?));
        }

        let cpu_offload = Self::init_cpu_offload(config)?;

        Ok(Self {
            block_pool: BlockPool::new(config.num_blocks),
            engines,
            block_size: config.block_size,
            metrics,
            prefix_cache: None,
            cpu_offload,
            layout,
        })
    }

    /// Create a KVCacheManager with prefix caching enabled.
    pub fn with_prefix_cache(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
    ) -> Result<Self, CacheError> {
        Self::with_prefix_cache_and_layout(config, metrics, KVCacheLayout::NHD)
    }

    /// Create a KVCacheManager with prefix caching and explicit layout.
    pub fn with_prefix_cache_and_layout(
        config: &CacheConfig,
        metrics: Arc<KVCacheMetrics>,
        layout: KVCacheLayout,
    ) -> Result<Self, CacheError> {
        let mut mgr = Self::with_metrics_and_layout(config, Arc::clone(&metrics), layout)?;
        mgr.prefix_cache = Some(PrefixCache::with_metrics(config.block_size, metrics));
        Ok(mgr)
    }

    /// Initialize CPU offload manager from config, if configured.
    fn init_cpu_offload(config: &CacheConfig) -> Result<Option<CpuOffloadManager>, CacheError> {
        match &config.cpu_offload {
            Some(offload_cfg) => {
                let mgr = CpuOffloadManager::new(
                    offload_cfg.clone(),
                    config.num_layers,
                    config.block_size,
                    config.num_kv_heads,
                    config.head_dim,
                    config.dtype,
                )?;
                Ok(Some(mgr))
            }
            None => Ok(None),
        }
    }

    /// Create a KVCacheManager with MLA compressed cache (for DeepSeek V2/V3).
    ///
    /// MLA cache stores compressed latent representations instead of full K/V tensors,
    /// achieving ~42x memory reduction for DeepSeek models.
    pub fn new_mla(config: &MLACacheConfig) -> Result<Self, CacheError> {
        Self::new_mla_with_metrics(config, Arc::new(KVCacheMetrics::new()))
    }

    /// Create a KVCacheManager with MLA cache and custom metrics.
    pub fn new_mla_with_metrics(
        config: &MLACacheConfig,
        metrics: Arc<KVCacheMetrics>,
    ) -> Result<Self, CacheError> {
        let mut engines = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            engines.push(CacheVariant::MLA(MLACacheEngine::new(config)?));
        }
        Ok(Self {
            block_pool: BlockPool::new(config.num_blocks),
            engines,
            block_size: config.block_size,
            metrics,
            prefix_cache: None,
            cpu_offload: None,
            // MLA uses its own layout, this field is only for standard cache
            layout: KVCacheLayout::NHD,
        })
    }

    /// Reset all cache state: zero tensors, reset scales, free all blocks.
    ///
    /// Ensures consistency when restarting generation — stale KV data
    /// from previous requests is cleared. Mirrors upstream vLLM's cache
    /// reset logic that touches all cache types (KV, multimodal, encoder).
    pub fn reset(&mut self) -> Result<(), CacheError> {
        self.block_pool.reset();
        for engine in &mut self.engines {
            match engine {
                CacheVariant::Standard(e) => e.reset()?,
                CacheVariant::MLA(e) => e.reset()?,
            }
        }
        if let Some(ref mut pc) = self.prefix_cache {
            // Returned block IDs don't need to be freed: block_pool.reset() already
            // marked all blocks as free.
            let _ = pc.clear();
        }
        Ok(())
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

    /// Get the CacheEngine for a specific layer (immutable).
    ///
    /// # Panics
    /// Panics if this is an MLA cache manager. Use `mla_engine()` instead.
    pub fn engine(&self, layer_idx: usize) -> &CacheEngine {
        self.engines[layer_idx].as_standard()
    }

    /// Get the CacheEngine for a specific layer (mutable).
    ///
    /// Required for quantized KV cache where write operations update scales.
    ///
    /// # Panics
    /// Panics if this is an MLA cache manager. Use `mla_engine_mut()` instead.
    pub fn engine_mut(&mut self, layer_idx: usize) -> &mut CacheEngine {
        self.engines[layer_idx].as_standard_mut()
    }

    /// Get the MLACacheEngine for a specific layer (immutable).
    ///
    /// # Panics
    /// Panics if this is a standard cache manager. Use `engine()` instead.
    pub fn mla_engine(&self, layer_idx: usize) -> &MLACacheEngine {
        self.engines[layer_idx].as_mla()
    }

    /// Get the MLACacheEngine for a specific layer (mutable).
    ///
    /// # Panics
    /// Panics if this is a standard cache manager. Use `engine_mut()` instead.
    pub fn mla_engine_mut(&mut self, layer_idx: usize) -> &mut MLACacheEngine {
        self.engines[layer_idx].as_mla_mut()
    }

    /// Check if this manager uses MLA compressed cache.
    pub fn is_mla(&self) -> bool {
        self.engines.first().map(|e| e.is_mla()).unwrap_or(false)
    }

    /// Get the memory layout of the standard KV cache.
    pub fn layout(&self) -> KVCacheLayout {
        self.layout
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free()
    }

    pub fn num_total_blocks(&self) -> usize {
        self.block_pool.num_total()
    }

    /// Free specific blocks back to the pool (used during speculative decode rollback
    /// and sliding window reclamation). Silently skips `NULL_BLOCK` sentinels.
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), CacheError> {
        let real_blocks: Vec<BlockId> = block_ids
            .iter()
            .copied()
            .filter(|&id| id != NULL_BLOCK)
            .collect();
        if !real_blocks.is_empty() {
            self.metrics.record_free(real_blocks.len());
            self.block_pool.free(&real_blocks)?;
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

    /// Enable prefix caching on this manager after construction.
    ///
    /// This initializes a PrefixCache using the manager's block size and metrics.
    /// No-op if prefix caching is already enabled.
    pub fn enable_prefix_cache(&mut self) {
        if self.prefix_cache.is_none() {
            self.prefix_cache = Some(PrefixCache::with_metrics(
                self.block_size,
                Arc::clone(&self.metrics),
            ));
        }
    }

    /// Get a reference to the prefix cache (if enabled).
    pub fn prefix_cache(&self) -> Option<&PrefixCache> {
        self.prefix_cache.as_ref()
    }

    /// Get a mutable reference to the prefix cache (if enabled).
    pub fn prefix_cache_mut(&mut self) -> Option<&mut PrefixCache> {
        self.prefix_cache.as_mut()
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

    /// Fork a block table for n>1 generation with KV cache block sharing.
    ///
    /// Creates a copy of `source` that shares the same physical blocks for
    /// prompt tokens. If prefix caching is enabled, increments reference counts
    /// on the cached blocks so they are protected from eviction while the fork
    /// is active. The forked table is ready for independent decode — the caller
    /// allocates new blocks for generated tokens.
    ///
    /// When prefix caching is disabled, the fork shares block IDs directly
    /// (safe because prompt KV data is read-only during decode) but the caller
    /// must ensure the source outlives the fork or manage lifetimes externally.
    pub fn fork_block_table(
        &mut self,
        source: &BlockTable,
        prompt_tokens: &[u32],
    ) -> BlockTable {
        // Increment ref counts on cached prefix blocks so they survive eviction
        if self.prefix_cache.is_some() {
            let _ = self.match_prefix(prompt_tokens);
        }
        source.clone()
    }

    /// Get prefix cache statistics.
    pub fn prefix_cache_stats(&self) -> Option<(usize, usize)> {
        self.prefix_cache
            .as_ref()
            .map(|c| (c.num_cached_blocks(), c.num_evictable_blocks()))
    }

    // ─── CPU offload integration ─────────────────────────────────────────────

    /// Check if CPU offloading is enabled.
    pub fn has_cpu_offload(&self) -> bool {
        self.cpu_offload.is_some()
    }

    /// Offload a GPU block to CPU storage by its content hash.
    ///
    /// The block data is copied from the GPU cache engines into CPU memory.
    /// This is a no-op if CPU offload is not configured or if the cache
    /// variant is MLA (only standard caches are supported).
    pub fn offload_block(
        &mut self,
        block_hash: u64,
        gpu_block_id: BlockId,
    ) -> Result<(), CacheError> {
        let offload = match &mut self.cpu_offload {
            Some(o) => o,
            None => return Ok(()),
        };

        let engine_refs: Vec<&CacheEngine> = self
            .engines
            .iter()
            .filter_map(|v| v.try_as_standard().ok())
            .collect();

        if engine_refs.is_empty() {
            return Ok(());
        }

        offload.store_block_from_refs(gpu_block_id, block_hash, &engine_refs)
    }

    /// Try to load a block from CPU offload cache back into GPU.
    ///
    /// Returns `Some(gpu_block_id)` if the block was found in CPU cache and
    /// loaded into a freshly allocated GPU block, or `None` if not found or
    /// offload is disabled.
    pub fn try_load_from_cpu(&mut self, block_hash: u64) -> Result<Option<BlockId>, CacheError> {
        // Check presence without mutable borrow.
        let has_block = self
            .cpu_offload
            .as_ref()
            .map(|o| o.has_block(block_hash))
            .unwrap_or(false);

        if !has_block {
            return Ok(None);
        }

        // Allocate a GPU block for the loaded data.
        let gpu_block_ids = self.block_pool.allocate(1)?;
        let gpu_block_id = gpu_block_ids[0];
        self.metrics.record_allocation(1);

        let mut engine_refs: Vec<&mut CacheEngine> = self
            .engines
            .iter_mut()
            .filter_map(|v| v.try_as_standard_mut().ok())
            .collect();

        if engine_refs.is_empty() {
            self.block_pool.free(&[gpu_block_id])?;
            self.metrics.record_free(1);
            return Ok(None);
        }

        let loaded = self
            .cpu_offload
            .as_mut()
            .expect("checked has_block above")
            .load_block_from_refs(block_hash, gpu_block_id, &mut engine_refs)?;

        if loaded {
            Ok(Some(gpu_block_id))
        } else {
            self.block_pool.free(&[gpu_block_id])?;
            self.metrics.record_free(1);
            Ok(None)
        }
    }

    /// Get a reference to the CPU offload metrics, if offloading is enabled.
    pub fn cpu_offload_metrics(&self) -> Option<&offload_metrics::CpuOffloadMetrics> {
        self.cpu_offload.as_ref().map(|o| o.metrics())
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
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
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
        let mut mgr = KVCacheManager::new(&config).unwrap();

        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();

        // Write to layer 0, slots 0,1,2 (block 0)
        mgr.engine_mut(0).write(&k, &v, &[0, 1, 2]).unwrap();

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

    // ─── fork_block_table tests ────────────────────────────────────────────────

    #[test]
    fn fork_block_table_creates_clone() {
        let mut config = test_config();
        config.num_blocks = 16;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 8).unwrap();
        table.advance(8);
        mgr.register_prefix(&prompt, table.block_ids());

        // Fork creates a clone with same block IDs
        let forked = mgr.fork_block_table(&table, &prompt);
        assert_eq!(forked.block_ids(), table.block_ids());
        assert_eq!(forked.num_tokens(), table.num_tokens());
    }

    #[test]
    fn fork_block_table_increments_refcount() {
        let mut config = test_config();
        config.num_blocks = 16;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 8).unwrap();
        table.advance(8);
        let block_ids = table.block_ids().to_vec();
        mgr.register_prefix(&prompt, &block_ids);

        // Fork increments ref_count via match_prefix
        let _forked = mgr.fork_block_table(&table, &prompt);

        // Release original owner — blocks should NOT be evictable (fork has reference)
        let to_free = mgr.release_prefix(&prompt, &block_ids);
        assert!(to_free.is_empty());
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 0); // Fork still holds reference

        // Release fork's reference — now evictable
        mgr.release_prefix(&prompt, &block_ids);
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 2);
    }

    #[test]
    fn fork_block_table_multiple_forks() {
        let mut config = test_config();
        config.num_blocks = 16;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        let prompt = vec![1, 2, 3, 4]; // 1 block
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 4).unwrap();
        table.advance(4);
        let block_ids = table.block_ids().to_vec();
        mgr.register_prefix(&prompt, &block_ids);

        // Fork 5 times (simulating n=5 best_of)
        let forks: Vec<_> = (0..5)
            .map(|_| mgr.fork_block_table(&table, &prompt))
            .collect();

        // All forks share the same physical blocks
        for fork in &forks {
            assert_eq!(fork.block_ids(), table.block_ids());
        }

        // Release all: owner + 5 forks = 6 references total
        // Need to release 6 times before blocks become evictable
        for _ in 0..6 {
            let (_, evictable) = mgr.prefix_cache_stats().unwrap();
            assert_eq!(evictable, 0);
            mgr.release_prefix(&prompt, &block_ids);
        }
        let (_, evictable) = mgr.prefix_cache_stats().unwrap();
        assert_eq!(evictable, 1);
    }

    #[test]
    fn fork_block_table_without_prefix_cache() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        assert!(!mgr.has_prefix_cache());

        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 8).unwrap();
        table.advance(8);

        // Fork without prefix cache still clones the table
        let forked = mgr.fork_block_table(&table, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(forked.block_ids(), table.block_ids());
        assert_eq!(forked.num_tokens(), 8);
    }

    #[test]
    fn fork_block_table_independent_decode() {
        let mut config = test_config();
        config.num_blocks = 20;
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut mgr = KVCacheManager::with_prefix_cache(&config, Arc::clone(&metrics)).unwrap();

        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks
        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 8).unwrap();
        table.advance(8);
        mgr.register_prefix(&prompt, table.block_ids());

        // Fork and allocate independent decode blocks
        let mut fork1 = mgr.fork_block_table(&table, &prompt);
        let mut fork2 = mgr.fork_block_table(&table, &prompt);

        mgr.allocate_for_request(&mut fork1, 4).unwrap(); // 1 new block
        mgr.allocate_for_request(&mut fork2, 4).unwrap(); // 1 new block

        // Shared prompt blocks are the same
        assert_eq!(fork1.block_ids()[0], fork2.block_ids()[0]);
        assert_eq!(fork1.block_ids()[1], fork2.block_ids()[1]);

        // Decode blocks are different (independently allocated)
        assert_ne!(fork1.block_ids()[2], fork2.block_ids()[2]);
    }

    // ─── CacheVariant and MLA cache tests ────────────────────────────────────────

    fn test_mla_config() -> MLACacheConfig {
        MLACacheConfig::new(
            32, // kv_lora_rank
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            4,  // num_heads
            4,  // block_size
            16, // num_blocks
            2,  // num_layers
            DType::F32,
            Device::Cpu,
        )
    }

    #[test]
    fn cache_variant_is_mla() {
        let standard = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        assert!(!standard.is_mla());

        let mla = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        assert!(mla.is_mla());
    }

    #[test]
    fn cache_variant_as_standard() {
        let mut variant = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        let _ = variant.as_standard();
        let _ = variant.as_standard_mut();
    }

    #[test]
    #[should_panic(expected = "Expected Standard cache, got MLA")]
    fn cache_variant_as_standard_panics_on_mla() {
        let variant = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        let _ = variant.as_standard();
    }

    #[test]
    fn cache_variant_as_mla() {
        let mut variant = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        let _ = variant.as_mla();
        let _ = variant.as_mla_mut();
    }

    #[test]
    #[should_panic(expected = "Expected MLA cache, got Standard")]
    fn cache_variant_as_mla_panics_on_standard() {
        let variant = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        let _ = variant.as_mla();
    }

    #[test]
    fn cache_variant_try_as_standard_ok() {
        let variant = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        assert!(variant.try_as_standard().is_ok());
    }

    #[test]
    fn cache_variant_try_as_standard_err() {
        let variant = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        let result = variant.try_as_standard();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(
            err,
            CacheError::CacheTypeMismatch {
                expected: "Standard",
                found: "MLA"
            }
        ));
    }

    #[test]
    fn cache_variant_try_as_mla_ok() {
        let variant = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        assert!(variant.try_as_mla().is_ok());
    }

    #[test]
    fn cache_variant_try_as_mla_err() {
        let variant = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        let result = variant.try_as_mla();
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(
            err,
            CacheError::CacheTypeMismatch {
                expected: "MLA",
                found: "Standard"
            }
        ));
    }

    #[test]
    fn cache_variant_try_as_standard_mut_ok() {
        let mut variant = CacheVariant::Standard(CacheEngine::new(&test_config()).unwrap());
        assert!(variant.try_as_standard_mut().is_ok());
    }

    #[test]
    fn cache_variant_try_as_mla_mut_ok() {
        let mut variant = CacheVariant::MLA(MLACacheEngine::new(&test_mla_config()).unwrap());
        assert!(variant.try_as_mla_mut().is_ok());
    }

    #[test]
    fn kv_cache_manager_new_mla() {
        let config = test_mla_config();
        let mgr = KVCacheManager::new_mla(&config).unwrap();

        assert!(mgr.is_mla());
        assert_eq!(mgr.num_free_blocks(), 16);
        assert_eq!(mgr.block_size(), 4);
    }

    #[test]
    fn kv_cache_manager_standard_is_not_mla() {
        let config = test_config();
        let mgr = KVCacheManager::new(&config).unwrap();

        assert!(!mgr.is_mla());
        assert_eq!(mgr.layout(), KVCacheLayout::NHD);
    }

    #[test]
    fn kv_cache_manager_with_hnd_layout() {
        let config = test_config();
        let mgr = KVCacheManager::with_layout(&config, KVCacheLayout::HND).unwrap();

        assert_eq!(mgr.layout(), KVCacheLayout::HND);
        // HND engine allocates [num_blocks, kv_heads, block_size, head_dim]
        assert_eq!(mgr.engine(0).k_cache().dims(), &[16, 2, 4, 8]);
        assert_eq!(mgr.engine(0).layout(), KVCacheLayout::HND);
    }

    #[test]
    fn kv_cache_manager_mla_engine_mut() {
        let config = test_mla_config();
        let mut mgr = KVCacheManager::new_mla(&config).unwrap();

        // Should work for MLA manager
        let _ = mgr.mla_engine_mut(0);
        let _ = mgr.mla_engine_mut(1);
    }

    #[test]
    #[should_panic(expected = "Expected Standard cache, got MLA")]
    fn kv_cache_manager_engine_mut_panics_on_mla() {
        let config = test_mla_config();
        let mut mgr = KVCacheManager::new_mla(&config).unwrap();

        // Should panic because this is an MLA manager
        let _ = mgr.engine_mut(0);
    }

    #[test]
    #[should_panic(expected = "Expected MLA cache, got Standard")]
    fn kv_cache_manager_mla_engine_mut_panics_on_standard() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();

        // Should panic because this is a standard manager
        let _ = mgr.mla_engine_mut(0);
    }

    #[test]
    fn kv_cache_manager_mla_allocate_and_free() {
        let config = test_mla_config();
        let mut mgr = KVCacheManager::new_mla(&config).unwrap();

        assert_eq!(mgr.num_free_blocks(), 16);

        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 10).unwrap(); // needs 3 blocks
        assert_eq!(mgr.num_free_blocks(), 13);

        mgr.free_request(&mut table).unwrap();
        assert_eq!(mgr.num_free_blocks(), 16);
    }

    #[test]
    fn kv_cache_manager_mla_write_read() {
        let config = test_mla_config();
        let mut mgr = KVCacheManager::new_mla(&config).unwrap();

        // Create test data
        let kv_c = Tensor::randn(0f32, 1f32, (2, 32), &Device::Cpu).unwrap();
        let k_pe = Tensor::randn(0f32, 1f32, (2, 8), &Device::Cpu).unwrap();

        // Write to MLA cache
        mgr.mla_engine_mut(0).write(&kv_c, &k_pe, &[0, 1]).unwrap();

        // Read back raw
        let (kv_c_out, k_pe_out) = mgr.mla_engine(0).read_raw(&[0], 2).unwrap();
        assert_eq!(kv_c_out.dims(), &[2, 32]);
        assert_eq!(k_pe_out.dims(), &[2, 8]);
    }

    // ─── CPU offload integration tests ───────────────────────────────────────

    fn test_config_with_offload() -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: Some(offload::CpuOffloadConfig {
                max_cpu_blocks: 4,
                use_pinned_memory: false,
                prefetch_count: 2,
            }),
        }
    }

    #[test]
    fn cpu_offload_disabled_by_default() {
        let config = test_config();
        let mgr = KVCacheManager::new(&config).unwrap();
        assert!(!mgr.has_cpu_offload());
        assert!(mgr.cpu_offload_metrics().is_none());
    }

    #[test]
    fn cpu_offload_enabled_when_configured() {
        let config = test_config_with_offload();
        let mgr = KVCacheManager::new(&config).unwrap();
        assert!(mgr.has_cpu_offload());
        assert!(mgr.cpu_offload_metrics().is_some());
    }

    #[test]
    fn cpu_offload_store_and_load_through_manager() {
        let config = test_config_with_offload();
        let mut mgr = KVCacheManager::new(&config).unwrap();

        // Write data to GPU block 0 in all layers.
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i + 1) as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();

        for layer in 0..2 {
            mgr.engine_mut(layer).write(&k, &v, &[0, 1, 2]).unwrap();
        }

        // Offload block 0 with hash 42.
        mgr.offload_block(42, 0).unwrap();

        // Verify metrics.
        let metrics = mgr.cpu_offload_metrics().unwrap();
        assert_eq!(metrics.stores(), 1);

        // Load it back into a new GPU block via try_load_from_cpu.
        let loaded_block = mgr.try_load_from_cpu(42).unwrap();
        assert!(loaded_block.is_some());

        let gpu_block = loaded_block.unwrap();

        // Verify the data is correct in the newly allocated GPU block.
        for layer in 0..2 {
            let (k_out, v_out) = mgr.engine(layer).read(&[gpu_block], 3).unwrap();
            let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
            let v_flat: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(k_flat, k_data);
            assert_eq!(v_flat, k_data);
        }
    }

    #[test]
    fn cpu_offload_miss_returns_none() {
        let config = test_config_with_offload();
        let mut mgr = KVCacheManager::new(&config).unwrap();

        let result = mgr.try_load_from_cpu(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn cpu_offload_noop_when_disabled() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();

        // These should be no-ops, not errors.
        mgr.offload_block(42, 0).unwrap();
        let result = mgr.try_load_from_cpu(42).unwrap();
        assert!(result.is_none());
    }
}
