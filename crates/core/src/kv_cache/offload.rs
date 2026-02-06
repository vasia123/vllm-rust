//! CPU offload manager for KV cache blocks.
//!
//! When GPU VRAM is limited (e.g., 8GB), this module allows evicted KV cache
//! blocks to be stored in CPU memory instead of being discarded. On a subsequent
//! request with the same prefix, the block can be loaded from CPU rather than
//! recomputed, saving significant prefill time.
//!
//! The CPU cache uses LRU eviction when full. Block data is copied (not moved),
//! so the GPU block is immediately available for reuse after offload.

use std::collections::{HashMap, VecDeque};

use candle_core::{DType, Device, Tensor};

use super::block_pool::BlockId;
use super::cache_engine::CacheEngine;
use super::error::CacheError;
use super::offload_metrics::CpuOffloadMetrics;

/// Identifier for a CPU-resident cache block.
pub type CpuBlockId = usize;

/// Configuration for CPU offload behavior.
#[derive(Debug, Clone)]
pub struct CpuOffloadConfig {
    /// Maximum number of CPU-resident blocks.
    pub max_cpu_blocks: usize,
    /// Whether to use pinned (page-locked) memory for faster transfers.
    ///
    /// NOTE: Pinned memory allocation requires CUDA runtime support.
    /// This flag is stored for future use when CUDA pinned memory is implemented.
    /// Currently all CPU tensors use standard allocations via candle_core::Device::Cpu.
    pub use_pinned_memory: bool,
    /// Number of blocks to prefetch ahead during decode.
    pub prefetch_count: usize,
}

impl Default for CpuOffloadConfig {
    fn default() -> Self {
        Self {
            max_cpu_blocks: 256,
            use_pinned_memory: false,
            prefetch_count: 2,
        }
    }
}

/// Entry tracking a single block stored in the CPU cache.
struct CpuCacheEntry {
    /// Index into cpu_k_cache / cpu_v_cache per-layer Vecs.
    cpu_block_id: CpuBlockId,
}

/// Manages CPU-side storage of evicted KV cache blocks.
///
/// Each layer gets its own pair of CPU tensors with shape
/// `[max_cpu_blocks, block_size, num_kv_heads, head_dim]`. Blocks are identified
/// by their content hash (from prefix cache) and stored/loaded by layer.
pub struct CpuOffloadManager {
    config: CpuOffloadConfig,
    /// CPU-side K cache tensors, one per layer.
    /// Shape per tensor: [max_cpu_blocks, block_size, num_kv_heads, head_dim]
    cpu_k_cache: Vec<Tensor>,
    /// CPU-side V cache tensors, one per layer.
    cpu_v_cache: Vec<Tensor>,
    /// Mapping: block_hash -> cpu cache entry.
    cpu_block_map: HashMap<u64, CpuCacheEntry>,
    /// LRU tracking: front = oldest, back = newest.
    cpu_lru: VecDeque<u64>,
    /// Free CPU block IDs available for allocation.
    cpu_free_list: Vec<CpuBlockId>,
    /// Blocks scheduled for prefetch (hash, target GPU block).
    prefetch_queue: Vec<(u64, BlockId)>,
    /// Per-layer block shape metadata.
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    /// Observability.
    metrics: CpuOffloadMetrics,
}

impl CpuOffloadManager {
    /// Allocate CPU buffers for KV cache offloading.
    ///
    /// Creates `num_layers` pairs of CPU tensors, each sized to hold
    /// `max_cpu_blocks` blocks.
    pub fn new(
        config: CpuOffloadConfig,
        num_layers: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self, CacheError> {
        let shape = (config.max_cpu_blocks, block_size, num_kv_heads, head_dim);

        let mut cpu_k_cache = Vec::with_capacity(num_layers);
        let mut cpu_v_cache = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            cpu_k_cache.push(Tensor::zeros(shape, dtype, &Device::Cpu)?);
            cpu_v_cache.push(Tensor::zeros(shape, dtype, &Device::Cpu)?);
        }

        // Initialize free list: all CPU block IDs are available.
        let cpu_free_list: Vec<CpuBlockId> = (0..config.max_cpu_blocks).rev().collect();

        Ok(Self {
            config,
            cpu_k_cache,
            cpu_v_cache,
            cpu_block_map: HashMap::new(),
            cpu_lru: VecDeque::new(),
            cpu_free_list,
            prefetch_queue: Vec::new(),
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            metrics: CpuOffloadMetrics::new(),
        })
    }

    /// Copy a GPU block to CPU storage.
    ///
    /// Reads the block from the GPU cache engine and writes it into a free
    /// CPU slot. If the CPU cache is full, LRU eviction is performed first.
    /// If the block hash is already in the CPU cache, this is a no-op.
    pub fn store_block(
        &mut self,
        gpu_block_id: BlockId,
        block_hash: u64,
        cache_engines: &[CacheEngine],
    ) -> Result<(), CacheError> {
        // Already stored -- nothing to do.
        if self.cpu_block_map.contains_key(&block_hash) {
            return Ok(());
        }

        // Ensure there is a free CPU slot, evicting if necessary.
        if self.cpu_free_list.is_empty() && self.evict_cpu_block().is_none() {
            // CPU cache is full and nothing is evictable (should not happen
            // because we always allow eviction from LRU, but guard anyway).
            return Ok(());
        }

        let cpu_id = self
            .cpu_free_list
            .pop()
            .expect("free list non-empty after eviction check");

        // Copy block data from each layer's GPU engine to CPU.
        for layer_idx in 0..self.num_layers {
            if layer_idx >= cache_engines.len() {
                break;
            }
            let engine = &cache_engines[layer_idx];
            self.copy_block_gpu_to_cpu(engine, gpu_block_id, layer_idx, cpu_id)?;
        }

        self.cpu_block_map.insert(
            block_hash,
            CpuCacheEntry {
                cpu_block_id: cpu_id,
            },
        );
        self.cpu_lru.push_back(block_hash);
        self.metrics.record_store();

        Ok(())
    }

    /// Load a CPU-cached block back into a GPU block.
    ///
    /// Returns `true` if the block was found in CPU cache and loaded,
    /// `false` if it was a miss.
    pub fn load_block(
        &mut self,
        block_hash: u64,
        gpu_block_id: BlockId,
        cache_engines: &mut [CacheEngine],
    ) -> Result<bool, CacheError> {
        let cpu_id = match self.cpu_block_map.get(&block_hash) {
            Some(entry) => {
                self.metrics.record_hit();
                entry.cpu_block_id
            }
            None => {
                self.metrics.record_miss();
                return Ok(false);
            }
        };

        // Touch LRU: move this hash to the back (most recently used).
        self.touch_lru(block_hash);

        // Copy data from CPU to each layer's GPU engine.
        for layer_idx in 0..self.num_layers {
            if layer_idx >= cache_engines.len() {
                break;
            }
            let engine = &mut cache_engines[layer_idx];
            self.copy_block_cpu_to_gpu(cpu_id, layer_idx, engine, gpu_block_id)?;
        }

        self.metrics.record_load();
        Ok(true)
    }

    /// Store a GPU block to CPU, using a slice of immutable engine references.
    ///
    /// This variant avoids the split-borrow issue when called from
    /// `KVCacheManager`, where `self.engines` and `self.cpu_offload` cannot
    /// both be mutably borrowed.
    pub fn store_block_from_refs(
        &mut self,
        gpu_block_id: BlockId,
        block_hash: u64,
        cache_engines: &[&CacheEngine],
    ) -> Result<(), CacheError> {
        if self.cpu_block_map.contains_key(&block_hash) {
            return Ok(());
        }

        if self.cpu_free_list.is_empty() && self.evict_cpu_block().is_none() {
            return Ok(());
        }

        let cpu_id = self
            .cpu_free_list
            .pop()
            .expect("free list non-empty after eviction check");

        for layer_idx in 0..self.num_layers {
            if layer_idx >= cache_engines.len() {
                break;
            }
            self.copy_block_gpu_to_cpu(cache_engines[layer_idx], gpu_block_id, layer_idx, cpu_id)?;
        }

        self.cpu_block_map.insert(
            block_hash,
            CpuCacheEntry {
                cpu_block_id: cpu_id,
            },
        );
        self.cpu_lru.push_back(block_hash);
        self.metrics.record_store();

        Ok(())
    }

    /// Load a CPU block to GPU, using a slice of mutable engine references.
    ///
    /// This variant avoids the split-borrow issue when called from
    /// `KVCacheManager`.
    pub fn load_block_from_refs(
        &mut self,
        block_hash: u64,
        gpu_block_id: BlockId,
        cache_engines: &mut [&mut CacheEngine],
    ) -> Result<bool, CacheError> {
        let cpu_id = match self.cpu_block_map.get(&block_hash) {
            Some(entry) => {
                self.metrics.record_hit();
                entry.cpu_block_id
            }
            None => {
                self.metrics.record_miss();
                return Ok(false);
            }
        };

        self.touch_lru(block_hash);

        for layer_idx in 0..self.num_layers {
            if layer_idx >= cache_engines.len() {
                break;
            }
            self.copy_block_cpu_to_gpu(cpu_id, layer_idx, cache_engines[layer_idx], gpu_block_id)?;
        }

        self.metrics.record_load();
        Ok(true)
    }

    /// Check if a block with the given hash is in the CPU cache.
    pub fn has_block(&self, block_hash: u64) -> bool {
        self.cpu_block_map.contains_key(&block_hash)
    }

    /// Evict the least recently used CPU block.
    ///
    /// Returns the hash of the evicted block, or `None` if the CPU cache is empty.
    pub fn evict_cpu_block(&mut self) -> Option<u64> {
        // Pop from front of LRU (oldest).
        while let Some(hash) = self.cpu_lru.pop_front() {
            if let Some(entry) = self.cpu_block_map.remove(&hash) {
                self.cpu_free_list.push(entry.cpu_block_id);
                self.metrics.record_eviction();
                return Some(hash);
            }
            // Hash was already removed (shouldn't happen in normal flow, but
            // defensive programming).
        }
        None
    }

    /// Schedule block hashes for prefetching from CPU to GPU.
    ///
    /// Only blocks that are actually in the CPU cache are queued.
    pub fn schedule_prefetch(&mut self, block_hashes: &[(u64, BlockId)]) {
        let max = self.config.prefetch_count;
        let mut count = 0;

        for &(hash, target_gpu_block) in block_hashes {
            if count >= max {
                break;
            }
            if self.has_block(hash) {
                self.prefetch_queue.push((hash, target_gpu_block));
                count += 1;
            }
        }
    }

    /// Execute all pending prefetches, loading CPU blocks into GPU.
    ///
    /// Drains the prefetch queue and loads each block. Returns the number
    /// of blocks successfully prefetched.
    pub fn execute_prefetches(
        &mut self,
        cache_engines: &mut [CacheEngine],
    ) -> Result<usize, CacheError> {
        let queue: Vec<(u64, BlockId)> = std::mem::take(&mut self.prefetch_queue);
        let mut loaded = 0;

        for (hash, gpu_block_id) in queue {
            if self.load_block(hash, gpu_block_id, cache_engines)? {
                self.metrics.record_prefetch();
                loaded += 1;
            }
        }

        Ok(loaded)
    }

    /// Number of blocks currently stored in CPU cache.
    pub fn num_stored_blocks(&self) -> usize {
        self.cpu_block_map.len()
    }

    /// Number of free CPU block slots.
    pub fn num_free_cpu_blocks(&self) -> usize {
        self.cpu_free_list.len()
    }

    /// Maximum CPU blocks this manager can hold.
    pub fn max_cpu_blocks(&self) -> usize {
        self.config.max_cpu_blocks
    }

    /// Get a reference to the offload metrics.
    pub fn metrics(&self) -> &CpuOffloadMetrics {
        &self.metrics
    }

    /// Number of pending prefetch operations.
    pub fn pending_prefetches(&self) -> usize {
        self.prefetch_queue.len()
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Copy a single block from GPU cache engine to CPU tensor storage.
    fn copy_block_gpu_to_cpu(
        &self,
        engine: &CacheEngine,
        gpu_block_id: BlockId,
        layer_idx: usize,
        cpu_block_id: CpuBlockId,
    ) -> Result<(), CacheError> {
        let total_gpu_slots = engine.k_cache().dims()[0] * engine.k_cache().dims()[1];
        let start_slot = gpu_block_id * self.block_size;
        let end_slot = start_slot + self.block_size;

        if end_slot > total_gpu_slots {
            return Err(CacheError::BlockNotAllocated {
                block_id: gpu_block_id,
            });
        }

        // Extract block data from GPU: narrow on the flattened slot dimension.
        let k_flat =
            engine
                .k_cache()
                .reshape((total_gpu_slots, self.num_kv_heads, self.head_dim))?;
        let v_flat =
            engine
                .v_cache()
                .reshape((total_gpu_slots, self.num_kv_heads, self.head_dim))?;

        let k_block = k_flat
            .narrow(0, start_slot, self.block_size)?
            .contiguous()?;
        let v_block = v_flat
            .narrow(0, start_slot, self.block_size)?
            .contiguous()?;

        // Move to CPU if not already there (for GPU -> CPU transfers).
        let k_cpu = k_block.to_device(&Device::Cpu)?;
        let v_cpu = v_block.to_device(&Device::Cpu)?;

        // Write into the CPU cache tensor at the assigned slot.
        let total_cpu_slots = self.config.max_cpu_blocks * self.block_size;
        let cpu_start = cpu_block_id * self.block_size;

        let cpu_k_flat = self.cpu_k_cache[layer_idx].reshape((
            total_cpu_slots,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let cpu_v_flat = self.cpu_v_cache[layer_idx].reshape((
            total_cpu_slots,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        // Build scatter indices for the block_size slots.
        let indices_vec: Vec<u32> = (cpu_start..cpu_start + self.block_size)
            .map(|s| s as u32)
            .collect();
        let indices = Tensor::from_vec(indices_vec, (self.block_size,), &Device::Cpu)?;
        let indices = indices
            .reshape((self.block_size, 1, 1))?
            .expand((self.block_size, self.num_kv_heads, self.head_dim))?
            .contiguous()?;

        cpu_k_flat.scatter_set(&indices, &k_cpu, 0)?;
        cpu_v_flat.scatter_set(&indices, &v_cpu, 0)?;

        Ok(())
    }

    /// Copy a single block from CPU tensor storage to GPU cache engine.
    fn copy_block_cpu_to_gpu(
        &self,
        cpu_block_id: CpuBlockId,
        layer_idx: usize,
        engine: &mut CacheEngine,
        gpu_block_id: BlockId,
    ) -> Result<(), CacheError> {
        let total_cpu_slots = self.config.max_cpu_blocks * self.block_size;
        let cpu_start = cpu_block_id * self.block_size;

        // Extract block from CPU cache.
        let cpu_k_flat = self.cpu_k_cache[layer_idx].reshape((
            total_cpu_slots,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let cpu_v_flat = self.cpu_v_cache[layer_idx].reshape((
            total_cpu_slots,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let k_block = cpu_k_flat
            .narrow(0, cpu_start, self.block_size)?
            .contiguous()?;
        let v_block = cpu_v_flat
            .narrow(0, cpu_start, self.block_size)?
            .contiguous()?;

        // Move to GPU device if the cache engine is on a different device.
        let target_device = engine.k_cache().device().clone();
        let k_gpu = k_block.to_device(&target_device)?;
        let v_gpu = v_block.to_device(&target_device)?;

        // Write into GPU engine at the target block's slots.
        let total_gpu_slots = engine.k_cache().dims()[0] * engine.k_cache().dims()[1];
        let gpu_start = gpu_block_id * self.block_size;

        let gpu_k_flat =
            engine
                .k_cache()
                .reshape((total_gpu_slots, self.num_kv_heads, self.head_dim))?;
        let gpu_v_flat =
            engine
                .v_cache()
                .reshape((total_gpu_slots, self.num_kv_heads, self.head_dim))?;

        let indices_vec: Vec<u32> = (gpu_start..gpu_start + self.block_size)
            .map(|s| s as u32)
            .collect();
        let indices = Tensor::from_vec(indices_vec, (self.block_size,), &target_device)?;
        let indices = indices
            .reshape((self.block_size, 1, 1))?
            .expand((self.block_size, self.num_kv_heads, self.head_dim))?
            .contiguous()?;

        gpu_k_flat.scatter_set(&indices, &k_gpu, 0)?;
        gpu_v_flat.scatter_set(&indices, &v_gpu, 0)?;

        Ok(())
    }

    /// Move a hash to the back of the LRU queue (most recently used).
    fn touch_lru(&mut self, hash: u64) {
        // Linear scan is acceptable here because the LRU queue size is bounded
        // by max_cpu_blocks (typically a few hundred).
        if let Some(pos) = self.cpu_lru.iter().position(|&h| h == hash) {
            self.cpu_lru.remove(pos);
        }
        self.cpu_lru.push_back(hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::quantization::KVCacheDtype;

    fn test_cache_config() -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn make_engines(config: &CacheConfig) -> Vec<CacheEngine> {
        (0..config.num_layers)
            .map(|_| CacheEngine::new(config).unwrap())
            .collect()
    }

    fn offload_config(max_blocks: usize) -> CpuOffloadConfig {
        CpuOffloadConfig {
            max_cpu_blocks: max_blocks,
            use_pinned_memory: false,
            prefetch_count: 2,
        }
    }

    #[test]
    fn new_creates_correct_state() {
        let cfg = test_cache_config();
        let mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        assert_eq!(mgr.num_stored_blocks(), 0);
        assert_eq!(mgr.num_free_cpu_blocks(), 4);
        assert_eq!(mgr.max_cpu_blocks(), 4);
        assert_eq!(mgr.pending_prefetches(), 0);
        assert_eq!(mgr.cpu_k_cache.len(), cfg.num_layers);
        assert_eq!(mgr.cpu_v_cache.len(), cfg.num_layers);
    }

    #[test]
    fn store_and_load_roundtrip() {
        let cfg = test_cache_config();
        let mut engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        // Write known data into GPU block 0 (slots 0..3) in each layer.
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i + 1) as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();

        for engine in engines.iter_mut() {
            engine.write(&k, &v, &[0, 1, 2]).unwrap();
        }

        // Store GPU block 0 to CPU under hash 42.
        mgr.store_block(0, 42, &engines).unwrap();

        assert_eq!(mgr.num_stored_blocks(), 1);
        assert!(mgr.has_block(42));
        assert!(!mgr.has_block(99));

        // Clear GPU block 0 in fresh engines, then load from CPU.
        let mut fresh_engines = make_engines(&cfg);
        let loaded = mgr.load_block(42, 0, &mut fresh_engines).unwrap();
        assert!(loaded);

        // Verify the data matches.
        for engine in &fresh_engines {
            let (k_out, v_out) = engine.read(&[0], 3).unwrap();
            let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
            let v_flat: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(k_flat, k_data);
            assert_eq!(v_flat, k_data);
        }

        // Verify metrics.
        assert_eq!(mgr.metrics().stores(), 1);
        assert_eq!(mgr.metrics().loads(), 1);
        assert_eq!(mgr.metrics().cpu_hits(), 1);
    }

    #[test]
    fn load_miss_returns_false() {
        let cfg = test_cache_config();
        let mut engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        let loaded = mgr.load_block(999, 0, &mut engines).unwrap();
        assert!(!loaded);
        assert_eq!(mgr.metrics().cpu_misses(), 1);
        assert_eq!(mgr.metrics().cpu_hits(), 0);
    }

    #[test]
    fn lru_eviction_when_full() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(2), 2, 4, 2, 8, DType::F32).unwrap();

        // Fill CPU cache with 2 blocks.
        mgr.store_block(0, 100, &engines).unwrap();
        mgr.store_block(1, 200, &engines).unwrap();
        assert_eq!(mgr.num_stored_blocks(), 2);
        assert_eq!(mgr.num_free_cpu_blocks(), 0);

        // Store a third block -- should evict hash 100 (LRU).
        mgr.store_block(2, 300, &engines).unwrap();
        assert_eq!(mgr.num_stored_blocks(), 2);
        assert!(!mgr.has_block(100)); // evicted
        assert!(mgr.has_block(200));
        assert!(mgr.has_block(300));
        assert_eq!(mgr.metrics().evictions(), 1);
    }

    #[test]
    fn lru_touch_on_load_updates_order() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(2), 2, 4, 2, 8, DType::F32).unwrap();

        // Fill CPU cache.
        mgr.store_block(0, 100, &engines).unwrap();
        mgr.store_block(1, 200, &engines).unwrap();

        // Load hash 100 to make it recently used.
        let mut mut_engines = make_engines(&cfg);
        mgr.load_block(100, 0, &mut mut_engines).unwrap();

        // Store a third block -- should evict hash 200 (now LRU), not 100.
        mgr.store_block(2, 300, &engines).unwrap();
        assert!(mgr.has_block(100)); // kept (recently accessed)
        assert!(!mgr.has_block(200)); // evicted (LRU)
        assert!(mgr.has_block(300));
    }

    #[test]
    fn duplicate_store_is_noop() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        mgr.store_block(0, 42, &engines).unwrap();
        mgr.store_block(0, 42, &engines).unwrap(); // duplicate
        assert_eq!(mgr.num_stored_blocks(), 1);
        assert_eq!(mgr.metrics().stores(), 1); // only counted once
    }

    #[test]
    fn evict_cpu_block_empty() {
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();
        assert_eq!(mgr.evict_cpu_block(), None);
    }

    #[test]
    fn evict_cpu_block_returns_lru() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        mgr.store_block(0, 10, &engines).unwrap();
        mgr.store_block(1, 20, &engines).unwrap();
        mgr.store_block(2, 30, &engines).unwrap();

        let evicted = mgr.evict_cpu_block();
        assert_eq!(evicted, Some(10)); // oldest
        assert_eq!(mgr.num_stored_blocks(), 2);
    }

    #[test]
    fn schedule_and_execute_prefetch() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        // Store some blocks.
        mgr.store_block(0, 100, &engines).unwrap();
        mgr.store_block(1, 200, &engines).unwrap();

        // Schedule prefetches: hash 100 -> GPU block 5, hash 999 -> GPU block 6 (miss).
        mgr.schedule_prefetch(&[(100, 5), (999, 6)]);
        assert_eq!(mgr.pending_prefetches(), 1); // only hash 100 is in CPU cache

        let mut mut_engines = make_engines(&cfg);
        let loaded = mgr.execute_prefetches(&mut mut_engines).unwrap();
        assert_eq!(loaded, 1);
        assert_eq!(mgr.metrics().prefetches(), 1);
        assert_eq!(mgr.pending_prefetches(), 0);
    }

    #[test]
    fn prefetch_respects_limit() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        // prefetch_count = 2
        let mut mgr = CpuOffloadManager::new(offload_config(8), 2, 4, 2, 8, DType::F32).unwrap();

        // Store 4 blocks.
        for i in 0..4u64 {
            mgr.store_block(i as usize, i + 1, &engines).unwrap();
        }

        // Schedule 4 prefetches -- only 2 should be queued (prefetch_count = 2).
        mgr.schedule_prefetch(&[(1, 0), (2, 1), (3, 2), (4, 3)]);
        assert_eq!(mgr.pending_prefetches(), 2);
    }

    #[test]
    fn store_block_to_different_gpu_blocks_different_hashes() {
        let cfg = test_cache_config();
        let mut engines = make_engines(&cfg);

        // Write different data to GPU blocks 0 and 1.
        let k0: Vec<f32> = (0..2 * 4 * 8).map(|i| i as f32).collect();
        let k1: Vec<f32> = (0..2 * 4 * 8).map(|i| (i + 100) as f32).collect();

        let k0t = Tensor::from_vec(k0.clone(), (2, 4, 8), &Device::Cpu).unwrap();
        let v0t = Tensor::from_vec(k0.clone(), (2, 4, 8), &Device::Cpu).unwrap();
        let k1t = Tensor::from_vec(k1.clone(), (2, 4, 8), &Device::Cpu).unwrap();
        let v1t = Tensor::from_vec(k1.clone(), (2, 4, 8), &Device::Cpu).unwrap();

        for engine in engines.iter_mut() {
            engine.write(&k0t, &v0t, &[0, 1, 2, 3]).unwrap();
            engine.write(&k1t, &v1t, &[4, 5, 6, 7]).unwrap();
        }

        let mut mgr = CpuOffloadManager::new(offload_config(4), 2, 4, 2, 8, DType::F32).unwrap();

        mgr.store_block(0, 10, &engines).unwrap();
        mgr.store_block(1, 20, &engines).unwrap();

        // Load them back into different GPU blocks of fresh engines.
        let mut fresh = make_engines(&cfg);
        mgr.load_block(10, 2, &mut fresh).unwrap(); // load hash 10 -> GPU block 2
        mgr.load_block(20, 3, &mut fresh).unwrap(); // load hash 20 -> GPU block 3

        // Verify block 2 has k0's data and block 3 has k1's data.
        for engine in &fresh {
            let (k2_out, _) = engine.read(&[2], 4).unwrap();
            let (k3_out, _) = engine.read(&[3], 4).unwrap();

            let k2_flat: Vec<f32> = k2_out.flatten_all().unwrap().to_vec1().unwrap();
            let k3_flat: Vec<f32> = k3_out.flatten_all().unwrap().to_vec1().unwrap();

            assert_eq!(k2_flat, k0);
            assert_eq!(k3_flat, k1);
        }
    }

    #[test]
    fn metrics_tracked_correctly() {
        let cfg = test_cache_config();
        let engines = make_engines(&cfg);
        let mut mut_engines = make_engines(&cfg);
        let mut mgr = CpuOffloadManager::new(offload_config(2), 2, 4, 2, 8, DType::F32).unwrap();

        // Store 3 blocks (will evict 1).
        mgr.store_block(0, 10, &engines).unwrap();
        mgr.store_block(1, 20, &engines).unwrap();
        mgr.store_block(2, 30, &engines).unwrap(); // evicts hash 10

        assert_eq!(mgr.metrics().stores(), 3);
        assert_eq!(mgr.metrics().evictions(), 1);

        // Load hit.
        mgr.load_block(20, 0, &mut mut_engines).unwrap();
        assert_eq!(mgr.metrics().loads(), 1);
        assert_eq!(mgr.metrics().cpu_hits(), 1);

        // Load miss.
        mgr.load_block(10, 0, &mut mut_engines).unwrap();
        assert_eq!(mgr.metrics().cpu_misses(), 1);

        let snap = mgr.metrics().snapshot();
        assert_eq!(snap.stores, 3);
        assert_eq!(snap.evictions, 1);
        assert_eq!(snap.loads, 1);
        assert_eq!(snap.cpu_hits, 1);
        assert_eq!(snap.cpu_misses, 1);
        let rate = snap.hit_rate.unwrap();
        assert!((rate - 0.5).abs() < 0.001);
    }
}
