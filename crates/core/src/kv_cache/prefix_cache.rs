use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::block_pool::BlockId;
use super::metrics::KVCacheMetrics;
use super::prefix_cache_stats::{
    PrefixCacheStats, PrefixCacheStatsSnapshot, SlidingWindowMetrics, SlidingWindowSnapshot,
};

/// Hash-based prefix cache for KV cache block reuse.
///
/// When multiple requests share a common token prefix, the KV cache blocks
/// computed for that prefix can be reused instead of recomputing. This cache
/// tracks which physical blocks hold computed KV data for specific token
/// sequences, enabling instant prefill for cached prefixes.
pub struct PrefixCache {
    block_size: usize,
    /// Block hash → cached block entry
    cache: HashMap<u64, CachedBlock>,
    /// Monotonic access counter for LRU eviction
    access_counter: u64,
    /// Optional metrics tracking (legacy KVCacheMetrics)
    metrics: Option<Arc<KVCacheMetrics>>,
    /// Detailed prefix cache statistics
    stats: Arc<PrefixCacheStats>,
    /// Sliding window metrics for recent hit rate
    sliding_window: SlidingWindowMetrics,
    /// Whether cache was reset (for sliding window signaling)
    was_reset: bool,
}

struct CachedBlock {
    block_id: BlockId,
    ref_count: usize,
    last_access: u64,
}

impl PrefixCache {
    /// Default sliding window size for recent metrics (number of requests).
    pub const DEFAULT_WINDOW_SIZE: u64 = 1000;

    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            cache: HashMap::new(),
            access_counter: 0,
            metrics: None,
            stats: Arc::new(PrefixCacheStats::new()),
            sliding_window: SlidingWindowMetrics::new(Self::DEFAULT_WINDOW_SIZE),
            was_reset: false,
        }
    }

    /// Create a PrefixCache with legacy KVCacheMetrics tracking enabled.
    pub fn with_metrics(block_size: usize, metrics: Arc<KVCacheMetrics>) -> Self {
        Self {
            block_size,
            cache: HashMap::new(),
            access_counter: 0,
            metrics: Some(metrics),
            stats: Arc::new(PrefixCacheStats::new()),
            sliding_window: SlidingWindowMetrics::new(Self::DEFAULT_WINDOW_SIZE),
            was_reset: false,
        }
    }

    /// Create a PrefixCache with custom sliding window size.
    pub fn with_window_size(block_size: usize, window_size: u64) -> Self {
        Self {
            block_size,
            cache: HashMap::new(),
            access_counter: 0,
            metrics: None,
            stats: Arc::new(PrefixCacheStats::new()),
            sliding_window: SlidingWindowMetrics::new(window_size),
            was_reset: false,
        }
    }

    /// Create a PrefixCache with all options.
    pub fn with_all_options(
        block_size: usize,
        metrics: Option<Arc<KVCacheMetrics>>,
        stats: Arc<PrefixCacheStats>,
        window_size: u64,
    ) -> Self {
        Self {
            block_size,
            cache: HashMap::new(),
            access_counter: 0,
            metrics,
            stats,
            sliding_window: SlidingWindowMetrics::new(window_size),
            was_reset: false,
        }
    }

    /// Get the underlying stats instance (for sharing across components).
    pub fn stats(&self) -> &Arc<PrefixCacheStats> {
        &self.stats
    }

    /// Match prefix blocks for a given prompt.
    ///
    /// Returns `(block_ids, num_cached_tokens)`: the physical blocks that
    /// already contain computed KV data, and how many tokens they cover.
    /// The caller should set `num_computed_tokens = num_cached_tokens` and
    /// pre-populate the block table with the returned block IDs.
    pub fn match_prefix(&mut self, prompt_tokens: &[u32]) -> (Vec<BlockId>, usize) {
        let hashes = compute_block_hashes(prompt_tokens, self.block_size);
        let mut matched_blocks = Vec::new();

        for hash in &hashes {
            if let Some(entry) = self.cache.get_mut(hash) {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                entry.ref_count += 1;
                matched_blocks.push(entry.block_id);
            } else {
                break; // prefix continuity broken
            }
        }

        // Record metrics: hits = matched blocks, misses = total needed - matched
        let total_full_blocks = hashes.len();
        let block_hits = matched_blocks.len();
        let block_misses = total_full_blocks.saturating_sub(block_hits);
        let num_cached = block_hits * self.block_size;
        let tokens_queried = prompt_tokens.len();

        // Record to legacy KVCacheMetrics
        if let Some(ref m) = self.metrics {
            m.record_cache_query(block_hits, block_misses);
        }

        // Record to detailed PrefixCacheStats
        self.stats
            .record_query(block_hits, block_misses, tokens_queried, num_cached);

        (matched_blocks, num_cached)
    }

    /// Record a request observation for sliding window metrics.
    ///
    /// This should be called once per request after match_prefix, providing
    /// the number of tokens queried and hit for that request. This enables
    /// the sliding window to track per-request hit rates.
    pub fn record_request(&mut self, tokens_queried: usize, tokens_hit: usize) {
        let reset = self.was_reset;
        self.was_reset = false;
        self.sliding_window
            .observe(1, tokens_queried as u64, tokens_hit as u64, reset);
    }

    /// Register blocks from a completed prefill into the cache.
    ///
    /// Only full blocks are registered (the partial last block is excluded).
    /// `prompt_tokens` is the full prompt, `block_ids` are the allocated blocks.
    ///
    /// The registering request has an implicit ownership reference (ref_count = 1).
    /// When another request matches, ref_count increases. When requests release,
    /// ref_count decreases. Blocks are evictable when ref_count = 0.
    pub fn register_blocks(&mut self, prompt_tokens: &[u32], block_ids: &[BlockId]) {
        let hashes = compute_block_hashes(prompt_tokens, self.block_size);

        for (i, hash) in hashes.iter().enumerate() {
            if i >= block_ids.len() {
                break;
            }
            self.cache.entry(*hash).or_insert_with(|| {
                self.access_counter += 1;
                CachedBlock {
                    block_id: block_ids[i],
                    ref_count: 1, // Owner has a reference
                    last_access: self.access_counter,
                }
            });
        }
    }

    /// Release a request's reference to cached blocks.
    ///
    /// Returns the block IDs that are NOT in the cache (should be freed to pool).
    /// Cached blocks have their ref_count decremented but stay allocated.
    pub fn release_blocks(&mut self, prompt_tokens: &[u32], block_ids: &[BlockId]) -> Vec<BlockId> {
        let hashes = compute_block_hashes(prompt_tokens, self.block_size);
        let cached_block_set: HashMap<BlockId, u64> = hashes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < block_ids.len())
            .map(|(i, &hash)| (block_ids[i], hash))
            .collect();

        let mut to_free = Vec::new();
        for &block_id in block_ids {
            if let Some(&hash) = cached_block_set.get(&block_id) {
                if let Some(entry) = self.cache.get_mut(&hash) {
                    if entry.block_id == block_id {
                        entry.ref_count = entry.ref_count.saturating_sub(1);
                        continue;
                    }
                }
            }
            to_free.push(block_id);
        }
        to_free
    }

    /// Evict up to `count` unreferenced blocks from the cache (LRU order).
    ///
    /// Returns the block IDs that were evicted (caller should free them to pool).
    pub fn evict(&mut self, count: usize) -> Vec<BlockId> {
        if count == 0 {
            return Vec::new();
        }

        // Collect evictable entries (ref_count == 0)
        let mut evictable: Vec<(u64, u64, BlockId)> = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.ref_count == 0)
            .map(|(&hash, entry)| (hash, entry.last_access, entry.block_id))
            .collect();

        // Sort by last_access ascending (LRU first)
        evictable.sort_by_key(|&(_, access, _)| access);

        let mut evicted = Vec::new();
        for (hash, _, block_id) in evictable.into_iter().take(count) {
            self.cache.remove(&hash);
            evicted.push(block_id);
        }

        // Record eviction in internal stats
        // Note: legacy KVCacheMetrics eviction is still recorded by the caller
        // (KVCacheManager) to avoid double-counting when manager coordinates.
        self.stats.record_eviction(evicted.len());

        evicted
    }

    /// Number of cached blocks currently stored.
    pub fn num_cached_blocks(&self) -> usize {
        self.cache.len()
    }

    /// Number of blocks with no active references (evictable).
    pub fn num_evictable_blocks(&self) -> usize {
        self.cache.values().filter(|e| e.ref_count == 0).count()
    }

    /// Get a snapshot of the detailed prefix cache statistics.
    pub fn get_stats(&self) -> PrefixCacheStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get a snapshot of the sliding window (recent) metrics.
    pub fn get_sliding_window_stats(&self) -> SlidingWindowSnapshot {
        self.sliding_window.snapshot()
    }

    /// Get the current hit rate from lifetime statistics.
    ///
    /// Returns `None` if no queries have been made.
    pub fn hit_rate(&self) -> Option<f64> {
        self.stats.block_hit_rate()
    }

    /// Get the token hit rate from lifetime statistics.
    ///
    /// Returns `None` if no tokens have been queried.
    pub fn token_hit_rate(&self) -> Option<f64> {
        self.stats.token_hit_rate()
    }

    /// Get the recent hit rate from the sliding window.
    ///
    /// This provides a more responsive view of recent cache performance.
    pub fn recent_hit_rate(&self) -> f64 {
        self.sliding_window.hit_rate()
    }

    /// Reset all statistics (both lifetime and sliding window).
    ///
    /// This does NOT clear the cache itself, only the counters.
    pub fn reset_stats(&mut self) {
        self.stats.reset();
        self.sliding_window.reset();
        self.was_reset = true;
    }

    /// Clear the cache and reset all statistics.
    ///
    /// Returns the block IDs that were evicted (all cached blocks).
    pub fn clear(&mut self) -> Vec<BlockId> {
        let evicted: Vec<BlockId> = self.cache.values().map(|e| e.block_id).collect();
        let num_evicted = evicted.len();

        self.cache.clear();
        self.access_counter = 0;

        // Record eviction in stats
        self.stats.record_eviction(num_evicted);
        self.reset_stats();

        evicted
    }

    /// Get the block size used by this cache.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Estimate memory used by cached blocks in bytes.
    ///
    /// Note: This is an estimate based on block count and size. The actual
    /// memory consumption depends on the KV cache tensor dimensions and dtype.
    pub fn estimated_cached_bytes(&self, bytes_per_block: usize) -> usize {
        self.cache.len() * bytes_per_block
    }
}

/// Compute chained block hashes for a token sequence.
///
/// Each block's hash incorporates the previous block's hash, ensuring
/// that only identical prefixes produce matching hashes.
fn compute_block_hashes(token_ids: &[u32], block_size: usize) -> Vec<u64> {
    let mut hashes = Vec::new();
    let mut prev_hash: u64 = 0;

    for chunk in token_ids.chunks(block_size) {
        if chunk.len() < block_size {
            break; // only full blocks are cacheable
        }
        let hash = hash_block(prev_hash, chunk);
        hashes.push(hash);
        prev_hash = hash;
    }
    hashes
}

/// Hash a single block: combines previous block hash with token IDs.
fn hash_block(prev_hash: u64, tokens: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prev_hash.hash(&mut hasher);
    for &t in tokens {
        t.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn empty_prompt_no_match() {
        let mut cache = PrefixCache::new(4);
        let (blocks, cached) = cache.match_prefix(&[]);
        assert!(blocks.is_empty());
        assert_eq!(cached, 0);
    }

    #[test]
    fn no_match_on_empty_cache() {
        let mut cache = PrefixCache::new(4);
        let (blocks, cached) = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(blocks.is_empty());
        assert_eq!(cached, 0);
    }

    #[test]
    fn register_and_match() {
        let mut cache = PrefixCache::new(4);
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks
        let block_ids = vec![10, 20];

        cache.register_blocks(&prompt, &block_ids);
        assert_eq!(cache.num_cached_blocks(), 2);

        let (matched, cached) = cache.match_prefix(&prompt);
        assert_eq!(matched, vec![10, 20]);
        assert_eq!(cached, 8);
    }

    #[test]
    fn partial_prefix_match() {
        let mut cache = PrefixCache::new(4);
        // Register [1,2,3,4,5,6,7,8]
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // Query with different second block
        let (matched, cached) = cache.match_prefix(&[1, 2, 3, 4, 9, 9, 9, 9]);
        assert_eq!(matched, vec![10]); // only first block matches
        assert_eq!(cached, 4);
    }

    #[test]
    fn partial_last_block_not_cached() {
        let mut cache = PrefixCache::new(4);
        // 6 tokens = 1 full block + 2 partial
        cache.register_blocks(&[1, 2, 3, 4, 5, 6], &[10, 20]);
        assert_eq!(cache.num_cached_blocks(), 1); // only 1 full block
    }

    #[test]
    fn release_decrements_refcount() {
        let mut cache = PrefixCache::new(4);
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
        cache.register_blocks(&prompt, &[10, 20]);
        // After register: ref_count = 1 (owner)

        // Match increments ref_count (ref_count = 2)
        cache.match_prefix(&prompt);
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release once (ref_count = 1) - still not evictable
        let to_free = cache.release_blocks(&prompt, &[10, 20]);
        assert!(to_free.is_empty());
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release again (ref_count = 0) - now evictable
        cache.release_blocks(&prompt, &[10, 20]);
        assert_eq!(cache.num_evictable_blocks(), 2);
    }

    #[test]
    fn release_returns_uncached_blocks() {
        let mut cache = PrefixCache::new(4);
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 2 full + partial
        cache.register_blocks(&prompt, &[10, 20, 30]);

        // Block 30 is the partial block, not cached
        let to_free = cache.release_blocks(&prompt, &[10, 20, 30]);
        assert_eq!(to_free, vec![30]);
    }

    #[test]
    fn evict_lru_order() {
        let mut cache = PrefixCache::new(4);
        // Register two different prefixes (ref_count = 1 each)
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.register_blocks(&[5, 6, 7, 8], &[20]);

        // Access second one to make it more recent (ref_count = 2)
        cache.match_prefix(&[5, 6, 7, 8]);

        // Release first prefix's owner reference (block 10: ref_count = 0)
        cache.release_blocks(&[1, 2, 3, 4], &[10]);

        // Release second prefix's references (match + owner)
        cache.release_blocks(&[5, 6, 7, 8], &[20]); // ref_count = 1
        cache.release_blocks(&[5, 6, 7, 8], &[20]); // ref_count = 0

        // Both are evictable, block 10 is LRU (registered first, not accessed)
        let evicted = cache.evict(1);
        assert_eq!(evicted, vec![10]);
        assert_eq!(cache.num_cached_blocks(), 1);
    }

    #[test]
    fn evict_skips_referenced_blocks() {
        let mut cache = PrefixCache::new(4);
        cache.register_blocks(&[1, 2, 3, 4], &[10]);

        // Match without releasing → ref_count > 0
        cache.match_prefix(&[1, 2, 3, 4]);

        let evicted = cache.evict(1);
        assert!(evicted.is_empty()); // can't evict referenced block
    }

    #[test]
    fn chained_hashes_prevent_false_match() {
        let mut cache = PrefixCache::new(4);
        // Register [A, B, C, D, E, F, G, H]
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // Different first block, same second block tokens
        // Should NOT match the second block because hash chains differ
        let (matched, _) = cache.match_prefix(&[9, 9, 9, 9, 5, 6, 7, 8]);
        assert!(matched.is_empty());
    }

    #[test]
    fn multiple_requests_same_prefix() {
        let mut cache = PrefixCache::new(4);
        let prefix = vec![1, 2, 3, 4, 5, 6, 7, 8];
        cache.register_blocks(&prefix, &[10, 20]);
        // ref_count = 1 (owner)

        // Two requests match the same prefix
        let (m1, _) = cache.match_prefix(&prefix); // ref_count = 2
        let (m2, _) = cache.match_prefix(&prefix); // ref_count = 3
        assert_eq!(m1, vec![10, 20]);
        assert_eq!(m2, vec![10, 20]);

        // Release one match → ref_count = 2, still not evictable
        cache.release_blocks(&prefix, &[10, 20]);
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release second match → ref_count = 1, still not evictable
        cache.release_blocks(&prefix, &[10, 20]);
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release owner → ref_count = 0, now evictable
        cache.release_blocks(&prefix, &[10, 20]);
        assert_eq!(cache.num_evictable_blocks(), 2);
    }

    #[test]
    fn compute_block_hashes_basic() {
        let hashes = compute_block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(hashes.len(), 2);
        // Same input should give same hashes
        let hashes2 = compute_block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(hashes, hashes2);
        // Different input gives different hashes
        let hashes3 = compute_block_hashes(&[1, 2, 3, 4, 9, 9, 9, 9], 4);
        assert_eq!(hashes[0], hashes3[0]); // first block same
        assert_ne!(hashes[1], hashes3[1]); // second block different
    }

    // ─── Metrics integration tests ────────────────────────────────────────────

    #[test]
    fn metrics_cache_hit() {
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut cache = PrefixCache::with_metrics(4, Arc::clone(&metrics));

        // Register prefix
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 full blocks
        cache.register_blocks(&prompt, &[10, 20]);

        // Match prefix → 2 hits, 0 misses
        cache.match_prefix(&prompt);

        assert_eq!(metrics.cache_queries(), 1);
        assert_eq!(metrics.cache_hits(), 2);
        assert_eq!(metrics.cache_misses(), 0);
        assert_eq!(metrics.hit_rate(), Some(1.0));
    }

    #[test]
    fn metrics_cache_miss() {
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut cache = PrefixCache::with_metrics(4, Arc::clone(&metrics));

        // Query empty cache → 0 hits, 2 misses
        cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(metrics.cache_queries(), 1);
        assert_eq!(metrics.cache_hits(), 0);
        assert_eq!(metrics.cache_misses(), 2);
        assert_eq!(metrics.hit_rate(), Some(0.0));
    }

    #[test]
    fn metrics_partial_hit() {
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut cache = PrefixCache::with_metrics(4, Arc::clone(&metrics));

        // Register only first block
        cache.register_blocks(&[1, 2, 3, 4], &[10]);

        // Query with 2 blocks → 1 hit, 1 miss
        cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(metrics.cache_hits(), 1);
        assert_eq!(metrics.cache_misses(), 1);
        let rate = metrics.hit_rate().unwrap();
        assert!((rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn metrics_eviction_tracked_by_caller() {
        // Note: eviction metrics are recorded by the caller (KVCacheManager),
        // not by PrefixCache itself, to avoid double-counting.
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut cache = PrefixCache::with_metrics(4, Arc::clone(&metrics));

        // Register two prefixes (ref_count = 1 each)
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.register_blocks(&[5, 6, 7, 8], &[20]);

        // Release owner references so blocks are evictable
        cache.release_blocks(&[1, 2, 3, 4], &[10]);
        cache.release_blocks(&[5, 6, 7, 8], &[20]);

        assert_eq!(metrics.blocks_evicted(), 0);

        // Evict 1 block - caller records metrics
        let evicted = cache.evict(1);
        metrics.record_eviction(evicted.len()); // simulating what manager does
        assert_eq!(metrics.blocks_evicted(), 1);

        // Evict remaining - caller records metrics
        let evicted = cache.evict(1);
        metrics.record_eviction(evicted.len());
        assert_eq!(metrics.blocks_evicted(), 2);
    }

    #[test]
    fn metrics_accumulate_across_queries() {
        let metrics = Arc::new(KVCacheMetrics::new());
        let mut cache = PrefixCache::with_metrics(4, Arc::clone(&metrics));

        // Register prefix
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // First query: 2 hits
        cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        cache.release_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // Second query with different prompt: 0 hits, 2 misses
        cache.match_prefix(&[9, 9, 9, 9, 9, 9, 9, 9]);

        assert_eq!(metrics.cache_queries(), 2);
        assert_eq!(metrics.cache_hits(), 2); // only first query hit
        assert_eq!(metrics.cache_misses(), 2); // second query missed
        let rate = metrics.hit_rate().unwrap();
        assert!((rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn no_metrics_works() {
        // Without metrics, should still work
        let mut cache = PrefixCache::new(4);
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        let (blocks, _) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(blocks, vec![10]);
        let evicted = cache.evict(1);
        assert_eq!(evicted.len(), 0); // still referenced
    }

    // ─── PrefixCacheStats integration tests ────────────────────────────────────

    #[test]
    fn stats_track_queries() {
        let mut cache = PrefixCache::new(4);

        // Register prefix
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // Query: 2 block hits, 8 tokens queried, 8 tokens hit
        let (_, num_cached) = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(num_cached, 8);

        let stats = cache.get_stats();
        assert_eq!(stats.num_hits, 2);
        assert_eq!(stats.num_misses, 0);
        assert_eq!(stats.tokens_queried, 8);
        assert_eq!(stats.tokens_hit, 8);
        assert_eq!(stats.block_hit_rate, Some(1.0));
        assert_eq!(stats.token_hit_rate, Some(1.0));
    }

    #[test]
    fn stats_track_partial_hit() {
        let mut cache = PrefixCache::new(4);

        // Register only first block
        cache.register_blocks(&[1, 2, 3, 4], &[10]);

        // Query 2 blocks (8 tokens): 1 hit, 1 miss, 4 tokens cached
        cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let stats = cache.get_stats();
        assert_eq!(stats.num_hits, 1);
        assert_eq!(stats.num_misses, 1);
        assert_eq!(stats.tokens_queried, 8);
        assert_eq!(stats.tokens_hit, 4);

        let block_rate = stats.block_hit_rate.unwrap();
        assert!((block_rate - 0.5).abs() < 0.001);

        let token_rate = stats.token_hit_rate.unwrap();
        assert!((token_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn stats_track_evictions() {
        let mut cache = PrefixCache::new(4);

        // Register and release
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.release_blocks(&[1, 2, 3, 4], &[10]);

        assert_eq!(cache.get_stats().num_evictions, 0);

        // Evict
        let evicted = cache.evict(1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(cache.get_stats().num_evictions, 1);
    }

    #[test]
    fn stats_hit_rate_methods() {
        let mut cache = PrefixCache::new(4);

        // No data yet
        assert_eq!(cache.hit_rate(), None);
        assert_eq!(cache.token_hit_rate(), None);

        // Register and query
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
        cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);

        assert_eq!(cache.hit_rate(), Some(1.0));
        assert_eq!(cache.token_hit_rate(), Some(1.0));
    }

    #[test]
    fn stats_reset() {
        let mut cache = PrefixCache::new(4);

        // Generate some stats
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.match_prefix(&[1, 2, 3, 4]);
        cache.release_blocks(&[1, 2, 3, 4], &[10]);
        cache.evict(1);

        let stats = cache.get_stats();
        assert!(stats.num_hits > 0 || stats.num_evictions > 0);

        // Reset stats
        cache.reset_stats();

        let stats = cache.get_stats();
        assert_eq!(stats.num_hits, 0);
        assert_eq!(stats.num_misses, 0);
        assert_eq!(stats.num_evictions, 0);
        assert_eq!(stats.num_queries, 0);
    }

    #[test]
    fn stats_clear_cache() {
        let mut cache = PrefixCache::new(4);

        // Register blocks
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
        assert_eq!(cache.num_cached_blocks(), 2);

        // Clear returns all block IDs
        let evicted = cache.clear();
        assert_eq!(evicted.len(), 2);
        assert_eq!(cache.num_cached_blocks(), 0);
    }

    // ─── Sliding window tests ──────────────────────────────────────────────────

    #[test]
    fn sliding_window_tracks_requests() {
        let mut cache = PrefixCache::with_window_size(4, 10);

        // Register and match
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);
        let (_, num_cached) = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        cache.record_request(8, num_cached);

        let sw = cache.get_sliding_window_stats();
        assert_eq!(sw.num_requests, 1);
        assert_eq!(sw.tokens_queried, 8);
        assert_eq!(sw.tokens_hit, 8);
        assert!((sw.hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn sliding_window_recent_hit_rate() {
        let mut cache = PrefixCache::with_window_size(4, 10);

        // Register prefix
        cache.register_blocks(&[1, 2, 3, 4], &[10]);

        // Request 1: full hit (4 tokens)
        let (_, c1) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.record_request(4, c1);

        // Request 2: partial hit (8 tokens, 4 hit)
        let (_, c2) = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8]);
        cache.record_request(8, c2);

        // Recent hit rate: (4 + 4) / (4 + 8) = 8 / 12 = 0.666...
        let rate = cache.recent_hit_rate();
        assert!((rate - 0.6666).abs() < 0.01);
    }

    #[test]
    fn sliding_window_reset_on_cache_reset() {
        let mut cache = PrefixCache::with_window_size(4, 10);

        // Generate some data
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        let (_, c) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.record_request(4, c);

        let sw = cache.get_sliding_window_stats();
        assert_eq!(sw.num_requests, 1);

        // Reset stats (simulates cache reset)
        cache.reset_stats();

        // Record new request - should signal reset to sliding window
        let (_, c) = cache.match_prefix(&[1, 2, 3, 4]);
        cache.record_request(4, c);

        let sw = cache.get_sliding_window_stats();
        // Window should have been reset before new observation
        assert_eq!(sw.num_requests, 1);
    }

    #[test]
    fn block_size_accessor() {
        let cache = PrefixCache::new(16);
        assert_eq!(cache.block_size(), 16);
    }

    #[test]
    fn estimated_cached_bytes() {
        let mut cache = PrefixCache::new(4);
        cache.register_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], &[10, 20]);

        // 2 blocks, 1024 bytes each
        let bytes = cache.estimated_cached_bytes(1024);
        assert_eq!(bytes, 2048);
    }

    #[test]
    fn with_all_options() {
        // PrefixCacheStats already imported at top of file via super::prefix_cache_stats
        let shared_stats = Arc::new(PrefixCacheStats::new());
        let metrics = Arc::new(KVCacheMetrics::new());

        let mut cache = PrefixCache::with_all_options(
            4,
            Some(Arc::clone(&metrics)),
            Arc::clone(&shared_stats),
            500,
        );

        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.match_prefix(&[1, 2, 3, 4]);

        // Both metrics systems should track the query
        assert_eq!(metrics.cache_hits(), 1);
        assert_eq!(shared_stats.num_hits(), 1);

        // Shared stats can be accessed externally
        assert_eq!(cache.stats().num_hits(), 1);
    }
}
