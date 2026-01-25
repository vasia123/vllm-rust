use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::block_pool::BlockId;

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
}

struct CachedBlock {
    block_id: BlockId,
    ref_count: usize,
    last_access: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            cache: HashMap::new(),
            access_counter: 0,
        }
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

        let num_cached = matched_blocks.len() * self.block_size;
        (matched_blocks, num_cached)
    }

    /// Register blocks from a completed prefill into the cache.
    ///
    /// Only full blocks are registered (the partial last block is excluded).
    /// `prompt_tokens` is the full prompt, `block_ids` are the allocated blocks.
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
                    ref_count: 0,
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

        // Match increments ref_count
        cache.match_prefix(&prompt);
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release decrements ref_count
        let to_free = cache.release_blocks(&prompt, &[10, 20]);
        assert!(to_free.is_empty()); // cached blocks not freed
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
        // Register two different prefixes
        cache.register_blocks(&[1, 2, 3, 4], &[10]);
        cache.register_blocks(&[5, 6, 7, 8], &[20]);

        // Access second one to make it more recent
        cache.match_prefix(&[5, 6, 7, 8]);
        // Release so it's evictable
        cache.release_blocks(&[5, 6, 7, 8], &[20]);

        // Evict 1 → should be block 10 (LRU)
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

        // Two requests match the same prefix
        let (m1, _) = cache.match_prefix(&prefix);
        let (m2, _) = cache.match_prefix(&prefix);
        assert_eq!(m1, vec![10, 20]);
        assert_eq!(m2, vec![10, 20]);

        // Release one → still not evictable (ref_count=1)
        cache.release_blocks(&prefix, &[10, 20]);
        assert_eq!(cache.num_evictable_blocks(), 0);

        // Release second → now evictable
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
}
