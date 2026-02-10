//! Encoder cache manager for multimodal models.
//!
//! Caches encoder outputs (vision embeddings, audio features) to avoid
//! recomputation when the same multimodal inputs appear across requests
//! or processing stages.
//!
//! # Design
//!
//! The cache operates at the granularity of individual multimodal items
//! within requests. Each item is identified by a content hash (`mm_hash`).
//!
//! - **Reference counting**: Multiple requests can reference the same cached item.
//! - **Lazy eviction**: When all references drop, items move to a freeable
//!   queue but remain cached. They're only evicted when space is needed.
//! - **FIFO eviction**: Oldest unreferenced items are evicted first.
//!
//! # Variants
//!
//! - [`EncoderCacheManager`]: Full caching with cross-request sharing and eviction.
//! - [`EncoderDecoderCacheManager`]: Simplified for encoder-decoder models (T5, BART)
//!   that only tracks scheduling budget without cross-request caching.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::request::RequestId;

/// Manages caching of encoder outputs for multimodal models.
///
/// Tracks cache occupancy in units of encoder embedding tokens (not input tokens).
/// Uses reference counting to share cached items across requests and FIFO eviction
/// when cache pressure requires reclaiming space.
pub struct EncoderCacheManager {
    /// Total cache capacity in encoder embedding tokens.
    cache_size: usize,
    /// Currently available capacity (unallocated slots).
    num_free_slots: usize,
    /// Free + reclaimable capacity (free_slots + unreferenced freeable items).
    num_freeable_slots: usize,
    /// mm_hash → set of request IDs referencing this cached item.
    cached: HashMap<String, HashSet<RequestId>>,
    /// Ordered queue of unreferenced items: (mm_hash, num_embeds).
    /// Oldest items at front for FIFO eviction.
    freeable: VecDeque<(String, usize)>,
    /// Hashes evicted since last call to `get_freed_mm_hashes()`.
    freed: Vec<String>,
}

impl EncoderCacheManager {
    /// Create a new cache with the given capacity in encoder embedding tokens.
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            num_free_slots: cache_size,
            num_freeable_slots: cache_size,
            cached: HashMap::new(),
            freeable: VecDeque::new(),
            freed: Vec::new(),
        }
    }

    /// Reset the cache to its initial state.
    ///
    /// Clears all cached entries and restores full capacity.
    /// Called when model weights are updated to invalidate stale embeddings.
    pub fn reset(&mut self) {
        self.cached.clear();
        self.freeable.clear();
        self.freed.clear();
        self.num_free_slots = self.cache_size;
        self.num_freeable_slots = self.cache_size;
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn num_free_slots(&self) -> usize {
        self.num_free_slots
    }

    pub fn num_freeable_slots(&self) -> usize {
        self.num_freeable_slots
    }

    /// Check if encoder output for a multimodal input is already cached.
    ///
    /// If cached, adds the request as a reference. If the item was unreferenced
    /// (in the freeable queue), it's restored to active status.
    ///
    /// Returns true if the item was found in cache.
    pub fn check_and_update_cache(&mut self, request_id: RequestId, mm_hash: &str) -> bool {
        let refs = match self.cached.get_mut(mm_hash) {
            Some(refs) => refs,
            None => return false,
        };

        // If no active references, restore from freeable queue
        if refs.is_empty() {
            if let Some(pos) = self.freeable.iter().position(|(h, _)| h == mm_hash) {
                let (_, num_embeds) = self.freeable.remove(pos).unwrap();
                self.num_freeable_slots -= num_embeds;
            }
        }

        refs.insert(request_id);
        true
    }

    /// Check if there's sufficient cache space for a multimodal input.
    ///
    /// If free space is insufficient but reclaimable space is available,
    /// evicts oldest unreferenced items until enough space is freed.
    ///
    /// Returns false if compute budget is exceeded or not enough space
    /// even after eviction.
    pub fn can_allocate(
        &mut self,
        num_embeds: usize,
        encoder_compute_budget: usize,
        num_embeds_to_schedule: usize,
    ) -> bool {
        if num_embeds > encoder_compute_budget {
            return false;
        }

        let total_needed = num_embeds + num_embeds_to_schedule;

        if total_needed <= self.num_free_slots {
            return true;
        }

        if total_needed > self.num_freeable_slots {
            return false;
        }

        // Evict oldest unreferenced items until we have enough space
        while total_needed > self.num_free_slots {
            let (mm_hash, num_free_embeds) = self
                .freeable
                .pop_front()
                .expect("freeable_slots >= total_needed but freeable queue empty");
            self.cached.remove(&mm_hash);
            self.freed.push(mm_hash);
            self.num_free_slots += num_free_embeds;
        }

        true
    }

    /// Allocate cache space for a multimodal input's encoder output.
    ///
    /// Must be called after `can_allocate()` returned true for the same item.
    pub fn allocate(&mut self, request_id: RequestId, mm_hash: &str, num_embeds: usize) {
        debug_assert!(
            self.num_free_slots >= num_embeds,
            "allocate() called without sufficient free slots: {} < {}",
            self.num_free_slots,
            num_embeds
        );
        debug_assert!(
            self.num_freeable_slots >= num_embeds,
            "allocate() called without sufficient freeable slots: {} < {}",
            self.num_freeable_slots,
            num_embeds
        );

        self.cached
            .entry(mm_hash.to_string())
            .or_default()
            .insert(request_id);
        self.num_free_slots -= num_embeds;
        self.num_freeable_slots -= num_embeds;
    }

    /// Get indices of multimodal features that are currently cached.
    ///
    /// `identifiers` is the list of mm_hash values for each feature in the request.
    /// Returns the set of indices whose identifier is present in the cache.
    pub fn get_cached_input_ids(&self, identifiers: &[&str]) -> HashSet<usize> {
        identifiers
            .iter()
            .enumerate()
            .filter(|(_, id)| self.cached.contains_key(**id))
            .map(|(i, _)| i)
            .collect()
    }

    /// Free a single encoder input reference for a request.
    ///
    /// When the reference count drops to zero, the item moves to the
    /// freeable queue (remains cached until eviction is triggered).
    pub fn free_encoder_input(
        &mut self,
        request_id: RequestId,
        mm_hash: &str,
        num_embeds: usize,
    ) {
        let refs = match self.cached.get_mut(mm_hash) {
            Some(refs) if !refs.is_empty() => refs,
            _ => return,
        };

        refs.remove(&request_id);

        if refs.is_empty() {
            self.freeable.push_back((mm_hash.to_string(), num_embeds));
            self.num_freeable_slots += num_embeds;
        }
    }

    /// Free all encoder input references held by a request.
    ///
    /// `features` is a slice of (mm_hash, num_embeds) for each multimodal
    /// feature in the request. Only features currently in the cache are freed.
    pub fn free(&mut self, request_id: RequestId, features: &[(&str, usize)]) {
        let identifiers: Vec<&str> = features.iter().map(|(h, _)| *h).collect();
        let cached_ids = self.get_cached_input_ids(&identifiers);
        for input_id in cached_ids {
            let (mm_hash, num_embeds) = features[input_id];
            self.free_encoder_input(request_id, mm_hash, num_embeds);
        }
    }

    /// Get and clear the list of recently evicted cache entries.
    ///
    /// Used to notify workers which encoder outputs can be removed
    /// from GPU memory.
    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.freed)
    }

    /// Whether a specific mm_hash is currently in the cache.
    #[cfg(test)]
    fn is_cached(&self, mm_hash: &str) -> bool {
        self.cached.contains_key(mm_hash)
    }

    /// Number of active references for a cached item.
    #[cfg(test)]
    fn ref_count(&self, mm_hash: &str) -> usize {
        self.cached.get(mm_hash).map_or(0, |refs| refs.len())
    }

    /// Whether an item is in the freeable queue.
    #[cfg(test)]
    fn is_freeable(&self, mm_hash: &str) -> bool {
        self.freeable.iter().any(|(h, _)| h == mm_hash)
    }
}

/// Simplified encoder cache manager for encoder-decoder models (T5, BART).
///
/// Does not support cross-request caching. Only tracks scheduling budget
/// to prevent over-allocation of encoder compute. Uses a two-phase free
/// to ensure the model runner finishes using encoder outputs before they
/// are released.
pub struct EncoderDecoderCacheManager {
    cache_size: usize,
    num_free_slots: usize,
    /// Hashes allocated this step (moved to `to_free` on `get_freed_mm_hashes`).
    allocated: Vec<String>,
    /// Hashes from the previous step, ready to be freed.
    to_free: Vec<String>,
}

impl EncoderDecoderCacheManager {
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            num_free_slots: cache_size,
            allocated: Vec::new(),
            to_free: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.num_free_slots = self.cache_size;
        self.allocated.clear();
        self.to_free.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn num_free_slots(&self) -> usize {
        self.num_free_slots
    }

    /// Always returns false — encoder-decoder models don't cache across requests.
    pub fn check_and_update_cache(
        &mut self,
        _request_id: RequestId,
        _mm_hash: &str,
    ) -> bool {
        false
    }

    /// Check if there's sufficient space for encoding.
    pub fn can_allocate(
        &self,
        num_embeds: usize,
        encoder_compute_budget: usize,
        num_embeds_to_schedule: usize,
    ) -> bool {
        if num_embeds > encoder_compute_budget {
            return false;
        }
        num_embeds + num_embeds_to_schedule <= self.num_free_slots
    }

    /// Allocate space for an encoder input.
    pub fn allocate(&mut self, mm_hash: &str, num_embeds: usize) {
        self.num_free_slots -= num_embeds;
        self.allocated.push(mm_hash.to_string());
    }

    /// Free all encoder inputs for a request.
    ///
    /// `features` is a slice of (mm_hash, num_embeds) for each feature.
    pub fn free(&mut self, features: &[(&str, usize)]) {
        for &(_, num_embeds) in features {
            self.num_free_slots += num_embeds;
        }
    }

    /// Free a single encoder input.
    pub fn free_encoder_input(&mut self, num_embeds: usize) {
        self.num_free_slots += num_embeds;
    }

    /// Get and clear recently freed hashes.
    ///
    /// Uses a two-phase approach: items freed this step move to `to_free`,
    /// items from the previous step are returned. This ensures the model
    /// runner has finished using the encoder outputs before they're freed.
    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        let to_free = std::mem::take(&mut self.to_free);
        self.to_free = std::mem::take(&mut self.allocated);
        to_free
    }

    #[cfg(test)]
    pub fn allocated(&self) -> &[String] {
        &self.allocated
    }

    #[cfg(test)]
    pub fn to_free(&self) -> &[String] {
        &self.to_free
    }
}

/// Compute encoder cache and compute budgets for multimodal models.
///
/// Returns `(compute_budget, cache_size)` in units of encoder embedding tokens.
///
/// Both budgets are at least as large as the largest single multimodal item
/// to ensure any item can be encoded in one step.
pub fn compute_mm_encoder_budget(
    max_num_encoder_input_tokens: usize,
    encoder_cache_size: usize,
    mm_max_tokens_per_item: &HashMap<String, usize>,
) -> (usize, usize) {
    if mm_max_tokens_per_item.is_empty() {
        return (0, 0);
    }

    let max_tokens_per_mm_item = *mm_max_tokens_per_item.values().max().unwrap();

    let compute_budget = max_num_encoder_input_tokens.max(max_tokens_per_mm_item);
    let cache_size = encoder_cache_size.max(max_tokens_per_mm_item);

    (compute_budget, cache_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== EncoderCacheManager Tests ====================

    #[test]
    fn basic_allocate_and_reuse() {
        let mut cache = EncoderCacheManager::new(10);
        let req_id = 1;

        // Not cached initially
        assert!(!cache.check_and_update_cache(req_id, "imgA"));
        assert!(cache.can_allocate(4, usize::MAX, 0));

        cache.allocate(req_id, "imgA", 4);
        assert_eq!(cache.num_free_slots(), 6);

        // Now it's cached
        assert!(cache.check_and_update_cache(req_id, "imgA"));
        assert_eq!(cache.ref_count("imgA"), 1);

        // Free twice to bring refcount to 0 (check_and_update added it once,
        // allocate added it once, so ref set = {1} — one discard is enough)
        cache.free_encoder_input(req_id, "imgA", 4);

        assert_eq!(cache.ref_count("imgA"), 0);
        assert!(cache.is_freeable("imgA"));
        assert_eq!(cache.num_freeable_slots(), 10);
        assert_eq!(cache.num_free_slots(), 6);
    }

    #[test]
    fn freeing_decreases_refcount_and_moves_to_freeable() {
        let mut mgr = EncoderCacheManager::new(10);
        let req_id = 2;

        assert!(mgr.can_allocate(5, usize::MAX, 0));
        mgr.allocate(req_id, "img3", 5);

        assert_eq!(mgr.ref_count("img3"), 1);

        mgr.free_encoder_input(req_id, "img3", 5);

        assert_eq!(mgr.ref_count("img3"), 0);
        assert!(mgr.is_freeable("img3"));
        assert_eq!(mgr.num_freeable_slots(), 10);
    }

    #[test]
    fn free_request_frees_all_inputs() {
        let mut mgr = EncoderCacheManager::new(10);
        let req_id = 3;

        assert!(mgr.can_allocate(2, usize::MAX, 0));
        mgr.allocate(req_id, "a", 2);

        assert!(mgr.can_allocate(3, usize::MAX, 0));
        mgr.allocate(req_id, "b", 3);

        assert_eq!(mgr.ref_count("a"), 1);
        assert_eq!(mgr.ref_count("b"), 1);

        mgr.free(req_id, &[("a", 2), ("b", 3)]);

        assert_eq!(mgr.ref_count("a"), 0);
        assert_eq!(mgr.ref_count("b"), 0);
        assert!(mgr.is_freeable("a"));
        assert!(mgr.is_freeable("b"));
        assert_eq!(mgr.num_freeable_slots(), 10);
    }

    #[test]
    fn eviction_when_cache_is_full() {
        let mut mgr = EncoderCacheManager::new(10);

        // Allocate and free req1's item (moves to freeable)
        mgr.allocate(1, "x", 6);
        mgr.free_encoder_input(1, "x", 6);

        // Allocate req2's item — should evict "x"
        assert!(mgr.can_allocate(5, usize::MAX, 0));
        mgr.allocate(2, "y", 5);

        assert!(!mgr.is_cached("x"));
        let freed = mgr.get_freed_mm_hashes();
        assert!(freed.contains(&"x".to_string()));
    }

    #[test]
    fn get_cached_input_ids() {
        let mut mgr = EncoderCacheManager::new(10);
        let req_id = 10;

        mgr.allocate(req_id, "m", 2);
        mgr.allocate(req_id, "o", 3);

        let cached_ids = mgr.get_cached_input_ids(&["m", "n", "o"]);
        assert_eq!(cached_ids, HashSet::from([0, 2]));
    }

    #[test]
    fn restores_from_freeable_on_cache_hit() {
        let mut mgr = EncoderCacheManager::new(10);
        let req_id = 20;

        mgr.allocate(req_id, "imgZ", 4);
        mgr.free_encoder_input(req_id, "imgZ", 4);

        assert!(mgr.is_freeable("imgZ"));

        // Should restore from freeable
        assert!(mgr.check_and_update_cache(req_id, "imgZ"));
        assert_eq!(mgr.ref_count("imgZ"), 1);
        assert!(!mgr.is_freeable("imgZ"));
        assert_eq!(mgr.num_freeable_slots(), 6);
    }

    #[test]
    fn get_freed_mm_hashes_clears_list() {
        let mut mgr = EncoderCacheManager::new(10);

        mgr.allocate(1, "a", 5);
        mgr.free_encoder_input(1, "a", 5);

        // Trigger eviction of "a"
        assert!(mgr.can_allocate(6, usize::MAX, 0));
        mgr.allocate(2, "b", 6);

        let freed = mgr.get_freed_mm_hashes();
        assert!(freed.contains(&"a".to_string()));

        // Second call should be empty
        assert!(mgr.get_freed_mm_hashes().is_empty());
    }

    #[test]
    fn multi_images_respect_space_limit() {
        let mut mgr = EncoderCacheManager::new(10);
        let compute_budget = 100;

        // First image fits
        assert!(mgr.can_allocate(5, compute_budget, 0));
        let scheduled = 5;

        // Second image (6 embeds) + already scheduled (5) = 11 > 10 cache
        assert!(!mgr.can_allocate(6, compute_budget - 5, scheduled));
    }

    #[test]
    fn multi_images_respect_compute_limit() {
        let mut mgr = EncoderCacheManager::new(100);

        // First image fits within compute budget of 10
        assert!(mgr.can_allocate(5, 10, 0));
        let scheduled = 5;
        let remaining_budget = 10 - 5;

        // Second image (6 embeds) exceeds remaining compute budget (5)
        assert!(!mgr.can_allocate(6, remaining_budget, scheduled));
    }

    #[test]
    fn reset_clears_all_state() {
        let mut mgr = EncoderCacheManager::new(20);

        mgr.allocate(1, "img1", 5);
        mgr.allocate(1, "img2", 3);
        mgr.allocate(2, "img3", 4);
        mgr.free_encoder_input(1, "img1", 5);
        mgr.free_encoder_input(1, "img2", 3);
        mgr.free_encoder_input(2, "img3", 4);

        // Trigger some eviction
        assert!(mgr.can_allocate(10, usize::MAX, 0));
        mgr.allocate(3, "img4", 10);

        assert!(mgr.num_free_slots() < 20);

        mgr.reset();

        assert!(mgr.cached.is_empty());
        assert!(mgr.freeable.is_empty());
        assert!(mgr.freed.is_empty());
        assert_eq!(mgr.num_free_slots(), 20);
        assert_eq!(mgr.num_freeable_slots(), 20);
    }

    #[test]
    fn reset_allows_fresh_allocations() {
        let mut mgr = EncoderCacheManager::new(10);

        mgr.allocate(1, "img1", 10);
        assert_eq!(mgr.num_free_slots(), 0);

        mgr.reset();

        assert!(mgr.can_allocate(8, usize::MAX, 0));
        mgr.allocate(2, "img2", 8);

        assert_eq!(mgr.num_free_slots(), 2);
        assert!(mgr.is_cached("img2"));
        assert!(!mgr.is_cached("img1"));
    }

    #[test]
    fn cross_request_sharing() {
        let mut mgr = EncoderCacheManager::new(10);

        // Two requests reference the same image
        mgr.allocate(1, "shared_img", 5);
        assert!(mgr.check_and_update_cache(2, "shared_img"));

        assert_eq!(mgr.ref_count("shared_img"), 2);
        assert_eq!(mgr.num_free_slots(), 5); // Only allocated once

        // Free one reference
        mgr.free_encoder_input(1, "shared_img", 5);
        assert_eq!(mgr.ref_count("shared_img"), 1);
        assert!(!mgr.is_freeable("shared_img")); // Still referenced

        // Free second reference
        mgr.free_encoder_input(2, "shared_img", 5);
        assert_eq!(mgr.ref_count("shared_img"), 0);
        assert!(mgr.is_freeable("shared_img"));
    }

    #[test]
    fn fifo_eviction_order() {
        let mut mgr = EncoderCacheManager::new(10);

        // Allocate and free three items in order
        mgr.allocate(1, "first", 3);
        mgr.allocate(2, "second", 3);
        mgr.allocate(3, "third", 4);

        mgr.free_encoder_input(1, "first", 3);
        mgr.free_encoder_input(2, "second", 3);
        mgr.free_encoder_input(3, "third", 4);

        // All are freeable now
        assert_eq!(mgr.num_free_slots(), 0);
        assert_eq!(mgr.num_freeable_slots(), 10);

        // Allocate 7 — should evict "first" (3) + "second" (3) = 6, then "third" (4) to get 10
        // Actually: need 7 free. Free=0, evict first(3)→free=3, evict second(3)→free=6,
        // evict third(4)→free=10. All three evicted.
        // Wait, 7 > 6 so we need to evict third too.
        assert!(mgr.can_allocate(7, usize::MAX, 0));
        let freed = mgr.get_freed_mm_hashes();
        assert_eq!(freed, vec!["first", "second", "third"]);
    }

    #[test]
    fn eviction_preserves_fifo_order_partial() {
        let mut mgr = EncoderCacheManager::new(12);

        mgr.allocate(1, "a", 4);
        mgr.allocate(2, "b", 4);
        mgr.allocate(3, "c", 4);

        mgr.free_encoder_input(1, "a", 4);
        mgr.free_encoder_input(2, "b", 4);
        mgr.free_encoder_input(3, "c", 4);

        // Need 5 more. Free=0, evict "a"(4)→free=4, evict "b"(4)→free=8. 5<=8, stop.
        assert!(mgr.can_allocate(5, usize::MAX, 0));
        let freed = mgr.get_freed_mm_hashes();
        assert_eq!(freed, vec!["a", "b"]);

        // "c" should still be cached (not evicted)
        assert!(mgr.is_cached("c"));
        assert!(mgr.is_freeable("c"));
    }

    #[test]
    fn zero_capacity_cache() {
        let mut mgr = EncoderCacheManager::new(0);
        assert!(!mgr.can_allocate(1, usize::MAX, 0));
        assert_eq!(mgr.num_free_slots(), 0);
    }

    // ==================== EncoderDecoderCacheManager Tests ====================

    #[test]
    fn enc_dec_basic_allocate_and_free() {
        let mut mgr = EncoderDecoderCacheManager::new(20);

        assert!(mgr.can_allocate(5, usize::MAX, 0));
        mgr.allocate("img1", 5);
        assert_eq!(mgr.num_free_slots(), 15);

        mgr.free(&[("img1", 5)]);
        assert_eq!(mgr.num_free_slots(), 20);
    }

    #[test]
    fn enc_dec_never_caches() {
        let mut mgr = EncoderDecoderCacheManager::new(20);
        assert!(!mgr.check_and_update_cache(1, "anything"));
    }

    #[test]
    fn enc_dec_two_phase_free() {
        let mut mgr = EncoderDecoderCacheManager::new(20);

        mgr.allocate("img1", 5);
        mgr.allocate("img2", 3);

        // First call: to_free is empty, allocated moves to to_free
        let freed = mgr.get_freed_mm_hashes();
        assert!(freed.is_empty());
        assert_eq!(mgr.to_free(), &["img1", "img2"]);
        assert!(mgr.allocated().is_empty());

        // Allocate more
        mgr.allocate("img3", 4);

        // Second call: to_free (img1, img2) returned, allocated (img3) moves to to_free
        let freed = mgr.get_freed_mm_hashes();
        assert_eq!(freed, vec!["img1", "img2"]);
        assert_eq!(mgr.to_free(), &["img3"]);
    }

    #[test]
    fn enc_dec_reset() {
        let mut mgr = EncoderDecoderCacheManager::new(20);

        mgr.allocate("img1", 5);
        mgr.allocate("img2", 3);
        mgr.free(&[("img1", 5)]);
        mgr.get_freed_mm_hashes();

        assert!(mgr.num_free_slots() < 20);

        mgr.reset();

        assert!(mgr.allocated().is_empty());
        assert!(mgr.to_free().is_empty());
        assert_eq!(mgr.num_free_slots(), 20);
    }

    #[test]
    fn enc_dec_reset_allows_fresh_allocations() {
        let mut mgr = EncoderDecoderCacheManager::new(10);

        mgr.allocate("img1", 10);
        assert_eq!(mgr.num_free_slots(), 0);

        mgr.reset();

        assert!(mgr.can_allocate(8, usize::MAX, 0));
        mgr.allocate("img2", 8);

        assert_eq!(mgr.num_free_slots(), 2);
        assert!(mgr.allocated().contains(&"img2".to_string()));
    }

    #[test]
    fn enc_dec_respects_compute_budget() {
        let mgr = EncoderDecoderCacheManager::new(100);

        // 6 embeds > 5 budget
        assert!(!mgr.can_allocate(6, 5, 0));
        // 5 embeds <= 5 budget
        assert!(mgr.can_allocate(5, 5, 0));
    }

    #[test]
    fn enc_dec_respects_space_with_scheduled() {
        let mgr = EncoderDecoderCacheManager::new(10);

        // 5 + 6 = 11 > 10 cache
        assert!(!mgr.can_allocate(5, usize::MAX, 6));
        // 4 + 6 = 10 <= 10 cache
        assert!(mgr.can_allocate(4, usize::MAX, 6));
    }

    // ==================== compute_mm_encoder_budget Tests ====================

    #[test]
    fn budget_empty_modalities() {
        let map = HashMap::new();
        let (compute, cache) = compute_mm_encoder_budget(100, 200, &map);
        assert_eq!(compute, 0);
        assert_eq!(cache, 0);
    }

    #[test]
    fn budget_single_modality() {
        let mut map = HashMap::new();
        map.insert("image".to_string(), 576);

        let (compute, cache) = compute_mm_encoder_budget(1024, 2048, &map);
        assert_eq!(compute, 1024);
        assert_eq!(cache, 2048);
    }

    #[test]
    fn budget_ensures_min_for_single_item() {
        let mut map = HashMap::new();
        map.insert("image".to_string(), 2000);

        // Config values smaller than max item → clamped up
        let (compute, cache) = compute_mm_encoder_budget(100, 200, &map);
        assert_eq!(compute, 2000);
        assert_eq!(cache, 2000);
    }

    #[test]
    fn budget_multiple_modalities() {
        let mut map = HashMap::new();
        map.insert("image".to_string(), 576);
        map.insert("video".to_string(), 1200);
        map.insert("audio".to_string(), 300);

        let (compute, cache) = compute_mm_encoder_budget(1024, 2048, &map);
        // max item = 1200, so compute = max(1024, 1200) = 1200
        assert_eq!(compute, 1200);
        // cache = max(2048, 1200) = 2048
        assert_eq!(cache, 2048);
    }
}
