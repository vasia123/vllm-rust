//! SHA-256-keyed LRU cache for vision encoder outputs.
//!
//! Repeated images across requests (multi-turn conversations, batches with
//! repeated context) produce identical `ProcessedImage` tensors from the
//! vision encoder. This cache skips the encoder call on a hit, which is the
//! dominant latency cost in multimodal preprocessing.
//!
//! The cache is transparent to callers: `MultimodalProcessor::process_image`
//! performs the lookup before calling the encoder and stores results on a miss.
//!
//! Eviction is LRU: the least-recently-used entry is evicted when the cache
//! reaches capacity.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::inputs::{ImageData, ImageSource, ProcessedImage};

/// Default maximum number of cached `ProcessedImage` entries.
///
/// At ~1–10 MB per embedding this keeps memory impact below ~1 GB for typical
/// VLM hidden sizes. Operators can override via `--mm-preprocessor-cache-size`
/// when that CLI flag is wired (future work).
pub const DEFAULT_CACHE_CAPACITY: usize = 128;

/// SHA-256-keyed LRU in-memory cache for preprocessed vision encoder outputs.
///
/// Thread-safety: this struct is not `Sync` by itself. Callers wrap it in
/// `Arc<Mutex<PreprocessorCache>>` (see `MultimodalProcessor`).
pub struct PreprocessorCache {
    entries: HashMap<String, Arc<ProcessedImage>>,
    /// Oldest key at the front, newest at the back. Maintained in access order
    /// so `pop_front()` always removes the least-recently-used entry.
    order: VecDeque<String>,
    capacity: usize,
}

impl PreprocessorCache {
    /// Create a new cache with the given maximum number of entries.
    ///
    /// # Panics
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "cache capacity must be > 0");
        Self {
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Compute a deterministic SHA-256 hex key for `image`.
    ///
    /// The key is `hex(SHA-256(prefix || raw_source_bytes [|| ":WxH"]))` where
    /// the prefix distinguishes source variants so collisions across types are
    /// impossible even for equal byte sequences.
    ///
    /// Returns `None` for `ImageSource::Embedding`: those are already encoded
    /// and bypass the vision encoder entirely.
    pub fn compute_key(image: &ImageData) -> Option<String> {
        let mut hasher = Sha256::new();
        match &image.source {
            ImageSource::Base64(data) => {
                hasher.update(b"b64:");
                hasher.update(data.as_bytes());
            }
            ImageSource::Bytes(bytes) => {
                hasher.update(b"bytes:");
                hasher.update(bytes.as_slice());
            }
            ImageSource::Url(url) => {
                // URL images are not pre-fetched; the key is the URL string.
                // Two requests for the same URL will share a cache entry only
                // if the remote content is stable (the common case for static
                // assets). Dynamic URLs should use Bytes or Base64 instead.
                hasher.update(b"url:");
                hasher.update(url.as_bytes());
            }
            ImageSource::Embedding(_) => return None,
        }
        // Same raw image resized to different dimensions → different encodings.
        if let Some((w, h)) = image.target_size {
            hasher.update(format!(":{w}x{h}").as_bytes());
        }
        Some(format!("{:x}", hasher.finalize()))
    }

    /// Return a cached `ProcessedImage`, promoting it to MRU position.
    ///
    /// Returns `None` on a cache miss.
    pub fn get(&mut self, key: &str) -> Option<Arc<ProcessedImage>> {
        if self.entries.contains_key(key) {
            // Promote to most-recently-used.
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_owned());
            self.entries.get(key).cloned()
        } else {
            None
        }
    }

    /// Insert a `ProcessedImage`, evicting the LRU entry when at capacity.
    ///
    /// Replacing an existing key moves it to MRU position without changing the
    /// entry count.
    pub fn insert(&mut self, key: String, value: ProcessedImage) {
        if self.entries.contains_key(&key) {
            // Replacement: update value and move to MRU.
            self.order.retain(|k| k != &key);
        } else if self.entries.len() >= self.capacity {
            // Evict the least-recently-used entry.
            if let Some(evict_key) = self.order.pop_front() {
                self.entries.remove(&evict_key);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, Arc::new(value));
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_processed_image() -> ProcessedImage {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((4, 64), DType::F32, &device).unwrap();
        ProcessedImage::new(tensor, 4)
    }

    // ── compute_key ──────────────────────────────────────────────────────────

    #[test]
    fn compute_key_bytes_differs_from_url() {
        let bytes_img = ImageData {
            source: ImageSource::Bytes(vec![0u8, 1, 2]),
            target_size: None,
        };
        let url_img = ImageData {
            source: ImageSource::Url("http://a.com/b.jpg".to_string()),
            target_size: None,
        };
        assert_ne!(
            PreprocessorCache::compute_key(&bytes_img).unwrap(),
            PreprocessorCache::compute_key(&url_img).unwrap()
        );
    }

    #[test]
    fn compute_key_target_size_distinguishes_same_source() {
        let img_a = ImageData {
            source: ImageSource::Url("http://x.com/img.jpg".to_string()),
            target_size: Some((224, 224)),
        };
        let img_b = ImageData {
            source: ImageSource::Url("http://x.com/img.jpg".to_string()),
            target_size: Some((336, 336)),
        };
        assert_ne!(
            PreprocessorCache::compute_key(&img_a).unwrap(),
            PreprocessorCache::compute_key(&img_b).unwrap()
        );
    }

    #[test]
    fn compute_key_same_inputs_produce_same_key() {
        let img = ImageData {
            source: ImageSource::Bytes(vec![1, 2, 3]),
            target_size: Some((224, 224)),
        };
        assert_eq!(
            PreprocessorCache::compute_key(&img).unwrap(),
            PreprocessorCache::compute_key(&img).unwrap()
        );
    }

    #[test]
    fn compute_key_embedding_returns_none() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((4, 64), DType::F32, &device).unwrap();
        let img = ImageData {
            source: ImageSource::Embedding(tensor),
            target_size: None,
        };
        assert!(PreprocessorCache::compute_key(&img).is_none());
    }

    // ── insert / get ─────────────────────────────────────────────────────────

    #[test]
    fn hit_after_insert() {
        let mut cache = PreprocessorCache::new(8);
        cache.insert("k1".to_string(), make_processed_image());
        assert_eq!(cache.len(), 1);
        assert!(cache.get("k1").is_some());
    }

    #[test]
    fn miss_for_unknown_key() {
        let mut cache = PreprocessorCache::new(8);
        assert!(cache.get("missing").is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn replacement_does_not_grow_len() {
        let mut cache = PreprocessorCache::new(4);
        cache.insert("a".to_string(), make_processed_image());
        cache.insert("a".to_string(), make_processed_image());
        assert_eq!(cache.len(), 1);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn lru_eviction_at_capacity() {
        let mut cache = PreprocessorCache::new(2);
        cache.insert("a".to_string(), make_processed_image());
        cache.insert("b".to_string(), make_processed_image());
        // Cache is full. Next insert should evict "a" (LRU).
        cache.insert("c".to_string(), make_processed_image());
        assert_eq!(cache.len(), 2);
        assert!(cache.get("a").is_none(), "LRU entry 'a' should be evicted");
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn access_promotes_to_mru_preventing_eviction() {
        let mut cache = PreprocessorCache::new(2);
        cache.insert("a".to_string(), make_processed_image());
        cache.insert("b".to_string(), make_processed_image());
        // Promote "a" to MRU; "b" becomes LRU.
        cache.get("a");
        // Inserting "c" should evict "b", not "a".
        cache.insert("c".to_string(), make_processed_image());
        assert!(cache.get("b").is_none(), "'b' should be evicted");
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn replacement_promotes_to_mru() {
        let mut cache = PreprocessorCache::new(2);
        cache.insert("a".to_string(), make_processed_image());
        cache.insert("b".to_string(), make_processed_image());
        // Replace "a" (currently LRU); it should become MRU.
        cache.insert("a".to_string(), make_processed_image());
        // Now "b" is LRU. Inserting "c" should evict "b".
        cache.insert("c".to_string(), make_processed_image());
        assert!(cache.get("b").is_none(), "'b' should be evicted");
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }
}
