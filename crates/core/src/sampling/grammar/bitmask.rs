//! Packed i32 bitmask for efficient token masking.
//!
//! Layout: `[max_batch, vocab_size.div_ceil(32)]` stored as `Vec<i32>`.
//! Each bit corresponds to one token ID: bit set = token allowed.

/// Packed bitmask for batched token masking.
///
/// Stores one row per batch element, each row a packed array of i32
/// where each bit represents whether the corresponding token is allowed.
/// This enables O(vocab_size/32) masking instead of O(vocab_size).
#[derive(Debug, Clone)]
pub struct PackedBitmask {
    /// Packed bitmask data, row-major: `[max_batch][words_per_row]`
    data: Vec<i32>,
    /// Number of i32 words per row
    words_per_row: usize,
    /// Maximum batch size (number of rows)
    max_batch: usize,
    /// Actual vocabulary size
    vocab_size: usize,
}

impl PackedBitmask {
    /// Create a new bitmask with all bits cleared (all tokens disallowed).
    pub fn new(max_batch: usize, vocab_size: usize) -> Self {
        let words_per_row = vocab_size.div_ceil(32);
        let data = vec![0i32; max_batch * words_per_row];
        Self {
            data,
            words_per_row,
            max_batch,
            vocab_size,
        }
    }

    /// Set all bits to zero (all tokens disallowed).
    pub fn set_all_zeros(&mut self) {
        self.data.fill(0);
    }

    /// Set all bits to one (all tokens allowed).
    pub fn set_all_allowed(&mut self) {
        self.data.fill(!0);
    }

    /// Set a single bit (mark token as allowed).
    #[inline]
    pub fn set_bit(&mut self, batch_index: usize, token_id: usize) {
        debug_assert!(batch_index < self.max_batch);
        debug_assert!(token_id < self.vocab_size);
        let word_idx = batch_index * self.words_per_row + token_id / 32;
        let bit_idx = token_id % 32;
        self.data[word_idx] |= 1i32 << bit_idx;
    }

    /// Clear a single bit (mark token as disallowed).
    #[inline]
    pub fn clear_bit(&mut self, batch_index: usize, token_id: usize) {
        debug_assert!(batch_index < self.max_batch);
        debug_assert!(token_id < self.vocab_size);
        let word_idx = batch_index * self.words_per_row + token_id / 32;
        let bit_idx = token_id % 32;
        self.data[word_idx] &= !(1i32 << bit_idx);
    }

    /// Test whether a bit is set (token is allowed).
    #[inline]
    pub fn get_bit(&self, batch_index: usize, token_id: usize) -> bool {
        debug_assert!(batch_index < self.max_batch);
        debug_assert!(token_id < self.vocab_size);
        let word_idx = batch_index * self.words_per_row + token_id / 32;
        let bit_idx = token_id % 32;
        (self.data[word_idx] >> bit_idx) & 1 != 0
    }

    /// Get mutable slice of a single row's packed words.
    pub fn row_mut(&mut self, batch_index: usize) -> &mut [i32] {
        let start = batch_index * self.words_per_row;
        &mut self.data[start..start + self.words_per_row]
    }

    /// Get immutable slice of a single row's packed words.
    pub fn row(&self, batch_index: usize) -> &[i32] {
        let start = batch_index * self.words_per_row;
        &self.data[start..start + self.words_per_row]
    }

    /// Copy a precomputed bitmask row into the given batch row.
    pub fn copy_row_from(&mut self, batch_index: usize, src: &[i32]) {
        debug_assert!(batch_index < self.max_batch);
        debug_assert!(src.len() >= self.words_per_row);
        let wpr = self.words_per_row;
        let row = self.row_mut(batch_index);
        row.copy_from_slice(&src[..wpr]);
    }

    /// Apply the bitmask to a logits slice: tokens with cleared bits get -inf.
    ///
    /// For each token, if its bit is 0 (disallowed), set the logit to NEG_INFINITY.
    /// Tokens with bit 1 (allowed) keep their original logit value.
    pub fn apply_to_logits(&self, logits: &mut [f32], batch_index: usize) {
        debug_assert!(batch_index < self.max_batch);
        let row_start = batch_index * self.words_per_row;

        // Process 32 tokens at a time using packed words
        for (word_idx, &word) in self.data[row_start..row_start + self.words_per_row]
            .iter()
            .enumerate()
        {
            let base_token = word_idx * 32;
            if word == !0i32 {
                // All bits set — all 32 tokens allowed, skip
                continue;
            }
            if word == 0 {
                // No bits set — mask all 32 tokens
                let end = (base_token + 32).min(logits.len());
                for logit in &mut logits[base_token..end] {
                    *logit = f32::NEG_INFINITY;
                }
                continue;
            }
            // Mixed — check individual bits
            for bit in 0..32 {
                let token_id = base_token + bit;
                if token_id >= logits.len() {
                    break;
                }
                if (word >> bit) & 1 == 0 {
                    logits[token_id] = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Number of i32 words per row.
    pub fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    /// Vocabulary size this bitmask was created for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bitmask_all_zeros() {
        let bm = PackedBitmask::new(2, 100);
        for batch in 0..2 {
            for token in 0..100 {
                assert!(!bm.get_bit(batch, token));
            }
        }
    }

    #[test]
    fn set_and_get_bit() {
        let mut bm = PackedBitmask::new(1, 64);
        bm.set_bit(0, 0);
        bm.set_bit(0, 31);
        bm.set_bit(0, 32);
        bm.set_bit(0, 63);

        assert!(bm.get_bit(0, 0));
        assert!(bm.get_bit(0, 31));
        assert!(bm.get_bit(0, 32));
        assert!(bm.get_bit(0, 63));
        assert!(!bm.get_bit(0, 1));
        assert!(!bm.get_bit(0, 30));
        assert!(!bm.get_bit(0, 33));
    }

    #[test]
    fn clear_bit() {
        let mut bm = PackedBitmask::new(1, 64);
        bm.set_bit(0, 10);
        assert!(bm.get_bit(0, 10));
        bm.clear_bit(0, 10);
        assert!(!bm.get_bit(0, 10));
    }

    #[test]
    fn set_all_allowed() {
        let mut bm = PackedBitmask::new(1, 100);
        bm.set_all_allowed();
        for token in 0..100 {
            assert!(bm.get_bit(0, token));
        }
    }

    #[test]
    fn set_all_zeros_clears() {
        let mut bm = PackedBitmask::new(1, 64);
        bm.set_all_allowed();
        bm.set_all_zeros();
        for token in 0..64 {
            assert!(!bm.get_bit(0, token));
        }
    }

    #[test]
    fn apply_to_logits_masks_disallowed() {
        let mut bm = PackedBitmask::new(1, 8);
        bm.set_bit(0, 2);
        bm.set_bit(0, 5);

        let mut logits = vec![1.0f32; 8];
        bm.apply_to_logits(&mut logits, 0);

        for (i, &l) in logits.iter().enumerate() {
            if i == 2 || i == 5 {
                assert_eq!(l, 1.0, "token {i} should be allowed");
            } else {
                assert_eq!(l, f32::NEG_INFINITY, "token {i} should be masked");
            }
        }
    }

    #[test]
    fn apply_to_logits_all_allowed_no_change() {
        let mut bm = PackedBitmask::new(1, 8);
        bm.set_all_allowed();

        let mut logits = vec![0.5f32; 8];
        bm.apply_to_logits(&mut logits, 0);

        for &l in &logits {
            assert_eq!(l, 0.5);
        }
    }

    #[test]
    fn multi_batch_isolation() {
        let mut bm = PackedBitmask::new(2, 64);
        bm.set_bit(0, 10);
        bm.set_bit(1, 20);

        assert!(bm.get_bit(0, 10));
        assert!(!bm.get_bit(0, 20));
        assert!(!bm.get_bit(1, 10));
        assert!(bm.get_bit(1, 20));
    }

    #[test]
    fn copy_row_from_precomputed() {
        let mut bm = PackedBitmask::new(1, 64);
        let precomputed = vec![0x0000_00FFi32, 0x0000_0000i32];
        bm.copy_row_from(0, &precomputed);

        for token in 0..8 {
            assert!(bm.get_bit(0, token), "token {token} should be set");
        }
        for token in 8..64 {
            assert!(!bm.get_bit(0, token), "token {token} should be clear");
        }
    }

    #[test]
    fn words_per_row_rounding() {
        let bm = PackedBitmask::new(1, 1);
        assert_eq!(bm.words_per_row(), 1);

        let bm = PackedBitmask::new(1, 32);
        assert_eq!(bm.words_per_row(), 1);

        let bm = PackedBitmask::new(1, 33);
        assert_eq!(bm.words_per_row(), 2);

        let bm = PackedBitmask::new(1, 128000);
        assert_eq!(bm.words_per_row(), 4000);
    }

    #[test]
    fn non_aligned_vocab_size() {
        // vocab_size not a multiple of 32
        let mut bm = PackedBitmask::new(1, 50);
        bm.set_bit(0, 49);
        assert!(bm.get_bit(0, 49));

        let mut logits = vec![1.0f32; 50];
        bm.apply_to_logits(&mut logits, 0);
        assert_eq!(logits[49], 1.0);
        assert_eq!(logits[48], f32::NEG_INFINITY);
    }
}
