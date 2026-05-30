//! Token-to-bytes mapping for grammar-based constraints.
//!
//! `VocabularyIndex` extracts the byte representation of every token
//! from a tokenizer, built once at backend init and shared across all
//! grammar instances via `Arc`.

use std::sync::Arc;

use crate::tokenizer::TokenizerWrapper;

/// Maps each token ID to its byte representation.
///
/// Built once from a tokenizer and shared (via `Arc`) across all
/// grammar instances for the lifetime of the backend.
#[derive(Debug, Clone)]
pub struct VocabularyIndex {
    /// Per-token byte sequence in the **decoded** form (what the
    /// model produces when it emits this token, post-detokenisation).
    /// Empty vec for tokens that cannot be decoded.
    token_bytes: Vec<Vec<u8>>,
    /// Per-token byte sequence in the **raw HF BPE** form (with the
    /// byte-alphabet alias chars `Ġ`/`Ċ`/etc. preserved). `None` if
    /// the underlying tokenizer didn't expose a `get_vocab()` map or
    /// the construction path didn't extract one (synthetic/test vocab).
    ///
    /// xgrammar's `VocabType::ByteLevel` path needs THIS form to map
    /// multi-byte BPE tokens back to grammar bytes correctly. The
    /// `from_tokenizer` constructor populates it from `tokenizer.get_vocab()`;
    /// `from_token_bytes` leaves it `None`.
    raw_token_bytes: Option<Vec<Vec<u8>>>,
    /// Total vocabulary size.
    vocab_size: usize,
}

impl VocabularyIndex {
    /// Build a vocabulary index by decoding each token individually.
    ///
    /// Also extracts the HF BPE raw token strings via
    /// `tokenizer.get_vocab()` and stores them alongside the decoded
    /// form. xgrammar prefers the raw form for `ByteLevel` vocab type
    /// where the `Ġ`/`Ċ` byte-alias chars matter; the decoded form
    /// is used elsewhere and matches the user-visible output bytes.
    pub fn from_tokenizer(tokenizer: &TokenizerWrapper) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut token_bytes = Vec::with_capacity(vocab_size);

        for token_id in 0..vocab_size {
            let bytes = tokenizer
                .decode(&[token_id as u32])
                .map(|s| s.into_bytes())
                .unwrap_or_default();
            token_bytes.push(bytes);
        }

        // Pull the raw vocab map and invert into id-indexed bytes.
        // `get_vocab(true)` includes special tokens; we keep every
        // position 0..vocab_size filled (falling back to the decoded
        // form for any token id missing from the raw map).
        let raw_map = tokenizer.get_vocab();
        let raw_token_bytes = if raw_map.is_empty() {
            None
        } else {
            let mut raw: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
            for (s, id) in &raw_map {
                let idx = *id as usize;
                if idx < vocab_size {
                    raw[idx] = s.as_bytes().to_vec();
                }
            }
            // Fill gaps with the decoded form so xgrammar's
            // TokenizerInfo gets a complete vocab regardless of
            // sparse special-token IDs.
            for (idx, slot) in raw.iter_mut().enumerate() {
                if slot.is_empty() {
                    *slot = token_bytes[idx].clone();
                }
            }
            Some(raw)
        };

        Self {
            token_bytes,
            raw_token_bytes,
            vocab_size,
        }
    }

    /// Build a vocabulary index from an `Arc<TokenizerWrapper>`.
    pub fn from_tokenizer_arc(tokenizer: &Arc<TokenizerWrapper>) -> Self {
        Self::from_tokenizer(tokenizer.as_ref())
    }

    /// Build a vocabulary index directly from a `(token_id → bytes)`
    /// list. Intended for tests / synthetic byte-level vocabularies
    /// (e.g. xgrammar integration smoke tests that need character-
    /// level grammar checks without spinning up a real HF tokenizer).
    pub fn from_token_bytes(token_bytes: Vec<Vec<u8>>) -> Self {
        let vocab_size = token_bytes.len();
        Self {
            token_bytes,
            raw_token_bytes: None,
            vocab_size,
        }
    }

    /// Borrow the raw HF BPE vocab (with `Ġ`/`Ċ` byte-alias
    /// markers preserved). Returns `None` for indexes that were
    /// built without an HF tokenizer (synthetic / test vocab).
    /// Iteration is in token-id order, same as [`Self::iter`].
    pub fn raw_token_bytes_iter(&self) -> Option<impl Iterator<Item = (u32, &[u8])>> {
        self.raw_token_bytes.as_ref().map(|raw| {
            raw.iter()
                .enumerate()
                .map(|(i, b)| (i as u32, b.as_slice()))
        })
    }

    /// Get the byte sequence for a token ID.
    #[inline]
    pub fn token_bytes(&self, token_id: u32) -> &[u8] {
        self.token_bytes
            .get(token_id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Iterate over all (token_id, bytes) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.token_bytes
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_testing_tokenizer() {
        let tokenizer = TokenizerWrapper::for_testing(10);
        let index = VocabularyIndex::from_tokenizer(&tokenizer);

        assert_eq!(index.vocab_size(), 10);

        // The test tokenizer uses "t0".."t9" tokens
        // decode(0) should give "t0", etc.
        let t0_bytes = index.token_bytes(0);
        assert!(!t0_bytes.is_empty(), "token 0 should have bytes");
    }

    #[test]
    fn out_of_range_returns_empty() {
        let tokenizer = TokenizerWrapper::for_testing(5);
        let index = VocabularyIndex::from_tokenizer(&tokenizer);

        assert!(index.token_bytes(999).is_empty());
    }

    #[test]
    fn iter_yields_all_tokens() {
        let tokenizer = TokenizerWrapper::for_testing(5);
        let index = VocabularyIndex::from_tokenizer(&tokenizer);

        let entries: Vec<_> = index.iter().collect();
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].0, 0);
        assert_eq!(entries[4].0, 4);
    }
}
