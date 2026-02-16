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
    /// `token_bytes[token_id]` = byte sequence for that token.
    /// Empty vec for tokens that cannot be decoded.
    token_bytes: Vec<Vec<u8>>,
    /// Total vocabulary size.
    vocab_size: usize,
}

impl VocabularyIndex {
    /// Build a vocabulary index by decoding each token individually.
    ///
    /// Tokens that fail to decode (e.g., partial multi-byte sequences in
    /// byte-level tokenizers) get an empty byte vector.
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

        Self {
            token_bytes,
            vocab_size,
        }
    }

    /// Build a vocabulary index from an `Arc<TokenizerWrapper>`.
    pub fn from_tokenizer_arc(tokenizer: &Arc<TokenizerWrapper>) -> Self {
        Self::from_tokenizer(tokenizer.as_ref())
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
