//! `StructuredOutputGrammar` adapter backed by upstream xgrammar
//! (mlc-ai/xgrammar) via the `xgrammar-rs` FFI crate.
//!
//! Provides full JSON Schema / regex / EBNF enforcement at parity
//! with the Python xgrammar library used by upstream vLLM. Replaces
//! the partial native Rust port at `super::json_schema` for the
//! JSON-Schema path when the `xgrammar` feature is enabled.
//!
//! ## Vocabulary mapping (known gap)
//!
//! [`super::vocabulary::VocabularyIndex`] decodes each token through
//! the HF tokenizer's `decode` method, yielding plain UTF-8 bytes.
//! That maps cleanly to xgrammar's `VocabType::Raw`. For Qwen3 /
//! Llama / Mistral families this is sufficient because their
//! tokenizers never split a single Unicode codepoint across two
//! tokens, so every token has a stable decoded form.
//!
//! For tokenizers that DO split codepoints (some older SentencePiece
//! variants), this routing would silently allow the matcher to
//! accept "valid" byte sequences that the tokenizer then emits as
//! invalid UTF-8. Tracking work: switch the bridge to consume the
//! tokenizer's raw BPE byte alphabet (`tokenizer.get_vocab()`) and
//! pick the matching `VocabType` family per detection. Documented in
//! the build-vocab path below as a TODO.

use std::sync::{Arc, Mutex};

use anyhow::Context;

use super::vocabulary::{RawVocabKind, VocabularyIndex};
use super::{PackedBitmask, StructuredOutputGrammar};

/// Pick the xgrammar `VocabType` for a vocabulary.
///
/// 1. Manual override wins: `VLLM_XGRAMMAR_VOCAB_TYPE=raw|byte_fallback|byte_level`
///    (case-insensitive). Unknown values are ignored with a warn-log.
/// 2. Heuristic auto-detect on the decoded bytes that
///    [`VocabularyIndex`] already exposes:
///    - `<0xNN>` SentencePiece byte-fallback markers ⇒ `ByteFallback`
///    - `Ġ` (U+0120) / `Ċ` (U+010A) HF BPE byte-alphabet glyphs ⇒ `ByteLevel`
///    - otherwise ⇒ `Raw` (Qwen / GPT-OSS / most modern decoded BPE)
///
/// **Known gap (documented):** for SentencePiece tokenizers whose
/// `decode([id])` already resolves byte-fallback markers into raw
/// bytes (Llama/Mistral with HF tokenizers crate), the marker pattern
/// isn't visible in the decoded vocab — detection lands on Raw and
/// we emit a WARN-log to surface the gap. The proper fix is to
/// consume the tokenizer's raw form via `tokenizer.get_vocab()`,
/// which is an architectural change to [`VocabularyIndex`]. Tracked
/// as a follow-up; for now, operators can set the env override.
pub(crate) fn detect_vocab_type(vocab: &VocabularyIndex) -> xgrammar_rs::VocabType {
    if let Ok(raw) = std::env::var("VLLM_XGRAMMAR_VOCAB_TYPE") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "raw" => return xgrammar_rs::VocabType::Raw,
            "byte_fallback" | "bytefallback" => return xgrammar_rs::VocabType::ByteFallback,
            "byte_level" | "bytelevel" => return xgrammar_rs::VocabType::ByteLevel,
            other => {
                tracing::warn!(
                    target: "vllm_core::xgrammar",
                    value = %other,
                    "VLLM_XGRAMMAR_VOCAB_TYPE has unknown value (allowed: raw, byte_fallback, byte_level); falling through to auto-detect"
                );
            }
        }
    }

    detect_vocab_type_from_pieces(vocab.iter().map(|(_, b)| b))
}

/// Heuristic core shared by the decoded and raw-HF-vocab paths.
///
/// Marker counting instead of any-single-token matching: Gemma's raw
/// SentencePiece vocab contains exactly one literal `Ġ` piece (a rare
/// character, not a byte-alias) — a "saw any Ġ ⇒ ByteLevel" rule would
/// misclassify the whole vocab on that single glyph, making xgrammar
/// decode it as a 0x20 space and forcing it into grammar-constrained
/// output. Priority:
///   1. `<0xNN>` byte-fallback pieces present ⇒ `ByteFallback`
///   2. `▁`-prefixed pieces outnumber `Ġ`/`Ċ`-prefixed ⇒ `ByteFallback`
///   3. `Ġ`/`Ċ`-prefixed pieces outnumber `▁`-prefixed ⇒ `ByteLevel`
///   4. otherwise ⇒ `Raw`
pub(crate) fn detect_vocab_type_from_pieces<'a>(
    pieces: impl Iterator<Item = &'a [u8]>,
) -> xgrammar_rs::VocabType {
    let mut saw_sp_byte_marker = false;
    let mut sp_space_count: usize = 0;
    let mut bpe_glyph_count: usize = 0;
    let mut non_ascii_count: usize = 0;
    let mut sample_count: usize = 0;

    for bytes in pieces {
        if bytes.is_empty() {
            continue;
        }
        sample_count += 1;
        // Cheap byte-window scan first — all markers are short.
        // SentencePiece byte-fallback tokens are exactly "<0xNN>" —
        // 6 ASCII bytes, easy substring match.
        if bytes.len() == 6
            && bytes[0] == b'<'
            && bytes[1] == b'0'
            && bytes[2] == b'x'
            && bytes[5] == b'>'
            && bytes[3].is_ascii_hexdigit()
            && bytes[4].is_ascii_hexdigit()
        {
            saw_sp_byte_marker = true;
        }
        // SentencePiece space marker: piece starting with U+2581 (▁,
        // UTF-8 = 0xE2 0x96 0x81).
        if bytes.len() >= 3 && bytes[0] == 0xE2 && bytes[1] == 0x96 && bytes[2] == 0x81 {
            sp_space_count += 1;
        }
        // HF BPE byte alphabet: piece starting with U+0120 (Ġ,
        // UTF-8 = 0xC4 0xA0) or U+010A (Ċ, UTF-8 = 0xC4 0x8A).
        if bytes.len() >= 2 && bytes[0] == 0xC4 && (bytes[1] == 0xA0 || bytes[1] == 0x8A) {
            bpe_glyph_count += 1;
        }
        // Track non-ASCII byte fraction so we can flag suspicious
        // "all-ASCII vocab" decisions below.
        if bytes.iter().any(|&b| b >= 0x80) {
            non_ascii_count += 1;
        }
    }

    if saw_sp_byte_marker || sp_space_count > bpe_glyph_count {
        return xgrammar_rs::VocabType::ByteFallback;
    }
    if bpe_glyph_count > 0 {
        return xgrammar_rs::VocabType::ByteLevel;
    }

    // Fell through to Raw. If the vocab looks suspiciously
    // ASCII-only (< 5% tokens with high-bit bytes) on a non-trivial
    // sample, the model is probably SentencePiece-family with
    // already-resolved byte-fallback — warn the operator to set
    // the override.
    if sample_count >= 1024 {
        let frac = non_ascii_count as f64 / sample_count as f64;
        if frac < 0.05 {
            tracing::warn!(
                target: "vllm_core::xgrammar",
                non_ascii_fraction = frac,
                sample_count,
                "xgrammar vocab-type auto-detect landed on Raw with very low non-ASCII fraction. \
                 If this is a SentencePiece family model (Llama/Mistral/Mixtral), set \
                 VLLM_XGRAMMAR_VOCAB_TYPE=byte_fallback to enforce grammars correctly."
            );
        }
    }
    xgrammar_rs::VocabType::Raw
}

/// Build an xgrammar `TokenizerInfo` from our `VocabularyIndex`.
/// Cached at `crate::sampling::grammar::compiler::GrammarCompiler`
/// level so the marshal cost is paid once per engine init, not per
/// request.
///
/// Routing:
///   - If the `VocabularyIndex` carries a `raw_token_bytes` view
///     (built from `tokenizer.get_vocab()`), pass those raw byte
///     strings + the vocab type derived from the tokenizer's `decoder`
///     definition (`RawVocabKind`, the same signal upstream xgrammar's
///     `HFTokenizerAnalyzer::DetectVocabType` keys off): `ByteLevel`
///     decoder (GPT-2/Qwen `Ġ`/`Ċ` aliases) ⇒ `ByteLevel`,
///     `ByteFallback` decoder (SentencePiece `▁` + `<0xNN>`, e.g.
///     Gemma) ⇒ `ByteFallback`. If no decoder kind is available, fall
///     back to the piece-content heuristic. Hardcoding `ByteLevel`
///     here made xgrammar read Gemma's single literal `Ġ` piece as a
///     space and emit it for grammar space literals.
///   - Otherwise (synthetic vocab built via `from_token_bytes`),
///     pass the decoded form + heuristic `detect_vocab_type`.
///
/// The env override `VLLM_XGRAMMAR_VOCAB_TYPE` still wins over both.
pub(crate) fn tokenizer_info_from_vocab(
    vocab: &VocabularyIndex,
    stop_token_ids: &[u32],
) -> anyhow::Result<Arc<xgrammar_rs::TokenizerInfo>> {
    // Env override takes precedence (and uses the decoded form, since
    // operators setting the env explicitly know what they want).
    let env_override = std::env::var("VLLM_XGRAMMAR_VOCAB_TYPE").ok();

    let (encoded, vocab_type) = match (env_override.as_deref(), vocab.raw_token_bytes_iter()) {
        // Operator-forced type — use decoded form.
        (Some(_), _) => {
            let enc: Vec<Vec<u8>> = vocab.iter().map(|(_, b)| b.to_vec()).collect();
            (enc, detect_vocab_type(vocab))
        }
        // Raw HF vocab available → vocab type from the tokenizer's
        // decoder definition (authoritative, mirrors upstream
        // xgrammar); piece-content heuristic only when the tokenizer
        // exposed no decoder.
        (None, Some(raw_iter)) => {
            let enc: Vec<Vec<u8>> = raw_iter.map(|(_, b)| b.to_vec()).collect();
            let vocab_type = match vocab.decoder_vocab_kind() {
                Some(RawVocabKind::ByteLevel) => xgrammar_rs::VocabType::ByteLevel,
                Some(RawVocabKind::ByteFallback) => xgrammar_rs::VocabType::ByteFallback,
                Some(RawVocabKind::Raw) => xgrammar_rs::VocabType::Raw,
                None => detect_vocab_type_from_pieces(enc.iter().map(|v| v.as_slice())),
            };
            (enc, vocab_type)
        }
        // No raw vocab and no override → decoded + heuristic.
        (None, None) => {
            let enc: Vec<Vec<u8>> = vocab.iter().map(|(_, b)| b.to_vec()).collect();
            (enc, detect_vocab_type(vocab))
        }
    };

    tracing::info!(
        target: "vllm_core::xgrammar",
        ?vocab_type,
        decoder_kind = ?vocab.decoder_vocab_kind(),
        vocab_size = vocab.vocab_size(),
        source = if vocab.raw_token_bytes_iter().is_some() && env_override.is_none() {
            "raw_hf_vocab"
        } else {
            "decoded_per_token"
        },
        "xgrammar TokenizerInfo built"
    );

    let info = xgrammar_rs::TokenizerInfo::from_encoded_vocab(
        &encoded,
        vocab_type,
        /* add_prefix_space = */ false,
        stop_token_ids,
    )
    .context("xgrammar TokenizerInfo::from_encoded_vocab")?;
    Ok(Arc::new(info))
}

/// Convenience: build an xgrammar-rs `GrammarCompiler` from our vocab.
pub(crate) fn make_xgrammar_compiler(
    vocab: &VocabularyIndex,
    stop_token_ids: &[u32],
) -> anyhow::Result<Arc<xgrammar_rs::GrammarCompiler>> {
    let tok = tokenizer_info_from_vocab(vocab, stop_token_ids)?;
    // max_threads 4 (upstream default 8): halves the CPU burst of a
    // pathological compile without measurably slowing legitimate schema
    // compiles (sub-second). max_memory_bytes bounds the upstream
    // grammar+rule caches — one pathological grammar's token masks
    // reach tens of MB at a 262k vocab, so "unlimited" (-1) is an OOM
    // vector on an 8-10GB host.
    let compiler = xgrammar_rs::GrammarCompiler::new_with_options(
        tok,
        xgrammar_rs::CompilerOptions {
            max_threads: 4,
            cache_enabled: true,
            max_memory_bytes: 256 * 1024 * 1024,
        },
    )
    .context("xgrammar GrammarCompiler::new_with_options")?;
    Ok(Arc::new(compiler))
}

/// Per-request grammar wrapper that holds the stateful xgrammar
/// `GrammarMatcher` plus the shared compiled grammar / tokenizer that
/// own its backing tables.
///
/// `xgrammar_rs::GrammarMatcher` is `Send` but `!Sync`. The trait
/// `StructuredOutputGrammar` requires `Send + Sync`, so we wrap the
/// matcher in a `Mutex`. In the engine pipeline each per-sequence
/// constraint is accessed serially, so the mutex contention is
/// effectively zero — but the type-level `Sync` requirement is met
/// and the wrapper composes safely with any future parallelism.
pub struct XGrammarGrammar {
    matcher: Mutex<xgrammar_rs::GrammarMatcher>,
    vocab_size: usize,
    // Held to keep the compiled grammar / tokenizer alive for the
    // matcher's lifetime. The `Arc`s also cheap-clone into new
    // matchers when the compiler factory re-instantiates.
    _grammar: Arc<xgrammar_rs::CompiledGrammar>,
}

impl XGrammarGrammar {
    pub(crate) fn from_compiled(
        grammar: Arc<xgrammar_rs::CompiledGrammar>,
        vocab_size: usize,
        max_rollback_tokens: i32,
    ) -> anyhow::Result<Self> {
        let matcher = grammar
            .new_matcher(max_rollback_tokens)
            .context("xgrammar CompiledGrammar::new_matcher")?;
        Ok(Self {
            matcher: Mutex::new(matcher),
            vocab_size,
            _grammar: grammar,
        })
    }
}

impl std::fmt::Debug for XGrammarGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XGrammarGrammar")
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl StructuredOutputGrammar for XGrammarGrammar {
    fn accept_tokens(&mut self, tokens: &[u32]) -> bool {
        let mut m = self.matcher.lock().expect("xgrammar matcher poisoned");
        for &t in tokens {
            match m.accept_token(t) {
                Ok(true) => {}
                Ok(false) | Err(_) => return false,
            }
        }
        true
    }

    fn fill_bitmask(&self, bitmask: &mut PackedBitmask, batch_index: usize) {
        // The `&self` signature means we cannot read-from-self and
        // write-to-bitmask simultaneously without splitting the
        // borrows — `row_mut` does that for us. Lock the matcher,
        // then write into the row.
        let m = self.matcher.lock().expect("xgrammar matcher poisoned");
        let row = bitmask.row_mut(batch_index);
        // If xgrammar reports "all allowed" (need_mask=false), we
        // explicitly fill the row with !0 to maintain the
        // adapter's invariant that a cleared row means "no token
        // allowed". The native ports follow the same convention.
        match m.fill_next_token_bitmask(row) {
            Ok(true) => {
                // Mask is already in `row`; nothing further to do.
            }
            Ok(false) => {
                // "Every token allowed" — set every bit.
                for w in row.iter_mut() {
                    *w = !0i32;
                }
            }
            Err(_) => {
                // Defensive: on a matcher error, mask everything
                // (caller falls through to error handling via the
                // adapter's terminated path on the next step).
                for w in row.iter_mut() {
                    *w = 0;
                }
            }
        }
    }

    fn rollback(&mut self, num_tokens: usize) {
        // `Rollback` errors only when num_tokens exceeds the matcher
        // budget — silently bound in that case, matching native
        // backend behaviour.
        let _ = self
            .matcher
            .lock()
            .expect("xgrammar matcher poisoned")
            .rollback(num_tokens);
    }

    fn is_terminated(&self) -> bool {
        self.matcher
            .lock()
            .expect("xgrammar matcher poisoned")
            .is_terminated()
    }

    fn reset(&mut self) {
        self.matcher
            .lock()
            .expect("xgrammar matcher poisoned")
            .reset();
    }

    fn xgrammar_matcher(&self) -> Option<&Mutex<xgrammar_rs::GrammarMatcher>> {
        Some(&self.matcher)
    }
}

/// Process-global thread-pooled batched matcher for parallel bitmask
/// fill across a constrained decode batch. `max_threads=0` → xgrammar
/// "auto" (number of CPU cores). Constructed once; cheap to share.
pub(crate) fn batch_matcher_handle() -> &'static xgrammar_rs::BatchMatcher {
    use std::sync::OnceLock;
    static CACHE: OnceLock<xgrammar_rs::BatchMatcher> = OnceLock::new();
    CACHE.get_or_init(|| xgrammar_rs::BatchMatcher::new(0).expect("BatchMatcher::new(auto) failed"))
}

// Note: spec-string-to-`Arc<xgrammar_rs::CompiledGrammar>` is folded
// into the caller (`super::compiler::xgrammar_compile_to_arc`) so the
// async fast path can share the compiled grammar across requests.
// `XGrammarGrammar::from_compiled` above is what callers reach for.

#[cfg(test)]
mod tests {
    use super::*;

    /// Save the current env var (if any) and restore on drop, so
    /// `VLLM_XGRAMMAR_VOCAB_TYPE`-touching tests don't leak state to
    /// each other.
    ///
    /// Process-wide env mutation is intrinsically global state, so
    /// the multi-threaded test harness can race two tests' set/clear
    /// pairs unless we serialise them. The guard holds a static
    /// `Mutex` for its lifetime, ensuring only one env-touching test
    /// runs at a time regardless of `--test-threads`.
    struct EnvGuard {
        prev: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    impl EnvGuard {
        fn set(value: &str) -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var("VLLM_XGRAMMAR_VOCAB_TYPE").ok();
            std::env::set_var("VLLM_XGRAMMAR_VOCAB_TYPE", value);
            Self { prev, _lock: lock }
        }
        fn clear() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var("VLLM_XGRAMMAR_VOCAB_TYPE").ok();
            std::env::remove_var("VLLM_XGRAMMAR_VOCAB_TYPE");
            Self { prev, _lock: lock }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var("VLLM_XGRAMMAR_VOCAB_TYPE", v),
                None => std::env::remove_var("VLLM_XGRAMMAR_VOCAB_TYPE"),
            }
            // _lock drops here, releasing the mutex.
        }
    }

    fn vocab_from(strings: &[&str]) -> VocabularyIndex {
        let bytes: Vec<Vec<u8>> = strings.iter().map(|s| s.as_bytes().to_vec()).collect();
        VocabularyIndex::from_token_bytes(bytes)
    }

    #[test]
    fn detect_raw_for_qwen_like_vocab() {
        let _g = EnvGuard::clear();
        // Plain UTF-8 fragments with no SP/BPE markers — Qwen3-style.
        let v = vocab_from(&["hello", " world", "привет", "你好", "!", "?", "{", "}"]);
        assert_eq!(detect_vocab_type(&v), xgrammar_rs::VocabType::Raw);
    }

    #[test]
    fn detect_byte_fallback_for_sentencepiece_markers() {
        let _g = EnvGuard::clear();
        // Any token of the exact form `<0xNN>` flips to ByteFallback.
        let v = vocab_from(&["hello", "<0x0A>", "world", "<0xFF>"]);
        assert_eq!(detect_vocab_type(&v), xgrammar_rs::VocabType::ByteFallback);
    }

    #[test]
    fn detect_byte_level_for_bpe_glyphs() {
        let _g = EnvGuard::clear();
        // U+0120 (Ġ) UTF-8 = 0xC4 0xA0 — HF BPE byte alphabet marker.
        let v = vocab_from(&["hello", "\u{0120}world"]);
        assert_eq!(detect_vocab_type(&v), xgrammar_rs::VocabType::ByteLevel);
    }

    #[test]
    fn env_override_wins_over_heuristic() {
        // Heuristic would say ByteFallback (sees <0x0A>), env says Raw.
        let _g = EnvGuard::set("raw");
        let v = vocab_from(&["<0x0A>", "x"]);
        assert_eq!(detect_vocab_type(&v), xgrammar_rs::VocabType::Raw);
    }

    #[test]
    fn env_override_accepts_case_variations() {
        let _g = EnvGuard::set("Byte_Fallback");
        // Pure-ASCII vocab — heuristic would land on Raw.
        let v = vocab_from(&["abc", "def"]);
        assert_eq!(detect_vocab_type(&v), xgrammar_rs::VocabType::ByteFallback);
    }

    #[test]
    fn raw_sentencepiece_vocab_with_stray_g_glyph_is_byte_fallback() {
        // Gemma 4's RAW HF vocab is SentencePiece-style: `▁`-prefixed
        // pieces + `<0xNN>` byte-fallback markers — and it ALSO contains a
        // single literal `Ġ` piece (a genuine rare character, id 245237).
        // Classifying this vocab as ByteLevel makes xgrammar decode that
        // `Ġ` piece as a 0x20 space, so a grammar with a literal space
        // happily forces token `Ġ` into the output (observed live:
        // `{"chosen":Ġ"2"}`). The stray glyph must not flip the decision.
        let pieces: Vec<&[u8]> = vec![
            "\u{2581}the".as_bytes(),
            "\u{2581}\"".as_bytes(),
            "<0x0A>".as_bytes(),
            "hello".as_bytes(),
            "\u{0120}".as_bytes(), // the stray Ġ
        ];
        assert_eq!(
            detect_vocab_type_from_pieces(pieces.into_iter()),
            xgrammar_rs::VocabType::ByteFallback
        );
    }

    #[test]
    fn raw_sentencepiece_vocab_without_byte_markers_is_byte_fallback() {
        // Even with the `<0xNN>` markers absent, a vocab dominated by
        // `▁`-prefixed pieces is SentencePiece-style.
        let pieces: Vec<&[u8]> = vec![
            "\u{2581}the".as_bytes(),
            "\u{2581}world".as_bytes(),
            "\u{2581}a".as_bytes(),
            "hello".as_bytes(),
            "\u{0120}".as_bytes(),
        ];
        assert_eq!(
            detect_vocab_type_from_pieces(pieces.into_iter()),
            xgrammar_rs::VocabType::ByteFallback
        );
    }

    #[test]
    fn raw_byte_level_vocab_stays_byte_level() {
        // GPT-2-style raw vocab: Ġ-prefixed pieces dominate, no ▁/<0xNN>.
        let pieces: Vec<&[u8]> = vec![
            "\u{0120}the".as_bytes(),
            "\u{0120}world".as_bytes(),
            "\u{010A}".as_bytes(),
            "hello".as_bytes(),
        ];
        assert_eq!(
            detect_vocab_type_from_pieces(pieces.into_iter()),
            xgrammar_rs::VocabType::ByteLevel
        );
    }
}
