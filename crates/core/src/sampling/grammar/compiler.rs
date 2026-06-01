//! Async grammar compilation with caching.
//!
//! `GrammarCompiler` compiles structured output specs (regex, JSON schema,
//! GBNF grammar) into `StructuredOutputGrammar` instances. DFA compilation
//! is offloaded to blocking threads via `tokio::task::spawn_blocking`.
//! Compiled templates are cached by content hash.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use super::vocabulary::VocabularyIndex;
use super::StructuredOutputGrammar;

/// What kind of structured output to compile.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructuredOutputOption {
    Regex,
    JsonSchema,
    Grammar,
}

/// A compiled grammar template that can be cloned to produce per-request instances.
///
/// For DFA-based grammars, this shares the precomputed state-to-bitmask map
/// and only clones the mutable state (current_state, history).
struct CompiledTemplate {
    /// Factory function to create a fresh grammar instance
    factory: Arc<dyn Fn() -> Box<dyn StructuredOutputGrammar> + Send + Sync>,
}

/// Async compiler for structured output grammars.
///
/// Caches compiled templates by content hash so repeated requests
/// with the same constraint pattern avoid recompilation.
///
/// With the `xgrammar` feature on, structured output (regex / JSON
/// schema / EBNF) compiles through the upstream xgrammar C++ library
/// for full constraint enforcement (`pattern`, `minLength`,
/// `additionalProperties:false`, etc.). The native Rust port at
/// `super::json_schema` / `super::regex_backend` /
/// `super::ebnf_backend` stays available as the fallback when the
/// feature is off — it has a smaller dependency footprint at the
/// cost of partial JSON-Schema enforcement.
pub struct GrammarCompiler {
    // Both fields are consumed only by the `xgrammar`-gated compile paths
    // (`xgrammar_compiler_handle`, `compile`, `compile_sync`). Without the
    // feature there is no backend, so they are intentionally unread.
    #[cfg_attr(not(feature = "xgrammar"), allow(dead_code))]
    vocab_index: Arc<VocabularyIndex>,
    /// Token ids that terminate the grammar at an accepting state. The
    /// model EOS belongs here: xgrammar only marks a stop token as
    /// *allowed* once the grammar reaches an accepting state, so a decode
    /// that has produced a complete document can emit EOS and stop
    /// cleanly. With an empty set xgrammar falls back to its own
    /// heuristically-detected end markers (`</s>`, `<|endoftext|>`),
    /// which need not match the model's real EOS — greedy then can't
    /// reach EOS at the accepting state and the engine emits trailing
    /// garbage (see the `diag_eos_termination_needs_stop_token_ids` test).
    #[cfg_attr(not(feature = "xgrammar"), allow(dead_code))]
    stop_token_ids: Vec<u32>,
    cache: Arc<Mutex<HashMap<u64, Arc<CompiledTemplate>>>>,
    /// Lazily-built upstream xgrammar compiler, shared across all
    /// `compile_*` calls on this `GrammarCompiler`. `None` while the
    /// `xgrammar` feature is off OR while no xgrammar-routed compile
    /// has been requested yet (in feature-on builds).
    #[cfg(feature = "xgrammar")]
    xgrammar_compiler: std::sync::OnceLock<Arc<xgrammar_rs::GrammarCompiler>>,
}

impl GrammarCompiler {
    /// Create a new compiler with the given vocabulary index and no
    /// configured stop tokens. Prefer [`Self::with_stop_tokens`] in the
    /// server so the grammar terminates on the model's real EOS.
    pub fn new(vocab_index: Arc<VocabularyIndex>) -> Self {
        Self::with_stop_tokens(vocab_index, Vec::new())
    }

    /// Create a compiler whose grammars terminate on `stop_token_ids`
    /// (typically the model EOS) once they reach an accepting state.
    pub fn with_stop_tokens(vocab_index: Arc<VocabularyIndex>, stop_token_ids: Vec<u32>) -> Self {
        Self {
            vocab_index,
            stop_token_ids,
            cache: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(feature = "xgrammar")]
            xgrammar_compiler: std::sync::OnceLock::new(),
        }
    }

    /// Lazily build (or fetch) the underlying xgrammar compiler bound
    /// to this `GrammarCompiler`'s vocabulary. The vocabulary marshal
    /// (token-bytes → xgrammar `TokenizerInfo`) happens on first
    /// call; subsequent calls just clone the `Arc`.
    #[cfg(feature = "xgrammar")]
    fn xgrammar_compiler_handle(&self) -> anyhow::Result<Arc<xgrammar_rs::GrammarCompiler>> {
        if let Some(c) = self.xgrammar_compiler.get() {
            return Ok(c.clone());
        }
        let built = super::xgrammar_backend::make_xgrammar_compiler(
            &self.vocab_index,
            &self.stop_token_ids,
        )?;
        // Race-safe: if another thread initialised first, drop ours.
        match self.xgrammar_compiler.set(built.clone()) {
            Ok(()) => Ok(built),
            Err(_) => Ok(self
                .xgrammar_compiler
                .get()
                .expect("OnceLock just set")
                .clone()),
        }
    }

    /// Compile a structured output specification into a grammar.
    ///
    /// Uses `spawn_blocking` for CPU-heavy DFA / NPDA compilation.
    /// Results are cached by content hash.
    ///
    /// With the `xgrammar` feature enabled, JSON-Schema / regex / EBNF
    /// route through upstream xgrammar — the compile cost (NPDA build,
    /// per-state token bitmask precompute) is incurred once per unique
    /// spec and amortised across every request via an `Arc`-shared
    /// `CompiledGrammar`. The cached factory only spins up a fresh
    /// `GrammarMatcher` per request (cheap stack-allocation).
    pub async fn compile(
        &self,
        option: StructuredOutputOption,
        spec: &str,
    ) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
        let spec_hash = hash_spec(&option, spec);

        // Check cache first
        {
            let cache = self.cache.lock().await;
            if let Some(template) = cache.get(&spec_hash) {
                return Ok((template.factory)());
            }
        }

        // ── xgrammar fast path ─────────────────────────────────────
        // Compile once into an Arc<CompiledGrammar>, then the factory
        // is a cheap `new_matcher` call per request.
        #[cfg(feature = "xgrammar")]
        {
            let xgr_compiler = self.xgrammar_compiler_handle()?;
            let spec_owned = spec.to_string();
            let option_for_compile = option.clone();
            let compiled = tokio::task::spawn_blocking(move || {
                xgrammar_compile_to_arc(&xgr_compiler, &option_for_compile, &spec_owned)
            })
            .await
            .map_err(|e| anyhow::anyhow!("xgrammar compile task panicked: {e}"))??;

            let vocab_size = self.vocab_index.vocab_size();
            let compiled_for_factory = compiled.clone();
            let factory: Arc<dyn Fn() -> Box<dyn StructuredOutputGrammar> + Send + Sync> =
                Arc::new(move || {
                    Box::new(
                        super::xgrammar_backend::XGrammarGrammar::from_compiled(
                            compiled_for_factory.clone(),
                            vocab_size,
                            /* max_rollback_tokens = */ 64,
                        )
                        .expect("cached xgrammar matcher creation must not fail"),
                    )
                });

            let template = Arc::new(CompiledTemplate { factory });
            {
                let mut cache = self.cache.lock().await;
                cache.insert(spec_hash, template.clone());
            }
            return Ok((template.factory)());
        }

        // ── No backend without xgrammar ────────────────────────────
        // The native Rust port was removed in v1.8 — Bug 10 proved
        // it could not enforce JSON-Schema `pattern + length` or
        // strict-object key boundaries correctly. Build with
        // `--features xgrammar` (or `cuda-full`, which implies it).
        #[cfg(not(feature = "xgrammar"))]
        {
            let _ = (option, spec, spec_hash);
            anyhow::bail!(
                "structured output is unavailable: rebuild with `--features xgrammar` \
                 (the native Rust grammar backends were removed in v1.8)"
            )
        }
    }

    /// Synchronous compilation (for use in non-async contexts).
    ///
    /// With `xgrammar` enabled this re-compiles from spec on every
    /// call (no factory caching) — acceptable for the test-only call
    /// sites; production goes through [`Self::compile`].
    pub fn compile_sync(
        &self,
        option: &StructuredOutputOption,
        spec: &str,
    ) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
        #[cfg(feature = "xgrammar")]
        {
            let xgr_compiler = self.xgrammar_compiler_handle()?;
            let arc = xgrammar_compile_to_arc(&xgr_compiler, option, spec)?;
            return Ok(Box::new(
                super::xgrammar_backend::XGrammarGrammar::from_compiled(
                    arc,
                    self.vocab_index.vocab_size(),
                    64,
                )?,
            ));
        }
        #[cfg(not(feature = "xgrammar"))]
        {
            let _ = (option, spec);
            anyhow::bail!("structured output is unavailable: rebuild with `--features xgrammar`")
        }
    }
}

/// Compile a structured-output spec to an `Arc<xgrammar_rs::CompiledGrammar>`.
/// Shared helper for both the async and sync compile paths under
/// `feature = "xgrammar"`.
#[cfg(feature = "xgrammar")]
fn xgrammar_compile_to_arc(
    compiler: &xgrammar_rs::GrammarCompiler,
    option: &StructuredOutputOption,
    spec: &str,
) -> anyhow::Result<Arc<xgrammar_rs::CompiledGrammar>> {
    use anyhow::Context;
    let compiled = match option {
        StructuredOutputOption::JsonSchema => {
            // Route through our schema→EBNF translator when the
            // schema uses constructs that xgrammar's CompileJSONSchema
            // mis-handles (`pattern + minLength`, strict object key
            // boundary). See `schema_to_ebnf` module docs for the
            // bug catalogue and the Phase 1/2 evidence.
            let parsed: serde_json::Value = serde_json::from_str(spec)
                .with_context(|| format!("invalid JSON schema: {spec}"))?;
            if super::schema_to_ebnf::schema_needs_ebnf_path(&parsed) {
                if let Some(ebnf) = super::schema_to_ebnf::try_schema_to_ebnf(&parsed)
                    .with_context(|| "schema_to_ebnf translator")?
                {
                    tracing::info!(
                        target: "vllm_core::xgrammar",
                        ebnf_lines = ebnf.lines().count(),
                        "JSON schema → EBNF translator path (bypassing CompileJSONSchema bugs)"
                    );
                    tracing::debug!(
                        target: "vllm_core::xgrammar",
                        "generated EBNF:\n{ebnf}"
                    );
                    compiler.compile_ebnf(&ebnf, "root").with_context(|| {
                        format!(
                            "xgrammar compile_ebnf on translated schema; \
                                 generated EBNF: {ebnf}"
                        )
                    })?
                } else {
                    tracing::info!(
                        target: "vllm_core::xgrammar",
                        "JSON schema → translator declined → falling back to CompileJSONSchema"
                    );
                    compiler
                        .compile_json_schema(spec, true, true)
                        .with_context(|| format!("xgrammar compile_json_schema for {spec}"))?
                }
            } else {
                tracing::info!(
                    target: "vllm_core::xgrammar",
                    "JSON schema → CompileJSONSchema direct (no EBNF triggers)"
                );
                compiler
                    .compile_json_schema(spec, /* any_ws = */ true, /* strict = */ true)
                    .with_context(|| format!("xgrammar compile_json_schema for {spec}"))?
            }
        }
        StructuredOutputOption::Regex => compiler
            .compile_regex(spec)
            .with_context(|| format!("xgrammar compile_regex for {spec}"))?,
        StructuredOutputOption::Grammar => compiler
            .compile_ebnf(spec, "root")
            .with_context(|| format!("xgrammar compile_ebnf for {spec}"))?,
    };
    Ok(Arc::new(compiled))
}

/// Hash a spec for cache key.
fn hash_spec(option: &StructuredOutputOption, spec: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    option.hash(&mut hasher);
    spec.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenizerWrapper;

    fn make_compiler() -> GrammarCompiler {
        let tok = TokenizerWrapper::for_testing(100);
        let vi = Arc::new(VocabularyIndex::from_tokenizer(&tok));
        GrammarCompiler::new(vi)
    }

    // The tests below exercise the compile_sync / compile pipeline,
    // which can only succeed when a structured-output backend is
    // linked in. After v1.8 (native port removed) that means
    // `feature = "xgrammar"` must be on; without it `compile_sync`
    // bails with a clear message. Gating these tests under the
    // feature keeps the default-feature build green while preserving
    // coverage where the backend exists.

    #[cfg(feature = "xgrammar")]
    #[test]
    fn compile_sync_regex() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Regex, "[a-z]+");
        assert!(result.is_ok());
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn compile_sync_grammar() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Grammar, r#"root ::= "hello""#);
        assert!(result.is_ok());
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn compile_sync_json_schema() {
        let compiler = make_compiler();
        let result =
            compiler.compile_sync(&StructuredOutputOption::JsonSchema, r#"{"type": "string"}"#);
        assert!(result.is_ok());
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn compile_sync_invalid_regex() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Regex, "[invalid");
        assert!(result.is_err());
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn compile_sync_invalid_json_schema() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::JsonSchema, "not json");
        assert!(result.is_err());
    }

    #[cfg(feature = "xgrammar")]
    #[tokio::test]
    async fn async_compile_regex() {
        let compiler = make_compiler();
        let result = compiler
            .compile(StructuredOutputOption::Regex, "[a-z]+")
            .await;
        assert!(result.is_ok());
    }

    #[cfg(feature = "xgrammar")]
    #[tokio::test]
    async fn async_compile_caches() {
        let compiler = make_compiler();

        // First compilation
        let r1 = compiler
            .compile(StructuredOutputOption::Regex, "[0-9]+")
            .await;
        assert!(r1.is_ok());

        // Second should hit cache
        let r2 = compiler
            .compile(StructuredOutputOption::Regex, "[0-9]+")
            .await;
        assert!(r2.is_ok());

        // Verify cache has one entry
        let cache = compiler.cache.lock().await;
        assert_eq!(cache.len(), 1);
    }

    #[cfg(not(feature = "xgrammar"))]
    #[test]
    fn compile_sync_bails_without_xgrammar() {
        // Document the contract: under default features the compiler
        // returns a clear actionable error, not silent success.
        // Note: `Box<dyn StructuredOutputGrammar>` is not `Debug`,
        // so we cannot use `.unwrap_err()` (which panic-prints the
        // Ok variant on failure). Pattern-match instead.
        let compiler = make_compiler();
        match compiler.compile_sync(&StructuredOutputOption::Regex, "[a-z]+") {
            Ok(_) => panic!("compile_sync must bail without xgrammar feature"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("xgrammar"),
                    "error must point at the missing feature: {msg}"
                );
            }
        }
    }

    #[test]
    fn hash_spec_different_options() {
        let h1 = hash_spec(&StructuredOutputOption::Regex, "abc");
        let h2 = hash_spec(&StructuredOutputOption::Grammar, "abc");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_spec_different_specs() {
        let h1 = hash_spec(&StructuredOutputOption::Regex, "abc");
        let h2 = hash_spec(&StructuredOutputOption::Regex, "def");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_spec_same_is_deterministic() {
        let h1 = hash_spec(&StructuredOutputOption::Regex, "test");
        let h2 = hash_spec(&StructuredOutputOption::Regex, "test");
        assert_eq!(h1, h2);
    }

    // ─── xgrammar end-to-end (Bug 10 regression coverage) ──────────
    //
    // Goes through the full vllm-rust compile dispatch under the
    // `xgrammar` feature, builds a byte-level synthetic vocab, and
    // verifies that the user's reproducer schema enforces:
    //   - `additionalProperties: false`
    //   - `minimum` / `maximum` on integers
    //   - `minLength` / `maxLength` on strings
    //   - no markdown bytes (`*`) anywhere at the start
    //
    // Mirrors `/tmp/grammar_strictness_check.py` at the API surface
    // immediately upstream of the HTTP layer.

    #[cfg(feature = "xgrammar")]
    fn byte_level_compiler() -> GrammarCompiler {
        let vocab: Vec<Vec<u8>> = (0u32..256).map(|b| vec![b as u8]).collect();
        let vi = Arc::new(VocabularyIndex::from_token_bytes(vocab));
        GrammarCompiler::new(vi)
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn xgrammar_compile_strictness_reproducer_via_vllm_rust() {
        // Exactly the schema in /tmp/grammar_strictness_check.py.
        let schema = r#"{
            "type":"object",
            "additionalProperties":false,
            "required":["foo","bar"],
            "properties":{
                "foo":{"type":"integer","minimum":1,"maximum":1000},
                "bar":{"type":"string","minLength":3,"maxLength":50}
            }
        }"#;
        let compiler = byte_level_compiler();
        let grammar = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, schema)
            .expect("xgrammar compile via vllm-rust dispatch");

        // At the very first decode step we must allow `{` and reject
        // any markdown-emphasis bytes — that's the bug repro.
        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, 256);
        grammar.fill_bitmask(&mut bm, 0);

        let allowed = |id: usize| bm.get_bit(0, id);
        assert!(allowed(b'{' as usize), "first byte: '{{' must be allowed");
        assert!(
            !allowed(b'*' as usize),
            "'*' (markdown emphasis) must be rejected at start of object"
        );
        assert!(
            !allowed(b'a' as usize),
            "letter 'a' must be rejected at start of object"
        );
        // is_terminated must be false at start
        assert!(!grammar.is_terminated());
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn xgrammar_dispatch_uses_cached_compiler_across_compiles() {
        // Two independent compiles share the same xgrammar
        // GrammarCompiler instance (and therefore its internal
        // schema-hash cache + tokenizer marshal). Verified by the
        // OnceLock seam in `GrammarCompiler::xgrammar_compiler_handle`.
        let compiler = byte_level_compiler();
        let _a = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, r#"{"type":"integer"}"#)
            .unwrap();
        let _b = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, r#"{"type":"string"}"#)
            .unwrap();
        // No assertion beyond "both compile without panic" — the
        // OnceLock contract guarantees both calls used the same
        // upstream xgrammar compiler; we cover that in the unit
        // tests of xgrammar-rs.
    }

    // ── Bug 10-v1.6 regression: nested-schema enforcement via EBNF ──
    //
    // The concept_test schema from /tmp/raw_output_probe.py exposed
    // two xgrammar CompileJSONSchema bugs that were verified in
    // xgrammar-rs/tests/nested_strictness.rs. Phase 3 routes such
    // schemas through `schema_to_ebnf` + `compile_ebnf` to bypass
    // those bugs. This test asserts the dispatch reaches the EBNF
    // path AND that the resulting matcher actually enforces the
    // failing constraints.

    #[cfg(feature = "xgrammar")]
    fn byte_level_compiler_256() -> GrammarCompiler {
        let vocab: Vec<Vec<u8>> = (0u32..256).map(|b| vec![b as u8]).collect();
        let vi = Arc::new(VocabularyIndex::from_token_bytes(vocab));
        GrammarCompiler::new(vi)
    }

    #[cfg(feature = "xgrammar")]
    const CONCEPT_SCHEMA: &str = r##"{
        "type":"object","additionalProperties":false,
        "required":["unique_features","description","year","season"],
        "properties":{
            "unique_features":{"type":"array","minItems":5,"maxItems":7,
                "items":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":8,"maxLength":80}},
            "description":{"type":"string","pattern":"^[А-Яа-яЁё0-9 ,.!?;:\\-—«»()…\\\"'\\n]+$","minLength":300,"maxLength":1600},
            "year":{"type":"integer","minimum":1,"maximum":9999},
            "season":{"type":"string","enum":["spring","summer","autumn","winter"]}
        }
    }"##;

    #[cfg(feature = "xgrammar")]
    fn min_features_array() -> Vec<u8> {
        let item = "\"привет1234\"".as_bytes();
        let mut out = Vec::new();
        out.push(b'[');
        for i in 0..5 {
            if i > 0 {
                out.push(b',');
            }
            out.extend_from_slice(item);
        }
        out.push(b']');
        out
    }

    #[cfg(feature = "xgrammar")]
    fn cyr_repeat(target_bytes: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(target_bytes + 4);
        while out.len() < target_bytes {
            out.extend_from_slice("ы".as_bytes());
        }
        out
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn xgrammar_ebnf_path_enforces_minlength_inside_string() {
        // T1 equivalent from xgrammar-rs/tests/nested_strictness.rs:
        // after the empty-description opening quote, the closing quote
        // must be REJECTED because minLength=300 is not satisfied. The
        // EBNF path enforces this; the direct CompileJSONSchema does not.
        let compiler = byte_level_compiler_256();
        let mut grammar = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, CONCEPT_SCHEMA)
            .expect("compile concept schema via EBNF dispatch");

        // Feed the prefix that lands at "after `\"description\":\"`".
        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"{\"unique_features\":");
        prefix.extend_from_slice(&min_features_array());
        prefix.extend_from_slice(b",\"description\":\"");
        let tokens: Vec<u32> = prefix.iter().map(|&b| b as u32).collect();
        assert!(
            grammar.accept_tokens(&tokens),
            "prefix should be acceptable up to the description opening quote"
        );

        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, 256);
        grammar.fill_bitmask(&mut bm, 0);
        assert!(
            !bm.get_bit(0, b'"' as usize),
            "T1 regression: closing-quote allowed at empty-description position — \
             EBNF path must enforce minLength=300"
        );
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn xgrammar_ebnf_path_enforces_key_boundary_strictness() {
        // T6 equivalent: after `\",` between description and the next
        // required key, only ASCII `"` must open the next key — byte
        // 0xE2 (start of U+201C `\u{201c}`) must be REJECTED.
        let compiler = byte_level_compiler_256();
        let mut grammar = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, CONCEPT_SCHEMA)
            .expect("compile concept schema via EBNF dispatch");

        let mut prefix = Vec::new();
        prefix.extend_from_slice(b"{\"unique_features\":");
        prefix.extend_from_slice(&min_features_array());
        prefix.extend_from_slice(b",\"description\":\"");
        prefix.extend_from_slice(&cyr_repeat(700));
        prefix.extend_from_slice(b"\",");
        let tokens: Vec<u32> = prefix.iter().map(|&b| b as u32).collect();
        assert!(
            grammar.accept_tokens(&tokens),
            "valid prefix up to key-boundary should be accepted"
        );

        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, 256);
        grammar.fill_bitmask(&mut bm, 0);
        assert!(
            bm.get_bit(0, b'"' as usize),
            "sanity: ASCII `\"` must still be allowed at key-start"
        );
        assert!(
            !bm.get_bit(0, 0xE2),
            "T6 regression: 0xE2 (typographic-quote start) allowed at key-boundary — \
             EBNF path must enforce strict key alternation"
        );
    }

    // ── A1 diagnosis: greedy colon-drop, real Qwen3 vocab ──
    //
    // Loads the actual Qwen3-8B tokenizer (skips if absent), compiles a
    // minimal integer-object schema through the production EBNF path,
    // walks the matcher to the position right after `{"foo"`, and
    // decodes the allowed-token set. The question: does the matcher
    // permit any token that would SKIP the mandatory `:` (e.g. a token
    // whose first byte is a digit)? Run with `--nocapture`.

    #[cfg(feature = "xgrammar")]
    fn find_qwen3_tokenizer() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let base = std::path::Path::new(&home)
            .join(".cache/huggingface/hub/models--turboderp--Qwen3-8B-exl3/snapshots");
        let snap = std::fs::read_dir(&base).ok()?.flatten().next()?;
        let p = snap.path().join("tokenizer.json");
        p.exists().then_some(p)
    }

    #[cfg(feature = "xgrammar")]
    #[test]
    fn diag_greedy_colon_allowed_set_real_qwen3() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP diag_greedy_colon: Qwen3 tokenizer not in HF cache");
            return;
        };
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load qwen3 tokenizer");
        let vi = Arc::new(VocabularyIndex::from_tokenizer(&tok));
        let vocab_size = vi.vocab_size();
        let compiler = GrammarCompiler::new(vi.clone());

        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo"],"properties":{"foo":{"type":"integer"}}}"#;
        let mut grammar = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, schema)
            .expect("compile schema");

        let prefix_ids = tok.encode("{\"foo\"").expect("encode prefix");
        eprintln!("prefix tokens = {prefix_ids:?}");
        for &id in &prefix_ids {
            eprintln!(
                "  tok {id} = {:?}",
                String::from_utf8_lossy(vi.token_bytes(id))
            );
        }
        assert!(
            grammar.accept_tokens(&prefix_ids),
            "prefix `{{\"foo\"` must be accepted by the matcher"
        );

        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, vocab_size);
        grammar.fill_bitmask(&mut bm, 0);

        let mut colon = vec![];
        let mut ws = vec![];
        let mut digit = vec![];
        let mut brace = vec![];
        let mut other = 0usize;
        let mut total = 0usize;
        for id in 0..vocab_size {
            if !bm.get_bit(0, id) {
                continue;
            }
            total += 1;
            let b = vi.token_bytes(id as u32);
            match b.first() {
                Some(b':') => colon.push(id),
                Some(b' ' | b'\n' | b'\r' | b'\t') => ws.push(id),
                Some(b'0'..=b'9') | Some(b'-') => digit.push(id),
                Some(b'}') => brace.push(id),
                _ => other += 1,
            }
        }
        eprintln!(
            "total_allowed={total} colon={} ws={} digit/minus={} brace={} other={}",
            colon.len(),
            ws.len(),
            digit.len(),
            brace.len(),
            other
        );
        let show = |label: &str, ids: &[usize]| {
            eprintln!("--- {label} ({}) ---", ids.len());
            for &id in ids.iter().take(25) {
                eprintln!(
                    "  {id} = {:?}",
                    String::from_utf8_lossy(vi.token_bytes(id as u32))
                );
            }
        };
        show("colon-first", &colon);
        show("digit/minus-first (COLON-SKIP if non-empty!)", &digit);
        show("ws-first", &ws);

        // Natural-flow probe: after `{"foo` (closing key-quote NOT yet
        // emitted — id 1 dropped). Real greedy decode reaches here and
        // the model's preferred next token is the fused `"key-close +
        // colon` (e.g. `":`). Confirm such fused tokens are allowed AND
        // that no value-starting digit is reachable without a colon.
        let mut g2 = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, schema)
            .expect("compile schema 2");
        assert!(
            g2.accept_tokens(&prefix_ids[..prefix_ids.len() - 1]),
            "prefix `{{\"foo` (no close quote) must be accepted"
        );
        let mut bm2 = crate::sampling::grammar::PackedBitmask::new(1, vocab_size);
        g2.fill_bitmask(&mut bm2, 0);
        let mut fused_colon = vec![];
        let mut digit2 = 0usize;
        for id in 0..vocab_size {
            if !bm2.get_bit(0, id) {
                continue;
            }
            let b = vi.token_bytes(id as u32);
            if b.contains(&b':') {
                fused_colon.push(id);
            }
            if matches!(b.first(), Some(b'0'..=b'9')) {
                digit2 += 1;
            }
        }
        eprintln!(
            "[after {{\"foo] tokens containing ':' = {}, digit-first = {digit2}",
            fused_colon.len()
        );
        show(
            "tokens-containing-colon (carry the mandatory ':')",
            &fused_colon,
        );
    }

    // ── A2 root-cause proof: stop_token_ids drive EOS at the accepting
    // state. Walks a full `{"foo": 42}` and checks whether the matcher
    // allows the model EOS (151645) afterward. With empty stop_token_ids
    // (today's `GrammarCompiler` hardcode) the allowed set is EMPTY → the
    // bitmask zeroes every logit → greedy argmax picks token 0 (`!`),
    // exactly the trailing-garbage symptom. With EOS configured, only the
    // EOS bit is set → greedy stops cleanly.

    #[cfg(feature = "xgrammar")]
    #[test]
    fn diag_eos_termination_needs_stop_token_ids() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP diag_eos_termination: Qwen3 tokenizer not in HF cache");
            return;
        };
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load qwen3 tokenizer");
        let vi = VocabularyIndex::from_tokenizer(&tok);
        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo"],"properties":{"foo":{"type":"integer"}}}"#;
        let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
        let ebnf = super::super::schema_to_ebnf::try_schema_to_ebnf(&parsed)
            .unwrap()
            .expect("schema translates to EBNF");
        let full = tok.encode("{\"foo\": 42}").expect("encode full json");
        const QWEN3_EOS: usize = 151645;

        let probe = |stop: &[u32]| -> (bool, bool, Vec<usize>) {
            let compiler =
                super::super::xgrammar_backend::make_xgrammar_compiler(&vi, stop).unwrap();
            let g = std::sync::Arc::new(compiler.compile_ebnf(&ebnf, "root").unwrap());
            let mut m = g.new_matcher(64).unwrap();
            for &t in &full {
                assert!(m.accept_token(t).unwrap(), "accept {t} in full json");
            }
            let terminated = m.is_terminated();
            let mut row = vec![0i32; vi.vocab_size().div_ceil(32)];
            let _ = m.fill_next_token_bitmask(&mut row).unwrap();
            let allowed: Vec<usize> = (0..vi.vocab_size())
                .filter(|&id| (row[id / 32] >> (id % 32)) & 1 != 0)
                .collect();
            let eos_allowed = (row[QWEN3_EOS / 32] >> (QWEN3_EOS % 32)) & 1 != 0;
            (eos_allowed, terminated, allowed)
        };

        let (eos_empty, term_empty, allowed_empty) = probe(&[]);
        let (eos_set, term_set, allowed_set) = probe(&[QWEN3_EOS as u32]);
        eprintln!(
            "stop=[]      : eos_allowed={eos_empty} terminated={term_empty} allowed={allowed_empty:?}"
        );
        for &id in &allowed_empty {
            eprintln!(
                "   stop=[] allowed {id} = {:?}",
                String::from_utf8_lossy(vi.token_bytes(id as u32))
            );
        }
        eprintln!(
            "stop=[151645]: eos_allowed={eos_set} terminated={term_set} allowed={allowed_set:?}"
        );

        // Root cause: with no stop token configured, the model EOS is NOT
        // in the allowed set at the accepting state — greedy cannot emit
        // it, so it picks whatever survives the bitmask (or token 0 when
        // the row collapses to empty after the next step), producing the
        // observed trailing garbage.
        assert!(
            !eos_empty,
            "ROOT CAUSE: empty stop_token_ids must leave EOS forbidden after `}}`"
        );
        // Fix: configuring EOS makes it allowed at the accepting state.
        assert!(
            eos_set,
            "FIX: EOS in stop_token_ids must make EOS allowed at the accepting state"
        );
    }

    // Production-wiring guard: `GrammarCompiler::with_stop_tokens` must
    // thread the EOS into the matcher so that after a complete document
    // the *only* allowed token is the model EOS — greedy then emits it
    // and the engine stops cleanly. Reverting the server to
    // `GrammarCompiler::new` (empty stop tokens) makes this fail.

    #[cfg(feature = "xgrammar")]
    #[test]
    fn with_stop_tokens_terminates_on_model_eos() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP with_stop_tokens_terminates_on_model_eos: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load qwen3 tokenizer");
        let vi = Arc::new(VocabularyIndex::from_tokenizer(&tok));
        let vocab_size = vi.vocab_size();
        let compiler = GrammarCompiler::with_stop_tokens(vi.clone(), vec![QWEN3_EOS]);
        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo"],"properties":{"foo":{"type":"integer"}}}"#;
        let mut grammar = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, schema)
            .expect("compile schema");
        let full = tok.encode("{\"foo\": 42}").expect("encode");
        assert!(grammar.accept_tokens(&full), "full json accepted");

        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, vocab_size);
        grammar.fill_bitmask(&mut bm, 0);
        assert!(
            bm.get_bit(0, QWEN3_EOS as usize),
            "model EOS must be allowed at the accepting state"
        );
        let allowed = (0..vocab_size).filter(|&id| bm.get_bit(0, id)).count();
        assert_eq!(
            allowed, 1,
            "after a complete document only the EOS may be allowed (got {allowed})"
        );
    }

    // B1 decision input: per-call cost of accept_token vs fill on a real
    // Qwen3 grammar. The engine does both once per constrained sequence
    // per step; the BatchMatcher already parallelises fill. This measures
    // whether sequential accept_token is worth a BatchAcceptToken FFI.
    // Run with `--nocapture --ignored` (ignored: timing, not an invariant).

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_accept_vs_fill_cost() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP bench_accept_vs_fill_cost: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load tokenizer");
        let vi = Arc::new(VocabularyIndex::from_tokenizer(&tok));
        let vocab_size = vi.vocab_size();
        let compiler = GrammarCompiler::with_stop_tokens(vi.clone(), vec![QWEN3_EOS]);
        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo","bar"],"properties":{"foo":{"type":"integer"},"bar":{"type":"string"}}}"#;

        let mut bm = crate::sampling::grammar::PackedBitmask::new(1, vocab_size);
        let doc = tok.encode("{\"foo\": 42, \"bar\": \"apple\"}").unwrap();

        // WARM fill: same (initial) state repeated — measures the cached
        // adaptive-token-mask path. NOT representative of decode.
        let g = compiler
            .compile_sync(&StructuredOutputOption::JsonSchema, schema)
            .unwrap();
        let warm_iters = 500u32;
        let tw = std::time::Instant::now();
        for _ in 0..warm_iters {
            g.fill_bitmask(&mut bm, 0);
        }
        let fill_warm = tw.elapsed() / warm_iters;

        // COLD fill: a fresh state at every step (the real decode loop).
        // accept one token, then fill — xgrammar must compute the
        // adaptive mask for a state it hasn't cached yet. We time only
        // the fills and report per-step.
        let passes = 60u32;
        let mut cold_total = std::time::Duration::ZERO;
        let mut fills = 0u32;
        for _ in 0..passes {
            let mut g = compiler
                .compile_sync(&StructuredOutputOption::JsonSchema, schema)
                .unwrap();
            for &t in &doc {
                let tf = std::time::Instant::now();
                g.fill_bitmask(&mut bm, 0);
                cold_total += tf.elapsed();
                fills += 1;
                let _ = g.accept_tokens(&[t]);
            }
        }
        let fill_cold = cold_total / fills;

        // accept_token per token (fresh matcher each pass).
        let ta = std::time::Instant::now();
        let mut tok_count = 0u32;
        for _ in 0..passes {
            let mut g = compiler
                .compile_sync(&StructuredOutputOption::JsonSchema, schema)
                .unwrap();
            for &t in &doc {
                let _ = g.accept_tokens(&[t]);
                tok_count += 1;
            }
        }
        let accept_per_tok = ta.elapsed() / tok_count;

        eprintln!("vocab_size={vocab_size}");
        eprintln!("fill WARM (cached state)  : {fill_warm:?}/call");
        eprintln!("fill COLD (new state/step): {fill_cold:?}/call  <-- real decode");
        eprintln!("accept_token (per token)  : {accept_per_tok:?}/call");
        let ratio = fill_cold.as_secs_f64() / accept_per_tok.as_secs_f64().max(1e-12);
        eprintln!("cold-fill / accept ratio  : {ratio:.1}x");
    }

    // Per-request construction cost. The server builds a *fresh*
    // GrammarCompiler per constrained request (chat.rs), which re-decodes
    // the whole vocab into a VocabularyIndex and re-marshals the 151k-token
    // xgrammar TokenizerInfo every time. This measures the TTFT tax that a
    // shared/cached compiler would remove.

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_per_request_constraint_construction() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = Arc::new(TokenizerWrapper::from_file(&tok_path).expect("load tokenizer"));
        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo"],"properties":{"foo":{"type":"integer"}}}"#;

        let n = 5u32;
        // (1) VocabularyIndex::from_tokenizer (per-token decode of 151k).
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let _vi = VocabularyIndex::from_tokenizer(&tok);
        }
        let vi_cost = t0.elapsed() / n;

        // (2) Full per-request path as the server does it today:
        //     new VocabularyIndex + new GrammarCompiler + compile_sync
        //     (first compile marshals TokenizerInfo + builds grammar).
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            let vi = Arc::new(VocabularyIndex::from_tokenizer(&tok));
            let compiler = GrammarCompiler::with_stop_tokens(vi, vec![QWEN3_EOS]);
            let _g = compiler
                .compile_sync(&StructuredOutputOption::JsonSchema, schema)
                .unwrap();
        }
        let full_cost = t1.elapsed() / n;

        eprintln!("VocabularyIndex::from_tokenizer : {vi_cost:?}/request");
        eprintln!("full per-request construction   : {full_cost:?}/request");
        eprintln!(
            "  (marshal+compile alone)       : {:?}/request",
            full_cost.saturating_sub(vi_cost)
        );
    }

    // Decides whether caching CompiledGrammar across same-schema requests
    // would warm the decode fill. If xgrammar's adaptive token-mask cache
    // lives in CompiledGrammar (shared by Arc), a second matcher built
    // from the SAME compiled grammar should see warm (~sub-µs) fills at
    // states the first matcher already visited. If it's per-matcher,
    // matcher #2 stays cold (~ms) and the caching lever is worthless.

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_compiled_grammar_cache_warms_fill() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load tokenizer");
        let vi = VocabularyIndex::from_tokenizer(&tok);
        let vocab_size = vi.vocab_size();
        let wpr = vocab_size.div_ceil(32);
        let schema = r#"{"type":"object","additionalProperties":false,"required":["foo","bar"],"properties":{"foo":{"type":"integer"},"bar":{"type":"string"}}}"#;
        let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
        let ebnf = super::super::schema_to_ebnf::try_schema_to_ebnf(&parsed)
            .unwrap()
            .unwrap();
        let doc = tok.encode("{\"foo\": 42, \"bar\": \"apple\"}").unwrap();

        let walk_fill_cost =
            |g: &std::sync::Arc<xgrammar_rs::CompiledGrammar>| -> std::time::Duration {
                let mut m = g.new_matcher(64).unwrap();
                let mut row = vec![0i32; wpr];
                let mut total = std::time::Duration::ZERO;
                for &t in &doc {
                    let s = std::time::Instant::now();
                    let _ = m.fill_next_token_bitmask(&mut row);
                    total += s.elapsed();
                    let _ = m.accept_token(t);
                }
                total / doc.len() as u32
            };

        let compiler =
            super::super::xgrammar_backend::make_xgrammar_compiler(&vi, &[QWEN3_EOS]).unwrap();

        // SHARED CompiledGrammar: matcher #1 (cold), then matcher #2 from
        // the SAME Arc — fills the same states matcher #1 already touched.
        let g_shared = std::sync::Arc::new(compiler.compile_ebnf(&ebnf, "root").unwrap());
        let m1 = walk_fill_cost(&g_shared);
        let m2 = walk_fill_cost(&g_shared);

        // FRESH CompiledGrammar (what compile_sync does per request).
        let g_fresh = std::sync::Arc::new(compiler.compile_ebnf(&ebnf, "root").unwrap());
        let m_fresh = walk_fill_cost(&g_fresh);

        eprintln!("matcher#1 on shared grammar (cold) : {m1:?}/fill");
        eprintln!("matcher#2 on shared grammar (warm?): {m2:?}/fill");
        eprintln!("matcher on FRESH grammar (cold)    : {m_fresh:?}/fill");
        let speedup = m1.as_secs_f64() / m2.as_secs_f64().max(1e-12);
        eprintln!("shared-reuse warm speedup (m1/m2)  : {speedup:.1}x");
    }

    // Decisive: is xgrammar's per-fill cost amortised when the matcher
    // sits in a SINGLE recurring grammar state (unbounded string content)?
    // Upstream xgrammar caches the adaptive token mask per grammar
    // stack-state, so filling step 2..N inside a string should be cheap
    // (only step 1 computes the mask). If every step stays ~ms, the cache
    // is NOT hitting on a recurring state → our compile path defeats it,
    // which is a fixable ~1000x integration slowdown (would obviate the
    // multi-week GPU fill port).

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_fill_cost_in_recurring_string_state() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load tokenizer");
        let vi = VocabularyIndex::from_tokenizer(&tok);
        let wpr = vi.vocab_size().div_ceil(32);
        // String value with no length bound → unbounded char-class star:
        // ONE recurring grammar state for every content character.
        let schema = r#"{"type":"object","additionalProperties":false,"required":["bar"],"properties":{"bar":{"type":"string"}}}"#;
        let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
        let ebnf = super::super::schema_to_ebnf::try_schema_to_ebnf(&parsed)
            .unwrap()
            .unwrap();
        eprintln!("EBNF:\n{ebnf}");
        let compiler =
            super::super::xgrammar_backend::make_xgrammar_compiler(&vi, &[QWEN3_EOS]).unwrap();
        let g = std::sync::Arc::new(compiler.compile_ebnf(&ebnf, "root").unwrap());
        let mut m = g.new_matcher(64).unwrap();

        // Drive into the string body: `{"bar": "`
        for &t in &tok.encode("{\"bar\": \"").unwrap() {
            assert!(m.accept_token(t).unwrap(), "prefix into string body");
        }
        // Now every accepted ASCII letter keeps us in the SAME string
        // content state. Time fill at each of 30 successive content chars.
        let mut row = vec![0i32; wpr];
        let letter = tok.encode("a").unwrap();
        let lt = *letter.first().unwrap();
        let mut per_step = Vec::new();
        for _ in 0..30 {
            let s = std::time::Instant::now();
            let _ = m.fill_next_token_bitmask(&mut row);
            per_step.push(s.elapsed());
            assert!(m.accept_token(lt).unwrap(), "accept content letter");
        }
        for (i, d) in per_step.iter().enumerate() {
            eprintln!("string-content fill step {i:2}: {d:?}");
        }
        let first = per_step[0];
        let rest_avg =
            per_step[1..].iter().sum::<std::time::Duration>() / (per_step.len() - 1) as u32;
        eprintln!("first fill: {first:?}, steps 1.. avg: {rest_avg:?}");
        eprintln!(
            "amortised? first/rest ratio = {:.1}x",
            first.as_secs_f64() / rest_avg.as_secs_f64().max(1e-12)
        );
    }

    // A/B: is the ~47ms string-content fill caused by OUR EBNF translator
    // or inherent to xgrammar? Compile the SAME schema via xgrammar's
    // native `compile_json_schema` and measure fill in the string state.
    // If native is fast, our `strchar_0* ` EBNF is the culprit (cheap fix:
    // route simple strings to native / emit a better string rule). If
    // native is also ~47ms, it's xgrammar-inherent (justifies GPU fill).

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_fill_native_jsonschema_vs_ebnf_string() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load tokenizer");
        let vi = VocabularyIndex::from_tokenizer(&tok);
        let wpr = vi.vocab_size().div_ceil(32);
        let schema = r#"{"type":"object","additionalProperties":false,"required":["bar"],"properties":{"bar":{"type":"string"}}}"#;
        let compiler =
            super::super::xgrammar_backend::make_xgrammar_compiler(&vi, &[QWEN3_EOS]).unwrap();

        let measure = |g: std::sync::Arc<xgrammar_rs::CompiledGrammar>| -> std::time::Duration {
            let mut m = g.new_matcher(64).unwrap();
            for &t in &tok.encode("{\"bar\": \"").unwrap() {
                if !m.accept_token(t).unwrap() {
                    return std::time::Duration::ZERO; // prefix rejected (grammar shape differs)
                }
            }
            let mut row = vec![0i32; wpr];
            let lt = *tok.encode("a").unwrap().first().unwrap();
            let mut tot = std::time::Duration::ZERO;
            let n = 15;
            for _ in 0..n {
                let s = std::time::Instant::now();
                let _ = m.fill_next_token_bitmask(&mut row);
                tot += s.elapsed();
                let _ = m.accept_token(lt);
            }
            tot / n
        };

        // Native xgrammar JSON-Schema path.
        let g_native =
            std::sync::Arc::new(compiler.compile_json_schema(schema, true, true).unwrap());
        let native = measure(g_native);

        // Our EBNF translator path.
        let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
        let ebnf = super::super::schema_to_ebnf::try_schema_to_ebnf(&parsed)
            .unwrap()
            .unwrap();
        let g_ebnf = std::sync::Arc::new(compiler.compile_ebnf(&ebnf, "root").unwrap());
        let ours = measure(g_ebnf);

        eprintln!("native compile_json_schema string-fill : {native:?}/step");
        eprintln!("our EBNF translator     string-fill     : {ours:?}/step");
    }

    // Find a string-rule EBNF formulation xgrammar fills fast. Hypothesis:
    // a rule-REFERENCE inside the content star (`strchar* ` where
    // `strchar ::= class | json_escape`) defeats xgrammar's FSM fast-path,
    // forcing an O(vocab) recompute each step; inlining the alternatives
    // as a single regex-like body should stay an FSM (µs fills).

    #[cfg(feature = "xgrammar")]
    #[test]
    #[ignore = "timing measurement, run manually with --nocapture"]
    fn bench_string_rule_variants() {
        let Some(tok_path) = find_qwen3_tokenizer() else {
            eprintln!("SKIP: tokenizer absent");
            return;
        };
        const QWEN3_EOS: u32 = 151645;
        let tok = TokenizerWrapper::from_file(&tok_path).expect("load tokenizer");
        let vi = VocabularyIndex::from_tokenizer(&tok);
        let wpr = vi.vocab_size().div_ceil(32);
        let compiler =
            super::super::xgrammar_backend::make_xgrammar_compiler(&vi, &[QWEN3_EOS]).unwrap();

        // Each variant is a full grammar whose root is a JSON string.
        let variants: Vec<(&str, String)> = vec![
            (
                "V0 current (rule-ref escape)",
                "root ::= str\n\
                 json_escape ::= (\"\\\\\\\"\" | \"\\\\\\\\\" | \"\\\\/\" | \"\\\\b\" | \"\\\\f\" | \"\\\\n\" | \"\\\\r\" | \"\\\\t\" | \"\\\\u\" [0-9A-Fa-f]{4})\n\
                 strchar ::= [^\"\\\\\\x00-\\x1f] | json_escape\n\
                 str ::= \"\\\"\" strchar* \"\\\"\"\n".to_string(),
            ),
            (
                "V1 pure negated class (no escapes)",
                "root ::= str\n\
                 str ::= \"\\\"\" [^\"\\\\\\x00-\\x1f]* \"\\\"\"\n".to_string(),
            ),
            (
                "V2 inline escape, no \\u",
                "root ::= str\n\
                 str ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" [\"\\\\/bfnrt])* \"\\\"\"\n".to_string(),
            ),
            (
                "V3 inline escape + \\u",
                "root ::= str\n\
                 str ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9A-Fa-f]{4}))* \"\\\"\"\n".to_string(),
            ),
            (
                "V4 explicit 4 hex (no {4})",
                "root ::= str\n\
                 str ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f]))* \"\\\"\"\n".to_string(),
            ),
            (
                "V5 escape via rule, hex explicit",
                "root ::= str\n\
                 esc ::= [\"\\\\/bfnrt] | \"u\" [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f]\n\
                 str ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" esc)* \"\\\"\"\n".to_string(),
            ),
            (
                "B-pure-class inline {5,40}",
                "root ::= str\n\
                 str ::= \"\\\"\" [^\"\\\\\\x00-\\x1f]{5,40} \"\\\"\"\n".to_string(),
            ),
            (
                "B-rule-ref {5,40}",
                "root ::= str\n\
                 sc ::= [^\"\\\\\\x00-\\x1f] | \"\\\\\" [\"\\\\/bfnrt]\n\
                 str ::= \"\\\"\" sc{5,40} \"\\\"\"\n".to_string(),
            ),
            (
                "B-alternation inline {5,40}",
                "root ::= str\n\
                 str ::= \"\\\"\" ([^\"\\\\\\x00-\\x1f] | \"\\\\\" [\"\\\\/bfnrt]){5,40} \"\\\"\"\n".to_string(),
            ),
        ];

        let lt = *tok.encode("a").unwrap().first().unwrap();
        let quote = *tok.encode("\"").unwrap().first().unwrap();
        for (name, ebnf) in &variants {
            let g = match compiler.compile_ebnf(ebnf, "root") {
                Ok(g) => std::sync::Arc::new(g),
                Err(e) => {
                    eprintln!("{name}: COMPILE ERR {e}");
                    continue;
                }
            };
            let mut m = g.new_matcher(64).unwrap();
            if !m.accept_token(quote).unwrap() {
                eprintln!("{name}: opening quote rejected");
                continue;
            }
            let mut row = vec![0i32; wpr];
            let n = 15;
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = m.fill_next_token_bitmask(&mut row);
                let _ = m.accept_token(lt);
            }
            eprintln!("{name}: {:?}/fill", t.elapsed() / n);
        }
    }
}
