//! Async grammar compilation with caching.
//!
//! `GrammarCompiler` compiles structured output specs (regex, JSON schema,
//! GBNF grammar) into `StructuredOutputGrammar` instances. DFA compilation
//! is offloaded to blocking threads via `tokio::task::spawn_blocking`.
//! Compiled templates are cached by content hash.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use super::ebnf_backend::ebnf_to_grammar;
use super::json_schema::json_schema_to_grammar;
use super::regex_backend::RegexDfaGrammar;
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
pub struct GrammarCompiler {
    vocab_index: Arc<VocabularyIndex>,
    cache: Arc<Mutex<HashMap<u64, Arc<CompiledTemplate>>>>,
}

impl GrammarCompiler {
    /// Create a new compiler with the given vocabulary index.
    pub fn new(vocab_index: Arc<VocabularyIndex>) -> Self {
        Self {
            vocab_index,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Compile a structured output specification into a grammar.
    ///
    /// Uses `spawn_blocking` for CPU-heavy DFA compilation.
    /// Results are cached by content hash.
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

        // Compile on a blocking thread to validate the spec
        let vocab_index = self.vocab_index.clone();
        let option_clone = option.clone();
        let spec_owned = spec.to_string();

        // Initial compile validates the spec (errors propagated to caller)
        tokio::task::spawn_blocking(move || compile_sync(&option_clone, &spec_owned, &vocab_index))
            .await
            .map_err(|e| anyhow::anyhow!("compilation task panicked: {e}"))??;

        // Factory re-compiles from cached spec (DFA compilation is the
        // expensive part, but the factory ensures each request gets fresh
        // mutable state)
        let vocab_for_factory = self.vocab_index.clone();
        let option_for_factory = option.clone();
        let spec_for_factory = spec.to_string();

        let factory: Arc<dyn Fn() -> Box<dyn StructuredOutputGrammar> + Send + Sync> =
            Arc::new(move || {
                compile_sync(&option_for_factory, &spec_for_factory, &vocab_for_factory)
                    .expect("cached compilation should not fail")
            });

        let template = Arc::new(CompiledTemplate { factory });

        // Store in cache
        {
            let mut cache = self.cache.lock().await;
            cache.insert(spec_hash, template.clone());
        }

        Ok((template.factory)())
    }

    /// Synchronous compilation (for use in non-async contexts).
    pub fn compile_sync(
        &self,
        option: &StructuredOutputOption,
        spec: &str,
    ) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
        compile_sync(option, spec, &self.vocab_index)
    }
}

/// Synchronous compilation dispatcher.
fn compile_sync(
    option: &StructuredOutputOption,
    spec: &str,
    vocab_index: &Arc<VocabularyIndex>,
) -> anyhow::Result<Box<dyn StructuredOutputGrammar>> {
    match option {
        StructuredOutputOption::Regex => {
            let grammar = RegexDfaGrammar::new(spec, vocab_index.clone())?;
            Ok(Box::new(grammar))
        }
        StructuredOutputOption::JsonSchema => {
            let schema: serde_json::Value = serde_json::from_str(spec)
                .map_err(|e| anyhow::anyhow!("invalid JSON schema: {e}"))?;
            json_schema_to_grammar(&schema, vocab_index.clone(), None)
        }
        StructuredOutputOption::Grammar => ebnf_to_grammar(spec, vocab_index.clone()),
    }
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

    #[test]
    fn compile_sync_regex() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Regex, "[a-z]+");
        assert!(result.is_ok());
    }

    #[test]
    fn compile_sync_grammar() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Grammar, r#"root ::= "hello""#);
        assert!(result.is_ok());
    }

    #[test]
    fn compile_sync_json_schema() {
        let compiler = make_compiler();
        let result =
            compiler.compile_sync(&StructuredOutputOption::JsonSchema, r#"{"type": "string"}"#);
        assert!(result.is_ok());
    }

    #[test]
    fn compile_sync_invalid_regex() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::Regex, "[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn compile_sync_invalid_json_schema() {
        let compiler = make_compiler();
        let result = compiler.compile_sync(&StructuredOutputOption::JsonSchema, "not json");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_compile_regex() {
        let compiler = make_compiler();
        let result = compiler
            .compile(StructuredOutputOption::Regex, "[a-z]+")
            .await;
        assert!(result.is_ok());
    }

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
}
