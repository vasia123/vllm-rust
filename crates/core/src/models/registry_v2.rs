//! Architecture registry — `phf::Map<arch_name, &dyn ArchFactory>`.
//!
//! Phase 1 lands the skeleton with **zero entries**: the legacy match-
//! arm dispatch in `mod.rs` keeps running. Phase 2 populates the map
//! and switches `from_config*` to thin shims over `lookup(arch)`.
//!
//! Why `phf` and not a runtime `HashMap`? Compile-time perfect hash
//! gives O(1) lookup with no startup cost, catches duplicate
//! arch_names at compile time (intended invariant — two factories
//! claiming the same name is a bug), and produces a static map
//! shareable across threads without any synchronisation.
//!
//! Behind `--features model-registry-v2` so the legacy code path
//! remains the live one until Phase 2 cuts over. The cfg gate lives
//! on the `pub mod registry_v2;` declaration in `mod.rs`; we don't
//! repeat it here.

use phf::phf_map;

use super::factory::ArchFactory;

/// Compile-time map from HuggingFace `architectures[]` strings to
/// `&'static dyn ArchFactory`. Phase 2 will populate this with ~560
/// entries (one per arch_name, including aliases — multiple keys
/// pointing at the same `&FOO_FACTORY` for legacy alias support).
///
/// Phase 1 keeps it empty so `cargo build --features
/// model-registry-v2` succeeds without forcing every factory file to
/// exist yet.
pub static REGISTRY: phf::Map<&'static str, &'static dyn ArchFactory> = phf_map! {};

/// Resolve a HuggingFace architecture name to its registered factory.
///
/// Returns `None` if the name is not registered. The dispatch shim in
/// `mod.rs` will translate that to `ModelError::UnsupportedArchitecture`.
pub fn lookup(arch: &str) -> Option<&'static dyn ArchFactory> {
    REGISTRY.get(arch).copied()
}

/// Iterator over `(arch_name, factory)` pairs — used by
/// `tests/registry_completeness.rs` (Phase 2) to assert every arch
/// from the old match-arm dispatch is still routable here.
pub fn entries() -> impl Iterator<Item = (&'static str, &'static dyn ArchFactory)> {
    REGISTRY.entries().map(|(k, v)| (*k, *v))
}

/// Total registered arch_names. Phase 2 will assert this matches the
/// expected count (~560 after legacy aliases) in a registry-completeness
/// integration test.
pub fn len() -> usize {
    REGISTRY.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_compiles_and_is_empty() {
        // Phase 1 invariant: registry is empty until Phase 2.
        assert_eq!(len(), 0);
        assert!(lookup("LlamaForCausalLM").is_none());
        assert_eq!(entries().count(), 0);
    }
}
