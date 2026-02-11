//! Prompt adapter manager for runtime adapter loading and caching.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device};
use thiserror::Error;

use super::loader::{PromptAdapterLoadError, PromptAdapterLoader};
use super::types::{PromptAdapter, PromptAdapterRequest};

/// Errors from the prompt adapter manager.
#[derive(Debug, Error)]
pub enum PromptAdapterManagerError {
    #[error("adapter not found: {0}")]
    AdapterNotFound(String),
    #[error("adapter ID already in use: {0}")]
    IdConflict(u32),
    #[error("adapter name already in use: {0}")]
    NameConflict(String),
    #[error("max adapters limit reached: {0}")]
    MaxAdaptersReached(usize),
    #[error("load error: {0}")]
    LoadError(#[from] PromptAdapterLoadError),
}

/// Configuration for the prompt adapter manager.
#[derive(Debug, Clone)]
pub struct PromptAdapterManagerConfig {
    /// Maximum number of adapters to keep loaded.
    pub max_adapters: usize,
    /// Maximum virtual tokens per adapter (for pre-allocation).
    pub max_virtual_tokens: usize,
}

impl Default for PromptAdapterManagerConfig {
    fn default() -> Self {
        Self {
            max_adapters: 8,
            max_virtual_tokens: 128,
        }
    }
}

/// Manages prompt adapters at runtime.
///
/// Provides adapter loading, caching, per-request lookup, and LRU eviction.
pub struct PromptAdapterManager {
    /// Loaded adapters by name.
    adapters: HashMap<String, Arc<PromptAdapter>>,
    /// Map from ID to name for fast lookup.
    id_to_name: HashMap<u32, String>,
    /// LRU generation per adapter name. Higher = more recently used.
    lru_generation: HashMap<String, u64>,
    /// Monotonically increasing generation counter.
    generation_counter: u64,
    /// Configuration.
    config: PromptAdapterManagerConfig,
    /// Loader instance.
    loader: PromptAdapterLoader,
    /// Next available adapter ID.
    next_id: u32,
}

impl PromptAdapterManager {
    pub fn new(device: Device, dtype: DType, config: PromptAdapterManagerConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            id_to_name: HashMap::new(),
            lru_generation: HashMap::new(),
            generation_counter: 0,
            config,
            loader: PromptAdapterLoader::new(device, dtype),
            next_id: 1,
        }
    }

    pub fn with_defaults(device: Device, dtype: DType) -> Self {
        Self::new(device, dtype, PromptAdapterManagerConfig::default())
    }

    /// Load an adapter from a directory path.
    ///
    /// Returns the adapter ID. If already loaded, returns its existing ID.
    pub fn load_adapter(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<u32, PromptAdapterManagerError> {
        let name = name.into();

        if let Some(adapter) = self.adapters.get(&name) {
            let id = adapter.id;
            self.touch(&name);
            return Ok(id);
        }

        if self.adapters.len() >= self.config.max_adapters {
            self.evict_lru()?;
        }

        let id = self.next_id;
        self.next_id += 1;

        let adapter = self.loader.load(path, &name, id)?;

        if adapter.num_virtual_tokens > self.config.max_virtual_tokens {
            return Err(PromptAdapterManagerError::LoadError(
                PromptAdapterLoadError::Config(format!(
                    "adapter has {} virtual tokens, max is {}",
                    adapter.num_virtual_tokens, self.config.max_virtual_tokens
                )),
            ));
        }

        self.adapters.insert(name.clone(), Arc::new(adapter));
        self.id_to_name.insert(id, name.clone());
        self.touch(&name);

        Ok(id)
    }

    /// Load an adapter with a specific ID.
    pub fn load_adapter_with_id(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
        id: u32,
    ) -> Result<(), PromptAdapterManagerError> {
        let name = name.into();

        if self.id_to_name.contains_key(&id) {
            return Err(PromptAdapterManagerError::IdConflict(id));
        }
        if self.adapters.contains_key(&name) {
            return Err(PromptAdapterManagerError::NameConflict(name));
        }

        if self.adapters.len() >= self.config.max_adapters {
            self.evict_lru()?;
        }

        let adapter = self.loader.load(path, &name, id)?;

        self.adapters.insert(name.clone(), Arc::new(adapter));
        self.id_to_name.insert(id, name.clone());
        self.touch(&name);

        if id >= self.next_id {
            self.next_id = id + 1;
        }

        Ok(())
    }

    /// Get an adapter by name.
    pub fn get(&self, name: &str) -> Option<Arc<PromptAdapter>> {
        self.adapters.get(name).cloned()
    }

    /// Get an adapter by ID.
    pub fn get_by_id(&self, id: u32) -> Option<Arc<PromptAdapter>> {
        self.id_to_name
            .get(&id)
            .and_then(|name| self.adapters.get(name))
            .cloned()
    }

    /// Get adapter for a request, updating LRU state.
    pub fn get_for_request(
        &mut self,
        request: &PromptAdapterRequest,
    ) -> Option<Arc<PromptAdapter>> {
        if self.adapters.contains_key(&request.name) {
            self.touch(&request.name);
            return self.adapters.get(&request.name).cloned();
        }

        if let Some(name) = self.id_to_name.get(&request.id) {
            let name = name.clone();
            self.touch(&name);
            return self.adapters.get(&name).cloned();
        }

        None
    }

    /// Check if an adapter is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }

    /// Unload an adapter by name.
    pub fn unload(&mut self, name: &str) -> Option<Arc<PromptAdapter>> {
        if let Some(adapter) = self.adapters.remove(name) {
            self.id_to_name.remove(&adapter.id);
            self.lru_generation.remove(name);
            Some(adapter)
        } else {
            None
        }
    }

    /// Get list of loaded adapter names.
    pub fn loaded_adapters(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// Number of loaded adapters.
    pub fn num_loaded(&self) -> usize {
        self.adapters.len()
    }

    /// Maximum number of adapters.
    pub fn max_adapters(&self) -> usize {
        self.config.max_adapters
    }

    /// Maximum virtual tokens per adapter.
    pub fn max_virtual_tokens(&self) -> usize {
        self.config.max_virtual_tokens
    }

    fn touch(&mut self, name: &str) {
        self.generation_counter += 1;
        if let Some(gen) = self.lru_generation.get_mut(name) {
            *gen = self.generation_counter;
        } else {
            self.lru_generation
                .insert(name.to_string(), self.generation_counter);
        }
    }

    fn evict_lru(&mut self) -> Result<(), PromptAdapterManagerError> {
        let victim = self
            .lru_generation
            .iter()
            .min_by_key(|(_, &gen)| gen)
            .map(|(name, _)| name.clone());

        let name = victim.ok_or(PromptAdapterManagerError::MaxAdaptersReached(
            self.config.max_adapters,
        ))?;

        self.lru_generation.remove(&name);
        if let Some(adapter) = self.adapters.remove(&name) {
            self.id_to_name.remove(&adapter.id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> PromptAdapterManager {
        PromptAdapterManager::with_defaults(Device::Cpu, DType::F32)
    }

    #[test]
    fn manager_creation() {
        let manager = test_manager();
        assert_eq!(manager.num_loaded(), 0);
        assert_eq!(manager.max_adapters(), 8);
        assert_eq!(manager.max_virtual_tokens(), 128);
    }

    #[test]
    fn manager_custom_config() {
        let config = PromptAdapterManagerConfig {
            max_adapters: 4,
            max_virtual_tokens: 64,
        };
        let manager = PromptAdapterManager::new(Device::Cpu, DType::F32, config);
        assert_eq!(manager.max_adapters(), 4);
        assert_eq!(manager.max_virtual_tokens(), 64);
    }

    #[test]
    fn manager_initially_empty() {
        let manager = test_manager();
        assert!(manager.loaded_adapters().is_empty());
        assert!(!manager.is_loaded("anything"));
        assert!(manager.get("anything").is_none());
        assert!(manager.get_by_id(1).is_none());
    }

    #[test]
    fn lru_touch_ordering() {
        let mut manager = test_manager();
        manager.touch("a");
        manager.touch("b");
        manager.touch("c");

        manager.touch("a");
        assert!(manager.lru_generation["a"] > manager.lru_generation["b"]);
        assert!(manager.lru_generation["a"] > manager.lru_generation["c"]);

        // "b" should be the LRU
        let min_name = manager
            .lru_generation
            .iter()
            .min_by_key(|(_, &gen)| gen)
            .map(|(name, _)| name.as_str())
            .unwrap();
        assert_eq!(min_name, "b");
    }

    #[test]
    fn unload_nonexistent_returns_none() {
        let mut manager = test_manager();
        assert!(manager.unload("nonexistent").is_none());
    }

    #[test]
    fn get_for_request_returns_none_when_not_loaded() {
        let mut manager = test_manager();
        let req = PromptAdapterRequest::new("test", 1, "/path", 20);
        assert!(manager.get_for_request(&req).is_none());
    }
}
