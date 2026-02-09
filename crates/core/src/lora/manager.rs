//! LoRA adapter manager for runtime adapter loading and caching.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device};
use thiserror::Error;

use super::loader::{LoraLoadError, LoraLoader};
use super::types::{LoraModel, LoraRequest};

/// Errors from the LoRA manager.
#[derive(Debug, Error)]
pub enum LoraManagerError {
    #[error("adapter not found: {0}")]
    AdapterNotFound(String),
    #[error("adapter ID already in use: {0}")]
    IdConflict(u32),
    #[error("adapter name already in use: {0}")]
    NameConflict(String),
    #[error("max adapters limit reached: {0}")]
    MaxAdaptersReached(usize),
    #[error("load error: {0}")]
    LoadError(#[from] LoraLoadError),
}

/// Configuration for the LoRA manager.
#[derive(Debug, Clone)]
pub struct LoraManagerConfig {
    /// Maximum number of adapters to keep loaded.
    pub max_adapters: usize,
    /// Whether to optimize (pre-merge scale) adapters on load.
    pub optimize_on_load: bool,
}

impl Default for LoraManagerConfig {
    fn default() -> Self {
        Self {
            max_adapters: 16,
            optimize_on_load: true,
        }
    }
}

/// Manages LoRA adapters at runtime.
///
/// Provides:
/// - Loading adapters from disk
/// - Caching loaded adapters
/// - Per-request adapter lookup
/// - LRU eviction when max_adapters is reached
pub struct LoraManager {
    /// Loaded adapters by name.
    adapters: HashMap<String, Arc<LoraModel>>,
    /// Map from ID to name for fast lookup.
    id_to_name: HashMap<u32, String>,
    /// LRU generation per adapter name. Higher = more recently used.
    lru_generation: HashMap<String, u64>,
    /// Monotonically increasing generation counter for O(1) LRU touch.
    generation_counter: u64,
    /// Configuration.
    config: LoraManagerConfig,
    /// Loader instance.
    loader: LoraLoader,
    /// Next available adapter ID.
    next_id: u32,
}

impl LoraManager {
    /// Create a new LoRA manager.
    pub fn new(device: Device, dtype: DType, config: LoraManagerConfig) -> Self {
        Self {
            adapters: HashMap::new(),
            id_to_name: HashMap::new(),
            lru_generation: HashMap::new(),
            generation_counter: 0,
            config,
            loader: LoraLoader::new(device, dtype),
            next_id: 1,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(device: Device, dtype: DType) -> Self {
        Self::new(device, dtype, LoraManagerConfig::default())
    }

    /// Load an adapter from a path.
    ///
    /// Returns the adapter ID. If an adapter with the same name is already
    /// loaded, returns its existing ID.
    pub fn load_adapter(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<u32, LoraManagerError> {
        let name = name.into();

        // Check if already loaded
        if let Some(model) = self.adapters.get(&name) {
            let id = model.id;
            self.touch(&name);
            return Ok(id);
        }

        // Check capacity
        if self.adapters.len() >= self.config.max_adapters {
            self.evict_lru()?;
        }

        // Allocate ID
        let id = self.next_id;
        self.next_id += 1;

        // Load the adapter
        let mut model = self.loader.load(path, &name, id)?;

        // Optimize if configured
        if self.config.optimize_on_load {
            model
                .optimize()
                .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?;
        }

        // Store
        self.adapters.insert(name.clone(), Arc::new(model));
        self.id_to_name.insert(id, name.clone());
        self.touch(&name);

        Ok(id)
    }

    /// Load an adapter with a specific ID.
    ///
    /// Useful for restoring state or when IDs need to match external references.
    pub fn load_adapter_with_id(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
        id: u32,
    ) -> Result<(), LoraManagerError> {
        let name = name.into();

        // Check for conflicts
        if self.id_to_name.contains_key(&id) {
            return Err(LoraManagerError::IdConflict(id));
        }
        if self.adapters.contains_key(&name) {
            return Err(LoraManagerError::NameConflict(name));
        }

        // Check capacity
        if self.adapters.len() >= self.config.max_adapters {
            self.evict_lru()?;
        }

        // Load the adapter
        let mut model = self.loader.load(path, &name, id)?;

        // Optimize if configured
        if self.config.optimize_on_load {
            model
                .optimize()
                .map_err(|e| LoraLoadError::WeightsLoad(e.to_string()))?;
        }

        // Store
        self.adapters.insert(name.clone(), Arc::new(model));
        self.id_to_name.insert(id, name.clone());
        self.touch(&name);

        // Update next_id if necessary
        if id >= self.next_id {
            self.next_id = id + 1;
        }

        Ok(())
    }

    /// Get an adapter by name.
    pub fn get(&self, name: &str) -> Option<Arc<LoraModel>> {
        self.adapters.get(name).cloned()
    }

    /// Get an adapter by ID.
    pub fn get_by_id(&self, id: u32) -> Option<Arc<LoraModel>> {
        self.id_to_name
            .get(&id)
            .and_then(|name| self.adapters.get(name))
            .cloned()
    }

    /// Get adapter for a LoRA request, updating LRU state.
    pub fn get_for_request(&mut self, request: &LoraRequest) -> Option<Arc<LoraModel>> {
        // Try by name first (most common case) — single clone
        if self.adapters.contains_key(&request.name) {
            self.touch(&request.name);
            return self.adapters.get(&request.name).cloned();
        }

        // Try by ID — resolve name once, then single clone
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
    pub fn unload(&mut self, name: &str) -> Option<Arc<LoraModel>> {
        if let Some(model) = self.adapters.remove(name) {
            self.id_to_name.remove(&model.id);
            self.lru_generation.remove(name);
            Some(model)
        } else {
            None
        }
    }

    /// Unload an adapter by ID.
    pub fn unload_by_id(&mut self, id: u32) -> Option<Arc<LoraModel>> {
        if let Some(name) = self.id_to_name.remove(&id) {
            self.lru_generation.remove(&name);
            self.adapters.remove(&name)
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

    /// Mark an adapter as recently used. O(1).
    fn touch(&mut self, name: &str) {
        self.generation_counter += 1;
        // Insert for new adapters, update for existing ones.
        if let Some(gen) = self.lru_generation.get_mut(name) {
            *gen = self.generation_counter;
        } else {
            self.lru_generation
                .insert(name.to_string(), self.generation_counter);
        }
    }

    /// Evict the least recently used adapter. O(n) in adapter count,
    /// but eviction is rare (only when at capacity).
    fn evict_lru(&mut self) -> Result<(), LoraManagerError> {
        let victim = self
            .lru_generation
            .iter()
            .min_by_key(|(_, &gen)| gen)
            .map(|(name, _)| name.clone());

        let name = victim.ok_or(LoraManagerError::MaxAdaptersReached(
            self.config.max_adapters,
        ))?;

        self.lru_generation.remove(&name);
        if let Some(model) = self.adapters.remove(&name) {
            self.id_to_name.remove(&model.id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> LoraManager {
        LoraManager::with_defaults(Device::Cpu, DType::F32)
    }

    #[test]
    fn test_manager_creation() {
        let manager = test_manager();
        assert_eq!(manager.num_loaded(), 0);
        assert_eq!(manager.max_adapters(), 16);
    }

    #[test]
    fn test_manager_custom_config() {
        let config = LoraManagerConfig {
            max_adapters: 4,
            optimize_on_load: false,
        };
        let manager = LoraManager::new(Device::Cpu, DType::F32, config);
        assert_eq!(manager.max_adapters(), 4);
    }

    #[test]
    fn test_lru_touch() {
        let mut manager = test_manager();
        // Simulate three adapters loaded in order a, b, c
        manager.touch("a");
        manager.touch("b");
        manager.touch("c");

        // After touching "a", it should have highest generation
        manager.touch("a");
        assert!(manager.lru_generation["a"] > manager.lru_generation["b"]);
        assert!(manager.lru_generation["a"] > manager.lru_generation["c"]);

        // "b" should be the LRU (lowest generation)
        let min_name = manager
            .lru_generation
            .iter()
            .min_by_key(|(_, &gen)| gen)
            .map(|(name, _)| name.as_str())
            .unwrap();
        assert_eq!(min_name, "b");

        // After touching "b", "c" becomes LRU
        manager.touch("b");
        let min_name = manager
            .lru_generation
            .iter()
            .min_by_key(|(_, &gen)| gen)
            .map(|(name, _)| name.as_str())
            .unwrap();
        assert_eq!(min_name, "c");
    }

    #[test]
    fn test_loaded_adapters_initially_empty() {
        let manager = test_manager();
        assert!(manager.loaded_adapters().is_empty());
        assert!(!manager.is_loaded("anything"));
    }
}
