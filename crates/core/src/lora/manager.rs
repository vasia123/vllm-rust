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
    /// LRU order for eviction (most recently used at end).
    lru_order: Vec<String>,
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
            lru_order: Vec::new(),
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
        self.lru_order.push(name);

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
        self.lru_order.push(name);

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

    /// Get adapter for a LoRA request.
    pub fn get_for_request(&mut self, request: &LoraRequest) -> Option<Arc<LoraModel>> {
        // Try by name first (most common case)
        if let Some(model) = self.adapters.get(&request.name).cloned() {
            self.touch(&request.name);
            return Some(model);
        }

        // Try by ID
        if let Some(name) = self.id_to_name.get(&request.id).cloned() {
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
            self.lru_order.retain(|n| n != name);
            Some(model)
        } else {
            None
        }
    }

    /// Unload an adapter by ID.
    pub fn unload_by_id(&mut self, id: u32) -> Option<Arc<LoraModel>> {
        if let Some(name) = self.id_to_name.remove(&id) {
            self.lru_order.retain(|n| n != &name);
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

    /// Mark an adapter as recently used.
    fn touch(&mut self, name: &str) {
        if let Some(pos) = self.lru_order.iter().position(|n| n == name) {
            let name = self.lru_order.remove(pos);
            self.lru_order.push(name);
        }
    }

    /// Evict the least recently used adapter.
    fn evict_lru(&mut self) -> Result<(), LoraManagerError> {
        if self.lru_order.is_empty() {
            return Err(LoraManagerError::MaxAdaptersReached(
                self.config.max_adapters,
            ));
        }

        let name = self.lru_order.remove(0);
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
        manager.lru_order = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        manager.touch("a");
        assert_eq!(manager.lru_order, vec!["b", "c", "a"]);

        manager.touch("b");
        assert_eq!(manager.lru_order, vec!["c", "a", "b"]);
    }

    #[test]
    fn test_loaded_adapters_initially_empty() {
        let manager = test_manager();
        assert!(manager.loaded_adapters().is_empty());
        assert!(!manager.is_loaded("anything"));
    }
}
