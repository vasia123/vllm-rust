//! Configuration persistence for vLLM server.
//!
//! Configuration is loaded with the following priority:
//! 1. CLI arguments (highest priority)
//! 2. Config file (~/.config/vllm-server/config.toml)
//! 3. Default values (lowest priority)

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Persistent configuration stored in TOML format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Model identifier (HuggingFace Hub format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Draft model for speculative decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,

    /// Number of speculative tokens per step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_speculative_tokens: Option<usize>,

    /// Port to listen on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,

    /// Host to bind to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,

    /// Number of KV cache blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_blocks: Option<usize>,

    /// Maximum concurrent requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_requests: Option<usize>,

    /// Decode steps per scheduler invocation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_step_count: Option<usize>,

    /// Maximum tokens per scheduling step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_step: Option<usize>,

    /// Enable chunked prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_chunked_prefill: Option<bool>,

    /// Enable prefix caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_prefix_caching: Option<bool>,

    /// Maximum requests per second (rate limit). 0 or None means no limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_requests_per_second: Option<u32>,

    /// Maximum pending requests in queue before rejecting with 503.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_queue_depth: Option<usize>,

    /// Comma-separated list of allowed CORS origins. "*" allows all origins.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_origins: Option<String>,

    /// Comma-separated list of allowed CORS HTTP methods.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_methods: Option<String>,

    /// Comma-separated list of allowed CORS headers. "*" allows all headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_headers: Option<String>,
}

impl ServerConfig {
    /// Get the default config file path.
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("vllm-server").join("config.toml"))
    }

    /// Load configuration from the default path.
    pub fn load() -> Self {
        Self::default_path()
            .and_then(|path| Self::load_from(&path).ok())
            .unwrap_or_default()
    }

    /// Load configuration from a specific path.
    pub fn load_from(path: &PathBuf) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(ConfigError::Io)?;
        toml::from_str(&content).map_err(ConfigError::Parse)
    }

    /// Save configuration to the default path.
    pub fn save(&self) -> Result<PathBuf, ConfigError> {
        let path = Self::default_path().ok_or(ConfigError::NoConfigDir)?;
        self.save_to(&path)?;
        Ok(path)
    }

    /// Save configuration to a specific path.
    pub fn save_to(&self, path: &PathBuf) -> Result<(), ConfigError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }

        let content = toml::to_string_pretty(self).map_err(ConfigError::Serialize)?;
        fs::write(path, content).map_err(ConfigError::Io)?;
        Ok(())
    }

    /// Merge with another config, preferring values from `other`.
    pub fn merge(&mut self, other: &ServerConfig) {
        if other.model.is_some() {
            self.model = other.model.clone();
        }
        if other.draft_model.is_some() {
            self.draft_model = other.draft_model.clone();
        }
        if other.num_speculative_tokens.is_some() {
            self.num_speculative_tokens = other.num_speculative_tokens;
        }
        if other.port.is_some() {
            self.port = other.port;
        }
        if other.host.is_some() {
            self.host = other.host.clone();
        }
        if other.num_blocks.is_some() {
            self.num_blocks = other.num_blocks;
        }
        if other.max_requests.is_some() {
            self.max_requests = other.max_requests;
        }
        if other.multi_step_count.is_some() {
            self.multi_step_count = other.multi_step_count;
        }
        if other.max_tokens_per_step.is_some() {
            self.max_tokens_per_step = other.max_tokens_per_step;
        }
        if other.enable_chunked_prefill.is_some() {
            self.enable_chunked_prefill = other.enable_chunked_prefill;
        }
        if other.enable_prefix_caching.is_some() {
            self.enable_prefix_caching = other.enable_prefix_caching;
        }
        if other.max_requests_per_second.is_some() {
            self.max_requests_per_second = other.max_requests_per_second;
        }
        if other.max_queue_depth.is_some() {
            self.max_queue_depth = other.max_queue_depth;
        }
        if other.allowed_origins.is_some() {
            self.allowed_origins = other.allowed_origins.clone();
        }
        if other.allowed_methods.is_some() {
            self.allowed_methods = other.allowed_methods.clone();
        }
        if other.allowed_headers.is_some() {
            self.allowed_headers = other.allowed_headers.clone();
        }
    }
}

/// Configuration errors.
#[derive(Debug)]
pub enum ConfigError {
    /// IO error reading/writing config file.
    Io(std::io::Error),
    /// Error parsing TOML.
    Parse(toml::de::Error),
    /// Error serializing to TOML.
    Serialize(toml::ser::Error),
    /// No config directory available.
    NoConfigDir,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "IO error: {}", e),
            ConfigError::Parse(e) => write!(f, "Parse error: {}", e),
            ConfigError::Serialize(e) => write!(f, "Serialize error: {}", e),
            ConfigError::NoConfigDir => write!(f, "No config directory available"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let config = ServerConfig {
            model: Some("Qwen/Qwen3-0.6B".to_string()),
            num_blocks: Some(512),
            max_requests: Some(8),
            ..Default::default()
        };

        config.save_to(&path).unwrap();
        let loaded = ServerConfig::load_from(&path).unwrap();

        assert_eq!(loaded.model, Some("Qwen/Qwen3-0.6B".to_string()));
        assert_eq!(loaded.num_blocks, Some(512));
        assert_eq!(loaded.max_requests, Some(8));
    }

    #[test]
    fn test_merge() {
        let mut base = ServerConfig {
            model: Some("base-model".to_string()),
            num_blocks: Some(256),
            ..Default::default()
        };

        let override_config = ServerConfig {
            num_blocks: Some(512),
            max_requests: Some(16),
            ..Default::default()
        };

        base.merge(&override_config);

        assert_eq!(base.model, Some("base-model".to_string())); // Unchanged
        assert_eq!(base.num_blocks, Some(512)); // Overridden
        assert_eq!(base.max_requests, Some(16)); // Added
    }
}
