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

    /// API key for Bearer token authentication. Set via `--api-key` or `VLLM_API_KEY` env.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Path to TLS certificate file (PEM format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssl_certfile: Option<String>,

    /// Path to TLS private key file (PEM format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ssl_keyfile: Option<String>,

    /// Model name returned by `/v1/models`. Defaults to model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub served_model_name: Option<String>,

    /// Data type for model weights (auto, bf16, fp16, fp32).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Override quantization method (none, fp8, gptq, awq, bnb, gguf).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,

    /// Fraction of GPU memory to use (0.0-1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_utilization: Option<f32>,

    /// Maximum model context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,

    /// Tensor parallel size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<usize>,

    /// Random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Allow custom code from HuggingFace.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_remote_code: Option<bool>,

    /// Maximum LoRA rank.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_lora_rank: Option<usize>,

    /// Override tokenizer path or HuggingFace ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,

    /// HuggingFace model revision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,

    /// Maximum tokens per scheduling step (alias for max_tokens_per_step).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_batched_tokens: Option<usize>,

    /// Scheduling policy (fcfs, priority).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduling_policy: Option<String>,

    /// Suppress per-request logging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_log_requests: Option<bool>,

    /// Override chat template file path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_path: Option<String>,

    /// Default assistant role name in responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_role: Option<String>,

    // ─── KV Cache / Memory ──────────────────────────────────────────────

    /// KV cache block size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,

    /// KV cache data type (auto, fp8, fp8_e5m2, fp8_e4m3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<String>,

    /// CPU swap space in GiB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_space: Option<f32>,

    /// CPU offload budget in GiB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_offload_gb: Option<f32>,

    /// Override GPU block count directly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu_blocks_override: Option<usize>,

    /// Disable CUDA graph capture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enforce_eager: Option<bool>,

    // ─── Model Loading ──────────────────────────────────────────────────

    /// Weight loading format (auto, safetensors, pt, npcache, dummy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_format: Option<String>,

    /// HuggingFace model cache directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_dir: Option<String>,

    /// Tokenizer mode (auto, slow).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_mode: Option<String>,

    /// Tokenizer revision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,

    /// Code revision for custom model code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_revision: Option<String>,

    /// Parallel weight loading workers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_parallel_loading_workers: Option<usize>,

    // ─── Scheduler Tuning ───────────────────────────────────────────────

    /// Maximum concurrent sequences (alias for max_requests).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_seqs: Option<usize>,

    /// Preemption mode (recompute, swap).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preemption_mode: Option<String>,

    /// Maximum partial prefills per step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_partial_prefills: Option<usize>,

    /// Token threshold for "long" prefill classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_prefill_token_threshold: Option<usize>,

    /// Streaming token interval in ms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_interval: Option<usize>,

    // ─── LoRA Configuration ─────────────────────────────────────────────

    /// Enable LoRA support globally.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_lora: Option<bool>,

    /// Max concurrent LoRA adapters per batch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_loras: Option<usize>,

    /// Extra vocab size for LoRA adapters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_extra_vocab_size: Option<usize>,

    /// LoRA weight data type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_dtype: Option<String>,

    /// CPU LoRA cache limit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_cpu_loras: Option<usize>,

    // ─── Speculative Decoding ───────────────────────────────────────────

    /// Speculative acceptance method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_decoding_acceptance_method: Option<String>,

    /// NGram prompt lookup max.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngram_prompt_lookup_max: Option<usize>,

    /// NGram prompt lookup min.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ngram_prompt_lookup_min: Option<usize>,

    // ─── Observability ──────────────────────────────────────────────────

    /// Suppress periodic stats logging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_log_stats: Option<bool>,

    /// Max logprobs per token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_logprobs: Option<usize>,

    /// OpenTelemetry OTLP endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub otlp_traces_endpoint: Option<String>,

    /// Server log level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,

    // ─── Multimodal ─────────────────────────────────────────────────────

    /// Multimodal items limit per prompt (JSON).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_mm_per_prompt: Option<String>,

    /// Disable multimodal preprocessor cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_mm_preprocessor_cache: Option<bool>,

    // ─── Pipeline Parallelism ───────────────────────────────────────────

    /// Pipeline parallel size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_parallel_size: Option<usize>,

    // ─── Generation Defaults ────────────────────────────────────────────

    /// Guided decoding backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guided_decoding_backend: Option<String>,

    /// Max sequence length for CUDA graph capture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_seq_len_to_capture: Option<usize>,

    /// Enable automatic tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_auto_tool_choice: Option<bool>,

    /// Return token IDs by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,
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
        if other.api_key.is_some() {
            self.api_key = other.api_key.clone();
        }
        if other.ssl_certfile.is_some() {
            self.ssl_certfile = other.ssl_certfile.clone();
        }
        if other.ssl_keyfile.is_some() {
            self.ssl_keyfile = other.ssl_keyfile.clone();
        }
        if other.served_model_name.is_some() {
            self.served_model_name = other.served_model_name.clone();
        }
        if other.dtype.is_some() {
            self.dtype = other.dtype.clone();
        }
        if other.quantization.is_some() {
            self.quantization = other.quantization.clone();
        }
        if other.gpu_memory_utilization.is_some() {
            self.gpu_memory_utilization = other.gpu_memory_utilization;
        }
        if other.max_model_len.is_some() {
            self.max_model_len = other.max_model_len;
        }
        if other.tensor_parallel_size.is_some() {
            self.tensor_parallel_size = other.tensor_parallel_size;
        }
        if other.seed.is_some() {
            self.seed = other.seed;
        }
        if other.trust_remote_code.is_some() {
            self.trust_remote_code = other.trust_remote_code;
        }
        if other.max_lora_rank.is_some() {
            self.max_lora_rank = other.max_lora_rank;
        }
        if other.tokenizer.is_some() {
            self.tokenizer = other.tokenizer.clone();
        }
        if other.revision.is_some() {
            self.revision = other.revision.clone();
        }
        if other.max_num_batched_tokens.is_some() {
            self.max_num_batched_tokens = other.max_num_batched_tokens;
        }
        if other.scheduling_policy.is_some() {
            self.scheduling_policy = other.scheduling_policy.clone();
        }
        if other.disable_log_requests.is_some() {
            self.disable_log_requests = other.disable_log_requests;
        }
        if other.chat_template_path.is_some() {
            self.chat_template_path = other.chat_template_path.clone();
        }
        if other.response_role.is_some() {
            self.response_role = other.response_role.clone();
        }
        // KV Cache / Memory
        if other.block_size.is_some() {
            self.block_size = other.block_size;
        }
        if other.kv_cache_dtype.is_some() {
            self.kv_cache_dtype = other.kv_cache_dtype.clone();
        }
        if other.swap_space.is_some() {
            self.swap_space = other.swap_space;
        }
        if other.cpu_offload_gb.is_some() {
            self.cpu_offload_gb = other.cpu_offload_gb;
        }
        if other.num_gpu_blocks_override.is_some() {
            self.num_gpu_blocks_override = other.num_gpu_blocks_override;
        }
        if other.enforce_eager.is_some() {
            self.enforce_eager = other.enforce_eager;
        }
        // Model Loading
        if other.load_format.is_some() {
            self.load_format = other.load_format.clone();
        }
        if other.download_dir.is_some() {
            self.download_dir = other.download_dir.clone();
        }
        if other.tokenizer_mode.is_some() {
            self.tokenizer_mode = other.tokenizer_mode.clone();
        }
        if other.tokenizer_revision.is_some() {
            self.tokenizer_revision = other.tokenizer_revision.clone();
        }
        if other.code_revision.is_some() {
            self.code_revision = other.code_revision.clone();
        }
        if other.max_parallel_loading_workers.is_some() {
            self.max_parallel_loading_workers = other.max_parallel_loading_workers;
        }
        // Scheduler Tuning
        if other.max_num_seqs.is_some() {
            self.max_num_seqs = other.max_num_seqs;
        }
        if other.preemption_mode.is_some() {
            self.preemption_mode = other.preemption_mode.clone();
        }
        if other.max_num_partial_prefills.is_some() {
            self.max_num_partial_prefills = other.max_num_partial_prefills;
        }
        if other.long_prefill_token_threshold.is_some() {
            self.long_prefill_token_threshold = other.long_prefill_token_threshold;
        }
        if other.stream_interval.is_some() {
            self.stream_interval = other.stream_interval;
        }
        // LoRA
        if other.enable_lora.is_some() {
            self.enable_lora = other.enable_lora;
        }
        if other.max_loras.is_some() {
            self.max_loras = other.max_loras;
        }
        if other.lora_extra_vocab_size.is_some() {
            self.lora_extra_vocab_size = other.lora_extra_vocab_size;
        }
        if other.lora_dtype.is_some() {
            self.lora_dtype = other.lora_dtype.clone();
        }
        if other.max_cpu_loras.is_some() {
            self.max_cpu_loras = other.max_cpu_loras;
        }
        // Speculative Decoding
        if other.spec_decoding_acceptance_method.is_some() {
            self.spec_decoding_acceptance_method =
                other.spec_decoding_acceptance_method.clone();
        }
        if other.ngram_prompt_lookup_max.is_some() {
            self.ngram_prompt_lookup_max = other.ngram_prompt_lookup_max;
        }
        if other.ngram_prompt_lookup_min.is_some() {
            self.ngram_prompt_lookup_min = other.ngram_prompt_lookup_min;
        }
        // Observability
        if other.disable_log_stats.is_some() {
            self.disable_log_stats = other.disable_log_stats;
        }
        if other.max_logprobs.is_some() {
            self.max_logprobs = other.max_logprobs;
        }
        if other.otlp_traces_endpoint.is_some() {
            self.otlp_traces_endpoint = other.otlp_traces_endpoint.clone();
        }
        if other.log_level.is_some() {
            self.log_level = other.log_level.clone();
        }
        // Multimodal
        if other.limit_mm_per_prompt.is_some() {
            self.limit_mm_per_prompt = other.limit_mm_per_prompt.clone();
        }
        if other.disable_mm_preprocessor_cache.is_some() {
            self.disable_mm_preprocessor_cache = other.disable_mm_preprocessor_cache;
        }
        // Pipeline Parallelism
        if other.pipeline_parallel_size.is_some() {
            self.pipeline_parallel_size = other.pipeline_parallel_size;
        }
        // Generation Defaults
        if other.guided_decoding_backend.is_some() {
            self.guided_decoding_backend = other.guided_decoding_backend.clone();
        }
        if other.max_seq_len_to_capture.is_some() {
            self.max_seq_len_to_capture = other.max_seq_len_to_capture;
        }
        if other.enable_auto_tool_choice.is_some() {
            self.enable_auto_tool_choice = other.enable_auto_tool_choice;
        }
        if other.return_tokens_as_token_ids.is_some() {
            self.return_tokens_as_token_ids = other.return_tokens_as_token_ids;
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

    #[test]
    fn test_new_fields_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config_new.toml");

        let config = ServerConfig {
            dtype: Some("fp16".to_string()),
            quantization: Some("gptq".to_string()),
            gpu_memory_utilization: Some(0.85),
            max_model_len: Some(4096),
            tensor_parallel_size: Some(1),
            seed: Some(42),
            trust_remote_code: Some(true),
            max_lora_rank: Some(32),
            tokenizer: Some("custom/tokenizer".to_string()),
            revision: Some("v1.0".to_string()),
            max_num_batched_tokens: Some(4096),
            scheduling_policy: Some("priority".to_string()),
            disable_log_requests: Some(true),
            chat_template_path: Some("/tmp/template.jinja".to_string()),
            response_role: Some("model".to_string()),
            ..Default::default()
        };

        config.save_to(&path).unwrap();
        let loaded = ServerConfig::load_from(&path).unwrap();

        assert_eq!(loaded.dtype, Some("fp16".to_string()));
        assert_eq!(loaded.quantization, Some("gptq".to_string()));
        assert_eq!(loaded.gpu_memory_utilization, Some(0.85));
        assert_eq!(loaded.max_model_len, Some(4096));
        assert_eq!(loaded.tensor_parallel_size, Some(1));
        assert_eq!(loaded.seed, Some(42));
        assert_eq!(loaded.trust_remote_code, Some(true));
        assert_eq!(loaded.max_lora_rank, Some(32));
        assert_eq!(loaded.tokenizer, Some("custom/tokenizer".to_string()));
        assert_eq!(loaded.revision, Some("v1.0".to_string()));
        assert_eq!(loaded.max_num_batched_tokens, Some(4096));
        assert_eq!(loaded.scheduling_policy, Some("priority".to_string()));
        assert_eq!(loaded.disable_log_requests, Some(true));
        assert_eq!(
            loaded.chat_template_path,
            Some("/tmp/template.jinja".to_string())
        );
        assert_eq!(loaded.response_role, Some("model".to_string()));
    }

    #[test]
    fn test_merge_new_fields() {
        let mut base = ServerConfig {
            dtype: Some("bf16".to_string()),
            seed: Some(0),
            ..Default::default()
        };

        let other = ServerConfig {
            dtype: Some("fp16".to_string()),
            max_lora_rank: Some(128),
            scheduling_policy: Some("priority".to_string()),
            ..Default::default()
        };

        base.merge(&other);

        assert_eq!(base.dtype, Some("fp16".to_string())); // Overridden
        assert_eq!(base.seed, Some(0)); // Unchanged
        assert_eq!(base.max_lora_rank, Some(128)); // Added
        assert_eq!(base.scheduling_policy, Some("priority".to_string())); // Added
    }
}
