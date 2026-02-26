use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use vllm_core::{
    engine::{
        spec_decode::MLPSpeculatorDraftProposer, start_engine, start_engine_with_draft,
        start_engine_with_proposer, AcceptanceMethod, EngineConfig, GenerationRequest, NGramConfig,
        NGramProposer, SpeculativeConfig,
    },
    kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager},
    loader,
    lora::LoraLoader,
    models,
    scheduler::{PreemptionMode, SchedulerConfig},
    tokenizer::{ChatTemplateEngine, TokenizerWrapper},
};

use vllm_server::api::{
    self,
    admin::{prometheus, types::RuntimeConfig},
    AdminState, AppState, AtomicEngineHandle, ProductionEngineBuilder,
};
use vllm_server::config::ServerConfig;
use vllm_server::logging;
use vllm_server::shutdown::shutdown_signal;

#[derive(Parser)]
#[command(name = "vllm-server", about = "Rust LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)] // CLI struct parsed once at startup
enum Command {
    /// Start the OpenAI-compatible HTTP server
    Serve {
        /// Model ID (HuggingFace Hub format)
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,

        /// Draft model ID for speculative decoding
        #[arg(long)]
        draft_model: Option<String>,

        /// Number of speculative tokens per step
        #[arg(long, default_value_t = 3)]
        num_speculative_tokens: usize,

        /// Port to listen on
        #[arg(long, default_value_t = 8000)]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Number of KV cache blocks
        #[arg(long, default_value_t = 512)]
        num_blocks: usize,

        /// Maximum concurrent requests
        #[arg(long, default_value_t = 8)]
        max_requests: usize,

        /// Decode steps per scheduler invocation (amortizes scheduling overhead)
        #[arg(long, default_value_t = 4)]
        multi_step_count: usize,

        /// LoRA adapters to load at startup (format: name=path, can be repeated)
        /// Example: --lora-adapter sql=./sql-adapter --lora-adapter code=./code-adapter
        #[arg(long = "lora-adapter")]
        lora_adapters: Vec<String>,

        /// Enable prefix caching for KV cache reuse
        #[arg(long)]
        enable_prefix_caching: bool,

        /// Enable chunked prefill for long prompts
        #[arg(long)]
        enable_chunked_prefill: bool,

        /// Graceful shutdown timeout in seconds (force shutdown after this duration)
        #[arg(long, default_value_t = 30)]
        shutdown_timeout: u64,

        /// Comma-separated list of allowed CORS origins ("*" allows all)
        #[arg(long, default_value = "*")]
        allowed_origins: String,

        /// Comma-separated list of allowed CORS HTTP methods
        #[arg(long, default_value = "GET,POST,OPTIONS")]
        allowed_methods: String,

        /// Comma-separated list of allowed CORS headers ("*" allows all)
        #[arg(long, default_value = "*")]
        allowed_headers: String,

        /// Maximum request body size in MiB (default: 32)
        #[arg(long, default_value_t = 32)]
        max_body_size_mb: usize,

        /// Tool call parser for extracting function calls from model output.
        /// Supported: hermes, glm4, json, llama, mistral, deepseek_v3, deepseek_v31,
        /// internlm2, jamba, pythonic, granite, granite-20b-fc, kimi_k2, phi4mini,
        /// longcat, xlam, gigachat3, functiongemma, hunyuan, ernie45, seed_oss,
        /// deepseek_v32, step3, qwen3coder
        #[arg(long, default_value = "hermes")]
        tool_call_parser: String,

        /// Reasoning parser for extracting chain-of-thought content from model output.
        /// Supported: deepseek_r1, deepseek_v3, qwen3, mistral, step3, step3p5,
        /// ernie45, granite, olmo3, seed_oss, minimax_m2, hunyuan_a13b,
        /// glm45, holo2, kimi_k2, identity.
        /// Empty string or "identity" disables reasoning extraction.
        #[arg(long, default_value = "")]
        reasoning_parser: String,

        /// Model name returned by /v1/models and echoed in responses.
        /// Defaults to the model identifier.
        #[arg(long)]
        served_model_name: Option<String>,

        /// Path to TLS certificate file (PEM format). Requires --ssl-keyfile.
        #[arg(long)]
        ssl_certfile: Option<String>,

        /// Path to TLS private key file (PEM format). Requires --ssl-certfile.
        #[arg(long)]
        ssl_keyfile: Option<String>,

        /// Data type for model weights and activations.
        /// auto: use model's native dtype (bf16 for most models)
        #[arg(long, default_value = "auto")]
        dtype: String,

        /// Override quantization method. By default, auto-detected from model config.
        /// Use "none" to force full-precision even if model has quantization config.
        #[arg(long)]
        quantization: Option<String>,

        /// Fraction of GPU memory to use for model + KV cache (0.0 to 1.0).
        /// When set, overrides --num-blocks by computing blocks from available VRAM.
        #[arg(long)]
        gpu_memory_utilization: Option<f32>,

        /// Maximum model context length. Overrides the model's max_position_embeddings.
        /// Useful for limiting memory usage or extending context with RoPE scaling.
        #[arg(long)]
        max_model_len: Option<usize>,

        /// Tensor parallel size (number of GPUs). Currently only 1 is supported.
        #[arg(long, default_value_t = 1)]
        tensor_parallel_size: usize,

        /// Random seed for reproducible sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,

        /// Allow loading models with custom code from HuggingFace Hub.
        #[arg(long)]
        trust_remote_code: bool,

        /// Maximum LoRA rank allowed for adapter loading.
        #[arg(long, default_value_t = 64)]
        max_lora_rank: usize,

        /// Override tokenizer path (HuggingFace Hub ID or local path).
        /// Defaults to the model path.
        #[arg(long)]
        tokenizer: Option<String>,

        /// HuggingFace model revision (branch, tag, or commit hash).
        #[arg(long, default_value = "main")]
        revision: String,

        /// Maximum number of tokens per scheduling step.
        /// Higher values increase throughput but use more memory.
        #[arg(long, default_value_t = 2048)]
        max_num_batched_tokens: usize,

        /// Scheduling policy for request ordering.
        #[arg(long, default_value = "fcfs")]
        scheduling_policy: String,

        /// Suppress per-request logging (arrival, completion, latency).
        #[arg(long)]
        disable_log_requests: bool,

        /// Override chat template with a Jinja template file path.
        #[arg(long)]
        chat_template: Option<String>,

        /// Default assistant role name in chat completion responses.
        #[arg(long, default_value = "assistant")]
        response_role: String,

        // ─── KV Cache / Memory ──────────────────────────────────────────
        /// KV cache block size (tokens per block).
        #[arg(long, default_value_t = 16)]
        block_size: usize,

        /// Data type for KV cache (auto, fp8, fp8_e5m2, fp8_e4m3).
        /// "auto" uses the model's compute dtype.
        #[arg(long, default_value = "auto")]
        kv_cache_dtype: String,

        /// CPU swap space in GiB for offloading KV cache.
        /// 0 disables CPU offloading.
        #[arg(long, default_value_t = 0.0)]
        swap_space: f32,

        /// CPU memory budget in GiB for KV cache offloading.
        /// Alias for swap-space behavior.
        #[arg(long, default_value_t = 0.0)]
        cpu_offload_gb: f32,

        /// Override GPU block count directly (bypasses auto-calculation).
        #[arg(long)]
        num_gpu_blocks_override: Option<usize>,

        /// Disable CUDA graph capture (use eager execution).
        #[arg(long)]
        enforce_eager: bool,

        // ─── Model Loading ──────────────────────────────────────────────
        /// Weight loading format: auto, safetensors, pt, npcache, dummy.
        #[arg(long, default_value = "auto")]
        load_format: String,

        /// Directory for downloading/caching HuggingFace models.
        /// Defaults to HF_HOME or ~/.cache/huggingface.
        #[arg(long)]
        download_dir: Option<String>,

        /// Tokenizer mode: auto, slow.
        /// "slow" forces the Python tokenizer fallback.
        #[arg(long, default_value = "auto")]
        tokenizer_mode: String,

        /// HuggingFace revision for the tokenizer (if different from model).
        #[arg(long)]
        tokenizer_revision: Option<String>,

        /// HuggingFace revision for custom model code (trust-remote-code).
        #[arg(long)]
        code_revision: Option<String>,

        /// HuggingFace API token for private model access.
        /// Can also be set via HUGGING_FACE_HUB_TOKEN or HF_TOKEN env vars.
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,

        /// Number of parallel workers for loading model weights.
        #[arg(long, default_value_t = 1)]
        max_parallel_loading_workers: usize,

        // ─── Scheduler Tuning ───────────────────────────────────────────
        /// Maximum number of sequences (same as --max-requests, vLLM-compatible alias).
        #[arg(long)]
        max_num_seqs: Option<usize>,

        /// Preemption mode when memory is exhausted: recompute or swap.
        #[arg(long, default_value = "recompute")]
        preemption_mode: String,

        /// Maximum number of partial prefills per scheduling step.
        #[arg(long)]
        max_num_partial_prefills: Option<usize>,

        /// Token count threshold for classifying a prefill as "long".
        #[arg(long)]
        long_prefill_token_threshold: Option<usize>,

        /// Delay between streaming token emission (milliseconds).
        /// 0 means emit every token immediately.
        #[arg(long, default_value_t = 0)]
        stream_interval: usize,

        // ─── LoRA Configuration ─────────────────────────────────────────
        /// Enable LoRA adapter support globally.
        #[arg(long)]
        enable_lora: bool,

        /// Maximum number of concurrent LoRA adapters in a single batch.
        #[arg(long, default_value_t = 1)]
        max_loras: usize,

        /// Extra vocabulary size reserved for LoRA adapters.
        #[arg(long, default_value_t = 256)]
        lora_extra_vocab_size: usize,

        /// Data type for LoRA weights (auto, fp16, bf16, fp32).
        #[arg(long)]
        lora_dtype: Option<String>,

        /// Maximum number of LoRA adapters cached on CPU.
        #[arg(long)]
        max_cpu_loras: Option<usize>,

        // ─── Speculative Decoding ───────────────────────────────────────
        /// Token acceptance method for speculative decoding.
        /// Supported: rejection_sampler, typical_acceptance_sampler.
        #[arg(long, default_value = "rejection_sampler")]
        spec_decoding_acceptance_method: String,

        /// Maximum n-gram size for prompt lookup speculative decoding.
        #[arg(long)]
        ngram_prompt_lookup_max: Option<usize>,

        /// Minimum n-gram size for prompt lookup speculative decoding.
        #[arg(long)]
        ngram_prompt_lookup_min: Option<usize>,

        // ─── Observability ──────────────────────────────────────────────
        /// Suppress periodic engine performance statistics logging.
        #[arg(long)]
        disable_log_stats: bool,

        /// Maximum number of log probabilities to return per token.
        #[arg(long, default_value_t = 20)]
        max_logprobs: usize,

        /// OpenTelemetry OTLP endpoint for trace export.
        #[arg(long)]
        otlp_traces_endpoint: Option<String>,

        /// Log level for the server (trace, debug, info, warn, error).
        #[arg(long, default_value = "info")]
        log_level: String,

        // ─── Multimodal / VLM ───────────────────────────────────────────
        /// Maximum multimodal items per prompt (JSON: {"image": 5, "video": 1}).
        #[arg(long)]
        limit_mm_per_prompt: Option<String>,

        /// Disable caching for multimodal preprocessor outputs.
        #[arg(long)]
        disable_mm_preprocessor_cache: bool,

        // ─── Pipeline Parallelism ───────────────────────────────────────
        /// Pipeline parallel size (number of pipeline stages).
        /// Currently only 1 is supported.
        #[arg(long, default_value_t = 1)]
        pipeline_parallel_size: usize,

        // ─── Generation Defaults ────────────────────────────────────────
        /// Backend for guided (structured) decoding: outlines, lm-format-enforcer.
        #[arg(long, default_value = "outlines")]
        guided_decoding_backend: String,

        /// Maximum sequence length for CUDA graph capture.
        /// Sequences longer than this use eager execution.
        #[arg(long, default_value_t = 8192)]
        max_seq_len_to_capture: usize,

        /// Enable automatic tool choice for function calling.
        #[arg(long)]
        enable_auto_tool_choice: bool,

        /// Return token IDs alongside text in completion responses by default.
        #[arg(long)]
        return_tokens_as_token_ids: bool,
    },
    /// Generate text from prompts (CLI mode)
    Generate {
        /// Model ID (HuggingFace Hub format)
        #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
        model: String,

        /// Draft model ID for speculative decoding
        #[arg(long)]
        draft_model: Option<String>,

        /// Number of speculative tokens per step
        #[arg(long, default_value_t = 3)]
        num_speculative_tokens: usize,

        /// Prompt(s) to generate from
        #[arg(long)]
        prompt: Vec<String>,

        /// Maximum tokens to generate per prompt
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        /// Decode steps per scheduler invocation
        #[arg(long, default_value_t = 4)]
        multi_step_count: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config from file first
    let file_config = ServerConfig::load();
    if let Some(path) = ServerConfig::default_path() {
        if path.exists() {
            eprintln!("Loaded config from: {}", path.display());
        }
    }

    let cli = Cli::parse();

    match cli.command {
        Command::Serve {
            model,
            draft_model,
            num_speculative_tokens,
            port,
            host,
            num_blocks,
            max_requests,
            multi_step_count,
            lora_adapters,
            enable_prefix_caching,
            enable_chunked_prefill,
            shutdown_timeout,
            allowed_origins,
            allowed_methods,
            allowed_headers,
            max_body_size_mb,
            tool_call_parser,
            reasoning_parser,
            served_model_name,
            ssl_certfile,
            ssl_keyfile,
            dtype,
            quantization,
            gpu_memory_utilization,
            max_model_len,
            tensor_parallel_size,
            seed,
            trust_remote_code,
            max_lora_rank,
            tokenizer,
            revision,
            max_num_batched_tokens,
            scheduling_policy,
            disable_log_requests,
            chat_template,
            response_role,
            // KV Cache / Memory
            block_size,
            kv_cache_dtype,
            swap_space,
            cpu_offload_gb,
            num_gpu_blocks_override,
            enforce_eager,
            // Model Loading
            load_format,
            download_dir,
            tokenizer_mode,
            tokenizer_revision,
            code_revision,
            hf_token,
            max_parallel_loading_workers,
            // Scheduler Tuning
            max_num_seqs,
            preemption_mode,
            max_num_partial_prefills,
            long_prefill_token_threshold,
            stream_interval,
            // LoRA
            enable_lora,
            max_loras,
            lora_extra_vocab_size,
            lora_dtype,
            max_cpu_loras,
            // Speculative Decoding
            spec_decoding_acceptance_method,
            ngram_prompt_lookup_max,
            ngram_prompt_lookup_min,
            // Observability
            disable_log_stats,
            max_logprobs,
            otlp_traces_endpoint,
            log_level,
            // Multimodal
            limit_mm_per_prompt,
            disable_mm_preprocessor_cache,
            // Pipeline Parallelism
            pipeline_parallel_size,
            // Generation Defaults
            guided_decoding_backend,
            max_seq_len_to_capture,
            enable_auto_tool_choice,
            return_tokens_as_token_ids,
        } => {
            // Merge CLI args with file config (CLI takes precedence)
            let model = if model == "Qwen/Qwen3-0.6B" {
                file_config.model.unwrap_or(model)
            } else {
                model
            };
            let draft_model = draft_model.or(file_config.draft_model);
            let num_speculative_tokens = if num_speculative_tokens == 3 {
                file_config
                    .num_speculative_tokens
                    .unwrap_or(num_speculative_tokens)
            } else {
                num_speculative_tokens
            };
            let port = if port == 8000 {
                file_config.port.unwrap_or(port)
            } else {
                port
            };
            let host = if host == "0.0.0.0" {
                file_config.host.unwrap_or(host)
            } else {
                host
            };
            let num_blocks = if num_blocks == 512 {
                file_config.num_blocks.unwrap_or(num_blocks)
            } else {
                num_blocks
            };
            let max_requests = if max_requests == 8 {
                file_config.max_requests.unwrap_or(max_requests)
            } else {
                max_requests
            };
            let multi_step_count = if multi_step_count == 4 {
                file_config.multi_step_count.unwrap_or(multi_step_count)
            } else {
                multi_step_count
            };
            // Bool flags: CLI flag (true) takes precedence, otherwise fall back to file config
            let enable_prefix_caching =
                enable_prefix_caching || file_config.enable_prefix_caching.unwrap_or(false);
            let enable_chunked_prefill =
                enable_chunked_prefill || file_config.enable_chunked_prefill.unwrap_or(false);

            // CORS: CLI defaults are the wildcard values; fall back to file config
            // only when CLI has the default.
            let allowed_origins = if allowed_origins == "*" {
                file_config.allowed_origins.unwrap_or(allowed_origins)
            } else {
                allowed_origins
            };
            let allowed_methods = if allowed_methods == "GET,POST,OPTIONS" {
                file_config.allowed_methods.unwrap_or(allowed_methods)
            } else {
                allowed_methods
            };
            let allowed_headers = if allowed_headers == "*" {
                file_config.allowed_headers.unwrap_or(allowed_headers)
            } else {
                allowed_headers
            };

            let cors_config = api::CorsConfig {
                allowed_origins,
                allowed_methods,
                allowed_headers,
            };

            // Resolve served model name: CLI > config file > model identifier
            let served_model_name = served_model_name
                .or(file_config.served_model_name)
                .unwrap_or_else(|| model.clone());

            // TLS: CLI > config file
            let ssl_certfile = ssl_certfile.or(file_config.ssl_certfile);
            let ssl_keyfile = ssl_keyfile.or(file_config.ssl_keyfile);

            // dtype: CLI > config file
            let dtype = if dtype == "auto" {
                file_config.dtype.unwrap_or_else(|| "auto".to_string())
            } else {
                dtype
            };

            // quantization: CLI > config file
            let quantization = quantization.or(file_config.quantization);

            // gpu_memory_utilization: CLI > config file
            let gpu_memory_utilization =
                gpu_memory_utilization.or(file_config.gpu_memory_utilization);

            // max_model_len: CLI > config file
            let max_model_len = max_model_len.or(file_config.max_model_len);

            // tensor_parallel_size: CLI > config file
            let tensor_parallel_size = if tensor_parallel_size == 1 {
                file_config.tensor_parallel_size.unwrap_or(1)
            } else {
                tensor_parallel_size
            };

            // seed: CLI > config file
            let seed = if seed == 0 {
                file_config.seed.unwrap_or(0)
            } else {
                seed
            };

            // trust_remote_code: CLI flag or config file
            let _trust_remote_code =
                trust_remote_code || file_config.trust_remote_code.unwrap_or(false);

            // max_lora_rank: CLI > config file
            let max_lora_rank = if max_lora_rank == 64 {
                file_config.max_lora_rank.unwrap_or(64)
            } else {
                max_lora_rank
            };

            // tokenizer: CLI > config file
            let tokenizer_override = tokenizer.or(file_config.tokenizer);

            // revision: CLI > config file
            let revision = if revision == "main" {
                file_config.revision.unwrap_or(revision)
            } else {
                revision
            };

            // max_num_batched_tokens: CLI > config file
            let max_num_batched_tokens = if max_num_batched_tokens == 2048 {
                file_config
                    .max_num_batched_tokens
                    .or(file_config.max_tokens_per_step)
                    .unwrap_or(2048)
            } else {
                max_num_batched_tokens
            };

            // scheduling_policy: CLI > config file
            let scheduling_policy = if scheduling_policy == "fcfs" {
                file_config
                    .scheduling_policy
                    .unwrap_or_else(|| "fcfs".to_string())
            } else {
                scheduling_policy
            };

            // disable_log_requests: CLI flag or config file
            let disable_log_requests =
                disable_log_requests || file_config.disable_log_requests.unwrap_or(false);

            // chat_template: CLI > config file
            let chat_template_override = chat_template.or(file_config.chat_template_path);

            // response_role: CLI > config file
            let response_role = if response_role == "assistant" {
                file_config
                    .response_role
                    .unwrap_or_else(|| "assistant".to_string())
            } else {
                response_role
            };

            // ─── New args: CLI > config file merge ───────────────────

            // KV Cache / Memory
            let block_size = if block_size == 16 {
                file_config.block_size.unwrap_or(16)
            } else {
                block_size
            };
            let kv_cache_dtype = if kv_cache_dtype == "auto" {
                file_config
                    .kv_cache_dtype
                    .unwrap_or_else(|| "auto".to_string())
            } else {
                kv_cache_dtype
            };
            let swap_space = if swap_space == 0.0 {
                file_config.swap_space.unwrap_or(0.0)
            } else {
                swap_space
            };
            let cpu_offload_gb = if cpu_offload_gb == 0.0 {
                file_config.cpu_offload_gb.unwrap_or(0.0)
            } else {
                cpu_offload_gb
            };
            let num_gpu_blocks_override =
                num_gpu_blocks_override.or(file_config.num_gpu_blocks_override);
            let enforce_eager = enforce_eager || file_config.enforce_eager.unwrap_or(false);

            // Model Loading
            let load_format = if load_format == "auto" {
                file_config
                    .load_format
                    .unwrap_or_else(|| "auto".to_string())
            } else {
                load_format
            };
            let download_dir = download_dir.or(file_config.download_dir);
            let tokenizer_mode = if tokenizer_mode == "auto" {
                file_config
                    .tokenizer_mode
                    .unwrap_or_else(|| "auto".to_string())
            } else {
                tokenizer_mode
            };
            let tokenizer_revision = tokenizer_revision.or(file_config.tokenizer_revision);
            let code_revision = code_revision.or(file_config.code_revision);
            let max_parallel_loading_workers = if max_parallel_loading_workers == 1 {
                file_config.max_parallel_loading_workers.unwrap_or(1)
            } else {
                max_parallel_loading_workers
            };

            // Scheduler Tuning
            let max_num_seqs = max_num_seqs.or(file_config.max_num_seqs);
            let preemption_mode = if preemption_mode == "recompute" {
                file_config
                    .preemption_mode
                    .unwrap_or_else(|| "recompute".to_string())
            } else {
                preemption_mode
            };
            let max_num_partial_prefills =
                max_num_partial_prefills.or(file_config.max_num_partial_prefills);
            let long_prefill_token_threshold =
                long_prefill_token_threshold.or(file_config.long_prefill_token_threshold);
            let stream_interval = if stream_interval == 0 {
                file_config.stream_interval.unwrap_or(0)
            } else {
                stream_interval
            };

            // max_num_seqs overrides max_requests if provided
            let max_requests = max_num_seqs.unwrap_or(max_requests);

            // LoRA
            let enable_lora = enable_lora || file_config.enable_lora.unwrap_or(false);
            let max_loras = if max_loras == 1 {
                file_config.max_loras.unwrap_or(1)
            } else {
                max_loras
            };
            let lora_extra_vocab_size = if lora_extra_vocab_size == 256 {
                file_config.lora_extra_vocab_size.unwrap_or(256)
            } else {
                lora_extra_vocab_size
            };
            let lora_dtype = lora_dtype.or(file_config.lora_dtype);
            let max_cpu_loras = max_cpu_loras.or(file_config.max_cpu_loras);

            // Speculative Decoding
            let spec_decoding_acceptance_method =
                if spec_decoding_acceptance_method == "rejection_sampler" {
                    file_config
                        .spec_decoding_acceptance_method
                        .unwrap_or_else(|| "rejection_sampler".to_string())
                } else {
                    spec_decoding_acceptance_method
                };
            let ngram_prompt_lookup_max =
                ngram_prompt_lookup_max.or(file_config.ngram_prompt_lookup_max);
            let ngram_prompt_lookup_min =
                ngram_prompt_lookup_min.or(file_config.ngram_prompt_lookup_min);

            // Observability
            let disable_log_stats =
                disable_log_stats || file_config.disable_log_stats.unwrap_or(false);
            let max_logprobs = if max_logprobs == 20 {
                file_config.max_logprobs.unwrap_or(20)
            } else {
                max_logprobs
            };
            let otlp_traces_endpoint = otlp_traces_endpoint.or(file_config.otlp_traces_endpoint);
            let log_level = if log_level == "info" {
                file_config.log_level.unwrap_or_else(|| "info".to_string())
            } else {
                log_level
            };

            // Multimodal
            let limit_mm_per_prompt = limit_mm_per_prompt.or(file_config.limit_mm_per_prompt);
            let disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
                || file_config.disable_mm_preprocessor_cache.unwrap_or(false);

            // Pipeline Parallelism
            let pipeline_parallel_size = if pipeline_parallel_size == 1 {
                file_config.pipeline_parallel_size.unwrap_or(1)
            } else {
                pipeline_parallel_size
            };

            // Generation Defaults
            let guided_decoding_backend = if guided_decoding_backend == "outlines" {
                file_config
                    .guided_decoding_backend
                    .unwrap_or_else(|| "outlines".to_string())
            } else {
                guided_decoding_backend
            };
            let max_seq_len_to_capture = if max_seq_len_to_capture == 8192 {
                file_config.max_seq_len_to_capture.unwrap_or(8192)
            } else {
                max_seq_len_to_capture
            };
            let enable_auto_tool_choice =
                enable_auto_tool_choice || file_config.enable_auto_tool_choice.unwrap_or(false);
            let return_tokens_as_token_ids = return_tokens_as_token_ids
                || file_config.return_tokens_as_token_ids.unwrap_or(false);

            // Validate tensor_parallel_size
            if tensor_parallel_size != 1 {
                anyhow::bail!(
                    "--tensor-parallel-size {} is not yet supported (only 1 is currently implemented)",
                    tensor_parallel_size
                );
            }

            // Parse scheduling policy
            let sched_policy = match scheduling_policy.as_str() {
                "fcfs" => vllm_core::scheduler::SchedulingPolicy::Fcfs,
                "priority" => vllm_core::scheduler::SchedulingPolicy::Priority,
                other => anyhow::bail!(
                    "Unknown scheduling policy '{}'. Supported: fcfs, priority",
                    other
                ),
            };

            // Parse preemption mode
            let parsed_preemption_mode = match preemption_mode.as_str() {
                "recompute" => PreemptionMode::Recompute,
                "swap" => PreemptionMode::Swap,
                other => anyhow::bail!(
                    "Unknown preemption mode '{}'. Supported: recompute, swap",
                    other
                ),
            };

            // Parse spec-decode acceptance method
            let parsed_acceptance_method =
                match spec_decoding_acceptance_method.to_lowercase().as_str() {
                    "rejection_sampler" | "rejection" => AcceptanceMethod::RejectionSampler,
                    "typical_acceptance_sampler" | "typical_acceptance" | "typical" => {
                        AcceptanceMethod::TypicalAcceptance {
                            posterior_threshold: 0.09,
                            posterior_alpha: 0.3,
                        }
                    }
                    other => anyhow::bail!(
                        "Unknown spec_decoding_acceptance_method '{}'. \
                         Supported: rejection_sampler, typical_acceptance_sampler",
                        other
                    ),
                };

            // Parse dtype
            let compute_dtype = match dtype.as_str() {
                "auto" | "bf16" | "bfloat16" => DType::BF16,
                "fp16" | "float16" | "half" => DType::F16,
                "fp32" | "float32" | "float" => DType::F32,
                other => anyhow::bail!(
                    "Unknown dtype '{}'. Supported: auto, bf16, fp16, fp32",
                    other
                ),
            };

            run_server(ServerLaunchConfig {
                model_id: model,
                draft_model_id: draft_model,
                num_speculative_tokens,
                host,
                port,
                num_blocks,
                max_requests,
                multi_step_count,
                lora_adapters,
                enable_prefix_caching,
                enable_chunked_prefill,
                shutdown_timeout,
                cors_config,
                max_body_size_mb,
                tool_call_parser,
                reasoning_parser,
                served_model_name,
                ssl_certfile,
                ssl_keyfile,
                dtype: compute_dtype,
                quantization,
                gpu_memory_utilization,
                max_model_len,
                seed,
                max_lora_rank,
                tokenizer_override,
                revision,
                max_num_batched_tokens,
                scheduling_policy: sched_policy,
                disable_log_requests,
                chat_template_override,
                response_role,
                // New args
                block_size,
                kv_cache_dtype,
                swap_space,
                cpu_offload_gb,
                num_gpu_blocks_override,
                enforce_eager,
                load_format,
                download_dir,
                tokenizer_mode,
                tokenizer_revision,
                code_revision,
                hf_token,
                max_parallel_loading_workers,
                preemption_mode: parsed_preemption_mode,
                max_num_partial_prefills,
                long_prefill_token_threshold,
                acceptance_method: parsed_acceptance_method,
                stream_interval,
                enable_lora,
                max_loras,
                lora_extra_vocab_size,
                lora_dtype,
                max_cpu_loras,
                ngram_prompt_lookup_max,
                ngram_prompt_lookup_min,
                disable_log_stats,
                max_logprobs,
                otlp_traces_endpoint,
                log_level,
                limit_mm_per_prompt,
                disable_mm_preprocessor_cache,
                guided_decoding_backend,
                max_seq_len_to_capture,
                enable_auto_tool_choice,
                return_tokens_as_token_ids,
                pipeline_parallel_size,
            })
            .await
        }
        Command::Generate {
            model,
            draft_model,
            num_speculative_tokens,
            prompt,
            max_tokens,
            multi_step_count,
        } => {
            let prompts = if prompt.is_empty() {
                vec!["Hello, world".to_string()]
            } else {
                prompt
            };
            run_generate(
                model,
                draft_model,
                num_speculative_tokens,
                prompts,
                max_tokens,
                multi_step_count,
            )
            .await
        }
    }
}

struct ServerLaunchConfig {
    model_id: String,
    draft_model_id: Option<String>,
    num_speculative_tokens: usize,
    host: String,
    port: u16,
    num_blocks: usize,
    max_requests: usize,
    multi_step_count: usize,
    lora_adapters: Vec<String>,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    shutdown_timeout: u64,
    cors_config: api::CorsConfig,
    max_body_size_mb: usize,
    tool_call_parser: String,
    reasoning_parser: String,
    served_model_name: String,
    ssl_certfile: Option<String>,
    ssl_keyfile: Option<String>,
    dtype: DType,
    quantization: Option<String>,
    gpu_memory_utilization: Option<f32>,
    max_model_len: Option<usize>,
    seed: u64,
    max_lora_rank: usize,
    tokenizer_override: Option<String>,
    revision: String,
    max_num_batched_tokens: usize,
    scheduling_policy: vllm_core::scheduler::SchedulingPolicy,
    disable_log_requests: bool,
    chat_template_override: Option<String>,
    response_role: String,
    // KV Cache / Memory
    block_size: usize,
    kv_cache_dtype: String,
    swap_space: f32,
    cpu_offload_gb: f32,
    num_gpu_blocks_override: Option<usize>,
    enforce_eager: bool,
    // Model Loading
    load_format: String,
    download_dir: Option<String>,
    tokenizer_mode: String,
    tokenizer_revision: Option<String>,
    code_revision: Option<String>,
    hf_token: Option<String>,
    max_parallel_loading_workers: usize,
    // Scheduler Tuning
    preemption_mode: PreemptionMode,
    max_num_partial_prefills: Option<usize>,
    long_prefill_token_threshold: Option<usize>,
    // Spec-decode acceptance
    acceptance_method: AcceptanceMethod,
    stream_interval: usize,
    // LoRA
    enable_lora: bool,
    max_loras: usize,
    lora_extra_vocab_size: usize,
    lora_dtype: Option<String>,
    max_cpu_loras: Option<usize>,
    // Speculative Decoding
    ngram_prompt_lookup_max: Option<usize>,
    ngram_prompt_lookup_min: Option<usize>,
    // Observability
    disable_log_stats: bool,
    max_logprobs: usize,
    otlp_traces_endpoint: Option<String>,
    log_level: String,
    // Multimodal
    limit_mm_per_prompt: Option<String>,
    disable_mm_preprocessor_cache: bool,
    // Generation Defaults
    guided_decoding_backend: String,
    max_seq_len_to_capture: usize,
    enable_auto_tool_choice: bool,
    return_tokens_as_token_ids: bool,
    // Parallelism
    pipeline_parallel_size: usize,
}

async fn run_server(cfg: ServerLaunchConfig) -> anyhow::Result<()> {
    logging::init_with_level(&cfg.log_level);
    prometheus::init_metrics();

    let ServerLaunchConfig {
        model_id,
        draft_model_id,
        num_speculative_tokens,
        host,
        port,
        mut num_blocks,
        max_requests,
        multi_step_count,
        lora_adapters,
        enable_prefix_caching,
        enable_chunked_prefill,
        shutdown_timeout,
        cors_config,
        max_body_size_mb,
        tool_call_parser,
        reasoning_parser,
        served_model_name,
        ssl_certfile,
        ssl_keyfile,
        dtype,
        quantization,
        gpu_memory_utilization,
        max_model_len: max_model_len_override,
        seed,
        max_lora_rank,
        tokenizer_override,
        revision,
        max_num_batched_tokens,
        scheduling_policy,
        disable_log_requests,
        chat_template_override,
        response_role,
        // New args
        block_size,
        kv_cache_dtype,
        swap_space,
        cpu_offload_gb,
        num_gpu_blocks_override,
        enforce_eager,
        load_format,
        download_dir,
        tokenizer_mode,
        tokenizer_revision,
        code_revision,
        hf_token,
        max_parallel_loading_workers,
        preemption_mode,
        max_num_partial_prefills,
        long_prefill_token_threshold,
        acceptance_method,
        stream_interval,
        enable_lora,
        max_loras,
        lora_extra_vocab_size,
        lora_dtype,
        max_cpu_loras,
        ngram_prompt_lookup_max,
        ngram_prompt_lookup_min,
        disable_log_stats,
        max_logprobs,
        otlp_traces_endpoint,
        log_level: _, // already consumed above via cfg.log_level
        limit_mm_per_prompt,
        disable_mm_preprocessor_cache,
        guided_decoding_backend,
        max_seq_len_to_capture,
        enable_auto_tool_choice,
        return_tokens_as_token_ids,
        pipeline_parallel_size,
    } = cfg;

    if seed != 0 {
        eprintln!("Using random seed: {seed}");
    }

    // Acknowledge new args (wire as features are implemented)
    let _ = &kv_cache_dtype; // Used below in CacheConfig
    let _ = enforce_eager; // TODO: wire to CUDA graph control
    let _ = &load_format; // TODO: wire to weight loading strategy
    let _ = &tokenizer_mode; // TODO: wire to tokenizer selection
    let _ = &code_revision; // TODO: wire to custom code loading
    let _ = max_parallel_loading_workers; // TODO: wire to parallel weight loading
                                          // max_loras: wired to SchedulerConfig.max_loras_per_batch below
    let _ = lora_extra_vocab_size; // TODO: wire to LoRA vocab extension
    let _ = &otlp_traces_endpoint; // TODO: wire to OpenTelemetry
                                   // Parse --limit-mm-per-prompt JSON ("{"image": 5, "video": 1}") into per-modality limits.
    let mm_limits: std::collections::HashMap<String, usize> =
        if let Some(ref json_str) = limit_mm_per_prompt {
            serde_json::from_str(json_str).map_err(|e| {
                anyhow::anyhow!(
                    "--limit-mm-per-prompt must be a JSON object like '{{\"image\": 5}}': {}",
                    e
                )
            })?
        } else {
            std::collections::HashMap::new()
        };
    let _ = disable_mm_preprocessor_cache; // TODO: wire to VLM cache
    let _ = &guided_decoding_backend; // TODO: wire to structured output backend
    let _ = max_seq_len_to_capture; // TODO: wire to CUDA graph capture

    // Pipeline parallelism validation.
    // TODO: wire pipeline_parallel_size to PipelineStagedModel stage construction.
    // Until stage-slicing is wired, reject pp > 1 gracefully.
    if pipeline_parallel_size > 1 {
        anyhow::bail!(
            "--pipeline-parallel-size {} is not yet supported (only 1 is currently implemented)",
            pipeline_parallel_size
        );
    }

    eprintln!("Loading model: {model_id}");
    let cache_dir = download_dir.as_deref().map(std::path::Path::new);
    let files =
        loader::fetch_model_with_auth(&model_id, &revision, hf_token.as_deref(), cache_dir)?;

    let device = Device::new_cuda(0)?;
    let dtype_label = match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "fp16",
        DType::F32 => "fp32",
        _ => "unknown",
    };

    // Handle quantization override
    let use_quantized = match quantization.as_deref() {
        Some("none") => false,
        Some(_) => true,
        None => loader::is_quantized(&files),
    };

    eprintln!("Loading weights to GPU ({dtype_label})...");
    let vb = loader::load_weights(&files.weights, dtype, &device)?;

    // Validate max_cpu_loras against the number of startup adapters.
    // Dynamic LoRA loading with an LRU cache is not yet implemented; this
    // check ensures the user-supplied limit is not violated at startup.
    // TODO: implement a proper LRU CPU adapter cache for dynamic loading.
    if let Some(max_cpu) = max_cpu_loras {
        if lora_adapters.len() > max_cpu {
            anyhow::bail!(
                "--max-cpu-loras {} is less than the number of adapters being loaded ({}). \
                 Either raise the limit or reduce the number of startup adapters.",
                max_cpu,
                lora_adapters.len()
            );
        }
    }

    // Parse LoRA adapter specs first
    let parsed_lora_specs: Vec<(&str, &str)> = lora_adapters
        .iter()
        .map(|spec| {
            let parts: Vec<&str> = spec.splitn(2, '=').collect();
            if parts.len() != 2 {
                anyhow::bail!(
                    "Invalid LoRA adapter spec '{}': expected format 'name=path'",
                    spec
                );
            }
            Ok((parts[0], parts[1]))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Build model - use LoRA-enabled variant if adapters specified
    eprintln!(
        "Building model ({} layers, quant={})...",
        files.config.num_hidden_layers,
        if use_quantized {
            loader::quantization_info(&files)
        } else {
            "none".to_string()
        },
    );

    let model: Box<dyn vllm_core::engine::ModelForward> = if !parsed_lora_specs.is_empty() {
        // Create LoRA-enabled model and register adapters
        let mut lora_model = models::from_config_with_lora(&files.config, vb)?;

        eprintln!("Loading {} LoRA adapter(s)...", parsed_lora_specs.len());
        // Use explicit --lora-dtype if specified; fall back to model dtype.
        let lora_load_dtype = match lora_dtype.as_deref() {
            Some("float16") | Some("fp16") | Some("half") => DType::F16,
            Some("bfloat16") | Some("bf16") => DType::BF16,
            Some("float32") | Some("fp32") | Some("float") => DType::F32,
            Some(other) => {
                anyhow::bail!(
                    "--lora-dtype '{}' is not supported; use float16, bfloat16, or float32",
                    other
                );
            }
            None => dtype,
        };
        let lora_loader = LoraLoader::new(device.clone(), lora_load_dtype);

        for (idx, (name, path)) in parsed_lora_specs.iter().enumerate() {
            eprintln!("  Loading adapter '{}' from: {}", name, path);
            let adapter = lora_loader.load(path, *name, (idx + 1) as u32)?;
            if adapter.rank > max_lora_rank {
                anyhow::bail!(
                    "LoRA adapter '{}' rank {} exceeds --max-lora-rank {}",
                    name,
                    adapter.rank,
                    max_lora_rank
                );
            }
            eprintln!(
                "    Loaded {} layer adapters (rank={}, alpha={})",
                adapter.num_adapters(),
                adapter.rank,
                adapter.alpha
            );
            lora_model.register_lora(&adapter);
        }

        eprintln!("Registered adapters: {:?}", lora_model.lora_adapters());
        Box::new(lora_model)
    } else {
        // Create regular model without LoRA
        models::from_config(&files.config, vb)?
    };

    // Resolve tokenizer: CLI override > tokenizer_revision > model default.
    // --tokenizer-revision allows fetching tokenizer files at a different
    // revision than the model weights (useful when a tokenizer is updated
    // independently of model weights).
    let tokenizer_path = if let Some(ref tok_override) = tokenizer_override {
        let tok_path = std::path::Path::new(tok_override);
        if tok_path.exists() {
            tok_path.to_path_buf()
        } else {
            // Try as HuggingFace model ID; use tokenizer_revision if specified.
            let tok_rev = tokenizer_revision.as_deref().unwrap_or("main");
            let tok_files = loader::fetch_model_with_auth(
                tok_override,
                tok_rev,
                hf_token.as_deref(),
                cache_dir,
            )?;
            tok_files.tokenizer
        }
    } else if let Some(ref tok_rev) = tokenizer_revision {
        // No override but a different tokenizer revision was requested;
        // re-fetch just the tokenizer files at the specified revision.
        let tok_files =
            loader::fetch_model_with_auth(&model_id, tok_rev, hf_token.as_deref(), cache_dir)?;
        tok_files.tokenizer
    } else {
        files.tokenizer.clone()
    };

    let tokenizer = TokenizerWrapper::from_file(&tokenizer_path)?;
    let tokenizer = Arc::new(tokenizer);

    // Resolve chat template: CLI override > tokenizer_config auto-detect
    let chat_template = if let Some(ref template_path) = chat_template_override {
        let path = std::path::Path::new(template_path);
        Some(Arc::new(
            ChatTemplateEngine::from_tokenizer_config(path).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load chat template from '{}': {}",
                    template_path,
                    e
                )
            })?,
        ))
    } else {
        files
            .tokenizer_config
            .as_ref()
            .and_then(|path| ChatTemplateEngine::from_tokenizer_config(path).ok())
            .map(Arc::new)
    };

    // Apply num_gpu_blocks_override if specified
    if let Some(override_blocks) = num_gpu_blocks_override {
        num_blocks = override_blocks;
        eprintln!("Using GPU blocks override: {num_blocks}");
    }

    // Compute num_blocks from GPU memory utilization if specified
    if let Some(utilization) = gpu_memory_utilization {
        if !(0.0..=1.0).contains(&utilization) {
            anyhow::bail!(
                "--gpu-memory-utilization must be between 0.0 and 1.0, got {utilization}"
            );
        }
        let kv_budget = estimate_kv_cache_budget(utilization, &files.config, dtype)?;
        let computed_blocks = CacheConfig::from_memory_budget(
            kv_budget,
            files.config.num_hidden_layers,
            files.config.num_key_value_heads,
            files.config.head_dim,
            block_size,
            dtype,
            device.clone(),
        );
        eprintln!(
            "GPU memory utilization {:.0}%: estimated {:.0} MiB for KV cache → {} blocks",
            utilization * 100.0,
            kv_budget as f64 / (1024.0 * 1024.0),
            computed_blocks.num_blocks,
        );
        num_blocks = computed_blocks.num_blocks;
    }

    // Parse kv_cache_dtype
    let parsed_kv_cache_dtype = match kv_cache_dtype.as_str() {
        "auto" => KVCacheDtype::Auto,
        "fp8" | "fp8_e4m3" | "fp8_e5m2" => KVCacheDtype::Fp8E4m3,
        other => anyhow::bail!(
            "Unknown kv-cache-dtype '{}'. Supported: auto, fp8, fp8_e4m3, fp8_e5m2",
            other
        ),
    };

    // Compute CPU offload config from swap_space or cpu_offload_gb
    let cpu_offload_config = {
        let offload_gb = if cpu_offload_gb > 0.0 {
            cpu_offload_gb
        } else {
            swap_space
        };
        if offload_gb > 0.0 {
            // Estimate max_cpu_blocks from GiB budget
            let bytes = (offload_gb * 1024.0 * 1024.0 * 1024.0) as usize;
            let bytes_per_block = 2
                * files.config.num_hidden_layers
                * files.config.num_key_value_heads
                * files.config.head_dim
                * block_size
                * dtype.size_in_bytes();
            let max_blocks = if bytes_per_block > 0 {
                bytes / bytes_per_block
            } else {
                256
            };
            Some(vllm_core::kv_cache::offload::CpuOffloadConfig {
                max_cpu_blocks: max_blocks,
                use_pinned_memory: false,
                prefetch_count: 2,
            })
        } else {
            None
        }
    };

    let cache_config = CacheConfig {
        block_size,
        num_blocks,
        num_layers: files.config.num_hidden_layers,
        num_kv_heads: files.config.num_key_value_heads,
        head_dim: files.config.head_dim,
        dtype,
        device: device.clone(),
        kv_cache_dtype: parsed_kv_cache_dtype,
        cpu_offload: cpu_offload_config,
    };
    eprintln!(
        "Allocating KV cache ({} blocks)...",
        cache_config.num_blocks
    );
    let kv_cache_mgr = KVCacheManager::new(&cache_config)?;

    let eos_token_id = files.config.eos_token_id;
    let engine_tokenizer = TokenizerWrapper::from_file(&tokenizer_path)?;

    // Build speculative config once; acceptance_method is always carried through.
    let spec_config_base = SpeculativeConfig {
        num_speculative_tokens,
        acceptance_method,
    };

    // Build scheduler config once; reused for all engine start paths.
    let scheduler_config = SchedulerConfig {
        max_running_requests: max_requests,
        max_tokens_per_step: max_num_batched_tokens,
        enable_chunked_prefill,
        scheduling_policy,
        max_loras_per_batch: max_loras,
        preemption_mode,
        max_num_partial_prefills: max_num_partial_prefills.unwrap_or(1),
        long_prefill_token_threshold: long_prefill_token_threshold.unwrap_or(0),
    };

    let handle = if let Some(ref draft_id) = draft_model_id {
        eprintln!("Loading draft model: {draft_id}");
        let draft_files =
            loader::fetch_model_with_auth(draft_id, "main", hf_token.as_deref(), cache_dir)?;

        eprintln!("Loading draft weights to GPU (bf16)...");
        let draft_vb = loader::load_weights(&draft_files.weights, dtype, &device)?;

        let draft_arch = draft_files
            .config
            .architectures
            .first()
            .map(|s| s.as_str())
            .unwrap_or("");

        if draft_arch == "MLPSpeculatorPreTrainedModel" {
            // MLP Speculator: stateless proposer backed by hidden-state MLP heads.
            // No draft KV cache required — uses target model's hidden states directly.
            eprintln!("Building MLP Speculator draft model...");
            let mlp_model = models::mlp_speculator_from_config(&draft_files.config, draft_vb)?;
            let proposer = MLPSpeculatorDraftProposer::new(mlp_model);

            let engine_config = EngineConfig::builder(scheduler_config, Some(spec_config_base))
                .enable_prefix_caching(enable_prefix_caching)
                .build();

            eprintln!("Starting engine (MLP Speculator, K={num_speculative_tokens})...");
            start_engine_with_proposer(
                model,
                Box::new(proposer),
                engine_tokenizer,
                kv_cache_mgr,
                engine_config,
            )
        } else {
            eprintln!(
                "Building draft model ({} layers)...",
                draft_files.config.num_hidden_layers
            );
            let draft_model = models::from_config(&draft_files.config, draft_vb)?;

            let draft_cache_config = CacheConfig {
                block_size: 16,
                num_blocks,
                num_layers: draft_files.config.num_hidden_layers,
                num_kv_heads: draft_files.config.num_key_value_heads,
                head_dim: draft_files.config.head_dim,
                dtype,
                device: device.clone(),
                kv_cache_dtype: KVCacheDtype::Auto,
                cpu_offload: None,
            };
            eprintln!(
                "Allocating draft KV cache ({} blocks)...",
                draft_cache_config.num_blocks
            );
            let draft_kv_cache = KVCacheManager::new(&draft_cache_config)?;

            let engine_config = EngineConfig::builder(scheduler_config, Some(spec_config_base))
                .enable_prefix_caching(enable_prefix_caching)
                .build();

            eprintln!("Starting engine (speculative, K={num_speculative_tokens})...");
            start_engine_with_draft(
                model,
                draft_model,
                engine_tokenizer,
                kv_cache_mgr,
                draft_kv_cache,
                engine_config,
            )
        }
    } else if let Some(max_n) = ngram_prompt_lookup_max {
        // N-gram prompt-lookup speculative decoding (no draft model required)
        let min_n = ngram_prompt_lookup_min.unwrap_or(1);
        let ngram_config = NGramConfig {
            min_n,
            max_n,
            num_speculative_tokens,
        };
        let engine_config = EngineConfig::builder(scheduler_config, Some(spec_config_base))
            .multi_step_count(multi_step_count)
            .enable_prefix_caching(enable_prefix_caching)
            .build();

        eprintln!(
            "Starting engine (NGram speculative, K={num_speculative_tokens}, n={min_n}..{max_n})..."
        );
        start_engine_with_proposer(
            model,
            Box::new(NGramProposer::new(ngram_config)),
            engine_tokenizer,
            kv_cache_mgr,
            engine_config,
        )
    } else {
        let engine_config = EngineConfig::builder(scheduler_config, None)
            .multi_step_count(multi_step_count)
            .enable_prefix_caching(enable_prefix_caching)
            .build();

        eprintln!("Starting engine (multi-step={multi_step_count})...");
        start_engine(model, engine_tokenizer, kv_cache_mgr, engine_config)
    };

    let (atomic_engine, engine_controller) = AtomicEngineHandle::new(handle);
    let accepting = Arc::new(AtomicBool::new(true));
    let engine_builder: Arc<ProductionEngineBuilder> = Arc::new(ProductionEngineBuilder);

    // Periodic engine stats logging. Logs running/waiting queue depths and
    // KV cache occupancy every 5 seconds. Suppressed by --disable-log-stats.
    if !disable_log_stats {
        let stats_handle = atomic_engine.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                if let Ok(stats) = stats_handle.get().get_stats().await {
                    let used = stats.num_total_blocks.saturating_sub(stats.num_free_blocks);
                    let pct = if stats.num_total_blocks > 0 {
                        100.0 * used as f64 / stats.num_total_blocks as f64
                    } else {
                        0.0
                    };
                    tracing::info!(
                        num_running = stats.num_running_requests,
                        num_waiting = stats.num_waiting_requests,
                        gpu_cache_usage_perc = format_args!("{:.1}", pct),
                        "Engine stats",
                    );
                }
            }
        });
    }

    // max_model_len: CLI override > model's max_position_embeddings > blocks * block_size
    let default_max_model_len = std::cmp::min(
        num_blocks * block_size,
        files.config.max_position_embeddings,
    );
    let max_model_len = max_model_len_override.unwrap_or(default_max_model_len);
    let state = AppState::new(
        atomic_engine.clone(),
        served_model_name,
        tokenizer,
        chat_template,
        eos_token_id,
        max_model_len,
        api::create_tool_call_parser(&tool_call_parser),
        api::create_reasoning_parser_arc(&reasoning_parser),
        accepting.clone(),
        response_role.clone(),
        enable_auto_tool_choice,
        return_tokens_as_token_ids,
        max_logprobs,
        mm_limits,
        stream_interval,
        enable_lora,
    );

    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let runtime_config = RuntimeConfig {
        model: model_id.clone(),
        draft_model: draft_model_id.clone(),
        num_speculative_tokens,
        num_blocks,
        block_size,
        max_requests,
        max_tokens_per_step: max_num_batched_tokens,
        enable_chunked_prefill,
        multi_step_count,
        enable_prefix_caching,
        dtype: dtype_label.to_string(),
        device: "cuda:0".to_string(),
    };

    let admin_state = AdminState::new(
        atomic_engine.clone(),
        engine_controller,
        model_id.clone(),
        start_time,
        runtime_config,
        accepting,
        engine_builder,
    );

    let cors_layer = api::build_cors_layer(&cors_config);
    let max_body_size = max_body_size_mb * 1024 * 1024;
    let app = api::create_full_router_with_options(
        state,
        admin_state,
        cors_layer,
        api::middleware::RateLimitState::unlimited(),
        max_body_size,
        !disable_log_requests,
    );
    let addr = format!("{host}:{port}");

    // Branch: TLS (axum-server with rustls) vs plain HTTP (axum::serve)
    match (ssl_certfile, ssl_keyfile) {
        (Some(certfile), Some(keyfile)) => {
            let tls_config =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&certfile, &keyfile)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to load TLS config: {e}"))?;

            let socket_addr: std::net::SocketAddr = addr
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid bind address '{addr}': {e}"))?;

            tracing::info!("Serving on https://{addr}/v1 (TLS)");
            tracing::info!("Admin panel: https://{addr}/admin/");

            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown_signal().await;
                shutdown_handle
                    .graceful_shutdown(Some(std::time::Duration::from_secs(shutdown_timeout)));
            });

            axum_server::bind_rustls(socket_addr, tls_config)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        (Some(_), None) => {
            anyhow::bail!("--ssl-certfile requires --ssl-keyfile");
        }
        (None, Some(_)) => {
            anyhow::bail!("--ssl-keyfile requires --ssl-certfile");
        }
        (None, None) => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            tracing::info!("Serving on http://{addr}/v1");
            tracing::info!("Admin panel: http://{addr}/admin/");

            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
    }

    tracing::info!(
        timeout_secs = shutdown_timeout,
        "Server stopped accepting connections, waiting for in-flight requests to complete"
    );

    // Give in-flight engine work a bounded window to finish before forcing shutdown.
    match tokio::time::timeout(
        std::time::Duration::from_secs(shutdown_timeout),
        atomic_engine.get().shutdown(),
    )
    .await
    {
        Ok(Ok(())) => {
            tracing::info!("Engine shut down cleanly");
        }
        Ok(Err(e)) => {
            tracing::error!("Engine shutdown returned an error: {e}");
        }
        Err(_) => {
            tracing::warn!(
                timeout_secs = shutdown_timeout,
                "Engine shutdown timed out, forcing exit"
            );
        }
    }

    tracing::info!("Shutdown complete");
    Ok(())
}

/// Estimate available GPU memory for KV cache given memory utilization target.
///
/// Queries total VRAM, estimates model size from config, and returns the
/// memory budget available for KV cache allocation.
fn estimate_kv_cache_budget(
    utilization: f32,
    config: &vllm_core::config::ModelConfig,
    dtype: DType,
) -> anyhow::Result<usize> {
    #[cfg(feature = "cuda")]
    {
        let (_free, total_vram) = vllm_core::kv_cache::config::gpu_memory_info()?;

        // Rough model size estimate:
        // params ≈ vocab * hidden + layers * (4 * hidden^2 + 3 * hidden * intermediate)
        let h = config.hidden_size;
        let v = config.vocab_size;
        let l = config.num_hidden_layers;
        let i = config.intermediate_size;
        let estimated_params = v * h + l * (4 * h * h + 3 * h * i);

        let kv_budget = vllm_core::kv_cache::config::estimate_kv_budget_bytes(
            total_vram,
            utilization,
            estimated_params,
            dtype,
        );

        if kv_budget == 0 {
            let bytes_per_param = if matches!(dtype, DType::F32) { 4 } else { 2 };
            let model_bytes = estimated_params * bytes_per_param;
            anyhow::bail!(
                "Insufficient GPU memory: {:.0} MiB usable ({:.0}% of {:.0} MiB), \
                 estimated model size {:.0} MiB",
                (total_vram as f64 * utilization as f64) / (1024.0 * 1024.0),
                utilization * 100.0,
                total_vram as f64 / (1024.0 * 1024.0),
                model_bytes as f64 / (1024.0 * 1024.0),
            );
        }

        Ok(kv_budget)
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (utilization, config, dtype);
        anyhow::bail!("--gpu-memory-utilization requires the 'cuda' feature")
    }
}

async fn run_generate(
    model_id: String,
    draft_model_id: Option<String>,
    num_speculative_tokens: usize,
    prompts: Vec<String>,
    max_tokens: usize,
    multi_step_count: usize,
) -> anyhow::Result<()> {
    eprintln!("Loading model: {model_id}");
    let files = loader::fetch_model(&model_id)?;

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    eprintln!("Loading weights to GPU (bf16)...");
    let vb = loader::load_weights(&files.weights, dtype, &device)?;

    eprintln!(
        "Building model ({} layers)...",
        files.config.num_hidden_layers
    );
    let model = models::from_config(&files.config, vb)?;

    let tokenizer = TokenizerWrapper::from_file(&files.tokenizer)?;

    let cache_config = CacheConfig {
        block_size: 16,
        num_blocks: 512,
        num_layers: files.config.num_hidden_layers,
        num_kv_heads: files.config.num_key_value_heads,
        head_dim: files.config.head_dim,
        dtype,
        device: device.clone(),
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    };
    eprintln!(
        "Allocating KV cache ({} blocks)...",
        cache_config.num_blocks
    );
    let kv_cache_mgr = KVCacheManager::new(&cache_config)?;

    let eos_token_id = files.config.eos_token_id;

    let handle = if let Some(ref draft_id) = draft_model_id {
        eprintln!("Loading draft model: {draft_id}");
        let draft_files = loader::fetch_model(draft_id)?;

        eprintln!("Loading draft weights to GPU (bf16)...");
        let draft_vb = loader::load_weights(&draft_files.weights, dtype, &device)?;

        let draft_arch = draft_files
            .config
            .architectures
            .first()
            .map(|s| s.as_str())
            .unwrap_or("");

        if draft_arch == "MLPSpeculatorPreTrainedModel" {
            eprintln!("Building MLP Speculator draft model...");
            let mlp_model = models::mlp_speculator_from_config(&draft_files.config, draft_vb)?;
            let proposer = MLPSpeculatorDraftProposer::new(mlp_model);

            let engine_config = EngineConfig::builder(
                SchedulerConfig {
                    max_running_requests: 8,
                    max_tokens_per_step: 2048,
                    enable_chunked_prefill: false,
                    scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                    max_loras_per_batch: 0,
                    ..SchedulerConfig::default()
                },
                Some(SpeculativeConfig {
                    num_speculative_tokens,
                    acceptance_method: AcceptanceMethod::RejectionSampler,
                }),
            )
            .build();

            eprintln!(
                "Starting engine ({} prompts, max {} tokens each, MLP Speculator K={})...",
                prompts.len(),
                max_tokens,
                num_speculative_tokens,
            );
            start_engine_with_proposer(
                model,
                Box::new(proposer),
                tokenizer,
                kv_cache_mgr,
                engine_config,
            )
        } else {
            eprintln!(
                "Building draft model ({} layers)...",
                draft_files.config.num_hidden_layers
            );
            let draft_model = models::from_config(&draft_files.config, draft_vb)?;

            let draft_cache_config = CacheConfig {
                block_size: 16,
                num_blocks: 512,
                num_layers: draft_files.config.num_hidden_layers,
                num_kv_heads: draft_files.config.num_key_value_heads,
                head_dim: draft_files.config.head_dim,
                dtype,
                device: device.clone(),
                kv_cache_dtype: KVCacheDtype::Auto,
                cpu_offload: None,
            };
            eprintln!(
                "Allocating draft KV cache ({} blocks)...",
                draft_cache_config.num_blocks
            );
            let draft_kv_cache = KVCacheManager::new(&draft_cache_config)?;

            let engine_config = EngineConfig::builder(
                SchedulerConfig {
                    max_running_requests: 8,
                    max_tokens_per_step: 2048,
                    enable_chunked_prefill: false,
                    scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                    max_loras_per_batch: 0,
                    ..SchedulerConfig::default()
                },
                Some(SpeculativeConfig {
                    num_speculative_tokens,
                    acceptance_method: AcceptanceMethod::RejectionSampler,
                }),
            )
            .build();

            eprintln!(
                "Starting engine ({} prompts, max {} tokens each, speculative K={})...",
                prompts.len(),
                max_tokens,
                num_speculative_tokens,
            );
            start_engine_with_draft(
                model,
                draft_model,
                tokenizer,
                kv_cache_mgr,
                draft_kv_cache,
                engine_config,
            )
        }
    } else {
        let engine_config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                max_loras_per_batch: 0,
                ..SchedulerConfig::default()
            },
            None,
        )
        .multi_step_count(multi_step_count)
        .build();

        eprintln!(
            "Starting engine ({} prompts, max {} tokens each, multi-step={})...",
            prompts.len(),
            max_tokens,
            multi_step_count
        );
        start_engine(model, tokenizer, kv_cache_mgr, engine_config)
    };

    let mut tasks = Vec::new();
    for prompt in &prompts {
        let h = handle.clone();
        let req = GenerationRequest {
            prompt: prompt.clone(),
            max_new_tokens: max_tokens,
            eos_token_id,
            ..Default::default()
        };
        tasks.push(tokio::spawn(async move {
            (req.prompt.clone(), h.generate(req).await)
        }));
    }

    for task in tasks {
        let (prompt, result) = task.await?;
        match result {
            Ok(gen) => println!("{prompt}{}", gen.generated_text),
            Err(e) => eprintln!("Error for prompt \"{prompt}\": {e}"),
        }
    }

    handle.shutdown().await?;
    Ok(())
}
