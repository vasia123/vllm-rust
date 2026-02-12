use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use vllm_core::{
    engine::{
        start_engine, start_engine_with_draft, EngineConfig, GenerationRequest, SpeculativeConfig,
    },
    kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager},
    loader,
    lora::LoraLoader,
    models,
    scheduler::SchedulerConfig,
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
}

async fn run_server(cfg: ServerLaunchConfig) -> anyhow::Result<()> {
    logging::init();
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
    } = cfg;

    if seed != 0 {
        eprintln!("Using random seed: {seed}");
    }
    let _ = disable_log_requests; // TODO: wire to per-request logging suppression
    let _ = response_role; // TODO: wire to chat completion response role field
    let _ = max_lora_rank; // TODO: validate adapter rank on load

    eprintln!("Loading model: {model_id}");
    let files = loader::fetch_model_with_revision(&model_id, &revision)?;

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
        let lora_loader = LoraLoader::new(device.clone(), dtype);

        for (idx, (name, path)) in parsed_lora_specs.iter().enumerate() {
            eprintln!("  Loading adapter '{}' from: {}", name, path);
            let adapter = lora_loader.load(path, *name, (idx + 1) as u32)?;
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

    // Resolve tokenizer: CLI override > model default
    let tokenizer_path = if let Some(ref tok_override) = tokenizer_override {
        // Load from override path (could be a HuggingFace model ID or local path)
        let tok_path = std::path::Path::new(tok_override);
        if tok_path.exists() {
            tok_path.to_path_buf()
        } else {
            // Try as HuggingFace model ID
            let tok_files = loader::fetch_model_with_revision(tok_override, "main")?;
            tok_files.tokenizer
        }
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
            16,
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

    let cache_config = CacheConfig {
        block_size: 16,
        num_blocks,
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
    let engine_tokenizer = TokenizerWrapper::from_file(&tokenizer_path)?;

    let handle = if let Some(ref draft_id) = draft_model_id {
        eprintln!("Loading draft model: {draft_id}");
        let draft_files = loader::fetch_model(draft_id)?;

        eprintln!("Loading draft weights to GPU (bf16)...");
        let draft_vb = loader::load_weights(&draft_files.weights, dtype, &device)?;

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

        let engine_config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: max_requests,
                max_tokens_per_step: max_num_batched_tokens,
                enable_chunked_prefill,
                scheduling_policy,
                max_loras_per_batch: 0,
            },
            Some(SpeculativeConfig {
                num_speculative_tokens,
            }),
        )
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
    } else {
        let engine_config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: max_requests,
                max_tokens_per_step: max_num_batched_tokens,
                enable_chunked_prefill,
                scheduling_policy,
                max_loras_per_batch: 0,
            },
            None,
        )
        .multi_step_count(multi_step_count)
        .enable_prefix_caching(enable_prefix_caching)
        .build();

        eprintln!("Starting engine (multi-step={multi_step_count})...");
        start_engine(model, engine_tokenizer, kv_cache_mgr, engine_config)
    };

    let (atomic_engine, engine_controller) = AtomicEngineHandle::new(handle);
    let accepting = Arc::new(AtomicBool::new(true));
    let engine_builder: Arc<ProductionEngineBuilder> = Arc::new(ProductionEngineBuilder);

    // max_model_len: CLI override > model's max_position_embeddings > blocks * block_size
    let default_max_model_len =
        std::cmp::min(num_blocks * 16, files.config.max_position_embeddings);
    let max_model_len = max_model_len_override.unwrap_or(default_max_model_len);
    let state = AppState::new(
        atomic_engine.clone(),
        served_model_name,
        tokenizer,
        chat_template,
        eos_token_id,
        max_model_len,
        api::create_tool_call_parser(&tool_call_parser),
        accepting.clone(),
    );

    let start_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let runtime_config = RuntimeConfig {
        model: model_id.clone(),
        draft_model: draft_model_id.clone(),
        num_speculative_tokens,
        num_blocks,
        block_size: 16,
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
            },
            Some(SpeculativeConfig {
                num_speculative_tokens,
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
    } else {
        let engine_config = EngineConfig::builder(
            SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
                max_loras_per_batch: 0,
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
