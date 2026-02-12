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

            run_server(
                model,
                draft_model,
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
            )
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

#[allow(clippy::too_many_arguments)]
async fn run_server(
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
) -> anyhow::Result<()> {
    logging::init();
    prometheus::init_metrics();

    eprintln!("Loading model: {model_id}");
    let files = loader::fetch_model(&model_id)?;

    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;

    eprintln!("Loading weights to GPU (bf16)...");
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
        "Building model ({} layers)...",
        files.config.num_hidden_layers
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

    let tokenizer = TokenizerWrapper::from_file(&files.tokenizer)?;
    let tokenizer = Arc::new(tokenizer);

    let chat_template = files
        .tokenizer_config
        .as_ref()
        .and_then(|path| ChatTemplateEngine::from_tokenizer_config(path).ok())
        .map(Arc::new);

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
    let engine_tokenizer = TokenizerWrapper::from_file(&files.tokenizer)?;

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
                max_tokens_per_step: 2048,
                enable_chunked_prefill,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
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
                max_tokens_per_step: 2048,
                enable_chunked_prefill,
                scheduling_policy: vllm_core::scheduler::SchedulingPolicy::Fcfs,
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

    let max_model_len = num_blocks * 16;
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
        max_tokens_per_step: 2048,
        enable_chunked_prefill,
        multi_step_count,
        enable_prefix_caching,
        dtype: "bf16".to_string(),
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
