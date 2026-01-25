use std::sync::Arc;

use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use vllm_core::{
    engine::{
        start_engine, start_engine_with_draft, EngineConfig, GenerationRequest, SpeculativeConfig,
    },
    kv_cache::{config::CacheConfig, KVCacheManager},
    loader, models,
    scheduler::SchedulerConfig,
    tokenizer::{ChatTemplateEngine, TokenizerWrapper},
};

use vllm_server::api::{self, AppState};

#[derive(Parser)]
#[command(name = "vllm-server", about = "Rust LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
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
        } => {
            run_server(
                model,
                draft_model,
                num_speculative_tokens,
                host,
                port,
                num_blocks,
                max_requests,
                multi_step_count,
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
) -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

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
        };
        eprintln!(
            "Allocating draft KV cache ({} blocks)...",
            draft_cache_config.num_blocks
        );
        let draft_kv_cache = KVCacheManager::new(&draft_cache_config)?;

        let engine_config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: max_requests,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

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
        let engine_config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: max_requests,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count,
            enable_prefix_caching: false,
        };

        eprintln!("Starting engine (multi-step={multi_step_count})...");
        start_engine(model, engine_tokenizer, kv_cache_mgr, engine_config)
    };

    let state = AppState {
        engine: handle.clone(),
        model_id: model_id.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
    };

    let app = api::create_router(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("Serving on http://{addr}/v1");

    axum::serve(listener, app).await?;

    handle.shutdown().await?;
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
        };
        eprintln!(
            "Allocating draft KV cache ({} blocks)...",
            draft_cache_config.num_blocks
        );
        let draft_kv_cache = KVCacheManager::new(&draft_cache_config)?;

        let engine_config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: Some(SpeculativeConfig {
                num_speculative_tokens,
            }),
            multi_step_count: 1,
            enable_prefix_caching: false,
        };

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
        let engine_config = EngineConfig {
            scheduler_config: SchedulerConfig {
                max_running_requests: 8,
                max_tokens_per_step: 2048,
                enable_chunked_prefill: false,
            },
            block_size: 16,
            speculative_config: None,
            multi_step_count,
            enable_prefix_caching: false,
        };

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
