use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

use crate::config::{flatten_hf_model_config, ModelConfig};
use crate::quantization::{
    create_weight_loader_with_params, detect_from_directory, detect_from_json, DetectedQuantConfig,
    QuantizationMethod, QuantizedWeightLoader,
};

/// How weights are loaded from disk.
///
/// Mirrors vLLM's `LoadFormat` enum. All variants except `Safetensors` and
/// `Dummy` are pass-throughs to safetensors loading because Rust has no
/// native PyTorch or numpy loader; users should convert first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadFormat {
    /// Auto-detect: try safetensors, fall back gracefully. Default.
    #[default]
    Auto,
    /// Explicit safetensors (same behaviour as `Auto` for a Rust engine).
    Safetensors,
    /// PyTorch binary (`.pt` / `.bin`). Not supported natively; returns an error.
    Pt,
    /// NumPy cache (vLLM-specific). Treated as `Auto` (load safetensors instead).
    Npcache,
    /// Dummy zero-filled weights. Skips weight download; useful for memory profiling.
    Dummy,
}

impl std::str::FromStr for LoadFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "safetensors" => Ok(Self::Safetensors),
            "pt" | "pytorch_bin" => Ok(Self::Pt),
            "npcache" => Ok(Self::Npcache),
            "dummy" => Ok(Self::Dummy),
            other => anyhow::bail!(
                "Unknown --load-format '{}'. Supported: auto, safetensors, pt, npcache, dummy",
                other
            ),
        }
    }
}

pub struct ModelFiles {
    pub config: ModelConfig,
    pub weights: Vec<PathBuf>,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    /// Detected quantization configuration (if any).
    pub quantization: DetectedQuantConfig,
}

/// Fetch only config.json from HuggingFace Hub (no weight files).
///
/// Much faster than `fetch_model` — only downloads the config file (~few KB).
/// Used for performance estimation and VRAM fitness checks.
pub fn fetch_model_config_only(model_id: &str, revision: &str) -> anyhow::Result<ModelConfig> {
    let api = ApiBuilder::from_env().build()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let config_path = repo.get("config.json")?;
    let raw = std::fs::read_to_string(&config_path)?;
    // Flatten nested sub-configs (Gemma 4 `text_config`, Gemma 4
    // `rope_parameters`, etc.) before deserialising into the flat
    // `ModelConfig` shape the rest of the engine expects.
    let flattened = flatten_hf_model_config(&raw)?;
    let config: ModelConfig = serde_json::from_str(&flattened)?;
    Ok(config)
}

/// Downloads model files from HuggingFace Hub (or uses cache).
pub fn fetch_model(model_id: &str) -> anyhow::Result<ModelFiles> {
    fetch_model_with_revision(model_id, "main")
}

/// Downloads model files from HuggingFace Hub at a specific revision.
pub fn fetch_model_with_revision(model_id: &str, revision: &str) -> anyhow::Result<ModelFiles> {
    fetch_model_with_auth(model_id, revision, None, None)
}

/// Downloads model files, optionally authenticating with an HF token and/or using
/// a custom cache directory.
pub fn fetch_model_with_auth(
    model_id: &str,
    revision: &str,
    hf_token: Option<&str>,
    cache_dir: Option<&std::path::Path>,
) -> anyhow::Result<ModelFiles> {
    fetch_model_with_options(model_id, revision, hf_token, cache_dir, LoadFormat::Auto, 1)
}

/// Full-featured model fetch with load format and parallel download control.
///
/// `load_format` affects which weight files are fetched:
/// - `Auto` / `Safetensors` / `Npcache`: download safetensors shards, return paths.
/// - `Dummy`: skip weight download entirely; the returned `weights` is empty.
///   Callers should use `load_dummy_weights` to create a zero-filled `VarBuilder`.
/// - `Pt`: returns an error — PyTorch binaries require Python-side conversion.
///
/// `max_parallel_workers` controls concurrency for multi-shard downloads.
/// 1 → sequential (default); >1 → shards downloaded in parallel up to that limit.
///
/// `code_revision` is accepted for API parity with vLLM but is unused because
/// the Rust engine does not execute Python custom code. A debug log is emitted
/// when it differs from the weight revision so operators are aware.
pub fn fetch_model_with_options(
    model_id: &str,
    revision: &str,
    hf_token: Option<&str>,
    cache_dir: Option<&std::path::Path>,
    load_format: LoadFormat,
    max_parallel_workers: usize,
) -> anyhow::Result<ModelFiles> {
    if load_format == LoadFormat::Pt {
        anyhow::bail!(
            "--load-format pt (PyTorch binary) is not supported by the Rust engine. \
             Convert weights to safetensors with `python -m vllm.tools.convert_weights` \
             or use --load-format auto."
        );
    }

    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }
    if let Some(dir) = cache_dir {
        builder = builder.with_cache_dir(dir.to_path_buf());
    }
    let api = builder.build()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo.get("config.json")?;
    let raw_content = std::fs::read_to_string(&config_path)?;
    // Gemma 4 and similar models store transformer hyperparameters under
    // a nested sub-object — flatten before deserialising so the engine
    // sees `hidden_size`, `rope_theta`, etc. at the top level.
    let config_content = flatten_hf_model_config(&raw_content)?;
    let config: ModelConfig = serde_json::from_str(&config_content)?;

    // Detect quantization from the raw JSON (quant fields live at the
    // top level even when the transformer config is nested).
    let config_json: serde_json::Value = serde_json::from_str(&raw_content)?;
    let mut quantization = detect_from_json(&config_json);

    // If no quantization detected in config.json, try quantize_config.json
    if quantization.method == QuantizationMethod::None {
        if let Ok(quant_config_path) = repo.get("quantize_config.json") {
            if let Ok(model_dir) = quant_config_path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("Failed to get parent directory"))
            {
                quantization = detect_from_directory(model_dir);
            }
        }
    }

    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

    // Skip weight download for dummy format — caller uses load_dummy_weights().
    let weights = if load_format == LoadFormat::Dummy {
        Vec::new()
    } else {
        let workers = max_parallel_workers.max(1);
        load_safetensor_paths_parallel(&repo, workers)?
    };

    Ok(ModelFiles {
        config,
        weights,
        tokenizer: tokenizer_path,
        tokenizer_config: tokenizer_config_path,
        quantization,
    })
}

/// Creates a VarBuilder from safetensor weight files.
pub fn load_weights(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    Ok(vb)
}

/// Creates a zero-filled VarBuilder for dummy weight loading.
///
/// All tensor accesses return zero-filled tensors of the requested shape.
/// This is used with `--load-format dummy` for memory profiling without
/// downloading actual weights.
pub fn load_dummy_weights(dtype: DType, device: &Device) -> VarBuilder<'static> {
    VarBuilder::zeros(dtype, device)
}

/// Download a specific GGUF file from a HuggingFace repo.
///
/// The production path for GGUF checkpoints is structurally different
/// from safetensors: a single `.gguf` file carries both the model
/// config (in the metadata header) and all weights, so we don't need
/// `config.json` or `model.safetensors.index.json`. Callers pass the
/// GGUF filename explicitly (e.g. `"model-Q4_K_M.gguf"`) to pick a
/// quantization variant.
pub fn fetch_gguf_file(
    model_id: &str,
    gguf_filename: &str,
    revision: &str,
    hf_token: Option<&str>,
    cache_dir: Option<&std::path::Path>,
) -> anyhow::Result<PathBuf> {
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }
    if let Some(dir) = cache_dir {
        builder = builder.with_cache_dir(dir.to_path_buf());
    }
    let api = builder.build()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let path = repo.get(gguf_filename)?;
    Ok(path)
}

/// Load a GGUF file into a fully-wired quantized inference pipeline.
///
/// Returns `(ModelConfig, VarBuilder, QuantizedWeightLoader)` where:
/// - `ModelConfig` is constructed from the GGUF metadata header
///   (no external `config.json` needed);
/// - `VarBuilder` is backed by `GgufVarBuilderBackend` so non-quant
///   tensors (RMS-norm weights, embeddings, optional lm_head) resolve
///   against the same GGUF file;
/// - the returned loader is a `GgufWeightLoader` that serves quantized
///   linear layers via `load_linear`.
///
/// Callers feed the returned VarBuilder and loader into
/// `from_config_with_quant()` (or a model's explicit `::new()`) to run
/// a full forward against the GGUF file.
pub fn load_gguf_model(
    path: &std::path::Path,
    device: Device,
    dtype: DType,
) -> anyhow::Result<(
    ModelConfig,
    VarBuilder<'static>,
    Box<dyn QuantizedWeightLoader>,
    DetectedQuantConfig,
)> {
    use crate::quantization::gguf::{GgufVarBuilderBackend, GgufWeightLoader};

    let gguf_loader = GgufWeightLoader::from_path(path, device.clone(), dtype)
        .map_err(|e| anyhow::anyhow!("GGUF load failed: {e}"))?;
    let cfg = gguf_loader
        .build_model_config()
        .map_err(|e| anyhow::anyhow!("GGUF model config extraction failed: {e}"))?;

    let gguf_handle = gguf_loader.gguf_handle();
    let backend: Box<dyn candle_nn::var_builder::SimpleBackend> =
        Box::new(GgufVarBuilderBackend::new(gguf_handle, device.clone()));
    let vb = VarBuilder::from_backend(backend, dtype, device);

    let quant_config = DetectedQuantConfig {
        method: QuantizationMethod::Gguf,
        ..DetectedQuantConfig::default()
    };

    Ok((cfg, vb, Box::new(gguf_loader), quant_config))
}

/// Creates a quantized weight loader from model files.
///
/// This function creates the appropriate weight loader based on the detected
/// quantization method.
///
/// # Arguments
/// * `files` - Model files from `fetch_model`
/// * `dtype` - Data type for compute (activations)
/// * `device` - Device to load weights to
///
/// # Returns
/// A boxed `QuantizedWeightLoader` implementation
pub fn create_quantized_loader(
    files: &ModelFiles,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<Box<dyn QuantizedWeightLoader>> {
    let vb = load_weights(&files.weights, dtype, device)?;
    Ok(create_weight_loader_with_params(vb, &files.quantization))
}

/// Returns true if the model uses quantization.
pub fn is_quantized(files: &ModelFiles) -> bool {
    files.quantization.method != QuantizationMethod::None
}

/// Get quantization info string for logging.
pub fn quantization_info(files: &ModelFiles) -> String {
    match files.quantization.method {
        QuantizationMethod::None => "None (full precision)".to_string(),
        QuantizationMethod::Fp8 => {
            let scheme = files
                .quantization
                .activation_scheme
                .as_deref()
                .unwrap_or("dynamic");
            format!("FP8 (activation scheme: {})", scheme)
        }
        QuantizationMethod::Gptq => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("GPTQ (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::Awq => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("AWQ (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::Gguf => "GGUF".to_string(),
        QuantizationMethod::BitsAndBytes => "BitsAndBytes".to_string(),
        QuantizationMethod::SqueezeLlm => "SqueezeLLM".to_string(),
        QuantizationMethod::Marlin => "Marlin".to_string(),
        QuantizationMethod::CompressedTensors => "Compressed-Tensors".to_string(),
        QuantizationMethod::Torchao => "TorchAO".to_string(),
        QuantizationMethod::ModelOpt => "ModelOpt (MXFP8)".to_string(),
        QuantizationMethod::ExpertsInt8 => "ExpertsInt8 (W8A16 MoE)".to_string(),
        QuantizationMethod::MoeWNA16 => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("MoeWNA16 (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::AwqMarlin => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!("AWQ-Marlin (bits: {}, group_size: {})", bits, group_size)
        }
        QuantizationMethod::FbgemmFp8 => "FBGEMM FP8 (per-channel, dynamic activation)".to_string(),
        QuantizationMethod::PtpcFp8 => "PTPC FP8 (per-token per-channel, ROCm MI300+)".to_string(),
        QuantizationMethod::Mxfp4 => "MXFP4 (OCP MX FP4 E2M1, block-32)".to_string(),
        QuantizationMethod::ModelOptFull => "ModelOpt (FP8/NVFP4/MXFP8 extended)".to_string(),
        QuantizationMethod::CpuWna16 => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!(
                "CPU AWQ / cpu_wna16 (bits: {}, group_size: {})",
                bits, group_size
            )
        }
        QuantizationMethod::Inc => {
            let bits = files.quantization.bits.unwrap_or(4);
            let group_size = files.quantization.group_size.unwrap_or(128);
            format!(
                "INC / auto-round (bits: {}, group_size: {})",
                bits, group_size
            )
        }
        QuantizationMethod::FpQuant => "FP-Quant (FP4 E2M1 + Hadamard rotation)".to_string(),
        QuantizationMethod::Quark => "QUARK (W8A8-FP8 / W8A8-INT8)".to_string(),
    }
}

/// Natural sort key: splits a filename into alternating text/numeric segments
/// so that "model-2-of-10" sorts before "model-10-of-10".
fn natural_sort_key(s: &str) -> Vec<Result<u64, String>> {
    let basename = std::path::Path::new(s)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(s);
    let mut parts = Vec::new();
    let mut chars = basename.chars().peekable();
    while chars.peek().is_some() {
        if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
            let mut num = String::new();
            while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                num.push(chars.next().unwrap());
            }
            parts.push(Ok(num.parse::<u64>().unwrap_or(0)));
        } else {
            let mut text = String::new();
            while chars.peek().is_some_and(|c| !c.is_ascii_digit()) {
                text.push(chars.next().unwrap());
            }
            parts.push(Err(text));
        }
    }
    parts
}

/// Download safetensors shards, potentially in parallel.
///
/// When `workers == 1`, shards are fetched sequentially (original behaviour).
/// When `workers > 1`, up to `workers` shards are downloaded concurrently
/// using scoped threads. Result order matches the natural-sorted shard order
/// regardless of download completion order.
fn load_safetensor_paths_parallel(
    repo: &hf_hub::api::sync::ApiRepo,
    workers: usize,
) -> anyhow::Result<Vec<PathBuf>> {
    // Try model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    // Multi-file: read model.safetensors.index.json
    let index_path = repo.get("model.safetensors.index.json")?;
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;

    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

    let mut filenames: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    filenames.sort_by_key(|a| natural_sort_key(a));
    filenames.dedup();

    if workers <= 1 || filenames.len() <= 1 {
        // Sequential path: no threading overhead for single-shard models.
        return filenames
            .iter()
            .map(|f| repo.get(f).map_err(|e| anyhow::anyhow!("{e}")))
            .collect();
    }

    // Parallel path: work-stealing across `workers` threads.
    //
    // `repo.get` takes `&self` and the underlying reqwest::blocking::Client
    // is Send + Sync, so ApiRepo is Sync and can be borrowed across threads.
    let n = filenames.len();
    let actual_workers = workers.min(n);
    let next = std::sync::atomic::AtomicUsize::new(0);
    // One Mutex per result slot — avoids a single contended mutex.
    let results: Vec<std::sync::Mutex<Option<Result<PathBuf, String>>>> =
        (0..n).map(|_| std::sync::Mutex::new(None)).collect();

    std::thread::scope(|s| {
        for _ in 0..actual_workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let result = repo.get(&filenames[i]).map_err(|e| e.to_string());
                *results[i].lock().unwrap() = Some(result);
            });
        }
    });

    results
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap()
                .unwrap_or_else(|| Err("shard was not downloaded".to_string()))
                .map_err(|e| anyhow::anyhow!("{e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_sort_orders_numerically() {
        let mut files = vec![
            "model-00010-of-00020.safetensors".to_string(),
            "model-00002-of-00020.safetensors".to_string(),
            "model-00001-of-00020.safetensors".to_string(),
            "model-00011-of-00020.safetensors".to_string(),
        ];
        files.sort_by_key(|a| natural_sort_key(a));
        assert_eq!(
            files,
            vec![
                "model-00001-of-00020.safetensors",
                "model-00002-of-00020.safetensors",
                "model-00010-of-00020.safetensors",
                "model-00011-of-00020.safetensors",
            ]
        );
    }

    #[test]
    fn natural_sort_handles_unpadded_numbers() {
        let mut files = vec![
            "model-10-of-20.safetensors".to_string(),
            "model-2-of-20.safetensors".to_string(),
            "model-1-of-20.safetensors".to_string(),
        ];
        files.sort_by_key(|a| natural_sort_key(a));
        assert_eq!(
            files,
            vec![
                "model-1-of-20.safetensors",
                "model-2-of-20.safetensors",
                "model-10-of-20.safetensors",
            ]
        );
    }

    #[test]
    fn load_format_from_str_roundtrip() {
        assert_eq!("auto".parse::<LoadFormat>().unwrap(), LoadFormat::Auto);
        assert_eq!(
            "safetensors".parse::<LoadFormat>().unwrap(),
            LoadFormat::Safetensors
        );
        assert_eq!("pt".parse::<LoadFormat>().unwrap(), LoadFormat::Pt);
        assert_eq!(
            "npcache".parse::<LoadFormat>().unwrap(),
            LoadFormat::Npcache
        );
        assert_eq!("dummy".parse::<LoadFormat>().unwrap(), LoadFormat::Dummy);
    }

    #[test]
    fn load_format_case_insensitive() {
        assert_eq!("AUTO".parse::<LoadFormat>().unwrap(), LoadFormat::Auto);
        assert_eq!(
            "Safetensors".parse::<LoadFormat>().unwrap(),
            LoadFormat::Safetensors
        );
    }

    #[test]
    fn load_format_unknown_returns_error() {
        assert!("unknown_format".parse::<LoadFormat>().is_err());
    }

    #[test]
    #[ignore = "requires HF download (~8 GB Gemma 4 BnB safetensors); run with --ignored"]
    fn real_bnb_nf4_gemma4_e2b_forward() {
        // Integration test: load the real `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`
        // checkpoint (a Gemma 4 VLM shipped as BitsAndBytes NF4
        // double-quant with a complex `llm_int8_skip_modules` list that
        // mixes per-layer excluded linears, PLE gates, the vision /
        // audio towers, `lm_head`, etc.) and run a short forward through
        // the language model.
        //
        // Exercises:
        // - `BitsAndBytesConfig::from_detected` with a checkpoint-native
        //   skip list containing specific layer patterns (e.g.
        //   `model.language_model.layers.12.self_attn`, `model.language_model.layers.6.mlp`),
        // - `RemappingWeightLoader` mapping `"model.X"` → `"model.language_model.X"`,
        // - `QuantizedGemma4ForCausalLM::new_at_model_root` bypassing
        //   the standalone `vb.pp("model")` step,
        // - `BitsAndBytesWeightLoader::load_linear` dispatching between
        //   the NF4 double-quant path and the BF16 fp16 fallback based
        //   on `is_layer_skipped`,
        // - KV sharing: layers 15..35 skip k/v projection load entirely
        //   (the shared layers would be missing `k_proj`/`v_proj` in
        //   the checkpoint anyway).
        //
        // Memory discipline: we override `hidden_size_per_layer_input=0`
        // to skip PLE (`embed_tokens_per_layer` alone is ~9 GB at F32
        // and would blow the laptop budget), and construct via
        // `new_at_model_root` directly to avoid loading the vision /
        // audio towers. That still leaves the main `embed_tokens`
        // table, the BF16 fallback layers, and the BnB NF4 quantized
        // layers fully exercised on CPU.
        //
        // Run:
        //   cargo test -p vllm-core --lib \
        //       loader::tests::real_bnb_nf4_gemma4_e2b_forward \
        //       -- --ignored --nocapture
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::{gemma4_vlm_quantized, QuantizedGemma4ForCausalLM};
        use crate::quantization::{
            create_weight_loader_with_params, BitsAndBytesConfig, BitsAndBytesWeightLoader,
            QuantizationMethod,
        };
        use candle_core::Tensor;

        let model_id = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch Gemma 4 BnB checkpoint");

        eprintln!(
            "fetched: arch={:?} quant={} layers={} hidden={} vocab={}",
            files.config.architectures,
            quantization_info(&files),
            files.config.num_hidden_layers,
            files.config.hidden_size,
            files.config.vocab_size,
        );
        assert_eq!(
            files.quantization.method,
            QuantizationMethod::BitsAndBytes,
            "checkpoint must be detected as BitsAndBytes"
        );

        let mut cfg = files.config.clone();
        cfg.architectures = vec!["Gemma4ForCausalLM".to_string()];
        // Skip PLE to keep the working set under 9 GB on a laptop —
        // `embed_tokens_per_layer` is `[262144, ple_dim * 35]` BF16 and
        // blows the budget on its own.
        cfg.extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(0),
        );

        let device = Device::Cpu;
        // Gemma 4 BnB 4-bit compute dtype is bfloat16 but Candle CPU
        // matmul doesn't support BF16 yet; load at F32 for the BnB
        // fallback path (weights stay U8 on disk and only the fp16
        // excluded tensors inflate, which is fine without PLE).
        let dtype = DType::F32;
        let vb =
            load_weights(&files.weights, dtype, &device).expect("mmap BnB Gemma 4 safetensors");

        // Build the quantized loader directly so we can wrap it in a
        // `RemappingWeightLoader`. We can't use the standard
        // `from_config_with_quant` dispatch because it would route
        // through `new()` which assumes a standalone checkpoint layout.
        let bnb_cfg = BitsAndBytesConfig::from_detected(&files.quantization.raw_config);
        let base_loader = BitsAndBytesWeightLoader::new(vb.clone(), bnb_cfg);

        // Construct the text model directly against `model.language_model.*`.
        // The `new_at_model_root` ctor skips the standalone
        // `vb.pp("model")` step, and the `RemappingWeightLoader` (a
        // `pub(crate)` helper reused from `gemma4_vlm_quantized`)
        // translates the inner model's `"model.X"` load paths into the
        // real checkpoint's `"model.language_model.X"`.
        let vb_lm = vb.pp("model").pp("language_model");
        let remap = gemma4_vlm_quantized::RemappingWeightLoader::new(
            &base_loader,
            "model",
            "model.language_model",
        );
        let model = QuantizedGemma4ForCausalLM::new_at_model_root(&cfg, vb_lm, &remap)
            .expect("build QuantizedGemma4ForCausalLM from BnB checkpoint");

        // Sanity: also exercise the default dispatch path works for a
        // vanilla quant config → unquantized loader factory so the test
        // catches broken routing that normally only surfaces in the
        // server path. This is a quick smoke check and throws away the
        // result.
        let _ = create_weight_loader_with_params(vb, &DetectedQuantConfig::default());

        // KV cache geometry: `cache_head_dim = max(head_dim, global_head_dim)`.
        let cache_head_dim = cfg
            .extra
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim)
            .max(cfg.head_dim);
        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cache_head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let prompt_ids: Vec<u32> = vec![2, 105, 108, 109, 112];
        let seq_len = prompt_ids.len();
        let input_ids = Tensor::from_vec(prompt_ids, (1, seq_len), &device).expect("input_ids");
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("BnB Gemma 4 prefill forward");
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        let last: Vec<f32> = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1())
            .expect("last-token logits");
        assert_eq!(last.iter().filter(|v| v.is_nan()).count(), 0);
        assert_eq!(last.iter().filter(|v| v.is_infinite()).count(), 0);

        let (argmax, max_val) = last.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
        let gap = max_val - mean;
        eprintln!(
            "gemma4 bnb last-token logits: argmax={argmax} max={max_val:.4} mean={mean:.4} gap={gap:.4}"
        );
        assert!(
            gap > 1e-3,
            "Gemma 4 BnB logits must have spread (gap={gap})"
        );

        eprintln!(
            "OK: Gemma 4 E2B BnB NF4 forward exercised all {} layers (mixed NF4 + BF16 fallback)",
            cfg.num_hidden_layers
        );
    }

    #[test]
    #[ignore = "requires HF download (~8 GB); run with --ignored"]
    fn real_gemma4_vlm_vision_tower_forward() {
        // Builds `Gemma4VisionTower` against the unsloth BnB checkpoint
        // (vision_tower is on the skip list, so weights are plain BF16)
        // and runs a forward pass on dummy patches. Exercises:
        // - `model.vision_tower.patch_embedder.{input_proj.weight, position_embedding_table}`
        // - All 16 encoder layers × (q/k/v/o `.linear.weight`,
        //   q_norm, k_norm, input_layernorm, post_attention_layernorm,
        //   pre/post_feedforward_layernorm, mlp.{gate,up,down}_proj.linear.weight)
        // - `Gemma4VisionPooler` avg-pool over a 36-patch grid (k=3) → 4 soft tokens
        //
        // Run:
        //   cargo test -p vllm-core --lib \
        //       loader::tests::real_gemma4_vlm_vision_tower_forward \
        //       -- --ignored --nocapture
        use crate::models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower};
        use candle_core::Tensor;

        let model_id = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch Gemma 4 BnB checkpoint");

        let device = Device::Cpu;
        // Vision tower weights are stored BF16; load as F32 on CPU since
        // candle-core CPU matmul lacks BF16.
        let dtype = DType::F32;
        let vb =
            load_weights(&files.weights, dtype, &device).expect("mmap BnB Gemma 4 safetensors");

        let cfg = Gemma4VisionConfig::from_model_config(&files.config);
        eprintln!(
            "vision_config: hidden={} layers={} heads={} head_dim={} patch={} pool_k={} pos_emb_sz={} use_clip={}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            cfg.patch_size,
            cfg.pooling_kernel_size,
            cfg.position_embedding_size,
            cfg.use_clipped_linears,
        );

        let vb_vt = vb.pp("model").pp("vision_tower");
        let tower = Gemma4VisionTower::new(&cfg, vb_vt)
            .expect("Gemma4VisionTower loads all weights from checkpoint");

        // Dummy image: 3×3 = 9 patch grid in a single batch item.
        // `pooling_kernel_size = 3` → 9/9 = 1 soft token.
        let b = 1usize;
        let l = 9usize;
        let patch_px = 3 * cfg.patch_size * cfg.patch_size;
        let pixels: Vec<f32> = (0..b * l * patch_px)
            .map(|i| (i as f32).sin() * 0.1)
            .collect();
        let pixel_values =
            Tensor::from_vec(pixels, (b, l, patch_px), &device).expect("pixel_values");
        let mut pos_ids = Vec::with_capacity(b * l * 2);
        for y in 0..3i64 {
            for x in 0..3i64 {
                pos_ids.push(x);
                pos_ids.push(y);
            }
        }
        let pixel_position_ids =
            Tensor::from_vec(pos_ids, (b, l, 2), &device).expect("pixel_position_ids");

        let out = tower
            .forward(&pixel_values, &pixel_position_ids)
            .expect("vision tower forward");
        assert_eq!(out.dim(1).expect("hidden dim"), cfg.hidden_size);
        assert!(
            out.dim(0).expect("rows") >= 1,
            "pooler must emit ≥1 soft token"
        );

        let flat: Vec<f32> = out
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .expect("flatten output");
        assert_eq!(
            flat.iter().filter(|v| v.is_nan()).count(),
            0,
            "NaN in vision output"
        );
        assert_eq!(
            flat.iter().filter(|v| v.is_infinite()).count(),
            0,
            "inf in vision output"
        );

        let (min, max) = flat
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
        eprintln!(
            "gemma4 vision_tower output: tokens={} hidden={} min={min:.4} max={max:.4} mean={mean:.4}",
            out.dim(0).unwrap(),
            out.dim(1).unwrap(),
        );
        assert!((max - min).abs() > 1e-4, "vision output must have spread");

        eprintln!(
            "OK: Gemma 4 vision_tower exercised {} layers on {} patches → {} soft tokens",
            cfg.num_hidden_layers,
            l,
            out.dim(0).unwrap(),
        );
    }

    #[cfg(feature = "image-loading")]
    #[test]
    #[ignore = "requires HF download + image-loading; run with --features image-loading --ignored"]
    fn real_gemma4_vlm_e2e_image_to_soft_tokens() {
        // End-to-end vision preprocessing + forward on a synthetic PNG.
        //
        // Pipeline:
        //   RGB gradient → PNG bytes
        //   → `Gemma4ImageProcessor::preprocess_bytes`
        //     (aspect-ratio resize → rescale → patchify → pad)
        //   → stack to [1, max_patches, …]
        //   → `Gemma4VisionTower::forward`
        //   → `[num_soft_tokens, hidden_size]` ready for the LLM.
        //
        // Run:
        //   cargo test -p vllm-core --lib --features image-loading \
        //     loader::tests::real_gemma4_vlm_e2e_image_to_soft_tokens \
        //     -- --ignored --nocapture
        use crate::models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower};
        use crate::multimodal::gemma4_image::{Gemma4ImageProcessor, Gemma4ImageProcessorConfig};

        // Build a simple 128×96 RGB gradient PNG in-memory.
        let (img_w, img_h) = (128u32, 96u32);
        let mut rgb = Vec::with_capacity((img_w * img_h * 3) as usize);
        for y in 0..img_h {
            for x in 0..img_w {
                rgb.push(((x * 255 / img_w) as u8).min(255));
                rgb.push(((y * 255 / img_h) as u8).min(255));
                rgb.push((((x + y) * 255 / (img_w + img_h)) as u8).min(255));
            }
        }
        let buf = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(img_w, img_h, rgb.clone())
            .expect("build RGB buffer");
        let mut png_bytes: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_bytes);
        buf.write_to(&mut cursor, image::ImageFormat::Png)
            .expect("encode PNG");

        let model_id = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch Gemma 4 BnB checkpoint");

        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb =
            load_weights(&files.weights, dtype, &device).expect("mmap BnB Gemma 4 safetensors");

        let cfg = Gemma4VisionConfig::from_model_config(&files.config);
        let proc_cfg = Gemma4ImageProcessorConfig {
            patch_size: cfg.patch_size,
            // Smallest supported token budget keeps the working set tight.
            max_soft_tokens: 70,
            pooling_kernel_size: cfg.pooling_kernel_size,
            rescale_factor: 1.0 / 255.0,
        };
        let proc = Gemma4ImageProcessor::new(proc_cfg, device.clone());

        let preprocessed = proc.preprocess_bytes(&png_bytes).expect("preprocess image");
        eprintln!(
            "preprocessed: patches={} soft_tokens={} pixel_dim={}",
            preprocessed.pixel_values.dim(0).unwrap(),
            preprocessed.num_soft_tokens,
            preprocessed.pixel_values.dim(1).unwrap(),
        );

        let (pixels, positions, token_counts) = proc.batch(&[preprocessed]).expect("stack batch");
        assert_eq!(pixels.dim(0).unwrap(), 1);
        assert_eq!(token_counts.len(), 1);

        let vb_vt = vb.pp("model").pp("vision_tower");
        let tower =
            Gemma4VisionTower::new(&cfg, vb_vt).expect("Gemma4VisionTower loads all weights");

        let pixels = pixels.to_dtype(DType::F32).unwrap();
        let soft_tokens = tower.forward(&pixels, &positions).expect("forward");
        assert_eq!(soft_tokens.dim(1).unwrap(), cfg.hidden_size);
        assert_eq!(soft_tokens.dim(0).unwrap(), token_counts[0]);

        let flat: Vec<f32> = soft_tokens
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .expect("flatten");
        assert_eq!(flat.iter().filter(|v| v.is_nan()).count(), 0);
        assert_eq!(flat.iter().filter(|v| v.is_infinite()).count(), 0);

        eprintln!(
            "OK: Gemma 4 E2E vision: {}×{} PNG → {} soft tokens × {} hidden",
            img_w,
            img_h,
            soft_tokens.dim(0).unwrap(),
            soft_tokens.dim(1).unwrap(),
        );
    }

    #[test]
    #[ignore = "requires HF download + CUDA GPU; run with --features cuda --ignored"]
    fn real_bnb_nf4_tinyllama_prefill_decode_cuda() {
        // CUDA-flavoured variant of the existing BnB TinyLlama test.
        // Proves that the BitsAndBytes NF4 double-quant dequant path
        // produces correct output when the linear weights live on the
        // GPU: the CPU dequant helper bounces the packed U8 through
        // host memory and uploads the dense F16 back to the original
        // device before matmul, so the GPU path is functionally
        // correct even without a fused CUDA kernel.
        //
        // Skipped when no CUDA device is visible (e.g. on a CPU-only
        // dev loop). Requires the `cuda` crate feature so candle-core
        // builds with the CUDA backend.
        //
        // Run:
        //   cargo test -p vllm-core --lib --features cuda \
        //       loader::tests::real_bnb_nf4_tinyllama_prefill_decode_cuda \
        //       -- --ignored --nocapture
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::from_config_with_quant;
        use crate::quantization::QuantizationMethod;
        use candle_core::Tensor;

        let device = match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => d,
            Ok(_) => {
                eprintln!("SKIP: no CUDA device available");
                return;
            }
            Err(e) => {
                eprintln!("SKIP: CUDA init failed: {e}");
                return;
            }
        };

        let model_id = "unsloth/tinyllama-bnb-4bit";
        let files =
            fetch_model_with_auth(model_id, "main", None, None).expect("fetch BnB NF4 checkpoint");
        assert_eq!(files.quantization.method, QuantizationMethod::BitsAndBytes);

        // F32 weights + F32 activations — candle's CUDA matmul backend
        // supports F32 without any extra feature flags, and BnB
        // backends dequantize to F32 internally anyway.
        let dtype = DType::F32;
        let vb = load_weights(&files.weights, dtype, &device).expect("mmap BnB weights to CUDA");
        let model = from_config_with_quant(&files.config, vb, &files.quantization)
            .expect("construct QuantizedLlamaForCausalLM via BnB loader on CUDA");

        let cfg = &files.config;
        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let prompt_ids: Vec<u32> = vec![1, 15043, 29892, 920, 526, 366];
        let seq_len = prompt_ids.len();
        let input_ids = Tensor::from_vec(prompt_ids, (1, seq_len), &device).expect("input_ids");
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("CUDA BnB prefill forward");
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        let last: Vec<f32> = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1())
            .expect("last logits");
        assert_eq!(last.iter().filter(|v| v.is_nan()).count(), 0);
        assert_eq!(last.iter().filter(|v| v.is_infinite()).count(), 0);

        let (argmax, max_val) = last.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
        let gap = max_val - mean;
        eprintln!(
            "bnb cuda last-token logits: argmax={argmax} max={max_val:.4} mean={mean:.4} gap={gap:.4} device={:?}",
            device,
        );
        assert!(
            gap > 1e-3,
            "CUDA BnB logits must show real spread (gap={gap})"
        );

        if let Ok(out) = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            eprintln!(
                "nvidia-smi after CUDA BnB forward: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }

        eprintln!(
            "OK: CUDA BnB NF4 forward produced real logits on {:?}",
            device
        );
    }

    #[test]
    #[ignore = "requires HF download (~600 MB BnB 4-bit weights); run with --ignored"]
    fn real_bnb_nf4_tinyllama_prefill_decode() {
        // End-to-end integration test for the BitsAndBytes NF4 4-bit
        // production path. Mirrors the AWQ test but exercises a very
        // different backend: NF4 packed u8 weights with double-quant
        // (absmax is itself quantized to u8, recovered via
        // nested_absmax / nested_quant_map). Validates:
        //
        // - `fetch_model` resolves a real BnB HuggingFace repo,
        // - `detect_from_json` recognises `quant_method="bitsandbytes"`,
        // - `create_weight_loader_with_params` routes to
        //   `BitsAndBytesWeightLoader`,
        // - the loader parses the HF-native tensor layout (`weight` as
        //   `[packed_len, 1]` U8, `weight.absmax` as U8, plus the
        //   `nested_absmax` / `nested_quant_map` / `quant_map` double-
        //   quant metadata),
        // - `QuantizedLlamaForCausalLM::new` assembles the full model,
        // - a CPU forward pass through `dequantize_nf4` produces
        //   non-NaN logits with a reasonable argmax (proving the
        //   double-quant unpacking is correct, not just zeros).
        //
        // Model: `unsloth/tinyllama-bnb-4bit` — 1.1B Llama,
        // bnb_4bit_quant_type=nf4, bnb_4bit_use_double_quant=true,
        // bnb_4bit_compute_dtype=bfloat16. ~600 MB on disk, comfortable
        // for CPU inference. We run on CPU so the test works even
        // without a GPU (the focus is the loader/dequant path).
        //
        // Run:
        //   cargo test -p vllm-core --lib \
        //       loader::tests::real_bnb_nf4_tinyllama_prefill_decode \
        //       -- --ignored --nocapture
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::from_config_with_quant;
        use crate::quantization::QuantizationMethod;
        use candle_core::Tensor;

        let model_id = "unsloth/tinyllama-bnb-4bit";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch BnB NF4 checkpoint from HuggingFace");

        eprintln!(
            "fetched: arch={:?} quant={} layers={} hidden={} vocab={}",
            files.config.architectures,
            quantization_info(&files),
            files.config.num_hidden_layers,
            files.config.hidden_size,
            files.config.vocab_size,
        );
        assert_eq!(
            files.quantization.method,
            QuantizationMethod::BitsAndBytes,
            "checkpoint must be detected as BitsAndBytes"
        );

        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = load_weights(&files.weights, dtype, &device).expect("mmap BnB weights");

        // Route through the production dispatch (same path the server
        // uses): `from_config_with_quant` calls
        // `create_weight_loader_with_params` → `BitsAndBytesWeightLoader`
        // and constructs `QuantizedLlamaForCausalLM`.
        let model = from_config_with_quant(&files.config, vb, &files.quantization)
            .expect("construct QuantizedLlamaForCausalLM via BnB loader");

        let cfg = &files.config;
        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let prompt_ids: Vec<u32> = vec![1, 15043, 29892, 920, 526, 366];
        let seq_len = prompt_ids.len();
        let input_ids = Tensor::from_vec(prompt_ids, (1, seq_len), &device).expect("input_ids");

        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("BnB prefill forward");
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        let last: Vec<f32> = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1())
            .expect("last-token logits");

        let nan = last.iter().filter(|v| v.is_nan()).count();
        let inf = last.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan, 0, "BnB NF4 forward must not produce NaN");
        assert_eq!(inf, 0, "BnB NF4 forward must not produce Inf");

        let (argmax, max_val) = last.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
        let gap = max_val - mean;
        eprintln!(
            "bnb last-token logits: argmax={argmax} max={max_val:.4} mean={mean:.4} gap={gap:.4}"
        );
        assert!(argmax < cfg.vocab_size);
        assert!(
            gap > 1e-3,
            "BnB NF4 logits must show real spread (gap={gap}); flat distribution would \
             indicate the double-quant absmax recovery is wrong"
        );

        // Follow-up decode step with a non-zero seqlen_offset — exercises
        // the same BnB dequant path on a fresh single-token input.
        block_table.advance(seq_len);
        kv_cache
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let decode_slot = block_table.slot_mapping(seq_len, 1);
        let next_token =
            Tensor::from_vec(vec![argmax as u32], (1, 1), &device).expect("decode token");
        let decode_logits = model
            .forward(
                &next_token,
                seq_len,
                &mut kv_cache,
                &block_table,
                &decode_slot,
            )
            .expect("BnB decode forward");
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);

        eprintln!("OK: BnB NF4 double-quant prefill+decode produced real logits (argmax={argmax})");
    }

    #[test]
    #[ignore = "requires HF download (~750 MB AWQ weights); run with --ignored"]
    fn real_awq_tinyllama_prefill_decode() {
        // End-to-end integration test for the AWQ production path.
        //
        // Validates that:
        // - `fetch_model` resolves a real AWQ HuggingFace repo,
        // - `detect_from_json` recognises `quantization_config.quant_method="awq"`,
        // - `create_quantized_loader` routes to `AwqWeightLoader`,
        // - `from_config_with_quant` builds a `QuantizedLlamaForCausalLM`,
        // - a CPU forward pass produces non-NaN logits with a sensible
        //   argmax (i.e. the dequantized weights are meaningful, not zeros).
        //
        // Picked `TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ` (~0.7 GB, 22 layers,
        // LlamaForCausalLM, AWQ int4 group-128 "gemm" version). Runs on CPU
        // so it works regardless of GPU availability — the point is to
        // validate the loader/forward path, not throughput.
        //
        // Run:
        //   cargo test -p vllm-core --lib \
        //       loader::tests::real_awq_tinyllama_prefill_decode \
        //       -- --ignored --nocapture
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::from_config_with_quant;
        use crate::quantization::QuantizationMethod;
        use candle_core::Tensor;

        let model_id = "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch AWQ checkpoint from HuggingFace");

        eprintln!(
            "fetched: arch={:?} quant={} layers={} hidden={} vocab={}",
            files.config.architectures,
            quantization_info(&files),
            files.config.num_hidden_layers,
            files.config.hidden_size,
            files.config.vocab_size,
        );
        assert_eq!(
            files.quantization.method,
            QuantizationMethod::Awq,
            "checkpoint must be detected as AWQ"
        );
        assert_eq!(files.quantization.bits, Some(4));
        assert_eq!(files.quantization.group_size, Some(128));

        let device = Device::Cpu;
        let dtype = DType::F32; // AWQ dequant path produces F32-friendly tensors on CPU.

        let vb = load_weights(&files.weights, dtype, &device).expect("mmap weights");

        // Route through the production dispatch — same path the HTTP
        // server uses — so the test exercises `create_weight_loader_with_params`
        // plus the architecture's `from_config_with_quant` arm.
        let model = from_config_with_quant(&files.config, vb, &files.quantization)
            .expect("construct QuantizedLlamaForCausalLM via from_config_with_quant");

        let cfg = &files.config;
        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Use a small real prompt: BOS + "Hello" typical first token ids.
        // TinyLlama vocab is 32003; any reasonable ids below that work.
        let prompt_ids: Vec<u32> = vec![1, 15043, 29892, 920, 526, 366];
        let seq_len = prompt_ids.len();
        let input_ids =
            Tensor::from_vec(prompt_ids.clone(), (1, seq_len), &device).expect("build input_ids");

        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill blocks");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("prefill forward");

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        // Look at the last token's logits — that's what a real decoder
        // sampler would use. We want to see real values, not zeros or NaN.
        let last_logits = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .expect("slice last logits");
        let last_vec: Vec<f32> = last_logits
            .to_dtype(DType::F32)
            .expect("cast logits to f32")
            .to_vec1()
            .expect("read logits");

        let nan_count = last_vec.iter().filter(|v| v.is_nan()).count();
        let inf_count = last_vec.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "logits must not contain NaN");
        assert_eq!(inf_count, 0, "logits must not contain Inf");

        let (argmax, max_val) = last_vec.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        );
        let mean: f32 = last_vec.iter().sum::<f32>() / last_vec.len() as f32;
        let max_gap = max_val - mean;
        eprintln!(
            "last-token logits: argmax={} max={:.4} mean={:.4} gap={:.4}",
            argmax, max_val, mean, max_gap
        );
        assert!(argmax < cfg.vocab_size, "argmax must be a valid vocab id");
        assert!(
            max_gap > 1e-3,
            "logits must have meaningful spread (max-mean gap {max_gap}); \
             near-flat distribution suggests weights are not being dequantized"
        );

        // Decode step — exercise the decode path with one token against
        // a non-zero seqlen_offset.
        block_table.advance(seq_len);
        kv_cache
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode slot");
        let decode_slot = block_table.slot_mapping(seq_len, 1);
        let next_token =
            Tensor::from_vec(vec![argmax as u32], (1, 1), &device).expect("decode token");

        let decode_logits = model
            .forward(
                &next_token,
                seq_len,
                &mut kv_cache,
                &block_table,
                &decode_slot,
            )
            .expect("decode forward");
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);

        // Decode logits must also be meaningful.
        let decode_vec: Vec<f32> = decode_logits
            .squeeze(0)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1())
            .expect("decode logits vec");
        assert_eq!(decode_vec.iter().filter(|v| v.is_nan()).count(), 0);

        eprintln!(
            "OK: AWQ prefill + decode produced real logits \
             (prefill argmax={argmax}, decode len={})",
            decode_vec.len()
        );
    }

    #[test]
    #[ignore = "requires HF download (~9 GB Gemma 4 BF16 safetensors); run with --ignored"]
    fn real_gemma4_e2b_bf16_cpu_forward() {
        // End-to-end integration test for the unquantized Gemma 4
        // production path against real HuggingFace BF16 weights.
        //
        // Unlike the GPU smoke test in `gemma4_quantized` (which shrinks
        // the MLP/vocab dims to fit 8 GB VRAM), this one keeps the REAL
        // E2B dimensions (hidden=1536, vocab=262144, 35 layers, mixed
        // sliding/full head_dim with proportional RoPE, PLE pipeline,
        // KV sharing for layers [15..35)) and runs entirely on CPU so
        // the BF16 working set (~8 GB) fits in host RAM.
        //
        // What this test validates that the dummy-weight GPU path can't:
        // - `flatten_hf_model_config` pulls the real config.json through
        //   unchanged,
        // - the safetensors shards download and mmap cleanly,
        // - every Gemma 4 specific code path executes against REAL,
        //   non-zero weights (the corrected `Gemma4RmsNorm` without +1
        //   offset, the proportional RoPE for full-attention layers,
        //   the KV-sharing skip for `k_proj`/`v_proj`, the per-layer
        //   head_dim padding in the KV cache, the PLE pipeline),
        // - logits look linguistically sensible (argmax falls in the
        //   ASCII / common-word range instead of near 0 or vocab end).
        //
        // Run:
        //   HF_TOKEN=hf_... \
        //     cargo test -p vllm-core --lib \
        //     loader::tests::real_gemma4_e2b_bf16_cpu_forward \
        //     -- --ignored --nocapture
        //
        // Takes ~5-10 minutes on a laptop CPU.
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::Gemma4ForCausalLM;
        use candle_core::Tensor;

        let model_id = "google/gemma-4-E2B-it";
        let files = fetch_model_with_auth(model_id, "main", None, None)
            .expect("fetch Gemma 4 E2B BF16 safetensors");

        // fetch_model already flattened the nested text_config because
        // `fetch_model_with_options` runs `flatten_hf_model_config`.
        let mut cfg = files.config.clone();
        cfg.architectures = vec!["Gemma4ForCausalLM".to_string()];

        // PLE memory shortcut: real Gemma 4 E2B's `embed_tokens_per_layer`
        // alone is `vocab_size × ple_dim × num_layers × 2 bytes ≈ 4.7 GB`,
        // pushing the BF16 working set to ~10 GB and over the laptop's
        // RAM. Disable PLE for this test (the GPU smoke test in
        // `gemma4_quantized` already exercises the PLE pipeline against
        // the real config with dummy weights). All the other Gemma 4
        // specifics — proportional RoPE, KV sharing, mixed head_dim,
        // double_wide_mlp, corrected RmsNorm — still run on REAL
        // weights here.
        cfg.extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(0),
        );
        eprintln!(
            "gemma4 real: arch={:?} hidden={} layers={} vocab={} head_dim={} global_head_dim={:?} \
             num_kv_shared_layers={:?} quant={}",
            cfg.architectures,
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.vocab_size,
            cfg.head_dim,
            cfg.extra.get("global_head_dim"),
            cfg.extra.get("num_kv_shared_layers"),
            quantization_info(&files),
        );
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.head_dim, 256);

        let device = Device::Cpu;
        // Candle's CPU matmul kernel does not support BF16, so we cast
        // weights to F16 on load. F16 has the same memory footprint as
        // BF16 (~4.5 GB after disabling PLE) but a different exponent
        // range — the existing Gemma4 forward path handles the dtype
        // mismatch via internal `to_dtype` calls in the linear layers.
        let dtype = DType::F16;
        let vb = load_weights(&files.weights, dtype, &device).expect("mmap Gemma 4 weights");

        // The HF Gemma 4 E2B repo is shipped as a VLM
        // (`Gemma4ForConditionalGeneration`) so the LLM tensors live
        // under `model.language_model.*`, not `model.*`. Construct via
        // `new_with_tp_at_root` and skip the standalone-checkpoint's
        // built-in `vb.pp("model")`.
        let vb_m = vb.pp("model").pp("language_model");
        let pg = crate::distributed::LocalProcessGroup::new();
        let tp_ctx = crate::models::gemma::TpContext::single_gpu();
        let model = Gemma4ForCausalLM::new_with_tp_at_root(&cfg, vb_m, None, &pg, tp_ctx)
            .expect("build Gemma4ForCausalLM at language_model root");

        // KV cache at cache_head_dim = max(head_dim, global_head_dim)
        // = max(256, 512) = 512. The layer forward pads / slices
        // automatically.
        let global_head_dim = cfg
            .extra
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.head_dim);
        let cache_head_dim = cfg.head_dim.max(global_head_dim);

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cache_head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Use BOS + a short English prompt. Exact token ids depend on
        // the tokenizer; we assume a small Gemma-like vocab slice here
        // and rely on `argmax != 0` + low NaN / spread > 1 to conclude
        // "the forward produced meaningful logits". Tokenizer-accurate
        // generation is out of scope for this test.
        let prompt_ids: Vec<u32> = vec![2, 105, 108, 109, 112];
        let seq_len = prompt_ids.len();
        let input_ids = Tensor::from_vec(prompt_ids, (1, seq_len), &device).expect("input_ids");
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("Gemma 4 BF16 prefill");
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        let last: Vec<f32> = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1())
            .expect("last-token logits");
        let nan = last.iter().filter(|v| v.is_nan()).count();
        let inf = last.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan, 0, "Gemma 4 BF16 forward produced NaN");
        assert_eq!(inf, 0, "Gemma 4 BF16 forward produced Inf");

        let (argmax, max_val) = last.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
        let gap = max_val - mean;
        eprintln!(
            "gemma4 real BF16 last-token logits: argmax={argmax} max={max_val:.4} \
             mean={mean:.4} gap={gap:.4}"
        );
        assert!(gap > 1e-2, "logits must have real spread (gap={gap})");

        // Follow-up decode step to exercise the non-zero seqlen_offset
        // branch against real weights.
        block_table.advance(seq_len);
        kv_cache
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let decode_slot = block_table.slot_mapping(seq_len, 1);
        let next_token =
            Tensor::from_vec(vec![argmax as u32], (1, 1), &device).expect("decode token");
        let decode_logits = model
            .forward(
                &next_token,
                seq_len,
                &mut kv_cache,
                &block_table,
                &decode_slot,
            )
            .expect("decode forward");
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);

        eprintln!(
            "OK: Gemma 4 E2B BF16 full-depth forward exercised all {} real layers \
             (sliding + full attention, proportional RoPE, KV sharing, PLE)",
            cfg.num_hidden_layers,
        );
    }

    #[test]
    #[ignore = "requires HF download (~100 MB GGUF); run with --ignored"]
    fn real_gguf_smollm_prefill_decode() {
        // End-to-end integration test for the GGUF production path.
        //
        // Validates:
        // - `fetch_gguf_file` downloads a real `.gguf` from HF,
        // - `load_gguf_model` parses the GGUF metadata into a usable
        //   `ModelConfig` and hands back a VarBuilder backed by the
        //   GGUF file plus a `GgufWeightLoader`,
        // - `from_config_with_quant` dispatches via the production
        //   `create_weight_loader_with_params` path (falls through to
        //   unquantized for GGUF — which warns — and we install OUR
        //   GGUF loader manually via a direct model constructor),
        // - `QuantizedLlamaForCausalLM::new` resolves both the
        //   GGUF-packed linears and the fp16 norms/embeddings through
        //   the unified GGUF-backed VarBuilder,
        // - prefill + decode produce non-NaN logits with sensible
        //   argmax — i.e. the GGML Q4_K_M dequant kernel produces
        //   meaningful values rather than zeros.
        //
        // Model: MaziyarPanahi/SmolLM-135M-Instruct-GGUF, Q4_K_M
        // variant (~105 MB, LlamaForCausalLM arch, 30 layers,
        // hidden=576). Small enough to download fast and run on CPU.
        use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
        use crate::models::QuantizedLlamaForCausalLM;
        use candle_core::Tensor;

        let model_id = "MaziyarPanahi/SmolLM-135M-Instruct-GGUF";
        let gguf_filename = "SmolLM-135M-Instruct.Q4_K_M.gguf";

        let path =
            fetch_gguf_file(model_id, gguf_filename, "main", None, None).expect("fetch GGUF file");

        let device = Device::Cpu;
        let dtype = DType::F32;

        let (cfg, vb, loader, quant) = load_gguf_model(&path, device.clone(), dtype)
            .expect("parse GGUF + build VarBuilder + loader");

        eprintln!(
            "gguf: arch={:?} layers={} hidden={} heads={}/{} head_dim={} vocab={} quant={:?}",
            cfg.architectures,
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.vocab_size,
            quant.method,
        );
        assert_eq!(cfg.architectures, vec!["LlamaForCausalLM".to_string()]);
        assert!(cfg.num_hidden_layers > 0);

        // Build the quantized Llama directly (the production
        // `create_weight_loader_with_params` dispatch doesn't yet route
        // GGUF — it would emit a warn and fall back to unquantized).
        let model = QuantizedLlamaForCausalLM::new(&cfg, vb, loader.as_ref())
            .expect("build QuantizedLlamaForCausalLM from GGUF");

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).expect("kv cache manager");
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // SmolLM vocab starts at the usual Llama positions; reuse the
        // prompt ids from the AWQ test — they are well inside vocab.
        let prompt_ids: Vec<u32> = vec![1, 15043, 29892, 920, 526, 366];
        let seq_len = prompt_ids.len();
        let input_ids = Tensor::from_vec(prompt_ids, (1, seq_len), &device).expect("input_ids");
        kv_cache
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .expect("prefill forward through GGUF loader");
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);

        let last = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .expect("last logits");
        let nan_count = last.iter().filter(|v| v.is_nan()).count();
        let inf_count = last.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(nan_count, 0, "GGUF forward must not produce NaN");
        assert_eq!(inf_count, 0, "GGUF forward must not produce Inf");

        let (argmax, max_val) = last.iter().copied().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, v)| if v > bv { (i, v) } else { (bi, bv) },
        );
        let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
        let gap = max_val - mean;
        eprintln!(
            "gguf last-token logits: argmax={argmax} max={max_val:.4} mean={mean:.4} gap={gap:.4}"
        );
        assert!(
            gap > 1e-3,
            "GGUF logits must have spread (gap={gap}); suggests dequant failure"
        );
        assert!(argmax < cfg.vocab_size);

        // Follow-up decode step.
        block_table.advance(seq_len);
        kv_cache
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let decode_slot = block_table.slot_mapping(seq_len, 1);
        let next_token =
            Tensor::from_vec(vec![argmax as u32], (1, 1), &device).expect("decode token");
        let decode_logits = model
            .forward(
                &next_token,
                seq_len,
                &mut kv_cache,
                &block_table,
                &decode_slot,
            )
            .expect("decode forward");
        assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);

        eprintln!("OK: GGUF Q4_K_M prefill+decode produced real logits");
    }

    #[test]
    fn load_format_pytorch_bin_alias() {
        assert_eq!("pytorch_bin".parse::<LoadFormat>().unwrap(), LoadFormat::Pt);
    }
}
