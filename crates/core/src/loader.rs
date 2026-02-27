use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

use crate::config::ModelConfig;
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
    let config_content = std::fs::read_to_string(&config_path)?;
    let config: ModelConfig = serde_json::from_str(&config_content)?;

    // Detect quantization from config.json
    let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
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
        files.sort_by(|a, b| natural_sort_key(a).cmp(&natural_sort_key(b)));
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
        files.sort_by(|a, b| natural_sort_key(a).cmp(&natural_sort_key(b)));
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
    fn load_format_pytorch_bin_alias() {
        assert_eq!("pytorch_bin".parse::<LoadFormat>().unwrap(), LoadFormat::Pt);
    }
}
