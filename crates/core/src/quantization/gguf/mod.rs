//! GGUF format support for loading quantized models.
//!
//! GGUF (GGML Universal Format) is a file format for storing quantized LLMs.
//! It is used by llama.cpp and other inference engines.
//!
//! # Supported quantization types
//!
//! - F32, F16, BF16 - Unquantized
//! - Q4_0, Q4_1 - 4-bit quantization
//! - Q5_0, Q5_1 - 5-bit quantization
//! - Q8_0, Q8_1 - 8-bit quantization
//! - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K - K-quant formats
//!
//! # References
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

mod dequant;
mod parser;

pub use dequant::{dequantize, GgmlType};
pub use parser::{GgufFile, GgufMetadata, GgufTensorInfo, GgufValue};

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::weight_loader::QuantizedWeightLoader;

/// GGUF quantization configuration.
#[derive(Debug, Clone, Default)]
pub struct GgufConfig {
    /// List of layers to skip quantization (use full precision)
    pub ignored_layers: Vec<String>,
}

impl GgufConfig {
    /// Create a new GGUF config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add layers to skip quantization.
    pub fn with_ignored_layers(mut self, layers: Vec<String>) -> Self {
        self.ignored_layers = layers;
        self
    }
}

impl QuantizationConfig for GgufConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        // GGUF dequantization produces F16 internally
        &[DType::F16, DType::BF16, DType::F32]
    }

    fn min_capability(&self) -> u32 {
        // GGUF can run on CPU (capability 0)
        0
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.ignored_layers
            .iter()
            .any(|ignored| layer_name.contains(ignored))
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(GgufLinear::new(
            in_features,
            out_features,
            bias,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// GGUF quantized linear layer.
///
/// Stores quantized weights and dequantizes them on-the-fly during forward pass.
#[derive(Debug)]
pub struct GgufLinear {
    /// Quantized weight data (raw bytes)
    qweight: Option<Tensor>,
    /// Weight quantization type
    qtype: GgmlType,
    /// Bias tensor (always unquantized)
    bias: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Device (stored for potential future use in dequantization)
    #[allow(dead_code)]
    device: Device,
}

impl GgufLinear {
    /// Create a new GGUF linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F16, device)?)
        } else {
            None
        };

        Ok(Self {
            qweight: None,
            qtype: GgmlType::F16,
            bias,
            in_features,
            out_features,
            device: device.clone(),
        })
    }

    /// Set the quantized weight data.
    pub fn set_qweight(&mut self, qweight: Tensor, qtype: GgmlType) {
        self.qweight = Some(qweight);
        self.qtype = qtype;
    }

    /// Get the quantization type.
    pub fn qtype(&self) -> GgmlType {
        self.qtype
    }

    /// Check if weights are loaded.
    pub fn has_weights(&self) -> bool {
        self.qweight.is_some()
    }
}

impl QuantizedLinear for GgufLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let qweight = self.qweight.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "GGUF layer has no weights loaded - call load_weights() first".to_string(),
            )
        })?;

        // Dequantize once per forward. `weight` lands as F16 with the
        // llama.cpp row layout `[out_features, in_features]`, which is
        // what we want for `x @ weight.t() = [*, out]`.
        let weight_f16 = dequantize(qweight, self.qtype, self.out_features, self.in_features)?;
        let orig_dtype = x.dtype();
        let compute_dtype = DType::F32;
        let weight = weight_f16.t()?.to_dtype(compute_dtype)?; // [in, out]
        let x_compute = x.to_dtype(compute_dtype)?;

        // Candle does not auto-broadcast a 2D weight over batched 3D
        // inputs, so collapse leading dims to a single row axis, matmul,
        // then restore the original shape.
        let dims = x_compute.dims().to_vec();
        if dims.is_empty() {
            candle_core::bail!("GgufLinear forward: scalar input is not supported");
        }
        let in_features = *dims.last().unwrap();
        if in_features != self.in_features {
            candle_core::bail!(
                "GgufLinear forward: last dim {in_features} != in_features {}",
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2d = x_compute.reshape((leading, in_features))?;
        let y2d = x2d.matmul(&weight)?;

        let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        let y = y2d.reshape(out_dims)?;

        let y = match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(compute_dtype)?)?,
            None => y,
        };
        y.to_dtype(orig_dtype)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(qweight) = weights.get("qweight") {
            self.qweight = Some(qweight.clone());
        }
        if let Some(qtype_tensor) = weights.get("qtype") {
            // qtype is stored as a scalar u8
            let qtype_val = qtype_tensor.to_vec0::<u8>()?;
            self.qtype = GgmlType::from_u32(qtype_val as u32);
        }
        if let Some(bias) = weights.get("bias") {
            self.bias = Some(bias.clone());
        }
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        // GGUF stores quantized weights as raw bytes (U8)
        DType::U8
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

/// Weight loader for GGUF files.
pub struct GgufWeightLoader {
    /// Parsed GGUF file
    gguf: std::sync::Arc<GgufFile>,
    /// Device to load tensors to
    device: Device,
    /// Compute dtype for activations
    dtype: DType,
    /// Config
    config: GgufConfig,
}

impl GgufWeightLoader {
    /// Create a new GGUF weight loader from a file path.
    pub fn from_path(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {e}")))?;

        Ok(Self {
            gguf: std::sync::Arc::new(gguf),
            device,
            dtype,
            config: GgufConfig::default(),
        })
    }

    /// Shared handle to the parsed GGUF file, so callers can hand the
    /// same data to both the `QuantizedWeightLoader` (for quantized
    /// linears) and a custom `VarBuilder` backend (for fp16 norms /
    /// embeddings).
    pub fn gguf_handle(&self) -> std::sync::Arc<GgufFile> {
        std::sync::Arc::clone(&self.gguf)
    }

    /// Extract the architecture name exported by the GGUF metadata key
    /// `general.architecture`. Short-hands like `llama` get mapped to
    /// the vLLM-style architecture list used by `ModelConfig`.
    pub fn vllm_architecture(&self) -> Option<Vec<String>> {
        let arch = self.architecture()?;
        let mapped = match arch {
            "llama" => "LlamaForCausalLM",
            "gemma" => "GemmaForCausalLM",
            "gemma2" => "Gemma2ForCausalLM",
            "gemma3" => "Gemma3ForCausalLM",
            "gemma4" => "Gemma4ForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "mistral" => "MistralForCausalLM",
            "phi3" => "Phi3ForCausalLM",
            other => {
                tracing::warn!("unknown GGUF architecture '{other}', falling back verbatim");
                return Some(vec![other.to_string()]);
            }
        };
        Some(vec![mapped.to_string()])
    }

    /// Build a `ModelConfig` from GGUF metadata. Pulls the standard
    /// llama.cpp keys (`{arch}.embedding_length`, `{arch}.block_count`,
    /// etc.) and falls back to conservative defaults for anything not
    /// present. Used by the GGUF production path so callers do not need
    /// a separate `config.json`.
    pub fn build_model_config(&self) -> Result<crate::config::ModelConfig> {
        let arch_name = self
            .architecture()
            .ok_or_else(|| candle_core::Error::Msg("GGUF missing general.architecture".into()))?
            .to_string();
        let metadata = self.gguf.metadata();
        let get_u64 = |k: &str| metadata.get(k).and_then(|v| v.as_u64());
        let get_u32 = |k: &str| metadata.get(k).and_then(|v| v.as_u32()).map(|v| v as u64);
        let get_f32 = |k: &str| metadata.get(k).and_then(|v| v.as_f32()).map(|v| v as f64);

        let hidden_size = get_u32(&format!("{arch_name}.embedding_length"))
            .or_else(|| get_u64(&format!("{arch_name}.embedding_length")))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.embedding_length"))
            })? as usize;
        let num_hidden_layers = get_u32(&format!("{arch_name}.block_count"))
            .or_else(|| get_u64(&format!("{arch_name}.block_count")))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.block_count"))
            })? as usize;
        let num_attention_heads = get_u32(&format!("{arch_name}.attention.head_count"))
            .or_else(|| get_u64(&format!("{arch_name}.attention.head_count")))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.attention.head_count"))
            })? as usize;
        let num_key_value_heads = get_u32(&format!("{arch_name}.attention.head_count_kv"))
            .or_else(|| get_u64(&format!("{arch_name}.attention.head_count_kv")))
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = get_u32(&format!("{arch_name}.feed_forward_length"))
            .or_else(|| get_u64(&format!("{arch_name}.feed_forward_length")))
            .unwrap_or(4 * hidden_size as u64) as usize;
        let max_position_embeddings = get_u32(&format!("{arch_name}.context_length"))
            .or_else(|| get_u64(&format!("{arch_name}.context_length")))
            .unwrap_or(4096) as usize;
        let rms_norm_eps =
            get_f32(&format!("{arch_name}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);
        let rope_theta = get_f32(&format!("{arch_name}.rope.freq_base")).unwrap_or(10000.0);

        let vocab_size = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .ok_or_else(|| {
                candle_core::Error::Msg("GGUF missing tokenizer.ggml.tokens array".into())
            })?;

        let head_dim = if num_attention_heads > 0 {
            hidden_size / num_attention_heads
        } else {
            hidden_size
        };

        let architectures = self
            .vllm_architecture()
            .unwrap_or_else(|| vec!["LlamaForCausalLM".to_string()]);

        Ok(crate::config::ModelConfig {
            architectures,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            head_dim,
            hidden_act: "silu".to_string(),
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        })
    }

    /// Create with a specific config.
    pub fn with_config(mut self, config: GgufConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the GGUF file metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        self.gguf.metadata()
    }

    /// Get a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Result<(Tensor, GgmlType)> {
        self.gguf.load_tensor(name, &self.device)
    }

    /// Get model architecture from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.gguf
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Get number of layers from metadata.
    pub fn num_layers(&self) -> Option<u32> {
        // Try common metadata keys
        let arch = self.architecture()?;
        let key = format!("{arch}.block_count");
        self.gguf.metadata().get(&key).and_then(|v| v.as_u32())
    }

    /// Get hidden size from metadata.
    pub fn hidden_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        let key = format!("{arch}.embedding_length");
        self.gguf.metadata().get(&key).and_then(|v| v.as_u32())
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.gguf.tensor_names()
    }
}

/// Map a vLLM-style tensor prefix to the canonical llama.cpp GGUF naming.
///
/// llama.cpp stores transformer blocks under `blk.N.*` and uses condensed
/// names (`attn_q`, `ffn_gate`, `output_norm`) rather than the HuggingFace
/// pytorch_model path. This mapping lets the GGUF loader accept vLLM-style
/// prefixes (`model.layers.5.self_attn.q_proj`) and still find the tensors.
///
/// Returns `None` when the input is not a recognised pattern — the caller
/// should then fall back to the literal path or report "tensor not found".
///
/// Gemma 4 specifics (PLE `per_layer_*`, MoE stacked experts `ffn_*_exps`,
/// `layer_scalar`) are NOT yet mapped here: stacked-expert loading requires
/// slicing an [N_experts, out, in] tensor into per-expert quantized blocks,
/// which the current `load_linear` signature does not support. Support will
/// be added once a real Gemma 4 GGUF checkpoint surfaces and the exact
/// tensor names are confirmed.
fn to_llama_cpp_prefix(prefix: &str) -> Option<String> {
    to_llama_cpp_prefixes(prefix).into_iter().next()
}

/// Return every llama.cpp-style candidate name for a given vLLM-style
/// prefix, in priority order. Some prefixes map to different llama.cpp
/// names across architectures — most notably `post_attention_layernorm`
/// is `ffn_norm` in llama/mistral but `post_attention_norm` in Gemma —
/// so we try each candidate in turn.
fn to_llama_cpp_prefixes(prefix: &str) -> Vec<String> {
    match prefix {
        "model.embed_tokens" => return vec!["token_embd".to_string()],
        "model.norm" => return vec!["output_norm".to_string()],
        "lm_head" => return vec!["output".to_string()],
        _ => {}
    }

    let rest = match prefix.strip_prefix("model.layers.") {
        Some(r) => r,
        None => return Vec::new(),
    };
    let (layer_idx_str, sub) = match rest.split_once('.') {
        Some(pair) => pair,
        None => return Vec::new(),
    };
    if layer_idx_str.parse::<u32>().is_err() {
        return Vec::new();
    }

    let subs: Vec<&str> = match sub {
        "self_attn.q_proj" => vec!["attn_q"],
        "self_attn.k_proj" => vec!["attn_k"],
        "self_attn.v_proj" => vec!["attn_v"],
        "self_attn.o_proj" => vec!["attn_output"],
        "self_attn.q_norm" => vec!["attn_q_norm"],
        "self_attn.k_norm" => vec!["attn_k_norm"],
        "mlp.gate_proj" => vec!["ffn_gate"],
        "mlp.up_proj" => vec!["ffn_up"],
        "mlp.down_proj" => vec!["ffn_down"],
        "input_layernorm" => vec!["attn_norm"],
        // Llama/Mistral: post-attention IS the pre-FFN norm → `ffn_norm`.
        // Gemma/Gemma2/Gemma3: separate `post_attention_norm` tensor.
        "post_attention_layernorm" => vec!["ffn_norm", "post_attention_norm"],
        // Gemma family has both pre/post FFN norms; plain Llama doesn't.
        "pre_feedforward_layernorm" => vec!["ffn_norm"],
        "post_feedforward_layernorm" => vec!["post_ffw_norm"],
        _ => return Vec::new(),
    };

    subs.into_iter()
        .map(|s| format!("blk.{layer_idx_str}.{s}"))
        .collect()
}

/// `VarBuilder` backend that resolves names against a GGUF file instead
/// of a safetensors archive. Used to hand the non-quantized tensors
/// (RMS-norm weights, embeddings) that a model construction pulls
/// through `vb.pp(...)` over to the same GGUF file that already backs
/// the quantized linears.
///
/// Resolution strategy, in order:
/// 1. Literal `{name}` — some GGUF exports keep HuggingFace names.
/// 2. `{to_llama_cpp_prefix(name).unwrap_or(name)}` — standard
///    llama.cpp shorthand (`blk.N.attn_norm`, `output_norm`, …).
///
/// Norm/embedding tensors in llama.cpp GGUF are typically stored as F32
/// or F16 and loaded directly via `load_tensor` → Candle F32.
pub struct GgufVarBuilderBackend {
    gguf: std::sync::Arc<GgufFile>,
    device: Device,
}

impl GgufVarBuilderBackend {
    /// Wrap a shared GGUF handle so the same parsed file serves both
    /// the quantized loader and the VarBuilder backend.
    pub fn new(gguf: std::sync::Arc<GgufFile>, device: Device) -> Self {
        Self { gguf, device }
    }

    /// Resolve `name` using the same fallback chain as
    /// `GgufWeightLoader::load_linear` — literal path, then
    /// `blk.N.*`-style mapping, then the bare prefix without `.weight`.
    fn try_load(&self, name: &str, target: DType) -> Result<Tensor> {
        // The name arrives with a trailing qualifier like `...weight`
        // that VarBuilder appends via `.pp(...).get(shape, "weight")`.
        // llama.cpp stores e.g. `blk.0.attn_norm.weight` literally, so
        // the literal path is tried first.
        let mut candidates: Vec<String> = vec![name.to_string()];
        if let Some((base, leaf)) = name.rsplit_once('.') {
            for mapped_base in to_llama_cpp_prefixes(base) {
                candidates.push(format!("{mapped_base}.{leaf}"));
            }
            if let Some(top) = to_llama_cpp_top_level(base) {
                candidates.push(format!("{top}.{leaf}"));
            }
        } else if let Some(mapped) = to_llama_cpp_top_level(name) {
            candidates.push(mapped.to_string());
        }

        for candidate in candidates {
            // Use the tensor-info metadata to get the LOGICAL shape
            // (vocab_size × hidden_size) rather than the byte-level
            // shape returned by `load_tensor`, which for quantized
            // types inflates the second dimension by the block ratio.
            let info = match self.gguf.tensor_info(&candidate) {
                Some(info) => info,
                None => continue,
            };
            let logical_shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
            let (tensor, qtype) = match self.gguf.load_tensor(&candidate, &self.device) {
                Ok(t) => t,
                Err(_) => continue,
            };

            // Non-quantized stored tensors (F32/F16/BF16) come back
            // ready to use; quantized tensors get dequantized on the
            // fly so the VarBuilder consumer sees a dense weight.
            let dense = match qtype {
                GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                    // Dense tensors just need a reshape to the logical
                    // shape; `load_tensor` may return them flattened.
                    if logical_shape.is_empty() {
                        tensor
                    } else {
                        tensor.reshape(logical_shape.clone())?
                    }
                }
                other => {
                    // llama.cpp stores weight matrices as `[out, in]`
                    // (rows × cols); the dequant helpers expect
                    // `(rows, cols)` in that order.
                    let (rows, cols) = match logical_shape.as_slice() {
                        [rows] => (*rows, 1usize),
                        [rows, cols] => (*rows, *cols),
                        _ => {
                            return Err(candle_core::Error::Msg(format!(
                                "GgufVarBuilderBackend: unexpected logical shape {:?} for {candidate}",
                                logical_shape
                            )));
                        }
                    };
                    dequantize(&tensor, other, rows, cols)?
                }
            };
            return dense.to_dtype(target);
        }
        Err(candle_core::Error::Msg(format!(
            "GgufVarBuilderBackend: tensor '{name}' not found (tried literal + llama.cpp mapping)"
        )))
    }
}

/// Minimal top-level name mapping for tensors that sit outside a
/// transformer block (`model.embed_tokens`, `model.norm`, `lm_head`).
/// `to_llama_cpp_prefix` handles the more general `blk.N.*` cases.
fn to_llama_cpp_top_level(name: &str) -> Option<&'static str> {
    match name {
        "model.embed_tokens" => Some("token_embd"),
        "model.norm" => Some("output_norm"),
        "lm_head" => Some("output"),
        _ => None,
    }
}

impl candle_nn::var_builder::SimpleBackend for GgufVarBuilderBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        _dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.try_load(name, dtype)?;
        let got = tensor.shape().clone();
        if got == s {
            return Ok(tensor);
        }

        // llama.cpp stores 2-D matrices in transposed order relative to
        // the HuggingFace / PyTorch convention: `token_embd.weight` is
        // `[n_embd, n_vocab]` on disk but Candle's `Embedding` expects
        // `[vocab, hidden]`, and linear weights are `[n_out, n_in]` on
        // disk but `candle_nn::Linear` consumes `[out, in]` already.
        // When the requested shape is a transpose of what we got, auto-
        // transpose and retry — this covers embeddings without breaking
        // already-correctly-shaped linears.
        if got.dims().len() == 2 && s.dims().len() == 2 {
            let got_dims = got.dims();
            let want_dims = s.dims();
            if got_dims[0] == want_dims[1] && got_dims[1] == want_dims[0] {
                let transposed = tensor.t()?.contiguous()?;
                if transposed.shape() == &s {
                    return Ok(transposed);
                }
            }
        }

        Err(candle_core::Error::Msg(format!(
            "GgufVarBuilderBackend: shape mismatch for '{name}' expected {:?}, got {:?}",
            s.dims(),
            got.dims()
        )))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.try_load(name, DType::F32).is_ok()
    }

    fn get_unchecked(&self, name: &str, dtype: DType, _dev: &Device) -> Result<Tensor> {
        self.try_load(name, dtype)
    }
}

impl QuantizedWeightLoader for GgufWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let mut linear = GgufLinear::new(in_features, out_features, bias, &self.device)?;

        // Fallback chain: literal `{prefix}.weight` first (preserves
        // existing behaviour for checkpoints that already ship vLLM-style
        // names), then the bare prefix, then a llama.cpp-style mapping.
        let weight_name = format!("{prefix}.weight");

        if let Ok((qweight, qtype)) = self.get_tensor(&weight_name) {
            linear.set_qweight(qweight, qtype);
        } else if let Ok((qweight, qtype)) = self.get_tensor(prefix) {
            linear.set_qweight(qweight, qtype);
        } else if let Some(mapped) = to_llama_cpp_prefix(prefix) {
            let mapped_weight = format!("{mapped}.weight");
            if let Ok((qweight, qtype)) = self.get_tensor(&mapped_weight) {
                linear.set_qweight(qweight, qtype);
            } else if let Ok((qweight, qtype)) = self.get_tensor(&mapped) {
                linear.set_qweight(qweight, qtype);
            }
        }

        if bias {
            let bias_name = format!("{prefix}.bias");
            if let Ok((bias_tensor, _)) = self.get_tensor(&bias_name) {
                let bias_f16 = bias_tensor.to_dtype(DType::F16)?;
                linear.bias = Some(bias_f16);
            } else if let Some(mapped) = to_llama_cpp_prefix(prefix) {
                let mapped_bias = format!("{mapped}.bias");
                if let Ok((bias_tensor, _)) = self.get_tensor(&mapped_bias) {
                    linear.bias = Some(bias_tensor.to_dtype(DType::F16)?);
                }
            }
        }

        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_config_default() {
        let config = GgufConfig::default();
        assert_eq!(config.method(), QuantizationMethod::Gguf);
        assert_eq!(config.min_capability(), 0);
        assert!(!config.is_layer_skipped("some_layer"));
    }

    #[test]
    fn test_gguf_config_ignored_layers() {
        let config = GgufConfig::new().with_ignored_layers(vec!["lm_head".to_string()]);
        assert!(config.is_layer_skipped("lm_head"));
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("self_attn"));
    }

    #[test]
    fn test_gguf_linear_creation() {
        let config = GgufConfig::default();
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_gguf_linear_validation() {
        let result = GgufLinear::new(0, 128, false, &Device::Cpu);
        assert!(result.is_err());

        let result = GgufLinear::new(64, 0, false, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_linear_forward_requires_weights() {
        let linear = GgufLinear::new(64, 128, false, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[2, 64], DType::F16, &Device::Cpu).unwrap();

        let result = linear.forward(&x);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no weights loaded"));
    }

    #[test]
    fn test_ggml_type_roundtrip() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8), GgmlType::Q8_0);
    }

    #[test]
    fn test_to_llama_cpp_prefix_top_level() {
        assert_eq!(
            to_llama_cpp_prefix("model.embed_tokens").as_deref(),
            Some("token_embd")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.norm").as_deref(),
            Some("output_norm")
        );
        assert_eq!(to_llama_cpp_prefix("lm_head").as_deref(), Some("output"));
    }

    #[test]
    fn test_to_llama_cpp_prefix_attention() {
        assert_eq!(
            to_llama_cpp_prefix("model.layers.5.self_attn.q_proj").as_deref(),
            Some("blk.5.attn_q")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.self_attn.k_proj").as_deref(),
            Some("blk.0.attn_k")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.11.self_attn.v_proj").as_deref(),
            Some("blk.11.attn_v")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.3.self_attn.o_proj").as_deref(),
            Some("blk.3.attn_output")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_qkv_norms() {
        // Gemma 4 QKV norms must map so checkpoints with per-head norms load.
        assert_eq!(
            to_llama_cpp_prefix("model.layers.2.self_attn.q_norm").as_deref(),
            Some("blk.2.attn_q_norm")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.2.self_attn.k_norm").as_deref(),
            Some("blk.2.attn_k_norm")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_mlp() {
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.gate_proj").as_deref(),
            Some("blk.7.ffn_gate")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.up_proj").as_deref(),
            Some("blk.7.ffn_up")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.down_proj").as_deref(),
            Some("blk.7.ffn_down")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_layernorms() {
        // Gemma family has four layernorms per block — verify all land on
        // the llama.cpp names. `post_attention_layernorm` carries two
        // candidates because plain Llama/Mistral store it as `ffn_norm`
        // (pre-FFN position) while Gemma uses a separate
        // `post_attention_norm` tensor.
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.input_layernorm").as_deref(),
            Some("blk.0.attn_norm")
        );
        let post_attn = to_llama_cpp_prefixes("model.layers.0.post_attention_layernorm");
        assert_eq!(
            post_attn,
            vec![
                "blk.0.ffn_norm".to_string(),
                "blk.0.post_attention_norm".to_string(),
            ]
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.pre_feedforward_layernorm").as_deref(),
            Some("blk.0.ffn_norm")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.post_feedforward_layernorm").as_deref(),
            Some("blk.0.post_ffw_norm")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_unknown_returns_none() {
        // Unknown Gemma 4 specifics (PLE, MoE experts) stay unmapped so the
        // caller falls back to the literal path.
        assert!(to_llama_cpp_prefix("model.per_layer_model_projection").is_none());
        assert!(to_llama_cpp_prefix("model.layers.0.per_layer_input_gate").is_none());
        assert!(to_llama_cpp_prefix("model.layers.0.moe.experts.0.gate_proj").is_none());
        assert!(to_llama_cpp_prefix("model.layers.not_a_number.self_attn.q_proj").is_none());
        assert!(to_llama_cpp_prefix("random_tensor").is_none());
    }
}
