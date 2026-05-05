//! Quantized Gemma 4 Vision-Language model.
//!
//! Thin wrapper around `QuantizedGemma4ForCausalLM` that adds the SigLIP
//! vision tower and the `Gemma4MultimodalEmbedder` (Linear + unweighted
//! RMSNorm). The vision components stay in full precision — SigLIP is small
//! enough (~150M params) that quantization savings are not worth the
//! kernel-level complexity, and unsloth / bartowski ship Gemma 4 VLMs with
//! FP16/BF16 vision towers even when the LLM is 4-bit.
//!
//! Weight layout (mirrors `gemma4_vlm.rs`):
//! - `vision_tower.*`                             → SigLIP encoder (FP)
//! - `embed_vision.embedding_projection.*`        → multimodal projector (FP)
//! - `language_model.model.*`                     → quantized Gemma 4 backbone
//! - `language_model.lm_head.*`                   → quantized head (if untied)
//!
//! Since `QuantizedGemma4ForCausalLM::new` embeds `model.*` as the top-level
//! weight-loader prefix, we wrap the provided loader with
//! `PrefixedWeightLoader` to prepend `language_model.` during VLM construction.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::gemma4_vision::Gemma4VisionTower;
use crate::quantization::{QuantizationMethod, QuantizedLinear, QuantizedWeightLoader};

use super::gemma4_quantized::QuantizedGemma4ForCausalLM;
use super::gemma4_vlm::Gemma4VLMConfig;

// ─── Remapping Weight Loader ─────────────────────────────────────────────
//
// Rewrites a path prefix on every `load_linear` call before forwarding to
// the underlying loader. The Gemma 4 quantized text model emits loader
// prefixes of the form `"model.layers.N.*"` and `"model.per_layer_*"`.
// In a VLM checkpoint those tensors actually live under
// `"model.language_model.layers.N.*"`, so the VLM wrapper installs a
// remapper that replaces the leading `"model."` with `"model.language_model."`.
//
// When `from_prefix` is the empty string the remapper degenerates to a
// plain prefix-prepend — the behaviour of the old `PrefixedWeightLoader`
// — which is useful for checkpoints where the text tensors live at
// `"<prefix>.embed_tokens"` etc. without a leading `"model."`.

pub struct RemappingWeightLoader<'a> {
    inner: &'a dyn QuantizedWeightLoader,
    from_prefix: String,
    to_prefix: String,
}

impl<'a> RemappingWeightLoader<'a> {
    pub fn new(
        inner: &'a dyn QuantizedWeightLoader,
        from_prefix: impl Into<String>,
        to_prefix: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            from_prefix: from_prefix.into(),
            to_prefix: to_prefix.into(),
        }
    }

    fn rewrite(&self, prefix: &str) -> String {
        if self.from_prefix.is_empty() {
            return if self.to_prefix.is_empty() {
                prefix.to_string()
            } else {
                format!("{}.{}", self.to_prefix, prefix)
            };
        }
        if let Some(rest) = prefix.strip_prefix(&self.from_prefix) {
            // Preserve the boundary dot: `"model.X"` → `"<to>.X"` by
            // re-inserting a single `.` after `to_prefix` when needed.
            let rest = rest.strip_prefix('.').unwrap_or(rest);
            if rest.is_empty() {
                self.to_prefix.clone()
            } else if self.to_prefix.is_empty() {
                rest.to_string()
            } else {
                format!("{}.{rest}", self.to_prefix)
            }
        } else {
            prefix.to_string()
        }
    }
}

impl<'a> QuantizedWeightLoader for RemappingWeightLoader<'a> {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let rewritten = self.rewrite(prefix);
        self.inner
            .load_linear(&rewritten, in_features, out_features, bias)
    }

    fn method(&self) -> QuantizationMethod {
        self.inner.method()
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
}

// ─── Multimodal Embedder (FP) ────────────────────────────────────────────
//
// Duplicated from `gemma4_vlm.rs::Gemma4MultimodalEmbedder` rather than
// reaching into a private type — the struct is trivial (a Linear + an
// unweighted RMSNorm) and keeping it local avoids cross-file coupling.

struct Gemma4MultimodalEmbedder {
    embedding_projection: candle_nn::Linear,
    eps: f64,
}

impl Gemma4MultimodalEmbedder {
    fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding_projection = candle_nn::linear_no_bias(
            multimodal_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;

        Ok(Self {
            embedding_projection,
            eps: rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.embedding_projection.forward(x)?;
        let dtype = projected.dtype();
        let projected_f32 = projected.to_dtype(DType::F32)?;
        let variance = projected_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = projected_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        normed.to_dtype(dtype)
    }
}

// ─── Quantized VLM ────────────────────────────────────────────────────────

pub struct QuantizedGemma4ForConditionalGeneration {
    vision_tower: Gemma4VisionTower,
    embed_vision: Gemma4MultimodalEmbedder,
    language_model: QuantizedGemma4ForCausalLM,
    #[allow(dead_code)]
    config: Gemma4VLMConfig,
    device: Device,
    dtype: DType,
}

impl QuantizedGemma4ForConditionalGeneration {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder<'static>,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let config = Gemma4VLMConfig::from_model_config(cfg);

        // Real HF `Gemma4ForConditionalGeneration` checkpoints wrap
        // every sub-module under a top-level `model.*` key:
        //   model.vision_tower.*
        //   model.audio_tower.*
        //   model.embed_vision.*
        //   model.language_model.{embed_tokens, layers, ...}
        //
        // Descend once here so sub-module VarBuilders are positioned at
        // the right root; `vb_m` / `vb` (the raw root) coexist so the
        // default-constructed `VarBuilder::zeros` path used by unit
        // tests (which has no `model.` wrapper) still works via the
        // `RemappingWeightLoader` fast path below.
        let vb_root = vb.pp("model");

        let vision_tower =
            Gemma4VisionTower::new(&config.vision_config, vb_root.pp("vision_tower"))?;

        let embed_vision = Gemma4MultimodalEmbedder::new(
            config.vision_hidden_size,
            cfg.hidden_size,
            config.vision_config.rms_norm_eps,
            vb_root.pp("embed_vision"),
        )?;

        // Language model lives at `model.language_model.*`. Hand the
        // inner quantized constructor a VarBuilder already positioned at
        // that namespace (via `new_at_model_root`) and wrap the base
        // weight_loader in a `RemappingWeightLoader` that turns the
        // text model's `"model.X"` load paths into
        // `"model.language_model.X"` against the safetensors root.
        let vb_lm = vb_root.pp("language_model");
        let lm_loader = RemappingWeightLoader::new(weight_loader, "model", "model.language_model");
        let language_model = QuantizedGemma4ForCausalLM::new_at_model_root(cfg, vb_lm, &lm_loader)?;

        Ok(Self {
            vision_tower,
            embed_vision,
            language_model,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Encode images: Gemma 4 vision tower → embedder projection.
    ///
    /// `pixel_values`       : `[B, L, 3·ps·ps]` flattened patches.
    /// `pixel_position_ids` : `i64 [B, L, 2]` (`-1, -1` marks padding).
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
    ) -> Result<Tensor> {
        let vision_features = self
            .vision_tower
            .forward(pixel_values, pixel_position_ids)?;
        self.embed_vision.forward(&vision_features)
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden_size) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.embed_vision.forward(&vision_emb)?;
            let projected = projected.squeeze(0)?;
            let emb_vec: Vec<Vec<f32>> = projected.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for QuantizedGemma4ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.language_model.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_decode_batch_with_embeddings(
            &embeddings,
            sequences,
            kv_cache_mgr,
        )
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let text_embeddings = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm_inputs) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm_inputs)?
        } else {
            text_embeddings
        };

        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::multimodal::ProcessedImage;
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

    fn test_vlm_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
        extra.insert(
            "hidden_size_per_layer_input".to_string(),
            serde_json::json!(16),
        );
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "num_channels": 3,
                "rms_norm_eps": 1e-6,
                "default_output_length": 4,
                "model_type": "siglip"
            }),
        );
        extra.insert("image_token_index".to_string(), serde_json::json!(262144));

        ModelConfig {
            architectures: vec!["Gemma4ForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: Some(256),
            attention_bias: Some(false),
            extra,
        }
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_remapping_loader_rewrites_model_prefix() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let base = create_weight_loader_with_params(vb, &DetectedQuantConfig::default());
        let remap = RemappingWeightLoader::new(base.as_ref(), "model", "model.language_model");

        // "model.X" should become "model.language_model.X".
        assert_eq!(
            remap.rewrite("model.layers.0.self_attn.q_proj"),
            "model.language_model.layers.0.self_attn.q_proj"
        );
        // Non-matching prefixes pass through unchanged.
        assert_eq!(remap.rewrite("lm_head"), "lm_head");
        // Exact match rewrites to the target prefix only.
        assert_eq!(remap.rewrite("model"), "model.language_model");

        assert_eq!(remap.method(), QuantizationMethod::None);

        // Empty `from_prefix` degenerates to plain prepend (matches the
        // legacy `PrefixedWeightLoader` behaviour).
        let prepend = RemappingWeightLoader::new(base.as_ref(), "", "language_model");
        assert_eq!(prepend.rewrite("model.X"), "language_model.model.X");
    }

    #[test]
    fn test_quantized_vlm_construction() {
        let device = Device::Cpu;
        let cfg = test_vlm_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let loader = create_weight_loader_with_params(vb.clone(), &DetectedQuantConfig::default());
        let model = QuantizedGemma4ForConditionalGeneration::new(&cfg, vb, loader.as_ref());

        assert!(
            model.is_ok(),
            "QuantizedGemma4ForConditionalGeneration should construct: {:?}",
            model.err()
        );
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_quantized_vlm_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &DetectedQuantConfig::default());
        let model =
            QuantizedGemma4ForConditionalGeneration::new(&cfg, vb, loader.as_ref()).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_vlm_multimodal_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &DetectedQuantConfig::default());
        let model =
            QuantizedGemma4ForConditionalGeneration::new(&cfg, vb, loader.as_ref()).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        let seq_len = 8;
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        let img_embedding = Tensor::randn(0f32, 1.0, (4, 32), &device).unwrap();
        let processed = ProcessedImage::new(img_embedding, 4);
        let mm_inputs = MultimodalInputs::with_images(vec![0u32; seq_len], vec![(0, processed)]);

        let logits = model
            .forward_multimodal(
                &input_ids,
                Some(&mm_inputs),
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }
}
