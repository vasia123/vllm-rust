//! Ovis2_5 vision-language model.
//!
//! Architecture:
//! - Vision: `Siglip2NavitModel` — SigLIP-2 NaViT ViT with optional 2D RoPE and window attention.
//!   Uses `hidden_stride` (default 2) spatial merge units.
//!   Key difference vs lfm2_vl.rs `Siglip2VisionTransformer`: accepts packed `grid_thws` input
//!   and applies `hidden_stride²` spatial merge when reshaping output.
//!   CPU path uses full attention (ignores windowing for simplicity/correctness).
//! - Projector: `VisualTokenizer` — Siglip2NavitModel → reshape `[N, stride²·D]` →
//!   Linear(stride²·D → head_dim) + LayerNorm → Softmax → pad indicator slots.
//! - Embedding: `vte.weight` — soft visual tokens @ vte `[vocab, hidden]` → `[N, hidden]`.
//! - Language: Qwen2 or Qwen3 (dispatched by `llm_type` config field).
//!
//! # Weight paths
//!
//! ```text
//! visual_tokenizer.vit.vision_model.embeddings.patch_embedding.{weight,bias}
//! visual_tokenizer.vit.vision_model.embeddings.position_embedding.weight  (if preserve_pe)
//! visual_tokenizer.vit.vision_model.encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.*
//! visual_tokenizer.vit.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}.*
//! visual_tokenizer.vit.vision_model.encoder.layers.{i}.{layer_norm1,layer_norm2}.*
//! visual_tokenizer.vit.vision_model.post_layernorm.*
//! visual_tokenizer.head.0.weight                  (linear, no bias)
//! visual_tokenizer.head.1.{weight,bias}            (LayerNorm)
//! vte.weight                                       [vocab_size, llm_hidden_size]
//! llm.model.* / llm.lm_head.*
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/ovis2_5.py`
//! `reference/vllm/vllm/model_executor/models/siglip2navit.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{ops::softmax, LayerNorm, Linear, VarBuilder};
use serde_json::Value;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;
use super::qwen3::Qwen3ForCausalLM;

// ─── Config ───────────────────────────────────────────────────────────────────

struct Siglip2NavitConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    patch_size: usize,
    num_channels: usize,
    /// Spatial merge stride (typically 2); output tokens = seq_len / hidden_stride².
    hidden_stride: usize,
    /// If 0 → Conv2d patch embed; if > 0 → Linear patch embed.
    num_patches: usize,
    layer_norm_eps: f64,
}

impl Siglip2NavitConfig {
    fn from_json(v: &Value) -> Self {
        let g = |k: &str, d: usize| v.get(k).and_then(|x| x.as_u64()).unwrap_or(d as u64) as usize;
        let f = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        Self {
            hidden_size: g("hidden_size", 1152),
            num_attention_heads: g("num_attention_heads", 16),
            num_hidden_layers: g("num_hidden_layers", 27),
            intermediate_size: g("intermediate_size", 4304),
            patch_size: g("patch_size", 14),
            num_channels: g("num_channels", 3),
            hidden_stride: g("hidden_stride", 2),
            num_patches: g("num_patches", 0),
            layer_norm_eps: f("layer_norm_eps", 1e-6),
        }
    }
}

struct Ovis25Config {
    vis: Siglip2NavitConfig,
    /// `visual_vocab_size` from the top-level HF config.
    visual_vocab_size: usize,
    llm_hidden_size: usize,
    llm_type: Ovis25LlmType,
}

enum Ovis25LlmType {
    Qwen2,
    Qwen3,
}

impl Ovis25Config {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vit_json = cfg
            .extra
            .get("vit_config")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));
        let vis = Siglip2NavitConfig::from_json(&vit_json);

        let visual_vocab_size = cfg
            .extra
            .get("visual_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as usize;

        let llm_type_str = cfg
            .extra
            .get("text_config")
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("qwen2");

        let llm_type = if llm_type_str.starts_with("qwen3") {
            Ovis25LlmType::Qwen3
        } else {
            Ovis25LlmType::Qwen2
        };

        Self {
            vis,
            visual_vocab_size,
            llm_hidden_size: cfg.hidden_size,
            llm_type,
        }
    }
}

// ─── Siglip2NavitModel ────────────────────────────────────────────────────────

/// Patch embedding for Siglip2NavitModel.
/// Uses Linear when `num_patches > 0`, otherwise Conv2d.
struct Siglip2NavitPatchEmbed {
    linear: Linear,
    use_linear: bool,
    patch_size: usize,
    num_channels: usize,
}

impl Siglip2NavitPatchEmbed {
    fn new(cfg: &Siglip2NavitConfig, vb: VarBuilder) -> Result<Self> {
        let in_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let linear = candle_nn::linear_no_bias(in_dim, cfg.hidden_size, vb.pp("patch_embedding"))?;
        Ok(Self {
            linear,
            use_linear: cfg.num_patches > 0,
            patch_size: cfg.patch_size,
            num_channels: cfg.num_channels,
        })
    }

    /// `pixel_values`: `[seq_len, C·ps·ps]` (already pre-patchified) OR `[N, C, H, W]`.
    /// Returns `[seq_len, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        if self.use_linear {
            // Already patchified: [seq_len, C·ps·ps]
            self.linear.forward(pixel_values)
        } else {
            // pixel_values: [N, C*temporal_patch_size, ps, ps] — reshape to [N, in_dim]
            let (n, _c, _h, _w) = pixel_values.dims4()?;
            let in_dim = self.num_channels * self.patch_size * self.patch_size;
            let flat = pixel_values.reshape((n, in_dim))?;
            self.linear.forward(&flat)
        }
    }
}

/// Standard pre-norm multi-head self-attention for Siglip2Navit.
/// Weight paths: `self_attn.{q_proj,k_proj,v_proj,out_proj}.*`
struct Siglip2NavitAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Siglip2NavitAttention {
    fn new(cfg: &Siglip2NavitConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let h = cfg.num_attention_heads;
        let hd = d / h;
        let q_proj = candle_nn::linear(d, d, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d, d, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d, d, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d, d, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: h,
            head_dim: hd,
        })
    }

    /// `x`: `[seq_len, hidden_size]` — full attention (no windowing on CPU).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, _d) = x.dims2()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [heads, seq_len, head_dim] for batched matmul
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let k = k
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let v = v
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(1, 2)?)? / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [heads, seq_len, head_dim]
        let out = out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

struct Siglip2NavitMlp {
    fc1: Linear,
    fc2: Linear,
}

impl Siglip2NavitMlp {
    fn new(cfg: &Siglip2NavitConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

struct Siglip2NavitEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Siglip2NavitAttention,
    layer_norm2: LayerNorm,
    mlp: Siglip2NavitMlp,
}

impl Siglip2NavitEncoderLayer {
    fn new(cfg: &Siglip2NavitConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm1: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layer_norm1"),
            )?,
            self_attn: Siglip2NavitAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm2: candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("layer_norm2"),
            )?,
            mlp: Siglip2NavitMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x;
        let x = (h + self.self_attn.forward(&self.layer_norm1.forward(h)?)?)?;
        let x = (&x + self.mlp.forward(&self.layer_norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

/// Siglip2NaViT vision transformer.
///
/// CPU path: full attention for all layers (window reordering omitted).
/// Returns `[seq_len, hidden_size]` — caller reshapes to merged tokens.
struct Siglip2NavitVisionTransformer {
    patch_embed: Siglip2NavitPatchEmbed,
    encoder: Vec<Siglip2NavitEncoderLayer>,
    post_layernorm: LayerNorm,
}

impl Siglip2NavitVisionTransformer {
    fn new(cfg: &Siglip2NavitConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = Siglip2NavitPatchEmbed::new(cfg, vb.pp("embeddings"))?;
        let enc_vb = vb.pp("encoder").pp("layers");
        let encoder = (0..cfg.num_hidden_layers)
            .map(|i| Siglip2NavitEncoderLayer::new(cfg, enc_vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let post_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            patch_embed,
            encoder,
            post_layernorm,
        })
    }

    /// `pixel_values`: `[seq_len, C·ps·ps]` (pre-patchified, NaViT style).
    /// Returns `[seq_len, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(pixel_values)?;
        for layer in &self.encoder {
            x = layer.forward(&x)?;
        }
        self.post_layernorm.forward(&x)
    }
}

// ─── VisualTokenizer ──────────────────────────────────────────────────────────

/// Ovis2_5 visual tokenizer.
///
/// Encodes patches → Siglip2NaViT → reshape stride² → head Linear+LN →
/// Softmax → pad 4 indicator slots → soft visual tokens `[N_merged, vocab_size]`.
struct Ovis25VisualTokenizer {
    vit: Siglip2NavitVisionTransformer,
    head_linear: Linear,
    head_norm: LayerNorm,
    hidden_stride: usize,
    visual_vocab_size: usize,
}

impl Ovis25VisualTokenizer {
    fn new(vis_cfg: &Siglip2NavitConfig, visual_vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        // NOTE: indicator IDs are [-301, -302, -303, -304] → 4 reserved slots
        const NUM_INDICATORS: usize = 4;
        let head_dim = visual_vocab_size - NUM_INDICATORS;
        let merge_dim = vis_cfg.hidden_size * vis_cfg.hidden_stride * vis_cfg.hidden_stride;

        let vit = Siglip2NavitVisionTransformer::new(vis_cfg, vb.pp("vit").pp("vision_model"))?;
        let head_linear = candle_nn::linear_no_bias(merge_dim, head_dim, vb.pp("head").pp("0"))?;
        let head_norm = candle_nn::layer_norm(head_dim, 1e-5, vb.pp("head").pp("1"))?;

        Ok(Self {
            vit,
            head_linear,
            head_norm,
            hidden_stride: vis_cfg.hidden_stride,
            visual_vocab_size,
        })
    }

    /// `pixel_values`: `[seq_len, C·ps·ps]` (packed NaViT patches).
    /// Returns `[N_merged, visual_vocab_size]` soft tokens.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let feats = self.vit.forward(pixel_values)?; // [seq_len, D]
        let (seq_len, hidden_size) = feats.dims2()?;
        let stride2 = self.hidden_stride * self.hidden_stride;
        let n_merged = seq_len / stride2;
        let feats = feats.reshape((n_merged, stride2 * hidden_size))?; // [N_merged, stride²·D]

        let logits = self.head_norm.forward(&self.head_linear.forward(&feats)?)?; // [N_merged, head_dim]
        let tokens = softmax(&logits, 1)?; // [N_merged, head_dim]

        // Pad 4 indicator slots with zeros → [N_merged, vocab_size]
        let n_indicator = self.visual_vocab_size - logits.dim(1)?;
        let pad = Tensor::zeros((n_merged, n_indicator), tokens.dtype(), tokens.device())?;
        Tensor::cat(&[&tokens, &pad], 1)
    }
}

// ─── LLM enum ─────────────────────────────────────────────────────────────────

enum Ovis25Llm {
    Qwen2(Box<Qwen2ForCausalLM>),
    Qwen3(Box<Qwen3ForCausalLM>),
}

impl Ovis25Llm {
    fn new(llm_type: &Ovis25LlmType, cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        match llm_type {
            Ovis25LlmType::Qwen2 => Ok(Self::Qwen2(Box::new(Qwen2ForCausalLM::new(cfg, vb)?))),
            Ovis25LlmType::Qwen3 => Ok(Self::Qwen3(Box::new(Qwen3ForCausalLM::new(cfg, vb)?))),
        }
    }

    fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.embed_text(input_ids),
            Self::Qwen3(m) => m.embed_text(input_ids),
        }
    }

    fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.forward_with_embeddings(
                embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Qwen3(m) => m.forward_with_embeddings(
                embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
        }
    }

    fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        match self {
            Self::Qwen2(m) => {
                m.forward_decode_batch_with_embeddings(embeddings, sequences, kv_cache_mgr)
            }
            Self::Qwen3(m) => {
                m.forward_decode_batch_with_embeddings(embeddings, sequences, kv_cache_mgr)
            }
        }
    }
}

// ─── merge_multimodal ─────────────────────────────────────────────────────────

fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }
    let (_b, seq_len, _d) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;
    for (position, processed) in &mm_inputs.image_embeddings {
        let emb: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
        let batch_idx = position / seq_len;
        let start_pos = position % seq_len;
        for (i, row) in emb.iter().enumerate() {
            let tgt = start_pos + i;
            if tgt < seq_len && batch_idx < merged.len() {
                merged[batch_idx][tgt] = row.clone();
            }
        }
    }
    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main Model ───────────────────────────────────────────────────────────────

/// Ovis2_5 vision-language model.
pub struct Ovis2_5ForConditionalGeneration {
    visual_tokenizer: Ovis25VisualTokenizer,
    /// Visual token embedding: `[visual_vocab_size, llm_hidden_size]`.
    vte_weight: Tensor,
    llm: Ovis25Llm,
    visual_vocab_size: usize,
    device: Device,
}

impl Ovis2_5ForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ovis_cfg = Ovis25Config::from_model_config(cfg);
        let device = vb.device().clone();

        let visual_tokenizer = Ovis25VisualTokenizer::new(
            &ovis_cfg.vis,
            ovis_cfg.visual_vocab_size,
            vb.pp("visual_tokenizer"),
        )?;

        let vte_weight = vb.pp("vte").get(
            (ovis_cfg.visual_vocab_size, ovis_cfg.llm_hidden_size),
            "weight",
        )?;

        let llm = Ovis25Llm::new(&ovis_cfg.llm_type, cfg, vb.pp("llm"))?;

        Ok(Self {
            visual_tokenizer,
            vte_weight,
            llm,
            visual_vocab_size: ovis_cfg.visual_vocab_size,
            device,
        })
    }

    /// Encode image patches to LLM-space embeddings.
    ///
    /// `pixel_values`: `[seq_len, C·ps·ps]` (pre-patchified).
    /// Returns `[N_merged, llm_hidden_size]`.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let soft_tokens = self.visual_tokenizer.forward(pixel_values)?; // [N_merged, vocab_size]
        let (n_merged, _) = soft_tokens.dims2()?;
        let tokens = soft_tokens.reshape((n_merged, self.visual_vocab_size))?;
        tokens.matmul(&self.vte_weight)
    }
}

impl crate::engine::ModelForward for Ovis2_5ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.llm.embed_text(input_ids)?;
        self.llm.forward_with_embeddings(
            &embeddings,
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
        let embeddings = self.llm.embed_text(input_ids)?;
        self.llm
            .forward_decode_batch_with_embeddings(&embeddings, sequences, kv_cache_mgr)
    }

    fn device(&self) -> &Device {
        &self.device
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
        let text_embeddings = self.llm.embed_text(input_ids)?;
        let embeddings = if let Some(mm) = multimodal_inputs {
            merge_multimodal(&text_embeddings, mm, &self.device)?
        } else {
            text_embeddings
        };
        self.llm.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use candle_core::DType;
    use serde_json::json;

    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vit_config".to_string(),
            json!({
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 8,
                "patch_size": 2,
                "num_channels": 1,
                "hidden_stride": 2,
                "num_patches": 16,
                "layer_norm_eps": 1e-6,
            }),
        );
        // visual_vocab_size = head_dim + 4 indicators; use 12 so head_dim=8
        extra.insert("visual_vocab_size".to_string(), json!(12u32));
        extra.insert("text_config".to_string(), json!({ "model_type": "qwen2" }));

        ModelConfig {
            architectures: vec!["Ovis2_5".to_string()],
            hidden_size: 32,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 64,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 4,
            num_blocks: 32,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_ovis2_5_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ovis2_5ForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_ovis2_5_encode_images() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ovis2_5ForConditionalGeneration::new(&cfg, vb).unwrap();

        // 16 patches (4×4 grid, hidden_stride=2 → 4 merged tokens)
        // patch_input_dim = 1 * 2 * 2 = 4
        let pixel_values = Tensor::zeros((16usize, 4), DType::F32, &device).unwrap();
        let result = model.encode_images(&pixel_values);
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        // 16 patches / 2² = 4 merged tokens, llm_hidden_size=32
        assert_eq!(result.unwrap().dims(), &[4, 32]);
    }

    #[test]
    fn test_ovis2_5_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Ovis2_5ForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);
        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }
}
