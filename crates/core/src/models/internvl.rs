//! InternVL2 vision-language model implementation.
//!
//! InternVL2 combines an InternViT vision encoder with various LLM backbones
//! (typically InternLM2) using a pixel-shuffle downsampler and MLP projector.
//!
//! Key architectural features:
//! - InternViT: Deep ViT with learned position embeddings, CLS token, layer scaling
//! - Optional QK normalization (RMSNorm on Q and K before attention)
//! - Pixel shuffle: spatial downsampling that reduces tokens by (1/downsample_ratio)^2
//! - mlp1 projector: LayerNorm + Linear + GELU + Linear
//!
//! Reference: InternVL2 (https://arxiv.org/abs/2404.16821)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    layer_norm, rms_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, RmsNorm, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::internlm2::InternLM2ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// InternViT vision encoder configuration.
#[derive(Debug, Clone)]
pub struct InternVLVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub qkv_bias: bool,
    pub qk_normalization: bool,
    pub layer_norm_eps: f64,
    pub use_rms_norm: bool,
    pub initializer_factor: f64,
}

impl Default for InternVLVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3200,
            intermediate_size: 12800,
            num_attention_heads: 25,
            num_hidden_layers: 45,
            image_size: 448,
            patch_size: 14,
            qkv_bias: true,
            qk_normalization: true,
            layer_norm_eps: 1e-6,
            use_rms_norm: true,
            initializer_factor: 0.1,
        }
    }
}

impl InternVLVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Total sequence length including CLS token.
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
        let norm_type = json
            .get("norm_type")
            .and_then(|v| v.as_str())
            .unwrap_or("rms_norm");
        Self {
            hidden_size: json
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.hidden_size as u64) as usize,
            intermediate_size: json
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.intermediate_size as u64)
                as usize,
            num_attention_heads: json
                .get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_attention_heads as u64)
                as usize,
            num_hidden_layers: json
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_hidden_layers as u64)
                as usize,
            image_size: json
                .get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_size as u64) as usize,
            patch_size: json
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize,
            qkv_bias: json
                .get("qkv_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.qkv_bias),
            qk_normalization: json
                .get("qk_normalization")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.qk_normalization),
            layer_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps),
            use_rms_norm: norm_type == "rms_norm",
            initializer_factor: json
                .get("initializer_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.initializer_factor),
        }
    }
}

/// Top-level InternVL configuration.
#[derive(Debug, Clone)]
pub struct InternVLConfig {
    pub model_config: ModelConfig,
    pub vision_config: InternVLVisionConfig,
    pub downsample_ratio: f64,
    pub select_layer: i32,
    pub image_token_id: u32,
}

impl InternVLConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(InternVLVisionConfig::from_json)
            .unwrap_or_default();

        let downsample_ratio = cfg
            .extra
            .get("downsample_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let select_layer = cfg
            .extra
            .get("select_layer")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1) as i32;

        // image_token_id for <IMG_CONTEXT>
        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151667) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            downsample_ratio,
            select_layer,
            image_token_id,
        }
    }

    /// Effective number of vision encoder layers to run.
    pub(crate) fn effective_depth(&self) -> usize {
        let total = self.vision_config.num_hidden_layers as i32;
        if self.select_layer < 0 {
            (total + self.select_layer + 1) as usize
        } else {
            self.select_layer as usize
        }
    }

    /// Number of image tokens per tile after pixel shuffle.
    pub fn num_image_tokens_per_tile(&self) -> usize {
        let grid = self.vision_config.image_size / self.vision_config.patch_size;
        let dr = self.downsample_ratio;
        ((grid as f64 * dr) as usize).pow(2)
    }

    /// Projector input dimension: vit_hidden * (1/downsample_ratio)^2.
    pub fn projector_input_dim(&self) -> usize {
        let scale = (1.0 / self.downsample_ratio) as usize;
        self.vision_config.hidden_size * scale * scale
    }
}

// ─── Vision Encoder ─────────────────────────────────────────────────────────

/// Norm that can be either LayerNorm or RmsNorm depending on config.
enum VisionNorm {
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
}

impl VisionNorm {
    fn new(hidden_size: usize, eps: f64, use_rms: bool, vb: VarBuilder) -> Result<Self> {
        if use_rms {
            Ok(Self::RmsNorm(rms_norm(hidden_size, eps, vb)?))
        } else {
            Ok(Self::LayerNorm(layer_norm(hidden_size, eps, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(ln) => ln.forward(x),
            Self::RmsNorm(rn) => rn.forward(x),
        }
    }
}

/// Vision embeddings: Conv2d patch embedding + learned position embedding + CLS token.
#[allow(dead_code)]
struct InternVisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Tensor,
    num_patches: usize,
}

#[allow(dead_code)]
impl InternVisionEmbeddings {
    fn new(cfg: &InternVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches = cfg.num_patches();

        // CLS token: [1, 1, hidden_size]
        let class_embedding = vb.get((1, 1, cfg.hidden_size), "class_embedding")?;

        // Conv2d patch embedding: [hidden_size, 3, patch_size, patch_size]
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = candle_nn::conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;

        // Learned position embedding: [1, num_patches+1, hidden_size]
        let position_embedding =
            vb.get((1, num_patches + 1, cfg.hidden_size), "position_embedding")?;

        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
            num_patches,
        })
    }

    /// Forward: pixel_values [B, 3, H, W] -> [B, num_patches+1, hidden_size]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;

        // Conv2d: [B, 3, H, W] -> [B, hidden_size, H/patch, W/patch]
        let patches = self.patch_embedding.forward(pixel_values)?;

        // Flatten spatial dims: [B, hidden_size, num_patches] -> [B, num_patches, hidden_size]
        let (_b, c, _h, _w) = patches.dims4()?;
        let patches = patches
            .reshape((batch_size, c, self.num_patches))?
            .transpose(1, 2)?;

        // Prepend CLS token
        let cls = self
            .class_embedding
            .broadcast_left(batch_size)?
            .reshape((batch_size, 1, c))?;
        let embeddings = Tensor::cat(&[cls, patches], 1)?;

        // Add position embeddings
        embeddings + &self.position_embedding
    }
}

/// Vision attention with fused QKV and optional QK normalization.
#[allow(dead_code)]
struct InternVisionAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl InternVisionAttention {
    fn new(cfg: &InternVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv = candle_nn::linear_b(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            cfg.qkv_bias,
            vb.pp("qkv"),
        )?;
        let proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("proj"))?;

        let (q_norm, k_norm) = if cfg.qk_normalization {
            (
                Some(rms_norm(
                    cfg.head_dim(),
                    cfg.layer_norm_eps,
                    vb.pp("q_norm"),
                )?),
                Some(rms_norm(
                    cfg.head_dim(),
                    cfg.layer_norm_eps,
                    vb.pp("k_norm"),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Bidirectional self-attention (no causal mask).
    /// x: [B, seq_len, hidden_size]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(2, 0, self.num_heads * self.head_dim)?;
        let k = qkv.narrow(
            2,
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;
        let v = qkv.narrow(
            2,
            2 * self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;

        // [B, seq_len, heads, head_dim] -> [B, heads, seq_len, head_dim]
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Optional QK normalization (per-head RMSNorm)
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q.contiguous()?)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k.contiguous()?)?
        } else {
            k
        };

        // Scaled dot-product attention (bidirectional)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, heads, seq_len, head_dim] -> [B, seq_len, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            b,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.proj.forward(&attn_output)
    }
}

/// Vision MLP: fc1 + GELU + fc2 (with bias).
#[allow(dead_code)]
struct InternVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl InternVisionMLP {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(intermediate_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

/// Vision encoder block with layer scaling.
///
/// Forward: x + attn(norm1(x)) * ls1 + mlp(norm2(x + attn(norm1(x)) * ls1)) * ls2
#[allow(dead_code)]
struct InternVisionBlock {
    norm1: VisionNorm,
    attn: InternVisionAttention,
    norm2: VisionNorm,
    mlp: InternVisionMLP,
    ls1: Tensor,
    ls2: Tensor,
}

#[allow(dead_code)]
impl InternVisionBlock {
    fn new(cfg: &InternVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = VisionNorm::new(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            cfg.use_rms_norm,
            vb.pp("norm1"),
        )?;
        let attn = InternVisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = VisionNorm::new(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            cfg.use_rms_norm,
            vb.pp("norm2"),
        )?;
        let mlp = InternVisionMLP::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;

        let ls1 = vb.get(cfg.hidden_size, "ls1")?;
        let ls2 = vb.get(cfg.hidden_size, "ls2")?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            ls1,
            ls2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let xs = self.norm1.forward(x)?;
        let xs = self.attn.forward(&xs)?;
        let xs = xs.broadcast_mul(&self.ls1)?;
        let x = (residual + xs)?;

        let residual = &x;
        let xs = self.norm2.forward(&x)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = xs.broadcast_mul(&self.ls2)?;
        residual + xs
    }
}

/// InternViT vision encoder.
#[allow(dead_code)]
pub(crate) struct InternVisionModel {
    embeddings: InternVisionEmbeddings,
    blocks: Vec<InternVisionBlock>,
}

#[allow(dead_code)]
impl InternVisionModel {
    pub(crate) fn new(cfg: &InternVLVisionConfig, depth: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = InternVisionEmbeddings::new(cfg, vb.pp("embeddings"))?;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(InternVisionBlock::new(
                cfg,
                vb.pp("encoder").pp("layers").pp(i),
            )?);
        }

        Ok(Self { embeddings, blocks })
    }

    /// Forward: pixel_values [B, 3, H, W] -> [B, num_patches+1, hidden_size]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.embeddings.forward(pixel_values)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

// ─── Pixel Shuffle + Projector ──────────────────────────────────────────────

/// Pixel shuffle: spatial downsampling by merging adjacent patches.
///
/// Input: [B, H*W, C] -> Output: [B, (H*s)*(W*s), C/(s*s)]
/// where s = downsample_ratio (typically 0.5, meaning spatial halved, channels 4x)
#[allow(dead_code)]
pub(crate) fn pixel_shuffle(x: &Tensor, grid_size: usize, downsample_ratio: f64) -> Result<Tensor> {
    let (batch, _seq, channels) = x.dims3()?;
    let h = grid_size;
    let w = grid_size;
    let scale = downsample_ratio;

    let new_h = (h as f64 * scale) as usize;
    let new_w = (w as f64 * scale) as usize;
    let channels_out = channels * h * w / (new_h * new_w);

    // Reshape: [B, H, W, C] -> rearrange for downsampling
    let x = x.reshape((batch, h, w, channels))?;

    // For downsample_ratio=0.5: group 2x2 spatial patches
    let factor = (1.0 / scale) as usize;
    // [B, H, W, C] -> [B, new_h, factor, new_w, factor, C] -> [B, new_h, new_w, factor*factor*C]
    let x = x.reshape((batch, new_h, factor, new_w, factor, channels))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?;
    let x = x.reshape((batch, new_h * new_w, channels_out))?;

    x.contiguous()
}

/// mlp1 projector: LayerNorm + Linear + GELU + Linear.
#[allow(dead_code)]
struct InternVLProjector {
    ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl InternVLProjector {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ln = layer_norm(input_dim, 1e-6, vb.pp("0"))?;
        let fc1 = candle_nn::linear(input_dim, output_dim, vb.pp("1"))?;
        let fc2 = candle_nn::linear(output_dim, output_dim, vb.pp("3"))?;
        Ok(Self { ln, fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// InternVL2 model for conditional generation.
///
/// Wraps InternViT vision encoder + pixel shuffle + mlp1 projector + InternLM2 LLM.
pub struct InternVLChatModel {
    #[allow(dead_code)]
    vision_model: InternVisionModel,
    #[allow(dead_code)]
    projector: InternVLProjector,
    language_model: InternLM2ForCausalLM,
    #[allow(dead_code)]
    config: InternVLConfig,
    device: Device,
    dtype: DType,
}

impl InternVLChatModel {
    pub fn new(cfg: &InternVLConfig, vb: VarBuilder) -> Result<Self> {
        let depth = cfg.effective_depth();
        let vision_model =
            InternVisionModel::new(&cfg.vision_config, depth, vb.pp("vision_model"))?;

        let proj_input = cfg.projector_input_dim();
        let projector =
            InternVLProjector::new(proj_input, cfg.model_config.hidden_size, vb.pp("mlp1"))?;

        let language_model = InternLM2ForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            projector,
            language_model,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ivl_cfg = InternVLConfig::from_model_config(cfg);
        Self::new(&ivl_cfg, vb)
    }

    /// Merge pre-encoded image embeddings with text embeddings.
    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
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

impl crate::engine::ModelForward for InternVLChatModel {
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
        // Decode: text-only (images already processed in prefill)
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

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["InternVLChatModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
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
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> InternVLVisionConfig {
        InternVLVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            qkv_bias: true,
            qk_normalization: true,
            layer_norm_eps: 1e-6,
            use_rms_norm: true,
            initializer_factor: 0.1,
        }
    }

    fn test_internvl_config() -> InternVLConfig {
        InternVLConfig {
            model_config: test_model_config(),
            vision_config: test_vision_config(),
            downsample_ratio: 0.5,
            select_layer: -1,
            image_token_id: 151667,
        }
    }

    // ── Config Tests ────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = InternVLVisionConfig::default();
        assert_eq!(cfg.hidden_size, 3200);
        assert_eq!(cfg.num_hidden_layers, 45);
        assert_eq!(cfg.num_attention_heads, 25);
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.image_size, 448);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches(), 1024); // (448/14)^2
        assert_eq!(cfg.seq_len(), 1025); // patches + CLS
    }

    #[test]
    fn test_vision_config_from_json() {
        let json = serde_json::json!({
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "image_size": 224,
            "patch_size": 16,
            "qkv_bias": false,
            "qk_normalization": false,
            "norm_type": "layer_norm"
        });
        let cfg = InternVLVisionConfig::from_json(&json);
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert!(!cfg.qkv_bias);
        assert!(!cfg.qk_normalization);
        assert!(!cfg.use_rms_norm);
    }

    #[test]
    fn test_internvl_config_effective_depth() {
        let mut cfg = test_internvl_config();
        cfg.vision_config.num_hidden_layers = 48;
        cfg.select_layer = -1;
        assert_eq!(cfg.effective_depth(), 48); // all layers

        cfg.select_layer = -12;
        assert_eq!(cfg.effective_depth(), 37); // 48 - 12 + 1
    }

    #[test]
    fn test_internvl_config_projector_dim() {
        let cfg = test_internvl_config();
        // downsample_ratio=0.5, scale=2, hidden=64
        assert_eq!(cfg.projector_input_dim(), 64 * 4); // 256
    }

    #[test]
    fn test_internvl_config_from_model_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "image_size": 28,
                "patch_size": 14
            }),
        );
        model_cfg
            .extra
            .insert("downsample_ratio".to_string(), serde_json::json!(0.5));
        model_cfg
            .extra
            .insert("select_layer".to_string(), serde_json::json!(-1));

        let cfg = InternVLConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.hidden_size, 128);
        assert_eq!(cfg.vision_config.num_hidden_layers, 4);
        assert_eq!(cfg.downsample_ratio, 0.5);
    }

    // ── Vision Components Tests ─────────────────────────────────────────

    #[test]
    fn test_vision_embeddings() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = InternVisionEmbeddings::new(&cfg, vb).unwrap();

        // [1, 3, 28, 28] -> [1, num_patches+1, hidden_size]
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = emb.forward(&pixel_values).unwrap();
        // 28/14 = 2, so 2*2 = 4 patches + 1 CLS = 5
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_attention() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = InternVisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_attention_no_qk_norm() {
        let device = Device::Cpu;
        let mut cfg = test_vision_config();
        cfg.qk_normalization = false;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = InternVisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_mlp() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = InternVisionMLP::new(64, 128, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_block() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = InternVisionBlock::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = block.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_vision_model() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternVisionModel::new(&cfg, 2, vb).unwrap();

        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]); // 4 patches + 1 CLS
    }

    // ── Pixel Shuffle Tests ─────────────────────────────────────────────

    #[test]
    fn test_pixel_shuffle_basic() {
        let device = Device::Cpu;
        // 4x4 grid, 64-dim -> downsample 0.5 -> 2x2 grid, 256-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 16, 64), &device).unwrap();
        let out = pixel_shuffle(&x, 4, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 4, 256]); // 2*2=4 tokens, 64*4=256 channels
    }

    #[test]
    fn test_pixel_shuffle_2x2() {
        let device = Device::Cpu;
        // 2x2 grid, 64-dim -> downsample 0.5 -> 1x1 grid, 256-dim
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device).unwrap();
        let out = pixel_shuffle(&x, 2, 0.5).unwrap();
        assert_eq!(out.dims(), &[1, 1, 256]);
    }

    #[test]
    fn test_pixel_shuffle_batched() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (3, 16, 32), &device).unwrap();
        let out = pixel_shuffle(&x, 4, 0.5).unwrap();
        assert_eq!(out.dims(), &[3, 4, 128]);
    }

    // ── Projector Tests ─────────────────────────────────────────────────

    #[test]
    fn test_projector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = InternVLProjector::new(256, 64, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 256), &device).unwrap();
        let output = proj.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    // ── Full Model Tests ────────────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_internvl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternVLChatModel::new(&cfg, vb);
        assert!(model.is_ok());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_model_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_internvl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternVLChatModel::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.model_config.num_hidden_layers,
            num_kv_heads: cfg.model_config.num_key_value_heads,
            head_dim: cfg.model_config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_model_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_internvl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternVLChatModel::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.model_config.num_hidden_layers,
            num_kv_heads: cfg.model_config.num_key_value_heads,
            head_dim: cfg.model_config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "norm_type": "rms_norm"
            }),
        );
        model_cfg
            .extra
            .insert("downsample_ratio".to_string(), serde_json::json!(0.5));

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = InternVLChatModel::from_model_config(&model_cfg, vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_vision_norm_rms() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = VisionNorm::new(64, 1e-6, true, vb).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device).unwrap();
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_vision_norm_layernorm() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = VisionNorm::new(64, 1e-6, false, vb).unwrap();
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device).unwrap();
        let output = norm.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }
}
