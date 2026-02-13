//! Llama 4 Vision-Language model (MLlama4).
//!
//! Combines a custom ViT vision encoder with the Llama4 MoE language model
//! via a pixel-shuffle MLP adapter and a linear multimodal projector.
//!
//! # Architecture
//!
//! 1. Unfold-based patch embedding (equivalent to nn.Unfold + linear)
//! 2. Class embedding prepended, then positional embedding added
//! 3. LayerNorm (pre + post) sandwiching a standard ViT encoder
//! 4. CLS token removed from output
//! 5. Pixel-shuffle MLP reduces spatial tokens (shuffle_ratio < 1 => downsample)
//! 6. Linear projector maps vision output dim to text hidden size
//! 7. Llama4ForCausalLM (MoE backbone) generates text
//!
//! # Weight mapping
//!
//! - `vision_model.patch_embedding.*` -> unfold convolution
//! - `vision_model.class_embedding` -> CLS token parameter
//! - `vision_model.positional_embedding_vlm` -> position embeddings
//! - `vision_model.layernorm_pre/post` -> LayerNorm
//! - `vision_model.model.layers.*` -> ViT encoder layers
//! - `vision_model.vision_adapter.*` -> pixel-shuffle MLP
//! - `multi_modal_projector.linear_1.*` -> linear projection
//! - `language_model.*` -> Llama4 backbone
//!
//! Reference: Meta Llama 4 technical report; vLLM mllama4.py

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::llama4::Llama4ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Vision encoder configuration extracted from `vision_config` in config.json.
#[derive(Debug, Clone)]
pub struct Llama4VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub pixel_shuffle_ratio: f64,
    pub projector_input_dim: usize,
    pub projector_output_dim: usize,
    pub multi_modal_projector_bias: bool,
    pub layer_norm_eps: f64,
}

impl Default for Llama4VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_attention_heads: 16,
            num_hidden_layers: 32,
            image_size: 560,
            patch_size: 14,
            num_channels: 3,
            pixel_shuffle_ratio: 0.5,
            projector_input_dim: 5120,
            projector_output_dim: 5120,
            multi_modal_projector_bias: false,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Llama4VisionConfig {
    /// Number of patches per image side.
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches (before CLS token).
    pub fn num_patches(&self) -> usize {
        self.patches_per_side() * self.patches_per_side()
    }

    /// Sequence length including CLS token.
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }

    /// Number of output tokens after pixel shuffle.
    pub fn num_output_tokens(&self) -> usize {
        let ds_ratio =
            (1.0 / (self.pixel_shuffle_ratio * self.pixel_shuffle_ratio)).round() as usize;
        self.num_patches() / ds_ratio
    }

    /// Hidden dimension after pixel shuffle (before MLP).
    pub fn shuffled_hidden_dim(&self) -> usize {
        let ds_ratio =
            (1.0 / (self.pixel_shuffle_ratio * self.pixel_shuffle_ratio)).round() as usize;
        self.hidden_size * ds_ratio
    }
}

/// Llama4 VL model configuration.
#[derive(Debug, Clone)]
pub struct Llama4VLConfig {
    pub model_config: ModelConfig,
    pub vision_config: Llama4VisionConfig,
    pub image_token_id: u32,
}

impl Llama4VLConfig {
    /// Parse Llama4 VL config from a ModelConfig.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = Llama4VisionConfig::default();

        let vision_config = if let Some(vc) = cfg.extra.get("vision_config") {
            let hidden_size = vc
                .get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.hidden_size as u64) as usize;
            let intermediate_size =
                vc.get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.intermediate_size as u64) as usize;
            let num_attention_heads =
                vc.get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_attention_heads as u64) as usize;
            let num_hidden_layers =
                vc.get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_hidden_layers as u64) as usize;
            let image_size = vc
                .get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.image_size as u64) as usize;
            let patch_size = vc
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize;
            let num_channels = vc
                .get("num_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_channels as u64) as usize;
            let pixel_shuffle_ratio = vc
                .get("pixel_shuffle_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.pixel_shuffle_ratio);
            let projector_input_dim =
                vc.get("projector_input_dim")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.projector_input_dim as u64) as usize;
            let projector_output_dim =
                vc.get("projector_output_dim")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.projector_output_dim as u64) as usize;
            let multi_modal_projector_bias = vc
                .get("multi_modal_projector_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(defaults.multi_modal_projector_bias);
            let layer_norm_eps = vc
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps);

            Llama4VisionConfig {
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_hidden_layers,
                image_size,
                patch_size,
                num_channels,
                pixel_shuffle_ratio,
                projector_input_dim,
                projector_output_dim,
                multi_modal_projector_bias,
                layer_norm_eps,
            }
        } else {
            defaults
        };

        let image_token_id = cfg
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(200080) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_token_id,
        }
    }
}

// ─── Vision MLP ─────────────────────────────────────────────────────────────

/// GELU-activated MLP used in the vision encoder and pixel-shuffle adapter.
struct Llama4VisionMLP {
    fc1: Linear,
    fc2: Linear,
    output_activation: bool,
}

impl Llama4VisionMLP {
    fn new(
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
        bias: bool,
        output_activation: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = if bias {
            candle_nn::linear(input_size, intermediate_size, vb.pp("fc1"))?
        } else {
            candle_nn::linear_no_bias(input_size, intermediate_size, vb.pp("fc1"))?
        };
        let fc2 = if bias {
            candle_nn::linear(intermediate_size, output_size, vb.pp("fc2"))?
        } else {
            candle_nn::linear_no_bias(intermediate_size, output_size, vb.pp("fc2"))?
        };
        Ok(Self {
            fc1,
            fc2,
            output_activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.fc1.forward(xs)?;
        xs = xs.gelu_erf()?;
        xs = self.fc2.forward(&xs)?;
        if self.output_activation {
            xs = xs.gelu_erf()?;
        }
        Ok(xs)
    }
}

// ─── Vision Attention ───────────────────────────────────────────────────────

/// Multi-head self-attention for vision encoder (no KV cache, no causal mask).
struct Llama4VisionAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Llama4VisionAttention {
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let qkv_size = 3 * hidden_size;
        let qkv_proj = candle_nn::linear(hidden_size, qkv_size, vb.pp("qkv_proj"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let qkv = self.qkv_proj.forward(xs)?;

        // Split into Q, K, V
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

        // Reshape: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention (no causal mask for encoder)
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?
            .contiguous()?;

        self.o_proj.forward(&attn_output)
    }
}

// ─── Vision Encoder Layer ───────────────────────────────────────────────────

struct Llama4VisionEncoderLayer {
    self_attn: Llama4VisionAttention,
    mlp: Llama4VisionMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl Llama4VisionEncoderLayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn =
            Llama4VisionAttention::new(hidden_size, num_attention_heads, vb.pp("self_attn"))?;
        let mlp = Llama4VisionMLP::new(
            hidden_size,
            intermediate_size,
            hidden_size,
            true,  // bias
            false, // no output activation
            vb.pp("mlp"),
        )?;
        let input_layernorm = layer_norm(hidden_size, layer_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = layer_norm(
            hidden_size,
            layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

// ─── Unfold Patch Embedding ─────────────────────────────────────────────────

/// Unfold-based patch embedding: extracts patches via reshape then projects.
///
/// Equivalent to nn.Unfold(kernel_size=patch_size, stride=patch_size) + Linear.
/// For each patch of size (C, P, P), flatten to (C*P*P) and project to hidden_size.
struct Llama4UnfoldConvolution {
    linear: Linear,
    patch_size: usize,
    num_channels: usize,
}

impl Llama4UnfoldConvolution {
    fn new(
        patch_size: usize,
        num_channels: usize,
        hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = num_channels * patch_size * patch_size;
        let linear = candle_nn::linear_no_bias(input_dim, hidden_size, vb.pp("linear"))?;
        Ok(Self {
            linear,
            patch_size,
            num_channels,
        })
    }

    /// Forward: [batch, channels, height, width] -> [batch, num_patches, hidden_size]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (batch, _channels, height, width) = pixel_values.dims4()?;
        let ph = height / self.patch_size;
        let pw = width / self.patch_size;

        // Reshape to extract patches:
        // [B, C, H, W] -> [B, C, ph, P, pw, P]
        let xs = pixel_values.reshape((
            batch,
            self.num_channels,
            ph,
            self.patch_size,
            pw,
            self.patch_size,
        ))?;
        // -> [B, ph, pw, C, P, P]
        let xs = xs.permute([0, 2, 4, 1, 3, 5])?;
        // -> [B, ph*pw, C*P*P]
        let xs = xs.reshape((
            batch,
            ph * pw,
            self.num_channels * self.patch_size * self.patch_size,
        ))?;
        let xs = xs.contiguous()?;

        self.linear.forward(&xs)
    }
}

// ─── Pixel Shuffle ──────────────────────────────────────────────────────────

/// Pixel shuffle operation for token reduction.
///
/// Rearranges spatial tokens to reduce the sequence length while increasing
/// the channel dimension. With shuffle_ratio=0.5:
///   - seq_len -> seq_len / 4
///   - hidden -> hidden * 4
///
/// This matches the Python `pixel_shuffle()` function in mllama4.py.
fn pixel_shuffle(xs: &Tensor, shuffle_ratio: f64) -> Result<Tensor> {
    let (batch, seq_len, hidden) = xs.dims3()?;
    let patch_size = (seq_len as f64).sqrt() as usize;

    // [B, seq_len, H] -> [B, h, w, H]
    let xs = xs.reshape((batch, patch_size, patch_size, hidden))?;

    let new_w = (patch_size as f64 * shuffle_ratio) as usize;
    let new_c1 = (hidden as f64 / shuffle_ratio) as usize;

    // [B, h, w, H] -> [B, h, new_w, new_c1]
    let xs = xs.reshape((batch, patch_size, new_w, new_c1))?;
    // -> [B, new_w, h, new_c1]
    let xs = xs.permute([0, 2, 1, 3])?.contiguous()?;

    let new_h = (patch_size as f64 * shuffle_ratio) as usize;
    let new_c2 = (new_c1 as f64 / shuffle_ratio) as usize;

    // -> [B, new_w, new_h, new_c2]
    let xs = xs.reshape((batch, new_w, new_h, new_c2))?;
    // -> [B, new_h, new_w, new_c2]
    let xs = xs.permute([0, 2, 1, 3])?.contiguous()?;

    // -> [B, new_h * new_w, new_c2]
    xs.reshape((batch, new_h * new_w, new_c2))
}

// ─── Pixel Shuffle MLP ─────────────────────────────────────────────────────

/// Pixel-shuffle MLP adapter: pixel_shuffle -> MLP with output GELU.
struct Llama4VisionPixelShuffleMLP {
    mlp: Llama4VisionMLP,
    pixel_shuffle_ratio: f64,
}

impl Llama4VisionPixelShuffleMLP {
    fn new(cfg: &Llama4VisionConfig, vb: VarBuilder) -> Result<Self> {
        // After pixel shuffle, the hidden dim = intermediate_size (because
        // hidden_size * (1/ratio^2) = hidden_size * 4 for ratio=0.5, and
        // projector_input_dim is typically set to match this).
        // The MLP takes intermediate_size as input and produces projector_output_dim.
        let mlp = Llama4VisionMLP::new(
            cfg.intermediate_size,
            cfg.projector_input_dim,
            cfg.projector_output_dim,
            cfg.multi_modal_projector_bias,
            true, // output_activation=True
            vb.pp("mlp"),
        )?;
        Ok(Self {
            mlp,
            pixel_shuffle_ratio: cfg.pixel_shuffle_ratio,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = pixel_shuffle(xs, self.pixel_shuffle_ratio)?;
        self.mlp.forward(&xs)
    }
}

// ─── Vision Model ───────────────────────────────────────────────────────────

/// Llama4 vision model: patch embedding + ViT encoder + pixel-shuffle adapter.
struct Llama4VisionModel {
    patch_embedding: Llama4UnfoldConvolution,
    class_embedding: Tensor,
    positional_embedding_vlm: Tensor,
    layernorm_pre: LayerNorm,
    layernorm_post: LayerNorm,
    encoder_layers: Vec<Llama4VisionEncoderLayer>,
    vision_adapter: Llama4VisionPixelShuffleMLP,
}

impl Llama4VisionModel {
    fn new(cfg: &Llama4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = Llama4UnfoldConvolution::new(
            cfg.patch_size,
            cfg.num_channels,
            cfg.hidden_size,
            vb.pp("patch_embedding"),
        )?;

        let class_embedding = vb.get(cfg.hidden_size, "class_embedding")?;
        let positional_embedding_vlm =
            vb.get((cfg.seq_len(), cfg.hidden_size), "positional_embedding_vlm")?;

        let layernorm_pre =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm_pre"))?;
        let layernorm_post =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm_post"))?;

        let vb_enc = vb.pp("model").pp("layers");
        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            encoder_layers.push(Llama4VisionEncoderLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
                cfg.layer_norm_eps,
                vb_enc.pp(i),
            )?);
        }

        let vision_adapter = Llama4VisionPixelShuffleMLP::new(cfg, vb.pp("vision_adapter"))?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            positional_embedding_vlm,
            layernorm_pre,
            layernorm_post,
            encoder_layers,
            vision_adapter,
        })
    }

    /// Forward: [batch, channels, height, width] -> [batch, num_output_tokens, vision_output_dim]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Patch embedding
        let mut hidden_state = self.patch_embedding.forward(pixel_values)?;
        let (num_tiles, _num_patches, hidden_dim) = hidden_state.dims3()?;

        // Append CLS token
        let cls = self
            .class_embedding
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((num_tiles, 1, hidden_dim))?;
        hidden_state = Tensor::cat(&[&hidden_state, &cls], 1)?;
        let _num_patches = hidden_state.dim(1)?;

        // Add positional embeddings (broadcast over batch/tile dimension)
        let pos_emb = self
            .positional_embedding_vlm
            .unsqueeze(0)?
            .to_dtype(hidden_state.dtype())?;
        hidden_state = hidden_state.broadcast_add(&pos_emb)?;
        hidden_state = self.layernorm_pre.forward(&hidden_state)?;

        // Encoder layers
        for layer in &self.encoder_layers {
            hidden_state = layer.forward(&hidden_state)?;
        }
        hidden_state = self.layernorm_post.forward(&hidden_state)?;

        // Remove CLS token (last position)
        let seq_len = hidden_state.dim(1)?;
        hidden_state = hidden_state.narrow(1, 0, seq_len - 1)?;

        // Pixel shuffle + MLP adapter
        self.vision_adapter.forward(&hidden_state)
    }
}

// ─── Multimodal Projector ───────────────────────────────────────────────────

/// Linear projector from vision output dim to text hidden size.
struct Llama4MultiModalProjector {
    linear_1: Linear,
}

impl Llama4MultiModalProjector {
    fn new(vision_output_dim: usize, text_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear_1 =
            candle_nn::linear_no_bias(vision_output_dim, text_hidden_size, vb.pp("linear_1"))?;
        Ok(Self { linear_1 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_1.forward(xs)
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Llama4 Vision-Language model for conditional generation.
///
/// Combines Llama4 vision encoder (ViT + pixel-shuffle) + linear projector
/// + Llama4 MoE language model.
pub struct Llama4VLForConditionalGeneration {
    vision_model: Llama4VisionModel,
    multi_modal_projector: Llama4MultiModalProjector,
    language_model: Llama4ForCausalLM,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl Llama4VLForConditionalGeneration {
    /// Create a new Llama4 VL model.
    pub fn new(cfg: &Llama4VLConfig, vb: VarBuilder) -> Result<Self> {
        let vision_model = Llama4VisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?;

        let multi_modal_projector = Llama4MultiModalProjector::new(
            cfg.vision_config.projector_output_dim,
            cfg.model_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;

        let language_model = Llama4ForCausalLM::new(&cfg.model_config, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            multi_modal_projector,
            language_model,
            image_token_id: cfg.image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Create from a generic ModelConfig.
    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vlm_cfg = Llama4VLConfig::from_model_config(cfg);
        Self::new(&vlm_cfg, vb)
    }

    /// Encode images through the full vision pipeline:
    /// vision encoder -> pixel-shuffle MLP -> linear projector.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_features = self.vision_model.forward(pixel_values)?;
        self.multi_modal_projector.forward(&vision_features)
    }

    /// Merge text embeddings with pre-processed multimodal image embeddings.
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
            // Project vision embeddings through multi-modal projector
            let vision_emb = processed.embedding.unsqueeze(0)?;
            let projected = self.multi_modal_projector.forward(&vision_emb)?;
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

impl crate::engine::ModelForward for Llama4VLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Text-only forward: delegate to Llama4
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

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        // Llama4 text model needs MoE config
        extra.insert("num_local_experts".into(), serde_json::Value::from(4));
        extra.insert("num_experts_per_tok".into(), serde_json::Value::from(1));
        extra.insert(
            "interleave_moe_layer_step".into(),
            serde_json::Value::from(2),
        );

        ModelConfig {
            architectures: vec!["Llama4VLForConditionalGeneration".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 4,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_vision_config() -> Llama4VisionConfig {
        // intermediate_size must equal hidden_size / pixel_shuffle_ratio^2 = 32 * 4 = 128
        // because pixel shuffle output dim IS intermediate_size (as in the real model).
        Llama4VisionConfig {
            hidden_size: 32,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            num_channels: 3,
            pixel_shuffle_ratio: 0.5,
            projector_input_dim: 256,
            projector_output_dim: 64,
            multi_modal_projector_bias: false,
            layer_norm_eps: 1e-5,
        }
    }

    fn test_vlm_config() -> Llama4VLConfig {
        Llama4VLConfig {
            model_config: test_model_config(),
            vision_config: test_vision_config(),
            image_token_id: 200080,
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

    // ── Config Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let defaults = Llama4VisionConfig::default();
        assert_eq!(defaults.hidden_size, 1280);
        assert_eq!(defaults.image_size, 560);
        assert_eq!(defaults.patch_size, 14);
        assert_eq!(defaults.pixel_shuffle_ratio, 0.5);
        assert_eq!(defaults.num_patches(), 1600); // (560/14)^2
        assert_eq!(defaults.seq_len(), 1601); // + CLS
        assert_eq!(defaults.num_output_tokens(), 400); // 1600 / 4
    }

    #[test]
    fn test_config_from_model_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "pixel_shuffle_ratio": 0.5,
                "projector_input_dim": 256,
                "projector_output_dim": 64
            }),
        );
        model_cfg
            .extra
            .insert("image_token_index".to_string(), serde_json::json!(200080));

        let cfg = Llama4VLConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.hidden_size, 32);
        assert_eq!(cfg.vision_config.image_size, 28);
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.vision_config.pixel_shuffle_ratio, 0.5);
        assert_eq!(cfg.image_token_id, 200080);
    }

    #[test]
    fn test_config_computed_values() {
        let cfg = test_vision_config();
        // 28/14 = 2 patches per side
        assert_eq!(cfg.patches_per_side(), 2);
        // 2*2 = 4 patches
        assert_eq!(cfg.num_patches(), 4);
        // 4 + 1 CLS = 5
        assert_eq!(cfg.seq_len(), 5);
        // With ratio=0.5: ds_ratio=4, 4/4=1 output token
        assert_eq!(cfg.num_output_tokens(), 1);
        // shuffled_hidden_dim = 32 * 4 = 128
        assert_eq!(cfg.shuffled_hidden_dim(), 128);
    }

    // ── Pixel Shuffle Tests ──────────────────────────────────────────────

    #[test]
    fn test_pixel_shuffle_shape() {
        let device = Device::Cpu;
        // 16 patches (4x4 grid), hidden=8
        let input = Tensor::zeros((1, 16, 8), DType::F32, &device).unwrap();
        let output = pixel_shuffle(&input, 0.5).unwrap();
        // With ratio=0.5: seq_len -> 16 * 0.5^2 = 4, hidden -> 8 / 0.5^2 = 32
        assert_eq!(output.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_pixel_shuffle_shape_4x4() {
        let device = Device::Cpu;
        let input = Tensor::zeros((2, 16, 32), DType::F32, &device).unwrap();
        let output = pixel_shuffle(&input, 0.5).unwrap();
        // seq_len: 16 -> 4, hidden: 32 -> 128
        assert_eq!(output.dims(), &[2, 4, 128]);
    }

    // ── Model Construction Tests ─────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Llama4VLForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Llama4 VL should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "pixel_shuffle_ratio": 0.5,
                "projector_input_dim": 256,
                "projector_output_dim": 64
            }),
        );

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(
            model.is_ok(),
            "from_model_config should work: {:?}",
            model.err()
        );
    }

    // ── Forward Tests ────────────────────────────────────────────────────

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
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
    fn test_multimodal_forward_no_images() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward_multimodal(
                &input_ids,
                None,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_multimodal_forward_with_images() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1, 2, 3], 0);

        // Vision encoder output: after pixel shuffle, 1 token with projector_output_dim=64
        // We need to provide pre-projected embeddings (vision_output_dim)
        let num_vision_tokens = cfg.vision_config.num_output_tokens(); // 1
        let seq_len = num_vision_tokens + 4; // 1 image token + 4 text tokens
        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();

        // Embedding with projector_output_dim (before multi_modal_projector maps to text hidden)
        let img_embedding = Tensor::randn(
            0f32,
            1.0,
            (num_vision_tokens, cfg.vision_config.projector_output_dim),
            &device,
        )
        .unwrap();
        let processed = ProcessedImage::new(img_embedding, num_vision_tokens);
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

        assert_eq!(logits.dims(), &[1, seq_len, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
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
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.model_config.vocab_size]);
        block_table.advance(3);

        // Decode step
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_vision_encoder_forward_shape() {
        let device = Device::Cpu;
        let cfg = test_vlm_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Llama4VLForConditionalGeneration::new(&cfg, vb).unwrap();

        // 28x28 image -> 2x2=4 patches -> after pixel shuffle (0.5): 1 token
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let encoded = model.encode_images(&pixel_values).unwrap();

        // Output: [1, 1, 64] (1 token, projected to text_hidden=64)
        assert_eq!(
            encoded.dims(),
            &[
                1,
                cfg.vision_config.num_output_tokens(),
                cfg.model_config.hidden_size
            ]
        );
    }

    #[test]
    fn test_unfold_patch_embedding_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // 14x14 patches on a 28x28 image -> 2x2=4 patches
        let unfold = Llama4UnfoldConvolution::new(14, 3, 32, vb).unwrap();
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let patches = unfold.forward(&pixel_values).unwrap();

        assert_eq!(patches.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_vision_mlp_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mlp = Llama4VisionMLP::new(32, 64, 32, true, false, vb).unwrap();
        let input = Tensor::randn(0f32, 1.0, (1, 4, 32), &device).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 4, 32]);
    }

    #[test]
    fn test_vision_attention_forward() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let attn = Llama4VisionAttention::new(32, 4, vb).unwrap();
        let input = Tensor::randn(0f32, 1.0, (1, 5, 32), &device).unwrap();
        let output = attn.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 5, 32]);
    }
}
