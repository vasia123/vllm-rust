//! GLM-4V vision-language model implementation.
//!
//! GLM-4V uses an EVA2-CLIP vision encoder with a GLU projector and Conv2d
//! downsampler to feed image features into a GLM4/ChatGLM language model.
//!
//! Key architectural features:
//! - EVA2-CLIP: Conv2d patch embedding + CLS token + position embedding
//! - Transformer layers with LayerNorm + standard attention + GELU MLP
//! - Post-norm residual: LayerNorm(attention(input)) + residual (not pre-norm!)
//! - EVA2CLIPGLU projector: linear_proj -> LayerNorm -> GELU -> SwiGLU gate -> dense
//! - Conv2d downsampler: stride-2 conv to reduce spatial resolution by 2x
//! - boi/eoi: begin/end of image tokens (learned parameters)
//!
//! Reference: vLLM's glm4v.py implementation.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, conv2d_no_bias, layer_norm, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::glm4::Glm4ForCausalLM;

// ─── Config ─────────────────────────────────────────────────────────────────

/// EVA2-CLIP vision encoder configuration.
#[derive(Debug, Clone)]
pub struct EVA2CLIPVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_hidden_layers: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub layer_norm_eps: f64,
    pub dropout_prob: f64,
    pub scaling_factor: f64,
}

impl Default for EVA2CLIPVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1792,
            intermediate_size: 15360,
            num_heads: 16,
            num_hidden_layers: 63,
            image_size: 1344,
            patch_size: 14,
            in_channels: 3,
            layer_norm_eps: 1e-6,
            dropout_prob: 0.0,
            scaling_factor: 1.0,
        }
    }
}

impl EVA2CLIPVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Number of patches in the grid (without CLS token).
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Total number of positions including CLS token.
    pub fn num_positions(&self) -> usize {
        self.num_patches() + 1
    }

    /// Grid size (square root of num_patches).
    pub fn grid_size(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Number of image tokens after conv2d downsampling (stride 2 reduces by 2x each dim).
    pub fn num_image_tokens(&self) -> usize {
        let g = self.grid_size() / 2;
        g * g
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
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
            num_heads: json
                .get("num_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_heads as u64) as usize,
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
            in_channels: json
                .get("in_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.in_channels as u64) as usize,
            layer_norm_eps: json
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.layer_norm_eps),
            dropout_prob: json
                .get("dropout_prob")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.dropout_prob),
            scaling_factor: json
                .get("scaling_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.scaling_factor),
        }
    }
}

// ─── EVA2-CLIP Patch Embedding ──────────────────────────────────────────────

/// EVA2-CLIP patch embedding: Conv2d + CLS token + position embedding.
#[allow(dead_code)]
struct EVA2CLIPPatchEmbedding {
    proj: candle_nn::Conv2d,
    cls_embedding: Tensor,
    position_embedding: Tensor,
    num_patches: usize,
}

#[allow(dead_code)]
impl EVA2CLIPPatchEmbedding {
    fn new(cfg: &EVA2CLIPVisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches = cfg.num_patches();
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let proj = conv2d(
            cfg.in_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("proj"),
        )?;

        // CLS token: [1, hidden_size]
        let cls_embedding = vb.get((1, cfg.hidden_size), "cls_embedding")?;

        // Position embedding as nn.Embedding: [num_positions, hidden_size]
        let position_embedding = vb.get(
            (cfg.num_positions(), cfg.hidden_size),
            "position_embedding.weight",
        )?;

        Ok(Self {
            proj,
            cls_embedding,
            position_embedding,
            num_patches,
        })
    }

    /// Forward: pixel_values [B, C, H, W] -> [B, num_patches+1, hidden_size]
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;

        // Conv2d: [B, C, H, W] -> [B, hidden_size, H/patch, W/patch]
        let x = self.proj.forward(pixel_values)?;

        // Flatten spatial dims: [B, hidden_size, num_patches] -> [B, num_patches, hidden_size]
        let (b, c, _h, _w) = x.dims4()?;
        let x = x.reshape((b, c, self.num_patches))?.transpose(1, 2)?;

        // Prepend CLS token: expand [1, hidden_size] -> [B, 1, hidden_size]
        let cls = self
            .cls_embedding
            .unsqueeze(0)?
            .broadcast_as((batch_size, 1, c))?
            .contiguous()?;
        let x = Tensor::cat(&[cls, x], 1)?;

        // Add position embeddings: [num_positions, hidden_size] -> broadcast to [B, num_positions, hidden_size]
        let pos = self.position_embedding.unsqueeze(0)?;
        x.broadcast_add(&pos)
    }
}

// ─── EVA2-CLIP Attention ────────────────────────────────────────────────────

/// EVA2-CLIP attention: fused QKV + bidirectional attention.
#[allow(dead_code)]
struct EVA2CLIPAttention {
    query_key_value: Linear,
    dense: Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl EVA2CLIPAttention {
    fn new(cfg: &EVA2CLIPVisionConfig, vb: VarBuilder) -> Result<Self> {
        let query_key_value = candle_nn::linear(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            vb.pp("query_key_value"),
        )?;
        let dense = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;

        Ok(Self {
            query_key_value,
            dense,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Bidirectional self-attention (no causal mask).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let hidden = self.num_heads * self.head_dim;

        let qkv = self.query_key_value.forward(x)?;
        let q = qkv.narrow(2, 0, hidden)?;
        let k = qkv.narrow(2, hidden, hidden)?;
        let v = qkv.narrow(2, 2 * hidden, hidden)?;

        // [B, seq, heads, head_dim] -> [B, heads, seq, head_dim]
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention (bidirectional)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, heads, seq, head_dim] -> [B, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, hidden))?;

        self.dense.forward(&attn_output)
    }
}

// ─── EVA2-CLIP MLP ──────────────────────────────────────────────────────────

/// EVA2-CLIP MLP: fc1 + GELU + fc2.
#[allow(dead_code)]
struct EVA2CLIPMLP {
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl EVA2CLIPMLP {
    fn new(cfg: &EVA2CLIPVisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── EVA2-CLIP Transformer Layer ────────────────────────────────────────────

/// EVA2-CLIP transformer layer with post-norm residual.
///
/// Forward:
/// ```text
/// attention_output = LayerNorm(attention(input))
/// hidden = input + attention_output
/// mlp_output = LayerNorm(mlp(hidden))
/// output = hidden + mlp_output
/// ```
///
/// This is post-norm: the norm is applied AFTER attention/MLP, not before.
#[allow(dead_code)]
struct EVA2CLIPTransformerLayer {
    input_layernorm: LayerNorm,
    attention: EVA2CLIPAttention,
    mlp: EVA2CLIPMLP,
    post_attention_layernorm: LayerNorm,
}

#[allow(dead_code)]
impl EVA2CLIPTransformerLayer {
    fn new(cfg: &EVA2CLIPVisionConfig, vb: VarBuilder) -> Result<Self> {
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let attention = EVA2CLIPAttention::new(cfg, vb.pp("attention"))?;
        let mlp = EVA2CLIPMLP::new(cfg, vb.pp("mlp"))?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            input_layernorm,
            attention,
            mlp,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Post-norm: norm AFTER attention, then add residual
        let attention_input = hidden_states;
        let attention_output = self
            .input_layernorm
            .forward(&self.attention.forward(attention_input)?)?;
        let hidden_states = (attention_input + attention_output)?;

        // Post-norm: norm AFTER MLP, then add residual
        let mlp_input = &hidden_states;
        let mlp_output = self
            .post_attention_layernorm
            .forward(&self.mlp.forward(mlp_input)?)?;
        mlp_input + mlp_output
    }
}

// ─── EVA2-CLIP GLU Projector ────────────────────────────────────────────────

/// GLU projector: linear_proj -> LayerNorm -> GELU -> SwiGLU gate -> dense.
///
/// ```text
/// x = linear_proj(x)                    # vision_hidden -> llm_hidden
/// x = gelu(layer_norm(x))               # LayerNorm + GELU
/// x = merged_proj(x)                    # llm_hidden -> [ffn_hidden, ffn_hidden]
/// x = silu(gate) * up                   # SiLU(first half) * second half
/// x = dense_4h_to_h(x)                  # ffn_hidden -> llm_hidden
/// ```
#[allow(dead_code)]
struct EVA2CLIPGLU {
    linear_proj: Linear,
    norm1: LayerNorm,
    merged_proj: Linear,
    dense_4h_to_h: Linear,
    ffn_hidden_size: usize,
}

#[allow(dead_code)]
impl EVA2CLIPGLU {
    fn new(
        vision_hidden_size: usize,
        llm_hidden_size: usize,
        ffn_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear_proj =
            candle_nn::linear_no_bias(vision_hidden_size, llm_hidden_size, vb.pp("linear_proj"))?;
        let norm1 = layer_norm(llm_hidden_size, 1e-5, vb.pp("norm1"))?;

        // merged_proj: llm_hidden -> 2 * ffn_hidden (gate + up, concatenated)
        let merged_proj =
            candle_nn::linear_no_bias(llm_hidden_size, 2 * ffn_hidden_size, vb.pp("merged_proj"))?;

        let dense_4h_to_h =
            candle_nn::linear_no_bias(ffn_hidden_size, llm_hidden_size, vb.pp("dense_4h_to_h"))?;

        Ok(Self {
            linear_proj,
            norm1,
            merged_proj,
            dense_4h_to_h,
            ffn_hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_proj.forward(x)?;
        let x = self.norm1.forward(&x)?.gelu_erf()?;

        // SwiGLU: split into gate and up, apply SiLU to gate, multiply
        let x = self.merged_proj.forward(&x)?;
        let gate = x.narrow(2, 0, self.ffn_hidden_size)?;
        let up = x.narrow(2, self.ffn_hidden_size, self.ffn_hidden_size)?;
        let x = (candle_nn::Activation::Silu.forward(&gate)? * up)?;

        self.dense_4h_to_h.forward(&x)
    }
}

// ─── Full EVA2-CLIP Model ───────────────────────────────────────────────────

/// Full EVA2-CLIP model with conv downsampler, GLU projector, and boi/eoi tokens.
#[allow(dead_code)]
struct EVA2CLIPModel {
    patch_embedding: EVA2CLIPPatchEmbedding,
    layers: Vec<EVA2CLIPTransformerLayer>,
    conv: candle_nn::Conv2d,
    linear_proj: EVA2CLIPGLU,
    boi: Tensor,
    eoi: Tensor,
    scaling_factor: f64,
    grid_size: usize,
}

#[allow(dead_code)]
impl EVA2CLIPModel {
    fn new(
        vision_cfg: &EVA2CLIPVisionConfig,
        llm_hidden_size: usize,
        ffn_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embedding = EVA2CLIPPatchEmbedding::new(vision_cfg, vb.pp("patch_embedding"))?;

        let mut layers = Vec::with_capacity(vision_cfg.num_hidden_layers);
        let vb_t = vb.pp("transformer").pp("layers");
        for i in 0..vision_cfg.num_hidden_layers {
            layers.push(EVA2CLIPTransformerLayer::new(vision_cfg, vb_t.pp(i))?);
        }

        // Conv2d downsampler: vision_hidden -> llm_hidden, kernel=2, stride=2
        let conv_cfg = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d_no_bias(
            vision_cfg.hidden_size,
            llm_hidden_size,
            2,
            conv_cfg,
            vb.pp("conv"),
        )?;

        let linear_proj = EVA2CLIPGLU::new(
            llm_hidden_size,
            llm_hidden_size,
            ffn_hidden_size,
            vb.pp("linear_proj"),
        )?;

        // boi/eoi: [1, 1, llm_hidden_size]
        let boi = vb.get((1, 1, llm_hidden_size), "boi")?;
        let eoi = vb.get((1, 1, llm_hidden_size), "eoi")?;

        Ok(Self {
            patch_embedding,
            layers,
            conv,
            linear_proj,
            boi,
            eoi,
            scaling_factor: vision_cfg.scaling_factor,
            grid_size: vision_cfg.grid_size(),
        })
    }

    /// Forward: pixel_values [B, C, H, W] -> [B, num_image_tokens+2, llm_hidden]
    ///
    /// The +2 accounts for boi and eoi tokens.
    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let batch_size = images.dim(0)?;

        // Patch embedding: [B, C, H, W] -> [B, num_patches+1, vision_hidden]
        let mut x = self.patch_embedding.forward(images)?;

        // Run through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Remove CLS token: [B, num_patches+1, D] -> [B, num_patches, D]
        x = x.narrow(1, 1, x.dim(1)? - 1)?;

        // Reshape to spatial grid: [B, grid, grid, D] -> [B, D, grid, grid]
        let (b, _s, h) = x.dims3()?;
        let x = x
            .reshape((b, self.grid_size, self.grid_size, h))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;

        // Conv2d downsample: [B, D_in, grid, grid] -> [B, D_out, grid/2, grid/2]
        let x = self.conv.forward(&x)?;

        // Flatten spatial: [B, D_out, h, w] -> [B, h*w, D_out]
        let (b, d, h, w) = x.dims4()?;
        let x = x.reshape((b, d, h * w))?.transpose(1, 2)?.contiguous()?;

        // GLU projector
        let x = self.linear_proj.forward(&x)?;

        // Add boi/eoi tokens
        let boi = self
            .boi
            .broadcast_as((batch_size, 1, x.dim(2)?))?
            .contiguous()?;
        let eoi = self
            .eoi
            .broadcast_as((batch_size, 1, x.dim(2)?))?
            .contiguous()?;
        let x = Tensor::cat(&[boi, x, eoi], 1)?;

        // Scale
        if (self.scaling_factor - 1.0).abs() > f64::EPSILON {
            x / self.scaling_factor
        } else {
            Ok(x)
        }
    }
}

// ─── Top-level GLM-4V Model ─────────────────────────────────────────────────

/// GLM-4V vision-language model.
///
/// Combines EVA2-CLIP vision encoder with GLM4 language model backbone.
pub struct Glm4VForConditionalGeneration {
    #[allow(dead_code)]
    vision_model: EVA2CLIPModel,
    language_model: Glm4ForCausalLM,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl Glm4VForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vision_cfg = cfg
            .extra
            .get("vision_config")
            .map(EVA2CLIPVisionConfig::from_json)
            .unwrap_or_default();

        let ffn_hidden_size = cfg
            .extra
            .get("ffn_hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.intermediate_size as u64) as usize;

        let image_token_id = cfg
            .extra
            .get("pad_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151329) as u32;

        let vision_model = EVA2CLIPModel::new(
            &vision_cfg,
            cfg.hidden_size,
            ffn_hidden_size,
            vb.pp("transformer").pp("vision"),
        )?;

        let language_model = Glm4ForCausalLM::new(cfg, vb.clone())?;

        Ok(Self {
            vision_model,
            language_model,
            image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Merge pre-encoded image embeddings with text embeddings.
    ///
    /// Replaces image placeholder tokens with vision embeddings at the
    /// corresponding positions.
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

impl crate::engine::ModelForward for Glm4VForConditionalGeneration {
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

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["GLM4VForCausalLM".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> EVA2CLIPVisionConfig {
        EVA2CLIPVisionConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            in_channels: 3,
            layer_norm_eps: 1e-6,
            dropout_prob: 0.0,
            scaling_factor: 1.0,
        }
    }

    fn test_model_config_with_vision() -> ModelConfig {
        let mut cfg = test_model_config();
        cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "num_hidden_layers": 2,
                "image_size": 28,
                "patch_size": 14,
                "in_channels": 3,
                "layer_norm_eps": 1e-6,
                "dropout_prob": 0.0,
                "scaling_factor": 1.0
            }),
        );
        cfg.extra
            .insert("ffn_hidden_size".to_string(), serde_json::json!(128));
        cfg.extra
            .insert("pad_token_id".to_string(), serde_json::json!(151329));
        cfg
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

    // ── Config Tests ────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = EVA2CLIPVisionConfig::default();
        assert_eq!(cfg.hidden_size, 1792);
        assert_eq!(cfg.num_hidden_layers, 63);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim(), 112);
        assert_eq!(cfg.image_size, 1344);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches(), 9216); // (1344/14)^2 = 96^2
        assert_eq!(cfg.num_positions(), 9217);
        assert_eq!(cfg.grid_size(), 96);
        assert_eq!(cfg.num_image_tokens(), 2304); // (96/2)^2 = 48^2
    }

    #[test]
    fn test_vision_config_from_json() {
        let json = serde_json::json!({
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "num_hidden_layers": 2,
            "image_size": 28,
            "patch_size": 14,
            "in_channels": 3,
            "layer_norm_eps": 1e-6,
            "scaling_factor": 2.0
        });
        let cfg = EVA2CLIPVisionConfig::from_json(&json);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.image_size, 28);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches(), 4); // (28/14)^2
        assert_eq!(cfg.num_positions(), 5);
        assert_eq!(cfg.grid_size(), 2);
        assert_eq!(cfg.num_image_tokens(), 1); // (2/2)^2 = 1
        assert!((cfg.scaling_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vision_config_small() {
        let cfg = test_vision_config();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.num_patches(), 4); // (28/14)^2
        assert_eq!(cfg.num_positions(), 5); // 4 + 1 CLS
        assert_eq!(cfg.grid_size(), 2);
        assert_eq!(cfg.num_image_tokens(), 1); // (2/2)^2
    }

    // ── Patch Embedding Tests ───────────────────────────────────────────

    #[test]
    fn test_patch_embedding_shapes() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = EVA2CLIPPatchEmbedding::new(&cfg, vb).unwrap();

        // [1, 3, 28, 28] -> [1, 5, 64] (4 patches + 1 CLS)
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = emb.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_patch_embedding_batch() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = EVA2CLIPPatchEmbedding::new(&cfg, vb).unwrap();

        // Batch of 2 images
        let pixel_values = Tensor::zeros((2, 3, 28, 28), DType::F32, &device).unwrap();
        let output = emb.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[2, 5, 64]);
    }

    // ── Attention Tests ─────────────────────────────────────────────────

    #[test]
    fn test_attention_shapes() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = EVA2CLIPAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_attention_batch() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = EVA2CLIPAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (2, 5, 64), &device).unwrap();
        let output = attn.forward(&x).unwrap();
        assert_eq!(output.dims(), &[2, 5, 64]);
    }

    // ── MLP Tests ───────────────────────────────────────────────────────

    #[test]
    fn test_mlp_shapes() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = EVA2CLIPMLP::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    // ── Transformer Layer Tests ─────────────────────────────────────────

    #[test]
    fn test_transformer_layer_post_norm() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let layer = EVA2CLIPTransformerLayer::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let output = layer.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    // ── GLU Projector Tests ─────────────────────────────────────────────

    #[test]
    fn test_glu_projector_shapes() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // vision_hidden=64, llm_hidden=64, ffn_hidden=128
        let glu = EVA2CLIPGLU::new(64, 64, 128, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 64), &device).unwrap();
        let output = glu.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_glu_projector_different_dims() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        // Different vision and LLM hidden sizes
        let glu = EVA2CLIPGLU::new(32, 64, 128, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 32), &device).unwrap();
        let output = glu.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    // ── Conv2d Downsampling Tests ───────────────────────────────────────

    #[test]
    fn test_conv2d_downsampling() {
        let device = Device::Cpu;
        let conv_cfg = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d_no_bias(
            64,
            64,
            2,
            conv_cfg,
            VarBuilder::zeros(DType::F32, &device).pp("conv"),
        )
        .unwrap();

        // [1, 64, 4, 4] -> [1, 64, 2, 2] (stride 2 halves each spatial dim)
        let x = Tensor::zeros((1, 64, 4, 4), DType::F32, &device).unwrap();
        let output = conv.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 64, 2, 2]);
    }

    #[test]
    fn test_conv2d_downsampling_2x2() {
        let device = Device::Cpu;
        let conv_cfg = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d_no_bias(
            64,
            64,
            2,
            conv_cfg,
            VarBuilder::zeros(DType::F32, &device).pp("conv"),
        )
        .unwrap();

        // [1, 64, 2, 2] -> [1, 64, 1, 1]
        let x = Tensor::zeros((1, 64, 2, 2), DType::F32, &device).unwrap();
        let output = conv.forward(&x).unwrap();
        assert_eq!(output.dims(), &[1, 64, 1, 1]);
    }

    // ── Full Vision Model Tests ─────────────────────────────────────────

    #[test]
    fn test_full_vision_model_forward() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let llm_hidden = 64;
        let ffn_hidden = 128;
        let model = EVA2CLIPModel::new(&cfg, llm_hidden, ffn_hidden, vb).unwrap();

        // image_size=28, patch_size=14 -> grid=2
        // Conv2d stride=2 -> grid=1 -> 1 spatial token
        // + boi + eoi = 3 tokens total
        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 3, 64]); // 1 patch + boi + eoi = 3 tokens
    }

    #[test]
    fn test_full_vision_model_batch() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = EVA2CLIPModel::new(&cfg, 64, 128, vb).unwrap();

        let pixel_values = Tensor::zeros((2, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_vision_model_with_scaling() {
        let device = Device::Cpu;
        let mut cfg = test_vision_config();
        cfg.scaling_factor = 2.0;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = EVA2CLIPModel::new(&cfg, 64, 128, vb).unwrap();

        let pixel_values = Tensor::zeros((1, 3, 28, 28), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 3, 64]);
    }

    // ── Top-level Model Tests ───────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config_with_vision();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = Glm4VForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "Model should construct: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_model_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config_with_vision();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_model_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config_with_vision();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
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
    fn test_multimodal_support_flag() {
        let device = Device::Cpu;
        let cfg = test_model_config_with_vision();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

        assert!(model.supports_multimodal());
    }

    #[test]
    fn test_model_multimodal_forward_no_images() {
        let device = Device::Cpu;
        let cfg = test_model_config_with_vision();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();

        // forward_multimodal with None should behave like text-only
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

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_image_token_id_default() {
        let cfg = test_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

        // Default pad_token_id when not in config
        assert_eq!(model.image_token_id, 151329);
    }

    #[test]
    fn test_image_token_id_from_config() {
        let mut cfg = test_model_config_with_vision();
        cfg.extra
            .insert("pad_token_id".to_string(), serde_json::json!(12345));
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Glm4VForConditionalGeneration::new(&cfg, vb).unwrap();

        assert_eq!(model.image_token_id, 12345);
    }

    // ── Additional Component Tests ──────────────────────────────────────

    #[test]
    fn test_vision_config_num_image_tokens_larger() {
        // Larger grid: image_size=56, patch_size=14 -> grid=4, conv stride 2 -> 2x2 = 4 tokens
        let cfg = EVA2CLIPVisionConfig {
            image_size: 56,
            patch_size: 14,
            ..test_vision_config()
        };
        assert_eq!(cfg.grid_size(), 4);
        assert_eq!(cfg.num_image_tokens(), 4); // (4/2)^2 = 4
        assert_eq!(cfg.num_patches(), 16); // 4^2
        assert_eq!(cfg.num_positions(), 17); // 16 + 1 CLS
    }

    #[test]
    fn test_full_vision_model_larger_grid() {
        let device = Device::Cpu;
        let cfg = EVA2CLIPVisionConfig {
            image_size: 56,
            patch_size: 14,
            ..test_vision_config()
        };
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = EVA2CLIPModel::new(&cfg, 64, 128, vb).unwrap();

        // grid=4, conv stride 2 -> 2x2 = 4 spatial tokens + boi + eoi = 6
        let pixel_values = Tensor::zeros((1, 3, 56, 56), DType::F32, &device).unwrap();
        let output = model.forward(&pixel_values).unwrap();
        assert_eq!(output.dims(), &[1, 6, 64]); // 4 patches + boi + eoi
    }
}
