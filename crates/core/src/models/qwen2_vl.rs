//! Qwen2-VL vision-language model implementation.
//!
//! Qwen2-VL combines a custom vision transformer (Qwen2VisionTransformer)
//! with the Qwen2 language model backbone, using a PatchMerger as the projector.
//!
//! Key architectural features:
//! - Conv3D-as-Linear patch embedding (temporal dimension handling)
//! - 2D RoPE for vision attention (partial rotary factor 0.5)
//! - MRoPE (Multi-dimensional RoPE) for language model (3D: temporal/height/width)
//! - PatchMerger reduces vision tokens by spatial_merge_size^2 (default 4x)
//!
//! Reference: Qwen2-VL (https://arxiv.org/abs/2409.12191)

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{causal_mask, paged_attention, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Qwen2-VL vision encoder configuration.
#[derive(Debug, Clone)]
pub struct Qwen2VLVisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
}

impl Default for Qwen2VLVisionConfig {
    fn default() -> Self {
        Self {
            depth: 32,
            embed_dim: 1280,
            num_heads: 16,
            mlp_ratio: 4.0,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
        }
    }
}

impl Qwen2VLVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn mlp_hidden_size(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio) as usize
    }

    /// Patch embedding input dimension: C * temporal_patch_size * patch_size * patch_size
    pub fn patch_input_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// Merger hidden dimension: embed_dim * spatial_merge_size^2
    pub fn merger_hidden_dim(&self) -> usize {
        self.embed_dim * self.spatial_merge_size * self.spatial_merge_size
    }

    /// Build from the outer `ModelConfig`, reading `extra["vision_config"]`.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        cfg.extra
            .get("vision_config")
            .map(Self::from_json)
            .unwrap_or_default()
    }

    fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();
        Self {
            depth: json
                .get("depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.depth as u64) as usize,
            embed_dim: json
                .get("embed_dim")
                .or_else(|| json.get("hidden_size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.embed_dim as u64) as usize,
            num_heads: json
                .get("num_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_heads as u64) as usize,
            mlp_ratio: json
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(defaults.mlp_ratio),
            in_channels: json
                .get("in_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.in_channels as u64) as usize,
            patch_size: json
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.patch_size as u64) as usize,
            temporal_patch_size: json
                .get("temporal_patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.temporal_patch_size as u64)
                as usize,
            spatial_merge_size: json
                .get("spatial_merge_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.spatial_merge_size as u64)
                as usize,
        }
    }
}

/// Top-level Qwen2-VL configuration.
#[derive(Debug, Clone)]
pub struct Qwen2VLConfig {
    pub model_config: ModelConfig,
    pub vision_config: Qwen2VLVisionConfig,
    pub mrope_section: Vec<usize>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen2VLConfig {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(Qwen2VLVisionConfig::from_json)
            .unwrap_or_default();

        let mrope_section = cfg
            .extra
            .get("rope_scaling")
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_else(|| vec![16, 24, 24]);

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151655) as u32;

        let video_token_id = cfg
            .extra
            .get("video_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151656) as u32;

        let vision_start_token_id = cfg
            .extra
            .get("vision_start_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151652) as u32;

        let vision_end_token_id = cfg
            .extra
            .get("vision_end_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151653) as u32;

        Self {
            model_config: cfg.clone(),
            vision_config,
            mrope_section,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
        }
    }
}

// ─── MRoPE (Multi-dimensional Rotary Position Embedding) ────────────────────

/// Multi-dimensional Rotary Position Embedding for Qwen2-VL.
///
/// Splits the rotary dimension into sections (temporal/height/width) and
/// applies different position IDs per section. For text-only tokens, all
/// sections use the same sequential position, degenerating to standard RoPE.
pub(crate) struct MRoPE {
    sin: Tensor,
    cos: Tensor,
    /// Half-dim count per section (e.g., [16, 24, 24] for head_dim=128).
    mrope_section: Vec<usize>,
}

impl MRoPE {
    pub(crate) fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        mrope_section: Vec<usize>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = head_dim;
        let half_dim = rotary_dim / 2;
        let section_sum: usize = mrope_section.iter().sum();
        if section_sum != half_dim {
            return Err(candle_core::Error::Msg(format!(
                "mrope_section sum ({section_sum}) must equal head_dim/2 ({half_dim})"
            )));
        }

        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f32).powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            mrope_section,
        })
    }

    /// Apply MRoPE with 3D position IDs.
    ///
    /// # Arguments
    /// * `q` - [batch, heads, seq_len, head_dim]
    /// * `k` - [batch, kv_heads, seq_len, head_dim]
    /// * `position_ids` - [3, seq_len] (temporal, height, width)
    pub(crate) fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Assemble per-section cos/sin from different position dimensions
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        let mut freq_offset = 0;

        for (dim_idx, &section_half_dim) in self.mrope_section.iter().enumerate() {
            // Position IDs for this dimension: [seq_len]
            let pos = position_ids.narrow(0, dim_idx, 1)?.squeeze(0)?;

            // Index into cos/sin cache: [seq_len, total_half_dim]
            let cos_full = self.cos.index_select(&pos.to_dtype(DType::U32)?, 0)?;
            let sin_full = self.sin.index_select(&pos.to_dtype(DType::U32)?, 0)?;

            // Extract this section's frequency range
            let cos_section = cos_full.narrow(1, freq_offset, section_half_dim)?;
            let sin_section = sin_full.narrow(1, freq_offset, section_half_dim)?;

            cos_parts.push(cos_section);
            sin_parts.push(sin_section);
            freq_offset += section_half_dim;
        }

        // Concatenate sections: [seq_len, half_dim] — must be contiguous for rope()
        let cos = Tensor::cat(&cos_parts, 1)?.contiguous()?;
        let sin = Tensor::cat(&sin_parts, 1)?.contiguous()?;

        // Apply standard neox-style rotation
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let q_rot = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    /// Apply MRoPE with scalar position offset (text-only decode).
    /// All 3 dimensions use the same position.
    pub(crate) fn apply_scalar(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let q_rot = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    /// Apply MRoPE with per-token scalar positions (batched decode).
    #[allow(dead_code)]
    pub(crate) fn apply_varlen(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let total_tokens = positions.len();
        let pos_tensor = Tensor::from_vec(
            positions.iter().map(|&p| p as u32).collect::<Vec<_>>(),
            (total_tokens,),
            self.sin.device(),
        )?;
        let cos = self.cos.index_select(&pos_tensor, 0)?;
        let sin = self.sin.index_select(&pos_tensor, 0)?;

        // [total_tokens, heads, head_dim] -> [1, heads, total_tokens, head_dim]
        let q = q.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

        let q_rot = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;

        let q_rot = q_rot.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let k_rot = k_rot.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        Ok((q_rot, k_rot))
    }
}

// ─── Vision Components ──────────────────────────────────────────────────────

#[allow(dead_code)]
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&candle_nn::ops::sigmoid(&(x * 1.702)?)?)
}

/// Patch embedding: Conv3D-as-Linear (kernel == stride, no overlap).
///
/// Input: [num_patches, C * temporal_patch_size * patch_size * patch_size]
/// Output: [num_patches, embed_dim]
#[allow(dead_code)]
struct Qwen2VLPatchEmbed {
    proj: Linear,
}

#[allow(dead_code)]
impl Qwen2VLPatchEmbed {
    fn new(cfg: &Qwen2VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let proj = candle_nn::linear(cfg.patch_input_dim(), cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

/// Vision self-attention with fused QKV and 2D RoPE.
#[allow(dead_code)]
struct Qwen2VLVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl Qwen2VLVisionAttention {
    fn new(cfg: &Qwen2VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv = candle_nn::linear(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv"))?;
        let proj = candle_nn::linear(cfg.embed_dim, cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward with optional precomputed 2D rotary embeddings.
    ///
    /// # Arguments
    /// * `x` - [num_tokens, embed_dim]
    /// * `rotary_emb` - Optional (cos, sin) each [num_tokens, rotary_dim/2]
    fn forward(&self, x: &Tensor, rotary_emb: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (num_tokens, _) = x.dims2()?;

        // Fused QKV: [num_tokens, 3 * embed_dim]
        let qkv = self.qkv.forward(x)?;

        // Split: [num_tokens, num_heads, head_dim] each
        let q = qkv.narrow(1, 0, self.num_heads * self.head_dim)?;
        let k = qkv.narrow(
            1,
            self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;
        let v = qkv.narrow(
            1,
            2 * self.num_heads * self.head_dim,
            self.num_heads * self.head_dim,
        )?;

        let q = q.reshape((num_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((num_tokens, self.num_heads, self.head_dim))?;
        let v = v.reshape((num_tokens, self.num_heads, self.head_dim))?;

        // Apply 2D RoPE to partial dims if provided
        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            let rotary_dim = cos.dim(1)? * 2;
            let q_rot = q.narrow(2, 0, rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;
            let k_rot = k.narrow(2, 0, rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;

            // Reshape for rope: [1, heads, tokens, rotary_dim]
            let q_rot = q_rot.transpose(0, 1)?.unsqueeze(0)?;
            let k_rot = k_rot.transpose(0, 1)?.unsqueeze(0)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, cos, sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, cos, sin)?;
            let q_rot = q_rot.squeeze(0)?.transpose(0, 1)?;
            let k_rot = k_rot.squeeze(0)?.transpose(0, 1)?;

            let q = Tensor::cat(&[q_rot, q_pass.contiguous()?], 2)?;
            let k = Tensor::cat(&[k_rot, k_pass.contiguous()?], 2)?;
            (q, k)
        } else {
            (q, k)
        };

        // Self-attention: [1, heads, tokens, head_dim]
        let q = q.transpose(0, 1)?.unsqueeze(0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [1, heads, tokens, head_dim] -> [tokens, embed_dim]
        let attn_output = attn_output
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_tokens, self.num_heads * self.head_dim))?;

        self.proj.forward(&attn_output)
    }
}

/// Vision MLP: fc1 + QuickGELU + fc2.
#[allow(dead_code)]
struct Qwen2VLVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

#[allow(dead_code)]
impl Qwen2VLVisionMLP {
    fn new(embed_dim: usize, mlp_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(embed_dim, mlp_hidden, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(mlp_hidden, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = quick_gelu(&x)?;
        self.fc2.forward(&x)
    }
}

/// Vision transformer block: LayerNorm + Attention + LayerNorm + MLP.
#[allow(dead_code)]
struct Qwen2VLVisionBlock {
    norm1: LayerNorm,
    attn: Qwen2VLVisionAttention,
    norm2: LayerNorm,
    mlp: Qwen2VLVisionMLP,
}

#[allow(dead_code)]
impl Qwen2VLVisionBlock {
    fn new(cfg: &Qwen2VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?;
        let attn = Qwen2VLVisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Qwen2VLVisionMLP::new(cfg.embed_dim, cfg.mlp_hidden_size(), vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, rotary_emb: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, rotary_emb)?;
        let x = (x + residual)?;
        let residual = &x;
        let xs = self.mlp.forward(&self.norm2.forward(&x)?)?;
        residual + xs
    }
}

/// PatchMerger: reduces vision tokens by spatial_merge_size^2.
///
/// Groups adjacent patches and projects them:
/// [num_merged_groups, merge_size^2, embed_dim] -> [num_merged_groups, hidden_size]
#[allow(dead_code)]
struct Qwen2VLPatchMerger {
    ln_q: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    spatial_merge_size: usize,
}

#[allow(dead_code)]
impl Qwen2VLPatchMerger {
    fn new(cfg: &Qwen2VLVisionConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let merger_hidden = cfg.merger_hidden_dim();
        let ln_q = layer_norm(cfg.embed_dim, 1e-6, vb.pp("ln_q"))?;
        // mlp.0 and mlp.2 (Linear layers in nn.Sequential)
        let mlp_fc1 = candle_nn::linear(merger_hidden, merger_hidden, vb.pp("mlp").pp("0"))?;
        let mlp_fc2 = candle_nn::linear(merger_hidden, hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            ln_q,
            mlp_fc1,
            mlp_fc2,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    /// Merge patches: input [num_tokens, embed_dim], grid [h, w] patches.
    ///
    /// Groups into spatial_merge_size^2 blocks, flattens, and projects.
    /// Returns [num_merged, hidden_size].
    fn forward(&self, x: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let x = self.ln_q.forward(x)?;
        let embed_dim = x.dim(1)?;
        let merge = self.spatial_merge_size;

        // Reshape [H*W, D] -> [H/m, m, W/m, m, D] -> [H/m, W/m, m*m, D] -> [H/m*W/m, m*m*D]
        let x = x.reshape((grid_h / merge, merge, grid_w / merge, merge, embed_dim))?;
        let x = x.permute((0, 2, 1, 3, 4))?;
        let num_merged = (grid_h / merge) * (grid_w / merge);
        let x = x.reshape((num_merged, merge * merge * embed_dim))?;

        // MLP: Linear + GELU + Linear
        let x = self.mlp_fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.mlp_fc2.forward(&x)
    }
}

// ─── Vision Transformer ─────────────────────────────────────────────────────

/// Qwen2-VL Vision Transformer.
///
/// Processes images into embeddings compatible with the language model.
#[allow(dead_code)]
struct Qwen2VisionTransformer {
    patch_embed: Qwen2VLPatchEmbed,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<Qwen2VLVisionBlock>,
    merger: Qwen2VLPatchMerger,
}

#[allow(dead_code)]
impl Qwen2VisionTransformer {
    fn new(cfg: &Qwen2VLVisionConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let patch_embed = Qwen2VLPatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        // 2D RoPE for vision: partial_rotary_factor=0.5, neox style
        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.head_dim(),
            8192, // Large enough for any image resolution
            10000.0,
            0.5,
            true,
            vb.dtype(),
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Qwen2VLVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let merger = Qwen2VLPatchMerger::new(cfg, hidden_size, vb.pp("merger"))?;

        Ok(Self {
            patch_embed,
            rotary_emb,
            blocks,
            merger,
        })
    }

    /// Encode an image into language model embeddings.
    ///
    /// # Arguments
    /// * `patches` - Flattened patches [num_patches, C*T*H*W]
    /// * `grid_h` - Number of patches along height
    /// * `grid_w` - Number of patches along width
    ///
    /// # Returns
    /// [num_merged_tokens, hidden_size]
    fn forward(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let num_tokens = patches.dim(0)?;

        // Patch embedding
        let mut x = self.patch_embed.forward(patches)?;

        // Compute 2D position IDs for RoPE
        let rotary_emb = self.compute_2d_rotary(num_tokens, grid_h, grid_w, x.device())?;

        // Transformer blocks
        let rotary_ref = rotary_emb
            .as_ref()
            .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor));
        for block in &self.blocks {
            x = block.forward(&x, rotary_ref)?;
        }

        // Merge patches to reduce token count
        self.merger.forward(&x, grid_h, grid_w)
    }

    /// Compute 2D rotary position embeddings for vision tokens.
    fn compute_2d_rotary(
        &self,
        _num_tokens: usize,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        // Create 2D position grid: (h_pos, w_pos) for each token
        let mut h_positions = Vec::with_capacity(grid_h * grid_w);
        let mut w_positions = Vec::with_capacity(grid_h * grid_w);
        for h in 0..grid_h {
            for w in 0..grid_w {
                h_positions.push(h as u32);
                w_positions.push(w as u32);
            }
        }

        // Stack into [num_tokens, 2] position IDs, then use the rotary embedding
        // For 2D RoPE: interleave h and w positions for the rotary dim
        // The rotary_dim is head_dim * 0.5, split equally between h and w
        let rotary_dim = self.rotary_emb.rotary_dim();
        let half_rotary = rotary_dim / 2;

        // Get cos/sin for height positions
        let h_pos = Tensor::from_vec(h_positions, (grid_h * grid_w,), device)?;
        let w_pos = Tensor::from_vec(w_positions, (grid_h * grid_w,), device)?;

        // Index into the precomputed cos/sin tables
        // cos/sin tables are [max_seq_len, rotary_dim/2]
        let cos_h = self.rotary_emb.cos().index_select(&h_pos, 0)?;
        let sin_h = self.rotary_emb.sin().index_select(&h_pos, 0)?;
        let cos_w = self.rotary_emb.cos().index_select(&w_pos, 0)?;
        let sin_w = self.rotary_emb.sin().index_select(&w_pos, 0)?;

        // Take first half_rotary dims for h, next half_rotary for w
        let cos_h = cos_h.narrow(1, 0, half_rotary)?;
        let sin_h = sin_h.narrow(1, 0, half_rotary)?;
        let cos_w = cos_w.narrow(1, 0, half_rotary)?;
        let sin_w = sin_w.narrow(1, 0, half_rotary)?;

        // Concatenate: [num_tokens, rotary_dim/2]
        let cos = Tensor::cat(&[cos_h, cos_w], 1)?;
        let sin = Tensor::cat(&[sin_h, sin_w], 1)?;

        Ok(Some((cos, sin)))
    }
}

// ─── Vision Encoder (no merger — for reuse by other VLMs) ───────────────────

/// Qwen2-VL vision encoder without the PatchMerger.
///
/// Produces per-patch features `[num_patches, embed_dim]` that downstream
/// models (e.g. Kanana-V) can post-process with their own projectors.
pub(crate) struct Qwen2VLVisionEncoder {
    patch_embed: Qwen2VLPatchEmbed,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<Qwen2VLVisionBlock>,
}

impl Qwen2VLVisionEncoder {
    pub(crate) fn new(cfg: &Qwen2VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = Qwen2VLPatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.head_dim(),
            8192,
            10000.0,
            0.5,
            true,
            vb.dtype(),
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Qwen2VLVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        Ok(Self {
            patch_embed,
            rotary_emb,
            blocks,
        })
    }

    /// Encode a single image: `patches [np, cps]` → `[np, embed_dim]`.
    pub(crate) fn encode(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(patches)?;

        // Compute 2D rotary embeddings (same as Qwen2VisionTransformer).
        let rotary_dim = self.rotary_emb.rotary_dim();
        let half_rotary = rotary_dim / 2;
        let device = patches.device();

        let mut h_positions = Vec::with_capacity(grid_h * grid_w);
        let mut w_positions = Vec::with_capacity(grid_h * grid_w);
        for h in 0..grid_h {
            for w in 0..grid_w {
                h_positions.push(h as u32);
                w_positions.push(w as u32);
            }
        }
        let h_pos = Tensor::from_vec(h_positions, (grid_h * grid_w,), device)?;
        let w_pos = Tensor::from_vec(w_positions, (grid_h * grid_w,), device)?;

        let cos_h = self.rotary_emb.cos().index_select(&h_pos, 0)?;
        let sin_h = self.rotary_emb.sin().index_select(&h_pos, 0)?;
        let cos_w = self.rotary_emb.cos().index_select(&w_pos, 0)?;
        let sin_w = self.rotary_emb.sin().index_select(&w_pos, 0)?;

        let cos_h = cos_h.narrow(1, 0, half_rotary)?;
        let sin_h = sin_h.narrow(1, 0, half_rotary)?;
        let cos_w = cos_w.narrow(1, 0, half_rotary)?;
        let sin_w = sin_w.narrow(1, 0, half_rotary)?;

        let cos = Tensor::cat(&[cos_h, cos_w], 1)?;
        let sin = Tensor::cat(&[sin_h, sin_w], 1)?;
        let rotary_emb = Some((&cos, &sin));

        for block in &self.blocks {
            x = block.forward(&x, rotary_emb.map(|(c, s)| (c as &Tensor, s as &Tensor)))?;
        }

        Ok(x)
    }
}

// ─── Language Model (Qwen2 backbone with MRoPE) ─────────────────────────────

pub(crate) struct Qwen2VLAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    mrope: MRoPE,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2VLAttention {
    pub(crate) fn new(cfg: &ModelConfig, mrope_section: &[usize], vb: VarBuilder) -> Result<Self> {
        let use_bias = cfg.attention_bias.unwrap_or(true);

        let q_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            use_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            use_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            use_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_b(cfg.hidden_size, cfg.hidden_size, false, vb.pp("o_proj"))?;

        let mrope = MRoPE::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            mrope_section.to_vec(),
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mrope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply MRoPE with 3D positions if available, else scalar
        let (q, k) = if let Some(pos_ids) = position_ids {
            self.mrope.apply(&q, &k, pos_ids)?
        } else {
            self.mrope.apply_scalar(&q, &k, seqlen_offset)?
        };

        paged_attention(
            &q,
            &k,
            &v,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
    }

    pub(crate) fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Decode: text-only, all dimensions use same position
        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.mrope.apply_varlen(&q, &k, &positions)?;

            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let max_blocks_per_seq = sequences
                .iter()
                .map(|s| s.block_ids.len())
                .max()
                .unwrap_or(1);
            let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
            for (i, seq) in sequences.iter().enumerate() {
                for (j, &block_id) in seq.block_ids.iter().enumerate() {
                    bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                }
            }
            let block_tables =
                Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

            let seq_lens_data: Vec<u32> = sequences
                .iter()
                .map(|s| (s.seqlen_offset + 1) as u32)
                .collect();
            let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
            let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
            )?;

            self.o_proj.forward(&attn_output)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = self.mrope.apply_scalar(&q_i, &k_i, seq.seqlen_offset)?;

                let attn_out = paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    None,
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?;
                outputs.push(attn_out);
            }

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.o_proj.forward(&attn_output)
        }
    }
}

pub(crate) struct Qwen2VLDecoderLayer {
    self_attn: Qwen2VLAttention,
    mlp_gate_proj: Linear,
    mlp_up_proj: Linear,
    mlp_down_proj: Linear,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2VLDecoderLayer {
    pub(crate) fn new(cfg: &ModelConfig, mrope_section: &[usize], vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen2VLAttention::new(cfg, mrope_section, vb.pp("self_attn"))?;
        let mlp_vb = vb.pp("mlp");
        let mlp_gate_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            mlp_vb.pp("gate_proj"),
        )?;
        let mlp_up_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            mlp_vb.pp("up_proj"),
        )?;
        let mlp_down_proj = candle_nn::linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            mlp_vb.pp("down_proj"),
        )?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp_gate_proj,
            mlp_up_proj,
            mlp_down_proj,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn swiglu_forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.mlp_gate_proj.forward(x)?;
        let up = self.mlp_up_proj.forward(x)?;
        self.mlp_down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            position_ids,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.swiglu_forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }

    pub(crate) fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.swiglu_forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Qwen2-VL model for conditional generation.
pub struct Qwen2VLForConditionalGeneration {
    /// Vision encoder — used by the multimodal processor to encode images.
    #[allow(dead_code)]
    visual: Qwen2VisionTransformer,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen2VLDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen2VLConfig,
    device: Device,
    dtype: DType,
}

impl Qwen2VLForConditionalGeneration {
    pub fn new(cfg: &Qwen2VLConfig, vb: VarBuilder) -> Result<Self> {
        let visual = Qwen2VisionTransformer::new(
            &cfg.vision_config,
            cfg.model_config.hidden_size,
            vb.pp("visual"),
        )?;

        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            cfg.model_config.vocab_size,
            cfg.model_config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(cfg.model_config.num_hidden_layers);
        for i in 0..cfg.model_config.num_hidden_layers {
            layers.push(Qwen2VLDecoderLayer::new(
                &cfg.model_config,
                &cfg.mrope_section,
                vb_m.pp("layers").pp(i),
            )?);
        }

        let norm = rms_norm(
            cfg.model_config.hidden_size,
            cfg.model_config.rms_norm_eps,
            vb_m.pp("norm"),
        )?;

        let lm_head = if cfg.model_config.tie_word_embeddings {
            let emb_w = vb_m.pp("embed_tokens").get(
                (cfg.model_config.vocab_size, cfg.model_config.hidden_size),
                "weight",
            )?;
            Linear::new(emb_w, None)
        } else {
            candle_nn::linear_no_bias(
                cfg.model_config.hidden_size,
                cfg.model_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };

        Ok(Self {
            visual,
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let qwen2vl_cfg = Qwen2VLConfig::from_model_config(cfg);
        Self::new(&qwen2vl_cfg, vb)
    }

    /// Compute 3D MRoPE position IDs for a mixed text+image sequence.
    ///
    /// For text tokens: all 3 dimensions use the same sequential position.
    /// For image tokens: temporal is fixed, H/W vary spatially.
    fn compute_position_ids(
        &self,
        input_ids: &[u32],
        mm_inputs: Option<&MultimodalInputs>,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let mut positions = vec![vec![0u32; seq_len]; 3];

        if mm_inputs.is_none() || !mm_inputs.is_some_and(|m| m.has_images()) {
            // Text only: sequential positions for all dimensions
            for (i, pos) in positions[0].iter_mut().enumerate() {
                *pos = i as u32;
            }
            positions[1] = positions[0].clone();
            positions[2] = positions[0].clone();
        } else {
            // Mixed text+image: assign spatial positions for image tokens
            let mut pos = 0u32;
            let mut i = 0;
            while i < seq_len {
                if input_ids[i] == self.config.image_token_id {
                    // Find the image info for this position
                    let grid_info = mm_inputs.and_then(|m| self.find_image_grid_at(m, i));
                    let (grid_h, grid_w) = grid_info.unwrap_or((1, 1));
                    let merge = self.config.vision_config.spatial_merge_size;
                    let merged_h = grid_h / merge;
                    let merged_w = grid_w / merge;
                    let num_image_tokens = merged_h * merged_w;

                    // Assign spatial positions for image tokens
                    for t in 0..num_image_tokens {
                        if i + t >= seq_len {
                            break;
                        }
                        let h = t / merged_w;
                        let w = t % merged_w;
                        positions[0][i + t] = pos; // temporal: same for all patches
                        positions[1][i + t] = pos + h as u32;
                        positions[2][i + t] = pos + w as u32;
                    }
                    let max_dim = merged_h.max(merged_w) as u32;
                    pos += max_dim;
                    i += num_image_tokens;
                } else {
                    positions[0][i] = pos;
                    positions[1][i] = pos;
                    positions[2][i] = pos;
                    pos += 1;
                    i += 1;
                }
            }
        }

        // Create [3, seq_len] tensor
        let flat: Vec<u32> = positions.into_iter().flatten().collect();
        Tensor::from_vec(flat, (3, seq_len), &self.device)
    }

    /// Find the grid dimensions for an image at the given token position.
    fn find_image_grid_at(
        &self,
        mm_inputs: &MultimodalInputs,
        pos: usize,
    ) -> Option<(usize, usize)> {
        for (img_pos, processed) in &mm_inputs.image_embeddings {
            if *img_pos == pos {
                return processed.grid_size;
            }
        }
        None
    }

    /// Merge pre-encoded vision embeddings with text embeddings.
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

impl crate::engine::ModelForward for Qwen2VLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                None, // No 3D positions for text-only
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
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
        let (_b_size, seq_len) = input_ids.dims2()?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        // Compute 3D position IDs
        let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let position_ids = self.compute_position_ids(&input_ids_vec, multimodal_inputs)?;

        // Text embeddings
        let text_embeddings = self.embed_tokens.forward(input_ids)?;

        // Merge with vision embeddings if present
        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm_inputs)?
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                Some(&position_ids),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
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
            architectures: vec!["Qwen2VLForConditionalGeneration".to_string()],
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
            rope_theta: 1000000.0,
            tie_word_embeddings: true,
            bos_token_id: 151643,
            eos_token_id: 151645,
            sliding_window: None,
            attention_bias: Some(true),
            extra: serde_json::Map::new(),
        }
    }

    fn test_vision_config() -> Qwen2VLVisionConfig {
        Qwen2VLVisionConfig {
            depth: 2,
            embed_dim: 64,
            num_heads: 4,
            mlp_ratio: 4.0,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
        }
    }

    fn test_qwen2vl_config() -> Qwen2VLConfig {
        Qwen2VLConfig {
            model_config: test_model_config(),
            vision_config: test_vision_config(),
            mrope_section: vec![2, 3, 3],
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
        }
    }

    // ── Config Tests ────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = Qwen2VLVisionConfig::default();
        assert_eq!(cfg.depth, 32);
        assert_eq!(cfg.embed_dim, 1280);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim(), 80);
        assert_eq!(cfg.mlp_hidden_size(), 5120);
        assert_eq!(cfg.patch_input_dim(), 3 * 2 * 14 * 14); // 1176
        assert_eq!(cfg.merger_hidden_dim(), 1280 * 4); // 5120
    }

    #[test]
    fn test_vision_config_from_json() {
        let json = serde_json::json!({
            "depth": 16,
            "embed_dim": 512,
            "num_heads": 8,
            "mlp_ratio": 2.0,
            "patch_size": 7,
            "temporal_patch_size": 1,
            "spatial_merge_size": 4
        });
        let cfg = Qwen2VLVisionConfig::from_json(&json);
        assert_eq!(cfg.depth, 16);
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.mlp_ratio, 2.0);
        assert_eq!(cfg.patch_size, 7);
        assert_eq!(cfg.temporal_patch_size, 1);
        assert_eq!(cfg.spatial_merge_size, 4);
    }

    #[test]
    fn test_qwen2vl_config_from_model_config() {
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 8,
                "embed_dim": 256,
                "num_heads": 4
            }),
        );
        model_cfg.extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({
                "mrope_section": [4, 6, 6]
            }),
        );
        model_cfg
            .extra
            .insert("image_token_id".to_string(), serde_json::json!(100));

        let cfg = Qwen2VLConfig::from_model_config(&model_cfg);
        assert_eq!(cfg.vision_config.depth, 8);
        assert_eq!(cfg.vision_config.embed_dim, 256);
        assert_eq!(cfg.mrope_section, vec![4, 6, 6]);
        assert_eq!(cfg.image_token_id, 100);
    }

    // ── MRoPE Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_mrope_creation() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 3, 3], DType::F32, &device);
        assert!(mrope.is_ok());
    }

    #[test]
    fn test_mrope_section_sum_mismatch() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 2, 2], DType::F32, &device);
        assert!(mrope.is_err());
    }

    #[test]
    fn test_mrope_scalar_application() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 3, 3], DType::F32, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 3, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 16), &device).unwrap();

        let (q_rot, k_rot) = mrope.apply_scalar(&q, &k, 0).unwrap();
        assert_eq!(q_rot.dims(), &[1, 4, 3, 16]);
        assert_eq!(k_rot.dims(), &[1, 2, 3, 16]);
    }

    #[test]
    fn test_mrope_3d_positions() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 3, 3], DType::F32, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();

        // 3D positions [3, 4]
        let position_ids =
            Tensor::from_vec(vec![0u32, 1, 2, 3, 0, 0, 1, 1, 0, 1, 0, 1], (3, 4), &device).unwrap();

        let (q_rot, k_rot) = mrope.apply(&q, &k, &position_ids).unwrap();
        assert_eq!(q_rot.dims(), &[1, 4, 4, 16]);
        assert_eq!(k_rot.dims(), &[1, 2, 4, 16]);
    }

    #[test]
    fn test_mrope_text_only_matches_scalar() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 3, 3], DType::F32, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 4, 3, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 16), &device).unwrap();

        // When all 3 dims have same positions [0,1,2], should match scalar offset=0
        let pos_ids =
            Tensor::from_vec(vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2], (3, 3), &device).unwrap();

        let (q_3d, k_3d) = mrope.apply(&q, &k, &pos_ids).unwrap();
        let (q_scalar, k_scalar) = mrope.apply_scalar(&q, &k, 0).unwrap();

        // Should produce identical results
        let q_diff: f32 = (q_3d - q_scalar)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(q_diff < 1e-5, "q diff: {q_diff}");

        let k_diff: f32 = (k_3d - k_scalar)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(k_diff < 1e-5, "k diff: {k_diff}");
    }

    // ── Vision Components Tests ─────────────────────────────────────────

    #[test]
    fn test_patch_embed() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pe = Qwen2VLPatchEmbed::new(&cfg, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, cfg.patch_input_dim()), &device).unwrap();
        let output = pe.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_attention() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = Qwen2VLVisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = attn.forward(&x, None).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_mlp() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = Qwen2VLVisionMLP::new(cfg.embed_dim, cfg.mlp_hidden_size(), vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_block() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = Qwen2VLVisionBlock::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = block.forward(&x, None).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_patch_merger() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let hidden_size = 64;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let merger = Qwen2VLPatchMerger::new(&cfg, hidden_size, vb).unwrap();

        // 4x4 grid of patches -> 2x2 merged = 4 tokens
        let x = Tensor::randn(0.0f32, 1.0, (16, cfg.embed_dim), &device).unwrap();
        let output = merger.forward(&x, 4, 4).unwrap();
        assert_eq!(output.dims(), &[4, hidden_size]);
    }

    #[test]
    fn test_vision_transformer() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let hidden_size = 64;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vit = Qwen2VisionTransformer::new(&cfg, hidden_size, vb).unwrap();

        // 4x4 grid, patch_input_dim = 3*2*14*14 = 1176
        let patches = Tensor::randn(0.0f32, 1.0, (16, cfg.patch_input_dim()), &device).unwrap();
        let output = vit.forward(&patches, 4, 4).unwrap();

        // 4x4 grid with merge_size=2 -> 2x2 = 4 merged tokens
        assert_eq!(output.dims(), &[4, hidden_size]);
    }

    // ── Full Model Tests ────────────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_qwen2vl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2VLForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_model_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_qwen2vl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2VLForConditionalGeneration::new(&cfg, vb).unwrap();

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
        let cfg = test_qwen2vl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2VLForConditionalGeneration::new(&cfg, vb).unwrap();

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

        // Each sequence generates 1 token of logits
        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_compute_position_ids_text_only() {
        let device = Device::Cpu;
        let cfg = test_qwen2vl_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let input_ids = vec![1, 2, 3, 4, 5];
        let pos_ids = model.compute_position_ids(&input_ids, None).unwrap();

        assert_eq!(pos_ids.dims(), &[3, 5]);
        let pos_data: Vec<u32> = pos_ids.flatten_all().unwrap().to_vec1().unwrap();
        // All 3 dimensions should be [0, 1, 2, 3, 4]
        assert_eq!(&pos_data[0..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&pos_data[5..10], &[0, 1, 2, 3, 4]);
        assert_eq!(&pos_data[10..15], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let mut model_cfg = test_model_config();
        model_cfg.extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 2,
                "embed_dim": 64,
                "num_heads": 4
            }),
        );
        model_cfg.extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({ "mrope_section": [2, 3, 3] }),
        );

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen2VLForConditionalGeneration::from_model_config(&model_cfg, vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_mrope_varlen() {
        let device = Device::Cpu;
        let mrope = MRoPE::new(16, 64, 10000.0, vec![2, 3, 3], DType::F32, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (3, 4, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (3, 2, 16), &device).unwrap();
        let positions = vec![0, 5, 10];

        let (q_rot, k_rot) = mrope.apply_varlen(&q, &k, &positions).unwrap();
        assert_eq!(q_rot.dims(), &[3, 4, 16]);
        assert_eq!(k_rot.dims(), &[3, 2, 16]);
    }

    #[test]
    fn test_quick_gelu() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![0.0f32, 1.0, -1.0], (3,), &device).unwrap();
        let y = quick_gelu(&x).unwrap();
        let values: Vec<f32> = y.to_vec1().unwrap();
        // quick_gelu(0) = 0
        assert!((values[0]).abs() < 1e-6);
        // quick_gelu(1) > 0
        assert!(values[1] > 0.5);
        // quick_gelu(-1) < 0
        assert!(values[2] < 0.0);
    }
}
