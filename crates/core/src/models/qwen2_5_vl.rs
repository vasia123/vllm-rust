//! Qwen2.5-VL vision-language model implementation.
//!
//! Evolutionary upgrade of Qwen2-VL with key vision encoder changes:
//! - Window attention: per-layer choice between full and windowed attention
//! - Gated MLP (SiLU): gate_up_proj + down_proj (replaces QuickGELU fc1+fc2)
//! - RMSNorm (replaces LayerNorm in vision encoder)
//! - Configurable intermediate_size and hidden_act
//!
//! LLM backbone is identical to Qwen2-VL (MRoPE, Qwen2 decoder layers).

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{causal_mask, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::qwen2_vl::{Qwen2VLConfig, Qwen2VLDecoderLayer};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Qwen2.5-VL vision encoder configuration.
#[derive(Debug, Clone)]
pub(crate) struct Qwen25VLVisionConfig {
    pub(crate) depth: usize,
    pub(crate) embed_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) in_channels: usize,
    pub(crate) patch_size: usize,
    pub(crate) temporal_patch_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) window_size: usize,
    pub(crate) fullatt_block_indexes: Vec<usize>,
    pub(crate) out_hidden_size: usize,
}

impl Default for Qwen25VLVisionConfig {
    fn default() -> Self {
        Self {
            depth: 32,
            embed_dim: 1280,
            num_heads: 16,
            intermediate_size: 3420,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            window_size: 112,
            fullatt_block_indexes: vec![7, 15, 23, 31],
            out_hidden_size: 3584,
        }
    }
}

impl Qwen25VLVisionConfig {
    fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    fn patch_input_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    fn merger_hidden_dim(&self) -> usize {
        self.embed_dim * self.spatial_merge_size * self.spatial_merge_size
    }

    pub(crate) fn from_json(json: &serde_json::Value) -> Self {
        let defaults = Self::default();

        let embed_dim = json
            .get("hidden_size")
            .or_else(|| json.get("embed_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(defaults.embed_dim as u64) as usize;

        let fullatt_block_indexes = json
            .get("fullatt_block_indexes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_else(|| defaults.fullatt_block_indexes.clone());

        Self {
            depth: json
                .get("depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.depth as u64) as usize,
            embed_dim,
            num_heads: json
                .get("num_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.num_heads as u64) as usize,
            intermediate_size: json
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.intermediate_size as u64)
                as usize,
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
            window_size: json
                .get("window_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.window_size as u64) as usize,
            fullatt_block_indexes,
            out_hidden_size: json
                .get("out_hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(defaults.out_hidden_size as u64) as usize,
        }
    }
}

/// Top-level Qwen2.5-VL configuration.
#[derive(Debug, Clone)]
struct Qwen25VLConfig {
    base: Qwen2VLConfig,
    vision_config: Qwen25VLVisionConfig,
}

impl Qwen25VLConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let base = Qwen2VLConfig::from_model_config(cfg);

        let vision_config = cfg
            .extra
            .get("vision_config")
            .map(Qwen25VLVisionConfig::from_json)
            .unwrap_or_default();

        Self {
            base,
            vision_config,
        }
    }
}

// ─── Vision Components ──────────────────────────────────────────────────────

/// Patch embedding: Conv3D-as-Linear (no bias, matching Qwen2.5-VL).
#[allow(dead_code)]
struct Qwen25VLPatchEmbed {
    proj: Linear,
}

#[allow(dead_code)]
impl Qwen25VLPatchEmbed {
    fn new(cfg: &Qwen25VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let proj = candle_nn::linear_no_bias(cfg.patch_input_dim(), cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

/// Vision MLP with gated activation (SiLU gate_and_mul).
///
/// gate_up_proj: [embed_dim] -> [2 * intermediate_size]
/// Split into gate and up halves, apply silu(gate) * up, then down_proj.
#[allow(dead_code)]
struct Qwen25VLVisionMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
}

#[allow(dead_code)]
impl Qwen25VLVisionMLP {
    fn new(embed_dim: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj =
            candle_nn::linear(embed_dim, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, embed_dim, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let chunks = gate_up.chunk(2, candle_core::D::Minus1)?;
        let gate = &chunks[0];
        let up = &chunks[1];
        let activated = (candle_nn::ops::silu(gate)? * up)?;
        self.down_proj.forward(&activated)
    }
}

/// Vision attention with fused QKV and 2D RoPE.
#[allow(dead_code)]
struct Qwen25VLVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

#[allow(dead_code)]
impl Qwen25VLVisionAttention {
    fn new(cfg: &Qwen25VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv = candle_nn::linear(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv"))?;
        let proj = candle_nn::linear(cfg.embed_dim, cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f64).powf(-0.5),
        })
    }

    /// Forward with 2D rotary embeddings and optional attention mask.
    ///
    /// # Arguments
    /// * `x` - [num_tokens, embed_dim]
    /// * `rotary_emb` - (cos, sin) each [num_tokens, rotary_dim/2]
    /// * `attention_mask` - Optional [1, 1, num_tokens, num_tokens] additive mask
    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (num_tokens, _) = x.dims2()?;

        let qkv = self.qkv.forward(x)?;
        let q_size = self.num_heads * self.head_dim;

        let q = qkv.narrow(1, 0, q_size)?;
        let k = qkv.narrow(1, q_size, q_size)?;
        let v = qkv.narrow(1, 2 * q_size, q_size)?;

        let q = q
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        // Apply 2D RoPE
        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            let rotary_dim = cos.dim(1)? * 2;
            let q_rot = q.narrow(2, 0, rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;
            let k_rot = k.narrow(2, 0, rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;

            let q_rot = q_rot.unsqueeze(0)?;
            let k_rot = k_rot.unsqueeze(0)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, cos, sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, cos, sin)?;
            let q_rot = q_rot.squeeze(0)?;
            let k_rot = k_rot.squeeze(0)?;

            let q = Tensor::cat(&[q_rot, q_pass.contiguous()?], 2)?;
            let k = Tensor::cat(&[k_rot, k_pass.contiguous()?], 2)?;
            (q, k)
        } else {
            (q, k)
        };

        // [heads, tokens, head_dim] -> [1, heads, tokens, head_dim]
        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        let attn_output = attn_output
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_tokens, self.num_heads * self.head_dim))?;

        self.proj.forward(&attn_output)
    }
}

/// Vision block: RMSNorm + Attention + RMSNorm + gated MLP.
#[allow(dead_code)]
struct Qwen25VLVisionBlock {
    norm1: RmsNorm,
    attn: Qwen25VLVisionAttention,
    norm2: RmsNorm,
    mlp: Qwen25VLVisionMLP,
}

#[allow(dead_code)]
impl Qwen25VLVisionBlock {
    fn new(cfg: &Qwen25VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = rms_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?;
        let attn = Qwen25VLVisionAttention::new(cfg, vb.pp("attn"))?;
        let norm2 = rms_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?;
        let mlp = Qwen25VLVisionMLP::new(cfg.embed_dim, cfg.intermediate_size, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x_attn = self.attn.forward(&x, rotary_emb, attention_mask)?;
        let x = (residual + &x_attn)?;
        let residual = &x;
        let xs = self.mlp.forward(&self.norm2.forward(&x)?)?;
        residual + xs
    }
}

/// PatchMerger with RMSNorm (matches Qwen2.5-VL).
#[allow(dead_code)]
struct Qwen25VLPatchMerger {
    ln_q: RmsNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    spatial_merge_size: usize,
}

#[allow(dead_code)]
impl Qwen25VLPatchMerger {
    fn new(cfg: &Qwen25VLVisionConfig, out_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let merger_hidden = cfg.merger_hidden_dim();
        let ln_q = rms_norm(cfg.embed_dim, 1e-6, vb.pp("ln_q"))?;
        let mlp_fc1 = candle_nn::linear(merger_hidden, merger_hidden, vb.pp("mlp").pp("0"))?;
        let mlp_fc2 = candle_nn::linear(merger_hidden, out_hidden_size, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            ln_q,
            mlp_fc1,
            mlp_fc2,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    fn forward(&self, x: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let x = self.ln_q.forward(x)?;
        let embed_dim = x.dim(1)?;
        let merge = self.spatial_merge_size;

        let x = x.reshape((grid_h / merge, merge, grid_w / merge, merge, embed_dim))?;
        let x = x.permute((0, 2, 1, 3, 4))?;
        let num_merged = (grid_h / merge) * (grid_w / merge);
        let x = x.reshape((num_merged, merge * merge * embed_dim))?;

        let x = self.mlp_fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.mlp_fc2.forward(&x)
    }
}

// ─── Window Attention Helpers ───────────────────────────────────────────────

/// Compute window indices for grouping tokens into spatial windows.
///
/// Returns (window_index, window_boundaries) where:
/// - window_index: permutation that groups tokens by window
/// - window_boundaries: cumulative sequence lengths for each window
#[allow(dead_code)]
fn get_window_index(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
) -> (Vec<usize>, Vec<usize>) {
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;
    let llm_grid_h = grid_h / spatial_merge_size;
    let llm_grid_w = grid_w / spatial_merge_size;

    if vit_merger_window_size == 0 || llm_grid_h == 0 || llm_grid_w == 0 {
        // Degenerate case: sequential order, single window
        let n = grid_t * llm_grid_h * llm_grid_w;
        return ((0..n).collect(), vec![n]);
    }

    // Pad grid dimensions to be divisible by window size
    let pad_h = if llm_grid_h.is_multiple_of(vit_merger_window_size) {
        0
    } else {
        vit_merger_window_size - llm_grid_h % vit_merger_window_size
    };
    let pad_w = if llm_grid_w.is_multiple_of(vit_merger_window_size) {
        0
    } else {
        vit_merger_window_size - llm_grid_w % vit_merger_window_size
    };
    let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    let total_llm_tokens = grid_t * llm_grid_h * llm_grid_w;
    let mut window_index = Vec::with_capacity(total_llm_tokens);
    let mut window_seqlens = Vec::new();

    for _t in 0..grid_t {
        // Build padded index grid for this temporal frame
        let base_offset = _t * llm_grid_h * llm_grid_w;
        let padded_h = llm_grid_h + pad_h;
        let padded_w = llm_grid_w + pad_w;
        let mut padded = vec![usize::MAX; padded_h * padded_w];
        for h in 0..llm_grid_h {
            for w in 0..llm_grid_w {
                padded[h * padded_w + w] = base_offset + h * llm_grid_w + w;
            }
        }

        // Extract windows
        for wh in 0..num_windows_h {
            for ww in 0..num_windows_w {
                let mut count = 0;
                for dh in 0..vit_merger_window_size {
                    for dw in 0..vit_merger_window_size {
                        let h = wh * vit_merger_window_size + dh;
                        let w = ww * vit_merger_window_size + dw;
                        let idx = padded[h * padded_w + w];
                        if idx != usize::MAX {
                            window_index.push(idx);
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    window_seqlens.push(count);
                }
            }
        }
    }

    (window_index, window_seqlens)
}

/// Build a block-diagonal attention mask from window boundaries.
///
/// Tokens within the same window can attend to each other; cross-window is masked.
/// Returns [1, 1, total_tokens, total_tokens] additive mask.
#[allow(dead_code)]
fn build_window_attention_mask(
    window_seqlens: &[usize],
    total_tokens: usize,
    spatial_merge_unit: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    // Scale seqlens by spatial_merge_unit (each merged token = merge_size^2 raw tokens)
    let scaled_total = total_tokens * spatial_merge_unit;
    let mut mask_data = vec![f32::NEG_INFINITY; scaled_total * scaled_total];

    let mut start = 0;
    for &seqlen in window_seqlens {
        let scaled_len = seqlen * spatial_merge_unit;
        for i in start..start + scaled_len {
            for j in start..start + scaled_len {
                if i < scaled_total && j < scaled_total {
                    mask_data[i * scaled_total + j] = 0.0;
                }
            }
        }
        start += scaled_len;
    }

    Tensor::from_vec(mask_data, (1, 1, scaled_total, scaled_total), device)?.to_dtype(dtype)
}

/// Invert a permutation: given perm[i] = j, returns inv[j] = i.
#[allow(dead_code)]
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ─── Vision Transformer ─────────────────────────────────────────────────────

/// Qwen2.5-VL Vision Transformer with window attention.
#[allow(dead_code)]
pub(crate) struct Qwen25VisionTransformer {
    patch_embed: Qwen25VLPatchEmbed,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<Qwen25VLVisionBlock>,
    merger: Qwen25VLPatchMerger,
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
}

#[allow(dead_code)]
impl Qwen25VisionTransformer {
    pub(crate) fn new(cfg: &Qwen25VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = Qwen25VLPatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.head_dim(),
            8192,
            10000.0,
            0.5,
            true,
            DType::F32,
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(Qwen25VLVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let merger = Qwen25VLPatchMerger::new(cfg, cfg.out_hidden_size, vb.pp("merger"))?;

        Ok(Self {
            patch_embed,
            rotary_emb,
            blocks,
            merger,
            spatial_merge_size: cfg.spatial_merge_size,
            window_size: cfg.window_size,
            patch_size: cfg.patch_size,
            fullatt_block_indexes: cfg.fullatt_block_indexes.clone(),
        })
    }

    /// Encode image patches into LLM-compatible embeddings.
    ///
    /// # Arguments
    /// * `patches` - [num_patches, C*T*H*W]
    /// * `grid_h` - Number of patches along height
    /// * `grid_w` - Number of patches along width
    pub(crate) fn forward(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let grid_t = 1; // Single image frame

        // Patch embedding
        let mut x = self.patch_embed.forward(patches)?;

        // Compute 2D rotary embeddings
        let rotary_emb = self.compute_2d_rotary(grid_h, grid_w, x.device())?;

        // Compute window indices
        let (window_index, window_seqlens) = get_window_index(
            grid_t,
            grid_h,
            grid_w,
            self.spatial_merge_size,
            self.window_size,
            self.patch_size,
        );
        let reverse_index = invert_permutation(&window_index);

        // Build attention masks
        let spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size;
        let llm_tokens = window_index.len();
        let raw_tokens = llm_tokens * spatial_merge_unit;

        // Permute tokens to window order (at merged-token granularity)
        // Each merged token = spatial_merge_unit raw tokens
        let mut permuted_indices = Vec::with_capacity(raw_tokens);
        for &wi in &window_index {
            for s in 0..spatial_merge_unit {
                permuted_indices.push((wi * spatial_merge_unit + s) as u32);
            }
        }
        let perm_tensor = Tensor::from_vec(permuted_indices, (raw_tokens,), x.device())?;
        x = x.index_select(&perm_tensor, 0)?;

        // Build window and full attention masks
        let window_mask = build_window_attention_mask(
            &window_seqlens,
            llm_tokens,
            spatial_merge_unit,
            x.dtype(),
            x.device(),
        )?;

        // Reorder rotary embeddings to match permuted token order
        let rotary_emb = rotary_emb.map(|(cos, sin)| {
            let cos = cos.index_select(&perm_tensor, 0).unwrap_or(cos);
            let sin = sin.index_select(&perm_tensor, 0).unwrap_or(sin);
            (cos, sin)
        });

        let rotary_ref = rotary_emb
            .as_ref()
            .map(|(cos, sin)| (cos as &Tensor, sin as &Tensor));

        // Process through transformer blocks
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mask = if self.fullatt_block_indexes.contains(&layer_idx) {
                None // Full attention: no mask
            } else {
                Some(&window_mask)
            };
            x = block.forward(&x, rotary_ref, mask)?;
        }

        // Un-permute back to original order
        let mut unperm_indices = Vec::with_capacity(raw_tokens);
        for &ri in &reverse_index {
            for s in 0..spatial_merge_unit {
                unperm_indices.push((ri * spatial_merge_unit + s) as u32);
            }
        }
        let unperm_tensor = Tensor::from_vec(unperm_indices, (raw_tokens,), x.device())?;
        x = x.index_select(&unperm_tensor, 0)?;

        // Merge patches
        self.merger.forward(&x, grid_h, grid_w)
    }

    fn compute_2d_rotary(
        &self,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let rotary_dim = self.rotary_emb.rotary_dim();
        let half_rotary = rotary_dim / 2;

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

        Ok(Some((cos, sin)))
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Qwen2.5-VL model for conditional generation.
///
/// Uses the upgraded vision encoder (window attention, gated MLP, RMSNorm)
/// with the same Qwen2 LLM backbone as Qwen2-VL.
pub struct Qwen25VLForConditionalGeneration {
    #[allow(dead_code)]
    visual: Qwen25VisionTransformer,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen2VLDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen25VLConfig,
    device: Device,
    dtype: DType,
}

impl Qwen25VLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let config = Qwen25VLConfig::from_model_config(cfg);

        let visual = Qwen25VisionTransformer::new(&config.vision_config, vb.pp("visual"))?;

        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen2VLDecoderLayer::new(
                cfg,
                &config.base.mrope_section,
                vb_m.pp("layers").pp(i),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_w = vb_m
                .pp("embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
            Linear::new(emb_w, None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            visual,
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn compute_position_ids(
        &self,
        input_ids: &[u32],
        mm_inputs: Option<&MultimodalInputs>,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let mut positions = vec![vec![0u32; seq_len]; 3];

        let image_token_id = self.config.base.image_token_id;
        let merge = self.config.vision_config.spatial_merge_size;

        if mm_inputs.is_none() || !mm_inputs.is_some_and(|m| m.has_images()) {
            for (i, pos) in positions[0].iter_mut().enumerate() {
                *pos = i as u32;
            }
            positions[1] = positions[0].clone();
            positions[2] = positions[0].clone();
        } else {
            let mut pos = 0u32;
            let mut i = 0;
            while i < seq_len {
                if input_ids[i] == image_token_id {
                    let grid_info = mm_inputs.and_then(|m| {
                        m.image_embeddings.iter().find_map(|(img_pos, processed)| {
                            if *img_pos == i {
                                processed.grid_size
                            } else {
                                None
                            }
                        })
                    });
                    let (grid_h, grid_w) = grid_info.unwrap_or((1, 1));
                    let merged_h = grid_h / merge;
                    let merged_w = grid_w / merge;
                    let num_image_tokens = merged_h * merged_w;

                    for t in 0..num_image_tokens {
                        if i + t >= seq_len {
                            break;
                        }
                        let h = t / merged_w;
                        let w = t % merged_w;
                        positions[0][i + t] = pos;
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

        let flat: Vec<u32> = positions.into_iter().flatten().collect();
        Tensor::from_vec(flat, (3, seq_len), &self.device)
    }

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

impl crate::engine::ModelForward for Qwen25VLForConditionalGeneration {
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
                None,
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

        let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let position_ids = self.compute_position_ids(&input_ids_vec, multimodal_inputs)?;

        let text_embeddings = self.embed_tokens.forward(input_ids)?;

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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 2,
                "hidden_size": 64,
                "num_heads": 4,
                "intermediate_size": 128,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "window_size": 56,
                "fullatt_block_indexes": [1],
                "out_hidden_size": 64
            }),
        );
        extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({ "mrope_section": [2, 3, 3] }),
        );
        extra.insert("image_token_id".to_string(), serde_json::json!(151655));
        extra.insert("video_token_id".to_string(), serde_json::json!(151656));

        ModelConfig {
            architectures: vec!["Qwen2_5_VLForConditionalGeneration".to_string()],
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
            extra,
        }
    }

    fn test_vision_config() -> Qwen25VLVisionConfig {
        Qwen25VLVisionConfig {
            depth: 2,
            embed_dim: 64,
            num_heads: 4,
            intermediate_size: 128,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            window_size: 56,
            fullatt_block_indexes: vec![1],
            out_hidden_size: 64,
        }
    }

    // ── Config Tests ────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_defaults() {
        let cfg = Qwen25VLVisionConfig::default();
        assert_eq!(cfg.depth, 32);
        assert_eq!(cfg.embed_dim, 1280);
        assert_eq!(cfg.intermediate_size, 3420);
        assert_eq!(cfg.window_size, 112);
        assert_eq!(cfg.fullatt_block_indexes, vec![7, 15, 23, 31]);
        assert_eq!(cfg.out_hidden_size, 3584);
    }

    #[test]
    fn test_vision_config_from_json() {
        let json = serde_json::json!({
            "depth": 8,
            "hidden_size": 512,
            "num_heads": 8,
            "intermediate_size": 1024,
            "window_size": 56,
            "fullatt_block_indexes": [3, 7],
            "out_hidden_size": 768
        });
        let cfg = Qwen25VLVisionConfig::from_json(&json);
        assert_eq!(cfg.depth, 8);
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.intermediate_size, 1024);
        assert_eq!(cfg.window_size, 56);
        assert_eq!(cfg.fullatt_block_indexes, vec![3, 7]);
        assert_eq!(cfg.out_hidden_size, 768);
    }

    #[test]
    fn test_config_from_model_config() {
        let cfg = test_model_config();
        let q25_cfg = Qwen25VLConfig::from_model_config(&cfg);
        assert_eq!(q25_cfg.vision_config.depth, 2);
        assert_eq!(q25_cfg.vision_config.embed_dim, 64);
        assert_eq!(q25_cfg.vision_config.window_size, 56);
        assert_eq!(q25_cfg.vision_config.fullatt_block_indexes, vec![1]);
        assert_eq!(q25_cfg.base.mrope_section, vec![2, 3, 3]);
    }

    // ── Window Attention Tests ──────────────────────────────────────────

    #[test]
    fn test_window_index_simple() {
        // 4x4 grid, merge=2, window_size=56, patch_size=14
        // vit_merger_window_size = 56 / 2 / 14 = 2
        // llm_grid = 4/2 = 2x2, exactly one 2x2 window
        let (index, seqlens) = get_window_index(1, 4, 4, 2, 56, 14);
        assert_eq!(index.len(), 4); // 2x2 = 4 merged tokens
        assert_eq!(seqlens.iter().sum::<usize>(), 4);
    }

    #[test]
    fn test_window_index_multiple_windows() {
        // 8x8 grid, merge=2, window_size=56, patch_size=14
        // vit_merger_window_size = 56/2/14 = 2
        // llm_grid = 4x4, 4 windows of 2x2 each
        let (index, seqlens) = get_window_index(1, 8, 8, 2, 56, 14);
        assert_eq!(index.len(), 16); // 4x4 = 16 merged tokens
        assert_eq!(seqlens.len(), 4); // 4 windows
    }

    #[test]
    fn test_invert_permutation() {
        let perm = vec![2, 0, 3, 1];
        let inv = invert_permutation(&perm);
        assert_eq!(inv, vec![1, 3, 0, 2]);
        // Verify round-trip
        for i in 0..perm.len() {
            assert_eq!(inv[perm[i]], i);
        }
    }

    // ── Vision Component Tests ──────────────────────────────────────────

    #[test]
    fn test_patch_embed() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let pe = Qwen25VLPatchEmbed::new(&cfg, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (4, cfg.patch_input_dim()), &device).unwrap();
        let output = pe.forward(&input).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_mlp() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mlp = Qwen25VLVisionMLP::new(cfg.embed_dim, cfg.intermediate_size, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = mlp.forward(&x).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_attention() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let attn = Qwen25VLVisionAttention::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = attn.forward(&x, None, None).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_vision_block() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = Qwen25VLVisionBlock::new(&cfg, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (4, cfg.embed_dim), &device).unwrap();
        let output = block.forward(&x, None, None).unwrap();
        assert_eq!(output.dims(), &[4, cfg.embed_dim]);
    }

    #[test]
    fn test_patch_merger() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let merger = Qwen25VLPatchMerger::new(&cfg, cfg.out_hidden_size, vb).unwrap();

        // 4x4 grid -> 2x2 merged = 4 tokens
        let x = Tensor::randn(0.0f32, 1.0, (16, cfg.embed_dim), &device).unwrap();
        let output = merger.forward(&x, 4, 4).unwrap();
        assert_eq!(output.dims(), &[4, cfg.out_hidden_size]);
    }

    #[test]
    fn test_vision_transformer() {
        let device = Device::Cpu;
        let cfg = test_vision_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vit = Qwen25VisionTransformer::new(&cfg, vb).unwrap();

        // 4x4 grid
        let patches = Tensor::randn(0.0f32, 1.0, (16, cfg.patch_input_dim()), &device).unwrap();
        let output = vit.forward(&patches, 4, 4).unwrap();

        // 4x4 grid with merge_size=2 -> 2x2 = 4 merged tokens
        assert_eq!(output.dims(), &[4, cfg.out_hidden_size]);
    }

    // ── Full Model Tests ────────────────────────────────────────────────

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen25VLForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_model_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen25VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
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

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_model_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen25VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
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
    fn test_compute_position_ids_text_only() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen25VLForConditionalGeneration::new(&cfg, vb).unwrap();

        let input_ids = vec![1, 2, 3, 4, 5];
        let pos_ids = model.compute_position_ids(&input_ids, None).unwrap();

        assert_eq!(pos_ids.dims(), &[3, 5]);
        let pos_data: Vec<u32> = pos_ids.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(&pos_data[0..5], &[0, 1, 2, 3, 4]);
        assert_eq!(&pos_data[5..10], &[0, 1, 2, 3, 4]);
        assert_eq!(&pos_data[10..15], &[0, 1, 2, 3, 4]);
    }
}
