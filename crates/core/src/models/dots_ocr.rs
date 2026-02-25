//! DotsOCR vision-language model.
//!
//! Architecture:
//! - Vision encoder: DotsVisionTransformer (Conv2d patch embed + RoPE +
//!   SwiGLU blocks + PatchMerger) at `vision_tower.*`
//! - Language model: Qwen2ForCausalLM at `model.*` / `lm_head.*` (standard HF paths)
//!
//! Weight paths (HF checkpoint; vLLM remaps `.attn.qkv_proj.` → `.attn.qkv.`,
//! `.attn.out_proj.` → `.attn.proj.`, `model.` → `language_model.model.`,
//! `lm_head.` → `language_model.lm_head.`):
//!
//! ```text
//! vision_tower.patch_embed.patchifier.proj.{weight,bias}
//! vision_tower.patch_embed.patchifier.norm.weight
//! vision_tower.blocks.{i}.{norm1,norm2}.weight
//! vision_tower.blocks.{i}.attn.qkv_proj.{weight,bias}   (our code: qkv_proj)
//! vision_tower.blocks.{i}.attn.out_proj.{weight,bias}   (our code: out_proj)
//! vision_tower.blocks.{i}.mlp.{fc1,fc3,fc2}.{weight,bias}
//! vision_tower.merger.ln_q.weight
//! vision_tower.merger.mlp.{0,2}.{weight,bias}
//! model.{embed_tokens,layers.*,norm}.*
//! lm_head.weight
//! ```
//!
//! Reference: reference/vllm/vllm/model_executor/models/dots_ocr.py

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d_no_bias, layer_norm, linear, linear_no_bias, Conv2dConfig, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;

// ─── DotsVisionConfig ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DotsVisionConfig {
    embed_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_heads: usize,
    num_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    rms_norm_eps: f64,
    use_bias: bool,
    post_norm: bool,
}

impl DotsVisionConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let v = cfg.extra.get("vision_config").cloned().unwrap_or_default();
        let get_u = |k: &str, d: u64| v.get(k).and_then(|x| x.as_u64()).unwrap_or(d) as usize;
        let get_b = |k: &str, d: bool| v.get(k).and_then(|x| x.as_bool()).unwrap_or(d);
        let get_f = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        Self {
            embed_dim: get_u("embed_dim", 1536),
            hidden_size: get_u("hidden_size", 1536),
            intermediate_size: get_u("intermediate_size", 4224),
            num_hidden_layers: get_u("num_hidden_layers", 42),
            num_heads: get_u("num_attention_heads", 12),
            num_channels: get_u("num_channels", 3),
            patch_size: get_u("patch_size", 14),
            temporal_patch_size: get_u("temporal_patch_size", 1),
            spatial_merge_size: get_u("spatial_merge_size", 2),
            rms_norm_eps: get_f("rms_norm_eps", 1e-5),
            use_bias: get_b("use_bias", false),
            post_norm: get_b("post_norm", true),
        }
    }

    fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

// ─── Vision Rotary Embedding ─────────────────────────────────────────────────

/// Computes outer-product rotary frequencies for DotsOCR vision positions.
///
/// `dim = head_dim / 2`; `inv_freq` has shape `[dim / 2]`.
/// `forward(seqlen)` returns `[seqlen, dim / 2]`.
#[allow(dead_code)]
struct DotsVisionRotaryEmbedding {
    inv_freq: Tensor,
}

#[allow(dead_code)]
impl DotsVisionRotaryEmbedding {
    fn new(dim: usize, theta: f64, device: &Device) -> Result<Self> {
        // inv_freq[i] = 1 / theta^(2i/dim) for i in 0..dim//2
        let half = dim / 2;
        let inv_freq_vals: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf((i * 2) as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq_vals, (half,), device)?;
        Ok(Self { inv_freq })
    }

    /// Returns `[seqlen, dim/2]` — the raw outer-product frequencies.
    fn forward(&self, seqlen: usize) -> Result<Tensor> {
        let device = self.inv_freq.device();
        let seq: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
        let seq = Tensor::from_vec(seq, (seqlen,), device)?;
        // outer product [seqlen, 1] @ [1, dim/2] → [seqlen, dim/2]
        seq.unsqueeze(1)?
            .broadcast_mul(&self.inv_freq.unsqueeze(0)?)
    }

    /// Build rotary embeddings for the given patch grid `(T, H, W)` entries.
    ///
    /// Returns `[total_patches, head_dim/2]` raw frequencies (before cos/sin).
    /// Spatial positions (h, w) are interleaved such that the final vector is
    /// `[h_freq_0, w_freq_0, h_freq_1, w_freq_1, ...]` via flatten.
    fn pos_emb_for_grids(&self, grids: &[(u32, u32, u32)], merge: usize) -> Result<Tensor> {
        let device = self.inv_freq.device();
        let max_grid = grids
            .iter()
            .flat_map(|(_, h, w)| [*h as usize, *w as usize])
            .max()
            .unwrap_or(1);

        let freq_full = self.forward(max_grid)?; // [max_grid, dim/2]

        let mut parts = Vec::new();
        for &(t, h, w) in grids {
            let (t, h, w) = (t as usize, h as usize, w as usize);
            // Build merged h/w position IDs: for each merged super-patch, record the
            // corresponding (h_id, w_id) by permuting the tile layout.
            let mh = h / merge;
            let mw = w / merge;

            // h_ids: arange(h) expanded along w, permuted by merge blocks, flattened
            let mut h_ids: Vec<u32> = Vec::with_capacity(mh * mw * merge * merge);
            let mut w_ids: Vec<u32> = Vec::with_capacity(mh * mw * merge * merge);
            for bh in 0..mh {
                for bw in 0..mw {
                    for mii in 0..merge {
                        for mjj in 0..merge {
                            h_ids.push((bh * merge + mii) as u32);
                            w_ids.push((bw * merge + mjj) as u32);
                        }
                    }
                }
            }
            let total = h_ids.len();
            let h_ids = Tensor::from_vec(h_ids, (total,), device)?;
            let w_ids = Tensor::from_vec(w_ids, (total,), device)?;

            let h_freq = freq_full.index_select(&h_ids, 0)?; // [total, dim/2]
            let w_freq = freq_full.index_select(&w_ids, 0)?; // [total, dim/2]

            // Interleave: [total, 2, dim/2] → [total, dim]
            let interleaved = Tensor::stack(&[h_freq, w_freq], 1)?
                .flatten_from(1)?
                .contiguous()?; // [total, dim]

            // Repeat for T frames
            if t > 1 {
                let mut repeated = Vec::new();
                for _ in 0..t {
                    repeated.push(interleaved.clone());
                }
                parts.push(Tensor::cat(&repeated, 0)?);
            } else {
                parts.push(interleaved);
            }
        }
        Tensor::cat(&parts, 0)
    }
}

// ─── Patch Embedding ─────────────────────────────────────────────────────────

/// Conv2d-based patch embedding + RmsNorm.
///
/// Input: `[N, C * T * ps * ps]` flat patches (vLLM pre-processing)
/// Output: `[N, embed_dim]`
#[allow(dead_code)]
struct DotsPatchEmbed {
    proj: candle_nn::Conv2d,
    norm: RmsNorm,
    num_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
}

#[allow(dead_code)]
impl DotsPatchEmbed {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        // NOTE: checkpoint stores Conv2d weight [embed_dim, C, ps, ps]
        let proj = conv2d_no_bias(
            cfg.num_channels,
            cfg.embed_dim,
            cfg.patch_size,
            conv_cfg,
            vb.pp("proj"),
        )?;
        let norm = rms_norm(cfg.embed_dim, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            proj,
            norm,
            num_channels: cfg.num_channels,
            temporal_patch_size: cfg.temporal_patch_size,
            patch_size: cfg.patch_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let num_patches = x.dim(0)?;
        // Reshape to [N, C, T, ps, ps] then take T=0 → [N, C, ps, ps]
        let x = x
            .reshape((
                num_patches,
                self.num_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ))?
            .narrow(2, 0, 1)?
            .squeeze(2)?; // [N, C, ps, ps]
        let x = self.proj.forward(&x)?; // [N, embed_dim, 1, 1]
        let x = x.flatten_from(1)?; // [N, embed_dim]
        self.norm.forward(&x)
    }
}

// ─── Vision Attention ─────────────────────────────────────────────────────────

/// DotsOCR vision self-attention with optional bias and 2D RoPE.
#[allow(dead_code)]
struct DotsVisionAttention {
    qkv: candle_nn::Linear,
    proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl DotsVisionAttention {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        // HF weight names: qkv_proj / out_proj (before vLLM mapper renames them)
        let qkv = if cfg.use_bias {
            linear(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv_proj"))?
        } else {
            linear_no_bias(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv_proj"))?
        };
        let proj = if cfg.use_bias {
            linear(cfg.embed_dim, cfg.embed_dim, vb.pp("out_proj"))?
        } else {
            linear_no_bias(cfg.embed_dim, cfg.embed_dim, vb.pp("out_proj"))?
        };
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward with precomputed rotary embeddings.
    ///
    /// * `x` — `[S, embed_dim]`
    /// * `rotary_pos_emb` — `[S, head_dim]` raw outer-product frequencies
    fn forward(&self, x: &Tensor, rotary_pos_emb: &Tensor) -> Result<Tensor> {
        let (num_tokens, _) = x.dims2()?;

        let qkv = self.qkv.forward(x)?; // [S, 3 * embed_dim]

        let q = qkv.narrow(1, 0, self.num_heads * self.head_dim)?.reshape((
            num_tokens,
            self.num_heads,
            self.head_dim,
        ))?;
        let k = qkv
            .narrow(
                1,
                self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
            )?
            .reshape((num_tokens, self.num_heads, self.head_dim))?;
        let v = qkv
            .narrow(
                1,
                2 * self.num_heads * self.head_dim,
                self.num_heads * self.head_dim,
            )?
            .reshape((num_tokens, self.num_heads, self.head_dim))?;

        // Apply RoPE (neox-style) to q and k using precomputed frequencies
        let (q, k) = self.apply_rope(q, k, rotary_pos_emb)?;

        // Scaled dot-product attention: [1, H, S, D]
        let q = q.transpose(0, 1)?.unsqueeze(0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [1, H, S, D]

        // [1, H, S, D] → [S, embed_dim]
        let out = out
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_tokens, self.num_heads * self.head_dim))?;

        self.proj.forward(&out)
    }

    /// Apply NeoxStyle rotary position embeddings.
    ///
    /// `rot_emb`: `[S, head_dim/2]` raw frequencies (matches `ApplyRotaryEmb` contract).
    fn apply_rope(&self, q: Tensor, k: Tensor, rot_emb: &Tensor) -> Result<(Tensor, Tensor)> {
        // [S, head_dim/2] → [S, 1, head_dim/2] for broadcast with [S, H, head_dim/2]
        let cos = rot_emb.cos()?.unsqueeze(1)?;
        let sin = rot_emb.sin()?.unsqueeze(1)?;

        let q = neox_rotate(&q, &cos, &sin)?;
        let k = neox_rotate(&k, &cos, &sin)?;
        Ok((q, k))
    }
}

/// NeoxStyle rotation matching vLLM's `ApplyRotaryEmb`.
///
/// `cos`, `sin`: `[S, 1, head_dim/2]`; `x`: `[S, H, head_dim]`.
/// Splits x into two halves, applies:  o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin.
#[allow(dead_code)]
fn neox_rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(2)? / 2;
    let x1 = x.narrow(2, 0, half)?; // [S, H, half]
    let x2 = x.narrow(2, half, half)?; // [S, H, half]
    let o1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let o2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    Tensor::cat(&[o1, o2], 2)
}

// ─── SwiGLU FFN ──────────────────────────────────────────────────────────────

/// DotsOCR vision SwiGLU FFN: `fc2(SiLU(fc1(x)) * fc3(x))`.
///
/// Weights in HF checkpoint: `fc1.weight`, `fc3.weight`, `fc2.weight`.
#[allow(dead_code)]
struct DotsSwiGLUFFN {
    fc1: candle_nn::Linear,
    fc3: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

#[allow(dead_code)]
impl DotsSwiGLUFFN {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        let make_fc = |in_: usize, out_: usize, vb_: VarBuilder| -> Result<candle_nn::Linear> {
            if cfg.use_bias {
                linear(in_, out_, vb_)
            } else {
                linear_no_bias(in_, out_, vb_)
            }
        };
        let fc1 = make_fc(cfg.embed_dim, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc3 = make_fc(cfg.embed_dim, cfg.intermediate_size, vb.pp("fc3"))?;
        let fc2 = make_fc(cfg.intermediate_size, cfg.embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc3, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.fc1.forward(x)?.silu()?;
        let u = self.fc3.forward(x)?;
        let h = (g * u)?;
        self.fc2.forward(&h)
    }
}

// ─── Vision Block ─────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct DotsVisionBlock {
    norm1: RmsNorm,
    attn: DotsVisionAttention,
    norm2: RmsNorm,
    mlp: DotsSwiGLUFFN,
}

#[allow(dead_code)]
impl DotsVisionBlock {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm(cfg.embed_dim, cfg.rms_norm_eps, vb.pp("norm1"))?,
            attn: DotsVisionAttention::new(cfg, vb.pp("attn"))?,
            norm2: rms_norm(cfg.embed_dim, cfg.rms_norm_eps, vb.pp("norm2"))?,
            mlp: DotsSwiGLUFFN::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rot_emb: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = (residual + self.attn.forward(&self.norm1.forward(x)?, rot_emb)?)?;
        let x = (&x + self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

// ─── Patch Merger ─────────────────────────────────────────────────────────────

/// Spatial patch merging: `LayerNorm → view(merge²) → Linear(GELU) → Linear`.
///
/// Reduces `[total_tokens, embed_dim]` → `[total_tokens / merge², hidden_size]`.
#[allow(dead_code)]
struct DotsPatchMerger {
    ln_q: candle_nn::LayerNorm,
    mlp_0: candle_nn::Linear,
    mlp_2: candle_nn::Linear,
    spatial_merge_size: usize,
}

#[allow(dead_code)]
impl DotsPatchMerger {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.embed_dim * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Ok(Self {
            ln_q: layer_norm(cfg.embed_dim, 1e-6, vb.pp("ln_q"))?,
            mlp_0: linear(hidden, hidden, vb.pp("mlp").pp(0))?,
            mlp_2: linear(hidden, cfg.hidden_size, vb.pp("mlp").pp(2))?,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let total = x.dim(0)?;
        let embed = x.dim(1)?;
        let merge_sq = self.spatial_merge_size * self.spatial_merge_size;
        let x = self.ln_q.forward(x)?;
        let x = x.reshape((total / merge_sq, embed * merge_sq))?;
        let x = self.mlp_0.forward(&x)?.gelu()?;
        self.mlp_2.forward(&x)
    }
}

// ─── Vision Transformer ───────────────────────────────────────────────────────

#[allow(dead_code)]
struct DotsVisionTransformer {
    patch_embed: DotsPatchEmbed,
    blocks: Vec<DotsVisionBlock>,
    post_trunk_norm: Option<RmsNorm>,
    merger: DotsPatchMerger,
    rotary_emb: DotsVisionRotaryEmbedding,
    config: DotsVisionConfig,
}

#[allow(dead_code)]
impl DotsVisionTransformer {
    fn new(cfg: &DotsVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = DotsPatchEmbed::new(cfg, vb.pp("patch_embed").pp("patchifier"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(DotsVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let post_trunk_norm = if cfg.post_norm {
            Some(rms_norm(
                cfg.embed_dim,
                cfg.rms_norm_eps,
                vb.pp("post_trunk_norm"),
            )?)
        } else {
            None
        };

        let merger = DotsPatchMerger::new(cfg, vb.pp("merger"))?;

        // head_dim / 2 is the RoPE dimension
        let rope_dim = cfg.head_dim() / 2;
        let rotary_emb = DotsVisionRotaryEmbedding::new(rope_dim, 10000.0, vb.device())?;

        Ok(Self {
            patch_embed,
            blocks,
            post_trunk_norm,
            merger,
            rotary_emb,
            config: cfg.clone(),
        })
    }

    /// Encode pixel patches for a batch of images described by `grid_thw`.
    ///
    /// * `pixel_values` — `[total_patches, C * T * ps * ps]`
    /// * `grid_thw` — `(T, H, W)` per image
    ///
    /// Returns `[merged_tokens, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor, grid_thw: &[(u32, u32, u32)]) -> Result<Tensor> {
        // Patch embedding
        let mut x = self.patch_embed.forward(pixel_values)?; // [N, embed_dim]

        // Positional rotary embeddings: [N, head_dim]
        let rot_emb = self
            .rotary_emb
            .pos_emb_for_grids(grid_thw, self.config.spatial_merge_size)?;
        let rot_emb = rot_emb.to_dtype(x.dtype())?;

        // Vision transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, &rot_emb)?;
        }

        if let Some(ref norm) = self.post_trunk_norm {
            x = norm.forward(&x)?;
        }

        self.merger.forward(&x)
    }
}

// ─── DotsOCRForCausalLM ───────────────────────────────────────────────────────

/// DotsOCR vision-language model: DotsVisionTransformer + Qwen2 LLM.
pub struct DotsOCRForCausalLM {
    #[allow(dead_code)]
    vision_tower: DotsVisionTransformer,
    language_model: Qwen2ForCausalLM,
    #[allow(dead_code)]
    image_token_id: u32,
    device: Device,
    dtype: DType,
}

impl DotsOCRForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_cfg = DotsVisionConfig::from_model_config(cfg);

        let vision_tower = DotsVisionTransformer::new(&vis_cfg, vb.pp("vision_tower"))?;

        // Language model loads from `model.*` and `lm_head.*` (standard HF paths)
        let language_model = Qwen2ForCausalLM::new(cfg, vb.clone())?;

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151665) as u32;

        Ok(Self {
            vision_tower,
            language_model,
            image_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Replace image-token positions with pre-computed vision embeddings.
    fn merge_vision_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_b, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            let img_emb = processed_image
                .embedding
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;
            for (i, emb) in img_emb.iter().enumerate() {
                let pos = start_pos + i;
                if pos < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for DotsOCRForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.language_model.forward(
            input_ids,
            seqlen_offset,
            kv_cache,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.language_model
            .forward_decode_batch(input_ids, sequences, kv_cache)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        mm_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Embed text tokens
        let text_emb = self.language_model.embed_text(input_ids)?;

        // Splice in vision features (pre-projected into LLM space)
        let embeddings = if let Some(mm) = mm_inputs {
            self.merge_vision_embeddings(&text_emb, mm)?
        } else {
            text_emb
        };

        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::engine::ModelForward;
    use crate::kv_cache::{BlockTable, CacheConfig, KVCacheDtype, KVCacheManager};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".into(),
            json!({
                "embed_dim": 32,
                "hidden_size": 64,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "patch_size": 4,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "rms_norm_eps": 1e-5,
                "use_bias": false,
                "post_norm": false
            }),
        );
        extra.insert("image_token_id".into(), json!(151665u64));
        ModelConfig {
            architectures: vec!["DotsOCRForCausalLM".to_string()],
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 4,
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
    fn test_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DotsOCRForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_forward() {
        use crate::engine::ModelForward;
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DotsOCRForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let bt = BlockTable::from_block_ids(vec![0, 1], 0);
        let sm: Vec<usize> = (0..4).collect();
        let ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let logits = model.forward(&ids, 0, &mut kv, &bt, &sm).unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        use crate::engine::DecodeSequenceMetadata;
        use crate::engine::ModelForward;
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DotsOCRForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let seqs = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 4,
                block_ids: vec![0],
                slot_mapping: vec![4],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 2,
                block_ids: vec![1],
                slot_mapping: vec![2],
            },
        ];
        let ids = Tensor::zeros((2, 1), DType::U32, &device).unwrap();
        let logits = model.forward_decode_batch(&ids, &seqs, &mut kv).unwrap();
        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_multimodal_text_only() {
        use crate::engine::ModelForward;
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DotsOCRForCausalLM::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let bt = BlockTable::from_block_ids(vec![0, 1], 0);
        let sm: Vec<usize> = (0..4).collect();
        let ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let logits = model
            .forward_multimodal(&ids, None, 0, &mut kv, &bt, &sm)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_vision_encoder() {
        // Smoke-test the vision encoder with dummy patches.
        // embed_dim=32, patch_size=4, num_channels=3, T=1, spatial_merge_size=2
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vis_cfg = DotsVisionConfig::from_model_config(&cfg);
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vision_tower = DotsVisionTransformer::new(&vis_cfg, vb.pp("vision_tower")).unwrap();

        // 4 patches, each = C * T * ps * ps = 3 * 1 * 4 * 4 = 48 values
        let patch_dim = vis_cfg.num_channels
            * vis_cfg.temporal_patch_size
            * vis_cfg.patch_size
            * vis_cfg.patch_size;
        // grid_thw = (T=1, H=4, W=4) — 16 patches, merge_size=2 → 4 merged tokens
        let num_patches = 16;
        let patches = Tensor::zeros((num_patches, patch_dim), DType::F32, &device).unwrap();
        let grids = vec![(1u32, 4u32, 4u32)];
        let out = vision_tower.forward(&patches, &grids).unwrap();
        // merged tokens = 16 / (2*2) = 4
        assert_eq!(out.dim(0).unwrap(), 4);
        assert_eq!(out.dim(1).unwrap(), vis_cfg.hidden_size);
    }
}
