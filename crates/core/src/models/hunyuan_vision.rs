//! HunYuan-VL vision-language model.
//!
//! Architecture: `HunYuanVisionTransformer` (vision encoder) + HunYuan Dense/MoE LLM.
//!
//! Vision encoder:
//! - `HunYuanVisionPatchEmbed`: Conv2d(patch_size×patch_size → hidden_size, bias=True) +
//!   bilinear-interpolated position embedding per image grid.
//! - N `HunYuanVisionBlock` layers: LayerNorm + QKV attention (bias=True) + LayerNorm + MLP
//!   (dense_h_to_4h GELU + dense_4h_to_h, both bias=True). Per-image parallel processing.
//! - `HunYuanVisionPatchMerger` (perceive): before_rms (RMSNorm) + Conv2d(2)×GELU×Conv2d(1)
//!   downsample + `image_newline` column separator + Linear projector + begin/end tokens +
//!   after_rms (RMSNorm). Outputs `[tokens_per_image, out_hidden_size]` per image.
//!
//! Weight paths (HF checkpoint):
//! - `visual.*`        — vision transformer (vit.vit.* or vit.* → visual.*)
//! - `language_model.*` — HunYuan dense/MoE LLM (model.* → language_model.model.*)
//!
//! Reference: `vllm/model_executor/models/hunyuan_vision.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::ProcessGroup;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm};
use crate::multimodal::MultimodalInputs;

use super::hunyuan::{HunYuanDenseV1ForCausalLM, HunYuanMoEV1ForCausalLM};
use super::tp_layers::TpContext;

// ─── Vision config ────────────────────────────────────────────────────────────

/// Vision configuration parsed from `vision_config` in the model JSON.
#[derive(Debug, Clone)]
pub(crate) struct HunYuanVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub max_image_size: usize,
    pub norm_eps: f64,
}

impl HunYuanVisionConfig {
    /// Parse from the `vision_config` sub-object in `ModelConfig::extra`.
    pub(crate) fn from_model_config(cfg: &ModelConfig) -> Self {
        let v = cfg.extra.get("vision_config");
        let get_usize = |key, default: usize| -> usize {
            v.and_then(|o| o.get(key))
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key, default: f64| -> f64 {
            v.and_then(|o| o.get(key))
                .and_then(|x| x.as_f64())
                .unwrap_or(default)
        };

        Self {
            hidden_size: get_usize("hidden_size", 1152),
            intermediate_size: get_usize("intermediate_size", 4304),
            num_hidden_layers: get_usize("num_hidden_layers", 27),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_channels: get_usize("num_channels", 3),
            patch_size: get_usize("patch_size", 16),
            spatial_merge_size: get_usize("spatial_merge_size", 2),
            out_hidden_size: get_usize("out_hidden_size", 4096),
            max_image_size: get_usize("max_image_size", 2048),
            norm_eps: get_f64("rms_norm_eps", 1e-5),
        }
    }
}

// ─── Vision patch embedding ───────────────────────────────────────────────────

/// Bilinear interpolation of position embeddings to a target (h × w) grid.
///
/// Source is a square `src_size × src_size` grid; target may be non-square.
/// Replicates `F.interpolate(..., mode='bilinear', align_corners=False, antialias=False)`.
fn bilinear_interp_pos_emb_2d(
    pos_emb: &Tensor, // [src_h * src_w, D]
    src_h: usize,
    src_w: usize,
    tgt_h: usize,
    tgt_w: usize,
) -> Result<Tensor> {
    if tgt_h == src_h && tgt_w == src_w {
        return Ok(pos_emb.clone());
    }

    let d = pos_emb.dim(1)?;
    let pe = pos_emb.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let pe_data = pe.to_vec2::<f32>()?;

    let mut out = vec![0f32; tgt_h * tgt_w * d];
    for ty in 0..tgt_h {
        for tx in 0..tgt_w {
            let sy = ((ty as f32 + 0.5) * src_h as f32 / tgt_h as f32 - 0.5)
                .clamp(0.0, (src_h - 1) as f32);
            let sx = ((tx as f32 + 0.5) * src_w as f32 / tgt_w as f32 - 0.5)
                .clamp(0.0, (src_w - 1) as f32);
            let sy0 = sy.floor() as usize;
            let sy1 = (sy0 + 1).min(src_h - 1);
            let sx0 = sx.floor() as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let wy1 = sy - sy0 as f32;
            let wy0 = 1.0 - wy1;
            let wx1 = sx - sx0 as f32;
            let wx0 = 1.0 - wx1;
            let dst = (ty * tgt_w + tx) * d;
            for k in 0..d {
                let v00 = pe_data[sy0 * src_w + sx0][k];
                let v01 = pe_data[sy0 * src_w + sx1][k];
                let v10 = pe_data[sy1 * src_w + sx0][k];
                let v11 = pe_data[sy1 * src_w + sx1][k];
                out[dst + k] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
            }
        }
    }

    let result = Tensor::from_vec(out, (tgt_h * tgt_w, d), pos_emb.device())?;
    result.to_dtype(pos_emb.dtype())
}

struct HunYuanVisionPatchEmbed {
    /// Conv2d(num_channels → hidden_size, kernel=patch_size, stride=patch_size, bias=True)
    patch_embedding: candle_nn::Conv2d,
    /// Learnable position embedding: [num_positions, hidden_size]
    /// num_positions = (max_image_size / patch_size)^2 + 1 (index 0 = CLS, skip it)
    position_embedding: Embedding,
    /// = max_image_size / patch_size = sqrt(num_positions - 1)
    position_edge: usize,
}

impl HunYuanVisionPatchEmbed {
    fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_patches_per_edge = vcfg.max_image_size / vcfg.patch_size;
        // +1 for a CLS-like token that we skip at forward time.
        let num_positions = num_patches_per_edge * num_patches_per_edge + 1;

        let patch_embedding = candle_nn::conv2d(
            vcfg.num_channels,
            vcfg.hidden_size,
            vcfg.patch_size,
            candle_nn::Conv2dConfig {
                stride: vcfg.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let position_embedding =
            embedding(num_positions, vcfg.hidden_size, vb.pp("position_embedding"))?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            position_edge: num_patches_per_edge,
        })
    }

    /// Forward through patch embed + interpolated position embed.
    ///
    /// * `pixel_values` – `[num_patches_total, C * patch_size^2]` flat (one row per patch)
    /// * `grid_thw`     – list of `(T, H, W)` grids, one per image; H and W are in patches
    ///
    /// Returns `[1, num_patches_total, hidden_size]`.
    fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Tensor> {
        let num_patches = pixel_values.dim(0)?;
        let patch_size = self.patch_embedding.weight().dim(2)?;
        let num_channels = self.patch_embedding.weight().dim(1)?;

        // Reshape flat patches → [num_patches, C, ps, ps] for the Conv2d.
        let pv = pixel_values
            .reshape((num_patches, num_channels, patch_size, patch_size))?
            .to_dtype(self.position_embedding.embeddings().dtype())?;

        // Patch embedding: each [C, ps, ps] → [hidden, 1, 1] → squeeze → [hidden].
        let patch_embeds = self.patch_embedding.forward(&pv)?; // [N, H, 1, 1]
        let patch_embeds = patch_embeds.squeeze(3)?.squeeze(2)?.unsqueeze(0)?; // [1, N, H]

        // Build per-image position embeddings via bilinear interpolation and sum.
        // position_embedding.weight[1:, :] gives the grid (skip CLS at index 0).
        let pe_grid = self.position_embedding.embeddings().narrow(
            0,
            1,
            self.position_edge * self.position_edge,
        )?; // [edge^2, H]

        let mut pos_embed_parts: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        for &[_, h, w] in grid_thw {
            let pe =
                bilinear_interp_pos_emb_2d(&pe_grid, self.position_edge, self.position_edge, h, w)?; // [h*w, H]
            pos_embed_parts.push(pe.unsqueeze(0)?); // [1, h*w, H]
        }
        let pos_embed = Tensor::cat(&pos_embed_parts, 1)?; // [1, total_patches, H]

        patch_embeds.broadcast_add(&pos_embed.to_dtype(patch_embeds.dtype())?)
    }
}

// ─── Vision attention ─────────────────────────────────────────────────────────

/// Standard full-attention for the vision encoder (no RoPE, no causal mask).
struct HunYuanVisionAttention {
    qkv: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl HunYuanVisionAttention {
    fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = vcfg.hidden_size;
        let num_heads = vcfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        // QKV fused with bias; in weights split as q/k/v in the checkpoint.
        // Python stacks them → [3*H, H]; we load as one weight.
        let qkv = linear(embed_dim, 3 * embed_dim, vb.pp("qkv"))?;
        let o_proj = candle_nn::linear_no_bias(embed_dim, embed_dim, vb.pp("o_proj"))?;
        Ok(Self {
            qkv,
            o_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[batch, seq_len, embed_dim]`; returns `[batch, seq_len, embed_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // [b, t, 3*D]
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

        // Reshape to [b, heads, t, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled dot-product attention (no mask — encoder style).
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(self.scale, 0.0)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v)?; // [b, heads, t, head_dim]

        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

// ─── Vision MLP ───────────────────────────────────────────────────────────────

struct HunYuanVisionMlp {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
}

impl HunYuanVisionMlp {
    fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: linear(
                vcfg.hidden_size,
                vcfg.intermediate_size,
                vb.pp("dense_h_to_4h"),
            )?,
            dense_4h_to_h: linear(
                vcfg.intermediate_size,
                vcfg.hidden_size,
                vb.pp("dense_4h_to_h"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dense_h_to_4h.forward(x)?.gelu_erf()?;
        self.dense_4h_to_h.forward(&x)
    }
}

// ─── Vision block ─────────────────────────────────────────────────────────────

struct HunYuanVisionBlock {
    input_layernorm: LayerNorm,
    self_attn: HunYuanVisionAttention,
    post_attention_layernorm: LayerNorm,
    mlp: HunYuanVisionMlp,
}

impl HunYuanVisionBlock {
    fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm_eps = 1e-6; // LayerNorm default per Python code (partial(nn.LayerNorm, eps=1e-6))
        Ok(Self {
            input_layernorm: layer_norm(vcfg.hidden_size, norm_eps, vb.pp("input_layernorm"))?,
            self_attn: HunYuanVisionAttention::new(vcfg, vb.pp("self_attn"))?,
            post_attention_layernorm: layer_norm(
                vcfg.hidden_size,
                norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: HunYuanVisionMlp::new(vcfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.input_layernorm.forward(x)?;
        let attn = self.self_attn.forward(&normed)?;
        let x = (x + attn)?;
        let normed2 = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        x + mlp_out
    }
}

// ─── Vision patch merger (perceive) ──────────────────────────────────────────

/// Downsamples and projects per-image patch features to LLM hidden size.
///
/// Input per image: `[1, h*w, hidden_size]`
/// Output per image: `[1, (h//s)*(w//s+1)+2, out_hidden_size]`
/// where `s = spatial_merge_size`.
struct HunYuanVisionPatchMerger {
    before_rms: RmsNorm,
    proj_conv1: candle_nn::Conv2d, // hidden → hidden*2, kernel=s, stride=s
    proj_conv2: candle_nn::Conv2d, // hidden*2 → hidden*4, kernel=1, stride=1
    mlp: Linear,                   // hidden*4 → out_hidden
    image_newline: Tensor,         // [hidden*4]
    image_begin: Tensor,           // [out_hidden]
    image_end: Tensor,             // [out_hidden]
    after_rms: RmsNorm,
}

impl HunYuanVisionPatchMerger {
    fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        let s = vcfg.spatial_merge_size;
        let in_ch = vcfg.hidden_size;
        let out_ch = vcfg.out_hidden_size;

        let before_rms = rms_norm(in_ch, vcfg.norm_eps, vb.pp("before_rms"))?;

        // proj is an nn.Sequential with indices 0 (Conv2d), 1 (GELU — no weights), 2 (Conv2d)
        let proj_conv1 = candle_nn::conv2d(
            in_ch,
            in_ch * 2,
            s,
            candle_nn::Conv2dConfig {
                stride: s,
                ..Default::default()
            },
            vb.pp("proj").pp("0"),
        )?;
        let proj_conv2 = candle_nn::conv2d(
            in_ch * 2,
            in_ch * 4,
            1,
            Default::default(),
            vb.pp("proj").pp("2"),
        )?;

        let mlp = linear(in_ch * 4, out_ch, vb.pp("mlp"))?;

        let image_newline = vb.get(in_ch * 4, "image_newline")?;
        let image_begin = vb.get(out_ch, "image_begin")?;
        let image_end = vb.get(out_ch, "image_end")?;

        let after_rms = rms_norm(out_ch, vcfg.norm_eps, vb.pp("after_rms"))?;

        Ok(Self {
            before_rms,
            proj_conv1,
            proj_conv2,
            mlp,
            image_newline,
            image_begin,
            image_end,
            after_rms,
        })
    }

    /// Forward for one image.
    ///
    /// * `x`    – `[1, h*w, hidden_size]`
    /// * `h, w` – patch grid dimensions before merge
    ///
    /// Returns `[1, (h//s)*(w//s+1)+2, out_hidden]`.
    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let x = self.before_rms.forward(x)?; // [1, h*w, C]
        let dtype = x.dtype();
        let in_ch = x.dim(2)?;

        // Reshape to [1, C, h, w] for Conv2d.
        let x = x.permute((0, 2, 1))?.reshape((1, in_ch, h, w))?;

        // Downsample with stride-s convolutions.
        let x = self.proj_conv1.forward(&x)?.gelu_erf()?;
        let x = self.proj_conv2.forward(&x)?; // [1, C*4, h//s, w//s]

        let (b, c, lh, lw) = x.dims4()?;
        let x = x.to_dtype(dtype)?;

        // Append image_newline column: shape [1, C*4, lh, 1]
        let newline = self
            .image_newline
            .reshape((1, c, 1, 1))?
            .expand((b, c, lh, 1))?
            .to_dtype(dtype)?;
        let x = Tensor::cat(&[&x, &newline], 3)?; // [1, C*4, lh, lw+1]

        // Flatten spatial → [1, lh*(lw+1), C*4]
        let x = x.reshape((b, c, lh * (lw + 1)))?.permute((0, 2, 1))?;

        // Project to out_hidden
        let x = self.mlp.forward(&x)?; // [1, lh*(lw+1), out_hidden]

        // Prepend begin token and append end token.
        let begin = self
            .image_begin
            .reshape((1, 1, x.dim(2)?))?
            .expand((b, 1, x.dim(2)?))?
            .to_dtype(dtype)?;
        let end = self
            .image_end
            .reshape((1, 1, x.dim(2)?))?
            .expand((b, 1, x.dim(2)?))?
            .to_dtype(dtype)?;
        let x = Tensor::cat(&[&begin, &x, &end], 1)?; // [1, tokens+2, out_hidden]

        self.after_rms.forward(&x)
    }
}

// ─── HunYuanVisionTransformer ─────────────────────────────────────────────────

pub struct HunYuanVisionTransformer {
    embeddings: HunYuanVisionPatchEmbed,
    layers: Vec<HunYuanVisionBlock>,
    perceive: HunYuanVisionPatchMerger,
}

impl HunYuanVisionTransformer {
    pub(crate) fn new(vcfg: &HunYuanVisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = HunYuanVisionPatchEmbed::new(vcfg, vb.pp("embeddings"))?;
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(vcfg.num_hidden_layers);
        for i in 0..vcfg.num_hidden_layers {
            layers.push(HunYuanVisionBlock::new(vcfg, vb_layers.pp(i))?);
        }
        let perceive = HunYuanVisionPatchMerger::new(vcfg, vb.pp("perceive"))?;
        Ok(Self {
            embeddings,
            layers,
            perceive,
        })
    }

    /// Encode pixel patches into image features.
    ///
    /// * `pixel_values` – `[num_patches_total, C*ps*ps]` flat (all images concatenated)
    /// * `grid_thw`     – `(T, H, W)` patch counts per image
    ///
    /// Returns one tensor per image: `[tokens_per_image, out_hidden_size]`.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[[usize; 3]]) -> Result<Vec<Tensor>> {
        // Patch + position embeddings → [1, total_patches, hidden]
        let mut hidden = self.embeddings.forward(pixel_values, grid_thw)?;

        // Per-image layer application (no cross-image attention).
        let split_lengths: Vec<usize> = grid_thw.iter().map(|&[_, h, w]| h * w).collect();
        for layer in &self.layers {
            let parts: Result<Vec<_>> = split_lengths
                .iter()
                .scan(0usize, |offset, &len| {
                    let part = hidden.narrow(1, *offset, len);
                    *offset += len;
                    Some(part)
                })
                .map(|t| t.and_then(|p| layer.forward(&p.contiguous()?)))
                .collect();
            hidden = Tensor::cat(&parts?, 1)?;
        }

        // Apply perceive (patch merger) per image.
        let mut image_embeds: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &[_, h, w] in grid_thw {
            let len = h * w;
            let item = hidden.narrow(1, offset, len)?.contiguous()?;
            let feat = self.perceive.forward(&item, h, w)?; // [1, tokens_per_image, out_h]
            image_embeds.push(feat.squeeze(0)?); // [tokens_per_image, out_h]
            offset += len;
        }

        Ok(image_embeds)
    }
}

// ─── HunYuanVLForConditionalGeneration ───────────────────────────────────────

/// HunYuan-VL: vision encoder + language model (Dense or MoE variant).
///
/// For multimodal inference:
/// 1. Encode images with `visual` → list of `[tokens_i, out_hidden]` tensors.
/// 2. Embed text tokens with `language_model.embed_text`.
/// 3. Splice image features at image placeholder positions.
/// 4. Forward through LLM with `language_model.forward_with_embeddings`.
pub enum HunYuanVLForConditionalGeneration {
    Dense {
        visual: HunYuanVisionTransformer,
        language_model: HunYuanDenseV1ForCausalLM,
    },
    Moe {
        visual: HunYuanVisionTransformer,
        language_model: HunYuanMoEV1ForCausalLM,
    },
}

impl HunYuanVLForConditionalGeneration {
    /// Dispatch to Dense or MoE variant based on `num_experts` in config.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let is_moe = cfg
            .extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .map(|n| n > 1)
            .unwrap_or(false);
        if is_moe {
            Self::new_moe(cfg, vb)
        } else {
            Self::new_dense(cfg, vb)
        }
    }

    pub fn new_dense(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vcfg = HunYuanVisionConfig::from_model_config(cfg);
        let visual = HunYuanVisionTransformer::new(&vcfg, vb.pp("visual"))?;
        let language_model = HunYuanDenseV1ForCausalLM::new(cfg, vb.pp("language_model"))?;
        Ok(Self::Dense {
            visual,
            language_model,
        })
    }

    pub fn new_moe(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vcfg = HunYuanVisionConfig::from_model_config(cfg);
        let visual = HunYuanVisionTransformer::new(&vcfg, vb.pp("visual"))?;
        let language_model = HunYuanMoEV1ForCausalLM::new(cfg, vb.pp("language_model"))?;
        Ok(Self::Moe {
            visual,
            language_model,
        })
    }

    pub fn new_dense_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vcfg = HunYuanVisionConfig::from_model_config(cfg);
        let visual = HunYuanVisionTransformer::new(&vcfg, vb.pp("visual"))?;
        let language_model =
            HunYuanDenseV1ForCausalLM::new_with_tp(cfg, vb.pp("language_model"), pg, tp_ctx)?;
        Ok(Self::Dense {
            visual,
            language_model,
        })
    }

    /// Encode image patches to features (one tensor per image).
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[usize; 3]],
    ) -> Result<Vec<Tensor>> {
        match self {
            Self::Dense { visual, .. } | Self::Moe { visual, .. } => {
                visual.forward(pixel_values, grid_thw)
            }
        }
    }

    /// Text-only prefill forward pass.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::Dense { language_model, .. } => language_model.forward(
                input_ids,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Moe { language_model, .. } => language_model.forward(
                input_ids,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
        }
    }

    /// Multimodal prefill: embed text, splice image features, run LLM.
    ///
    /// `image_embeds` is a flat `[total_image_tokens, hidden_size]` tensor.
    /// `image_positions` contains the token indices in `input_ids` where image
    /// features should replace text embeddings.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        image_embeds: &Tensor,
        image_positions: &[usize],
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (embeddings, kv_cache_mgr, block_table, slot_mapping) = match self {
            Self::Dense { language_model, .. } => {
                let mut emb = language_model.embed_text(input_ids)?;
                // Splice image features at the specified positions.
                if !image_positions.is_empty() {
                    emb = splice_image_embeds(emb, image_embeds, image_positions)?;
                }
                (emb, kv_cache_mgr, block_table, slot_mapping)
            }
            Self::Moe { language_model, .. } => {
                let mut emb = language_model.embed_text(input_ids)?;
                if !image_positions.is_empty() {
                    emb = splice_image_embeds(emb, image_embeds, image_positions)?;
                }
                (emb, kv_cache_mgr, block_table, slot_mapping)
            }
        };

        match self {
            Self::Dense { language_model, .. } => language_model.forward_with_embeddings(
                &embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Moe { language_model, .. } => language_model.forward_with_embeddings(
                &embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Dense { language_model, .. } => language_model.device(),
            Self::Moe { language_model, .. } => language_model.device(),
        }
    }

    pub fn tp_context(&self) -> &TpContext {
        match self {
            Self::Dense { language_model, .. } => language_model.tp_context(),
            Self::Moe { language_model, .. } => language_model.tp_context(),
        }
    }
}

impl crate::engine::ModelForward for HunYuanVLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        // Text-only decode path.
        HunYuanVLForConditionalGeneration::forward(
            self,
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
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
    ) -> candle_core::Result<Tensor> {
        let (image_embeds, image_positions) =
            if let Some(mm) = multimodal_inputs.filter(|m| m.has_images()) {
                let mut all_embeds = Vec::new();
                let mut positions = Vec::new();
                for (pos, processed) in &mm.image_embeddings {
                    let emb = &processed.embedding;
                    let n = emb.dim(0)?;
                    all_embeds.push(emb.clone());
                    for i in 0..n {
                        positions.push(pos + i);
                    }
                }
                let flat = Tensor::cat(&all_embeds, 0)?;
                (flat, positions)
            } else {
                // No images — just run text-only forward.
                return HunYuanVLForConditionalGeneration::forward(
                    self,
                    input_ids,
                    seqlen_offset,
                    kv_cache_mgr,
                    block_table,
                    slot_mapping,
                );
            };

        HunYuanVLForConditionalGeneration::forward_multimodal(
            self,
            input_ids,
            &image_embeds,
            &image_positions,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        HunYuanVLForConditionalGeneration::device(self)
    }
}

/// Splice `image_embeds` into `text_embeddings` at the given token positions.
///
/// * `text_embeddings`  – `[1, seq_len, hidden]`
/// * `image_embeds`     – `[total_img_tokens, hidden]`
/// * `image_positions`  – indices into the seq_len dimension (sorted, contiguous)
fn splice_image_embeds(
    text_embeddings: Tensor,
    image_embeds: &Tensor,
    image_positions: &[usize],
) -> Result<Tensor> {
    let seq_len = text_embeddings.dim(1)?;
    let hidden = text_embeddings.dim(2)?;

    // Build a 2D embedding tensor from the 3D text_embeddings.
    let mut rows: Vec<Tensor> = Vec::with_capacity(seq_len);
    let pos_set: std::collections::HashSet<usize> = image_positions.iter().copied().collect();

    let mut img_idx = 0usize;
    for i in 0..seq_len {
        if pos_set.contains(&i) {
            let img_row = image_embeds.narrow(0, img_idx, 1)?; // [1, hidden]
            rows.push(img_row);
            img_idx += 1;
        } else {
            let text_row = text_embeddings.narrow(1, i, 1)?.squeeze(0)?; // [1, hidden]
            rows.push(text_row);
        }
    }

    let combined = Tensor::cat(&rows, 0)?.unsqueeze(0)?; // [1, seq_len, hidden]
    debug_assert_eq!(combined.dims(), &[1, seq_len, hidden]);
    Ok(combined)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use candle_core::{DType, Device};

    /// Tiny ModelConfig for VLM tests.
    fn vl_model_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["HunYuanVLForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_hidden_layers: 1,
            head_dim: 16,
            vocab_size: 256,
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            intermediate_size: 64,
            tie_word_embeddings: false,
            extra: {
                let mut m = serde_json::Map::new();
                m.insert(
                    "vision_config".to_string(),
                    serde_json::json!({
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "num_channels": 3,
                        "patch_size": 4,
                        "spatial_merge_size": 2,
                        "out_hidden_size": 32,
                        "max_image_size": 8,
                        "rms_norm_eps": 1e-5,
                    }),
                );
                m
            },
            ..Default::default()
        }
    }

    fn make_kv_cache(cfg: &ModelConfig, device: &Device) -> (KVCacheManager, BlockTable) {
        let mut kv = KVCacheManager::new(&CacheConfig {
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
        .unwrap();
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 16).unwrap();
        (kv, bt)
    }

    #[test]
    fn test_hunyuan_vl_dense_construction() {
        let cfg = vl_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = HunYuanVLForConditionalGeneration::new_dense(&cfg, vb);
        assert!(
            model.is_ok(),
            "HunYuanVLForConditionalGeneration (Dense) should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_hunyuan_vl_vision_encoder_shape() {
        let cfg = vl_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let vcfg = HunYuanVisionConfig::from_model_config(&cfg);

        let encoder = HunYuanVisionTransformer::new(&vcfg, vb).unwrap();

        // 1 image, 4×4 grid (patch_size=4, image=16px × 16px → 4×4 patches)
        let grid_thw = [[1usize, 4, 4]];
        let num_patches = 16;
        // pixel_values: [num_patches, C * ps^2] = [16, 3*4*4=48]
        let pixel_values =
            Tensor::zeros((num_patches, 3 * 4 * 4), candle_core::DType::F32, &device).unwrap();

        let image_feats = encoder.forward(&pixel_values, &grid_thw).unwrap();
        assert_eq!(image_feats.len(), 1, "One feature tensor per image");

        // spatial_merge_size=2: llm_h=4/2=2, llm_w=4/2=2, tokens=2*(2+1)+2=8
        let expected_tokens = (4 / 2) * (4 / 2 + 1) + 2;
        assert_eq!(
            image_feats[0].dims(),
            &[expected_tokens, vcfg.out_hidden_size],
            "Vision encoder output shape mismatch"
        );
    }

    #[test]
    fn test_hunyuan_vl_text_forward() {
        let cfg = vl_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = HunYuanVLForConditionalGeneration::new_dense(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 4), candle_core::DType::U32, &device).unwrap();
        let (mut kv, bt) = make_kv_cache(&cfg, &device);
        let slot_mapping = [0usize, 1, 2, 3];

        let logits = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            logits.is_ok(),
            "Text-only forward should succeed: {:?}",
            logits.err()
        );
        let logits = logits.unwrap();
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_hunyuan_vl_multimodal_forward() {
        let cfg = vl_model_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = HunYuanVLForConditionalGeneration::new_dense(&cfg, vb).unwrap();

        // Sequence: 2 text + 3 image + 2 text = 7 tokens
        let seq_len = 7;
        let input_ids = Tensor::zeros((1, seq_len), candle_core::DType::U32, &device).unwrap();
        let image_embeds =
            Tensor::zeros((3, cfg.hidden_size), candle_core::DType::F32, &device).unwrap();
        let image_positions = vec![2, 3, 4]; // image tokens at positions 2,3,4

        let (mut kv, bt) = make_kv_cache(&cfg, &device);
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let logits = model.forward_multimodal(
            &input_ids,
            &image_embeds,
            &image_positions,
            0,
            &mut kv,
            &bt,
            &slot_mapping,
        );
        assert!(
            logits.is_ok(),
            "Multimodal forward should succeed: {:?}",
            logits.err()
        );
        let logits = logits.unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_hunyuan_vl_moe_construction() {
        let mut cfg = vl_model_config();
        cfg.architectures = vec!["HunYuanMoEV1ForCausalLM".to_string()];
        // Set MoE config via extra
        cfg.extra
            .insert("num_experts".to_string(), serde_json::json!(4));
        cfg.extra
            .insert("moe_topk".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("moe_intermediate_size".to_string(), serde_json::json!(32));
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = HunYuanVLForConditionalGeneration::new_moe(&cfg, vb);
        assert!(
            model.is_ok(),
            "HunYuanVLForConditionalGeneration (MoE) should construct: {:?}",
            model.err()
        );
    }
}
