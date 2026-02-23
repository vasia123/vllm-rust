//! Kanana-V vision-language model.
//!
//! Architecture: Qwen2-VL vision encoder (no merger) + DynamicCAbstractor
//! (RegNet stages + bilinear pos-emb resampling) + LlamaForCausalLM.
//!
//! # Pipeline
//!
//! ```text
//! pixel_values [np, cps]         (np = total patches, cps = C*T*pH*pW)
//! grid_thw     [ni, 3]           (T, H, W in patch units per image)
//!
//! → Qwen2VLVisionEncoder          [np, embed_dim]
//! → DynamicCAbstractor            [np/m², output_hidden_size]
//!     pos_emb resampling (bilinear, optional)
//!     s1: RegStage (depth blocks, embed_dim → hidden_size)
//!     PatchMerge  ([1,D,H,W] → [1,D*m²,H/m,W/m])
//!     s2: RegStage (depth blocks, hidden_size*m² → hidden_size)
//!     readout MLP (hidden_size → output_hidden_size)
//! → merge_multimodal               [B, S, output_hidden_size]
//! → LlamaForCausalLM               logits [B, S, vocab]
//! ```
//!
//! # Weight paths
//!
//! ```text
//! vision_model.patch_embed.proj.{weight,bias}
//! vision_model.blocks.{i}.{norm1,norm2,attn,mlp}.*
//! abstractor.pos_emb                                   (optional)
//! abstractor.net.0.blocks.block{k}.{conv1,conv2,conv3,shortcut}.*   (s1)
//! abstractor.net.2.blocks.block{k}.*                                 (s2)
//! abstractor.readout.{0,2,4,...}.{weight,bias}
//! model.model.embed_tokens.weight
//! model.model.layers.{i}.*
//! model.lm_head.weight
//! ```
//!
//! # Note on depth=0
//!
//! The Python reference supports `depth=0` (PatchMerge only, no RegNet), but the
//! readout MLP dimensions are ambiguous when `merge_size > 1` because PatchMerge
//! expands channels by `m²` while `build_mlp` is called with `encoder_hidden_size`
//! (pre-merge width).  No known Kanana-V checkpoint uses `depth=0`, so this
//! implementation requires `depth ≥ 1`.
//!
//! # References
//!
//! `reference/vllm/vllm/model_executor/models/kanana_v.py`
//! `reference/vllm/vllm/model_executor/models/qwen2_vl.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::models::llama::LlamaForCausalLM;
use crate::models::moonvit::bilinear_interp_2d;
use crate::models::qwen2_vl::{Qwen2VLVisionConfig, Qwen2VLVisionEncoder};
use crate::multimodal::MultimodalInputs;

// ─── Config ──────────────────────────────────────────────────────────────────

/// DynamicCAbstractor configuration, read from `cfg.extra["projector_config"]`.
#[derive(Debug, Clone)]
struct KananaAbstractorConfig {
    /// Number of RegStage bottleneck blocks per stage (must be ≥ 1).
    depth: usize,
    /// Vision encoder output dim (= Qwen2VLVisionConfig.embed_dim).
    encoder_hidden_size: usize,
    /// Hidden size inside the RegNet stages.
    hidden_size: usize,
    /// LLM hidden size (readout output = language_model.hidden_size).
    output_hidden_size: usize,
    /// Depth of the readout MLP (number of Linear layers).
    mlp_depth: usize,
    /// Spatial merge factor: PatchMerge reduces H and W by this factor.
    merge_size: usize,
    /// Whether a learnable bilinear-resampled position embedding is used.
    pos_emb: bool,
    /// Number of tokens in the default position embedding grid (N = pos_emb_size).
    pos_emb_size: usize,
}

impl KananaAbstractorConfig {
    fn from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Self {
        let proj = extra
            .get("projector_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let usize_field = |obj: &serde_json::Map<_, _>, key: &str, default: usize| {
            obj.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let bool_field = |obj: &serde_json::Map<_, _>, key: &str, default: bool| {
            obj.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
        };

        KananaAbstractorConfig {
            depth: usize_field(&proj, "depth", 1),
            encoder_hidden_size: usize_field(&proj, "encoder_hidden_size", 1152),
            hidden_size: usize_field(&proj, "hidden_size", 1024),
            output_hidden_size: usize_field(&proj, "output_hidden_size", 2048),
            mlp_depth: usize_field(&proj, "mlp_depth", 1),
            merge_size: usize_field(&proj, "merge_size", 2),
            pos_emb: bool_field(&proj, "pos_emb", false),
            pos_emb_size: usize_field(&proj, "pos_emb_size", 0),
        }
    }
}

// ─── LayerNorm2d ─────────────────────────────────────────────────────────────

/// Channel-first LayerNorm for `[B, C, H, W]` tensors (timm `LayerNorm2d`).
///
/// Permutes to `[B, H, W, C]`, applies `LayerNorm` on the last dimension C,
/// then permutes back to `[B, C, H, W]`.
struct LayerNorm2d {
    ln: LayerNorm,
}

impl LayerNorm2d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: layer_norm(channels, 1e-6, vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let x = self.ln.forward(&x)?;
        x.permute((0, 3, 1, 2))?.contiguous()
    }
}

// ─── ConvNormAct ─────────────────────────────────────────────────────────────

/// Conv2d (no bias) + LayerNorm2d + optional SiLU activation.
///
/// Mirrors timm `ConvNormAct` with `norm_layer=LayerNorm2d, act_layer=SiLU`.
/// Weight paths: `vb.pp("conv").weight`, `vb.pp("bn").{weight,bias}`.
struct ConvNormAct {
    conv: Conv2d,
    norm: LayerNorm2d,
    act: bool,
}

impl ConvNormAct {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        padding: usize,
        act: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = candle_nn::conv2d_no_bias(
            in_c,
            out_c,
            kernel,
            Conv2dConfig {
                padding,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;
        let norm = LayerNorm2d::new(out_c, vb.pp("bn"))?;
        Ok(Self { conv, norm, act })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        if self.act {
            x.silu()
        } else {
            Ok(x)
        }
    }
}

// ─── RegBottleneck ───────────────────────────────────────────────────────────

/// Single timm-style RegNet bottleneck block (stride=1, groups=1, bottle_ratio=1.0).
///
/// Structure: conv1(1×1,+act) → conv2(3×3,+act) → conv3(1×1,no act) + skip → SiLU.
/// Shortcut is a 1×1 projection when `in_c ≠ out_c`, else identity.
///
/// Weight paths under `vb`:
/// - `conv{1,2,3}.conv.weight`, `conv{1,2,3}.bn.{weight,bias}`
/// - `shortcut.conv.weight`, `shortcut.bn.{weight,bias}`  (when in_c ≠ out_c)
struct RegBottleneck {
    conv1: ConvNormAct,
    conv2: ConvNormAct,
    conv3: ConvNormAct,
    shortcut: Option<ConvNormAct>,
}

impl RegBottleneck {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        // bottle_ratio = 1.0 → bottleneck_chs = out_c (no dimension reduction)
        let conv1 = ConvNormAct::new(in_c, out_c, 1, 0, true, vb.pp("conv1"))?;
        let conv2 = ConvNormAct::new(out_c, out_c, 3, 1, true, vb.pp("conv2"))?;
        let conv3 = ConvNormAct::new(out_c, out_c, 1, 0, false, vb.pp("conv3"))?;
        let shortcut = if in_c != out_c {
            Some(ConvNormAct::new(
                in_c,
                out_c,
                1,
                0,
                false,
                vb.pp("shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            conv3,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let skip = match &self.shortcut {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };
        let x = self.conv1.forward(x)?;
        let x = self.conv2.forward(&x)?;
        let x = self.conv3.forward(&x)?;
        (x + skip)?.silu()
    }
}

// ─── RegStage ────────────────────────────────────────────────────────────────

/// A stack of `depth` `RegBottleneck` blocks (timm `RegStage`).
///
/// Weight paths: `vb.pp("blocks").pp("block{1}"), …, .pp("block{depth}")` (1-indexed).
struct RegStage {
    blocks: Vec<RegBottleneck>,
}

impl RegStage {
    fn new(depth: usize, in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        let mut prev_c = in_c;
        for i in 0..depth {
            blocks.push(RegBottleneck::new(
                prev_c,
                out_c,
                vb_blocks.pp(format!("block{}", i + 1)),
            )?);
            prev_c = out_c;
        }
        Ok(Self { blocks })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

// ─── ReadoutMLP ──────────────────────────────────────────────────────────────

/// Sequential `[Linear → SiLU → Linear → …]` readout projector.
///
/// `mlp_depth` Linear layers with SiLU between them.
/// PyTorch `nn.Sequential` indices: first Linear at 0, SiLU at 1,
/// second Linear at 2, etc.  Weight paths: `vb.pp("0")`, `vb.pp("2")`, …
struct ReadoutMLP {
    layers: Vec<Linear>,
}

impl ReadoutMLP {
    fn new(mlp_depth: usize, in_size: usize, out_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(mlp_depth);
        layers.push(candle_nn::linear(in_size, out_size, vb.pp("0"))?);
        for i in 1..mlp_depth {
            // Each extra Linear follows a SiLU: indices 2, 4, 6, …
            layers.push(candle_nn::linear(out_size, out_size, vb.pp(i * 2))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.layers[0].forward(x)?;
        for layer in self.layers.iter().skip(1) {
            x = layer.forward(&x.silu()?)?;
        }
        Ok(x)
    }
}

// ─── DynamicCAbstractor ──────────────────────────────────────────────────────

/// C-Abstractor: bilinear pos-emb resampling + two RegNet stages + readout MLP.
///
/// Processes each image independently and concatenates the results.
/// Only `depth ≥ 1` is supported (see module-level note on depth=0).
struct DynamicCAbstractor {
    /// Optional learnable position embedding `[1, pos_emb_size, encoder_hidden_size]`.
    pos_emb: Option<Tensor>,
    /// Side length of the square pos_emb grid (pos_emb_side² = pos_emb_size).
    pos_emb_side: usize,
    s1: RegStage,
    s2: RegStage,
    readout: ReadoutMLP,
    merge_size: usize,
    encoder_hidden_size: usize,
}

impl DynamicCAbstractor {
    fn new(cfg: &KananaAbstractorConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.depth == 0 {
            candle_core::bail!(
                "DynamicCAbstractor depth=0 is not supported: PatchMerge expands channels \
                 by merge_size² but the readout MLP expects encoder_hidden_size inputs. \
                 All known Kanana-V checkpoints use depth ≥ 1."
            );
        }

        let pos_emb = if cfg.pos_emb && cfg.pos_emb_size > 0 {
            Some(vb.get(
                (1usize, cfg.pos_emb_size, cfg.encoder_hidden_size),
                "pos_emb",
            )?)
        } else {
            None
        };
        let pos_emb_side = if cfg.pos_emb_size > 0 {
            ((cfg.pos_emb_size as f64).sqrt() as usize).max(1)
        } else {
            1
        };

        // net.0 = s1, net.1 = PatchMerge (no weights), net.2 = s2
        let s1 = RegStage::new(
            cfg.depth,
            cfg.encoder_hidden_size,
            cfg.hidden_size,
            vb.pp("net").pp("0"),
        )?;
        let s2 = RegStage::new(
            cfg.depth,
            cfg.merge_size * cfg.merge_size * cfg.hidden_size,
            cfg.hidden_size,
            vb.pp("net").pp("2"),
        )?;
        let readout = ReadoutMLP::new(
            cfg.mlp_depth,
            cfg.hidden_size,
            cfg.output_hidden_size,
            vb.pp("readout"),
        )?;

        Ok(Self {
            pos_emb,
            pos_emb_side,
            s1,
            s2,
            readout,
            merge_size: cfg.merge_size,
            encoder_hidden_size: cfg.encoder_hidden_size,
        })
    }

    /// Bilinear-resample `pos_emb [1, N, D]` from `(old_h, old_w)` to `(new_h, new_w)`.
    ///
    /// Delegates to `bilinear_interp_2d` which operates on `[H, W, D]` tensors.
    fn resample_pos_emb(
        pos_emb: &Tensor,
        old_h: usize,
        old_w: usize,
        new_h: usize,
        new_w: usize,
    ) -> Result<Tensor> {
        let (_, _, d) = pos_emb.dims3()?;
        // [1, N, D] → [old_h, old_w, D]
        let x = pos_emb.reshape((old_h, old_w, d))?;
        // bilinear_interp_2d: [H_src, W_src, D] → [H_dst*W_dst, D]
        let x = bilinear_interp_2d(&x, new_h, new_w)?;
        // [new_h*new_w, D] → [1, new_h*new_w, D]
        x.unsqueeze(0)
    }

    /// PatchMerge: `[1, D, H, W]` → `[1, D*m², H/m, W/m]`.
    fn patch_merge(x: &Tensor, m: usize) -> Result<Tensor> {
        let (b, d, h, w) = x.dims4()?;
        // [B, D, H, W] → [B, D, H/m, m, W/m, m] → [B, D, m, m, H/m, W/m] → [B, D*m², H/m, W/m]
        let x = x.reshape((b, d, h / m, m, w / m, m))?;
        let x = x.permute((0, 1, 3, 5, 2, 4))?.contiguous()?;
        x.reshape((b, d * m * m, h / m, w / m))
    }

    /// Process a single image: `embeds [H*W, D]`, grid `(H, W)` → `[(H/m)*(W/m), out_dim]`.
    fn forward_one(&self, embeds: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let d = embeds.dim(1)?;
        // Reshape to [1, H, W, D]
        let x = embeds.reshape((1usize, h, w, d))?;

        // Add resampled pos_emb if present
        let x = if let Some(ref pe) = self.pos_emb {
            let resampled = Self::resample_pos_emb(pe, self.pos_emb_side, self.pos_emb_side, h, w)?;
            let resampled = resampled.reshape((1usize, h, w, self.encoder_hidden_size))?;
            (x + resampled)?
        } else {
            x
        };

        // [1, H, W, D] → [1, D, H, W]
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;

        // s1: embed_dim → hidden_size
        let x = self.s1.forward(&x)?;
        // PatchMerge: [1, hidden_size, H, W] → [1, hidden_size*m², H/m, W/m]
        let x = Self::patch_merge(&x, self.merge_size)?;
        // s2: hidden_size*m² → hidden_size
        let x = self.s2.forward(&x)?;

        // [1, hidden_size, H/m, W/m] → [(H/m)*(W/m), hidden_size]
        let (_, d_out, h_out, w_out) = x.dims4()?;
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let x = x.reshape((h_out * w_out, d_out))?;

        self.readout.forward(&x)
    }

    /// Run abstractor over all images in the batch.
    ///
    /// `flattened` – all patch features concatenated `[total_patches, embed_dim]`
    /// `grid_thw`  – `(T, H, W)` for each image (T must be 1)
    fn forward(&self, flattened: &Tensor, grid_thw: &[(usize, usize, usize)]) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            if t != 1 {
                candle_core::bail!("Kanana-V: video (T>1) not supported, got T={t}");
            }
            let n = h * w;
            let embeds_i = flattened.narrow(0, offset, n)?;
            offset += n;
            outputs.push(self.forward_one(&embeds_i, h, w)?);
        }
        Tensor::cat(&outputs, 0)
    }
}

// ─── merge_multimodal ────────────────────────────────────────────────────────

/// Splice pre-computed image embeddings into the text embedding sequence.
///
/// Follows the same positional-insertion pattern as `glm4_1v.rs`: each
/// `(pos, ProcessedImage)` pair overwrites `img.num_tokens` consecutive
/// positions starting at `pos` in the flattened `[B, S, D]` tensor.
fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }

    let (_batch_size, seq_len, _hidden) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_dtype(DType::F32)?.to_vec3::<f32>()?;

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

    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main model ──────────────────────────────────────────────────────────────

/// Kanana-V vision-language model.
pub struct KananaVForConditionalGeneration {
    vision_encoder: Qwen2VLVisionEncoder,
    abstractor: DynamicCAbstractor,
    language_model: LlamaForCausalLM,
    #[allow(dead_code)]
    vision_cfg: Qwen2VLVisionConfig,
    device: Device,
}

impl KananaVForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vision_cfg = Qwen2VLVisionConfig::from_model_config(cfg);
        let abs_cfg = KananaAbstractorConfig::from_extra(&cfg.extra);

        let vision_encoder = Qwen2VLVisionEncoder::new(&vision_cfg, vb.pp("vision_model"))?;
        let abstractor = DynamicCAbstractor::new(&abs_cfg, vb.pp("abstractor"))?;
        // LlamaForCausalLM uses vb.pp("model") internally for model.* weights
        // and vb.pp("lm_head") for the LM head.
        let language_model = LlamaForCausalLM::new(cfg, vb.pp("model"))?;

        Ok(Self {
            vision_encoder,
            abstractor,
            language_model,
            vision_cfg,
            device: vb.device().clone(),
        })
    }

    /// Encode raw patch values through vision encoder + DynamicCAbstractor.
    ///
    /// Returns `[total_output_tokens, output_hidden_size]`.
    ///
    /// Called by the processor before `forward_multimodal`.
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[(usize, usize, usize)],
    ) -> Result<Tensor> {
        // Per-image ViT encode (no merger)
        let mut patch_features: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            let n = t * h * w;
            let patches = pixel_values.narrow(0, offset, n)?;
            offset += n;
            let features = self.vision_encoder.encode(&patches, h, w)?;
            patch_features.push(features);
        }
        let all_features = Tensor::cat(&patch_features, 0)?;
        // Project via DynamicCAbstractor
        self.abstractor.forward(&all_features, grid_thw)
    }
}

impl crate::engine::ModelForward for KananaVForConditionalGeneration {
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
        let embeddings = if let Some(mm) = multimodal_inputs {
            merge_multimodal(&text_embeddings, mm, &self.device)?
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

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::kv_cache::KVCacheDtype;
    use candle_core::DType;
    use serde_json::json;

    /// Minimal config: ViT depth=1, abstractor depth=1, LLM 2 layers.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        extra.insert(
            "vision_config".to_string(),
            json!({
                "depth": 1,
                "embed_dim": 8,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "in_channels": 1,
                "spatial_merge_size": 2
            }),
        );

        extra.insert(
            "projector_config".to_string(),
            json!({
                "depth": 1,
                "encoder_hidden_size": 8,
                "hidden_size": 8,
                "output_hidden_size": 16,
                "mlp_depth": 1,
                "merge_size": 2,
                "pos_emb": false,
                "pos_emb_size": 4
            }),
        );

        ModelConfig {
            architectures: vec!["KananaVForConditionalGeneration".to_string()],
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 1000,
            max_position_embeddings: 512,
            head_dim: 8,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: None,
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let cache_config = CacheConfig {
            block_size: 4,
            num_blocks: 64,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cache_config).unwrap()
    }

    #[test]
    fn test_kanana_v_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KananaVForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
    }

    #[test]
    fn test_kanana_v_vision_encoder() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KananaVForConditionalGeneration::new(&cfg, vb).unwrap();
        let vcfg = Qwen2VLVisionConfig::from_model_config(&cfg);

        // 4 patches (2×2 grid), patch_size=2, in_channels=1, temporal=1 → cps=4
        let cps = vcfg.patch_input_dim();
        let pixel_values = Tensor::zeros((4usize, cps), DType::F32, &device).unwrap();
        let result = model.vision_encoder.encode(&pixel_values, 2, 2);
        assert!(result.is_ok(), "vision encode failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[4, vcfg.embed_dim]);
    }

    #[test]
    fn test_kanana_v_abstractor() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KananaVForConditionalGeneration::new(&cfg, vb).unwrap();

        // 4 patches (2×2 grid), merge_size=2 → 1 output token per image
        let features = Tensor::zeros((4usize, 8usize), DType::F32, &device).unwrap();
        let grid_thw = [(1usize, 2usize, 2usize)];
        let result = model.abstractor.forward(&features, &grid_thw);
        assert!(result.is_ok(), "abstractor failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 16]);
    }

    #[test]
    fn test_kanana_v_text_only_prefill() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KananaVForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();

        use crate::engine::ModelForward;
        let result = model.forward(&input_ids, 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_kanana_v_encode_images_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = KananaVForConditionalGeneration::new(&cfg, vb).unwrap();
        let vcfg = Qwen2VLVisionConfig::from_model_config(&cfg);

        // Two images: 4 patches each (2×2 grid)
        let cps = vcfg.patch_input_dim();
        let pixel_values = Tensor::zeros((8usize, cps), DType::F32, &device).unwrap();
        let grid_thw = [(1usize, 2usize, 2usize), (1usize, 2usize, 2usize)];

        let result = model.encode_images(&pixel_values, &grid_thw);
        assert!(result.is_ok(), "encode_images failed: {:?}", result.err());
        // 2 images × 1 output token each (4 patches / merge_size²=4) = 2 tokens
        assert_eq!(result.unwrap().dims(), &[2, 16]);
    }
}
