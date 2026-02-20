//! HyperCLOVA-X Vision (HCXVision) model.
//!
//! Architecture: CLIP/SigLIP vision encoder + MM projector + LLaMA-style LLM.
//!
//! Supported projector types:
//! - `linear`      — single `nn.Linear`
//! - `mlp`         — `Linear(in → hidden) + GELU + Linear(hidden → out)`
//! - `inverted_mlp`— `Linear(in → 2*hidden) + GELU + Linear(2*hidden → out)`
//! - `cabstractor` — C-Abstractor (Honeybee): RegNet residual stages + adaptive pooling
//!
//! ## C-Abstractor weight layout (under `mm_projector.*`)
//!
//! ```text
//! pos_emb                                → [1, num_patches, vis_hidden]  (optional)
//! prenorm.{weight,bias}                  → LayerNorm (optional)
//! net.0.{0,1,2}.{conv1,conv2,conv3,se,shortcut}.*  → s1 RegStage (3 blocks)
//! net.2.{0,1,2}.*                        → s2 RegStage (3 blocks)
//! readout.0.{weight,bias}                → Linear(hidden → out)
//! readout.2.{weight,bias}                → Linear(out → out)
//! ```
//!
//! Each RegBottleneckBlock (depth=3 blocks per stage):
//! ```text
//! conv1.conv.weight   [out, in, 1, 1]
//! conv1.bn.{weight,bias}  [out]
//! conv2.conv.weight   [out, out, 3, 3]
//! conv2.bn.{weight,bias}  [out]
//! se.fc.{weight,bias}     [se_chs=out//4, out, 1, 1] / [se_chs]
//! se.fcs.{weight,bias}    [out, se_chs, 1, 1] / [out]
//! conv3.conv.weight   [out, out, 1, 1]
//! conv3.bn.{weight,bias}  [out]
//! shortcut.conv.weight    [out, in, 1, 1]       (only when in_chs != out_chs)
//! shortcut.bn.{weight,bias}  [out]
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/hyperclovax_vision.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;
use crate::multimodal::{VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::llama::LlamaForCausalLM;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum MmProjectorType {
    Linear,
    Mlp,
    InvertedMlp,
    CAbstractor,
}

impl MmProjectorType {
    fn from_str(s: &str) -> Self {
        match s {
            "linear" => Self::Linear,
            "mlp" => Self::Mlp,
            "inverted_mlp" => Self::InvertedMlp,
            "cabstractor" => Self::CAbstractor,
            _ => Self::CAbstractor, // default
        }
    }
}

struct HcxConfig {
    vis_cfg: VisionEncoderConfig,
    /// Skip CLS token for CLIP (not needed for SigLIP).
    skip_cls: bool,
    mm_projector_type: MmProjectorType,
    /// Number of output visual queries (must be a perfect square).
    num_queries_image: usize,
    proj_pos_emb: bool,
    proj_prenorm: bool,
    /// Number of visual patches (image_size / patch_size)².
    num_input_patches: usize,
}

impl HcxConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let vc = cfg
            .extra
            .get("vision_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let get_usize = |key: &str, default: usize| -> usize {
            vc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            vc.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let model_type = vc
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("clip_vision_model");

        let encoder_type = if model_type.contains("siglip") {
            VisionEncoderType::SigLip
        } else {
            VisionEncoderType::Clip
        };
        let skip_cls = encoder_type == VisionEncoderType::Clip;

        let image_size = get_usize("image_size", 224);
        let patch_size = get_usize("patch_size", 14);
        let vis_cfg = VisionEncoderConfig {
            encoder_type,
            hidden_size: get_usize("hidden_size", 1024),
            intermediate_size: get_usize("intermediate_size", 4096),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_hidden_layers: get_usize("num_hidden_layers", 24),
            image_size,
            patch_size,
            num_channels: get_usize("num_channels", 3),
            layer_norm_eps: get_f64("layer_norm_eps", 1e-5),
        };

        let num_input_patches = (image_size / patch_size).pow(2);

        let mm_projector_type = cfg
            .extra
            .get("mm_projector_type")
            .and_then(|v| v.as_str())
            .map(MmProjectorType::from_str)
            .unwrap_or(MmProjectorType::CAbstractor);

        let num_queries_image = cfg
            .extra
            .get("num_queries_vis_abstractor_image")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(81); // default 9×9=81 for 3B model

        let proj_pos_emb = cfg
            .extra
            .get("proj_pos_emb")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let proj_prenorm = cfg
            .extra
            .get("proj_prenorm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            vis_cfg,
            skip_cls,
            mm_projector_type,
            num_queries_image,
            proj_pos_emb,
            proj_prenorm,
            num_input_patches,
        }
    }
}

// ─── LayerNorm2d ─────────────────────────────────────────────────────────────

/// Apply LayerNorm on the channel dimension of a 4D `[B, C, H, W]` tensor.
///
/// timm's `LayerNorm2d` normalizes over channels (dim=1), equivalent to
/// standard LayerNorm after transposing to `[B, H, W, C]`.
struct LayerNorm2d {
    weight: Tensor, // [C]
    bias: Tensor,   // [C]
    eps: f64,
}

impl LayerNorm2d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(channels, "weight")?;
        let bias = vb.get(channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-5,
        })
    }

    /// `x`: `[B, C, H, W]` → same shape.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w_size) = x.dims4()?;
        // Transpose to [B, H, W, C], normalize last dim, transpose back
        let x = x.permute((0, 2, 3, 1))?.contiguous()?; // [B, H, W, C]
        let x = x.reshape((b * h * w_size, c))?;

        // Manual LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
        let mean = x.mean_keepdim(1)?; // [B*H*W, 1]
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(1)?; // [B*H*W, 1]
        let std = (var + self.eps)?.sqrt()?;
        let x = diff.broadcast_div(&std)?;

        let wt = self.weight.to_dtype(x.dtype())?;
        let bt = self.bias.to_dtype(x.dtype())?;
        let x = x
            .broadcast_mul(&wt.unsqueeze(0)?)?
            .broadcast_add(&bt.unsqueeze(0)?)?;

        x.reshape((b, h, w_size, c))?.permute((0, 3, 1, 2))
    }
}

// ─── Squeeze-Excitation ───────────────────────────────────────────────────────

/// Squeeze-Excitation channel attention (timm implementation).
///
/// Uses 1×1 Conv2d for both layers; se_channels = round(in_channels * 0.25).
struct SeModule {
    fc: Conv2d,  // [se_chs, in_chs, 1, 1]
    fcs: Conv2d, // [in_chs, se_chs, 1, 1]
}

impl SeModule {
    fn new(in_channels: usize, vb: VarBuilder) -> Result<Self> {
        let se_chs = (in_channels / 4).max(1);
        let cfg = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let fc = conv2d(in_channels, se_chs, 1, cfg, vb.pp("fc"))?;
        let fcs = conv2d(se_chs, in_channels, 1, cfg, vb.pp("fcs"))?;
        Ok(Self { fc, fcs })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // Global average pool → [B, C, 1, 1]
        let pool = x.reshape((b, c, h * w))?.mean_keepdim(2)?.unsqueeze(3)?;
        let gate = self.fc.forward(&pool)?;
        let gate = gate.silu()?;
        let gate = self.fcs.forward(&gate)?;
        let gate = candle_nn::ops::sigmoid(&gate)?;
        x * gate.broadcast_as(x.shape())?
    }
}

// ─── RegBottleneckBlock ───────────────────────────────────────────────────────

/// Residual bottleneck block with SE attention and LayerNorm2d.
///
/// Matches timm's `BottleneckBlock` with `norm_layer=LayerNorm2d, act_layer=SiLU`.
/// Weight paths under the block's VarBuilder root:
/// `conv1.{conv,bn}.*, conv2.{conv,bn}.*, se.{fc,fcs}.*, conv3.{conv,bn}.*`
/// and optionally `shortcut.{conv,bn}.*` when in_chs != out_chs.
struct RegBottleneckBlock {
    conv1_w: Tensor, // [out, in, 1, 1]
    conv1_bn: LayerNorm2d,
    conv2_w: Tensor, // [out, out, 3, 3]
    conv2_bn: LayerNorm2d,
    se: SeModule,
    conv3_w: Tensor, // [out, out, 1, 1]
    conv3_bn: LayerNorm2d,
    shortcut: Option<(Tensor, LayerNorm2d)>, // conv.weight + bn
}

impl RegBottleneckBlock {
    fn new(in_chs: usize, out_chs: usize, vb: VarBuilder) -> Result<Self> {
        // conv1: 1×1, stride=1 (no bias — norm absorbs it)
        let conv1_w = vb.get((out_chs, in_chs, 1, 1), "conv1.conv.weight")?;
        let conv1_bn = LayerNorm2d::new(out_chs, vb.pp("conv1.bn"))?;

        // conv2: 3×3, stride=1, padding=1
        let conv2_w = vb.get((out_chs, out_chs, 3, 3), "conv2.conv.weight")?;
        let conv2_bn = LayerNorm2d::new(out_chs, vb.pp("conv2.bn"))?;

        // SE
        let se = SeModule::new(out_chs, vb.pp("se"))?;

        // conv3: 1×1, no activation
        let conv3_w = vb.get((out_chs, out_chs, 1, 1), "conv3.conv.weight")?;
        let conv3_bn = LayerNorm2d::new(out_chs, vb.pp("conv3.bn"))?;

        // Shortcut (only when in_chs != out_chs)
        let shortcut = if in_chs != out_chs {
            let sc_w = vb.get((out_chs, in_chs, 1, 1), "shortcut.conv.weight")?;
            let sc_bn = LayerNorm2d::new(out_chs, vb.pp("shortcut.bn"))?;
            Some((sc_w, sc_bn))
        } else {
            None
        };

        Ok(Self {
            conv1_w,
            conv1_bn,
            conv2_w,
            conv2_bn,
            se,
            conv3_w,
            conv3_bn,
            shortcut,
        })
    }

    /// `x`: `[B, C, H, W]` → `[B, out_chs, H, W]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Shortcut (no bias — LayerNorm2d absorbs it)
        let sc = if let Some((ref w, ref bn)) = self.shortcut {
            // padding=0, stride=1, dilation=1, groups=1
            let s = x.conv2d(w, 0, 1, 1, 1)?;
            bn.forward(&s)?
        } else {
            x.clone()
        };

        // conv1: 1×1
        let h = x.conv2d(&self.conv1_w, 0, 1, 1, 1)?;
        let h = self.conv1_bn.forward(&h)?.silu()?;

        // conv2: 3×3 (padding=1 to preserve spatial)
        let h = h.conv2d(&self.conv2_w, 1, 1, 1, 1)?;
        let h = self.conv2_bn.forward(&h)?.silu()?;

        // SE channel attention
        let h = self.se.forward(&h)?;

        // conv3: 1×1, no activation after
        let h = h.conv2d(&self.conv3_w, 0, 1, 1, 1)?;
        let h = self.conv3_bn.forward(&h)?;

        // Residual + final SiLU
        (sc + h)?.silu()
    }
}

// ─── RegStage ─────────────────────────────────────────────────────────────────

/// Sequence of `depth` RegBottleneckBlocks.
///
/// Block 0: `in_chs → out_chs` (may have shortcut).
/// Blocks 1..: `out_chs → out_chs` (no shortcut).
struct RegStage {
    blocks: Vec<RegBottleneckBlock>,
}

impl RegStage {
    fn new(depth: usize, in_chs: usize, out_chs: usize, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| {
                let block_in = if i == 0 { in_chs } else { out_chs };
                RegBottleneckBlock::new(block_in, out_chs, vb.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
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

// ─── Adaptive Average Pool 2D ─────────────────────────────────────────────────

/// Reduce `[B, C, H, W]` → `[B, C, oh, ow]` via average pooling.
///
/// For the divisible case (H % oh == 0 and W % ow == 0), uses an efficient
/// reshape-and-mean. The non-divisible case uses a loop over output positions.
fn adaptive_avg_pool_2d(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    if h == oh && w == ow {
        return Ok(x.clone());
    }

    if h % oh == 0 && w % ow == 0 {
        let kh = h / oh;
        let kw = w / ow;
        // [B, C, H, W] → [B, C, oh, kh, ow, kw]
        let x = x.reshape((b, c, oh, kh, ow, kw))?;
        // Average over kh (dim=3), then kw (dim=4 after collapse)
        let x = x.mean(3)?; // [B, C, oh, ow, kw]
        return x.mean(4); // [B, C, oh, ow]
    }

    // General case: loop over output positions.
    // Reshape to [b*c, h*w] for simple 1D indexing.
    let x_f = x.to_dtype(DType::F32)?.reshape((b * c, h * w))?;
    let data = x_f.to_vec2::<f32>()?;
    let mut out = vec![0.0f32; b * c * oh * ow];

    for bc in 0..b * c {
        for oi in 0..oh {
            for oj in 0..ow {
                let h_start = (oi * h) / oh;
                let h_end = ((oi + 1) * h) / oh;
                let w_start = (oj * w) / ow;
                let w_end = ((oj + 1) * w) / ow;
                let count = (h_end - h_start) * (w_end - w_start);
                let mut sum = 0.0f32;
                for hi in h_start..h_end {
                    for wi in w_start..w_end {
                        sum += data[bc][hi * w + wi];
                    }
                }
                out[bc * oh * ow + oi * ow + oj] = sum / count as f32;
            }
        }
    }

    Tensor::from_vec(out, (b, c, oh, ow), x.device())?.to_dtype(x.dtype())
}

// ─── C-Abstractor Projector ──────────────────────────────────────────────────

/// C-Abstractor: ResNet stages + adaptive pooling + MLP readout.
///
/// Based on Honeybee's C-Abstractor. Processes `[B, L, D]` vision tokens:
/// 1. Optional prenorm (LayerNorm) + positional embedding
/// 2. Reshape to `[B, D, H, W]` (H=W=√L)
/// 3. s1 RegStage (3 residual blocks): `[B, D, H, W]`
/// 4. AdaptiveAvgPool2d to `[B, D, hw_out, hw_out]` (hw_out = √num_queries)
/// 5. s2 RegStage (3 residual blocks): `[B, D, hw_out, hw_out]`
/// 6. Reshape to `[B, num_queries, D]` → readout MLP → `[B, num_queries, out_dim]`
struct HCXCAbstractor {
    pos_emb: Option<Tensor>,    // [1, L, D]
    prenorm: Option<LayerNorm>, // optional
    s1: RegStage,
    s2: RegStage,
    hw_out: usize, // sqrt(num_queries)
    fc1: Linear,   // readout.0
    fc2: Linear,   // readout.2
}

struct CAbstractorConfig {
    num_queries: usize,
    num_input_patches: usize,
    enc_hidden: usize,
    out_hidden: usize,
    pos_emb: bool,
    prenorm: bool,
    depth: usize,
}

impl HCXCAbstractor {
    fn new(cfg: &CAbstractorConfig, vb: VarBuilder) -> Result<Self> {
        let CAbstractorConfig {
            num_queries,
            num_input_patches,
            enc_hidden,
            out_hidden,
            pos_emb,
            prenorm,
            depth,
        } = *cfg;
        let pos_emb_param = if pos_emb {
            Some(vb.get((1, num_input_patches, enc_hidden), "pos_emb")?)
        } else {
            None
        };

        let prenorm_layer = if prenorm {
            Some(layer_norm(enc_hidden, 1e-5, vb.pp("prenorm"))?)
        } else {
            None
        };

        let hw_out = (num_queries as f64).sqrt() as usize;
        debug_assert_eq!(
            hw_out * hw_out,
            num_queries,
            "num_queries must be a perfect square"
        );

        // net.0 = s1, net.1 = sampler (no weights), net.2 = s2
        let s1 = RegStage::new(depth, enc_hidden, enc_hidden, vb.pp("net").pp(0))?;
        let s2 = RegStage::new(depth, enc_hidden, enc_hidden, vb.pp("net").pp(2))?;

        // Readout MLP: Linear(D → out) + SiLU + Linear(out → out)
        // (build_mlp with mlp_depth=2: layers[0]=Linear, layers[1]=SiLU, layers[2]=Linear)
        let fc1 = linear(enc_hidden, out_hidden, vb.pp("readout").pp(0))?;
        let fc2 = linear(out_hidden, out_hidden, vb.pp("readout").pp(2))?;

        Ok(Self {
            pos_emb: pos_emb_param,
            prenorm: prenorm_layer,
            s1,
            s2,
            hw_out,
            fc1,
            fc2,
        })
    }

    /// `x`: `[B, L, D]` vision tokens → `[B, num_queries, out_dim]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, d) = x.dims3()?;

        // Optional prenorm
        let x = if let Some(ref ln) = self.prenorm {
            ln.forward(x)?
        } else {
            x.clone()
        };

        // Optional positional embedding
        let x = if let Some(ref pe) = self.pos_emb {
            let pe = pe.to_dtype(x.dtype())?;
            x.broadcast_add(&pe)?
        } else {
            x
        };

        // Reshape to spatial: [B, L, D] → [B, D, H, W]
        let hw = (l as f64).sqrt() as usize;
        let x = x
            .reshape((b, hw, hw, d))?
            .permute((0, 3, 1, 2))?
            .contiguous()?; // [B, D, H, W]

        // s1 → adaptive pool → s2
        let x = self.s1.forward(&x)?;
        let x = adaptive_avg_pool_2d(&x, self.hw_out, self.hw_out)?;
        let x = self.s2.forward(&x)?;

        // Back to sequence: [B, D, hw, hw] → [B, hw², D]
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let x = x.reshape((b, self.hw_out * self.hw_out, d))?;

        // Readout MLP: Linear → SiLU → Linear
        let x = self.fc1.forward(&x)?;
        let x = x.silu()?;
        self.fc2.forward(&x)
    }
}

// ─── Other Projectors ─────────────────────────────────────────────────────────

struct HcxLinearProjector {
    linear: Linear,
}

impl HcxLinearProjector {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(in_dim, out_dim, vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

struct HcxMlpProjector {
    fc1: Linear,
    fc2: Linear,
}

impl HcxMlpProjector {
    /// `inverted`: if true, fc1 expands to 2× hidden (inverted_mlp variant).
    fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        inverted: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mid = if inverted { 2 * hidden_dim } else { hidden_dim };
        let fc1 = linear(in_dim, mid, vb.pp("fc1"))?;
        let fc2 = linear(mid, out_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

enum MmProjector {
    Linear(HcxLinearProjector),
    Mlp(HcxMlpProjector),
    CAbstractor(HCXCAbstractor),
}

impl MmProjector {
    fn new(cfg: &HcxConfig, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let vis_hidden = cfg.vis_cfg.hidden_size;
        match cfg.mm_projector_type {
            MmProjectorType::Linear => Ok(Self::Linear(HcxLinearProjector::new(
                vis_hidden,
                text_hidden,
                vb,
            )?)),
            MmProjectorType::Mlp => Ok(Self::Mlp(HcxMlpProjector::new(
                vis_hidden,
                vis_hidden,
                text_hidden,
                false,
                vb,
            )?)),
            MmProjectorType::InvertedMlp => Ok(Self::Mlp(HcxMlpProjector::new(
                vis_hidden,
                vis_hidden,
                text_hidden,
                true,
                vb,
            )?)),
            MmProjectorType::CAbstractor => {
                // depth=3 as in Python (timm RegBlock default)
                let cabstractor_cfg = CAbstractorConfig {
                    num_queries: cfg.num_queries_image,
                    num_input_patches: cfg.num_input_patches,
                    enc_hidden: vis_hidden,
                    out_hidden: text_hidden,
                    pos_emb: cfg.proj_pos_emb,
                    prenorm: cfg.proj_prenorm,
                    depth: 3,
                };
                Ok(Self::CAbstractor(HCXCAbstractor::new(
                    &cabstractor_cfg,
                    vb,
                )?))
            }
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(p) => p.forward(x),
            Self::Mlp(p) => p.forward(x),
            Self::CAbstractor(p) => p.forward(x),
        }
    }
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// HyperCLOVA-X Vision: CLIP/SigLIP encoder + projector + LLM.
pub struct HCXVisionForCausalLM {
    vision_model: VisionEncoder,
    mm_projector: MmProjector,
    language_model: LlamaForCausalLM,
    skip_cls: bool,
    device: Device,
    dtype: DType,
}

impl HCXVisionForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hcx_cfg = HcxConfig::from_model_config(cfg);

        let vision_model = VisionEncoder::new(&hcx_cfg.vis_cfg, vb.pp("vision_model"))?;
        let mm_projector = MmProjector::new(&hcx_cfg, cfg.hidden_size, vb.pp("mm_projector"))?;
        // Language model — use LLaMA for all text config model_types
        let language_model = LlamaForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            vision_model,
            mm_projector,
            language_model,
            skip_cls: hcx_cfg.skip_cls,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(cfg, vb)
    }

    /// Encode images through vision tower + projector → `[total_tokens, text_hidden]`.
    ///
    /// `pixel_values`: `[B, C, H, W]`.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vis_out = self.vision_model.forward(pixel_values)?; // [B, seq_len, D]

        // Skip CLS token for CLIP
        let vis_out = if self.skip_cls {
            let (_b, seq, _d) = vis_out.dims3()?;
            vis_out.narrow(1, 1, seq - 1)?.contiguous()? // [B, patches, D]
        } else {
            vis_out
        };

        self.mm_projector.forward(&vis_out)
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;
            for (i, row) in emb.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = row.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for HCXVisionForCausalLM {
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype, KVCacheManager};
    use candle_core::{DType, Device};
    use serde_json::json;

    /// Tiny config: small CLIP + C-Abstractor + small LLaMA.
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();

        // HCX VLM config
        extra.insert("mm_projector_type".to_string(), json!("cabstractor"));
        // 4 queries (2×2 = square), to match 4 patches from 4×4 image with patch_size=2
        extra.insert("num_queries_vis_abstractor_image".to_string(), json!(4));
        extra.insert("proj_pos_emb".to_string(), json!(true));
        extra.insert("proj_prenorm".to_string(), json!(false));

        // Small CLIP-style vision config
        extra.insert(
            "vision_config".to_string(),
            json!({
                "model_type": "clip_vision_model",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "image_size": 4,
                "patch_size": 2,
                "num_channels": 1,
                "layer_norm_eps": 1e-5
            }),
        );

        ModelConfig {
            architectures: vec!["HCXVisionForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
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
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        let cache_cfg = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        KVCacheManager::new(&cache_cfg).unwrap()
    }

    #[test]
    fn test_hcxvision_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = HCXVisionForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "HCXVisionForCausalLM construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_layer_norm_2d_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let ln = LayerNorm2d::new(8, vb).unwrap();
        let x = Tensor::zeros((2usize, 8usize, 4usize, 4usize), DType::F32, &device).unwrap();
        let out = ln.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_cabstractor_shape() {
        // 4 patches (image 4×4, patch 2×2 → 2×2 grid), vis_hidden=16, text_hidden=64
        // num_queries=4, hw_out=2
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let abstractor_cfg = CAbstractorConfig {
            num_queries: 4,
            num_input_patches: 4,
            enc_hidden: 16,
            out_hidden: 64,
            pos_emb: true,
            prenorm: false,
            depth: 1, // depth=1 for fast test (vs 3 in production)
        };
        let abstractor = HCXCAbstractor::new(&abstractor_cfg, vb).unwrap();

        let x = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &device).unwrap();
        let out = abstractor.forward(&x).unwrap();
        // [B=1, num_queries=4, out_hidden=64]
        assert_eq!(out.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_text_forward() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = HCXVisionForCausalLM::new(&cfg, vb).unwrap();
        let mut kv = make_cache(&cfg, &device);

        let input_ids = Tensor::zeros((1usize, 4usize), DType::U32, &device).unwrap();
        let block_table = crate::kv_cache::BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let out = model.forward(&input_ids, 0, &mut kv, &block_table, &slot_mapping);
        assert!(out.is_ok(), "text forward failed: {:?}", out.err());
        let out = out.unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_encode_images_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = HCXVisionForCausalLM::new(&cfg, vb).unwrap();

        // image_size=4, patch_size=2, in_channels=1 → [1, 1, 4, 4]
        let pixel_values =
            Tensor::zeros((1usize, 1usize, 4usize, 4usize), DType::F32, &device).unwrap();
        let out = model.encode_images(&pixel_values);
        assert!(out.is_ok(), "encode_images failed: {:?}", out.err());
        let out = out.unwrap();
        // C-Abstractor with depth=3 and 4 queries: [1, 4, text_hidden=64]
        assert_eq!(out.dims(), &[1, 4, 64]);
    }
}
