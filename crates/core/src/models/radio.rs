#![allow(dead_code)]
//! RadioModel vision encoder for NemotronH_Nano_VL_V2 and NemotronParse.
//!
//! RadioModel is an InternViT-based vision encoder with:
//! - `ViTPatchGenerator`: unfold-based patch embedding with learnable CLS + register tokens
//!   and interpolatable absolute positional encoding.
//! - `RadioVisionBlock`: same pre-norm + ls1/ls2 block as InternViT, but layer-scale
//!   weights are NOT loaded from the checkpoint (they stay at the default 0.1 init).
//! - `RadioModel`: wraps the InternVisionModel, strips CLS/register tokens from output.
//!
//! Weight prefix convention (from Python `RadioModel.load_weights`):
//!   `radio_model.model.patch_generator.*`  → `model.patch_generator.*`
//!   `radio_model.model.blocks.{i}.*`       → `model.encoder.layers.{i}.*`
//!   (ls1 / ls2 are intentionally skipped; they keep default 0.1 init)

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

use crate::layers::{rms_norm, RmsNorm};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Subset of RadioConfig fields used by the vision encoder.
#[derive(Debug, Clone)]
pub(crate) struct RadioVisionConfig {
    pub hidden_size: usize,
    /// Typically `hidden_size * mlp_ratio` (e.g. 1280 * 4 = 5120).
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Maximum image size for CPE (Conditional Positional Encoding) interpolation.
    pub cpe_max_size: usize,
    pub qkv_bias: bool,
    pub qk_normalization: bool,
    pub layer_norm_eps: f64,
    /// Number of CLS tokens (1 by default; `len(unique_teachers)` when
    /// `cls_token_per_teacher=true`).
    pub num_cls_tokens: usize,
    /// Number of register tokens prepended alongside CLS.
    pub num_registers: usize,
}

impl RadioVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of patch tokens (no CLS, no registers).
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// CLS + register tokens prepended to the sequence.
    pub fn num_skip(&self) -> usize {
        self.num_cls_tokens + self.num_registers
    }

    /// CPE mode: true when `cpe_max_size > image_size`.
    pub fn cpe_mode(&self) -> bool {
        self.cpe_max_size > self.image_size
    }
}

impl Default for RadioVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            patch_size: 16,
            image_size: 512,
            cpe_max_size: 512,
            qkv_bias: false,
            qk_normalization: false,
            layer_norm_eps: 1e-6,
            num_cls_tokens: 1,
            num_registers: 0,
        }
    }
}

// ─── ClsToken ────────────────────────────────────────────────────────────────

/// Learnable CLS + register tokens prepended to patch embeddings.
///
/// Weight: `cls_token.token` of shape `[num_cls + num_registers, hidden_size]`.
struct ClsToken {
    /// Combined CLS + register parameter. None when disabled.
    token: Option<Tensor>,
    num_cls: usize,
    num_registers: usize,
    enabled: bool,
}

impl ClsToken {
    fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        if cfg.num_cls_tokens > 0 || cfg.num_registers > 0 {
            let total = cfg.num_cls_tokens + cfg.num_registers;
            let token = vb.get((total, cfg.hidden_size), "token")?;
            Ok(Self {
                token: Some(token),
                num_cls: cfg.num_cls_tokens,
                num_registers: cfg.num_registers,
                enabled: true,
            })
        } else {
            Ok(Self {
                token: None,
                num_cls: 0,
                num_registers: 0,
                enabled: false,
            })
        }
    }

    /// Prepend CLS + register tokens to `x` of shape `[B, S, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let Some(ref tok) = self.token else {
            return Ok(x.clone());
        };
        let batch = x.dim(0)?;
        // [num_total, D] → [B, num_total, D]
        let tok = tok
            .unsqueeze(0)?
            .broadcast_as((batch, tok.dim(0)?, tok.dim(1)?))?;
        Tensor::cat(&[tok, x.clone()], 1)
    }

    fn num_total_prepended(&self) -> usize {
        if self.enabled {
            self.num_cls + self.num_registers
        } else {
            0
        }
    }
}

// ─── ViTPatchGenerator ───────────────────────────────────────────────────────

/// Converts `[B,3,H,W]` pixel values into patch embeddings `[B, S, D]`.
///
/// Steps:
///   1. `Im2Patches`: unfold image into `[B, py*px, 3*ps*ps]`
///   2. `ViTPatchLinear`: linear projection → `[B, py*px, hidden_size]`
///   3. Add absolute positional encoding (interpolated if needed)
///   4. Prepend CLS + register tokens via `ClsToken`
pub(crate) struct ViTPatchGenerator {
    patch_size: usize,
    num_rows: usize,
    num_cols: usize,
    /// `[1, num_rows*num_cols, hidden_size]`
    pos_embed: Tensor,
    embedder: Linear,
    cls_token: ClsToken,
    cpe_mode: bool,
}

impl ViTPatchGenerator {
    pub(crate) fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let max_size = if cfg.cpe_mode() {
            cfg.cpe_max_size
        } else {
            cfg.image_size
        };
        let num_rows = max_size / cfg.patch_size;
        let num_cols = max_size / cfg.patch_size;
        let num_patches = num_rows * num_cols;
        let patch_flat = 3 * cfg.patch_size * cfg.patch_size;

        let pos_embed = vb.get((1, num_patches, cfg.hidden_size), "pos_embed")?;
        let embedder = candle_nn::linear_no_bias(patch_flat, cfg.hidden_size, vb.pp("embedder"))?;
        let cls_token = ClsToken::new(cfg, vb.pp("cls_token"))?;

        Ok(Self {
            patch_size: cfg.patch_size,
            num_rows,
            num_cols,
            pos_embed,
            embedder,
            cls_token,
            cpe_mode: cfg.cpe_mode(),
        })
    }

    /// Forward: `pixel_values [B,3,H,W]` → `[B, num_skip + H/ps * W/ps, D]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;
        let py = h / self.patch_size;
        let px = w / self.patch_size;

        // Im2Patches: [B,3,H,W] → [B, py*px, 3*ps*ps]
        let patches = im2patches(x, self.patch_size)?;

        // Project patches: [B, py*px, 3*ps*ps] → [B, py*px, hidden_size]
        let patches = self.embedder.forward(&patches)?;

        // Add positional encoding
        let pos = self.get_pos_enc(b, py, px)?;
        let patches = (patches + pos)?;

        // Prepend CLS + register tokens
        self.cls_token.forward(&patches)
    }

    fn get_pos_enc(&self, _batch: usize, py: usize, px: usize) -> Result<Tensor> {
        if (py, px) == (self.num_rows, self.num_cols) {
            return Ok(self.pos_embed.clone());
        }

        // Interpolate: reshape to [1, D, num_rows, num_cols], bilinear → [1, D, py, px],
        // then flatten to [1, py*px, D].
        let d = self.pos_embed.dim(2)?;
        let pe = self
            .pos_embed
            .reshape((1, self.num_rows, self.num_cols, d))?
            .permute((0, 3, 1, 2))?; // [1, D, rows, cols]

        // candle bilinear interpolation via avg_pool2d + upsample approximation:
        // For exact interpolation we use Tensor::upsample_bilinear2d.
        let pe = pe.interpolate2d(py, px)?;

        // [1, D, py, px] → [1, py*px, D]
        let pe = pe.permute((0, 2, 3, 1))?.reshape((1, py * px, d))?;
        Ok(pe)
    }

    pub(crate) fn num_skip(&self) -> usize {
        self.cls_token.num_total_prepended()
    }
}

/// Unfold `[B,3,H,W]` into `[B, (H/ps)*(W/ps), 3*ps*ps]`.
///
/// Equivalent to Python `rearrange(x, "b c (py yy) (px xx) -> b (py px) (c yy xx)")`.
fn im2patches(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let py = h / patch_size;
    let px = w / patch_size;
    let ps = patch_size;

    // [B, C, py*ps, px*ps] → [B, C, py, ps, px, ps]
    let x = x.reshape((b, c, py, ps, px, ps))?;
    // [B, C, py, ps, px, ps] → [B, py, px, C, ps, ps]
    let x = x.permute((0, 2, 4, 1, 3, 5))?;
    // [B, py, px, C, ps, ps] → [B, py*px, C*ps*ps]
    x.reshape((b, py * px, c * ps * ps))
}

// ─── RadioVisionAttention ────────────────────────────────────────────────────

struct RadioVisionAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    head_dim: usize,
}

impl RadioVisionAttention {
    fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv = candle_nn::linear_b(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            cfg.qkv_bias,
            vb.pp("qkv"),
        )?;
        let proj = candle_nn::linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("proj"))?;

        let (q_norm, k_norm) = if cfg.qk_normalization {
            let hd = cfg.head_dim();
            (
                Some(rms_norm(hd, cfg.layer_norm_eps, vb.pp("q_norm"))?),
                Some(rms_norm(hd, cfg.layer_norm_eps, vb.pp("k_norm"))?),
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

    /// Bidirectional self-attention. `x: [B, S, D]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let qkv = self.qkv.forward(x)?;
        let h = self.num_heads;
        let d = self.head_dim;

        let q = qkv.narrow(2, 0, h * d)?;
        let k = qkv.narrow(2, h * d, h * d)?;
        let v = qkv.narrow(2, 2 * h * d, h * d)?;

        let q = q.reshape((b, s, h, d))?.transpose(1, 2)?;
        let k = k.reshape((b, s, h, d))?.transpose(1, 2)?;
        let v = v.reshape((b, s, h, d))?.transpose(1, 2)?;

        let q = if let Some(ref n) = self.q_norm {
            n.forward(&q.contiguous()?)?
        } else {
            q
        };
        let k = if let Some(ref n) = self.k_norm {
            n.forward(&k.contiguous()?)?
        } else {
            k
        };

        let scale = 1.0 / (d as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))?;
        self.proj.forward(&out)
    }
}

// ─── RadioVisionMLP ──────────────────────────────────────────────────────────

struct RadioVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl RadioVisionMLP {
    fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── RadioVisionBlock ────────────────────────────────────────────────────────

/// Pre-norm transformer block with layer scaling.
///
/// Like InternVisionBlock but ls1/ls2 are NOT loaded from the checkpoint —
/// they keep the fixed 0.1 default (Python skips them during weight loading).
struct RadioVisionBlock {
    norm1: LayerNorm,
    attn: RadioVisionAttention,
    norm2: LayerNorm,
    mlp: RadioVisionMLP,
}

impl RadioVisionBlock {
    fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("norm2"))?;
        let attn = RadioVisionAttention::new(cfg, vb.pp("attn"))?;
        let mlp = RadioVisionMLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        use candle_core::Module;
        // Layer-scale 0.1 matches Python init; weights are intentionally not loaded.
        let xs = self.attn.forward(&self.norm1.forward(x)?)?;
        let x = (x + (xs * 0.1)?)?;
        let xs = self.mlp.forward(&self.norm2.forward(&x)?)?;
        &x + &(xs * 0.1)?
    }
}

// ─── RadioInternVisionModel ──────────────────────────────────────────────────

/// InternViT-based encoder with ViTPatchGenerator embedding.
pub(crate) struct RadioInternVisionModel {
    pub(crate) patch_generator: ViTPatchGenerator,
    layers: Vec<RadioVisionBlock>,
}

impl RadioInternVisionModel {
    fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_generator = ViTPatchGenerator::new(cfg, vb.pp("patch_generator"))?;

        let vb_blocks = vb.pp("blocks");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(RadioVisionBlock::new(cfg, vb_blocks.pp(i))?);
        }

        Ok(Self {
            patch_generator,
            layers,
        })
    }

    /// `x [B,3,H,W]` → `[B, num_skip + S, D]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.patch_generator.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        Ok(h)
    }
}

// ─── RadioModel ──────────────────────────────────────────────────────────────

/// RADIO vision encoder.
///
/// Returns `(summary, features)`:
/// - `summary`: `[B, num_cls_tokens * D]` — averaged CLS token(s)
/// - `features`: `[B, S, D]` — patch features with CLS/registers stripped
pub(crate) struct RadioModel {
    model: RadioInternVisionModel,
    num_cls_tokens: usize,
    num_skip: usize,
}

impl RadioModel {
    /// Construct from a `RadioVisionConfig`.
    ///
    /// `vb` should point to the root of the RADIO model (i.e., the level that
    /// has `model.patch_generator.*` and `model.blocks.*` children).
    pub(crate) fn new(cfg: &RadioVisionConfig, vb: VarBuilder) -> Result<Self> {
        let model = RadioInternVisionModel::new(cfg, vb.pp("model"))?;
        let num_cls_tokens = cfg.num_cls_tokens;
        let num_skip = model.patch_generator.num_skip();
        Ok(Self {
            model,
            num_cls_tokens,
            num_skip,
        })
    }

    /// Forward: `pixel_values [B,3,H,W]` → `(summary, features)`.
    ///
    /// `summary [B, num_cls*D]` — CLS token features (flattened).
    /// `features [B, S, D]`    — spatial patch features (CLS/registers stripped).
    pub(crate) fn forward(&self, pixel_values: &Tensor) -> Result<(Tensor, Tensor)> {
        let y = self.model.forward(pixel_values)?;
        self.extract_final(&y)
    }

    fn extract_final(&self, y: &Tensor) -> Result<(Tensor, Tensor)> {
        // `y [B, num_skip + S, D]`
        let d = y.dim(2)?;

        let summary = y.narrow(1, 0, self.num_cls_tokens)?;
        // flatten cls dim: [B, num_cls, D] → [B, num_cls*D]
        let b = y.dim(0)?;
        let summary = summary.reshape((b, self.num_cls_tokens * d))?;

        let total_s = y.dim(1)?;
        let feat_len = total_s - self.num_skip;
        let features = y.narrow(1, self.num_skip, feat_len)?;

        Ok((summary, features))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn small_cfg() -> RadioVisionConfig {
        RadioVisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            patch_size: 4,
            image_size: 16,
            cpe_max_size: 16,
            qkv_bias: false,
            qk_normalization: false,
            layer_norm_eps: 1e-6,
            num_cls_tokens: 1,
            num_registers: 0,
        }
    }

    fn make_vb(cfg: &RadioVisionConfig, device: &Device) -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        // Initialise all required tensors
        let m = vb.pp("model");
        let pg = m.pp("patch_generator");

        let num_max_patches = (cfg.cpe_max_size / cfg.patch_size).pow(2);
        let patch_flat = 3 * cfg.patch_size * cfg.patch_size;

        // pos_embed, embedder
        pg.get((1, num_max_patches, cfg.hidden_size), "pos_embed")
            .unwrap();
        pg.pp("embedder")
            .get((cfg.hidden_size, patch_flat), "weight")
            .unwrap();

        // cls_token
        let num_cls_total = cfg.num_cls_tokens + cfg.num_registers;
        if num_cls_total > 0 {
            pg.pp("cls_token")
                .get((num_cls_total, cfg.hidden_size), "token")
                .unwrap();
        }

        // blocks
        for i in 0..cfg.num_hidden_layers {
            let b = m.pp("blocks").pp(i);
            for norm in ["norm1", "norm2"] {
                b.pp(norm).get(cfg.hidden_size, "weight").unwrap();
                b.pp(norm).get(cfg.hidden_size, "bias").unwrap();
            }
            // attn
            let attn = b.pp("attn");
            attn.get((3 * cfg.hidden_size, cfg.hidden_size), "weight")
                .unwrap();
            attn.get((cfg.hidden_size, cfg.hidden_size), "weight_proj")
                .unwrap_or_else(|_| {
                    attn.pp("proj")
                        .get((cfg.hidden_size, cfg.hidden_size), "weight")
                        .unwrap()
                });
            // mlp
            let mlp = b.pp("mlp");
            mlp.pp("fc1")
                .get((cfg.intermediate_size, cfg.hidden_size), "weight")
                .unwrap();
            mlp.pp("fc2")
                .get((cfg.hidden_size, cfg.intermediate_size), "weight")
                .unwrap();
        }

        (varmap, vb)
    }

    #[test]
    fn test_im2patches() {
        let device = Device::Cpu;
        let ps = 4usize;
        let img = Tensor::zeros((1, 3, 16, 16), DType::F32, &device).unwrap();
        let patches = im2patches(&img, ps).unwrap();
        // 16/4 * 16/4 = 16 patches, each 3*4*4=48 values
        assert_eq!(patches.shape().dims(), &[1, 16, 48]);
    }

    #[test]
    fn test_cls_token_prepend() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = small_cfg();
        vb.pp("cls_token")
            .get((cfg.num_cls_tokens, cfg.hidden_size), "token")
            .unwrap();
        let cls = ClsToken::new(&cfg, vb).unwrap();
        let x = Tensor::zeros((2, 10, cfg.hidden_size), DType::F32, &device).unwrap();
        let out = cls.forward(&x).unwrap();
        // 1 CLS + 10 patches = 11
        assert_eq!(out.dim(1).unwrap(), 11);
    }

    #[test]
    fn test_radio_model_output_shape() {
        let device = Device::Cpu;
        let cfg = small_cfg();
        let (_varmap, vb) = make_vb(&cfg, &device);

        // Need separate VarBuilder for RadioModel since make_vb uses vb.pp("model") internally
        let model_vb = vb; // radio.rs RadioModel::new does vb.pp("model") internally
        let radio = RadioModel::new(&cfg, model_vb).unwrap();

        let img = Tensor::zeros((1, 3, 16, 16), DType::F32, &device).unwrap();
        let (summary, features) = radio.forward(&img).unwrap();

        // summary: [1, 1 * 32] = [1, 32]
        assert_eq!(summary.shape().dims(), &[1, 32]);
        // features: [1, 16, 32] (16 patches, no CLS stripped)
        assert_eq!(features.shape().dims(), &[1, 16, 32]);
    }

    #[test]
    fn test_radio_model_with_registers() {
        let device = Device::Cpu;
        let mut cfg = small_cfg();
        cfg.num_registers = 4; // 1 CLS + 4 registers = 5 skip tokens

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Init weights
        let m = vb.pp("model");
        let pg = m.pp("patch_generator");
        let num_max_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let patch_flat = 3 * cfg.patch_size * cfg.patch_size;

        pg.get((1, num_max_patches, cfg.hidden_size), "pos_embed")
            .unwrap();
        pg.pp("embedder")
            .get((cfg.hidden_size, patch_flat), "weight")
            .unwrap();
        let total = cfg.num_cls_tokens + cfg.num_registers;
        pg.pp("cls_token")
            .get((total, cfg.hidden_size), "token")
            .unwrap();

        for i in 0..cfg.num_hidden_layers {
            let b = m.pp("blocks").pp(i);
            for norm in ["norm1", "norm2"] {
                b.pp(norm).get(cfg.hidden_size, "weight").unwrap();
                b.pp(norm).get(cfg.hidden_size, "bias").unwrap();
            }
            let attn = b.pp("attn");
            attn.get((3 * cfg.hidden_size, cfg.hidden_size), "weight")
                .unwrap();
            attn.pp("proj")
                .get((cfg.hidden_size, cfg.hidden_size), "weight")
                .unwrap();
            let mlp = b.pp("mlp");
            mlp.pp("fc1")
                .get((cfg.intermediate_size, cfg.hidden_size), "weight")
                .unwrap();
            mlp.pp("fc2")
                .get((cfg.hidden_size, cfg.intermediate_size), "weight")
                .unwrap();
        }

        let radio = RadioModel::new(&cfg, vb).unwrap();

        let img = Tensor::zeros((1, 3, 16, 16), DType::F32, &device).unwrap();
        let (summary, features) = radio.forward(&img).unwrap();

        assert_eq!(summary.shape().dims(), &[1, 32]); // 1 CLS * 32 hidden
        assert_eq!(features.shape().dims(), &[1, 16, 32]); // 16 patches
    }

    #[test]
    fn test_pos_embed_interpolation() {
        // Test that forward works when input size differs from cpe_max_size
        let device = Device::Cpu;
        let mut cfg = small_cfg();
        cfg.cpe_max_size = 32; // larger than image_size=16 → cpe_mode=true
        cfg.num_hidden_layers = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let m = vb.pp("model");
        let pg = m.pp("patch_generator");

        let max_patches = (cfg.cpe_max_size / cfg.patch_size).pow(2); // 8*8=64
        let patch_flat = 3 * cfg.patch_size * cfg.patch_size;

        pg.get((1, max_patches, cfg.hidden_size), "pos_embed")
            .unwrap();
        pg.pp("embedder")
            .get((cfg.hidden_size, patch_flat), "weight")
            .unwrap();
        pg.pp("cls_token")
            .get((1, cfg.hidden_size), "token")
            .unwrap();

        let b = m.pp("blocks").pp(0);
        for norm in ["norm1", "norm2"] {
            b.pp(norm).get(cfg.hidden_size, "weight").unwrap();
            b.pp(norm).get(cfg.hidden_size, "bias").unwrap();
        }
        b.pp("attn")
            .get((3 * cfg.hidden_size, cfg.hidden_size), "weight")
            .unwrap();
        b.pp("attn")
            .pp("proj")
            .get((cfg.hidden_size, cfg.hidden_size), "weight")
            .unwrap();
        b.pp("mlp")
            .pp("fc1")
            .get((cfg.intermediate_size, cfg.hidden_size), "weight")
            .unwrap();
        b.pp("mlp")
            .pp("fc2")
            .get((cfg.hidden_size, cfg.intermediate_size), "weight")
            .unwrap();

        let radio = RadioModel::new(&cfg, vb).unwrap();

        // 16x16 image with ps=4 → 4x4=16 patches
        let img = Tensor::zeros((1, 3, 16, 16), DType::F32, &device).unwrap();
        let (summary, features) = radio.forward(&img).unwrap();
        assert_eq!(summary.shape().dims(), &[1, 32]);
        assert_eq!(features.shape().dims(), &[1, 16, 32]);
    }
}
