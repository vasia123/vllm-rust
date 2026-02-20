//! MoonViT vision encoder (Kimi-VL).
//!
//! Architecture overview:
//! - `MoonVisionPatchEmbed`: Conv2d patch projection + learnable 2D positional embedding
//! - `Rope2DPosEmb`: precomputed 2D RoPE with interleaved x/y frequencies (complex multiply)
//! - `MoonVitEncoder`: N transformer blocks with pre-norm + 2D RoPE + final LayerNorm
//! - `patch_merger`: spatial grouping — merge_kernel_size² patch tokens per output token
//!
//! # Input format
//!
//! `pixel_values`: `[L, in_channels, patch_size, patch_size]` — pre-extracted patches,
//! all images concatenated along the batch dimension (L = sum(h_i × w_i)).
//! `grid_hws`: `&[[h, w]]` — patch-grid dimensions per image.
//!
//! # Weight paths
//!
//! ```text
//! patch_embed.proj.*              → Conv2d projection
//! patch_embed.pos_emb.weight      → Learnable2DInterpPosEmb
//! encoder.blocks.{i}.norm0.*     → LayerNorm pre-attention
//! encoder.blocks.{i}.norm1.*     → LayerNorm pre-MLP
//! encoder.blocks.{i}.wqkv.*     → fused QKV (bias=true)
//! encoder.blocks.{i}.wo.*       → output projection (bias=true)
//! encoder.blocks.{i}.mlp.fc0.*  → MLP first linear (bias=true)
//! encoder.blocks.{i}.mlp.fc1.*  → MLP second linear (bias=true)
//! encoder.final_layernorm.*      → final LayerNorm
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/moonvit.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    layer_norm, linear, ops::softmax_last_dim, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};

// ─── Config ──────────────────────────────────────────────────────────────────

/// MoonViT vision encoder configuration.
#[derive(Debug, Clone)]
pub struct MoonVitConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub init_pos_emb_height: usize,
    pub init_pos_emb_width: usize,
    pub merge_kernel_size: [usize; 2],
    /// Maximum grid height for RoPE table (hardcoded 512 in Python).
    pub rope_max_height: usize,
    /// Maximum grid width for RoPE table (hardcoded 512 in Python).
    pub rope_max_width: usize,
    pub rope_theta: f32,
}

impl MoonVitConfig {
    /// Construct from the outer ModelConfig's `extra["vision_config"]` map.
    pub fn from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Self {
        let vc = extra
            .get("vision_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let get_usize = |map: &serde_json::Map<String, serde_json::Value>,
                         key: &str,
                         default: usize|
         -> usize {
            map.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let merge_kernel_size = vc
            .get("merge_kernel_size")
            .and_then(|v| v.as_array())
            .map(|arr| {
                let h = arr.first().and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                let w = arr.get(1).and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                [h, w]
            })
            .unwrap_or([2, 2]);

        Self {
            hidden_size: get_usize(&vc, "hidden_size", 1024),
            num_hidden_layers: get_usize(&vc, "num_hidden_layers", 24),
            num_attention_heads: get_usize(&vc, "num_attention_heads", 16),
            intermediate_size: get_usize(&vc, "intermediate_size", 4096),
            in_channels: get_usize(&vc, "in_channels", 3),
            patch_size: get_usize(&vc, "patch_size", 14),
            init_pos_emb_height: get_usize(&vc, "init_pos_emb_height", 32),
            init_pos_emb_width: get_usize(&vc, "init_pos_emb_width", 32),
            merge_kernel_size,
            // Python hardcodes 512; allow override via config
            rope_max_height: get_usize(&vc, "rope_max_height", 512),
            rope_max_width: get_usize(&vc, "rope_max_width", 512),
            rope_theta: vc
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0) as f32,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ─── 2D Rotary Position Embedding ────────────────────────────────────────────

/// Precomputed 2D rotary positional embeddings.
///
/// Stores interleaved x/y complex frequencies as real cos/sin tables of shape
/// `[max_h, max_w, head_dim/2]`.  Indices are laid out as:
/// ```text
/// [2i]   = frequency for column (x) position, i-th component
/// [2i+1] = frequency for row (y) position, i-th component
/// ```
/// This matches the Python `Rope2DPosEmb.precomputed_freqs_cis` layout where
/// x_cis and y_cis are interleaved: `cat([x_cis, y_cis], dim=-1)`.
pub(crate) struct Rope2DPosEmb {
    cos_table: Tensor, // [max_h, max_w, head_dim/2]
    sin_table: Tensor,
    #[allow(dead_code)]
    max_height: usize,
    #[allow(dead_code)]
    max_width: usize,
    half_dim: usize, // head_dim / 2
}

impl Rope2DPosEmb {
    /// Build precomputed tables.  `dim` = head_dim, must be divisible by 4.
    pub(crate) fn new(
        dim: usize,
        max_height: usize,
        max_width: usize,
        theta: f32,
        device: &Device,
    ) -> Result<Self> {
        assert!(
            dim.is_multiple_of(4),
            "head_dim must be divisible by 4 for 2D RoPE"
        );
        let n_freqs = dim / 4; // Python: arange(0, dim, 4) → dim/4 frequencies
        let half_dim = dim / 2;

        // freq[i] = 1 / theta^(4i / dim)  for i in 0..n_freqs
        let freqs: Vec<f32> = (0..n_freqs)
            .map(|i| 1.0f32 / theta.powf((4 * i) as f32 / dim as f32))
            .collect();

        let mut cos_data = vec![0.0f32; max_height * max_width * half_dim];
        let mut sin_data = vec![0.0f32; max_height * max_width * half_dim];

        for h in 0..max_height {
            for w in 0..max_width {
                let base = (h * max_width + w) * half_dim;
                for i in 0..n_freqs {
                    let x_angle = w as f32 * freqs[i]; // column / x-axis
                    let y_angle = h as f32 * freqs[i]; // row / y-axis
                    cos_data[base + 2 * i] = x_angle.cos();
                    cos_data[base + 2 * i + 1] = y_angle.cos();
                    sin_data[base + 2 * i] = x_angle.sin();
                    sin_data[base + 2 * i + 1] = y_angle.sin();
                }
            }
        }

        let cos_table = Tensor::from_vec(cos_data, (max_height, max_width, half_dim), device)?;
        let sin_table = Tensor::from_vec(sin_data, (max_height, max_width, half_dim), device)?;

        Ok(Self {
            cos_table,
            sin_table,
            max_height,
            max_width,
            half_dim,
        })
    }

    /// Select freqs for a sequence of image grids.
    ///
    /// Returns `(cos, sin)` each of shape `[S, head_dim/2]`
    /// where `S = sum(h_i * w_i)`.
    pub(crate) fn get_freqs_by_seqlens(&self, grid_hws: &[[usize; 2]]) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        for &[h, w] in grid_hws {
            let cos = self
                .cos_table
                .narrow(0, 0, h)?
                .narrow(1, 0, w)?
                .contiguous()?
                .reshape((h * w, self.half_dim))?;
            let sin = self
                .sin_table
                .narrow(0, 0, h)?
                .narrow(1, 0, w)?
                .contiguous()?
                .reshape((h * w, self.half_dim))?;
            cos_parts.push(cos);
            sin_parts.push(sin);
        }
        Ok((Tensor::cat(&cos_parts, 0)?, Tensor::cat(&sin_parts, 0)?))
    }

    /// Select freqs for 3D grids (temporal + spatial), repeating spatial freqs T times per image.
    ///
    /// `grid_thws`: `[[t, h, w], ...]` — T frames, H×W spatial grid per image.
    /// Returns `(cos, sin)` each of shape `[sum(t_i * h_i * w_i), head_dim/2]`.
    pub(crate) fn get_freqs_by_seqlens_3d(
        &self,
        grid_thws: &[[usize; 3]],
    ) -> Result<(Tensor, Tensor)> {
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        for &[t, h, w] in grid_thws {
            let cos = self
                .cos_table
                .narrow(0, 0, h)?
                .narrow(1, 0, w)?
                .contiguous()?
                .reshape((h * w, self.half_dim))?;
            let sin = self
                .sin_table
                .narrow(0, 0, h)?
                .narrow(1, 0, w)?
                .contiguous()?
                .reshape((h * w, self.half_dim))?;
            // Repeat spatial freqs T times — mirrors Python's .repeat(t, 1)
            for _ in 0..t {
                cos_parts.push(cos.clone());
                sin_parts.push(sin.clone());
            }
        }
        Ok((Tensor::cat(&cos_parts, 0)?, Tensor::cat(&sin_parts, 0)?))
    }
}

/// Apply interleaved 2D RoPE via complex multiplication.
///
/// Treats consecutive pairs `(q[2i], q[2i+1])` as complex numbers and multiplies
/// by `freqs_cos[i] + i*freqs_sin[i]`, replicating `view_as_complex` semantics.
///
/// - `q`, `k`: `[S, heads, head_dim]`
/// - `freqs_cos`, `freqs_sin`: `[S, head_dim/2]`
pub(crate) fn apply_rope_2d(
    q: &Tensor,
    k: &Tensor,
    freqs_cos: &Tensor,
    freqs_sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (s, heads, head_dim) = q.dims3()?;
    let half = head_dim / 2;

    // Cast freq tables to match input dtype (important for f16/bf16 inference)
    let cos = freqs_cos.to_dtype(q.dtype())?.unsqueeze(1)?; // [S, 1, half]
    let sin = freqs_sin.to_dtype(q.dtype())?.unsqueeze(1)?; // [S, 1, half]

    let rotate = |x: &Tensor| -> Result<(Tensor, Tensor, Tensor)> {
        // View as interleaved pairs: [S, heads, half, 2]
        let x4 = x.reshape((s, heads, half, 2))?;
        let x_re = x4.narrow(3, 0, 1)?.squeeze(3)?; // [S, heads, half]
        let x_im = x4.narrow(3, 1, 1)?.squeeze(3)?;
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let re_out = ((x_re.broadcast_mul(&cos))? - (x_im.broadcast_mul(&sin))?)?;
        let im_out = ((x_re.broadcast_mul(&sin))? + (x_im.broadcast_mul(&cos))?)?;
        Ok((re_out, im_out, x.clone()))
    };

    let interleave = |re: Tensor, im: Tensor| -> Result<Tensor> {
        // Stack → [S, heads, half, 2] → reshape → [S, heads, head_dim]
        Tensor::stack(&[&re, &im], 3)?
            .contiguous()?
            .reshape((s, heads, head_dim))
    };

    let (q_re, q_im, _) = rotate(q)?;
    let (k_re, k_im, _) = rotate(k)?;
    let q_out = interleave(q_re, q_im)?;
    let k_out = interleave(k_re, k_im)?;

    Ok((q_out, k_out))
}

// ─── Learnable 2D Positional Embedding ───────────────────────────────────────

/// Learned positional embedding with bilinear interpolation for variable grid sizes.
///
/// Weight shape: `[H_init, W_init, D]`.  When the requested grid matches the
/// stored height/width the weight is used as-is; otherwise it is bilinearly
/// interpolated using a pure-Rust implementation (no GPU kernel needed).
struct Learnable2DInterpPosEmb {
    weight: Tensor, // [H, W, D]
    init_height: usize,
    init_width: usize,
}

impl Learnable2DInterpPosEmb {
    fn new(height: usize, width: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((height, width, dim), "weight")?;
        Ok(Self {
            weight,
            init_height: height,
            init_width: width,
        })
    }

    /// Add positional embedding to `x: [S, D]`.
    fn forward(&self, x: &Tensor, grid_hws: &[[usize; 2]]) -> Result<Tensor> {
        let mut pos_embs = Vec::new();
        for &[h, w] in grid_hws {
            let emb = if h == self.init_height && w == self.init_width {
                let d = self.weight.dim(2)?;
                self.weight.reshape((h * w, d))?
            } else {
                bilinear_interp_2d(&self.weight, h, w)?
            };
            pos_embs.push(emb);
        }
        let pos_cat = Tensor::cat(&pos_embs, 0)?.to_dtype(x.dtype())?;
        x + pos_cat
    }
}

/// Bilinear interpolation of `weight: [H_src, W_src, D]` to `[H_dst*W_dst, D]`.
pub(crate) fn bilinear_interp_2d(weight: &Tensor, h_dst: usize, w_dst: usize) -> Result<Tensor> {
    let (h_src, w_src, d) = weight.dims3()?;
    let device = weight.device();
    let dtype = weight.dtype();

    let w_data = weight.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let h_scale = if h_dst > 1 {
        (h_src - 1) as f32 / (h_dst - 1) as f32
    } else {
        0.0
    };
    let w_scale = if w_dst > 1 {
        (w_src - 1) as f32 / (w_dst - 1) as f32
    } else {
        0.0
    };

    let mut out = vec![0.0f32; h_dst * w_dst * d];
    for i in 0..h_dst {
        for j in 0..w_dst {
            let y = i as f32 * h_scale;
            let x = j as f32 * w_scale;
            let y0 = y.floor() as usize;
            let y1 = (y0 + 1).min(h_src - 1);
            let x0 = x.floor() as usize;
            let x1 = (x0 + 1).min(w_src - 1);
            let wy = y - y0 as f32;
            let wx = x - x0 as f32;
            let base = (i * w_dst + j) * d;
            for di in 0..d {
                out[base + di] = (1.0 - wy) * (1.0 - wx) * w_data[y0][x0][di]
                    + (1.0 - wy) * wx * w_data[y0][x1][di]
                    + wy * (1.0 - wx) * w_data[y1][x0][di]
                    + wy * wx * w_data[y1][x1][di];
            }
        }
    }

    Tensor::from_vec(out, (h_dst * w_dst, d), device)?.to_dtype(dtype)
}

// ─── Patch Embed ─────────────────────────────────────────────────────────────

/// Project pre-extracted patches and add learnable positional embedding.
///
/// Pixel values are pre-extracted patches: `[L, C, ps, ps]`.
/// A Conv2d with `kernel_size=stride=patch_size` maps each patch to `[out_dim]`.
struct MoonVisionPatchEmbed {
    proj: Conv2d, // (in_channels → hidden, kernel=ps, stride=ps)
    pos_emb: Learnable2DInterpPosEmb,
}

impl MoonVisionPatchEmbed {
    fn new(cfg: &MoonVitConfig, vb: VarBuilder) -> Result<Self> {
        let ps = cfg.patch_size;
        let conv_cfg = Conv2dConfig {
            stride: ps,
            padding: 0,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(
            cfg.in_channels,
            cfg.hidden_size,
            ps,
            conv_cfg,
            vb.pp("proj"),
        )?;
        let pos_emb = Learnable2DInterpPosEmb::new(
            cfg.init_pos_emb_height,
            cfg.init_pos_emb_width,
            cfg.hidden_size,
            vb.pp("pos_emb"),
        )?;
        Ok(Self { proj, pos_emb })
    }

    /// Input: `pixel_values [L, C, ps, ps]`, grid sizes `[(h, w)]`.
    /// Output: `[L, hidden_dim]`.
    fn forward(&self, x: &Tensor, grid_hws: &[[usize; 2]]) -> Result<Tensor> {
        // [L, C, ps, ps] → [L, hidden, 1, 1] → [L, hidden]
        let x = self.proj.forward(x)?;
        let (l, c, _, _) = x.dims4()?;
        let x = x.reshape((l, c))?;
        self.pos_emb.forward(&x, grid_hws)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

struct MoonVitMlp {
    fc0: Linear, // bias=true: in_dim → hidden_dim
    fc1: Linear, // bias=true: hidden_dim → out_dim
}

impl MoonVitMlp {
    fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc0 = linear(in_dim, hidden_dim, vb.pp("fc0"))?;
        let fc1 = linear(hidden_dim, out_dim, vb.pp("fc1"))?;
        Ok(Self { fc0, fc1 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python uses ACT2FN["gelu_pytorch_tanh"] ≈ GELU; gelu_erf is close enough
        self.fc1.forward(&self.fc0.forward(x)?.gelu_erf()?)
    }
}

// ─── Encoder Layer ───────────────────────────────────────────────────────────

pub(crate) struct MoonVitEncoderLayer {
    norm0: LayerNorm,
    norm1: LayerNorm,
    wqkv: Linear, // fused QKV, bias=true: [hidden → 3*hidden]
    wo: Linear,   // output proj, bias=true: [hidden → hidden]
    mlp: MoonVitMlp,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MoonVitEncoderLayer {
    pub(crate) fn new(cfg: &MoonVitConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        // Python: nn.LayerNorm(hidden_dim) — default eps=1e-5
        let norm0 = layer_norm(h, 1e-5, vb.pp("norm0"))?;
        let norm1 = layer_norm(h, 1e-5, vb.pp("norm1"))?;
        // attn_bias: True in Python config
        let wqkv = linear(h, 3 * h, vb.pp("wqkv"))?;
        let wo = linear(h, h, vb.pp("wo"))?;
        let mlp = MoonVitMlp::new(h, cfg.intermediate_size, h, vb.pp("mlp"))?;
        Ok(Self {
            norm0,
            norm1,
            wqkv,
            wo,
            mlp,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Forward.
    ///
    /// `x`: `[S, hidden]` — packed patches (all images concatenated).
    /// `freqs_cos`, `freqs_sin`: `[S, head_dim/2]`.
    ///
    /// NOTE: Attention is computed over the full packed sequence without per-image
    /// masking.  In the Python reference, `cu_seqlens` prevents cross-image
    /// attention, but this is an acceptable approximation for inference since ViT
    /// encoders are typically run one image at a time.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        freqs_cos: &Tensor,
        freqs_sin: &Tensor,
    ) -> Result<Tensor> {
        let s = x.dim(0)?;

        // ── Attention block ──
        let residual = x.clone();
        let x_norm = self.norm0.forward(x)?;

        let qkv = self.wqkv.forward(&x_norm)?; // [S, 3*H]
        let qkv = qkv.reshape((s, 3, self.num_heads, self.head_dim))?;
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?; // [S, heads, head_dim]
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?.contiguous()?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?.contiguous()?;

        let (q, k) = apply_rope_2d(&q, &k, freqs_cos, freqs_sin)?;

        // SDPA: reshape to [heads, S, head_dim]
        let q = q.permute((1, 0, 2))?;
        let k = k.permute((1, 0, 2))?;
        let v = v.permute((1, 0, 2))?;

        let attn = (q.matmul(&k.transpose(1, 2)?)? * self.scale)?; // [heads, S, S]
        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [heads, S, head_dim]

        // [heads, S, head_dim] → [S, hidden]
        let out = out
            .permute((1, 0, 2))?
            .contiguous()?
            .reshape((s, self.num_heads * self.head_dim))?;
        let out = self.wo.forward(&out)?;
        let x = (residual + out)?;

        // ── MLP block ──
        let residual = x.clone();
        let mlp_out = self.mlp.forward(&self.norm1.forward(&x)?)?;
        residual + mlp_out
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

struct MoonVitEncoder {
    rope_2d: Rope2DPosEmb,
    blocks: Vec<MoonVitEncoderLayer>,
    final_layernorm: LayerNorm,
}

impl MoonVitEncoder {
    fn new(cfg: &MoonVitConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let rope_2d = Rope2DPosEmb::new(
            head_dim,
            cfg.rope_max_height,
            cfg.rope_max_width,
            cfg.rope_theta,
            vb.device(),
        )?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| MoonVitEncoderLayer::new(cfg, vb.pp("blocks").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let final_layernorm = layer_norm(cfg.hidden_size, 1e-5, vb.pp("final_layernorm"))?;
        Ok(Self {
            rope_2d,
            blocks,
            final_layernorm,
        })
    }

    /// `x`: `[L, hidden]` — packed patches.
    /// `grid_hws`: grid sizes per image.
    fn forward(&self, x: &Tensor, grid_hws: &[[usize; 2]]) -> Result<Tensor> {
        let (freqs_cos, freqs_sin) = self.rope_2d.get_freqs_by_seqlens(grid_hws)?;
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, &freqs_cos, &freqs_sin)?;
        }
        self.final_layernorm.forward(&x)
    }
}

// ─── Patch Merger ────────────────────────────────────────────────────────────

/// Group spatial patches spatially.
///
/// For each image with grid `(h, w)` and merge kernel `(kh, kw)`:
/// - Input slice: `[h*w, D]`
/// - Reshape to `[new_h, kh, new_w, kw, D]`, permute `(0,2,1,3,4)`, reshape to
///   `[new_h*new_w, kh*kw, D]`
///
/// Returns one tensor per image.
pub fn patch_merger(
    x: &Tensor,
    grid_hws: &[[usize; 2]],
    merge_kernel_size: [usize; 2],
) -> Result<Vec<Tensor>> {
    let d = x.dim(1)?;
    let [kh, kw] = merge_kernel_size;
    let mut outputs = Vec::with_capacity(grid_hws.len());
    let mut offset = 0usize;

    for &[h, w] in grid_hws {
        let seq = x.narrow(0, offset, h * w)?; // [h*w, D]
        let new_h = h / kh;
        let new_w = w / kw;
        // [h*w, D] → [new_h, kh, new_w, kw, D] → permute(0,2,1,3,4) → [new_h*new_w, kh*kw, D]
        let merged = seq
            .reshape((new_h, kh, new_w, kw, d))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((new_h * new_w, kh * kw, d))?;
        outputs.push(merged);
        offset += h * w;
    }

    Ok(outputs)
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// MoonViT pretrained vision encoder.
pub struct MoonVitPretrainedModel {
    patch_embed: MoonVisionPatchEmbed,
    encoder: MoonVitEncoder,
    pub merge_kernel_size: [usize; 2],
    pub cfg: MoonVitConfig,
}

impl MoonVitPretrainedModel {
    pub fn new(cfg: &MoonVitConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = MoonVisionPatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let encoder = MoonVitEncoder::new(cfg, vb.pp("encoder"))?;
        Ok(Self {
            patch_embed,
            encoder,
            merge_kernel_size: cfg.merge_kernel_size,
            cfg: cfg.clone(),
        })
    }

    /// Encode pixel patches through embed → transformer → patch merger.
    ///
    /// `pixel_values`: `[L, C, ps, ps]` — pre-extracted patches for all images.
    /// `grid_hws`: patch-grid dimensions `(h, w)` per image.
    ///
    /// Returns one tensor per image: `[new_h*new_w, kh*kw, hidden]`.
    pub fn forward(&self, pixel_values: &Tensor, grid_hws: &[[usize; 2]]) -> Result<Vec<Tensor>> {
        let x = self.patch_embed.forward(pixel_values, grid_hws)?; // [L, D]
        let x = self.encoder.forward(&x, grid_hws)?; // [L, D]
        patch_merger(&x, grid_hws, self.merge_kernel_size)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_cfg() -> MoonVitConfig {
        // Tiny config for fast CPU tests
        // head_dim = 8, head_dim/4 = 2 frequencies → head_dim%4==0 ✓
        MoonVitConfig {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2, // head_dim = 8
            intermediate_size: 32,
            in_channels: 1,
            patch_size: 2,
            init_pos_emb_height: 4,
            init_pos_emb_width: 4,
            merge_kernel_size: [2, 2],
            rope_max_height: 16,
            rope_max_width: 16,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_moonvit_new() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MoonVitPretrainedModel::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "MoonVitPretrainedModel construction failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_rope2d_freqs_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let head_dim = cfg.head_dim(); // 8
        let rope = Rope2DPosEmb::new(head_dim, 4, 4, 10000.0, &device).unwrap();
        // Grid 2×2 → 4 patches; head_dim/2 = 4
        let (cos, sin) = rope.get_freqs_by_seqlens(&[[2, 2]]).unwrap();
        assert_eq!(cos.dims(), &[4, head_dim / 2]);
        assert_eq!(sin.dims(), &[4, head_dim / 2]);
        // Two images: 2×2 + 2×2 → total 8 patches
        let (cos2, _) = rope.get_freqs_by_seqlens(&[[2, 2], [2, 2]]).unwrap();
        assert_eq!(cos2.dim(0).unwrap(), 8);
    }

    #[test]
    fn test_apply_rope2d_shape() {
        let device = Device::Cpu;
        let s = 4usize;
        let heads = 2usize;
        let head_dim = 8usize;
        let half = head_dim / 2;
        let q = Tensor::zeros((s, heads, head_dim), DType::F32, &device).unwrap();
        let k = Tensor::zeros((s, heads, head_dim), DType::F32, &device).unwrap();
        let cos = Tensor::ones((s, half), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((s, half), DType::F32, &device).unwrap();
        // Identity rope (cos=1, sin=0) should leave tensors unchanged
        let (q_out, k_out) = apply_rope_2d(&q, &k, &cos, &sin).unwrap();
        assert_eq!(q_out.dims(), q.dims());
        assert_eq!(k_out.dims(), k.dims());
    }

    #[test]
    fn test_patch_merger_shape() {
        let device = Device::Cpu;
        // 4 patches (2×2 grid), D=16, merge (2,2)
        let x = Tensor::zeros((4usize, 16usize), DType::F32, &device).unwrap();
        let grid_hws = [[2usize, 2usize]];
        let out = patch_merger(&x, &grid_hws, [2, 2]).unwrap();
        assert_eq!(out.len(), 1);
        // new_h*new_w = 1×1 = 1, kh*kw = 4, D = 16
        assert_eq!(out[0].dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_moonvit_forward_shape() {
        let device = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MoonVitPretrainedModel::new(&cfg, vb).unwrap();

        // One image: 4×4 patch grid = 16 patches; patch_size=2, in_channels=1
        let ps = cfg.patch_size;
        let pixel_values =
            Tensor::zeros((16usize, cfg.in_channels, ps, ps), DType::F32, &device).unwrap();
        let grid_hws = [[4usize, 4usize]];

        let out = model.forward(&pixel_values, &grid_hws).unwrap();
        assert_eq!(out.len(), 1);
        // After merge (2,2): new_h*new_w = 2×2 = 4, kh*kw = 4, D = 16
        assert_eq!(out[0].dims(), &[4, 4, cfg.hidden_size]);
    }
}
