//! Chameleon unified token model implementation.
//!
//! Architecture:
//! - Image tokenization: Frozen VQVAE encoder + vector quantizer
//! - Unified vocabulary: Images encoded as discrete tokens in shared BPE space
//! - LLM: Llama-like transformer with Q/K RMSNorm per head
//! - Optional Swin-style layer normalization
//!
//! ## Image handling
//!
//! Chameleon tokenizes images into discrete BPE tokens via a VQVAE encoder.
//! In the standard vLLM flow, the Python multimodal preprocessor runs the VQVAE
//! before inference, producing image BPE token IDs that are embedded in `input_ids`
//! alongside text tokens.  The LLM then processes them uniformly.
//!
//! This Rust implementation loads all VQVAE weights (`model.vqmodel.*`) and
//! implements the full encoding pipeline (`ChameleonVQVAE::encode`).  The
//! vocabulary mapping (`vocabulary_map` in `config.json`) maps VQGAN codebook
//! indices to BPE token IDs; an `Option`-wrapped `ChameleonVocabMapping` is
//! built when the map is present in `ModelConfig.extra`.
//!
//! ## Weight paths
//! - `model.vqmodel.encoder.*`   — VQVAE convolutional encoder
//! - `model.vqmodel.quantize.*`  — vector quantizer codebook (`embedding.weight`)
//! - `model.embed_tokens.*`      — shared text+image BPE embedding
//! - `model.layers.*`            — Llama-style transformer layers
//! - `lm_head.*`                 — output projection
//!
//! Reference: reference/vllm/vllm/model_executor/models/chameleon.py

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{causal_mask, paged_attention, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ChameleonConfig {
    model_config: ModelConfig,
    swin_norm: bool,
    logit_scale: f64,
}

impl ChameleonConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let swin_norm = cfg
            .extra
            .get("swin_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let logit_scale = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            model_config: cfg.clone(),
            swin_norm,
            logit_scale,
        }
    }
}

// ─── Per-Head RMSNorm (ChameleonLayerNorm) ──────────────────────────────────

/// Per-head RMSNorm applied to Q and K projections.
///
/// Unlike standard RMSNorm, this applies normalization per head independently.
struct ChameleonQKNorm {
    norms: Vec<RmsNorm>,
    num_heads: usize,
    head_dim: usize,
}

impl ChameleonQKNorm {
    fn new(num_heads: usize, head_dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let mut norms = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            norms.push(rms_norm(head_dim, eps, vb.pp(i))?);
        }
        Ok(Self {
            norms,
            num_heads,
            head_dim,
        })
    }

    /// Apply per-head normalization.
    ///
    /// Input shape: (batch, seq_len, num_heads * head_dim)
    /// Output shape: same
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Reshape to (batch, seq_len, num_heads, head_dim)
        let x = x.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        let mut head_outputs = Vec::with_capacity(self.num_heads);
        for (h, norm) in self.norms.iter().enumerate() {
            // Extract head h: (batch, seq_len, head_dim)
            let head = x.narrow(2, h, 1)?.squeeze(2)?;
            let normed = norm.forward(&head)?;
            head_outputs.push(normed.unsqueeze(2)?);
        }

        let result = Tensor::cat(&head_outputs, 2)?;
        result.reshape((batch, seq_len, self.num_heads * self.head_dim))
    }
}

// ─── Attention ──────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct ChameleonAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: ChameleonQKNorm,
    k_norm: ChameleonQKNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

impl ChameleonAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let total_qkv = (num_heads + 2 * num_kv_heads) * head_dim;
        let qkv_proj = candle_nn::linear(cfg.hidden_size, total_qkv, vb.pp("qkv_proj"))?;
        let o_proj = candle_nn::linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = ChameleonQKNorm::new(num_heads, head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm =
            ChameleonQKNorm::new(num_kv_heads, head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let rotary = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups: num_heads / num_kv_heads,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;

        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_dim)?;
        let k = qkv.narrow(2, q_dim, kv_dim)?;
        let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?;

        // Per-head Q/K normalization
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Reshape: (batch, seq_len, heads, head_dim) → (batch, heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // RoPE
        let (q, k) = self.rotary.apply(&q, &k, seqlen_offset)?;

        // Paged attention with KV cache
        paged_attention(
            &q,
            &k,
            &v,
            mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
    }
}

// ─── MLP ────────────────────────────────────────────────────────────────────

/// SwiGLU MLP: gate_up_proj → silu(gate) * up → down_proj
struct ChameleonMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl ChameleonMLP {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = candle_nn::linear(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj =
            candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(2, 0, self.intermediate_size)?;
        let up = gate_up.narrow(2, self.intermediate_size, self.intermediate_size)?;
        let x = (candle_nn::Activation::Silu.forward(&gate)? * up)?;
        self.down_proj.forward(&x)
    }
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct ChameleonDecoderLayer {
    self_attn: ChameleonAttention,
    mlp: ChameleonMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl ChameleonDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = ChameleonAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = ChameleonMLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(
            &x,
            mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let x = (residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

// ─── VQVAE ──────────────────────────────────────────────────────────────────

/// Configuration parsed from `config.extra["vq_config"]`.
///
/// Mirrors `ChameleonVQVAEConfig` in HuggingFace transformers.
#[derive(Debug, Clone)]
struct VqVaeConfig {
    /// Number of codebook entries (default 8192).
    num_embeddings: usize,
    /// Codebook entry dimension (default 256).
    embed_dim: usize,
    /// Latent channels at encoder output (default 256).
    latent_channels: usize,
    /// Input image resolution (default 512).
    resolution: usize,
    /// Input channels (3 for RGB).
    in_channels: usize,
    /// Base channel count (default 128).
    base_channels: usize,
    /// Per-resolution channel multipliers (default [1,1,2,2,4]).
    channel_multiplier: Vec<usize>,
    /// ResNet blocks per resolution (default 2).
    num_res_blocks: usize,
    /// Resolutions at which attention blocks are added (default []).
    attn_resolutions: Vec<usize>,
    /// Whether to use vanilla attention or none (default "vanilla").
    use_attn: bool,
}

impl VqVaeConfig {
    fn from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        let vq = extra.get("vq_config")?.as_object()?;
        let get_usize = |key: &str, default: usize| -> usize {
            vq.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let channel_multiplier: Vec<usize> = vq
            .get("channel_multiplier")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_else(|| vec![1, 1, 2, 2, 4]);
        let attn_resolutions: Vec<usize> = vq
            .get("attn_resolutions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        let attn_type = vq
            .get("attn_type")
            .and_then(|v| v.as_str())
            .unwrap_or("vanilla");
        Some(Self {
            num_embeddings: get_usize("num_embeddings", 8192),
            embed_dim: get_usize("embed_dim", 256),
            latent_channels: get_usize("latent_channels", 256),
            resolution: get_usize("resolution", 512),
            in_channels: get_usize("in_channels", 3),
            base_channels: get_usize("base_channels", 128),
            channel_multiplier,
            num_res_blocks: get_usize("num_res_blocks", 2),
            attn_resolutions,
            use_attn: attn_type == "vanilla",
        })
    }
}

/// GroupNorm for NCHW tensors — applies `candle_nn::GroupNorm`.
fn vq_group_norm(num_channels: usize, vb: VarBuilder) -> Result<GroupNorm> {
    candle_nn::group_norm(32, num_channels, 1e-6, vb)
}

/// SiLU: x * sigmoid(x).
#[allow(dead_code)]
fn silu(x: &Tensor) -> Result<Tensor> {
    x * candle_nn::ops::sigmoid(x)?
}

/// Conv2d wrapper with standard defaults.
fn vq_conv(
    in_ch: usize,
    out_ch: usize,
    ks: usize,
    stride: usize,
    padding: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        stride,
        padding,
        ..Default::default()
    };
    candle_nn::conv2d(in_ch, out_ch, ks, cfg, vb)
}

/// ResNet block: GroupNorm + SiLU + Conv + GroupNorm + SiLU + Conv + residual.
///
/// Mirrors `ChameleonVQVAEEncoderResnetBlock` (no dropout at inference).
#[allow(dead_code)]
struct VqResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

#[allow(dead_code)]
impl VqResnetBlock {
    fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = vq_group_norm(in_channels, vb.pp("norm1"))?;
        let conv1 = vq_conv(in_channels, out_channels, 3, 1, 1, vb.pp("conv1"))?;
        let norm2 = vq_group_norm(out_channels, vb.pp("norm2"))?;
        let conv2 = vq_conv(out_channels, out_channels, 3, 1, 1, vb.pp("conv2"))?;
        let shortcut = if in_channels != out_channels {
            // nin_shortcut: 1x1 conv (no conv_shortcut path used in Chameleon encoder)
            Some(vq_conv(
                in_channels,
                out_channels,
                1,
                1,
                0,
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = silu(&self.norm1.forward(x)?)?;
        let h = self.conv1.forward(&h)?;
        let h = silu(&self.norm2.forward(&h)?)?;
        let h = self.conv2.forward(&h)?;
        match &self.shortcut {
            Some(sc) => h + sc.forward(residual)?,
            None => h + residual,
        }
    }
}

/// Attention block for high-resolution feature maps.
///
/// Mirrors `ChameleonVQVAEEncoderAttnBlock`: GroupNorm + Q/K/V 1×1 convs
/// + scaled dot-product attention + proj_out + residual.
#[allow(dead_code)]
struct VqAttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

#[allow(dead_code)]
impl VqAttnBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm = vq_group_norm(channels, vb.pp("norm"))?;
        let q = vq_conv(channels, channels, 1, 1, 0, vb.pp("q"))?;
        let k = vq_conv(channels, channels, 1, 1, 0, vb.pp("k"))?;
        let v = vq_conv(channels, channels, 1, 1, 0, vb.pp("v"))?;
        let proj_out = vq_conv(channels, channels, 1, 1, 0, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.norm.forward(x)?;
        let q = self.q.forward(&h)?;
        let k = self.k.forward(&h)?;
        let v = self.v.forward(&h)?;

        let (b, c, h_size, w_size) = q.dims4()?;
        let hw = h_size * w_size;

        // q: [B, HW, C]; k: [B, C, HW]
        let q = q.reshape((b, c, hw))?.permute((0, 2, 1))?;
        let k = k.reshape((b, c, hw))?;

        // attention weights [B, HW, HW], scaled
        let scale = (c as f64).powf(-0.5);
        let attn = q.matmul(&k)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&attn, 2)?;

        // attend to values: v [B, C, HW] → out [B, C, H, W]
        let v = v.reshape((b, c, hw))?;
        let attn_t = attn.permute((0, 2, 1))?;
        let out = v.matmul(&attn_t)?;
        let out = out.reshape((b, c, h_size, w_size))?;

        let out = self.proj_out.forward(&out)?;
        out + residual
    }
}

/// Stride-2 downsampler with asymmetric zero-padding.
///
/// Mirrors `ChameleonVQVAEEncoderConvDownsample`: pad right/bottom by 1,
/// then Conv2d(kernel=3, stride=2, padding=0).
#[allow(dead_code)]
struct VqDownsample {
    conv: Conv2d,
}

#[allow(dead_code)]
impl VqDownsample {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv = vq_conv(channels, channels, 3, 2, 0, vb.pp("conv"))?;
        Ok(Self { conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Asymmetric padding: (left=0, right=1, top=0, bottom=1)
        let x = x.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?;
        self.conv.forward(&x)
    }
}

/// One resolution stage in the encoder: `num_res_blocks` ResNet blocks,
/// optional attention blocks, and (if not the last stage) a downsampler.
#[allow(dead_code)]
struct VqDownStage {
    blocks: Vec<VqResnetBlock>,
    attn: Vec<VqAttnBlock>,
    downsample: Option<VqDownsample>,
}

#[allow(dead_code)]
impl VqDownStage {
    fn forward(&self, mut x: Tensor) -> Result<Tensor> {
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if i < self.attn.len() {
                x = self.attn[i].forward(&x)?;
            }
        }
        if let Some(ds) = &self.downsample {
            x = ds.forward(&x)?;
        }
        Ok(x)
    }
}

/// VQVAE encoder: multi-scale convolutional encoder.
///
/// Mirrors `ChameleonVQVAEEncoder`.  Maps pixel values `[B, C, H, W]` →
/// latent codes `[B, latent_channels, H/2^(num_res-1), W/2^(num_res-1)]`.
struct VqEncoder {
    conv_in: Conv2d,
    down: Vec<VqDownStage>,
    mid_block1: VqResnetBlock,
    mid_attn: Option<VqAttnBlock>,
    mid_block2: VqResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl VqEncoder {
    fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let conv_in = vq_conv(
            cfg.in_channels,
            cfg.base_channels,
            3,
            1,
            1,
            vb.pp("conv_in"),
        )?;

        let num_res = cfg.channel_multiplier.len();
        let in_ch_mult: Vec<usize> = std::iter::once(1)
            .chain(cfg.channel_multiplier.iter().copied())
            .collect();

        let vb_down = vb.pp("down");
        let mut down = Vec::with_capacity(num_res);
        let mut curr_res = cfg.resolution;
        let mut block_in = cfg.base_channels;

        for (i_level, &in_ch_m) in in_ch_mult.iter().enumerate().take(num_res) {
            let block_out = cfg.base_channels * cfg.channel_multiplier[i_level];
            let block_in_start = cfg.base_channels * in_ch_m;

            let vb_level = vb_down.pp(i_level);
            let vb_block = vb_level.pp("block");
            let vb_attn = vb_level.pp("attn");

            let mut blocks = Vec::with_capacity(cfg.num_res_blocks);
            let mut attn_blocks = Vec::new();

            let mut current_in = block_in_start;
            for i_block in 0..cfg.num_res_blocks {
                blocks.push(VqResnetBlock::new(
                    current_in,
                    block_out,
                    vb_block.pp(i_block),
                )?);
                current_in = block_out;
                if cfg.use_attn && cfg.attn_resolutions.contains(&curr_res) {
                    attn_blocks.push(VqAttnBlock::new(block_out, vb_attn.pp(i_block))?);
                }
            }

            let downsample = if i_level < num_res - 1 {
                curr_res /= 2;
                Some(VqDownsample::new(block_out, vb_level.pp("downsample"))?)
            } else {
                None
            };

            block_in = block_out;
            down.push(VqDownStage {
                blocks,
                attn: attn_blocks,
                downsample,
            });
        }

        let vb_mid = vb.pp("mid");
        let mid_block1 = VqResnetBlock::new(block_in, block_in, vb_mid.pp("block_1"))?;
        let mid_attn = if cfg.use_attn {
            Some(VqAttnBlock::new(block_in, vb_mid.pp("attn_1"))?)
        } else {
            None
        };
        let mid_block2 = VqResnetBlock::new(block_in, block_in, vb_mid.pp("block_2"))?;

        let norm_out = vq_group_norm(block_in, vb.pp("norm_out"))?;
        let conv_out = vq_conv(block_in, cfg.latent_channels, 3, 1, 1, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            down,
            mid_block1,
            mid_attn,
            mid_block2,
            norm_out,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for stage in &self.down {
            h = stage.forward(h)?;
        }
        h = self.mid_block1.forward(&h)?;
        if let Some(attn) = &self.mid_attn {
            h = attn.forward(&h)?;
        }
        h = self.mid_block2.forward(&h)?;
        h = silu(&self.norm_out.forward(&h)?)?;
        self.conv_out.forward(&h)
    }
}

/// Vector quantizer: nearest-neighbour lookup in a learned codebook.
///
/// Mirrors `ChameleonVQVAEVectorQuantizer`.  At inference (no gradients),
/// only the argmin indices are needed.
#[allow(dead_code)]
struct VqQuantizer {
    embedding: candle_nn::Embedding,
    embedding_dim: usize,
}

#[allow(dead_code)]
impl VqQuantizer {
    fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let embedding =
            candle_nn::embedding(cfg.num_embeddings, cfg.embed_dim, vb.pp("embedding"))?;
        Ok(Self {
            embedding,
            embedding_dim: cfg.embed_dim,
        })
    }

    /// Quantize `z` and return the flat codebook indices.
    ///
    /// `z` shape: `[B, embed_dim, H, W]`
    /// Returns: `[B * H * W]` u32 codebook indices.
    fn encode(&self, z: &Tensor) -> Result<Tensor> {
        // permute to [B, H, W, C] then flatten to [N, C]
        let z = z.permute((0, 2, 3, 1))?.contiguous()?;
        let (b, h, w, _c) = z.dims4()?;
        let z_flat = z.reshape((b * h * w, self.embedding_dim))?;

        // distances = z^2 + e^2 - 2*z*e^T
        let e = self.embedding.embeddings(); // [K, D]
        let z_sq = z_flat.sqr()?.sum_keepdim(1)?; // [N, 1]
        let e_sq = e.sqr()?.sum_keepdim(1)?.t()?; // [1, K]
        let ze = z_flat.matmul(&e.t()?)?; // [N, K]
        let dists = (z_sq.broadcast_add(&e_sq)? - ze.affine(2.0, 0.0)?)?;

        dists.argmin_keepdim(1)?.squeeze(1)?.to_dtype(DType::U32)
    }
}

/// Chameleon VQVAE: encoder + vector quantizer.
///
/// Weight prefix in saved checkpoint: `model.vqmodel.*`.
pub(crate) struct ChameleonVQVAE {
    #[allow(dead_code)]
    encoder: VqEncoder,
    #[allow(dead_code)]
    quantize: VqQuantizer,
}

#[allow(dead_code)]
impl ChameleonVQVAE {
    fn new(cfg: &VqVaeConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = VqEncoder::new(cfg, vb.pp("encoder"))?;
        let quantize = VqQuantizer::new(cfg, vb.pp("quantize"))?;
        Ok(Self { encoder, quantize })
    }

    /// Encode pixel values to flat codebook indices.
    ///
    /// `pixel_values`: `[B, C, H, W]` normalized to `[-1, 1]`.
    /// Returns: `[B, H', W']` u32 codebook indices.
    pub(crate) fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let z = self.encoder.forward(pixel_values)?;
        let (b, _c, h, w) = z.dims4()?;
        let indices = self.quantize.encode(&z)?;
        indices.reshape((b, h, w))
    }
}

/// Vocabulary mapping from VQGAN codebook index → BPE token ID.
///
/// Built from `config.extra["vocabulary_map"]` which is a JSON object mapping
/// token names (`"IMGIMGABBA"`) to BPE integer IDs.
///
/// Name encoding: `"IMGIMG"` prefix + N chars (A=0, B=1, …, J=9) + trailing char.
/// Example: `"IMGIMGABBA"` → `"0110"` → codebook index 110.
pub(crate) struct ChameleonVocabMapping {
    /// `mapping[codebook_idx] = bpe_token_id`
    #[allow(dead_code)]
    mapping: Vec<u32>,
}

#[allow(dead_code)]
impl ChameleonVocabMapping {
    fn from_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        let vocab_map = extra.get("vocabulary_map")?.as_object()?;
        if vocab_map.is_empty() {
            return None;
        }

        let char_to_digit = |c: char| -> Option<u32> {
            if ('A'..='J').contains(&c) {
                Some(c as u32 - 'A' as u32)
            } else {
                None
            }
        };

        let mut pairs: Vec<(usize, u32)> = Vec::new();
        for (name, val) in vocab_map {
            if !name.starts_with("IMGIMG") {
                continue;
            }
            let bpe_id = val.as_u64()? as u32;
            // Decode: strip "IMGIMG" prefix and trailing char
            let encoded = &name[6..name.len().saturating_sub(1)];
            let mut codebook_idx: u32 = 0;
            for c in encoded.chars() {
                let d = char_to_digit(c)?;
                codebook_idx = codebook_idx * 10 + d;
            }
            pairs.push((codebook_idx as usize, bpe_id));
        }
        if pairs.is_empty() {
            return None;
        }

        let max_idx = pairs.iter().map(|(i, _)| *i).max()?;
        let mut mapping = vec![0u32; max_idx + 1];
        for (idx, bpe_id) in pairs {
            mapping[idx] = bpe_id;
        }
        Some(Self { mapping })
    }

    /// Convert a batch of codebook indices to BPE token IDs (CPU).
    pub(crate) fn convert(&self, indices: &[u32]) -> Vec<u32> {
        indices
            .iter()
            .map(|&i| self.mapping.get(i as usize).copied().unwrap_or(0))
            .collect()
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Chameleon unified token model.
///
/// Uses VQVAE to tokenize images into discrete tokens within a shared vocabulary.
/// The LLM is Llama-like with per-head Q/K RMSNorm on attention projections.
pub struct ChameleonForConditionalGeneration {
    embed_tokens: TpEmbedding,
    layers: Vec<ChameleonDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    #[allow(dead_code)]
    config: ChameleonConfig,
    #[allow(dead_code)]
    logit_scale: f64,
    /// VQVAE encoder for converting raw pixel values to codebook indices.
    /// Present when `vq_config` is available in the model config.
    #[allow(dead_code)]
    vqvae: Option<ChameleonVQVAE>,
    /// Vocabulary mapping from VQGAN codebook index to BPE token ID.
    #[allow(dead_code)]
    vocab_mapping: Option<ChameleonVocabMapping>,
    device: Device,
    dtype: DType,
}

impl ChameleonForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = ChameleonConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(ChameleonDecoderLayer::new(cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        // Load VQVAE if vq_config is present in model config
        let vq_cfg = VqVaeConfig::from_extra(&cfg.extra);
        let vqvae = vq_cfg
            .as_ref()
            .map(|c| ChameleonVQVAE::new(c, vb_m.pp("vqmodel")))
            .transpose()?;
        let vocab_mapping = ChameleonVocabMapping::from_extra(&cfg.extra);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            logit_scale: config.logit_scale,
            config,
            vqvae,
            vocab_mapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        _multimodal_inputs: Option<&MultimodalInputs>,
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

        // In Chameleon, images are already tokenized as discrete BPE tokens
        // via VQVAE + vocabulary mapping, so they go through embed_tokens directly.
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
        }

        let xs = self.norm.forward(&xs)?;

        // Apply logit scaling
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        if (self.logit_scale - 1.0).abs() > 1e-6 {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
    }
}

impl crate::engine::ModelForward for ChameleonForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            None,
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut batch_outputs = Vec::with_capacity(sequences.len());
            let cache_engine = kv_cache_mgr.engine_mut(layer_idx);
            for (seq_idx, seq_meta) in sequences.iter().enumerate() {
                let x_single = xs.narrow(0, seq_idx, 1)?;
                let block_table = BlockTable::from_block_ids(seq_meta.block_ids.clone(), 0);
                let out = layer.forward(
                    &x_single,
                    None,
                    seq_meta.seqlen_offset,
                    cache_engine,
                    &block_table,
                    &seq_meta.slot_mapping,
                )?;
                batch_outputs.push(out);
            }
            xs = Tensor::cat(&batch_outputs, 0)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        if (self.logit_scale - 1.0).abs() > 1e-6 {
            logits * self.logit_scale
        } else {
            Ok(logits)
        }
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
        // Chameleon images are discrete BPE tokens produced by the VQVAE encoder.
        //
        // Standard flow: Python multimodal preprocessor runs ChameleonVQVAE
        // → image BPE token IDs are already embedded in `input_ids` alongside
        // text tokens.  No additional multimodal processing is needed here —
        // `forward_inner` handles them like regular text tokens.
        //
        // Native-Rust flow: if raw pixel tensors are provided via
        // `MultimodalInputs`, the caller would need to encode them using
        // `self.vqvae.encode(pixel_tensor)` + `self.vocab_mapping.convert()`,
        // then substitute the resulting BPE token IDs into `input_ids` at the
        // image placeholder positions before calling `forward_inner`.
        // This path is not currently wired to the server's multimodal processor
        // since the standard vLLM pipeline always pre-tokenizes.
        self.forward_inner(
            input_ids,
            multimodal_inputs,
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("swin_norm".to_string(), serde_json::json!(false));
        extra.insert("logit_scale".to_string(), serde_json::json!(1.0));

        ModelConfig {
            architectures: vec!["ChameleonForConditionalGeneration".to_string()],
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ── VQVAE tests ──────────────────────────────────────────────────────────

    // GroupNorm(32) requires channels divisible by 32.
    fn make_tiny_vqcfg() -> VqVaeConfig {
        VqVaeConfig {
            num_embeddings: 8,
            embed_dim: 4,
            latent_channels: 4, // conv_out output — no GroupNorm on this
            resolution: 8,
            in_channels: 3,
            base_channels: 32, // must be divisible by num_groups=32
            channel_multiplier: vec![1, 2],
            num_res_blocks: 1,
            attn_resolutions: vec![],
            use_attn: false,
        }
    }

    #[test]
    fn test_vq_resnet_block() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        // channels must be divisible by num_groups=32
        let block = VqResnetBlock::new(32, 64, vb).unwrap();
        let x = Tensor::zeros((1, 32, 4, 4), DType::F32, &device).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 64, 4, 4]);
    }

    #[test]
    fn test_vq_encoder_shape() {
        let device = Device::Cpu;
        let cfg = make_tiny_vqcfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = VqEncoder::new(&cfg, vb).unwrap();
        // Resolution 8, channel_multiplier [1, 2] → 1 downsample stage → H/2=4
        let x = Tensor::zeros((1, 3, 8, 8), DType::F32, &device).unwrap();
        let z = encoder.forward(&x).unwrap();
        // conv_out maps to latent_channels; resolution halved once → 4×4
        assert_eq!(z.dim(1).unwrap(), cfg.latent_channels);
        assert_eq!(z.dim(2).unwrap(), 4);
        assert_eq!(z.dim(3).unwrap(), 4);
    }

    #[test]
    fn test_vq_quantizer_indices_in_range() {
        let device = Device::Cpu;
        let cfg = make_tiny_vqcfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let q = VqQuantizer::new(&cfg, vb).unwrap();
        // z [1, embed_dim=4, 2, 2]
        let z = Tensor::randn(0.0f32, 1.0, (1, 4, 2, 2), &device).unwrap();
        let idx = q.encode(&z).unwrap();
        assert_eq!(idx.dims(), &[4]); // 1 * 2 * 2 flattened
        let idx_vec = idx.to_vec1::<u32>().unwrap();
        for i in idx_vec {
            assert!((i as usize) < cfg.num_embeddings);
        }
    }

    #[test]
    fn test_vqvae_encode_shape() {
        let device = Device::Cpu;
        let cfg = make_tiny_vqcfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let vqvae = ChameleonVQVAE::new(&cfg, vb).unwrap();
        let pixels = Tensor::zeros((1, 3, 8, 8), DType::F32, &device).unwrap();
        let indices = vqvae.encode(&pixels).unwrap();
        // Expected: [B=1, H'=4, W'=4]
        assert_eq!(indices.dims(), &[1, 4, 4]);
    }

    #[test]
    fn test_vocab_mapping_from_extra() {
        // Build a minimal vocabulary_map with 4 IMGIMG entries
        let mut vocab_map = serde_json::Map::new();
        // "IMGIMGAAA A" → codebook index 0 → BPE token 1000
        vocab_map.insert("IMGIMGAAAA".to_string(), serde_json::json!(1000u32));
        vocab_map.insert("IMGIMGAABA".to_string(), serde_json::json!(1001u32));
        vocab_map.insert("IMGIMGAACA".to_string(), serde_json::json!(1002u32));
        // non-IMGIMG entry should be ignored
        vocab_map.insert("<image>".to_string(), serde_json::json!(9));

        let mut extra = serde_json::Map::new();
        extra.insert(
            "vocabulary_map".to_string(),
            serde_json::Value::Object(vocab_map),
        );

        let mapping = ChameleonVocabMapping::from_extra(&extra).unwrap();
        // IMGIMGAAAA → "AA" = "00" → idx 0, bpe 1000
        assert_eq!(mapping.convert(&[0]), vec![1000]);
        // IMGIMGAABA → "AB" = "01" → idx 1, bpe 1001
        assert_eq!(mapping.convert(&[1]), vec![1001]);
        // IMGIMGAACA → "AC" = "02" → idx 2, bpe 1002
        assert_eq!(mapping.convert(&[2]), vec![1002]);
    }

    #[test]
    fn test_model_with_vq_config() {
        let device = Device::Cpu;
        let mut cfg = test_model_config();
        // Add a minimal vq_config to exercise VQVAE loading
        cfg.extra.insert(
            "vq_config".to_string(),
            serde_json::json!({
                "num_embeddings": 8,
                "embed_dim": 4,
                "latent_channels": 4,
                "resolution": 8,
                "in_channels": 3,
                "base_channels": 32,
                "channel_multiplier": [1, 2],
                "num_res_blocks": 1,
                "attn_type": "none",
            }),
        );
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "construction failed with vq_config: {:?}",
            model.err()
        );
        let m = model.unwrap();
        assert!(
            m.vqvae.is_some(),
            "VQVAE should be loaded when vq_config present"
        );
    }

    // ── LLM tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_qk_norm() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let norm = ChameleonQKNorm::new(4, 16, 1e-6, vb).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (1, 3, 64), &device).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 3, 64]);
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = ChameleonForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }
}
