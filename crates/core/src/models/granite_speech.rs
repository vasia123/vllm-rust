//! IBM Granite Speech: Conformer CTC encoder + BLIP2 QFormer projector + Granite LLM.
//!
//! Architecture:
//! ```text
//! mel [B, T, 160] (Mel-spectrogram features)
//!   └── GraniteSpeechCTCEncoder (input_linear → N conformer blocks → hidden)
//!         → [B, T, hidden_dim=512]
//!   └── GraniteSpeechEncoderProjector (QFormer windowed cross-attn → linear)
//!         → [B, num_audio_tokens, text_hidden]
//! text_ids → GraniteForCausalLM (embed)
//!   scatter audio embeddings at <|audio|> positions
//!   → Granite transformer → logits
//! ```
//!
//! ## Conformer block (Macaron-style)
//! `0.5*FF1 + self + Attn + self + Conv + self + 0.5*FF2 + self + PostNorm`
//! - Attention: Shaw's relative position embeddings, block attention (context_size=160)
//! - Conv: LayerNorm → up_conv → GLU → depthwise → BatchNorm1d → SiLU → down_conv
//!
//! ## Mid-layer CTC auxiliary
//! At layer `num_layers // 2`: `hidden += out_mid(softmax(out(hidden)))`
//!
//! ## QFormer projector
//! Windowed processing (window_size=160 frames per block):
//! 1. Pad encoder output to multiple of window_size
//! 2. Reshape to `[B*nblocks, window_size, hidden_dim]`
//! 3. Cross-attend with learnable queries `[1, num_queries, qformer_hidden]`
//! 4. Project to `[B, nblocks*num_queries, text_hidden]`
//!
//! ## Weight paths
//! - `encoder.*` → GraniteSpeechCTCEncoder
//! - `projector.query` → learnable query
//! - `projector.qformer.*` → BLIP2-style QFormer
//! - `projector.linear.*` → final projection
//! - `language_model.*` → GraniteForCausalLM
//!
//! The encoder and projector are used at audio pre-processing time.
//! At model-forward time the pre-encoded `ProcessedAudio` embeddings are scattered in.

use candle_core::{Device, Result, Tensor};
use candle_nn::{
    batch_norm, conv1d, embedding, layer_norm, linear, ops, BatchNormConfig, Conv1dConfig,
    Embedding, LayerNorm, Linear, Module, ModuleT, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::granite::GraniteForCausalLM;

// ─── Encoder Config ───────────────────────────────────────────────────────────

struct EncoderCfg {
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    dim_head: usize,
    max_pos_emb: usize,
    context_size: usize,
    input_dim: usize,
    output_dim: usize,
    feedforward_mult: usize,
    conv_expansion_factor: usize,
    conv_kernel_size: usize,
}

impl EncoderCfg {
    fn from_json(v: &serde_json::Value) -> Self {
        let g = |key, default: usize| {
            v.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(default)
        };
        Self {
            hidden_dim: g("hidden_dim", 512),
            num_layers: g("num_layers", 17),
            num_heads: g("num_heads", 4),
            dim_head: g("dim_head", 128),
            max_pos_emb: g("max_pos_emb", 160),
            context_size: g("context_size", 160),
            input_dim: g("input_dim", 160),
            output_dim: g("output_dim", 4608),
            feedforward_mult: g("feedforward_mult", 4),
            conv_expansion_factor: g("conv_expansion_factor", 2),
            conv_kernel_size: g("conv_kernel_size", 31),
        }
    }
}

// ─── Projector Config ─────────────────────────────────────────────────────────

struct ProjectorCfg {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    encoder_hidden_size: usize, // = encoder hidden_dim
    cross_attention_frequency: usize,
    layer_norm_eps: f64,
    // from top-level config
    downsample_rate: usize,
    window_size: usize,
    text_hidden_size: usize,
}

impl ProjectorCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;
        let proj = extra.get("projector_config").cloned().unwrap_or_default();
        let enc = extra.get("encoder_config").cloned().unwrap_or_default();

        let gp = |key, default: usize| {
            proj.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let encoder_hidden_dim = enc
            .get("hidden_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(512);

        Self {
            hidden_size: gp("hidden_size", 1024),
            num_hidden_layers: gp("num_hidden_layers", 2),
            num_attention_heads: gp("num_attention_heads", 16),
            intermediate_size: gp("intermediate_size", 4096),
            encoder_hidden_size: encoder_hidden_dim,
            cross_attention_frequency: gp("cross_attention_frequency", 1),
            layer_norm_eps: proj
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-12),
            downsample_rate: extra
                .get("downsample_rate")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(8),
            window_size: extra
                .get("window_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(160),
            text_hidden_size: cfg.hidden_size,
        }
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn num_queries(&self) -> usize {
        self.window_size / self.downsample_rate
    }
}

// ─── Conformer Feed-Forward ───────────────────────────────────────────────────

#[allow(dead_code)]
struct ConformerFFN {
    pre_norm: LayerNorm,
    up_proj: Linear,
    down_proj: Linear,
}

#[allow(dead_code)]
impl ConformerFFN {
    fn new(cfg: &EncoderCfg, vb: VarBuilder) -> Result<Self> {
        let pre_norm = layer_norm(cfg.hidden_dim, 1e-5, vb.pp("pre_norm"))?;
        let up_proj = linear(
            cfg.hidden_dim,
            cfg.hidden_dim * cfg.feedforward_mult,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear(
            cfg.hidden_dim * cfg.feedforward_mult,
            cfg.hidden_dim,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            pre_norm,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.pre_norm.forward(xs)?;
        let xs = self.up_proj.forward(&xs)?;
        let xs = xs.silu()?;
        self.down_proj.forward(&xs)
    }
}

// ─── Conformer Attention (Shaw's relative positions, block attention) ──────────

#[allow(dead_code)]
struct ConformerAttention {
    pre_norm: LayerNorm,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    rel_pos_emb: Embedding,
    num_heads: usize,
    dim_head: usize,
    context_size: usize,
    scale: f64,
}

#[allow(dead_code)]
impl ConformerAttention {
    fn new(cfg: &EncoderCfg, vb: VarBuilder) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.dim_head;
        let pre_norm = layer_norm(cfg.hidden_dim, 1e-5, vb.pp("pre_norm"))?;
        let to_q = linear(cfg.hidden_dim, inner_dim, vb.pp("to_q"))?;
        let to_kv = linear(cfg.hidden_dim, inner_dim * 2, vb.pp("to_kv"))?;
        let to_out = linear(inner_dim, cfg.hidden_dim, vb.pp("to_out"))?;
        let rel_pos_emb = embedding(2 * cfg.max_pos_emb + 1, cfg.dim_head, vb.pp("rel_pos_emb"))?;
        Ok(Self {
            pre_norm,
            to_q,
            to_kv,
            to_out,
            rel_pos_emb,
            num_heads: cfg.num_heads,
            dim_head: cfg.dim_head,
            context_size: cfg.context_size,
            scale: (cfg.dim_head as f64).powf(-0.5),
        })
    }

    /// `attention_dists`: precomputed `[context_size, context_size]` U32 relative distances.
    fn forward(&self, hidden_states: &Tensor, attention_dists: &Tensor) -> Result<Tensor> {
        let hidden_states = self.pre_norm.forward(hidden_states)?;
        let (bsz, num_features, _) = hidden_states.dims3()?;
        let num_blocks = num_features.div_ceil(self.context_size);
        let padded_len = num_blocks * self.context_size;
        let remainder = num_features % self.context_size;

        let hidden_states = if padded_len > num_features {
            let pad = Tensor::zeros(
                (bsz, padded_len - num_features, hidden_states.dim(2)?),
                hidden_states.dtype(),
                hidden_states.device(),
            )?;
            Tensor::cat(&[&hidden_states, &pad], 1)?
        } else {
            hidden_states.clone()
        };

        let q = self.to_q.forward(&hidden_states)?; // [B, T_padded, inner_dim]
        let kv = self.to_kv.forward(&hidden_states)?; // [B, T_padded, inner_dim*2]
        let inner_dim = self.num_heads * self.dim_head;
        let k = kv.narrow(2, 0, inner_dim)?;
        let v = kv.narrow(2, inner_dim, inner_dim)?;

        // Merge B and nb → single batch dim for 4D matmul support.
        // Reshape to [B*nb, nh, cs, dim_head].
        let reshape_qkv = |t: &Tensor| -> Result<Tensor> {
            t.reshape((
                bsz * num_blocks,
                self.context_size,
                self.num_heads,
                self.dim_head,
            ))?
            .transpose(1, 2)?
            .contiguous()
        };
        let q = reshape_qkv(&q)?; // [B*nb, nh, cs, dim_head]
        let k = reshape_qkv(&k)?;
        let v = reshape_qkv(&v)?;

        // Shaw's relative position bias:
        // attention_dists: [cs, cs] U32; rel_pos_emb: [2*max_pos_emb+1, dim_head]
        // Compute per-block position attention [B*nb, nh, cs, cs].
        let dist_flat = attention_dists.flatten_all()?; // [cs*cs]
        let rpe = self.rel_pos_emb.forward(&dist_flat)?; // [cs*cs, dim_head]
        let rpe = rpe.reshape((self.context_size, self.context_size, self.dim_head))?;
        // q: [B*nb, nh, cs, dim_head] → [B*nb, nh, cs, 1, dim_head]
        let q_exp = q.unsqueeze(3)?;
        // rpe: [cs, cs, dim_head] → [1, 1, cs, cs, dim_head]
        let rpe_exp = rpe.unsqueeze(0)?.unsqueeze(0)?;
        // Sum over last dim: [B*nb, nh, cs, cs]
        let pos_attn = (q_exp.broadcast_mul(&rpe_exp)?.sum(4)? * self.scale)?;

        // Content attention scores [B*nb, nh, cs, cs]
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let mut attn_weights = (attn_weights + pos_attn)?; // [B*nb, nh, cs, cs]

        // Mask padding in last block when remainder > 0
        if remainder > 0 {
            let cs = self.context_size;
            let mask_data: Vec<f32> = (0..cs)
                .flat_map(|i| {
                    (0..cs).map(move |j| {
                        if i < remainder && j < remainder {
                            0.0_f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_vec(mask_data, (cs, cs), hidden_states.device())?
                .to_dtype(attn_weights.dtype())?;
            // attn_weights: [B*nb, nh, cs, cs]; last `bsz` items are the last block per batch
            // Simpler: reshape to [B, nb, nh, cs, cs] and mask last nb slot
            let aw5 = attn_weights.reshape((bsz, num_blocks, self.num_heads, cs, cs))?;
            let mask_exp = mask.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?; // [1,1,1,cs,cs]
            let last = aw5.narrow(1, num_blocks - 1, 1)?.broadcast_add(&mask_exp)?;
            let aw5 = if num_blocks > 1 {
                let others = aw5.narrow(1, 0, num_blocks - 1)?;
                Tensor::cat(&[&others, &last], 1)?
            } else {
                last
            };
            attn_weights = aw5.reshape((bsz * num_blocks, self.num_heads, cs, cs))?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        // out: [B*nb, nh, cs, dim_head] → [B*nb, cs, nh*dim_head] → [B, T_padded, inner_dim]
        let inner_dim = self.num_heads * self.dim_head;
        let out = attn_weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, padded_len, inner_dim))?;
        // Trim to original length and apply output projection
        self.to_out.forward(&out.narrow(1, 0, num_features)?)
    }
}

// ─── Conformer Conv Module ────────────────────────────────────────────────────

#[allow(dead_code)]
struct ConformerConvModule {
    norm: LayerNorm,
    up_conv: candle_nn::Conv1d,
    depth_conv: candle_nn::Conv1d,
    batch_norm: candle_nn::BatchNorm,
    down_conv: candle_nn::Conv1d,
    inner_dim: usize,
}

#[allow(dead_code)]
impl ConformerConvModule {
    fn new(cfg: &EncoderCfg, vb: VarBuilder) -> Result<Self> {
        let inner_dim = cfg.hidden_dim * cfg.conv_expansion_factor;
        let norm = layer_norm(cfg.hidden_dim, 1e-5, vb.pp("norm"))?;

        let up_conv = conv1d(
            cfg.hidden_dim,
            inner_dim * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("up_conv"),
        )?;

        // Depthwise conv with symmetric padding (handled manually).
        let kernel = cfg.conv_kernel_size;
        let depth_conv = conv1d(
            inner_dim,
            inner_dim,
            kernel,
            Conv1dConfig {
                groups: inner_dim,
                padding: 0, // we pad manually to match Python (pad, pad - offset)
                ..Default::default()
            },
            vb.pp("depth_conv").pp("conv"),
        )?;

        let batch_norm = batch_norm(inner_dim, BatchNormConfig::default(), vb.pp("batch_norm"))?;

        let down_conv = conv1d(
            inner_dim,
            cfg.hidden_dim,
            1,
            Conv1dConfig::default(),
            vb.pp("down_conv"),
        )?;

        Ok(Self {
            norm,
            up_conv,
            depth_conv,
            batch_norm,
            down_conv,
            inner_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, T, hidden_dim]
        let xs = self.norm.forward(xs)?;
        // Conv1d expects [B, C, T]
        let xs = xs.permute([0, 2, 1])?;

        // Up-projection with GLU
        let xs = self.up_conv.forward(&xs)?; // [B, inner_dim*2, T]
        let c = xs.dim(1)?; // inner_dim * 2
        let xs_a = xs.narrow(1, 0, c / 2)?;
        let xs_b = xs.narrow(1, c / 2, c / 2)?;
        let xs = (xs_a * ops::sigmoid(&xs_b)?)?; // [B, inner_dim, T]

        // Depthwise conv with manual asymmetric padding
        // kernel_size=k: pad_left = k//2, pad_right = k//2 - (k+1)%2
        let k = self.depth_conv.weight().dim(2)?; // should match conv_kernel_size
        let pad_left = k / 2;
        let pad_right = pad_left - (k + 1) % 2;
        let t = xs.dim(2)?;
        let xs = if pad_left > 0 || pad_right > 0 {
            let left = Tensor::zeros(
                (xs.dim(0)?, self.inner_dim, pad_left),
                xs.dtype(),
                xs.device(),
            )?;
            let right = Tensor::zeros(
                (xs.dim(0)?, self.inner_dim, pad_right),
                xs.dtype(),
                xs.device(),
            )?;
            Tensor::cat(&[&left, &xs, &right], 2)?
        } else {
            xs
        };
        let xs = self.depth_conv.forward(&xs)?; // [B, inner_dim, T]

        // BatchNorm + SiLU
        let xs = self.batch_norm.forward_t(&xs, false)?;
        let xs = xs.silu()?;

        // Down-projection back to [B, T, hidden_dim]
        let xs = self.down_conv.forward(&xs)?;
        // Trim to original T (should match if padding is correct, but guard)
        let xs = xs.narrow(2, 0, t)?;
        xs.permute([0, 2, 1])
    }
}

// ─── Conformer Block ──────────────────────────────────────────────────────────

#[allow(dead_code)]
struct ConformerBlock {
    ff1: ConformerFFN,
    attn: ConformerAttention,
    conv: ConformerConvModule,
    ff2: ConformerFFN,
    post_norm: LayerNorm,
}

#[allow(dead_code)]
impl ConformerBlock {
    fn new(cfg: &EncoderCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ff1: ConformerFFN::new(cfg, vb.pp("ff1"))?,
            attn: ConformerAttention::new(cfg, vb.pp("attn"))?,
            conv: ConformerConvModule::new(cfg, vb.pp("conv"))?,
            ff2: ConformerFFN::new(cfg, vb.pp("ff2"))?,
            post_norm: layer_norm(cfg.hidden_dim, 1e-5, vb.pp("post_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_dists: &Tensor) -> Result<Tensor> {
        // Macaron-style: each sub-module contributes via residual
        let xs = (xs + &(self.ff1.forward(xs)? * 0.5)?)?;
        let xs = (&xs + &self.attn.forward(&xs, attention_dists)?)?;
        let xs = (&xs + &self.conv.forward(&xs)?)?;
        let xs = (&xs + &(self.ff2.forward(&xs)? * 0.5)?)?;
        self.post_norm.forward(&xs)
    }
}

// ─── Granite Speech CTC Encoder ───────────────────────────────────────────────

#[allow(dead_code)]
struct GraniteSpeechCTCEncoder {
    input_linear: Linear,
    layers: Vec<ConformerBlock>,
    out: Linear,
    out_mid: Linear,
    // Precomputed relative position distances [context_size, context_size] U32
    attention_dists: Tensor,
    num_layers: usize,
}

#[allow(dead_code)]
impl GraniteSpeechCTCEncoder {
    fn new(cfg: &EncoderCfg, vb: VarBuilder) -> Result<Self> {
        let input_linear = linear(cfg.input_dim, cfg.hidden_dim, vb.pp("input_linear"))?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_layers {
            layers.push(ConformerBlock::new(cfg, vb_l.pp(i))?);
        }

        let out = linear(cfg.hidden_dim, cfg.output_dim, vb.pp("out"))?;
        let out_mid = linear(cfg.output_dim, cfg.hidden_dim, vb.pp("out_mid"))?;

        // Precompute Shaw's relative distances: dist[i,j] = clamp(i-j, -cs, cs) + max_pos_emb
        let cs = cfg.context_size as i64;
        let max_pe = cfg.max_pos_emb as i64;
        let dist_data: Vec<u32> = (0..cs)
            .flat_map(|i| {
                (0..cs).map(move |j| {
                    let d = (i - j).clamp(-cs, cs);
                    (d + max_pe) as u32
                })
            })
            .collect();
        let attention_dists = Tensor::from_vec(
            dist_data,
            (cfg.context_size, cfg.context_size),
            &Device::Cpu,
        )?;

        Ok(Self {
            input_linear,
            layers,
            out,
            out_mid,
            attention_dists,
            num_layers: cfg.num_layers,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, T, input_dim=160]
        let mut xs = self.input_linear.forward(xs)?; // [B, T, hidden_dim]

        // Move attention_dists to same device as xs
        let attention_dists = self.attention_dists.to_device(xs.device())?;

        let mid = self.num_layers / 2;
        for (idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, &attention_dists)?;

            if idx + 1 == mid {
                // Mid-layer CTC auxiliary: add soft CTC logits back into hidden
                let mid_out = self.out.forward(&xs)?; // [B, T, output_dim]
                let mid_soft = candle_nn::ops::softmax_last_dim(&mid_out)?;
                let mid_back = self.out_mid.forward(&mid_soft)?; // [B, T, hidden_dim]
                xs = (&xs + &mid_back)?;
            }
        }
        Ok(xs) // [B, T, hidden_dim]
    }
}

// ─── QFormer (BLIP2-style) ────────────────────────────────────────────────────
//
// Two-layer QFormer: self-attention among queries + cross-attention to encoder.
// Weight paths follow BLIP2 naming used by Granite Speech projector.

#[allow(dead_code)]
struct GsQAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    out_ln: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl GsQAttn {
    fn new(
        q_hidden: usize,
        kv_hidden: usize,
        proj_cfg: &ProjectorCfg,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n = proj_cfg.num_attention_heads;
        let d = proj_cfg.head_dim();
        let all = n * d;
        Ok(Self {
            q: linear(q_hidden, all, vb.pp("attention").pp("query"))?,
            k: linear(kv_hidden, all, vb.pp("attention").pp("key"))?,
            v: linear(kv_hidden, all, vb.pp("attention").pp("value"))?,
            out: linear(all, proj_cfg.hidden_size, vb.pp("output").pp("dense"))?,
            out_ln: layer_norm(
                proj_cfg.hidden_size,
                proj_cfg.layer_norm_eps,
                vb.pp("output").pp("LayerNorm"),
            )?,
            num_heads: n,
            head_dim: d,
        })
    }

    fn forward(&self, q_hs: &Tensor, kv_hs: &Tensor) -> Result<Tensor> {
        let (b, q_len, _) = q_hs.dims3()?;
        let kv_len = kv_hs.dim(1)?;
        let scale = (self.head_dim as f64).powf(-0.5);

        let q = self
            .q
            .forward(q_hs)?
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(kv_hs)?
            .reshape((b, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(kv_hs)?
            .reshape((b, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, q_len, ()))?;
        let out = self.out.forward(&out)?;
        self.out_ln.forward(&(out + q_hs)?)
    }
}

#[allow(dead_code)]
struct GsQFormerLayer {
    self_attn: GsQAttn,
    cross_attn: Option<GsQAttn>,
    intermediate: Linear,
    output: Linear,
    output_ln: LayerNorm,
}

#[allow(dead_code)]
impl GsQFormerLayer {
    fn new(idx: usize, proj_cfg: &ProjectorCfg, vb: VarBuilder) -> Result<Self> {
        let self_attn = GsQAttn::new(
            proj_cfg.hidden_size,
            proj_cfg.hidden_size,
            proj_cfg,
            vb.pp("attention"),
        )?;

        let cross_attn = if idx.is_multiple_of(proj_cfg.cross_attention_frequency) {
            Some(GsQAttn::new(
                proj_cfg.hidden_size,
                proj_cfg.encoder_hidden_size,
                proj_cfg,
                vb.pp("crossattention"),
            )?)
        } else {
            None
        };

        let intermediate = linear(
            proj_cfg.hidden_size,
            proj_cfg.intermediate_size,
            vb.pp("intermediate_query").pp("dense"),
        )?;
        let output = linear(
            proj_cfg.intermediate_size,
            proj_cfg.hidden_size,
            vb.pp("output_query").pp("dense"),
        )?;
        let output_ln = layer_norm(
            proj_cfg.hidden_size,
            proj_cfg.layer_norm_eps,
            vb.pp("output_query").pp("LayerNorm"),
        )?;

        Ok(Self {
            self_attn,
            cross_attn,
            intermediate,
            output,
            output_ln,
        })
    }

    fn forward(&self, queries: &Tensor, encoder_hs: &Tensor) -> Result<Tensor> {
        // Self-attention among queries
        let xs = self.self_attn.forward(queries, queries)?;

        // Cross-attention to encoder output
        let xs = if let Some(ca) = &self.cross_attn {
            ca.forward(&xs, encoder_hs)?
        } else {
            xs
        };

        // FFN (GELU)
        let res = xs.clone();
        let xs = self.intermediate.forward(&xs)?.gelu_erf()?;
        let xs = self.output.forward(&xs)?;
        self.output_ln.forward(&(xs + res)?)
    }
}

#[allow(dead_code)]
struct GraniteSpeechEncoderProjector {
    query: Tensor,                // [1, num_queries, hidden_size]
    qformer_layernorm: LayerNorm, // applied to queries before qformer
    qformer_layers: Vec<GsQFormerLayer>,
    linear: Linear,
    window_size: usize,
    downsample_rate: usize,
}

#[allow(dead_code)]
impl GraniteSpeechEncoderProjector {
    fn new(proj_cfg: &ProjectorCfg, vb: VarBuilder) -> Result<Self> {
        let num_queries = proj_cfg.num_queries();
        let query = vb.get((1, num_queries, proj_cfg.hidden_size), "query")?;

        let qformer_ln = layer_norm(
            proj_cfg.hidden_size,
            proj_cfg.layer_norm_eps,
            vb.pp("qformer").pp("layernorm"),
        )?;

        let mut qformer_layers = Vec::with_capacity(proj_cfg.num_hidden_layers);
        let vb_ql = vb.pp("qformer").pp("encoder").pp("layer");
        for i in 0..proj_cfg.num_hidden_layers {
            qformer_layers.push(GsQFormerLayer::new(i, proj_cfg, vb_ql.pp(i))?);
        }

        let linear = linear(
            proj_cfg.hidden_size,
            proj_cfg.text_hidden_size,
            vb.pp("linear"),
        )?;

        Ok(Self {
            query,
            qformer_layernorm: qformer_ln,
            qformer_layers,
            linear,
            window_size: proj_cfg.window_size,
            downsample_rate: proj_cfg.downsample_rate,
        })
    }

    /// encoder_hidden_states: `[B, T, encoder_hidden_dim]`
    /// Returns: `[B, nblocks * num_queries, text_hidden_size]`
    fn forward(&self, encoder_hs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = encoder_hs.dims3()?;
        let ws = self.window_size;
        let nblocks = seq_len.div_ceil(ws);
        let padded = nblocks * ws;

        // Pad encoder output to multiple of window_size
        let encoder_hs = if padded > seq_len {
            let pad = Tensor::zeros(
                (batch_size, padded - seq_len, encoder_hs.dim(2)?),
                encoder_hs.dtype(),
                encoder_hs.device(),
            )?;
            Tensor::cat(&[encoder_hs, &pad], 1)?
        } else {
            encoder_hs.clone()
        };

        // Reshape to [B*nblocks, ws, encoder_dim]
        let enc_dim = encoder_hs.dim(2)?;
        let encoder_hs = encoder_hs.reshape((batch_size * nblocks, ws, enc_dim))?;

        // Initialize queries: [1, num_queries, hidden_size] → [B*nblocks, num_queries, hidden_size]
        let num_queries = self.query.dim(1)?;
        let queries = self.query.broadcast_left(batch_size * nblocks)?.reshape((
            batch_size * nblocks,
            num_queries,
            self.query.dim(2)?,
        ))?;

        // Apply layernorm to queries
        let mut queries = self.qformer_layernorm.forward(&queries)?;

        // QFormer layers: self-attn + cross-attn to encoder
        for layer in &self.qformer_layers {
            queries = layer.forward(&queries, &encoder_hs)?;
        }
        // queries: [B*nblocks, num_queries, hidden_size]

        // Reshape back to [B, nblocks * num_queries, hidden_size]
        let hidden = queries.dim(2)?;
        let queries = queries.reshape((batch_size, nblocks * num_queries, hidden))?;

        self.linear.forward(&queries)
    }
}

// ─── Main Model ───────────────────────────────────────────────────────────────

/// IBM Granite Speech for conditional generation.
///
/// Conformer CTC encoder + BLIP2 QFormer projector + Granite LLM.
pub struct GraniteSpeechForConditionalGeneration {
    #[allow(dead_code)]
    encoder: GraniteSpeechCTCEncoder,
    #[allow(dead_code)]
    projector: GraniteSpeechEncoderProjector,
    language_model: GraniteForCausalLM,
    audio_token_index: u32,
    device: Device,
}

impl GraniteSpeechForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let extra = &cfg.extra;

        let enc_json = extra.get("encoder_config").cloned().unwrap_or_default();
        let enc_cfg = EncoderCfg::from_json(&enc_json);

        let proj_cfg = ProjectorCfg::from_model_config(cfg);

        let audio_token_index = extra
            .get("audio_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as u32;

        let encoder = GraniteSpeechCTCEncoder::new(&enc_cfg, vb.pp("encoder"))?;
        let projector = GraniteSpeechEncoderProjector::new(&proj_cfg, vb.pp("projector"))?;
        let language_model = GraniteForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            encoder,
            projector,
            language_model,
            audio_token_index,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for GraniteSpeechForConditionalGeneration {
    fn device(&self) -> &Device {
        &self.device
    }

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
        let text_embeds = self.language_model.embed_text(input_ids)?;

        let embeddings = if let Some(mm) = multimodal_inputs {
            if mm.has_audio() {
                scatter_audio_into_text(&text_embeds, mm, self.audio_token_index)?
            } else {
                text_embeds
            }
        } else {
            text_embeds
        };

        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

// ─── Audio scatter ────────────────────────────────────────────────────────────

fn scatter_audio_into_text(
    text_embeds: &Tensor,
    mm: &MultimodalInputs,
    audio_token_index: u32,
) -> Result<Tensor> {
    if mm.audio_embeddings.is_empty() {
        return Ok(text_embeds.clone());
    }

    let (b, s, d) = text_embeds.dims3()?;

    let mut audio_clips: Vec<(usize, Tensor)> = mm
        .audio_embeddings
        .iter()
        .map(|(pos, pa)| (*pos, pa.embedding.clone()))
        .collect();
    audio_clips.sort_by_key(|(pos, _)| *pos);

    let flat_embeds = text_embeds.reshape((b * s, d))?;
    let token_ids = &mm.token_ids;

    let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
    let mut clip_idx = 0usize;
    let mut clip_offset = 0usize;

    for (seq_idx, &tok) in token_ids.iter().enumerate() {
        if tok == audio_token_index && clip_idx < audio_clips.len() {
            let clip = &audio_clips[clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, clip_offset, 1)?.squeeze(0)?);
            clip_offset += 1;
            if clip_offset >= clip_len {
                clip_idx += 1;
                clip_offset = 0;
            }
        } else {
            rows.push(flat_embeds.narrow(0, seq_idx, 1)?.squeeze(0)?);
        }
    }

    Tensor::stack(&rows, 0)?.reshape((b, s, d))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::config::ModelConfig;
    use crate::kv_cache::{BlockTable, CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::ProcessedAudio;

    /// Very small config to keep test runtime reasonable.
    /// encoder: hidden=8, layers=2, heads=2, head_dim=4, context=4, mels=4,
    ///   ffn_mult=2, conv_exp=2, kernel=3, out_dim=8, max_pos=4
    /// projector: hidden=8, layers=2, heads=2, inter=16, cross_freq=1, win=4, down=2
    /// LM: hidden=8, layers=1, heads=2, kv=2, inter=16, vocab=32
    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_index".into(), json!(10u32));
        extra.insert("downsample_rate".into(), json!(2u32));
        extra.insert("window_size".into(), json!(4u32));
        extra.insert(
            "encoder_config".into(),
            json!({
                "hidden_dim": 8,
                "num_layers": 2,
                "num_heads": 2,
                "dim_head": 4,
                "max_pos_emb": 4,
                "context_size": 4,
                "input_dim": 4,
                "output_dim": 8,
                "feedforward_mult": 2,
                "conv_expansion_factor": 2,
                "conv_kernel_size": 3
            }),
        );
        extra.insert(
            "projector_config".into(),
            json!({
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "cross_attention_frequency": 1
            }),
        );
        ModelConfig {
            architectures: vec!["GraniteSpeechForConditionalGeneration".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    fn make_cache(cfg: &ModelConfig, dev: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: dev.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn granite_speech_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        GraniteSpeechForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn granite_speech_encoder_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GraniteSpeechForConditionalGeneration::new(&cfg, vb).unwrap();

        // mel features: [B, T, input_dim=4]
        let features = Tensor::zeros((1usize, 8usize, 4usize), DType::F32, &dev).unwrap();
        let out = model.encoder.forward(&features).unwrap();
        assert_eq!(out.dim(0).unwrap(), 1);
        assert_eq!(out.dim(2).unwrap(), 8); // hidden_dim
    }

    #[test]
    fn granite_speech_projector_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GraniteSpeechForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encoder output: [B, T, hidden_dim=8]
        let enc_out = Tensor::zeros((1usize, 8usize, 8usize), DType::F32, &dev).unwrap();
        let proj = model.projector.forward(&enc_out).unwrap();
        // nblocks = ceil(8/4) = 2, num_queries = 4/2 = 2; output = [1, 4, 8]
        assert_eq!(proj.dim(0).unwrap(), 1);
        assert_eq!(proj.dim(2).unwrap(), 8); // text_hidden_size
    }

    #[test]
    fn granite_speech_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GraniteSpeechForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &dev).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
    }

    #[test]
    fn granite_speech_multimodal_scatter() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GraniteSpeechForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);

        let audio_token_index: u32 = 10;
        // Sequence: [1, audio, audio, 2]
        let token_ids = vec![1u32, audio_token_index, audio_token_index, 2u32];

        // Pre-encoded audio: 2 tokens × hidden=8
        let audio_emb = Tensor::ones((2usize, 8usize), DType::F32, &dev).unwrap();
        let processed = ProcessedAudio::new(audio_emb, 2);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(1, processed)]);

        let input_ids = Tensor::from_vec(token_ids, (1usize, 4usize), &dev)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();

        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let logits = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
    }
}
