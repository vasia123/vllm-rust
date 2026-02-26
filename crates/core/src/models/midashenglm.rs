//! MiDasheng audio language model.
//!
//! `MiDashengLMModel` is an audio-language model that combines:
//! 1. `DashengAudioTransformer` — ViT-style audio encoder with 2D patch embed,
//!    separable time×freq positional embeddings, BatchNorm2D, and LayerScale blocks
//! 2. `DashengProjector` — subsample+project: `reshape(k×dim) → Linear(GELU) → Linear`
//! 3. `Qwen2ForCausalLM` — language backbone
//!
//! Audio is pre-encoded by the processor; `forward_multimodal` scatters
//! `ProcessedAudio.embedding` tensors at audio placeholder positions.
//!
//! Weight paths:
//! - `audio_encoder.init_bn.{weight,bias,running_mean,running_var}`
//! - `audio_encoder.patch_embed.proj.{weight,bias}`
//! - `audio_encoder.{time_pos_embed,freq_pos_embed}`
//! - `audio_encoder.blocks.{i}.{norm1,norm2}.{weight,bias}`
//! - `audio_encoder.blocks.{i}.attn.{qkv,proj}.{weight,bias}`
//! - `audio_encoder.blocks.{i}.{ls1,ls2}.gamma`
//! - `audio_encoder.blocks.{i}.mlp.{fc1,fc2}.{weight,bias}`
//! - `audio_encoder.norm.{weight,bias}`
//! - `audio_projector.net.{0,2}.weight`
//! - `decoder.model.*`, `decoder.lm_head.*`
//!
//! Reference: `reference/vllm/vllm/model_executor/models/midashenglm.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::qwen2::Qwen2ForCausalLM;

// ─── BatchNorm 2D (inference-only) ────────────────────────────────────────

/// Inference-only BatchNorm2D: uses pre-loaded running mean/var.
///
/// Applied on the channel (frequency) dimension of `[B, C, H, T]` tensors.
/// Forward: `(x - mean) / sqrt(var + eps) * weight + bias`
struct DashengBn2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl DashengBn2d {
    fn new(num_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(num_features, "weight")?;
        let bias = vb.get(num_features, "bias")?;
        let running_mean = vb.get(num_features, "running_mean")?;
        let running_var = vb.get(num_features, "running_var")?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    /// `x` shape: `[B, C, H, T]` — normalises along channel dim C.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, t) = xs.dims4()?;
        // Reshape to [B, C, H*T] for channel-wise normalisation.
        let xs = xs.reshape((b, c, h * t))?;
        let mean = self.running_mean.reshape((1, c, 1))?;
        let var = self.running_var.reshape((1, c, 1))?;
        let w = self.weight.reshape((1, c, 1))?;
        let b_bias = self.bias.reshape((1, c, 1))?;

        // (x - mean) / sqrt(var + eps) * weight + bias
        let denom = var.affine(1.0, self.eps)?.sqrt()?;
        let normed = xs.broadcast_sub(&mean)?.broadcast_div(&denom)?;
        let out = normed.broadcast_mul(&w)?.broadcast_add(&b_bias)?;
        out.reshape((b, c, h, t))
    }
}

// ─── LayerScale ───────────────────────────────────────────────────────────

struct DashengLayerScale {
    gamma: Tensor,
}

impl DashengLayerScale {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [B, S, D]; gamma: [D]
        xs.broadcast_mul(&self.gamma)
    }
}

// ─── DashengAttention ─────────────────────────────────────────────────────

struct DashengAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl DashengAttention {
    fn new(dim: usize, num_heads: usize, qkv_bias: bool, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);
        let qkv = if qkv_bias {
            linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj = linear_no_bias(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let qkv = self.qkv.forward(xs)?;
        // [B, N, 3*D] → [B, N, 3, H, head_dim]
        let qkv = qkv
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        // [3, B, H, N, head_dim] — split on dim 0
        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?;

        let attn = (q.matmul(&k.transpose(2, 3)?)?.affine(self.scale, 0.0))?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        // [B, H, N, head_dim] → [B, N, H*head_dim]
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, n, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

// ─── DashengMlp ───────────────────────────────────────────────────────────

struct DashengMlp {
    fc1: Linear,
    fc2: Linear,
}

impl DashengMlp {
    fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(hidden_features, in_features, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

// ─── DashengBlock ─────────────────────────────────────────────────────────

struct DashengBlock {
    norm1: LayerNorm,
    attn: DashengAttention,
    ls1: Option<DashengLayerScale>,
    norm2: LayerNorm,
    mlp: DashengMlp,
    ls2: Option<DashengLayerScale>,
}

impl DashengBlock {
    fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        has_layer_scale: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let attn = DashengAttention::new(dim, num_heads, qkv_bias, vb.pp("attn"))?;
        let ls1 = if has_layer_scale {
            Some(DashengLayerScale::new(dim, vb.pp("ls1"))?)
        } else {
            None
        };
        let norm2 = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let hidden = (dim as f64 * mlp_ratio) as usize;
        let mlp = DashengMlp::new(dim, hidden, vb.pp("mlp"))?;
        let ls2 = if has_layer_scale {
            Some(DashengLayerScale::new(dim, vb.pp("ls2"))?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(&self.norm1.forward(xs)?)?;
        let attn_out = match &self.ls1 {
            Some(ls) => ls.forward(&attn_out)?,
            None => attn_out,
        };
        let xs = (xs + attn_out)?;

        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        let mlp_out = match &self.ls2 {
            Some(ls) => ls.forward(&mlp_out)?,
            None => mlp_out,
        };
        xs + mlp_out
    }
}

// ─── DashengAudioTransformer ──────────────────────────────────────────────

/// ViT-style audio encoder with 2D patch embedding and separable pos embeddings.
///
/// Input: log-mel spectrogram `[B, n_mels, t]`
/// Output: sequence of audio embeddings `[B, n_patches, embed_dim]`
#[allow(dead_code)]
struct DashengAudioTransformer {
    init_bn: DashengBn2d,
    patch_embed: Conv2d,
    time_pos_embed: Tensor,
    freq_pos_embed: Tensor,
    blocks: Vec<DashengBlock>,
    norm: LayerNorm,
    /// Patch count in the frequency dimension.
    freq_grid: usize,
    /// Max time patches the positional embedding covers per chunk.
    time_chunk: usize,
    embed_dim: usize,
}

#[allow(dead_code)]
impl DashengAudioTransformer {
    fn new(cfg: &DashengEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let n_mels = cfg.n_mels;
        let embed_dim = cfg.embed_dim;

        let init_bn = DashengBn2d::new(n_mels, vb.pp("init_bn"))?;

        let patch_embed_cfg = Conv2dConfig {
            stride: cfg.patch_stride,
            ..Default::default()
        };
        let patch_embed = candle_nn::conv2d(
            cfg.input_channels,
            embed_dim,
            cfg.patch_size,
            patch_embed_cfg,
            vb.pp("patch_embed").pp("proj"),
        )?;

        let freq_grid = n_mels / cfg.patch_stride;
        // Total time patches for the full target_length (used for pos embed size).
        let time_total = cfg.target_length / cfg.patch_stride;
        // Chunk size: quarter of total (matching the Python split strategy).
        let time_chunk = time_total.div_ceil(4);

        let time_pos_embed = vb.get((1, embed_dim, 1, time_total), "time_pos_embed")?;
        let freq_pos_embed = vb.get((1, embed_dim, freq_grid, 1), "freq_pos_embed")?;

        let blocks = (0..cfg.depth)
            .map(|i| {
                DashengBlock::new(
                    embed_dim,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.qkv_bias,
                    cfg.init_values.is_some(),
                    vb.pp("blocks").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = candle_nn::layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;

        Ok(Self {
            init_bn,
            patch_embed,
            time_pos_embed,
            freq_pos_embed,
            blocks,
            norm,
            freq_grid,
            time_chunk,
            embed_dim,
        })
    }

    /// Forward through audio encoder.
    ///
    /// `features`: log-mel spectrogram `[B, n_mels, t]` (pre-extracted).
    /// Returns: `[B, num_patches, embed_dim]`
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let (b, _n_mels, _t) = features.dims3()?;

        // [B, n_mels, t] → [B, n_mels, 1, t] for BatchNorm2D (C=n_mels, H=1, W=t).
        let x = features.unsqueeze(2)?;
        let x = self.init_bn.forward(&x)?;
        // Restore to [B, 1, n_mels, t] for Conv2d (in_chans=1).
        let x = x.permute((0, 2, 1, 3))?;

        // Patch embed: [B, 1, n_mels, t] → [B, embed_dim, f_grid, t_grid]
        let x = self.patch_embed.forward(&x)?;
        let t_grid = x.dim(3)?;

        // Process in time chunks to bound sequence length per block stack.
        let num_chunks = t_grid.div_ceil(self.time_chunk);
        let mut outputs = Vec::with_capacity(num_chunks);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * self.time_chunk;
            let chunk_t = (self.time_chunk).min(t_grid - start);

            // [B, embed_dim, f_grid, chunk_t]
            let chunk = x.narrow(3, start, chunk_t)?;

            // Add time and frequency positional embeddings (both broadcast).
            let time_emb = self.time_pos_embed.narrow(3, start, chunk_t)?;
            let chunk = chunk
                .broadcast_add(&time_emb)?
                .broadcast_add(&self.freq_pos_embed)?;

            // [B, embed_dim, f_grid, chunk_t] → [B, f_grid*chunk_t, embed_dim]
            let seq = self.freq_grid * chunk_t;
            let chunk = chunk
                .reshape((b, self.embed_dim, seq))?
                .permute((0, 2, 1))?
                .contiguous()?;

            let mut chunk = chunk;
            for block in &self.blocks {
                chunk = block.forward(&chunk)?;
            }
            outputs.push(self.norm.forward(&chunk)?);
        }

        Tensor::cat(&outputs, 1)
    }
}

// ─── DashengProjector ─────────────────────────────────────────────────────

/// Audio projector: subsample k frames, project to LLM hidden dim.
///
/// `[B, s*k, in_dim]` → reshape → `[B, s, k*in_dim]` → fc1+GELU → fc2 → `[B, s, out_dim]`
#[allow(dead_code)]
struct DashengProjector {
    fc1: Linear,
    fc2: Linear,
    k: usize,
}

#[allow(dead_code)]
impl DashengProjector {
    fn new(in_dim: usize, out_dim: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_no_bias(in_dim * k, out_dim, vb.pp("net").pp(0))?;
        let fc2 = linear_no_bias(out_dim, out_dim, vb.pp("net").pp(2))?;
        Ok(Self { fc1, fc2, k })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, d) = xs.dims3()?;
        // Discard tail frames that don't fit evenly.
        let keep = (s / self.k) * self.k;
        let xs = if keep < s {
            xs.narrow(1, 0, keep)?
        } else {
            xs.clone()
        };
        let new_s = keep / self.k;
        let xs = xs.reshape((b, new_s, self.k * d))?;
        let xs = self.fc1.forward(&xs)?.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

// ─── Encoder config ───────────────────────────────────────────────────────

/// Dasheng audio encoder hyper-parameters (from `audio_encoder_config` in model config).
#[derive(Debug, Clone)]
struct DashengEncoderConfig {
    n_mels: usize,
    embed_dim: usize,
    depth: usize,
    num_heads: usize,
    mlp_ratio: f64,
    qkv_bias: bool,
    init_values: Option<f64>,
    input_channels: usize,
    patch_size: usize,
    patch_stride: usize,
    target_length: usize,
}

impl DashengEncoderConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let enc = cfg
            .extra
            .get("audio_encoder_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let get_usize = |key: &str, default: usize| -> usize {
            enc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            enc.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };
        let get_bool = |key: &str, default: bool| -> bool {
            enc.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
        };

        let init_values = enc.get("init_values").and_then(|v| v.as_f64());

        Self {
            n_mels: get_usize("n_mels", 128),
            embed_dim: get_usize("embed_dim", 768),
            depth: get_usize("depth", 12),
            num_heads: get_usize("num_heads", 12),
            mlp_ratio: get_f64("mlp_ratio", 4.0),
            qkv_bias: get_bool("qkv_bias", true),
            init_values,
            input_channels: get_usize("input_channels", 1),
            patch_size: get_usize("patch_size", 16),
            patch_stride: get_usize("patch_stride", 16),
            target_length: get_usize("target_length", 1024),
        }
    }
}

// ─── MiDashengLMModel ─────────────────────────────────────────────────────

/// Audio language model: DashengAudioTransformer + DashengProjector + Qwen2.
pub struct MiDashengLMModel {
    #[allow(dead_code)]
    audio_encoder: DashengAudioTransformer,
    #[allow(dead_code)]
    audio_projector: DashengProjector,
    decoder: Qwen2ForCausalLM,
    audio_token_id: u32,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl MiDashengLMModel {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let enc_cfg = DashengEncoderConfig::from_model_config(cfg);

        let audio_encoder = DashengAudioTransformer::new(&enc_cfg, vb.pp("audio_encoder"))?;

        let subsample_factor = cfg
            .extra
            .get("subsample_factor")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(5);

        let audio_projector = DashengProjector::new(
            enc_cfg.embed_dim,
            cfg.hidden_size,
            subsample_factor,
            vb.pp("audio_projector"),
        )?;

        let decoder = Qwen2ForCausalLM::new(cfg, vb.pp("decoder"))?;

        let audio_token_id = cfg
            .extra
            .get("audio_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151646) as u32;

        Ok(Self {
            audio_encoder,
            audio_projector,
            decoder,
            audio_token_id,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }
}

// ─── ModelForward ─────────────────────────────────────────────────────────

impl ModelForward for MiDashengLMModel {
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
        self.decoder.forward(
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
        let embeddings = self.decoder.embed_text(input_ids)?;
        self.decoder
            .forward_decode_batch_with_embeddings(&embeddings, sequences, kv_cache_mgr)
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
        let text_embeds = self.decoder.embed_text(input_ids)?;

        let embeddings = if let Some(mm) = multimodal_inputs {
            if mm.has_audio() {
                scatter_audio_into_text(&text_embeds, mm, self.audio_token_id)?
            } else {
                text_embeds
            }
        } else {
            text_embeds
        };

        self.decoder.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

// ─── Multimodal scatter ───────────────────────────────────────────────────

fn scatter_audio_into_text(
    text_embeds: &Tensor,
    mm: &MultimodalInputs,
    audio_token_id: u32,
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
        if tok == audio_token_id && clip_idx < audio_clips.len() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    use crate::kv_cache::{BlockTable, CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::ProcessedAudio;

    fn test_config() -> ModelConfig {
        let mut enc = serde_json::Map::new();
        enc.insert("n_mels".into(), serde_json::json!(16));
        enc.insert("embed_dim".into(), serde_json::json!(32));
        enc.insert("depth".into(), serde_json::json!(2));
        enc.insert("num_heads".into(), serde_json::json!(2));
        enc.insert("mlp_ratio".into(), serde_json::json!(2.0));
        enc.insert("qkv_bias".into(), serde_json::json!(true));
        enc.insert("input_channels".into(), serde_json::json!(1));
        enc.insert("patch_size".into(), serde_json::json!(4));
        enc.insert("patch_stride".into(), serde_json::json!(4));
        enc.insert("target_length".into(), serde_json::json!(64));

        let mut extra = serde_json::Map::new();
        extra.insert(
            "audio_encoder_config".into(),
            serde_json::Value::Object(enc),
        );
        extra.insert("subsample_factor".into(), serde_json::json!(2));
        extra.insert("audio_token_id".into(), serde_json::json!(10u32));

        ModelConfig {
            architectures: vec!["MiDashengLMModel".to_string()],
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
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
    fn test_midashenglm_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiDashengLMModel::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
    }

    #[test]
    fn test_dasheng_bn2d_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let bn = DashengBn2d::new(4, vb).unwrap();
        let x = Tensor::zeros((2, 4, 1, 8), DType::F32, &device).unwrap();
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4, 1, 8]);
    }

    #[test]
    fn test_dasheng_projector_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let proj = DashengProjector::new(8, 16, 2, vb).unwrap();
        let x = Tensor::zeros((1, 6, 8), DType::F32, &device).unwrap();
        let out = proj.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_dasheng_block_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = DashengBlock::new(16, 2, 2.0, true, false, vb).unwrap();
        let x = Tensor::zeros((1, 4, 16), DType::F32, &device).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_midashenglm_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiDashengLMModel::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &device);
        let seq_len = 4usize;
        let mut bt = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_mgr, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 1);
    }

    #[test]
    fn test_midashenglm_multimodal_scatter() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = MiDashengLMModel::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &device);
        let audio_token_id: u32 = 10;
        let token_ids = vec![1u32, audio_token_id, audio_token_id, 2u32];

        let audio_emb = Tensor::ones((2usize, 32usize), DType::F32, &device).unwrap();
        let processed = ProcessedAudio::new(audio_emb, 2);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(1, processed)]);

        let input_ids = Tensor::from_vec(token_ids, (1usize, 4usize), &device)
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

    #[test]
    fn test_midashenglm_encoder_config() {
        let cfg = test_config();
        let enc_cfg = DashengEncoderConfig::from_model_config(&cfg);
        assert_eq!(enc_cfg.n_mels, 16);
        assert_eq!(enc_cfg.embed_dim, 32);
        assert_eq!(enc_cfg.depth, 2);
        assert_eq!(enc_cfg.target_length, 64);
    }
}
