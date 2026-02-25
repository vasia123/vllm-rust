//! GLM-ASR: transformer audio encoder with GQA + partial RoPE + LLaMA LLM.
//!
//! ```text
//! mel [1, n_mels, T]
//!   └── GlmAsrEncoder
//!         conv1: Conv1d(n_mels→hidden, k=3, s=1, p=1) → GELU
//!         conv2: Conv1d(hidden→hidden, k=3, s=2, p=1) → GELU  [T → T/2]
//!         N × GlmAsrEncoderLayer (pre-norm, GQA attention with partial RoPE, MLP)
//!         LayerNorm
//!         → [1, T/2, hidden]
//!   merge: truncate T/2 to multiple of merge_factor, reshape → [T/2/M, hidden*M]
//!   └── GlmAsrMultiModalProjector (Linear → GELU → Linear)
//!         → [T/2/M, text_hidden]
//! text_ids → LlamaForCausalLM
//!   scatter audio embeddings at audio placeholder positions
//! ```
//!
//! ## Encoder attention
//! GQA with partial RoPE: only first `rotary_dim = head_dim * partial_rotary_factor`
//! dimensions receive rotary embeddings; remaining dimensions are passed through unchanged.
//! Bias on q/v projections; no bias on k projection.
//!
//! ## Weight paths (HuggingFace format)
//! - `audio_tower.conv1.{weight,bias}`
//! - `audio_tower.conv2.{weight,bias}`
//! - `audio_tower.layers.{i}.input_layernorm.{weight,bias}`
//! - `audio_tower.layers.{i}.self_attn.{q_proj,v_proj}.{weight,bias}`
//! - `audio_tower.layers.{i}.self_attn.k_proj.weight`  (no bias)
//! - `audio_tower.layers.{i}.self_attn.o_proj.{weight,bias}`
//! - `audio_tower.layers.{i}.mlp.{fc1,fc2}.{weight,bias}`
//! - `audio_tower.layers.{i}.post_attention_layernorm.{weight,bias}`
//! - `audio_tower.norm.{weight,bias}`
//! - `multi_modal_projector.linear_{1,2}.weight`  (no bias)
//! - `language_model.{model.*,lm_head.*}`
//!
//! ## HF config layout
//! ```json
//! {
//!   "audio_config": {
//!     "hidden_size": 1280, "intermediate_size": 5120,
//!     "num_hidden_layers": 32, "num_attention_heads": 20,
//!     "num_key_value_heads": 4, "num_mel_bins": 128,
//!     "layer_norm_eps": 1e-5, "hidden_act": "gelu",
//!     "rope_theta": 10000.0, "rope_parameters": {"partial_rotary_factor": 0.5}
//!   },
//!   "text_config": {"hidden_size": 4096, ...},
//!   "merge_factor": 4,
//!   "audio_token_id": 151646
//! }
//! ```

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    conv1d, layer_norm, linear_b, linear_no_bias, ops::softmax_last_dim, Conv1d, Conv1dConfig,
    LayerNorm, Linear, VarBuilder,
};

use crate::config::ModelConfig;
use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::multimodal::MultimodalInputs;

use super::llama::LlamaForCausalLM;

// ─── Config ──────────────────────────────────────────────────────────────────

struct GlmAsrCfg {
    num_mel_bins: usize,
    audio_hidden: usize,
    audio_intermediate: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    rotary_dim: usize,
    rope_theta: f64,
    text_hidden: usize,
    audio_token_id: u32,
}

impl GlmAsrCfg {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;
        let audio = extra.get("audio_config").cloned().unwrap_or_default();
        let text = extra.get("text_config").cloned().unwrap_or_default();

        let audio_hidden = audio
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1280);

        let num_heads = audio
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(20);

        let head_dim = audio_hidden / num_heads;

        let partial_rotary_factor = audio
            .get("rope_parameters")
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .or_else(|| audio.get("partial_rotary_factor").and_then(|v| v.as_f64()))
            .unwrap_or(0.5);
        let rotary_dim = ((head_dim as f64) * partial_rotary_factor) as usize;

        let rope_theta = audio
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| audio.get("rope_theta").and_then(|v| v.as_f64()))
            .unwrap_or(10000.0);

        let merge_factor = extra
            .get("merge_factor")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(4);

        let audio_intermediate = audio
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(audio_hidden * merge_factor);

        let text_hidden = text
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        Self {
            num_mel_bins: audio
                .get("num_mel_bins")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(128),
            audio_hidden,
            audio_intermediate,
            num_layers: audio
                .get("num_hidden_layers")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(32),
            num_heads,
            num_kv_heads: audio
                .get("num_key_value_heads")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(num_heads),
            head_dim,
            intermediate_size: audio
                .get("mlp_intermediate_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(audio_hidden * 4),
            layer_norm_eps: audio
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5),
            rotary_dim,
            rope_theta,
            text_hidden,
            audio_token_id: extra
                .get("audio_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(151646) as u32,
        }
    }
}

// ─── Rotary helpers ───────────────────────────────────────────────────────────

/// Compute cos/sin for standard RoPE given `seq_len` positions.
///
/// Returns `(cos, sin)` each of shape `[seq_len, rotary_dim/2]`.
fn compute_rope_cos_sin(
    seq_len: usize,
    rotary_dim: usize,
    rope_theta: f64,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = rotary_dim / 2;
    // inv_freq: [half]
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| {
            let exp = (2 * i) as f64 / rotary_dim as f64;
            (1.0 / rope_theta.powf(exp)) as f32
        })
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

    // positions: [seq_len]
    let positions = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;

    // freqs: [seq_len, half]  via outer product
    let freqs = positions
        .unsqueeze(1)?
        .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply partial RoPE to `x` (shape `[B, S, H, head_dim]`).
///
/// Only the first `rotary_dim` channels are rotated; the rest pass through.
/// `cos` and `sin` have shape `[S, rotary_dim/2]`.
fn apply_partial_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let half = rotary_dim / 2;

    // x_rot: [B, S, H, rotary_dim]
    let x_rot = x.narrow(3, 0, rotary_dim)?;
    let x1 = x_rot.narrow(3, 0, half)?;
    let x2 = x_rot.narrow(3, half, half)?;

    // cos/sin: [S, half] → [1, S, 1, half]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    let rotated = Tensor::cat(
        &[
            x1.broadcast_mul(&cos)?
                .broadcast_sub(&x2.broadcast_mul(&sin)?)?,
            x2.broadcast_mul(&cos)?
                .broadcast_add(&x1.broadcast_mul(&sin)?)?,
        ],
        3,
    )?;

    if rotary_dim == head_dim {
        Ok(rotated)
    } else {
        let x_pass = x.narrow(3, rotary_dim, head_dim - rotary_dim)?;
        Tensor::cat(&[rotated, x_pass], 3)
    }
}

// ─── Encoder components ───────────────────────────────────────────────────────

struct GlmAsrEncoderAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
}

impl GlmAsrEncoderAttention {
    fn new(cfg: &GlmAsrCfg, vb: VarBuilder) -> Result<Self> {
        let q_size = cfg.num_heads * cfg.head_dim;
        let kv_size = cfg.num_kv_heads * cfg.head_dim;

        Ok(Self {
            q_proj: linear_b(cfg.audio_hidden, q_size, true, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.audio_hidden, kv_size, vb.pp("k_proj"))?,
            v_proj: linear_b(cfg.audio_hidden, kv_size, true, vb.pp("v_proj"))?,
            o_proj: linear_b(q_size, cfg.audio_hidden, true, vb.pp("o_proj"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, s, _) = hidden.dims3()?;
        let dtype = hidden.dtype();
        let device = hidden.device();

        let q = self.q_proj.forward(hidden)?;
        let k = self.k_proj.forward(hidden)?;
        let v = self.v_proj.forward(hidden)?;

        // Reshape to [B, S, H, D]
        let q = q.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, s, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, s, self.num_kv_heads, self.head_dim))?;

        // Compute and apply partial RoPE
        let (cos, sin) = compute_rope_cos_sin(s, self.rotary_dim, self.rope_theta, dtype, device)?;
        let q = apply_partial_rope(&q, &cos, &sin, self.rotary_dim, self.head_dim)?;
        let k = apply_partial_rope(&k, &cos, &sin, self.rotary_dim, self.head_dim)?;

        // GQA: repeat k/v heads to match query heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let ratio = self.num_heads / self.num_kv_heads;
            (k.repeat((1, 1, ratio, 1))?, v.repeat((1, 1, ratio, 1))?)
        } else {
            (k, v)
        };

        // [B, S, H, D] → [B, H, S, D]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt().recip();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let attn = softmax_last_dim(&scores)?.matmul(&v)?;

        // [B, H, S, D] → [B, S, H*D]
        let out =
            attn.transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

struct GlmAsrEncoderMlp {
    fc1: Linear,
    fc2: Linear,
}

impl GlmAsrEncoderMlp {
    fn new(cfg: &GlmAsrCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear_b(cfg.audio_hidden, cfg.intermediate_size, true, vb.pp("fc1"))?,
            fc2: linear_b(cfg.intermediate_size, cfg.audio_hidden, true, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(xs)?.gelu_erf()?)
    }
}

struct GlmAsrEncoderLayer {
    self_attn: GlmAsrEncoderAttention,
    mlp: GlmAsrEncoderMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl GlmAsrEncoderLayer {
    fn new(cfg: &GlmAsrCfg, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: GlmAsrEncoderAttention::new(cfg, vb.pp("self_attn"))?,
            mlp: GlmAsrEncoderMlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: layer_norm(
                cfg.audio_hidden,
                cfg.layer_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: layer_norm(
                cfg.audio_hidden,
                cfg.layer_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = (residual + self.self_attn.forward(&xs)?)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (residual + self.mlp.forward(&xs)?)?;
        Ok(xs)
    }
}

/// GLM-ASR audio encoder.
///
/// Architecture: Conv1d×2(GELU) → N pre-norm transformer layers with GQA +
/// partial RoPE → LayerNorm.
///
/// Input: `[B, n_mels, T]` → output: `[B, T/2, hidden]`
/// (T/2 from the stride-2 conv2; in practice B=1 for audio encoding).
#[allow(dead_code)]
struct GlmAsrEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    layers: Vec<GlmAsrEncoderLayer>,
    norm: LayerNorm,
}

#[allow(dead_code)]
impl GlmAsrEncoder {
    fn new(cfg: &GlmAsrCfg, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv1d(
            cfg.num_mel_bins,
            cfg.audio_hidden,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = conv1d(
            cfg.audio_hidden,
            cfg.audio_hidden,
            3,
            Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let layers = (0..cfg.num_layers)
            .map(|i| GlmAsrEncoderLayer::new(cfg, vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let norm = layer_norm(cfg.audio_hidden, cfg.layer_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            conv1,
            conv2,
            layers,
            norm,
        })
    }

    /// `mel`: `[B, n_mels, T]` → `[B, T/2, hidden]`
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(mel)?.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;
        // [B, hidden, T/2] → [B, T/2, hidden]
        let xs = xs.transpose(1, 2)?;
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.norm.forward(&xs)
    }
}

// ─── Projector ────────────────────────────────────────────────────────────────

/// Projects merged audio frames to LLM hidden size.
///
/// Input: `[*, audio_intermediate]` where `audio_intermediate = audio_hidden * merge_factor`.
/// Architecture: Linear → GELU → Linear (no bias in either layer).
#[allow(dead_code)]
struct GlmAsrMultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
}

#[allow(dead_code)]
impl GlmAsrMultiModalProjector {
    fn new(audio_intermediate: usize, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_no_bias(audio_intermediate, text_hidden * 2, vb.pp("linear_1"))?,
            linear_2: linear_no_bias(text_hidden * 2, text_hidden, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_2
            .forward(&self.linear_1.forward(xs)?.gelu_erf()?)
    }
}

// ─── Top-level model ─────────────────────────────────────────────────────────

/// GLM-ASR: audio encoder + projector + LLaMA LLM.
///
/// Audio is pre-encoded by the processor; `forward_multimodal` scatters
/// `ProcessedAudio.embedding` tensors at audio placeholder positions.
pub struct GlmAsrForConditionalGeneration {
    #[allow(dead_code)]
    audio_tower: GlmAsrEncoder,
    #[allow(dead_code)]
    multi_modal_projector: GlmAsrMultiModalProjector,
    language_model: LlamaForCausalLM,
    audio_token_id: u32,
    device: Device,
}

impl GlmAsrForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let glm_cfg = GlmAsrCfg::from_model_config(cfg);
        let audio_token_id = glm_cfg.audio_token_id;

        let audio_tower = GlmAsrEncoder::new(&glm_cfg, vb.pp("audio_tower"))?;
        let multi_modal_projector = GlmAsrMultiModalProjector::new(
            glm_cfg.audio_intermediate,
            glm_cfg.text_hidden,
            vb.pp("multi_modal_projector"),
        )?;
        let language_model = LlamaForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            audio_tower,
            multi_modal_projector,
            language_model,
            audio_token_id,
            device: vb.device().clone(),
        })
    }
}

impl ModelForward for GlmAsrForConditionalGeneration {
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
                scatter_audio_into_text(&text_embeds, mm, self.audio_token_id)?
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

// ─── Multimodal scatter ───────────────────────────────────────────────────────

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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::config::ModelConfig;
    use crate::kv_cache::{BlockTable, CacheConfig, KVCacheDtype, KVCacheManager};
    use crate::multimodal::ProcessedAudio;

    fn make_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_id".into(), json!(10u32));
        extra.insert("merge_factor".into(), json!(2u32));
        extra.insert(
            "audio_config".into(),
            json!({
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "num_mel_bins": 8,
                "mlp_intermediate_size": 16,
                "layer_norm_eps": 1e-5,
                "hidden_act": "gelu",
                "rope_theta": 10000.0,
                "rope_parameters": {"partial_rotary_factor": 0.5}
            }),
        );
        extra.insert("text_config".into(), json!({ "hidden_size": 8 }));
        ModelConfig {
            architectures: vec!["GlmAsrForConditionalGeneration".to_string()],
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
    fn glmasr_new() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let _model = GlmAsrForConditionalGeneration::new(&cfg, vb).unwrap();
    }

    #[test]
    fn glmasr_encoder_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let glm_cfg = GlmAsrCfg::from_model_config(&cfg);
        let encoder = GlmAsrEncoder::new(&glm_cfg, vb.pp("audio_tower")).unwrap();

        // mel: [1, n_mels=8, T=16]
        let mel = Tensor::zeros((1usize, 8usize, 16usize), DType::F32, &dev).unwrap();
        let out = encoder.forward(&mel).unwrap();
        // conv2 halves T: T/2=8; output: [1, 8, hidden=8]
        assert_eq!(out.dims(), &[1, 8, 8]);
    }

    #[test]
    fn glmasr_forward_shape() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GlmAsrForConditionalGeneration::new(&cfg, vb).unwrap();

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
    fn glmasr_decode_batch_shape() {
        use crate::engine::DecodeSequenceMetadata;

        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GlmAsrForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let mut bt0 = BlockTable::new(16);
        let mut bt1 = BlockTable::new(16);
        kv_mgr.allocate_for_request(&mut bt0, 4).unwrap();
        kv_mgr.allocate_for_request(&mut bt1, 4).unwrap();
        let slot0 = bt0.slot_mapping(4, 1);
        let slot1 = bt1.slot_mapping(4, 1);

        let input_ids = Tensor::zeros((2usize, 1usize), DType::U32, &dev).unwrap();
        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 0,
                seqlen_offset: 4,
                block_ids: bt0.block_ids().to_vec(),
                slot_mapping: slot0,
            },
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 4,
                block_ids: bt1.block_ids().to_vec(),
                slot_mapping: slot1,
            },
        ];
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_mgr)
            .unwrap();
        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn glmasr_multimodal_scatter() {
        let dev = Device::Cpu;
        let cfg = make_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let model = GlmAsrForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv_mgr = make_cache(&cfg, &dev);
        let audio_token_id: u32 = 10;
        let token_ids = vec![1u32, audio_token_id, audio_token_id, 2u32];

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

    #[test]
    fn glmasr_projector_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        // audio_intermediate=16, text_hidden=8; linear_1: 16→16, linear_2: 16→8
        let projector =
            GlmAsrMultiModalProjector::new(16, 8, vb.pp("multi_modal_projector")).unwrap();
        let input = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &dev).unwrap();
        let out = projector.forward(&input).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8]);
    }
}
