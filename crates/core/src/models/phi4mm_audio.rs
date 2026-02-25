//! Phi4MM audio components: ConformerEncoder + WindowQformer + AudioEmbedding.
//!
//! This module implements the audio encoder side of `Phi4MMForCausalLM`.
//! The conformer encoder runs during preprocessing (Python side); at Rust
//! inference time we use pre-encoded audio embeddings and scatter them into
//! the token sequence.  Only `audio_projection` and `audio_projection_for_vision`
//! are needed on the Rust inference critical path.
//!
//! ## Architecture (phi4mm_audio.py `AudioEmbedding`)
//! ```text
//! mel [B, T, n_mels]
//!   └── ConformerEncoder (stub — preprocessing only)
//!         NemoConvSubsampling or WindowQformer (optional, preprocessing only)
//!         → [B, T', audio_dim]
//!   linear downsample (merge `downsample_rate` consecutive frames):
//!         → [B, T'/R, audio_dim*R]
//!   └── audio_projection  (MLP or linear → text_hidden)
//!         → [B, T'/R, text_hidden]
//! ```
//!
//! ## Weight paths (after phi4mm.py weight remapping)
//! Under `embed_tokens_extend.*`:
//! - `encoder.*` (conformer — many sub-weights, not loaded in our stub)
//! - MLP projection (projection_cls="mlp", depth=2, indexed sequential):
//!   - `audio_projection.0.{weight,bias}`   — Linear(audio_dim*R → text_hidden)
//!   - `audio_projection.2.{weight,bias}`   — Linear(text_hidden → text_hidden)
//!   - same for `audio_projection_for_vision.{0,2}.*`
//! - Linear projection (projection_cls="linear"):
//!   - `audio_projection.{weight,bias}`
//!   - same for `audio_projection_for_vision.*`
//!
//! ## HF config fields consumed here
//! ```json
//! {
//!   "hidden_size":  4096,                    // text LLM hidden size
//!   "audio_processor": {
//!     "name": "cascades",
//!     "config": { "input_size": 80, "attention_dim": 1024, ... }
//!   },
//!   "embd_layer": {
//!     "audio_embd_layer": {
//!       "projection_cls": "mlp",             // "linear" | "mlp"
//!       "downsample_rate": 2,
//!       "use_qformer": false,
//!       "use_conv_downsample": false
//!     }
//!   }
//! }
//! ```

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use serde_json::Value;

use crate::config::ModelConfig;

// ─── Config ──────────────────────────────────────────────────────────────────

pub(crate) struct Phi4MMAudioCfg {
    pub(crate) audio_dim: usize,
    pub(crate) text_hidden: usize,
    pub(crate) downsample_rate: usize,
    pub(crate) use_qformer: bool,
    pub(crate) use_conv_downsample: bool,
    pub(crate) projection_cls: String,
    pub(crate) audio_token_id: u32,
}

impl Phi4MMAudioCfg {
    pub(crate) fn from_model_config(cfg: &ModelConfig) -> Option<Self> {
        let extra = &cfg.extra;

        // Conformer hidden size from audio_processor.config.attention_dim
        let audio_processor = extra.get("audio_processor")?;
        let ap_config = audio_processor.get("config")?;
        let audio_dim = ap_config
            .get("attention_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)?;

        // Audio embedding config
        let embd_layer = extra.get("embd_layer")?;
        let audio_embd = embd_layer.get("audio_embd_layer")?;
        let audio_embd_map = audio_embd.as_object()?;

        let projection_cls = audio_embd_map
            .get("projection_cls")
            .and_then(|v| v.as_str())
            .unwrap_or("mlp")
            .to_string();

        let downsample_rate = audio_embd_map
            .get("downsample_rate")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(1);

        let use_qformer = audio_embd_map
            .get("use_qformer")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let use_conv_downsample = audio_embd_map
            .get("use_conv_downsample")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let text_hidden = extra
            .get("n_embd")
            .or(Some(&Value::Null))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(cfg.hidden_size);

        let audio_token_id = extra
            .get("audio_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(200011) as u32;

        Some(Self {
            audio_dim,
            text_hidden,
            downsample_rate,
            use_qformer,
            use_conv_downsample,
            projection_cls,
            audio_token_id,
        })
    }

    /// Input size to `audio_projection`: `audio_dim * effective_downsample_rate`.
    pub(crate) fn projection_in_size(&self) -> usize {
        // If qformer or conv_ds is used, linear_downsample_rate=1 (the qformer/conv
        // already handles temporal compression).
        let rate = if self.use_qformer || self.use_conv_downsample {
            1
        } else {
            self.downsample_rate
        };
        self.audio_dim * rate
    }
}

// ─── Stub conformer ───────────────────────────────────────────────────────────

/// Stub for the NeMo-style Conformer encoder used in Phi4MM audio preprocessing.
///
/// The conformer runs during Python preprocessing and is NOT invoked during
/// Rust inference.  This struct exists to satisfy weight loading; weights are
/// intentionally ignored (not loaded) since they are not needed for the
/// pre-encoded embedding path.
///
/// TODO: Implement full conformer for native Rust audio encoding from raw mel
/// spectrograms. Requires: FeedForward (with GLU), ConvModule (depthwise-sep
/// conv + GLU), MultiHeadedAttention (GQA + T5 relative bias), positional
/// encoding, and NemoConvSubsampling. Reference: phi4mm_audio.py ConformerEncoder.
#[allow(dead_code)]
pub(crate) struct Phi4MMConformerEncoder;

#[allow(dead_code)]
impl Phi4MMConformerEncoder {
    pub(crate) fn new(_cfg: &Phi4MMAudioCfg, _vb: VarBuilder) -> Result<Self> {
        Ok(Self)
    }
}

// ─── Projection layers ────────────────────────────────────────────────────────

/// Audio projection: maps encoder output → text LLM hidden space.
///
/// Supports two layout variants from `projection_cls`:
/// - `"linear"`: single `Linear(in_size → text_hidden)`
///   weight paths: `weight`, `bias`
/// - `"mlp"`: two-layer MLP matching PyTorch `nn.Sequential`
///   weight paths: `0.{weight,bias}`, `2.{weight,bias}`
///   (index 1 is GELU, no weights)
#[allow(dead_code)]
pub(crate) struct Phi4MMAudioProjection {
    linear_0: Linear,         // in_size → text_hidden
    linear_2: Option<Linear>, // text_hidden → text_hidden  (mlp only)
}

#[allow(dead_code)]
impl Phi4MMAudioProjection {
    pub(crate) fn new(
        in_size: usize,
        text_hidden: usize,
        projection_cls: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        match projection_cls {
            "linear" => Ok(Self {
                linear_0: linear(in_size, text_hidden, vb)?,
                linear_2: None,
            }),
            _ => {
                // "mlp" (default, depth=2): Linear → GELU → Linear
                // Sequential indices: 0=Linear, 1=GELU (no weights), 2=Linear
                Ok(Self {
                    linear_0: linear(in_size, text_hidden, vb.pp("0"))?,
                    linear_2: Some(linear(text_hidden, text_hidden, vb.pp("2"))?),
                })
            }
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear_0.forward(xs)?;
        match &self.linear_2 {
            Some(l) => l.forward(&xs.gelu_erf()?),
            None => Ok(xs),
        }
    }
}

// ─── AudioEmbedding ──────────────────────────────────────────────────────────

/// Phi4MM audio embedding module (`embed_tokens_extend` in the model).
///
/// Contains the conformer encoder (preprocessing only, stub) and the
/// audio projection layers used at inference time.
pub(crate) struct Phi4MMAudioEmbedding {
    #[allow(dead_code)]
    encoder: Phi4MMConformerEncoder,
    #[allow(dead_code)]
    pub(crate) audio_projection: Phi4MMAudioProjection,
    #[allow(dead_code)]
    pub(crate) audio_projection_for_vision: Phi4MMAudioProjection,
    pub(crate) audio_token_id: u32,
}

impl Phi4MMAudioEmbedding {
    pub(crate) fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Option<Self>> {
        let Some(audio_cfg) = Phi4MMAudioCfg::from_model_config(cfg) else {
            return Ok(None);
        };

        let audio_token_id = audio_cfg.audio_token_id;
        let proj_in = audio_cfg.projection_in_size();
        let text_hidden = audio_cfg.text_hidden;
        let projection_cls = audio_cfg.projection_cls.clone();

        let encoder = Phi4MMConformerEncoder::new(&audio_cfg, vb.pp("encoder"))?;
        let audio_projection = Phi4MMAudioProjection::new(
            proj_in,
            text_hidden,
            &projection_cls,
            vb.pp("audio_projection"),
        )?;
        let audio_projection_for_vision = Phi4MMAudioProjection::new(
            proj_in,
            text_hidden,
            &projection_cls,
            vb.pp("audio_projection_for_vision"),
        )?;

        Ok(Some(Self {
            encoder,
            audio_projection,
            audio_projection_for_vision,
            audio_token_id,
        }))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

    pub fn make_audio_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_id".into(), json!(200011u32));
        extra.insert(
            "audio_processor".into(),
            json!({
                "name": "cascades",
                "config": {
                    "input_size": 8,
                    "attention_dim": 8,
                    "attention_heads": 2,
                    "linear_units": 16,
                    "num_block": 1
                }
            }),
        );
        extra.insert(
            "embd_layer".into(),
            json!({
                "audio_embd_layer": {
                    "embedding_cls": "cascades",
                    "projection_cls": "mlp",
                    "downsample_rate": 2,
                    "use_qformer": false,
                    "use_conv_downsample": false
                }
            }),
        );
        ModelConfig {
            architectures: vec!["Phi4MMForCausalLM".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        }
    }

    #[test]
    fn phi4mm_audio_cfg_parsed() {
        let cfg = make_audio_cfg();
        let audio_cfg = Phi4MMAudioCfg::from_model_config(&cfg).unwrap();
        assert_eq!(audio_cfg.audio_dim, 8);
        assert_eq!(audio_cfg.text_hidden, 8);
        assert_eq!(audio_cfg.downsample_rate, 2);
        // projection_in = audio_dim * downsample_rate = 8 * 2 = 16
        assert_eq!(audio_cfg.projection_in_size(), 16);
    }

    #[test]
    fn phi4mm_audio_embedding_new() {
        let dev = Device::Cpu;
        let cfg = make_audio_cfg();
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let result = Phi4MMAudioEmbedding::new(&cfg, vb).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn phi4mm_audio_projection_mlp_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        // projection_in=16, text_hidden=8
        let proj = Phi4MMAudioProjection::new(16, 8, "mlp", vb).unwrap();
        let xs = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &dev).unwrap();
        let out = proj.forward(&xs).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8]);
    }

    #[test]
    fn phi4mm_audio_projection_linear_shape() {
        let dev = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &dev);
        let proj = Phi4MMAudioProjection::new(16, 8, "linear", vb).unwrap();
        let xs = Tensor::zeros((1usize, 4usize, 16usize), DType::F32, &dev).unwrap();
        let out = proj.forward(&xs).unwrap();
        assert_eq!(out.dims(), &[1, 4, 8]);
    }
}
