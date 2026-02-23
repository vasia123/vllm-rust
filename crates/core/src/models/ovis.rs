//! Ovis vision-language model.
//!
//! Architecture: AIMv2 ViT + VisualTokenizer (soft vocabulary tokenization)
//! + VisualEmbedding (soft-token matmul) + LLM backbone (Llama / Qwen2).
//!
//! # Pipeline
//!
//! ```text
//! pixel_values [np, C, H, W]          (np = total patches across all images)
//!
//! → AIMv2Model                        [np, L, D]   (L = (H/patch_size)²)
//!     preprocessor: Conv2d patch embed + RMSNorm + learnable pos_embed
//!     trunk: depth × (RMSNorm → QKV-SDPA → proj; RMSNorm → fc1/fc3/fc2 SwiGLU)
//!
//! → VisualTokenizer                   [np*T, vocab_size]
//!     (optional) hidden_stride² merge: [np, L, D] → [np, L/s², D*s²]
//!     head: Linear(D*s², vocab_size−5) + LayerNorm
//!     tokenize: softmax | st_argmax | gumbel_argmax → pad to vocab_size
//!
//! → VisualEmbedding (matmul)          [np*T, hidden_size]
//!     soft_tokens @ vte.weight        (vte is [vocab_size, hidden_size])
//!
//! → merge_multimodal                  [B, S, hidden_size]
//!     overwrite image_pad_token positions in text embeddings
//!
//! → LLM (Llama | Qwen2)              logits [B, S, vocab]
//! ```
//!
//! # Weight paths
//!
//! ```text
//! visual_tokenizer.backbone.preprocessor.patchifier.proj.weight
//! visual_tokenizer.backbone.preprocessor.patchifier.norm.weight
//! visual_tokenizer.backbone.preprocessor.pos_embed
//! visual_tokenizer.backbone.trunk.blocks.{i}.{norm_1,norm_2,attn,mlp}.*
//! visual_tokenizer.head.0.weight        (Linear gate_up)
//! visual_tokenizer.head.1.{weight,bias} (LayerNorm)
//! vte.weight                            [vocab_size, hidden_size]
//! llm.model.*  /  llm.lm_head.*        (LlamaForCausalLM / Qwen2ForCausalLM)
//! ```
//!
//! # Tokenize functions
//!
//! - `softmax`: soft probability distribution over vocab
//! - `st_argmax`: straight-through argmax (one-hot at inference)
//! - `gumbel_argmax`: gumbel-softmax hard (→ one-hot at inference, no noise)
//!
//! # References
//!
//! `reference/vllm/vllm/model_executor/models/ovis.py`
//! `reference/vllm/vllm/model_executor/models/aimv2.py`
//! `reference/vllm/vllm/transformers_utils/configs/ovis.py`

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};
use serde_json::Value;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::RmsNorm;
use crate::models::llama::LlamaForCausalLM;
use crate::models::qwen2::Qwen2ForCausalLM;
use crate::multimodal::MultimodalInputs;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct AIMv2Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    rms_norm_eps: f64,
    qkv_bias: bool,
    use_bias: bool,
}

impl Default for AIMv2Config {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 24,
            num_attention_heads: 8,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            use_bias: false,
        }
    }
}

impl AIMv2Config {
    fn from_json(v: &Value) -> Self {
        let get_usize = |k: &str, d: usize| -> usize {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        let get_bool =
            |k: &str, d: bool| -> bool { v.get(k).and_then(|x| x.as_bool()).unwrap_or(d) };
        let get_f64 = |k: &str, d: f64| -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(d) };
        let def = Self::default();
        Self {
            hidden_size: get_usize("hidden_size", def.hidden_size),
            intermediate_size: get_usize("intermediate_size", def.intermediate_size),
            num_hidden_layers: get_usize("num_hidden_layers", def.num_hidden_layers),
            num_attention_heads: get_usize("num_attention_heads", def.num_attention_heads),
            num_channels: get_usize("num_channels", def.num_channels),
            image_size: get_usize("image_size", def.image_size),
            patch_size: get_usize("patch_size", def.patch_size),
            rms_norm_eps: get_f64("rms_norm_eps", def.rms_norm_eps),
            qkv_bias: get_bool("qkv_bias", def.qkv_bias),
            use_bias: get_bool("use_bias", def.use_bias),
        }
    }

    fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
enum TokenizeFunction {
    Softmax,
    StArgmax,
    GumbelArgmax,
}

#[derive(Debug, Clone)]
struct OvisTokenizerConfig {
    vocab_size: usize,
    tokenize_function: TokenizeFunction,
    hidden_stride: usize,
    drop_cls_token: bool,
    aimv2: AIMv2Config,
}

#[derive(Debug, Clone, PartialEq)]
enum OvisLlmType {
    Llama,
    Qwen2,
}

#[derive(Debug, Clone)]
struct OvisConfig {
    tokenizer: OvisTokenizerConfig,
    llm_type: OvisLlmType,
    /// Token ID used to mark image patch positions in input_ids.
    image_pad_token_id: u32,
    hidden_size: usize,
}

impl OvisConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let extra = &cfg.extra;

        // Parse visual_tokenizer_config
        let vt = extra
            .get("visual_tokenizer_config")
            .cloned()
            .unwrap_or(Value::Null);
        let backbone = vt.get("backbone_config").cloned().unwrap_or(Value::Null);
        let aimv2 = AIMv2Config::from_json(&backbone);
        let vocab_size = vt
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(16384) as usize;
        let hidden_stride = vt
            .get("hidden_stride")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let drop_cls_token = vt
            .get("drop_cls_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let tokenize_function = match vt
            .get("tokenize_function")
            .and_then(|v| v.as_str())
            .unwrap_or("softmax")
        {
            "gumbel_argmax" => TokenizeFunction::GumbelArgmax,
            "st_argmax" => TokenizeFunction::StArgmax,
            _ => TokenizeFunction::Softmax,
        };

        // Parse LLM type
        let llm_model_type = extra
            .get("text_config")
            .and_then(|v| v.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("llama");
        let llm_type = if llm_model_type.starts_with("qwen2") || llm_model_type.starts_with("qwen3")
        {
            OvisLlmType::Qwen2
        } else {
            OvisLlmType::Llama
        };

        // Image pad token id
        let image_pad_token_id: u32 = match llm_model_type {
            "qwen2" | "qwen2_5" | "qwen3" => 151655,
            "gemma2" => 7,
            _ => 128002, // llama default
        };

        let hidden_size = cfg.hidden_size;

        OvisConfig {
            tokenizer: OvisTokenizerConfig {
                vocab_size,
                tokenize_function,
                hidden_stride,
                drop_cls_token,
                aimv2,
            },
            llm_type,
            image_pad_token_id,
            hidden_size,
        }
    }
}

// ─── AIMv2 ViT ───────────────────────────────────────────────────────────────

/// Conv2d + RMSNorm patch embedding (no CLS token).
struct AIMv2PatchEmbed {
    proj: candle_nn::Conv2d,
    norm: RmsNorm,
}

impl AIMv2PatchEmbed {
    fn new(cfg: &AIMv2Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("proj"),
        )?;
        let norm = crate::layers::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self { proj, norm })
    }

    /// pixel_values `[B, C, H, W]` → patch tokens `[B, L, D]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(x)?; // [B, D, h, w]
        let (b, d, h, w) = x.dims4()?;
        let x = x.permute((0, 2, 3, 1))?.reshape((b, h * w, d))?;
        self.norm.forward(&x)
    }
}

/// Patch embed + learnable positional embeddings.
struct AIMv2Preprocessor {
    patchifier: AIMv2PatchEmbed,
    pos_embed: Tensor, // [1, num_patches, D]
}

impl AIMv2Preprocessor {
    fn new(cfg: &AIMv2Config, vb: VarBuilder) -> Result<Self> {
        let patchifier = AIMv2PatchEmbed::new(cfg, vb.pp("patchifier"))?;
        let pos_embed = vb.get((1, cfg.num_patches(), cfg.hidden_size), "pos_embed")?;
        Ok(Self {
            patchifier,
            pos_embed,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = self.patchifier.forward(x)?; // [B, L, D]
        let l = tokens.dim(1)?;
        // Slice positional embedding to the actual sequence length
        let pos = self.pos_embed.narrow(1, 0, l)?;
        tokens.broadcast_add(&pos)
    }
}

/// Standard multi-head self-attention for AIMv2.
struct AIMv2Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl AIMv2Attention {
    fn new(cfg: &AIMv2Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let qkv = if cfg.qkv_bias {
            candle_nn::linear(d, 3 * d, vb.pp("qkv"))?
        } else {
            candle_nn::linear_no_bias(d, 3 * d, vb.pp("qkv"))?
        };
        let proj = if cfg.use_bias {
            candle_nn::linear(d, d, vb.pp("proj"))?
        } else {
            candle_nn::linear_no_bias(d, d, vb.pp("proj"))?
        };
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _d) = x.dims3()?;
        let qkv = self.qkv.forward(x)?; // [B, L, 3D]
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
        // [B, L, nh*hd] → [B, nh, L, hd]
        let reshape_head = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, l, self.num_heads, self.head_dim))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let q = reshape_head(q)?;
        let k = reshape_head(k)?;
        let v = reshape_head(v)?;

        // Scaled dot-product attention (no causal mask for vision)
        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?; // [B, nh, L, hd]
        let out = out
            .permute((0, 2, 1, 3))?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

/// SwiGLU feed-forward network for AIMv2.
struct AIMv2Mlp {
    fc1: Linear, // gate
    fc3: Linear, // up
    fc2: Linear, // down
}

impl AIMv2Mlp {
    fn new(cfg: &AIMv2Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let h = cfg.intermediate_size;
        let bias = cfg.use_bias;
        let mk = if bias {
            candle_nn::linear
        } else {
            candle_nn::linear_no_bias
        };
        Ok(Self {
            fc1: mk(d, h, vb.pp("fc1"))?,
            fc3: mk(d, h, vb.pp("fc3"))?,
            fc2: mk(h, d, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: gate * sigmoid(gate) * up  — equivalently silu(gate) * up
        let gate = self.fc1.forward(x)?;
        let up = self.fc3.forward(x)?;
        let act = (gate.silu()? * up)?;
        self.fc2.forward(&act)
    }
}

/// Pre-norm transformer block: norm → attn → res; norm → mlp → res.
struct AIMv2Block {
    norm_1: RmsNorm,
    attn: AIMv2Attention,
    norm_2: RmsNorm,
    mlp: AIMv2Mlp,
}

impl AIMv2Block {
    fn new(cfg: &AIMv2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_1: crate::layers::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_1"))?,
            attn: AIMv2Attention::new(cfg, vb.pp("attn"))?,
            norm_2: crate::layers::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_2"))?,
            mlp: AIMv2Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm_1.forward(x)?)?)?;
        let x = (&x + self.mlp.forward(&self.norm_2.forward(&x)?)?)?;
        Ok(x)
    }
}

/// AIMv2 vision transformer (no CLS token, optional post-norm).
pub(crate) struct AIMv2Model {
    preprocessor: AIMv2Preprocessor,
    blocks: Vec<AIMv2Block>,
    post_trunk_norm: Option<RmsNorm>,
}

impl AIMv2Model {
    fn new(cfg: &AIMv2Config, require_post_norm: bool, vb: VarBuilder) -> Result<Self> {
        let preprocessor = AIMv2Preprocessor::new(cfg, vb.pp("preprocessor"))?;
        let trunk_vb = vb.pp("trunk");
        let blocks_vb = trunk_vb.pp("blocks");
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| AIMv2Block::new(cfg, blocks_vb.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let post_trunk_norm = if require_post_norm {
            Some(crate::layers::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                trunk_vb.pp("post_trunk_norm"),
            )?)
        } else {
            None
        };
        Ok(Self {
            preprocessor,
            blocks,
            post_trunk_norm,
        })
    }

    /// Forward: `pixel_values [B, C, H, W]` → `[B, L, D]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.preprocessor.forward(pixel_values)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        if let Some(norm) = &self.post_trunk_norm {
            x = norm.forward(&x)?;
        }
        Ok(x)
    }
}

// ─── VisualTokenizer ─────────────────────────────────────────────────────────

/// Converts raw image patches to soft visual tokens via AIMv2 + vocabulary projection.
struct VisualTokenizer {
    backbone: AIMv2Model,
    head_linear: Linear,  // Linear(D*s², vocab_size-5), no bias
    head_norm: LayerNorm, // LayerNorm(vocab_size-5)
    cfg: OvisTokenizerConfig,
}

impl VisualTokenizer {
    fn new(cfg: &OvisTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        // AIMv2 backbone has no post_trunk_norm in Ovis2 (require_post_norm=false)
        let backbone = AIMv2Model::new(&cfg.aimv2, false, vb.pp("backbone"))?;
        let in_dim = cfg.aimv2.hidden_size * cfg.hidden_stride * cfg.hidden_stride;
        let head_vocab = cfg.vocab_size - 5; // 5 reserved indicator IDs
        let head_vb = vb.pp("head");
        let head_linear = candle_nn::linear_no_bias(in_dim, head_vocab, head_vb.pp("0"))?;
        let head_norm = layer_norm(head_vocab, 1e-5, head_vb.pp("1"))?;
        Ok(Self {
            backbone,
            head_linear,
            head_norm,
            cfg: cfg.clone(),
        })
    }

    /// Encode pixel_values `[B, C, H, W]` → merged features `[B, T, D*s²]`.
    fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut features = self.backbone.forward(pixel_values)?; // [B, L, D]

        if self.cfg.drop_cls_token {
            let l = features.dim(1)?;
            features = features.narrow(1, 1, l - 1)?;
        }

        let s = self.cfg.hidden_stride;
        if s > 1 {
            let (b, l, d) = features.dims3()?;
            let sqrt_l = (l as f64).sqrt() as usize;
            // Pad if needed so spatial dims are divisible by s
            let pad = (s - (sqrt_l % s)) % s;
            let (sqrt_lp, lp) = (sqrt_l + pad, (sqrt_l + pad) * (sqrt_l + pad));
            let features = if pad > 0 {
                // Zero-pad the spatial dimensions: [B, l, d] → [B, l_padded, d]
                let zeros = Tensor::zeros((b, lp - l, d), features.dtype(), features.device())?;
                Tensor::cat(&[&features, &zeros], 1)?
            } else {
                features
            };
            // [B, sqrt_lp, sqrt_lp, D]
            let f = features.reshape((b, sqrt_lp, sqrt_lp, d))?;
            // Fold s×s patches: [B, sqrt_lp/s, s, sqrt_lp/s, s, D]
            let hs = sqrt_lp / s;
            let f = f.reshape((b, hs, s, hs, s, d))?;
            // [B, hs, hs, s, s, D] → [B, hs, hs, s*s*D]
            let f = f
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((b, hs * hs, s * s * d))?;
            return Ok(f);
        }

        Ok(features)
    }

    /// Tokenize logits to soft one-hot or softmax.
    fn tokenize(&self, logits: &Tensor) -> Result<Tensor> {
        match &self.cfg.tokenize_function {
            TokenizeFunction::Softmax => candle_nn::ops::softmax(logits, D::Minus1),
            TokenizeFunction::StArgmax | TokenizeFunction::GumbelArgmax => {
                // Hard one-hot from argmax (inference: no gumbel noise)
                let idx = logits.argmax(D::Minus1)?; // [B, T]
                let vocab = logits.dim(D::Minus1)?;
                let device = logits.device();
                // Broadcast compare: [B, T, 1] eq [vocab_size]
                let range = Tensor::arange(0u32, vocab as u32, device)?;
                let idx_exp = idx.unsqueeze(D::Minus1)?;
                idx_exp.broadcast_eq(&range)?.to_dtype(logits.dtype())
            }
        }
    }

    /// pixel_values `[B, C, H, W]` → soft tokens `[B, T, vocab_size]`.
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let features = self.encode(pixel_values)?; // [B, T, D*s²]
        let logits = self.head_linear.forward(&features)?; // [B, T, vocab-5]
        let logits = self.head_norm.forward(&logits)?;
        let tokens = self.tokenize(&logits)?; // [B, T, vocab-5]
                                              // Pad 5 zeros for indicator IDs: [B, T, vocab]
        let (b, t, _) = tokens.dims3()?;
        let pad = Tensor::zeros((b, t, 5), tokens.dtype(), tokens.device())?;
        Tensor::cat(&[&tokens, &pad], D::Minus1)
    }
}

// ─── OvisLlm enum ────────────────────────────────────────────────────────────

/// Unified wrapper for the LLM backbone.
enum OvisLlm {
    Llama(Box<LlamaForCausalLM>),
    Qwen2(Box<Qwen2ForCausalLM>),
}

impl OvisLlm {
    fn new(llm_type: &OvisLlmType, cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        match llm_type {
            OvisLlmType::Llama => Ok(Self::Llama(Box::new(LlamaForCausalLM::new(cfg, vb)?))),
            OvisLlmType::Qwen2 => Ok(Self::Qwen2(Box::new(Qwen2ForCausalLM::new(cfg, vb)?))),
        }
    }

    fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.embed_text(input_ids),
            Self::Qwen2(m) => m.embed_text(input_ids),
        }
    }

    fn forward_with_embeddings(
        &self,
        embeddings: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward_with_embeddings(
                embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
            Self::Qwen2(m) => m.forward_with_embeddings(
                embeddings,
                seqlen_offset,
                kv_cache_mgr,
                block_table,
                slot_mapping,
            ),
        }
    }

    fn forward_decode_batch_with_embeddings(
        &self,
        embeddings: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        match self {
            Self::Llama(m) => {
                m.forward_decode_batch_with_embeddings(embeddings, sequences, kv_cache_mgr)
            }
            Self::Qwen2(m) => {
                m.forward_decode_batch_with_embeddings(embeddings, sequences, kv_cache_mgr)
            }
        }
    }
}

// ─── merge_multimodal ────────────────────────────────────────────────────────

/// Splice pre-computed image embeddings into the text embedding sequence.
///
/// Replaces positions marked by `im_pad_token_id` in `input_ids` with
/// consecutive rows from `image_embeds`.
fn merge_multimodal(
    text_embeds: &Tensor,
    image_embeds: &Tensor,
    input_ids: &Tensor,
    im_pad_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    let (batch_size, seq_len, hidden) = text_embeds.dims3()?;
    let ids_flat = input_ids
        .to_dtype(DType::U32)?
        .reshape(batch_size * seq_len)?;
    let ids_vec = ids_flat.to_vec1::<u32>()?;

    let mut merged = text_embeds.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let img_rows = image_embeds.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut img_cursor = 0usize;
    for (flat_pos, &id) in ids_vec.iter().enumerate() {
        if id == im_pad_token_id && img_cursor < img_rows.len() {
            let b = flat_pos / seq_len;
            let s = flat_pos % seq_len;
            if b < batch_size && s < seq_len {
                merged[b][s] = img_rows[img_cursor].clone();
                img_cursor += 1;
            }
        }
    }

    let _ = hidden; // used implicitly through merged structure
    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main model ──────────────────────────────────────────────────────────────

/// Ovis vision-language model.
pub struct OvisForConditionalGeneration {
    visual_tokenizer: VisualTokenizer,
    /// Visual token embedding weight: `[vocab_size, hidden_size]`.
    vte_weight: Tensor,
    llm: OvisLlm,
    cfg: OvisConfig,
    device: Device,
}

impl OvisForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ovis_cfg = OvisConfig::from_model_config(cfg);
        let device = vb.device().clone();

        let visual_tokenizer =
            VisualTokenizer::new(&ovis_cfg.tokenizer, vb.pp("visual_tokenizer"))?;

        let vte_weight = vb.pp("vte").get(
            (ovis_cfg.tokenizer.vocab_size, ovis_cfg.hidden_size),
            "weight",
        )?;

        // LLM is at "llm.*" in checkpoint: llm.model.* / llm.lm_head.*
        let llm = OvisLlm::new(&ovis_cfg.llm_type, cfg, vb.pp("llm"))?;

        Ok(Self {
            visual_tokenizer,
            vte_weight,
            llm,
            cfg: ovis_cfg,
            device,
        })
    }

    /// Encode all image patches through the visual tokenizer + vte.
    ///
    /// Returns `[total_visual_tokens, hidden_size]`.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [np, C, H, W]
        let soft_tokens = self.visual_tokenizer.forward(pixel_values)?; // [np, T, vocab]
        let (np, t, _) = soft_tokens.dims3()?;
        let tokens_flat = soft_tokens.reshape((np * t, self.cfg.tokenizer.vocab_size))?;
        // soft visual tokens @ vte_weight: [np*T, vocab] × [vocab, hidden] → [np*T, hidden]
        tokens_flat.matmul(&self.vte_weight)
    }
}

impl crate::engine::ModelForward for OvisForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Text-only path (no image)
        self.llm.forward_with_embeddings(
            &self.llm.embed_text(input_ids)?,
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
        self.llm.forward_decode_batch_with_embeddings(
            &self.llm.embed_text(input_ids)?,
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
        let text_embeds = self.llm.embed_text(input_ids)?;
        let merged = if let Some(mm) = multimodal_inputs {
            if mm.has_images() {
                // Collect all pixel_values into one batch
                let all_pv: Vec<&Tensor> = mm
                    .image_embeddings
                    .iter()
                    .map(|(_, pi)| &pi.embedding)
                    .collect();
                // image_embeddings here are pre-computed visual token embeddings
                // stored at positions in the sequence
                merge_multimodal(
                    &text_embeds,
                    &Tensor::cat(&all_pv, 0)?,
                    input_ids,
                    self.cfg.image_pad_token_id,
                    &self.device,
                )?
            } else {
                text_embeds
            }
        } else {
            text_embeds
        };
        self.llm.forward_with_embeddings(
            &merged,
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
    use crate::kv_cache::KVCacheDtype;
    use candle_core::DType;

    fn tiny_aimv2_cfg() -> AIMv2Config {
        AIMv2Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_channels: 1,
            image_size: 8,
            patch_size: 4,
            rms_norm_eps: 1e-5,
            qkv_bias: false,
            use_bias: false,
        }
    }

    fn tiny_tokenizer_cfg() -> OvisTokenizerConfig {
        OvisTokenizerConfig {
            vocab_size: 16,
            tokenize_function: TokenizeFunction::Softmax,
            hidden_stride: 1,
            drop_cls_token: false,
            aimv2: tiny_aimv2_cfg(),
        }
    }

    #[test]
    fn test_ovis_aimv2_new() {
        let device = Device::Cpu;
        let cfg = tiny_aimv2_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AIMv2Model::new(&cfg, false, vb);
        assert!(model.is_ok(), "{:?}", model.err());
    }

    #[test]
    fn test_ovis_aimv2_forward() {
        let device = Device::Cpu;
        let cfg = tiny_aimv2_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = AIMv2Model::new(&cfg, false, vb).unwrap();
        // image_size=8, patch_size=4 → 4 patches per image
        let pixel_values = Tensor::zeros((2, 1, 8, 8), DType::F32, &device).unwrap();
        let out = model.forward(&pixel_values).unwrap();
        assert_eq!(out.dims(), &[2, 4, 16]); // [B, L, D]
    }

    #[test]
    fn test_ovis_visual_tokenizer_forward() {
        let device = Device::Cpu;
        let cfg = tiny_tokenizer_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tokenizer = VisualTokenizer::new(&cfg, vb).unwrap();
        // 2 patches, image_size=8, patch_size=4
        let pixel_values = Tensor::ones((2, 1, 8, 8), DType::F32, &device).unwrap();
        let out = tokenizer.forward(&pixel_values).unwrap();
        // hidden_stride=1 → T=4, vocab_size=16
        assert_eq!(out.dims(), &[2, 4, 16]);
    }

    #[test]
    fn test_ovis_visual_tokenizer_st_argmax() {
        let device = Device::Cpu;
        let mut cfg = tiny_tokenizer_cfg();
        cfg.tokenize_function = TokenizeFunction::StArgmax;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tokenizer = VisualTokenizer::new(&cfg, vb).unwrap();
        let pixel_values = Tensor::ones((1, 1, 8, 8), DType::F32, &device).unwrap();
        let out = tokenizer.forward(&pixel_values).unwrap();
        // One-hot: each row sums to 1, dtype F32
        assert_eq!(out.dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_ovis_hidden_stride() {
        let device = Device::Cpu;
        let mut cfg = tiny_tokenizer_cfg();
        cfg.hidden_stride = 2;
        // AIMv2 with 4 patches, stride=2 → merge 2×2 → 1 token per image
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tokenizer = VisualTokenizer::new(&cfg, vb).unwrap();
        let pixel_values = Tensor::zeros((1, 1, 8, 8), DType::F32, &device).unwrap();
        let out = tokenizer.forward(&pixel_values).unwrap();
        // T = 4/4 = 1 (merged), vocab_size=16
        assert_eq!(out.dims(), &[1, 1, 16]);
    }

    fn tiny_ovis_model_config() -> ModelConfig {
        use serde_json::json;
        ModelConfig {
            architectures: vec!["OvisForConditionalGeneration".to_string()],
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 256,
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
            extra: serde_json::from_value(json!({
                "visual_tokenizer_config": {
                    "vocab_size": 16,
                    "tokenize_function": "softmax",
                    "hidden_stride": 1,
                    "drop_cls_token": false,
                    "backbone_config": {
                        "model_type": "aimv2",
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 2,
                        "num_channels": 1,
                        "image_size": 8,
                        "patch_size": 4,
                        "rms_norm_eps": 1e-5,
                        "qkv_bias": false,
                        "use_bias": false
                    }
                },
                "text_config": {
                    "model_type": "llama"
                }
            }))
            .unwrap(),
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        use crate::kv_cache::config::CacheConfig;
        let cache_cfg = CacheConfig {
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
        KVCacheManager::new(&cache_cfg).unwrap()
    }

    #[test]
    fn test_ovis_forward_text_only() {
        let device = Device::Cpu;
        let cfg = tiny_ovis_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OvisForConditionalGeneration::new(&cfg, vb).unwrap();
        let mut kv_cache = make_cache(&cfg, &device);

        let seq_len = 4usize;
        let mut bt = crate::kv_cache::BlockTable::new(4);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();

        use crate::engine::ModelForward;
        let out = model.forward(&input_ids, 0, &mut kv_cache, &bt, &slot_mapping);
        assert!(out.is_ok(), "{:?}", out.err());
        let out = out.unwrap();
        assert_eq!(out.dims()[0], 1);
        assert_eq!(out.dims()[2], cfg.vocab_size);
    }
}
