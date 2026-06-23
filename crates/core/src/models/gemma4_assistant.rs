//! Gemma 4 MTP "assistant" drafter (`arch = gemma4_assistant`).
//!
//! Unlike the DeepSeek-style MTP that lives inside the target checkpoint, the
//! Gemma 4 assistant is a SEPARATE lightweight decoder used for speculative
//! decoding. Its distinguishing traits (mirrored from
//! `reference/llama.cpp/src/models/gemma4-assistant.cpp`):
//!
//! - **Hidden-state fusion**: the input is `concat(target_token_embedding,
//!   backbone_hidden)` of width `2 * n_embd_backbone`, projected to the
//!   assistant's (small) hidden via `mtp.pre_projection`. The token embedding
//!   comes from the TARGET model's table (scaled by `sqrt(n_embd_backbone)`),
//!   not the assistant's own — the assistant's `token_embd` is the tied
//!   `lm_head` only.
//! - **Q-only attention**: every layer computes only Q (+ q_norm + RoPE) and
//!   reads K/V from the TARGET model's KV cache. There are no K/V projections.
//!   The caller supplies the (already per-layer-shaped) target K/V; this module
//!   never allocates or writes a KV cache. SWA layers read the target's last
//!   sliding layer, the full layer reads the target's last full layer (see
//!   `reference/llama.cpp/src/llama-model.cpp` `share()`).
//! - **Output**: `output_norm` → tied `lm_head` logits, plus `mtp.post_projection`
//!   back to backbone width as the feedback hidden for the next draft step.
//! - **Scale = 1.0** attention (q/k RMSNorms carry the magnitude — same as the
//!   Gemma 4 backbone's naive path), GELU-tanh parallel MLP, per-layer
//!   `layer_output_scale` scalar.
//!
//! Every draft step processes a single token, all at the SAME position
//! (`n_past`) — the assistant has no KV of its own, so there is never a prefill
//! pass and draft tokens do not attend to each other, only to the backbone
//! context. Weights are F16; we load them dense (the model is ~80 MB) rather
//! than through the quantized-linear path.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::layers::attention::repeat_kv;
use crate::layers::{causal_mask, rms_norm, RmsNorm, RotaryEmbedding};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Parsed `gemma4_assistant.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Gemma4AssistantConfig {
    pub n_layer: usize,
    /// Assistant's own (small) hidden width.
    pub n_embd: usize,
    /// Backbone/target hidden width — the fusion input and feedback width.
    pub n_embd_backbone: usize,
    pub n_ff: usize,
    pub n_head: usize,
    /// KV-head count PER LAYER. Gemma 4 assistants are heterogeneous: the 12B
    /// drafter has `[8, 8, 8, 1]` (sliding layers GQA-8, full layer MQA-1),
    /// while E4B ships a scalar (broadcast to every layer).
    pub n_head_kv: Vec<usize>,
    pub vocab_size: usize,
    pub rms_eps: f64,
    pub rope_base_full: f64,
    pub rope_base_swa: f64,
    /// Per-head dim on full-attention layers (GGUF `attention.key_length`).
    pub head_dim_full: usize,
    /// Per-head dim on sliding layers (GGUF `attention.key_length_swa`).
    pub head_dim_swa: usize,
    pub sliding_window: usize,
    /// `true` = sliding layer, `false` = full layer. One entry per layer.
    pub is_swa: Vec<bool>,
}

fn meta_get<'a>(meta: &'a HashMap<String, Value>, key: &str) -> Result<&'a Value> {
    meta.get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("assistant GGUF missing metadata '{key}'")))
}

fn meta_u64(meta: &HashMap<String, Value>, key: &str) -> Result<u64> {
    let v = meta_get(meta, key)?;
    if let Ok(x) = v.to_u32() {
        return Ok(x as u64);
    }
    if let Ok(x) = v.to_u64() {
        return Ok(x);
    }
    if let Ok(x) = v.to_i32() {
        return Ok(x as u64);
    }
    if let Ok(x) = v.to_i64() {
        return Ok(x as u64);
    }
    Err(candle_core::Error::Msg(format!(
        "assistant GGUF metadata '{key}' is not an integer"
    )))
}

fn meta_f64(meta: &HashMap<String, Value>, key: &str) -> Result<f64> {
    let v = meta_get(meta, key)?;
    if let Ok(x) = v.to_f32() {
        return Ok(x as f64);
    }
    if let Ok(x) = v.to_f64() {
        return Ok(x);
    }
    Err(candle_core::Error::Msg(format!(
        "assistant GGUF metadata '{key}' is not a float"
    )))
}

/// First non-empty integer value for `key`, trying `meta`'s key under either
/// assistant arch prefix (the two community converters disagree, e.g.
/// `n_embd_backbone` vs `embedding_length_out`). Returns `Ok` for the first
/// present key, else `Err`.
fn meta_u64_any(meta: &HashMap<String, Value>, keys: &[String]) -> Result<u64> {
    for k in keys {
        if meta.contains_key(k) {
            return meta_u64(meta, k);
        }
    }
    Err(candle_core::Error::Msg(format!(
        "assistant GGUF missing any of {keys:?}"
    )))
}

/// Parse `head_count_kv`, which is either a scalar (E4B → broadcast to every
/// layer) or a per-layer array (12B → `[8, 8, 8, 1]`). Returns one entry per
/// layer.
fn meta_head_kv(meta: &HashMap<String, Value>, key: &str, n_layer: usize) -> Result<Vec<usize>> {
    let v = meta_get(meta, key)?;
    if let Ok(arr) = v.to_vec() {
        let out: Vec<usize> = arr
            .iter()
            .map(|e| {
                e.to_u32()
                    .map(|x| x as usize)
                    .or_else(|_| e.to_i32().map(|x| x as usize))
                    .or_else(|_| e.to_u64().map(|x| x as usize))
                    .map_err(|_| candle_core::Error::Msg("head_count_kv element not int".into()))
            })
            .collect::<Result<Vec<usize>>>()?;
        if out.len() != n_layer {
            return Err(candle_core::Error::Msg(format!(
                "head_count_kv array len {} != block_count {n_layer}",
                out.len()
            )));
        }
        Ok(out)
    } else {
        let scalar = meta_u64(meta, key)? as usize;
        Ok(vec![scalar; n_layer])
    }
}

impl Gemma4AssistantConfig {
    /// Build from the GGUF metadata map and the `token_embd.weight` shape
    /// (`[vocab, n_embd]`, candle orientation).
    pub fn from_metadata(
        meta: &HashMap<String, Value>,
        token_embd_shape: &[usize],
    ) -> Result<Self> {
        let arch = meta_get(meta, "general.architecture")?
            .to_string()
            .map_err(|_| candle_core::Error::Msg("general.architecture not a string".into()))?
            .clone();
        // Two community converters use different spellings: AtomicChat's E4B is
        // `gemma4_assistant`, unsloth's 12B is `gemma4-assistant`. Both are valid.
        if arch != "gemma4_assistant" && arch != "gemma4-assistant" {
            return Err(candle_core::Error::Msg(format!(
                "expected arch 'gemma4_assistant'/'gemma4-assistant', got '{arch}'"
            )));
        }
        let k = |s: &str| format!("{arch}.{s}");

        let n_layer = meta_u64(meta, &k("block_count"))? as usize;
        let is_swa: Vec<bool> = {
            let v = meta_get(meta, &k("attention.sliding_window_pattern"))?;
            let arr = v.to_vec().map_err(|_| {
                candle_core::Error::Msg("sliding_window_pattern not an array".into())
            })?;
            arr.iter()
                .map(|e| {
                    e.to_bool().map_err(|_| {
                        candle_core::Error::Msg("sliding_window_pattern element not bool".into())
                    })
                })
                .collect::<Result<Vec<bool>>>()?
        };
        if is_swa.len() != n_layer {
            return Err(candle_core::Error::Msg(format!(
                "sliding_window_pattern len {} != block_count {n_layer}",
                is_swa.len()
            )));
        }

        Ok(Self {
            n_layer,
            n_embd: meta_u64(meta, &k("embedding_length"))? as usize,
            // AtomicChat E4B uses `n_embd_backbone`; unsloth 12B uses
            // `embedding_length_out`. Both name the target's hidden width.
            n_embd_backbone: meta_u64_any(meta, &[k("n_embd_backbone"), k("embedding_length_out")])?
                as usize,
            n_ff: meta_u64(meta, &k("feed_forward_length"))? as usize,
            n_head: meta_u64(meta, &k("attention.head_count"))? as usize,
            n_head_kv: meta_head_kv(meta, &k("attention.head_count_kv"), n_layer)?,
            vocab_size: token_embd_shape[0],
            rms_eps: meta_f64(meta, &k("attention.layer_norm_rms_epsilon"))?,
            rope_base_full: meta_f64(meta, &k("rope.freq_base"))?,
            rope_base_swa: meta_f64(meta, &k("rope.freq_base_swa"))?,
            head_dim_full: meta_u64(meta, &k("attention.key_length"))? as usize,
            head_dim_swa: meta_u64(meta, &k("attention.key_length_swa"))? as usize,
            sliding_window: meta_u64(meta, &k("attention.sliding_window"))? as usize,
            is_swa,
        })
    }

    fn head_dim(&self, layer_idx: usize) -> usize {
        if self.is_swa[layer_idx] {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    fn n_head_kv_at(&self, layer_idx: usize) -> usize {
        self.n_head_kv[layer_idx]
    }

    /// KV-head count of the assistant's sliding layers (all share the same
    /// target sliding-layer KV). Falls back to the first layer if none sliding.
    pub fn swa_kv_heads(&self) -> usize {
        self.is_swa
            .iter()
            .position(|&s| s)
            .map(|i| self.n_head_kv[i])
            .unwrap_or(self.n_head_kv[0])
    }

    /// KV-head count of the assistant's full layer.
    pub fn full_kv_heads(&self) -> usize {
        self.is_swa
            .iter()
            .position(|&s| !s)
            .map(|i| self.n_head_kv[i])
            .unwrap_or(self.n_head_kv[self.n_layer - 1])
    }
}

// ─── Layer ───────────────────────────────────────────────────────────────────

struct Gemma4AssistantLayer {
    attn_norm: RmsNorm,
    q_proj: Linear,
    q_norm: RmsNorm,
    o_proj: Linear,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    post_ffw_norm: RmsNorm,
    /// Per-layer output scalar (`blk.N.layer_output_scale.weight`, shape `[1]`).
    output_scale: Tensor,
    rotary: RotaryEmbedding,
    is_swa: bool,
    sliding_window: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
}

impl Gemma4AssistantLayer {
    /// Per-head RMSNorm over the last dim of a `[b, h, s, d]` tensor.
    fn per_head_norm(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        norm.forward(&x.reshape((b * h * s, d))?)?
            .reshape((b, h, s, d))
    }

    /// One decoder layer. `inpl` is `[1, q_len, n_embd]`; K/V are the target's
    /// cached `[1, n_head_kv, ctx, head_dim]` for this layer's attention type.
    fn forward(&self, inpl: &Tensor, position: usize, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, q_len, _) = inpl.dims3()?;

        let cur = self.attn_norm.forward(inpl)?;
        let q = self
            .q_proj
            .forward(&cur)?
            .reshape((b, q_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let q = Self::per_head_norm(&q, &self.q_norm)?;
        let (q, _) = self.rotary.apply(&q, &q, position)?;

        let attn = self.attention(&q, k, v, q_len)?;
        let attn = self.o_proj.forward(&attn)?;
        let attn = self.post_attention_norm.forward(&attn)?;
        let attn_out = (attn + inpl)?;

        let ff = self.ffn_norm.forward(&attn_out)?;
        let gate = self.gate_proj.forward(&ff)?.gelu()?; // GELU-tanh (ggml LLM_FFN_GELU)
        let up = self.up_proj.forward(&ff)?;
        let ff = self.down_proj.forward(&(gate * up)?)?;
        let ff = self.post_ffw_norm.forward(&ff)?;
        let cur = (ff + attn_out)?;

        cur.broadcast_mul(&self.output_scale)
    }

    /// Q-only attention against externally supplied target K/V. No `1/sqrt(d)`
    /// scaling (Gemma 4 folds it into q/k RMSNorm). The query is the newest
    /// position, so the mask is causal (windowed for sliding layers).
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, q_len: usize) -> Result<Tensor> {
        let (b, _, _, head_dim) = q.dims4()?;
        let ctx = k.dim(2)?;
        let offset = ctx - q_len;

        let groups = self.n_head / self.n_head_kv;
        let k = repeat_kv(k.clone(), groups)?;
        let v = repeat_kv(v.clone(), groups)?;

        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let mask = if self.is_swa {
            sliding_window_mask(
                q_len,
                ctx,
                offset,
                self.sliding_window,
                q.dtype(),
                q.device(),
            )?
        } else {
            causal_mask(q_len, offset, q.dtype(), q.device())?
        };
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [b, n_head, q_len, head_dim]
        out.transpose(1, 2)?
            .reshape((b, q_len, self.n_head * head_dim))
    }
}

/// Sliding-window causal mask: key `j` is visible to query at `offset+i` iff
/// `j <= offset+i` and `offset+i - j < window`. Shape `[1, 1, q_len, ctx]`.
fn sliding_window_mask(
    q_len: usize,
    ctx: usize,
    offset: usize,
    window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let data: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            let qpos = i + offset;
            (0..ctx).map(move |j| {
                if j > qpos || qpos - j >= window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, q_len, ctx), device)?.to_dtype(dtype)
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// The Gemma 4 MTP assistant drafter.
pub struct Gemma4Assistant {
    pre_projection: Linear,
    post_projection: Linear,
    layers: Vec<Gemma4AssistantLayer>,
    output_norm: RmsNorm,
    /// Tied `token_embd` over the 262k vocab, kept QUANTIZED-resident (a
    /// `QMatMul`, not dense). For the 12B drafter this is `[262144, 1024]`:
    /// 536 MB dense (bf16) vs ~284 MB at Q8_0 — the saving that lets 12B +
    /// drafter leave forward-scratch headroom on an 8 GB GPU.
    lm_head: QMatMul,
    config: Gemma4AssistantConfig,
    /// RoPE cos/sin tables cover `[0, max_pos)`; the drafter must not be asked
    /// for a position beyond this (the caller falls back to plain decode).
    max_pos: usize,
    device: Device,
    dtype: DType,
}

impl Gemma4Assistant {
    pub fn config(&self) -> &Gemma4AssistantConfig {
        &self.config
    }

    /// Largest RoPE position the drafter can encode.
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Load the assistant from a GGUF file. `max_pos` caps the RoPE cos/sin
    /// cache (use the target's max sequence length; the draft position never
    /// exceeds it). Weights are dequantized to `dtype`.
    pub fn load(
        path: &std::path::Path,
        max_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        use crate::quantization::gguf::header::GgufHeader;
        use crate::quantization::gguf::{GgufVarBuilderBackend, GgufWeightLoader};

        let header = {
            let f = std::fs::File::open(path)
                .map_err(|e| candle_core::Error::Msg(format!("open assistant GGUF: {e}")))?;
            GgufHeader::read(std::io::BufReader::new(f))?
        };
        let token_embd_shape = header
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| {
                candle_core::Error::Msg("assistant GGUF missing token_embd.weight".into())
            })?
            .shape
            .clone();
        let config = Gemma4AssistantConfig::from_metadata(&header.metadata, &token_embd_shape)?;

        let loader = GgufWeightLoader::from_path(path, device.clone(), dtype)?;
        let tensors = loader.shared_tensors();

        // Tied lm_head: keep `token_embd` QUANTIZED-resident (`QMatMul` over the
        // raw QTensor) instead of dequantizing the 262k×n_embd table to dense.
        let lm_head = {
            use crate::quantization::gguf::GgufTensor;
            match tensors.get("token_embd.weight") {
                Some(GgufTensor::Candle(qt)) => QMatMul::from_arc(qt.clone())
                    .map_err(|e| candle_core::Error::Msg(format!("assistant lm_head: {e}")))?,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "assistant token_embd.weight missing or not a candle QTensor".into(),
                    ))
                }
            }
        };

        let backend = Box::new(GgufVarBuilderBackend::new(
            Arc::clone(&tensors),
            device.clone(),
        ));
        let vb = VarBuilder::from_backend(backend, dtype, device.clone());

        Self::new(&config, vb, lm_head, max_pos, dtype, device)
    }

    /// Build from a GGUF-backed `VarBuilder` whose `get` resolves literal
    /// llama.cpp tensor names (`blk.N.*`, `mtp.*`, `token_embd.weight`, ...).
    pub fn new(
        config: &Gemma4AssistantConfig,
        vb: VarBuilder,
        lm_head: QMatMul,
        max_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let nb = config.n_embd_backbone;
        let ne = config.n_embd;

        // The fusion projections are named `mtp.*` by AtomicChat (E4B) and
        // `nextn.*` by unsloth (12B). Try both.
        let get_either = |shape: (usize, usize), a: &str, b: &str| -> Result<Tensor> {
            vb.get(shape, a).or_else(|_| vb.get(shape, b))
        };
        let pre_w = get_either(
            (ne, 2 * nb),
            "mtp.pre_projection.weight",
            "nextn.pre_projection.weight",
        )?;
        let pre_projection = Linear::new(pre_w, None);
        let post_w = get_either(
            (nb, ne),
            "mtp.post_projection.weight",
            "nextn.post_projection.weight",
        )?;
        let post_projection = Linear::new(post_w, None);

        // rope_freqs (full-layer freq factors), if the GGUF ships them.
        let rope_freqs: Option<Vec<f32>> =
            match vb.get((config.head_dim_full / 2,), "rope_freqs.weight") {
                Ok(t) => Some(t.to_dtype(DType::F32)?.to_vec1::<f32>()?),
                Err(_) => None,
            };

        let mut layers = Vec::with_capacity(config.n_layer);
        for il in 0..config.n_layer {
            let head_dim = config.head_dim(il);
            let is_swa = config.is_swa[il];
            let p = format!("blk.{il}");

            let q_proj = Linear::new(
                vb.get(
                    (config.n_head * head_dim, ne),
                    &format!("{p}.attn_q.weight"),
                )?,
                None,
            );
            let o_proj = Linear::new(
                vb.get(
                    (ne, config.n_head * head_dim),
                    &format!("{p}.attn_output.weight"),
                )?,
                None,
            );
            let gate_proj = Linear::new(
                vb.get((config.n_ff, ne), &format!("{p}.ffn_gate.weight"))?,
                None,
            );
            let up_proj = Linear::new(
                vb.get((config.n_ff, ne), &format!("{p}.ffn_up.weight"))?,
                None,
            );
            let down_proj = Linear::new(
                vb.get((ne, config.n_ff), &format!("{p}.ffn_down.weight"))?,
                None,
            );

            let attn_norm = rms_norm(ne, config.rms_eps, vb.pp(format!("{p}.attn_norm")))?;
            let q_norm = rms_norm(head_dim, config.rms_eps, vb.pp(format!("{p}.attn_q_norm")))?;
            let post_attention_norm = rms_norm(
                ne,
                config.rms_eps,
                vb.pp(format!("{p}.post_attention_norm")),
            )?;
            let ffn_norm = rms_norm(ne, config.rms_eps, vb.pp(format!("{p}.ffn_norm")))?;
            let post_ffw_norm = rms_norm(ne, config.rms_eps, vb.pp(format!("{p}.post_ffw_norm")))?;
            let output_scale = vb.get((1,), &format!("{p}.layer_output_scale.weight"))?;

            let rotary = if is_swa {
                RotaryEmbedding::new(head_dim, max_pos, config.rope_base_swa, dtype, device)?
            } else if let Some(ff) = &rope_freqs {
                RotaryEmbedding::new_with_freq_factors(
                    head_dim,
                    max_pos,
                    config.rope_base_full,
                    ff,
                    dtype,
                    device,
                )?
            } else {
                RotaryEmbedding::new(head_dim, max_pos, config.rope_base_full, dtype, device)?
            };

            layers.push(Gemma4AssistantLayer {
                attn_norm,
                q_proj,
                q_norm,
                o_proj,
                post_attention_norm,
                ffn_norm,
                gate_proj,
                up_proj,
                down_proj,
                post_ffw_norm,
                output_scale,
                rotary,
                is_swa,
                sliding_window: config.sliding_window,
                n_head: config.n_head,
                n_head_kv: config.n_head_kv_at(il),
                head_dim,
            });
        }

        let output_norm = rms_norm(ne, config.rms_eps, vb.pp("output_norm"))?;
        // `lm_head` (tied `token_embd`) is supplied quantized-resident by the caller.

        Ok(Self {
            pre_projection,
            post_projection,
            layers,
            output_norm,
            lm_head,
            config: config.clone(),
            max_pos,
            device: device.clone(),
            dtype,
        })
    }

    /// One draft forward.
    ///
    /// - `target_embed`: `[1, q_len, n_embd_backbone]` — the target model's
    ///   token embedding of the draft input id(s), already scaled by
    ///   `sqrt(n_embd_backbone)` (i.e. the target's `embed_text` output).
    /// - `backbone_hidden`: `[1, q_len, n_embd_backbone]` — the FIXED target
    ///   pre-final-norm hidden state for this proposal round.
    /// - `position`: the draft position (`n_past`), shared by all draft tokens.
    /// - `kv_swa` / `kv_full`: target K/V `(k, v)` for the sliding / full
    ///   attention layers, each `[1, n_head_kv, ctx, head_dim]` already sliced
    ///   to this assistant's per-layer geometry.
    ///
    /// Returns `(logits [1, q_len, vocab], h_next [1, q_len, n_embd_backbone])`.
    pub fn forward(
        &self,
        target_embed: &Tensor,
        backbone_hidden: &Tensor,
        position: usize,
        kv_swa: (&Tensor, &Tensor),
        kv_full: (&Tensor, &Tensor),
    ) -> Result<(Tensor, Tensor)> {
        let xh = Tensor::cat(&[target_embed, backbone_hidden], D::Minus1)?;
        let mut inpl = self.pre_projection.forward(&xh)?;
        for layer in &self.layers {
            let (k, v) = if layer.is_swa { kv_swa } else { kv_full };
            inpl = layer.forward(&inpl, position, k, v)?;
        }
        let normed = self.output_norm.forward(&inpl)?;
        // The quantized `QMatMul` lm_head dequantizes its weight and computes in
        // F32, so it needs an F32 activation (the rest of the model runs in the
        // target compute dtype, e.g. bf16). Logits stay F32 for sampling.
        let logits = self.lm_head.forward(&normed.to_dtype(DType::F32)?)?;
        let h_next = self.post_projection.forward(&inpl)?;
        Ok((logits, h_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASSISTANT: &str =
        "/home/vasis/gguf-models/gemma-4-E4B-assistant/gemma-4-E4B-it-assistant.F16.gguf";

    #[test]
    fn loads_and_forwards_real_assistant() {
        if !std::path::Path::new(ASSISTANT).exists() {
            eprintln!("skip: assistant model file not present");
            return;
        }
        let device = Device::Cpu;
        let dtype = DType::F32;
        let model = Gemma4Assistant::load(std::path::Path::new(ASSISTANT), 256, dtype, &device)
            .expect("load assistant");

        let cfg = model.config();
        assert_eq!(cfg.n_layer, 4);
        assert_eq!(cfg.n_embd, 256);
        assert_eq!(cfg.n_embd_backbone, 2560);
        assert_eq!(cfg.head_dim_swa, 256);
        assert_eq!(cfg.head_dim_full, 512);
        assert_eq!(cfg.is_swa, vec![true, true, true, false]);

        // Synthetic single-token draft against a 5-token backbone context.
        let ctx = 5usize;
        let position = ctx - 1;
        let nb = cfg.n_embd_backbone;
        let target_embed = Tensor::randn(0f32, 1.0, (1, 1, nb), &device).unwrap();
        let backbone_hidden = Tensor::randn(0f32, 1.0, (1, 1, nb), &device).unwrap();
        let kvh_swa = cfg.swa_kv_heads();
        let kvh_full = cfg.full_kv_heads();
        let k_swa = Tensor::randn(0f32, 1.0, (1, kvh_swa, ctx, cfg.head_dim_swa), &device).unwrap();
        let v_swa = Tensor::randn(0f32, 1.0, (1, kvh_swa, ctx, cfg.head_dim_swa), &device).unwrap();
        let k_full =
            Tensor::randn(0f32, 1.0, (1, kvh_full, ctx, cfg.head_dim_full), &device).unwrap();
        let v_full =
            Tensor::randn(0f32, 1.0, (1, kvh_full, ctx, cfg.head_dim_full), &device).unwrap();

        let (logits, h_next) = model
            .forward(
                &target_embed,
                &backbone_hidden,
                position,
                (&k_swa, &v_swa),
                (&k_full, &v_full),
            )
            .expect("assistant forward");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(h_next.dims(), &[1, 1, nb]);
        // Logits must be finite (no NaN/Inf from the fused projections + attn).
        let max = logits
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max.is_finite(), "logits not finite: {max}");
    }

    const ASSISTANT_12B: &str =
        "/home/vasis/gguf-models/gemma-4-12B-assistant/gemma-4-12b-it-F16-MTP.gguf";

    #[test]
    fn loads_12b_assistant_per_layer_kv_and_hyphen_arch() {
        if !std::path::Path::new(ASSISTANT_12B).exists() {
            eprintln!("skip: 12B assistant file not present");
            return;
        }
        let device = Device::Cpu;
        let model = Gemma4Assistant::load(
            std::path::Path::new(ASSISTANT_12B),
            256,
            DType::F32,
            &device,
        )
        .expect("load 12B assistant (hyphen arch + embedding_length_out + array head_count_kv)");
        let cfg = model.config();
        // unsloth 12B conventions differ from AtomicChat E4B.
        assert_eq!(cfg.n_layer, 4);
        assert_eq!(cfg.n_embd, 1024);
        assert_eq!(cfg.n_embd_backbone, 3840); // from `embedding_length_out`
        assert_eq!(cfg.n_head, 16);
        // Per-layer KV heads: sliding layers GQA-8, full layer MQA-1.
        assert_eq!(cfg.n_head_kv, vec![8, 8, 8, 1]);
        assert_eq!(cfg.swa_kv_heads(), 8);
        assert_eq!(cfg.full_kv_heads(), 1);

        // Forward with per-layer-correct synthetic KV (8 heads swa, 1 head full).
        let ctx = 6usize;
        let nb = cfg.n_embd_backbone;
        let te = Tensor::randn(0f32, 1.0, (1, 1, nb), &device).unwrap();
        let bh = Tensor::randn(0f32, 1.0, (1, 1, nb), &device).unwrap();
        let ks = Tensor::randn(0f32, 1.0, (1, 8, ctx, cfg.head_dim_swa), &device).unwrap();
        let vs = Tensor::randn(0f32, 1.0, (1, 8, ctx, cfg.head_dim_swa), &device).unwrap();
        let kf = Tensor::randn(0f32, 1.0, (1, 1, ctx, cfg.head_dim_full), &device).unwrap();
        let vf = Tensor::randn(0f32, 1.0, (1, 1, ctx, cfg.head_dim_full), &device).unwrap();
        let (logits, h_next) = model
            .forward(&te, &bh, ctx - 1, (&ks, &vs), (&kf, &vf))
            .expect("12B assistant forward");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(h_next.dims(), &[1, 1, nb]);
        let mx = logits
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(mx.is_finite(), "12B logits not finite: {mx}");
    }
}
