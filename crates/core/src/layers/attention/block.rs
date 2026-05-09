//! Universal config-driven attention block.
//!
//! This module provides [`AttentionBlock`], a TP-aware reusable building block
//! that captures the structural variations across the attention modules of 80%+
//! of supported decoder-only architectures:
//!
//! - **Vanilla MHA / GQA / MQA** (Llama, Mistral, Qwen2, Yi, Solar, …)
//! - **Per-head QK RMSNorm** (Qwen3, Olmo2, Cohere, Bailing-MoE, GLM4-MoE)
//! - **Per-head QK LayerNorm** (Persimmon)
//! - **QK-norm ordering** before RoPE (default) or after RoPE (Hunyuan)
//! - **Attention-logit soft cap** (Gemma2)
//! - **Sliding-window mask** (Mistral, Gemma3 local layers)
//! - **Bias** on any subset of Q/K/V/O via [`AttentionBias`]
//! - **Custom attention scaling** (Gemma2 hidden-size-based, Granite multiplier)
//! - **Custom projection names** (`wo`/`out_proj`/`dense`, fused `qkv_proj` /
//!   `Wqkv` / `query_key_value` / `c_attn`, custom q_norm/k_norm names)
//! - **Fused QKV** (concat-style `[Q | K | V]` layout)
//! - **Bypass RoPE** per-layer (Llama4 alternating non-RoPE layers, GPT-2/OPT
//!   with absolute position embeddings)
//! - **Partial RoPE** through `RotaryEmbedding::new_partial` (GLM4, GPT-J,
//!   GPT-NeoX, Nemotron, ChatGLM, MiniMax-M2, …) — block treats RoPE as opaque
//! - **Asymmetric Q/K/V input dim** via `with_qkv_input_size` (Eagle3 layer 0
//!   projects `2 * hidden_size` while o_proj output stays `hidden_size`)
//!
//! Architectures that need exotic attention math — DeepSeek MLA absorption,
//! Mamba/Jamba/Bamba SSM, GLA hybrids, ALiBi (MPT/Bloom/Jais/Baichuan),
//! linear attention (MiniMax, Kimi-Linear), gated attention (AfMoE),
//! output gating (Qwen3-Next), asymmetric V-dim (MiMoV2-Flash), and
//! flat-dim QK norm (OLMoE, FlexOlmo) stay bespoke in their model files.
//!
//! # Design
//!
//! The block is intentionally **structural**: it owns the projections (Q/K/V/O),
//! the rotary embedding, optional QK normalization, and an [`AttentionConfig`]
//! describing per-layer behavior. The numerics are unchanged from the bespoke
//! implementations — we delegate to the same primitives (`paged_attention`,
//! `repeat_kv`, `causal_mask`, `apply_per_head_norm`) so migrations are bit-exact.
//!
//! Two execution paths exist:
//!
//! - **Fast path** (`softcap = None`, `sliding_window = None`, no custom mask):
//!   delegates to [`paged_attention`]. This is the hot path for every Llama-class
//!   model.
//!
//! - **Manual path** (any of the above set): explicit `cache_engine.write` +
//!   `cache_engine.read` + matmul attention with custom mask and softcap. Bit
//!   identical to the existing Gemma2/Gemma3 implementations.
//!
//! # Example
//!
//! ```ignore
//! let attn_cfg = AttentionConfig::gqa(num_heads, num_kv_heads, head_dim, hidden)
//!     .with_qk_norm(QkNormVariant::PerHead)
//!     .with_softcap(50.0);
//!
//! let block = AttentionBlock::new(&attn_cfg, vb.pp("self_attn"), pg, rope)?;
//! let out = block.forward(&xs, mask, offset, cache, bt, slots, &tp_ctx)?;
//! ```
//!
//! # Why not also wrap exotic attentions?
//!
//! Past attempts in vLLM-style codebases to make a single attention "do it all"
//! end up with O(2^N) configuration flags and unreadable forward kernels. We
//! deliberately leave bespoke attentions bespoke. The plan is **80% covered by
//! `AttentionBlock`, 20% bespoke** — pragmatic, reviewable, performant.
//!
//! The decision is recorded in [ADR-0010] and enforced at CI by
//! `crates/core/tests/no_local_attention.rs`, which lists every bespoke
//! attention with a one-line reason and fails when a new one is added
//! without justification.
//!
//! [ADR-0010]: ../../../../docs/adr/0010-attention-block-consolidation.md

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

use candle_nn::{layer_norm, LayerNorm, Module};

use crate::distributed::ProcessGroup;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine};
use crate::layers::attention::ops::{apply_per_head_norm, repeat_kv};
use crate::layers::{causal_mask, paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
use crate::models::tp_layers::{TpContext, TpLinear};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Variant of QK normalization applied between projection and attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkNormVariant {
    /// Per-head RMSNorm on Q and K (head_dim-sized weight).
    /// Used by Qwen3, Olmo2, Cohere, Bailing-MoE.
    PerHead,
    /// Per-head LayerNorm on Q and K (head_dim-sized weight + bias).
    /// Used by Persimmon.
    PerHeadLayerNorm,
}

/// Per-forward shared state for batched decode attention.
///
/// All quantities here depend only on the active `[DecodeSequenceMetadata]`
/// (positions, block IDs, KV lengths, slot mapping) — not on which
/// transformer layer is currently running. So they can be built **once** at
/// the start of a forward batch and reused across every attention layer,
/// avoiding the previous per-layer pattern of building scratch `Vec<u32>`s
/// and uploading them with `Tensor::from_vec` 36 times per token on
/// Qwen3-4B.
///
/// Construct via [`build_decode_batch_shared`], hand to
/// [`AttentionBlock::forward_decode_batch_with_shared`] (or via
/// `ForwardContext::decode_shared` when a model routes through it).
#[derive(Debug, Clone)]
pub struct DecodeBatchShared {
    /// Per-sequence query positions (`seqlen_offset`), host-side. Used by
    /// `RotaryEmbedding::apply_varlen` which already expects host slice.
    pub positions: std::sync::Arc<Vec<usize>>,
    /// Flattened per-step slot mapping for all sequences (host-side).
    /// Used by `cache_engine.write_batch`.
    pub all_slot_mapping: std::sync::Arc<Vec<usize>>,
    /// `[batch_size, max_blocks_per_seq]` U32 block IDs on the same
    /// device as the model.
    pub block_tables: Tensor,
    /// `[batch_size]` U32 sequence lengths (current decode position + 1).
    pub seq_lens: Tensor,
    /// Padding extent of `block_tables` along axis 1.
    pub max_blocks_per_seq: usize,
    /// `max(seq_lens)` — used as the kernel's upper bound.
    pub max_seq_len: usize,
}

/// Build the per-forward shared decode state on `device`.
///
/// One pass through `sequences`: collects positions, slot mapping,
/// block IDs (padded), seq_lens. Two `Tensor::from_vec` host→device
/// uploads (block_tables, seq_lens) — same as one attention layer used
/// to do, but now amortised across all 36 layers of a Qwen3 forward.
pub fn build_decode_batch_shared(
    sequences: &[DecodeSequenceMetadata],
    device: &Device,
) -> Result<DecodeBatchShared> {
    let batch_size = sequences.len();
    if batch_size == 0 {
        candle_core::bail!("build_decode_batch_shared: empty sequences");
    }

    let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();

    let all_slot_mapping: Vec<usize> = sequences
        .iter()
        .flat_map(|s| s.slot_mapping.iter().copied())
        .collect();

    let max_blocks_per_seq = sequences
        .iter()
        .map(|s| s.block_ids.len())
        .max()
        .unwrap_or(1);
    let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
    for (i, seq) in sequences.iter().enumerate() {
        for (j, &block_id) in seq.block_ids.iter().enumerate() {
            bt_data[i * max_blocks_per_seq + j] = block_id as u32;
        }
    }
    let block_tables = Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), device)?;

    let seq_lens_data: Vec<u32> = sequences
        .iter()
        .map(|s| (s.seqlen_offset + 1) as u32)
        .collect();
    let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
    let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), device)?;

    Ok(DecodeBatchShared {
        positions: std::sync::Arc::new(positions),
        all_slot_mapping: std::sync::Arc::new(all_slot_mapping),
        block_tables,
        seq_lens,
        max_blocks_per_seq,
        max_seq_len,
    })
}

/// Bias presence on the four projections.
///
/// Most models have bias=false everywhere (Llama family). Some (Qwen2)
/// have bias on Q/K/V but not O. The struct captures all four independently.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionBias {
    pub q: bool,
    pub k: bool,
    pub v: bool,
    pub o: bool,
}

impl AttentionBias {
    pub const NONE: Self = Self {
        q: false,
        k: false,
        v: false,
        o: false,
    };
    pub const QKV_ONLY: Self = Self {
        q: true,
        k: true,
        v: true,
        o: false,
    };
    pub const ALL: Self = Self {
        q: true,
        k: true,
        v: true,
        o: true,
    };
}

// Compile-time invariants on the named bias presets. If a future edit
// changes the meaning of NONE/QKV_ONLY/ALL inadvertently, the build fails.
const _: () = {
    assert!(!AttentionBias::NONE.q);
    assert!(AttentionBias::QKV_ONLY.q && AttentionBias::QKV_ONLY.k && AttentionBias::QKV_ONLY.v);
    assert!(!AttentionBias::QKV_ONLY.o);
    assert!(AttentionBias::ALL.o);
};

/// VarBuilder names for the QKV/O projections.
///
/// Different model families use different conventions:
/// - Llama family:  separate `q_proj`, `k_proj`, `v_proj`, `o_proj` (default).
/// - InternLM2:     `q_proj`, `k_proj`, `v_proj`, `wo`.
/// - Exaone:        `q_proj`, `k_proj`, `v_proj`, `out_proj`.
/// - Phi3:          fused `qkv_proj` + `o_proj`.
/// - ChatGLM:       fused `query_key_value` + `dense`.
/// - GPT-2 / BigCode: fused `c_attn` + `c_proj`.
///
/// `qkv` is only used when `AttentionConfig::qkv_fused` is true. The
/// `q_norm`/`k_norm` names are only consulted when `qk_norm` is set.
#[derive(Debug, Clone, Copy)]
pub struct ProjNames {
    pub q: &'static str,
    pub k: &'static str,
    pub v: &'static str,
    pub o: &'static str,
    pub qkv: &'static str,
    pub q_norm: &'static str,
    pub k_norm: &'static str,
}

impl Default for ProjNames {
    fn default() -> Self {
        Self {
            q: "q_proj",
            k: "k_proj",
            v: "v_proj",
            o: "o_proj",
            qkv: "qkv_proj",
            q_norm: "q_norm",
            k_norm: "k_norm",
        }
    }
}

/// Configuration for a single attention block.
///
/// Constructed via builder-style helpers. Defaults match the most common case
/// (vanilla GQA, no QK-norm, no softcap, no sliding window, no bias, scale =
/// 1/sqrt(head_dim)).
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,

    /// Optional bias on each projection. Default: no bias.
    pub bias: AttentionBias,

    /// QK normalization (Qwen3, Olmo2, Cohere). Default: None.
    pub qk_norm: Option<QkNormVariant>,
    /// Epsilon for the QK RMSNorm (only used when `qk_norm` is set).
    pub qk_norm_eps: f64,

    /// Soft-capping value for attention logits (Gemma2). When set, applies
    /// `cap * tanh(logits / cap)` before mask + softmax.
    pub softcap: Option<f64>,

    /// Sliding-window size (Mistral, Gemma3 local layers). When set, mask
    /// allows attention only to positions within `window` tokens.
    pub sliding_window: Option<usize>,

    /// Optional override for attention scaling.
    /// Default (None) means `1/sqrt(head_dim)`. Used by Gemma2 which has
    /// a hidden-size-based scaling.
    pub scale: Option<f64>,

    /// VarBuilder names for the projections. Default: q_proj/k_proj/v_proj/o_proj.
    pub proj_names: ProjNames,

    /// If true, skip RoPE entirely. Used by Llama4's "no-RoPE" layers (every
    /// fourth layer or so), and similar architectures that interleave RoPE and
    /// non-RoPE attention layers. The `RotaryEmbedding` passed at construction
    /// time is still owned by the block but never invoked.
    pub bypass_rope: bool,

    /// If true, use a single fused `qkv_proj` weight that produces
    /// `[Q | K | V]` concatenated along the last axis with shapes
    /// `Q: num_heads*head_dim`, `K: num_kv_heads*head_dim`,
    /// `V: num_kv_heads*head_dim`. Used by Phi3, GPT-2/BigCode, ChatGLM,
    /// DBRX, Falcon-style architectures.
    pub qkv_fused: bool,

    /// QK-norm ordering relative to RoPE.
    ///
    /// `false` (default, used by Qwen3/Olmo2/Cohere): apply QK-norm BEFORE RoPE.
    /// `true` (used by Hunyuan): apply QK-norm AFTER RoPE on the rotated Q/K.
    ///
    /// Only consulted when `qk_norm` is set.
    pub qk_norm_after_rope: bool,

    /// Optional override for the input dimension of the Q/K/V projections.
    ///
    /// Defaults to `hidden_size` (the residual stream dimension). Used by
    /// architectures whose Q/K/V projections accept a wider input than the
    /// hidden state — for example Eagle 3's first decoder layer, where
    /// `qkv_input_size = 2 * hidden_size` because the layer-0 input is the
    /// concatenation of token embeddings and target hidden states. The
    /// `o_proj` output dimension is always `hidden_size`.
    pub qkv_input_size: Option<usize>,
}

impl AttentionConfig {
    /// Standard GQA configuration: no QK-norm, no softcap, no sliding window,
    /// no bias, default scale.
    pub fn gqa(num_heads: usize, num_kv_heads: usize, head_dim: usize, hidden_size: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            bias: AttentionBias::NONE,
            qk_norm: None,
            qk_norm_eps: 1e-6,
            softcap: None,
            sliding_window: None,
            scale: None,
            proj_names: ProjNames::default(),
            bypass_rope: false,
            qkv_fused: false,
            qk_norm_after_rope: false,
            qkv_input_size: None,
        }
    }

    /// Override the Q/K/V projection input dimension (default = `hidden_size`).
    /// `o_proj` output dimension is always `hidden_size`.
    pub fn with_qkv_input_size(mut self, size: usize) -> Self {
        self.qkv_input_size = Some(size);
        self
    }

    pub fn with_bypass_rope(mut self) -> Self {
        self.bypass_rope = true;
        self
    }

    pub fn with_qkv_fused(mut self) -> Self {
        self.qkv_fused = true;
        self
    }

    pub fn with_qk_norm_after_rope(mut self) -> Self {
        self.qk_norm_after_rope = true;
        self
    }

    pub fn with_proj_names(mut self, names: ProjNames) -> Self {
        self.proj_names = names;
        self
    }

    pub fn with_bias(mut self, bias: AttentionBias) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_qk_norm(mut self, variant: QkNormVariant, eps: f64) -> Self {
        self.qk_norm = Some(variant);
        self.qk_norm_eps = eps;
        self
    }

    pub fn with_softcap(mut self, cap: f64) -> Self {
        self.softcap = Some(cap);
        self
    }

    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Effective attention scale.
    fn effective_scale(&self) -> f64 {
        self.scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }

    /// Whether the manual (non-paged_attention) path is required.
    fn needs_manual_path(&self) -> bool {
        self.softcap.is_some() || self.sliding_window.is_some() || self.scale.is_some()
    }
}

// ─── Block ───────────────────────────────────────────────────────────────────

/// TP-aware universal attention block.
///
/// Owns the QKVO projections, optional QK-norms, and a pre-built rotary
/// embedding. The configuration is captured at construction time; `forward`
/// is purely computational.
///
/// **Sharding** (TP world_size > 1): Q/K/V are column-parallel (split output
/// heads), O is row-parallel (reduce partial outputs). The internal
/// `num_heads` / `num_kv_heads` fields hold the **per-GPU** counts after
/// sharding — matching the existing bespoke convention.
/// QKV projection layout: separate (Llama-style) or fused (Phi3-style).
enum QkvProjection {
    Separate {
        q: TpLinear,
        k: TpLinear,
        v: TpLinear,
    },
    /// Concatenated layout `[Q | K | V]` along the last axis.
    Fused(TpLinear),
}

/// Per-head normalization layers between QKV projection and attention.
enum QkNormPair {
    PerHeadRms { q: RmsNorm, k: RmsNorm },
    PerHeadLayerNorm { q: LayerNorm, k: LayerNorm },
}

pub struct AttentionBlock {
    qkv: QkvProjection,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    qk_norm: Option<QkNormPair>,

    // Per-GPU counts (post-sharding).
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,

    softcap: Option<f64>,
    sliding_window: Option<usize>,
    /// Effective scale used in the manual path (the fast paged_attention path
    /// uses its own internal default scale; we never override it there).
    manual_scale: f64,
    needs_manual: bool,
    bypass_rope: bool,
    qk_norm_after_rope: bool,
}

impl AttentionBlock {
    /// Build a new attention block.
    ///
    /// `vb` should point at the attention module (e.g. `vb.pp("self_attn")`).
    /// Children expected: `q_proj`, `k_proj`, `v_proj`, `o_proj`, and (if
    /// `qk_norm` is set) `q_norm` / `k_norm`.
    pub fn new(
        cfg: &AttentionConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        rotary_emb: RotaryEmbedding,
    ) -> Result<Self> {
        let world_size = pg.world_size();

        // TP divisibility: enforced exactly as the bespoke implementations do.
        if world_size > 1 {
            if !cfg.num_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({}) must be divisible by world_size ({world_size})",
                    cfg.num_heads
                )));
            }
            if !cfg.num_kv_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({}) must be divisible by world_size ({world_size})",
                    cfg.num_kv_heads
                )));
            }
        }

        let names = cfg.proj_names;
        let qkv = if cfg.qkv_fused {
            // The fused projection cannot have heterogeneous bias on Q/K/V, so
            // we require all three flags to agree.
            if cfg.bias.q != cfg.bias.k || cfg.bias.k != cfg.bias.v {
                return Err(candle_core::Error::Msg(
                    "fused QKV requires bias.q == bias.k == bias.v".to_string(),
                ));
            }
            let qkv_dim = (cfg.num_heads + 2 * cfg.num_kv_heads) * cfg.head_dim;
            QkvProjection::Fused(TpLinear::column_parallel(
                cfg.qkv_input_size.unwrap_or(cfg.hidden_size),
                qkv_dim,
                cfg.bias.q,
                false,
                vb.pp(names.qkv),
                pg,
            )?)
        } else {
            let qkv_in = cfg.qkv_input_size.unwrap_or(cfg.hidden_size);
            QkvProjection::Separate {
                q: TpLinear::column_parallel(
                    qkv_in,
                    cfg.num_heads * cfg.head_dim,
                    cfg.bias.q,
                    false,
                    vb.pp(names.q),
                    pg,
                )?,
                k: TpLinear::column_parallel(
                    qkv_in,
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.bias.k,
                    false,
                    vb.pp(names.k),
                    pg,
                )?,
                v: TpLinear::column_parallel(
                    qkv_in,
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.bias.v,
                    false,
                    vb.pp(names.v),
                    pg,
                )?,
            }
        };
        let o_proj = TpLinear::row_parallel(
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.bias.o,
            true,
            vb.pp(names.o),
            pg,
        )?;

        let qk_norm = match cfg.qk_norm {
            None => None,
            Some(QkNormVariant::PerHead) => Some(QkNormPair::PerHeadRms {
                q: rms_norm(cfg.head_dim, cfg.qk_norm_eps, vb.pp(names.q_norm))?,
                k: rms_norm(cfg.head_dim, cfg.qk_norm_eps, vb.pp(names.k_norm))?,
            }),
            Some(QkNormVariant::PerHeadLayerNorm) => Some(QkNormPair::PerHeadLayerNorm {
                q: layer_norm(cfg.head_dim, cfg.qk_norm_eps, vb.pp(names.q_norm))?,
                k: layer_norm(cfg.head_dim, cfg.qk_norm_eps, vb.pp(names.k_norm))?,
            }),
        };

        Ok(Self {
            qkv,
            o_proj,
            rotary_emb,
            qk_norm,
            num_heads: cfg.num_heads / world_size,
            num_kv_heads: cfg.num_kv_heads / world_size,
            head_dim: cfg.head_dim,
            softcap: cfg.softcap,
            sliding_window: cfg.sliding_window,
            manual_scale: cfg.effective_scale(),
            needs_manual: cfg.needs_manual_path(),
            bypass_rope: cfg.bypass_rope,
            qk_norm_after_rope: cfg.qk_norm_after_rope,
        })
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    // ── Prefill path ─────────────────────────────────────────────────────────

    /// Prefill attention forward.
    ///
    /// Accepts `xs` of shape `[batch, q_len, hidden]` and returns
    /// `[batch, q_len, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let (q, k, v) = self.project_qkv(xs, b_sz, q_len, tp_ctx)?;
        // Default order: QK-norm → RoPE. Some architectures (Hunyuan) flip
        // this to RoPE → QK-norm; controlled by `qk_norm_after_rope`.
        let (q, k) = if self.qk_norm_after_rope {
            let (q, k) = if self.bypass_rope {
                (q, k)
            } else {
                self.rotary_emb.apply(&q, &k, seqlen_offset)?
            };
            self.apply_qk_norm(&q, &k)?
        } else {
            let (q, k) = self.apply_qk_norm(&q, &k)?;
            if self.bypass_rope {
                (q, k)
            } else {
                self.rotary_emb.apply(&q, &k, seqlen_offset)?
            }
        };

        let attn_output = if self.needs_manual {
            self.manual_prefill(
                &q,
                &k,
                &v,
                seqlen_offset,
                cache_engine,
                block_table,
                slot_mapping,
            )?
        } else {
            paged_attention(
                &q,
                &k,
                &v,
                attention_mask,
                seqlen_offset,
                cache_engine,
                block_table.block_ids(),
                slot_mapping,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )?
        };

        self.o_proj.forward(&attn_output, tp_ctx)
    }

    // ── Decode path ──────────────────────────────────────────────────────────

    /// Batched-decode attention forward.
    ///
    /// Accepts `xs` of shape `[batch, 1, hidden]` (one token per sequence)
    /// and returns `[batch, 1, hidden]`.
    pub fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch_with_shared(xs, sequences, cache_engine, tp_ctx, None)
    }

    /// Like [`forward_decode_batch`], but accepts a [`DecodeBatchShared`] that
    /// pre-builds the per-sequence host vectors and GPU tensors that are
    /// constant across every attention layer of one forward pass.
    ///
    /// On Qwen3-4B this is the difference between 36 host→device uploads
    /// per token (block_tables, seq_lens, plus the implicit syncs from
    /// each `Tensor::from_vec` call inside `cuda_decode_batch`) and a
    /// single upload reused by every layer.
    pub fn forward_decode_batch_with_shared(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
        shared: Option<&DecodeBatchShared>,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let (q, k, v) = self.project_qkv(xs, batch_size, 1, tp_ctx)?;
        // Apply QK-norm before RoPE in the default ordering. For Hunyuan-style
        // (qk_norm_after_rope), this is deferred to inside the per-seq loop.
        let (q, k) = if self.qk_norm_after_rope {
            (q, k)
        } else {
            self.apply_qk_norm(&q, &k)?
        };

        // Fast batched CUDA decode path: only when on a CUDA device, no softcap
        // / sliding-window / custom scale and no post-RoPE QK norm (the kernel
        // applies RoPE internally and we cannot inject a norm step in between).
        // The device check matters when `cuda-kernels` is compiled in but the
        // model is being exercised on CPU (e.g. `*_decode_batch` tests).
        #[cfg(feature = "cuda-kernels")]
        if q.device().is_cuda() && !self.needs_manual && !self.qk_norm_after_rope {
            return self.cuda_decode_batch(&q, &k, &v, sequences, cache_engine, tp_ctx, shared);
        }
        let _ = shared;

        // Per-sequence loop fallback (CPU and manual paths). RoPE position
        // depends on sequence offset, and the slot_mapping / block_ids are
        // per-sequence — neither is amenable to a single batched call.
        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

            let (q_i, k_i) = if self.bypass_rope {
                (q_i, k_i)
            } else {
                self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?
            };
            // Hunyuan-style: QK-norm AFTER RoPE.
            let (q_i, k_i) = if self.qk_norm_after_rope {
                self.apply_qk_norm(&q_i, &k_i)?
            } else {
                (q_i, k_i)
            };

            let attn_out = if self.needs_manual {
                self.manual_decode_one(
                    &q_i,
                    &k_i,
                    &v_i,
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                )?
            } else {
                paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    None,
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?
            };
            outputs.push(attn_out);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.o_proj.forward(&attn_output, tp_ctx)
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Project `xs` to Q, K, V with shape `[batch, num_*_heads, seq, head_dim]`.
    fn project_qkv(
        &self,
        xs: &Tensor,
        batch: usize,
        seq: usize,
        tp_ctx: &TpContext,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (q_flat, k_flat, v_flat) = match &self.qkv {
            QkvProjection::Separate { q, k, v } => (
                q.forward(xs, tp_ctx)?,
                k.forward(xs, tp_ctx)?,
                v.forward(xs, tp_ctx)?,
            ),
            QkvProjection::Fused(qkv) => {
                let qkv_out = qkv.forward(xs, tp_ctx)?;
                let q_dim = self.num_heads * self.head_dim;
                let kv_dim = self.num_kv_heads * self.head_dim;
                let q = qkv_out.narrow(D::Minus1, 0, q_dim)?;
                let k = qkv_out.narrow(D::Minus1, q_dim, kv_dim)?;
                let v = qkv_out.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?;
                (q, k, v)
            }
        };

        let q = q_flat
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k_flat
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v_flat
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn apply_qk_norm(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        match &self.qk_norm {
            None => Ok((q.clone(), k.clone())),
            Some(QkNormPair::PerHeadRms { q: qn, k: kn }) => {
                Ok((apply_per_head_norm(q, qn)?, apply_per_head_norm(k, kn)?))
            }
            Some(QkNormPair::PerHeadLayerNorm { q: qn, k: kn }) => Ok((
                apply_per_head_layer_norm(q, qn)?,
                apply_per_head_layer_norm(k, kn)?,
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn manual_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, _, q_len, _) = q.dims4()?;
        let device = q.device();
        let dtype = q.dtype();

        // Write new K/V into the paged cache.
        let k_for_cache = k.squeeze(0)?.contiguous()?;
        let v_for_cache = v.squeeze(0)?.contiguous()?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        let num_tokens = seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(block_table.block_ids(), num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        let kv_len = k_full.dim(2)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        let q_scaled = (q * self.manual_scale)?;
        let mut logits = q_scaled.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;
        if let Some(cap) = self.softcap {
            logits = soft_cap(&logits, cap)?;
        }

        let mask = self.build_mask(q_len, kv_len, seqlen_offset, dtype, device)?;
        logits = logits.broadcast_add(&mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let attn = probs.matmul(&v_full)?;

        attn.transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))
    }

    #[allow(clippy::too_many_arguments)]
    fn manual_decode_one(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_ids: &[crate::kv_cache::BlockId],
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_, _, q_len, _) = q.dims4()?;
        let device = q.device();
        let dtype = q.dtype();

        let k_for_cache = k.squeeze(0)?.contiguous()?;
        let v_for_cache = v.squeeze(0)?.contiguous()?;
        cache_engine
            .write(&k_for_cache, &v_for_cache, slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        let num_tokens = seqlen_offset + q_len;
        let (k_full, v_full) = cache_engine
            .read(block_ids, num_tokens)
            .map_err(|e| candle_core::Error::Msg(format!("cache read: {e}")))?;

        let kv_len = k_full.dim(2)?;
        let num_kv_groups = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(k_full, num_kv_groups)?;
        let v_full = repeat_kv(v_full, num_kv_groups)?;

        let q_scaled = (q * self.manual_scale)?;
        let mut logits = q_scaled.matmul(&k_full.transpose(D::Minus2, D::Minus1)?)?;
        if let Some(cap) = self.softcap {
            logits = soft_cap(&logits, cap)?;
        }

        // For decode (q_len = 1) we still need a sliding-window mask if configured;
        // otherwise no mask is needed (single token is always causal).
        if let Some(window) = self.sliding_window {
            let mask = sliding_window_mask(q_len, kv_len, seqlen_offset, window, dtype, device)?;
            logits = logits.broadcast_add(&mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
        let attn = probs.matmul(&v_full)?;

        attn.transpose(1, 2)?
            .reshape((1, q_len, self.num_heads * self.head_dim))
    }

    /// Fast batched CUDA decode path (extracted from per-model bespoke code).
    ///
    /// Only invoked when `softcap`, `sliding_window`, and custom `scale` are all
    /// unset — the kernel computes attention with the standard `1/sqrt(head_dim)`
    /// scale and a plain causal mask.
    #[cfg(feature = "cuda-kernels")]
    #[allow(clippy::too_many_arguments)]
    fn cuda_decode_batch(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
        shared: Option<&DecodeBatchShared>,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        // [batch, heads, 1, head_dim] → [batch, heads, head_dim]
        let q = q.squeeze(2)?;
        let k = k.squeeze(2)?;
        let v = v.squeeze(2)?;

        // RoPE positions — identical across attention layers within one
        // forward, so reuse from `shared` when available; otherwise build
        // a per-layer scratch vector (the old slow path).
        let positions_owned: Vec<usize>;
        let positions: &[usize] = match shared {
            Some(s) => &s.positions,
            None => {
                positions_owned = sequences.iter().map(|s| s.seqlen_offset).collect();
                &positions_owned
            }
        };
        let (q, k) = if self.bypass_rope {
            (q, k)
        } else {
            self.rotary_emb.apply_varlen(&q, &k, positions)?
        };

        // KV write — slot_mapping is also constant across layers; reuse.
        let slot_mapping_owned: Vec<usize>;
        let all_slot_mapping: &[usize] = match shared {
            Some(s) => &s.all_slot_mapping,
            None => {
                slot_mapping_owned = sequences
                    .iter()
                    .flat_map(|s| s.slot_mapping.iter().copied())
                    .collect();
                &slot_mapping_owned
            }
        };
        cache_engine
            .write_batch(&k, &v, all_slot_mapping)
            .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

        // block_tables / seq_lens — same shape and contents for every
        // attention layer of one forward, so the GPU upload happens once
        // per token (in `engine/helpers.rs`) instead of 36× via
        // `Tensor::from_vec` here. We clone the cached `Tensor`s on the
        // shared path; that's just an `Arc` refcount bump — no GPU work.
        let (block_tables, seq_lens, max_blocks_per_seq, max_seq_len): (
            Tensor,
            Tensor,
            usize,
            usize,
        ) = match shared {
            Some(s) => (
                s.block_tables.clone(),
                s.seq_lens.clone(),
                s.max_blocks_per_seq,
                s.max_seq_len,
            ),
            None => {
                let max_blocks_per_seq = sequences
                    .iter()
                    .map(|s| s.block_ids.len())
                    .max()
                    .unwrap_or(1);
                let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
                for (i, seq) in sequences.iter().enumerate() {
                    for (j, &block_id) in seq.block_ids.iter().enumerate() {
                        bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                    }
                }
                let bt = Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

                let seq_lens_data: Vec<u32> = sequences
                    .iter()
                    .map(|s| (s.seqlen_offset + 1) as u32)
                    .collect();
                let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
                let sl = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

                (bt, sl, max_blocks_per_seq, max_seq_len)
            }
        };

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // 2026-05-09: route through `paged_attention_auto` instead of
        // V1 directly — `paged_attention_bench` shows V2 wins at every
        // measured seq_len (V1 is the legacy path; V2 split-K paginates
        // the reduction). The auto-dispatcher picks the partition size
        // based on max_seq_len so short / long contexts get the right
        // tradeoff. See V2_SEQ_LEN_THRESHOLD = 0 in cuda_kernels.rs.
        let attn_output = crate::cuda_kernels::paged_attention_auto(
            &q,
            cache_engine.k_cache(),
            cache_engine.v_cache(),
            &block_tables,
            &seq_lens,
            scale,
            self.num_heads,
            self.num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            self.head_dim,
            cache_engine.block_size(),
        )?;

        // [batch, hidden] → [batch, 1, hidden] to match residual shape.
        self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
    }

    fn build_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        seqlen_offset: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        match self.sliding_window {
            Some(window) => {
                sliding_window_mask(q_len, kv_len, seqlen_offset, window, dtype, device)
            }
            None => causal_mask(q_len, seqlen_offset, dtype, device),
        }
    }
}

/// Apply LayerNorm per attention head (mirror of `apply_per_head_norm` but for
/// LayerNorm). Reshapes `[b, h, s, d]` → `[b*h*s, d]`, applies norm, reshapes back.
fn apply_per_head_layer_norm(x: &Tensor, norm: &LayerNorm) -> Result<Tensor> {
    let (b, h, s, d) = x.dims4()?;
    let x = x.reshape((b * h * s, d))?;
    let x = norm.forward(&x)?;
    x.reshape((b, h, s, d))
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Soft capping: `cap * tanh(x / cap)`.
///
/// Identical to the per-model `soft_cap` helpers in Gemma2/Gemma3/Gemma4 — kept
/// here to avoid circular dependencies (these models import from `layers`).
pub fn soft_cap(xs: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(xs.clone());
    }
    let scaled = (xs / cap)?;
    scaled.tanh()? * cap
}

/// Sliding-window causal mask of shape `[1, 1, q_len, kv_len]`.
///
/// Allows attention only to positions in the window
/// `[query_pos - window + 1, query_pos]`.
pub fn sliding_window_mask(
    q_len: usize,
    kv_len: usize,
    seqlen_offset: usize,
    window_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![f32::NEG_INFINITY; q_len * kv_len];
    for i in 0..q_len {
        let query_pos = seqlen_offset + i;
        for j in 0..kv_len {
            let is_causal = j <= query_pos;
            let is_in_window = query_pos < window_size || j > query_pos - window_size;
            if is_causal && is_in_window {
                mask[i * kv_len + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ─── DecodeBatchShared builder ───────────────────────────────────────

    fn fake_metadata(
        num_seqs: usize,
        offsets: &[usize],
        block_lists: &[Vec<usize>],
        slot_lists: &[Vec<usize>],
    ) -> Vec<crate::engine::DecodeSequenceMetadata> {
        assert_eq!(num_seqs, offsets.len());
        assert_eq!(num_seqs, block_lists.len());
        assert_eq!(num_seqs, slot_lists.len());

        let mut sequences = Vec::with_capacity(num_seqs);
        for i in 0..num_seqs {
            sequences.push(crate::engine::DecodeSequenceMetadata {
                request_id: i as u64,
                seqlen_offset: offsets[i],
                block_ids: block_lists[i].clone(),
                slot_mapping: slot_lists[i].clone(),
            });
        }
        sequences
    }

    #[test]
    fn build_decode_batch_shared_single_seq() {
        let device = Device::Cpu;
        let seqs = fake_metadata(1, &[5], &[vec![0]], &[vec![5]]);
        let shared = build_decode_batch_shared(&seqs, &device).unwrap();

        assert_eq!(shared.positions.as_slice(), &[5]);
        assert_eq!(shared.all_slot_mapping.as_slice(), &[5]);
        assert_eq!(shared.max_blocks_per_seq, 1);
        assert_eq!(shared.max_seq_len, 6); // seqlen_offset + 1
        assert_eq!(shared.block_tables.dims(), &[1, 1]);
        assert_eq!(shared.seq_lens.dims(), &[1]);

        let bt: Vec<u32> = shared
            .block_tables
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let sl: Vec<u32> = shared.seq_lens.to_vec1().unwrap();
        assert_eq!(bt, vec![0]);
        assert_eq!(sl, vec![6]);
    }

    #[test]
    fn build_decode_batch_shared_multiple_sequences_pad_block_table() {
        // Two sequences with different block-list lengths — block_tables
        // gets padded along axis 1 to max_blocks_per_seq.
        let device = Device::Cpu;
        let seqs = fake_metadata(
            2,
            &[15, 7],
            &[vec![3, 9, 12], vec![5]],
            &[vec![15], vec![7]],
        );
        let shared = build_decode_batch_shared(&seqs, &device).unwrap();

        assert_eq!(shared.positions.as_slice(), &[15, 7]);
        assert_eq!(shared.all_slot_mapping.as_slice(), &[15, 7]);
        assert_eq!(shared.max_blocks_per_seq, 3);
        assert_eq!(shared.max_seq_len, 16);

        let bt: Vec<u32> = shared
            .block_tables
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Row 0: [3, 9, 12]; row 1: [5, 0, 0] (zero-padded).
        assert_eq!(bt, vec![3, 9, 12, 5, 0, 0]);

        let sl: Vec<u32> = shared.seq_lens.to_vec1().unwrap();
        assert_eq!(sl, vec![16, 8]); // seqlen_offset + 1
    }

    #[test]
    fn build_decode_batch_shared_empty_errors() {
        let device = Device::Cpu;
        let res = build_decode_batch_shared(&[], &device);
        assert!(res.is_err(), "empty sequences must error");
    }

    #[test]
    fn decode_batch_shared_clone_is_cheap() {
        // Clone should be Arc/Tensor refcount bumps, not deep copies.
        // Exercising the path here also lets the borrow checker confirm
        // DecodeBatchShared: Clone is well-formed.
        let device = Device::Cpu;
        let seqs = fake_metadata(1, &[2], &[vec![1]], &[vec![2]]);
        let shared = build_decode_batch_shared(&seqs, &device).unwrap();
        let clone1 = shared.clone();
        let clone2 = shared.clone();
        assert!(std::sync::Arc::ptr_eq(&shared.positions, &clone1.positions));
        assert!(std::sync::Arc::ptr_eq(
            &shared.all_slot_mapping,
            &clone2.all_slot_mapping
        ));
    }

    #[test]
    fn config_defaults() {
        let cfg = AttentionConfig::gqa(8, 4, 64, 512);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.hidden_size, 512);
        assert!(cfg.qk_norm.is_none());
        assert!(cfg.softcap.is_none());
        assert!(cfg.sliding_window.is_none());
        assert!(cfg.scale.is_none());
        assert!(!cfg.needs_manual_path());
        assert!((cfg.effective_scale() - 1.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn config_softcap_forces_manual_path() {
        let cfg = AttentionConfig::gqa(8, 4, 64, 512).with_softcap(50.0);
        assert!(cfg.needs_manual_path());
    }

    #[test]
    fn config_sliding_window_forces_manual_path() {
        let cfg = AttentionConfig::gqa(8, 4, 64, 512).with_sliding_window(128);
        assert!(cfg.needs_manual_path());
    }

    #[test]
    fn config_custom_scale_forces_manual_path() {
        // Custom scale routes through the manual path because the fast
        // paged_attention kernel uses its own internal scale.
        let cfg = AttentionConfig::gqa(8, 4, 64, 512).with_scale(0.5);
        assert!(cfg.needs_manual_path());
    }

    #[test]
    fn config_qk_norm_does_not_force_manual_path() {
        // QK-norm is applied before paged_attention — the fast path is fine.
        let cfg = AttentionConfig::gqa(8, 4, 64, 512).with_qk_norm(QkNormVariant::PerHead, 1e-6);
        assert!(!cfg.needs_manual_path());
    }

    #[test]
    fn soft_cap_zero_is_identity() {
        let device = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (4,), &device).unwrap();
        let y = soft_cap(&x, 0.0).unwrap();
        let xv: Vec<f32> = x.to_vec1().unwrap();
        let yv: Vec<f32> = y.to_vec1().unwrap();
        for (a, b) in xv.iter().zip(yv.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn soft_cap_bounds_logits() {
        // tanh saturates at ±1 → output is bounded by ±cap.
        let device = Device::Cpu;
        let big = Tensor::from_vec(vec![1000.0f32, -1000.0, 0.0], (3,), &device).unwrap();
        let cap = 50.0;
        let y = soft_cap(&big, cap).unwrap();
        let yv: Vec<f32> = y.to_vec1().unwrap();
        assert!((yv[0] - (cap as f32)).abs() < 1e-3);
        assert!((yv[1] + (cap as f32)).abs() < 1e-3);
        assert!(yv[2].abs() < 1e-6);
    }

    #[test]
    fn sliding_window_mask_shape() {
        let device = Device::Cpu;
        let mask = sliding_window_mask(8, 16, 0, 4, DType::F32, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 8, 16]);
    }

    #[test]
    fn sliding_window_mask_clips_far_past() {
        // Window=2 means at query_pos=5 we only see positions {4, 5}.
        let device = Device::Cpu;
        let mask = sliding_window_mask(1, 6, 5, 2, DType::F32, &device).unwrap();
        let v: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Position 4 and 5 attendable (0.0); 0,1,2,3 masked (-inf).
        assert!(v[0].is_infinite() && v[0] < 0.0);
        assert!(v[3].is_infinite() && v[3] < 0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 0.0);
    }

    #[test]
    fn sliding_window_mask_at_start_is_causal_only() {
        // When query_pos < window_size, every causal position is in-window.
        let device = Device::Cpu;
        let mask = sliding_window_mask(4, 4, 0, 16, DType::F32, &device).unwrap();
        let v: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Same as plain causal mask for 4x4.
        for i in 0..4 {
            for j in 0..4 {
                let val = v[i * 4 + j];
                if j > i {
                    assert!(val.is_infinite() && val < 0.0, "({i},{j}) should be -inf");
                } else {
                    assert_eq!(val, 0.0, "({i},{j}) should be 0");
                }
            }
        }
    }

    // Invariants on the AttentionBias presets are checked at compile time
    // via the `const _: () = { ... };` block near their definition.
}
