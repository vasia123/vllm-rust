//! Shared decoder-only building blocks.
//!
//! These primitives are the **vocabulary** of the model zoo. They are
//! intentionally generic — accept dimensions as parameters, never
//! `ModelConfig` — and architectures compose them in their own files.
//!
//! The big ticket is [`attention::AttentionBlock`] (see [ADR-0010]),
//! a config-driven attention block covering ~80% of decoder-only
//! models. Around it sit:
//!
//! - [`mlp::SwiGluMlp`] / [`mlp::MlpConfig`] — gate/up/down GLU MLP
//!   with selectable activation (SiLU / GELU / GeLU PyTorch-tanh)
//!   and optional bias.
//! - [`normalization::RmsNorm`] / [`normalization::RmsNormVariant`] —
//!   Standard / ScalePlusOne (Gemma) / Unweighted (Gemma 4 `v_norm`,
//!   router norms).
//! - [`rotary::RotaryEmbedding`] (`new`, `new_partial`) — full and
//!   partial RoPE; treated as opaque by `AttentionBlock`.
//! - [`rotary::XDRotaryEmbedding`] / multi-modal rotary helpers — used
//!   by Qwen2/3-VL bespoke attentions (MRoPE).
//! - [`alibi::AlibiAttentionBias`] — used by bespoke ALiBi attentions
//!   (Bloom, MPT, JAIS, Baichuan); not part of `AttentionBlock`.
//! - [`mask::causal_mask`] — the standard upper-triangle mask. The
//!   sliding-window variant lives inside `AttentionBlock`'s manual
//!   path.
//! - [`cross_attention`] — encoder-decoder cross-attention used by
//!   T5, Whisper, BART-style models.
//! - [`attention::paged_attention`] / [`attention::batched_paged_attention_decode`]
//!   — the attention numerics primitive. `AttentionBlock` is built on
//!   top of these; bespoke attentions also call them directly.
//!
//! Adding a new layer type is fine when it's used by 3+ models;
//! one-off layer math stays in the model file (see the
//! `BESPOKE_ATTENTION_FILES` whitelist in
//! `crates/core/tests/no_local_attention.rs`).
//!
//! [ADR-0010]: ../../../../docs/adr/0010-attention-block-consolidation.md

pub mod alibi;
pub mod attention;
pub mod cross_attention;
pub mod mask;
pub mod mlp;
pub mod normalization;
pub mod rotary;

pub use alibi::{apply_alibi_bias, compute_alibi_slopes, AlibiAttentionBias};
pub use attention::{
    apply_per_head_norm, batched_paged_attention_decode, paged_attention, repeat_kv,
};
pub use mask::causal_mask;
pub use mlp::{fused_swiglu, GluActivation, MlpConfig, SwiGluMlp};
pub use normalization::{rms_norm, rms_norm_gemma, rms_norm_unweighted, RmsNorm, RmsNormVariant};
pub use rotary::{
    get_mrope_interleaved_id_list, MRoPEInterleaved, RotaryEmbedding, XDRotaryEmbedding,
};
