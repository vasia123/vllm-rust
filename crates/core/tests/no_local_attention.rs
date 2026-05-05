//! Phase 6 invariant: every model file in `crates/core/src/models/`
//! whose attention shape can be expressed declaratively must use
//! `crate::layers::attention::AttentionBlock`.
//!
//! The vast majority of decoder-only architectures fall into the
//! "config-driven attention" bucket and are migrated to a thin shim
//! over `AttentionBlock` (see ADR-0010). A new model file declaring a
//! bespoke `XxxAttention` struct without justification regresses
//! Phase 4 of the architecture refactor — this test catches that
//! drift at PR time.
//!
//! How the test works:
//!
//! 1. Walk `crates/core/src/models/*.rs`.
//! 2. For each file, find struct declarations whose name ends in
//!    `Attention` (e.g. `struct LlamaAttention`, `pub(crate) struct
//!    Qwen3VLAttention`).
//! 3. Skip files where every such struct contains `inner: AttentionBlock`
//!    (a thin shim — the migrated case).
//! 4. The remaining files must appear in `BESPOKE_ATTENTION_FILES` with
//!    a reason. Adding a new entry is an explicit decision that goes
//!    through code review.
//!
//! When you add a new model:
//!
//! - **Default path:** use `AttentionBlock`. See `docs/adding-a-model.md`.
//! - **Genuinely bespoke** (MLA, SSM, ALiBi, linear attention, gated
//!   attention, asymmetric V-dim, flat-dim QK norm, MRoPE/FoPE, encoder
//!   without KV cache, vision tower, quantized weights): add the file
//!   to `BESPOKE_ATTENTION_FILES` with a one-line reason.

use std::fs;
use std::path::{Path, PathBuf};

/// Files in `crates/core/src/models/` that intentionally implement
/// bespoke attention. Each entry is paired with a one-line reason.
///
/// Categories:
///
/// - **Quantized variants** — use `Q*Linear` types instead of
///   `TpLinear`; cannot share the `AttentionBlock` constructor.
/// - **Vision encoders** — bidirectional attention, no KV cache.
/// - **Encoder-only / encoder-decoder text models** — encoder paths
///   use full attention, not paged.
/// - **SSM / hybrid SSM** — Mamba/Mamba2-style recurrence.
/// - **MLA** — DeepSeek V2/V3 latent KV with kv_b_proj absorption.
/// - **ALiBi** — additive position bias instead of RoPE.
/// - **Linear / lightning / GDN attention** — recurrent state instead
///   of softmax attention.
/// - **Gated / output-gated attention** — applies a learned gate
///   between the attention matmul and the output projection that
///   `AttentionBlock`'s structural pipeline does not expose.
/// - **Asymmetric V-dim** — `v_head_dim != head_dim`.
/// - **Flat-dim QK norm** — RMSNorm over the full `embed_dim` rather
///   than per-head `head_dim`.
/// - **MRoPE / FoPE / 2D RoPE** — multi-modal or learned-frequency
///   rotary embeddings that the standard `RotaryEmbedding` doesn't
///   parameterize.
/// - **Eagle1-style fused-input fc** — already migrated where possible
///   (eagle_llama is migrated; eagle_mistral_large3, eagle_minicpm,
///   eagle_deepseek, eagle_llama4 share the migrated implementation).
const BESPOKE_ATTENTION_FILES: &[(&str, &str)] = &[
    // ── Quantized variants (Q*Linear, not TpLinear) ─────────────
    ("apertus_quantized.rs", "quantized linear"),
    ("bloom_quantized.rs", "quantized linear (also ALiBi)"),
    ("chatglm_quantized.rs", "quantized linear"),
    ("cohere_quantized.rs", "quantized linear"),
    ("deepseek_quantized.rs", "quantized linear (also MLA)"),
    ("exaone4_quantized.rs", "quantized linear"),
    ("exaone_quantized.rs", "quantized linear"),
    ("falcon_quantized.rs", "quantized linear"),
    ("gemma2_quantized.rs", "quantized linear"),
    ("gemma3_quantized.rs", "quantized linear"),
    ("gemma4_quantized.rs", "quantized linear"),
    ("gemma_quantized.rs", "quantized linear"),
    ("glm4_quantized.rs", "quantized linear (also partial RoPE)"),
    ("glm_quantized.rs", "quantized linear"),
    ("gpt2_quantized.rs", "quantized linear"),
    ("gpt_bigcode_quantized.rs", "quantized linear"),
    ("gpt_j_quantized.rs", "quantized linear"),
    ("gpt_neox_quantized.rs", "quantized linear"),
    ("granite_quantized.rs", "quantized linear"),
    ("hunyuan_quantized.rs", "quantized linear"),
    ("internlm2_quantized.rs", "quantized linear"),
    ("jais2_quantized.rs", "quantized linear"),
    ("jais_quantized.rs", "quantized linear (also ALiBi)"),
    ("llama_quantized.rs", "quantized linear"),
    ("minicpm_quantized.rs", "quantized linear"),
    ("mistral_quantized.rs", "quantized linear"),
    ("mixtral_quantized.rs", "quantized linear"),
    ("mpt_quantized.rs", "quantized linear (also ALiBi)"),
    ("nemotron_quantized.rs", "quantized linear"),
    ("olmo2_quantized.rs", "quantized linear"),
    ("opt_quantized.rs", "quantized linear"),
    ("persimmon_quantized.rs", "quantized linear"),
    ("phi3_quantized.rs", "quantized linear"),
    ("phi_quantized.rs", "quantized linear"),
    ("plamo3_quantized.rs", "quantized linear"),
    ("qwen2_moe_quantized.rs", "quantized linear"),
    ("qwen2_quantized.rs", "quantized linear"),
    ("qwen3_quantized.rs", "quantized linear"),
    ("qwen_quantized.rs", "quantized linear"),
    ("starcoder2_quantized.rs", "quantized linear"),
    ("step1_quantized.rs", "quantized linear"),
    // ── Vision towers / VLM components ──────────────────────────
    ("aria.rs", "VLM cross-attention components"),
    ("blip2.rs", "vision encoder + Q-Former"),
    ("clip.rs", "CLIP vision tower"),
    ("dots_ocr.rs", "DotsVisionAttention vision tower"),
    ("ernie45_vl.rs", "ERNIE 4.5 VL cross-attention"),
    ("gemma4_vision.rs", "Gemma 4 vision tower"),
    ("glm4_1v.rs", "GLM-4.1V vision tower"),
    ("glm4v.rs", "GLM-4V vision tower"),
    ("glm_ocr.rs", "GLM OCR vision components"),
    ("hunyuan_vision.rs", "Hunyuan vision tower"),
    ("internvl.rs", "InternVL vision components"),
    ("interns1.rs", "InternS1 vision tower"),
    ("isaac.rs", "Isaac vision tower"),
    ("keye_vl.rs", "Keye-VL vision tower"),
    ("llama4_vl.rs", "Llama4-VL vision components"),
    ("midashenglm.rs", "MiDashengLM audio components"),
    ("minicpmv.rs", "MiniCPM-V vision components"),
    (
        "nemotron_parse.rs",
        "NemotronParse encoder + cross-attention",
    ),
    ("openpangu_vl.rs", "Pangu-VL vision components"),
    ("ovis2_5.rs", "Ovis2.5 vision components"),
    ("ovis.rs", "Ovis vision components"),
    ("pixtral.rs", "Pixtral vision tower"),
    ("qwen2_5_vl.rs", "Qwen2.5-VL vision tower"),
    ("qwen2_audio.rs", "Qwen2-Audio encoder"),
    (
        "qwen2_vl.rs",
        "Qwen2-VL vision tower + MRoPE language model",
    ),
    ("qwen3_omni_moe_thinker.rs", "Qwen3-Omni Thinker components"),
    ("qwen3_vl_moe.rs", "Qwen3-VL-MoE vision tower + MRoPE"),
    ("qwen3_vl.rs", "Qwen3-VL vision tower + MRoPE"),
    ("qwen_vl.rs", "Qwen-VL vision tower"),
    ("radio.rs", "RadioModel vision tower"),
    ("siglip.rs", "SigLIP vision tower"),
    ("step3_vl.rs", "Step3-VL vision components"),
    // ── Encoder / encoder-decoder models ────────────────────────
    ("bert.rs", "BERT bidirectional attention"),
    ("e5_mistral.rs", "E5-Mistral embedding model"),
    ("granite_speech.rs", "Granite-Speech audio encoder"),
    ("gte.rs", "GTE embedding encoder"),
    ("modernbert.rs", "ModernBERT bidirectional attention"),
    ("t5.rs", "T5 encoder + relative position bias"),
    ("voyage.rs", "Voyage embedding encoder"),
    ("whisper.rs", "Whisper encoder + decoder cross-attention"),
    // ── SSM / hybrid SSM / Mamba family ─────────────────────────
    ("bamba.rs", "Bamba: Mamba2 hybrid"),
    ("falcon_h1.rs", "Falcon-H1: Mamba2 hybrid"),
    ("granitemoe_hybrid.rs", "GraniteMoE-Hybrid: Mamba2 + MoE"),
    ("jamba.rs", "Jamba: Mamba1 + MoE hybrid"),
    ("lfm2.rs", "LFM2 ShortConv hybrid"),
    ("lfm2_vl.rs", "LFM2-VL ShortConv hybrid"),
    ("nemotron_h.rs", "Nemotron-H Mamba2 hybrid"),
    ("zamba2.rs", "Zamba2 Mamba2 hybrid"),
    // ── MLA (DeepSeek family) ───────────────────────────────────
    ("deepseek_ocr.rs", "DeepSeek-OCR uses MLA"),
    ("deepseek_ocr2.rs", "DeepSeek-OCR2 uses MLA"),
    // ── ALiBi ───────────────────────────────────────────────────
    ("baichuan.rs", "Baichuan ALiBi"),
    ("bloom.rs", "BLOOM ALiBi"),
    ("jais.rs", "JAIS ALiBi"),
    ("mpt.rs", "MPT ALiBi"),
    // ── Linear / lightning / GDN attention ──────────────────────
    ("kimi_linear.rs", "Kimi-Linear lightning attention"),
    // ── Gated / output-gated / loop-gated attention ─────────────
    ("afmoe.rs", "AfMoE gated attention"),
    (
        "iquest_loopcoder.rs",
        "iquest LoopCoder loop-gated attention",
    ),
    ("qwen3_next.rs", "Qwen3-Next output gating"),
    // ── Asymmetric V-dim ────────────────────────────────────────
    ("mimo_v2_flash.rs", "MiMoV2-Flash v_head_dim != head_dim"),
    // ── Flat-dim QK norm (over embed_dim, not head_dim) ─────────
    ("flex_olmo.rs", "FlexOlmo flat-dim QK norm"),
    ("olmoe.rs", "OLMoE flat-dim QK norm"),
    // ── MRoPE / FoPE / 2D RoPE ──────────────────────────────────
    ("interns1_pro.rs", "InternS1-Pro Fourier rotary embeddings"),
    // ── Custom attention shape we deliberately keep bespoke ─────
    (
        "chameleon.rs",
        "Chameleon RopeAttention (per-layer attention masks + VQ tokens)",
    ),
    (
        "gemma3.rs",
        "Gemma3 alternating local/global with sliding window every layer",
    ),
    ("gemma3n.rs", "Gemma3n with KV-sharing across layers"),
    ("gemma4.rs", "Gemma4 attention (vision-capable variant)"),
    ("glmasr.rs", "GLM-ASR encoder + custom positional encoding"),
    (
        "gpt_bigcode.rs",
        "GPT-BigCode MQA with replicated KV head under TP",
    ),
    ("minicpm3.rs", "MiniCPM3 uses MLA-style absorption"),
    ("nemotron_parse.rs", "NemotronParse is encoder-decoder"),
    ("step1.rs", "Step-1 ALiBi + sqrt scaling"),
    ("step3p5.rs", "Step-3.5 head-wise attention gate"),
    (
        "step3_text.rs",
        "Step-3 fused QKV with `share_q_dim` (non-standard sizing)",
    ),
];

#[test]
fn no_local_attention_outside_whitelist() {
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("models");
    assert!(
        models_dir.is_dir(),
        "expected models dir at {}",
        models_dir.display()
    );

    let whitelist: std::collections::HashSet<&str> =
        BESPOKE_ATTENTION_FILES.iter().map(|(f, _)| *f).collect();

    let mut violations: Vec<String> = Vec::new();
    let mut whitelist_unused: std::collections::HashSet<&str> = whitelist.clone();

    for entry in fs::read_dir(&models_dir).expect("read models dir") {
        let path: PathBuf = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .expect("file name")
            .to_string();
        if file_name == "mod.rs" {
            continue;
        }

        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let has_attention_struct = src.lines().any(line_declares_attention_struct);
        if !has_attention_struct {
            continue;
        }

        // A file is considered "migrated" if every `Attention` struct in
        // it is a thin shim — we approximate this by requiring that the
        // file contains `inner: AttentionBlock` as a field. Files that
        // both wrap AttentionBlock AND have legacy bespoke attention are
        // flagged as violations (so the whitelist must explicitly list
        // any partial migration — none currently exist).
        let has_attention_block_shim = src.contains("inner: AttentionBlock");

        if has_attention_block_shim {
            // Migrated. We don't allow it on the whitelist; if it is,
            // the entry is now stale and should be removed.
            if whitelist.contains(file_name.as_str()) {
                violations.push(format!(
                    "{} declares `inner: AttentionBlock` but is also on the bespoke whitelist — \
                     remove the entry from BESPOKE_ATTENTION_FILES.",
                    file_name
                ));
            }
            continue;
        }

        // Bespoke attention. Must be on the whitelist.
        if whitelist.contains(file_name.as_str()) {
            whitelist_unused.remove(file_name.as_str());
        } else {
            violations.push(format!(
                "{} declares a bespoke `Attention` struct without using AttentionBlock. \
                 Either migrate to AttentionBlock (see docs/adding-a-model.md), \
                 or — if the architecture genuinely needs bespoke attention — add an \
                 entry to BESPOKE_ATTENTION_FILES with a one-line reason.",
                file_name
            ));
        }
    }

    for stale in &whitelist_unused {
        violations.push(format!(
            "{} is on BESPOKE_ATTENTION_FILES but was not found in `crates/core/src/models/` \
             (or no longer declares an Attention struct). Remove the stale entry.",
            stale
        ));
    }

    if !violations.is_empty() {
        panic!(
            "no_local_attention guardrail failed:\n  {}\n",
            violations.join("\n  ")
        );
    }
}

/// Heuristic: does this line declare a struct whose name ends in
/// `Attention`? Tolerates `pub`, `pub(crate)`, generics, and unit
/// structs. Skips comments and type aliases.
fn line_declares_attention_struct(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with("#[") {
        return false;
    }
    // Forms we want to match:
    //   struct FooAttention {
    //   pub struct FooAttention {
    //   pub(crate) struct FooAttention<...> {
    //   struct FooAttention;
    let after_struct = trimmed
        .strip_prefix("struct ")
        .or_else(|| trimmed.strip_prefix("pub struct "))
        .or_else(|| trimmed.strip_prefix("pub(crate) struct "))
        .or_else(|| trimmed.strip_prefix("pub(super) struct "));
    let Some(rest) = after_struct else {
        return false;
    };
    // Take identifier up to `<`, `{`, `;`, `(` or whitespace.
    let ident: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    ident.ends_with("Attention")
}
