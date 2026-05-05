//! Phase 8 invariant: every model file in `crates/core/src/models/`
//! that declares a vision-tower struct must either route through
//! [`crate::multimodal::VisionEncoder`] (the canonical CLIP/SigLIP
//! tower, see ADR-0012) or appear in `BESPOKE_VISION_TOWER_FILES`
//! with a one-line reason.
//!
//! Why a guardrail and not a forced collapse: vision encoders carry
//! significant model-specific shape (MRoPE, window attention, 2D RoPE,
//! conditional positional embedding, custom patch handling, audio
//! Mel/Conv1d encoders). ADR-0012 records the decision to keep them
//! bespoke and use this test as the discipline mechanism instead.
//!
//! When you add a new VLM:
//!
//! - **Default path:** if the vision tower is plain CLIP or SigLIP,
//!   route through `multimodal::VisionEncoder` (no whitelist entry
//!   needed).
//! - **Genuinely bespoke** (model-specific RoPE / attention / patch
//!   handling): add the file to `BESPOKE_VISION_TOWER_FILES` with a
//!   one-line reason.
//!
//! Trigger to canonicalise: when ≥3 bespoke files share the same
//! `// reason:` line, that's the signal to extend
//! `multimodal::VisionEncoder` with a new variant.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Files in `crates/core/src/models/` that intentionally implement a
/// bespoke vision (or audio) encoder. Each entry is paired with a
/// one-line reason explaining what model-specific shape forced the
/// bespoke implementation.
///
/// Categories:
///
/// - **Custom RoPE in vision tower** — multi-modal / 2D / Fourier
///   rotary embeddings that don't fit the standard `RotaryEmbedding`.
/// - **Window / dynamic-resolution attention** — Qwen-VL family.
/// - **Custom patch / image-token packing** — Pixtral, Gemma4-Vision.
/// - **Conditional positional embedding** — RadioViT/InternViT.
/// - **Audio encoders** — not vision; Mel-spectrogram + Conv1d
///   front-end.
/// - **Wrappers** — files that own the canonical encoder under a
///   model-specific config (kept bespoke for naming/path reasons).
const BESPOKE_VISION_TOWER_FILES: &[(&str, &str)] = &[
    // ── Canonical CLIP/SigLIP wrappers (kept bespoke for paths) ──
    (
        "clip.rs",
        "ClipVisionConfig + ClipVisionTransformer (path: vision_model)",
    ),
    (
        "siglip.rs",
        "SiglipVisionConfig + SiglipVisionTransformer (path: vision_model)",
    ),
    ("moonvit.rs", "MoonViT 2D RoPE + per-patch input"),
    // ── Qwen-VL family (window attention + MRoPE in vision) ──────
    ("qwen_vl.rs", "Qwen-VL bespoke vision tower"),
    ("qwen2_vl.rs", "Qwen2-VL window attention + MRoPE"),
    ("qwen2_5_vl.rs", "Qwen2.5-VL window attention + MRoPE"),
    ("qwen3_vl.rs", "Qwen3-VL window attention + MRoPE"),
    (
        "qwen3_omni_moe_thinker.rs",
        "Qwen3-Omni Thinker vision + audio components",
    ),
    // ── Custom patch handling / 2D RoPE / unique token packing ───
    ("pixtral.rs", "Pixtral 2D RoPE in vision tower"),
    (
        "gemma4_vision.rs",
        "Gemma4 vision: GeLU-tanh + custom patch + image-token packing",
    ),
    ("dots_ocr.rs", "DotsVisionAttention OCR vision tower"),
    (
        "hunyuan_vision.rs",
        "Hunyuan vision tower with adaptive resolution",
    ),
    (
        "isaac.rs",
        "Isaac Siglip2 + bilinear pos-emb + pixel_shuffle",
    ),
    ("keye_vl.rs", "Keye-VL Siglip + 2D RoPE"),
    (
        "step3_vl.rs",
        "Step3-VL adaptive resolution vision components",
    ),
    ("openpangu_vl.rs", "Pangu-VL custom vision pipeline"),
    ("glm4_1v.rs", "GLM-4.1V vision tower"),
    ("glm4v.rs", "GLM-4V vision tower"),
    ("glm_ocr.rs", "GLM OCR vision components"),
    (
        "ernie45_vl.rs",
        "ERNIE 4.5 VL cross-attention + custom vision",
    ),
    ("llama4_vl.rs", "Llama4-VL vision components"),
    // ── DeepSeek OCR family ──────────────────────────────────────
    ("deepseek_ocr.rs", "DeepSeek-OCR custom vision pipeline"),
    (
        "deepseek_ocr2.rs",
        "DeepSeek-OCR2 SAM-style ViT + dual-mode mask",
    ),
    // ── RadioViT / InternViT (CPE + ViT-H blocks) ────────────────
    (
        "radio.rs",
        "RadioViT: conditional positional embedding + ViT-H blocks",
    ),
    (
        "internvl.rs",
        "InternVL routes through InternViT/RadioModel",
    ),
    ("interns1.rs", "InternS1 InternViT/RadioModel components"),
    // ── Other VLMs with bespoke vision pipelines ─────────────────
    ("minicpmv.rs", "MiniCPM-V SigLIP + adaptive resampler"),
    ("ovis2_5.rs", "Ovis2.5 vision components"),
    (
        "lfm2_vl.rs",
        "LFM2-VL vision (uses ShortConv hybrid backbone)",
    ),
    ("bagel.rs", "Bagel custom multimodal pipeline"),
    ("molmo2.rs", "Molmo2 vision tower + ImageProjectorMLP"),
    ("minicpmo.rs", "MiniCPM-O omni-modal (vision + audio)"),
    ("gemma3n_vlm.rs", "Gemma3n VLM with audio encoder"),
    // ── Audio encoders (not vision; share the discipline pattern) ─
    ("qwen2_audio.rs", "Qwen2-Audio Mel + Conv1d encoder"),
    ("granite_speech.rs", "Granite-Speech audio encoder"),
    ("whisper.rs", "Whisper encoder + decoder cross-attention"),
    (
        "funaudiochat.rs",
        "FunAudioChat audio encoder + Conv1d front-end",
    ),
];

#[test]
fn no_local_vision_tower_outside_whitelist() {
    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("models");
    assert!(
        models_dir.is_dir(),
        "expected models dir at {}",
        models_dir.display()
    );

    let whitelist: HashSet<&str> = BESPOKE_VISION_TOWER_FILES.iter().map(|(f, _)| *f).collect();

    let mut violations: Vec<String> = Vec::new();
    let mut whitelist_unused: HashSet<&str> = whitelist.clone();

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

        let has_vision_struct = src.lines().any(line_declares_vision_tower_struct);
        if !has_vision_struct {
            continue;
        }

        // A file is considered "canonical" if it routes through
        // `crate::multimodal::VisionEncoder` (the CLIP/SigLIP tower).
        // Also accept files that use the canonical types as building
        // blocks — they're still on the canonical path.
        let uses_canonical = src.contains("multimodal::VisionEncoder")
            || src.contains("multimodal::vision::VisionEncoder")
            || src.contains("crate::multimodal::vision_tower")
            || src.contains("VisionTowerFactory");

        if uses_canonical && !whitelist.contains(file_name.as_str()) {
            // Canonical path; no whitelist needed.
            continue;
        }

        if whitelist.contains(file_name.as_str()) {
            whitelist_unused.remove(file_name.as_str());
        } else {
            violations.push(format!(
                "{} declares a bespoke vision/audio tower struct without using \
                 multimodal::VisionEncoder. Either route through the canonical \
                 tower (see ADR-0012), or add an entry to \
                 BESPOKE_VISION_TOWER_FILES with a one-line reason.",
                file_name
            ));
        }
    }

    for stale in &whitelist_unused {
        violations.push(format!(
            "{} is on BESPOKE_VISION_TOWER_FILES but no longer declares a vision \
             tower struct (or no longer exists in models/). Remove the stale entry.",
            stale
        ));
    }

    if !violations.is_empty() {
        panic!(
            "no_local_vision_tower guardrail failed:\n  {}\n",
            violations.join("\n  ")
        );
    }
}

/// Heuristic: is this line a `struct Foo*Vision*` / `struct
/// FooViT*` / `struct FooImageEncoder*` / `struct FooMel*` / `struct
/// FooAudioEncoder*` declaration? Tolerates `pub` and `pub(crate)`.
fn line_declares_vision_tower_struct(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with("#[") {
        return false;
    }
    let after_struct = trimmed
        .strip_prefix("struct ")
        .or_else(|| trimmed.strip_prefix("pub struct "))
        .or_else(|| trimmed.strip_prefix("pub(crate) struct "))
        .or_else(|| trimmed.strip_prefix("pub(super) struct "));
    let Some(rest) = after_struct else {
        return false;
    };
    let ident: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let lc = ident.to_lowercase();
    // Heuristics — broad enough to catch the bespoke encoders, narrow
    // enough not to catch every `Vision*Config` / `*Patch*` helper.
    let looks_vision = lc.contains("visiontransformer")
        || lc.contains("visiontower")
        || lc.contains("visionattention")
        || lc.contains("visionencoder")
        || lc.contains("visionpatchembed")
        || lc.contains("visionmlp")
        || lc.contains("visionmodel")
        || lc.contains("imageencoder")
        || lc.contains("imageprojector");
    let looks_vit = lc.ends_with("vit")
        || lc.contains("vitblock")
        || lc.contains("vitencoder")
        || lc.contains("vittransformer")
        || lc.contains("clipvision")
        || lc.contains("siglip");
    let looks_audio = lc.contains("melspectrogram")
        || lc.contains("audioencoder")
        || lc.contains("whisperencoder")
        || lc.contains("speechencoder");
    looks_vision || looks_vit || looks_audio
}
