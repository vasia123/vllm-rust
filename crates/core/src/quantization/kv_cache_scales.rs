//! Loading per-layer KV cache quantisation scales from a model checkpoint.
//!
//! FP8-quantised checkpoints (NVIDIA AMMO/ModelOpt, Qwen2.5-FP8,
//! GPT-OSS, RedHat-AI/Llama-3-FP8, …) ship per-layer scalar `k_scale`
//! and `v_scale` tensors that pin the FP8 quantisation range of the
//! KV cache the model was calibrated on. When they are present in
//! the safetensors file we want to honour them (one-shot
//! [`KVScales::set`] before the first cache write) instead of running
//! our first-batch absmax calibration: the checkpoint scales reflect
//! an offline calibration over a representative dataset and are
//! strictly better than our online heuristic.
//!
//! The naming is not standardised across hubs — vLLM upstream
//! `maybe_remap_kv_scale_name` carries ~10 patterns. We support the
//! handful that actually ship in production today; unknown patterns
//! fall through silently and the first-write calibration takes over.
//!
//! Sentinel handling mirrors vLLM: a checkpoint value of `-1.0`
//! marks "scales were registered as `nn.Parameter(-1.0)` but never
//! filled" and is treated as missing.

use std::collections::HashMap;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

/// Per-layer scale pair extracted from a checkpoint. Either or both
/// components may be `None` if the checkpoint only ships one side.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LayerKvScales {
    pub k_scale: Option<f32>,
    pub v_scale: Option<f32>,
}

impl LayerKvScales {
    /// Returns `true` if at least one scale is present.
    pub fn any_present(&self) -> bool {
        self.k_scale.is_some() || self.v_scale.is_some()
    }
}

/// Candidate weight-name patterns for a single layer's K scale.
/// `{i}` is replaced by the layer index at call time. Order matters:
/// more-specific patterns first so we don't pick up a stale duplicate
/// in mixed checkpoints.
const K_SCALE_PATTERNS: &[&str] = &[
    // NVIDIA AMMO / ModelOpt FP8: scales live on the projection
    // module after vLLM's checkpoint rewrite.
    "model.layers.{i}.self_attn.k_proj.k_scale",
    "model.layers.{i}.self_attn.qkv_proj.k_scale",
    // vLLM-canonical attention-level naming
    "model.layers.{i}.self_attn.attn.k_scale",
    // Qwen2.5-FP8 / GPT-OSS plain naming
    "model.layers.{i}.self_attn.k_scale",
    // Mistral / Persimmon / older HF transformer naming
    "transformer.h.{i}.attn.k_scale",
    "model.layers.{i}.attn.k_scale",
];

const V_SCALE_PATTERNS: &[&str] = &[
    "model.layers.{i}.self_attn.v_proj.v_scale",
    "model.layers.{i}.self_attn.qkv_proj.v_scale",
    "model.layers.{i}.self_attn.attn.v_scale",
    "model.layers.{i}.self_attn.v_scale",
    "transformer.h.{i}.attn.v_scale",
    "model.layers.{i}.attn.v_scale",
];

/// Deprecated single fused-scale patterns. When matched the value
/// is duplicated to both K and V (legacy convention from early FP8
/// checkpoints).
const FUSED_KV_SCALE_PATTERNS: &[&str] = &[
    "model.layers.{i}.self_attn.kv_scale",
    "model.layers.{i}.self_attn.attn.kv_scale",
    "transformer.h.{i}.attn.kv_scale",
];

/// vLLM sentinel for "registered but never filled" scales —
/// `nn.Parameter(torch.tensor(-1.0))`. Treated as missing.
const SCALE_SENTINEL: f32 = -1.0;
const SENTINEL_TOL: f32 = 1e-6;

/// Attempt to load per-layer KV cache scales from the provided
/// VarBuilder. Returns a map from `layer_idx` to the (k, v) scale
/// pair for every layer where at least one scale was found.
///
/// Behaviour:
/// - Tries each pattern in [`K_SCALE_PATTERNS`] / [`V_SCALE_PATTERNS`]
///   in order; the first hit wins.
/// - Falls back to [`FUSED_KV_SCALE_PATTERNS`] if separate scales
///   were not found — the fused value applies to both K and V.
/// - Skips checkpoint values equal to [`SCALE_SENTINEL`] (-1.0).
/// - Silently ignores layers with no matching key — the caller then
///   keeps the first-write calibration path for those layers.
///
/// Errors are returned only for genuine I/O problems (e.g. the
/// tensor exists but cannot be read or is the wrong dtype); a
/// missing tensor is *not* an error — that is the steady-state
/// outcome for non-FP8 checkpoints.
pub fn try_load_kv_cache_scales(
    vb: &VarBuilder,
    num_layers: usize,
) -> candle_core::Result<HashMap<usize, LayerKvScales>> {
    let mut out: HashMap<usize, LayerKvScales> = HashMap::new();

    for layer_idx in 0..num_layers {
        let mut scales = LayerKvScales {
            k_scale: lookup_first_present(vb, K_SCALE_PATTERNS, layer_idx)?,
            v_scale: lookup_first_present(vb, V_SCALE_PATTERNS, layer_idx)?,
        };

        // Fall back to the deprecated fused scalar if either side is
        // still missing — older checkpoints ship one scalar that
        // applies to both K and V.
        if scales.k_scale.is_none() || scales.v_scale.is_none() {
            if let Some(fused) = lookup_first_present(vb, FUSED_KV_SCALE_PATTERNS, layer_idx)? {
                if scales.k_scale.is_none() {
                    scales.k_scale = Some(fused);
                }
                if scales.v_scale.is_none() {
                    scales.v_scale = Some(fused);
                }
            }
        }

        if scales.any_present() {
            out.insert(layer_idx, scales);
        }
    }

    Ok(out)
}

/// Try each pattern in turn; return the first non-sentinel scalar
/// found, or `None` if no pattern matches.
fn lookup_first_present(
    vb: &VarBuilder,
    patterns: &[&str],
    layer_idx: usize,
) -> candle_core::Result<Option<f32>> {
    for pat in patterns {
        let name = pat.replace("{i}", &layer_idx.to_string());
        if !vb.contains_tensor(&name) {
            continue;
        }
        // Shape is always scalar `[1]` or `[]`. We accept both —
        // `.flatten_all()` collapses to 1-D, and we read the first
        // element. Any tensor under these well-known names is
        // expected to be a single F32 scalar; if it isn't, surface
        // the error so a malformed checkpoint doesn't silently
        // poison the calibration.
        let t: Tensor = vb.get_unchecked(&name)?;
        let value = read_scalar_f32(&t, &name)?;
        if (value - SCALE_SENTINEL).abs() < SENTINEL_TOL {
            // -1.0 sentinel ⇒ treat as missing, keep scanning so a
            // later pattern (e.g. fused) can take over.
            continue;
        }
        return Ok(Some(value));
    }
    Ok(None)
}

/// Read a 1-element tensor as f32. Accepts shape `[]` or `[1]`,
/// any dtype convertible to F32. Bails on multi-element tensors
/// because per-layer KV scales are always per-tensor scalars in the
/// formats we recognise — per-channel / per-token scales (vLLM's
/// "per_token_head" path) need a different code path and are not
/// matched by these key patterns in the first place.
fn read_scalar_f32(t: &Tensor, name: &str) -> candle_core::Result<f32> {
    let total = t.elem_count();
    if total != 1 {
        return Err(candle_core::Error::Msg(format!(
            "kv_cache_scales: expected a single-element tensor for \
             {name}, got shape {:?} ({total} elements). Per-channel \
             KV scales are not yet supported.",
            t.shape()
        )));
    }
    let v: Vec<f32> = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(v[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap as StdHashMap;

    /// Build an in-memory VarBuilder backed by named scalar tensors.
    fn vb_from_scalars(scalars: &[(&str, f32)]) -> VarBuilder<'static> {
        let device = Device::Cpu;
        let mut tensors: StdHashMap<String, Tensor> = StdHashMap::new();
        for (name, value) in scalars {
            let t = Tensor::from_slice(&[*value], 1, &device).unwrap();
            tensors.insert((*name).to_string(), t);
        }
        VarBuilder::from_tensors(tensors, DType::F32, &device)
    }

    #[test]
    fn returns_empty_when_no_scales_present() {
        let vb = vb_from_scalars(&[("model.embed_tokens.weight", 0.0)]);
        let out = try_load_kv_cache_scales(&vb, 4).unwrap();
        assert!(out.is_empty(), "no scale tensors → empty map");
    }

    #[test]
    fn picks_up_modelopt_proj_scales() {
        let vb = vb_from_scalars(&[
            ("model.layers.0.self_attn.k_proj.k_scale", 0.125),
            ("model.layers.0.self_attn.v_proj.v_scale", 0.0625),
        ]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        assert_eq!(out.len(), 1);
        let s = out.get(&0).unwrap();
        assert_eq!(s.k_scale, Some(0.125));
        assert_eq!(s.v_scale, Some(0.0625));
    }

    #[test]
    fn picks_up_qwen_attn_level_scales() {
        let vb = vb_from_scalars(&[
            ("model.layers.5.self_attn.k_scale", 0.5),
            ("model.layers.5.self_attn.v_scale", 0.25),
        ]);
        let out = try_load_kv_cache_scales(&vb, 8).unwrap();
        let s = out.get(&5).expect("layer 5 must be in map");
        assert_eq!(s.k_scale, Some(0.5));
        assert_eq!(s.v_scale, Some(0.25));
        assert!(!out.contains_key(&0), "layers without scales stay unmapped");
    }

    #[test]
    fn fused_kv_scale_duplicates_to_both_sides() {
        let vb = vb_from_scalars(&[("model.layers.0.self_attn.kv_scale", 0.7)]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        let s = out.get(&0).unwrap();
        assert_eq!(s.k_scale, Some(0.7));
        assert_eq!(s.v_scale, Some(0.7));
    }

    #[test]
    fn separate_scale_wins_over_fused_when_both_present() {
        let vb = vb_from_scalars(&[
            ("model.layers.0.self_attn.k_scale", 0.1),
            ("model.layers.0.self_attn.v_scale", 0.2),
            ("model.layers.0.self_attn.kv_scale", 0.9),
        ]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        let s = out.get(&0).unwrap();
        assert_eq!(s.k_scale, Some(0.1));
        assert_eq!(s.v_scale, Some(0.2));
    }

    #[test]
    fn sentinel_minus_one_falls_through_to_next_pattern() {
        // The proj-level pattern stores the -1.0 sentinel (registered
        // but never filled); the attention-level pattern carries the
        // real value. Loader must skip the sentinel and pick up the
        // real one.
        let vb = vb_from_scalars(&[
            ("model.layers.0.self_attn.k_proj.k_scale", -1.0),
            ("model.layers.0.self_attn.k_scale", 0.42),
        ]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        assert_eq!(out.get(&0).unwrap().k_scale, Some(0.42));
    }

    #[test]
    fn sentinel_only_means_no_scale() {
        let vb = vb_from_scalars(&[("model.layers.0.self_attn.k_proj.k_scale", -1.0)]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        assert!(
            out.is_empty(),
            "all-sentinel checkpoints stay on first-write calibration"
        );
    }

    #[test]
    fn fused_pattern_fills_only_missing_side() {
        // K has a real scale; V is absent; fused exists.
        let vb = vb_from_scalars(&[
            ("model.layers.0.self_attn.k_scale", 0.1),
            ("model.layers.0.self_attn.kv_scale", 0.9),
        ]);
        let out = try_load_kv_cache_scales(&vb, 1).unwrap();
        let s = out.get(&0).unwrap();
        assert_eq!(s.k_scale, Some(0.1), "explicit K wins over fused fallback");
        assert_eq!(s.v_scale, Some(0.9), "fused fills the missing V side");
    }

    #[test]
    fn transformer_h_naming_supported() {
        let vb = vb_from_scalars(&[
            ("transformer.h.3.attn.k_scale", 0.3),
            ("transformer.h.3.attn.v_scale", 0.4),
        ]);
        let out = try_load_kv_cache_scales(&vb, 4).unwrap();
        let s = out.get(&3).unwrap();
        assert_eq!(s.k_scale, Some(0.3));
        assert_eq!(s.v_scale, Some(0.4));
    }
}
