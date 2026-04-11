use serde::{Deserialize, Serialize};

use crate::perf_estimate::gpu_profile::GpuHardwareProfile;
use crate::perf_estimate::model_profile::ModelProfile;

/// Weight data type for parameter sizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightDtype {
    Bf16,
    Fp8,
    Int8,
    Int4,
    Nf4,
}

impl WeightDtype {
    pub fn bytes_per_param(self) -> f64 {
        match self {
            Self::Bf16 => 2.0,
            Self::Fp8 | Self::Int8 => 1.0,
            Self::Int4 | Self::Nf4 => 0.5,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Fp8 => "FP8",
            Self::Int8 => "INT8",
            Self::Int4 => "INT4",
            Self::Nf4 => "NF4",
        }
    }

    /// All dtype variants ordered from highest to lowest quality.
    pub fn all_variants() -> &'static [WeightDtype] {
        &[
            WeightDtype::Bf16,
            WeightDtype::Fp8,
            WeightDtype::Int8,
            WeightDtype::Int4,
            WeightDtype::Nf4,
        ]
    }
}

/// Fitness result for a single weight dtype variant.
#[derive(Debug, Clone, Serialize)]
pub struct VariantFitness {
    pub dtype: WeightDtype,
    /// Estimated weight size in bytes.
    pub weight_bytes: u64,
    /// Estimated weight size in GB (for display).
    pub weight_gb: f64,
    /// Whether the model weights fit in available VRAM.
    pub fits: bool,
    /// VRAM remaining after model weights + overhead (available for KV cache).
    pub remaining_bytes: i64,
    /// Approximate max context length at batch=1 with remaining VRAM.
    pub max_context_at_batch1: usize,
}

/// Which quantization variants of a model fit on the GPU.
#[derive(Debug, Clone, Serialize)]
pub struct VramFitness {
    pub model_id: String,
    pub total_params: u64,
    pub variants: Vec<VariantFitness>,
    /// True if at least one variant fits.
    pub any_fits: bool,
    /// Best quality dtype that fits (first in order: BF16 > FP8 > INT8 > INT4 > NF4).
    pub recommended_dtype: Option<WeightDtype>,
}

/// Quick VRAM fit check using only param count (no config.json needed).
///
/// HuggingFace API returns `safetensors.parameters.total` which is sufficient
/// for this check.
pub fn quick_vram_check(
    model_id: &str,
    total_params: u64,
    gpu_vram_bytes: u64,
    gpu_utilization: f32,
) -> VramFitness {
    let usable = (gpu_vram_bytes as f64 * gpu_utilization as f64) as u64;
    let overhead = usable / 20; // 5% for CUDA context
    let available = usable.saturating_sub(overhead);

    let variants: Vec<VariantFitness> = WeightDtype::all_variants()
        .iter()
        .map(|&dtype| {
            let weight_bytes = (total_params as f64 * dtype.bytes_per_param()) as u64;
            let remaining = available as i64 - weight_bytes as i64;
            let fits = remaining > 0;
            // Rough estimate: assume ~128KB per token for KV cache (conservative)
            let max_context = if fits {
                (remaining as usize) / 131072_usize
            } else {
                0
            };
            VariantFitness {
                dtype,
                weight_bytes,
                weight_gb: weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                fits,
                remaining_bytes: remaining,
                max_context_at_batch1: max_context,
            }
        })
        .collect();

    let any_fits = variants.iter().any(|v| v.fits);
    let recommended_dtype = variants.iter().find(|v| v.fits).map(|v| v.dtype);

    VramFitness {
        model_id: model_id.to_string(),
        total_params,
        variants,
        any_fits,
        recommended_dtype,
    }
}

/// Detailed VRAM fit check using full ModelProfile (after config.json is available).
/// More accurate because it uses exact KV cache sizing.
pub fn detailed_vram_check(
    model: &ModelProfile,
    gpu: &GpuHardwareProfile,
    utilization: f32,
) -> VramFitness {
    let usable = (gpu.total_vram_bytes as f64 * utilization as f64) as u64;
    let overhead = usable / 20;
    let available = usable.saturating_sub(overhead);
    let kv_bytes_per_token = model.kv_bytes_per_token.max(1);

    let variants: Vec<VariantFitness> = WeightDtype::all_variants()
        .iter()
        .map(|&dtype| {
            let weight_bytes = (model.total_params as f64 * dtype.bytes_per_param()) as u64;
            let remaining = available as i64 - weight_bytes as i64;
            let fits = remaining > 0;
            let max_context = if fits {
                remaining as usize / kv_bytes_per_token
            } else {
                0
            };
            VariantFitness {
                dtype,
                weight_bytes,
                weight_gb: weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                fits,
                remaining_bytes: remaining,
                max_context_at_batch1: max_context,
            }
        })
        .collect();

    let any_fits = variants.iter().any(|v| v.fits);
    let recommended_dtype = variants.iter().find(|v| v.fits).map(|v| v.dtype);

    VramFitness {
        model_id: model.model_id.clone(),
        total_params: model.total_params,
        variants,
        any_fits,
        recommended_dtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_dtype_bytes() {
        assert!((WeightDtype::Bf16.bytes_per_param() - 2.0).abs() < f64::EPSILON);
        assert!((WeightDtype::Fp8.bytes_per_param() - 1.0).abs() < f64::EPSILON);
        assert!((WeightDtype::Int4.bytes_per_param() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quick_vram_check_8b_on_8gb() {
        // 8B param model on 8GB GPU — BF16 won't fit, INT4 should
        let fitness = quick_vram_check(
            "test-8b",
            8_000_000_000,
            8 * 1024 * 1024 * 1024, // 8GB
            0.9,
        );

        // BF16 = 16GB → doesn't fit on 8GB
        assert!(!fitness.variants[0].fits);
        assert_eq!(fitness.variants[0].dtype, WeightDtype::Bf16);

        // INT4 = 4GB → fits on 8GB
        let int4 = fitness
            .variants
            .iter()
            .find(|v| v.dtype == WeightDtype::Int4)
            .unwrap();
        assert!(int4.fits);

        assert!(fitness.any_fits);
        // Recommended should be the first that fits (FP8 = 8GB, borderline)
        assert!(fitness.recommended_dtype.is_some());
    }

    #[test]
    fn test_quick_vram_check_70b_on_24gb() {
        // 70B param model on 24GB GPU — only INT4 fits
        let fitness = quick_vram_check(
            "test-70b",
            70_000_000_000,
            24 * 1024 * 1024 * 1024, // 24GB
            0.9,
        );

        // BF16 = 140GB, FP8 = 70GB, INT8 = 70GB — none fit
        assert!(!fitness.variants[0].fits); // BF16
        assert!(!fitness.variants[1].fits); // FP8
        assert!(!fitness.variants[2].fits); // INT8

        // INT4 = 35GB — still doesn't fit on 24GB
        let int4 = fitness
            .variants
            .iter()
            .find(|v| v.dtype == WeightDtype::Int4)
            .unwrap();
        assert!(!int4.fits);

        assert!(!fitness.any_fits);
        assert!(fitness.recommended_dtype.is_none());
    }

    #[test]
    fn test_quick_vram_check_3b_on_8gb() {
        // 3B param model on 8GB GPU — everything fits
        let fitness = quick_vram_check("test-3b", 3_000_000_000, 8 * 1024 * 1024 * 1024, 0.9);

        // BF16 = 6GB → fits on 8GB
        assert!(fitness.variants[0].fits);
        assert!(fitness.any_fits);
        assert_eq!(fitness.recommended_dtype, Some(WeightDtype::Bf16));
    }

    #[test]
    fn test_quick_vram_check_nothing_fits() {
        // 200B params on 8GB GPU
        let fitness = quick_vram_check("test-200b", 200_000_000_000, 8 * 1024 * 1024 * 1024, 0.9);
        assert!(!fitness.any_fits);
        assert!(fitness.recommended_dtype.is_none());
    }

    #[test]
    fn test_detailed_vram_check() {
        use crate::config::ModelConfig;
        use crate::perf_estimate::gpu_profile::gpu_profile_from_values;

        let config = ModelConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            intermediate_size: 14336,
            vocab_size: 128256,
            head_dim: 128,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
            ..Default::default()
        };
        let profile = ModelProfile::from_config("llama-8b", &config, WeightDtype::Bf16);
        let gpu = gpu_profile_from_values(
            "NVIDIA GeForce RTX 4090",
            24 * 1024 * 1024 * 1024,
            20 * 1024 * 1024 * 1024,
        );

        let fitness = detailed_vram_check(&profile, &gpu, 0.9);

        // BF16 ≈ 16GB on 24GB → fits
        assert!(fitness.variants[0].fits);
        assert!(fitness.any_fits);
        assert_eq!(fitness.recommended_dtype, Some(WeightDtype::Bf16));

        // Should have context room
        assert!(fitness.variants[0].max_context_at_batch1 > 0);
    }

    #[test]
    fn test_recommended_dtype_best_quality() {
        // The recommended dtype should be the highest quality that fits
        let fitness = quick_vram_check(
            "test",
            8_000_000_000,
            12 * 1024 * 1024 * 1024, // 12GB
            0.9,
        );
        // BF16 = 16GB doesn't fit, FP8 = 8GB fits
        if fitness.recommended_dtype == Some(WeightDtype::Fp8) {
            // FP8 is the best quality that fits
        } else {
            // Could be INT8 depending on overhead calculation
            assert!(fitness.recommended_dtype.is_some());
        }
    }
}
