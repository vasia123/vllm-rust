//! PTPC FP8 quantization — Per-Token Per-Channel dynamic FP8 (ROCm/AMD).
//!
//! `PTPCFp8Config` is a thin wrapper over [`Fp8Config`] that:
//! - Uses `is_checkpoint_fp8_serialized = false` (dynamic-only checkpoints).
//! - Requires `min_capability = 94` (AMD Instinct MI300 and newer).
//! - Reports [`QuantizationMethod::PtpcFp8`] so detection round-trips cleanly.
//!
//! Weight format is identical to standard FP8: U8 weights + F32 scales,
//! loaded by the same [`Fp8WeightLoader`].
//!
//! Reference: `vllm/model_executor/layers/quantization/ptpc_fp8.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::fp8::Fp8Config;

// ─── PtpcFp8Config ───────────────────────────────────────────────────────────

/// PTPC FP8 quantization configuration (AMD MI300+).
///
/// Delegates all weight loading and layer creation to [`Fp8Config`];
/// only `method()` and `min_capability()` differ.
#[derive(Debug, Clone, Default)]
pub struct PtpcFp8Config {
    inner: Fp8Config,
}

impl PtpcFp8Config {
    /// Create from a detected quantization config.
    ///
    /// Only `activation_scheme` and `ignored_layers` are meaningful here;
    /// block-wise quantization is not used by PTPC FP8 checkpoints.
    pub fn from_detected(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        // PTPC FP8 is always dynamic (static raises an error in the Python reference).
        let activation_scheme = raw_config
            .get("activation_scheme")
            .and_then(|v| v.as_str())
            .unwrap_or("dynamic");

        let inner = Fp8Config::from_detected(Some(8), Some(activation_scheme), raw_config);
        Self { inner }
    }
}

impl QuantizationConfig for PtpcFp8Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::PtpcFp8
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        self.inner.supported_act_dtypes()
    }

    /// AMD MI300 or newer required.
    fn min_capability(&self) -> u32 {
        94
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.inner.is_layer_skipped(layer_name)
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Delegate to Fp8Config — same Fp8Linear layer, same load path.
        self.inner
            .create_linear(in_features, out_features, bias, device)
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn default_config() -> PtpcFp8Config {
        PtpcFp8Config::from_detected(&HashMap::new())
    }

    #[test]
    fn test_ptpc_fp8_method() {
        let cfg = default_config();
        assert_eq!(cfg.method(), QuantizationMethod::PtpcFp8);
    }

    #[test]
    fn test_ptpc_fp8_min_capability() {
        let cfg = default_config();
        assert_eq!(cfg.min_capability(), 94);
    }

    #[test]
    fn test_ptpc_fp8_ignored_layers() {
        let mut raw = HashMap::new();
        raw.insert("ignored_layers".to_string(), serde_json::json!(["lm_head"]));
        let cfg = PtpcFp8Config::from_detected(&raw);
        assert!(cfg.is_layer_skipped("model.lm_head.weight"));
        assert!(!cfg.is_layer_skipped("model.layers.0.mlp.gate_proj"));
    }

    #[test]
    fn test_ptpc_fp8_create_linear() {
        let cfg = default_config();
        let linear = cfg.create_linear(64, 128, false, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
        assert!(!l.has_bias());
    }

    #[test]
    fn test_ptpc_fp8_create_linear_with_bias() {
        let cfg = default_config();
        let linear = cfg.create_linear(32, 64, true, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert!(l.has_bias());
    }

    #[test]
    fn test_ptpc_fp8_linear_is_fp8_type() {
        // The created linear should be an Fp8Linear under the hood.
        let cfg = default_config();
        let linear = cfg.create_linear(64, 64, false, &Device::Cpu).unwrap();
        // Fp8Linear initialises with BF16 zero weights.
        assert_eq!(linear.weight_dtype(), DType::BF16);
    }
}
