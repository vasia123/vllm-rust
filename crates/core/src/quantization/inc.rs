//! INC (Intel Neural Compressor) quantization — router to GPTQ or AWQ.
//!
//! INC is a meta-quantization config that dispatches to an underlying method
//! based on the checkpoint's `packing_format`:
//!
//! - `"auto_round:auto_gptq"` (default) → [`GptqConfig`]
//! - `"auto_round:auto_awq"` → [`AwqConfig`]
//!
//! Both backends additionally auto-upgrade to Marlin when the layer shape and
//! bit-width are compatible (4-bit symmetric GPTQ / 4-bit AWQ on Ampere+).
//!
//! The `quant_method` JSON key may be `"inc"` or `"auto-round"`; both route
//! here via the detection layer.
//!
//! Reference: `vllm/model_executor/layers/quantization/inc.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result};

use super::awq::AwqConfig;
use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::gptq::GptqConfig;

// ─── Backend enum ─────────────────────────────────────────────────────────────

/// Resolved backend for an INC checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncBackend {
    Gptq,
    Awq,
}

impl IncBackend {
    /// Detect from `packing_format` and explicit `backend` fields.
    fn from_config(raw_config: &HashMap<String, serde_json::Value>) -> Self {
        let packing_format = raw_config
            .get("packing_format")
            .and_then(|v| v.as_str())
            .unwrap_or("auto_round:auto_gptq");

        let backend = raw_config
            .get("backend")
            .or_else(|| raw_config.get("vllm_backend"))
            .and_then(|v| v.as_str())
            .unwrap_or("auto");

        // Explicit backend override takes precedence.
        if backend.contains("awq") {
            return Self::Awq;
        }
        if backend.contains("gptq") || backend.contains("marlin") {
            return Self::Gptq;
        }

        // Fall back to packing_format.
        if packing_format.contains("awq") {
            Self::Awq
        } else {
            // "auto_round:auto_gptq" is the default
            Self::Gptq
        }
    }
}

// ─── IncConfig ────────────────────────────────────────────────────────────────

/// INC quantization configuration — routes to GPTQ or AWQ delegate.
///
/// Resolves the backend at construction time so all `QuantizationConfig`
/// calls are simple delegation to the inner config.
#[derive(Debug, Clone)]
pub struct IncConfig {
    inner: IncInner,
}

#[derive(Debug, Clone)]
enum IncInner {
    Gptq(GptqConfig),
    Awq(AwqConfig),
}

impl IncConfig {
    /// Create from a detected quantization config.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        desc_act: Option<bool>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let backend = IncBackend::from_config(raw_config);

        // sym default is true for INC (unlike GPTQ which defaults false in some files)
        let sym_override = raw_config
            .get("sym")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let mut rc = raw_config.clone();
        rc.entry("sym".to_string())
            .or_insert_with(|| serde_json::Value::Bool(sym_override));

        let inner = match backend {
            IncBackend::Gptq => {
                IncInner::Gptq(GptqConfig::from_detected(bits, group_size, desc_act, &rc))
            }
            IncBackend::Awq => {
                IncInner::Awq(AwqConfig::from_detected(bits, group_size, raw_config))
            }
        };

        Self { inner }
    }
}

impl QuantizationConfig for IncConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Inc
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        match &self.inner {
            IncInner::Gptq(c) => c.supported_act_dtypes(),
            IncInner::Awq(c) => c.supported_act_dtypes(),
        }
    }

    fn min_capability(&self) -> u32 {
        // Python says 60; keep 70 to ensure basic integer matmul support.
        70
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        match &self.inner {
            IncInner::Gptq(c) => c.is_layer_skipped(layer_name),
            IncInner::Awq(c) => c.is_layer_skipped(layer_name),
        }
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        match &self.inner {
            IncInner::Gptq(c) => c.create_linear(in_features, out_features, bias, device),
            IncInner::Awq(c) => c.create_linear(in_features, out_features, bias, device),
        }
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

    fn gptq_config() -> IncConfig {
        let mut raw = HashMap::new();
        raw.insert(
            "packing_format".to_string(),
            serde_json::json!("auto_round:auto_gptq"),
        );
        IncConfig::from_detected(Some(4), Some(128), None, &raw)
    }

    fn awq_config() -> IncConfig {
        let mut raw = HashMap::new();
        raw.insert(
            "packing_format".to_string(),
            serde_json::json!("auto_round:auto_awq"),
        );
        IncConfig::from_detected(Some(4), Some(128), None, &raw)
    }

    #[test]
    fn test_inc_method() {
        assert_eq!(gptq_config().method(), QuantizationMethod::Inc);
        assert_eq!(awq_config().method(), QuantizationMethod::Inc);
    }

    #[test]
    fn test_inc_min_capability() {
        assert_eq!(gptq_config().min_capability(), 70);
    }

    #[test]
    fn test_inc_backend_gptq_by_default() {
        // Default packing_format is auto_round:auto_gptq
        let cfg = IncConfig::from_detected(None, None, None, &HashMap::new());
        assert!(matches!(cfg.inner, IncInner::Gptq(_)));
    }

    #[test]
    fn test_inc_backend_awq_via_packing_format() {
        let cfg = awq_config();
        assert!(matches!(cfg.inner, IncInner::Awq(_)));
    }

    #[test]
    fn test_inc_backend_override_via_explicit_backend() {
        let mut raw = HashMap::new();
        // packing_format says gptq but backend says awq — backend wins
        raw.insert(
            "packing_format".to_string(),
            serde_json::json!("auto_round:auto_gptq"),
        );
        raw.insert("backend".to_string(), serde_json::json!("awq"));
        let cfg = IncConfig::from_detected(Some(4), Some(128), None, &raw);
        assert!(matches!(cfg.inner, IncInner::Awq(_)));
    }

    #[test]
    fn test_inc_create_linear_gptq_path() {
        let cfg = gptq_config();
        let linear = cfg.create_linear(64, 128, false, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
    }

    #[test]
    fn test_inc_create_linear_awq_path() {
        let cfg = awq_config();
        let linear = cfg.create_linear(64, 128, false, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
    }
}
