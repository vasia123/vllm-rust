//! CPU AWQ (cpu_wna16) quantization — weight-only INT4 for CPU inference.
//!
//! `cpu_awq` / `cpu_wna16` is a CPU-targeted weight-only quantization method
//! that reuses the exact same weight format as AWQ (packed INT4 `qweight`,
//! `qzeros`, and per-group `scales`).  The Python implementation is a thin
//! subclass of `AwqConfig` that overrides `min_capability` to 0 and opts out
//! of all CUDA-specific kernel paths.
//!
//! In Rust this is a wrapper over [`AwqConfig`] that:
//! - Reports [`QuantizationMethod::CpuWna16`] for round-trip identity.
//! - Sets `min_capability = 0` so that CPU-only deployments pass the
//!   capability gate.
//! - Disables Marlin (GPU-only) by forcing `use_marlin = false`.
//!
//! Reference: `vllm/model_executor/layers/quantization/cpu_wna16.py`

use std::collections::HashMap;

use candle_core::{DType, Device, Result};

use super::awq::AwqConfig;
use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

// ─── CpuWna16Config ──────────────────────────────────────────────────────────

/// CPU AWQ quantization configuration (cpu_wna16 / cpu_awq).
///
/// Delegates all weight loading and layer creation to [`AwqConfig`]; only
/// `method()` and `min_capability()` differ.  Marlin is explicitly disabled
/// because it requires a CUDA-capable GPU (compute ≥ 8.0).
#[derive(Debug, Clone)]
pub struct CpuWna16Config {
    inner: AwqConfig,
}

impl CpuWna16Config {
    /// Create from a detected quantization config.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let inner = AwqConfig::from_detected(bits, group_size, raw_config)
            // Marlin requires Ampere GPU — not available for CPU-only inference.
            .with_marlin(false);
        Self { inner }
    }
}

impl QuantizationConfig for CpuWna16Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::CpuWna16
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        self.inner.supported_act_dtypes()
    }

    /// No GPU required — works on CPU.
    fn min_capability(&self) -> u32 {
        0
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

    fn default_config() -> CpuWna16Config {
        CpuWna16Config::from_detected(None, None, &HashMap::new())
    }

    #[test]
    fn test_cpu_wna16_method() {
        let cfg = default_config();
        assert_eq!(cfg.method(), QuantizationMethod::CpuWna16);
    }

    #[test]
    fn test_cpu_wna16_min_capability() {
        // CPU-only: no GPU required.
        let cfg = default_config();
        assert_eq!(cfg.min_capability(), 0);
    }

    #[test]
    fn test_cpu_wna16_no_marlin() {
        // Marlin is GPU-only and must be disabled for CPU inference.
        let cfg = default_config();
        // The underlying AwqConfig.use_marlin should be false.
        assert!(!cfg.inner.use_marlin);
    }

    #[test]
    fn test_cpu_wna16_create_linear() {
        let cfg = default_config();
        let linear = cfg.create_linear(64, 128, false, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert_eq!(l.in_features(), 64);
        assert_eq!(l.out_features(), 128);
        assert!(!l.has_bias());
    }

    #[test]
    fn test_cpu_wna16_create_linear_with_bias() {
        let cfg = default_config();
        let linear = cfg.create_linear(32, 64, true, &Device::Cpu);
        assert!(linear.is_ok());
        let l = linear.unwrap();
        assert!(l.has_bias());
    }
}
