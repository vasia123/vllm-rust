//! AWQ-Marlin quantization.
//!
//! AWQ-Marlin loads weights using the AWQ checkpoint format (qweight, scales,
//! qzeros with interleaved nibble ordering) and routes inference through the
//! Marlin INT4 kernel.  The weight loader deinterleaves AWQ's nibble ordering
//! to produce standard GPTQ sequential packing, then passes the result to
//! `MarlinLinear` which handles the final GPTQ → Marlin tile repack.
//!
//! AWQ nibble ordering (interleaved even-odd):
//!   u32 encodes [v0, v2, v4, v6, v1, v3, v5, v7] in nibbles 0..7
//!
//! GPTQ sequential ordering:
//!   u32 encodes [v0, v1, v2, v3, v4, v5, v6, v7] in nibbles 0..7
//!
//! Applies `undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}` per u32 word.
//!
//! References:
//! - vLLM: `vllm/model_executor/layers/quantization/awq_marlin.py`
//! - CUDA kernel: `csrc/quantization/marlin/awq_marlin_repack.cu`

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::marlin::{MarlinConfig, MarlinLinear};

// ─── AwqMarlinConfig ─────────────────────────────────────────────────────────

/// AWQ-Marlin quantization configuration.
///
/// Asymmetric INT4 with group scales and zero points, accelerated by the
/// Marlin kernel.  Requires GPU compute capability ≥ 80 (Ampere).
#[derive(Debug, Clone)]
pub struct AwqMarlinConfig {
    inner: MarlinConfig,
    /// Whether lm_head is quantized (false = skip lm_head).
    lm_head_quantized: bool,
}

impl AwqMarlinConfig {
    /// Create from detected quantization fields.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        // AWQ-Marlin only supports 4-bit in vLLM; default to 4 for other bit widths.
        let _ = bits; // preserved for API symmetry
        let group_size = group_size.map(|g| g as i32).unwrap_or(128);
        let inner = MarlinConfig::awq_int4(group_size);

        let lm_head_quantized = raw_config
            .get("lm_head_quantized")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Self {
            inner,
            lm_head_quantized,
        }
    }

    /// Access the underlying Marlin config.
    pub fn marlin_config(&self) -> &MarlinConfig {
        &self.inner
    }

    /// Group size used for quantization.
    pub fn group_size(&self) -> i32 {
        self.inner.group_size
    }
}

impl QuantizationConfig for AwqMarlinConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::AwqMarlin
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        80 // Marlin requires Ampere
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        !self.lm_head_quantized && layer_name.contains("lm_head")
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(MarlinLinear::new(
            in_features,
            out_features,
            bias,
            self.inner.clone(),
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

// ─── AWQ nibble deinterleaving ────────────────────────────────────────────────

/// Repack `qweight` from AWQ nibble ordering to GPTQ sequential ordering (CPU).
///
/// AWQ packs 8 int4 values per u32 in interleaved even-odd order:
/// nibble positions 0..7 hold `[v0, v2, v4, v6, v1, v3, v5, v7]`.
///
/// GPTQ packs them sequentially:
/// nibble positions 0..7 hold `[v0, v1, v2, v3, v4, v5, v6, v7]`.
///
/// Applies `undo_pack = {0,4,1,5,2,6,3,7}` per 32-bit word.
///
/// NOTE: The vLLM CUDA reference performs AWQ → Marlin directly via
/// `awq_marlin_repack.cu`.  This CPU fallback produces GPTQ ordering, which
/// `MarlinLinear::process_weights` then converts to Marlin tile format via
/// `repack_gptq_to_marlin`.  The result is numerically equivalent.
/// A direct CUDA path can be added by implementing the PTX function
/// `awq_marlin_repack_int4` in `marlin_gemm.cu`.
pub(crate) fn repack_awq_nibbles(qweight: &Tensor) -> Result<Tensor> {
    let shape = qweight.shape().clone();
    let device = qweight.device().clone();

    // Bring to CPU u32 for bit manipulation.
    let flat = qweight
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let words = flat.to_vec1::<u32>()?;

    let reordered: Vec<u32> = words.iter().map(|&w| awq_to_gptq_u32(w)).collect();

    Tensor::from_vec(reordered, &shape, &Device::Cpu)?.to_device(&device)
}

/// Convert one u32 word from AWQ interleaved nibble ordering to GPTQ
/// sequential nibble ordering.
///
/// `undo_pack = {0, 4, 1, 5, 2, 6, 3, 7}`:
/// output nibble `i` ← AWQ nibble `undo_pack[i]`.
#[inline]
fn awq_to_gptq_u32(w: u32) -> u32 {
    let n0 = (w) & 0xF;
    let n1 = (w >> 4) & 0xF;
    let n2 = (w >> 8) & 0xF;
    let n3 = (w >> 12) & 0xF;
    let n4 = (w >> 16) & 0xF;
    let n5 = (w >> 20) & 0xF;
    let n6 = (w >> 24) & 0xF;
    let n7 = (w >> 28) & 0xF;
    // output = [n0, n4, n1, n5, n2, n6, n3, n7]
    n0 | (n4 << 4) | (n1 << 8) | (n5 << 12) | (n2 << 16) | (n6 << 20) | (n3 << 24) | (n7 << 28)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_awq_marlin_config_default() {
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &HashMap::new());
        assert_eq!(config.method(), QuantizationMethod::AwqMarlin);
        assert_eq!(config.group_size(), 128);
        assert_eq!(config.min_capability(), 80);
    }

    #[test]
    fn test_awq_marlin_config_lm_head_skip() {
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &HashMap::new());
        // By default lm_head is skipped.
        assert!(config.is_layer_skipped("lm_head"));
        assert!(!config.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn test_awq_marlin_config_lm_head_quantized() {
        let mut raw = HashMap::new();
        raw.insert(
            "lm_head_quantized".to_string(),
            serde_json::Value::Bool(true),
        );
        let config = AwqMarlinConfig::from_detected(Some(4), Some(128), &raw);
        assert!(!config.is_layer_skipped("lm_head"));
    }

    #[test]
    fn test_awq_to_gptq_u32_identity_all_zero() {
        // All-zero word stays zero.
        assert_eq!(awq_to_gptq_u32(0x0000_0000), 0x0000_0000);
    }

    #[test]
    fn test_awq_to_gptq_u32_known_values() {
        // Pack 8 distinct nibbles in AWQ order: v=[1,2,3,4,5,6,7,8]
        // AWQ u32: n0=v[0]=1, n1=v[2]=3, n2=v[4]=5, n3=v[6]=7,
        //          n4=v[1]=2, n5=v[3]=4, n6=v[5]=6, n7=v[7]=8
        let awq: u32 =
            1 | (3 << 4) | (5 << 8) | (7 << 12) | (2 << 16) | (4 << 20) | (6 << 24) | (8 << 28);
        let gptq = awq_to_gptq_u32(awq);
        // GPTQ u32: [v0,v1,v2,v3,v4,v5,v6,v7] = [1,2,3,4,5,6,7,8]
        let expected: u32 =
            1 | (2 << 4) | (3 << 8) | (4 << 12) | (5 << 16) | (6 << 20) | (7 << 24) | (8 << 28);
        assert_eq!(gptq, expected);
    }

    #[test]
    fn test_repack_awq_nibbles_shape_preserved() {
        let device = Device::Cpu;
        // qweight shape for int4: [K/8, N] — use K=128, N=64 → [16, 64]
        let qweight = Tensor::zeros((16usize, 64usize), DType::U32, &device).unwrap();
        let result = repack_awq_nibbles(&qweight).unwrap();
        assert_eq!(result.dims(), &[16, 64]);
    }

    #[test]
    fn test_repack_awq_nibbles_idempotent_all_zero() {
        // All-zero input: deinterleave of zeros is zeros.
        let device = Device::Cpu;
        let qweight = Tensor::zeros((8usize, 32usize), DType::U32, &device).unwrap();
        let result = repack_awq_nibbles(&qweight).unwrap();
        let flat: Vec<u32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&v| v == 0));
    }
}
