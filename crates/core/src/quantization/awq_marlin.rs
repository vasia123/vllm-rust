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

#[cfg(feature = "marlin")]
use super::marlin_cuda;

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
        Ok(Box::new(AwqMarlinLinear::new(
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

/// Repack `qweight` per-word: AWQ interleaved nibble order → GPTQ sequential.
///
/// Shape is preserved.  Suitable for `qzeros` (where AWQ and GPTQ share
/// the same `[num_groups, N/8]` layout) but **NOT** for `qweight`, where
/// AWQ packs along the output axis (`[K, N/8]`) and GPTQ packs along the
/// input axis (`[K/8, N]`) — for that, use [`awq_to_gptq_qweight`].
///
/// Applies `undo_pack = {0,4,1,5,2,6,3,7}` per 32-bit word.
pub(crate) fn repack_awq_nibbles(qweight: &Tensor) -> Result<Tensor> {
    let shape = qweight.shape().clone();
    let device = qweight.device().clone();

    // GPU fast path: parallel per-word transform via CUDA kernel.
    // Any sm_80+ device is supported (the kernel uses only integer ops).
    #[cfg(feature = "marlin")]
    if device.is_cuda() {
        let dims = shape.dims();
        if dims.len() == 2 {
            // qweight shape [K/8, N]: reconstruct size_k and size_n
            return marlin_cuda::awq_marlin_repack(qweight, dims[0] * 8, dims[1]);
        }
    }

    // CPU fallback: bring to host, transform word-by-word, return to original device.
    let flat = qweight
        .to_dtype(DType::U32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?;
    let words = flat.to_vec1::<u32>()?;

    let reordered: Vec<u32> = words.iter().map(|&w| awq_to_gptq_u32(w)).collect();

    Tensor::from_vec(reordered, &shape, &Device::Cpu)?.to_device(&device)
}

/// Repack AWQ `qweight` (`[K, N/8]`, packed along the output axis with
/// interleaved nibble ordering) into GPTQ sequential layout (`[K/8, N]`,
/// packed along the input axis).
///
/// This is what vLLM's `awq_marlin_repack` CUDA kernel does; we provide
/// a CPU implementation here so the loader can run AWQ checkpoints
/// through the Marlin kernel (or GPTQ fallback) without any per-forward
/// dequantization.  The transform is one-shot at load time.
///
/// Steps:
/// 1. For each `(k, n)`, extract the int4 nibble from the AWQ word
///    `qweight_awq[k, n/8]` at position `undo_pack[n % 8]`.
/// 2. Pack 8 consecutive `k`-values (k = k_block * 8 + i, i in 0..8) into
///    a single u32 in **sequential** nibble order:
///    `gptq_word = sum_i nib(k_block*8 + i, n) << (i * 4)`.
///
/// Result lives on the same device as the input.
pub(crate) fn awq_to_gptq_qweight(
    awq_qweight: &Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<Tensor> {
    const PACK_FACTOR: usize = 8;
    if !in_features.is_multiple_of(PACK_FACTOR) {
        candle_core::bail!(
            "awq_to_gptq_qweight: in_features ({in_features}) must be divisible by 8"
        );
    }
    if !out_features.is_multiple_of(PACK_FACTOR) {
        candle_core::bail!(
            "awq_to_gptq_qweight: out_features ({out_features}) must be divisible by 8"
        );
    }
    let packed_n = out_features / PACK_FACTOR;
    let packed_k = in_features / PACK_FACTOR;

    if awq_qweight.dtype() != DType::U32 {
        candle_core::bail!(
            "awq_to_gptq_qweight: qweight must be U32, got {:?}",
            awq_qweight.dtype()
        );
    }
    if awq_qweight.dims() != [in_features, packed_n] {
        candle_core::bail!(
            "awq_to_gptq_qweight: qweight shape {:?} != expected [{in_features}, {packed_n}]",
            awq_qweight.dims()
        );
    }

    let original_device = awq_qweight.device().clone();
    let words: Vec<u32> = awq_qweight
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;

    let mut out = vec![0u32; packed_k * out_features];
    for n in 0..out_features {
        let undo_shift = AWQ_UNDO_PACK_SHIFTS[n % PACK_FACTOR];
        let n_block = n / PACK_FACTOR;
        for kb in 0..packed_k {
            let k_base = kb * PACK_FACTOR;
            let mut word: u32 = 0;
            for i in 0..PACK_FACTOR {
                let k = k_base + i;
                let awq_word = words[k * packed_n + n_block];
                let nib = (awq_word >> undo_shift) & 0xF;
                word |= nib << (i * 4);
            }
            out[kb * out_features + n] = word;
        }
    }

    Tensor::from_vec(out, (packed_k, out_features), &Device::Cpu)?.to_device(&original_device)
}

/// Bit-shift positions per output lane in an AWQ packed word.
///
/// AWQ encodes 8 int4 outputs in a u32 with `undo_pack = {0,4,1,5,2,6,3,7}`,
/// so output `i` lives at bit offset `undo_pack[i] * 4`.
const AWQ_UNDO_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];

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

// ─── AwqMarlinLinear ─────────────────────────────────────────────────────────

/// Linear layer that loads AWQ checkpoints and runs through Marlin
/// (or, when the `marlin` feature is off, through the GPTQ CUDA fallback).
///
/// On `load_weights`, this layer:
/// 1. Repacks AWQ `qweight` (`[K, N/8]`, output-axis interleaved) into
///    GPTQ sequential layout (`[K/8, N]`) via [`awq_to_gptq_qweight`].
/// 2. Repacks AWQ `qzeros` (`[num_groups, N/8]`) into sequential nibble
///    ordering via [`repack_awq_nibbles`] (shape preserved).
/// 3. Hands the GPTQ-style tensors to the inner [`MarlinLinear`], whose
///    own `load_weights` then performs the GPTQ → Marlin tile repack.
///
/// All forward passes delegate to `MarlinLinear::forward`, which uses
/// the Marlin INT4 kernel when `feature = "marlin"`, otherwise falls
/// back to `gptq_cuda::gptq_gemm` (via `cuda-kernels`).  Either way is
/// dramatically faster than `AwqLinear`'s per-forward CPU dequant.
#[derive(Debug)]
pub struct AwqMarlinLinear {
    inner: MarlinLinear,
    in_features: usize,
    out_features: usize,
}

impl AwqMarlinLinear {
    /// Create a new AWQ-Marlin linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        marlin_config: MarlinConfig,
        device: &Device,
    ) -> Result<Self> {
        let inner = MarlinLinear::new(in_features, out_features, has_bias, marlin_config, device)?;
        Ok(Self {
            inner,
            in_features,
            out_features,
        })
    }
}

impl QuantizedLinear for AwqMarlinLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        let mut repacked: HashMap<String, Tensor> = HashMap::with_capacity(weights.len());

        if let Some(qw) = weights.get("qweight") {
            let gptq_qw = awq_to_gptq_qweight(qw, self.in_features, self.out_features)?;
            repacked.insert("qweight".to_string(), gptq_qw);
        }
        if let Some(qz) = weights.get("qzeros") {
            // qzeros share the same shape between AWQ and GPTQ
            // (`[num_groups, N/8]`); only the nibble order inside each
            // u32 word changes.
            let gptq_qz = repack_awq_nibbles(qz)?;
            repacked.insert("qzeros".to_string(), gptq_qz);
        }
        // Scales and bias are layout-compatible: pass through.
        if let Some(s) = weights.get("scales") {
            repacked.insert("scales".to_string(), s.clone());
        }
        if let Some(b) = weights.get("bias") {
            repacked.insert("bias".to_string(), b.clone());
        }

        self.inner.load_weights(&repacked)
    }

    fn weight_dtype(&self) -> DType {
        self.inner.weight_dtype()
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.inner.has_bias()
    }
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
    fn test_awq_to_gptq_qweight_shape_transposes_packing_axis() {
        // AWQ qweight shape: [K, N/8]; GPTQ output: [K/8, N].
        let device = Device::Cpu;
        let k = 64usize;
        let n = 32usize;
        let qw = Tensor::zeros((k, n / 8), DType::U32, &device).unwrap();
        let gptq = awq_to_gptq_qweight(&qw, k, n).unwrap();
        assert_eq!(gptq.dims(), &[k / 8, n]);
    }

    #[test]
    fn test_awq_to_gptq_qweight_known_value() {
        // Build a 1×1 (in nibble units) AWQ matrix where K=8, N=8.
        // AWQ shape: [K=8, N/8=1]. Each row k stores 8 nibbles for the
        // 8 outputs of column-block 0, in undo_pack order.
        // GPTQ shape after repack: [K/8=1, N=8]. Column n stores 8
        // nibbles for inputs k=0..7, in sequential order.
        //
        // Use a unique value per (k, n): nib(k, n) = (k + 1) (mod 16),
        // independent of n. After repack, every GPTQ word equals
        // sum_i (i+1) << (i*4).
        let device = Device::Cpu;
        let k = 8usize;
        let n = 8usize;
        let mut awq_words = Vec::with_capacity(k);
        for ki in 0..k {
            let v = ((ki + 1) & 0xF) as u32;
            // All 8 output lanes hold the same value v, so the AWQ word
            // is v repeated in every nibble — identical regardless of
            // undo_pack order.
            let mut word: u32 = 0;
            for shift in (0..32).step_by(4) {
                word |= v << shift;
            }
            awq_words.push(word);
        }
        let awq = Tensor::from_vec(awq_words, (k, n / 8), &device).unwrap();
        let gptq = awq_to_gptq_qweight(&awq, k, n).unwrap();
        let out: Vec<u32> = gptq.flatten_all().unwrap().to_vec1().unwrap();

        let mut expected_word: u32 = 0;
        for ki in 0..k {
            let v = ((ki + 1) & 0xF) as u32;
            expected_word |= v << (ki * 4);
        }
        assert_eq!(out.len(), n);
        for &w in &out {
            assert_eq!(w, expected_word);
        }
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
