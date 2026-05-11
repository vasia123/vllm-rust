//! EXL3 quantization (ExLlamaV3 trellis-coded vector quantization).
//!
//! EXL3 (turboderp/ExLlamaV3) is a QTIP-style trellis-coded vector
//! quantization scheme. Each linear layer stores:
//!
//! - `{prefix}.trellis` — `(k/16, n/16, 16*K)` uint16, the trellis codes.
//!   K = bits-per-weight ∈ {2,3,4,5,6,8}.
//! - `{prefix}.suh` or `{prefix}.su` — input Hadamard sign vector.
//!   `suh` is fp16 (`±1.0`); `su` is int16-packed bitfield to be unpacked.
//! - `{prefix}.svh` or `{prefix}.sv` — output Hadamard sign vector.
//! - `{prefix}.mcg` — optional uint32 flag (`0xCBAC1FED`) toggling the
//!   `mcg` codebook multiplier.
//! - `{prefix}.mul1` — optional uint32 flag (`0x83DCD12D`) toggling the
//!   `mul1` codebook multiplier.
//! - `{prefix}.bias` — optional fp16 bias.
//!
//! Forward pass:
//!   `y = had_r_128(input * suh) @ decode(trellis); y = had_r_128(y) * svh`
//!
//! Activation dtype is fp16 throughout.
//!
//! This module currently contains Phase-1 scaffolding only — the actual
//! kernels live in `crate::cuda_kernels::exl3` (Phase 3-4) and are not
//! yet wired up. `Exl3Linear::forward` will return an error until then.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use serde_json::Value;

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};

/// EXL3 codebook multipliers used by ExLlamaV3.
pub const EXL3_MCG_MULTIPLIER: u32 = 0xCBAC_1FED;
pub const EXL3_MUL1_MULTIPLIER: u32 = 0x83DC_D12D;

/// Per-tensor storage info as serialised in `quantization_config.tensor_storage`.
///
/// ExLlamaV3 emits this map so loaders can pre-allocate without parsing each
/// tensor header. We mirror only the fields we actually need at runtime.
#[derive(Debug, Clone, Default)]
pub struct Exl3TensorInfo {
    /// Original tensor shape `[out_features, in_features]` for linear weights.
    pub shape: Vec<usize>,
    /// On-disk dtype name (`"int16"`, `"float16"`, `"uint32"`).
    pub dtype: String,
}

/// EXL3 quantization configuration.
#[derive(Debug, Clone)]
pub struct Exl3Config {
    /// Bits per weight. ExLlamaV3 supports 2..=8.
    /// Determined per-tensor from `trellis.shape[-1] / 16`; the value in
    /// `Config` is the dominant bpw for sanity checks only.
    pub bits_per_weight: u32,
    /// True when `.mcg` flag tensors are present and the codebook multiplier
    /// `EXL3_MCG_MULTIPLIER` should be applied during trellis decode.
    pub mcg_default: bool,
    /// True when `.mul1` flag tensors are present.
    pub mul1_default: bool,
    /// Layer-name patterns to exclude from quantization (kept dense).
    pub ignored_layers: Vec<String>,
    /// Per-tensor storage hints from `quantization_config.tensor_storage`.
    /// Empty when the producer didn't emit it.
    pub tensor_storage: HashMap<String, Exl3TensorInfo>,
}

impl Exl3Config {
    /// Build a config from `quantization_config` raw JSON.
    pub fn from_detected(raw: &HashMap<String, Value>) -> Self {
        // EXL3 checkpoints encode `bits` as float (e.g. 3.0, 3.5, 4.0)
        // and `bits_per_weight` may be either int or absent. We round
        // down because the kernel selects K=floor(bits), with the
        // actual stored bit-width carried by the trellis tensor's
        // last dim (validated at load time).
        let read_numeric = |key: &str| -> Option<u32> {
            raw.get(key)
                .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)))
                .map(|n| n as u32)
        };
        let bits_per_weight = read_numeric("bits_per_weight")
            .or_else(|| read_numeric("bits"))
            .unwrap_or(4);

        let mcg_default = raw
            .get("mcg_multiplier")
            .and_then(|v| v.as_u64())
            .map(|m| m as u32 == EXL3_MCG_MULTIPLIER)
            .unwrap_or(false);
        let mul1_default = raw
            .get("mul1_multiplier")
            .and_then(|v| v.as_u64())
            .map(|m| m as u32 == EXL3_MUL1_MULTIPLIER)
            .unwrap_or(false);

        let ignored_layers = raw
            .get("modules_to_not_convert")
            .or_else(|| raw.get("ignored_layers"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let tensor_storage = raw
            .get("tensor_storage")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let info = v.as_object()?;
                        let shape = info
                            .get("shape")?
                            .as_array()?
                            .iter()
                            .filter_map(|x| x.as_u64().map(|n| n as usize))
                            .collect();
                        let dtype = info
                            .get("dtype")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string();
                        Some((k.clone(), Exl3TensorInfo { shape, dtype }))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            bits_per_weight,
            mcg_default,
            mul1_default,
            ignored_layers,
            tensor_storage,
        }
    }
}

impl Default for Exl3Config {
    fn default() -> Self {
        Self {
            bits_per_weight: 4,
            mcg_default: false,
            mul1_default: false,
            ignored_layers: Vec::new(),
            tensor_storage: HashMap::new(),
        }
    }
}

impl QuantizationConfig for Exl3Config {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Exl3
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        // EXL3 kernels are fp16-only. Backbone weights are loaded as fp16.
        &[DType::F16]
    }

    fn min_capability(&self) -> u32 {
        // ExLlamaV3 ships sm_80+ binaries.
        80
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.ignored_layers
            .iter()
            .any(|pat| layer_name.contains(pat))
    }

    fn create_linear(
        &self,
        _in_features: usize,
        _out_features: usize,
        _bias: bool,
        _device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Direct construction without on-disk tensors is not meaningful for
        // EXL3 — every parameter (trellis, suh, svh) must come from the
        // checkpoint. Use `Exl3WeightLoader::load_linear` instead.
        Err(candle_core::Error::Msg(
            "Exl3Config::create_linear: use Exl3WeightLoader::load_linear; \
             EXL3 weights must be loaded from a checkpoint."
                .to_string(),
        ))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// EXL3 quantized linear layer.
///
/// Forward (Phase 4b.5): `y = exl3_gemm(x, trellis, suh, svh)` — the
/// Hadamard transforms and trellis-decode GEMM are fused inside the
/// kernel; the dispatcher allocates output + A_had scratch + locks.
#[derive(Debug)]
pub struct Exl3Linear {
    pub trellis: Tensor,
    pub suh: Tensor,
    pub svh: Tensor,
    pub mcg: bool,
    pub mul1: bool,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
    pub bits_per_weight: u32,
}

impl QuantizedLinear for Exl3Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda-kernels")]
        {
            use super::exl3_cuda::{exl3_gemm, Exl3Codebook};

            // Accept fp16 or bf16/fp32 input; cast to fp16 since the
            // EXL3 kernels are fp16-only.
            let x_fp16 = if x.dtype() == DType::F16 {
                x.clone()
            } else {
                x.to_dtype(DType::F16)?
            };

            // Flatten leading dims to 2D [M, K]. Restore on the way out.
            let dims = x_fp16.dims().to_vec();
            let k = self.in_features;
            if *dims.last().unwrap_or(&0) != k {
                candle_core::bail!(
                    "Exl3Linear::forward: input last dim ({:?}) must equal in_features ({})",
                    dims.last(),
                    k
                );
            }
            let leading: usize = dims[..dims.len() - 1].iter().product();
            let x2 = if dims.len() == 2 {
                x_fp16
            } else {
                x_fp16.reshape((leading, k))?
            };

            let codebook = Exl3Codebook::from_flags(self.mcg, self.mul1);
            let y2 = exl3_gemm(
                &x2,
                &self.trellis,
                Some(&self.suh),
                Some(&self.svh),
                self.bits_per_weight,
                codebook,
            )?;

            // Restore the leading dims.
            let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_dims.push(self.out_features);
            let y = if dims.len() == 2 {
                y2
            } else {
                y2.reshape(out_dims.as_slice())?
            };

            // Bias addition. EXL3 bias is fp16.
            let y = match &self.bias {
                Some(b) => y.broadcast_add(b)?,
                None => y,
            };

            // Cast back to the activation dtype if it wasn't fp16.
            if x.dtype() != DType::F16 {
                y.to_dtype(x.dtype())
            } else {
                Ok(y)
            }
        }
        #[cfg(not(feature = "cuda-kernels"))]
        {
            let _ = x;
            Err(candle_core::Error::Msg(
                "Exl3Linear::forward requires the cuda-kernels feature".to_string(),
            ))
        }
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        // EXL3 tensors are loaded by `Exl3WeightLoader` in one shot; the
        // generic state-dict path is unused.
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        // Trellis is uint16 on disk but we report the activation dtype,
        // which is what callers expect from `weight_dtype` (used for
        // KV-cache / activation routing decisions).
        DType::F16
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_parses_minimum_fields() {
        let mut raw = HashMap::new();
        raw.insert("bits_per_weight".to_string(), json!(3));
        let cfg = Exl3Config::from_detected(&raw);
        assert_eq!(cfg.bits_per_weight, 3);
        assert!(!cfg.mcg_default);
        assert!(!cfg.mul1_default);
    }

    #[test]
    fn config_picks_up_codebook_flags() {
        let mut raw = HashMap::new();
        raw.insert("bits_per_weight".to_string(), json!(4));
        raw.insert(
            "mcg_multiplier".to_string(),
            json!(EXL3_MCG_MULTIPLIER as u64),
        );
        raw.insert(
            "mul1_multiplier".to_string(),
            json!(EXL3_MUL1_MULTIPLIER as u64),
        );
        let cfg = Exl3Config::from_detected(&raw);
        assert!(cfg.mcg_default);
        assert!(cfg.mul1_default);
    }

    #[test]
    fn config_parses_ignored_layers() {
        let mut raw = HashMap::new();
        raw.insert(
            "modules_to_not_convert".to_string(),
            json!(["lm_head", "embed_tokens"]),
        );
        let cfg = Exl3Config::from_detected(&raw);
        assert_eq!(cfg.ignored_layers.len(), 2);
        assert!(cfg.is_layer_skipped("model.lm_head"));
        assert!(cfg.is_layer_skipped("model.embed_tokens.weight"));
        assert!(!cfg.is_layer_skipped("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn config_method_and_dtypes() {
        let cfg = Exl3Config::default();
        assert_eq!(cfg.method(), QuantizationMethod::Exl3);
        assert_eq!(cfg.supported_act_dtypes(), &[DType::F16]);
        assert_eq!(cfg.min_capability(), 80);
    }

    #[test]
    fn create_linear_errors_with_helpful_message() {
        let cfg = Exl3Config::default();
        let err = cfg
            .create_linear(4096, 4096, false, &Device::Cpu)
            .err()
            .expect("EXL3 create_linear must require checkpoint tensors");
        assert!(format!("{err}").contains("Exl3WeightLoader"));
    }

    #[test]
    fn linear_forward_errors_on_cpu() {
        // Forward dispatches to the CUDA kernel; on CPU it must fail
        // gracefully (the underlying CustomOp1::cpu_fwd bails).
        let trellis = Tensor::zeros((1, 1, 64), DType::I16, &Device::Cpu).unwrap();
        let suh = Tensor::zeros(16, DType::F16, &Device::Cpu).unwrap();
        let svh = Tensor::zeros(16, DType::F16, &Device::Cpu).unwrap();
        let lin = Exl3Linear {
            trellis,
            suh,
            svh,
            mcg: false,
            mul1: false,
            bias: None,
            in_features: 16,
            out_features: 16,
            bits_per_weight: 4,
        };
        let x = Tensor::zeros((1, 16), DType::F16, &Device::Cpu).unwrap();
        assert!(lin.forward(&x).is_err());
    }

    #[test]
    fn linear_forward_validates_input_shape() {
        // Wrong K dim must error before we touch the kernel.
        let trellis = Tensor::zeros((1, 1, 64), DType::I16, &Device::Cpu).unwrap();
        let suh = Tensor::zeros(16, DType::F16, &Device::Cpu).unwrap();
        let svh = Tensor::zeros(16, DType::F16, &Device::Cpu).unwrap();
        let lin = Exl3Linear {
            trellis,
            suh,
            svh,
            mcg: false,
            mul1: false,
            bias: None,
            in_features: 16,
            out_features: 16,
            bits_per_weight: 4,
        };
        let x = Tensor::zeros((1, 32), DType::F16, &Device::Cpu).unwrap();
        let err = lin.forward(&x).err().unwrap();
        assert!(format!("{err}").contains("in_features"));
    }
}
