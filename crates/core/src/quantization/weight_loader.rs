//! Weight loaders for different quantization methods.
//!
//! This module provides traits and implementations for loading quantized
//! weights from HuggingFace model checkpoints (safetensors format).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// Request a tensor from a VarBuilder with an explicit storage dtype,
/// bypassing the VarBuilder's default activation dtype. Used for loading
/// packed integer tensors (GPTQ qweight, AWQ qweight/qzeros, etc.) that
/// must keep their raw bit pattern even when the model's activation
/// dtype is F16/BF16/F32.
///
/// candle 0.10 routes the on-disk-dtype → activation-dtype → target-dtype
/// chain through CUDA cast kernels. The integer-cast variants
/// (e.g. I32→U32) hit `CUDA_ERROR_NOT_FOUND, "named symbol not found"`
/// because we don't ship those kernels — and the older "via BF16" cast
/// path corrupts packed nibbles. Workaround: when the VarBuilder is on a
/// CUDA device and we want an integer dtype, clone the VarBuilder onto
/// `Device::Cpu` (cheap — backend is reference-counted), fetch there
/// (CPU casts are bitwise / native), then move the result back to GPU.
/// This is one-shot at load time. See Stage 13-H follow-up notes.
fn vb_get_as<S: Into<candle_core::Shape>>(
    vb: &VarBuilder<'static>,
    shape: S,
    name: &str,
    dtype: DType,
) -> Result<Tensor> {
    let dev = vb.device().clone();
    let needs_cpu_workaround =
        dev.is_cuda() && matches!(dtype, DType::U8 | DType::U32 | DType::I64);
    if !needs_cpu_workaround {
        return vb.get_with_hints_dtype(shape, name, Init::Const(0.0), dtype);
    }
    let cpu_vb = vb.clone().set_device(Device::Cpu);
    let t_cpu = cpu_vb.get_with_hints_dtype(shape, name, Init::Const(0.0), dtype)?;
    t_cpu.to_device(&dev)
}

use super::awq::{AwqConfig, AwqLinear};
use super::awq_marlin::{repack_awq_nibbles, AwqMarlinConfig, AwqMarlinLinear};
use super::bitsandbytes::{BitsAndBytesConfig, BitsAndBytesLinear, BnbQuantType};
use super::config::{QuantizationConfig, QuantizedLinear};
use super::exl3::{Exl3Config, Exl3Linear};
use super::fbgemm_fp8::{FbgemmFp8Config, FbgemmFp8Linear};
use super::fp8::{Fp8Config, Fp8Linear};
use super::gptq::{GptqConfig, GptqLinear};
use super::marlin::MarlinLinear;
use super::mxfp4::{MxFp4Config, MxFp4Linear, MXFP4_BLOCK_SIZE};
use super::mxfp8::{MxFp8Config, MxFp8Linear, MXFP8_BLOCK_SIZE};
use super::{create_config_from_directory, DetectedQuantConfig, QuantizationMethod};

/// Trait for loading quantized weights from safetensors.
///
/// Each quantization method has a specific weight naming convention:
/// - Unquantized: `{prefix}.weight`, `{prefix}.bias`
/// - GPTQ: `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`, `{prefix}.g_idx`
/// - FP8: `{prefix}.weight` (U8), `{prefix}.weight_scale`, `{prefix}.input_scale`
/// - AWQ: `{prefix}.qweight`, `{prefix}.scales`, `{prefix}.qzeros`
pub trait QuantizedWeightLoader: Send + Sync {
    /// Load a quantized linear layer from the checkpoint.
    ///
    /// # Arguments
    /// * `prefix` - Weight name prefix (e.g., "model.layers.0.self_attn.q_proj")
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `bias` - Whether the layer has bias
    ///
    /// # Returns
    /// A boxed `QuantizedLinear` implementation with weights loaded
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>>;

    /// Get the quantization method this loader handles.
    fn method(&self) -> QuantizationMethod;

    /// Get the device for this loader.
    fn device(&self) -> &Device;

    /// Get the compute dtype for activations.
    fn dtype(&self) -> DType;
}

/// Weight loader for unquantized (full precision) models.
pub struct UnquantizedWeightLoader {
    vb: VarBuilder<'static>,
    device: Device,
    dtype: DType,
}

impl UnquantizedWeightLoader {
    /// Create a new unquantized weight loader.
    pub fn new(vb: VarBuilder<'static>) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self { vb, device, dtype }
    }
}

impl QuantizedWeightLoader for UnquantizedWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // Load weight tensor
        let weight = vb.get((out_features, in_features), "weight")?;

        // Optionally load bias
        let bias_tensor = if bias {
            Some(vb.get(out_features, "bias")?)
        } else {
            None
        };

        Ok(Box::new(LoadedUnquantizedLinear {
            weight,
            bias: bias_tensor,
            in_features,
            out_features,
        }))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::None
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Unquantized linear layer with pre-loaded weights.
#[derive(Debug)]
struct LoadedUnquantizedLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl QuantizedLinear for LoadedUnquantizedLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 2026-05-09: for 3D `[B, S, H]` inputs, FLATTEN to 2D and do a
        // single GEMM instead of `weight.broadcast_left(B) → batched
        // matmul with stride_b=0`. cuBLAS picks a markedly better
        // algorithm for the 2D path on small-M, large-N shapes (e.g.
        // an unquantized lm_head). +32 % e2e at c=8 on Qwen3-4B-AWQ
        // in side-by-side bench (118.8 → 157.1 tps); see
        // `docs/perf/2026-05-09-lm-head-flatten-win.md`.
        match x.dims().len() {
            3 => {
                let dims = x.dims();
                let (b, s, h) = (dims[0], dims[1], dims[2]);
                let x_flat = x.reshape((b * s, h))?;
                let y_flat = x_flat.matmul(&self.weight.t()?)?;
                let y = y_flat.reshape((b, s, self.out_features))?;
                match &self.bias {
                    Some(bias) => y.broadcast_add(bias),
                    None => Ok(y),
                }
            }
            _ => {
                let y = x.matmul(&self.weight.t()?)?;
                match &self.bias {
                    Some(bias) => y.broadcast_add(bias),
                    None => Ok(y),
                }
            }
        }
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        // Weights already loaded during construction
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
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

/// Weight loader for GPTQ quantized models.
pub struct GptqWeightLoader {
    vb: VarBuilder<'static>,
    config: GptqConfig,
    device: Device,
    dtype: DType,
}

impl GptqWeightLoader {
    /// Create a new GPTQ weight loader.
    pub fn new(vb: VarBuilder<'static>, config: GptqConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for GptqWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // Create GPTQ linear layer
        let mut linear = GptqLinear::new(
            in_features,
            out_features,
            bias,
            self.config.bits,
            self.config.group_size,
            &self.device,
        )?;

        // Calculate packed dimensions for GPTQ
        let pack_factor = 32 / self.config.bits as usize;
        let packed_in = in_features.div_ceil(pack_factor);
        let num_groups = if self.config.group_size <= 0 {
            1
        } else {
            in_features.div_ceil(self.config.group_size as usize)
        };

        // Load quantized weights
        let mut weights = HashMap::new();

        // Try to load qweight - GPTQ stores as (in_features/pack_factor, out_features)
        if let Ok(qweight) = vb.get((packed_in, out_features), "qweight") {
            weights.insert("qweight".to_string(), qweight);
        }

        // Load scales
        if let Ok(scales) = vb.get((num_groups, out_features), "scales") {
            weights.insert("scales".to_string(), scales);
        }

        // Load qzeros - packed format
        let packed_out = out_features.div_ceil(pack_factor);
        if let Ok(qzeros) = vb.get((num_groups, packed_out), "qzeros") {
            weights.insert("qzeros".to_string(), qzeros);
        }

        // Optional g_idx for desc_act
        if self.config.desc_act {
            if let Ok(g_idx) = vb.get(in_features, "g_idx") {
                weights.insert("g_idx".to_string(), g_idx);
            }
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gptq
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for FP8 quantized models.
pub struct Fp8WeightLoader {
    vb: VarBuilder<'static>,
    config: Fp8Config,
    device: Device,
    dtype: DType,
}

impl Fp8WeightLoader {
    /// Create a new FP8 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: Fp8Config) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for Fp8WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Check if layer should be skipped (not quantized)
        if self.config.is_layer_skipped(prefix) {
            // Fall back to unquantized
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);

        // Create FP8 linear layer
        let mut linear = Fp8Linear::new(
            in_features,
            out_features,
            bias,
            self.config.activation_scheme,
            self.config.weight_block_size,
            &self.device,
        )?;

        let mut weights = HashMap::new();

        // FP8 weights are stored as U8 (E4M3 format) - (out_features, in_features)
        if let Ok(weight) = vb.get((out_features, in_features), "weight") {
            weights.insert("weight".to_string(), weight);
        }

        // Weight scale for dequantization
        if let Ok(scale) = vb.get((), "weight_scale") {
            weights.insert("weight_scale".to_string(), scale);
        } else if let Ok(scale) = vb.get(1, "weight_scale") {
            // Per-tensor scale stored as 1D
            weights.insert("weight_scale".to_string(), scale.squeeze(0)?);
        }

        // Input scale for static quantization
        if let Ok(scale) = vb.get((), "input_scale") {
            weights.insert("input_scale".to_string(), scale);
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Fp8
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for EXL3 (ExLlamaV3) quantized models.
///
/// Phase-1 stub: `load_linear` parses the on-disk tensors and produces an
/// `Exl3Linear` whose `forward` errors out. Phase 4 lands the GEMM/Hadamard
/// kernels and unblocks real inference. We still load the tensors here so
/// Phase-5 model wiring can be exercised on CPU without crashing on
/// missing keys.
pub struct Exl3WeightLoader {
    vb: VarBuilder<'static>,
    config: Exl3Config,
    device: Device,
    dtype: DType,
}

impl Exl3WeightLoader {
    pub fn new(vb: VarBuilder<'static>, config: Exl3Config) -> Self {
        let device = vb.device().clone();
        // EXL3 forces fp16 on the activation path regardless of what the
        // VarBuilder advertises — backbone weights (embed/lm_head/norm)
        // are also fp16 in published EXL3 checkpoints.
        Self {
            vb,
            config,
            device,
            dtype: DType::F16,
        }
    }

    pub fn config(&self) -> &Exl3Config {
        &self.config
    }
}

impl QuantizedWeightLoader for Exl3WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let vb = self.vb.pp(prefix);

        // `trellis` shape is `(k/16, n/16, 16*K)` with k = in_features,
        // n = out_features, K = bpw. We don't know the exact bpw for this
        // layer until we read the tensor's last dim — `bits_per_weight`
        // in the config is the dominant value, not authoritative per-layer.
        // The on-disk dtype is int16; candle 0.10.2 has native I16 which
        // preserves byte layout (each i16 is 2 bytes on GPU as well), so
        // we can pass the slice directly to the kernel which interprets
        // it as `uint16_t*` — the bit pattern is identical.
        let k_blocks = in_features / 16;
        let n_blocks = out_features / 16;
        let last_dim = 16 * self.config.bits_per_weight as usize;
        let trellis = vb.get_with_hints_dtype(
            (k_blocks, n_blocks, last_dim),
            "trellis",
            Init::Const(0.0),
            DType::I16,
        )?;
        // Bits-per-weight from the actual tensor shape, in case the config
        // value disagrees.
        let bits_per_weight = (trellis.dims()[2] / 16) as u32;

        let suh = vb.get_with_hints_dtype(in_features, "suh", Init::Const(0.0), DType::F16)?;
        let svh = vb.get_with_hints_dtype(out_features, "svh", Init::Const(0.0), DType::F16)?;

        let bias_tensor = if bias {
            Some(vb.get_with_hints_dtype(out_features, "bias", Init::Const(0.0), DType::F16)?)
        } else {
            None
        };

        Ok(Box::new(Exl3Linear {
            trellis,
            suh,
            svh,
            mcg: self.config.mcg_default,
            mul1: self.config.mul1_default,
            bias: bias_tensor,
            in_features,
            out_features,
            bits_per_weight,
        }))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Exl3
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for AWQ quantized models.
pub struct AwqWeightLoader {
    vb: VarBuilder<'static>,
    config: AwqConfig,
    device: Device,
    dtype: DType,
}

impl AwqWeightLoader {
    /// Create a new AWQ weight loader.
    pub fn new(vb: VarBuilder<'static>, config: AwqConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for AwqWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Respect `modules_to_not_convert` — HF AWQ checkpoints leave
        // certain layers (`lm_head` by default) in plain fp16/bf16
        // instead of packing them. Fall back to the unquantized loader
        // for those paths.
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);

        // HuggingFace AWQ gemm format (this is the ONLY native layout vLLM
        // publishes for AWQ checkpoints): qweight is packed along the
        // output axis, shape `[in_features, out_features / pack_factor]`,
        // stored as I32. The old code expected the GPTQ-style
        // `[in_features / pack_factor, out_features]` which silently
        // mismatched and left the layer with unquantized zeros.
        let pack_factor = 32 / self.config.bits as usize;
        let packed_out = out_features.div_ceil(pack_factor);
        let num_groups = if self.config.group_size <= 0 {
            1
        } else {
            in_features.div_ceil(self.config.group_size as usize)
        };

        let mut weights = HashMap::new();

        // `vb_get_as` overrides the VarBuilder's default activation
        // dtype so qweight/qzeros arrive as U32 (the raw packed bits)
        // instead of being numerically cast to the activation dtype.
        // Scales stay at F16 which is the HuggingFace-native dtype.
        let qweight =
            vb_get_as(&vb, (in_features, packed_out), "qweight", DType::U32).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "AWQ qweight load failed for {prefix} with expected shape \
                     [{in_features}, {packed_out}]: {e}"
                ))
            })?;
        weights.insert("qweight".to_string(), qweight);

        let scales =
            vb_get_as(&vb, (num_groups, out_features), "scales", DType::F16).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "AWQ scales load failed for {prefix} with expected shape \
                     [{num_groups}, {out_features}]: {e}"
                ))
            })?;
        weights.insert("scales".to_string(), scales);

        let qzeros =
            vb_get_as(&vb, (num_groups, packed_out), "qzeros", DType::U32).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "AWQ qzeros load failed for {prefix} with expected shape \
                     [{num_groups}, {packed_out}]: {e}"
                ))
            })?;
        weights.insert("qzeros".to_string(), qzeros);

        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        // Route through `AwqMarlinLinear` when the AWQ config and tensor
        // shapes both clear the Marlin gate. `AwqConfig::create_linear`
        // applies the same gate, but the production loader path does not
        // go through `create_linear` — without this hook, every AWQ run
        // silently falls back to the CPU dequant fast path of `AwqLinear`,
        // which is ~1000× slower than the Marlin INT4 / decode-GEMV
        // kernels we ship.
        if self.config.use_marlin
            && self
                .config
                .can_use_marlin_for_shape(in_features, out_features)
        {
            if let Some(marlin_config) = self.config.to_marlin_config() {
                let mut marlin_linear = AwqMarlinLinear::new(
                    in_features,
                    out_features,
                    bias,
                    marlin_config,
                    &self.device,
                )?;
                marlin_linear.load_weights(&weights)?;
                return Ok(Box::new(marlin_linear));
            }
        }

        let mut linear = AwqLinear::new(
            in_features,
            out_features,
            bias,
            self.config.bits,
            self.config.group_size,
            &self.device,
        )?;
        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Awq
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for BitsAndBytes quantized models.
pub struct BitsAndBytesWeightLoader {
    vb: VarBuilder<'static>,
    config: BitsAndBytesConfig,
    device: Device,
    dtype: DType,
}

impl BitsAndBytesWeightLoader {
    /// Create a new BitsAndBytes weight loader.
    pub fn new(vb: VarBuilder<'static>, config: BitsAndBytesConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for BitsAndBytesWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Skip quantization for ignored layers
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);
        let total_elements = out_features * in_features;

        let mut linear = BitsAndBytesLinear::new(
            in_features,
            out_features,
            bias,
            self.config.quant_type,
            self.config.block_size,
            &self.device,
        )?;

        let mut weights = HashMap::new();

        match self.config.quant_type {
            BnbQuantType::NF4 => {
                // NF4: packed uint8. HF BnB checkpoints store the
                // packed tensor as 2D `[packed_len, 1]` with dtype U8;
                // fall back to flat `[packed_len]` for in-memory /
                // unit-test construction paths.
                let packed_len = total_elements.div_ceil(2);
                let weight_tensor = vb_get_as(&vb, (packed_len, 1), "weight", DType::U8)
                    .or_else(|_| vb_get_as(&vb, packed_len, "weight", DType::U8));
                let weight = weight_tensor.map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "BnB NF4 weight load failed for {prefix} (expected U8 shape \
                         [{packed_len}, 1] or [{packed_len}]): {e}"
                    ))
                })?;
                weights.insert("weight".to_string(), weight);

                // Per-block absmax. Double-quant checkpoints store
                // absmax as U8 (itself quantized) plus nested metadata;
                // single-quant stores it as F32 directly. Try U8 first
                // (the HF-native layout), fall back to F32 for legacy.
                let num_blocks = total_elements.div_ceil(self.config.block_size);
                let absmax_tensor = vb_get_as(&vb, num_blocks, "weight.absmax", DType::U8)
                    .or_else(|_| vb_get_as(&vb, num_blocks, "weight.absmax", DType::F32));
                let absmax = absmax_tensor.map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "BnB NF4 absmax load failed for {prefix} \
                         (expected shape [{num_blocks}]): {e}"
                    ))
                })?;
                weights.insert("absmax".to_string(), absmax);

                // Optional double-quant metadata — only present in
                // `bnb_4bit_use_double_quant=True` checkpoints.
                let nested_block_size = 256usize;
                let num_outer_blocks = num_blocks.div_ceil(nested_block_size);
                if let Ok(n) = vb_get_as(&vb, num_outer_blocks, "weight.nested_absmax", DType::F32)
                {
                    weights.insert("nested_absmax".to_string(), n);
                }
                if let Ok(m) = vb_get_as(&vb, 256, "weight.nested_quant_map", DType::F32) {
                    weights.insert("nested_quant_map".to_string(), m);
                }
                if let Ok(m) = vb_get_as(&vb, 16, "weight.quant_map", DType::F32) {
                    weights.insert("quant_map".to_string(), m);
                }
            }
            BnbQuantType::INT8 => {
                // INT8: shape [out_features, in_features]
                if let Ok(w) = vb.get((out_features, in_features), "weight") {
                    weights.insert("weight".to_string(), w);
                }
                // INT8 scales (SCB naming convention)
                if let Ok(s) = vb.get(out_features, "weight.SCB") {
                    weights.insert("SCB".to_string(), s);
                }
            }
        }

        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::BitsAndBytes
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for MXFP8 (ModelOpt) quantized models.
pub struct MxFp8WeightLoader {
    vb: VarBuilder<'static>,
    config: MxFp8Config,
    device: Device,
    dtype: DType,
}

impl MxFp8WeightLoader {
    /// Create a new MXFP8 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: MxFp8Config) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for MxFp8WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Skip quantization for ignored layers
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        // Validate K divisibility by block size
        if !in_features.is_multiple_of(MXFP8_BLOCK_SIZE) {
            candle_core::bail!(
                "MXFP8 requires in_features ({}) divisible by block_size ({})",
                in_features,
                MXFP8_BLOCK_SIZE,
            );
        }

        let vb = self.vb.pp(prefix);
        let mut linear = MxFp8Linear::new(in_features, out_features, bias, &self.device)?;

        let mut weights = HashMap::new();

        // MXFP8 weights: U8 (FP8 E4M3), shape [out_features, in_features]
        if let Ok(weight) = vb.get((out_features, in_features), "weight") {
            weights.insert("weight".to_string(), weight);
        }

        // E8M0 block scales: U8, shape [out_features, in_features / 32]
        let num_blocks = in_features / MXFP8_BLOCK_SIZE;
        if let Ok(scale) = vb.get((out_features, num_blocks), "weight_scale") {
            weights.insert("weight_scale".to_string(), scale);
        }

        // Optional bias
        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::ModelOpt
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Weight loader for compressed-tensors models.
///
/// Delegates to the appropriate inner loader (FP8, GPTQ, INT8) based on
/// the scheme resolved from the model's quantization config.
pub struct CompressedTensorsWeightLoader {
    /// Inner weight loader determined by the resolved scheme
    inner: Box<dyn QuantizedWeightLoader>,
    config: super::compressed_tensors::CompressedTensorsConfig,
}

impl CompressedTensorsWeightLoader {
    /// Create a new compressed-tensors weight loader.
    pub fn new(
        vb: VarBuilder<'static>,
        config: super::compressed_tensors::CompressedTensorsConfig,
    ) -> Self {
        // Determine the inner loader based on the default scheme.
        // The config's create_linear already delegates appropriately,
        // but we also need to load weights in the right format.
        let inner: Box<dyn QuantizedWeightLoader> = {
            let quant_config: Box<dyn QuantizationConfig> = Box::new(config.clone());
            let min_cap = quant_config.min_capability();

            if min_cap >= 89 {
                // FP8 scheme: use Fp8WeightLoader
                let fp8_config = Fp8Config::dynamic();
                Box::new(Fp8WeightLoader::new(vb, fp8_config))
            } else if min_cap >= 75 {
                // GPTQ/Marlin or INT8: use GptqWeightLoader as default
                let gptq_config = GptqConfig::int4(128);
                Box::new(GptqWeightLoader::new(vb.clone(), gptq_config))
            } else {
                Box::new(UnquantizedWeightLoader::new(vb))
            }
        };

        Self { inner, config }
    }
}

impl QuantizedWeightLoader for CompressedTensorsWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        if self.config.is_layer_skipped(prefix) {
            // Unquantized fallback for skipped layers
            return self
                .inner
                .load_linear(prefix, in_features, out_features, bias);
        }
        self.inner
            .load_linear(prefix, in_features, out_features, bias)
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::CompressedTensors
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
}

// ─── ExpertsInt8 Weight Loader ─────────────────────────────────────────────

/// Weight loader for ExpertsInt8 quantization.
///
/// Non-MoE layers are loaded as unquantized. MoE expert layers
/// are loaded as FP16/BF16 and quantized to INT8 online via
/// `ExpertsInt8Linear::load_weights()`.
pub struct ExpertsInt8WeightLoader {
    inner: UnquantizedWeightLoader,
    config: super::experts_int8::ExpertsInt8Config,
}

impl ExpertsInt8WeightLoader {
    /// Create a new ExpertsInt8 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: super::experts_int8::ExpertsInt8Config) -> Self {
        let inner = UnquantizedWeightLoader::new(vb);
        Self { inner, config }
    }

    /// Create an empty INT8 linear for MoE expert weights.
    ///
    /// Returns an `ExpertsInt8Linear` initialized with zero weights.
    /// Model code should call `load_weights()` with the FP16/BF16 weight
    /// tensor from VarBuilder — this triggers online INT8 quantization.
    pub fn create_int8_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        self.config
            .create_int8_linear(in_features, out_features, bias, self.inner.device())
    }
}

impl QuantizedWeightLoader for ExpertsInt8WeightLoader {
    /// Load a linear layer. For ExpertsInt8, non-MoE layers return unquantized.
    /// Models should call `load_int8_linear()` for MoE expert layers.
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        self.inner
            .load_linear(prefix, in_features, out_features, bias)
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::ExpertsInt8
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
}

// ─── MoeWNA16 Weight Loader ───────────────────────────────────────────────

/// Weight loader for MoeWNA16 quantization.
///
/// Non-MoE layers are loaded as unquantized. MoE expert layers
/// are loaded via GPTQ weight loader (packed INT4/INT8 format).
pub struct MoeWNA16WeightLoader {
    /// Unquantized loader for non-MoE layers
    unquant_inner: UnquantizedWeightLoader,
    /// GPTQ loader for MoE expert layers
    gptq_inner: GptqWeightLoader,
    #[allow(dead_code)] // Used by tests / future code
    config: super::moe_wna16::MoeWNA16Config,
}

impl MoeWNA16WeightLoader {
    /// Create a new MoeWNA16 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: super::moe_wna16::MoeWNA16Config) -> Self {
        let gptq_config = GptqConfig::from_detected(
            Some(config.weight_bits()),
            Some(config.group_size()),
            Some(false),
            &HashMap::new(),
        );
        let unquant_inner = UnquantizedWeightLoader::new(vb.clone());
        let gptq_inner = GptqWeightLoader::new(vb, gptq_config);
        Self {
            unquant_inner,
            gptq_inner,
            config,
        }
    }

    /// Load a GPTQ-quantized linear for MoE expert weights.
    pub fn load_expert_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        self.gptq_inner
            .load_linear(prefix, in_features, out_features, bias)
    }
}

impl QuantizedWeightLoader for MoeWNA16WeightLoader {
    /// Load a linear layer. For MoeWNA16, non-MoE layers return unquantized.
    /// Models should call `load_expert_linear()` for MoE expert layers.
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        self.unquant_inner
            .load_linear(prefix, in_features, out_features, bias)
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::MoeWNA16
    }

    fn device(&self) -> &Device {
        self.unquant_inner.device()
    }

    fn dtype(&self) -> DType {
        self.unquant_inner.dtype()
    }
}

// ─── AwqMarlin Weight Loader ──────────────────────────────────────────────────

/// Weight loader for AWQ-Marlin quantized models.
///
/// Loads AWQ checkpoint tensors (`qweight`, `scales`, `qzeros`) and
/// deinterleaves AWQ's interleaved nibble packing into GPTQ sequential order
/// before routing to `MarlinLinear`.  The Marlin GPTQ→tile repack happens
/// inside `MarlinLinear::process_weights()`.
pub struct AwqMarlinWeightLoader {
    vb: VarBuilder<'static>,
    config: AwqMarlinConfig,
    device: Device,
    dtype: DType,
}

impl AwqMarlinWeightLoader {
    /// Create a new AWQ-Marlin weight loader.
    pub fn new(vb: VarBuilder<'static>, config: AwqMarlinConfig) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for AwqMarlinWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Skip lm_head unless explicitly quantized.
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);
        let mc = self.config.marlin_config();
        let pack_factor = 8usize; // 32 / 4 bits = 8 int4 values per u32
        let packed_in = in_features / pack_factor;
        let num_groups = mc.num_groups(in_features);
        let packed_out = out_features / pack_factor;

        let mut linear =
            MarlinLinear::new(in_features, out_features, bias, mc.clone(), &self.device)?;

        let mut weights = HashMap::new();

        // Load and deinterleave AWQ qweight → GPTQ nibble ordering.
        if let Ok(qw) = vb.get((packed_in, out_features), "qweight") {
            let reordered = repack_awq_nibbles(&qw)?;
            weights.insert("qweight".to_string(), reordered);
        }

        // Scales are already in the standard [num_groups, out_features] format.
        if let Ok(s) = vb.get((num_groups, out_features), "scales") {
            weights.insert("scales".to_string(), s);
        }

        // Zero points (AWQ packed format [num_groups, out_features/pack_factor]).
        // NOTE: For the Marlin CUDA kernel, qzeros need additional repacking to
        // Marlin format (`marlin_awq_repack_zp`).  This CPU path stores them as-is;
        // a GPU-accelerated conversion will be needed when the
        // `awq_marlin_repack_int4` PTX function is added to `marlin_gemm.cu`.
        if let Ok(z) = vb.get((num_groups, packed_out), "qzeros") {
            weights.insert("qzeros".to_string(), z);
        }

        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::AwqMarlin
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

// ─── FbgemmFp8 Weight Loader ──────────────────────────────────────────────────

/// Weight loader for FBGEMM FP8 quantized models.
///
/// Loads FP8 E4M3 weights (stored as U8) plus per-channel F32 scales
/// (`[out, 1]` or `[out]`) from the checkpoint.
pub struct FbgemmFp8WeightLoader {
    vb: VarBuilder<'static>,
    config: FbgemmFp8Config,
    device: Device,
    dtype: DType,
}

impl FbgemmFp8WeightLoader {
    /// Create a new FBGEMM FP8 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: FbgemmFp8Config) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for FbgemmFp8WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        // Fall back to unquantized for skipped layers.
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        let vb = self.vb.pp(prefix);
        let mut linear = FbgemmFp8Linear::new(
            in_features,
            out_features,
            bias,
            self.config.input_scale_ub,
            &self.device,
        )?;

        let mut weights = HashMap::new();

        // FP8 weight: U8 (E4M3), shape [out_features, in_features].
        if let Ok(w) = vb.get((out_features, in_features), "weight") {
            weights.insert("weight".to_string(), w);
        }

        // Per-channel scale: F32, shape [out_features, 1] or [out_features].
        // Try both shapes; the [out, 1] form is what HuggingFace Meta-Llama checkpoints use.
        if let Ok(s) = vb.get((out_features, 1usize), "weight_scale") {
            weights.insert("weight_scale".to_string(), s);
        } else if let Ok(s) = vb.get(out_features, "weight_scale") {
            weights.insert("weight_scale".to_string(), s);
        }

        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::FbgemmFp8
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

// ─── MxFp4 Weight Loader ──────────────────────────────────────────────────────

/// Weight loader for OCP Microscaling FP4 (MXFP4 E2M1) quantized models.
///
/// Weight tensor layout on disk:
///   - `weight`: `[out_features, in_features / 2]` U8 — two FP4 E2M1 nibbles per byte
///     (lower nibble = first element, upper nibble = second element).
///   - `weight_scale`: `[out_features, in_features / 32]` U8 — E8M0 block exponents
///     (one scale per 32-element K-dimension block, bias 127).
///   - `bias`: `[out_features]` BF16 — optional.
///
/// The CPU forward path dequantizes on every call (unpack → decode scales →
/// BF16 matmul).  GPU-accelerated dispatch is tracked separately as Phase 3.3.
pub struct MxFp4WeightLoader {
    vb: VarBuilder<'static>,
    config: MxFp4Config,
    device: Device,
    dtype: DType,
}

impl MxFp4WeightLoader {
    /// Create a new MXFP4 weight loader.
    pub fn new(vb: VarBuilder<'static>, config: MxFp4Config) -> Self {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        Self {
            vb,
            config,
            device,
            dtype,
        }
    }
}

impl QuantizedWeightLoader for MxFp4WeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        if self.config.is_layer_skipped(prefix) {
            let vb = self.vb.pp(prefix);
            let weight = vb.get((out_features, in_features), "weight")?;
            let bias_tensor = if bias {
                Some(vb.get(out_features, "bias")?)
            } else {
                None
            };
            return Ok(Box::new(LoadedUnquantizedLinear {
                weight,
                bias: bias_tensor,
                in_features,
                out_features,
            }));
        }

        if !in_features.is_multiple_of(MXFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "MXFP4 requires in_features ({}) divisible by block_size ({})",
                in_features,
                MXFP4_BLOCK_SIZE,
            );
        }

        let vb = self.vb.pp(prefix);
        let mut linear = MxFp4Linear::new(in_features, out_features, bias, &self.device)?;
        let mut weights = HashMap::new();

        // Packed FP4: [out_features, in_features / 2] U8
        let packed_k = in_features / 2;
        if let Ok(weight) = vb.get((out_features, packed_k), "weight") {
            weights.insert("weight".to_string(), weight);
        }

        // E8M0 block scales: [out_features, in_features / MXFP4_BLOCK_SIZE] U8
        let num_blocks = in_features / MXFP4_BLOCK_SIZE;
        if let Ok(scale) = vb.get((out_features, num_blocks), "weight_scale") {
            weights.insert("weight_scale".to_string(), scale);
        }

        if bias {
            if let Ok(b) = vb.get(out_features, "bias") {
                weights.insert("bias".to_string(), b);
            }
        }

        linear.load_weights(&weights)?;
        Ok(Box::new(linear))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Mxfp4
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Create an appropriate weight loader for a model directory.
///
/// This function detects the quantization method from the model config
/// and returns the corresponding weight loader.
pub fn create_weight_loader(
    model_dir: &Path,
    vb: VarBuilder<'static>,
) -> Box<dyn QuantizedWeightLoader> {
    let config = create_config_from_directory(model_dir);
    create_weight_loader_from_config(vb, config)
}

/// Create a weight loader from a detected quantization config.
pub fn create_weight_loader_from_detected(
    vb: VarBuilder<'static>,
    detected: &DetectedQuantConfig,
) -> Box<dyn QuantizedWeightLoader> {
    let config = super::create_config(detected);
    create_weight_loader_from_config(vb, config)
}

/// Create a weight loader from a quantization config.
pub fn create_weight_loader_from_config(
    vb: VarBuilder<'static>,
    config: Box<dyn QuantizationConfig>,
) -> Box<dyn QuantizedWeightLoader> {
    match config.method() {
        QuantizationMethod::Gptq => {
            // Downcast to GptqConfig
            let gptq_config = GptqConfig::from_detected(None, None, None, &HashMap::new());
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        QuantizationMethod::Fp8 => {
            let fp8_config = Fp8Config::default();
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        QuantizationMethod::Awq => {
            let awq_config = AwqConfig::default();
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        QuantizationMethod::BitsAndBytes => {
            let bnb_config = BitsAndBytesConfig::default();
            Box::new(BitsAndBytesWeightLoader::new(vb, bnb_config))
        }
        QuantizationMethod::ModelOpt => {
            let mxfp8_config = MxFp8Config::default();
            Box::new(MxFp8WeightLoader::new(vb, mxfp8_config))
        }
        QuantizationMethod::CompressedTensors => {
            let ct_config =
                super::compressed_tensors::CompressedTensorsConfig::from_detected(&HashMap::new());
            Box::new(CompressedTensorsWeightLoader::new(vb, ct_config))
        }
        QuantizationMethod::ExpertsInt8 => {
            let ei8_config = super::experts_int8::ExpertsInt8Config::from_detected(&HashMap::new());
            Box::new(ExpertsInt8WeightLoader::new(vb, ei8_config))
        }
        QuantizationMethod::MoeWNA16 => {
            let wna16_config = super::moe_wna16::MoeWNA16Config::from_detected(&HashMap::new());
            Box::new(MoeWNA16WeightLoader::new(vb, wna16_config))
        }
        QuantizationMethod::AwqMarlin => {
            let awq_marlin_config = AwqMarlinConfig::from_detected(None, None, &HashMap::new());
            Box::new(AwqMarlinWeightLoader::new(vb, awq_marlin_config))
        }
        QuantizationMethod::FbgemmFp8 => {
            let cfg = FbgemmFp8Config::from_detected(&HashMap::new());
            Box::new(FbgemmFp8WeightLoader::new(vb, cfg))
        }
        QuantizationMethod::Mxfp4 => {
            let cfg = MxFp4Config::from_detected(&HashMap::new());
            Box::new(MxFp4WeightLoader::new(vb, cfg))
        }
        QuantizationMethod::PtpcFp8 => {
            // Same weights as standard FP8 — reuse Fp8WeightLoader.
            let fp8_config = Fp8Config::dynamic();
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        // CPU AWQ has the same packed-INT4 weight format as standard AWQ.
        QuantizationMethod::CpuWna16 => {
            let awq_config = AwqConfig::default();
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        // INC defaults to GPTQ packing; AWQ packing cannot be distinguished at
        // this point without the raw config, so fall back to GPTQ layout.
        QuantizationMethod::Inc => {
            let gptq_config = GptqConfig::from_detected(None, None, None, &HashMap::new());
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        // Standalone Marlin — same layout as GPTQ, the Marlin kernel path
        // in `GptqLinear` engages automatically on supported hardware.
        QuantizationMethod::Marlin => {
            let gptq_config = GptqConfig::from_detected(None, None, None, &HashMap::new());
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        other => {
            if !matches!(other, QuantizationMethod::None) {
                tracing::warn!(
                    method = %other,
                    "no production weight loader for quantization method; falling back to unquantized — \
                     weights will NOT be dequantized"
                );
            }
            Box::new(UnquantizedWeightLoader::new(vb))
        }
    }
}

/// Create a weight loader with explicit config parameters.
pub fn create_weight_loader_with_params(
    vb: VarBuilder<'static>,
    detected: &DetectedQuantConfig,
) -> Box<dyn QuantizedWeightLoader> {
    match detected.method {
        QuantizationMethod::Gptq => {
            let gptq_config = GptqConfig::from_detected(
                detected.bits,
                detected.group_size,
                detected.desc_act,
                &detected.raw_config,
            );
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        QuantizationMethod::Fp8 => {
            let fp8_config = Fp8Config::from_detected(
                detected.bits,
                detected.activation_scheme.as_deref(),
                &detected.raw_config,
            );
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        QuantizationMethod::Awq => {
            let awq_config =
                AwqConfig::from_detected(detected.bits, detected.group_size, &detected.raw_config);
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        QuantizationMethod::BitsAndBytes => {
            let bnb_config = BitsAndBytesConfig::from_detected(&detected.raw_config);
            Box::new(BitsAndBytesWeightLoader::new(vb, bnb_config))
        }
        QuantizationMethod::ModelOpt => {
            let mxfp8_config = MxFp8Config::from_detected(&detected.raw_config);
            Box::new(MxFp8WeightLoader::new(vb, mxfp8_config))
        }
        QuantizationMethod::CompressedTensors => {
            let ct_config = super::compressed_tensors::CompressedTensorsConfig::from_detected(
                &detected.raw_config,
            );
            Box::new(CompressedTensorsWeightLoader::new(vb, ct_config))
        }
        QuantizationMethod::ExpertsInt8 => {
            let ei8_config =
                super::experts_int8::ExpertsInt8Config::from_detected(&detected.raw_config);
            Box::new(ExpertsInt8WeightLoader::new(vb, ei8_config))
        }
        QuantizationMethod::MoeWNA16 => {
            let wna16_config =
                super::moe_wna16::MoeWNA16Config::from_detected(&detected.raw_config);
            Box::new(MoeWNA16WeightLoader::new(vb, wna16_config))
        }
        QuantizationMethod::AwqMarlin => {
            let awq_marlin_config = AwqMarlinConfig::from_detected(
                detected.bits,
                detected.group_size,
                &detected.raw_config,
            );
            Box::new(AwqMarlinWeightLoader::new(vb, awq_marlin_config))
        }
        QuantizationMethod::FbgemmFp8 => {
            let cfg = FbgemmFp8Config::from_detected(&detected.raw_config);
            Box::new(FbgemmFp8WeightLoader::new(vb, cfg))
        }
        QuantizationMethod::Mxfp4 => {
            let cfg = MxFp4Config::from_detected(&detected.raw_config);
            Box::new(MxFp4WeightLoader::new(vb, cfg))
        }
        QuantizationMethod::PtpcFp8 => {
            // Same weight format as Fp8 — dynamic, no serialised FP8.
            let fp8_config = Fp8Config::from_detected(
                detected.bits,
                detected.activation_scheme.as_deref(),
                &detected.raw_config,
            );
            Box::new(Fp8WeightLoader::new(vb, fp8_config))
        }
        // CPU AWQ shares the packed-INT4 layout with standard AWQ.
        QuantizationMethod::CpuWna16 => {
            let awq_config =
                AwqConfig::from_detected(detected.bits, detected.group_size, &detected.raw_config);
            Box::new(AwqWeightLoader::new(vb, awq_config))
        }
        // INC (Intel Neural Compressor) defaults to GPTQ packing format.
        // The `packing_format` field lives in `detected.raw_config`; we
        // route through GPTQ which covers the common case and lets
        // Marlin engage for INT4 weights automatically.
        QuantizationMethod::Inc => {
            let gptq_config = GptqConfig::from_detected(
                detected.bits,
                detected.group_size,
                detected.desc_act,
                &detected.raw_config,
            );
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        // Standalone Marlin checkpoints (`gptq_marlin` in config.json) use
        // the GPTQ layout with Marlin upgrade. Route through GPTQ so the
        // Marlin kernel path kicks in when the hardware supports it.
        QuantizationMethod::Marlin => {
            let gptq_config = GptqConfig::from_detected(
                detected.bits,
                detected.group_size,
                detected.desc_act,
                &detected.raw_config,
            );
            Box::new(GptqWeightLoader::new(vb, gptq_config))
        }
        QuantizationMethod::Exl3 => {
            let exl3_config = Exl3Config::from_detected(&detected.raw_config);
            Box::new(Exl3WeightLoader::new(vb, exl3_config))
        }
        other => {
            // Anything else (Gguf / Torchao / Quark / FpQuant / ModelOptFull
            // / SqueezeLlm / None) silently returns an unquantized loader.
            // Log a loud warning for non-`None` fall-through so users notice
            // their quant is being downgraded to zero-weight unquantized.
            if !matches!(other, QuantizationMethod::None) {
                tracing::warn!(
                    method = %other,
                    "no production weight loader for quantization method; falling back to unquantized — \
                     weights will NOT be dequantized. Add an explicit dispatch arm or pick a supported method."
                );
            }
            Box::new(UnquantizedWeightLoader::new(vb))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unquantized_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = UnquantizedWeightLoader::new(vb);
        assert_eq!(loader.method(), QuantizationMethod::None);
    }

    #[test]
    fn test_gptq_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = GptqConfig::int4(128);
        let loader = GptqWeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_fp8_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = Fp8Config::dynamic();
        let loader = Fp8WeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_create_weight_loader_default() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_from_detected(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::None);
    }

    #[test]
    fn test_create_weight_loader_gptq() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_create_weight_loader_cpu_wna16_routes_to_awq() {
        // Drift-fix regression: `create_weight_loader_with_params` used
        // to silently fall through to unquantized on CpuWna16 even though
        // `from_config` routed it correctly.
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::CpuWna16,
            bits: Some(4),
            group_size: Some(128),
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_create_weight_loader_inc_routes_to_gptq() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Inc,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_create_weight_loader_marlin_routes_to_gptq() {
        // Standalone Marlin (`gptq_marlin` in config.json) uses the GPTQ
        // layout — routing through GPTQ lets the Marlin kernel engage
        // automatically on supported hardware.
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Marlin,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Gptq);
    }

    #[test]
    fn test_create_weight_loader_fp8() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Fp8,
            bits: Some(8),
            group_size: None,
            desc_act: None,
            activation_scheme: Some("dynamic".to_string()),
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Fp8);
    }

    #[test]
    fn test_awq_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = AwqConfig::int4(128);
        let loader = AwqWeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_create_weight_loader_awq() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Awq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Awq);
    }

    #[test]
    fn test_loaded_unquantized_linear_forward() {
        let device = Device::Cpu;
        let weight = Tensor::ones((8, 4), DType::F32, &device).unwrap();
        let linear = LoadedUnquantizedLinear {
            weight,
            bias: None,
            in_features: 4,
            out_features: 8,
        };

        let x = Tensor::ones((2, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
        assert_eq!(linear.in_features(), 4);
        assert_eq!(linear.out_features(), 8);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_loaded_unquantized_linear_with_bias() {
        let device = Device::Cpu;
        let weight = Tensor::ones((8, 4), DType::F32, &device).unwrap();
        let bias = Tensor::ones(8, DType::F32, &device).unwrap();
        let linear = LoadedUnquantizedLinear {
            weight,
            bias: Some(bias),
            in_features: 4,
            out_features: 8,
        };

        let x = Tensor::ones((2, 4), DType::F32, &device).unwrap();
        let y = linear.forward(&x).unwrap();

        assert_eq!(y.dims(), &[2, 8]);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_mxfp8_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let config = MxFp8Config::default();
        let loader = MxFp8WeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::ModelOpt);
    }

    #[test]
    fn test_create_weight_loader_mxfp8() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::ModelOpt,
            bits: None,
            group_size: None,
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::ModelOpt);
    }

    // ─── MxFp4 Weight Loader tests ───────────────────────────────────────────

    #[test]
    fn test_mxfp4_loader_method() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let config = MxFp4Config::default();
        let loader = MxFp4WeightLoader::new(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Mxfp4);
    }

    #[test]
    fn test_create_weight_loader_mxfp4_from_config() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let config: Box<dyn QuantizationConfig> = Box::new(MxFp4Config::default());
        let loader = create_weight_loader_from_config(vb, config);
        assert_eq!(loader.method(), QuantizationMethod::Mxfp4);
    }

    #[test]
    fn test_create_weight_loader_mxfp4_with_params() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Mxfp4,
            bits: Some(4),
            group_size: Some(32),
            desc_act: None,
            activation_scheme: None,
            raw_config: HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb, &detected);
        assert_eq!(loader.method(), QuantizationMethod::Mxfp4);
    }

    #[test]
    fn test_mxfp4_loader_load_linear_zeros() {
        // Tests that load_linear succeeds when VarBuilder has no tensors —
        // the linear layer is created with zero-initialised placeholders.
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let config = MxFp4Config::default();
        let loader = MxFp4WeightLoader::new(vb, config);

        // in_features must be divisible by MXFP4_BLOCK_SIZE (32)
        let linear = loader.load_linear("layer", 64, 32, false).unwrap();
        assert_eq!(linear.in_features(), 64);
        assert_eq!(linear.out_features(), 32);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_mxfp4_loader_requires_divisible_in_features() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::BF16, &device);
        let config = MxFp4Config::default();
        let loader = MxFp4WeightLoader::new(vb, config);
        // in_features = 10 is not divisible by 32
        let result = loader.load_linear("layer", 10, 4, false);
        assert!(result.is_err());
    }
}
