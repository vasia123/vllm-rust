//! Marlin optimized INT4 GPTQ kernel integration.
//!
//! Marlin is a highly optimized INT4 GPTQ kernel that provides 2-3x speedup
//! over standard dequantize+GEMM operations. Key features:
//! - Asynchronous global memory loads
//! - Optimized warp shuffles for data exchange
//! - Packed INT4 format with custom layout for efficient MMA operations
//!
//! # Weight Format
//!
//! Marlin uses a specialized weight packing format optimized for tensor core
//! operations:
//! - Weights are packed in tiles of 16x16 (GPTQ_MARLIN_TILE = 16)
//! - Scales and zero points are permuted for coalesced memory access
//! - Supports group sizes: -1 (per-channel), 32, 64, 128
//!
//! # Supported Types
//!
//! - INT4 symmetric (uint4b8): GPTQ-style with 8-offset bias
//! - INT8 symmetric (uint8b128): 128-offset bias
//! - FP8 (float8_e4m3fn): FP8 quantized weights
//!
//! # Requirements
//!
//! - GPU compute capability >= 8.0 (Ampere or newer)
//! - N dimension divisible by 64 (GPTQ_MARLIN_MIN_THREAD_N)
//! - K dimension divisible by 128 (GPTQ_MARLIN_MIN_THREAD_K)

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
#[cfg(feature = "marlin")]
use super::marlin_cuda;

// Marlin kernel constants from reference implementation
/// Tile size for Marlin kernel operations
pub const GPTQ_MARLIN_TILE: usize = 16;
/// Minimum thread tile size in N dimension
pub const GPTQ_MARLIN_MIN_THREAD_N: usize = 64;
/// Minimum thread tile size in K dimension
pub const GPTQ_MARLIN_MIN_THREAD_K: usize = 128;
/// Maximum parallel partitions
pub const GPTQ_MARLIN_MAX_PARALLEL: usize = 16;

/// Supported group sizes for Marlin kernel
pub const MARLIN_SUPPORTED_GROUP_SIZES: &[i32] = &[-1, 32, 64, 128];

/// Scalar types supported by Marlin
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarlinScalarType {
    /// 4-bit unsigned with 8-offset symmetric bias (GPTQ style)
    Uint4b8,
    /// 8-bit unsigned with 128-offset symmetric bias
    Uint8b128,
    /// 4-bit unsigned with runtime zero point (AWQ style)
    Uint4,
    /// 8-bit unsigned with runtime zero point
    Uint8,
    /// FP8 E4M3
    Float8E4m3fn,
}

impl MarlinScalarType {
    /// Get the number of bits for this type
    pub fn bits(&self) -> u32 {
        match self {
            Self::Uint4b8 | Self::Uint4 => 4,
            Self::Uint8b128 | Self::Uint8 => 8,
            Self::Float8E4m3fn => 8,
        }
    }

    /// Check if this type has a zero point
    pub fn has_zero_point(&self) -> bool {
        matches!(self, Self::Uint4 | Self::Uint8)
    }

    /// Get pack factor (elements per u32)
    pub fn pack_factor(&self) -> usize {
        32 / self.bits() as usize
    }
}

/// Marlin configuration for optimized INT4/INT8 GPTQ inference.
#[derive(Debug, Clone)]
pub struct MarlinConfig {
    /// Quantization bits (4 or 8)
    pub bits: u32,
    /// Group size for quantization (-1 for per-channel)
    pub group_size: i32,
    /// Whether to use descending activation order
    pub desc_act: bool,
    /// Whether quantization is symmetric
    pub is_sym: bool,
    /// Scalar type for weights
    pub scalar_type: MarlinScalarType,
    /// Use FP32 reduction for better precision
    pub use_fp32_reduce: bool,
    /// Whether this linear layer fuses activation-and-multiply (SwiGLU).
    /// Affects weight repacking and intermediate buffer sizing for MoE.
    /// True for gate+up projections, false for down projections and
    /// non-gated architectures (e.g. Nemotron-H).
    pub is_act_and_mul: bool,
}

impl MarlinConfig {
    /// Create a new Marlin config for 4-bit GPTQ.
    pub fn gptq_int4(group_size: i32) -> Self {
        Self {
            bits: 4,
            group_size,
            desc_act: false,
            is_sym: true,
            scalar_type: MarlinScalarType::Uint4b8,
            use_fp32_reduce: true,
            is_act_and_mul: true,
        }
    }

    /// Create a new Marlin config for 8-bit GPTQ.
    pub fn gptq_int8(group_size: i32) -> Self {
        Self {
            bits: 8,
            group_size,
            desc_act: false,
            is_sym: true,
            scalar_type: MarlinScalarType::Uint8b128,
            use_fp32_reduce: true,
            is_act_and_mul: true,
        }
    }

    /// Create a new Marlin config for AWQ (asymmetric with zero points).
    pub fn awq_int4(group_size: i32) -> Self {
        Self {
            bits: 4,
            group_size,
            desc_act: false,
            is_sym: false,
            scalar_type: MarlinScalarType::Uint4,
            use_fp32_reduce: true,
            is_act_and_mul: true,
        }
    }

    /// Create from detected configuration.
    pub fn from_detected(
        bits: Option<u32>,
        group_size: Option<usize>,
        desc_act: Option<bool>,
        is_sym: Option<bool>,
        raw_config: &HashMap<String, serde_json::Value>,
    ) -> Self {
        let bits = bits.unwrap_or(4);
        let group_size = group_size.map(|g| g as i32).unwrap_or(128);
        let desc_act = desc_act.unwrap_or(false);
        let is_sym = is_sym.unwrap_or(true);

        // Determine scalar type based on symmetry and bits
        let scalar_type = match (bits, is_sym) {
            (4, true) => MarlinScalarType::Uint4b8,
            (4, false) => MarlinScalarType::Uint4,
            (8, true) => MarlinScalarType::Uint8b128,
            (8, false) => MarlinScalarType::Uint8,
            _ => MarlinScalarType::Uint4b8,
        };

        let use_fp32_reduce = raw_config
            .get("use_fp32_reduce")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let is_act_and_mul = raw_config
            .get("is_act_and_mul")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            bits,
            group_size,
            desc_act,
            is_sym,
            scalar_type,
            use_fp32_reduce,
            is_act_and_mul,
        }
    }

    /// Enable desc_act optimization.
    pub fn with_desc_act(mut self, enabled: bool) -> Self {
        self.desc_act = enabled;
        self
    }

    /// Set FP32 reduction for better precision.
    pub fn with_fp32_reduce(mut self, enabled: bool) -> Self {
        self.use_fp32_reduce = enabled;
        self
    }

    /// Set whether this layer uses fused activation-and-multiply.
    pub fn with_act_and_mul(mut self, enabled: bool) -> Self {
        self.is_act_and_mul = enabled;
        self
    }

    /// Calculate number of groups for a given input size.
    pub fn num_groups(&self, in_features: usize) -> usize {
        if self.group_size <= 0 {
            1 // Per-channel quantization
        } else {
            in_features.div_ceil(self.group_size as usize)
        }
    }

    /// Check if the given group size is supported.
    pub fn is_group_size_supported(&self) -> bool {
        MARLIN_SUPPORTED_GROUP_SIZES.contains(&self.group_size)
    }
}

impl Default for MarlinConfig {
    fn default() -> Self {
        Self::gptq_int4(128)
    }
}

impl QuantizationConfig for MarlinConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Marlin
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        &[DType::F16, DType::BF16]
    }

    fn min_capability(&self) -> u32 {
        80 // Ampere required for Marlin
    }

    fn is_layer_skipped(&self, _layer_name: &str) -> bool {
        false // Marlin quantizes all linear layers
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
            self.clone(),
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// Check if Marlin is supported for the given configuration.
pub fn check_marlin_supported(
    scalar_type: MarlinScalarType,
    group_size: i32,
    has_zp: bool,
    device_capability: Option<u32>,
) -> bool {
    let capability = device_capability.unwrap_or(0);

    // Marlin requires at least SM75 (Turing), but optimal on SM80+ (Ampere)
    if capability > 0 && capability < 75 {
        return false;
    }

    // Check scalar type support
    let supported_types = if has_zp {
        // AWQ style with runtime zero points
        vec![MarlinScalarType::Uint4]
    } else {
        // GPTQ style with symmetric bias
        vec![
            MarlinScalarType::Uint4b8,
            MarlinScalarType::Uint8b128,
            MarlinScalarType::Float8E4m3fn,
        ]
    };

    if !supported_types.contains(&scalar_type) {
        return false;
    }

    // Check group size
    if !MARLIN_SUPPORTED_GROUP_SIZES.contains(&group_size) {
        return false;
    }

    true
}

/// Check if Marlin supports the given shape.
pub fn check_marlin_supports_shape(
    output_size: usize,
    input_size: usize,
    group_size: i32,
) -> Result<()> {
    // Validate output size (N dimension)
    if output_size % GPTQ_MARLIN_MIN_THREAD_N != 0 {
        candle_core::bail!(
            "Marlin requires output_size ({}) to be divisible by {} (GPTQ_MARLIN_MIN_THREAD_N)",
            output_size,
            GPTQ_MARLIN_MIN_THREAD_N
        );
    }

    // Validate input size (K dimension)
    if input_size % GPTQ_MARLIN_MIN_THREAD_K != 0 {
        candle_core::bail!(
            "Marlin requires input_size ({}) to be divisible by {} (GPTQ_MARLIN_MIN_THREAD_K)",
            input_size,
            GPTQ_MARLIN_MIN_THREAD_K
        );
    }

    // Check group size divides input evenly
    if group_size > 0 && (input_size % group_size as usize != 0) {
        candle_core::bail!(
            "Marlin requires input_size ({}) to be divisible by group_size ({})",
            input_size,
            group_size
        );
    }

    Ok(())
}

/// Create workspace tensor for Marlin kernel.
pub fn marlin_make_workspace(output_size: usize, device: &Device) -> Result<Tensor> {
    let max_workspace_size = (output_size / GPTQ_MARLIN_MIN_THREAD_N) * GPTQ_MARLIN_MAX_PARALLEL;
    Tensor::zeros(max_workspace_size, DType::U32, device)
}

/// Permutation indices for scale reordering.
fn get_scale_perms() -> (Vec<usize>, Vec<usize>) {
    // Full permutation for grouped scales
    let mut scale_perm = Vec::with_capacity(64);
    for i in 0..8 {
        for j in 0..8 {
            scale_perm.push(i + 8 * j);
        }
    }

    // Single permutation for per-channel scales
    let mut scale_perm_single = Vec::with_capacity(32);
    for i in 0..4 {
        scale_perm_single.extend([
            2 * i,
            2 * i + 1,
            2 * i + 8,
            2 * i + 9,
            2 * i + 16,
            2 * i + 17,
            2 * i + 24,
            2 * i + 25,
        ]);
    }

    (scale_perm, scale_perm_single)
}

/// Permute scales for Marlin format.
///
/// Marlin requires scales to be permuted for efficient coalesced access.
pub fn marlin_permute_scales(
    scales: &Tensor,
    size_k: usize,
    size_n: usize,
    group_size: i32,
) -> Result<Tensor> {
    let (scale_perm, scale_perm_single) = get_scale_perms();

    // Choose permutation based on whether we have group scales or per-channel
    let perm = if group_size > 0 && group_size < size_k as i32 {
        &scale_perm
    } else {
        &scale_perm_single
    };

    // Reshape and permute
    let perm_len = perm.len();
    let reshaped = scales.reshape(((), perm_len))?;

    // Create permutation tensor
    let perm_tensor = Tensor::new(
        perm.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        scales.device(),
    )?;

    // Apply permutation using index_select
    let permuted = reshaped.index_select(&perm_tensor, 1)?;

    // Reshape back
    permuted.reshape(((), size_n))?.contiguous()
}

/// Repack weights from standard GPTQ format to Marlin format.
///
/// Standard GPTQ: weights packed row-major in INT32
/// Marlin: weights packed in 16x16 tiles for tensor core operations
pub fn repack_gptq_to_marlin(
    qweight: &Tensor,
    g_idx_sort_indices: Option<&Tensor>,
    size_k: usize,
    size_n: usize,
    bits: u32,
) -> Result<Tensor> {
    let pack_factor = 32 / bits as usize;

    // Validate input shape
    let qweight_dims = qweight.dims();
    if qweight_dims.len() != 2 {
        candle_core::bail!("qweight must be 2D, got {}D", qweight_dims.len());
    }

    let expected_k_packed = size_k / pack_factor;
    if qweight_dims[0] != expected_k_packed {
        candle_core::bail!(
            "qweight dim 0 ({}) doesn't match expected packed K ({} = {} / {})",
            qweight_dims[0],
            expected_k_packed,
            size_k,
            pack_factor
        );
    }

    if qweight_dims[1] != size_n {
        candle_core::bail!(
            "qweight dim 1 ({}) doesn't match size_n ({})",
            qweight_dims[1],
            size_n
        );
    }

    // For now, use the CUDA kernel for repacking when available
    #[cfg(feature = "marlin")]
    if qweight.device().is_cuda() {
        return marlin_cuda::repack_gptq_to_marlin(
            qweight,
            g_idx_sort_indices,
            size_k,
            size_n,
            bits,
        );
    }

    // CPU fallback: just return the original weights
    // In practice, Marlin requires CUDA, so this path is mainly for testing
    let _ = g_idx_sort_indices;
    Ok(qweight.clone())
}

/// Marlin optimized linear layer.
///
/// Uses Marlin kernel for INT4 GPTQ inference with 2-3x speedup
/// over standard dequantize+GEMM operations.
#[derive(Debug)]
pub struct MarlinLinear {
    /// Repacked quantized weights in Marlin format [K/tile, N/tile, tile, tile]
    qweight: Tensor,
    /// Permuted scales [num_groups, N]
    scales: Tensor,
    /// Zero points (for AWQ, empty for GPTQ)
    qzeros: Tensor,
    /// Optional bias [N]
    bias: Option<Tensor>,
    /// G_idx for desc_act (optional)
    g_idx: Option<Tensor>,
    /// Sort indices for g_idx
    g_idx_sort_indices: Option<Tensor>,
    /// Workspace tensor for Marlin kernel
    #[allow(dead_code)] // Used in forward_marlin when marlin feature is enabled
    workspace: Tensor,
    /// Configuration
    config: MarlinConfig,
    /// Input features (K dimension)
    in_features: usize,
    /// Output features (N dimension)
    out_features: usize,
    /// Whether weights are loaded and repacked
    is_initialized: bool,
}

impl MarlinLinear {
    /// Create a new Marlin linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        config: MarlinConfig,
        device: &Device,
    ) -> Result<Self> {
        // Validate inputs
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }

        // Validate shape for Marlin
        check_marlin_supports_shape(out_features, in_features, config.group_size)?;

        // Validate group size
        if !config.is_group_size_supported() {
            candle_core::bail!(
                "Marlin does not support group_size {}. Supported: {:?}",
                config.group_size,
                MARLIN_SUPPORTED_GROUP_SIZES
            );
        }

        let pack_factor = config.scalar_type.pack_factor();
        let num_groups = config.num_groups(in_features);

        // Initialize empty tensors (will be loaded later)
        let packed_k = in_features / pack_factor;
        let qweight = Tensor::zeros((packed_k, out_features), DType::U32, device)?;
        let scales = Tensor::zeros((num_groups, out_features), DType::F16, device)?;

        // Zero points only needed for asymmetric quantization
        let qzeros = if config.scalar_type.has_zero_point() {
            Tensor::zeros((num_groups, out_features / pack_factor), DType::U32, device)?
        } else {
            Tensor::zeros(0, DType::U32, device)?
        };

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F16, device)?)
        } else {
            None
        };

        // Create workspace
        let workspace = marlin_make_workspace(out_features, device)?;

        Ok(Self {
            qweight,
            scales,
            qzeros,
            bias,
            g_idx: None,
            g_idx_sort_indices: None,
            workspace,
            config,
            in_features,
            out_features,
            is_initialized: false,
        })
    }

    /// Check if Marlin CUDA kernel can be used.
    #[cfg(feature = "marlin")]
    fn can_use_marlin_kernel(&self) -> bool {
        self.is_initialized && self.qweight.device().is_cuda()
    }

    #[cfg(not(feature = "marlin"))]
    #[allow(dead_code)] // Called from forward() but method body is feature-gated
    fn can_use_marlin_kernel(&self) -> bool {
        false
    }

    /// Forward using Marlin CUDA kernel.
    #[cfg(feature = "marlin")]
    fn forward_marlin(&self, x: &Tensor) -> Result<Tensor> {
        let is_k_full = !self.config.desc_act || self.g_idx_sort_indices.is_none();

        marlin_cuda::marlin_gemm(
            x,
            &self.qweight,
            &self.scales,
            self.qzeros.as_ref().filter(|z| z.elem_count() > 0),
            self.g_idx.as_ref(),
            self.g_idx_sort_indices.as_ref(),
            &self.workspace,
            self.bias.as_ref(),
            self.config.scalar_type,
            self.in_features,
            self.out_features,
            is_k_full,
            self.config.use_fp32_reduce,
        )
    }

    /// Fallback forward using standard GPTQ dequantization.
    fn forward_fallback(&self, _x: &Tensor) -> Result<Tensor> {
        // When Marlin kernel is not available, fall back to standard GPTQ
        // This is slower but ensures correctness
        #[cfg(feature = "cuda-kernels")]
        {
            use super::gptq_cuda;

            let scales_bf16 = if self.scales.dtype() == DType::BF16 {
                self.scales.clone()
            } else {
                self.scales.to_dtype(DType::BF16)?
            };

            return gptq_cuda::gptq_gemm(
                x,
                &self.qweight,
                &scales_bf16,
                &self.qzeros,
                self.bias.as_ref(),
                self.config.group_size,
            );
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            candle_core::bail!(
                "Marlin/GPTQ forward requires CUDA. Enable cuda-kernels or marlin feature."
            );
        }
    }

    /// Process weights after loading: repack to Marlin format.
    fn process_weights(&mut self) -> Result<()> {
        if !self.is_initialized {
            return Ok(());
        }

        // Handle activation order (desc_act)
        if self.config.desc_act {
            if let Some(ref g_idx) = self.g_idx {
                // Sort g_idx and get sort indices
                let sorted = g_idx.arg_sort_last_dim(false)?;
                self.g_idx_sort_indices = Some(sorted.to_dtype(DType::U32)?);

                // Reorder g_idx by sort indices
                if let Some(ref sort_indices) = self.g_idx_sort_indices {
                    let sorted_g_idx = g_idx.index_select(sort_indices, 0)?;
                    self.g_idx = Some(sorted_g_idx);
                }
            }
        }

        // Repack weights to Marlin format
        let repacked = repack_gptq_to_marlin(
            &self.qweight,
            self.g_idx_sort_indices.as_ref(),
            self.in_features,
            self.out_features,
            self.config.bits,
        )?;
        self.qweight = repacked;

        // Permute scales
        let permuted_scales = marlin_permute_scales(
            &self.scales,
            self.in_features,
            self.out_features,
            self.config.group_size,
        )?;
        self.scales = permuted_scales;

        Ok(())
    }
}

impl QuantizedLinear for MarlinLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.is_initialized {
            candle_core::bail!("Marlin layer has no weights loaded - call load_weights() first");
        }

        // Use Marlin kernel when available
        #[cfg(feature = "marlin")]
        if self.can_use_marlin_kernel() {
            return self.forward_marlin(x);
        }

        // Fallback to standard GPTQ
        self.forward_fallback(x)
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        if let Some(w) = weights.get("qweight") {
            self.qweight = w.clone();
            self.is_initialized = true;
        }
        if let Some(s) = weights.get("scales") {
            self.scales = s.clone();
        }
        if let Some(z) = weights.get("qzeros") {
            self.qzeros = z.clone();
        }
        if let Some(b) = weights.get("bias") {
            self.bias = Some(b.clone());
        }
        if let Some(g) = weights.get("g_idx") {
            self.g_idx = Some(g.clone());
        }

        // Process weights for Marlin format
        self.process_weights()?;

        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::U32 // Packed format
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

    #[test]
    fn test_marlin_config_gptq_int4() {
        let config = MarlinConfig::gptq_int4(128);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert!(config.is_sym);
        assert_eq!(config.scalar_type, MarlinScalarType::Uint4b8);
        assert!(config.is_group_size_supported());
    }

    #[test]
    fn test_marlin_config_awq() {
        let config = MarlinConfig::awq_int4(64);
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 64);
        assert!(!config.is_sym);
        assert_eq!(config.scalar_type, MarlinScalarType::Uint4);
    }

    #[test]
    fn test_marlin_scalar_type() {
        assert_eq!(MarlinScalarType::Uint4b8.bits(), 4);
        assert_eq!(MarlinScalarType::Uint4b8.pack_factor(), 8);
        assert!(!MarlinScalarType::Uint4b8.has_zero_point());

        assert_eq!(MarlinScalarType::Uint4.bits(), 4);
        assert!(MarlinScalarType::Uint4.has_zero_point());

        assert_eq!(MarlinScalarType::Uint8b128.bits(), 8);
        assert_eq!(MarlinScalarType::Uint8b128.pack_factor(), 4);
    }

    #[test]
    fn test_check_marlin_supported() {
        // GPTQ INT4 with valid group sizes
        assert!(check_marlin_supported(
            MarlinScalarType::Uint4b8,
            128,
            false,
            Some(80)
        ));
        assert!(check_marlin_supported(
            MarlinScalarType::Uint4b8,
            -1,
            false,
            Some(80)
        ));

        // Invalid group size
        assert!(!check_marlin_supported(
            MarlinScalarType::Uint4b8,
            100, // Not in MARLIN_SUPPORTED_GROUP_SIZES
            false,
            Some(80)
        ));

        // Wrong device capability
        assert!(!check_marlin_supported(
            MarlinScalarType::Uint4b8,
            128,
            false,
            Some(70) // Too old
        ));

        // AWQ with zero points
        assert!(check_marlin_supported(
            MarlinScalarType::Uint4,
            128,
            true,
            Some(80)
        ));
    }

    #[test]
    fn test_check_marlin_supports_shape() {
        // Valid shapes
        assert!(check_marlin_supports_shape(4096, 4096, 128).is_ok());
        assert!(check_marlin_supports_shape(1024, 2048, -1).is_ok());

        // Invalid N dimension
        assert!(check_marlin_supports_shape(100, 4096, 128).is_err());

        // Invalid K dimension
        assert!(check_marlin_supports_shape(4096, 100, 128).is_err());
    }

    #[test]
    fn test_marlin_config_method() {
        let config = MarlinConfig::default();
        assert_eq!(config.method(), QuantizationMethod::Marlin);
        assert_eq!(config.min_capability(), 80);
    }

    #[test]
    fn test_marlin_num_groups() {
        let config = MarlinConfig::gptq_int4(128);
        assert_eq!(config.num_groups(4096), 32);
        assert_eq!(config.num_groups(128), 1);
        assert_eq!(config.num_groups(256), 2);

        let config_perchannel = MarlinConfig::gptq_int4(-1);
        assert_eq!(config_perchannel.num_groups(4096), 1);
    }

    #[test]
    fn test_scale_permutations() {
        let (scale_perm, scale_perm_single) = get_scale_perms();

        // Full permutation has 64 elements
        assert_eq!(scale_perm.len(), 64);

        // Single permutation has 32 elements
        assert_eq!(scale_perm_single.len(), 32);

        // Check first few elements of scale_perm
        assert_eq!(scale_perm[0], 0);
        assert_eq!(scale_perm[1], 8);
        assert_eq!(scale_perm[8], 1);
    }

    #[test]
    fn test_marlin_linear_validation() {
        // Zero features should fail
        let result = MarlinLinear::new(0, 128, false, MarlinConfig::default(), &Device::Cpu);
        assert!(result.is_err());

        let result = MarlinLinear::new(128, 0, false, MarlinConfig::default(), &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_marlin_linear_creation() {
        // Valid shape that meets Marlin requirements
        let config = MarlinConfig::gptq_int4(128);
        let linear = MarlinLinear::new(4096, 4096, true, config, &Device::Cpu);

        assert!(linear.is_ok());
        let linear = linear.unwrap();
        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_marlin_config_is_act_and_mul_default() {
        let config = MarlinConfig::gptq_int4(128);
        assert!(config.is_act_and_mul, "default should be true (SwiGLU)");
    }

    #[test]
    fn test_marlin_config_with_act_and_mul() {
        let config = MarlinConfig::gptq_int4(128).with_act_and_mul(false);
        assert!(!config.is_act_and_mul);
    }

    #[test]
    fn test_marlin_config_from_detected_act_and_mul() {
        let mut raw = HashMap::new();
        raw.insert(
            "is_act_and_mul".to_string(),
            serde_json::Value::Bool(false),
        );
        let config = MarlinConfig::from_detected(Some(4), Some(128), None, None, &raw);
        assert!(!config.is_act_and_mul);

        // Default when not present in raw_config
        let config_default =
            MarlinConfig::from_detected(Some(4), Some(128), None, None, &HashMap::new());
        assert!(config_default.is_act_and_mul);
    }

    #[test]
    fn test_marlin_linear_requires_loaded_weights() {
        let config = MarlinConfig::gptq_int4(128);
        let linear = MarlinLinear::new(4096, 4096, false, config, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[2, 4096], DType::F16, &Device::Cpu).unwrap();
        let result = linear.forward(&x);

        // Should fail because weights aren't loaded
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no weights loaded"));
    }
}
