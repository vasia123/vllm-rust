//! Configuration for MLA (Multi-head Latent Attention) cache.
//!
//! MLA compresses KV cache by storing low-rank latent representations
//! instead of full K/V tensors, achieving 42x memory reduction.

use candle_core::{DType, Device};

use super::quantization::KVCacheDtype;

/// Configuration for MLA cache (DeepSeek V2/V3).
///
/// MLA stores compressed latent representations:
/// - `kv_c_cache`: Compressed KV latent [num_blocks, block_size, kv_lora_rank]
/// - `k_pe_cache`: RoPE key [num_blocks, block_size, qk_rope_head_dim]
///
/// This reduces memory by ~42x compared to standard KV cache.
#[derive(Debug, Clone)]
pub struct MLACacheConfig {
    /// Low-rank dimension for KV compression (512 for DeepSeek V3)
    pub kv_lora_rank: usize,
    /// RoPE dimension for keys (64 for DeepSeek V3)
    pub qk_rope_head_dim: usize,
    /// Non-RoPE dimension for keys (128 for DeepSeek V3)
    pub qk_nope_head_dim: usize,
    /// Value head dimension (128 for DeepSeek V3)
    pub v_head_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Tokens per block
    pub block_size: usize,
    /// Total number of blocks
    pub num_blocks: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Compute dtype for activations
    pub dtype: DType,
    /// Target device
    pub device: Device,
    /// Optional quantization for MLA cache
    pub kv_cache_dtype: KVCacheDtype,
}

impl MLACacheConfig {
    /// Create MLA cache config from model parameters.
    ///
    /// # Arguments
    /// * `kv_lora_rank` - Low-rank dimension (typically 512)
    /// * `qk_rope_head_dim` - RoPE key dimension (typically 64)
    /// * `qk_nope_head_dim` - Non-RoPE key dimension (typically 128)
    /// * `v_head_dim` - Value head dimension (typically 128)
    /// * `num_heads` - Number of attention heads
    /// * `block_size` - Tokens per block
    /// * `num_blocks` - Total blocks to allocate
    /// * `num_layers` - Number of transformer layers
    /// * `dtype` - Compute dtype
    /// * `device` - Target device
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
        num_heads: usize,
        block_size: usize,
        num_blocks: usize,
        num_layers: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            num_heads,
            block_size,
            num_blocks,
            num_layers,
            dtype,
            device,
            kv_cache_dtype: KVCacheDtype::Auto,
        }
    }

    /// Set quantization mode for MLA cache.
    pub fn with_kv_cache_dtype(mut self, dtype: KVCacheDtype) -> Self {
        self.kv_cache_dtype = dtype;
        self
    }

    /// Calculate bytes per token for MLA cache.
    ///
    /// MLA stores:
    /// - kv_c: [kv_lora_rank] per token
    /// - k_pe: [qk_rope_head_dim] per token
    ///
    /// Total: (kv_lora_rank + qk_rope_head_dim) * elem_size
    pub fn bytes_per_token(&self) -> usize {
        let elem_size = self.kv_cache_dtype.element_size(self.dtype);
        (self.kv_lora_rank + self.qk_rope_head_dim) * elem_size
    }

    /// Calculate bytes per token for standard KV cache (for comparison).
    ///
    /// Standard cache stores:
    /// - K: [num_heads, head_dim] per token
    /// - V: [num_heads, head_dim] per token
    ///
    /// Total: 2 * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * elem_size
    pub fn standard_bytes_per_token(&self) -> usize {
        let head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let elem_size = self.kv_cache_dtype.element_size(self.dtype);
        2 * self.num_heads * head_dim * elem_size
    }

    /// Calculate memory reduction ratio vs standard cache.
    pub fn memory_reduction_ratio(&self) -> f64 {
        let standard = self.standard_bytes_per_token() as f64;
        let mla = self.bytes_per_token() as f64;
        standard / mla
    }

    /// Calculate bytes per block for MLA cache.
    pub fn bytes_per_block(&self) -> usize {
        self.bytes_per_token() * self.block_size * self.num_layers
    }

    /// Calculate total memory usage for MLA cache.
    pub fn total_memory_bytes(&self) -> usize {
        self.bytes_per_block() * self.num_blocks
    }

    /// Compute num_blocks from memory budget.
    #[allow(clippy::too_many_arguments)]
    pub fn from_memory_budget(
        budget_bytes: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
        num_heads: usize,
        block_size: usize,
        num_layers: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        let mut config = Self::new(
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
            num_heads,
            block_size,
            0, // Will be computed
            num_layers,
            dtype,
            device,
        );

        let bytes_per_block = config.bytes_per_block();
        config.num_blocks = if bytes_per_block > 0 {
            budget_bytes / bytes_per_block
        } else {
            0
        };

        config
    }

    /// Get the storage dtype for cache tensors.
    pub fn storage_dtype(&self) -> DType {
        self.kv_cache_dtype.storage_dtype(self.dtype)
    }

    /// Returns true if cache uses quantization.
    pub fn is_quantized(&self) -> bool {
        self.kv_cache_dtype.is_quantized()
    }
}

/// Dimensions extracted from DeepSeek MLA configuration.
#[derive(Debug, Clone, Copy)]
pub struct MLADims {
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
}

impl MLADims {
    /// Default dimensions for DeepSeek V3.
    pub fn deepseek_v3() -> Self {
        Self {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
        }
    }

    /// Extract MLA dimensions from model config extra fields.
    pub fn from_config_extra(extra: &serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        let kv_lora_rank = extra.get("kv_lora_rank")?.as_u64()? as usize;
        let qk_rope_head_dim = extra.get("qk_rope_head_dim")?.as_u64()? as usize;
        let qk_nope_head_dim = extra.get("qk_nope_head_dim")?.as_u64()? as usize;
        let v_head_dim = extra.get("v_head_dim")?.as_u64()? as usize;

        Some(Self {
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            v_head_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_config_creation() {
        let config = MLACacheConfig::new(
            512,  // kv_lora_rank
            64,   // qk_rope_head_dim
            128,  // qk_nope_head_dim
            128,  // v_head_dim
            128,  // num_heads (DeepSeek V3)
            16,   // block_size
            1000, // num_blocks
            60,   // num_layers
            DType::BF16,
            Device::Cpu,
        );

        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.qk_rope_head_dim, 64);
    }

    #[test]
    fn test_memory_reduction_ratio() {
        let config = MLACacheConfig::new(
            512, // kv_lora_rank
            64,  // qk_rope_head_dim
            128, // qk_nope_head_dim
            128, // v_head_dim
            128, // num_heads
            16,
            1000,
            60,
            DType::BF16,
            Device::Cpu,
        );

        // Standard: 2 * 128 * (128+64) * 2 = 98,304 bytes/token
        // MLA: (512 + 64) * 2 = 1,152 bytes/token
        // Ratio: 98,304 / 1,152 = 85.33x
        let ratio = config.memory_reduction_ratio();
        assert!(ratio > 80.0 && ratio < 90.0, "ratio={}", ratio);
    }

    #[test]
    fn test_bytes_per_token() {
        let config = MLACacheConfig::new(
            512,
            64,
            128,
            128,
            128,
            16,
            1000,
            60,
            DType::BF16,
            Device::Cpu,
        );

        // MLA: (512 + 64) * 2 bytes = 1,152 bytes
        assert_eq!(config.bytes_per_token(), 1152);

        // Standard: 2 * 128 * (128+64) * 2 = 98,304 bytes
        assert_eq!(config.standard_bytes_per_token(), 98304);
    }

    #[test]
    fn test_from_memory_budget() {
        // 1 GB budget
        let budget = 1024 * 1024 * 1024;
        let config = MLACacheConfig::from_memory_budget(
            budget,
            512,
            64,
            128,
            128,
            128,
            16,
            60,
            DType::BF16,
            Device::Cpu,
        );

        // bytes_per_block = (512 + 64) * 2 * 16 * 60 = 1,105,920 bytes
        // num_blocks = 1GB / 1,105,920 ≈ 971
        assert!(config.num_blocks > 900 && config.num_blocks < 1000);
    }

    #[test]
    fn test_mla_dims_from_config() {
        let mut extra = serde_json::Map::new();
        extra.insert("kv_lora_rank".into(), serde_json::json!(512));
        extra.insert("qk_rope_head_dim".into(), serde_json::json!(64));
        extra.insert("qk_nope_head_dim".into(), serde_json::json!(128));
        extra.insert("v_head_dim".into(), serde_json::json!(128));

        let dims = MLADims::from_config_extra(&extra).unwrap();
        assert_eq!(dims.kv_lora_rank, 512);
        assert_eq!(dims.qk_rope_head_dim, 64);
        assert_eq!(dims.qk_nope_head_dim, 128);
        assert_eq!(dims.v_head_dim, 128);
    }

    #[test]
    fn test_mla_dims_missing_fields() {
        let extra = serde_json::Map::new();
        assert!(MLADims::from_config_extra(&extra).is_none());
    }

    #[test]
    fn test_with_quantization() {
        let config = MLACacheConfig::new(
            512,
            64,
            128,
            128,
            128,
            16,
            1000,
            60,
            DType::BF16,
            Device::Cpu,
        )
        .with_kv_cache_dtype(KVCacheDtype::Fp8E4m3);

        assert!(config.is_quantized());

        // FP8: (512 + 64) * 1 = 576 bytes
        assert_eq!(config.bytes_per_token(), 576);
    }
}
