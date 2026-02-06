use candle_core::{DType, Device};

use super::offload::CpuOffloadConfig;
use super::quantization::KVCacheDtype;

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Compute dtype for model activations (BF16/F16)
    pub dtype: DType,
    pub device: Device,
    /// KV cache storage dtype: Auto (no quantization), Fp8E4m3, or Int8
    pub kv_cache_dtype: KVCacheDtype,
    /// Optional CPU offload configuration. When `Some`, evicted GPU blocks
    /// are stored in CPU memory for potential reuse instead of being discarded.
    pub cpu_offload: Option<CpuOffloadConfig>,
}

impl CacheConfig {
    /// Compute num_blocks from available GPU memory budget.
    ///
    /// bytes_per_block_per_layer = 2(K+V) * num_kv_heads * block_size * head_dim * elem_size
    /// num_blocks = budget_bytes / (num_layers * bytes_per_block_per_layer)
    ///
    /// When using quantized KV cache (FP8/INT8), elem_size = 1 byte instead of 2,
    /// enabling 2x more blocks from the same memory budget.
    pub fn from_memory_budget(
        budget_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self::from_memory_budget_with_kv_dtype(
            budget_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            block_size,
            dtype,
            device,
            KVCacheDtype::Auto,
        )
    }

    /// Compute num_blocks from available GPU memory budget with explicit KV cache dtype.
    #[allow(clippy::too_many_arguments)]
    pub fn from_memory_budget_with_kv_dtype(
        budget_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        device: Device,
        kv_cache_dtype: KVCacheDtype,
    ) -> Self {
        let elem_size = kv_cache_dtype.element_size(dtype);
        let bytes_per_block_per_layer = 2 * num_kv_heads * block_size * head_dim * elem_size;
        let total_per_block = num_layers * bytes_per_block_per_layer;
        let num_blocks = if total_per_block > 0 {
            budget_bytes / total_per_block
        } else {
            0
        };

        Self {
            block_size,
            num_blocks,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype,
            device,
            kv_cache_dtype,
            cpu_offload: None,
        }
    }

    /// Returns the element size used for KV cache storage.
    pub fn kv_element_size(&self) -> usize {
        self.kv_cache_dtype.element_size(self.dtype)
    }

    /// Returns the storage dtype for KV cache tensors.
    pub fn kv_storage_dtype(&self) -> DType {
        self.kv_cache_dtype.storage_dtype(self.dtype)
    }

    /// Returns true if KV cache uses quantization.
    pub fn is_quantized(&self) -> bool {
        self.kv_cache_dtype.is_quantized()
    }

    /// Calculate memory usage per block across all layers.
    pub fn bytes_per_block(&self) -> usize {
        let elem_size = self.kv_element_size();
        // 2 (K+V) * heads * block_tokens * head_dim * layers * elem_size
        2 * self.num_kv_heads * self.block_size * self.head_dim * self.num_layers * elem_size
    }

    /// Calculate total KV cache memory usage.
    pub fn total_memory_bytes(&self) -> usize {
        self.bytes_per_block() * self.num_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_memory_budget_computes_correct_blocks() {
        // Qwen3-0.6B: 28 layers, 8 kv_heads, 128 head_dim, BF16
        // Per block per layer: 2 * 8 * 16 * 128 * 2 = 65536 bytes
        // All layers per block: 28 * 65536 = 1,835,008 bytes (~1.75 MB)
        // 900 MB budget: 900*1024*1024 / 1835008 ≈ 514 blocks
        let budget = 900 * 1024 * 1024;
        let config =
            CacheConfig::from_memory_budget(budget, 28, 8, 128, 16, DType::BF16, Device::Cpu);
        assert_eq!(config.num_blocks, 514);
        assert_eq!(config.block_size, 16);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.kv_cache_dtype, KVCacheDtype::Auto);
    }

    #[test]
    fn from_memory_budget_fp8_doubles_blocks() {
        // Same config but with FP8: element size is 1 byte instead of 2
        // So we should get approximately 2x the blocks
        let budget = 900 * 1024 * 1024;

        let config_bf16 =
            CacheConfig::from_memory_budget(budget, 28, 8, 128, 16, DType::BF16, Device::Cpu);

        let config_fp8 = CacheConfig::from_memory_budget_with_kv_dtype(
            budget,
            28,
            8,
            128,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Fp8E4m3,
        );

        // FP8 should have exactly 2x the blocks (element size halved)
        assert_eq!(config_fp8.num_blocks, config_bf16.num_blocks * 2);
        assert_eq!(config_fp8.kv_cache_dtype, KVCacheDtype::Fp8E4m3);
    }

    #[test]
    fn from_memory_budget_int8_doubles_blocks() {
        let budget = 900 * 1024 * 1024;

        let config_bf16 =
            CacheConfig::from_memory_budget(budget, 28, 8, 128, 16, DType::BF16, Device::Cpu);

        let config_int8 = CacheConfig::from_memory_budget_with_kv_dtype(
            budget,
            28,
            8,
            128,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Int8,
        );

        assert_eq!(config_int8.num_blocks, config_bf16.num_blocks * 2);
        assert_eq!(config_int8.kv_cache_dtype, KVCacheDtype::Int8);
    }

    #[test]
    fn kv_element_size_correct() {
        let config = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Auto,
        );
        assert_eq!(config.kv_element_size(), 2); // BF16 = 2 bytes

        let config_fp8 = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Fp8E4m3,
        );
        assert_eq!(config_fp8.kv_element_size(), 1); // FP8 = 1 byte
    }

    #[test]
    fn kv_storage_dtype_correct() {
        let config = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Auto,
        );
        assert_eq!(config.kv_storage_dtype(), DType::BF16);

        let config_fp8 = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Fp8E4m3,
        );
        assert_eq!(config_fp8.kv_storage_dtype(), DType::U8);
    }

    #[test]
    fn is_quantized_correct() {
        let config_auto = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Auto,
        );
        assert!(!config_auto.is_quantized());

        let config_fp8 = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Fp8E4m3,
        );
        assert!(config_fp8.is_quantized());

        let config_int8 = CacheConfig::from_memory_budget_with_kv_dtype(
            1024 * 1024,
            1,
            1,
            64,
            16,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Int8,
        );
        assert!(config_int8.is_quantized());
    }

    #[test]
    fn bytes_per_block_correct() {
        // 1 layer, 2 kv_heads, 4 block_size, 8 head_dim, BF16 (2 bytes)
        // bytes = 2 * 2 * 4 * 8 * 1 * 2 = 256 bytes
        let config = CacheConfig {
            block_size: 4,
            num_blocks: 10,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::BF16,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        assert_eq!(config.bytes_per_block(), 256);

        // Same config with FP8: bytes = 2 * 2 * 4 * 8 * 1 * 1 = 128 bytes
        let config_fp8 = CacheConfig {
            kv_cache_dtype: KVCacheDtype::Fp8E4m3,
            ..config
        };
        assert_eq!(config_fp8.bytes_per_block(), 128);
    }

    #[test]
    fn total_memory_bytes_correct() {
        let config = CacheConfig {
            block_size: 4,
            num_blocks: 10,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::BF16,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        // 256 bytes per block * 10 blocks = 2560 bytes
        assert_eq!(config.total_memory_bytes(), 2560);
    }
}
