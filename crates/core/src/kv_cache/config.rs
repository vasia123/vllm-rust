use candle_core::{DType, Device};

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
    pub device: Device,
}

impl CacheConfig {
    /// Compute num_blocks from available GPU memory budget.
    ///
    /// bytes_per_block_per_layer = 2(K+V) * num_kv_heads * block_size * head_dim * dtype_size
    /// num_blocks = budget_bytes / (num_layers * bytes_per_block_per_layer)
    pub fn from_memory_budget(
        budget_bytes: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        let elem_size = dtype.size_in_bytes();
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
        }
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
    }
}
