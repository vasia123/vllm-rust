//! Configuration for fused MoE kernels.

/// Block size configuration for fused MoE kernels.
#[derive(Debug, Clone, Copy)]
pub struct FusedMoEBlockConfig {
    /// Block size for M dimension (tokens).
    pub block_size_m: usize,
    /// Block size for N dimension (output features).
    pub block_size_n: usize,
    /// Block size for K dimension (input features).
    pub block_size_k: usize,
    /// Group size for M dimension (L2 cache optimization).
    pub group_size_m: usize,
}

impl Default for FusedMoEBlockConfig {
    fn default() -> Self {
        Self {
            block_size_m: 64,
            block_size_n: 64,
            block_size_k: 32,
            group_size_m: 8,
        }
    }
}

impl FusedMoEBlockConfig {
    /// Configuration optimized for small batches.
    pub fn small_batch() -> Self {
        Self {
            block_size_m: 16,
            block_size_n: 64,
            block_size_k: 64,
            group_size_m: 8,
        }
    }

    /// Configuration optimized for large batches.
    pub fn large_batch() -> Self {
        Self {
            block_size_m: 128,
            block_size_n: 128,
            block_size_k: 32,
            group_size_m: 8,
        }
    }

    /// Select optimal configuration based on problem size.
    pub fn auto_select(num_tokens: usize, hidden_size: usize, intermediate_size: usize) -> Self {
        // Heuristic based on vLLM's autotuning results
        if num_tokens < 64 {
            Self::small_batch()
        } else if num_tokens >= 256 && hidden_size >= 4096 && intermediate_size >= 11008 {
            Self::large_batch()
        } else {
            Self::default()
        }
    }
}

/// Configuration for the fused MoE layer.
#[derive(Debug, Clone)]
pub struct FusedMoEConfig {
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts activated per token.
    pub top_k: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Intermediate (FFN) dimension size.
    pub intermediate_size: usize,
    /// Whether to renormalize routing weights.
    pub renormalize: bool,
    /// Block configuration for CUDA kernels.
    pub block_config: FusedMoEBlockConfig,
}

impl FusedMoEConfig {
    /// Create a new fused MoE configuration.
    pub fn new(
        num_experts: usize,
        top_k: usize,
        hidden_size: usize,
        intermediate_size: usize,
        renormalize: bool,
    ) -> Self {
        let block_config = FusedMoEBlockConfig::auto_select(
            128, // Default assumption, will be overridden at runtime
            hidden_size,
            intermediate_size,
        );

        Self {
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            renormalize,
            block_config,
        }
    }

    /// Get the block size used for token alignment.
    pub fn alignment_block_size(&self) -> usize {
        self.block_config.block_size_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FusedMoEBlockConfig::default();
        assert_eq!(config.block_size_m, 64);
        assert_eq!(config.block_size_n, 64);
    }

    #[test]
    fn test_auto_select() {
        let small = FusedMoEBlockConfig::auto_select(32, 4096, 11008);
        assert_eq!(small.block_size_m, 16);

        let large = FusedMoEBlockConfig::auto_select(512, 4096, 11008);
        assert_eq!(large.block_size_m, 128);
    }
}
