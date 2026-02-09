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
    /// Whether to reuse the input buffer as the output buffer.
    /// Saves one `[num_tokens, hidden_size]` allocation per forward pass.
    /// Must be disabled when shared experts are present (they need the
    /// original input). Currently a no-op in Candle (tensors are immutable);
    /// will take effect once mutable-buffer CUDA kernels are implemented.
    pub inplace: bool,
    /// Whether the first projection uses fused activation-and-multiply
    /// (SwiGLU: `SiLU(gate(x)) * up(x)`). When true, the w13 weight
    /// shape is `[E, 2*intermediate_size, hidden_size]` and the
    /// intermediate buffer is `2*intermediate_size` wide. When false,
    /// a plain activation is applied and w13 has shape
    /// `[E, intermediate_size, hidden_size]`.
    pub is_act_and_mul: bool,
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
            inplace: false,
            is_act_and_mul: true,
        }
    }

    /// Get the block size used for token alignment.
    pub fn alignment_block_size(&self) -> usize {
        self.block_config.block_size_m
    }

    /// Expected first-projection (w13) N-dimension size.
    ///
    /// When `is_act_and_mul` is true (SwiGLU), the gate+up weights are
    /// stacked, so N = 2 * intermediate_size. When false, N = intermediate_size.
    pub fn w13_n_dim(&self) -> usize {
        if self.is_act_and_mul {
            2 * self.intermediate_size
        } else {
            self.intermediate_size
        }
    }

    /// Intermediate buffer size for a given number of tokens.
    ///
    /// The intermediate buffer holds the output of the first projection
    /// (gate+up or plain), before the down projection.
    pub fn intermediate_cache_size(&self, num_tokens: usize) -> usize {
        num_tokens * self.w13_n_dim()
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

    #[test]
    fn test_fused_moe_config_inplace_default() {
        let config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        assert!(!config.inplace, "inplace should default to false");
    }

    #[test]
    fn test_fused_moe_config_inplace_set() {
        let mut config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        config.inplace = true;
        assert!(config.inplace);
    }

    #[test]
    fn test_fused_moe_config_is_act_and_mul_default() {
        let config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        assert!(config.is_act_and_mul, "is_act_and_mul should default to true (SwiGLU)");
    }

    #[test]
    fn test_w13_n_dim_with_act_and_mul() {
        let config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        assert_eq!(config.w13_n_dim(), 2 * 11008);
    }

    #[test]
    fn test_w13_n_dim_without_act_and_mul() {
        let mut config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        config.is_act_and_mul = false;
        assert_eq!(config.w13_n_dim(), 11008);
    }

    #[test]
    fn test_intermediate_cache_size() {
        let config = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        // With is_act_and_mul=true: 32 tokens * 2 * 11008
        assert_eq!(config.intermediate_cache_size(32), 32 * 2 * 11008);

        let mut config_no_mul = FusedMoEConfig::new(8, 2, 4096, 11008, true);
        config_no_mul.is_act_and_mul = false;
        // Without: 32 tokens * 11008
        assert_eq!(config_no_mul.intermediate_cache_size(32), 32 * 11008);
    }
}
