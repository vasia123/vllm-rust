//! Automatic attention backend selection based on model requirements and hardware.
//!
//! Selects the best available attention backend by evaluating:
//! - Compile-time feature flags (`flash-attn`, `flashinfer`)
//! - Head dimension constraints (FlashAttention requires head_dim % 8 == 0)
//! - Model-specific needs (sliding window, MLA)
//! - GPU compute capability (FlashAttention requires SM 80+)
//!
//! The fallback chain is: FlashInfer -> FlashAttention -> Naive.

use super::backend::AttentionBackend;
use super::flash::FlashAttentionBackend;
use super::flashinfer::FlashInferBackend;
use super::naive::NaiveAttentionBackend;

/// GPU compute capability, matching NVIDIA SM versions.
///
/// Used to gate backend availability: FlashAttention-2 requires SM 80+ (Ampere),
/// FlashInfer requires SM 80+.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuCapability {
    /// Major compute capability (e.g., 8 for SM 80)
    pub major: u32,
    /// Minor compute capability (e.g., 0 for SM 80)
    pub minor: u32,
}

impl GpuCapability {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// SM 70 (Volta) - V100
    pub const SM_70: Self = Self::new(7, 0);
    /// SM 75 (Turing) - T4, RTX 2080
    pub const SM_75: Self = Self::new(7, 5);
    /// SM 80 (Ampere) - A100
    pub const SM_80: Self = Self::new(8, 0);
    /// SM 86 (Ampere) - RTX 3090
    pub const SM_86: Self = Self::new(8, 6);
    /// SM 89 (Ada Lovelace) - RTX 4090, L4
    pub const SM_89: Self = Self::new(8, 9);
    /// SM 90 (Hopper) - H100
    pub const SM_90: Self = Self::new(9, 0);

    /// Check if this capability meets a minimum requirement.
    pub fn meets_minimum(&self, minimum: &GpuCapability) -> bool {
        (self.major, self.minor) >= (minimum.major, minimum.minor)
    }
}

impl std::fmt::Display for GpuCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}.{}", self.major, self.minor)
    }
}

/// Requirements for the attention backend, derived from model configuration.
#[derive(Debug, Clone)]
pub struct AttentionRequirements {
    /// Number of query attention heads
    pub num_heads: usize,
    /// Number of key/value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Whether the model uses sliding window attention
    pub sliding_window: bool,
    /// Whether the model uses Multi-head Latent Attention (DeepSeek V2/V3)
    pub needs_mla: bool,
    /// GPU compute capability, if known
    pub gpu_capability: Option<GpuCapability>,
}

impl AttentionRequirements {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window: false,
            needs_mla: false,
            gpu_capability: None,
        }
    }

    pub fn with_sliding_window(mut self, enabled: bool) -> Self {
        self.sliding_window = enabled;
        self
    }

    pub fn with_mla(mut self, enabled: bool) -> Self {
        self.needs_mla = enabled;
        self
    }

    pub fn with_gpu_capability(mut self, capability: GpuCapability) -> Self {
        self.gpu_capability = Some(capability);
        self
    }
}

/// Reason a backend was selected or rejected during auto-selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionReason {
    /// Backend was selected as the best available option.
    Selected {
        backend: &'static str,
        reason: &'static str,
    },
    /// Backend was rejected for a specific reason.
    Rejected {
        backend: &'static str,
        reason: &'static str,
    },
}

/// Result of attention backend selection, including the chosen backend
/// and the reasoning behind the decision.
pub struct SelectionResult {
    /// The selected backend.
    pub backend: Box<dyn AttentionBackend>,
    /// Why this backend was selected.
    pub reason: &'static str,
    /// Full selection trace: each backend considered and why it was picked or rejected.
    pub trace: Vec<SelectionReason>,
}

impl std::fmt::Debug for SelectionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionResult")
            .field("backend", &self.backend.name())
            .field("reason", &self.reason)
            .field("trace", &self.trace)
            .finish()
    }
}

/// Selects the best attention backend based on model requirements and hardware.
///
/// Evaluation order (most specialized to most generic):
/// 1. FlashInfer - best decode performance with paged KV cache
/// 2. FlashAttention - best prefill performance
/// 3. Naive - always available fallback
///
/// Each candidate is checked against:
/// - Compile-time availability (feature flags)
/// - Head dimension alignment constraints
/// - Sliding window support
/// - MLA compatibility
/// - GPU compute capability
pub struct AttentionBackendSelector;

impl AttentionBackendSelector {
    /// Select the best backend for the given requirements.
    ///
    /// Returns a `SelectionResult` containing the backend and selection reasoning.
    pub fn select(requirements: &AttentionRequirements) -> SelectionResult {
        let mut trace = Vec::new();

        // MLA models use their own attention path, not the standard backends.
        // Fall through to naive as the generic backend for non-MLA operations
        // that may still occur (e.g., shared attention layers).
        if requirements.needs_mla {
            trace.push(SelectionReason::Rejected {
                backend: "flashinfer",
                reason: "MLA models use dedicated MLAAttention, not standard backends",
            });
            trace.push(SelectionReason::Rejected {
                backend: "flash-attention",
                reason: "MLA models use dedicated MLAAttention, not standard backends",
            });
            trace.push(SelectionReason::Selected {
                backend: "naive",
                reason: "fallback for MLA models (primary attention handled by MLAAttention)",
            });
            return SelectionResult {
                backend: Box::new(NaiveAttentionBackend::new()),
                reason: "fallback for MLA models (primary attention handled by MLAAttention)",
                trace,
            };
        }

        // Try FlashInfer first
        if let Some(result) = Self::try_flashinfer(requirements, &mut trace) {
            return result;
        }

        // Try FlashAttention
        if let Some(result) = Self::try_flash_attention(requirements, &mut trace) {
            return result;
        }

        // Naive always works
        trace.push(SelectionReason::Selected {
            backend: "naive",
            reason: "universal fallback",
        });
        SelectionResult {
            backend: Box::new(NaiveAttentionBackend::new()),
            reason: "universal fallback",
            trace,
        }
    }

    /// Attempt to select FlashInfer backend.
    fn try_flashinfer(
        requirements: &AttentionRequirements,
        trace: &mut Vec<SelectionReason>,
    ) -> Option<SelectionResult> {
        // Compile-time availability check
        if !cfg!(feature = "flashinfer") {
            trace.push(SelectionReason::Rejected {
                backend: "flashinfer",
                reason: "feature 'flashinfer' not enabled at compile time",
            });
            return None;
        }

        // FlashInfer requires head_dim to be a power of 2 and in {64, 128, 256}
        if !Self::is_valid_flashinfer_head_dim(requirements.head_dim) {
            trace.push(SelectionReason::Rejected {
                backend: "flashinfer",
                reason: "head_dim must be 64, 128, or 256 for FlashInfer",
            });
            return None;
        }

        // GPU compute capability check: FlashInfer requires SM 80+
        if let Some(ref cap) = requirements.gpu_capability {
            if !cap.meets_minimum(&GpuCapability::SM_80) {
                trace.push(SelectionReason::Rejected {
                    backend: "flashinfer",
                    reason: "requires GPU compute capability SM 80+ (Ampere or newer)",
                });
                return None;
            }
        }

        // FlashInfer supports sliding window natively
        let reason = if requirements.sliding_window {
            "best decode performance with paged KV cache and sliding window support"
        } else {
            "best decode performance with paged KV cache"
        };

        trace.push(SelectionReason::Selected {
            backend: "flashinfer",
            reason,
        });
        Some(SelectionResult {
            backend: Box::new(FlashInferBackend::new()),
            reason,
            trace: trace.clone(),
        })
    }

    /// Attempt to select FlashAttention backend.
    fn try_flash_attention(
        requirements: &AttentionRequirements,
        trace: &mut Vec<SelectionReason>,
    ) -> Option<SelectionResult> {
        // Compile-time availability check
        if !cfg!(feature = "flash-attn") {
            trace.push(SelectionReason::Rejected {
                backend: "flash-attention",
                reason: "feature 'flash-attn' not enabled at compile time",
            });
            return None;
        }

        // FlashAttention-2 requires head_dim % 8 == 0 and head_dim <= 256
        if !Self::is_valid_flash_attn_head_dim(requirements.head_dim) {
            trace.push(SelectionReason::Rejected {
                backend: "flash-attention",
                reason: "head_dim must be divisible by 8 and <= 256 for FlashAttention",
            });
            return None;
        }

        // GPU compute capability check: FlashAttention-2 requires SM 80+
        if let Some(ref cap) = requirements.gpu_capability {
            if !cap.meets_minimum(&GpuCapability::SM_80) {
                trace.push(SelectionReason::Rejected {
                    backend: "flash-attention",
                    reason: "requires GPU compute capability SM 80+ (Ampere or newer)",
                });
                return None;
            }
        }

        // FlashAttention-2 does not natively support sliding window in our integration.
        // It can still be used; the model applies the window mask separately.
        let reason = if requirements.sliding_window {
            "optimized prefill attention (sliding window applied via mask)"
        } else {
            "optimized prefill attention"
        };

        trace.push(SelectionReason::Selected {
            backend: "flash-attention",
            reason,
        });
        Some(SelectionResult {
            backend: Box::new(FlashAttentionBackend::new()),
            reason,
            trace: trace.clone(),
        })
    }

    /// Check if head_dim is valid for FlashAttention-2.
    ///
    /// FlashAttention-2 requires head_dim % 8 == 0 and head_dim <= 256.
    fn is_valid_flash_attn_head_dim(head_dim: usize) -> bool {
        head_dim > 0 && head_dim % 8 == 0 && head_dim <= 256
    }

    /// Check if head_dim is valid for FlashInfer.
    ///
    /// FlashInfer supports head dimensions that are powers of 2, specifically
    /// {64, 128, 256} in current implementations.
    fn is_valid_flashinfer_head_dim(head_dim: usize) -> bool {
        matches!(head_dim, 64 | 128 | 256)
    }

    /// Convenience: select and return just the backend without the trace.
    pub fn select_backend(requirements: &AttentionRequirements) -> Box<dyn AttentionBackend> {
        Self::select(requirements).backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GpuCapability tests
    // =========================================================================

    #[test]
    fn test_gpu_capability_ordering() {
        assert!(GpuCapability::SM_70 < GpuCapability::SM_75);
        assert!(GpuCapability::SM_75 < GpuCapability::SM_80);
        assert!(GpuCapability::SM_80 < GpuCapability::SM_86);
        assert!(GpuCapability::SM_86 < GpuCapability::SM_89);
        assert!(GpuCapability::SM_89 < GpuCapability::SM_90);
    }

    #[test]
    fn test_gpu_capability_meets_minimum() {
        let ampere = GpuCapability::SM_80;
        let hopper = GpuCapability::SM_90;
        let volta = GpuCapability::SM_70;

        assert!(ampere.meets_minimum(&GpuCapability::SM_80));
        assert!(hopper.meets_minimum(&GpuCapability::SM_80));
        assert!(!volta.meets_minimum(&GpuCapability::SM_80));
    }

    #[test]
    fn test_gpu_capability_display() {
        assert_eq!(GpuCapability::SM_80.to_string(), "SM 8.0");
        assert_eq!(GpuCapability::SM_90.to_string(), "SM 9.0");
        assert_eq!(GpuCapability::new(7, 5).to_string(), "SM 7.5");
    }

    #[test]
    fn test_gpu_capability_equality() {
        let a = GpuCapability::new(8, 0);
        let b = GpuCapability::SM_80;
        assert_eq!(a, b);
    }

    // =========================================================================
    // AttentionRequirements tests
    // =========================================================================

    #[test]
    fn test_requirements_builder() {
        let req = AttentionRequirements::new(32, 8, 128)
            .with_sliding_window(true)
            .with_mla(false)
            .with_gpu_capability(GpuCapability::SM_80);

        assert_eq!(req.num_heads, 32);
        assert_eq!(req.num_kv_heads, 8);
        assert_eq!(req.head_dim, 128);
        assert!(req.sliding_window);
        assert!(!req.needs_mla);
        assert_eq!(req.gpu_capability, Some(GpuCapability::SM_80));
    }

    #[test]
    fn test_requirements_defaults() {
        let req = AttentionRequirements::new(32, 8, 128);

        assert!(!req.sliding_window);
        assert!(!req.needs_mla);
        assert!(req.gpu_capability.is_none());
    }

    // =========================================================================
    // Head dimension validation tests
    // =========================================================================

    #[test]
    fn test_flash_attn_head_dim_valid() {
        // Common valid head dims
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(64));
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(128));
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(256));
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(96));
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(80));
        assert!(AttentionBackendSelector::is_valid_flash_attn_head_dim(8));
    }

    #[test]
    fn test_flash_attn_head_dim_invalid() {
        assert!(!AttentionBackendSelector::is_valid_flash_attn_head_dim(0));
        assert!(!AttentionBackendSelector::is_valid_flash_attn_head_dim(7));
        assert!(!AttentionBackendSelector::is_valid_flash_attn_head_dim(65));
        assert!(!AttentionBackendSelector::is_valid_flash_attn_head_dim(257));
        assert!(!AttentionBackendSelector::is_valid_flash_attn_head_dim(512));
    }

    #[test]
    fn test_flashinfer_head_dim_valid() {
        assert!(AttentionBackendSelector::is_valid_flashinfer_head_dim(64));
        assert!(AttentionBackendSelector::is_valid_flashinfer_head_dim(128));
        assert!(AttentionBackendSelector::is_valid_flashinfer_head_dim(256));
    }

    #[test]
    fn test_flashinfer_head_dim_invalid() {
        assert!(!AttentionBackendSelector::is_valid_flashinfer_head_dim(0));
        assert!(!AttentionBackendSelector::is_valid_flashinfer_head_dim(32));
        assert!(!AttentionBackendSelector::is_valid_flashinfer_head_dim(96));
        assert!(!AttentionBackendSelector::is_valid_flashinfer_head_dim(160));
        assert!(!AttentionBackendSelector::is_valid_flashinfer_head_dim(512));
    }

    // =========================================================================
    // Selection logic tests (CPU-only, no feature flags enabled)
    //
    // Without flash-attn/flashinfer features, the selector must always
    // fall back to naive. We verify the rejection trace is correct.
    // =========================================================================

    #[test]
    fn test_select_fallback_to_naive_no_features() {
        // Standard LLaMA-like config
        let req = AttentionRequirements::new(32, 8, 128);
        let result = AttentionBackendSelector::select(&req);

        // Without features compiled, both FlashInfer and FlashAttention are rejected
        assert_eq!(result.backend.name(), expected_naive_name());
        assert_eq!(result.reason, "universal fallback");
    }

    #[test]
    fn test_select_with_sliding_window() {
        // Mistral-like config with sliding window
        let req = AttentionRequirements::new(32, 8, 128).with_sliding_window(true);
        let result = AttentionBackendSelector::select(&req);

        // Verify backend is valid (the specific backend depends on compiled features)
        assert!(!result.backend.name().is_empty());
        assert!(!result.trace.is_empty());
    }

    #[test]
    fn test_select_mla_model() {
        // DeepSeek V2 MLA config
        let req = AttentionRequirements::new(128, 1, 192).with_mla(true);
        let result = AttentionBackendSelector::select(&req);

        // MLA models always get naive backend for standard attention operations
        assert_eq!(result.backend.name(), "naive");
        assert!(result.reason.contains("MLA"));

        // Trace should show all standard backends rejected for MLA
        let rejected_count = result
            .trace
            .iter()
            .filter(|r| matches!(r, SelectionReason::Rejected { .. }))
            .count();
        assert_eq!(
            rejected_count, 2,
            "FlashInfer and FlashAttention should both be rejected for MLA"
        );
    }

    #[test]
    fn test_select_odd_head_dim() {
        // Unusual head_dim that fails FlashAttention alignment
        let req = AttentionRequirements::new(16, 16, 65);
        let result = AttentionBackendSelector::select(&req);

        // Both FlashInfer (not 64/128/256) and FlashAttention (not % 8) should reject
        let flash_rejected = result.trace.iter().any(|r| {
            matches!(
                r,
                SelectionReason::Rejected { backend: "flash-attention", reason }
                if reason.contains("head_dim")
            )
        });

        let flashinfer_rejected = result.trace.iter().any(|r| {
            matches!(
                r,
                SelectionReason::Rejected { backend: "flashinfer", reason }
                if reason.contains("head_dim")
            )
        });

        // At minimum, head_dim=65 should be rejected by both specialized backends
        // (they may also be rejected for missing feature flags first)
        let any_head_dim_rejection = flash_rejected || flashinfer_rejected;
        let any_feature_rejection = result.trace.iter().any(|r| {
            matches!(
                r,
                SelectionReason::Rejected { reason, .. }
                if reason.contains("feature") || reason.contains("head_dim")
            )
        });
        assert!(any_head_dim_rejection || any_feature_rejection);

        // Should still get a valid backend (naive)
        assert_eq!(result.backend.name(), expected_naive_name());
    }

    #[test]
    fn test_select_gpu_too_old_for_flash() {
        // Volta GPU (SM 7.0) - too old for FlashAttention/FlashInfer
        let req = AttentionRequirements::new(32, 8, 128).with_gpu_capability(GpuCapability::SM_70);
        let result = AttentionBackendSelector::select(&req);

        // Should still get a valid backend
        assert!(!result.backend.name().is_empty());

        // If features are enabled, we expect SM rejection; if not, feature rejection.
        // Either way, specialized backends should be in the trace.
        assert!(
            result.trace.len() >= 2,
            "should have at least two rejection entries plus selection"
        );
    }

    #[test]
    fn test_select_modern_gpu_standard_config() {
        // H100 (SM 9.0) with standard LLaMA config
        let req = AttentionRequirements::new(32, 8, 128).with_gpu_capability(GpuCapability::SM_90);
        let result = AttentionBackendSelector::select(&req);

        assert!(!result.backend.name().is_empty());
        // On a modern GPU with correct head_dim, the best available backend should be selected
    }

    #[test]
    fn test_select_trace_completeness() {
        let req = AttentionRequirements::new(32, 8, 128);
        let result = AttentionBackendSelector::select(&req);

        // The trace should contain at least one Selected entry
        let selected_count = result
            .trace
            .iter()
            .filter(|r| matches!(r, SelectionReason::Selected { .. }))
            .count();
        assert_eq!(selected_count, 1, "exactly one backend should be selected");
    }

    #[test]
    fn test_select_convenience_method() {
        let req = AttentionRequirements::new(32, 8, 128);
        let backend = AttentionBackendSelector::select_backend(&req);
        assert!(!backend.name().is_empty());
    }

    // =========================================================================
    // Fallback chain tests
    // =========================================================================

    #[test]
    fn test_fallback_chain_head_dim_32() {
        // head_dim=32: rejected by FlashInfer (not 64/128/256),
        // accepted by FlashAttention (32 % 8 == 0), or naive fallback
        let req = AttentionRequirements::new(16, 4, 32);
        let result = AttentionBackendSelector::select(&req);

        let fi_rejected = result.trace.iter().any(|r| {
            matches!(
                r,
                SelectionReason::Rejected {
                    backend: "flashinfer",
                    ..
                }
            )
        });
        assert!(fi_rejected, "FlashInfer should reject head_dim=32");

        // Result should be either flash-attention or naive, both valid
        assert!(!result.backend.name().is_empty());
    }

    #[test]
    fn test_fallback_chain_head_dim_96() {
        // head_dim=96: rejected by FlashInfer (not 64/128/256),
        // accepted by FlashAttention (96 % 8 == 0), or naive fallback
        let req = AttentionRequirements::new(32, 8, 96);
        let result = AttentionBackendSelector::select(&req);

        let fi_rejected = result.trace.iter().any(|r| {
            matches!(
                r,
                SelectionReason::Rejected {
                    backend: "flashinfer",
                    ..
                }
            )
        });
        assert!(fi_rejected, "FlashInfer should reject head_dim=96");

        assert!(!result.backend.name().is_empty());
    }

    #[test]
    fn test_fallback_chain_mla_overrides_everything() {
        // MLA should always result in naive, regardless of other params
        let req = AttentionRequirements::new(32, 8, 128)
            .with_sliding_window(true)
            .with_mla(true)
            .with_gpu_capability(GpuCapability::SM_90);
        let result = AttentionBackendSelector::select(&req);

        assert_eq!(result.backend.name(), "naive");
    }

    // =========================================================================
    // Feature-flag-aware tests
    //
    // These tests verify behavior that changes based on compile-time features.
    // =========================================================================

    #[cfg(not(any(feature = "flash-attn", feature = "flashinfer")))]
    #[test]
    fn test_no_features_always_naive() {
        let configs = [
            AttentionRequirements::new(32, 8, 128),
            AttentionRequirements::new(64, 8, 64),
            AttentionRequirements::new(16, 16, 256),
        ];

        for req in &configs {
            let result = AttentionBackendSelector::select(req);
            assert_eq!(
                result.backend.name(),
                "naive",
                "without features, backend must be naive for heads={} kv={} dim={}",
                req.num_heads,
                req.num_kv_heads,
                req.head_dim
            );
        }
    }

    #[cfg(all(feature = "flash-attn", not(feature = "flashinfer")))]
    #[test]
    fn test_flash_attn_only_selects_flash() {
        let req = AttentionRequirements::new(32, 8, 128);
        let result = AttentionBackendSelector::select(&req);
        assert_eq!(result.backend.name(), "flash-attention");
    }

    #[cfg(all(feature = "flash-attn", not(feature = "flashinfer")))]
    #[test]
    fn test_flash_attn_rejects_bad_head_dim() {
        let req = AttentionRequirements::new(32, 8, 65);
        let result = AttentionBackendSelector::select(&req);
        // FlashAttention rejected, falls through to naive
        assert_eq!(result.backend.name(), "naive");
    }

    #[cfg(feature = "flashinfer")]
    #[test]
    fn test_flashinfer_selected_for_standard_config() {
        let req = AttentionRequirements::new(32, 8, 128);
        let result = AttentionBackendSelector::select(&req);
        assert_eq!(result.backend.name(), "flashinfer");
    }

    #[cfg(feature = "flashinfer")]
    #[test]
    fn test_flashinfer_rejected_bad_head_dim_falls_to_flash_or_naive() {
        let req = AttentionRequirements::new(32, 8, 96);
        let result = AttentionBackendSelector::select(&req);
        // FlashInfer rejected for head_dim=96, should get flash-attn or naive
        assert_ne!(result.backend.name(), "flashinfer");
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_select_single_head() {
        let req = AttentionRequirements::new(1, 1, 64);
        let result = AttentionBackendSelector::select(&req);
        assert!(!result.backend.name().is_empty());
    }

    #[test]
    fn test_select_many_heads() {
        // Large model config (e.g., GPT-3 175B scale)
        let req = AttentionRequirements::new(96, 96, 128);
        let result = AttentionBackendSelector::select(&req);
        assert!(!result.backend.name().is_empty());
    }

    #[test]
    fn test_select_gqa_config() {
        // GQA with 4:1 ratio
        let req = AttentionRequirements::new(32, 8, 128);
        let result = AttentionBackendSelector::select(&req);
        assert!(!result.backend.name().is_empty());
    }

    #[test]
    fn test_select_mqa_config() {
        // MQA: single KV head
        let req = AttentionRequirements::new(32, 1, 128);
        let result = AttentionBackendSelector::select(&req);
        assert!(!result.backend.name().is_empty());
    }

    /// Helper: expected naive backend name depends on whether features are enabled,
    /// because FlashAttentionBackend and FlashInferBackend with feature disabled
    /// have fallback names, but NaiveAttentionBackend is always "naive".
    fn expected_naive_name() -> &'static str {
        "naive"
    }
}
