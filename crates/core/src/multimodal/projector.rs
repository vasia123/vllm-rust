//! Vision-language projector implementations.
//!
//! Projectors map vision encoder outputs to the language model's hidden space.
//! Different VLMs use different projector architectures:
//! - LLaVA 1.5: MLP projector (2-layer MLP with GELU)
//! - LLaVA 1.6+: Resampler or MLP projector
//! - Qwen-VL: Resampler
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::multimodal::projector::MultimodalProjector;
//!
//! let projector = MultimodalProjector::new_mlp(vision_hidden, llm_hidden, vb)?;
//! let llm_embeddings = projector.project(&vision_embeddings)?;
//! ```

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// Type of multimodal projector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectorType {
    /// Simple linear projection.
    Linear,
    /// 2-layer MLP with GELU activation (LLaVA 1.5 style).
    Mlp,
    /// MLP with identity connection (some LLaVA variants).
    MlpGelu,
}

/// Configuration for multimodal projector.
#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    /// Type of projector.
    pub projector_type: ProjectorType,
    /// Vision encoder hidden size.
    pub vision_hidden_size: usize,
    /// Language model hidden size.
    pub llm_hidden_size: usize,
    /// Intermediate size for MLP projectors (usually = llm_hidden_size).
    pub intermediate_size: Option<usize>,
}

impl ProjectorConfig {
    /// Create a linear projector config.
    pub fn linear(vision_hidden_size: usize, llm_hidden_size: usize) -> Self {
        Self {
            projector_type: ProjectorType::Linear,
            vision_hidden_size,
            llm_hidden_size,
            intermediate_size: None,
        }
    }

    /// Create an MLP projector config (LLaVA 1.5 style).
    pub fn mlp(vision_hidden_size: usize, llm_hidden_size: usize) -> Self {
        Self {
            projector_type: ProjectorType::Mlp,
            vision_hidden_size,
            llm_hidden_size,
            intermediate_size: Some(llm_hidden_size),
        }
    }

    /// Create an MLP GELU projector config.
    pub fn mlp_gelu(
        vision_hidden_size: usize,
        llm_hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        Self {
            projector_type: ProjectorType::MlpGelu,
            vision_hidden_size,
            llm_hidden_size,
            intermediate_size: Some(intermediate_size),
        }
    }
}

/// Multimodal projector that maps vision embeddings to LLM space.
pub struct MultimodalProjector {
    projector_type: ProjectorType,
    linear1: Linear,
    linear2: Option<Linear>,
}

impl MultimodalProjector {
    /// Create a new multimodal projector from config.
    pub fn new(cfg: &ProjectorConfig, vb: VarBuilder) -> Result<Self> {
        match cfg.projector_type {
            ProjectorType::Linear => {
                Self::new_linear(cfg.vision_hidden_size, cfg.llm_hidden_size, vb)
            }
            ProjectorType::Mlp | ProjectorType::MlpGelu => {
                let intermediate = cfg.intermediate_size.unwrap_or(cfg.llm_hidden_size);
                Self::new_mlp(
                    cfg.vision_hidden_size,
                    intermediate,
                    cfg.llm_hidden_size,
                    vb,
                )
            }
        }
    }

    /// Create a simple linear projector.
    pub fn new_linear(
        vision_hidden_size: usize,
        llm_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let linear1 = linear(vision_hidden_size, llm_hidden_size, vb.pp("linear"))?;

        Ok(Self {
            projector_type: ProjectorType::Linear,
            linear1,
            linear2: None,
        })
    }

    /// Create a 2-layer MLP projector (LLaVA 1.5 style).
    pub fn new_mlp(
        vision_hidden_size: usize,
        intermediate_size: usize,
        llm_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // LLaVA uses "linear_1" and "linear_2" naming
        let linear1 = linear(vision_hidden_size, intermediate_size, vb.pp("linear_1"))?;
        let linear2 = linear(intermediate_size, llm_hidden_size, vb.pp("linear_2"))?;

        Ok(Self {
            projector_type: ProjectorType::Mlp,
            linear1,
            linear2: Some(linear2),
        })
    }

    /// Project vision embeddings to LLM embedding space.
    ///
    /// # Arguments
    /// * `vision_embeddings` - Tensor [batch, num_tokens, vision_hidden_size]
    ///
    /// # Returns
    /// Tensor [batch, num_tokens, llm_hidden_size]
    pub fn project(&self, vision_embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(vision_embeddings)?;

        match self.projector_type {
            ProjectorType::Linear => Ok(hidden),
            ProjectorType::Mlp | ProjectorType::MlpGelu => {
                let hidden = hidden.gelu_erf()?;
                self.linear2.as_ref().unwrap().forward(&hidden)
            }
        }
    }

    /// Get the projector type.
    pub fn projector_type(&self) -> ProjectorType {
        self.projector_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_linear_projector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let cfg = ProjectorConfig::linear(1024, 4096);
        let projector = MultimodalProjector::new(&cfg, vb).unwrap();

        assert_eq!(projector.projector_type(), ProjectorType::Linear);

        // Test forward
        let input = Tensor::zeros((1, 577, 1024), DType::F32, &device).unwrap();
        let output = projector.project(&input).unwrap();

        assert_eq!(output.dims(), &[1, 577, 4096]);
    }

    #[test]
    fn test_mlp_projector() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let cfg = ProjectorConfig::mlp(1024, 4096);
        let projector = MultimodalProjector::new(&cfg, vb).unwrap();

        assert_eq!(projector.projector_type(), ProjectorType::Mlp);

        // Test forward
        let input = Tensor::zeros((1, 577, 1024), DType::F32, &device).unwrap();
        let output = projector.project(&input).unwrap();

        assert_eq!(output.dims(), &[1, 577, 4096]);
    }

    #[test]
    fn test_projector_config() {
        let cfg = ProjectorConfig::linear(1024, 4096);
        assert_eq!(cfg.vision_hidden_size, 1024);
        assert_eq!(cfg.llm_hidden_size, 4096);
        assert!(cfg.intermediate_size.is_none());

        let cfg = ProjectorConfig::mlp(1024, 4096);
        assert_eq!(cfg.intermediate_size, Some(4096));

        let cfg = ProjectorConfig::mlp_gelu(1024, 4096, 2048);
        assert_eq!(cfg.intermediate_size, Some(2048));
    }
}
