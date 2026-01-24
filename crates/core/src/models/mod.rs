pub mod llama;
pub mod qwen3;

pub use llama::LlamaForCausalLM;
pub use qwen3::Qwen3ForCausalLM;

use candle_nn::VarBuilder;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::engine::ModelForward;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error("model load error: {0}")]
    Load(#[from] candle_core::Error),
}

/// Construct the appropriate model from config.architectures[0].
pub fn from_config(
    cfg: &ModelConfig,
    vb: VarBuilder,
) -> Result<Box<dyn ModelForward>, ModelError> {
    let arch = cfg
        .architectures
        .first()
        .ok_or_else(|| ModelError::UnsupportedArchitecture("empty architectures list".into()))?;
    match arch.as_str() {
        "Qwen3ForCausalLM" => Ok(Box::new(Qwen3ForCausalLM::new(cfg, vb)?)),
        "LlamaForCausalLM" => Ok(Box::new(LlamaForCausalLM::new(cfg, vb)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.into())),
    }
}
