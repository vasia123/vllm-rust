//! LoRA (Low-Rank Adaptation) support for efficient fine-tuning.
//!
//! This module provides infrastructure for loading and applying LoRA adapters
//! to base models, enabling per-request adapter selection without model reload.

mod context;
mod linear;
mod loader;
mod manager;
mod types;

pub use context::LoraContext;
pub use linear::LinearWithLora;
pub use loader::{LoraLoadError, LoraLoader};
pub use manager::{LoraManager, LoraManagerConfig, LoraManagerError};
pub use types::{LoraAdapter, LoraConfig, LoraModel, LoraRequest};
