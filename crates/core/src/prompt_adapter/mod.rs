//! Prompt adapter (soft prompt tuning) support.
//!
//! Prompt adapters prepend learned "virtual token" embeddings to the input
//! sequence. Unlike LoRA, which modifies weight matrices across many layers,
//! prompt adapters only affect the embedding layer — making them lightweight
//! and composable with other adapter types.
//!
//! The adapter stores a tensor of shape `[num_virtual_tokens, embedding_dim]`.
//! During prefill, the first `num_virtual_tokens` positions have their
//! embeddings **replaced** with the learned virtual embeddings.

mod loader;
mod manager;
mod types;

pub use loader::{PromptAdapterLoadError, PromptAdapterLoader};
pub use manager::{PromptAdapterManager, PromptAdapterManagerConfig, PromptAdapterManagerError};
pub use types::{PromptAdapter, PromptAdapterConfig, PromptAdapterRequest};
