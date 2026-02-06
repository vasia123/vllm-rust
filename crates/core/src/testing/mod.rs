//! Shared test utilities for vllm-core.
//!
//! This module provides reusable test helpers, mock implementations,
//! and tiny model configurations for integration testing.

mod mock_model;
mod tiny_config;

pub use mock_model::MockModelForward;
pub use tiny_config::{tiny_bert_config, tiny_llama_config};
