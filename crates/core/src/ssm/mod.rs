//! State Space Model (SSM) infrastructure for Mamba-style models.
//!
//! SSM models use recurrent state instead of KV cache. Each layer
//! maintains a state tensor that gets updated during generation.

pub mod selective_scan;
pub mod state;

pub use selective_scan::selective_scan;
pub use state::{SSMState, SSMStateManager};
