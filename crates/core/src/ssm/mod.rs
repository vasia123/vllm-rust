//! State Space Model (SSM) infrastructure for Mamba-style models.
//!
//! SSM models use recurrent state instead of KV cache. Each layer
//! maintains a state tensor that gets updated during generation.

pub mod causal_conv1d;
pub mod gated_layer_norm;
pub mod selective_scan;
pub mod ssd;
pub mod state;

pub use causal_conv1d::{causal_conv1d_decode, causal_conv1d_prefill};
pub use gated_layer_norm::rms_norm_gated;
pub use selective_scan::selective_scan;
pub use ssd::{ssd_chunk_scan, ssd_sequential};
pub use state::{SSMState, SSMStateManager};
