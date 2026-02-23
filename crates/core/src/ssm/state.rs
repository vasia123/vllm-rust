//! SSM state management for Mamba-style models.
//!
//! Unlike transformer models that use KV cache, SSM models maintain
//! recurrent state tensors that get updated at each generation step.
//! Each layer has two state components:
//!
//! - **SSM state**: the hidden recurrence state `h` (shape: [batch, d_inner, d_state])
//! - **Conv state**: the causal convolution buffer (shape: [batch, d_inner, d_conv-1])

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use thiserror::Error;

/// Per-layer SSM recurrence state.
///
/// Holds the hidden state tensor `h` of shape [batch, d_inner, d_state].
#[derive(Debug, Clone)]
pub struct SSMState {
    pub tensor: Tensor,
}

impl SSMState {
    /// Create a zero-initialized SSM state.
    pub fn zeros(d_inner: usize, d_state: usize, dtype: DType, device: &Device) -> Result<Self> {
        let tensor = Tensor::zeros((1, d_inner, d_state), dtype, device)?;
        Ok(Self { tensor })
    }

    /// Reset this state to zeros in-place (returns a new zeroed state with same shape).
    pub fn reset(&mut self) -> Result<()> {
        let dims = self.tensor.dims().to_vec();
        let dtype = self.tensor.dtype();
        let device = self.tensor.device().clone();
        self.tensor = Tensor::zeros(dims.as_slice(), dtype, &device)?;
        Ok(())
    }
}

/// Per-layer causal convolution state buffer.
///
/// Holds the convolution history of shape [batch, d_inner, d_conv-1].
/// During decode, new values are shifted in and the oldest value is dropped.
#[derive(Debug, Clone)]
pub struct SSMConvState {
    pub tensor: Tensor,
}

impl SSMConvState {
    /// Create a zero-initialized conv state.
    pub fn zeros(d_inner: usize, d_conv: usize, dtype: DType, device: &Device) -> Result<Self> {
        // Conv state holds d_conv-1 past values for causal convolution
        let conv_state_len = d_conv.saturating_sub(1);
        let tensor = Tensor::zeros((1, d_inner, conv_state_len), dtype, device)?;
        Ok(Self { tensor })
    }

    /// Reset this state to zeros in-place.
    pub fn reset(&mut self) -> Result<()> {
        let dims = self.tensor.dims().to_vec();
        let dtype = self.tensor.dtype();
        let device = self.tensor.device().clone();
        self.tensor = Tensor::zeros(dims.as_slice(), dtype, &device)?;
        Ok(())
    }
}

/// Combined per-request SSM state across all layers.
///
/// Each layer has both an SSM recurrence state and a convolution state buffer.
#[derive(Debug, Clone)]
pub struct RequestSSMState {
    pub ssm_states: Vec<SSMState>,
    pub conv_states: Vec<SSMConvState>,
}

#[derive(Debug, Error)]
pub enum SSMStateError {
    #[error("request {0} not found in state manager")]
    RequestNotFound(u64),
    #[error("request {0} already has allocated state")]
    RequestAlreadyExists(u64),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Manages SSM states across layers and sequences.
///
/// Each active request gets a set of per-layer SSM and conv states.
/// States are allocated when a request begins and freed when it completes.
pub struct SSMStateManager {
    num_layers: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    /// Channel count for conv state buffer. Equals `d_inner` for Mamba-1;
    /// for Mamba-2 it is `d_inner + 2 * n_groups * d_state` (the conv_dim).
    d_conv_channels: usize,
    dtype: DType,
    device: Device,
    states: HashMap<u64, RequestSSMState>,
}

impl SSMStateManager {
    /// Create a new state manager.
    ///
    /// # Arguments
    /// * `num_layers` - Number of Mamba blocks in the model
    /// * `d_inner` - Inner dimension (expand * hidden_size)
    /// * `d_state` - SSM state dimension (typically 16)
    /// * `d_conv` - Convolution kernel size (typically 4)
    /// * `dtype` - Data type for state tensors
    /// * `device` - Device for state tensors
    pub fn new(
        num_layers: usize,
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            num_layers,
            d_inner,
            d_state,
            d_conv,
            d_conv_channels: d_inner,
            dtype,
            device,
            states: HashMap::new(),
        }
    }

    /// Create a state manager where the conv state has a different channel
    /// count from the SSM state.
    ///
    /// Used by Mamba-2 where `conv_dim = d_inner + 2 * n_groups * d_state`.
    ///
    /// # Arguments
    /// * `d_conv_channels` - Number of channels for conv state (overrides `d_inner`)
    pub fn new_with_conv_channels(
        num_layers: usize,
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        d_conv_channels: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            num_layers,
            d_inner,
            d_state,
            d_conv,
            d_conv_channels,
            dtype,
            device,
            states: HashMap::new(),
        }
    }

    /// Allocate states for a new request.
    ///
    /// Creates zero-initialized SSM and conv states for each layer.
    /// Returns a mutable reference to the allocated states.
    pub fn allocate_state(
        &mut self,
        request_id: u64,
    ) -> std::result::Result<&mut RequestSSMState, SSMStateError> {
        if self.states.contains_key(&request_id) {
            return Err(SSMStateError::RequestAlreadyExists(request_id));
        }

        let mut ssm_states = Vec::with_capacity(self.num_layers);
        let mut conv_states = Vec::with_capacity(self.num_layers);

        for _ in 0..self.num_layers {
            ssm_states.push(SSMState::zeros(
                self.d_inner,
                self.d_state,
                self.dtype,
                &self.device,
            )?);
            conv_states.push(SSMConvState::zeros(
                self.d_conv_channels,
                self.d_conv,
                self.dtype,
                &self.device,
            )?);
        }

        self.states.insert(
            request_id,
            RequestSSMState {
                ssm_states,
                conv_states,
            },
        );

        // Safe: we just inserted the key above
        Ok(self.states.get_mut(&request_id).expect("just inserted"))
    }

    /// Get mutable reference to states for an existing request.
    pub fn get_state(&mut self, request_id: u64) -> Option<&mut RequestSSMState> {
        self.states.get_mut(&request_id)
    }

    /// Free states for a completed request.
    pub fn free_state(&mut self, request_id: u64) {
        self.states.remove(&request_id);
    }

    /// Reset all states for a request to zeros.
    pub fn reset_state(&mut self, request_id: u64) -> std::result::Result<(), SSMStateError> {
        let state = self
            .states
            .get_mut(&request_id)
            .ok_or(SSMStateError::RequestNotFound(request_id))?;

        for ssm_state in &mut state.ssm_states {
            ssm_state.reset()?;
        }
        for conv_state in &mut state.conv_states {
            conv_state.reset()?;
        }
        Ok(())
    }

    /// Number of active requests with allocated state.
    pub fn num_active_requests(&self) -> usize {
        self.states.len()
    }

    /// Check if a request has allocated state.
    pub fn has_state(&self, request_id: u64) -> bool {
        self.states.contains_key(&request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_NUM_LAYERS: usize = 2;
    const TEST_D_INNER: usize = 128;
    const TEST_D_STATE: usize = 8;
    const TEST_D_CONV: usize = 4;

    fn create_test_manager() -> SSMStateManager {
        SSMStateManager::new(
            TEST_NUM_LAYERS,
            TEST_D_INNER,
            TEST_D_STATE,
            TEST_D_CONV,
            DType::F32,
            Device::Cpu,
        )
    }

    #[test]
    fn allocate_state_creates_correct_shapes() {
        let mut mgr = create_test_manager();
        let state = mgr.allocate_state(1).expect("allocate");

        assert_eq!(state.ssm_states.len(), TEST_NUM_LAYERS);
        assert_eq!(state.conv_states.len(), TEST_NUM_LAYERS);

        for ssm in &state.ssm_states {
            assert_eq!(ssm.tensor.dims(), &[1, TEST_D_INNER, TEST_D_STATE]);
        }

        for conv in &state.conv_states {
            assert_eq!(conv.tensor.dims(), &[1, TEST_D_INNER, TEST_D_CONV - 1]);
        }
    }

    #[test]
    fn allocate_duplicate_request_fails() {
        let mut mgr = create_test_manager();
        mgr.allocate_state(1).expect("first allocate");

        let result = mgr.allocate_state(1);
        assert!(result.is_err());
        match result.unwrap_err() {
            SSMStateError::RequestAlreadyExists(id) => assert_eq!(id, 1),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn get_state_returns_allocated() {
        let mut mgr = create_test_manager();
        mgr.allocate_state(42).expect("allocate");

        let state = mgr.get_state(42);
        assert!(state.is_some());

        let state = state.expect("should exist");
        assert_eq!(state.ssm_states.len(), TEST_NUM_LAYERS);
    }

    #[test]
    fn get_state_returns_none_for_unknown() {
        let mut mgr = create_test_manager();
        assert!(mgr.get_state(999).is_none());
    }

    #[test]
    fn free_state_removes_request() {
        let mut mgr = create_test_manager();
        mgr.allocate_state(1).expect("allocate");
        assert!(mgr.has_state(1));

        mgr.free_state(1);
        assert!(!mgr.has_state(1));
        assert!(mgr.get_state(1).is_none());
    }

    #[test]
    fn free_nonexistent_is_noop() {
        let mut mgr = create_test_manager();
        mgr.free_state(999); // should not panic
    }

    #[test]
    fn reset_state_zeros_all_layers() {
        let mut mgr = create_test_manager();
        mgr.allocate_state(1).expect("allocate");

        // Modify the state to non-zero values
        {
            let state = mgr.get_state(1).expect("should exist");
            state.ssm_states[0].tensor =
                Tensor::ones((1, TEST_D_INNER, TEST_D_STATE), DType::F32, &Device::Cpu)
                    .expect("ones");
            state.conv_states[0].tensor =
                Tensor::ones((1, TEST_D_INNER, TEST_D_CONV - 1), DType::F32, &Device::Cpu)
                    .expect("ones");
        }

        // Verify non-zero
        {
            let state = mgr.get_state(1).expect("should exist");
            let sum: f32 = state.ssm_states[0]
                .tensor
                .sum_all()
                .expect("sum")
                .to_scalar()
                .expect("scalar");
            assert!(sum > 0.0, "state should be non-zero before reset");
        }

        // Reset
        mgr.reset_state(1).expect("reset");

        // Verify zeroed
        {
            let state = mgr.get_state(1).expect("should exist");
            for ssm in &state.ssm_states {
                let sum: f32 = ssm
                    .tensor
                    .sum_all()
                    .expect("sum")
                    .to_scalar()
                    .expect("scalar");
                assert_eq!(sum, 0.0, "SSM state should be zero after reset");
            }
            for conv in &state.conv_states {
                let sum: f32 = conv
                    .tensor
                    .sum_all()
                    .expect("sum")
                    .to_scalar()
                    .expect("scalar");
                assert_eq!(sum, 0.0, "conv state should be zero after reset");
            }
        }
    }

    #[test]
    fn reset_nonexistent_fails() {
        let mut mgr = create_test_manager();
        let result = mgr.reset_state(999);
        assert!(result.is_err());
        match result.unwrap_err() {
            SSMStateError::RequestNotFound(id) => assert_eq!(id, 999),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn multiple_requests_independent() {
        let mut mgr = create_test_manager();
        mgr.allocate_state(1).expect("allocate 1");
        mgr.allocate_state(2).expect("allocate 2");

        assert_eq!(mgr.num_active_requests(), 2);
        assert!(mgr.has_state(1));
        assert!(mgr.has_state(2));

        mgr.free_state(1);
        assert_eq!(mgr.num_active_requests(), 1);
        assert!(!mgr.has_state(1));
        assert!(mgr.has_state(2));
    }

    #[test]
    fn ssm_state_zeros_correct_dtype() {
        let state = SSMState::zeros(64, 16, DType::F32, &Device::Cpu).expect("zeros");
        assert_eq!(state.tensor.dtype(), DType::F32);
        assert_eq!(state.tensor.dims(), &[1, 64, 16]);
    }

    #[test]
    fn conv_state_zeros_correct_shape() {
        let state = SSMConvState::zeros(64, 4, DType::F32, &Device::Cpu).expect("zeros");
        assert_eq!(state.tensor.dims(), &[1, 64, 3]); // d_conv-1 = 3
    }

    #[test]
    fn conv_state_d_conv_1_gives_zero_buffer() {
        // Edge case: d_conv=1 means no history needed
        let state = SSMConvState::zeros(64, 1, DType::F32, &Device::Cpu).expect("zeros");
        assert_eq!(state.tensor.dims(), &[1, 64, 0]);
    }
}
