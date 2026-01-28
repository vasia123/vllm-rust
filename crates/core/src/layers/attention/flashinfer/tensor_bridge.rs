//! Tensor conversion between candle and FlashInfer.
//!
//! Provides utilities for converting candle Tensors to raw pointers
//! and managing CUDA stream synchronization.

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "flashinfer")]
use candle_core::DType;

/// Extract raw CUDA device pointer from a candle Tensor.
///
/// # Safety
/// The returned pointer is only valid as long as the Tensor is alive.
/// Caller must ensure proper synchronization with CUDA operations.
#[cfg(feature = "flashinfer")]
pub fn tensor_to_cuda_ptr<T: candle_core::cuda_backend::WrapErr>(
    tensor: &Tensor,
) -> Result<*const T> {
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::Storage;

    let storage = tensor.storage_and_layout().0;
    match &*storage {
        Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<T>()?;
            Ok(slice.device_ptr() as *const T)
        }
        _ => Err(candle_core::Error::Msg(
            "Tensor must be on CUDA device".to_string(),
        )),
    }
}

/// Extract mutable raw CUDA device pointer from a candle Tensor.
#[cfg(feature = "flashinfer")]
pub fn tensor_to_cuda_ptr_mut<T: candle_core::cuda_backend::WrapErr>(
    tensor: &mut Tensor,
) -> Result<*mut T> {
    tensor_to_cuda_ptr::<T>(tensor).map(|p| p as *mut T)
}

/// Trait for types that can be used with FlashInfer operations.
#[cfg(feature = "flashinfer")]
pub trait FlashInferDtype: Sized {
    /// The candle DType corresponding to this type.
    fn candle_dtype() -> DType;

    /// Check if a tensor has the correct dtype for this operation.
    fn check_dtype(tensor: &Tensor) -> Result<()> {
        if tensor.dtype() != Self::candle_dtype() {
            return Err(candle_core::Error::Msg(format!(
                "Expected dtype {:?}, got {:?}",
                Self::candle_dtype(),
                tensor.dtype()
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "flashinfer")]
impl FlashInferDtype for half::f16 {
    fn candle_dtype() -> DType {
        DType::F16
    }
}

#[cfg(feature = "flashinfer")]
impl FlashInferDtype for half::bf16 {
    fn candle_dtype() -> DType {
        DType::BF16
    }
}

#[cfg(feature = "flashinfer")]
impl FlashInferDtype for f32 {
    fn candle_dtype() -> DType {
        DType::F32
    }
}

/// Convert candle DType to flashinfer-rs DType equivalent.
#[cfg(feature = "flashinfer")]
pub fn candle_to_flashinfer_dtype(dtype: DType) -> Result<flashinfer_rs::DType> {
    match dtype {
        DType::F16 => Ok(flashinfer_rs::DType::Float16),
        DType::BF16 => Ok(flashinfer_rs::DType::BFloat16),
        DType::F32 => Ok(flashinfer_rs::DType::Float32),
        _ => Err(candle_core::Error::Msg(format!(
            "Unsupported dtype for FlashInfer: {:?}",
            dtype
        ))),
    }
}

/// Check if a device is CUDA and suitable for FlashInfer.
pub fn is_cuda_available(device: &Device) -> bool {
    device.is_cuda()
}

/// Get CUDA device ordinal from candle Device.
#[cfg(feature = "flashinfer")]
pub fn get_cuda_device_ordinal(device: &Device) -> Result<i32> {
    match device {
        Device::Cuda(cuda_device) => Ok(cuda_device.ordinal() as i32),
        _ => Err(candle_core::Error::Msg("Device must be CUDA".to_string())),
    }
}

/// Ensure a tensor is contiguous for FlashInfer operations.
pub fn ensure_contiguous(tensor: &Tensor) -> Result<Tensor> {
    if tensor.is_contiguous() {
        Ok(tensor.clone())
    } else {
        tensor.contiguous()
    }
}

/// Verify tensor shapes match expected attention layout.
///
/// FlashInfer expects:
/// - Q: [batch_size, num_qo_heads, head_dim] for decode
/// - Q: [total_tokens, num_qo_heads, head_dim] for prefill
/// - KV Cache: [num_pages, page_size, num_kv_heads, head_dim]
pub fn verify_attention_shapes(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_decode: bool,
) -> Result<()> {
    // Q shape check
    let q_dims = q.dims();
    if q_dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "Q must be 3D, got {}D",
            q_dims.len()
        )));
    }
    if q_dims[1] != num_qo_heads {
        return Err(candle_core::Error::Msg(format!(
            "Q num_heads mismatch: expected {}, got {}",
            num_qo_heads, q_dims[1]
        )));
    }
    if q_dims[2] != head_dim {
        return Err(candle_core::Error::Msg(format!(
            "Q head_dim mismatch: expected {}, got {}",
            head_dim, q_dims[2]
        )));
    }

    // KV cache shape check: [num_pages, page_size, num_kv_heads, head_dim]
    let k_dims = k_cache.dims();
    let v_dims = v_cache.dims();

    if k_dims.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "K cache must be 4D, got {}D",
            k_dims.len()
        )));
    }
    if v_dims.len() != 4 {
        return Err(candle_core::Error::Msg(format!(
            "V cache must be 4D, got {}D",
            v_dims.len()
        )));
    }

    if k_dims[2] != num_kv_heads {
        return Err(candle_core::Error::Msg(format!(
            "K cache num_kv_heads mismatch: expected {}, got {}",
            num_kv_heads, k_dims[2]
        )));
    }
    if k_dims[3] != head_dim {
        return Err(candle_core::Error::Msg(format!(
            "K cache head_dim mismatch: expected {}, got {}",
            head_dim, k_dims[3]
        )));
    }

    // For decode, batch_size (Q dim 0) should match intended batch
    if is_decode && q_dims[0] == 0 {
        return Err(candle_core::Error::Msg(
            "Empty batch for decode attention".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_is_cuda_available() {
        let cpu_device = Device::Cpu;
        assert!(!is_cuda_available(&cpu_device));
    }

    #[test]
    fn test_ensure_contiguous_already_contiguous() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 3, 4), DType::F32, &device).unwrap();
        let result = ensure_contiguous(&tensor).unwrap();
        assert!(result.is_contiguous());
    }

    #[test]
    fn test_verify_attention_shapes_valid() {
        let device = Device::Cpu;
        let batch_size = 2;
        let num_qo_heads = 32;
        let num_kv_heads = 8;
        let head_dim = 128;
        let num_pages = 100;
        let page_size = 16;

        let q = Tensor::zeros((batch_size, num_qo_heads, head_dim), DType::F32, &device).unwrap();
        let k_cache = Tensor::zeros(
            (num_pages, page_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();
        let v_cache = Tensor::zeros(
            (num_pages, page_size, num_kv_heads, head_dim),
            DType::F32,
            &device,
        )
        .unwrap();

        let result = verify_attention_shapes(
            &q,
            &k_cache,
            &v_cache,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_attention_shapes_invalid_q() {
        let device = Device::Cpu;
        // Wrong number of heads
        let q = Tensor::zeros((2, 16, 128), DType::F32, &device).unwrap();
        let k_cache = Tensor::zeros((100, 16, 8, 128), DType::F32, &device).unwrap();
        let v_cache = Tensor::zeros((100, 16, 8, 128), DType::F32, &device).unwrap();

        let result = verify_attention_shapes(&q, &k_cache, &v_cache, 32, 8, 128, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_attention_shapes_invalid_cache() {
        let device = Device::Cpu;
        // Wrong head_dim in cache
        let q = Tensor::zeros((2, 32, 128), DType::F32, &device).unwrap();
        let k_cache = Tensor::zeros((100, 16, 8, 64), DType::F32, &device).unwrap(); // Wrong!
        let v_cache = Tensor::zeros((100, 16, 8, 128), DType::F32, &device).unwrap();

        let result = verify_attention_shapes(&q, &k_cache, &v_cache, 32, 8, 128, true);
        assert!(result.is_err());
    }
}
