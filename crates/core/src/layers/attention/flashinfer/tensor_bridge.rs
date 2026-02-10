//! Tensor conversion between candle and FlashInfer.
//!
//! Provides utilities for converting candle Tensors to raw CUDA pointers
//! for passing to FlashInfer's FFI layer.

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "flashinfer")]
use candle_core::DType;

/// Extract raw CUDA device pointer from a candle Tensor as `*const c_void`.
///
/// The returned pointer is the raw GPU memory address suitable for FFI.
///
/// # Safety
/// The returned pointer is only valid as long as the Tensor is alive.
/// Caller must ensure proper synchronization with CUDA operations.
#[cfg(feature = "flashinfer")]
pub fn tensor_to_device_ptr(tensor: &Tensor) -> Result<*const std::ffi::c_void> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::Storage;

    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    match &*storage {
        Storage::Cuda(cuda_storage) => {
            // Get the raw CUdeviceptr from the underlying storage.
            // We match on the dtype to extract the appropriately-typed CudaSlice,
            // then read its cu_device_ptr field via the stream-based API.
            let cuda_device = match tensor.device() {
                Device::Cuda(d) => d,
                _ => unreachable!(),
            };
            let stream = cuda_device.cuda_stream();

            let base_ptr = match tensor.dtype() {
                DType::F16 => {
                    let slice = cuda_storage.as_cuda_slice::<half::f16>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::BF16 => {
                    let slice = cuda_storage.as_cuda_slice::<half::bf16>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::F32 => {
                    let slice = cuda_storage.as_cuda_slice::<f32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::U8 => {
                    let slice = cuda_storage.as_cuda_slice::<u8>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::U32 => {
                    let slice = cuda_storage.as_cuda_slice::<u32>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::I64 => {
                    let slice = cuda_storage.as_cuda_slice::<i64>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
                DType::F64 => {
                    let slice = cuda_storage.as_cuda_slice::<f64>()?;
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    ptr as *const std::ffi::c_void
                }
            };

            // Add byte offset for non-zero start offsets
            let elem_size = tensor.dtype().size_in_bytes();
            let ptr = unsafe { (base_ptr as *const u8).add(offset * elem_size) };
            Ok(ptr as *const std::ffi::c_void)
        }
        _ => Err(candle_core::Error::Msg(
            "Tensor must be on CUDA device".to_string(),
        )),
    }
}

/// Extract raw CUDA stream handle from a candle CUDA device.
///
/// Returns the raw `CUstream` pointer suitable for passing to FlashInfer FFI
/// as `*mut c_void`.
#[cfg(feature = "flashinfer")]
pub fn get_cuda_stream_ptr(device: &Device) -> Result<*mut std::ffi::c_void> {
    match device {
        Device::Cuda(cuda_device) => {
            let stream = cuda_device.cuda_stream();
            Ok(stream.cu_stream() as *mut std::ffi::c_void)
        }
        _ => Err(candle_core::Error::Msg(
            "Device must be CUDA for FlashInfer".to_string(),
        )),
    }
}

/// Allocate a GPU tensor of i32-compatible data for FFI use.
///
/// Candle doesn't have an i32 dtype, so we store as u32 (same size, 4 bytes).
/// For FFI calls expecting `*const i32`, cast via `tensor_to_device_ptr() as *const i32`.
/// This is correct because u32 and i32 have the same size and bit representation
/// for the non-negative values used in FlashInfer metadata (indices, lengths, indptrs).
#[cfg(feature = "flashinfer")]
pub fn alloc_gpu_i32(data: &[i32], device: &Device) -> Result<Tensor> {
    let data_u32: Vec<u32> = data.iter().map(|&x| x as u32).collect();
    Tensor::from_vec(data_u32, (data.len(),), device)
}

/// Convert candle DType to flashinfer-rs FFI DType.
#[cfg(feature = "flashinfer")]
pub fn candle_to_ffi_dtype(dtype: DType) -> Result<flashinfer_rs::ffi::DType> {
    match dtype {
        DType::F16 => Ok(flashinfer_rs::ffi::DType::Float16),
        DType::BF16 => Ok(flashinfer_rs::ffi::DType::BFloat16),
        DType::F32 => Ok(flashinfer_rs::ffi::DType::Float32),
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
