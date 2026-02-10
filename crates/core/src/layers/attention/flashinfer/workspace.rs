//! Workspace buffer management for FlashInfer.
//!
//! FlashInfer requires pre-allocated workspace memory for plan/run operations.

use candle_core::{DType, Device, Result, Tensor};

/// Default workspace size: 128 MB
/// This is the recommended size from FlashInfer documentation.
pub const DEFAULT_WORKSPACE_SIZE: usize = 128 * 1024 * 1024;

/// Workspace buffer for FlashInfer operations.
///
/// FlashInfer's BatchPrefillHandler and BatchDecodeHandler require workspace
/// memory for intermediate computations. This buffer is pre-allocated once
/// and reused across operations.
#[derive(Debug)]
pub struct WorkspaceBuffer {
    /// Raw buffer tensor (U8)
    buffer: Tensor,
    /// Buffer size in bytes
    size: usize,
    /// Device this buffer is allocated on
    device: Device,
}

impl WorkspaceBuffer {
    /// Create a new workspace buffer with default size (128 MB).
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_size(DEFAULT_WORKSPACE_SIZE, device)
    }

    /// Create a new workspace buffer with custom size.
    pub fn with_size(size: usize, device: &Device) -> Result<Self> {
        let buffer = Tensor::zeros(size, DType::U8, device)?;
        Ok(Self {
            buffer,
            size,
            device: device.clone(),
        })
    }

    /// Get the raw buffer tensor.
    pub fn buffer(&self) -> &Tensor {
        &self.buffer
    }

    /// Get buffer size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device this buffer is allocated on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a raw pointer to the buffer data (for FFI).
    ///
    /// # Safety
    /// The returned pointer is only valid as long as this WorkspaceBuffer
    /// is alive and not modified.
    #[cfg(feature = "flashinfer")]
    pub fn as_ptr(&self) -> Result<*mut u8> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::Storage;

        let cuda_device = match self.buffer.device() {
            Device::Cuda(d) => d,
            _ => {
                return Err(candle_core::Error::Msg(
                    "WorkspaceBuffer must be on CUDA device".to_string(),
                ))
            }
        };
        let stream = cuda_device.cuda_stream();

        let storage = self.buffer.storage_and_layout().0;
        match &*storage {
            Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<u8>()?;
                let (ptr, _guard) = slice.device_ptr(&stream);
                Ok(ptr as *mut u8)
            }
            _ => Err(candle_core::Error::Msg(
                "WorkspaceBuffer must be on CUDA device".to_string(),
            )),
        }
    }

    /// Check if this workspace is on a CUDA device.
    pub fn is_cuda(&self) -> bool {
        self.device.is_cuda()
    }

    /// Resize the workspace buffer if needed.
    ///
    /// Only reallocates if the new size is larger than current.
    pub fn ensure_size(&mut self, required_size: usize) -> Result<()> {
        if required_size > self.size {
            self.buffer = Tensor::zeros(required_size, DType::U8, &self.device)?;
            self.size = required_size;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let device = Device::Cpu;
        let workspace = WorkspaceBuffer::new(&device).unwrap();

        assert_eq!(workspace.size(), DEFAULT_WORKSPACE_SIZE);
        assert!(!workspace.is_cuda());
    }

    #[test]
    fn test_workspace_custom_size() {
        let device = Device::Cpu;
        let custom_size = 64 * 1024 * 1024; // 64 MB
        let workspace = WorkspaceBuffer::with_size(custom_size, &device).unwrap();

        assert_eq!(workspace.size(), custom_size);
    }

    #[test]
    fn test_workspace_ensure_size_no_realloc() {
        let device = Device::Cpu;
        let mut workspace = WorkspaceBuffer::new(&device).unwrap();
        let original_size = workspace.size();

        // Request smaller size - should not reallocate
        workspace.ensure_size(1024).unwrap();
        assert_eq!(workspace.size(), original_size);
    }

    #[test]
    fn test_workspace_ensure_size_realloc() {
        let device = Device::Cpu;
        let initial_size = 1024;
        let mut workspace = WorkspaceBuffer::with_size(initial_size, &device).unwrap();

        // Request larger size - should reallocate
        let new_size = 2048;
        workspace.ensure_size(new_size).unwrap();
        assert_eq!(workspace.size(), new_size);
    }

    #[test]
    fn test_workspace_buffer_shape() {
        let device = Device::Cpu;
        let size = 1024;
        let workspace = WorkspaceBuffer::with_size(size, &device).unwrap();

        assert_eq!(workspace.buffer().dims(), &[size]);
        assert_eq!(workspace.buffer().dtype(), DType::U8);
    }
}
