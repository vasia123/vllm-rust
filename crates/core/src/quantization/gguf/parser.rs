//! GGUF file format parser.
//!
//! GGUF v2/v3 format specification:
//! - Magic: "GGUF" (4 bytes)
//! - Version: u32 (2 or 3)
//! - Tensor count: u64
//! - Metadata KV count: u64
//! - Metadata KV pairs
//! - Tensor infos
//! - Padding to alignment
//! - Tensor data

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use super::dequant::GgmlType;

/// GGUF file magic number.
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// Minimum supported GGUF version.
const GGUF_VERSION_MIN: u32 = 2;

/// Maximum supported GGUF version.
const GGUF_VERSION_MAX: u32 = 3;

/// Default alignment for tensor data.
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    /// Unsigned 8-bit integer
    U8(u8),
    /// Signed 8-bit integer
    I8(i8),
    /// Unsigned 16-bit integer
    U16(u16),
    /// Signed 16-bit integer
    I16(i16),
    /// Unsigned 32-bit integer
    U32(u32),
    /// Signed 32-bit integer
    I32(i32),
    /// Unsigned 64-bit integer
    U64(u64),
    /// Signed 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Boolean
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Array of values
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Get value as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get value as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U8(v) => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I8(v) => Some(*v as u32),
            GgufValue::I16(v) => Some(*v as u32),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get value as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) => Some(*v as u64),
            GgufValue::I16(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            GgufValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get value as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get value as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            GgufValue::U8(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Get value as array.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

/// GGUF metadata (key-value pairs).
pub type GgufMetadata = HashMap<String, GgufValue>;

/// Information about a tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Shape (dimensions)
    pub shape: Vec<u64>,
    /// GGML type
    pub ggml_type: GgmlType,
    /// Offset from start of tensor data section
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Calculate the number of elements in the tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Calculate the size in bytes based on GGML type.
    pub fn size_bytes(&self) -> u64 {
        let n_elements = self.n_elements();
        let (block_size, type_size) = self.ggml_type.block_info();

        if block_size == 0 {
            return 0;
        }

        // Number of blocks needed
        let n_blocks = n_elements.div_ceil(block_size as u64);
        n_blocks * type_size as u64
    }
}

/// Parsed GGUF file.
pub struct GgufFile {
    /// File version
    version: u32,
    /// Metadata key-value pairs
    metadata: GgufMetadata,
    /// Tensor information
    tensors: HashMap<String, GgufTensorInfo>,
    /// File path
    path: std::path::PathBuf,
    /// Offset to tensor data section
    data_offset: u64,
    /// Alignment for tensor data (stored for potential future use)
    #[allow(dead_code)]
    alignment: u64,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate magic
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"),
            ));
        }

        // Read version
        let version = read_u32(&mut reader)?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported GGUF version: {version} (supported: {GGUF_VERSION_MIN}-{GGUF_VERSION_MAX})"
                ),
            ));
        }

        // Read counts
        let tensor_count = read_u64(&mut reader)?;
        let metadata_kv_count = read_u64(&mut reader)?;

        // Parse metadata
        let mut metadata = GgufMetadata::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = read_metadata_kv(&mut reader)?;
            metadata.insert(key, value);
        }

        // Get alignment from metadata or use default
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Parse tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let info = read_tensor_info(&mut reader)?;
            tensors.insert(info.name.clone(), info);
        }

        // Calculate data offset (aligned)
        let current_pos = reader.stream_position()?;
        let data_offset = align_offset(current_pos, alignment);

        Ok(Self {
            version,
            metadata,
            tensors,
            path: path.to_path_buf(),
            data_offset,
            alignment,
        })
    }

    /// Get the GGUF version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load a tensor from the file.
    pub fn load_tensor(&self, name: &str, device: &Device) -> Result<(Tensor, GgmlType)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("Tensor not found: {name}")))?;

        // Open file and seek to tensor data
        let file = File::open(&self.path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to open file: {e}")))?;
        let mut reader = BufReader::new(file);

        // Calculate absolute offset
        let tensor_offset = self.data_offset + info.offset;
        reader
            .seek(SeekFrom::Start(tensor_offset))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to seek: {e}")))?;

        // Read tensor data
        let size = info.size_bytes() as usize;
        let mut data = vec![0u8; size];
        reader
            .read_exact(&mut data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read tensor data: {e}")))?;

        // Create tensor from raw bytes
        // For quantized types, we store the raw bytes as U8 tensor
        // Dequantization happens during forward pass
        let tensor = if info.ggml_type.is_quantized() {
            Tensor::from_vec(data, (size,), device)?
        } else {
            // For unquantized types, convert directly
            match info.ggml_type {
                GgmlType::F32 => {
                    let floats: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
                    Tensor::from_vec(floats, shape.as_slice(), device)?
                }
                GgmlType::F16 => {
                    let halfs: Vec<half::f16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
                    Tensor::from_vec(halfs, shape.as_slice(), device)?
                }
                GgmlType::BF16 => {
                    let bfloats: Vec<half::bf16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
                    Tensor::from_vec(bfloats, shape.as_slice(), device)?
                }
                _ => Tensor::from_vec(data, (size,), device)?,
            }
        };

        Ok((tensor, info.ggml_type))
    }

    /// Load a tensor and dequantize it to the target dtype.
    pub fn load_tensor_dequant(
        &self,
        name: &str,
        device: &Device,
        target_dtype: DType,
    ) -> Result<Tensor> {
        let (tensor, qtype) = self.load_tensor(name, device)?;

        if !qtype.is_quantized() {
            // Already unquantized, just convert dtype if needed
            return tensor.to_dtype(target_dtype);
        }

        let info = self.tensor_info(name).unwrap();
        let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();

        // Dequantize
        let n_elements = info.n_elements() as usize;
        let out_rows = if shape.len() >= 2 { shape[0] } else { 1 };
        let out_cols = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            n_elements
        };

        let dequantized = super::dequant::dequantize(&tensor, qtype, out_rows, out_cols)?;
        dequantized.to_dtype(target_dtype)
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("tensor_count", &self.tensors.len())
            .field("metadata_count", &self.metadata.len())
            .field("path", &self.path)
            .finish()
    }
}

// Helper functions for reading binary data

fn read_u8<R: Read>(reader: &mut R) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8<R: Read>(reader: &mut R) -> std::io::Result<i8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16<R: Read>(reader: &mut R) -> std::io::Result<i16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64<R: Read>(reader: &mut R) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> std::io::Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Invalid UTF-8: {e}"),
        )
    })
}

fn read_metadata_value<R: Read>(reader: &mut R, value_type: u32) -> std::io::Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::U8(read_u8(reader)?)),
        1 => Ok(GgufValue::I8(read_i8(reader)?)),
        2 => Ok(GgufValue::U16(read_u16(reader)?)),
        3 => Ok(GgufValue::I16(read_i16(reader)?)),
        4 => Ok(GgufValue::U32(read_u32(reader)?)),
        5 => Ok(GgufValue::I32(read_i32(reader)?)),
        6 => Ok(GgufValue::F32(read_f32(reader)?)),
        7 => Ok(GgufValue::Bool(read_u8(reader)? != 0)),
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => {
            // Array
            let array_type = read_u32(reader)?;
            let array_len = read_u64(reader)? as usize;
            let mut values = Vec::with_capacity(array_len);
            for _ in 0..array_len {
                values.push(read_metadata_value(reader, array_type)?);
            }
            Ok(GgufValue::Array(values))
        }
        10 => Ok(GgufValue::U64(read_u64(reader)?)),
        11 => Ok(GgufValue::I64(read_i64(reader)?)),
        12 => Ok(GgufValue::F64(read_f64(reader)?)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown metadata value type: {value_type}"),
        )),
    }
}

fn read_metadata_kv<R: Read>(reader: &mut R) -> std::io::Result<(String, GgufValue)> {
    let key = read_string(reader)?;
    let value_type = read_u32(reader)?;
    let value = read_metadata_value(reader, value_type)?;
    Ok((key, value))
}

fn read_tensor_info<R: Read>(reader: &mut R) -> std::io::Result<GgufTensorInfo> {
    let name = read_string(reader)?;
    let n_dims = read_u32(reader)?;

    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(read_u64(reader)?);
    }

    let ggml_type_raw = read_u32(reader)?;
    let ggml_type = GgmlType::from_u32(ggml_type_raw);

    let offset = read_u64(reader)?;

    Ok(GgufTensorInfo {
        name,
        n_dims,
        shape,
        ggml_type,
        offset,
    })
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_value_conversions() {
        let val = GgufValue::U32(42);
        assert_eq!(val.as_u32(), Some(42));
        assert_eq!(val.as_u64(), Some(42));
        assert_eq!(val.as_str(), None);

        let val = GgufValue::String("test".to_string());
        assert_eq!(val.as_str(), Some("test"));
        assert_eq!(val.as_u32(), None);

        let val = GgufValue::F32(3.14);
        assert!((val.as_f32().unwrap() - 3.14).abs() < 0.001);

        let val = GgufValue::Bool(true);
        assert_eq!(val.as_bool(), Some(true));
    }

    #[test]
    fn test_tensor_info_size() {
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            shape: vec![4096, 4096],
            ggml_type: GgmlType::F16,
            offset: 0,
        };

        assert_eq!(info.n_elements(), 4096 * 4096);
        // F16: 2 bytes per element
        assert_eq!(info.size_bytes(), 4096 * 4096 * 2);
    }

    #[test]
    fn test_tensor_info_quantized_size() {
        // Q4_0: 32 elements per block, 18 bytes per block (16 bytes for 32 4-bit values + 2 for scale)
        let info = GgufTensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            shape: vec![32, 32],
            ggml_type: GgmlType::Q4_0,
            offset: 0,
        };

        assert_eq!(info.n_elements(), 1024);
        // 1024 elements / 32 per block = 32 blocks * 18 bytes = 576 bytes
        assert_eq!(info.size_bytes(), 576);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(100, 32), 128);
    }
}
