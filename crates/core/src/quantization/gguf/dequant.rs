//! GGML dequantization routines.
//!
//! This module implements CPU dequantization for various GGML quantization types.
//! For GPU acceleration, CUDA kernels should be used instead.

use candle_core::{DType, Result, Tensor};

/// GGML quantization types.
///
/// These match the enum values from llama.cpp's ggml.h.
/// Names follow GGML convention (Q4_0, Q4_K, etc.) for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // deprecated
    // Q4_3 = 5, // deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
    /// Unknown or unsupported type
    Unknown = 255,
}

impl GgmlType {
    /// Convert from raw u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            _ => Self::Unknown,
        }
    }

    /// Get the raw u32 value.
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// Check if this type is quantized (not full precision).
    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Get block size and type size in bytes.
    ///
    /// Returns (elements_per_block, bytes_per_block).
    pub fn block_info(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::BF16 => (1, 2),
            Self::F64 => (1, 8),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::Q4_0 => (32, 18), // 32 elements, 2 bytes scale + 16 bytes data
            Self::Q4_1 => (32, 20), // 32 elements, 2 bytes scale + 2 bytes min + 16 bytes data
            Self::Q5_0 => (32, 22), // 32 elements, 2 bytes scale + 4 bytes high bits + 16 bytes data
            Self::Q5_1 => (32, 24), // 32 elements, 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes data
            Self::Q8_0 => (32, 34), // 32 elements, 2 bytes scale + 32 bytes data
            Self::Q8_1 => (32, 36), // 32 elements, 4 bytes scale + 32 bytes data (scale is f32)
            Self::Q2_K => (256, 84), // Super-block of 256, complex structure
            Self::Q3_K => (256, 110), // Super-block of 256
            Self::Q4_K => (256, 144), // Super-block of 256
            Self::Q5_K => (256, 176), // Super-block of 256
            Self::Q6_K => (256, 210), // Super-block of 256
            Self::Q8_K => (256, 292), // Super-block of 256
            _ => (0, 0),            // Unknown or unsupported
        }
    }

    /// Get the DType for unquantized equivalent.
    pub fn to_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F64 => DType::F64,
            _ => DType::F16, // Default output for quantized types
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::F64 => write!(f, "F64"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2_K => write!(f, "Q2_K"),
            Self::Q3_K => write!(f, "Q3_K"),
            Self::Q4_K => write!(f, "Q4_K"),
            Self::Q5_K => write!(f, "Q5_K"),
            Self::Q6_K => write!(f, "Q6_K"),
            Self::Q8_K => write!(f, "Q8_K"),
            Self::I8 => write!(f, "I8"),
            Self::I16 => write!(f, "I16"),
            Self::I32 => write!(f, "I32"),
            Self::I64 => write!(f, "I64"),
            Self::IQ2_XXS => write!(f, "IQ2_XXS"),
            Self::IQ2_XS => write!(f, "IQ2_XS"),
            Self::IQ3_XXS => write!(f, "IQ3_XXS"),
            Self::IQ1_S => write!(f, "IQ1_S"),
            Self::IQ4_NL => write!(f, "IQ4_NL"),
            Self::IQ3_S => write!(f, "IQ3_S"),
            Self::IQ2_S => write!(f, "IQ2_S"),
            Self::IQ4_XS => write!(f, "IQ4_XS"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Dequantize a tensor from GGML quantized format to F16.
///
/// # Arguments
/// * `data` - Raw quantized data as U8 tensor
/// * `qtype` - GGML quantization type
/// * `rows` - Number of rows in output
/// * `cols` - Number of columns in output
///
/// # Returns
/// Dequantized tensor in F16 format with shape [rows, cols]
pub fn dequantize(data: &Tensor, qtype: GgmlType, rows: usize, cols: usize) -> Result<Tensor> {
    // Get raw bytes
    let bytes = data.to_vec1::<u8>()?;
    let device = data.device();

    match qtype {
        GgmlType::F32 => {
            // Already unquantized - just reshape
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let tensor = Tensor::from_vec(floats, (rows, cols), device)?;
            tensor.to_dtype(DType::F16)
        }
        GgmlType::F16 => {
            // Already F16 - just reshape
            let halfs: Vec<half::f16> = bytes
                .chunks_exact(2)
                .map(|chunk| half::f16::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Tensor::from_vec(halfs, (rows, cols), device)
        }
        GgmlType::BF16 => {
            let bfloats: Vec<half::bf16> = bytes
                .chunks_exact(2)
                .map(|chunk| half::bf16::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let tensor = Tensor::from_vec(bfloats, (rows, cols), device)?;
            tensor.to_dtype(DType::F16)
        }
        GgmlType::Q4_0 => dequantize_q4_0(&bytes, rows, cols, device),
        GgmlType::Q4_1 => dequantize_q4_1(&bytes, rows, cols, device),
        GgmlType::Q5_0 => dequantize_q5_0(&bytes, rows, cols, device),
        GgmlType::Q5_1 => dequantize_q5_1(&bytes, rows, cols, device),
        GgmlType::Q8_0 => dequantize_q8_0(&bytes, rows, cols, device),
        GgmlType::Q8_1 => dequantize_q8_1(&bytes, rows, cols, device),
        GgmlType::Q2_K => dequantize_q2_k(&bytes, rows, cols, device),
        GgmlType::Q3_K => dequantize_q3_k(&bytes, rows, cols, device),
        GgmlType::Q4_K => dequantize_q4_k(&bytes, rows, cols, device),
        GgmlType::Q5_K => dequantize_q5_k(&bytes, rows, cols, device),
        GgmlType::Q6_K => dequantize_q6_k(&bytes, rows, cols, device),
        _ => candle_core::bail!("Unsupported GGML quantization type: {qtype}"),
    }
}

/// Dequantize Q4_0 format.
/// Block structure: [f16 scale, u8[16] quants]
/// Each u8 contains 2 4-bit values.
fn dequantize_q4_0(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 18;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q4_0: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        // Read scale (f16)
        let scale_bytes = [bytes[block_offset], bytes[block_offset + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read quantized values (16 bytes = 32 4-bit values)
        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..16 {
            let quant = bytes[block_offset + 2 + i];
            let q0 = (quant & 0x0F) as i32 - 8; // Low 4 bits
            let q1 = (quant >> 4) as i32 - 8; // High 4 bits

            output[out_offset + i] = q0 as f32 * scale;
            output[out_offset + i + 16] = q1 as f32 * scale;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q4_1 format.
/// Block structure: [f16 scale, f16 min, u8[16] quants]
fn dequantize_q4_1(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 20;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q4_1: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        // Read scale and min (both f16)
        let scale_bytes = [bytes[block_offset], bytes[block_offset + 1]];
        let min_bytes = [bytes[block_offset + 2], bytes[block_offset + 3]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
        let min = half::f16::from_le_bytes(min_bytes).to_f32();

        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..16 {
            let quant = bytes[block_offset + 4 + i];
            let q0 = (quant & 0x0F) as f32;
            let q1 = (quant >> 4) as f32;

            output[out_offset + i] = q0 * scale + min;
            output[out_offset + i + 16] = q1 * scale + min;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q5_0 format.
/// Block structure: [f16 scale, u8[4] high_bits, u8[16] quants]
fn dequantize_q5_0(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 22;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q5_0: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        // Read scale (f16)
        let scale_bytes = [bytes[block_offset], bytes[block_offset + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read high bits (4 bytes = 32 bits, one per element)
        let high_bits = u32::from_le_bytes([
            bytes[block_offset + 2],
            bytes[block_offset + 3],
            bytes[block_offset + 4],
            bytes[block_offset + 5],
        ]);

        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..16 {
            let quant = bytes[block_offset + 6 + i];

            // Low 4 bits + high bit for element i
            let high0 = ((high_bits >> i) & 1) as i32;
            let q0 = ((quant & 0x0F) as i32 | (high0 << 4)) - 16;

            // High 4 bits + high bit for element i+16
            let high1 = ((high_bits >> (i + 16)) & 1) as i32;
            let q1 = ((quant >> 4) as i32 | (high1 << 4)) - 16;

            output[out_offset + i] = q0 as f32 * scale;
            output[out_offset + i + 16] = q1 as f32 * scale;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q5_1 format.
/// Block structure: [f16 scale, f16 min, u8[4] high_bits, u8[16] quants]
fn dequantize_q5_1(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 24;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q5_1: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        let scale_bytes = [bytes[block_offset], bytes[block_offset + 1]];
        let min_bytes = [bytes[block_offset + 2], bytes[block_offset + 3]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();
        let min = half::f16::from_le_bytes(min_bytes).to_f32();

        let high_bits = u32::from_le_bytes([
            bytes[block_offset + 4],
            bytes[block_offset + 5],
            bytes[block_offset + 6],
            bytes[block_offset + 7],
        ]);

        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..16 {
            let quant = bytes[block_offset + 8 + i];

            let high0 = (high_bits >> i) & 1;
            let q0 = (quant & 0x0F) as u32 | (high0 << 4);

            let high1 = (high_bits >> (i + 16)) & 1;
            let q1 = (quant >> 4) as u32 | (high1 << 4);

            output[out_offset + i] = q0 as f32 * scale + min;
            output[out_offset + i + 16] = q1 as f32 * scale + min;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q8_0 format.
/// Block structure: [f16 scale, i8[32] quants]
fn dequantize_q8_0(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 34;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q8_0: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        let scale_bytes = [bytes[block_offset], bytes[block_offset + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..32 {
            let quant = bytes[block_offset + 2 + i] as i8;
            output[out_offset + i] = quant as f32 * scale;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q8_1 format.
/// Block structure: [f32 scale, f32 sum, i8[32] quants]
fn dequantize_q8_1(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 32;
    const BYTES_PER_BLOCK: usize = 36;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q8_1: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;

        let scale = f32::from_le_bytes([
            bytes[block_offset],
            bytes[block_offset + 1],
            bytes[block_offset + 2],
            bytes[block_offset + 3],
        ]);

        // d (sum) is at offset 4-7, but we don't use it for dequantization
        // It's used for dot product optimization

        let out_offset = block_idx * BLOCK_SIZE;

        for i in 0..32 {
            let quant = bytes[block_offset + 8 + i] as i8;
            output[out_offset + i] = quant as f32 * scale;
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

// K-quant formats are more complex. Basic implementations below.

/// Dequantize Q2_K format (super-block of 256 elements).
fn dequantize_q2_k(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 84;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q2_K: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let out_offset = block_idx * BLOCK_SIZE;

        // Q2_K structure:
        // scales: 16 bytes (4 bits each, 16 sub-blocks)
        // qs: 64 bytes (2 bits per element, 256 elements)
        // d: f16 (2 bytes)
        // dmin: f16 (2 bytes)

        let scales = &bytes[block_offset..block_offset + 16];
        let qs = &bytes[block_offset + 16..block_offset + 80];

        let d =
            half::f16::from_le_bytes([bytes[block_offset + 80], bytes[block_offset + 81]]).to_f32();
        let dmin =
            half::f16::from_le_bytes([bytes[block_offset + 82], bytes[block_offset + 83]]).to_f32();

        // Process 16 sub-blocks of 16 elements each
        for j in 0..16 {
            let sc = scales[j];
            let scale = (sc & 0x0F) as f32;
            let min = (sc >> 4) as f32;

            for l in 0..16 {
                let q_idx = j * 4 + l / 4;
                let shift = (l % 4) * 2;
                let q = ((qs[q_idx] >> shift) & 0x03) as f32;

                output[out_offset + j * 16 + l] = d * scale * q - dmin * min;
            }
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q3_K format.
fn dequantize_q3_k(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 110;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q3_K: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let out_offset = block_idx * BLOCK_SIZE;

        // Q3_K structure:
        // hmask: 32 bytes (high bits for 256 elements)
        // qs: 64 bytes (low 2 bits for 256 elements)
        // scales: 12 bytes (6-bit scales for 16 sub-blocks)
        // d: f16 (2 bytes)

        let hmask = &bytes[block_offset..block_offset + 32];
        let qs = &bytes[block_offset + 32..block_offset + 96];
        let scales_bytes = &bytes[block_offset + 96..block_offset + 108];

        let d = half::f16::from_le_bytes([bytes[block_offset + 108], bytes[block_offset + 109]])
            .to_f32();

        // Decode 6-bit scales from packed 12 bytes
        let mut scales = [0i8; 16];
        for i in 0..8 {
            let a = scales_bytes[i];
            let b = scales_bytes[i + 4];
            let _c = scales_bytes[i + 8];

            scales[i] = ((a & 0x3F) as i8) - 32;
            scales[i + 8] = (((a >> 6) | ((b & 0x0F) << 2)) as i8) - 32;
        }

        // Process elements
        for j in 0..256 {
            let scale_idx = j / 16;
            let scale = scales[scale_idx] as f32;

            let q_idx = j / 4;
            let q_shift = (j % 4) * 2;
            let q_low = ((qs[q_idx] >> q_shift) & 0x03) as i32;

            let h_idx = j / 8;
            let h_shift = j % 8;
            let q_high = ((hmask[h_idx] >> h_shift) & 1) as i32;

            let q = q_low | (q_high << 2);
            output[out_offset + j] = d * scale * (q as f32 - 4.0);
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q4_K format.
fn dequantize_q4_k(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 144;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q4_K: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let out_offset = block_idx * BLOCK_SIZE;

        // Q4_K structure:
        // d: f16 (2 bytes)
        // dmin: f16 (2 bytes)
        // scales: 12 bytes (6-bit values)
        // qs: 128 bytes (4-bit values)

        let d = half::f16::from_le_bytes([bytes[block_offset], bytes[block_offset + 1]]).to_f32();
        let dmin =
            half::f16::from_le_bytes([bytes[block_offset + 2], bytes[block_offset + 3]]).to_f32();

        let scales_bytes = &bytes[block_offset + 4..block_offset + 16];
        let qs = &bytes[block_offset + 16..block_offset + 144];

        // Decode scales and mins from packed bytes
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = scales_bytes[i + 4] & 0x3F;
            mins[i] = scales_bytes[i] >> 6 | ((scales_bytes[i + 4] >> 4) & 0x0C);
            mins[i + 4] = scales_bytes[i + 8] & 0x3F;
        }

        // Process 8 sub-blocks of 32 elements
        for j in 0..8 {
            let sc = scales[j] as f32;
            let m = mins[j] as f32;

            for l in 0..32 {
                let idx = j * 32 + l;
                let q_idx = j * 16 + l / 2;
                let q = if l % 2 == 0 {
                    (qs[q_idx] & 0x0F) as f32
                } else {
                    (qs[q_idx] >> 4) as f32
                };

                output[out_offset + idx] = d * sc * q - dmin * m;
            }
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q5_K format.
fn dequantize_q5_k(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 176;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q5_K: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let out_offset = block_idx * BLOCK_SIZE;

        // Q5_K structure:
        // d: f16, dmin: f16, scales: 12 bytes, qh: 32 bytes, qs: 128 bytes

        let d = half::f16::from_le_bytes([bytes[block_offset], bytes[block_offset + 1]]).to_f32();
        let dmin =
            half::f16::from_le_bytes([bytes[block_offset + 2], bytes[block_offset + 3]]).to_f32();

        let scales_bytes = &bytes[block_offset + 4..block_offset + 16];
        let qh = &bytes[block_offset + 16..block_offset + 48];
        let qs = &bytes[block_offset + 48..block_offset + 176];

        // Decode scales
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            scales[i + 4] = scales_bytes[i + 4] & 0x3F;
            mins[i] = scales_bytes[i] >> 6 | ((scales_bytes[i + 4] >> 4) & 0x0C);
            mins[i + 4] = scales_bytes[i + 8] & 0x3F;
        }

        // Process elements
        for j in 0..8 {
            let sc = scales[j] as f32;
            let m = mins[j] as f32;

            for l in 0..32 {
                let idx = j * 32 + l;
                let q_idx = j * 16 + l / 2;
                let q_low = if l % 2 == 0 {
                    (qs[q_idx] & 0x0F) as u32
                } else {
                    (qs[q_idx] >> 4) as u32
                };

                let qh_idx = j * 4 + l / 8;
                let qh_shift = l % 8;
                let q_high = ((qh[qh_idx] >> qh_shift) & 1) as u32;

                let q = (q_low | (q_high << 4)) as f32;
                output[out_offset + idx] = d * sc * q - dmin * m;
            }
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

/// Dequantize Q6_K format.
fn dequantize_q6_k(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    const BLOCK_SIZE: usize = 256;
    const BYTES_PER_BLOCK: usize = 210;

    let n_elements = rows * cols;
    let n_blocks = n_elements / BLOCK_SIZE;

    if bytes.len() < n_blocks * BYTES_PER_BLOCK {
        candle_core::bail!(
            "Q6_K: insufficient data: expected {} bytes, got {}",
            n_blocks * BYTES_PER_BLOCK,
            bytes.len()
        );
    }

    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * BYTES_PER_BLOCK;
        let out_offset = block_idx * BLOCK_SIZE;

        // Q6_K structure:
        // ql: 128 bytes (low 4 bits)
        // qh: 64 bytes (high 2 bits)
        // scales: 16 bytes (8-bit scales)
        // d: f16 (2 bytes)

        let ql = &bytes[block_offset..block_offset + 128];
        let qh = &bytes[block_offset + 128..block_offset + 192];
        let scales = &bytes[block_offset + 192..block_offset + 208];
        let d = half::f16::from_le_bytes([bytes[block_offset + 208], bytes[block_offset + 209]])
            .to_f32();

        // Process 16 sub-blocks of 16 elements
        for (j, &scale_byte) in scales.iter().enumerate() {
            let scale = (scale_byte as i8) as f32;

            for l in 0..16 {
                let idx = j * 16 + l;

                // Get low 4 bits
                let ql_idx = j * 8 + l / 2;
                let q_low = if l % 2 == 0 {
                    (ql[ql_idx] & 0x0F) as u32
                } else {
                    (ql[ql_idx] >> 4) as u32
                };

                // Get high 2 bits
                let qh_idx = j * 4 + l / 4;
                let qh_shift = (l % 4) * 2;
                let q_high = ((qh[qh_idx] >> qh_shift) & 0x03) as u32;

                let q = (q_low | (q_high << 4)) as i32 - 32;
                output[out_offset + idx] = d * scale * q as f32;
            }
        }
    }

    let tensor = Tensor::from_vec(output, (rows, cols), device)?;
    tensor.to_dtype(DType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(8), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_u32(30), GgmlType::BF16);
        assert_eq!(GgmlType::from_u32(999), GgmlType::Unknown);
    }

    #[test]
    fn test_ggml_type_is_quantized() {
        assert!(!GgmlType::F32.is_quantized());
        assert!(!GgmlType::F16.is_quantized());
        assert!(!GgmlType::BF16.is_quantized());
        assert!(GgmlType::Q4_0.is_quantized());
        assert!(GgmlType::Q8_0.is_quantized());
        assert!(GgmlType::Q4_K.is_quantized());
    }

    #[test]
    fn test_ggml_type_block_info() {
        assert_eq!(GgmlType::F32.block_info(), (1, 4));
        assert_eq!(GgmlType::F16.block_info(), (1, 2));
        assert_eq!(GgmlType::Q4_0.block_info(), (32, 18));
        assert_eq!(GgmlType::Q8_0.block_info(), (32, 34));
        assert_eq!(GgmlType::Q4_K.block_info(), (256, 144));
    }

    #[test]
    fn test_dequantize_q4_0() {
        let device = candle_core::Device::Cpu;

        // Create a simple Q4_0 block (32 elements)
        // Scale = 1.0 (as f16), all quants = 0x88 (values 8 and 8 -> 0 and 0 after -8)
        let mut data = vec![0u8; 18];
        let scale_f16 = half::f16::from_f32(1.0);
        data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..16 {
            data[2 + i] = 0x88; // Both nibbles = 8, which becomes 0 after -8
        }

        let tensor = Tensor::from_vec(data, (18,), &device).unwrap();
        let result = dequantize_q4_0(&tensor.to_vec1::<u8>().unwrap(), 1, 32, &device).unwrap();

        assert_eq!(result.dims(), &[1, 32]);

        // All values should be 0 (8-8=0, 0*1.0=0)
        let values: Vec<f32> = result.to_dtype(DType::F32).unwrap().to_vec2().unwrap()[0].clone();
        for v in values {
            assert!((v - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dequantize_q8_0() {
        let device = candle_core::Device::Cpu;

        // Create a simple Q8_0 block (32 elements)
        // Scale = 0.5 (as f16), all quants = 2 -> output = 1.0
        let mut data = vec![0u8; 34];
        let scale_f16 = half::f16::from_f32(0.5);
        data[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            data[2 + i] = 2; // signed value = 2
        }

        let tensor = Tensor::from_vec(data, (34,), &device).unwrap();
        let result = dequantize_q8_0(&tensor.to_vec1::<u8>().unwrap(), 1, 32, &device).unwrap();

        assert_eq!(result.dims(), &[1, 32]);

        let values: Vec<f32> = result.to_dtype(DType::F32).unwrap().to_vec2().unwrap()[0].clone();
        for v in values {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }
}
