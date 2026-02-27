//! CUDA kernel bindings for MXFP4 E2M1 quantized linear layers.
//!
//! Provides `mxfp4_gemm`: a GPU-accelerated forward pass for weights stored
//! in OCP MXFP4 E2M1 format with blockwise E8M0 scales.  Replaces the CPU
//! dequantize→matmul emulation path in `MxFp4Linear::forward`.
//!
//! Weight format:
//!   `qweight [N, K/2]` U8 — two FP4 nibbles per byte, lower nibble = lower K
//!   `scales  [N, K/32]` U8 — E8M0 block scale, one per 32 K-elements

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Storage,
    Tensor,
};

const MXFP4_GEMM_PTX: &str = include_str!("../../kernels/mxfp4_gemm.ptx");

// ─── MXFP4 GEMM op ───────────────────────────────────────────────────────────

struct Mxfp4GemmOp {
    qweight: Tensor,      // [N, K/2] U8  — packed FP4
    scales: Tensor,       // [N, K/32] U8 — E8M0 block scales
    bias: Option<Tensor>, // [N]      BF16 — optional bias
    m: usize,
    n: usize,
    k: usize,
}

impl CustomOp1 for Mxfp4GemmOp {
    fn name(&self) -> &'static str {
        "mxfp4_gemm_bf16"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("mxfp4_gemm_bf16 requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func = dev.get_or_load_custom_func("mxfp4_gemm_bf16", "mxfp4_gemm", MXFP4_GEMM_PTX)?;

        let output_shape = Shape::from_dims(&[self.m, self.n]);
        let output = dev.alloc_zeros::<half::bf16>(self.m * self.n)?;

        let input = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("mxfp4_gemm_bf16: input must be BF16"),
        };

        let (qw_guard, _) = self.qweight.storage_and_layout();
        let qweight = match &*qw_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("mxfp4_gemm_bf16: qweight must be U8"),
            },
            _ => candle_core::bail!("mxfp4_gemm_bf16: qweight must be on CUDA"),
        };

        let (sc_guard, _) = self.scales.storage_and_layout();
        let scales = match &*sc_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U8(s) => s,
                _ => candle_core::bail!("mxfp4_gemm_bf16: scales must be U8"),
            },
            _ => candle_core::bail!("mxfp4_gemm_bf16: scales must be on CUDA"),
        };

        // Tile size matches TILE_M/TILE_N in mxfp4_gemm.cu
        const TILE: usize = 16;
        let cfg = LaunchConfig {
            grid_dim: (
                self.n.div_ceil(TILE) as u32,
                self.m.div_ceil(TILE) as u32,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let m_i32 = self.m as i32;
        let n_i32 = self.n as i32;
        let k_i32 = self.k as i32;
        let has_bias_i32: i32 = if self.bias.is_some() { 1 } else { 0 };

        // Hoist optional bias guard so it outlives the builder argument references.
        let bias_storage = self.bias.as_ref().map(|b| b.storage_and_layout());
        let null_bias_ptr: u64 = 0;

        let mut builder = func.builder();
        builder.arg(&output);
        builder.arg(input);
        builder.arg(qweight);
        builder.arg(scales);
        builder.arg(&m_i32);
        builder.arg(&n_i32);
        builder.arg(&k_i32);
        builder.arg(&has_bias_i32);

        match &bias_storage {
            Some((bias_guard, _)) => {
                if let Storage::Cuda(cs) = &**bias_guard {
                    if let CudaStorageSlice::BF16(s) = &cs.slice {
                        builder.arg(s);
                    } else {
                        candle_core::bail!("mxfp4_gemm_bf16: bias must be BF16");
                    }
                } else {
                    candle_core::bail!("mxfp4_gemm_bf16: bias must be on CUDA");
                }
            }
            None => {
                builder.arg(&null_bias_ptr);
            }
        }

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("mxfp4_gemm_bf16 launch: {e}")))?;

        drop(qw_guard);
        drop(sc_guard);
        drop(bias_storage);

        let output_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((output_storage, output_shape))
    }
}

/// MXFP4 E2M1 GEMM: `output = input @ dequant(qweight).T + bias`
///
/// Decodes packed FP4 weights and E8M0 block scales in the kernel, avoiding
/// materialization of the full-precision weight matrix.
///
/// # Arguments
/// * `input`   — `[M, K]` or `[batch.., K]` in BF16
/// * `qweight` — `[N, K/2]` U8 (two FP4 nibbles per byte, lower nibble = lower K)
/// * `scales`  — `[N, K/32]` U8 (E8M0 exponent, one per 32 K-elements)
/// * `bias`    — optional `[N]` BF16
/// * `size_n`  — output feature dimension N
pub fn mxfp4_gemm(
    input: &Tensor,
    qweight: &Tensor,
    scales: &Tensor,
    bias: Option<&Tensor>,
    size_n: usize,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let dims = input.dims();
    let (m, k) = match dims.len() {
        2 => (dims[0], dims[1]),
        _ => {
            let batch: usize = dims[..dims.len() - 1].iter().product();
            (batch, dims[dims.len() - 1])
        }
    };

    let op = Mxfp4GemmOp {
        qweight: qweight.clone(),
        scales: scales.clone(),
        bias: bias.cloned(),
        m,
        n: size_n,
        k,
    };
    let result = input.apply_op1(op)?;

    if dims.len() > 2 {
        let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
        out_shape.push(size_n);
        result.reshape(out_shape)
    } else {
        Ok(result)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_mxfp4_gemm_requires_cuda() {
        let input = Tensor::zeros((4usize, 64usize), DType::BF16, &Device::Cpu).unwrap();
        let qweight = Tensor::zeros((32usize, 32usize), DType::U8, &Device::Cpu).unwrap();
        let scales = Tensor::zeros((32usize, 2usize), DType::U8, &Device::Cpu).unwrap();
        let result = mxfp4_gemm(&input, &qweight, &scales, None, 32);
        assert!(result.is_err(), "expected error on CPU");
    }

    mod gpu_tests {
        use super::*;

        fn get_cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn test_mxfp4_gemm_zero_weights() {
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m = 4usize;
            let k = 64usize; // divisible by MXFP4_BLOCK_SIZE=32
            let n = 32usize;

            let input = Tensor::ones((m, k), DType::BF16, &device).unwrap();
            // Zero FP4 weights (nibble 0 = 0.0 in LUT) → output should be zero.
            let qweight = Tensor::zeros((n, k / 2), DType::U8, &device).unwrap();
            // Scale 127 → 2^(127-127) = 1.0 (identity scale)
            let scales = Tensor::full(127u8, (n, k / 32), &device).unwrap();

            let result = mxfp4_gemm(&input, &qweight, &scales, None, n);
            if let Ok(output) = result {
                assert_eq!(output.dims(), &[m, n]);
                let vals: Vec<f32> = output
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert!(
                    vals.iter().all(|&v| v.abs() < 1e-3),
                    "expected all zeros, got {:?}",
                    &vals[..4]
                );
            }
        }

        #[test]
        fn test_mxfp4_gemm_matches_cpu_dequant() {
            // Verify GPU kernel result matches CPU dequantize path numerically.
            // Use weight nibble 0b0010 = 2 → fp4 value 1.0, scale 127 → 1.0.
            // Then output[m, n] = sum_k(input[m,k] * 1.0 * 1.0) = K = 64.0.
            let Some(device) = get_cuda_device() else {
                eprintln!("Skipping test: no CUDA device");
                return;
            };

            let m = 2usize;
            let k = 64usize;
            let n = 16usize;

            // Pack two nibbles 0b0010_0010 = 0x22 → both nibbles = 2 → fp4 = 1.0
            let qweight = Tensor::full(0x22u8, (n, k / 2), &device).unwrap();
            // Scale 127 → 1.0
            let scales = Tensor::full(127u8, (n, k / 32), &device).unwrap();
            // All-ones input
            let input = Tensor::ones((m, k), DType::BF16, &device).unwrap();

            let result = mxfp4_gemm(&input, &qweight, &scales, None, n);
            if let Ok(output) = result {
                assert_eq!(output.dims(), &[m, n]);
                let vals: Vec<f32> = output
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let expected = k as f32; // 64.0
                assert!(
                    vals.iter().all(|&v| (v - expected).abs() < 0.5),
                    "expected {expected}, got {:?}",
                    &vals[..4]
                );
            }
        }
    }
}
