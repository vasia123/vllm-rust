//! Stage 15.C — Minimum-viable Marlin tile-MMA dispatcher.
//!
//! Wires the `marlin_tile_mma_int4_bf16_m16n16k16` kernel (in
//! `kernels/marlin_tile_mma.cu`) into a Rust-callable function.
//!
//! v1 hard-coded shapes: M = N = K = 16, BF16 activations, INT4 AWQ
//! weights laid out per [`super::awq_marlin::awq_to_marlin_tile_repack_cpu`]
//! (warp_id = 0 quarter of the K=16/N=64 tile, 32 u32), per-channel BF16
//! scales. See `docs/perf/marlin-tile-mma-step-15c-design.md` for the
//! full kernel design and the upcoming tensor-core path.
//!
//! The current kernel body is a software dot-product scaffold (proves
//! the build + dispatch pipeline). Stage 15.C-body replaces it with a
//! tensor-core mma.m16n8k16 path; the Rust dispatcher does not change.

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Storage, Tensor,
};

const MARLIN_TILE_MMA_PTX: &str = include_str!("../../kernels/marlin_tile_mma.ptx");

/// CustomOp wrapping the tile-MMA dispatch. The activation `A` is the
/// Op's primary tensor (so we get its CudaStorage via the trait); the
/// repacked weight `B` and per-channel scales `S` are held as struct
/// fields and resolved through `storage_and_layout()` inside `cuda_fwd`.
struct MarlinTileMmaV1Op {
    b_tile: Tensor,
    scales: Tensor,
}

impl CustomOp1 for MarlinTileMmaV1Op {
    fn name(&self) -> &'static str {
        "marlin_tile_mma_int4_bf16_m16n16k16"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("marlin_tile_mma requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func = dev.get_or_load_custom_func(
            "marlin_tile_mma_int4_bf16_m16n16k16",
            "marlin_tile_mma",
            MARLIN_TILE_MMA_PTX,
        )?;

        let a = match &storage.slice {
            CudaStorageSlice::BF16(s) => s,
            _ => candle_core::bail!("marlin_tile_mma: A must be BF16"),
        };

        let (b_guard, _) = self.b_tile.storage_and_layout();
        let b = match &*b_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("marlin_tile_mma: B must be U32"),
            },
            _ => candle_core::bail!("marlin_tile_mma: B must be on CUDA"),
        };

        let (s_guard, _) = self.scales.storage_and_layout();
        let scales = match &*s_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::BF16(s) => s,
                _ => candle_core::bail!("marlin_tile_mma: S must be BF16"),
            },
            _ => candle_core::bail!("marlin_tile_mma: S must be on CUDA"),
        };

        // v1 hard-coded: M = N = K = 16. Output [16, 16] BF16 = 256 elements.
        let output = dev.alloc_zeros::<half::bf16>(256)?;

        // Scaffold uses 256 threads (one per output cell) + 1 block. The
        // tensor-core body will switch this to (32, 1, 1) (1 warp).
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(a);
        builder.arg(b);
        builder.arg(scales);
        builder.arg(&output);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_tile_mma launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[16, 16])))
    }
}

/// Dispatch the v1 single-tile Marlin tile-MMA kernel.
///
/// # Arguments
/// * `a` — `[16, 16]` BF16 activations on CUDA.
/// * `b_tile` — `[32]` u32, the warp_id=0 quarter of an
///   `awq_to_marlin_tile_repack_cpu(.., K=16, N=64)` output (i.e. the
///   first 32 of the 128 u32 in the single tile).
/// * `scales` — `[16]` BF16 per-channel scale.
///
/// Returns `[16, 16]` BF16 row-major output on the same device.
pub fn dispatch_marlin_tile_mma_v1(a: &Tensor, b_tile: &Tensor, scales: &Tensor) -> Result<Tensor> {
    if a.dims() != [16, 16] || a.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: A must be [16,16] BF16, got {:?} {:?}",
            a.dims(),
            a.dtype()
        );
    }
    if b_tile.dims() != [32] || b_tile.dtype() != DType::U32 {
        candle_core::bail!(
            "marlin_tile_mma v1: B must be [32] U32, got {:?} {:?}",
            b_tile.dims(),
            b_tile.dtype()
        );
    }
    if scales.dims() != [16] || scales.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: S must be [16] BF16, got {:?} {:?}",
            scales.dims(),
            scales.dtype()
        );
    }

    let op = MarlinTileMmaV1Op {
        b_tile: b_tile.clone(),
        scales: scales.clone(),
    };
    a.apply_op1(op)
}

#[cfg(all(test, feature = "gpu-test-small"))]
mod gpu_tests {
    use super::*;
    use crate::quantization::awq_marlin::awq_to_marlin_tile_repack_cpu;
    use candle_core::Device;

    /// Stage 15.C numeric-correctness lock-down.
    ///
    /// Build a deterministic AWQ qweight + per-channel scales + activations,
    /// dispatch the v1 kernel, compare against a CPU reference that does
    /// the same dequant + multiply + matmul. BF16 reduction over K=16 has
    /// limited precision; tolerance 5e-2 absolute (well within bf16 mantissa
    /// noise for K=16 sums of nibbles 0..15 × scales 0..1 × activations -1..1).
    #[test]
    fn test_marlin_tile_mma_v1_matches_cpu_reference() {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };

        let m: usize = 16;
        let n: usize = 16;
        let k: usize = 16;

        // 1. Activations A[M, K] BF16: deterministic small floats in [-1, 1].
        let a_vals: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.137) % 2.0) - 1.0))
            .collect();
        let a = Tensor::from_vec(a_vals.clone(), (m, k), &cuda).unwrap();

        // 2. AWQ qweight: K=16, N=16 logical nibbles. AWQ format = [K, N/8] u32.
        // Pack a deterministic logical nibble at every (k, n).
        let nibble = |kk: usize, nn: usize| -> u32 { ((kk * 113 + nn * 29 + 7) as u32) % 16 };
        const AWQ_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
        let mut awq = vec![0u32; k * (n / 8)];
        for kk in 0..k {
            for nn in 0..n {
                awq[kk * (n / 8) + nn / 8] |= nibble(kk, nn) << AWQ_PACK_SHIFTS[nn % 8];
            }
        }

        // 3. Pad to N=64 (single Marlin tile), repack, take first 32 u32 slice.
        // The producer's tile_n_size is 64; padding columns 16..64 with zeros
        // is harmless because the v1 kernel only reads the warp_id=0 quarter.
        let mut awq_padded = vec![0u32; k * (64 / 8)];
        for kk in 0..k {
            for col_pack in 0..(n / 8) {
                awq_padded[kk * 8 + col_pack] = awq[kk * (n / 8) + col_pack];
            }
        }
        let awq_padded_t = Tensor::from_vec(awq_padded, (k, 64 / 8), &Device::Cpu).unwrap();
        let tile_full = awq_to_marlin_tile_repack_cpu(&awq_padded_t, 16, 64).unwrap();
        // tile_full shape [1, 128]; take first 32 u32 (warp_id=0 quarter,
        // i.e. positions th_id*4 + 0 for th_id in 0..32, which are
        // 0, 4, 8, ..., 124).
        let tile_full_vec: Vec<u32> = tile_full.flatten_all().unwrap().to_vec1().unwrap();
        let mut b_v1 = vec![0u32; 32];
        for th_id in 0..32 {
            b_v1[th_id] = tile_full_vec[th_id * 4];
        }
        let b_tile = Tensor::from_vec(b_v1, 32, &cuda).unwrap();

        // 4. Per-channel scales [N=16] BF16.
        let scales_vec: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.05 + (i as f32) * 0.01))
            .collect();
        let scales = Tensor::from_vec(scales_vec.clone(), n, &cuda).unwrap();

        // 5. Dispatch.
        let c = dispatch_marlin_tile_mma_v1(&a, &b_tile, &scales).unwrap();
        let c_host: Vec<half::bf16> = c.flatten_all().unwrap().to_vec1().unwrap();

        // 6. CPU reference: dequant nibble * per-channel scale * activation, sum over K.
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let w = nibble(kk, col) as f32 * scales_vec[col].to_f32();
                    let av = a_vals[row * k + kk].to_f32();
                    acc += av * w;
                }
                let got = c_host[row * n + col].to_f32();
                let diff = (got - acc).abs();
                assert!(
                    diff < 5e-2,
                    "mismatch at (row={row}, col={col}): cpu={acc:.6} gpu={got:.6} diff={diff:.6}"
                );
            }
        }
    }
}
