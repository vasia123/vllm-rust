//! Stage 15.C/D — Marlin tile-MMA dispatcher.
//!
//! Wires the `marlin_tile_mma_int4_bf16_m16n16k` kernel (in
//! `kernels/marlin_tile_mma.cu`) into a Rust-callable function.
//!
//! v1 fixed M = N = 16, K any multiple of 16, BF16 activations, INT4 AWQ
//! weights laid out per [`super::awq_marlin::awq_to_marlin_tile_repack_cpu`]
//! (warp_id = 0 quarters of each k-tile concatenated, 32 u32 per tile),
//! per-channel BF16 scales. See `docs/perf/marlin-tile-mma-step-15c-design.md`
//! for the full kernel design and the upcoming tensor-core path.
//!
//! The current kernel body is a software dot-product scaffold (proves
//! the build + dispatch pipeline). Stage 15.D-body replaces it with a
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
    k: i32,
}

impl CustomOp1 for MarlinTileMmaV1Op {
    fn name(&self) -> &'static str {
        "marlin_tile_mma_int4_bf16_m16n16k"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("marlin_tile_mma requires CUDA")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &storage.device;
        let func = dev.get_or_load_custom_func(
            "marlin_tile_mma_int4_bf16_m16n16k",
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
        builder.arg(&self.k);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("marlin_tile_mma launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::BF16(output),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[16, 16])))
    }
}

/// Dispatch the v1 multi-K Marlin tile-MMA kernel.
///
/// # Arguments
/// * `a` — `[16, K]` BF16 activations on CUDA. `K` must be a multiple of 16.
/// * `b_tile` — `[(K/16) * 32]` u32, concatenated warp_id=0 quarters of
///   `awq_to_marlin_tile_repack_cpu(.., K, N=64)` (one 32-u32 quarter per
///   k-tile, 16 K-rows per tile). For K=16 this is 32 u32 (single tile);
///   for K=32 it's 64 u32, etc.
/// * `scales` — `[16]` BF16 per-channel scale (group_size = K).
///
/// Returns `[16, 16]` BF16 row-major output on the same device.
pub fn dispatch_marlin_tile_mma_v1(a: &Tensor, b_tile: &Tensor, scales: &Tensor) -> Result<Tensor> {
    if a.dims().len() != 2 || a.dims()[0] != 16 || a.dtype() != DType::BF16 {
        candle_core::bail!(
            "marlin_tile_mma v1: A must be [16, K] BF16, got {:?} {:?}",
            a.dims(),
            a.dtype()
        );
    }
    let k = a.dims()[1];
    if !k.is_multiple_of(16) {
        candle_core::bail!("marlin_tile_mma v1: K ({}) must be a multiple of 16", k);
    }
    let k_tiles = k / 16;
    let expected_b_len = k_tiles * 32;
    if b_tile.dims() != [expected_b_len] || b_tile.dtype() != DType::U32 {
        candle_core::bail!(
            "marlin_tile_mma v1: B must be [{}] U32, got {:?} {:?}",
            expected_b_len,
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
        k: k as i32,
    };
    a.apply_op1(op)
}

#[cfg(all(test, feature = "gpu-test-small"))]
mod gpu_tests {
    use super::*;
    use crate::quantization::awq_marlin::awq_to_marlin_tile_repack_cpu;
    use candle_core::Device;

    /// Numeric-correctness lock-down for arbitrary K (multiple of 16).
    ///
    /// Build a deterministic AWQ qweight + per-channel scales + activations,
    /// dispatch the v1 kernel, compare against a CPU reference that does the
    /// same dequant + multiply + matmul. BF16 reduction tolerance scales
    /// with K (each summand is up to ~16 in magnitude × bf16 mantissa noise).
    fn run_lock_down(k: usize, tolerance: f32) {
        let Ok(cuda) = Device::new_cuda(0) else {
            eprintln!("no CUDA device — skipping");
            return;
        };
        assert!(k.is_multiple_of(16), "K must be multiple of 16");

        let m: usize = 16;
        let n: usize = 16;

        // 1. Activations A[M, K] BF16: deterministic small floats in [-1, 1].
        let a_vals: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i as f32 * 0.137) % 2.0) - 1.0))
            .collect();
        let a = Tensor::from_vec(a_vals.clone(), (m, k), &cuda).unwrap();

        // 2. AWQ qweight: [K, N/8] u32. Pack a deterministic logical nibble.
        let nibble = |kk: usize, nn: usize| -> u32 { ((kk * 113 + nn * 29 + 7) as u32) % 16 };
        const AWQ_PACK_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];
        let mut awq = vec![0u32; k * (n / 8)];
        for kk in 0..k {
            for nn in 0..n {
                awq[kk * (n / 8) + nn / 8] |= nibble(kk, nn) << AWQ_PACK_SHIFTS[nn % 8];
            }
        }

        // 3. Pad N=16 → N=64 to clear the producer's tile_n_size; only the
        // warp_id=0 quarter of the resulting tiles is consumed.
        let mut awq_padded = vec![0u32; k * (64 / 8)];
        for kk in 0..k {
            for col_pack in 0..(n / 8) {
                awq_padded[kk * 8 + col_pack] = awq[kk * (n / 8) + col_pack];
            }
        }
        let awq_padded_t = Tensor::from_vec(awq_padded, (k, 64 / 8), &Device::Cpu).unwrap();
        let tile_full = awq_to_marlin_tile_repack_cpu(&awq_padded_t, k, 64).unwrap();
        // tile_full shape [k/16, 128]; for each k-tile take the warp_id=0
        // quarter (positions th_id*4 + 0 for th_id in 0..32) and concat.
        let tile_full_vec: Vec<u32> = tile_full.flatten_all().unwrap().to_vec1().unwrap();
        let k_tiles = k / 16;
        let mut b_v1 = vec![0u32; k_tiles * 32];
        for kt in 0..k_tiles {
            let row_base = kt * 128;
            for th_id in 0..32 {
                b_v1[kt * 32 + th_id] = tile_full_vec[row_base + th_id * 4];
            }
        }
        let b_tile = Tensor::from_vec(b_v1, k_tiles * 32, &cuda).unwrap();

        // 4. Per-channel scales [N=16] BF16.
        let scales_vec: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.05 + (i as f32) * 0.01))
            .collect();
        let scales = Tensor::from_vec(scales_vec.clone(), n, &cuda).unwrap();

        // 5. Dispatch.
        let c = dispatch_marlin_tile_mma_v1(&a, &b_tile, &scales).unwrap();
        let c_host: Vec<half::bf16> = c.flatten_all().unwrap().to_vec1().unwrap();

        // 6. CPU reference.
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
                    diff < tolerance,
                    "K={k} mismatch at (row={row}, col={col}): cpu={acc:.6} gpu={got:.6} diff={diff:.6}"
                );
            }
        }
    }

    #[test]
    fn test_marlin_tile_mma_v1_k16_matches_cpu_reference() {
        run_lock_down(16, 5e-2);
    }

    #[test]
    fn test_marlin_tile_mma_v1_k32_matches_cpu_reference() {
        run_lock_down(32, 1e-1);
    }

    #[test]
    fn test_marlin_tile_mma_v1_k64_matches_cpu_reference() {
        run_lock_down(64, 2e-1);
    }
}
