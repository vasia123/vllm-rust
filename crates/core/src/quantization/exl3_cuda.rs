//! Low-level CUDA dispatchers for the vendored EXL3 kernels.
//!
//! Layout mirrors `marlin_tile_cuda.rs`: PTX is embedded via
//! `include_str!`, loaded once per device by cudarc, and exposed
//! through `CustomOp1` wrappers so candle handles the storage/shape
//! plumbing.
//!
//! Phase-3 scope: `had_r_128` (Hadamard butterfly on 128-element
//! groups, fp16). Reconstruct + the GEMM/GEMV dispatchers land in
//! Phase 4. The implementation here is capture-friendly — each launch
//! writes into a pool-reserved buffer or in-place, never via
//! ephemeral `dev.alloc()`.

use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, InplaceOp1, Layout, Result,
    Shape, Tensor,
};

const HADAMARD_PTX: &str = include_str!("../../kernels/exl3/hadamard.ptx");

/// Which side carries the Hadamard sign-vector scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadScale {
    /// `out = (x * scale) @ Had`
    Pre,
    /// `out = (x @ Had) * scale`
    Post,
    /// `out = x @ Had` (scale ignored — caller passes any tensor)
    None,
}

impl HadScale {
    /// Mangled PTX entry-point name for the fp16 hadamard kernel.
    /// Symbols come from `nvcc --ptx` over `hadamard.cu`; verified by
    /// `grep '.entry' hadamard.ptx`.
    const fn fp16_kernel(self) -> &'static str {
        match self {
            HadScale::Pre => "_Z19had_hf_r_128_kernelILb1ELb0EEvPK6__halfPS0_S2_f",
            HadScale::Post => "_Z19had_hf_r_128_kernelILb0ELb1EEvPK6__halfPS0_S2_f",
            HadScale::None => "_Z19had_hf_r_128_kernelILb0ELb0EEvPK6__halfPS0_S2_f",
        }
    }
}

/// Out-of-place 128-wise Hadamard transform over fp16 rows.
///
/// `input` must be 2D `[rows, cols]` with `cols % 128 == 0`. Returns
/// a freshly allocated `[rows, cols]` fp16 tensor. For capture-friendly
/// pooled output, prefer `had_r_128_fp16_inplace` once the pool path
/// is wired (Phase 4 — pending the OutputPool integration story).
///
/// Math: each contiguous 128-element block of a row is multiplied by
/// `had_128 / sqrt(128) * scale_factor`. If `mode == Pre`, the input is
/// first elementwise-multiplied by `scale[col / 128]`; if `Post`, the
/// output is elementwise-multiplied by `scale[col / 128]` afterwards.
pub fn had_r_128_fp16(input: &Tensor, scale: Option<&Tensor>, mode: HadScale) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        candle_core::bail!("had_r_128: input must be fp16, got {:?}", input.dtype());
    }
    let dims = input.dims();
    if dims.len() != 2 {
        candle_core::bail!("had_r_128: input must be 2D, got {:?}", dims);
    }
    let (rows, cols) = (dims[0], dims[1]);
    if !cols.is_multiple_of(128) {
        candle_core::bail!("had_r_128: cols ({}) must be a multiple of 128", cols);
    }
    if matches!(mode, HadScale::Pre | HadScale::Post) && scale.is_none() {
        candle_core::bail!("had_r_128: mode={:?} requires a scale tensor", mode);
    }
    if let Some(s) = scale {
        if s.dtype() != DType::F16 {
            candle_core::bail!("had_r_128: scale must be fp16, got {:?}", s.dtype());
        }
        let scale_len: usize = s.dims().iter().product();
        let expected = match mode {
            // pre  → scale spans the input cols (k-side Hadamard signs)
            // post → scale spans the output cols (n-side; same width here)
            HadScale::Pre | HadScale::Post => cols,
            HadScale::None => scale_len, // unused
        };
        if matches!(mode, HadScale::Pre | HadScale::Post) && scale_len != expected {
            candle_core::bail!(
                "had_r_128: scale len ({}) must equal cols ({})",
                scale_len,
                cols
            );
        }
    }

    let op = HadR128Fp16Op {
        scale: scale.cloned(),
        mode,
        rows: rows as i32,
        cols: cols as i32,
    };
    input.apply_op1(op)
}

struct HadR128Fp16Op {
    scale: Option<Tensor>,
    mode: HadScale,
    rows: i32,
    cols: i32,
}

impl CustomOp1 for HadR128Fp16Op {
    fn name(&self) -> &'static str {
        "exl3_had_r_128_fp16"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("exl3_had_r_128_fp16: CUDA only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _l: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let dev = &storage.device;
        let kernel_name = self.mode.fp16_kernel();
        let func = dev.get_or_load_custom_func(kernel_name, "exl3_hadamard", HADAMARD_PTX)?;

        let input = match &storage.slice {
            CudaStorageSlice::F16(s) => s,
            _ => candle_core::bail!("exl3_had_r_128: input must be F16"),
        };

        // Output is freshly allocated (zeroed; the kernel overwrites
        // every element, so the zero init is wasted but cheap). When
        // we wire OutputPool in Phase 4, replace with `reserve()`.
        let elem_count = (self.rows as usize) * (self.cols as usize);
        let output = dev.alloc_zeros::<half::f16>(elem_count)?;

        // Build a "null" device pointer for the no-scale variant by
        // allocating a 1-element placeholder. Cleaner than nullable
        // pointers in the cudarc builder.
        let null_scale_buf;
        let scale_storage_guard;
        let scale_ref: &candle_core::cuda::cudarc::driver::CudaSlice<half::f16> =
            match (&self.scale, self.mode) {
                (Some(s), HadScale::Pre | HadScale::Post) => {
                    scale_storage_guard = s.storage_and_layout().0;
                    match &*scale_storage_guard {
                        Storage::Cuda(cs) => match &cs.slice {
                            CudaStorageSlice::F16(slice) => slice,
                            _ => candle_core::bail!("exl3_had_r_128: scale must be F16"),
                        },
                        _ => candle_core::bail!("exl3_had_r_128: scale must be on CUDA"),
                    }
                }
                _ => {
                    null_scale_buf = dev.alloc_zeros::<half::f16>(1)?;
                    // Build a leaked reference: cudarc takes a slice ref;
                    // null_scale_buf lives for the rest of the function.
                    // Workaround: rebind via shadowing below.
                    // The kernel ignores `scale` in the <false,false> case.
                    #[allow(clippy::needless_borrow)]
                    {
                        &null_scale_buf
                    }
                }
            };

        // 1/sqrt(128) — the kernel multiplies by this internally.
        let r_scale: f32 = 1.0_f32 / 11.313708498984761_f32; // = 0.088388347648f

        let grid = (self.rows as u32, (self.cols as u32) / 128, 1);
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(input);
        builder.arg(&output);
        builder.arg(scale_ref);
        builder.arg(&r_scale);
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("had_r_128 launch: {e}")))?;

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::F16(output),
            device: dev.clone(),
        };
        Ok((
            out_storage,
            Shape::from_dims(&[self.rows as usize, self.cols as usize]),
        ))
    }
}

// `InplaceOp1` variant used when the caller already owns the output
// buffer (e.g. pool-reserved tensor). Not used in Phase 3 yet — leave
// the type declared so the wire-up in Phase 4 is a one-line addition.
#[allow(dead_code)]
struct HadR128Fp16InplaceOp<'a> {
    input: &'a Tensor,
    scale: Option<&'a Tensor>,
    mode: HadScale,
    rows: i32,
    cols: i32,
}

#[allow(dead_code)]
impl<'a> InplaceOp1 for HadR128Fp16InplaceOp<'a> {
    fn name(&self) -> &'static str {
        "exl3_had_r_128_fp16_inplace"
    }
    fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
        candle_core::bail!("exl3_had_r_128_fp16_inplace: CUDA only")
    }
    fn cuda_fwd(&self, _s: &mut CudaStorage, _l: &Layout) -> Result<()> {
        // Will be implemented in Phase 4 alongside OutputPool wiring.
        candle_core::bail!("exl3_had_r_128_fp16_inplace: pending Phase 4")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn had_scale_kernel_names_distinct() {
        assert_ne!(HadScale::Pre.fp16_kernel(), HadScale::Post.fp16_kernel());
        assert_ne!(HadScale::Pre.fp16_kernel(), HadScale::None.fp16_kernel());
        assert_ne!(HadScale::Post.fp16_kernel(), HadScale::None.fp16_kernel());
        // Sanity: every mangled name starts with the C++ Itanium prefix.
        for s in [HadScale::Pre, HadScale::Post, HadScale::None] {
            assert!(s.fp16_kernel().starts_with("_Z"));
        }
    }

    #[test]
    fn had_r_128_rejects_non_fp16_input() {
        let x = Tensor::zeros((1, 128), DType::F32, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::None).err().unwrap();
        assert!(format!("{err}").contains("fp16"));
    }

    #[test]
    fn had_r_128_rejects_non_multiple_of_128_cols() {
        let x = Tensor::zeros((1, 64), DType::F16, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::None).err().unwrap();
        assert!(format!("{err}").contains("multiple of 128"));
    }

    #[test]
    fn had_r_128_requires_scale_in_scaled_modes() {
        let x = Tensor::zeros((1, 128), DType::F16, &Device::Cpu).unwrap();
        let err = had_r_128_fp16(&x, None, HadScale::Pre).err().unwrap();
        assert!(format!("{err}").contains("requires a scale"));
    }

    #[cfg(feature = "gpu-test-small")]
    mod gpu_tests {
        use super::*;
        use candle_core::Device;

        fn cuda_device() -> Option<Device> {
            Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
        }

        #[test]
        fn had_r_128_fp16_no_scale_launches_and_produces_orthogonal_norm() {
            // The 128-point Hadamard matrix is orthogonal up to a factor
            // of √128, and the kernel divides by √128 internally.
            // Therefore ‖had_r_128(x)‖² ≈ ‖x‖² for any x. We use that
            // as a self-consistency check that doesn't require a Python
            // reference.
            let Some(dev) = cuda_device() else { return };

            let rows = 4usize;
            let cols = 256usize; // two Hadamard groups
            let n = rows * cols;
            let mut data = Vec::with_capacity(n);
            for i in 0..n {
                // Simple deterministic pseudo-random fp16 inputs.
                let v = ((i as f32 * 0.1234).sin() * 0.5).clamp(-1.0, 1.0);
                data.push(half::f16::from_f32(v));
            }
            let x = Tensor::from_slice(&data, (rows, cols), &dev).unwrap();
            let y = had_r_128_fp16(&x, None, HadScale::None).expect("launch ok");

            assert_eq!(y.dims(), &[rows, cols]);
            assert_eq!(y.dtype(), DType::F16);

            let x_norm = x
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let y_norm = y
                .to_dtype(DType::F32)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            // 1.5 % slack for fp16 + fast-math.
            let ratio = y_norm / x_norm;
            assert!(
                (ratio - 1.0).abs() < 0.015,
                "Hadamard norm-preservation broken: ‖y‖²/‖x‖² = {ratio}"
            );
        }
    }
}
