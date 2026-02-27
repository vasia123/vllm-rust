//! CUDA kernel binding for the Mamba2 SSD sequential scan.
//!
//! [`ssd_scan`] calls the `ssd_scan_f32` PTX kernel (one block per (batch, head)
//! pair, one thread per head_dim position).  The kernel parallelises the
//! embarrassingly parallel `(B, H, P)` dimensions while keeping the sequential
//! recurrence over `L` inside the kernel.
//!
//! # Fallback
//!
//! Returns an error (and the caller falls back to [`super::ssd::ssd_sequential`])
//! when either:
//! * `head_dim > 1024` — CUDA block size limit
//! * `d_state > 256`   — compile-time register array bound (`MAX_D_STATE`)
//!
//! # Output packing
//!
//! The kernel writes `y [B,L,H,P]` followed by `hn [B,H,P,N]` into a single
//! flat allocation.  The Rust wrapper splits this flat tensor into the two
//! return values via `narrow` + `reshape`.

use candle_core::{
    cuda::{CudaStorage, CudaStorageSlice},
    CpuStorage, CustomOp1, Layout, Result, Shape, Storage, Tensor,
};

const SSD_SCAN_PTX: &str = include_str!("../../kernels/ssd_scan.ptx");
const MAX_D_STATE: usize = 256;

// ─── CustomOp1 wrapper ────────────────────────────────────────────────────────

struct SsdScanOp {
    dt: Tensor, // [B, L, H]
    a: Tensor,  // [H]
    b: Tensor,  // [B, L, G, N]
    c: Tensor,  // [B, L, G, N]
    d: Tensor,  // [H]
    h0: Tensor, // [B, H, P, N]
    // Cached dimensions
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    n_groups: usize,
    d_state: usize,
    heads_per_group: usize,
}

impl CustomOp1 for SsdScanOp {
    fn name(&self) -> &'static str {
        "ssd_scan_f32"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ssd_scan_f32: CPU path not supported via this op")
    }

    fn cuda_fwd(&self, x_cs: &CudaStorage, _: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &x_cs.device;
        let func = dev.get_or_load_custom_func("ssd_scan_f32", "ssd_scan", SSD_SCAN_PTX)?;

        let (batch, seq_len, num_heads, head_dim) =
            (self.batch, self.seq_len, self.num_heads, self.head_dim);
        let (n_groups, d_state) = (self.n_groups, self.d_state);

        let y_elems = batch * seq_len * num_heads * head_dim;
        let hn_elems = batch * num_heads * head_dim * d_state;
        let out_elems = y_elems + hn_elems;

        let out_buf = dev.alloc_zeros::<f32>(out_elems)?;

        // Extract F32 slice from x (the primary CustomOp1 input).
        let x_f32 = match &x_cs.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("ssd_scan_f32: x must be F32"),
        };

        // Extract F32 slices from auxiliary tensors; guards must outlive launch.
        let (dt_guard, _) = self.dt.storage_and_layout();
        let (a_guard, _) = self.a.storage_and_layout();
        let (b_guard, _) = self.b.storage_and_layout();
        let (c_guard, _) = self.c.storage_and_layout();
        let (d_guard, _) = self.d.storage_and_layout();
        let (h0_guard, _) = self.h0.storage_and_layout();

        macro_rules! f32_slice {
            ($guard:ident, $name:literal) => {
                match &*$guard {
                    Storage::Cuda(cs) => match &cs.slice {
                        CudaStorageSlice::F32(s) => s,
                        _ => candle_core::bail!(concat!("ssd_scan_f32: ", $name, " must be F32")),
                    },
                    _ => candle_core::bail!(concat!("ssd_scan_f32: ", $name, " must be on CUDA")),
                }
            };
        }

        let dt_f32 = f32_slice!(dt_guard, "dt");
        let a_f32 = f32_slice!(a_guard, "a");
        let b_f32 = f32_slice!(b_guard, "b");
        let c_f32 = f32_slice!(c_guard, "c");
        let d_f32 = f32_slice!(d_guard, "d");
        let h0_f32 = f32_slice!(h0_guard, "h0");

        let cfg = LaunchConfig {
            grid_dim: ((batch * num_heads) as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            // Shared memory: b_sm[d_state] + c_sm[d_state]
            shared_mem_bytes: (2 * d_state * std::mem::size_of::<f32>()) as u32,
        };

        // Integer kernel parameters (all i32).
        let y_elems_i = y_elems as i32;
        let seq_len_i = seq_len as i32;
        let num_heads_i = num_heads as i32;
        let head_dim_i = head_dim as i32;
        let n_groups_i = n_groups as i32;
        let d_state_i = d_state as i32;
        let hpg_i = self.heads_per_group as i32;

        let mut builder = func.builder();
        builder.arg(&out_buf);
        builder.arg(x_f32);
        builder.arg(dt_f32);
        builder.arg(a_f32);
        builder.arg(b_f32);
        builder.arg(c_f32);
        builder.arg(d_f32);
        builder.arg(h0_f32);
        builder.arg(&y_elems_i);
        builder.arg(&seq_len_i);
        builder.arg(&num_heads_i);
        builder.arg(&head_dim_i);
        builder.arg(&n_groups_i);
        builder.arg(&d_state_i);
        builder.arg(&hpg_i);

        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("ssd_scan_f32 launch: {e}")))?;

        // Drop guards before constructing output (releases read locks).
        drop(dt_guard);
        drop(a_guard);
        drop(b_guard);
        drop(c_guard);
        drop(d_guard);
        drop(h0_guard);

        let out_storage = CudaStorage {
            slice: CudaStorageSlice::F32(out_buf),
            device: dev.clone(),
        };
        Ok((out_storage, Shape::from_dims(&[out_elems])))
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// GPU-accelerated Mamba2 SSD sequential scan.
///
/// Dispatches to the `ssd_scan_f32` CUDA kernel when all tensors are on a CUDA
/// device, `head_dim ≤ 1024`, and `d_state ≤ 256`.
///
/// Returns `Err` (not `bail!`) with a descriptive message when constraints
/// aren't met, so callers can fall back to the CPU path.
///
/// # Arguments
/// Same signature as [`super::ssd::ssd_sequential`].
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    state: &Tensor,
    heads_per_group: usize,
) -> Result<(Tensor, Tensor)> {
    let x_dims = x.dims();
    let (batch, seq_len, num_heads, head_dim) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    let b_dims = b.dims();
    let (n_groups, d_state) = (b_dims[2], b_dims[3]);

    // Validate constraints before launching.
    if head_dim > 1024 {
        candle_core::bail!(
            "ssd_scan_f32: head_dim {head_dim} exceeds CUDA block-size limit (1024)"
        );
    }
    if d_state > MAX_D_STATE {
        candle_core::bail!("ssd_scan_f32: d_state {d_state} exceeds MAX_D_STATE ({MAX_D_STATE})");
    }

    // Make all inputs contiguous so kernel strides are implicit.
    let x = x.contiguous()?;
    let dt = dt.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    let d = d.contiguous()?;
    let state = state.contiguous()?;

    let op = SsdScanOp {
        dt,
        a,
        b,
        c,
        d,
        h0: state,
        batch,
        seq_len,
        num_heads,
        head_dim,
        n_groups,
        d_state,
        heads_per_group,
    };

    // apply_op1 calls cuda_fwd and returns the packed flat tensor.
    let packed = x.apply_op1(op)?;

    // Split flat output: [y_elems | hn_elems]
    let y_elems = batch * seq_len * num_heads * head_dim;
    let hn_elems = batch * num_heads * head_dim * d_state;

    let y = packed
        .narrow(0, 0, y_elems)?
        .reshape(&[batch, seq_len, num_heads, head_dim])?;
    let hn = packed
        .narrow(0, y_elems, hn_elems)?
        .reshape(&[batch, num_heads, head_dim, d_state])?;

    Ok((y, hn))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::ssd::ssd_sequential;
    use candle_core::{DType, Device};

    fn get_cuda() -> Option<Device> {
        Device::cuda_if_available(0).ok().filter(|d| d.is_cuda())
    }

    #[test]
    fn ssd_cuda_requires_f32() {
        let Some(dev) = get_cuda() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };
        // Pass BF16 tensor — should produce an error.
        let x = Tensor::zeros((1usize, 4usize, 2usize, 8usize), DType::BF16, &dev).unwrap();
        let dt = Tensor::ones((1usize, 4usize, 2usize), DType::BF16, &dev).unwrap();
        let a = Tensor::ones((2usize,), DType::BF16, &dev).unwrap();
        let b = Tensor::zeros((1usize, 4usize, 1usize, 4usize), DType::BF16, &dev).unwrap();
        let c = Tensor::zeros((1usize, 4usize, 1usize, 4usize), DType::BF16, &dev).unwrap();
        let d = Tensor::zeros((2usize,), DType::BF16, &dev).unwrap();
        let h0 = Tensor::zeros((1usize, 2usize, 8usize, 4usize), DType::BF16, &dev).unwrap();
        assert!(ssd_scan(&x, &dt, &a, &b, &c, &d, &h0, 2).is_err());
    }

    /// GPU result must numerically match the CPU sequential reference.
    #[test]
    fn ssd_cuda_matches_cpu() {
        let Some(cuda_dev) = get_cuda() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;

        const B: usize = 1;
        const L: usize = 8;
        const H: usize = 4;
        const P: usize = 8; // head_dim
        const G: usize = 2; // n_groups
        const N: usize = 4; // d_state
        const HPG: usize = H / G;

        // Build inputs on CPU, then copy to CUDA.
        let x_cpu = Tensor::randn(0f32, 0.1, (B, L, H, P), &cpu).unwrap();
        let dt_cpu = (Tensor::ones((B, L, H), DType::F32, &cpu).unwrap() * 0.1f64).unwrap();
        let a_cpu = (Tensor::ones((H,), DType::F32, &cpu).unwrap() * -1.0f64).unwrap();
        let b_cpu = Tensor::randn(0f32, 0.1, (B, L, G, N), &cpu).unwrap();
        let c_cpu = Tensor::randn(0f32, 0.1, (B, L, G, N), &cpu).unwrap();
        let d_cpu = Tensor::zeros((H,), DType::F32, &cpu).unwrap();
        let h0_cpu = Tensor::zeros((B, H, P, N), DType::F32, &cpu).unwrap();

        let (y_ref, hn_ref) = ssd_sequential(
            &x_cpu, &dt_cpu, &a_cpu, &b_cpu, &c_cpu, &d_cpu, &h0_cpu, HPG,
        )
        .unwrap();

        // Copy to CUDA.
        let to_cuda = |t: &Tensor| t.to_device(&cuda_dev).unwrap();
        let (x_gpu, dt_gpu, a_gpu, b_gpu, c_gpu, d_gpu, h0_gpu) = (
            to_cuda(&x_cpu),
            to_cuda(&dt_cpu),
            to_cuda(&a_cpu),
            to_cuda(&b_cpu),
            to_cuda(&c_cpu),
            to_cuda(&d_cpu),
            to_cuda(&h0_cpu),
        );

        let (y_gpu, hn_gpu) = ssd_scan(
            &x_gpu, &dt_gpu, &a_gpu, &b_gpu, &c_gpu, &d_gpu, &h0_gpu, HPG,
        )
        .unwrap();

        // Compare: copy GPU result to CPU, then diff.
        let y_gpu_cpu = y_gpu.to_device(&cpu).unwrap();
        let hn_gpu_cpu = hn_gpu.to_device(&cpu).unwrap();

        let y_diff: f32 = ((&y_ref - &y_gpu_cpu).unwrap())
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(y_diff < 1e-4, "y max diff GPU vs CPU = {y_diff}");

        let hn_diff: f32 = ((&hn_ref - &hn_gpu_cpu).unwrap())
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(hn_diff < 1e-4, "hn max diff GPU vs CPU = {hn_diff}");
    }

    /// Check output shapes are correct.
    #[test]
    fn ssd_cuda_output_shape() {
        let Some(dev) = get_cuda() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };

        let x = Tensor::zeros((2usize, 16usize, 8usize, 16usize), DType::F32, &dev).unwrap();
        let dt = Tensor::ones((2usize, 16usize, 8usize), DType::F32, &dev).unwrap();
        let a = (Tensor::ones((8usize,), DType::F32, &dev).unwrap() * -1.0f64).unwrap();
        let b = Tensor::zeros((2usize, 16usize, 4usize, 8usize), DType::F32, &dev).unwrap();
        let c = Tensor::zeros((2usize, 16usize, 4usize, 8usize), DType::F32, &dev).unwrap();
        let d = Tensor::zeros((8usize,), DType::F32, &dev).unwrap();
        let h0 = Tensor::zeros((2usize, 8usize, 16usize, 8usize), DType::F32, &dev).unwrap();

        let (y, hn) = ssd_scan(&x, &dt, &a, &b, &c, &d, &h0, 2).unwrap();
        assert_eq!(y.dims(), [2, 16, 8, 16]);
        assert_eq!(hn.dims(), [2, 8, 16, 8]);
    }
}
