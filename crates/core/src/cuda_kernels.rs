#[cfg(feature = "cuda-fused-activations")]
use candle_core::CustomOp2;
use candle_core::{
    cuda::CudaStorageSlice, CpuStorage, CudaStorage, CustomOp1, DType, InplaceOp2, Layout, Result,
    Shape, Storage, Tensor,
};

use crate::kv_cache::quantization::KVCacheDtype;

/// Clone a CudaStorageSlice by pattern-matching each variant.
/// CudaStorageSlice does not derive Clone, but inner CudaSlice<T> does.
#[cfg(feature = "cuda")]
fn clone_cuda_storage_slice(slice: &CudaStorageSlice) -> CudaStorageSlice {
    match slice {
        CudaStorageSlice::U8(s) => CudaStorageSlice::U8(s.clone()),
        CudaStorageSlice::U32(s) => CudaStorageSlice::U32(s.clone()),
        CudaStorageSlice::I64(s) => CudaStorageSlice::I64(s.clone()),
        CudaStorageSlice::BF16(s) => CudaStorageSlice::BF16(s.clone()),
        CudaStorageSlice::F16(s) => CudaStorageSlice::F16(s.clone()),
        CudaStorageSlice::F32(s) => CudaStorageSlice::F32(s.clone()),
        CudaStorageSlice::F64(s) => CudaStorageSlice::F64(s.clone()),
        // candle 0.10 added I16/I32/F8E4M3/F6E2M3/F6E3M2/F4/F8E8M0; we don't
        // use them in any kernel that exercises this clone helper, so panic
        // loudly if one ever reaches here rather than silently dropping data.
        _ => panic!("clone_cuda_storage_slice: unsupported variant in candle 0.10"),
    }
}

const PTX: &str = include_str!("../kernels/paged_attention.ptx");

#[cfg(feature = "cuda-fused-activations")]
const SWIGLU_PTX: &str = include_str!("../kernels/swiglu.ptx");

/// SM version this binary was compiled against (set by build.rs).
/// Returns 0 when cuda-kernels feature was not enabled at build time.
pub fn cuda_target_sm() -> u32 {
    option_env!("CUDA_TARGET_SM")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Check if FP8 CUDA kernels are available (requires sm_89+).
pub fn fp8_kernels_available() -> bool {
    cuda_target_sm() >= 89
}

const NUM_WARPS: usize = 4;

/// Validate the V2 paged-attention contract `max(seq_lens) <= max_seq_len`.
///
/// `max_seq_len` controls how many partition slots the V2 path allocates and
/// how many `tmp_out` partitions Stage-1 actually writes. The Stage-2 reduce
/// kernel iterates over `ceil(seq_len / PARTITION_SIZE)` partitions per
/// sequence — if any `seq_lens[i] > max_seq_len`, reduce reads beyond the
/// initialised range and silently produces wrong outputs. The check is gated
/// behind `debug_assertions` so release builds avoid the device→host sync.
fn debug_assert_seq_lens_within_bound(seq_lens: &Tensor, max_seq_len: usize) -> Result<()> {
    if cfg!(debug_assertions) {
        let lens: Vec<u32> = seq_lens.to_vec1()?;
        if let Some(&actual_max) = lens.iter().max() {
            if (actual_max as usize) > max_seq_len {
                candle_core::bail!(
                    "paged_attention_v2: contract violation — seq_lens.max()={} exceeds max_seq_len={}",
                    actual_max,
                    max_seq_len
                );
            }
        }
    }
    Ok(())
}

/// Storage dtype for PagedAttention kernels.
///
/// Implementations name the kernel symbols and provide CudaStorageSlice
/// projections for their concrete float type. Adding a new dtype (e.g. fp8
/// activations) is one impl block — no kernel-launch code changes.
trait PagedAttnDtype:
    candle_core::cuda::cudarc::driver::DeviceRepr
    + candle_core::cuda::cudarc::driver::ValidAsZeroBits
    + Default
    + Copy
    + 'static
{
    /// Kernel symbols for Q dtype × KV cache dtype combinations. See
    /// `crates/core/kernels/paged_attention.cu` for the underlying
    /// implementations. `_AUTO` means KV cache stored at the same dtype
    /// as Q (native fp16 / bf16); FP8 / INT8 entries handle quantized KV
    /// cache with inline kernel-side dequantization driven by per-tensor
    /// `k_scale` / `v_scale` device pointers.
    const KERNEL_V1_AUTO: &'static str;
    const KERNEL_V1_AUTO_ALIBI: &'static str;
    const KERNEL_V1_FP8_E4M3: &'static str;
    const KERNEL_V1_FP8_E5M2: &'static str;
    const KERNEL_V1_INT8: &'static str;
    // Phase 10: ALiBi + FP8/INT8 KV cache combinations.
    const KERNEL_V1_FP8_E4M3_ALIBI: &'static str;
    const KERNEL_V1_FP8_E5M2_ALIBI: &'static str;
    const KERNEL_V1_INT8_ALIBI: &'static str;
    const KERNEL_V2_AUTO: &'static str;
    const KERNEL_V2_AUTO_ALIBI: &'static str;
    const KERNEL_V2_FP8_E4M3: &'static str;
    const KERNEL_V2_FP8_E5M2: &'static str;
    const KERNEL_V2_INT8: &'static str;
    const KERNEL_V2_FP8_E4M3_ALIBI: &'static str;
    const KERNEL_V2_FP8_E5M2_ALIBI: &'static str;
    const KERNEL_V2_INT8_ALIBI: &'static str;
    const KERNEL_V2_REDUCE: &'static str;
    const NAME: &'static str;

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>>;
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice;
}

impl PagedAttnDtype for half::bf16 {
    const KERNEL_V1_AUTO: &'static str = "paged_attention_v1_bf16";
    const KERNEL_V1_AUTO_ALIBI: &'static str = "paged_attention_v1_bf16_alibi";
    const KERNEL_V1_FP8_E4M3: &'static str = "paged_attention_v1_bf16_fp8e4m3";
    const KERNEL_V1_FP8_E5M2: &'static str = "paged_attention_v1_bf16_fp8e5m2";
    const KERNEL_V1_INT8: &'static str = "paged_attention_v1_bf16_int8";
    const KERNEL_V1_FP8_E4M3_ALIBI: &'static str = "paged_attention_v1_bf16_fp8e4m3_alibi";
    const KERNEL_V1_FP8_E5M2_ALIBI: &'static str = "paged_attention_v1_bf16_fp8e5m2_alibi";
    const KERNEL_V1_INT8_ALIBI: &'static str = "paged_attention_v1_bf16_int8_alibi";
    const KERNEL_V2_AUTO: &'static str = "paged_attention_v2_bf16";
    const KERNEL_V2_AUTO_ALIBI: &'static str = "paged_attention_v2_bf16_alibi";
    const KERNEL_V2_FP8_E4M3: &'static str = "paged_attention_v2_bf16_fp8e4m3";
    const KERNEL_V2_FP8_E5M2: &'static str = "paged_attention_v2_bf16_fp8e5m2";
    const KERNEL_V2_INT8: &'static str = "paged_attention_v2_bf16_int8";
    const KERNEL_V2_FP8_E4M3_ALIBI: &'static str = "paged_attention_v2_bf16_fp8e4m3_alibi";
    const KERNEL_V2_FP8_E5M2_ALIBI: &'static str = "paged_attention_v2_bf16_fp8e5m2_alibi";
    const KERNEL_V2_INT8_ALIBI: &'static str = "paged_attention_v2_bf16_int8_alibi";
    const KERNEL_V2_REDUCE: &'static str = "paged_attention_v2_reduce_bf16";
    const NAME: &'static str = "bf16";

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>> {
        match s {
            CudaStorageSlice::BF16(s) => Some(s),
            _ => None,
        }
    }
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice {
        CudaStorageSlice::BF16(s)
    }
}

impl PagedAttnDtype for half::f16 {
    const KERNEL_V1_AUTO: &'static str = "paged_attention_v1_f16";
    const KERNEL_V1_AUTO_ALIBI: &'static str = "paged_attention_v1_f16_alibi";
    const KERNEL_V1_FP8_E4M3: &'static str = "paged_attention_v1_f16_fp8e4m3";
    const KERNEL_V1_FP8_E5M2: &'static str = "paged_attention_v1_f16_fp8e5m2";
    const KERNEL_V1_INT8: &'static str = "paged_attention_v1_f16_int8";
    const KERNEL_V1_FP8_E4M3_ALIBI: &'static str = "paged_attention_v1_f16_fp8e4m3_alibi";
    const KERNEL_V1_FP8_E5M2_ALIBI: &'static str = "paged_attention_v1_f16_fp8e5m2_alibi";
    const KERNEL_V1_INT8_ALIBI: &'static str = "paged_attention_v1_f16_int8_alibi";
    const KERNEL_V2_AUTO: &'static str = "paged_attention_v2_f16";
    const KERNEL_V2_AUTO_ALIBI: &'static str = "paged_attention_v2_f16_alibi";
    const KERNEL_V2_FP8_E4M3: &'static str = "paged_attention_v2_f16_fp8e4m3";
    const KERNEL_V2_FP8_E5M2: &'static str = "paged_attention_v2_f16_fp8e5m2";
    const KERNEL_V2_INT8: &'static str = "paged_attention_v2_f16_int8";
    const KERNEL_V2_FP8_E4M3_ALIBI: &'static str = "paged_attention_v2_f16_fp8e4m3_alibi";
    const KERNEL_V2_FP8_E5M2_ALIBI: &'static str = "paged_attention_v2_f16_fp8e5m2_alibi";
    const KERNEL_V2_INT8_ALIBI: &'static str = "paged_attention_v2_f16_int8_alibi";
    const KERNEL_V2_REDUCE: &'static str = "paged_attention_v2_reduce_f16";
    const NAME: &'static str = "f16";

    fn slice_from(
        s: &CudaStorageSlice,
    ) -> Option<&candle_core::cuda::cudarc::driver::CudaSlice<Self>> {
        match s {
            CudaStorageSlice::F16(s) => Some(s),
            _ => None,
        }
    }
    fn into_storage_slice(
        s: candle_core::cuda::cudarc::driver::CudaSlice<Self>,
    ) -> CudaStorageSlice {
        CudaStorageSlice::F16(s)
    }
}

struct PagedAttnOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    /// KV cache storage dtype. `Auto` => K/V cache stored at Q dtype
    /// (fp16/bf16). Quantized variants (`Fp8E4m3`, `Fp8E5m2`, `Int8`)
    /// store K/V as packed U8 bytes and require per-tensor `k_scale` /
    /// `v_scale` device pointers for inline kernel-side dequantization.
    kv_cache_dtype: KVCacheDtype,
    /// Per-tensor scalar F32 scale for K cache (length-1 device tensor).
    /// Required when `kv_cache_dtype != Auto`. Default value 1.0 maps
    /// the symmetric input range linearly to the FP8/INT8 storage and
    /// matches the write-side scale.
    k_scale: Option<Tensor>,
    /// Per-tensor scalar F32 scale for V cache. See [`Self::k_scale`].
    v_scale: Option<Tensor>,
}

impl CustomOp1 for PagedAttnOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1 requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v1::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v1::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention expects bf16 or f16 Q tensor"),
        }
    }
}

// ============================================================================
// Paged-attention scale plumbing helpers
// ============================================================================
//
// Per-tensor scales (`k_scale`, `v_scale`) for quantized KV cache live in
// `CacheEngine::scales` as length-1 F32 device tensors. We need to extract
// the device `CudaSlice<f32>` and pass it to the kernel — but Tensor's
// storage guard has lifetime tied to the Tensor, so callers must hold the
// guard alive across the kernel-launch closure. These helpers centralise
// the boilerplate while preserving the lifetime contract.
//
// `validate_scales_for_kv_dtype` performs eager validation: quantized cache
// dtypes require BOTH scales to be present and F32. Auto-mode silently
// drops scales (they aren't used). Diagnoses misconfigurations at the
// API boundary rather than as a runtime kernel-launch failure.
fn validate_scales_for_kv_dtype(
    kv_cache_dtype: KVCacheDtype,
    k_scale: &Option<Tensor>,
    v_scale: &Option<Tensor>,
) -> Result<()> {
    if kv_cache_dtype == KVCacheDtype::Auto {
        return Ok(());
    }
    let k = k_scale.as_ref().ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "paged_attention: kv_cache_dtype={:?} requires `k_scale` tensor (per-tensor F32 scalar)",
            kv_cache_dtype
        ))
    })?;
    let v = v_scale.as_ref().ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "paged_attention: kv_cache_dtype={:?} requires `v_scale` tensor (per-tensor F32 scalar)",
            kv_cache_dtype
        ))
    })?;
    if k.dtype() != DType::F32 {
        candle_core::bail!("paged_attention: k_scale must be F32, got {:?}", k.dtype());
    }
    if v.dtype() != DType::F32 {
        candle_core::bail!("paged_attention: v_scale must be F32, got {:?}", v.dtype());
    }
    if k.elem_count() != 1 || v.elem_count() != 1 {
        candle_core::bail!(
            "paged_attention: per-tensor scales must have 1 element each, got k={} v={}",
            k.elem_count(),
            v.elem_count()
        );
    }
    Ok(())
}

// WORKAROUND: cudarc 0.19 does not expose `impl PushKernelArg<Option<&CudaSlice<T>>>`
// — its trait-impl matrix only covers `&CudaSlice<T>` / `&CudaSlice<T> mut`.
// To pass an *optional* device pointer (kernel param `const float* k_scale_ptr`)
// we need 8 bytes of zero serialised into the kernel-arg slot when the
// Rust side has `None`. The `&u64` `PushKernelArg<&T: DeviceRepr>` impl
// memcpys 8 bytes from the host stack into the launch's args array;
// cuLaunchKernel then forwards those 8 bytes to the kernel which sees
// a nullptr `const float*`. The paged_attention kernel helpers test
// against nullptr and fall back to scale=1.0.
//
// Drop this whole workaround when cudarc grows native `Option<&CudaSlice>`
// support (upstream issue tracker: cudarc/issues). Until then, keep
// this static const + the `push_scale_args` helper as the single chokepoint.
static NULL_DEV_PTR_VALUE: u64 = 0;

/// Extract a length-1 F32 device slice (per-tensor scale) from a tensor's
/// storage guard. Validated at the API boundary (see
/// [`validate_scales_for_kv_dtype`]) so this only handles the happy path.
fn extract_f32_scalar<'g, D>(
    guard: &'g D,
    name: &str,
) -> Result<&'g candle_core::cuda::cudarc::driver::CudaSlice<f32>>
where
    D: std::ops::Deref<Target = Storage>,
{
    match &**guard {
        Storage::Cuda(cs) => match &cs.slice {
            CudaStorageSlice::F32(s) => Ok(s),
            _ => candle_core::bail!(
                "paged_attention: {} scale must be F32 on CUDA, got non-F32 slice",
                name
            ),
        },
        _ => candle_core::bail!("paged_attention: {} scale must be on CUDA", name),
    }
}

/// Extract a native-dtype (`T`) K/V cache slice from a storage guard.
/// Used on the Auto path where K/V cache shares Q's dtype.
fn expect_native_kv_slice<'g, T: PagedAttnDtype>(
    guard: &'g (impl std::ops::Deref<Target = Storage>),
    name: &str,
) -> Result<&'g candle_core::cuda::cudarc::driver::CudaSlice<T>> {
    match &**guard {
        Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention: {} cache dtype must match Q ({}) on Auto KV path",
                name,
                T::NAME
            ))
        }),
        _ => candle_core::bail!("paged_attention: {} cache must be on CUDA", name),
    }
}

/// Extract a U8 K/V cache slice from a storage guard. Used on the
/// quantized path (FP8 E4M3/E5M2, INT8) where K/V cache is stored as
/// raw bytes regardless of the underlying numeric format. The byte-vs-
/// signed-vs-FP8 reinterpretation is done kernel-side by the matching
/// `load_kv_to_f32` template specialization.
fn expect_u8_kv_slice<'g>(
    guard: &'g (impl std::ops::Deref<Target = Storage>),
    name: &str,
) -> Result<&'g candle_core::cuda::cudarc::driver::CudaSlice<u8>> {
    match &**guard {
        Storage::Cuda(cs) => match &cs.slice {
            CudaStorageSlice::U8(s) => Ok(s),
            _ => candle_core::bail!(
                "paged_attention: quantized {} cache must be U8 byte storage, found other slice",
                name
            ),
        },
        _ => candle_core::bail!("paged_attention: {} cache must be on CUDA", name),
    }
}

/// Push the `(k_scale_ptr, v_scale_ptr)` trailing kernel args. For Auto
/// mode the slices are `None` and we push two device-side null pointers
/// via [`NULL_DEV_PTR_VALUE`].
fn push_scale_args<'b>(
    builder: &mut candle_core::cuda::cudarc::driver::LaunchArgs<'b>,
    k_scale: Option<&'b candle_core::cuda::cudarc::driver::CudaSlice<f32>>,
    v_scale: Option<&'b candle_core::cuda::cudarc::driver::CudaSlice<f32>>,
) {
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    match k_scale {
        Some(s) => {
            builder.arg(s);
        }
        None => {
            builder.arg(&NULL_DEV_PTR_VALUE);
        }
    }
    match v_scale {
        Some(s) => {
            builder.arg(s);
        }
        None => {
            builder.arg(&NULL_DEV_PTR_VALUE);
        }
    }
}

impl PagedAttnOp {
    /// Pick the kernel symbol for the current Q dtype × KV cache dtype
    /// combination. `Auto` keeps the legacy path; quantized variants
    /// route to the FP8/INT8 inline-dequant kernels.
    fn kernel_symbol_v1<T: PagedAttnDtype>(&self) -> &'static str {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => T::KERNEL_V1_AUTO,
            KVCacheDtype::Fp8E4m3 => T::KERNEL_V1_FP8_E4M3,
            KVCacheDtype::Fp8E5m2 => T::KERNEL_V1_FP8_E5M2,
            KVCacheDtype::Int8 => T::KERNEL_V1_INT8,
        }
    }

    fn run_v1<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        validate_scales_for_kv_dtype(self.kv_cache_dtype, &self.k_scale, &self.v_scale)?;

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention: Q dtype mismatch with selected kernel ({})",
                T::NAME
            ))
        })?;

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        // Per-tensor F32 scales — guards must outlive the kernel launch.
        let k_scale_guard_layout = self.k_scale.as_ref().map(|t| t.storage_and_layout());
        let v_scale_guard_layout = self.v_scale.as_ref().map(|t| t.storage_and_layout());
        let k_scale_slice = k_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "k_scale"))
            .transpose()?;
        let v_scale_slice = v_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "v_scale"))
            .transpose()?;

        let head_dim = self.head_dim;
        let elem_count = num_seqs * self.num_heads * head_dim;
        // SAFETY: every (seq, head) block writes a full `head_dim` slice
        // of output; every byte is covered before any read.
        let output_slice = unsafe { dev.alloc::<T>(elem_count) }?;

        let kernel_sym = self.kernel_symbol_v1::<T>();
        let func = dev.get_or_load_custom_func(kernel_sym, "paged_attention", PTX)?;

        // Shared memory: q_smem[head_dim] + reduce_smem[NUM_WARPS] + logits[max_seq_len]
        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;

        // The K/V slice element type differs between Auto (Q dtype) and
        // quantized (raw u8 byte storage). Each branch holds its own
        // storage guards alive through the kernel launch; common scalar
        // args + scale-ptr args are appended uniformly.
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => {
                let (k_guard, _) = self.k_cache.storage_and_layout();
                let k_slice = expect_native_kv_slice::<T>(&k_guard, "K")?;
                let (v_guard, _) = self.v_cache.storage_and_layout();
                let v_slice = expect_native_kv_slice::<T>(&v_guard, "V")?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                push_scale_args(&mut builder, k_scale_slice, v_scale_slice);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("paged_attention launch: {e}")))?;

                drop(k_guard);
                drop(v_guard);
            }
            KVCacheDtype::Fp8E4m3 | KVCacheDtype::Fp8E5m2 | KVCacheDtype::Int8 => {
                let (k_guard, _) = self.k_cache.storage_and_layout();
                let k_slice = expect_u8_kv_slice(&k_guard, "K")?;
                let (v_guard, _) = self.v_cache.storage_and_layout();
                let v_slice = expect_u8_kv_slice(&v_guard, "V")?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                push_scale_args(&mut builder, k_scale_slice, v_scale_slice);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("paged_attention launch: {e}")))?;

                drop(k_guard);
                drop(v_guard);
            }
        }

        drop(bt_guard);
        drop(sl_guard);
        drop(k_scale_guard_layout);
        drop(v_scale_guard_layout);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v1 decode kernel.
///
/// Replaces per-sequence cache read + GQA repeat + matmul + softmax + matmul
/// with a single CUDA kernel launch. Supports configurable head dimensions
/// and block sizes for compatibility with different model architectures.
///
/// # Arguments
/// - `q`: Query tensor `[num_seqs, num_heads, head_dim]` bf16, contiguous
/// - `k_cache`: K cache `[num_blocks, block_size, num_kv_heads, head_dim]` bf16
/// - `v_cache`: V cache `[num_blocks, block_size, num_kv_heads, head_dim]` bf16
/// - `block_tables`: Physical block IDs `[num_seqs, max_blocks_per_seq]` u32
/// - `seq_lens`: KV length per sequence `[num_seqs]` u32
/// - `scale`: Attention scale factor (1/sqrt(head_dim))
/// - `num_heads`: Number of query heads
/// - `num_kv_heads`: Number of KV heads (for GQA)
/// - `max_blocks_per_seq`: Second dim of block_tables
/// - `max_seq_len`: Maximum sequence length in this batch (for shared memory sizing)
/// - `head_dim`: Head dimension (e.g. 64, 96, 128, 256)
/// - `block_size`: KV cache block size (e.g. 8, 16, 32)
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;

    let op = PagedAttnOp {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens.contiguous()?,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        // Auto KV cache by default; callers that need quantized KV must
        // use `paged_attention_cuda_with_kv_dtype`.
        kv_cache_dtype: KVCacheDtype::Auto,
        k_scale: None,
        v_scale: None,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// Explicit-KV-dtype variant of [`paged_attention_cuda`]. Use when the
/// KV cache is FP8/INT8 packed; pass per-tensor `k_scale` / `v_scale`
/// length-1 F32 device tensors (Auto mode ignores them and may pass
/// `None`).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda_with_kv_dtype(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;

    let op = PagedAttnOp {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens.contiguous()?,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        kv_cache_dtype,
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

// ============================================================================
// PagedAttention V2: Split-K for long sequences
// ============================================================================

/// Default V2 partition size used by `paged_attention_auto` for sequences
/// in the medium-context band.  See `select_partition_size` for the
/// adaptive policy that picks the actual value used at launch.
///
/// Kept as a module constant primarily for tests that want the same
/// default the production dispatcher uses.
pub const DEFAULT_V2_PARTITION_SIZE: usize = 128;

/// Threshold: use V2 when max_seq_len exceeds this value.
///
/// `paged_attention_bench` (commit aca4b69) measured V1 losing to V2 at
/// every sequence length on RTX 4060 Laptop, including seq_len = 64
/// (V1 = 56.7 µs vs V2 p=64 = 54.0 µs).  V1 also has a hard ceiling
/// around seq_len ≈ 12k where its `logits[max_seq_len]` shared-memory
/// allocation overflows the per-block 100 KB Ada limit, so very long
/// contexts have no choice but V2 anyway.  Set the threshold to 0 to
/// always pick V2.
const V2_SEQ_LEN_THRESHOLD: usize = 0;

/// Pick the V2 partition size for a given `max_seq_len`.
///
/// Empirically derived in `crates/core/benches/paged_attention_bench.rs`
/// (commit aca4b69) on Qwen3-4B-AWQ shape (num_q_heads=32, num_kv_heads=8,
/// head_dim=128, block_size=16, batch=1) on RTX 4060 Laptop, sm_89:
///
///   seq_len ≤ 4096  → p=64  wins or ties on every measured point.
///   seq_len > 4096  → p=256 wins (smaller p inflates partition count
///                     and reduce-kernel overhead; larger p loses
///                     occupancy below 8k tokens).
///
/// The same shape on different SM counts may want a different boundary
/// (an Ada full-fat 4090 has 128 SMs vs the Laptop's 24 — the optimum
/// p shifts accordingly).  Re-run the microbench on the target box and
/// adjust if you see a regression.
pub fn select_v2_partition_size(max_seq_len: usize) -> usize {
    if max_seq_len <= 4096 {
        64
    } else {
        256
    }
}

/// Hard cap on `head_dim` for V2's warp-level K-pass (the
/// `head_dim % 32 == 0 && head_dim <= 512` branch in
/// `paged_attention_v2_impl`).  Above this the kernel automatically
/// falls back to its legacy per-token block-reduce path, which is
/// correct but ~5× slower.  We document it as a public constant so
/// callers can warn or pick an alternative attention backend instead.
pub const PAGED_ATTN_V2_WARP_KPATH_MAX_HEAD_DIM: usize = 512;

/// Stage 1: compute partitioned attention outputs.
struct PagedAttnV2Op {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    /// Caller's stated upper bound on `seq_lens[i]`. The kernel itself only
    /// reads `max_num_partitions` (derived from this), but we keep the field
    /// to validate the contract `max(seq_lens) <= max_seq_len` at the public
    /// API entry — violating it would make the reduce kernel read
    /// uninitialised partitions.
    #[allow(dead_code)]
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    /// Tokens per V2 partition (Stage 1 grid Z dim = ⌈seq_len/partition_size⌉).
    /// Picked at launch by the dispatcher; the kernel reads it as a runtime arg.
    partition_size: usize,
    max_num_partitions: usize,
    /// See [`PagedAttnOp::kv_cache_dtype`].
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
}

impl CustomOp1 for PagedAttnV2Op {
    fn name(&self) -> &'static str {
        "paged_attention_v2"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v2 requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention_v2: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v2::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v2::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v2 expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnV2Op {
    /// Pick the V2 Stage-1 kernel symbol for the current Q dtype × KV
    /// cache dtype combination.
    fn kernel_symbol_v2<T: PagedAttnDtype>(&self) -> &'static str {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => T::KERNEL_V2_AUTO,
            KVCacheDtype::Fp8E4m3 => T::KERNEL_V2_FP8_E4M3,
            KVCacheDtype::Fp8E5m2 => T::KERNEL_V2_FP8_E5M2,
            KVCacheDtype::Int8 => T::KERNEL_V2_INT8,
        }
    }

    fn run_v2<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        validate_scales_for_kv_dtype(self.kv_cache_dtype, &self.k_scale, &self.v_scale)?;

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v2: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        // Per-tensor F32 scales — see `run_v1` for the same pattern.
        let k_scale_guard_layout = self.k_scale.as_ref().map(|t| t.storage_and_layout());
        let v_scale_guard_layout = self.v_scale.as_ref().map(|t| t.storage_and_layout());
        let k_scale_slice = k_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "k_scale"))
            .transpose()?;
        let v_scale_slice = v_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "v_scale"))
            .transpose()?;

        let head_dim = self.head_dim;
        let max_num_partitions = self.max_num_partitions;

        // Allocate intermediate buffers
        let tmp_out_size = num_seqs * self.num_heads * max_num_partitions * head_dim;
        let meta_size = num_seqs * self.num_heads * max_num_partitions;
        let tmp_out_slice = dev.alloc_zeros::<f32>(tmp_out_size)?;
        let exp_sums_slice = dev.alloc_zeros::<f32>(meta_size)?;
        let max_logits_slice = dev.alloc_zeros::<f32>(meta_size)?;

        // Final output
        let out_size = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(out_size)?;

        // Stage 1: partitioned attention kernel
        let kernel_sym = self.kernel_symbol_v2::<T>();
        let v2_func = dev.get_or_load_custom_func(kernel_sym, "paged_attention", PTX)?;

        // Shared memory: q[head_dim] + reduce[NUM_WARPS] + logits[partition_size]
        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.partition_size) * std::mem::size_of::<f32>()) as u32;

        let v2_cfg = LaunchConfig {
            grid_dim: (
                self.num_heads as u32,
                num_seqs as u32,
                max_num_partitions as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;
        let partition_size_i32 = self.partition_size as i32;
        let max_partitions_i32 = max_num_partitions as i32;

        match self.kv_cache_dtype {
            KVCacheDtype::Auto => {
                let (k_guard, _) = self.k_cache.storage_and_layout();
                let k_slice = expect_native_kv_slice::<T>(&k_guard, "K")?;
                let (v_guard, _) = self.v_cache.storage_and_layout();
                let v_slice = expect_native_kv_slice::<T>(&v_guard, "V")?;

                let mut builder = v2_func.builder();
                builder.arg(&tmp_out_slice);
                builder.arg(&exp_sums_slice);
                builder.arg(&max_logits_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(&partition_size_i32);
                builder.arg(&max_partitions_i32);
                push_scale_args(&mut builder, k_scale_slice, v_scale_slice);

                unsafe { builder.launch(v2_cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("paged_attention_v2 launch: {e}"))
                })?;
                drop(k_guard);
                drop(v_guard);
            }
            KVCacheDtype::Fp8E4m3 | KVCacheDtype::Fp8E5m2 | KVCacheDtype::Int8 => {
                let (k_guard, _) = self.k_cache.storage_and_layout();
                let k_slice = expect_u8_kv_slice(&k_guard, "K")?;
                let (v_guard, _) = self.v_cache.storage_and_layout();
                let v_slice = expect_u8_kv_slice(&v_guard, "V")?;

                let mut builder = v2_func.builder();
                builder.arg(&tmp_out_slice);
                builder.arg(&exp_sums_slice);
                builder.arg(&max_logits_slice);
                builder.arg(q_slice);
                builder.arg(k_slice);
                builder.arg(v_slice);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(&partition_size_i32);
                builder.arg(&max_partitions_i32);
                push_scale_args(&mut builder, k_scale_slice, v_scale_slice);

                unsafe { builder.launch(v2_cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("paged_attention_v2 launch: {e}"))
                })?;
                drop(k_guard);
                drop(v_guard);
            }
        }

        // Stage 2: reduce kernel
        let reduce_func =
            dev.get_or_load_custom_func(T::KERNEL_V2_REDUCE, "paged_attention", PTX)?;

        // Reduce shared memory: max_logits[max_num_partitions] + exp_sums[max_num_partitions] + warp_reduce[NUM_WARPS]
        let reduce_shared_bytes =
            ((2 * max_num_partitions + NUM_WARPS) * std::mem::size_of::<f32>()) as u32;

        let reduce_cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: reduce_shared_bytes,
        };

        {
            let mut builder = reduce_func.builder();
            builder.arg(&output_slice);
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(sl_slice);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&partition_size_i32);
            builder.arg(&max_partitions_i32);

            // SAFETY: reduce kernel launch with validated partition outputs
            unsafe { builder.launch(reduce_cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("paged_attention_v2 reduce: {e}")))?;
        }

        // K/V guards were dropped inside the per-dtype match above; bt
        // and sl guards stay alive across the reduce launch since the
        // reduce kernel reads `sl_slice` again.
        drop(bt_guard);
        drop(sl_guard);
        drop(k_scale_guard_layout);
        drop(v_scale_guard_layout);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v2 decode kernel for long sequences.
///
/// Uses split-K partitioning: the sequence is divided into partitions of
/// `DEFAULT_V2_PARTITION_SIZE` tokens, each processed by an independent
/// thread block. A reduce kernel merges partitions using numerically
/// stable log-sum-exp.  Use [`paged_attention_v2_cuda_with_partition_size`]
/// to override the partition size (e.g. from a benchmark-driven adaptive
/// selector).
///
/// # Arguments
/// Same as [`paged_attention_cuda`].
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    paged_attention_v2_cuda_with_partition_size(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        DEFAULT_V2_PARTITION_SIZE,
    )
}

/// Same as [`paged_attention_v2_cuda`] but lets the caller pick the V2
/// partition size.  Used by the adaptive selector and by the parity tests
/// (which sweep the partition size to verify behaviour at boundaries).
///
/// `partition_size` must be > 0.  Anything <= 0 is rejected at the public
/// boundary.  The kernel itself is correct for any positive value.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda_with_partition_size(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
) -> Result<Tensor> {
    if partition_size == 0 {
        candle_core::bail!("paged_attention_v2: partition_size must be > 0");
    }
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;
    let max_num_partitions = max_seq_len.div_ceil(partition_size);

    let seq_lens_c = seq_lens.contiguous()?;
    debug_assert_seq_lens_within_bound(&seq_lens_c, max_seq_len)?;

    let op = PagedAttnV2Op {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens_c,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        partition_size,
        max_num_partitions,
        // Auto KV cache by default; see
        // `paged_attention_v2_cuda_with_kv_dtype` for the quantized
        // entry point.
        kv_cache_dtype: KVCacheDtype::Auto,
        k_scale: None,
        v_scale: None,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// Explicit-KV-dtype variant of [`paged_attention_v2_cuda_with_partition_size`].
/// `kv_cache_dtype = Auto` reproduces the legacy behaviour; quantized
/// variants require length-1 F32 `k_scale` / `v_scale` device tensors
/// matching the write-side calibration.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda_with_kv_dtype(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Tensor> {
    if partition_size == 0 {
        candle_core::bail!("paged_attention_v2: partition_size must be > 0");
    }
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;
    let max_num_partitions = max_seq_len.div_ceil(partition_size);

    let seq_lens_c = seq_lens.contiguous()?;
    debug_assert_seq_lens_within_bound(&seq_lens_c, max_seq_len)?;

    let op = PagedAttnV2Op {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens_c,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        partition_size,
        max_num_partitions,
        kv_cache_dtype,
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// In-place sibling of [`PagedAttnV2Op`] used by the pooled fast path.
///
/// All four working buffers (final output + tmp_out + exp_sums +
/// max_logits) are reserved from the global [`OutputPool`] **before**
/// `inplace_op2` is called, so device addresses stay stable across
/// forwards — the precondition for CUDA Graph capture replay.
///
/// Worst-case sizing: callers pass the **maximum possible**
/// `max_num_partitions` (derived from `--max-model-len / partition_size`)
/// rather than the per-call value. The Stage-1 kernel still receives
/// the actual seq_lens array and only does meaningful work on
/// partitions covering each sequence's real KV length; partitions
/// beyond actual length write zero/skip-identity (alloc_zeros init in
/// the original path masked this — pool reuse needs the kernel itself
/// to be correct for un-touched partitions, which it is).
struct PagedAttnV2InplaceOp<'a> {
    k_cache: &'a Tensor,
    v_cache: &'a Tensor,
    block_tables: &'a Tensor,
    seq_lens: &'a Tensor,
    tmp_out: &'a Tensor,
    exp_sums: &'a Tensor,
    max_logits: &'a Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
    /// Worst-case partition count from `worst_case_max_seq_len /
    /// partition_size`. Sizes the pool scratch buffers (`tmp_out`,
    /// `exp_sums`, `max_logits`) and is passed to the kernel as the
    /// per-(seq, head) partition **stride** — required to match the
    /// allocated buffer layout regardless of actual seq_len.
    max_num_partitions: usize,
    /// Active partition count derived from `actual_max_seq_len`. Drives
    /// the Stage 1 grid Z dimension so we only launch as many partition
    /// blocks as can have work. Without this, short decode positions
    /// (e.g. seq_len=32 with worst_case=131_072 and partition_size=256)
    /// pay 511 empty-block launches per (head, seq) — the documented
    /// 5-10× per-call regression at c=8 with prompt=256.
    active_max_partitions: usize,
}

impl<'a> InplaceOp2 for PagedAttnV2InplaceOp<'a> {
    fn name(&self) -> &'static str {
        "paged_attention_v2_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("paged_attention_v2_inplace requires CUDA")
    }

    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        q_storage: &CudaStorage,
        q_layout: &Layout,
    ) -> Result<()> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention_v2_inplace: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => {
                self.run_v2_inplace::<half::bf16>(out_storage, q_storage, num_seqs)
            }
            CudaStorageSlice::F16(_) => {
                self.run_v2_inplace::<half::f16>(out_storage, q_storage, num_seqs)
            }
            _ => candle_core::bail!("paged_attention_v2_inplace expects bf16 or f16 Q tensor"),
        }
    }
}

impl<'a> PagedAttnV2InplaceOp<'a> {
    fn run_v2_inplace<T: PagedAttnDtype>(
        &self,
        out_storage: &mut CudaStorage,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v2_inplace: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;
        let out_slice = match &mut out_storage.slice {
            CudaStorageSlice::BF16(s) => {
                if T::NAME != "bf16" {
                    candle_core::bail!(
                        "paged_attention_v2_inplace: out is BF16 but Q is {}",
                        T::NAME
                    );
                }
                // SAFETY: T == bf16 by NAME match.
                unsafe {
                    &mut *(s as *mut candle_core::cuda::cudarc::driver::CudaSlice<half::bf16>
                        as *mut candle_core::cuda::cudarc::driver::CudaSlice<T>)
                }
            }
            CudaStorageSlice::F16(s) => {
                if T::NAME != "f16" {
                    candle_core::bail!(
                        "paged_attention_v2_inplace: out is F16 but Q is {}",
                        T::NAME
                    );
                }
                // SAFETY: T == f16 by NAME match.
                unsafe {
                    &mut *(s as *mut candle_core::cuda::cudarc::driver::CudaSlice<half::f16>
                        as *mut candle_core::cuda::cudarc::driver::CudaSlice<T>)
                }
            }
            _ => candle_core::bail!("paged_attention_v2_inplace: out must be BF16 or F16"),
        };

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let k_slice = match &*k_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2_inplace: K cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention_v2_inplace: K cache must be on CUDA"),
        };

        let (v_guard, _) = self.v_cache.storage_and_layout();
        let v_slice = match &*v_guard {
            Storage::Cuda(cs) => T::slice_from(&cs.slice).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "paged_attention_v2_inplace: V cache dtype must match Q ({})",
                    T::NAME
                ))
            })?,
            _ => candle_core::bail!("paged_attention_v2_inplace: V cache must be on CUDA"),
        };

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        let (tmp_guard, _) = self.tmp_out.storage_and_layout();
        let tmp_out_slice = match &*tmp_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("tmp_out must be F32"),
            },
            _ => candle_core::bail!("tmp_out must be on CUDA"),
        };
        let (es_guard, _) = self.exp_sums.storage_and_layout();
        let exp_sums_slice = match &*es_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("exp_sums must be F32"),
            },
            _ => candle_core::bail!("exp_sums must be on CUDA"),
        };
        let (ml_guard, _) = self.max_logits.storage_and_layout();
        let max_logits_slice = match &*ml_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("max_logits must be F32"),
            },
            _ => candle_core::bail!("max_logits must be on CUDA"),
        };

        let head_dim = self.head_dim;
        let max_num_partitions = self.max_num_partitions;
        // Grid Z covers only partitions that can actually have work.
        // The kernel early-returns at `partition_start_token >= seq_len`
        // (paged_attention.cu:571) — but block launch itself costs ~µs
        // per block, and at worst_case=131_072 with partition_size=256
        // we'd launch 511 such empty blocks per (head, seq) for a
        // 256-token decode. The kernel reads `partition_idx = blockIdx.z`
        // for its own math and `max_num_partitions` for buffer stride —
        // these are independent, so shrinking grid Z is correctness-safe
        // as long as we keep the stride argument worst-case.
        let active_max_partitions = self.active_max_partitions;

        // Stage 1: partitioned attention kernel. The pooled-V2 path
        // currently supports `Auto` KV cache only — quantized cache flows
        // through `paged_attention_v2_cuda_with_kv_dtype`.
        let v2_func = dev.get_or_load_custom_func(T::KERNEL_V2_AUTO, "paged_attention", PTX)?;
        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.partition_size) * std::mem::size_of::<f32>()) as u32;
        let v2_cfg = LaunchConfig {
            grid_dim: (
                self.num_heads as u32,
                num_seqs as u32,
                active_max_partitions as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;
        let partition_size_i32 = self.partition_size as i32;
        let max_partitions_i32 = max_num_partitions as i32;

        {
            let mut builder = v2_func.builder();
            builder.arg(tmp_out_slice);
            builder.arg(exp_sums_slice);
            builder.arg(max_logits_slice);
            builder.arg(q_slice);
            builder.arg(k_slice);
            builder.arg(v_slice);
            builder.arg(bt_slice);
            builder.arg(sl_slice);
            builder.arg(&self.scale);
            builder.arg(&num_heads_i32);
            builder.arg(&num_kv_heads_i32);
            builder.arg(&max_blocks_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&block_size_i32);
            builder.arg(&partition_size_i32);
            builder.arg(&max_partitions_i32);
            // Pooled-V2 currently routes Auto KV cache only — push two
            // device nullptrs for `(k_scale_ptr, v_scale_ptr)` so the
            // kernel side falls back to scale=1.0. The kernel signature
            // requires these slots whether or not the cache is quantized.
            push_scale_args(&mut builder, None, None);
            // SAFETY: validated kernel params; pooled scratch buffers are
            // exclusively held by this op for the duration of the launch.
            unsafe { builder.launch(v2_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("paged_attention_v2_inplace stage1: {e}"))
            })?;
        }

        // Stage 2: reduce kernel
        let reduce_func =
            dev.get_or_load_custom_func(T::KERNEL_V2_REDUCE, "paged_attention", PTX)?;
        let reduce_shared_bytes =
            ((2 * max_num_partitions + NUM_WARPS) * std::mem::size_of::<f32>()) as u32;
        let reduce_cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: reduce_shared_bytes,
        };
        {
            let mut builder = reduce_func.builder();
            builder.arg(out_slice);
            builder.arg(tmp_out_slice);
            builder.arg(exp_sums_slice);
            builder.arg(max_logits_slice);
            builder.arg(sl_slice);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&partition_size_i32);
            builder.arg(&max_partitions_i32);
            unsafe { builder.launch(reduce_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("paged_attention_v2_inplace reduce: {e}"))
            })?;
        }

        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);
        drop(tmp_guard);
        drop(es_guard);
        drop(ml_guard);
        Ok(())
    }
}

/// Zero every element of a pool-backed tensor in-place via a single
/// `cuMemsetD8Async` (capture-safe; recorded as a memset node when
/// invoked inside `cuStreamBeginCapture`). Used by Phase CR.1 to wipe
/// `paged_attention_v2_cuda_pooled`'s scratch buffers each forward so
/// captured-graph replay doesn't read stale partition slots written by
/// a prior decode step.
#[cfg(feature = "cuda-kernels")]
pub fn zero_pool_tensor_dtod_count() -> usize {
    use std::sync::atomic::Ordering;
    static C: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    C.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda-kernels")]
fn zero_pool_tensor_dtod(t: &Tensor) -> Result<()> {
    use std::sync::atomic::Ordering;
    static C: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    C.fetch_add(1, Ordering::Relaxed);
    if std::env::var("CR_PRINT_ZPT").is_ok() {
        eprintln!("zero_pool_tensor_dtod call #{}", C.load(Ordering::Relaxed));
    }
    crate::engine::cr_trace::mark_op("zero_pool_tensor_dtod");
    use candle_core::cuda::CudaStorageSlice;
    use candle_core::{CpuStorage, InplaceOp1, Layout};

    struct ZeroFillOp;
    impl InplaceOp1 for ZeroFillOp {
        fn name(&self) -> &'static str {
            "zero_pool_tensor_dtod"
        }
        fn cpu_fwd(&self, _s: &mut CpuStorage, _l: &Layout) -> Result<()> {
            candle_core::bail!("zero_pool_tensor_dtod is CUDA-only")
        }
        fn cuda_fwd(&self, storage: &mut candle_core::CudaStorage, layout: &Layout) -> Result<()> {
            use candle_core::cuda::cudarc::driver::sys::{cuMemsetD8Async, CUresult};
            use candle_core::cuda::cudarc::driver::DevicePtr;

            let stream = storage.device.cuda_stream();
            macro_rules! ms {
                ($slice:expr, $elem_size:expr) => {{
                    let (raw, _guard) = $slice.device_ptr(&stream);
                    let bytes = layout.shape().elem_count() * $elem_size;
                    let ptr = raw + layout.start_offset() as u64 * $elem_size as u64;
                    let res = unsafe { cuMemsetD8Async(ptr, 0, bytes, stream.cu_stream()) };
                    if res != CUresult::CUDA_SUCCESS {
                        candle_core::bail!("cuMemsetD8Async failed: {:?}", res);
                    }
                }};
            }
            match &storage.slice {
                CudaStorageSlice::U8(s) => ms!(s, 1usize),
                CudaStorageSlice::U32(s) => ms!(s, 4usize),
                CudaStorageSlice::I64(s) => ms!(s, 8usize),
                CudaStorageSlice::BF16(s) => ms!(s, 2usize),
                CudaStorageSlice::F16(s) => ms!(s, 2usize),
                CudaStorageSlice::F32(s) => ms!(s, 4usize),
                CudaStorageSlice::F64(s) => ms!(s, 8usize),
                _ => candle_core::bail!("zero_pool_tensor_dtod: unsupported dtype"),
            }
            Ok(())
        }
    }

    t.inplace_op1(&ZeroFillOp)
}

#[cfg(not(feature = "cuda-kernels"))]
fn zero_pool_tensor_dtod(_t: &Tensor) -> Result<()> {
    Ok(())
}

/// Pool-backed PagedAttention V2 for the decode hot path. Reserves the
/// final output and the three intermediate buffers (tmp_out, exp_sums,
/// max_logits) from [`OutputPool`] at **worst-case** size derived from
/// `worst_case_max_seq_len` so device addresses stay stable across
/// forwards.
///
/// Decode-only gate (num_seqs ≤ 64) — prefill or any batch larger
/// than the captured budget falls through to the non-pooled
/// [`paged_attention_v2_cuda_with_partition_size`].
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda_pooled(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    worst_case_max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
) -> Result<Tensor> {
    paged_attention_v2_cuda_pooled_with_kv_dtype(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        worst_case_max_seq_len,
        head_dim,
        block_size,
        partition_size,
        KVCacheDtype::Auto,
        None,
        None,
    )
}

/// KV-dtype-aware variant of [`paged_attention_v2_cuda_pooled`].
///
/// The pooled fast-path keeps the legacy `PagedAttnV2InplaceOp` which
/// supports `Auto` KV cache only — quantized cache + worst-case-sized
/// scratch + captured CUDA graph is a follow-up integration step
/// (kernel side already handles FP8/INT8 dequant via `paged_attention_v2_*`
/// entry points; pooled in-place op needs to thread scale ptrs through
/// to `PagedAttnV2InplaceOp::cuda_fwd` too).
///
/// For oversize batches (num_seqs > POOL_MAX_NUM_SEQS) we fall through
/// to the non-pooled `paged_attention_v2_cuda_with_kv_dtype`, which DOES
/// handle quantized cache; that path is the production target for
/// prefill / large-batch decode of FP8 KV models.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda_pooled_with_kv_dtype(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    worst_case_max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Tensor> {
    use crate::engine::output_pool::OutputPool;
    crate::engine::cr_trace::mark_op("paged_attention_v2_cuda_pooled");

    if kv_cache_dtype != KVCacheDtype::Auto {
        // Quantized KV cache + pooled-V2 captured-graph path is not yet
        // wired. Until `PagedAttnV2InplaceOp` carries scale pointers
        // through to its launch builder, route the call through the
        // non-pooled wrapper instead — the only cost is per-call scratch
        // allocation (no pool reuse for FP8 KV models on the captured
        // path). EXL3 + FP8 typically runs with `--enforce-eager` so
        // this fallback is the production path anyway.
        return paged_attention_v2_cuda_with_kv_dtype(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            partition_size,
            kv_cache_dtype,
            k_scale,
            v_scale,
        );
    }
    let _ = (k_scale, v_scale); // Auto path ignores scales by definition.

    /// Decode-shape budget; matches the other pool wrappers.
    const POOL_MAX_NUM_SEQS: usize = 64;

    if partition_size == 0 {
        candle_core::bail!("paged_attention_v2_pooled: partition_size must be > 0");
    }

    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;
    if num_seqs > POOL_MAX_NUM_SEQS {
        return paged_attention_v2_cuda_with_partition_size(
            &q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            partition_size,
        );
    }

    let max_num_partitions = worst_case_max_seq_len.div_ceil(partition_size).max(1);
    // Active partition count = ceil(actual_max_seq_len / partition_size).
    // Bounded above by max_num_partitions (the caller-stated worst case
    // is always ≥ actual). Drives grid Z; see PagedAttnV2InplaceOp doc.
    let active_max_partitions = max_seq_len.div_ceil(partition_size).max(1);

    let seq_lens_c = seq_lens.contiguous()?;
    debug_assert_seq_lens_within_bound(&seq_lens_c, worst_case_max_seq_len)?;
    let block_tables_c = block_tables.contiguous()?;

    // Reserve output + scratch buffers from the pool. Worst-case sizes
    // → fixed addresses across forwards regardless of actual seq_len.
    let dtype = q.dtype();
    let device = q.device();
    // PERF NOTE: see exl3_gemm/marlin_gemm — Result<PooledTensor> ABI
    // regression. paged_attention V2 reserve sites kept on legacy
    // `reserve()` for the same reason.
    let output = OutputPool::global().reserve(&[num_seqs, num_heads, head_dim], dtype, device)?;
    let tmp_out = OutputPool::global().reserve(
        &[num_seqs, num_heads, max_num_partitions, head_dim],
        candle_core::DType::F32,
        device,
    )?;
    let exp_sums = OutputPool::global().reserve(
        &[num_seqs, num_heads, max_num_partitions],
        candle_core::DType::F32,
        device,
    )?;
    let max_logits = OutputPool::global().reserve(
        &[num_seqs, num_heads, max_num_partitions],
        candle_core::DType::F32,
        device,
    )?;

    // Phase CR.1: zero the scratch buffers via captured memset nodes so
    // captured graph replays start every paged-attn invocation from a
    // clean slate. Without this, pool slots retain values written by
    // prior forwards (or by capture warmup's dummy seq_len=1), and the
    // V2 reduce path could read stale max_logits/exp_sums from invalid
    // partitions. The memset is a captured node (cuMemsetD8Async),
    // capture-safe.
    zero_pool_tensor_dtod(&tmp_out)?;
    zero_pool_tensor_dtod(&exp_sums)?;
    zero_pool_tensor_dtod(&max_logits)?;

    let op = PagedAttnV2InplaceOp {
        k_cache,
        v_cache,
        block_tables: &block_tables_c,
        seq_lens: &seq_lens_c,
        tmp_out: &tmp_out,
        exp_sums: &exp_sums,
        max_logits: &max_logits,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        head_dim,
        block_size,
        partition_size,
        max_num_partitions,
        active_max_partitions,
    };

    output.inplace_op2(&q, &op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// Auto-selecting paged attention: uses V1 for short sequences, V2 for long.
///
/// The threshold is 512 tokens (one partition). Below this, V1 has lower
/// overhead. Above, V2 benefits from split-K parallelism and bounded shared
/// memory.
///
/// This is the recommended entry point for model code.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
) -> Result<Tensor> {
    paged_attention_auto_with_kv_dtype(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        KVCacheDtype::Auto,
        None,
        None,
    )
}

/// KV-dtype-aware variant of [`paged_attention_auto`]. Routes to V1 or
/// V2 based on `max_seq_len` and selects the matching FP8/INT8 kernel
/// when `kv_cache_dtype != Auto`. Per-tensor `k_scale` / `v_scale`
/// (F32 length-1 device tensors) must be supplied for quantized cache;
/// pass `None` for `Auto`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto_with_kv_dtype(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Tensor> {
    if max_seq_len > V2_SEQ_LEN_THRESHOLD {
        paged_attention_v2_cuda_with_kv_dtype(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            select_v2_partition_size(max_seq_len),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    } else {
        paged_attention_cuda_with_kv_dtype(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    }
}

// ============================================================================
// PagedAttention with ALiBi
// ============================================================================

/// Paged attention op with ALiBi positional bias support.
///
/// The CUDA kernel adds `alibi_slopes[head] * (key_pos - query_pos)` to each
/// attention logit, implementing ALiBi (Attention with Linear Biases) for
/// models like Bloom and MPT.
struct PagedAttnAlibiOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    alibi_slopes: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    // Phase 10 — ALiBi + FP8/INT8 KV cache. `Auto` selects the legacy
    // native-dtype path (same kernel symbol layout as before this
    // phase); quantised variants route through the new
    // `KERNEL_V1_*_ALIBI` entry points which take per-tensor scale
    // pointers alongside `alibi_slopes`.
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
}

impl CustomOp1 for PagedAttnAlibiOp {
    fn name(&self) -> &'static str {
        "paged_attention_v1_alibi"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v1_alibi requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v1_alibi::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v1_alibi::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v1_alibi expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnAlibiOp {
    /// Pick the kernel symbol for the current Q dtype × KV cache dtype
    /// combination. `Auto` keeps the legacy native-dtype symbol;
    /// quantised variants route to the Phase 10 entry points.
    fn kernel_symbol_v1<T: PagedAttnDtype>(&self) -> &'static str {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => T::KERNEL_V1_AUTO_ALIBI,
            KVCacheDtype::Fp8E4m3 => T::KERNEL_V1_FP8_E4M3_ALIBI,
            KVCacheDtype::Fp8E5m2 => T::KERNEL_V1_FP8_E5M2_ALIBI,
            KVCacheDtype::Int8 => T::KERNEL_V1_INT8_ALIBI,
        }
    }

    fn run_v1_alibi<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v1_alibi: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        // Validate FP8/INT8 scales presence when in a quantised mode.
        validate_scales_for_kv_dtype(self.kv_cache_dtype, &self.k_scale, &self.v_scale)?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let (v_guard, _) = self.v_cache.storage_and_layout();

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        let (alibi_guard, _) = self.alibi_slopes.storage_and_layout();
        let alibi_slice = match &*alibi_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("alibi_slopes must be F32"),
            },
            _ => candle_core::bail!("alibi_slopes must be on CUDA"),
        };

        // Per-tensor F32 scale slices for quantised paths (None on Auto).
        let k_scale_guard_layout = self.k_scale.as_ref().map(|t| t.storage_and_layout());
        let v_scale_guard_layout = self.v_scale.as_ref().map(|t| t.storage_and_layout());
        let k_scale_slice = k_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "k_scale"))
            .transpose()?;
        let v_scale_slice = v_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "v_scale"))
            .transpose()?;

        let head_dim = self.head_dim;
        let elem_count = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(elem_count)?;

        let kernel_sym = self.kernel_symbol_v1::<T>();
        let func = dev.get_or_load_custom_func(kernel_sym, "paged_attention", PTX)?;

        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.max_seq_len) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;

        let mut builder = func.builder();
        builder.arg(&output_slice);
        builder.arg(q_slice);
        // K/V slice extraction is dtype-conditional: Auto uses `slice<T>`,
        // FP8/INT8 use raw U8 bytes (the kernel template's `Cache_t`
        // marker drives the inline dequant inside `load_kv_to_f32`).
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => {
                let k_native = expect_native_kv_slice::<T>(&k_guard, "K")?;
                let v_native = expect_native_kv_slice::<T>(&v_guard, "V")?;
                builder.arg(k_native);
                builder.arg(v_native);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(alibi_slice);
                push_scale_args(&mut builder, None, None);
            }
            KVCacheDtype::Fp8E4m3 | KVCacheDtype::Fp8E5m2 | KVCacheDtype::Int8 => {
                let k_u8 = expect_u8_kv_slice(&k_guard, "K")?;
                let v_u8 = expect_u8_kv_slice(&v_guard, "V")?;
                builder.arg(k_u8);
                builder.arg(v_u8);
                builder.arg(bt_slice);
                builder.arg(sl_slice);
                builder.arg(&self.scale);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);
                builder.arg(&max_blocks_i32);
                builder.arg(&head_dim_i32);
                builder.arg(&block_size_i32);
                builder.arg(alibi_slice);
                push_scale_args(&mut builder, k_scale_slice, v_scale_slice);
            }
        }

        // SAFETY: kernel launch with validated parameters and contiguous buffers
        unsafe { builder.launch(cfg) }
            .map_err(|e| candle_core::Error::Msg(format!("paged_attention alibi launch: {e}")))?;

        drop(alibi_guard);
        drop(k_scale_guard_layout);
        drop(v_scale_guard_layout);
        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Fused PagedAttention v1 decode kernel with ALiBi positional bias.
///
/// Identical to [`paged_attention_cuda`] but adds ALiBi (Attention with Linear
/// Biases) to the attention logits inside the kernel. This is used by models
/// like Bloom and MPT that use ALiBi instead of RoPE for positional encoding.
///
/// The kernel applies: `logit[h, pos] += alibi_slopes[h] * (pos - (seq_len - 1))`
/// which gives non-positive bias to past tokens (closer = less negative).
///
/// # Arguments
/// Same as [`paged_attention_cuda`], plus:
/// - `alibi_slopes`: Per-head ALiBi slopes `[num_heads]` f32, on CUDA
///
/// # Returns
/// Output tensor `[num_seqs, num_heads * head_dim]` bf16
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_cuda_alibi(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    alibi_slopes: &Tensor,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let num_seqs = q.dim(0)?;

    let alibi_slopes = alibi_slopes
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?;

    let op = PagedAttnAlibiOp {
        k_cache: k_cache.clone(),
        v_cache: v_cache.clone(),
        block_tables: block_tables.contiguous()?,
        seq_lens: seq_lens.contiguous()?,
        alibi_slopes,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        head_dim,
        block_size,
        // Legacy public API: pin Auto KV cache so the dispatch routes
        // through the unchanged `KERNEL_V1_AUTO_ALIBI` symbol. Phase
        // 10 quantised paths must use the new
        // `paged_attention_cuda_alibi_with_kv_dtype` wrapper.
        kv_cache_dtype: KVCacheDtype::Auto,
        k_scale: None,
        v_scale: None,
    };

    let output = q.apply_op1_no_bwd(&op)?;
    output.reshape((num_seqs, num_heads * head_dim))
}

/// V2 paged attention with ALiBi support.
struct PagedAttnV2AlibiOp {
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    seq_lens: Tensor,
    alibi_slopes: Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    /// Same contract as in `PagedAttnV2Op::max_seq_len`.
    #[allow(dead_code)]
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    /// Same role as in [`PagedAttnV2Op::partition_size`].
    partition_size: usize,
    max_num_partitions: usize,
    // Phase 10 — see `PagedAttnAlibiOp::kv_cache_dtype`. `Auto` keeps
    // the legacy native-dtype path; FP8/INT8 routes through the new
    // `KERNEL_V2_*_ALIBI` symbols.
    kv_cache_dtype: KVCacheDtype,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
}

impl CustomOp1 for PagedAttnV2AlibiOp {
    fn name(&self) -> &'static str {
        "paged_attention_v2_alibi"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("paged_attention_v2_alibi requires CUDA")
    }

    fn cuda_fwd(&self, q_storage: &CudaStorage, q_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        if q_layout.start_offset() != 0 {
            candle_core::bail!("paged_attention_v2_alibi: Q must be contiguous from offset 0");
        }
        let num_seqs = q_layout.dims()[0];
        match &q_storage.slice {
            CudaStorageSlice::BF16(_) => self.run_v2_alibi::<half::bf16>(q_storage, num_seqs),
            CudaStorageSlice::F16(_) => self.run_v2_alibi::<half::f16>(q_storage, num_seqs),
            _ => candle_core::bail!("paged_attention_v2_alibi expects bf16 or f16 Q tensor"),
        }
    }
}

impl PagedAttnV2AlibiOp {
    /// See `PagedAttnAlibiOp::kernel_symbol_v1`.
    fn kernel_symbol_v2<T: PagedAttnDtype>(&self) -> &'static str {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => T::KERNEL_V2_AUTO_ALIBI,
            KVCacheDtype::Fp8E4m3 => T::KERNEL_V2_FP8_E4M3_ALIBI,
            KVCacheDtype::Fp8E5m2 => T::KERNEL_V2_FP8_E5M2_ALIBI,
            KVCacheDtype::Int8 => T::KERNEL_V2_INT8_ALIBI,
        }
    }

    fn run_v2_alibi<T: PagedAttnDtype>(
        &self,
        q_storage: &CudaStorage,
        num_seqs: usize,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &q_storage.device;
        let q_slice = T::slice_from(&q_storage.slice).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "paged_attention_v2_alibi: Q dtype mismatch ({})",
                T::NAME
            ))
        })?;

        validate_scales_for_kv_dtype(self.kv_cache_dtype, &self.k_scale, &self.v_scale)?;

        let (k_guard, _) = self.k_cache.storage_and_layout();
        let (v_guard, _) = self.v_cache.storage_and_layout();

        let k_scale_guard_layout = self.k_scale.as_ref().map(|t| t.storage_and_layout());
        let v_scale_guard_layout = self.v_scale.as_ref().map(|t| t.storage_and_layout());
        let k_scale_slice = k_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "k_scale"))
            .transpose()?;
        let v_scale_slice = v_scale_guard_layout
            .as_ref()
            .map(|(g, _)| extract_f32_scalar(g, "v_scale"))
            .transpose()?;

        let (bt_guard, _) = self.block_tables.storage_and_layout();
        let bt_slice = match &*bt_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("block_tables must be U32"),
            },
            _ => candle_core::bail!("block_tables must be on CUDA"),
        };

        let (sl_guard, _) = self.seq_lens.storage_and_layout();
        let sl_slice = match &*sl_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::U32(s) => s,
                _ => candle_core::bail!("seq_lens must be U32"),
            },
            _ => candle_core::bail!("seq_lens must be on CUDA"),
        };

        let (alibi_guard, _) = self.alibi_slopes.storage_and_layout();
        let alibi_slice = match &*alibi_guard {
            Storage::Cuda(cs) => match &cs.slice {
                CudaStorageSlice::F32(s) => s,
                _ => candle_core::bail!("alibi_slopes must be F32"),
            },
            _ => candle_core::bail!("alibi_slopes must be on CUDA"),
        };

        let head_dim = self.head_dim;
        let max_num_partitions = self.max_num_partitions;

        let tmp_out_size = num_seqs * self.num_heads * max_num_partitions * head_dim;
        let meta_size = num_seqs * self.num_heads * max_num_partitions;
        let tmp_out_slice = dev.alloc_zeros::<f32>(tmp_out_size)?;
        let exp_sums_slice = dev.alloc_zeros::<f32>(meta_size)?;
        let max_logits_slice = dev.alloc_zeros::<f32>(meta_size)?;

        let out_size = num_seqs * self.num_heads * head_dim;
        let output_slice = dev.alloc_zeros::<T>(out_size)?;

        // Stage 1: partitioned attention with ALiBi. Phase 10 — symbol
        // depends on kv_cache_dtype.
        let v2_sym = self.kernel_symbol_v2::<T>();
        let v2_func = dev.get_or_load_custom_func(v2_sym, "paged_attention", PTX)?;

        let shared_mem_bytes =
            ((head_dim + NUM_WARPS + self.partition_size) * std::mem::size_of::<f32>()) as u32;

        let v2_cfg = LaunchConfig {
            grid_dim: (
                self.num_heads as u32,
                num_seqs as u32,
                max_num_partitions as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };

        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = self.num_kv_heads as i32;
        let max_blocks_i32 = self.max_blocks_per_seq as i32;
        let head_dim_i32 = head_dim as i32;
        let block_size_i32 = self.block_size as i32;
        let partition_size_i32 = self.partition_size as i32;
        let max_partitions_i32 = max_num_partitions as i32;

        {
            let mut builder = v2_func.builder();
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(q_slice);
            // See V1 dispatch above — Auto uses native slice<T>, FP8/INT8
            // route through U8 byte slice. The kernel template's KV
            // marker drives inline dequant via `load_kv_to_f32`.
            match self.kv_cache_dtype {
                KVCacheDtype::Auto => {
                    let k_native = expect_native_kv_slice::<T>(&k_guard, "K")?;
                    let v_native = expect_native_kv_slice::<T>(&v_guard, "V")?;
                    builder.arg(k_native);
                    builder.arg(v_native);
                    builder.arg(bt_slice);
                    builder.arg(sl_slice);
                    builder.arg(&self.scale);
                    builder.arg(&num_heads_i32);
                    builder.arg(&num_kv_heads_i32);
                    builder.arg(&max_blocks_i32);
                    builder.arg(&head_dim_i32);
                    builder.arg(&block_size_i32);
                    builder.arg(&partition_size_i32);
                    builder.arg(&max_partitions_i32);
                    builder.arg(alibi_slice);
                    push_scale_args(&mut builder, None, None);
                }
                KVCacheDtype::Fp8E4m3 | KVCacheDtype::Fp8E5m2 | KVCacheDtype::Int8 => {
                    let k_u8 = expect_u8_kv_slice(&k_guard, "K")?;
                    let v_u8 = expect_u8_kv_slice(&v_guard, "V")?;
                    builder.arg(k_u8);
                    builder.arg(v_u8);
                    builder.arg(bt_slice);
                    builder.arg(sl_slice);
                    builder.arg(&self.scale);
                    builder.arg(&num_heads_i32);
                    builder.arg(&num_kv_heads_i32);
                    builder.arg(&max_blocks_i32);
                    builder.arg(&head_dim_i32);
                    builder.arg(&block_size_i32);
                    builder.arg(&partition_size_i32);
                    builder.arg(&max_partitions_i32);
                    builder.arg(alibi_slice);
                    push_scale_args(&mut builder, k_scale_slice, v_scale_slice);
                }
            }

            // SAFETY: kernel launch with validated parameters
            unsafe { builder.launch(v2_cfg) }
                .map_err(|e| candle_core::Error::Msg(format!("paged_attention_v2_alibi: {e}")))?;
        }

        // Stage 2: reduce (same kernel family as non-ALiBi — reduction is ALiBi-agnostic).
        let reduce_func =
            dev.get_or_load_custom_func(T::KERNEL_V2_REDUCE, "paged_attention", PTX)?;

        let reduce_shared_bytes =
            ((2 * max_num_partitions + NUM_WARPS) * std::mem::size_of::<f32>()) as u32;

        let reduce_cfg = LaunchConfig {
            grid_dim: (self.num_heads as u32, num_seqs as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: reduce_shared_bytes,
        };

        {
            let mut builder = reduce_func.builder();
            builder.arg(&output_slice);
            builder.arg(&tmp_out_slice);
            builder.arg(&exp_sums_slice);
            builder.arg(&max_logits_slice);
            builder.arg(sl_slice);
            builder.arg(&num_heads_i32);
            builder.arg(&head_dim_i32);
            builder.arg(&partition_size_i32);
            builder.arg(&max_partitions_i32);

            // SAFETY: reduce kernel launch
            unsafe { builder.launch(reduce_cfg) }.map_err(|e| {
                candle_core::Error::Msg(format!("paged_attention_v2_alibi reduce: {e}"))
            })?;
        }

        drop(alibi_guard);
        drop(k_scale_guard_layout);
        drop(v_scale_guard_layout);
        drop(k_guard);
        drop(v_guard);
        drop(bt_guard);
        drop(sl_guard);

        let output_storage = CudaStorage {
            slice: T::into_storage_slice(output_slice),
            device: dev.clone(),
        };
        let output_shape = Shape::from_dims(&[num_seqs, self.num_heads, head_dim]);
        Ok((output_storage, output_shape))
    }
}

/// Auto-selecting paged attention with ALiBi: V1 for short, V2 for long.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_auto_alibi(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    alibi_slopes: &Tensor,
) -> Result<Tensor> {
    if max_seq_len > V2_SEQ_LEN_THRESHOLD {
        let q = q.contiguous()?;
        let num_seqs = q.dim(0)?;
        let partition_size = DEFAULT_V2_PARTITION_SIZE;
        let max_num_partitions = max_seq_len.div_ceil(partition_size);

        let alibi_slopes = alibi_slopes
            .to_dtype(candle_core::DType::F32)?
            .contiguous()?;

        let seq_lens_c = seq_lens.contiguous()?;
        debug_assert_seq_lens_within_bound(&seq_lens_c, max_seq_len)?;

        let op = PagedAttnV2AlibiOp {
            k_cache: k_cache.clone(),
            v_cache: v_cache.clone(),
            block_tables: block_tables.contiguous()?,
            seq_lens: seq_lens_c,
            alibi_slopes,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            partition_size,
            max_num_partitions,
            // Legacy public API: pin Auto KV cache (Phase 10 quantised
            // path is exposed separately).
            kv_cache_dtype: KVCacheDtype::Auto,
            k_scale: None,
            v_scale: None,
        };

        let output = q.apply_op1_no_bwd(&op)?;
        output.reshape((num_seqs, num_heads * head_dim))
    } else {
        paged_attention_cuda_alibi(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            head_dim,
            block_size,
            alibi_slopes,
        )
    }
}

// ============================================================================
// Fused SwiGLU Activation
// ============================================================================

/// Fused SwiGLU operation: output = silu(gate) * up
///
/// Computes the SwiGLU activation in a single kernel pass, avoiding
/// materialization of intermediate results and saving memory bandwidth.
#[cfg(feature = "cuda-fused-activations")]
pub struct FusedSwiGluOp;

#[cfg(feature = "cuda-fused-activations")]
impl CustomOp2 for FusedSwiGluOp {
    fn name(&self) -> &'static str {
        "fused_swiglu"
    }

    fn cpu_fwd(
        &self,
        gate_storage: &CpuStorage,
        gate_layout: &Layout,
        up_storage: &CpuStorage,
        up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // CPU fallback implementation using standard operations
        let gate_shape = gate_layout.shape();
        let up_shape = up_layout.shape();

        if gate_shape != up_shape {
            candle_core::bail!(
                "fused_swiglu: gate and up shapes must match, got {:?} and {:?}",
                gate_shape,
                up_shape
            );
        }

        // For CPU, we do element-wise: silu(gate) * up
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        match (gate_storage, up_storage) {
            (CpuStorage::F32(gate_data), CpuStorage::F32(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<f32> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let silu_g = g / (1.0 + (-g).exp());
                        silu_g * u
                    })
                    .collect();

                Ok((CpuStorage::F32(result), gate_shape.clone()))
            }
            (CpuStorage::BF16(gate_data), CpuStorage::BF16(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<half::bf16> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let g_f32 = g.to_f32();
                        let u_f32 = u.to_f32();
                        let silu_g = g_f32 / (1.0 + (-g_f32).exp());
                        half::bf16::from_f32(silu_g * u_f32)
                    })
                    .collect();

                Ok((CpuStorage::BF16(result), gate_shape.clone()))
            }
            (CpuStorage::F16(gate_data), CpuStorage::F16(up_data)) => {
                let gate_slice = gate_layout.contiguous_offsets();
                let up_slice = up_layout.contiguous_offsets();

                let (gate_start, gate_end) = match gate_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: gate must be contiguous"),
                };
                let (up_start, up_end) = match up_slice {
                    Some((s, e)) => (s, e),
                    None => candle_core::bail!("fused_swiglu: up must be contiguous"),
                };

                let gate_data = &gate_data[gate_start..gate_end];
                let up_data = &up_data[up_start..up_end];

                let result: Vec<half::f16> = gate_data
                    .iter()
                    .zip(up_data.iter())
                    .map(|(&g, &u)| {
                        let g_f32 = g.to_f32();
                        let u_f32 = u.to_f32();
                        let silu_g = g_f32 / (1.0 + (-g_f32).exp());
                        half::f16::from_f32(silu_g * u_f32)
                    })
                    .collect();

                Ok((CpuStorage::F16(result), gate_shape.clone()))
            }
            _ => candle_core::bail!(
                "fused_swiglu: unsupported dtype combination or mismatched dtypes"
            ),
        }
    }

    fn cuda_fwd(
        &self,
        gate_storage: &CudaStorage,
        gate_layout: &Layout,
        up_storage: &CudaStorage,
        up_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &gate_storage.device;
        let gate_shape = gate_layout.shape();
        let up_shape = up_layout.shape();

        if gate_shape != up_shape {
            candle_core::bail!(
                "fused_swiglu: gate and up shapes must match, got {:?} and {:?}",
                gate_shape,
                up_shape
            );
        }

        if gate_layout.start_offset() != 0 {
            candle_core::bail!("fused_swiglu: gate must be contiguous from offset 0");
        }
        if up_layout.start_offset() != 0 {
            candle_core::bail!("fused_swiglu: up must be contiguous from offset 0");
        }

        // Calculate dimensions
        // Shape is [..., hidden_size], we need num_tokens and hidden_size
        let dims = gate_shape.dims();
        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();
        let num_tokens = if num_tokens == 0 { 1 } else { num_tokens };

        let elem_count = num_tokens * hidden_size;
        let block_size = std::cmp::min(hidden_size, 1024);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = hidden_size as i32;

        match (&gate_storage.slice, &up_storage.slice) {
            (CudaStorageSlice::BF16(gate_slice), CudaStorageSlice::BF16(up_slice)) => {
                // SAFETY: SwiGLU writes every output element; uninit safe.
                let output_slice = unsafe { dev.alloc::<half::bf16>(elem_count) }?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_bf16", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::BF16(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            (CudaStorageSlice::F16(gate_slice), CudaStorageSlice::F16(up_slice)) => {
                // SAFETY: SwiGLU writes every output element; uninit safe.
                let output_slice = unsafe { dev.alloc::<half::f16>(elem_count) }?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_fp16", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::F16(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            (CudaStorageSlice::F32(gate_slice), CudaStorageSlice::F32(up_slice)) => {
                let output_slice = dev.alloc_zeros::<f32>(elem_count)?;

                let func =
                    dev.get_or_load_custom_func("fused_swiglu_fp32", "swiglu", SWIGLU_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&hidden_size_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("fused_swiglu launch: {e}")))?;

                let output_storage = CudaStorage {
                    slice: CudaStorageSlice::F32(output_slice),
                    device: dev.clone(),
                };
                Ok((output_storage, gate_shape.clone()))
            }
            _ => candle_core::bail!("fused_swiglu: unsupported dtype, expected BF16, F16, or F32"),
        }
    }
}

/// Fused SwiGLU activation: output = silu(gate) * up
///
/// Computes the SwiGLU activation in a single CUDA kernel pass when the
/// `cuda-fused-activations` feature is enabled. Falls back to CPU implementation
/// for CPU tensors.
///
/// This saves memory bandwidth by avoiding materialization of the intermediate
/// silu(gate) result.
///
/// # Arguments
/// - `gate`: Gate tensor from gate_proj linear layer `[..., hidden_size]`
/// - `up`: Up tensor from up_proj linear layer `[..., hidden_size]`
///
/// # Returns
/// Output tensor `[..., hidden_size]` with same dtype as inputs
///
/// # Panics
/// - If gate and up have different shapes
/// - If gate and up have different dtypes
/// - If tensors are not contiguous
#[cfg(feature = "cuda-fused-activations")]
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;

    gate.apply_op2_no_bwd(&up, &FusedSwiGluOp)
}

/// In-place sibling for the pooled SiLU+Mul fast path on **separate**
/// gate/up tensors (not packed). Receiver = pre-allocated output
/// buffer; `up` is read via the struct ref. Uses kernel
/// `silu_and_mul_separate_bf16`.
///
/// Migration target for `QuantizedSwiGluMlp::forward`, which today
/// does `candle_nn::ops::silu(&gate)? * up` (two ad-hoc allocations:
/// silu intermediate + multiplication output) — a 6.3 MB-per-forward
/// hot-path slot. Pool replaces both with one stable-address receiver
/// buffer.
#[cfg(feature = "cuda-fused-activations")]
struct SiluAndMulSeparateInplaceOp<'a> {
    up: &'a Tensor,
}

#[cfg(feature = "cuda-fused-activations")]
impl<'a> InplaceOp2 for SiluAndMulSeparateInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "silu_and_mul_separate_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("silu_and_mul_separate_inplace requires CUDA")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        gate_storage: &CudaStorage,
        gate_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &gate_storage.device;
        let dims = gate_layout.shape().dims();
        if dims.is_empty() {
            candle_core::bail!("silu_and_mul_separate_inplace: gate must have ≥ 1 dim");
        }
        let d = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();
        if num_tokens == 0 {
            return Ok(());
        }

        let (up_storage, _up_layout) = self.up.storage_and_layout();
        let up_storage = match &*up_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("silu_and_mul_separate_inplace: up must be on CUDA"),
        };

        let block_size = std::cmp::min(d, 1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_i32 = d as i32;

        match (
            &mut out_storage.slice,
            &gate_storage.slice,
            &up_storage.slice,
        ) {
            (
                CudaStorageSlice::BF16(out_slice),
                CudaStorageSlice::BF16(gate_slice),
                CudaStorageSlice::BF16(up_slice),
            ) => {
                let func = dev.get_or_load_custom_func(
                    "silu_and_mul_separate_bf16",
                    "activations",
                    ACTIVATIONS_PTX,
                )?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&d_i32);
                // SAFETY: kernel writes every output element; receiver is
                // exclusively held via &mut out_storage.
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("silu_and_mul_separate_inplace launch: {e}"))
                })?;
            }
            (
                CudaStorageSlice::F16(out_slice),
                CudaStorageSlice::F16(gate_slice),
                CudaStorageSlice::F16(up_slice),
            ) => {
                // Phase 11.2.C: F16 sibling for the EXL3-Llama decode
                // forward (activations forced to F16 by main.rs).
                let func = dev.get_or_load_custom_func(
                    "silu_and_mul_separate_fp16",
                    "activations",
                    ACTIVATIONS_PTX,
                )?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(gate_slice);
                builder.arg(up_slice);
                builder.arg(&d_i32);
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("silu_and_mul_separate_inplace launch: {e}"))
                })?;
            }
            _ => candle_core::bail!(
                "silu_and_mul_separate_inplace: only bf16/f16 supported (out/gate/up dtype mismatch)"
            ),
        }
        Ok(())
    }
}

/// Pool-backed `silu(gate) * up` for the decode hot path. Reserves
/// the output from [`OutputPool`] so device addresses stay stable
/// across forwards. Decode-only gate (num_tokens ≤ 64) — prefill
/// falls through to the candle `silu()? * up` path (broadcast_mul).
#[cfg(feature = "cuda-fused-activations")]
pub fn silu_and_mul_separate_pooled(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    crate::engine::cr_trace::mark_op("silu_and_mul_separate_pooled");
    use crate::engine::output_pool::OutputPool;

    /// Decode-shape budget; matches the other pool wrappers.
    const POOL_MAX_NUM_TOKENS: usize = 64;

    if gate.shape() != up.shape() {
        candle_core::bail!(
            "silu_and_mul_separate_pooled: gate {:?} and up {:?} shape mismatch",
            gate.shape(),
            up.shape()
        );
    }
    let gate_dt = gate.dtype();
    if gate_dt != candle_core::DType::BF16 && gate_dt != candle_core::DType::F16 {
        // Fallback for unsupported dtypes (F32 etc.) — kernels exist
        // only for BF16 (Qwen3-AWQ path) and F16 (Phase 11.2.C, EXL3
        // path).
        let activated = candle_nn::ops::silu(gate)?;
        return activated.broadcast_mul(up);
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let dims = gate.dims();
    let num_tokens: usize = dims[..dims.len().saturating_sub(1)].iter().product();
    if num_tokens > POOL_MAX_NUM_TOKENS {
        let activated = candle_nn::ops::silu(&gate)?;
        return activated.broadcast_mul(&up);
    }

    let shape: Vec<usize> = dims.to_vec();
    let dtype = gate.dtype();
    let output = OutputPool::global().reserve(&shape, dtype, gate.device())?;
    let op = SiluAndMulSeparateInplaceOp { up: &up };
    output.inplace_op2(&gate, &op)?;
    Ok(output)
}

/// In-place sibling for the pool-backed embedding lookup. Receiver =
/// pre-allocated `[num_tokens, hidden_size]` BF16 output; primary
/// "input" passed via inplace_op2 is the int32-shaped `input_ids`
/// (U32 storage; bit-pattern matches int for vocab ≤ 2^31). Weight
/// matrix is held by the struct.
///
/// Backs `embedding_pooled` — captures the embedding output into a
/// stable-address pool buffer so the captured CUDA graph reads from
/// a consistent device pointer across replays.
#[cfg(feature = "cuda-fused-activations")]
struct EmbeddingLookupInplaceOp<'a> {
    weight: &'a Tensor,
    hidden_size: usize,
}

#[cfg(feature = "cuda-fused-activations")]
impl<'a> InplaceOp2 for EmbeddingLookupInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "embedding_lookup_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("embedding_lookup_inplace requires CUDA")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        ids_storage: &CudaStorage,
        ids_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &ids_storage.device;
        let dims = ids_layout.shape().dims();
        let num_tokens: usize = dims.iter().product();
        if num_tokens == 0 {
            return Ok(());
        }

        let (w_storage, _w_layout) = self.weight.storage_and_layout();
        let w_storage = match &*w_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("embedding_lookup_inplace: weight must be on CUDA"),
        };

        let block_size = std::cmp::min(self.hidden_size, 1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = self.hidden_size as i32;

        match (&mut out_storage.slice, &w_storage.slice, &ids_storage.slice) {
            (
                CudaStorageSlice::BF16(out_slice),
                CudaStorageSlice::BF16(w_slice),
                CudaStorageSlice::U32(ids_slice),
            ) => {
                let func = dev.get_or_load_custom_func(
                    "embedding_lookup_bf16",
                    "activations",
                    ACTIVATIONS_PTX,
                )?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(w_slice);
                builder.arg(ids_slice);
                builder.arg(&hidden_i32);
                // SAFETY: kernel writes every element of out; receiver held mutably.
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("embedding_lookup_inplace launch: {e}"))
                })?;
            }
            (
                CudaStorageSlice::F16(out_slice),
                CudaStorageSlice::F16(w_slice),
                CudaStorageSlice::U32(ids_slice),
            ) => {
                // Phase 11.2.C: F16 sibling for the EXL3-Llama embed
                // lookup (compute dtype = F16).
                let func = dev.get_or_load_custom_func(
                    "embedding_lookup_fp16",
                    "activations",
                    ACTIVATIONS_PTX,
                )?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(w_slice);
                builder.arg(ids_slice);
                builder.arg(&hidden_i32);
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("embedding_lookup_fp16 launch: {e}"))
                })?;
            }
            _ => {
                candle_core::bail!(
                    "embedding_lookup_inplace: only BF16/F16 weight + U32 ids supported"
                )
            }
        }
        Ok(())
    }
}

/// In-place sibling for the pool-backed BF16 element-wise add.
/// Receiver = pre-allocated `[..., hidden]` BF16 output. The first
/// addend (`a`) is passed via inplace_op2; the second (`b`) is held
/// by the struct.
#[cfg(feature = "cuda-fused-activations")]
struct Bf16AddInplaceOp<'a> {
    b: &'a Tensor,
}

#[cfg(feature = "cuda-fused-activations")]
impl<'a> InplaceOp2 for Bf16AddInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "bf16_add_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("bf16_add_inplace requires CUDA")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        a_storage: &CudaStorage,
        a_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &a_storage.device;
        let dims = a_layout.shape().dims();
        if dims.is_empty() {
            candle_core::bail!("bf16_add_inplace: input must have ≥ 1 dim");
        }
        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();
        let num_tokens = if num_tokens == 0 { 1 } else { num_tokens };

        let (b_storage, _b_layout) = self.b.storage_and_layout();
        let b_storage = match &*b_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("bf16_add_inplace: b must be on CUDA"),
        };

        let block_size = std::cmp::min(hidden_size, 1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i32 = hidden_size as i32;

        match (&mut out_storage.slice, &a_storage.slice, &b_storage.slice) {
            (
                CudaStorageSlice::BF16(out_slice),
                CudaStorageSlice::BF16(a_slice),
                CudaStorageSlice::BF16(b_slice),
            ) => {
                let func =
                    dev.get_or_load_custom_func("add_bf16", "activations", ACTIVATIONS_PTX)?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(a_slice);
                builder.arg(b_slice);
                builder.arg(&hidden_i32);
                // SAFETY: kernel writes every output element; receiver held mutably.
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("bf16_add_inplace launch: {e}"))
                })?;
            }
            (
                CudaStorageSlice::F16(out_slice),
                CudaStorageSlice::F16(a_slice),
                CudaStorageSlice::F16(b_slice),
            ) => {
                // Phase 11.2.C: F16 sibling for the EXL3-Llama decode
                // forward (residual-add inputs are F16).
                let func =
                    dev.get_or_load_custom_func("add_fp16", "activations", ACTIVATIONS_PTX)?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(a_slice);
                builder.arg(b_slice);
                builder.arg(&hidden_i32);
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("add_fp16 inplace launch: {e}"))
                })?;
            }
            _ => candle_core::bail!("add_inplace: only BF16/F16 supported"),
        }
        Ok(())
    }
}

/// Pool-backed element-wise BF16 add `c = a + b`. Reserves the
/// output from [`OutputPool`] (decode-only gate, num_tokens ≤ 64) and
/// runs the `add_bf16` kernel into the receiver. Falls back to
/// candle's `Tensor::add` for prefill / non-BF16 / non-CUDA.
///
/// Migration target for the residual-add sites in the per-layer
/// decoder: `(xs + residual)?` and `residual + xs`. Candle's `+`
/// allocates a fresh output each call — 72 fresh allocations per
/// Qwen3-4B decode forward, all of which leaked unstable device
/// pointers into the captured CUDA graph.
#[cfg(feature = "cuda-fused-activations")]
pub fn half_add_pooled(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    crate::engine::cr_trace::mark_op("half_add_pooled");
    use crate::engine::output_pool::OutputPool;

    /// Decode-shape budget; matches the other pool wrappers.
    const POOL_MAX_NUM_TOKENS: usize = 64;

    let a_dt = a.dtype();
    let b_dt = b.dtype();
    if a_dt != b_dt
        || !matches!(a_dt, candle_core::DType::BF16 | candle_core::DType::F16)
        || !a.device().is_cuda()
    {
        // Unsupported dtype / CPU — defer to candle.
        return (a + b)?.contiguous();
    }
    if a.shape() != b.shape() {
        candle_core::bail!(
            "half_add_pooled: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }

    let a = a.contiguous()?;
    let b_c = b.contiguous()?;
    let dims = a.dims();
    let num_tokens: usize = dims[..dims.len().saturating_sub(1)].iter().product();
    if num_tokens > POOL_MAX_NUM_TOKENS {
        return (&a + &b_c)?.contiguous();
    }

    let shape: Vec<usize> = dims.to_vec();
    let output = OutputPool::global().reserve(&shape, a_dt, a.device())?;
    let op = Bf16AddInplaceOp { b: &b_c };
    output.inplace_op2(&a, &op)?;
    Ok(output)
}

/// In-place BF16 matmul `Y = X @ W^T` where X is `[m, k]`, W is
/// `[n, k]`, Y is `[m, n]`. Receiver = output Y; primary "input" =
/// X. Weight W is held by the struct.
///
/// Uses cuBLAS HGEMM via cudarc to write into a pre-allocated pool
/// buffer. Migration target for `TiedEmbeddingHead::forward` —
/// candle's `Tensor::matmul` allocates the result fresh per call,
/// which is the last unstable-address allocation on the Qwen3-AWQ
/// decode hot path blocking CUDA Graph capture replay.
#[cfg(feature = "cuda-kernels")]
struct Bf16MatmulInplaceOp<'a> {
    weight: &'a Tensor,
    m: usize,
    n: usize,
    k: usize,
}

#[cfg(feature = "cuda-kernels")]
impl<'a> InplaceOp2 for Bf16MatmulInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "bf16_matmul_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("bf16_matmul_inplace requires CUDA")
    }

    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        x_storage: &CudaStorage,
        _x_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::cublas::{sys, Gemm, GemmConfig};

        let (w_storage, _w_layout) = self.weight.storage_and_layout();
        let w_storage = match &*w_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("bf16_matmul_inplace: weight must be on CUDA"),
        };

        let device = match self.weight.device() {
            candle_core::Device::Cuda(d) => d,
            _ => candle_core::bail!("bf16_matmul_inplace: weight must be on CUDA"),
        };
        // Reuse candle's process-wide cuBLAS handle (bound to the
        // device's default stream at startup). Constructing a fresh
        // CudaBlas inside the captured stream invalidates the capture
        // (cuBLAS init does host syncs / non-capture-safe state).
        let blas = device.cublas_handle();

        match (&mut out_storage.slice, &x_storage.slice, &w_storage.slice) {
            (
                CudaStorageSlice::BF16(out_slice),
                CudaStorageSlice::BF16(x_slice),
                CudaStorageSlice::BF16(w_slice),
            ) => {
                // Y = X @ W^T where row-major shapes: X=[m,k], W=[n,k], Y=[m,n].
                // cuBLAS is column-major; treating row-major as transposed:
                //   X (row-major [m,k]) ≡ column-major [k,m].
                //   W (row-major [n,k]) ≡ column-major [k,n].
                //   Y (row-major [m,n]) ≡ column-major [n,m].
                // Compute Y_col = (W_col)^T @ X_col → [n,k]^T·[k,m] = [n,m]. ✓
                let cfg = GemmConfig::<half::bf16> {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: self.n as i32,
                    n: self.m as i32,
                    k: self.k as i32,
                    alpha: half::bf16::from_f32(1.0),
                    lda: self.k as i32,
                    ldb: self.k as i32,
                    beta: half::bf16::from_f32(0.0),
                    ldc: self.n as i32,
                };
                // SAFETY: shapes validated; cuBLAS BF16 GEMM into pool buffer.
                unsafe {
                    blas.gemm(cfg, w_slice, x_slice, out_slice).map_err(|e| {
                        candle_core::Error::Msg(format!("bf16_matmul_inplace gemm: {e}"))
                    })?;
                }
            }
            (
                CudaStorageSlice::F16(out_slice),
                CudaStorageSlice::F16(x_slice),
                CudaStorageSlice::F16(w_slice),
            ) => {
                // Phase 11.2.C: F16 sibling for EXL3-Llama lm_head matmul.
                // Same row-major → column-major cuBLAS gymnastics as the
                // BF16 case, just with `half::f16` operand type (cuBLAS
                // HGEMM).
                let cfg = GemmConfig::<half::f16> {
                    transa: sys::cublasOperation_t::CUBLAS_OP_T,
                    transb: sys::cublasOperation_t::CUBLAS_OP_N,
                    m: self.n as i32,
                    n: self.m as i32,
                    k: self.k as i32,
                    alpha: half::f16::from_f32(1.0),
                    lda: self.k as i32,
                    ldb: self.k as i32,
                    beta: half::f16::from_f32(0.0),
                    ldc: self.n as i32,
                };
                unsafe {
                    blas.gemm(cfg, w_slice, x_slice, out_slice).map_err(|e| {
                        candle_core::Error::Msg(format!("f16_matmul_inplace gemm: {e}"))
                    })?;
                }
            }
            _ => candle_core::bail!("matmul_inplace: only BF16/F16 supported"),
        }

        Ok(())
    }
}

/// Pool-backed BF16 matmul `Y = X @ W^T`. Reserves Y from
/// [`OutputPool`] (decode-only gate `m ≤ 64`) and runs cuBLAS HGEMM
/// into the receiver. Falls back to candle's `Tensor::matmul` for
/// prefill, non-BF16 dtypes, or non-CUDA devices.
#[cfg(feature = "cuda-kernels")]
pub fn half_matmul_pooled(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    use crate::engine::output_pool::OutputPool;
    crate::engine::cr_trace::mark_op("half_matmul_pooled");

    /// Decode-shape budget; matches the other pool wrappers.
    const POOL_MAX_M: usize = 64;

    let in_dt = input.dtype();
    let w_dt = weight.dtype();
    if in_dt != w_dt
        || !matches!(in_dt, candle_core::DType::BF16 | candle_core::DType::F16)
        || !input.device().is_cuda()
    {
        // Unsupported dtype / CPU — defer to candle.
        return input.matmul(&weight.t()?);
    }

    let in_dims = input.dims();
    if in_dims.len() < 2 {
        candle_core::bail!("half_matmul_pooled: input must have ≥ 2 dims");
    }
    let k = in_dims[in_dims.len() - 1];
    let m: usize = in_dims[..in_dims.len() - 1].iter().product();

    let w_dims = weight.dims();
    if w_dims.len() != 2 || w_dims[1] != k {
        return input.matmul(&weight.t()?);
    }
    let n = w_dims[0];

    if m > POOL_MAX_M {
        return input.matmul(&weight.t()?);
    }

    let input = input.contiguous()?;
    let weight_c = weight.contiguous()?;

    let mut out_shape: Vec<usize> = in_dims[..in_dims.len() - 1].to_vec();
    out_shape.push(n);
    let output = OutputPool::global().reserve(&out_shape, in_dt, input.device())?;

    let input_2d = input.reshape((m, k))?;
    let output_2d = output.reshape((m, n))?;

    let op = Bf16MatmulInplaceOp {
        weight: &weight_c,
        m,
        n,
        k,
    };
    output_2d.inplace_op2(&input_2d, &op)?;

    if in_dims.len() == 2 {
        Ok(output)
    } else {
        output.reshape(out_shape)
    }
}

/// Pool-backed embedding lookup: gather rows of `weight` indexed by
/// `input_ids` into a stable-address pool buffer. Replaces
/// `Embedding::forward` on the decode hot path so the captured CUDA
/// graph's recorded device pointer to the embedding output stays
/// consistent across replays.
///
/// Decode-only gate (num_tokens ≤ 64); larger inputs fall through to
/// the candle `Embedding::forward` path inside the caller.
#[cfg(feature = "cuda-fused-activations")]
pub fn embedding_pooled(input_ids: &Tensor, weight: &Tensor) -> Result<Tensor> {
    use crate::engine::output_pool::OutputPool;
    crate::engine::cr_trace::mark_op("embedding_pooled");

    /// Decode-shape budget; matches the other pool wrappers.
    const POOL_MAX_NUM_TOKENS: usize = 64;

    if !input_ids.device().is_cuda() {
        candle_core::bail!("embedding_pooled: requires CUDA device");
    }
    if !matches!(
        weight.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16
    ) {
        candle_core::bail!("embedding_pooled: weight must be BF16 or F16");
    }
    if input_ids.dtype() != candle_core::DType::U32 {
        candle_core::bail!("embedding_pooled: input_ids must be U32");
    }

    let ids_dims = input_ids.dims();
    let num_tokens: usize = ids_dims.iter().product();
    if num_tokens == 0 || num_tokens > POOL_MAX_NUM_TOKENS {
        candle_core::bail!(
            "embedding_pooled: num_tokens {num_tokens} out of decode budget [1, {POOL_MAX_NUM_TOKENS}]"
        );
    }

    let weight_dims = weight.dims();
    if weight_dims.len() != 2 {
        candle_core::bail!("embedding_pooled: weight must be 2D [vocab, hidden]");
    }
    let hidden_size = weight_dims[1];

    let mut out_shape: Vec<usize> = ids_dims.to_vec();
    out_shape.push(hidden_size);

    let input_ids = input_ids.contiguous()?;
    let output = OutputPool::global().reserve(&out_shape, weight.dtype(), input_ids.device())?;
    let op = EmbeddingLookupInplaceOp {
        weight,
        hidden_size,
    };
    output.inplace_op2(&input_ids, &op)?;
    Ok(output)
}

/// Check if fused SwiGLU CUDA kernel is available.
///
/// Returns true if:
/// - The `cuda-fused-activations` feature is enabled
/// - The tensor is on a CUDA device
#[cfg(feature = "cuda-fused-activations")]
pub fn fused_swiglu_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-fused-activations"))]
pub fn fused_swiglu_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// RMSNorm CUDA Kernel
// ============================================================================

#[cfg(feature = "cuda-layernorm")]
const LAYERNORM_PTX: &str = include_str!("../kernels/layernorm.ptx");

/// Fused RMSNorm operation: output = (input / sqrt(mean(input^2) + eps)) * weight
///
/// Uses vectorized loads for BF16/FP16 (4 elements per thread per iteration)
/// and warp+block-level reduction for the variance computation.
#[cfg(feature = "cuda-layernorm")]
struct RmsNormOp {
    weight: Tensor,
    epsilon: f32,
}

#[cfg(feature = "cuda-layernorm")]
impl CustomOp1 for RmsNormOp {
    fn name(&self) -> &'static str {
        "rms_norm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("rms_norm CUDA kernel does not support CPU")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::{bf16, f16};

        let dev = &input_storage.device;
        let input_shape = input_layout.shape();
        let dims = input_shape.dims();

        if dims.is_empty() {
            candle_core::bail!("rms_norm: input must have at least 1 dimension");
        }

        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        if num_tokens == 0 {
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                input_shape.clone(),
            ));
        }

        let (weight_storage, _weight_layout) = self.weight.storage_and_layout();
        let weight_storage = match &*weight_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rms_norm: weight must be on CUDA device"),
        };

        // Block size: threads per block. One block per token.
        // For large num_tokens, smaller blocks give better SM utilization.
        let max_block_size: u32 = if num_tokens < 256 { 1024 } else { 256 };
        let block_size = std::cmp::min(hidden_size as u32, max_block_size);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = hidden_size as i32;

        match (&input_storage.slice, &weight_storage.slice) {
            (CudaStorageSlice::BF16(input_slice), CudaStorageSlice::BF16(weight_slice)) => {
                // SAFETY: RMSNorm fills every output element via the
                // per-token block; uninit safe.
                let output_slice = unsafe { dev.alloc::<bf16>(num_tokens * hidden_size) }?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_bf16", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            (CudaStorageSlice::F16(input_slice), CudaStorageSlice::F16(weight_slice)) => {
                let output_slice = dev.alloc_zeros::<f16>(num_tokens * hidden_size)?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_fp16", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F16(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            (CudaStorageSlice::F32(input_slice), CudaStorageSlice::F32(weight_slice)) => {
                let output_slice = dev.alloc_zeros::<f32>(num_tokens * hidden_size)?;
                let func =
                    dev.get_or_load_custom_func("rms_norm_f32", "layernorm", LAYERNORM_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(weight_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rms_norm launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F32(output_slice),
                        device: dev.clone(),
                    },
                    input_shape.clone(),
                ))
            }
            _ => candle_core::bail!("rms_norm: unsupported dtype combination"),
        }
    }
}

/// CUDA-accelerated RMSNorm.
///
/// Applies RMSNorm using a vectorized CUDA kernel that fuses the variance
/// computation, normalization, and weight scaling into a single pass.
///
/// # Arguments
/// - `input`: Input tensor `[..., hidden_size]`, contiguous in last dim
/// - `weight`: Weight tensor `[hidden_size]`
/// - `epsilon`: Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Returns
/// Normalized tensor with same shape and dtype as input
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    let input = input.contiguous()?;
    let op = RmsNormOp {
        weight: weight.clone(),
        epsilon,
    };
    input.apply_op1_no_bwd(&op)
}

/// In-place sibling of [`RmsNormOp`] used by the pooled fast path.
///
/// `Tensor::inplace_op2(&input, &RmsNormInplaceOp { ... })` writes the
/// normalised output into the receiver tensor's storage. The receiver is
/// a buffer reserved from [`crate::engine::output_pool::OutputPool`] —
/// it lives across forward passes, so the per-call `dev.alloc()` that
/// `RmsNormOp::cuda_fwd` performs disappears. This is the precondition
/// for CUDA Graph capture: the captured graph records the receiver's
/// device pointer, which stays stable across replays.
///
/// The kernel launch logic mirrors [`RmsNormOp::cuda_fwd`] exactly
/// (same dispatch table, same launch config); only the output pointer
/// changes.
#[cfg(feature = "cuda-layernorm")]
struct RmsNormInplaceOp<'a> {
    weight: &'a Tensor,
    epsilon: f32,
}

#[cfg(feature = "cuda-layernorm")]
impl<'a> InplaceOp2 for RmsNormInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "rms_norm_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("rms_norm_inplace requires CUDA")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        in_storage: &CudaStorage,
        in_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &in_storage.device;
        let dims = in_layout.shape().dims();

        if dims.is_empty() {
            candle_core::bail!("rms_norm_inplace: input must have at least 1 dimension");
        }

        let hidden_size = dims[dims.len() - 1];
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        if num_tokens == 0 {
            return Ok(());
        }

        let (weight_storage, _weight_layout) = self.weight.storage_and_layout();
        let weight_storage = match &*weight_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rms_norm_inplace: weight must be on CUDA device"),
        };

        let max_block_size: u32 = if num_tokens < 256 { 1024 } else { 256 };
        let block_size = std::cmp::min(hidden_size as u32, max_block_size);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let hidden_size_i32 = hidden_size as i32;

        match (
            &mut out_storage.slice,
            &in_storage.slice,
            &weight_storage.slice,
        ) {
            (
                CudaStorageSlice::BF16(out_slice),
                CudaStorageSlice::BF16(in_slice),
                CudaStorageSlice::BF16(w_slice),
            ) => {
                let func =
                    dev.get_or_load_custom_func("rms_norm_bf16", "layernorm", LAYERNORM_PTX)?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(in_slice);
                builder.arg(w_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);
                // SAFETY: kernel launch with validated parameters; output buffer
                // owned by the pool and exclusively held by `&mut out_storage`.
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("rms_norm_inplace launch: {e}"))
                })?;
            }
            (
                CudaStorageSlice::F16(out_slice),
                CudaStorageSlice::F16(in_slice),
                CudaStorageSlice::F16(w_slice),
            ) => {
                let func =
                    dev.get_or_load_custom_func("rms_norm_fp16", "layernorm", LAYERNORM_PTX)?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(in_slice);
                builder.arg(w_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("rms_norm_inplace launch: {e}"))
                })?;
            }
            (
                CudaStorageSlice::F32(out_slice),
                CudaStorageSlice::F32(in_slice),
                CudaStorageSlice::F32(w_slice),
            ) => {
                let func =
                    dev.get_or_load_custom_func("rms_norm_f32", "layernorm", LAYERNORM_PTX)?;
                let mut builder = func.builder();
                builder.arg(out_slice);
                builder.arg(in_slice);
                builder.arg(w_slice);
                builder.arg(&self.epsilon);
                builder.arg(&hidden_size_i32);
                unsafe { builder.launch(cfg) }.map_err(|e| {
                    candle_core::Error::Msg(format!("rms_norm_inplace launch: {e}"))
                })?;
            }
            _ => candle_core::bail!("rms_norm_inplace: unsupported dtype combination"),
        }

        Ok(())
    }
}

/// Pool-backed RMSNorm for the decode hot path: reserves the output
/// tensor from the process-global [`OutputPool`] and runs the kernel
/// through the [`InplaceOp2`] path, eliminating the per-call
/// `dev.alloc()` that the non-pooled [`rms_norm_cuda`] performs. The
/// CUDA Graph capture path requires every per-layer scratch tensor to
/// live at a stable device address — this is the migration target for
/// `crate::layers::RmsNorm`.
///
/// Falls back to the non-pooled path on prefill-shape inputs:
/// prefill's `num_tokens` varies per request (prompt length), and a
/// 36-layer × 3-call/layer pass through the pool would allocate
/// hundreds of multi-MB buffers per forward — fast OOM on small
/// (8 GiB) GPUs. Decode is bounded by `MAX_DECODE_NUM_TOKENS`
/// (= max captured batch size 32, with safety margin), where
/// `num_tokens = batch_size × 1`.
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda_pooled(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    use std::sync::atomic::Ordering;
    RMS_NORM_CUDA_POOLED_COUNT.fetch_add(1, Ordering::Relaxed);
    crate::engine::cr_trace::mark_op("rms_norm_cuda_pooled");
    use crate::engine::output_pool::OutputPool;

    /// Inputs with up to this many tokens (= batch_size for decode) go
    /// through the pool. Prefill (much larger) falls through to fresh
    /// alloc to avoid pool memory pressure.
    const POOL_MAX_NUM_TOKENS: usize = 64;

    let input = input.contiguous()?;
    let dims = input.dims();
    let num_tokens: usize = dims[..dims.len().saturating_sub(1)].iter().product();
    if num_tokens > POOL_MAX_NUM_TOKENS {
        // Prefill or any non-hot-path caller — fall back to non-pooled.
        return rms_norm_cuda(&input, weight, epsilon);
    }

    let shape: Vec<usize> = dims.to_vec();
    let dtype = input.dtype();

    let output = OutputPool::global().reserve(&shape, dtype, input.device())?;
    let op = RmsNormInplaceOp { weight, epsilon };
    output.inplace_op2(&input, &op)?;
    Ok(output)
}

/// Diagnostic counter for `rms_norm_cuda_pooled` invocations. Kept as
/// permanent observability infrastructure so future perf sessions can
/// verify dispatch routing without a fresh code patch — see
/// `memory/cuda_layernorm_4pct_regression.md` for the investigation
/// that motivated this. Read via `Ordering::Relaxed`.
#[cfg(feature = "cuda-layernorm")]
pub static RMS_NORM_CUDA_POOLED_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Check if CUDA RMSNorm kernel is available.
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-layernorm"))]
pub fn rms_norm_cuda_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// Fused RoPE CUDA Kernel
// ============================================================================

#[cfg(feature = "cuda-kernels")]
const ROPE_PTX: &str = include_str!("../kernels/rope.ptx");

/// Apply fused RoPE to a single tensor (Q or K) using a CUDA kernel.
///
/// Uses CustomOp2: primary operand is the tensor to transform, secondary is
/// a packed {positions, cos_sin_cache} tensor. The kernel modifies data in-place
/// on a freshly allocated copy, avoiding mutation of the input tensor.
///
/// # Arguments
/// - `tensor`: Query or Key tensor `[num_tokens, num_heads * head_size]`
/// - `positions`: Position indices `[num_tokens]` i32
/// - `cos_sin_cache`: Precomputed `[max_position, rot_dim]` f32 where each row is
///   `[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`
/// - `rot_dim`: Rotary dimension (= 2 * half_dim)
/// - `head_size`: Full head dimension
/// - `num_heads`: Number of heads for this tensor
/// - `is_neox`: true for NeoX-style (split halves), false for GPT-J (interleaved)
#[cfg(feature = "cuda-kernels")]
struct RopeOp {
    positions: Tensor,
    cos_sin_cache: Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    is_neox: bool,
}

#[cfg(feature = "cuda-kernels")]
impl CustomOp1 for RopeOp {
    fn name(&self) -> &'static str {
        "rope_fused"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("rope_fused CUDA kernel does not support CPU")
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::bf16;

        let dev = &input_storage.device;
        let shape = input_layout.shape();
        let num_tokens = shape.dims()[0];

        if num_tokens == 0 {
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                shape.clone(),
            ));
        }

        let total_elems: usize = shape.elem_count();
        let query_stride = (self.num_heads * self.head_size) as i32;
        let rot_dim_i32 = self.rot_dim as i32;
        let head_size_i32 = self.head_size as i32;
        let num_heads_i32 = self.num_heads as i32;
        // key is null — we process one tensor at a time
        let num_kv_heads_i32 = 0i32;
        let key_stride = 0i32;

        let block_dim = std::cmp::min(self.num_heads * self.rot_dim / 2, 512) as u32;
        let block_dim = std::cmp::max(block_dim, 1);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel_name = if self.is_neox {
            "rotary_embedding_neox_bf16"
        } else {
            "rotary_embedding_gptj_bf16"
        };

        // Get positions storage. The kernel reads each entry as `int`
        // (4 bytes); for non-negative position indices ≤ 2^31 the bit
        // pattern of u32 and i32 is identical, so the U32 path hands the
        // device pointer to the kernel directly — avoiding a synchronous
        // device→host→device round-trip on every RoPE call (which also
        // blocked CUDA-graph capture).
        //
        // I64 positions still need a one-pass conversion; we materialise
        // an i32 buffer on host. This path is currently unreachable from
        // `apply_varlen` (which always builds U32) but kept as a defensive
        // fallback. A one-shot warn tags any future regression.
        let (pos_storage, _pos_layout) = self.positions.storage_and_layout();
        let pos_storage = match &*pos_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope: positions must be on CUDA"),
        };
        let pos_i32_owned: Option<candle_core::cuda::cudarc::driver::CudaSlice<i32>> =
            match &pos_storage.slice {
                CudaStorageSlice::U32(_) => None,
                CudaStorageSlice::I64(_) => {
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| {
                        tracing::warn!(
                            target: "vllm_core::rope",
                            "RoPE positions arrived as I64 — host-pull conversion path active. \
                             Caller should pass U32 to avoid the per-call host sync."
                        );
                    });
                    let pos_vec: Vec<i32> = self
                        .positions
                        .to_vec1::<i64>()?
                        .iter()
                        .map(|&v| v as i32)
                        .collect();
                    Some(dev.clone_htod(&pos_vec)?)
                }
                _ => candle_core::bail!(
                    "rope: positions must be i64 or u32, got {:?}",
                    self.positions.dtype()
                ),
            };

        // Get cos_sin_cache storage
        let (cache_storage, _cache_layout) = self.cos_sin_cache.storage_and_layout();
        let cache_storage = match &*cache_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope: cos_sin_cache must be on CUDA"),
        };
        let cache_slice = match &cache_storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("rope: cos_sin_cache must be f32"),
        };

        match &input_storage.slice {
            CudaStorageSlice::BF16(input_slice) => {
                // Copy input to output buffer (kernel modifies in-place).
                // The buffer is fully overwritten by `memcpy_dtod` before
                // the kernel reads it; the implicit zero-init in
                // `alloc_zeros` was dead work — switch to uninit alloc.
                // SAFETY: every byte is initialised by `memcpy_dtod` prior
                // to any read.
                let mut output_slice = unsafe { dev.alloc::<bf16>(total_elems) }?;
                dev.memcpy_dtod(input_slice, &mut output_slice)?;

                let func = dev.get_or_load_custom_func(kernel_name, "rope", ROPE_PTX)?;

                let mut builder = func.builder();
                // Positions arg: U32 source goes directly (kernel reads as `int`,
                // bit pattern matches for non-negative indices); I64 source uses
                // the host-converted I32 slice.
                if let CudaStorageSlice::U32(s) = &pos_storage.slice {
                    builder.arg(s);
                } else if let Some(s) = pos_i32_owned.as_ref() {
                    builder.arg(s);
                } else {
                    candle_core::bail!("rope: positions dtype unsupported (must be U32 or I64)");
                }
                builder.arg(&output_slice); // query (modified in-place)
                builder.arg(&0u64); // key = nullptr
                builder.arg(cache_slice); // cos_sin_cache
                builder.arg(&rot_dim_i32);
                builder.arg(&query_stride);
                builder.arg(&key_stride); // unused since key=null
                builder.arg(&head_size_i32);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32); // 0

                // SAFETY: kernel launch with validated params, output_slice is freshly allocated
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    shape.clone(),
                ))
            }
            _ => candle_core::bail!("rope_fused: only bf16 supported"),
        }
    }
}

/// Apply fused RoPE CUDA kernel to Q and K tensors.
///
/// Each tensor is processed with a single kernel launch (2 launches total).
/// The kernel replaces ~6 candle ops (split, mul, sub, mul, add, cat) per tensor.
///
/// # Arguments
/// - `q`: Query `[num_tokens, num_heads, head_size]` bf16
/// - `k`: Key `[num_tokens, num_kv_heads, head_size]` bf16
/// - `positions`: Position for each token `[num_tokens]`
/// - `cos_sin_cache`: `[max_position, rot_dim]` f32 — interleaved cos/sin
/// - `rot_dim`: Rotary dimension (= 2 * half_dim)
/// - `head_size`: Full head dimension
/// - `num_heads`: Number of Q heads
/// - `num_kv_heads`: Number of KV heads
/// - `is_neox`: NeoX vs GPT-J style
#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn rotary_embedding_cuda(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    cos_sin_cache: &Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    let num_tokens = positions.dim(0)?;
    if num_tokens == 0 {
        return Ok((q.clone(), k.clone()));
    }

    // Flatten [num_tokens, heads, head_size] → [num_tokens, heads * head_size]
    let q_flat = q.reshape((num_tokens, num_heads * head_size))?;
    let k_flat = k.reshape((num_tokens, num_kv_heads * head_size))?;

    let q_op = RopeOp {
        positions: positions.clone(),
        cos_sin_cache: cos_sin_cache.clone(),
        rot_dim,
        head_size,
        num_heads,
        is_neox,
    };
    let q_out = q_flat.contiguous()?.apply_op1_no_bwd(&q_op)?;

    let k_op = RopeOp {
        positions: positions.clone(),
        cos_sin_cache: cos_sin_cache.clone(),
        rot_dim,
        head_size,
        num_heads: num_kv_heads,
        is_neox,
    };
    let k_out = k_flat.contiguous()?.apply_op1_no_bwd(&k_op)?;

    // Reshape back to [num_tokens, heads, head_size]
    let q_out = q_out.reshape(q.shape())?;
    let k_out = k_out.reshape(k.shape())?;

    Ok((q_out, k_out))
}

/// Check if CUDA RoPE kernel is available.
#[cfg(feature = "cuda-kernels")]
pub fn rotary_embedding_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn rotary_embedding_cuda_available(_tensor: &Tensor) -> bool {
    false
}

/// In-place sibling of [`RopeOp`] used by the pooled fast path.
///
/// `out.inplace_op2(&input, &RopeInplaceOp { ... })` first copies the
/// input bytes into the receiver via `memcpy_dtod`, then launches the
/// in-place RoPE kernel against the receiver. Receiver is a buffer
/// reserved from the global [`OutputPool`]; its device pointer stays
/// stable across forwards, which is what CUDA Graph capture needs.
#[cfg(feature = "cuda-kernels")]
struct RopeInplaceOp<'a> {
    positions: &'a Tensor,
    cos_sin_cache: &'a Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    is_neox: bool,
}

#[cfg(feature = "cuda-kernels")]
impl<'a> InplaceOp2 for RopeInplaceOp<'a> {
    fn name(&self) -> &'static str {
        "rope_fused_inplace"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("rope_fused_inplace requires CUDA")
    }

    fn cuda_fwd(
        &self,
        out_storage: &mut CudaStorage,
        _out_layout: &Layout,
        in_storage: &CudaStorage,
        in_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        let dev = &in_storage.device;
        let shape = in_layout.shape();
        let num_tokens = shape.dims()[0];

        if num_tokens == 0 {
            return Ok(());
        }

        let total_elems = shape.elem_count();
        let query_stride = (self.num_heads * self.head_size) as i32;
        let rot_dim_i32 = self.rot_dim as i32;
        let head_size_i32 = self.head_size as i32;
        let num_heads_i32 = self.num_heads as i32;
        let num_kv_heads_i32 = 0i32;
        let key_stride = 0i32;

        let block_dim = std::cmp::min(self.num_heads * self.rot_dim / 2, 512) as u32;
        let block_dim = std::cmp::max(block_dim, 1);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase 11.2.C: pick the kernel variant by output dtype so the
        // EXL3 F16 path uses the F16 RoPE kernel instead of falling
        // through to the slow candle path (which allocates positional
        // index_select tensors fresh per layer — the source of the
        // ILLEGAL_ADDRESS at capture replay).
        let kernel_name_bf16 = if self.is_neox {
            "rotary_embedding_neox_bf16"
        } else {
            "rotary_embedding_gptj_bf16"
        };
        let kernel_name_fp16 = if self.is_neox {
            "rotary_embedding_neox_fp16"
        } else {
            // No GPT-J fp16 kernel — fall back to neox-style for any
            // F16 user (Llama-EXL3 is neox-style).
            "rotary_embedding_neox_fp16"
        };

        let (pos_storage, _) = self.positions.storage_and_layout();
        let pos_storage = match &*pos_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope_inplace: positions must be on CUDA"),
        };

        let (cache_storage, _) = self.cos_sin_cache.storage_and_layout();
        let cache_storage = match &*cache_storage {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("rope_inplace: cos_sin_cache must be on CUDA"),
        };
        let cache_slice = match &cache_storage.slice {
            CudaStorageSlice::F32(s) => s,
            _ => candle_core::bail!("rope_inplace: cos_sin_cache must be f32"),
        };

        match (&mut out_storage.slice, &in_storage.slice) {
            (CudaStorageSlice::BF16(out_slice), CudaStorageSlice::BF16(in_slice)) => {
                // Copy input into receiver — kernel modifies receiver in place.
                let in_view = in_slice.slice(0..total_elems);
                let mut out_view = out_slice.slice_mut(0..total_elems);
                dev.memcpy_dtod(&in_view, &mut out_view)?;

                let func = dev.get_or_load_custom_func(kernel_name_bf16, "rope", ROPE_PTX)?;

                let mut builder = func.builder();
                if let CudaStorageSlice::U32(s) = &pos_storage.slice {
                    builder.arg(s);
                } else {
                    candle_core::bail!(
                        "rope_inplace: positions must be U32 (got {:?})",
                        self.positions.dtype()
                    );
                }
                builder.arg(out_slice);
                builder.arg(&0u64); // key = nullptr
                builder.arg(cache_slice);
                builder.arg(&rot_dim_i32);
                builder.arg(&query_stride);
                builder.arg(&key_stride);
                builder.arg(&head_size_i32);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);

                // SAFETY: kernel launch with validated params; out_slice exclusively
                // held via &mut out_storage and just overwritten by memcpy_dtod.
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope_inplace launch: {e}")))?;
            }
            (CudaStorageSlice::F16(out_slice), CudaStorageSlice::F16(in_slice)) => {
                // Phase 11.2.C: F16 RoPE for the EXL3 path.
                let in_view = in_slice.slice(0..total_elems);
                let mut out_view = out_slice.slice_mut(0..total_elems);
                dev.memcpy_dtod(&in_view, &mut out_view)?;

                let func = dev.get_or_load_custom_func(kernel_name_fp16, "rope", ROPE_PTX)?;

                let mut builder = func.builder();
                if let CudaStorageSlice::U32(s) = &pos_storage.slice {
                    builder.arg(s);
                } else {
                    candle_core::bail!(
                        "rope_inplace: positions must be U32 (got {:?})",
                        self.positions.dtype()
                    );
                }
                builder.arg(out_slice);
                builder.arg(&0u64);
                builder.arg(cache_slice);
                builder.arg(&rot_dim_i32);
                builder.arg(&query_stride);
                builder.arg(&key_stride);
                builder.arg(&head_size_i32);
                builder.arg(&num_heads_i32);
                builder.arg(&num_kv_heads_i32);

                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("rope_inplace launch: {e}")))?;
            }
            _ => candle_core::bail!("rope_inplace: only bf16/f16 supported"),
        }
        Ok(())
    }
}

/// Pool-backed RoPE for the decode hot path. Wraps Q and K each through
/// the pool — output tensors live at stable device addresses so the
/// captured CUDA graph's recorded memcpy_dtod + rotary_embedding kernel
/// reads/writes consistent pointers across replays.
///
/// Falls back to the non-pooled path when num_tokens exceeds the
/// decode-shape budget (prefill etc.) — same rationale as
/// `rms_norm_cuda_pooled`.
#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn rotary_embedding_cuda_pooled(
    q: &Tensor,
    k: &Tensor,
    positions: &Tensor,
    cos_sin_cache: &Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    use crate::engine::output_pool::OutputPool;
    crate::engine::cr_trace::mark_op("rotary_embedding_cuda_pooled");

    /// Same threshold as `rms_norm_cuda_pooled` (decode-shape budget).
    const POOL_MAX_NUM_TOKENS: usize = 64;

    let num_tokens = positions.dim(0)?;
    if num_tokens == 0 {
        return Ok((q.clone(), k.clone()));
    }

    if num_tokens > POOL_MAX_NUM_TOKENS {
        return rotary_embedding_cuda(
            q,
            k,
            positions,
            cos_sin_cache,
            rot_dim,
            head_size,
            num_heads,
            num_kv_heads,
            is_neox,
        );
    }

    // Reshape to flat [num_tokens, n_heads * head_size] like the non-pooled path.
    let q_flat = q
        .reshape((num_tokens, num_heads * head_size))?
        .contiguous()?;
    let k_flat = k
        .reshape((num_tokens, num_kv_heads * head_size))?
        .contiguous()?;

    let q_shape = q_flat.dims().to_vec();
    let k_shape = k_flat.dims().to_vec();
    let dtype = q_flat.dtype();
    let device = q_flat.device();

    let q_out = OutputPool::global().reserve(&q_shape, dtype, device)?;
    let q_op = RopeInplaceOp {
        positions,
        cos_sin_cache,
        rot_dim,
        head_size,
        num_heads,
        is_neox,
    };
    q_out.inplace_op2(&q_flat, &q_op)?;

    let k_out = OutputPool::global().reserve(&k_shape, dtype, device)?;
    let k_op = RopeInplaceOp {
        positions,
        cos_sin_cache,
        rot_dim,
        head_size,
        num_heads: num_kv_heads,
        is_neox,
    };
    k_out.inplace_op2(&k_flat, &k_op)?;

    let q_out = q_out.reshape(q.shape())?;
    let k_out = k_out.reshape(k.shape())?;
    Ok((q_out, k_out))
}

// ============================================================================
// Type-Safe PooledTensor Wrappers (Phase TS.2)
// ============================================================================
//
// These siblings of the *_pooled functions above accept `&PooledTensor`
// inputs and return `PooledTensor` outputs, propagating the pool-backed
// storage invariant through the call chain. Captured-eligible decode
// forwards (QuantizedLlama / Qwen3) use these so the compiler rejects
// fresh-alloc regressions.
//
// Each function internally calls its legacy `_pooled` sibling — the
// legacy functions already reserve outputs from `OutputPool::global()`,
// so wrapping their result in `PooledTensor::from_pool_unchecked` is
// safe by construction. Some functions additionally tighten the typed
// version (e.g., refusing dtype/device fallbacks that the legacy version
// silently routes through candle).

use crate::engine::output_pool::PooledTensor;

/// Type-safe wrapper for [`rms_norm_cuda_pooled`].
#[cfg(feature = "cuda-layernorm")]
pub fn rms_norm_cuda_pooled_typed(
    input: &PooledTensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<PooledTensor> {
    let out = rms_norm_cuda_pooled(input.as_tensor(), weight, epsilon)?;
    // SAFETY: `rms_norm_cuda_pooled` reserves output via OutputPool::global().
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`silu_and_mul_separate_pooled`].
#[cfg(feature = "cuda-fused-activations")]
pub fn silu_and_mul_separate_pooled_typed(
    gate: &PooledTensor,
    up: &PooledTensor,
) -> Result<PooledTensor> {
    let out = silu_and_mul_separate_pooled(gate.as_tensor(), up.as_tensor())?;
    // SAFETY: legacy reserves output from pool when dtype/device match;
    // for the F32/CPU fallback path it returns a fresh candle tensor.
    // We bail above on those to keep the invariant.
    if !matches!(
        out.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16
    ) {
        candle_core::bail!(
            "silu_and_mul_separate_pooled_typed: legacy returned non-pool dtype {:?}",
            out.dtype()
        );
    }
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`half_add_pooled`].
#[cfg(feature = "cuda-fused-activations")]
pub fn half_add_pooled_typed(a: &PooledTensor, b: &PooledTensor) -> Result<PooledTensor> {
    let out = half_add_pooled(a.as_tensor(), b.as_tensor())?;
    if !matches!(
        out.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F16
    ) {
        candle_core::bail!(
            "half_add_pooled_typed: legacy returned non-pool dtype {:?}",
            out.dtype()
        );
    }
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`half_matmul_pooled`].
pub fn half_matmul_pooled_typed(input: &PooledTensor, weight: &Tensor) -> Result<PooledTensor> {
    let in_dt = input.dtype();
    if in_dt != weight.dtype()
        || !matches!(in_dt, candle_core::DType::BF16 | candle_core::DType::F16)
        || !input.device().is_cuda()
    {
        // Typed version refuses fallbacks that would return a fresh
        // candle Tensor (line 2463 in legacy). Caller must ensure
        // dtypes match and tensor is on CUDA.
        candle_core::bail!(
            "half_matmul_pooled_typed: requires CUDA BF16/F16 (got {:?} on {:?})",
            in_dt,
            input.device()
        );
    }
    let out = half_matmul_pooled(input.as_tensor(), weight)?;
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`embedding_pooled`].
#[cfg(feature = "cuda-fused-activations")]
pub fn embedding_pooled_typed(input_ids: &PooledTensor, weight: &Tensor) -> Result<PooledTensor> {
    let out = embedding_pooled(input_ids.as_tensor(), weight)?;
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

/// Type-safe wrapper for [`rotary_embedding_cuda_pooled`]. Q and K are
/// pool-backed; positions and cos_sin_cache come from stable sources
/// (build_decode_batch_shared / model layer state) and remain `&Tensor`.
#[allow(clippy::too_many_arguments)]
pub fn rotary_embedding_cuda_pooled_typed(
    q: &PooledTensor,
    k: &PooledTensor,
    positions: &Tensor,
    cos_sin_cache: &Tensor,
    rot_dim: usize,
    head_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    is_neox: bool,
) -> Result<(PooledTensor, PooledTensor)> {
    let (q_out, k_out) = rotary_embedding_cuda_pooled(
        q.as_tensor(),
        k.as_tensor(),
        positions,
        cos_sin_cache,
        rot_dim,
        head_size,
        num_heads,
        num_kv_heads,
        is_neox,
    )?;
    // SAFETY: rotary_embedding_cuda_pooled reserves both outputs from
    // OutputPool::global() when num_tokens ≤ 64. Above 64 (prefill), it
    // falls through to rotary_embedding_cuda which returns fresh tensors
    // — captured forwards never enter that branch (batch ≤ 16 cap).
    // We could detect this by checking shape vs pool budget; for now
    // trust the captured-path contract.
    Ok((
        unsafe { PooledTensor::from_pool_unchecked(q_out) },
        unsafe { PooledTensor::from_pool_unchecked(k_out) },
    ))
}

/// Type-safe wrapper for [`paged_attention_v2_cuda_pooled`].
///
/// `q` is pool-backed; `k_cache`/`v_cache` are CacheEngine-owned (stable
/// for engine lifetime); `block_tables`/`seq_lens` are from
/// `build_decode_batch_shared` (TS.3 will tighten these to `&PooledTensor`).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_v2_cuda_pooled_typed(
    q: &PooledTensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    seq_lens: &Tensor,
    scale: f32,
    num_heads: usize,
    num_kv_heads: usize,
    max_blocks_per_seq: usize,
    max_seq_len: usize,
    worst_case_max_seq_len: usize,
    head_dim: usize,
    block_size: usize,
    partition_size: usize,
) -> Result<PooledTensor> {
    let out = paged_attention_v2_cuda_pooled(
        q.as_tensor(),
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        num_kv_heads,
        max_blocks_per_seq,
        max_seq_len,
        worst_case_max_seq_len,
        head_dim,
        block_size,
        partition_size,
    )?;
    Ok(unsafe { PooledTensor::from_pool_unchecked(out) })
}

// ============================================================================
// GELU/GeGLU Activation CUDA Kernels
// ============================================================================

#[cfg(feature = "cuda-fused-activations")]
const ACTIVATIONS_PTX: &str = include_str!("../kernels/activations.ptx");

/// GELU activation variant.
#[cfg(feature = "cuda-fused-activations")]
#[derive(Debug, Clone, Copy)]
pub enum GeluVariant {
    /// Exact GELU using erf
    Exact,
    /// Tanh approximation (faster)
    Tanh,
}

/// Fused gated activation: output = act(gate) * up
///
/// Input is `[..., 2 * d]` where first half is gate and second half is up.
/// Output is `[..., d]`.
///
/// # Arguments
/// - `input`: Concatenated gate+up tensor `[..., 2 * d]`
/// - `variant`: Which GELU variant to use
///
/// # Returns
/// Activated tensor `[..., d]`
#[cfg(feature = "cuda-fused-activations")]
pub struct GeluAndMulOp {
    variant: GeluVariant,
}

#[cfg(feature = "cuda-fused-activations")]
impl CustomOp1 for GeluAndMulOp {
    fn name(&self) -> &'static str {
        "gelu_and_mul"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let shape = layout.shape();
        let dims = shape.dims();
        if dims.is_empty() {
            candle_core::bail!("gelu_and_mul: input must have at least 1 dimension");
        }
        let last_dim = dims[dims.len() - 1];
        if !last_dim.is_multiple_of(2) {
            candle_core::bail!("gelu_and_mul: last dimension must be even, got {last_dim}");
        }
        let d = last_dim / 2;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = d;
        let out_shape = Shape::from_dims(&out_dims);

        match storage {
            CpuStorage::F32(data) => {
                let offsets = layout.contiguous_offsets();
                let (start, _end) = offsets.ok_or_else(|| {
                    candle_core::Error::Msg("gelu_and_mul: input must be contiguous".to_string())
                })?;
                let data = &data[start..];
                let mut result = Vec::with_capacity(num_tokens * d);
                for t in 0..num_tokens {
                    let gate = &data[t * last_dim..t * last_dim + d];
                    let up = &data[t * last_dim + d..t * last_dim + last_dim];
                    for i in 0..d {
                        let g = gate[i];
                        let u = up[i];
                        let act = match self.variant {
                            GeluVariant::Exact => {
                                g * 0.5 * (1.0 + libm::erff(g * std::f32::consts::FRAC_1_SQRT_2))
                            }
                            GeluVariant::Tanh => {
                                let beta = (2.0f32 / std::f32::consts::PI).sqrt();
                                let inner = beta * (g + 0.044715 * g * g * g);
                                0.5 * g * (1.0 + inner.tanh())
                            }
                        };
                        result.push(act * u);
                    }
                }
                Ok((CpuStorage::F32(result), out_shape))
            }
            _ => candle_core::bail!("gelu_and_mul: CPU fallback only supports F32"),
        }
    }

    #[cfg(feature = "cuda-kernels")]
    fn cuda_fwd(
        &self,
        input_storage: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};
        use half::{bf16, f16};

        let dev = &input_storage.device;
        let shape = input_layout.shape();
        let dims = shape.dims();

        let last_dim = dims[dims.len() - 1];
        let d = last_dim / 2;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        if num_tokens == 0 {
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = d;
            return Ok((
                CudaStorage {
                    slice: clone_cuda_storage_slice(&input_storage.slice),
                    device: dev.clone(),
                },
                Shape::from_dims(&out_dims),
            ));
        }

        let d_i32 = d as i32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (std::cmp::min(d as u32, 1024), 1, 1),
            shared_mem_bytes: 0,
        };

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = d;
        let out_shape = Shape::from_dims(&out_dims);

        let suffix = match self.variant {
            GeluVariant::Exact => "gelu_and_mul",
            GeluVariant::Tanh => "gelu_tanh_and_mul",
        };

        match &input_storage.slice {
            CudaStorageSlice::BF16(input_slice) => {
                // SAFETY: gelu/silu kernel writes every output element.
                let output_slice = unsafe { dev.alloc::<bf16>(num_tokens * d) }?;
                let func_name = format!("{suffix}_bf16");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::BF16(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            CudaStorageSlice::F16(input_slice) => {
                let output_slice = dev.alloc_zeros::<f16>(num_tokens * d)?;
                let func_name = format!("{suffix}_fp16");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F16(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            CudaStorageSlice::F32(input_slice) => {
                let output_slice = dev.alloc_zeros::<f32>(num_tokens * d)?;
                let func_name = format!("{suffix}_f32");
                let func =
                    dev.get_or_load_custom_func(&func_name, "activations", ACTIVATIONS_PTX)?;

                let mut builder = func.builder();
                builder.arg(&output_slice);
                builder.arg(input_slice);
                builder.arg(&d_i32);

                // SAFETY: kernel launch with validated parameters
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gelu_and_mul launch: {e}")))?;

                Ok((
                    CudaStorage {
                        slice: CudaStorageSlice::F32(output_slice),
                        device: dev.clone(),
                    },
                    out_shape,
                ))
            }
            _ => candle_core::bail!("gelu_and_mul: unsupported dtype"),
        }
    }
}

/// Fused GELU-gated activation: output = gelu(gate) * up
///
/// Input is concatenated `[gate, up]` along the last dimension.
///
/// # Arguments
/// - `input`: Tensor `[..., 2 * d]` where first half is gate, second is up
/// - `variant`: GELU variant (exact erf or tanh approximation)
///
/// # Returns
/// Tensor `[..., d]`
#[cfg(feature = "cuda-fused-activations")]
pub fn gelu_and_mul(input: &Tensor, variant: GeluVariant) -> Result<Tensor> {
    let input = input.contiguous()?;
    input.apply_op1_no_bwd(&GeluAndMulOp { variant })
}

/// Check if CUDA GELU activation kernels are available.
#[cfg(feature = "cuda-fused-activations")]
pub fn gelu_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-fused-activations"))]
pub fn gelu_cuda_available(_tensor: &Tensor) -> bool {
    false
}

// ============================================================================
// reshape_and_cache CUDA Kernels
// ============================================================================

#[cfg(feature = "cuda-kernels")]
const CACHE_OPS_PTX: &str = include_str!("../kernels/cache_ops.ptx");

/// Cache layout for CUDA kernel dispatch.
#[cfg(feature = "cuda-kernels")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaCacheLayout {
    /// `[num_blocks, block_size, num_kv_heads, head_dim]`
    NHD,
    /// `[num_blocks, num_kv_heads, block_size, head_dim]`
    HND,
}

/// Write K/V from model output into paged KV cache using a CUDA kernel.
///
/// This replaces the Candle scatter path with a direct kernel launch,
/// avoiding intermediate tensor allocations for index expansion.
///
/// # Arguments
/// - `key`: Input key tensor `[num_tokens, num_kv_heads, head_dim]`, contiguous
/// - `value`: Input value tensor `[num_tokens, num_kv_heads, head_dim]`, contiguous
/// - `key_cache`: KV cache key tensor (modified in-place)
/// - `value_cache`: KV cache value tensor (modified in-place)
/// - `slot_mapping`: Physical slot per token `[num_tokens]` (i32, on CUDA)
/// - `num_kv_heads`: Number of KV heads
/// - `head_dim`: Head dimension
/// - `block_size`: Cache block size
/// - `layout`: Cache layout (NHD or HND)
#[cfg(feature = "cuda-kernels")]
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_cuda(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    layout: CudaCacheLayout,
) -> Result<()> {
    use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

    let num_tokens = key.dim(0)?;
    if num_tokens == 0 {
        return Ok(());
    }

    let (key_guard, _) = key.storage_and_layout();
    let (value_guard, _) = value.storage_and_layout();
    let (kc_guard, _) = key_cache.storage_and_layout();
    let (vc_guard, _) = value_cache.storage_and_layout();
    let (sm_guard, _) = slot_mapping.storage_and_layout();

    let dev = match &*key_guard {
        Storage::Cuda(cs) => &cs.device,
        _ => candle_core::bail!("reshape_and_cache: key must be on CUDA"),
    };

    let sm_slice = match &*sm_guard {
        Storage::Cuda(cs) => match &cs.slice {
            CudaStorageSlice::I64(_) => {
                candle_core::bail!("reshape_and_cache: slot_mapping must be U32, got I64")
            }
            CudaStorageSlice::U32(s) => s,
            _ => candle_core::bail!("reshape_and_cache: slot_mapping must be U32"),
        },
        _ => candle_core::bail!("reshape_and_cache: slot_mapping must be on CUDA"),
    };

    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let block_size_i32 = block_size as i32;

    // Select kernel name and grid based on layout and dtype
    match layout {
        CudaCacheLayout::NHD => {
            let kv_stride = num_kv_heads * head_dim;
            let block_dim = std::cmp::min(kv_stride, 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            match &*key_guard {
                Storage::Cuda(key_cs) => match (&key_cs.slice, &*value_guard) {
                    (CudaStorageSlice::BF16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::BF16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_bf16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        // SAFETY: all pointers validated, contiguous, same device
                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache NHD launch: {e}"))
                        })?;
                    }
                    (CudaStorageSlice::F16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::F16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_fp16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache NHD launch: {e}"))
                        })?;
                    }
                    _ => candle_core::bail!(
                        "reshape_and_cache: unsupported dtype, expected BF16 or F16"
                    ),
                },
                _ => candle_core::bail!("reshape_and_cache: key not on CUDA"),
            }
        }
        CudaCacheLayout::HND => {
            let block_dim = std::cmp::min(head_dim, 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, num_kv_heads as u32, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            match &*key_guard {
                Storage::Cuda(key_cs) => match (&key_cs.slice, &*value_guard) {
                    (CudaStorageSlice::BF16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::BF16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::BF16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_hnd_bf16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        // SAFETY: all pointers validated, contiguous, same device
                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache HND launch: {e}"))
                        })?;
                    }
                    (CudaStorageSlice::F16(k_slice), Storage::Cuda(v_cs)) => {
                        let v_slice = match &v_cs.slice {
                            CudaStorageSlice::F16(s) => s,
                            _ => candle_core::bail!("reshape_and_cache: value dtype mismatch"),
                        };
                        let kc_slice = match &*kc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: key_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: key_cache not on CUDA"),
                        };
                        let vc_slice = match &*vc_guard {
                            Storage::Cuda(cs) => match &cs.slice {
                                CudaStorageSlice::F16(s) => s,
                                _ => candle_core::bail!(
                                    "reshape_and_cache: value_cache dtype mismatch"
                                ),
                            },
                            _ => candle_core::bail!("reshape_and_cache: value_cache not on CUDA"),
                        };

                        let func = dev.get_or_load_custom_func(
                            "reshape_and_cache_hnd_fp16",
                            "cache_ops",
                            CACHE_OPS_PTX,
                        )?;

                        let mut builder = func.builder();
                        builder.arg(k_slice);
                        builder.arg(v_slice);
                        builder.arg(kc_slice);
                        builder.arg(vc_slice);
                        builder.arg(sm_slice);
                        builder.arg(&num_kv_heads_i32);
                        builder.arg(&head_dim_i32);
                        builder.arg(&block_size_i32);

                        unsafe { builder.launch(cfg) }.map_err(|e| {
                            candle_core::Error::Msg(format!("reshape_and_cache HND launch: {e}"))
                        })?;
                    }
                    _ => candle_core::bail!(
                        "reshape_and_cache: unsupported dtype, expected BF16 or F16"
                    ),
                },
                _ => candle_core::bail!("reshape_and_cache: key not on CUDA"),
            }
        }
    }

    drop(key_guard);
    drop(value_guard);
    drop(kc_guard);
    drop(vc_guard);
    drop(sm_guard);

    Ok(())
}

/// Check if reshape_and_cache CUDA kernel is available.
#[cfg(feature = "cuda-kernels")]
pub fn reshape_and_cache_cuda_available(tensor: &Tensor) -> bool {
    tensor.device().is_cuda()
}

#[cfg(not(feature = "cuda-kernels"))]
pub fn reshape_and_cache_cuda_available(_tensor: &Tensor) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_silu_formula() {
        // Verify our SiLU implementation matches expected values
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let test_cases = [
            (0.0f32, 0.0f32),         // silu(0) = 0
            (1.0f32, 0.7310586f32),   // silu(1) ≈ 0.731
            (-1.0f32, -0.2689414f32), // silu(-1) ≈ -0.269
            (2.0f32, 1.7615942f32),   // silu(2) ≈ 1.762
        ];

        for (input, expected) in test_cases {
            let computed = input / (1.0 + (-input).exp());
            assert!(
                (computed - expected).abs() < 1e-5,
                "silu({}) = {}, expected {}",
                input,
                computed,
                expected
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_cpu_f32() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![1.0f32, 2.0, -1.0, 0.5], (2, 2), &device).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (2, 2), &device).unwrap();

        let result = fused_swiglu(&gate, &up).unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Expected: silu(gate) * up = silu(gate) * 1 = silu(gate)
        let expected: Vec<f32> = vec![1.0f32, 2.0, -1.0, 0.5]
            .iter()
            .map(|&g| g / (1.0 + (-g).exp()))
            .collect();

        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-5,
                "fused_swiglu mismatch: got {}, expected {}",
                r,
                e
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_cpu_bf16() {
        let device = Device::Cpu;
        let gate_f32 = vec![1.0f32, 2.0, -1.0, 0.5];
        let up_f32 = vec![2.0f32, 0.5, 1.0, 3.0];

        let gate = Tensor::from_vec(gate_f32.clone(), (2, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let up = Tensor::from_vec(up_f32.clone(), (2, 2), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let result = fused_swiglu(&gate, &up).unwrap();
        let result_f32 = result.to_dtype(DType::F32).unwrap();
        let result_data: Vec<f32> = result_f32.flatten_all().unwrap().to_vec1().unwrap();

        // Expected: silu(gate) * up
        let expected: Vec<f32> = gate_f32
            .iter()
            .zip(up_f32.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        for (r, e) in result_data.iter().zip(expected.iter()) {
            // BF16 has lower precision, use larger tolerance
            assert!(
                (r - e).abs() < 0.05,
                "fused_swiglu bf16 mismatch: got {}, expected {}",
                r,
                e
            );
        }
    }

    #[cfg(feature = "cuda-fused-activations")]
    #[test]
    fn test_fused_swiglu_shape_mismatch() {
        let device = Device::Cpu;
        let gate = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let up = Tensor::zeros((2, 8), DType::F32, &device).unwrap();

        let result = fused_swiglu(&gate, &up);
        assert!(result.is_err());
    }

    /// Paged-attention F16 vs BF16 parity: identical inputs cast to either
    /// dtype must produce outputs that match within F16 mantissa tolerance.
    /// Covers V1 (short seq), V2 (long seq, partitioned), and the auto path.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attention_f16_bf16_parity() {
        let Ok(dev) = Device::new_cuda(0) else { return };

        let num_seqs = 1usize;
        let num_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 128usize;
        let block_size = 16usize;
        let seq_len = 16usize; // one full block, fits V1
        let max_blocks_per_seq = 1usize;

        // Random-ish but deterministic input data.
        let q_f32: Vec<f32> = (0..num_seqs * num_heads * head_dim)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5)
            .collect();
        let k_f32: Vec<f32> = (0..max_blocks_per_seq * block_size * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..max_blocks_per_seq * block_size * num_kv_heads * head_dim)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_heads, head_dim), &dev).unwrap();
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let block_tables =
            Tensor::from_vec(vec![0u32], (num_seqs, max_blocks_per_seq), &dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, &dev).unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        // BF16 path
        let q_bf16 = q.to_dtype(DType::BF16).unwrap();
        let k_bf16 = k_cache.to_dtype(DType::BF16).unwrap();
        let v_bf16 = v_cache.to_dtype(DType::BF16).unwrap();
        let out_bf16 = paged_attention_cuda(
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &block_tables,
            &seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        // F16 path
        let q_f16 = q.to_dtype(DType::F16).unwrap();
        let k_f16 = k_cache.to_dtype(DType::F16).unwrap();
        let v_f16 = v_cache.to_dtype(DType::F16).unwrap();
        let out_f16 = paged_attention_cuda(
            &q_f16,
            &k_f16,
            &v_f16,
            &block_tables,
            &seq_lens,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

        let bf16_vec: Vec<f32> = out_bf16.flatten_all().unwrap().to_vec1().unwrap();
        let f16_vec: Vec<f32> = out_f16.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(bf16_vec.len(), f16_vec.len());

        // F16 has 10 mantissa bits, BF16 has 7 — different rounding paths.
        // Tolerance: 5e-2 absolute, matches values <0.5 in our synthetic input.
        let max_diff = bf16_vec
            .iter()
            .zip(f16_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 5e-2,
            "paged_attention v1 f16/bf16 max diff {max_diff} exceeds 5e-2"
        );
    }

    /// V2 split-K parity: long sequence forces multi-partition reduce path.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attention_v2_f16_bf16_parity() {
        let Ok(dev) = Device::new_cuda(0) else { return };

        let num_seqs = 1usize;
        let num_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 128usize;
        let block_size = 16usize;
        // > V2_SEQ_LEN_THRESHOLD (512) and > PARTITION_SIZE (512) → exercises reduce.
        let seq_len = 1024usize;
        let max_blocks_per_seq = seq_len / block_size; // 64 blocks

        let q_f32: Vec<f32> = (0..num_seqs * num_heads * head_dim)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5)
            .collect();
        let kv_elements = max_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_heads, head_dim), &dev).unwrap();
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap();
        let bt: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
        let block_tables = Tensor::from_vec(bt, (num_seqs, max_blocks_per_seq), &dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, &dev).unwrap();

        let scale = 1.0 / (head_dim as f32).sqrt();

        let run = |dtype: DType| {
            paged_attention_auto(
                &q.to_dtype(dtype).unwrap(),
                &k_cache.to_dtype(dtype).unwrap(),
                &v_cache.to_dtype(dtype).unwrap(),
                &block_tables,
                &seq_lens,
                scale,
                num_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };

        let bf16_vec = run(DType::BF16);
        let f16_vec = run(DType::F16);
        assert_eq!(bf16_vec.len(), f16_vec.len());

        let max_diff = bf16_vec
            .iter()
            .zip(f16_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // V2 sums across partitions accumulate more rounding drift than V1.
        assert!(
            max_diff < 1e-1,
            "paged_attention v2 f16/bf16 max diff {max_diff} exceeds 1e-1"
        );
    }

    // ─── Boundary parity matrix for V1 ↔ V2 (Step 2 of Phase 2 hardening) ───
    //
    // These tests exercise paths that the existing parity tests do not cover:
    //
    //  - seq_len at partition boundaries (seq_len ≈ k·partition_size ± 1) so
    //    the multi-partition log-sum-exp reduce is exercised on edge counts;
    //  - partition_size sweep (64, 128, 256, 512) at fixed seq_len so the new
    //    runtime-arg dispatch is verified against itself;
    //  - head_dim sweep (64, 96, 128, 256, 80) so both the warp K-pass
    //    (head_dim divisible by 32, ≤ 512) and the legacy block-reduce
    //    fallback (head_dim=80 → not divisible by 32) are walked;
    //  - GQA num_queries_per_kv ∈ {1, 4, 8} so MHA, Qwen3-style 4× sharing,
    //    and Llama-70B-style 8× sharing all run;
    //  - ALiBi enabled / disabled so the alibi_slope addition path is tested.
    //
    // The tolerance is tighter than the BF16↔F16 dtype-mismatch test above
    // (5e-3 absolute) since we are comparing two implementations of the same
    // math at the same dtype: V1 against V2-with-multi-partition-reduce.

    #[cfg(feature = "cuda-kernels")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn_v1_v2_parity_case_bf16(
        dev: &Device,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
        partition_size: usize,
        alibi_slopes: Option<&[f32]>,
    ) -> f32 {
        use crate::cuda_kernels::{
            paged_attention_auto_alibi, paged_attention_cuda, paged_attention_cuda_alibi,
            paged_attention_v2_cuda_with_partition_size,
        };

        // Deterministic synthetic inputs.
        let num_seqs = 1usize;
        let max_blocks_per_seq = seq_len.div_ceil(block_size);
        let q_f32: Vec<f32> = (0..num_seqs * num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.0123 + (head_dim as f32) * 0.001).sin() * 0.5)
            .collect();
        let kv_elements = max_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0271 + (seq_len as f32) * 0.001).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0411 + (head_dim as f32) * 0.002).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_q_heads, head_dim), dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let bt: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
        let block_tables = Tensor::from_vec(bt, (num_seqs, max_blocks_per_seq), dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, dev).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let alibi_tensor = alibi_slopes.map(|s| {
            assert_eq!(
                s.len(),
                num_q_heads,
                "alibi_slopes length must equal num_q_heads"
            );
            Tensor::from_vec(s.to_vec(), num_q_heads, dev).unwrap()
        });

        let v1 = match alibi_tensor.as_ref() {
            None => paged_attention_cuda(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &seq_lens,
                scale,
                num_q_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
            ),
            Some(a) => paged_attention_cuda_alibi(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &seq_lens,
                scale,
                num_q_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
                a,
            ),
        }
        .unwrap();

        // Suppress unused warning for partition_size in the ALiBi branch.
        let _ = partition_size;

        let v2 = match alibi_tensor.as_ref() {
            None => paged_attention_v2_cuda_with_partition_size(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &seq_lens,
                scale,
                num_q_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
                partition_size,
            ),
            // ALiBi V2 currently uses the default partition_size (the public
            // `paged_attention_auto_alibi` does not expose the override).
            // For seq_len > V2_SEQ_LEN_THRESHOLD this routes through the V2
            // ALiBi kernel, which is the path the parity test cares about.
            // A `_with_partition_size` overload for ALiBi can be added later
            // if/when the adaptive selector wants to tune ALiBi separately.
            Some(a) => paged_attention_auto_alibi(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &seq_lens,
                scale,
                num_q_heads,
                num_kv_heads,
                max_blocks_per_seq,
                seq_len,
                head_dim,
                block_size,
                a,
            ),
        }
        .unwrap();

        let v1_f: Vec<f32> = v1
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let v2_f: Vec<f32> = v2
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(v1_f.len(), v2_f.len());
        v1_f.iter()
            .zip(v2_f.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// V1↔V2 parity, seq_len × partition_size sweep with Qwen3-shape fixed
    /// (num_q_heads=32, num_kv_heads=8, head_dim=128, block_size=16).
    ///
    /// Walks every combination of seq_len ∈ {boundaries near 64, 128, 256,
    /// 512, 1024, 2048, 4096} and partition_size ∈ {64, 128, 256, 512}.
    /// Specifically pins the V2 parameter-size dispatch we just refactored.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_seq_len_x_partition_size_matrix() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let num_q_heads = 32;
        let num_kv_heads = 8; // Qwen3-style GQA, group=4
        let head_dim = 128;
        let block_size = 16;

        // Boundary-rich seq_len set: covers 1× / multi-partition transitions
        // for every partition_size we sweep.
        let seq_lens: &[usize] = &[
            65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 2049, 4097,
        ];
        let partition_sizes: &[usize] = &[64, 128, 256, 512];

        for &seq_len in seq_lens {
            for &part in partition_sizes {
                let max_diff = run_paged_attn_v1_v2_parity_case_bf16(
                    &dev,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    block_size,
                    seq_len,
                    part,
                    None,
                );
                assert!(
                    max_diff < 5e-3,
                    "V1↔V2 parity FAIL seq_len={seq_len} partition_size={part} \
                     max_diff={max_diff} (threshold 5e-3)"
                );
            }
        }
    }

    /// V1↔V2 parity across head_dim variants. Walks both warp K-pass
    /// (head_dim ∈ {64, 96, 128, 256}, divisible by 32) and the legacy
    /// block-reduce fallback (head_dim=80, not divisible by 32 → falls into
    /// the `else` branch we kept around exactly for cases like this).
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_head_dim_sweep() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let num_q_heads = 4;
        let num_kv_heads = 4; // MHA; varying GQA orthogonal to head_dim test
        let block_size = 16;
        let seq_len = 512;
        let partition_size = 128;

        for &head_dim in &[64usize, 80, 96, 128, 256] {
            let max_diff = run_paged_attn_v1_v2_parity_case_bf16(
                &dev,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                seq_len,
                partition_size,
                None,
            );
            assert!(
                max_diff < 5e-3,
                "V1↔V2 parity FAIL head_dim={head_dim} max_diff={max_diff} (threshold 5e-3); \
                 head_dim=80 exercises the legacy block-reduce fallback path"
            );
        }
    }

    /// V1↔V2 parity across GQA sharing factors (num_queries_per_kv).
    /// 1 = MHA, 4 = Qwen3-4B-AWQ, 8 = Llama-3-70B style.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_gqa_sweep() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let head_dim = 128;
        let block_size = 16;
        let seq_len = 512;
        let partition_size = 128;
        let num_q_heads = 8;
        for &num_q_per_kv in &[1usize, 4, 8] {
            assert!(num_q_heads % num_q_per_kv == 0);
            let num_kv_heads = num_q_heads / num_q_per_kv;
            let max_diff = run_paged_attn_v1_v2_parity_case_bf16(
                &dev,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_size,
                seq_len,
                partition_size,
                None,
            );
            assert!(
                max_diff < 5e-3,
                "V1↔V2 parity FAIL num_q_per_kv={num_q_per_kv} max_diff={max_diff}"
            );
        }
    }

    /// V1_alibi ↔ V2_alibi parity with deliberately-non-trivial slopes.
    /// Catches regressions in the ALiBi-bias term inside both kernels;
    /// the existing parity test does not exercise ALiBi at all.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    /// V2 with FP8 E4M3 KV cache must produce output close to V2 with
    /// the same (unquantized) K/V tensors at fp16/bf16. The acceptable
    /// error budget reflects FP8 E4M3's per-value precision (~12.5%
    /// max relative on any one byte, much less after dot-product RMS).
    /// scale=1.0 maps the [-448, 448] FP8 range directly so all test
    /// values (±0.5) survive without saturation.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_fp8e4m3_matches_auto() {
        use crate::kv_cache::quantization::{quantize_fp8, KVCacheDtype};

        let Ok(dev) = Device::new_cuda(0) else { return };
        let max_diff = run_paged_attn_v2_quantized_case::<half::bf16>(
            &dev,
            KVCacheDtype::Fp8E4m3,
            8,   // num_q_heads
            2,   // num_kv_heads (GQA group=4)
            128, // head_dim
            16,  // block_size
            128, // seq_len
            128, // partition_size
            1.0, // k_scale
            1.0, // v_scale
            |t, scale| quantize_fp8(t, scale),
            DType::BF16,
        );
        assert!(
            max_diff < 0.05,
            "paged_attn V2 FP8 E4M3 vs Auto: max_diff={max_diff} (threshold 0.05)"
        );
    }

    /// V2 with INT8 KV cache + properly calibrated scale must match the
    /// Auto baseline closely. INT8 with per-tensor scale = max_abs/127
    /// gives ~max_abs/127 absolute quantization step per byte; for our
    /// ±0.5 test fixture that is ~4e-3 per byte, RMS-aggregated across
    /// `seq_len` tokens.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_int8_matches_auto() {
        use crate::kv_cache::quantization::{quantize_int8, KVCacheDtype};

        let Ok(dev) = Device::new_cuda(0) else { return };
        let int8_scale = 0.5 / 127.0; // test data abs range capped at 0.5
        let max_diff = run_paged_attn_v2_quantized_case::<half::bf16>(
            &dev,
            KVCacheDtype::Int8,
            8,
            2,
            128,
            16,
            128,
            128,
            int8_scale,
            int8_scale,
            |t, scale| quantize_int8(t, scale),
            DType::BF16,
        );
        assert!(
            max_diff < 0.02,
            "paged_attn V2 INT8 vs Auto: max_diff={max_diff} (threshold 0.02)"
        );
    }

    /// Helper: run V2 with quantized KV cache and compare against the
    /// Auto-mode baseline computed on the same raw input. Returns
    /// `max(|out_quant - out_auto|)` element-wise.
    #[cfg(feature = "cuda-kernels")]
    #[allow(clippy::too_many_arguments)]
    fn run_paged_attn_v2_quantized_case<T>(
        dev: &Device,
        kv_cache_dtype: crate::kv_cache::quantization::KVCacheDtype,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
        partition_size: usize,
        k_scale_val: f32,
        v_scale_val: f32,
        quantize: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
        compute_dtype: DType,
    ) -> f32
    where
        T: PagedAttnDtype + 'static,
    {
        use crate::cuda_kernels::{
            paged_attention_v2_cuda_with_kv_dtype, paged_attention_v2_cuda_with_partition_size,
        };

        let num_seqs = 1usize;
        let max_blocks_per_seq = seq_len.div_ceil(block_size);
        let q_f32: Vec<f32> = (0..num_seqs * num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5)
            .collect();
        let kv_elements = max_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..kv_elements)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32.clone(), (num_seqs, num_q_heads, head_dim), dev)
            .unwrap()
            .to_dtype(compute_dtype)
            .unwrap();
        let k_full = Tensor::from_vec(
            k_f32.clone(),
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            dev,
        )
        .unwrap()
        .to_dtype(compute_dtype)
        .unwrap();
        let v_full = Tensor::from_vec(
            v_f32.clone(),
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            dev,
        )
        .unwrap()
        .to_dtype(compute_dtype)
        .unwrap();

        let bt: Vec<u32> = (0..max_blocks_per_seq as u32).collect();
        let block_tables = Tensor::from_vec(bt, (num_seqs, max_blocks_per_seq), dev).unwrap();
        let seq_lens = Tensor::from_vec(vec![seq_len as u32], num_seqs, dev).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Baseline: Auto KV cache.
        let auto_out = paged_attention_v2_cuda_with_partition_size(
            &q,
            &k_full,
            &v_full,
            &block_tables,
            &seq_lens,
            scale,
            num_q_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
            partition_size,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        // Quantize K/V to U8 storage; the per-tensor scale is supplied
        // externally — we don't use CacheEngine's write path because
        // we want to compare the raw paged_attn kernel against the
        // exact same input data, without any reshape/scatter noise.
        let k_scale_t = Tensor::from_vec(vec![k_scale_val], 1, dev).unwrap();
        let v_scale_t = Tensor::from_vec(vec![v_scale_val], 1, dev).unwrap();
        let k_quant = quantize(&k_full, &k_scale_t).unwrap();
        let v_quant = quantize(&v_full, &v_scale_t).unwrap();

        let quant_out = paged_attention_v2_cuda_with_kv_dtype(
            &q,
            &k_quant,
            &v_quant,
            &block_tables,
            &seq_lens,
            scale,
            num_q_heads,
            num_kv_heads,
            max_blocks_per_seq,
            seq_len,
            head_dim,
            block_size,
            partition_size,
            kv_cache_dtype,
            Some(&k_scale_t),
            Some(&v_scale_t),
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        auto_out
            .iter()
            .zip(quant_out.iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
    }

    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_alibi_parity() {
        let Ok(dev) = Device::new_cuda(0) else { return };
        let num_q_heads = 4;
        let num_kv_heads = 4;
        let head_dim = 128;
        let block_size = 16;
        let seq_len = 512;
        let partition_size = 128;

        // BLOOM-style halved-decay slopes; varied per-head so each warp
        // sees a different bias and any cross-head bug shows up.
        let slopes: Vec<f32> = (0..num_q_heads)
            .map(|h| 0.5_f32.powi(h as i32 + 1))
            .collect();

        let max_diff = run_paged_attn_v1_v2_parity_case_bf16(
            &dev,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            seq_len,
            partition_size,
            Some(&slopes),
        );
        assert!(
            max_diff < 5e-3,
            "V1↔V2 ALiBi parity max_diff={max_diff} (threshold 5e-3)"
        );
    }

    /// Regression guard for the active-partitions grid-Z optimisation in
    /// `PagedAttnV2InplaceOp::cuda_fwd`. Before that change, the pooled
    /// kernel launched `worst_case_max_seq_len / partition_size` blocks
    /// in grid Z; now it launches only `actual_max_seq_len /
    /// partition_size`. Both should produce **bit-identical** output for
    /// every (head, seq) cell — the un-launched partitions were
    /// guaranteed-empty (kernel does `partition_start_token >= seq_len ?
    /// return : ...`) and the reduce kernel only reads
    /// `ceil(seq_len / partition_size)` partitions anyway. Comparing two
    /// pooled forwards at worst_case_a vs worst_case_b confirms this
    /// guarantee survives across regenerations of the kernel or the
    /// pool's zero-init contract.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    fn test_paged_attn_v2_pooled_worst_case_invariance() {
        use crate::cuda_kernels::paged_attention_v2_cuda_pooled;
        let Ok(dev) = Device::new_cuda(0) else { return };

        // Use a Qwen3-ish shape but kept tiny so the test fits in
        // pool slots reserved for other tests. Pool capacity is sized
        // per-process; a >POOL_MAX_NUM_TOKENS request would reroute.
        let num_q_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 64;
        let block_size = 16;
        let num_seqs = 2;
        let actual_seq_len: usize = 256;
        let max_blocks_per_seq = actual_seq_len.div_ceil(block_size);

        let q_f32: Vec<f32> = (0..num_seqs * num_q_heads * head_dim)
            .map(|i| (i as f32 * 0.0173).sin() * 0.5)
            .collect();
        let kv_elems = max_blocks_per_seq * block_size * num_kv_heads * head_dim;
        let k_f32: Vec<f32> = (0..kv_elems)
            .map(|i| (i as f32 * 0.0271).cos() * 0.5)
            .collect();
        let v_f32: Vec<f32> = (0..kv_elems)
            .map(|i| (i as f32 * 0.0411).sin() * 0.5)
            .collect();

        let q = Tensor::from_vec(q_f32, (num_seqs, num_q_heads, head_dim), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // Each sequence reads its own copy of the kv blocks. Re-using
        // the same block ids across sequences is fine — they read
        // identical data, output diverges via Q only.
        let k_cache = Tensor::from_vec(
            k_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let v_cache = Tensor::from_vec(
            v_f32,
            (max_blocks_per_seq, block_size, num_kv_heads, head_dim),
            &dev,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

        let bt: Vec<u32> = (0..(num_seqs * max_blocks_per_seq) as u32)
            .map(|i| (i as usize % max_blocks_per_seq) as u32)
            .collect();
        let block_tables = Tensor::from_vec(bt, (num_seqs, max_blocks_per_seq), &dev).unwrap();
        let seq_lens =
            Tensor::from_vec(vec![actual_seq_len as u32; num_seqs], num_seqs, &dev).unwrap();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // partition_size pinned so the worst-case sweep below produces
        // a meaningful active/inactive ratio (1 active vs 64 worst).
        let partition_size = 64;

        let call = |worst_case: usize| {
            paged_attention_v2_cuda_pooled(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &seq_lens,
                scale,
                num_q_heads,
                num_kv_heads,
                max_blocks_per_seq,
                actual_seq_len,
                worst_case,
                head_dim,
                block_size,
                partition_size,
            )
            .expect("paged_attention_v2_cuda_pooled")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };

        let tight = call(actual_seq_len);
        let inflated = call(actual_seq_len * 64); // 64× headroom
        assert_eq!(
            tight.len(),
            inflated.len(),
            "shape mismatch between worst_case values"
        );
        let max_diff = tight
            .iter()
            .zip(inflated.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // The kernel does identical math for every active partition; any
        // diff would mean the grid-Z change is exercising a previously
        // un-tested code path. Bit-exact tolerance accounts for
        // BF16↔F32 round-trip noise on tensor.to_dtype.
        assert!(
            max_diff < 1e-6,
            "worst_case-invariance violated: max_diff={max_diff}"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase CR.8: capture-replay correctness mini-repro
    //
    // Apples-to-apples pool-slot diff (Phase CR.6) localised the wrong-
    // tokens regression to two specific kernels under cuGraph replay:
    // RoPE K (block_dim=256) and paged_attention_v2 main+reduce. These
    // tests isolate each kernel in a self-contained capture-replay
    // harness so the divergence reproduces without a full server. Once
    // a test fails deterministically here, `compute-sanitizer` can be
    // run against the test binary in seconds (test footprint ~300 MB)
    // — orders of magnitude lighter than instrumenting the full
    // production server (which OOMs WSL2 on a 16 GB Windows host).
    //
    // Both tests are `#[ignore]` (capture window tests are serialised
    // by `--test-threads=1` to avoid sharing the GPU's default cuMemPool
    // across captures).
    // ─────────────────────────────────────────────────────────────────

    /// Helper: build cos/sin cache for a synthetic NeoX RoPE in the
    /// format `rotary_embedding_neox_fp16` expects (interleaved
    /// `[cos_0..cos_{half-1}, sin_0..sin_{half-1}]` per position).
    #[cfg(feature = "cuda-kernels")]
    fn build_cos_sin_cache(
        rot_dim: usize,
        max_pos: usize,
        rope_theta: f32,
        dev: &Device,
    ) -> Tensor {
        let half = rot_dim / 2;
        let mut data = Vec::<f32>::with_capacity(max_pos * rot_dim);
        for pos in 0..max_pos {
            // cos block
            for i in 0..half {
                let inv_freq = 1.0_f32 / rope_theta.powf((2 * i) as f32 / rot_dim as f32);
                data.push((pos as f32 * inv_freq).cos());
            }
            // sin block
            for i in 0..half {
                let inv_freq = 1.0_f32 / rope_theta.powf((2 * i) as f32 / rot_dim as f32);
                data.push((pos as f32 * inv_freq).sin());
            }
        }
        Tensor::from_vec(data, (max_pos, rot_dim), dev).unwrap()
    }

    /// Helper: launch RoPE for Q+K via the pooled wrapper, return
    /// (q_out, k_out) Tensors that share storage with pool slots.
    #[cfg(feature = "cuda-kernels")]
    #[allow(clippy::too_many_arguments)]
    fn run_rope_pooled(
        q_in: &Tensor,
        k_in: &Tensor,
        positions: &Tensor,
        cos_sin: &Tensor,
        rot_dim: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> (Tensor, Tensor) {
        rotary_embedding_cuda_pooled(
            q_in,
            k_in,
            positions,
            cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
            /* is_neox = */ true,
        )
        .expect("rotary_embedding_cuda_pooled")
    }

    /// Mini-repro: does the captured RoPE K kernel produce the same
    /// output at replay as the eager kernel given the same inputs?
    ///
    /// Pool-slot diff (Phase CR.6) showed F16 [1, 512] slot 2 — the
    /// RoPE K output — diverges between eager+V2-pool and capture+V2-pool
    /// forward 1, while RoPE Q (F16 [1, 2048] slot 1) matches. Same
    /// kernel template (`rotary_embedding_neox_fp16`), different
    /// launch config (Q: block_dim=512 num_heads=32; K: block_dim=256
    /// num_kv_heads=8). This test isolates the K invocation at the K's
    /// production shape from Llama-3.2-1B.
    ///
    /// Pass condition: replay produces byte-identical output to eager.
    /// Fail condition: divergence reproduces → run compute-sanitizer
    /// against this test binary to identify root cause.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn rope_k_capture_replay_matches_eager() {
        use crate::engine::output_pool::OutputPool;
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
            CUstreamCaptureMode,
        };

        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        // Llama-3.2-1B decode K shape (the divergent one).
        let num_tokens = 1usize;
        let num_heads = 32usize; // Q heads
        let num_kv_heads = 8usize; // K heads
        let head_dim = 64usize;
        let rot_dim = 64usize;
        let rope_theta = 500000.0_f32; // Llama-3.2

        // Synthetic inputs (deterministic, non-trivial).
        let mut q_data = Vec::<half::f16>::with_capacity(num_tokens * num_heads * head_dim);
        for i in 0..(num_tokens * num_heads * head_dim) {
            let v = ((i as f32 * 0.0031).sin() * 0.5) as f32;
            q_data.push(half::f16::from_f32(v));
        }
        let mut k_data = Vec::<half::f16>::with_capacity(num_tokens * num_kv_heads * head_dim);
        for i in 0..(num_tokens * num_kv_heads * head_dim) {
            let v = ((i as f32 * 0.0027).cos() * 0.5) as f32;
            k_data.push(half::f16::from_f32(v));
        }
        let cos_sin = build_cos_sin_cache(rot_dim, 64, rope_theta, &dev_t);

        // Positions pool-backed at stable address (matches build_decode_batch_shared).
        let positions_src =
            Tensor::from_vec(vec![5u32; num_tokens], (num_tokens,), &dev_t).expect("positions_src");
        let positions_dst = OutputPool::global()
            .reserve_pooled(&[num_tokens], DType::U32, &dev_t)
            .expect("positions_dst reserve")
            .into_tensor();
        crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&positions_dst, &positions_src)
            .expect("positions memcpy");

        // Q/K input tensors created as fresh allocs each iteration; the
        // pool path reshape+contiguous keeps them in place.
        let q_input_a =
            Tensor::from_slice(&q_data, (num_tokens, num_heads * head_dim), &dev_t).unwrap();
        let k_input_a =
            Tensor::from_slice(&k_data, (num_tokens, num_kv_heads * head_dim), &dev_t).unwrap();

        // ── Pre-warm pool: run eagerly twice so pool slots stabilise. ──
        let _ = run_rope_pooled(
            &q_input_a,
            &k_input_a,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );
        OutputPool::global().reset_cursors();
        let _ = run_rope_pooled(
            &q_input_a,
            &k_input_a,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );

        // Sync — no pending work allowed before capture.
        unsafe {
            let r = candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
            assert_eq!(r, CUresult::CUDA_SUCCESS, "pre-eager sync");
        }

        // ── Path A: EAGER reference. Reset cursors so kernel writes go ──
        //          to slot 0/1 of each pool entry. Capture k_out bytes.
        OutputPool::global().reset_cursors();
        let (_q_eager, k_eager) = run_rope_pooled(
            &q_input_a,
            &k_input_a,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let eager_k_bytes: Vec<u16> = {
            // Read out via to_vec for F16 → u16 raw bits.
            let f16s: Vec<half::f16> = k_eager.flatten_all().unwrap().to_vec1().unwrap();
            f16s.iter().map(|v| v.to_bits()).collect()
        };

        // ── Path B: CAPTURE + REPLAY. ──
        // Warmup pass (eager call inside our function but before capture
        // begins) re-populates pool, then we capture a second forward.
        OutputPool::global().reset_cursors();
        let _ = run_rope_pooled(
            &q_input_a,
            &k_input_a,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        OutputPool::global().reset_cursors();
        let stream = cuda_dev.cuda_stream().cu_stream();
        let begin = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        assert_eq!(begin, CUresult::CUDA_SUCCESS, "cuStreamBeginCapture_v2");

        // Capture forward: pool reserves return same slot addresses as
        // the warmup pass above (cursor was reset).
        let (_q_cap, k_cap) = run_rope_pooled(
            &q_input_a,
            &k_input_a,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );

        let mut graph: CUgraph = std::ptr::null_mut();
        let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
        assert_eq!(end, CUresult::CUDA_SUCCESS, "cuStreamEndCapture");
        assert!(!graph.is_null());

        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        assert_eq!(inst, CUresult::CUDA_SUCCESS, "cuGraphInstantiateWithFlags");

        // Replay: same inputs (positions_dst, q_input_a, k_input_a) are
        // still populated. The captured graph re-executes its memcpy +
        // kernel chain against the same pool slots.
        let launch = unsafe { cuGraphLaunch(exec, stream) };
        assert_eq!(launch, CUresult::CUDA_SUCCESS, "cuGraphLaunch");
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let replay_k_bytes: Vec<u16> = {
            let f16s: Vec<half::f16> = k_cap.flatten_all().unwrap().to_vec1().unwrap();
            f16s.iter().map(|v| v.to_bits()).collect()
        };

        unsafe {
            cuGraphExecDestroy(exec);
            cuGraphDestroy(graph);
        }

        // Bit-identical: same kernel, same inputs, deterministic → must match.
        if eager_k_bytes != replay_k_bytes {
            let mismatches: Vec<(usize, u16, u16)> = eager_k_bytes
                .iter()
                .zip(replay_k_bytes.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect();
            panic!(
                "RoPE K diverges under capture-replay vs eager.\n\
                 Total elements: {}, mismatches (first 8): {:?}\n\
                 This reproduces the Phase CR.6 finding — run compute-sanitizer\n\
                 on this test binary to identify the root cause.",
                eager_k_bytes.len(),
                mismatches
            );
        }
    }

    /// Mini-repro v2: RoPE K under capture-replay where Q/K inputs are
    /// themselves pool slots written by a preceding captured memcpy.
    /// This mimics the production flow where q_proj/k_proj outputs are
    /// pool-backed activations, NOT fresh-alloc tensors.
    ///
    /// If `rope_k_capture_replay_matches_eager` passed but this fails,
    /// the bug is sensitive to whether the kernel's input is a pool-slot
    /// pointer reused by a captured upstream op vs. a stable external
    /// tensor — strong hint that the captured graph mis-orders ops or
    /// has a memcpy-then-kernel dependency issue.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn rope_k_pool_chained_capture_replay_matches_eager() {
        use crate::engine::output_pool::OutputPool;
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
            CUstreamCaptureMode,
        };

        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        let num_tokens = 1usize;
        let num_heads = 32usize;
        let num_kv_heads = 8usize;
        let head_dim = 64usize;
        let rot_dim = 64usize;
        let rope_theta = 500000.0_f32;

        let mut q_data = Vec::<half::f16>::with_capacity(num_tokens * num_heads * head_dim);
        for i in 0..(num_tokens * num_heads * head_dim) {
            q_data.push(half::f16::from_f32(
                ((i as f32 * 0.0031).sin() * 0.5) as f32,
            ));
        }
        let mut k_data = Vec::<half::f16>::with_capacity(num_tokens * num_kv_heads * head_dim);
        for i in 0..(num_tokens * num_kv_heads * head_dim) {
            k_data.push(half::f16::from_f32(
                ((i as f32 * 0.0027).cos() * 0.5) as f32,
            ));
        }
        let cos_sin = build_cos_sin_cache(rot_dim, 64, rope_theta, &dev_t);

        let positions_src =
            Tensor::from_vec(vec![5u32; num_tokens], (num_tokens,), &dev_t).unwrap();
        let positions_dst = OutputPool::global()
            .reserve_pooled(&[num_tokens], DType::U32, &dev_t)
            .unwrap()
            .into_tensor();
        crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&positions_dst, &positions_src)
            .unwrap();

        // Fresh source tensors holding the canonical Q/K input bytes —
        // these are written into pool slots at the start of each path
        // (eager + capture warmup + capture record) via captured memcpy.
        let q_src =
            Tensor::from_slice(&q_data, (num_tokens, num_heads * head_dim), &dev_t).unwrap();
        let k_src =
            Tensor::from_slice(&k_data, (num_tokens, num_kv_heads * head_dim), &dev_t).unwrap();

        // Helper: reserve Q/K pool slots, memcpy src→dst, then run RoPE.
        // This mirrors what happens at production: q_proj writes into a
        // pool slot (which here we simulate via memcpy from a fresh src),
        // then RoPE reads that slot. Both ops are captured in sequence.
        let exec_path = |label: &str| -> Vec<u16> {
            let q_pool = OutputPool::global()
                .reserve_pooled(&[num_tokens, num_heads * head_dim], DType::F16, &dev_t)
                .unwrap()
                .into_tensor();
            let k_pool = OutputPool::global()
                .reserve_pooled(&[num_tokens, num_kv_heads * head_dim], DType::F16, &dev_t)
                .unwrap()
                .into_tensor();
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&q_pool, &q_src).unwrap();
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&k_pool, &k_src).unwrap();
            let (_q_out, k_out) = run_rope_pooled(
                &q_pool,
                &k_pool,
                &positions_dst,
                &cos_sin,
                rot_dim,
                head_dim,
                num_heads,
                num_kv_heads,
            );
            let _ = label;
            let f16s: Vec<half::f16> = k_out.flatten_all().unwrap().to_vec1().unwrap();
            f16s.iter().map(|v| v.to_bits()).collect()
        };

        // Pre-warm pool (2 forwards eager).
        OutputPool::global().reset_cursors();
        let _ = exec_path("warmup1");
        OutputPool::global().reset_cursors();
        let _ = exec_path("warmup2");
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        // Path A: eager reference.
        OutputPool::global().reset_cursors();
        let eager_k = exec_path("eager");
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        // Path B: capture warmup + capture record.
        OutputPool::global().reset_cursors();
        let _ = exec_path("capture-warmup");
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        OutputPool::global().reset_cursors();
        let stream = cuda_dev.cuda_stream().cu_stream();
        let begin = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        assert_eq!(begin, CUresult::CUDA_SUCCESS);

        // The exec_path inside capture window: pool reserves return the
        // same slot addresses as the warmup pass; captured memcpys and
        // RoPE op are recorded into the graph.
        let q_pool = OutputPool::global()
            .reserve_pooled(&[num_tokens, num_heads * head_dim], DType::F16, &dev_t)
            .unwrap()
            .into_tensor();
        let k_pool = OutputPool::global()
            .reserve_pooled(&[num_tokens, num_kv_heads * head_dim], DType::F16, &dev_t)
            .unwrap()
            .into_tensor();
        crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&q_pool, &q_src).unwrap();
        crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&k_pool, &k_src).unwrap();
        let (_q_cap, k_cap) = run_rope_pooled(
            &q_pool,
            &k_pool,
            &positions_dst,
            &cos_sin,
            rot_dim,
            head_dim,
            num_heads,
            num_kv_heads,
        );

        let mut graph: CUgraph = std::ptr::null_mut();
        let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
        assert_eq!(end, CUresult::CUDA_SUCCESS);

        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        assert_eq!(inst, CUresult::CUDA_SUCCESS);

        // Replay.
        let launch = unsafe { cuGraphLaunch(exec, stream) };
        assert_eq!(launch, CUresult::CUDA_SUCCESS);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let f16s: Vec<half::f16> = k_cap.flatten_all().unwrap().to_vec1().unwrap();
        let replay_k: Vec<u16> = f16s.iter().map(|v| v.to_bits()).collect();

        unsafe {
            cuGraphExecDestroy(exec);
            cuGraphDestroy(graph);
        }

        if eager_k != replay_k {
            let mismatches: Vec<(usize, u16, u16)> = eager_k
                .iter()
                .zip(replay_k.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect();
            panic!(
                "RoPE K (pool-chained) diverges: {} elems, first 8 mismatches: {:?}",
                eager_k.len(),
                mismatches
            );
        }
    }

    /// Mini-repro v3: paged_attention V2 (main + reduce) under
    /// capture-replay. CR.6 pool diff showed F16 [1, 32, 64] slot 0
    /// (V2 output) diverges. Tests whether V2 kernels in isolation
    /// reproduce the bug.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn paged_attn_v2_pool_capture_replay_matches_eager() {
        use crate::engine::output_pool::OutputPool;
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
            CUstreamCaptureMode,
        };

        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        // Llama-3.2-1B production shape.
        let num_seqs = 1usize;
        let num_heads = 32usize;
        let num_kv_heads = 8usize;
        let head_dim = 64usize;
        let block_size = 16usize;
        let num_blocks = 8usize;
        let max_blocks_per_seq = 1usize;
        let max_seq_len = 6usize;
        let worst_case_max_seq_len = 1024usize; // partition_size=64 → max_num_partitions=16
        let partition_size = 64usize;

        // Synthetic Q (pool-backed, captured src→dst memcpy in replay path).
        let mut q_data = Vec::<half::f16>::with_capacity(num_seqs * num_heads * head_dim);
        for i in 0..(num_seqs * num_heads * head_dim) {
            q_data.push(half::f16::from_f32(
                ((i as f32 * 0.0019).sin() * 0.3) as f32,
            ));
        }
        let q_src = Tensor::from_slice(&q_data, (num_seqs, num_heads, head_dim), &dev_t).unwrap();

        // K/V cache: pre-fill blocks with non-trivial values so attention
        // produces meaningful (non-zero) output. NHD layout: [num_blocks, block_size, num_kv_heads, head_dim].
        let kv_elems = num_blocks * block_size * num_kv_heads * head_dim;
        let mut k_cache_data = Vec::<half::f16>::with_capacity(kv_elems);
        let mut v_cache_data = Vec::<half::f16>::with_capacity(kv_elems);
        for i in 0..kv_elems {
            k_cache_data.push(half::f16::from_f32(
                ((i as f32 * 0.0021).cos() * 0.4) as f32,
            ));
            v_cache_data.push(half::f16::from_f32(
                ((i as f32 * 0.0023).sin() * 0.4) as f32,
            ));
        }
        let k_cache = Tensor::from_slice(
            &k_cache_data,
            (num_blocks, block_size, num_kv_heads, head_dim),
            &dev_t,
        )
        .unwrap();
        let v_cache = Tensor::from_slice(
            &v_cache_data,
            (num_blocks, block_size, num_kv_heads, head_dim),
            &dev_t,
        )
        .unwrap();

        // Block tables held as fresh Tensor for test lifetime.
        let bt_dst = Tensor::from_vec(
            vec![0u32; num_seqs * max_blocks_per_seq],
            (num_seqs, max_blocks_per_seq),
            &dev_t,
        )
        .unwrap();

        // sl_dst held by test scope for entire test lifetime — its
        // storage is stable, captureable. We use Tensor::from_vec
        // (not pool) here because cross-test pool state pollution
        // makes debug-build to_vec1 read garbage from sl pool slots;
        // for this mini-repro a stable Tensor::from_vec lifetime is
        // enough. Must run in release (--release) — debug mode hits
        // `debug_assert_seq_lens_within_bound` inside paged_attn_v2
        // which has a separate test-pollution issue with to_vec1.
        let sl_dst =
            Tensor::from_vec(vec![max_seq_len as u32; num_seqs], (num_seqs,), &dev_t).unwrap();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let exec_path = || -> Vec<u16> {
            let q_pool = OutputPool::global()
                .reserve_pooled(&[num_seqs, num_heads, head_dim], DType::F16, &dev_t)
                .unwrap()
                .into_tensor();
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&q_pool, &q_src).unwrap();
            let out = paged_attention_v2_cuda_pooled(
                &q_pool,
                &k_cache,
                &v_cache,
                &bt_dst,
                &sl_dst,
                scale,
                num_heads,
                num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                worst_case_max_seq_len,
                head_dim,
                block_size,
                partition_size,
            )
            .unwrap();
            let f16s: Vec<half::f16> = out.flatten_all().unwrap().to_vec1().unwrap();
            f16s.iter().map(|v| v.to_bits()).collect()
        };

        OutputPool::global().reset_cursors();
        let _ = exec_path();
        OutputPool::global().reset_cursors();
        let _ = exec_path();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        OutputPool::global().reset_cursors();
        let eager_out = exec_path();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        OutputPool::global().reset_cursors();
        let _ = exec_path();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        OutputPool::global().reset_cursors();
        let stream = cuda_dev.cuda_stream().cu_stream();
        let begin = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        assert_eq!(begin, CUresult::CUDA_SUCCESS);

        let q_pool = OutputPool::global()
            .reserve_pooled(&[num_seqs, num_heads, head_dim], DType::F16, &dev_t)
            .unwrap()
            .into_tensor();
        crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&q_pool, &q_src).unwrap();
        let out_cap = paged_attention_v2_cuda_pooled(
            &q_pool,
            &k_cache,
            &v_cache,
            &bt_dst,
            &sl_dst,
            scale,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            max_seq_len,
            worst_case_max_seq_len,
            head_dim,
            block_size,
            partition_size,
        )
        .unwrap();

        let mut graph: CUgraph = std::ptr::null_mut();
        let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
        assert_eq!(end, CUresult::CUDA_SUCCESS);

        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        assert_eq!(inst, CUresult::CUDA_SUCCESS);

        let launch = unsafe { cuGraphLaunch(exec, stream) };
        assert_eq!(launch, CUresult::CUDA_SUCCESS);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let f16s: Vec<half::f16> = out_cap.flatten_all().unwrap().to_vec1().unwrap();
        let replay_out: Vec<u16> = f16s.iter().map(|v| v.to_bits()).collect();

        unsafe {
            cuGraphExecDestroy(exec);
            cuGraphDestroy(graph);
        }

        if eager_out != replay_out {
            let mismatches: Vec<(usize, u16, u16)> = eager_out
                .iter()
                .zip(replay_out.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect();
            panic!(
                "paged_attn V2 diverges under capture-replay: {} elems, first 8: {:?}",
                eager_out.len(),
                mismatches
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Phase CR.9: multi-layer mini-decoder capture-replay repro
    //
    // Single-kernel tests (CR.8) all pass — bug not in any individual
    // kernel's cuGraph handling. Production wrong-tokens manifests via
    // ~600 captured nodes across 16 attention layers. This test stitches
    // 2 layers of the EXACT typed pool-backed forward sequence
    // (`QuantizedLlamaDecoderLayer::forward_decode_batch_with_shared_pooled`)
    // with bf16_matmul standing in for quant linears (bug pattern
    // observed in EXL3 AND AWQ-Marlin → not quant-specific). If 2-layer
    // forward diverges → minimal repro for sanitizer/cuda-gdb. If not
    // → escalate to N=4..16 layers.
    // ─────────────────────────────────────────────────────────────────

    #[cfg(feature = "cuda-kernels")]
    struct MiniLayerWeights {
        input_norm: Tensor, // [hidden_size]
        q_proj: Tensor,     // [num_heads*head_dim, hidden_size]
        k_proj: Tensor,     // [num_kv_heads*head_dim, hidden_size]
        v_proj: Tensor,     // [num_kv_heads*head_dim, hidden_size]
        o_proj: Tensor,     // [hidden_size, num_heads*head_dim]
        post_attn_norm: Tensor,
        gate_proj: Tensor, // [intermediate_size, hidden_size]
        up_proj: Tensor,   // [intermediate_size, hidden_size]
        down_proj: Tensor, // [hidden_size, intermediate_size]
    }

    #[cfg(feature = "cuda-kernels")]
    fn mini_decoder_make_weights(
        cfg: &MiniDecoderCfg,
        dev: &Device,
        seed_base: u32,
    ) -> Vec<MiniLayerWeights> {
        let mut s = seed_base;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            // Tiny scale → numerical stability across many matmuls.
            half::bf16::from_f32((((s >> 16) as f32) / 32768.0 - 1.0) * 0.02)
        };
        let mut mk = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            let data: Vec<half::bf16> = (0..n).map(|_| next()).collect();
            Tensor::from_slice(&data, shape, dev).unwrap()
        };
        let mk_ones = |n: usize| {
            // RMSNorm weights start near 1.
            let data: Vec<half::bf16> = (0..n).map(|_| half::bf16::from_f32(1.0)).collect();
            Tensor::from_slice(&data, &[n], dev).unwrap()
        };

        let h = cfg.hidden_size;
        let nq = cfg.num_heads * cfg.head_dim;
        let nk = cfg.num_kv_heads * cfg.head_dim;
        let i = cfg.intermediate_size;

        (0..cfg.num_layers)
            .map(|_| MiniLayerWeights {
                input_norm: mk_ones(h),
                q_proj: mk(&[nq, h]),
                k_proj: mk(&[nk, h]),
                v_proj: mk(&[nk, h]),
                o_proj: mk(&[h, nq]),
                post_attn_norm: mk_ones(h),
                gate_proj: mk(&[i, h]),
                up_proj: mk(&[i, h]),
                down_proj: mk(&[h, i]),
            })
            .collect()
    }

    #[cfg(feature = "cuda-kernels")]
    #[derive(Clone)]
    struct MiniDecoderCfg {
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        num_layers: usize,
        rms_eps: f32,
    }

    /// Run ONE decoder layer with the same op sequence as
    /// `QuantizedLlamaDecoderLayer::forward_decode_batch_with_shared_pooled`
    /// but with bf16_matmul replacing quant linears.
    #[cfg(feature = "cuda-kernels")]
    #[allow(clippy::too_many_arguments)]
    /// D2 bisect-gate: which op groups to keep in `mini_decoder_layer_forward`.
    /// Set `CR_BISECT_KEEP` to a string of single-letter flags to restrict
    /// the forward to those op groups. Default (env unset) keeps everything,
    /// preserving the original capture-replay-correctness contract.
    ///
    /// Letters:
    /// - `N` rms_norm (input + post_attn + final)
    /// - `M` bf16_matmul (q/k/v/o/gate/up/down)
    /// - `R` rotary_embedding
    /// - `C` reshape_and_cache
    /// - `P` paged_attention_v2
    /// - `S` silu_and_mul
    /// - `A` bf16_add (residual sums)
    #[cfg(feature = "cuda-kernels")]
    fn bisect_keep(c: char) -> bool {
        use std::sync::OnceLock;
        static MASK: OnceLock<Option<String>> = OnceLock::new();
        let mask = MASK.get_or_init(|| std::env::var("CR_BISECT_KEEP").ok());
        match mask {
            None => true,
            Some(s) => s.contains(c),
        }
    }

    // `mini_decoder_layer_forward` and downstream replay helpers/tests
    // invoke `rms_norm_cuda_pooled` / `half_add_pooled` /
    // `silu_and_mul_separate_pooled` / `embedding_pooled` which live
    // behind the `cuda-layernorm` and `cuda-fused-activations`
    // sub-features. Gating only on `cuda-kernels` made the lib test
    // binary fail to compile in the default `cuda-kernels`-only build
    // (the historical baseline documented in Cargo.toml). The
    // `mini-decoder-tests` meta-feature aggregates the actual
    // dependencies — see Cargo.toml for the rationale.
    #[cfg(feature = "mini-decoder-tests")]
    #[allow(clippy::too_many_arguments)]
    fn mini_decoder_layer_forward(
        xs: &Tensor,
        layer: &MiniLayerWeights,
        cfg: &MiniDecoderCfg,
        positions_dev: &Tensor,
        slot_mapping_dev: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        cos_sin: &Tensor,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        worst: usize,
        partition_size: usize,
        block_size: usize,
    ) -> Tensor {
        let batch_size = xs.dim(0).unwrap();

        // residual = xs
        let residual = xs.clone();

        // input_norm
        let xs_n = if bisect_keep('N') {
            rms_norm_cuda_pooled(xs, &layer.input_norm, cfg.rms_eps).unwrap()
        } else {
            xs.clone()
        };

        // q/k/v projections (half_matmul_pooled = input @ weight.t)
        let (q, k, v) = if bisect_keep('M') {
            (
                half_matmul_pooled(&xs_n, &layer.q_proj).unwrap(),
                half_matmul_pooled(&xs_n, &layer.k_proj).unwrap(),
                half_matmul_pooled(&xs_n, &layer.v_proj).unwrap(),
            )
        } else {
            // Use zero-cost view tensors that match the shapes the rest
            // of the forward expects. We reuse `xs_n` reshaped if the
            // shape happens to match — but for simplicity use pool reserves
            // so downstream tensors still have backing storage.
            let m = batch_size;
            let dev = xs_n.device();
            let dt = xs_n.dtype();
            let qz = crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[m, 1, cfg.num_heads * cfg.head_dim], dt, dev)
                .unwrap()
                .into_tensor();
            let kz = crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[m, 1, cfg.num_kv_heads * cfg.head_dim], dt, dev)
                .unwrap()
                .into_tensor();
            let vz = crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[m, 1, cfg.num_kv_heads * cfg.head_dim], dt, dev)
                .unwrap()
                .into_tensor();
            (qz, kz, vz)
        };

        // Reshape q/k/v to (batch, num_heads, head_dim) like the attention
        // forward does (via reshape+transpose+squeeze).
        let q3 = q
            .reshape((batch_size, 1, cfg.num_heads, cfg.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .squeeze(2)
            .unwrap();
        let k3 = k
            .reshape((batch_size, 1, cfg.num_kv_heads, cfg.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .squeeze(2)
            .unwrap();
        let v3 = v
            .reshape((batch_size, 1, cfg.num_kv_heads, cfg.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .squeeze(2)
            .unwrap();

        // RoPE (pool-backed)
        let (q_rope, k_rope) = if bisect_keep('R') {
            rotary_embedding_cuda_pooled(
                &q3,
                &k3,
                positions_dev,
                cos_sin,
                cfg.head_dim,
                cfg.head_dim,
                cfg.num_heads,
                cfg.num_kv_heads,
                /* is_neox = */ true,
            )
            .unwrap()
        } else {
            (q3.clone(), k3.clone())
        };
        let q_for_attn = q_rope
            .reshape((batch_size, cfg.num_heads, cfg.head_dim))
            .unwrap();
        let k_for_cache = k_rope
            .reshape((batch_size, cfg.num_kv_heads, cfg.head_dim))
            .unwrap();

        // KV cache write
        if bisect_keep('C') {
            reshape_and_cache_cuda(
                &k_for_cache,
                &v3,
                k_cache,
                v_cache,
                slot_mapping_dev,
                cfg.num_kv_heads,
                cfg.head_dim,
                block_size,
                CudaCacheLayout::NHD,
            )
            .unwrap();
        }

        // Paged attention V2 (worst-case sized, pool-backed)
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let attn = if bisect_keep('P') {
            paged_attention_v2_cuda_pooled(
                &q_for_attn,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                scale,
                cfg.num_heads,
                cfg.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                worst,
                cfg.head_dim,
                block_size,
                partition_size,
            )
            .unwrap()
        } else {
            // Reserve a stable pool slot of the same shape so downstream
            // o_proj / residual still has a tensor of the expected layout.
            crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(
                    &[batch_size, cfg.num_heads * cfg.head_dim],
                    xs_n.dtype(),
                    xs_n.device(),
                )
                .unwrap()
                .into_tensor()
        };

        // o_proj: attn shape [batch, num_heads*head_dim] → unsqueeze for [B,1,H*D]
        let attn_3d = attn.unsqueeze(1).unwrap();
        let attn_out = if bisect_keep('M') {
            half_matmul_pooled(&attn_3d, &layer.o_proj).unwrap()
        } else {
            attn_3d.clone()
        };

        // residual add (post-attention)
        let xs1 = if bisect_keep('A') {
            half_add_pooled(&attn_out, &residual).unwrap()
        } else {
            attn_out
        };
        let residual2 = xs1.clone();

        // post_attn_norm
        let xs_n2 = if bisect_keep('N') {
            rms_norm_cuda_pooled(&xs1, &layer.post_attn_norm, cfg.rms_eps).unwrap()
        } else {
            xs1.clone()
        };

        // MLP: gate + up → silu_and_mul → down
        let (gate, up) = if bisect_keep('M') {
            (
                half_matmul_pooled(&xs_n2, &layer.gate_proj).unwrap(),
                half_matmul_pooled(&xs_n2, &layer.up_proj).unwrap(),
            )
        } else {
            let dt = xs_n2.dtype();
            let dev = xs_n2.device();
            let g = crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[batch_size, 1, cfg.intermediate_size], dt, dev)
                .unwrap()
                .into_tensor();
            let u = crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[batch_size, 1, cfg.intermediate_size], dt, dev)
                .unwrap()
                .into_tensor();
            (g, u)
        };
        let activated = if bisect_keep('S') {
            silu_and_mul_separate_pooled(&gate, &up).unwrap()
        } else {
            gate.clone()
        };
        let mlp_out = if bisect_keep('M') {
            half_matmul_pooled(&activated, &layer.down_proj).unwrap()
        } else {
            // Reserve a slot of post-MLP shape so the residual add below
            // has a well-shaped operand.
            let dt = residual2.dtype();
            let dev = residual2.device();
            crate::engine::output_pool::OutputPool::global()
                .reserve_pooled(&[batch_size, 1, cfg.hidden_size], dt, dev)
                .unwrap()
                .into_tensor()
        };

        // residual add (post-MLP)
        if bisect_keep('A') {
            half_add_pooled(&residual2, &mlp_out).unwrap()
        } else {
            residual2
        }
    }

    #[cfg(feature = "mini-decoder-tests")]
    #[allow(clippy::too_many_arguments)]
    fn mini_decoder_forward(
        xs0: &Tensor,
        layers: &[MiniLayerWeights],
        final_norm: &Tensor,
        cfg: &MiniDecoderCfg,
        positions_dev: &Tensor,
        slot_mapping_dev: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        k_caches: &[Tensor],
        v_caches: &[Tensor],
        cos_sin: &Tensor,
        max_blocks_per_seq: usize,
        max_seq_len: usize,
        worst: usize,
        partition_size: usize,
        block_size: usize,
    ) -> Tensor {
        let mut xs = xs0.clone();
        for (li, layer) in layers.iter().enumerate() {
            xs = mini_decoder_layer_forward(
                &xs,
                layer,
                cfg,
                positions_dev,
                slot_mapping_dev,
                block_tables,
                seq_lens,
                &k_caches[li],
                &v_caches[li],
                cos_sin,
                max_blocks_per_seq,
                max_seq_len,
                worst,
                partition_size,
                block_size,
            );
        }
        // Final norm
        rms_norm_cuda_pooled(&xs, final_norm, cfg.rms_eps).unwrap()
    }

    /// Stress test: many memsets per capture, two rounds.
    /// If 2nd round drops nodes → cuGraph has size/state issue.
    /// If both retain same count → bug is specific op interaction.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn double_capture_many_memsets() {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphGetNodes, cuGraphInstantiateWithFlags,
            cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec,
            CUresult, CUstreamCaptureMode,
        };
        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }
        let t = Tensor::from_vec(vec![1u32; 16], (16,), &dev_t).unwrap();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let stream = cuda_dev.cuda_stream().cu_stream();
        let mut sizes = Vec::new();
        for round in 0..2 {
            unsafe {
                cuStreamBeginCapture_v2(
                    stream,
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                );
            }
            for _ in 0..400 {
                zero_pool_tensor_dtod(&t).unwrap();
            }
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe { cuStreamEndCapture(stream, &mut graph) };

            let mut n: usize = 0;
            unsafe { cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) };
            sizes.push(n);
            eprintln!("Round {round}: {n} nodes");

            let mut exec: CUgraphExec = std::ptr::null_mut();
            unsafe {
                cuGraphInstantiateWithFlags(&mut exec, graph, 0);
                cuGraphLaunch(exec, stream);
                candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
                cuGraphExecDestroy(exec);
                cuGraphDestroy(graph);
            }
            let _ = CUresult::CUDA_SUCCESS;
        }
        assert_eq!(sizes[0], sizes[1], "got {sizes:?}");
    }

    /// Two captures, each FULL cycle (Instantiate + Launch + Destroy).
    /// If 2nd capture loses nodes → cuGraphInstantiate or cuGraphLaunch
    /// from the 1st leaves stream state that breaks 2nd capture.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn double_capture_with_full_cycle() {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphGetNodes, cuGraphInstantiateWithFlags,
            cuGraphLaunch, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec,
            CUresult, CUstreamCaptureMode,
        };
        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }
        let t = Tensor::from_vec(vec![1u32; 16], (16,), &dev_t).unwrap();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let stream = cuda_dev.cuda_stream().cu_stream();
        let mut sizes = Vec::new();
        for round in 0..2 {
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS);
            zero_pool_tensor_dtod(&t).unwrap();
            zero_pool_tensor_dtod(&t).unwrap();
            zero_pool_tensor_dtod(&t).unwrap();
            let mut graph: CUgraph = std::ptr::null_mut();
            unsafe { cuStreamEndCapture(stream, &mut graph) };

            let mut n: usize = 0;
            unsafe { cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n) };
            sizes.push(n);
            eprintln!("Round {round}: {n} nodes (full cycle)");

            let mut exec: CUgraphExec = std::ptr::null_mut();
            unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
            unsafe { cuGraphLaunch(exec, stream) };
            unsafe { candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize() };
            unsafe {
                cuGraphExecDestroy(exec);
                cuGraphDestroy(graph);
            }
        }
        assert_eq!(sizes[0], sizes[1], "got {sizes:?}");
    }

    /// Two sequential captures of a multi-op forward — WITHOUT
    /// instantiate/launch between. If 2nd capture still loses nodes →
    /// the bug is in EndCapture/BeginCapture sequence itself, not in
    /// Instantiate/Launch state leaking.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn double_capture_without_instantiate() {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphGetNodes, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph,
            CUresult, CUstreamCaptureMode,
        };
        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        let t = Tensor::from_vec(vec![1u32; 16], (16,), &dev_t).unwrap();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let stream = cuda_dev.cuda_stream().cu_stream();
        let mut sizes = Vec::new();
        for round in 0..2 {
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS, "round {round} BeginCapture");
            // Mix: memset (from CR.1 path) + a simple op
            zero_pool_tensor_dtod(&t).unwrap();
            zero_pool_tensor_dtod(&t).unwrap();
            zero_pool_tensor_dtod(&t).unwrap();
            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            assert_eq!(end, CUresult::CUDA_SUCCESS);

            let mut n: usize = 0;
            unsafe {
                cuGraphGetNodes(graph, std::ptr::null_mut(), &mut n);
            }
            sizes.push(n);
            eprintln!("Round {round}: {n} nodes (no instantiate)");
            unsafe {
                cuGraphDestroy(graph);
            }
        }
        assert_eq!(
            sizes[0], sizes[1],
            "double capture (no instantiate) should produce same node count, got {sizes:?}"
        );
    }

    /// Bare-bones reproducer: capture a single `cuMemsetD8Async` call
    /// twice in the same process. If the second capture has fewer
    /// nodes than the first → cudarc/candle has a state issue across
    /// consecutive captures.
    #[cfg(feature = "cuda-kernels")]
    #[test]
    #[ignore]
    fn minimal_double_capture_memset() {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphGetNodes, cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph,
            CUgraphNode, CUresult, CUstreamCaptureMode,
        };
        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return,
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        let t = Tensor::from_vec(vec![1u32; 16], (16,), &dev_t).unwrap();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let count_nodes = |g: CUgraph| -> usize {
            let mut n: usize = 0;
            unsafe {
                cuGraphGetNodes(g, std::ptr::null_mut(), &mut n);
            }
            n
        };

        let stream = cuda_dev.cuda_stream().cu_stream();

        for round in 0..2 {
            let begin = unsafe {
                cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            };
            assert_eq!(begin, CUresult::CUDA_SUCCESS, "round {round} BeginCapture");

            // The captured operation: zero out the tensor's storage.
            zero_pool_tensor_dtod(&t).unwrap();

            let mut graph: CUgraph = std::ptr::null_mut();
            let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
            assert_eq!(end, CUresult::CUDA_SUCCESS, "round {round} EndCapture");

            let n = count_nodes(graph);
            eprintln!("Round {round}: captured graph has {n} nodes");
            unsafe {
                cuGraphDestroy(graph);
            }
        }
    }

    /// Dump a captured CUDA graph's DAG (nodes + types + edges) to a
    /// file. Use to diff DAGs between solo and after-prior runs of the
    /// same capture, to test the "missing dependency edge" hypothesis.
    #[cfg(feature = "cuda-kernels")]
    fn dump_graph_dag(
        graph: candle_core::cuda::cudarc::driver::sys::CUgraph,
        path: &str,
    ) -> std::io::Result<()> {
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphGetEdges, cuGraphGetNodes, cuGraphNodeGetType, CUgraphNode, CUgraphNodeType,
        };
        use std::io::Write;
        unsafe {
            let mut num_nodes: usize = 0;
            cuGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes);
            let mut nodes: Vec<CUgraphNode> = vec![std::ptr::null_mut(); num_nodes];
            cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_nodes);

            let mut types: Vec<u32> = Vec::with_capacity(num_nodes);
            for n in &nodes {
                let mut t: CUgraphNodeType = std::mem::zeroed();
                cuGraphNodeGetType(*n, &mut t);
                types.push(t as u32);
            }

            let mut num_edges: usize = 0;
            cuGraphGetEdges(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut num_edges,
            );
            let mut from: Vec<CUgraphNode> = vec![std::ptr::null_mut(); num_edges];
            let mut to: Vec<CUgraphNode> = vec![std::ptr::null_mut(); num_edges];
            cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut num_edges);

            // Map raw ptrs to indices.
            let ptr_to_idx = |p: CUgraphNode| -> usize {
                nodes.iter().position(|n| *n == p).unwrap_or(usize::MAX)
            };
            let mut edges: Vec<(usize, usize)> = from
                .iter()
                .zip(to.iter())
                .map(|(f, t)| (ptr_to_idx(*f), ptr_to_idx(*t)))
                .collect();
            // Sort for stable diff between runs.
            edges.sort();

            let mut f = std::fs::File::create(path)?;
            writeln!(f, "nodes={num_nodes} edges={num_edges}")?;
            for (i, t) in types.iter().enumerate() {
                let t_name = match t {
                    0 => "KERNEL",
                    1 => "MEMCPY",
                    2 => "MEMSET",
                    3 => "HOST",
                    4 => "GRAPH",
                    5 => "EMPTY",
                    6 => "WAIT_EVENT",
                    7 => "EVENT_RECORD",
                    10 => "MEM_ALLOC",
                    11 => "MEM_FREE",
                    _ => "OTHER",
                };
                writeln!(f, "node[{i}] = {t_name} ({t})")?;
            }
            for (s, d) in &edges {
                writeln!(f, "edge {s} -> {d}")?;
            }
            // Per-node fan-in/fan-out for quick visual scan.
            writeln!(f, "--- in-degree (= number of incoming edges):")?;
            let mut in_deg = vec![0usize; num_nodes];
            for (_s, d) in &edges {
                in_deg[*d] += 1;
            }
            for (i, d) in in_deg.iter().enumerate() {
                if *d == 0 {
                    writeln!(f, "  node[{i}] (in=0) — ROOT")?;
                }
            }
        }
        Ok(())
    }

    /// Inner harness for the multi-layer capture-replay correctness
    /// check. Takes a layer count, `input_shift` flag (capture vs replay
    /// payload differ), and `with_embedding` flag (when true, the
    /// captured graph starts with `embedding_pooled` index_select like
    /// production — input is U32 token IDs, not pre-embedded hidden).
    #[cfg(feature = "mini-decoder-tests")]
    #[allow(clippy::too_many_arguments)]
    fn run_mini_decoder_capture_replay(
        num_layers: usize,
        input_shift: bool,
        with_embedding: bool,
    ) -> std::result::Result<(), String> {
        use crate::engine::output_pool::OutputPool;
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
            CUstreamCaptureMode,
        };

        // D2 hypothesis: process-shared Device avoids the cross-invocation
        // CudaContext drop/recreate that triggers Bug A. With CR_SHARED_DEV=1,
        // both test invocations reuse one Device instead of creating a
        // fresh one each call. Default (env unset) preserves the original
        // per-call Device which reproduces Bug A.
        let dev_t = if std::env::var("CR_SHARED_DEV").is_ok() {
            use std::sync::OnceLock;
            static SHARED: OnceLock<Device> = OnceLock::new();
            SHARED
                .get_or_init(|| {
                    let d = Device::new_cuda_with_stream(0).expect("Device::new_cuda_with_stream");
                    if let Device::Cuda(c) = &d {
                        unsafe {
                            c.disable_event_tracking();
                        }
                    }
                    d
                })
                .clone()
        } else {
            match Device::new_cuda_with_stream(0) {
                Ok(d) if d.is_cuda() => d,
                _ => return Ok(()),
            }
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        // CR.11: gated by env var so we can run tests with and without
        // the clear to verify it is the cross-test pollution that
        // breaks replay correctness. With `CR_CLEAR_POOL=1`, both runs
        // pass; without, the 2nd test fails 100% mismatches.
        if std::env::var("CR_CLEAR_POOL")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            crate::engine::output_pool::OutputPool::global().clear_all();
        }

        let cfg = MiniDecoderCfg {
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            intermediate_size: 8192,
            num_layers,
            rms_eps: 1e-5,
        };
        let batch_size = 1usize;
        let block_size = 16usize;
        let num_blocks = 8usize;
        let max_blocks_per_seq = 1usize;
        let max_seq_len = 6usize;
        let worst = 1024usize;
        let partition_size = 64usize;

        let layers = mini_decoder_make_weights(&cfg, &dev_t, 0xDEAD_BEEF);
        let final_norm: Tensor = {
            let v: Vec<half::bf16> = (0..cfg.hidden_size)
                .map(|_| half::bf16::from_f32(1.0))
                .collect();
            Tensor::from_slice(&v, &[cfg.hidden_size], &dev_t).unwrap()
        };
        let kv_elems = num_blocks * block_size * cfg.num_kv_heads * cfg.head_dim;
        let mut k_caches: Vec<Tensor> = Vec::with_capacity(cfg.num_layers);
        let mut v_caches: Vec<Tensor> = Vec::with_capacity(cfg.num_layers);
        for li in 0..cfg.num_layers {
            let mut s = 0x1000 + li as u32;
            let mut next = || {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                half::bf16::from_f32((((s >> 16) as f32) / 32768.0 - 1.0) * 0.1)
            };
            let kd: Vec<half::bf16> = (0..kv_elems).map(|_| next()).collect();
            let vd: Vec<half::bf16> = (0..kv_elems).map(|_| next()).collect();
            k_caches.push(
                Tensor::from_slice(
                    &kd,
                    &[num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim],
                    &dev_t,
                )
                .unwrap(),
            );
            v_caches.push(
                Tensor::from_slice(
                    &vd,
                    &[num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim],
                    &dev_t,
                )
                .unwrap(),
            );
        }
        let cos_sin = build_cos_sin_cache(cfg.head_dim, 1024, 500000.0, &dev_t);
        let positions_dev =
            Tensor::from_vec(vec![5u32; batch_size], (batch_size,), &dev_t).unwrap();
        let slot_mapping_dev =
            Tensor::from_vec(vec![5u32; batch_size], (batch_size,), &dev_t).unwrap();
        let block_tables = Tensor::from_vec(
            vec![0u32; batch_size * max_blocks_per_seq],
            (batch_size, max_blocks_per_seq),
            &dev_t,
        )
        .unwrap();
        let seq_lens =
            Tensor::from_vec(vec![max_seq_len as u32; batch_size], (batch_size,), &dev_t).unwrap();
        // Two payloads: "warmup" (xs_a) and "real" (xs_b). When
        // input_shift=true, we memcpy xs_a into xs0 before warmup +
        // capture record, and xs_b before eager baseline + replay. This
        // mimics production where the captured graph is recorded with a
        // dummy input_ids buffer (zeros) and replayed against real
        // input_ids streamed in via cuda_memcpy_inplace.
        let xs_a_data: Vec<half::bf16> = (0..(batch_size * cfg.hidden_size))
            .map(|i| half::bf16::from_f32(((i as f32 * 0.013).sin() * 0.05) as f32))
            .collect();
        let xs_b_data: Vec<half::bf16> = (0..(batch_size * cfg.hidden_size))
            .map(|i| half::bf16::from_f32(((i as f32 * 0.019).cos() * 0.05) as f32))
            .collect();
        let xs_a =
            Tensor::from_slice(&xs_a_data, &[batch_size, 1, cfg.hidden_size], &dev_t).unwrap();
        let xs_b =
            Tensor::from_slice(&xs_b_data, &[batch_size, 1, cfg.hidden_size], &dev_t).unwrap();

        // xs0 is the persistent input buffer that the captured graph
        // references. We memcpy xs_a or xs_b into it depending on phase.
        let xs0 =
            Tensor::from_slice(&xs_a_data, &[batch_size, 1, cfg.hidden_size], &dev_t).unwrap();

        // Embedding setup (only used when with_embedding=true). Mimics
        // production's `embedding_pooled(input_ids, weight)` step that
        // produces the first hidden state via index_select.
        let vocab_size = 256usize;
        let mut s = 0xCAFE_BABE_u32;
        let mut emb_next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            half::bf16::from_f32((((s >> 16) as f32) / 32768.0 - 1.0) * 0.05)
        };
        let emb_data: Vec<half::bf16> = (0..(vocab_size * cfg.hidden_size))
            .map(|_| emb_next())
            .collect();
        let embed_weight =
            Tensor::from_slice(&emb_data, &[vocab_size, cfg.hidden_size], &dev_t).unwrap();
        // input_ids (U32) — persistent buffer used by both warmup and
        // replay; data is overwritten per phase.
        let input_ids_a =
            Tensor::from_vec(vec![5u32; batch_size], (batch_size, 1), &dev_t).unwrap();
        let input_ids_b =
            Tensor::from_vec(vec![13u32; batch_size], (batch_size, 1), &dev_t).unwrap();
        let input_ids = Tensor::from_vec(vec![5u32; batch_size], (batch_size, 1), &dev_t).unwrap();

        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        let load_xs = |src: &Tensor| {
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&xs0, src).unwrap();
        };
        let load_ids = |src: &Tensor| {
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(&input_ids, src).unwrap();
        };

        let forward_once = || -> Tensor {
            let initial = if with_embedding {
                let embed_out = embedding_pooled(&input_ids, &embed_weight).unwrap();
                // embedding_pooled output shape is [num_tokens, hidden] —
                // mini layer expects [batch, 1, hidden]. Reshape to match.
                embed_out.reshape((batch_size, 1, cfg.hidden_size)).unwrap()
            } else {
                xs0.clone()
            };
            mini_decoder_forward(
                &initial,
                &layers,
                &final_norm,
                &cfg,
                &positions_dev,
                &slot_mapping_dev,
                &block_tables,
                &seq_lens,
                &k_caches,
                &v_caches,
                &cos_sin,
                max_blocks_per_seq,
                max_seq_len,
                worst,
                partition_size,
                block_size,
            )
        };
        let tensor_to_u16_bytes = |t: &Tensor| -> Vec<u16> {
            let bf16s: Vec<half::bf16> = t.flatten_all().unwrap().to_vec1().unwrap();
            bf16s.iter().map(|v| v.to_bits()).collect()
        };

        // Pool pre-warm: eager forwards with whatever payload (xs_a).
        load_xs(&xs_a);
        load_ids(&input_ids_a);
        OutputPool::global().reset_cursors();
        let _ = forward_once();
        OutputPool::global().reset_cursors();
        let _ = forward_once();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        // Eager baseline: load REAL payload (xs_b when shifting), run.
        if input_shift {
            load_xs(&xs_b);
            load_ids(&input_ids_b);
        }
        OutputPool::global().reset_cursors();
        let eager_out = forward_once();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let eager_bytes = tensor_to_u16_bytes(&eager_out);
        drop(eager_out);

        // Capture warmup: load DUMMY payload (xs_a). The captured
        // graph's first kernel reads xs0/input_ids → captures pointer.
        load_xs(&xs_a);
        load_ids(&input_ids_a);
        OutputPool::global().reset_cursors();
        let _ = forward_once();
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        // Capture record: still dummy payload in xs0.
        OutputPool::global().reset_cursors();
        // CR.15 / D1: bracket the captured forward with cr_trace so we
        // can diff (op, dtype, shape, slot_idx) sequences across rounds.
        // The process-static counter labels round 0 vs round 1 when this
        // function is invoked twice via two separate `#[test]` entries
        // sharing one process under `--test-threads=1`.
        {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static ROUND: AtomicUsize = AtomicUsize::new(0);
            let r = ROUND.fetch_add(1, Ordering::SeqCst);
            crate::engine::cr_trace::begin_round(&format!("mini_capture_{r}"));
        }
        let stream = cuda_dev.cuda_stream().cu_stream();
        let begin = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        if begin != CUresult::CUDA_SUCCESS {
            crate::engine::cr_trace::end_round();
            return Err(format!("BeginCapture failed: {begin:?}"));
        }
        let captured_out = forward_once();
        let mut graph: CUgraph = std::ptr::null_mut();
        let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
        crate::engine::cr_trace::end_round();
        if end != CUresult::CUDA_SUCCESS {
            return Err(format!("EndCapture failed: {end:?}"));
        }

        // CR.12: dump DAG if requested. CR_DUMP_DAG_PREFIX writes to
        // `<prefix>.<counter>.txt`, counter increments per capture in
        // this process. Both captures dumped when multiple tests run.
        if let Ok(p) = std::env::var("CR_DUMP_DAG_PREFIX") {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static CAP_IDX: AtomicUsize = AtomicUsize::new(0);
            let idx = CAP_IDX.fetch_add(1, Ordering::SeqCst);
            let full = format!("{p}.{idx}.txt");
            let _ = dump_graph_dag(graph, &full);
            eprintln!("dumped captured DAG to {full}");
        }
        if let Ok(p) = std::env::var("CR_DUMP_DAG") {
            let _ = dump_graph_dag(graph, &p);
            eprintln!("dumped captured DAG to {p}");
        }

        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if inst != CUresult::CUDA_SUCCESS {
            return Err(format!("Instantiate failed: {inst:?}"));
        }

        // Replay: load REAL payload, then launch captured graph.
        // Captured kernel chain reads xs0/input_ids and propagates.
        if input_shift {
            load_xs(&xs_b);
            load_ids(&input_ids_b);
        }
        let launch = unsafe { cuGraphLaunch(exec, stream) };
        if launch != CUresult::CUDA_SUCCESS {
            return Err(format!("Launch failed: {launch:?}"));
        }
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let replay_bytes = tensor_to_u16_bytes(&captured_out);

        unsafe {
            cuGraphExecDestroy(exec);
            cuGraphDestroy(graph);
        }

        if eager_bytes != replay_bytes {
            let mismatches: Vec<(usize, u16, u16)> = eager_bytes
                .iter()
                .zip(replay_bytes.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect();
            let n_diff = eager_bytes
                .iter()
                .zip(replay_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            return Err(format!(
                "Mini-decoder ({num_layers}-layer) diverges: {} elems, {} mismatches ({:.1}%), first 8: {:?}",
                eager_bytes.len(),
                n_diff,
                100.0 * n_diff as f64 / eager_bytes.len() as f64,
                mismatches
            ));
        }
        Ok(())
    }

    /// 1-layer, no input shift — minimum repro size.
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_1layer_capture_replay_matches_eager() {
        if let Err(e) = run_mini_decoder_capture_replay(1, false, false) {
            panic!("{e}");
        }
    }

    /// 2-layer, no input shift (same payload for eager and replay).
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_2layer_capture_replay_matches_eager() {
        if let Err(e) = run_mini_decoder_capture_replay(2, false, false) {
            panic!("{e}");
        }
    }

    /// 16-layer, no input shift.
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_16layer_capture_replay_matches_eager() {
        if let Err(e) = run_mini_decoder_capture_replay(16, false, false) {
            panic!("{e}");
        }
    }

    /// 16-layer WITH input shift.
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_16layer_capture_replay_with_input_shift() {
        if let Err(e) = run_mini_decoder_capture_replay(16, true, false) {
            panic!("{e}");
        }
    }

    /// 16-layer + input shift + EMBEDDING — full production-like flow:
    /// input_ids → embedding_pooled → decoder layers → final_norm.
    /// Capture recorded with input_ids_a (token=5), replay against
    /// input_ids_b (token=13). If the captured graph's embedding/index_select
    /// is the culprit → diverges here.
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_16layer_capture_replay_with_embedding() {
        if let Err(e) = run_mini_decoder_capture_replay(16, true, true) {
            panic!("{e}");
        }
    }

    /// Same as `mini_decoder_16layer_capture_replay_with_input_shift`
    /// but named differently to trigger ordering — runs AFTER another
    /// capture-replay test. Confirms second-capture-after-first-capture
    /// pattern (production: batch=1 → batch=2 → batch=4 → ...).
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn zz_mini_decoder_second_with_input_shift_after_prior() {
        if let Err(e) = run_mini_decoder_capture_replay(16, true, false) {
            panic!("{e}");
        }
    }

    /// 16-layer + 2 SEQUENTIAL decode forwards (replay the captured
    /// graph twice with advancing seq_lens/positions/slot_mapping
    /// between calls). Production runs 12 decode forwards in sequence;
    /// the wrong-tokens regression manifests from forward 2 onwards.
    /// If forward 1's KV cache write is silently corrupt under replay,
    /// forward 2 reads bad KV and diverges from eager's forward 2.
    /// This test diffs forward 1 AND forward 2 outputs between eager
    /// and capture-replay paths.
    #[cfg(feature = "mini-decoder-tests")]
    #[test]
    #[ignore]
    fn mini_decoder_16layer_capture_replay_two_forwards() {
        if let Err(e) = run_mini_decoder_two_forwards_capture_replay(16) {
            panic!("{e}");
        }
    }

    /// Two-forward variant of [`run_mini_decoder_capture_replay`].
    /// Compares BOTH forward 1 and forward 2 outputs between eager and
    /// capture-replay. Between forwards, advances seq_lens (6→7),
    /// positions (5→6), slot_mapping (5→6) — so forward 2 reads what
    /// forward 1 wrote to KV cache.
    #[cfg(feature = "mini-decoder-tests")]
    fn run_mini_decoder_two_forwards_capture_replay(
        num_layers: usize,
    ) -> std::result::Result<(), String> {
        use crate::engine::output_pool::OutputPool;
        use candle_core::cuda::cudarc::driver::sys::{
            cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiateWithFlags, cuGraphLaunch,
            cuStreamBeginCapture_v2, cuStreamEndCapture, CUgraph, CUgraphExec, CUresult,
            CUstreamCaptureMode,
        };

        let dev_t = match Device::new_cuda_with_stream(0) {
            Ok(d) if d.is_cuda() => d,
            _ => return Ok(()),
        };
        let cuda_dev = match &dev_t {
            Device::Cuda(c) => c.clone(),
            _ => unreachable!(),
        };
        unsafe {
            cuda_dev.disable_event_tracking();
        }

        let cfg = MiniDecoderCfg {
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            intermediate_size: 8192,
            num_layers,
            rms_eps: 1e-5,
        };
        let batch_size = 1usize;
        let block_size = 16usize;
        let num_blocks = 8usize;
        let max_blocks_per_seq = 1usize;
        let worst = 1024usize;
        let partition_size = 64usize;

        let layers = mini_decoder_make_weights(&cfg, &dev_t, 0xDEAD_BEEF);
        let final_norm: Tensor = {
            let v: Vec<half::bf16> = (0..cfg.hidden_size)
                .map(|_| half::bf16::from_f32(1.0))
                .collect();
            Tensor::from_slice(&v, &[cfg.hidden_size], &dev_t).unwrap()
        };
        let kv_elems = num_blocks * block_size * cfg.num_kv_heads * cfg.head_dim;

        // Build TWO sets of KV caches: one for eager, one for capture
        // replay. Pre-populated identically; each path's forward writes
        // independently so we can verify forward 1's KV write
        // correctness via forward 2 read.
        let mk_kv = |seed: u32| -> Tensor {
            let mut s = seed;
            let data: Vec<half::bf16> = (0..kv_elems)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    half::bf16::from_f32((((s >> 16) as f32) / 32768.0 - 1.0) * 0.1)
                })
                .collect();
            Tensor::from_slice(
                &data,
                &[num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim],
                &dev_t,
            )
            .unwrap()
        };
        let mut k_eager: Vec<Tensor> = (0..num_layers)
            .map(|li| mk_kv(0x1000 + li as u32))
            .collect();
        let mut v_eager: Vec<Tensor> = (0..num_layers)
            .map(|li| mk_kv(0x2000 + li as u32))
            .collect();
        let mut k_cap: Vec<Tensor> = (0..num_layers)
            .map(|li| mk_kv(0x1000 + li as u32))
            .collect();
        let mut v_cap: Vec<Tensor> = (0..num_layers)
            .map(|li| mk_kv(0x2000 + li as u32))
            .collect();

        let cos_sin = build_cos_sin_cache(cfg.head_dim, 1024, 500000.0, &dev_t);

        // Persistent metadata buffers — overwritten per forward.
        let positions_dev =
            Tensor::from_vec(vec![0u32; batch_size], (batch_size,), &dev_t).unwrap();
        let slot_mapping_dev =
            Tensor::from_vec(vec![0u32; batch_size], (batch_size,), &dev_t).unwrap();
        let block_tables = Tensor::from_vec(
            vec![0u32; batch_size * max_blocks_per_seq],
            (batch_size, max_blocks_per_seq),
            &dev_t,
        )
        .unwrap();
        let seq_lens = Tensor::from_vec(vec![0u32; batch_size], (batch_size,), &dev_t).unwrap();

        let load_u32_1 = |t: &Tensor, val: u32| {
            let src = Tensor::from_vec(vec![val; batch_size], t.shape(), &dev_t).unwrap();
            crate::engine::cuda_graph_runner::cuda_memcpy_inplace(t, &src).unwrap();
        };

        // Hidden state input (constant across forwards for simplicity —
        // production threads argmax of forward N as input to forward N+1
        // but we keep this fixed to isolate KV cache state effects).
        let xs0_data: Vec<half::bf16> = (0..(batch_size * cfg.hidden_size))
            .map(|i| half::bf16::from_f32(((i as f32 * 0.013).sin() * 0.05) as f32))
            .collect();
        let xs0 = Tensor::from_slice(&xs0_data, &[batch_size, 1, cfg.hidden_size], &dev_t).unwrap();

        let run_fwd = |k: &[Tensor], v: &[Tensor], max_seq: usize| -> Tensor {
            mini_decoder_forward(
                &xs0,
                &layers,
                &final_norm,
                &cfg,
                &positions_dev,
                &slot_mapping_dev,
                &block_tables,
                &seq_lens,
                k,
                v,
                &cos_sin,
                max_blocks_per_seq,
                max_seq,
                worst,
                partition_size,
                block_size,
            )
        };
        let bytes_of = |t: &Tensor| -> Vec<u16> {
            let f: Vec<half::bf16> = t.flatten_all().unwrap().to_vec1().unwrap();
            f.iter().map(|v| v.to_bits()).collect()
        };

        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }

        // ── Path A: EAGER both forwards on k_eager/v_eager ───────────
        // Forward 1: position=5, seq_len=6, slot=5
        load_u32_1(&positions_dev, 5);
        load_u32_1(&slot_mapping_dev, 5);
        load_u32_1(&seq_lens, 6);
        // Pre-warm pool
        OutputPool::global().reset_cursors();
        let _ = run_fwd(&k_eager, &v_eager, 6);
        OutputPool::global().reset_cursors();
        let _ = run_fwd(&k_eager, &v_eager, 6);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        // Restore k_eager/v_eager (warmups wrote to slot 5). Re-populate.
        for li in 0..num_layers {
            k_eager[li] = mk_kv(0x1000 + li as u32);
            v_eager[li] = mk_kv(0x2000 + li as u32);
        }
        OutputPool::global().reset_cursors();
        let eager_fwd1 = run_fwd(&k_eager, &v_eager, 6);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let eager_fwd1_bytes = bytes_of(&eager_fwd1);
        drop(eager_fwd1);
        // Forward 2: position=6, seq_len=7, slot=6 — reads slot 5 (just written by forward 1)
        load_u32_1(&positions_dev, 6);
        load_u32_1(&slot_mapping_dev, 6);
        load_u32_1(&seq_lens, 7);
        OutputPool::global().reset_cursors();
        let eager_fwd2 = run_fwd(&k_eager, &v_eager, 7);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let eager_fwd2_bytes = bytes_of(&eager_fwd2);
        drop(eager_fwd2);

        // ── Path B: CAPTURE warmup + record + REPLAY twice ───────────
        // Forward 1 (warmup + record).
        load_u32_1(&positions_dev, 5);
        load_u32_1(&slot_mapping_dev, 5);
        load_u32_1(&seq_lens, 6);
        OutputPool::global().reset_cursors();
        let _ = run_fwd(&k_cap, &v_cap, 6);
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        // Restore.
        for li in 0..num_layers {
            k_cap[li] = mk_kv(0x1000 + li as u32);
            v_cap[li] = mk_kv(0x2000 + li as u32);
        }
        OutputPool::global().reset_cursors();
        let stream = cuda_dev.cuda_stream().cu_stream();
        let begin = unsafe {
            cuStreamBeginCapture_v2(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        };
        if begin != CUresult::CUDA_SUCCESS {
            return Err(format!("BeginCapture: {begin:?}"));
        }
        let captured_fwd1_out = run_fwd(&k_cap, &v_cap, 6);
        let mut graph: CUgraph = std::ptr::null_mut();
        let end = unsafe { cuStreamEndCapture(stream, &mut graph) };
        if end != CUresult::CUDA_SUCCESS {
            return Err(format!("EndCapture: {end:?}"));
        }
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if inst != CUresult::CUDA_SUCCESS {
            return Err(format!("Instantiate: {inst:?}"));
        }
        // Replay 1 (forward 1).
        let l1 = unsafe { cuGraphLaunch(exec, stream) };
        if l1 != CUresult::CUDA_SUCCESS {
            return Err(format!("Launch 1: {l1:?}"));
        }
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let cap_fwd1_bytes = bytes_of(&captured_fwd1_out);
        // Forward 2: advance metadata, replay same graph.
        load_u32_1(&positions_dev, 6);
        load_u32_1(&slot_mapping_dev, 6);
        load_u32_1(&seq_lens, 7);
        // NB: max_seq_len kernel-arg captured into the graph reflects
        // FORWARD 1's value (6). We can't change it. But seq_lens TENSOR
        // contents we just bumped to 7. paged_attn V2 uses the device
        // seq_lens for its bounds, so it should adapt. The captured
        // max_seq_len const is only used for kernel grid sizing (max
        // partitions); since partition_size is fixed and seq_lens fits
        // within worst_case_max_seq_len, it's fine.
        let l2 = unsafe { cuGraphLaunch(exec, stream) };
        if l2 != CUresult::CUDA_SUCCESS {
            return Err(format!("Launch 2: {l2:?}"));
        }
        unsafe {
            candle_core::cuda::cudarc::driver::sys::cuCtxSynchronize();
        }
        let cap_fwd2_bytes = bytes_of(&captured_fwd1_out);

        unsafe {
            cuGraphExecDestroy(exec);
            cuGraphDestroy(graph);
        }

        let mut errs = Vec::new();
        if eager_fwd1_bytes != cap_fwd1_bytes {
            let nd = eager_fwd1_bytes
                .iter()
                .zip(cap_fwd1_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            errs.push(format!(
                "FWD 1 diverges: {nd}/{} mismatches",
                eager_fwd1_bytes.len()
            ));
        }
        if eager_fwd2_bytes != cap_fwd2_bytes {
            let nd = eager_fwd2_bytes
                .iter()
                .zip(cap_fwd2_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            errs.push(format!(
                "FWD 2 diverges: {nd}/{} mismatches",
                eager_fwd2_bytes.len()
            ));
        }
        if !errs.is_empty() {
            return Err(errs.join("; "));
        }
        Ok(())
    }
}
