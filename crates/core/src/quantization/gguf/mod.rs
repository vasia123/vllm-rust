//! GGUF format support for loading quantized models.
//!
//! GGUF (GGML Universal Format) is the llama.cpp checkpoint format. This
//! module wires GGUF checkpoints into the engine by leaning entirely on
//! candle's `quantized` subsystem:
//!
//! - [`candle_core::quantized::gguf_file::Content`] parses the header
//!   (metadata + tensor table) and loads each tensor straight onto the
//!   target device as a [`QTensor`] — **quantized-resident**, no dense
//!   materialization.
//! - [`QMatMul`] runs the forward as candle's fused dequant+matmul
//!   (CUDA MMVQ for M≤8, q8_1 GEMM otherwise; a fused CPU `matmul_t`
//!   path). The weight never leaves quantized form, so an 8 GB GPU can
//!   hold models whose dense f16 weights would not fit, AND decode is
//!   fast — the old path dequantized the WHOLE weight to f32 every
//!   forward (see `benches/gguf_qmatmul_bench.rs`: ~29-45× slower at
//!   decode).
//!
//! Supported quant types are whatever candle's `GgmlDType` covers:
//! F32/F16/BF16 (unquantized), Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, and the
//! K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.
//!
//! # References
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub mod header;
pub mod iq;

use candle_core::quantized::gguf_file;
#[cfg(feature = "cuda-kernels")]
use candle_core::quantized::GgmlDType;
use candle_core::quantized::{QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

use self::header::{GgufHeader, GgufTensorDtype};
use self::iq::IqType;
use super::config::{QuantizationConfig, QuantizationMethod, QuantizedLinear};
use super::weight_loader::QuantizedWeightLoader;

/// GGUF quantization configuration.
#[derive(Debug, Clone, Default)]
pub struct GgufConfig {
    /// List of layers to skip quantization (use full precision)
    pub ignored_layers: Vec<String>,
}

impl GgufConfig {
    /// Create a new GGUF config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add layers to skip quantization.
    pub fn with_ignored_layers(mut self, layers: Vec<String>) -> Self {
        self.ignored_layers = layers;
        self
    }
}

impl QuantizationConfig for GgufConfig {
    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn supported_act_dtypes(&self) -> &[DType] {
        // GgufLinear casts activations to F32 at the QMatMul boundary, so
        // it accepts any of the common float activation dtypes.
        &[DType::F16, DType::BF16, DType::F32]
    }

    fn min_capability(&self) -> u32 {
        // GGUF can run on CPU (capability 0)
        0
    }

    fn is_layer_skipped(&self, layer_name: &str) -> bool {
        self.ignored_layers
            .iter()
            .any(|ignored| layer_name.contains(ignored))
    }

    fn create_linear(
        &self,
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Box<dyn QuantizedLinear>> {
        Ok(Box::new(GgufLinear::new(
            in_features,
            out_features,
            bias,
            device,
        )?))
    }

    fn clone_box(&self) -> Box<dyn QuantizationConfig> {
        Box::new(self.clone())
    }
}

/// GGUF quantized linear layer, backed by candle's [`QMatMul`].
///
/// The quantized weight stays resident on-device as a [`QTensor`]; the
/// forward delegates to candle's fused dequant+matmul kernel rather than
/// dequantizing the whole matrix to dense f32 first.
#[derive(Debug)]
pub struct GgufLinear {
    /// Fused quantized matmul over the on-device quantized weight.
    /// `None` until weights are installed (via the loader's `load_linear`
    /// or the trait `load_weights`); `forward` errors in that state.
    qmm: Option<Arc<QMatMul>>,
    /// Bias tensor, dense f32 (added after the matmul).
    bias: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
}

impl GgufLinear {
    /// Create a new GGUF linear layer with no weights yet installed.
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Self> {
        if in_features == 0 {
            candle_core::bail!("in_features must be non-zero");
        }
        if out_features == 0 {
            candle_core::bail!("out_features must be non-zero");
        }

        let bias = if has_bias {
            Some(Tensor::zeros(out_features, DType::F32, device)?)
        } else {
            None
        };

        Ok(Self {
            qmm: None,
            bias,
            in_features,
            out_features,
        })
    }

    /// Build a linear directly from a quantized matmul + optional bias.
    /// This is the path the GGUF loader uses — the weight is already a
    /// device-resident `QTensor` wrapped in `QMatMul`.
    pub fn from_qmatmul(
        qmm: Arc<QMatMul>,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            qmm: Some(qmm),
            bias,
            in_features,
            out_features,
        }
    }

    /// Install the quantized matmul after construction.
    pub fn set_qmatmul(&mut self, qmm: Arc<QMatMul>) {
        self.qmm = Some(qmm);
    }

    /// Check if weights are loaded.
    pub fn has_weights(&self) -> bool {
        self.qmm.is_some()
    }

    /// Core forward: cast to f32 at the kernel boundary, run the fused
    /// quantized matmul, add bias, restore the input dtype.
    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        let qmm = self.qmm.as_ref().ok_or_else(|| {
            candle_core::Error::Msg(
                "GGUF layer has no weights loaded - call load_weights() first".to_string(),
            )
        })?;

        // candle's quantized kernels are f32-in / f32-out. Cast and make
        // contiguous so the last-dim stride matches the kernel contract;
        // QMatMul handles arbitrary leading dims (collapses to rows).
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
        let y = qmm.forward(&x_f32)?;

        let y = match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        y.to_dtype(orig_dtype)
    }
}

impl QuantizedLinear for GgufLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x)
    }

    fn forward_pooled(
        &self,
        x: &crate::engine::output_pool::PooledTensor,
    ) -> Result<crate::engine::output_pool::PooledTensor> {
        // candle's `QMatMul` allocates a fresh output buffer (it is not
        // pool-backed). That is sound only OUTSIDE CUDA Graph capture —
        // a captured graph would record this transient pointer and
        // replay stale memory. GGUF models force `enforce_eager` (see the
        // server GGUF path), so capture never wraps this call; guard
        // loudly in case that invariant is ever broken.
        if crate::engine::output_pool::is_capturing() {
            candle_core::bail!(
                "GgufLinear::forward_pooled invoked under CUDA Graph capture; \
                 GGUF must run eager (enforce_eager)"
            );
        }
        let out = self.forward_inner(x.as_tensor())?;
        // SAFETY: not under capture (guarded above), so the fresh-alloc
        // pointer is never recorded/replayed by a captured graph.
        Ok(unsafe { crate::engine::output_pool::PooledTensor::from_pool_unchecked(out) })
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        // GGUF weights are quantized blocks loaded as `QTensor` by the
        // dedicated loader (`GgufWeightLoader::load_linear`), not dense
        // tensors in a state dict. This trait entry point is a no-op for
        // GGUF; the loader installs `qmm` directly.
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        // Compute dtype at the QMatMul boundary.
        DType::F32
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

/// A GGUF weight tensor loaded onto the device, kept quantized-resident.
///
/// candle-native types ride candle's [`QTensor`]; the I-quants candle lacks
/// (`IQ*`) keep their raw GGUF block bytes as a U8 [`Tensor`] plus the
/// [`IqType`] needed to dequantize them (see [`iq`]).
#[derive(Clone)]
pub enum GgufTensor {
    /// A candle-native quantized/dense tensor.
    Candle(Arc<QTensor>),
    /// An I-quant weight: raw block bytes (U8) + type + logical shape
    /// `[out, in]`.
    Iq {
        bytes: Tensor,
        iq: IqType,
        shape: Vec<usize>,
    },
}

impl GgufTensor {
    /// Dequantize the whole tensor to a dense f32 [`Tensor`] in its logical
    /// shape. Used for non-linear weights pulled through the VarBuilder and
    /// for bias tensors.
    fn dequantize_dense(&self, device: &Device) -> Result<Tensor> {
        match self {
            GgufTensor::Candle(qt) => qt.dequantize(device),
            GgufTensor::Iq { bytes, iq, shape } => {
                let n: usize = shape.iter().product();
                iq::dequantize_iq(bytes, *iq, n)?.reshape(shape.clone())
            }
        }
    }
}

/// GGUF linear layer over an I-quant weight (candle has no I-quant matmul).
///
/// The weight stays I-quant-resident on the device as raw block bytes; the
/// forward dequantizes it to a dense f32 matrix (`kernels/iq_dequant.cu` on
/// CUDA, the scalar port on CPU) and matmuls. This is the simple, correct
/// dequant-then-matmul path — a fused MMVQ kernel that skips materializing
/// the dense weight is a later perf optimisation (tracked separately).
pub struct IqLinear {
    /// Raw I-quant block bytes (U8) on the compute device.
    bytes: Tensor,
    iq: IqType,
    /// Bias, dense f32 (added after the matmul); `None` for most GGUF models.
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl IqLinear {
    /// Build from the device-resident I-quant bytes + logical shape
    /// `[out, in]`.
    pub fn new(
        bytes: Tensor,
        iq: IqType,
        out_features: usize,
        in_features: usize,
        bias: Option<Tensor>,
    ) -> Self {
        Self {
            bytes,
            iq,
            bias,
            in_features,
            out_features,
        }
    }

    fn forward_inner(&self, x: &Tensor) -> Result<Tensor> {
        // y = x @ Wᵀ. candle's quantized kernels are f32; match that here so
        // the result is numerically consistent with the GgufLinear path.
        // Collapse leading dims to a 2-D [M, in] matmul, then restore.
        let orig_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?.contiguous()?;
        let dims = x_f32.dims().to_vec();
        let in_f = *dims.last().unwrap_or(&0);
        if in_f != self.in_features {
            candle_core::bail!(
                "IqLinear: input dim {in_f} != in_features {}",
                self.in_features
            );
        }
        let m: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        let x_2d = x_f32.reshape((m, self.in_features))?;

        let y = if x_2d.device().is_cuda() && m <= iq::IQ_GEMV_MAX_M {
            // Decode (small M): fused GEMV — dot the I-quant blocks against the
            // activation directly, no dense weight materialized. One pass over
            // the weight bytes.
            iq::iq_matmul(
                &self.bytes,
                &x_2d,
                self.iq,
                self.out_features,
                self.in_features,
                m,
            )?
        } else {
            // Prefill (large M) / CPU: dequantize the whole weight to dense f32
            // once, then a single cuBLAS/candle GEMM amortizes it over all rows.
            let n = self.out_features * self.in_features;
            let w = iq::dequantize_iq(&self.bytes, self.iq, n)?
                .reshape((self.out_features, self.in_features))?;
            x_2d.matmul(&w.t()?.contiguous()?)?
        };

        let mut out_dims = dims;
        if let Some(last) = out_dims.last_mut() {
            *last = self.out_features;
        }
        let y = y.reshape(out_dims)?;
        let y = match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        y.to_dtype(orig_dtype)
    }
}

impl QuantizedLinear for IqLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x)
    }

    fn forward_pooled(
        &self,
        x: &crate::engine::output_pool::PooledTensor,
    ) -> Result<crate::engine::output_pool::PooledTensor> {
        // Like GgufLinear: the dequant + matmul allocates fresh buffers (not
        // pool-backed), sound only OUTSIDE CUDA Graph capture. GGUF forces
        // enforce_eager, so capture never wraps this; guard loudly otherwise.
        if crate::engine::output_pool::is_capturing() {
            candle_core::bail!(
                "IqLinear::forward_pooled invoked under CUDA Graph capture; \
                 GGUF must run eager (enforce_eager)"
            );
        }
        let out = self.forward_inner(x.as_tensor())?;
        // SAFETY: not under capture (guarded above), so the fresh-alloc
        // pointer is never recorded/replayed by a captured graph.
        Ok(unsafe { crate::engine::output_pool::PooledTensor::from_pool_unchecked(out) })
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        // I-quant weights are installed by the GGUF loader, not via a state
        // dict — no-op here (mirrors GgufLinear).
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        DType::F32
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

/// Warn once if `CANDLE_DEQUANTIZE_ALL` is set: candle's
/// `QMatMul::from_arc` then silently dequantizes every weight to dense
/// f32, which defeats quantized residency and can OOM an 8 GB GPU on a
/// model that otherwise fits (e.g. Gemma 4 E4B's PLE table).
fn warn_if_dequantize_all() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if let Ok(v) = std::env::var("CANDLE_DEQUANTIZE_ALL") {
            if !v.is_empty() && v != "0" {
                tracing::warn!(
                    "CANDLE_DEQUANTIZE_ALL={v} is set — GGUF weights will be \
                     dequantized to dense f32, losing quantized residency. \
                     This may OOM on memory-constrained GPUs. Unset it for \
                     the intended quantized-resident GGUF path."
                );
            }
        }
    });
}

/// Backing store for a [`QuantizedEmbedding`].
enum EmbTable {
    /// CPU-resident `QTensor`; gather by slicing the row's block bytes and
    /// dequantizing the small slice. Works for any GGUF quant type.
    CpuQTensor(Arc<QTensor>),
    /// GPU-resident raw Q6_K block bytes (U8 `Tensor`, row-major over
    /// `[num_embeddings, embedding_dim]`); gather via the
    /// `gather_dequant_q6k` CUDA kernel. No host residency, no full-table
    /// dequant.
    GpuQ6kBytes(Tensor),
}

/// Quantized embedding table that materializes only the rows a forward
/// selects, instead of dequantizing the whole `[num_embeddings,
/// embedding_dim]` table.
///
/// This exists for Gemma 4's Per-Layer Embedding table: dense f16 it is
/// ~5.6 GB ([262144, 42·256]) and would OOM an 8 GB GPU at construction if
/// served through candle's `embedding()` (which materializes the whole
/// weight via `VarBuilder::get`). Two backings:
///
/// - [`EmbTable::GpuQ6kBytes`] (production): the Q6_K table stays
///   quantized-resident in VRAM (~2.3 GB) and a forward runs the
///   `gather_dequant_q6k` kernel over just the batch's rows.
/// - [`EmbTable::CpuQTensor`] (fallback / CPU / tests): the table is a
///   CPU `QTensor`; a forward slices the row blocks and dequantizes the
///   slice. Row `r` is a contiguous run of `embedding_dim / block_size`
///   blocks (Gemma PLE row = 10752/256 = 42 Q6_K blocks, exact block
///   alignment), so a gather is a byte-range copy + small dequant.
pub struct QuantizedEmbedding {
    table: EmbTable,
    num_embeddings: usize,
    embedding_dim: usize,
    /// Output dtype/device for the gathered dense rows (the compute side).
    out_dtype: DType,
    out_device: Device,
}

impl QuantizedEmbedding {
    /// CPU-backed embedding from a 2-D `QTensor`. `embedding_dim` must be a
    /// multiple of the quant block size (true for GGUF embeddings).
    pub fn new(qt: Arc<QTensor>, out_dtype: DType, out_device: Device) -> Result<Self> {
        let dims = qt.shape().dims().to_vec();
        let (num_embeddings, embedding_dim) = match dims.as_slice() {
            [n, d] => (*n, *d),
            _ => candle_core::bail!("QuantizedEmbedding expects a 2-D table, got {dims:?}"),
        };
        if embedding_dim % qt.dtype().block_size() != 0 {
            candle_core::bail!(
                "QuantizedEmbedding: embedding_dim {embedding_dim} not a multiple of block size {}",
                qt.dtype().block_size()
            );
        }
        Ok(Self {
            table: EmbTable::CpuQTensor(qt),
            num_embeddings,
            embedding_dim,
            out_dtype,
            out_device,
        })
    }

    /// GPU-backed Q6_K embedding from raw block bytes (U8 `Tensor` on
    /// CUDA), gathered via the custom kernel. The table never goes dense
    /// and never touches host RAM.
    pub fn new_gpu_q6k(
        bytes: Tensor,
        num_embeddings: usize,
        embedding_dim: usize,
        out_dtype: DType,
        out_device: Device,
    ) -> Result<Self> {
        if !embedding_dim.is_multiple_of(256) {
            candle_core::bail!(
                "QuantizedEmbedding(Q6_K): embedding_dim {embedding_dim} not a multiple of 256"
            );
        }
        Ok(Self {
            table: EmbTable::GpuQ6kBytes(bytes),
            num_embeddings,
            embedding_dim,
            out_dtype,
            out_device,
        })
    }

    /// Number of embedding rows.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Gather rows for `ids` (any shape) → `[*ids.shape, embedding_dim]` in
    /// `out_dtype` on `out_device`. Materializes only the gathered rows.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match &self.table {
            EmbTable::CpuQTensor(qt) => self.forward_cpu(qt, ids),
            EmbTable::GpuQ6kBytes(bytes) => self.forward_gpu_q6k(bytes, ids),
        }
    }

    fn forward_cpu(&self, qt: &Arc<QTensor>, ids: &Tensor) -> Result<Tensor> {
        let block = qt.dtype().block_size();
        let tsize = qt.dtype().type_size();
        let blocks_per_row = self.embedding_dim / block;
        let row_bytes = blocks_per_row * tsize;

        let mut out_shape = ids.dims().to_vec();
        let flat = ids.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()?;

        // `data()` on a CPU-resident QTensor borrows the block bytes (no
        // copy); gathering is a byte-range memcpy per row.
        let data = qt.data()?;
        let mut buf: Vec<u8> = Vec::with_capacity(flat.len() * row_bytes);
        for &id in &flat {
            let id = id as usize;
            if id >= self.num_embeddings {
                candle_core::bail!(
                    "QuantizedEmbedding: id {id} out of range (num_embeddings={})",
                    self.num_embeddings
                );
            }
            let start = id * row_bytes;
            buf.extend_from_slice(&data[start..start + row_bytes]);
        }

        // Dequantize the gathered rows only (CPU), then cast + move.
        //
        // NOTE: `Cow::Borrowed`, not `Cow::Owned`. candle's internal
        // `as_t_slice` takes the `Cow` by value and returns a `&[Block]`
        // into it; with `Cow::Owned` the buffer drops when that helper
        // returns and the subsequent `.to_vec()` reads freed memory
        // (observed as garbage rows). Borrowing keeps `buf` (this scope's
        // local) alive across the call so the slice stays valid.
        let storage = QStorage::from_data(Cow::Borrowed(&buf), &Device::Cpu, qt.dtype())?;
        let gathered = QTensor::new(storage, (flat.len(), self.embedding_dim))?;
        let dense = gathered.dequantize(&Device::Cpu)?;

        out_shape.push(self.embedding_dim);
        dense
            .reshape(out_shape)?
            .to_dtype(self.out_dtype)?
            .to_device(&self.out_device)
    }

    #[cfg(feature = "cuda-kernels")]
    fn forward_gpu_q6k(&self, bytes: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let mut out_shape = ids.dims().to_vec();
        let dense = crate::quantization::gguf_cuda::gather_dequant_q6k(
            bytes,
            ids,
            self.num_embeddings,
            self.embedding_dim,
        )?;
        out_shape.push(self.embedding_dim);
        dense
            .reshape(out_shape)?
            .to_dtype(self.out_dtype)?
            .to_device(&self.out_device)
    }

    #[cfg(not(feature = "cuda-kernels"))]
    fn forward_gpu_q6k(&self, _bytes: &Tensor, _ids: &Tensor) -> Result<Tensor> {
        candle_core::bail!("QuantizedEmbedding GPU Q6_K path requires the cuda-kernels feature")
    }
}

/// Weight loader for GGUF files.
///
/// Holds the parsed [`GgufHeader`] (metadata + tensor table — our own
/// parser, since candle's cannot read I-quant files) and a one-shot map of
/// every tensor loaded onto the target device. candle-native linears wrap a
/// `QTensor` in a [`QMatMul`]; I-quant linears use [`IqLinear`];
/// norm/embedding tensors are dequantized on demand by
/// [`GgufVarBuilderBackend`].
pub struct GgufWeightLoader {
    /// Parsed GGUF header (metadata + tensor infos, I-quant-aware).
    header: Arc<GgufHeader>,
    /// Every tensor, loaded once onto `device`, kept quantized-resident.
    /// Large embedding tables routed to a [`QuantizedEmbedding`] (see
    /// [`EMBED_TABLES`]) are NOT in here — they would waste VRAM as unused
    /// tensors; they are read from the file on demand instead.
    tensors: Arc<HashMap<String, GgufTensor>>,
    /// Path to the GGUF file, for on-demand reads of skipped embedding
    /// tables.
    path: std::path::PathBuf,
    /// Device to load tensors to
    device: Device,
    /// Compute dtype for activations
    dtype: DType,
    /// Config
    #[allow(dead_code)]
    config: GgufConfig,
}

/// GGUF tensor names whose tables are huge embeddings served through a
/// [`QuantizedEmbedding`] (per-row gather) rather than a dense QTensor —
/// so they are skipped during the bulk load to avoid wasting VRAM/host
/// RAM on a copy that is never used as a dense tensor.
const EMBED_TABLES: &[&str] = &["per_layer_token_embd.weight"];

impl GgufWeightLoader {
    /// Open a GGUF file, parse its header, and load every tensor onto
    /// `device` — candle-native types as a quantized-resident `QTensor`,
    /// I-quants as raw block bytes (see [`GgufTensor`]).
    pub fn from_path(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        warn_if_dequantize_all();

        let header = {
            let f = std::fs::File::open(path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to open GGUF file: {e}")))?;
            GgufHeader::read(std::io::BufReader::new(f))
                .map_err(|e| candle_core::Error::Msg(format!("Failed to parse GGUF header: {e}")))?
        };

        // Single pass: load each tensor onto the device, quantized. Large
        // embedding tables (EMBED_TABLES) are skipped — they are served by
        // a `QuantizedEmbedding` that reads them from the file on demand,
        // so a dense copy here would only waste memory.
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(path)
                .map_err(|e| candle_core::Error::Msg(format!("reopen GGUF: {e}")))?,
        );
        let mut tensors: HashMap<String, GgufTensor> =
            HashMap::with_capacity(header.tensor_infos.len());
        for (name, info) in &header.tensor_infos {
            if EMBED_TABLES.contains(&name.as_str()) {
                continue;
            }
            let bytes = header.read_tensor_bytes(&mut reader, name).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to read tensor {name}: {e}"))
            })?;
            let tensor = match info.dtype {
                GgufTensorDtype::Candle(d) => {
                    // `Cow::Borrowed`, NOT `Cow::Owned`: candle's internal
                    // `as_t_slice` takes the `Cow` by value and returns a
                    // `&[Block]` into it via `from_raw_parts`; with
                    // `Cow::Owned` the buffer drops when that helper returns
                    // and `load_quantized` then reads freed memory (SIGSEGV).
                    // Borrowing keeps `bytes` (this scope's local) alive
                    // across the upload. Same hazard as `QuantizedEmbedding`.
                    let storage = QStorage::from_data(Cow::Borrowed(&bytes), &device, d)?;
                    let qt = QTensor::new(storage, info.shape.clone()).map_err(|e| {
                        candle_core::Error::Msg(format!("Failed to load tensor {name}: {e}"))
                    })?;
                    GgufTensor::Candle(Arc::new(qt))
                }
                GgufTensorDtype::Iq(iq) => {
                    let n_bytes = bytes.len();
                    let bytes_t = Tensor::from_vec(bytes, (n_bytes,), &device)?;
                    GgufTensor::Iq {
                        bytes: bytes_t,
                        iq,
                        shape: info.shape.clone(),
                    }
                }
            };
            tensors.insert(name.clone(), tensor);
        }

        Ok(Self {
            header: Arc::new(header),
            tensors: Arc::new(tensors),
            path: path.to_path_buf(),
            device,
            dtype,
            config: GgufConfig::default(),
        })
    }

    /// Read a tensor's raw quantized bytes straight from the GGUF file
    /// (used for embedding tables skipped during the bulk load).
    fn read_tensor_bytes(&self, gguf_name: &str) -> Result<Vec<u8>> {
        let mut f = std::io::BufReader::new(
            std::fs::File::open(&self.path)
                .map_err(|e| candle_core::Error::Msg(format!("reopen GGUF: {e}")))?,
        );
        self.header.read_tensor_bytes(&mut f, gguf_name)
    }

    /// Shared handle to the device-resident tensor map, so a `VarBuilder`
    /// backend can resolve non-quantized (dense) tensors against the same
    /// data.
    pub fn shared_tensors(&self) -> Arc<HashMap<String, GgufTensor>> {
        Arc::clone(&self.tensors)
    }

    /// Look up a metadata string value.
    fn meta_str(&self, key: &str) -> Option<String> {
        self.header
            .metadata
            .get(key)
            .and_then(|v| v.to_string().ok())
            .cloned()
    }

    /// Look up a metadata integer (upcasts U8/U16/U32 → u64).
    fn meta_u64(&self, key: &str) -> Option<u64> {
        self.header.metadata.get(key).and_then(|v| v.to_u64().ok())
    }

    /// End-of-generation token ids: the metadata `eos_token_id` plus any
    /// chat end-of-turn markers present in the vocab. GGUF rarely stores an
    /// explicit EOT id, so — like llama.cpp — we derive it by looking up the
    /// marker strings in `tokenizer.ggml.tokens` (index == token id). Lets
    /// the server stop on `<end_of_turn>` (Gemma-instruct) etc. instead of
    /// leaking it into the output.
    fn eog_token_ids(&self) -> Vec<u32> {
        const EOT_MARKERS: &[&str] = &[
            "<end_of_turn>",
            "<|im_end|>",
            "<|eot_id|>",
            "<|end|>",
            "<|endoftext|>",
        ];
        let mut ids: Vec<u32> = Vec::new();
        if let Some(eos) = self.meta_u64("tokenizer.ggml.eos_token_id") {
            ids.push(eos as u32);
        }
        let tokens = self
            .header
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok());
        if let Some(tokens) = &tokens {
            for (id, tok) in tokens.iter().enumerate() {
                if let Ok(s) = tok.to_string() {
                    if EOT_MARKERS.contains(&s.as_str()) && !ids.contains(&(id as u32)) {
                        ids.push(id as u32);
                    }
                }
            }
        }
        // Gemma's end-of-turn is canonically token 106. Some GGUF converters
        // mangle its string (this E4B checkpoint stores it as "<turn|>", which
        // the marker search above misses — and llama.cpp flags it too), so add
        // it by id for any gemma architecture, mirroring llama.cpp treating
        // 106 as an EOG token for this model.
        if matches!(self.architecture().as_deref(), Some(a) if a.starts_with("gemma")) {
            const GEMMA_END_OF_TURN: u32 = 106;
            let in_vocab = tokens
                .as_ref()
                .is_none_or(|t| GEMMA_END_OF_TURN < t.len() as u32);
            if in_vocab && !ids.contains(&GEMMA_END_OF_TURN) {
                ids.push(GEMMA_END_OF_TURN);
            }
        }
        ids
    }

    /// Look up a metadata float (F32 or F64).
    fn meta_f64(&self, key: &str) -> Option<f64> {
        self.header
            .metadata
            .get(key)
            .and_then(|v| v.to_f32().ok().map(f64::from).or_else(|| v.to_f64().ok()))
    }

    /// The GGUF architecture string (`general.architecture`).
    pub fn architecture(&self) -> Option<String> {
        self.meta_str("general.architecture")
    }

    /// Map the GGUF architecture short-hand to the vLLM-style
    /// architecture list used by `ModelConfig`.
    pub fn vllm_architecture(&self) -> Option<Vec<String>> {
        let arch = self.architecture()?;
        let mapped = match arch.as_str() {
            "llama" => "LlamaForCausalLM",
            "gemma" => "GemmaForCausalLM",
            "gemma2" => "Gemma2ForCausalLM",
            "gemma3" => "Gemma3ForCausalLM",
            "gemma4" => "Gemma4ForCausalLM",
            "qwen2" => "Qwen2ForCausalLM",
            "qwen3" => "Qwen3ForCausalLM",
            "mistral" => "MistralForCausalLM",
            "phi3" => "Phi3ForCausalLM",
            other => {
                tracing::warn!("unknown GGUF architecture '{other}', falling back verbatim");
                return Some(vec![other.to_string()]);
            }
        };
        Some(vec![mapped.to_string()])
    }

    /// Build a `ModelConfig` from GGUF metadata (no external
    /// `config.json` needed). Pulls the standard llama.cpp keys
    /// (`{arch}.embedding_length`, `{arch}.block_count`, …) and falls
    /// back to conservative defaults for anything absent.
    pub fn build_model_config(&self) -> Result<crate::config::ModelConfig> {
        let arch_name = self
            .architecture()
            .ok_or_else(|| candle_core::Error::Msg("GGUF missing general.architecture".into()))?;

        let hidden_size = self
            .meta_u64(&format!("{arch_name}.embedding_length"))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.embedding_length"))
            })? as usize;
        let num_hidden_layers = self
            .meta_u64(&format!("{arch_name}.block_count"))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.block_count"))
            })? as usize;
        let num_attention_heads = self
            .meta_u64(&format!("{arch_name}.attention.head_count"))
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("GGUF missing {arch_name}.attention.head_count"))
            })? as usize;
        let num_key_value_heads = self
            .meta_u64(&format!("{arch_name}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = self
            .meta_u64(&format!("{arch_name}.feed_forward_length"))
            .unwrap_or(4 * hidden_size as u64) as usize;
        let max_position_embeddings = self
            .meta_u64(&format!("{arch_name}.context_length"))
            .unwrap_or(4096) as usize;
        let rms_norm_eps = self
            .meta_f64(&format!("{arch_name}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let rope_theta = self
            .meta_f64(&format!("{arch_name}.rope.freq_base"))
            .unwrap_or(10000.0);

        let vocab_size = self
            .header
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| arr.len())
            .ok_or_else(|| {
                candle_core::Error::Msg("GGUF missing tokenizer.ggml.tokens array".into())
            })?;

        // head_dim: the DEFAULT (sliding-attention) head dim. Derive from
        // layer 0's q_proj width (out / num_heads) — robust where head_dim
        // != hidden/heads (Gemma 4 E4B: 2560/8 = 320 naive, real 256).
        //
        // Gemma 4 is heterogeneous: full-attention layers use a LARGER head
        // dim (E4B sliding 256, full 512) — that is `global_head_dim`, set
        // from `{arch}.attention.key_length` in `gemma4_config_extras`,
        // while this `head_dim` is the sliding one (`key_length_swa`). Layer
        // 0 is a sliding layer in E4B, so the q_proj derivation yields 256.
        let head_dim = self
            .qtensor_out_dim("model.layers.0.self_attn.q_proj")
            .filter(|_| num_attention_heads > 0)
            .map(|out| out / num_attention_heads)
            .unwrap_or_else(|| {
                if num_attention_heads > 0 {
                    hidden_size / num_attention_heads
                } else {
                    hidden_size
                }
            });

        // Gemma 4 stores the GLOBAL (full-attention) attention params in
        // metadata (`head_count_kv`, `key_length`); sliding layers can carry a
        // DIFFERENT kv-head count (12B: sliding 8 × 256, full 1 × 512). Derive
        // the sliding kv-head count from layer 0's k_proj (a sliding layer):
        // kv = k_out / head_dim. The metadata `head_count_kv` is the GLOBAL
        // value, surfaced as `num_global_key_value_heads` in
        // `gemma4_config_extras`. E4B is non-heterogeneous in kv heads, so this
        // derivation yields the same value the metadata would.
        let num_key_value_heads = if arch_name == "gemma4" && head_dim > 0 {
            self.qtensor_out_dim("model.layers.0.self_attn.k_proj")
                .map(|k_out| k_out / head_dim)
                .filter(|&kv| kv > 0)
                .unwrap_or(num_key_value_heads)
        } else {
            num_key_value_heads
        };

        let architectures = self
            .vllm_architecture()
            .unwrap_or_else(|| vec!["LlamaForCausalLM".to_string()]);

        // Gemma 4 needs PLE / sliding / soft-cap / per-attn-type RoPE
        // settings in `extra` to activate its specialised forward; other
        // architectures use an empty `extra`.
        let (sliding_window, mut extra) = if arch_name == "gemma4" {
            self.gemma4_config_extras(vocab_size)
        } else if arch_name == "gemma3" {
            self.gemma3_config_extras()
        } else {
            (None, serde_json::Map::new())
        };

        // End-of-generation token set: the metadata `eos_token_id` plus any
        // chat end-of-turn markers in the vocab. Gemma-instruct ends a turn
        // with `<end_of_turn>` (106), NOT `<eos>` (1), so a single eos lets
        // `<end_of_turn>` leak into the output. The server reads
        // `extra["eos_token_ids"]` into its additional-EOS stop set.
        let eos_token_ids = self.eog_token_ids();
        if eos_token_ids.len() > 1 {
            extra.insert(
                "eos_token_ids".to_string(),
                serde_json::json!(eos_token_ids),
            );
        }

        // Whether the tokenizer prepends BOS / appends EOS. EmbeddingGemma
        // needs both (`[bos] … [eos]`) to match the reference; our HTTP
        // tokenizer doesn't add them, so the embedding path applies them from
        // these flags. Mirrors `tokenizer.ggml.add_{bos,eos}_token`.
        if let Some(add_bos) = self
            .header
            .metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok())
        {
            extra.insert("add_bos_token".to_string(), serde_json::json!(add_bos));
        }
        if let Some(add_eos) = self
            .header
            .metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.to_bool().ok())
        {
            extra.insert("add_eos_token".to_string(), serde_json::json!(add_eos));
        }

        // Embedded chat template + the bos/eos token STRINGS, so the server
        // can render `/v1/chat/completions` without an external
        // `tokenizer_config.json`. The server uses this only as the
        // last-resort source (it prefers `--chat-template` / sibling files);
        // the strings feed `{{ bos_token }}` / `{{ eos_token }}` in the
        // template.
        if let Some(tmpl) = self.meta_str("tokenizer.chat_template") {
            extra.insert("chat_template".to_string(), serde_json::json!(tmpl));
        }
        if let Some(tokens) = self
            .header
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
        {
            let tok_str = |id: Option<u64>| -> Option<String> {
                id.and_then(|i| tokens.get(i as usize))
                    .and_then(|v| v.to_string().ok())
                    .cloned()
            };
            if let Some(s) = tok_str(self.meta_u64("tokenizer.ggml.bos_token_id")) {
                extra.insert("bos_token".to_string(), serde_json::json!(s));
            }
            if let Some(s) = tok_str(self.meta_u64("tokenizer.ggml.eos_token_id")) {
                extra.insert("eos_token".to_string(), serde_json::json!(s));
            }
        }

        Ok(crate::config::ModelConfig {
            architectures,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            head_dim,
            hidden_act: "silu".to_string(),
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings: true,
            // Read the real special-token ids from GGUF metadata. These are
            // arch-specific (Llama bos=1/eos=2, but Gemma bos=2/eos=1), so a
            // hardcoded default silently feeds Gemma `<eos>` as its leading
            // token — and Gemma is famously degenerate without a correct
            // `<bos>` (the model loses the thread and re-emits turn markers).
            // Fall back to the Llama-style defaults only when the keys are
            // absent.
            bos_token_id: self
                .meta_u64("tokenizer.ggml.bos_token_id")
                .map(|v| v as u32)
                .or(Some(1)),
            eos_token_id: self
                .meta_u64("tokenizer.ggml.eos_token_id")
                .map(|v| v as u32)
                .or(Some(2)),
            sliding_window,
            attention_bias: Some(false),
            extra,
        })
    }

    /// First (outermost) dimension of a tensor resolved from a vLLM-style
    /// prefix — the output feature count for a linear weight. candle's
    /// `Content` reverses the on-disk dims, so the stored shape is
    /// `[out, in]` and `dims()[0]` is `out`.
    fn qtensor_out_dim(&self, prefix: &str) -> Option<usize> {
        let mut names = vec![format!("{prefix}.weight"), prefix.to_string()];
        if let Some(m) = to_llama_cpp_prefix(prefix) {
            names.push(format!("{m}.weight"));
            names.push(m);
        }
        names
            .iter()
            .find_map(|n| self.header.tensor_infos.get(n))
            .and_then(|info| info.shape.first().copied())
    }

    /// Populate the Gemma-4-specific `ModelConfig.extra` keys from GGUF
    /// metadata so `Gemma4ExtraConfig::from_model_config` activates the
    /// PLE path, sliding-window layers, soft-capping and per-attention-type
    /// RoPE. Returns `(sliding_window, extra)`. Confirmed against the real
    /// `gemma-4-E4B-it-Q4_K_M.gguf` header.
    fn gemma4_config_extras(
        &self,
        vocab_size: usize,
    ) -> (Option<usize>, serde_json::Map<String, serde_json::Value>) {
        use serde_json::{json, Value};
        let arch = "gemma4";
        let mut extra: serde_json::Map<String, Value> = serde_json::Map::new();

        // Per-Layer Embeddings.
        if let Some(ple) = self.meta_u64(&format!("{arch}.embedding_length_per_layer_input")) {
            extra.insert("hidden_size_per_layer_input".into(), json!(ple));
        }
        extra.insert("vocab_size_per_layer_input".into(), json!(vocab_size));

        // KV-cache sharing: the last N layers reuse an earlier layer's KV.
        if let Some(shared) = self.meta_u64(&format!("{arch}.attention.shared_kv_layers")) {
            extra.insert("num_kv_shared_layers".into(), json!(shared));
        }

        // Heterogeneous head_dim: full-attention layers use a larger head
        // dim than sliding layers (E4B: sliding 256 = key_length_swa, full
        // 512 = key_length). `global_head_dim` drives the full-layer q/k/v
        // and q_norm/k_norm sizing plus the per-layer KV cache geometry;
        // the typed `head_dim` is the sliding one. Only meaningful when it
        // differs from the sliding head dim.
        if let Some(full_hd) = self.meta_u64(&format!("{arch}.attention.key_length")) {
            extra.insert("global_head_dim".into(), json!(full_hd));
        }

        // Heterogeneous kv-head count: in the 12B `head_count_kv` is a
        // PER-LAYER array (`[8,8,8,8,8,1,…]` — sliding 8, full 1), not a
        // scalar, so `meta_u64` can't read it. Read the array and surface the
        // full-attention (global) kv-head count — the smaller value, since
        // full/global attention uses no more kv heads than sliding. The model
        // applies it to full layers via `num_global_key_value_heads`; sliding
        // layers use the derived `num_key_value_heads`. A scalar `head_count_kv`
        // (non-heterogeneous models) is handled by `num_key_value_heads` alone.
        if let Some(gguf_file::Value::Array(arr)) = self
            .header
            .metadata
            .get(&format!("{arch}.attention.head_count_kv"))
        {
            // The array elements are I32 here; candle's `to_i64`/`to_u64` are
            // strict (I64 / unsigned only) and reject I32, so match the
            // integer variants explicitly.
            let as_int = |v: &gguf_file::Value| -> Option<i64> {
                match v {
                    gguf_file::Value::U8(x) => Some(*x as i64),
                    gguf_file::Value::I8(x) => Some(*x as i64),
                    gguf_file::Value::U16(x) => Some(*x as i64),
                    gguf_file::Value::I16(x) => Some(*x as i64),
                    gguf_file::Value::U32(x) => Some(*x as i64),
                    gguf_file::Value::I32(x) => Some(*x as i64),
                    gguf_file::Value::U64(x) => Some(*x as i64),
                    gguf_file::Value::I64(x) => Some(*x),
                    _ => None,
                }
            };
            let global_kv = arr.iter().filter_map(as_int).filter(|&x| x > 0).min();
            if let Some(g) = global_kv {
                extra.insert("num_global_key_value_heads".into(), json!(g));
            }
        }

        // Soft-capping.
        if let Some(cap) = self.meta_f64(&format!("{arch}.final_logit_softcapping")) {
            extra.insert("final_logit_softcapping".into(), json!(cap));
        }
        if let Some(cap) = self.meta_f64(&format!("{arch}.attn_logit_softcapping")) {
            extra.insert("attn_logit_softcapping".into(), json!(cap));
        }

        // Per-layer attention type from the sliding-window bool pattern
        // (true → sliding_attention, false → full_attention). Mirrors the
        // HF `layer_types` array the model consumes directly.
        if let Some(gguf_file::Value::Array(pat)) = self
            .header
            .metadata
            .get(&format!("{arch}.attention.sliding_window_pattern"))
        {
            let layer_types: Vec<Value> = pat
                .iter()
                .map(|v| {
                    // candle's `Value::to_bool` is strict (Bool only).
                    let sliding = v.to_bool().unwrap_or(true);
                    Value::String(
                        if sliding {
                            "sliding_attention"
                        } else {
                            "full_attention"
                        }
                        .to_string(),
                    )
                })
                .collect();
            let full = layer_types
                .iter()
                .filter(|v| v.as_str() == Some("full_attention"))
                .count();
            tracing::info!(
                "Gemma 4 GGUF: {} layers, {full} full-attention (rest sliding)",
                layer_types.len()
            );
            extra.insert("layer_types".into(), Value::Array(layer_types));

            // K-as-V: some Gemma 4 checkpoints (e.g. the 12B) omit `attn_v`
            // on full-attention layers — there V is the K projection output
            // (llama.cpp `gemma4.cpp`: `Vcur = wv ? wv@x : Kcur`). Detect a
            // full-attention layer with no `attn_v.weight` and flip the flag
            // that activates the model's existing K-as-V path; otherwise the
            // loader requests a v_proj that does not exist and load fails.
            let k_eq_v = pat.iter().enumerate().any(|(i, v)| {
                let full = !v.to_bool().unwrap_or(true);
                full && !self
                    .header
                    .tensor_infos
                    .contains_key(&format!("blk.{i}.attn_v.weight"))
            });
            if k_eq_v {
                extra.insert("attention_k_eq_v".into(), json!(true));
                tracing::info!(
                    "Gemma 4 GGUF: full-attention layers omit attn_v → V = K projection"
                );
            }
        }

        // Per-attention-type RoPE. Full-attention layers use a PARTIAL,
        // "proportional" rotary (Gemma 4 architectural constants:
        // partial_rotary_factor 0.25, rope_type "proportional"); sliding
        // layers use the default full rotary. The GGUF header carries only
        // the freq bases (freq_base / freq_base_swa), NOT the partial
        // factor or rope type, so those are supplied from the known Gemma 4
        // spec — without them the full layers rotate the wrong number of
        // dims and decode is incoherent.
        let full = self
            .meta_f64(&format!("{arch}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let swa = self
            .meta_f64(&format!("{arch}.rope.freq_base_swa"))
            .unwrap_or(10_000.0);
        let mut rope = serde_json::Map::new();
        rope.insert(
            "full_attention".into(),
            json!({
                "rope_theta": full,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            }),
        );
        rope.insert(
            "sliding_attention".into(),
            json!({ "rope_theta": swa, "rope_type": "default" }),
        );
        extra.insert("rope_parameters".into(), Value::Object(rope));

        let sliding_window = self
            .meta_u64(&format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize);

        (sliding_window, extra)
    }

    /// Populate the Gemma-3-specific `ModelConfig.extra`. Besides the usual
    /// chat-model settings (soft-caps, local RoPE), this detects the
    /// **EmbeddingGemma** variant: a bidirectional Gemma3 encoder
    /// (`gemma3.attention.causal == false`) with mean pooling
    /// (`gemma3.pooling_type`) and a sentence-transformers projector
    /// (`dense.0`/`dense.1` weights). Those three keys flip the quantized
    /// model into embedding mode. Returns `(sliding_window, extra)`.
    fn gemma3_config_extras(&self) -> (Option<usize>, serde_json::Map<String, serde_json::Value>) {
        use serde_json::{json, Value};
        let arch = "gemma3";
        let mut extra: serde_json::Map<String, Value> = serde_json::Map::new();

        // llama.cpp's GGUF conversion adds 1 to every `*norm.weight`, so the
        // Gemma `(1 + weight)` scaling is already baked in — apply norms plainly.
        extra.insert("gemma_norm_prefolded".into(), json!(true));

        // Soft-capping (Gemma 3 chat models set these; embedding models don't).
        if let Some(cap) = self.meta_f64(&format!("{arch}.final_logit_softcapping")) {
            extra.insert("final_logit_softcapping".into(), json!(cap));
        }
        if let Some(cap) = self.meta_f64(&format!("{arch}.attn_logit_softcapping")) {
            extra.insert("attn_logit_softcapping".into(), json!(cap));
        }

        // Separate local (sliding-layer) RoPE base.
        if let Some(swa) = self.meta_f64(&format!("{arch}.rope.freq_base_swa")) {
            extra.insert("rope_local_base_freq".into(), json!(swa));
        }

        // Per-layer sliding/global pattern. Gemma 3 interleaves sliding-window
        // and full-attention layers (5:1), each with its own RoPE base. The
        // GGUF stores a per-layer bool array (true = sliding); index it
        // directly rather than guessing a stride.
        if let Some(gguf_file::Value::Array(pat)) = self
            .header
            .metadata
            .get(&format!("{arch}.attention.sliding_window_pattern"))
        {
            let flags: Vec<Value> = pat
                .iter()
                .map(|v| Value::Bool(v.to_bool().unwrap_or(true)))
                .collect();
            if !flags.is_empty() {
                let sliding = flags.iter().filter(|v| v.as_bool() == Some(true)).count();
                tracing::info!(
                    "Gemma 3 GGUF: {} layers, {sliding} sliding / {} full",
                    flags.len(),
                    flags.len() - sliding
                );
                extra.insert("layer_is_sliding".into(), Value::Array(flags));
            }
        }

        // Query pre-attention scalar = head_dim (key_length) → scaling
        // 1/sqrt(head_dim). The struct default (sqrt(head_dim)) is not the
        // Gemma 3 value; set it explicitly from the GGUF geometry.
        if let Some(kl) = self.meta_u64(&format!("{arch}.attention.key_length")) {
            extra.insert("query_pre_attn_scalar".into(), json!(kl));
        }

        // ── EmbeddingGemma detection ──────────────────────────────────────
        // Bidirectional encoder: `causal == false`.
        let causal = self
            .header
            .metadata
            .get(&format!("{arch}.attention.causal"))
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);
        if !causal {
            extra.insert("is_encoder_only".into(), json!(true));
        }

        // Pooling: llama.cpp pooling_type 0=none,1=mean,2=cls,3=last.
        if let Some(pt) = self.meta_u64(&format!("{arch}.pooling_type")) {
            let name = match pt {
                1 => Some("mean"),
                2 => Some("cls"),
                3 => Some("last"),
                _ => None,
            };
            if let Some(name) = name {
                extra.insert("embedding_pooling".into(), json!(name));
            }
        }

        // Sentence-transformers projector: presence of `dense.0`/`dense.1`
        // (BF16, no bias) signals the Matryoshka head applied after pooling.
        if let (Some(d0), Some(d1)) = (
            self.header.tensor_infos.get("dense.0.weight"),
            self.header.tensor_infos.get("dense.1.weight"),
        ) {
            // candle reverses on-disk dims → stored `[out, in]`; dims()[0] is out.
            let mid = d0.shape.first().copied();
            let out = d1.shape.first().copied();
            if let (Some(mid), Some(out)) = (mid, out) {
                extra.insert("has_st_projector".into(), json!(true));
                extra.insert("st_projector_mid".into(), json!(mid));
                extra.insert("st_projector_out".into(), json!(out));
                tracing::info!(
                    "Gemma 3 GGUF: EmbeddingGemma ST projector detected (dense {mid}→{out})"
                );
            }
        }

        let sliding_window = self
            .meta_u64(&format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize);

        (sliding_window, extra)
    }

    /// Create with a specific config.
    pub fn with_config(mut self, config: GgufConfig) -> Self {
        self.config = config;
        self
    }

    /// Number of transformer blocks (`{arch}.block_count`).
    pub fn num_layers(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.meta_u64(&format!("{arch}.block_count"))
            .map(|v| v as u32)
    }

    /// Hidden size (`{arch}.embedding_length`).
    pub fn hidden_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.meta_u64(&format!("{arch}.embedding_length"))
            .map(|v| v as u32)
    }

    /// All tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Resolve a vLLM-style prefix to a device-resident [`GgufTensor`],
    /// trying the literal name, the bare prefix, then the llama.cpp
    /// `blk.N.*` mapping.
    fn resolve_tensor(&self, prefix: &str) -> Option<&GgufTensor> {
        let mut candidates: Vec<String> = vec![format!("{prefix}.weight"), prefix.to_string()];
        if let Some(mapped) = to_llama_cpp_prefix(prefix) {
            candidates.push(format!("{mapped}.weight"));
            candidates.push(mapped);
        }
        candidates.iter().find_map(|name| self.tensors.get(name))
    }

    /// Resolve an optional bias tensor for a linear layer, dequantized to
    /// dense f32.
    fn resolve_bias(&self, prefix: &str) -> Option<Tensor> {
        let mut candidates: Vec<String> = vec![format!("{prefix}.bias")];
        if let Some(mapped) = to_llama_cpp_prefix(prefix) {
            candidates.push(format!("{mapped}.bias"));
        }
        for name in &candidates {
            if let Some(t) = self.tensors.get(name) {
                if let Ok(dense) = t.dequantize_dense(&self.device) {
                    return dense.to_dtype(DType::F32).ok();
                }
            }
        }
        None
    }
}

/// Map a vLLM-style tensor prefix to the canonical llama.cpp GGUF naming.
///
/// llama.cpp stores transformer blocks under `blk.N.*` and uses condensed
/// names (`attn_q`, `ffn_gate`, `output_norm`) rather than the HuggingFace
/// pytorch_model path. This mapping lets the GGUF loader accept vLLM-style
/// prefixes (`model.layers.5.self_attn.q_proj`) and still find the tensors.
///
/// Returns `None` when the input is not a recognised pattern — the caller
/// should then fall back to the literal path or report "tensor not found".
///
/// Gemma 4 E4B PLE tensors ARE mapped (confirmed against the real
/// `gemma-4-E4B-it-Q4_K_M.gguf` header): `per_layer_token_embd`,
/// `per_layer_model_proj`, `per_layer_proj_norm`, and the per-layer
/// `inp_gate` / `proj` / `post_norm`. E4B has no MoE (dense FFN), so no
/// stacked-expert mapping is needed. The llama.cpp-only `rope_freqs` and
/// `layer_output_scale` tensors are intentionally unmapped (our model
/// computes RoPE itself and does not use a per-layer output scale).
fn to_llama_cpp_prefix(prefix: &str) -> Option<String> {
    to_llama_cpp_prefixes(prefix).into_iter().next()
}

/// Return every llama.cpp-style candidate name for a given vLLM-style
/// prefix, in priority order. Some prefixes map to different llama.cpp
/// names across architectures — most notably `post_attention_layernorm`
/// is `ffn_norm` in llama/mistral but `post_attention_norm` in Gemma —
/// so we try each candidate in turn.
fn to_llama_cpp_prefixes(prefix: &str) -> Vec<String> {
    match prefix {
        "model.embed_tokens" => return vec!["token_embd".to_string()],
        "model.norm" => return vec!["output_norm".to_string()],
        "lm_head" => return vec!["output".to_string()],
        // Gemma 4 Per-Layer Embeddings (PLE), top-level tensors.
        "model.embed_tokens_per_layer" => return vec!["per_layer_token_embd".to_string()],
        "model.per_layer_model_projection" => return vec!["per_layer_model_proj".to_string()],
        "model.per_layer_projection_norm" => return vec!["per_layer_proj_norm".to_string()],
        _ => {}
    }

    let rest = match prefix.strip_prefix("model.layers.") {
        Some(r) => r,
        None => return Vec::new(),
    };
    let (layer_idx_str, sub) = match rest.split_once('.') {
        Some(pair) => pair,
        None => return Vec::new(),
    };
    if layer_idx_str.parse::<u32>().is_err() {
        return Vec::new();
    }

    let subs: Vec<&str> = match sub {
        "self_attn.q_proj" => vec!["attn_q"],
        "self_attn.k_proj" => vec!["attn_k"],
        "self_attn.v_proj" => vec!["attn_v"],
        "self_attn.o_proj" => vec!["attn_output"],
        "self_attn.q_norm" => vec!["attn_q_norm"],
        "self_attn.k_norm" => vec!["attn_k_norm"],
        "mlp.gate_proj" => vec!["ffn_gate"],
        "mlp.up_proj" => vec!["ffn_up"],
        "mlp.down_proj" => vec!["ffn_down"],
        // Gemma 4 PLE per-layer tensors.
        "per_layer_input_gate" => vec!["inp_gate"],
        "per_layer_projection" => vec!["proj"],
        "post_per_layer_input_norm" => vec!["post_norm"],
        // Gemma 4 per-layer output scale (a layer-level leaf, not a
        // sub-module; resolved via the full-name path in the VarBuilder
        // backend). Values are learned (~0.06-0.6), not 1.0, so loading the
        // real tensor instead of the identity fallback is correctness-
        // critical.
        "layer_scalar" => vec!["layer_output_scale"],
        "input_layernorm" => vec!["attn_norm"],
        // Gemma/Gemma2/Gemma3/Gemma4 ship a SEPARATE `post_attention_norm`
        // (the pre-FFN norm is `ffn_norm`). Llama/Mistral have no
        // `post_attention_norm` — their post-attention IS the pre-FFN
        // `ffn_norm`. Try `post_attention_norm` FIRST: present → Gemma uses
        // it; absent → Llama falls back to `ffn_norm`. (Trying `ffn_norm`
        // first was a bug: on Gemma both tensors exist, so
        // `post_attention_layernorm` wrongly loaded `ffn_norm` — the
        // pre-FFN weights — blowing up the residual stream.)
        "post_attention_layernorm" => vec!["post_attention_norm", "ffn_norm"],
        // Gemma family has both pre/post FFN norms; plain Llama doesn't.
        "pre_feedforward_layernorm" => vec!["ffn_norm"],
        "post_feedforward_layernorm" => vec!["post_ffw_norm"],
        _ => return Vec::new(),
    };

    subs.into_iter()
        .map(|s| format!("blk.{layer_idx_str}.{s}"))
        .collect()
}

/// Minimal top-level name mapping for tensors that sit outside a
/// transformer block (`model.embed_tokens`, `model.norm`, `lm_head`).
/// `to_llama_cpp_prefix` handles the more general `blk.N.*` cases.
fn to_llama_cpp_top_level(name: &str) -> Option<&'static str> {
    match name {
        "model.embed_tokens" => Some("token_embd"),
        "model.norm" => Some("output_norm"),
        "lm_head" => Some("output"),
        "model.embed_tokens_per_layer" => Some("per_layer_token_embd"),
        "model.per_layer_model_projection" => Some("per_layer_model_proj"),
        "model.per_layer_projection_norm" => Some("per_layer_proj_norm"),
        _ => None,
    }
}

/// `VarBuilder` backend that resolves names against the GGUF tensor map
/// instead of a safetensors archive. Hands the non-quantized tensors
/// (RMS-norm weights, embeddings) a model construction pulls through
/// `vb.pp(...)` over to the same GGUF file that backs the quantized
/// linears — dequantizing each `QTensor` to a dense tensor on demand.
///
/// Resolution strategy, in order:
/// 1. Literal `{name}` — some GGUF exports keep HuggingFace names.
/// 2. `{to_llama_cpp_prefix(name)}` — standard llama.cpp shorthand
///    (`blk.N.attn_norm`, `output_norm`, …).
pub struct GgufVarBuilderBackend {
    tensors: Arc<HashMap<String, GgufTensor>>,
    device: Device,
}

impl GgufVarBuilderBackend {
    /// Wrap the shared device-resident tensor map so the same data serves
    /// both the quantized loader and the VarBuilder backend.
    pub fn new(tensors: Arc<HashMap<String, GgufTensor>>, device: Device) -> Self {
        Self { tensors, device }
    }

    /// Resolve `name` using the same fallback chain as
    /// `GgufWeightLoader::resolve_tensor` — literal path, then
    /// `blk.N.*`-style mapping, then the bare top-level name — and
    /// dequantize the matched `QTensor` to a dense tensor.
    fn try_load(&self, name: &str, target: DType) -> Result<Tensor> {
        // VarBuilder appends a trailing qualifier like `...weight` via
        // `.pp(...).get(shape, "weight")`. llama.cpp stores e.g.
        // `blk.0.attn_norm.weight` literally, so the literal path is first.
        let mut candidates: Vec<String> = vec![name.to_string()];
        if let Some((base, leaf)) = name.rsplit_once('.') {
            for mapped_base in to_llama_cpp_prefixes(base) {
                candidates.push(format!("{mapped_base}.{leaf}"));
            }
            if let Some(top) = to_llama_cpp_top_level(base) {
                candidates.push(format!("{top}.{leaf}"));
            }
        } else if let Some(mapped) = to_llama_cpp_top_level(name) {
            candidates.push(mapped.to_string());
        }
        // Layer-level leaves that are NOT sub-modules (e.g. Gemma 4's
        // `model.layers.N.layer_scalar`) don't split into a base+leaf the
        // block mapper recognises. Try mapping the FULL name as a prefix
        // too, with and without a `.weight` suffix.
        for mapped in to_llama_cpp_prefixes(name) {
            candidates.push(format!("{mapped}.weight"));
            candidates.push(mapped);
        }

        for candidate in candidates {
            let t = match self.tensors.get(&candidate) {
                Some(t) => t,
                None => continue,
            };
            // The header reverses the on-disk dims, so a 2-D weight comes
            // back with candle's logical shape (e.g. token_embd as
            // `[vocab, hidden]`); `dequantize_dense` yields a dense tensor in
            // that shape directly — no manual reshape.
            let dense = t.dequantize_dense(&self.device)?;
            return dense.to_dtype(target);
        }
        Err(candle_core::Error::Msg(format!(
            "GgufVarBuilderBackend: tensor '{name}' not found (tried literal + llama.cpp mapping)"
        )))
    }
}

impl candle_nn::var_builder::SimpleBackend for GgufVarBuilderBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        _dev: &Device,
    ) -> Result<Tensor> {
        let tensor = self.try_load(name, dtype)?;
        let got = tensor.shape().clone();
        if got == s {
            return Ok(tensor);
        }

        // Some exports / consumers disagree on the 2-D matrix orientation
        // (e.g. an embedding stored `[hidden, vocab]` where the consumer
        // wants `[vocab, hidden]`). When the requested shape is exactly a
        // transpose of what we got, auto-transpose and retry — harmless
        // for already-correctly-shaped tensors (that branch returns above).
        if got.dims().len() == 2 && s.dims().len() == 2 {
            let got_dims = got.dims();
            let want_dims = s.dims();
            if got_dims[0] == want_dims[1] && got_dims[1] == want_dims[0] {
                let transposed = tensor.t()?.contiguous()?;
                if transposed.shape() == &s {
                    return Ok(transposed);
                }
            }
        }

        Err(candle_core::Error::Msg(format!(
            "GgufVarBuilderBackend: shape mismatch for '{name}' expected {:?}, got {:?}",
            s.dims(),
            got.dims()
        )))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.try_load(name, DType::F32).is_ok()
    }

    fn get_unchecked(&self, name: &str, dtype: DType, _dev: &Device) -> Result<Tensor> {
        self.try_load(name, dtype)
    }
}

impl QuantizedWeightLoader for GgufWeightLoader {
    fn load_linear(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Box<dyn QuantizedLinear>> {
        let tensor = self.resolve_tensor(prefix).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "GGUF: no quantized weight found for '{prefix}' \
                 (tried literal + llama.cpp mapping)"
            ))
        })?;
        let bias_tensor = if bias {
            self.resolve_bias(prefix)
        } else {
            None
        };
        match tensor {
            GgufTensor::Candle(qt) => {
                let qmm = Arc::new(QMatMul::from_arc(Arc::clone(qt))?);
                Ok(Box::new(GgufLinear::from_qmatmul(
                    qmm,
                    bias_tensor,
                    in_features,
                    out_features,
                )))
            }
            GgufTensor::Iq { bytes, iq, shape } => {
                // Size the linear from the tensor's ACTUAL shape, not the
                // caller's declared in/out. The Gemma 4 K-as-V fallback loads
                // `k_proj`'s bytes under the `v_proj` name, so the real dims
                // (K's) are what the dequant + matmul must use — using the
                // declared V dims mis-sizes the dequant (have/need mismatch).
                let (out, inn) = match shape.as_slice() {
                    [o, i] => (*o, *i),
                    other => candle_core::bail!(
                        "IqLinear expects a 2-D weight for '{prefix}', got {other:?}"
                    ),
                };
                Ok(Box::new(IqLinear::new(
                    bytes.clone(),
                    *iq,
                    out,
                    inn,
                    bias_tensor,
                )))
            }
        }
    }

    fn load_quantized_embedding(
        &self,
        prefix: &str,
        out_dtype: DType,
        out_device: &Device,
    ) -> Result<Option<crate::quantization::QuantizedEmbedding>> {
        // Resolve the GGUF tensor name for this embedding prefix.
        let gguf_name = match to_llama_cpp_prefix(prefix) {
            Some(mapped) => format!("{mapped}.weight"),
            None => return Ok(None),
        };
        let info = match self.header.tensor_infos.get(&gguf_name) {
            Some(info) => info,
            None => return Ok(None),
        };
        let dims = info.shape.clone();
        if dims.len() != 2 {
            return Ok(None); // not a 2-D embedding table
        }
        // The quantized-embedding gather paths (GPU Q6_K kernel, CPU QTensor
        // slice) are candle-native only. I-quant embedding tables would need
        // an I-quant gather; none of the supported checkpoints use one
        // (Gemma 4 PLE is Q6_K, token_embd is a K-quant), so fall back to the
        // dense VarBuilder load for the unlikely IQ embedding.
        let ggml_dtype = match info.dtype {
            GgufTensorDtype::Candle(d) => d,
            GgufTensorDtype::Iq(_) => return Ok(None),
        };

        // Fast GPU path: Q6_K table + CUDA device + the gather kernel —
        // keep the quantized bytes resident in VRAM and gather per-row via
        // the custom kernel (never goes dense).
        //
        // VRAM-aware placement: an embedding table is a GATHER (a few rows
        // per token), not a matmul — it does not need GPU compute. On a
        // VRAM-constrained card, parking a multi-GB table (Gemma 4 PLE is
        // ~2.3 GB Q6_K) in VRAM starves the prefill/decode scratch and the
        // KV cache, so we keep it on GPU ONLY when there is comfortable
        // headroom left, otherwise the host (CPU gather). Mirrors
        // llama.cpp keeping embeddings off the GPU under memory pressure.
        #[cfg(feature = "cuda-kernels")]
        if ggml_dtype == GgmlDType::Q6K && out_device.is_cuda() {
            let elem_count: usize = dims.iter().product();
            let table_bytes = elem_count / ggml_dtype.block_size() * ggml_dtype.type_size();
            // Headroom to leave free AFTER the table for scratch (~1.5 GB)
            // + a usable KV cache (~0.5 GB).
            const GPU_PLACE_HEADROOM: usize = 2 * 1024 * 1024 * 1024;
            let fits_on_gpu = match crate::kv_cache::config::gpu_memory_info() {
                Ok((free, _total)) => free.saturating_sub(table_bytes) >= GPU_PLACE_HEADROOM,
                Err(_) => true, // can't measure → assume GPU (prior behaviour)
            };
            if fits_on_gpu {
                tracing::info!(
                    "GGUF embedding '{gguf_name}' ({:.2} GB Q6_K) → GPU gather kernel",
                    table_bytes as f64 / 1e9
                );
                let bytes = self.read_tensor_bytes(&gguf_name)?;
                let n_bytes = bytes.len();
                let table = Tensor::from_vec(bytes, (n_bytes,), out_device)?;
                return Ok(Some(crate::quantization::QuantizedEmbedding::new_gpu_q6k(
                    table,
                    dims[0],
                    dims[1],
                    out_dtype,
                    out_device.clone(),
                )?));
            }
            tracing::info!(
                "GGUF embedding '{gguf_name}' ({:.2} GB Q6_K) → CPU gather \
                 (insufficient VRAM headroom; frees it for scratch + KV)",
                table_bytes as f64 / 1e9
            );
        }

        // CPU-resident QTensor, per-row block-slice gather. Used for any
        // quant type, non-CUDA devices, OR (above) when a large table is
        // better kept off a VRAM-constrained GPU.
        let bytes = self.read_tensor_bytes(&gguf_name)?;
        let storage = QStorage::from_data(Cow::Borrowed(&bytes), &Device::Cpu, ggml_dtype)?;
        let qt = Arc::new(QTensor::new(storage, dims)?);
        Ok(Some(crate::quantization::QuantizedEmbedding::new(
            qt,
            out_dtype,
            out_device.clone(),
        )?))
    }

    fn method(&self) -> QuantizationMethod {
        QuantizationMethod::Gguf
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod iq_linear_tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Fixture {
        raw_hex: String,
        golden_f32: Vec<f32>,
    }

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// `IqLinear::forward` must equal a dense `x @ Wᵀ` over the SAME
    /// dequantized weight. Uses the IQ3_XXS golden fixture (1536 values) as a
    /// `[6, 256]` weight; exercises the full dequant→matmul path on CPU.
    #[test]
    fn iq_linear_forward_matches_dense_matmul() {
        let f: Fixture = serde_json::from_str(include_str!("iq/testdata/iq3_xxs.json")).unwrap();
        let (out_features, in_features) = (6usize, 256usize);
        assert_eq!(f.golden_f32.len(), out_features * in_features);

        let dev = Device::Cpu;
        let bytes = hex_to_bytes(&f.raw_hex);
        let n_bytes = bytes.len();
        let bytes_t = Tensor::from_vec(bytes, (n_bytes,), &dev).unwrap();
        let lin = IqLinear::new(bytes_t, IqType::Iq3Xxs, out_features, in_features, None);

        // Reference dense weight from the golden values.
        let w = Tensor::from_vec(f.golden_f32.clone(), (out_features, in_features), &dev).unwrap();

        // Deterministic input [M, in].
        let m = 3usize;
        let xv: Vec<f32> = (0..m * in_features)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let x = Tensor::from_vec(xv, (m, in_features), &dev).unwrap();

        let y = lin.forward(&x).unwrap();
        assert_eq!(y.dims(), &[m, out_features]);
        let reference = x.matmul(&w.t().unwrap()).unwrap();

        let diff = (y - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff <= 1e-4, "IqLinear vs dense matmul diverged: {diff}");
    }

    /// `from_path` must round-trip candle-native tensors: write a tiny GGUF,
    /// load it back, and check the dequantized values match. Guards the
    /// `Cow::Owned` freed-memory hazard in the candle-native load arm — with
    /// `Cow::Owned`, candle's `as_t_slice` returns a slice into a temporary
    /// that drops before `load_quantized` copies it, so the CUDA upload reads
    /// freed memory (garbage weights / SIGSEGV). Only the CUDA path exercises
    /// that hazard, so the assertion is meaningful there; on CPU it still
    /// checks the loader end-to-end.
    #[test]
    fn from_path_roundtrips_candle_tensors() {
        use candle_core::quantized::{gguf_file, GgmlDType, QTensor};

        let dev = match Device::cuda_if_available(0) {
            Ok(d) => d, // prefer CUDA (where the freed-memory bug bites)
            Err(_) => Device::Cpu,
        };
        let cpu = Device::Cpu;

        // [out, in] with in % 256 == 0 for the K-quant block size.
        let w = Tensor::randn(0f32, 1.0, (32usize, 256usize), &cpu).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
        let reference = q.dequantize(&cpu).unwrap();

        let path = std::env::temp_dir().join("vllm_iq_from_path_roundtrip.gguf");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let arch = gguf_file::Value::String("llama".to_string());
            gguf_file::write(
                &mut f,
                &[("general.architecture", &arch)],
                &[("blk.0.attn_q.weight", &q)],
            )
            .unwrap();
        }

        let loader = GgufWeightLoader::from_path(&path, dev.clone(), DType::F32).unwrap();
        let tensors = loader.shared_tensors();
        let loaded = tensors.get("blk.0.attn_q.weight").expect("tensor loaded");
        let dense = loaded
            .dequantize_dense(&dev)
            .unwrap()
            .to_device(&cpu)
            .unwrap();

        let diff = (dense - reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            diff <= 1e-5,
            "from_path candle-tensor round-trip diverged (freed-memory bug?): {diff}"
        );
    }

    #[test]
    fn iq_linear_trait_metadata() {
        let dev = Device::Cpu;
        // 1 block (256 elems) of zero IQ3_XXS bytes → weight [1, 256].
        let bytes = vec![0u8; IqType::Iq3Xxs.type_size()];
        let n = bytes.len();
        let bytes_t = Tensor::from_vec(bytes, (n,), &dev).unwrap();
        let lin = IqLinear::new(bytes_t, IqType::Iq3Xxs, 1, 256, None);
        assert_eq!(lin.in_features(), 256);
        assert_eq!(lin.out_features(), 1);
        assert!(!lin.has_bias());
        assert_eq!(lin.weight_dtype(), DType::F32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_config_default() {
        let config = GgufConfig::default();
        assert_eq!(config.method(), QuantizationMethod::Gguf);
        assert_eq!(config.min_capability(), 0);
        assert!(!config.is_layer_skipped("some_layer"));
    }

    #[test]
    fn test_gguf_config_ignored_layers() {
        let config = GgufConfig::new().with_ignored_layers(vec!["lm_head".to_string()]);
        assert!(config.is_layer_skipped("lm_head"));
        assert!(config.is_layer_skipped("model.lm_head.weight"));
        assert!(!config.is_layer_skipped("self_attn"));
    }

    #[test]
    fn test_gguf_linear_creation() {
        let config = GgufConfig::default();
        let linear = config
            .create_linear(4096, 4096, false, &Device::Cpu)
            .unwrap();

        assert_eq!(linear.in_features(), 4096);
        assert_eq!(linear.out_features(), 4096);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_gguf_linear_validation() {
        let result = GgufLinear::new(0, 128, false, &Device::Cpu);
        assert!(result.is_err());

        let result = GgufLinear::new(64, 0, false, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_linear_forward_requires_weights() {
        let linear = GgufLinear::new(64, 128, false, &Device::Cpu).unwrap();
        let x = Tensor::ones(&[2, 64], DType::F32, &Device::Cpu).unwrap();

        let result = linear.forward(&x);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no weights loaded"));
    }

    #[test]
    fn test_gguf_linear_forward_qmatmul_roundtrip() {
        // Build a real QMatMul over a Q4_K weight and verify the
        // GgufLinear forward tracks a dense dequant+matmul reference that
        // uses the SAME quantized weight.
        //
        // NOTE: this is a RELATIVE-error check, not bit-equality. ggml's
        // matmul (candle `k_quants::matmul`) quantizes the ACTIVATION to
        // the weight's `VecDotType` — Q8_K for Q4_K — before the integer
        // dot product. So `QMatMul::forward(x)` carries ~1-2% activation
        // quantization error on top of the (common) weight quantization.
        // The naive `x @ dequant(W)ᵀ` reference does not, so a tight
        // absolute tolerance would wrongly flag the correct fused kernel.
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
        let device = Device::Cpu;
        let (out, inn) = (256usize, 256usize); // QK_K=256 divides `in`
        let w = Tensor::randn(0.0f32, 1.0, (out, inn), &device).unwrap();
        let qt = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
        let w_dense = qt.dequantize(&device).unwrap();
        let qmm = Arc::new(QMatMul::from_qtensor(qt).unwrap());

        let linear = GgufLinear::from_qmatmul(qmm, None, inn, out);
        let x = Tensor::randn(0.0f32, 1.0, (1, 3, inn), &device).unwrap();
        let y = linear.forward(&x).unwrap();

        // Reference: x @ w_dense.t() (full-precision activation).
        let x2d = x.reshape((3, inn)).unwrap();
        let y_ref = x2d.matmul(&w_dense.t().unwrap()).unwrap();
        let y2d = y.reshape((3, out)).unwrap();

        // Relative RMS error = ‖y − y_ref‖ / ‖y_ref‖.
        let err = (&y2d - &y_ref).unwrap();
        let num = err
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let den = y_ref
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let rel_rms = (num / den).sqrt();
        assert!(
            rel_rms < 0.03,
            "QMatMul forward relative-RMS error {rel_rms} exceeds the \
             Q8_K activation-quantization bound (~1-2%); a real kernel \
             bug would push this far higher"
        );
    }

    #[test]
    fn test_gguf_linear_forward_preserves_dtype() {
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
        let device = Device::Cpu;
        let w = Tensor::randn(0.0f32, 1.0, (256usize, 256usize), &device).unwrap();
        let qt = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
        let qmm = Arc::new(QMatMul::from_qtensor(qt).unwrap());
        let linear = GgufLinear::from_qmatmul(qmm, None, 256, 256);

        // BF16 in → BF16 out (cast happens at the kernel boundary).
        let x = Tensor::randn(0.0f32, 1.0, (1usize, 2usize, 256usize), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let y = linear.forward(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn test_to_llama_cpp_prefix_top_level() {
        assert_eq!(
            to_llama_cpp_prefix("model.embed_tokens").as_deref(),
            Some("token_embd")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.norm").as_deref(),
            Some("output_norm")
        );
        assert_eq!(to_llama_cpp_prefix("lm_head").as_deref(), Some("output"));
    }

    #[test]
    fn test_to_llama_cpp_prefix_attention() {
        assert_eq!(
            to_llama_cpp_prefix("model.layers.5.self_attn.q_proj").as_deref(),
            Some("blk.5.attn_q")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.self_attn.k_proj").as_deref(),
            Some("blk.0.attn_k")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.11.self_attn.v_proj").as_deref(),
            Some("blk.11.attn_v")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.3.self_attn.o_proj").as_deref(),
            Some("blk.3.attn_output")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_qkv_norms() {
        // Gemma 4 QKV norms must map so checkpoints with per-head norms load.
        assert_eq!(
            to_llama_cpp_prefix("model.layers.2.self_attn.q_norm").as_deref(),
            Some("blk.2.attn_q_norm")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.2.self_attn.k_norm").as_deref(),
            Some("blk.2.attn_k_norm")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_mlp() {
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.gate_proj").as_deref(),
            Some("blk.7.ffn_gate")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.up_proj").as_deref(),
            Some("blk.7.ffn_up")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.7.mlp.down_proj").as_deref(),
            Some("blk.7.ffn_down")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_layernorms() {
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.input_layernorm").as_deref(),
            Some("blk.0.attn_norm")
        );
        // Gemma's separate post-attention norm must be tried FIRST so it
        // wins over `ffn_norm` (the pre-FFN norm) when both exist.
        let post_attn = to_llama_cpp_prefixes("model.layers.0.post_attention_layernorm");
        assert_eq!(
            post_attn,
            vec![
                "blk.0.post_attention_norm".to_string(),
                "blk.0.ffn_norm".to_string(),
            ]
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.pre_feedforward_layernorm").as_deref(),
            Some("blk.0.ffn_norm")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.0.post_feedforward_layernorm").as_deref(),
            Some("blk.0.post_ffw_norm")
        );
    }

    #[test]
    fn test_quantized_embedding_gather_matches_full_dequant() {
        // A gathered row must equal the corresponding row of the full
        // dequantized table (gather is a byte-range slice of the same
        // blocks, then the identical dequant) — bit-for-bit.
        use candle_core::quantized::{GgmlDType, QTensor};
        let device = Device::Cpu;
        let (num, dim) = (8usize, 256usize); // QK_K=256 → 1 block per row
        let table = Tensor::randn(0.0f32, 1.0, (num, dim), &device).unwrap();
        let qt = Arc::new(QTensor::quantize(&table, GgmlDType::Q4K).unwrap());
        let full = qt.dequantize(&device).unwrap(); // [num, dim]

        let emb = QuantizedEmbedding::new(Arc::clone(&qt), DType::F32, device.clone()).unwrap();

        // Single-id gathers must be bit-exact (same blocks, same dequant).
        for id in [0u32, 2, 5, 7] {
            let ids = Tensor::from_vec(vec![id], (1,), &device).unwrap();
            let g = emb.forward(&ids).unwrap().reshape((1, dim)).unwrap();
            let r = full.narrow(0, id as usize, 1).unwrap();
            let d = (g - r).unwrap().abs().unwrap().max_all().unwrap();
            assert_eq!(
                d.to_scalar::<f32>().unwrap(),
                0.0,
                "single id {id} mismatch"
            );
        }

        // Multi-id, multi-dim shape: [2,2] ids → [2,2,dim], each row
        // matching the full table including a repeated id.
        let ids = Tensor::from_vec(vec![2u32, 5, 2, 0], (2, 2), &device).unwrap();
        let got = emb.forward(&ids).unwrap();
        assert_eq!(got.dims(), &[2, 2, dim]);
        let got2d = got.reshape((4, dim)).unwrap();
        for (row, &id) in [2usize, 5, 2, 0].iter().enumerate() {
            let g = got2d.narrow(0, row, 1).unwrap();
            let r = full.narrow(0, id, 1).unwrap();
            let d = (g - r).unwrap().abs().unwrap().max_all().unwrap();
            assert_eq!(
                d.to_scalar::<f32>().unwrap(),
                0.0,
                "row {row} (id {id}) mismatch"
            );
        }
    }

    #[test]
    fn test_to_llama_cpp_prefix_gemma4_ple() {
        // Gemma 4 E4B Per-Layer Embedding tensors (real GGUF names).
        assert_eq!(
            to_llama_cpp_prefix("model.embed_tokens_per_layer").as_deref(),
            Some("per_layer_token_embd")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.per_layer_model_projection").as_deref(),
            Some("per_layer_model_proj")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.per_layer_projection_norm").as_deref(),
            Some("per_layer_proj_norm")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.3.per_layer_input_gate").as_deref(),
            Some("blk.3.inp_gate")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.3.per_layer_projection").as_deref(),
            Some("blk.3.proj")
        );
        assert_eq!(
            to_llama_cpp_prefix("model.layers.3.post_per_layer_input_norm").as_deref(),
            Some("blk.3.post_norm")
        );
    }

    #[test]
    fn test_to_llama_cpp_prefix_unknown_returns_none() {
        // E4B has no MoE; stacked-expert names stay unmapped.
        assert!(to_llama_cpp_prefix("model.layers.0.moe.experts.0.gate_proj").is_none());
        assert!(to_llama_cpp_prefix("model.layers.not_a_number.self_attn.q_proj").is_none());
        assert!(to_llama_cpp_prefix("random_tensor").is_none());
    }
}
