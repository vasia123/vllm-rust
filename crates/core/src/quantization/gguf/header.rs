//! Minimal GGUF header parser that understands I-quant tensor types.
//!
//! candle's [`gguf_file::Content::read`] aborts at the tensor table the
//! moment it sees an unknown GGML type id — and it does not know any of the
//! I-quants (`IQ*`). The Unsloth "UD" dynamic checkpoints mix several
//! I-quant types per file, so candle can't even read their header. This
//! parser reads the same layout candle does (it is byte-for-byte compatible:
//! same version/alignment/dimension-reversal handling) but classifies each
//! tensor's dtype as either a candle [`GgmlDType`] or one of our [`IqType`]s.
//!
//! Metadata values reuse candle's public [`gguf_file::Value`] so every
//! existing metadata reader in the loader keeps working unchanged.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::GgmlDType;
use candle_core::Result;

use super::iq::IqType;

/// GGUF default tensor-data alignment (overridable via
/// `general.alignment`); mirrors candle's `DEFAULT_ALIGNMENT`.
const DEFAULT_ALIGNMENT: u64 = 32;

/// A plain (non-quantized) integer tensor type GGUF can carry but candle's
/// `GgmlDType` does not model — e.g. an ordering/index table. Never a matmul
/// weight; we classify it so the header parses, then load it lazily.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufIntType {
    I8,
    I16,
    I32,
    I64,
}

impl GgufIntType {
    /// Bytes per element (block size is always 1 for plain integers).
    fn type_size(self) -> usize {
        match self {
            GgufIntType::I8 => 1,
            GgufIntType::I16 => 2,
            GgufIntType::I32 => 4,
            GgufIntType::I64 => 8,
        }
    }
}

/// How a tensor's blocks are encoded: a candle-native GGML type, one of the
/// I-quants candle lacks, or a plain integer table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorDtype {
    /// A type candle's `GgmlDType` handles (F32/F16/BF16, legacy quants,
    /// K-quants).
    Candle(GgmlDType),
    /// An I-quant handled by [`crate::quantization::gguf::iq`].
    Iq(IqType),
    /// A plain integer table (GGML `I8`/`I16`/`I32`/`I64`). Not a weight —
    /// e.g. the `mtp.token_ordering` index of the Gemma 4 assistant's masked
    /// embedder. Carried so the header parses; the eager loader skips it.
    Int(GgufIntType),
}

impl GgufTensorDtype {
    /// Elements per block. Both candle types and I-quants use `QK_K = 256`
    /// for the K-/I-quants and the legacy block sizes otherwise.
    pub fn block_size(self) -> usize {
        match self {
            GgufTensorDtype::Candle(d) => d.block_size(),
            GgufTensorDtype::Iq(iq) => iq.block_size(),
            GgufTensorDtype::Int(_) => 1,
        }
    }

    /// On-disk byte size of one block.
    pub fn type_size(self) -> usize {
        match self {
            GgufTensorDtype::Candle(d) => d.type_size(),
            GgufTensorDtype::Iq(iq) => iq.type_size(),
            GgufTensorDtype::Int(i) => i.type_size(),
        }
    }

    /// `true` for the I-quant variants candle cannot load.
    pub fn is_iq(self) -> bool {
        matches!(self, GgufTensorDtype::Iq(_))
    }
}

/// One tensor's header entry. `shape` is in candle's logical orientation
/// (on-disk dims reversed), so a 2-D weight is `[out, in]` — exactly what
/// the loader's existing shape logic assumes.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub shape: Vec<usize>,
    pub dtype: GgufTensorDtype,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total element count across all dimensions.
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Raw byte length of this tensor's quantized blocks.
    pub fn n_bytes(&self) -> usize {
        self.elem_count() / self.dtype.block_size() * self.dtype.type_size()
    }
}

/// Parsed GGUF header: metadata + tensor table + where tensor data starts.
/// Drop-in replacement for the parts of candle's `Content` the loader uses,
/// extended to carry I-quant tensors.
pub struct GgufHeader {
    pub metadata: HashMap<String, Value>,
    pub tensor_infos: HashMap<String, GgufTensorInfo>,
    pub tensor_data_offset: u64,
}

/// Little-endian primitive reads over an arbitrary reader.
struct LeReader<R> {
    r: R,
}

impl<R: Read + Seek> LeReader<R> {
    fn new(r: R) -> Self {
        Self { r }
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// GGUF string: a u64 length (v2/v3) followed by UTF-8 bytes. v1 used a
    /// u32 length — handled via `count_is_u64`.
    fn string(&mut self, count_is_u64: bool) -> Result<String> {
        let len = if count_is_u64 {
            self.u64()? as usize
        } else {
            self.u32()? as usize
        };
        let mut buf = vec![0u8; len];
        self.r.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| candle_core::Error::Msg(format!("GGUF utf8: {e}")))
    }

    /// Read a typed metadata value, recursing for arrays. `value_type` is the
    /// GGUF metadata value-type id (0..=12), matching candle's `ValueType`.
    fn value(&mut self, value_type: u32, count_is_u64: bool) -> Result<Value> {
        Ok(match value_type {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16({
                let mut b = [0u8; 2];
                self.r.read_exact(&mut b)?;
                u16::from_le_bytes(b)
            }),
            3 => Value::I16({
                let mut b = [0u8; 2];
                self.r.read_exact(&mut b)?;
                i16::from_le_bytes(b)
            }),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(self.f32()?),
            7 => match self.u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                b => candle_core::bail!("GGUF: unexpected bool value {b}"),
            },
            8 => Value::String(self.string(count_is_u64)?),
            9 => {
                let elem_type = self.u32()?;
                let len = if count_is_u64 {
                    self.u64()? as usize
                } else {
                    self.u32()? as usize
                };
                let mut vs = Vec::with_capacity(len);
                for _ in 0..len {
                    vs.push(self.value(elem_type, count_is_u64)?);
                }
                Value::Array(vs)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(self.f64()?),
            v => candle_core::bail!("GGUF: unrecognized metadata value-type {v}"),
        })
    }

    fn stream_position(&mut self) -> Result<u64> {
        Ok(self.r.stream_position()?)
    }
}

impl GgufHeader {
    /// Parse a GGUF header from `reader`. Accepts versions 1/2/3 (modern
    /// files are v3). Errors on a non-`GGUF` magic or a tensor type id that
    /// is neither a candle GGML type nor a supported I-quant.
    pub fn read<R: Read + Seek>(reader: R) -> Result<Self> {
        let mut r = LeReader::new(reader);

        let mut magic = [0u8; 4];
        r.r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            candle_core::bail!("not a GGUF file (bad magic {magic:?})");
        }
        let version = r.u32()?;
        // v1 used u32 counts; v2/v3 use u64. Same for string/array lengths.
        let count_is_u64 = version >= 2;
        let read_count = |r: &mut LeReader<R>| -> Result<usize> {
            Ok(if count_is_u64 {
                r.u64()? as usize
            } else {
                r.u32()? as usize
            })
        };

        let tensor_count = read_count(&mut r)?;
        let metadata_kv_count = read_count(&mut r)?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = r.string(count_is_u64)?;
            let value_type = r.u32()?;
            let value = r.value(value_type, count_is_u64)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string(count_is_u64)?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = if count_is_u64 {
                    r.u64()? as usize
                } else {
                    r.u32()? as usize
                };
                dims.push(d);
            }
            // candle reverses on-disk dims so a 2-D weight is logical
            // `[out, in]`; mirror that so the loader's shape logic is identical.
            dims.reverse();

            let ggml_type_id = r.u32()?;
            let offset = r.u64()?;
            let dtype = classify_dtype(ggml_type_id).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "GGUF tensor '{name}': unsupported GGML type id {ggml_type_id}"
                ))
            })?;
            tensor_infos.insert(
                name,
                GgufTensorInfo {
                    shape: dims,
                    dtype,
                    offset,
                },
            );
        }

        let position = r.stream_position()?;
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => *v as u64,
            Some(Value::U16(v)) => *v as u64,
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        let tensor_data_offset = position.div_ceil(alignment) * alignment;

        Ok(Self {
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    /// Read a tensor's raw quantized block bytes straight from `reader`
    /// (which must be positioned anywhere; this seeks). Used for both the
    /// I-quant path and any tensor served from raw bytes.
    pub fn read_tensor_bytes<R: Read + Seek>(&self, reader: &mut R, name: &str) -> Result<Vec<u8>> {
        let info = self.tensor_infos.get(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("GGUF tensor '{name}' not found in header"))
        })?;
        let n_bytes = info.n_bytes();
        reader.seek(SeekFrom::Start(self.tensor_data_offset + info.offset))?;
        let mut buf = vec![0u8; n_bytes];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Map a raw GGML tensor type id to our dtype classification, or `None` if
/// it is neither a candle-native type nor a supported I-quant.
fn classify_dtype(id: u32) -> Option<GgufTensorDtype> {
    // candle-native ids (see candle_core::quantized::GgmlDType::from_u32).
    let candle = match id {
        0 => Some(GgmlDType::F32),
        1 => Some(GgmlDType::F16),
        2 => Some(GgmlDType::Q4_0),
        3 => Some(GgmlDType::Q4_1),
        6 => Some(GgmlDType::Q5_0),
        7 => Some(GgmlDType::Q5_1),
        8 => Some(GgmlDType::Q8_0),
        9 => Some(GgmlDType::Q8_1),
        10 => Some(GgmlDType::Q2K),
        11 => Some(GgmlDType::Q3K),
        12 => Some(GgmlDType::Q4K),
        13 => Some(GgmlDType::Q5K),
        14 => Some(GgmlDType::Q6K),
        15 => Some(GgmlDType::Q8K),
        30 => Some(GgmlDType::BF16),
        _ => None,
    };
    if let Some(d) = candle {
        return Some(GgufTensorDtype::Candle(d));
    }
    // GGML plain-integer ids (24=I8, 25=I16, 26=I32, 27=I64). Not weights, but
    // some checkpoints (e.g. the Gemma 4 assistant's masked-embedder ordering)
    // carry them, so classify rather than bail.
    let int = match id {
        24 => Some(GgufIntType::I8),
        25 => Some(GgufIntType::I16),
        26 => Some(GgufIntType::I32),
        27 => Some(GgufIntType::I64),
        _ => None,
    };
    if let Some(i) = int {
        return Some(GgufTensorDtype::Int(i));
    }
    IqType::from_ggml_type_id(id).map(GgufTensorDtype::Iq)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL: &str = "/home/vasis/.cache/dnd-llm/gemma4-12b-iq3/gemma-4-12b-it-UD-IQ3_XXS.gguf";

    #[test]
    fn parses_real_gemma4_iq3_header() {
        if !std::path::Path::new(MODEL).exists() {
            eprintln!("skip: model file not present");
            return;
        }
        let f = std::fs::File::open(MODEL).unwrap();
        let h = GgufHeader::read(std::io::BufReader::new(f)).unwrap();

        // Header-level facts (cross-checked with scripts/gguf_types.py).
        assert_eq!(h.tensor_infos.len(), 667);
        assert_eq!(
            h.metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .map(String::as_str),
            Some("gemma4")
        );

        // The whole point: I-quant tensors are classified, not rejected.
        let v = &h.tensor_infos["blk.0.attn_v.weight"];
        assert_eq!(v.dtype, GgufTensorDtype::Iq(IqType::Iq3Xxs));
        // candle orientation reverses on-disk dims [3840, 2048] → [out, in] =
        // [2048, 3840] (v_proj maps hidden 3840 → kv 2048).
        assert_eq!(v.shape, vec![2048, 3840]);
        // n_bytes consistent with the IQ3_XXS block geometry.
        assert_eq!(v.n_bytes(), v.elem_count() / 256 * 98);

        let k = &h.tensor_infos["blk.0.attn_k.weight"];
        assert_eq!(k.dtype, GgufTensorDtype::Iq(IqType::Iq2S));

        // A candle-native tensor stays candle-native.
        assert_eq!(
            h.tensor_infos["token_embd.weight"].dtype,
            GgufTensorDtype::Candle(GgmlDType::Q3K)
        );
        assert!(matches!(
            h.tensor_infos["output_norm.weight"].dtype,
            GgufTensorDtype::Candle(GgmlDType::F32)
        ));

        // Histogram of IQ types must match the known mix.
        let n_iq = h.tensor_infos.values().filter(|t| t.dtype.is_iq()).count();
        assert_eq!(n_iq, 187 + 76 + 50 + 10 + 5);
    }

    const ASSISTANT: &str =
        "/home/vasis/gguf-models/gemma-4-E4B-assistant/gemma-4-E4B-it-assistant.F16.gguf";

    #[test]
    fn classifies_plain_integer_types() {
        // GGML plain-integer ids must classify (so headers carrying index
        // tables — e.g. the Gemma 4 assistant masked-embedder ordering —
        // parse instead of bailing). 26 = I32 is the one that appears in the
        // wild; the siblings round out the family.
        assert_eq!(
            classify_dtype(24),
            Some(GgufTensorDtype::Int(GgufIntType::I8))
        );
        assert_eq!(
            classify_dtype(25),
            Some(GgufTensorDtype::Int(GgufIntType::I16))
        );
        assert_eq!(
            classify_dtype(26),
            Some(GgufTensorDtype::Int(GgufIntType::I32))
        );
        assert_eq!(
            classify_dtype(27),
            Some(GgufTensorDtype::Int(GgufIntType::I64))
        );
        // An I32 table is 4 bytes/elem, block size 1.
        let d = GgufTensorDtype::Int(GgufIntType::I32);
        assert_eq!(d.block_size(), 1);
        assert_eq!(d.type_size(), 4);
        assert!(!d.is_iq());
        // Still nothing for a genuinely unknown id.
        assert_eq!(classify_dtype(9999), None);
    }

    #[test]
    fn parses_gemma4_assistant_header() {
        if !std::path::Path::new(ASSISTANT).exists() {
            eprintln!("skip: assistant model file not present");
            return;
        }
        let f = std::fs::File::open(ASSISTANT).unwrap();
        let h = GgufHeader::read(std::io::BufReader::new(f)).unwrap();

        // Whole point of the I32 fix: the header parses despite the masked-
        // embedder `mtp.token_ordering` index tensor (GGML I32 / id 26).
        assert_eq!(h.tensor_infos.len(), 51);
        assert_eq!(
            h.metadata
                .get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .map(String::as_str),
            Some("gemma4_assistant")
        );
        assert_eq!(
            h.tensor_infos["mtp.token_ordering.weight"].dtype,
            GgufTensorDtype::Int(GgufIntType::I32)
        );

        // Q-only attention: a Q projection exists, K/V projections do not.
        assert!(h.tensor_infos.contains_key("blk.0.attn_q.weight"));
        assert!(!h.tensor_infos.contains_key("blk.0.attn_k.weight"));
        assert!(!h.tensor_infos.contains_key("blk.0.attn_v.weight"));

        // Hidden-state fusion projections, candle orientation [out, in].
        // pre:  fuse [target_embd | backbone_hidden] (2*2560) → assistant 256.
        assert_eq!(
            h.tensor_infos["mtp.pre_projection.weight"].shape,
            vec![256, 5120]
        );
        // post: assistant 256 → backbone 2560 (feedback hidden).
        assert_eq!(
            h.tensor_infos["mtp.post_projection.weight"].shape,
            vec![2560, 256]
        );
    }
}
