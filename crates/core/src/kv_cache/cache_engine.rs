use candle_core::{DType, Tensor};

use super::block_pool::BlockId;
use super::config::{CacheConfig, KVCacheLayout};
use super::error::CacheError;
use super::quantization::{
    compute_int8_scale, dequantize_fp8, dequantize_int8, quantize_fp8, quantize_int8, KVCacheDtype,
    KVScales,
};

/// Owns pre-allocated GPU tensors for one layer's KV cache.
/// Performs block-level read/write via scatter_set and index_select.
///
/// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
/// This layout allows direct reshape to [total_slots, kv_heads, head_dim] for scatter/gather.
///
/// Supports optional quantization (FP8/INT8) for 2x memory reduction:
/// - On write: quantizes K/V tensors and updates scales
/// - On read: dequantizes back to compute dtype
pub struct CacheEngine {
    k_cache: Tensor,
    v_cache: Tensor,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// KV cache quantization dtype
    kv_cache_dtype: KVCacheDtype,
    /// Compute dtype for activations (BF16/F16/F32)
    compute_dtype: DType,
    /// Scales for quantized mode (None if Auto/unquantized)
    scales: Option<KVScales>,
    /// Memory layout (NHD or HND)
    layout: KVCacheLayout,
}

impl CacheEngine {
    /// Pre-allocate cache tensors with NHD layout, filled with zeros.
    ///
    /// When `kv_cache_dtype` is FP8 or INT8, allocates U8 tensors and
    /// initializes scales.
    pub fn new(config: &CacheConfig) -> Result<Self, CacheError> {
        Self::with_layout(config, KVCacheLayout::NHD)
    }

    /// Pre-allocate cache tensors with explicit layout, filled with zeros.
    ///
    /// The layout determines tensor shape:
    /// - NHD: `[num_blocks, block_size, num_kv_heads, head_dim]`
    /// - HND: `[num_blocks, num_kv_heads, block_size, head_dim]`
    pub fn with_layout(config: &CacheConfig, layout: KVCacheLayout) -> Result<Self, CacheError> {
        let shape = layout.cache_shape(
            config.num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        );

        // Use storage dtype based on quantization setting
        let storage_dtype = config.kv_storage_dtype();
        let k_cache = Tensor::zeros(shape, storage_dtype, &config.device)?;
        let v_cache = Tensor::zeros(shape, storage_dtype, &config.device)?;

        // Initialize scales if quantized
        let scales = if config.kv_cache_dtype.is_quantized() {
            Some(KVScales::new(&config.device)?)
        } else {
            None
        };

        Ok(Self {
            k_cache,
            v_cache,
            num_blocks: config.num_blocks,
            block_size: config.block_size,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            kv_cache_dtype: config.kv_cache_dtype,
            compute_dtype: config.dtype,
            scales,
            layout,
        })
    }

    /// Returns true if this cache engine uses quantization.
    pub fn is_quantized(&self) -> bool {
        self.kv_cache_dtype.is_quantized()
    }

    /// Get the current K scale (for quantized mode).
    pub fn k_scale(&self) -> Option<&Tensor> {
        self.scales.as_ref().map(|s| &s.k_scale)
    }

    /// Get the current V scale (for quantized mode).
    pub fn v_scale(&self) -> Option<&Tensor> {
        self.scales.as_ref().map(|s| &s.v_scale)
    }

    /// Raw K cache tensor (shape depends on layout).
    pub fn k_cache(&self) -> &Tensor {
        &self.k_cache
    }

    /// Raw V cache tensor (shape depends on layout).
    pub fn v_cache(&self) -> &Tensor {
        &self.v_cache
    }

    /// Get the memory layout of this cache engine.
    pub fn layout(&self) -> KVCacheLayout {
        self.layout
    }

    /// Quantize write data based on kv_cache_dtype setting.
    ///
    /// Input and output are in [new_tokens, kv_heads, head_dim] layout.
    /// `k_orig` and `v_orig` are the original (pre-transposed) tensors for calibration.
    fn quantize_write_data(
        &mut self,
        k_prepared: &Tensor,
        v_prepared: &Tensor,
        k_orig: &Tensor,
        v_orig: &Tensor,
        new_tokens: usize,
    ) -> Result<(Tensor, Tensor), CacheError> {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => Ok((k_prepared.clone(), v_prepared.clone())),
            KVCacheDtype::Fp8E4m3 => {
                let scales = self.scales.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("FP8 mode requires scales".to_string())
                })?;
                scales.calibrate(k_orig, v_orig)?;

                let k_2d = k_prepared.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;
                let v_2d = v_prepared.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;

                let k_quant = quantize_fp8(&k_2d, &scales.k_scale)?;
                let v_quant = quantize_fp8(&v_2d, &scales.v_scale)?;

                Ok((
                    k_quant.reshape((new_tokens, self.num_kv_heads, self.head_dim))?,
                    v_quant.reshape((new_tokens, self.num_kv_heads, self.head_dim))?,
                ))
            }
            KVCacheDtype::Int8 => {
                let scales = self.scales.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("INT8 mode requires scales".to_string())
                })?;

                scales.k_scale = compute_int8_scale(k_orig)?;
                scales.v_scale = compute_int8_scale(v_orig)?;

                let k_2d = k_prepared.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;
                let v_2d = v_prepared.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;

                let k_quant = quantize_int8(&k_2d, &scales.k_scale)?;
                let v_quant = quantize_int8(&v_2d, &scales.v_scale)?;

                Ok((
                    k_quant.reshape((new_tokens, self.num_kv_heads, self.head_dim))?,
                    v_quant.reshape((new_tokens, self.num_kv_heads, self.head_dim))?,
                ))
            }
        }
    }

    /// Scatter data into the cache, handling NHD/HND layout differences.
    ///
    /// `k_src`, `v_src`: [new_tokens, kv_heads, head_dim]
    /// `slot_mapping`: physical slot per token (length = new_tokens)
    ///
    /// GPU fast path (cuda-kernels feature, BF16/F16 only): uses CUDA kernel directly.
    /// CPU / quantized fallback:
    /// - NHD: reshape cache to [total_slots, H, D] (zero-copy view), scatter on dim 0.
    /// - HND: reshape cache to [total_rows, D], build HND-native indices, scatter on dim 0.
    fn scatter_into_cache(
        &mut self,
        k_src: &Tensor,
        v_src: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        // GPU fast path: CUDA kernel for unquantized BF16/F16
        #[cfg(feature = "cuda-kernels")]
        if self.k_cache.device().is_cuda() && !self.is_quantized() {
            return self.scatter_into_cache_cuda(k_src, v_src, slot_mapping);
        }

        self.scatter_into_cache_candle(k_src, v_src, slot_mapping)
    }

    /// CUDA kernel fast path for scatter.
    #[cfg(feature = "cuda-kernels")]
    fn scatter_into_cache_cuda(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        use crate::cuda_kernels::{reshape_and_cache_cuda, CudaCacheLayout};

        let new_tokens = slot_mapping.len();
        let device = self.k_cache.device().clone();

        // Build slot_mapping as U32 tensor on CUDA
        let slot_tensor = Tensor::from_vec(
            slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            (new_tokens,),
            &device,
        )?;

        let cuda_layout = match self.layout {
            KVCacheLayout::NHD => CudaCacheLayout::NHD,
            KVCacheLayout::HND => CudaCacheLayout::HND,
        };

        reshape_and_cache_cuda(
            &k_src.contiguous()?,
            &v_src.contiguous()?,
            &self.k_cache,
            &self.v_cache,
            &slot_tensor,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            cuda_layout,
        )?;

        Ok(())
    }

    /// Candle-based scatter fallback (CPU + quantized GPU).
    fn scatter_into_cache_candle(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        let new_tokens = slot_mapping.len();
        let device = self.k_cache.device().clone();

        match self.layout {
            KVCacheLayout::NHD => {
                // NHD [num_blocks, block_size, kv_heads, head_dim]
                // reshape to [total_slots, kv_heads, head_dim] is a view (no copy)
                let total_slots = self.num_blocks * self.block_size;
                let flat_shape = (total_slots, self.num_kv_heads, self.head_dim);
                let k_flat = self.k_cache.reshape(flat_shape)?;
                let v_flat = self.v_cache.reshape(flat_shape)?;

                // Build NHD indices: [new_tokens] → expand to [new_tokens, H, D]
                let indices = Tensor::from_vec(
                    slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
                    (new_tokens,),
                    &device,
                )?;
                let indices = indices
                    .reshape((new_tokens, 1, 1))?
                    .expand((new_tokens, self.num_kv_heads, self.head_dim))?
                    .contiguous()?;

                k_flat.scatter_set(&indices, k_src, 0)?;
                v_flat.scatter_set(&indices, v_src, 0)?;
            }
            KVCacheLayout::HND => {
                // HND [num_blocks, kv_heads, block_size, head_dim]
                // Flatten to [num_blocks * kv_heads * block_size, head_dim] and
                // build HND-native row indices. Each (token, head) pair maps to
                // a unique row without transposing the cache.
                let total_rows = self.num_blocks * self.num_kv_heads * self.block_size;
                let k_flat = self.k_cache.reshape((total_rows, self.head_dim))?;
                let v_flat = self.v_cache.reshape((total_rows, self.head_dim))?;

                // Build HND indices: for slot s, head h:
                //   block = s / block_size, offset = s % block_size
                //   row = block * (H * block_size) + h * block_size + offset
                let head_block_stride = self.num_kv_heads * self.block_size;
                let mut idx_data =
                    Vec::with_capacity(new_tokens * self.num_kv_heads * self.head_dim);
                for &slot in slot_mapping {
                    let block = slot / self.block_size;
                    let offset = slot % self.block_size;
                    let block_base = block * head_block_stride;
                    for h in 0..self.num_kv_heads {
                        let row = block_base + h * self.block_size + offset;
                        for _d in 0..self.head_dim {
                            idx_data.push(row as u32);
                        }
                    }
                }
                let hnd_indices = Tensor::from_vec(
                    idx_data,
                    (new_tokens * self.num_kv_heads, self.head_dim),
                    &device,
                )?;

                // Reshape src from [new_tokens, H, D] to [new_tokens * H, D]
                let k_2d = k_src.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;
                let v_2d = v_src.reshape((new_tokens * self.num_kv_heads, self.head_dim))?;

                k_flat.scatter_set(&hnd_indices, &k_2d, 0)?;
                v_flat.scatter_set(&hnd_indices, &v_2d, 0)?;
            }
        }

        Ok(())
    }

    /// Flatten selected blocks to [total_capacity, kv_heads, head_dim], handling layout.
    ///
    /// For NHD: direct reshape (no copy).
    /// For HND: transpose dims 1↔2 then reshape (requires copy).
    fn flatten_read_blocks(
        &self,
        k_raw: &Tensor,
        v_raw: &Tensor,
        _num_blocks_used: usize,
        total_capacity: usize,
    ) -> Result<(Tensor, Tensor), CacheError> {
        let flat_shape = (total_capacity, self.num_kv_heads, self.head_dim);

        match self.layout {
            KVCacheLayout::NHD => {
                // NHD: [B, block_size, kv_heads, head_dim] → reshape
                Ok((k_raw.reshape(flat_shape)?, v_raw.reshape(flat_shape)?))
            }
            KVCacheLayout::HND => {
                // HND: [B, kv_heads, block_size, head_dim] → transpose to NHD → reshape
                let k_nhd = k_raw.transpose(1, 2)?.contiguous()?;
                let v_nhd = v_raw.transpose(1, 2)?.contiguous()?;
                Ok((k_nhd.reshape(flat_shape)?, v_nhd.reshape(flat_shape)?))
            }
        }
    }

    /// Dequantize read data based on kv_cache_dtype.
    ///
    /// Input and output are in [total_capacity, kv_heads, head_dim] layout.
    fn dequantize_read_data(
        &self,
        k_flat: &Tensor,
        v_flat: &Tensor,
        total_capacity: usize,
    ) -> Result<(Tensor, Tensor), CacheError> {
        match self.kv_cache_dtype {
            KVCacheDtype::Auto => Ok((k_flat.clone(), v_flat.clone())),
            KVCacheDtype::Fp8E4m3 => {
                let scales = self.scales.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("FP8 mode requires scales".to_string())
                })?;
                let k_2d = k_flat.reshape((total_capacity * self.num_kv_heads, self.head_dim))?;
                let v_2d = v_flat.reshape((total_capacity * self.num_kv_heads, self.head_dim))?;
                let k_dq = dequantize_fp8(&k_2d, &scales.k_scale, self.compute_dtype)?;
                let v_dq = dequantize_fp8(&v_2d, &scales.v_scale, self.compute_dtype)?;
                Ok((
                    k_dq.reshape((total_capacity, self.num_kv_heads, self.head_dim))?,
                    v_dq.reshape((total_capacity, self.num_kv_heads, self.head_dim))?,
                ))
            }
            KVCacheDtype::Int8 => {
                let scales = self.scales.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("INT8 mode requires scales".to_string())
                })?;
                let k_2d = k_flat.reshape((total_capacity * self.num_kv_heads, self.head_dim))?;
                let v_2d = v_flat.reshape((total_capacity * self.num_kv_heads, self.head_dim))?;
                let k_dq = dequantize_int8(&k_2d, &scales.k_scale, self.compute_dtype)?;
                let v_dq = dequantize_int8(&v_2d, &scales.v_scale, self.compute_dtype)?;
                Ok((
                    k_dq.reshape((total_capacity, self.num_kv_heads, self.head_dim))?,
                    v_dq.reshape((total_capacity, self.num_kv_heads, self.head_dim))?,
                ))
            }
        }
    }

    /// Reset cache contents to zeros without reallocating.
    ///
    /// This ensures consistency when restarting generation — stale KV data
    /// from previous requests is cleared.
    pub fn reset(&mut self) -> Result<(), CacheError> {
        self.k_cache = self.k_cache.zeros_like()?;
        self.v_cache = self.v_cache.zeros_like()?;
        if let Some(ref mut scales) = self.scales {
            scales.reset()?;
        }
        Ok(())
    }

    /// Write K, V for new tokens into their assigned slots.
    ///
    /// k, v shape: [num_kv_heads, new_tokens, head_dim]
    /// slot_mapping: physical slot IDs (length = new_tokens)
    ///
    /// For quantized mode, data is quantized before writing and scales are
    /// updated based on observed values.
    pub fn write(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        let new_tokens = slot_mapping.len();

        // Input [kv_heads, new_tokens, head_dim] → [new_tokens, kv_heads, head_dim]
        let k_transposed = k.transpose(0, 1)?.contiguous()?;
        let v_transposed = v.transpose(0, 1)?.contiguous()?;

        // Apply quantization if enabled
        let (k_src, v_src) =
            self.quantize_write_data(&k_transposed, &v_transposed, k, v, new_tokens)?;

        // Scatter into cache (layout-aware)
        self.scatter_into_cache(&k_src, &v_src, slot_mapping)?;

        Ok(())
    }

    /// Write K, V tokens in [new_tokens, kv_heads, head_dim] layout.
    /// Used by batched decode where tokens from different sequences are concatenated.
    pub fn write_batch(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        let new_tokens = slot_mapping.len();

        // Input already in [new_tokens, kv_heads, head_dim]
        let k_contiguous = k.contiguous()?;
        let v_contiguous = v.contiguous()?;

        // Apply quantization if enabled
        let (k_src, v_src) =
            self.quantize_write_data(&k_contiguous, &v_contiguous, k, v, new_tokens)?;

        self.scatter_into_cache(&k_src, &v_src, slot_mapping)?;

        Ok(())
    }

    /// Read K, V for multiple sequences concatenated contiguously.
    /// Returns (k, v) each with shape [total_tokens, num_kv_heads, head_dim]
    /// in compute dtype (dequantized if cache is quantized).
    /// sequences: list of (block_ids, num_tokens) for each sequence.
    pub fn read_contiguous_multi(
        &self,
        sequences: &[(&[BlockId], usize)],
    ) -> Result<(Tensor, Tensor), CacheError> {
        let mut k_parts = Vec::with_capacity(sequences.len());
        let mut v_parts = Vec::with_capacity(sequences.len());

        for &(block_ids, num_tokens) in sequences {
            if num_tokens == 0 {
                continue;
            }
            let num_blocks_used = block_ids.len();
            let indices = Tensor::from_vec(
                block_ids.iter().map(|&b| b as u32).collect::<Vec<_>>(),
                (num_blocks_used,),
                self.k_cache.device(),
            )?;

            // index_select on dim 0: selects blocks
            let k_raw = self.k_cache.index_select(&indices, 0)?;
            let v_raw = self.v_cache.index_select(&indices, 0)?;

            // Flatten to [total_capacity, kv_heads, head_dim] (layout-aware)
            let total_capacity = num_blocks_used * self.block_size;
            let (k_flat, v_flat) =
                self.flatten_read_blocks(&k_raw, &v_raw, num_blocks_used, total_capacity)?;

            // Apply dequantization
            let (k, v) = self.dequantize_read_data(&k_flat, &v_flat, total_capacity)?;

            let k = k.narrow(0, 0, num_tokens)?;
            let v = v.narrow(0, 0, num_tokens)?;

            k_parts.push(k);
            v_parts.push(v);
        }

        let k = Tensor::cat(&k_parts, 0)?;
        let v = Tensor::cat(&v_parts, 0)?;
        Ok((k, v))
    }

    /// Read K, V for all tokens of a request.
    ///
    /// block_ids: ordered physical block IDs from BlockTable
    /// num_tokens: total valid tokens (to narrow partial last block)
    ///
    /// Returns (k, v) each with shape [1, num_kv_heads, num_tokens, head_dim]
    /// in compute dtype (dequantized if cache is quantized).
    pub fn read(
        &self,
        block_ids: &[BlockId],
        num_tokens: usize,
    ) -> Result<(Tensor, Tensor), CacheError> {
        let num_blocks_used = block_ids.len();

        let indices = Tensor::from_vec(
            block_ids.iter().map(|&b| b as u32).collect::<Vec<_>>(),
            (num_blocks_used,),
            self.k_cache.device(),
        )?;

        // index_select on dim 0: selects blocks
        let k_raw = self.k_cache.index_select(&indices, 0)?;
        let v_raw = self.v_cache.index_select(&indices, 0)?;

        // Flatten to [total_capacity, kv_heads, head_dim] (layout-aware)
        let total_capacity = num_blocks_used * self.block_size;
        let (k_flat, v_flat) =
            self.flatten_read_blocks(&k_raw, &v_raw, num_blocks_used, total_capacity)?;

        // Apply dequantization
        let (k, v) = self.dequantize_read_data(&k_flat, &v_flat, total_capacity)?;

        // Narrow to actual tokens, transpose to [kv_heads, num_tokens, head_dim]
        let k = k.narrow(0, 0, num_tokens)?;
        let k = k.transpose(0, 1)?.contiguous()?;
        let k = k.unsqueeze(0)?;

        let v = v.narrow(0, 0, num_tokens)?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let v = v.unsqueeze(0)?;

        Ok((k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::KVCacheLayout;
    use candle_core::{DType, Device, Tensor};

    fn test_config(num_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    fn test_config_fp8(num_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Fp8E4m3,
            cpu_offload: None,
        }
    }

    fn test_config_int8(num_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Int8,
            cpu_offload: None,
        }
    }

    #[test]
    fn new_allocates_correct_shape() {
        let config = test_config(8);
        let engine = CacheEngine::new(&config).unwrap();
        // NHD layout: [num_blocks, block_size, kv_heads, head_dim]
        assert_eq!(engine.k_cache.dims(), &[8, 4, 2, 8]);
        assert_eq!(engine.v_cache.dims(), &[8, 4, 2, 8]);
        assert_eq!(engine.layout(), KVCacheLayout::NHD);
        assert!(!engine.is_quantized());
    }

    #[test]
    fn new_with_hnd_layout_allocates_correct_shape() {
        let config = test_config(8);
        let engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();
        // HND layout: [num_blocks, kv_heads, block_size, head_dim]
        assert_eq!(engine.k_cache.dims(), &[8, 2, 4, 8]);
        assert_eq!(engine.v_cache.dims(), &[8, 2, 4, 8]);
        assert_eq!(engine.layout(), KVCacheLayout::HND);
    }

    #[test]
    fn new_fp8_allocates_u8_tensors() {
        let config = test_config_fp8(8);
        let engine = CacheEngine::new(&config).unwrap();
        assert_eq!(engine.k_cache.dims(), &[8, 4, 2, 8]);
        assert_eq!(engine.k_cache.dtype(), DType::U8);
        assert_eq!(engine.v_cache.dtype(), DType::U8);
        assert!(engine.is_quantized());
        assert!(engine.k_scale().is_some());
        assert!(engine.v_scale().is_some());
    }

    #[test]
    fn new_int8_allocates_u8_tensors() {
        let config = test_config_int8(8);
        let engine = CacheEngine::new(&config).unwrap();
        assert_eq!(engine.k_cache.dtype(), DType::U8);
        assert_eq!(engine.v_cache.dtype(), DType::U8);
        assert!(engine.is_quantized());
    }

    #[test]
    fn write_read_roundtrip() {
        let config = test_config(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // Write 3 tokens to block 2 (slots 8, 9, 10)
        // k shape: [kv_heads=2, new_tokens=3, head_dim=8]
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![8, 9, 10]; // block 2, offsets 0,1,2

        engine.write(&k, &v, &slot_mapping).unwrap();

        // Read back from block 2, 3 tokens
        let (k_out, _v_out) = engine.read(&[2], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        // Read output [1, kv_heads, tokens, head_dim] should match input [kv_heads, tokens, head_dim]
        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let k_orig_flat: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k_read_flat, k_orig_flat);
    }

    #[test]
    fn write_read_roundtrip_fp8() {
        let config = test_config_fp8(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // Write 3 tokens with reasonable values for FP8
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 0.1).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![8, 9, 10];

        engine.write(&k, &v, &slot_mapping).unwrap();

        // Read back
        let (k_out, _) = engine.read(&[2], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        // FP8 has limited precision, allow some error
        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, read) in k_data.iter().zip(k_read_flat.iter()) {
            let abs_error = (orig - read).abs();
            let rel_error = if orig.abs() > 1e-6 {
                abs_error / orig.abs()
            } else {
                abs_error
            };
            assert!(
                rel_error < 0.2 || abs_error < 0.5,
                "FP8 roundtrip error too large: orig={}, read={}, rel_error={}",
                orig,
                read,
                rel_error
            );
        }
    }

    #[test]
    fn write_read_roundtrip_int8() {
        let config = test_config_int8(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 0.1).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![8, 9, 10];

        engine.write(&k, &v, &slot_mapping).unwrap();

        let (k_out, _) = engine.read(&[2], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, read) in k_data.iter().zip(k_read_flat.iter()) {
            let abs_error = (orig - read).abs();
            let rel_error = if orig.abs() > 1e-6 {
                abs_error / orig.abs()
            } else {
                abs_error
            };
            assert!(
                rel_error < 0.15 || abs_error < 0.3,
                "INT8 roundtrip error too large: orig={}, read={}, rel_error={}",
                orig,
                read,
                rel_error
            );
        }
    }

    #[test]
    fn write_non_contiguous_slots() {
        let config = test_config(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // Write 2 tokens: slot 1 (block 0, offset 1) and slot 12 (block 3, offset 0)
        let k_data: Vec<f32> = (0..2 * 2 * 8).map(|i| (i + 1) as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 2, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 2, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![1, 12];

        engine.write(&k, &v, &slot_mapping).unwrap();

        // Read block 0 (2 tokens): slot 0 = zeros, slot 1 = first token
        let (k_out, _) = engine.read(&[0], 2).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 2, 8]);

        // In output [1, kv_heads=2, tokens=2, head_dim=8]:
        // token 0 (slot 0) should be all zeros
        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        // head 0, token 0: positions 0..8
        assert!(k_flat[0..8].iter().all(|&x| x == 0.0));
        // head 1, token 0: positions 16..24
        assert!(k_flat[16..24].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn read_multi_block_partial() {
        let config = test_config(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // Write 6 tokens: 4 in block 1 + 2 in block 5
        let k_data: Vec<f32> = (0..2 * 6 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 6, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 6, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![4, 5, 6, 7, 20, 21];

        engine.write(&k, &v, &slot_mapping).unwrap();

        // Read blocks [1, 5] with 6 valid tokens (narrows away 2 unused slots in block 5)
        let (k_out, _) = engine.read(&[1, 5], 6).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 6, 8]);

        // Verify data matches original
        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let k_orig_flat: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k_read_flat, k_orig_flat);
    }

    #[test]
    fn memory_reduction_fp8() {
        // Verify that FP8 config uses half the memory
        let budget = 1024 * 1024; // 1 MB
        let config_auto =
            CacheConfig::from_memory_budget(budget, 1, 2, 8, 4, DType::BF16, Device::Cpu);
        let config_fp8 = CacheConfig::from_memory_budget_with_kv_dtype(
            budget,
            1,
            2,
            8,
            4,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Fp8E4m3,
        );

        // FP8 should get 2x the blocks
        assert_eq!(config_fp8.num_blocks, config_auto.num_blocks * 2);

        // Both engines allocate successfully
        let engine_auto = CacheEngine::new(&config_auto).unwrap();
        let engine_fp8 = CacheEngine::new(&config_fp8).unwrap();

        assert!(!engine_auto.is_quantized());
        assert!(engine_fp8.is_quantized());
    }

    #[test]
    fn memory_reduction_int8() {
        // Verify that INT8 config also uses half the memory
        let budget = 1024 * 1024; // 1 MB
        let config_auto =
            CacheConfig::from_memory_budget(budget, 1, 2, 8, 4, DType::BF16, Device::Cpu);
        let config_int8 = CacheConfig::from_memory_budget_with_kv_dtype(
            budget,
            1,
            2,
            8,
            4,
            DType::BF16,
            Device::Cpu,
            KVCacheDtype::Int8,
        );

        // INT8 should get 2x the blocks
        assert_eq!(config_int8.num_blocks, config_auto.num_blocks * 2);

        let engine_int8 = CacheEngine::new(&config_int8).unwrap();
        assert!(engine_int8.is_quantized());
    }

    #[test]
    fn scale_calibration_during_write() {
        let config = test_config_fp8(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // Initial scales should be 1.0 (stored as [1] shape tensors)
        let initial_k_scale: Vec<f32> = engine
            .k_scale()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let initial_v_scale: Vec<f32> = engine
            .v_scale()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!((initial_k_scale[0] - 1.0).abs() < 1e-6);
        assert!((initial_v_scale[0] - 1.0).abs() < 1e-6);

        // Write data with larger range
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| (i as f32) * 2.0).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();

        engine.write(&k, &v, &[0, 1, 2]).unwrap();

        // Scales should now be calibrated based on the data range (may be scalar or [1])
        let k_scale: Vec<f32> = engine
            .k_scale()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let v_scale: Vec<f32> = engine
            .v_scale()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(k_scale[0] > 0.0, "K scale should be positive");
        assert!(v_scale[0] > 0.0, "V scale should be positive");
    }

    #[test]
    fn write_batch_fp8_roundtrip() {
        let config = test_config_fp8(8);
        let mut engine = CacheEngine::new(&config).unwrap();

        // write_batch uses [new_tokens, kv_heads, head_dim] layout
        // Write to contiguous slots in block 0 only
        let new_tokens = 3;
        let num_heads = 2;
        let head_dim = 8;

        // Shape: [new_tokens, heads, head_dim]
        let k_data: Vec<f32> = (0..new_tokens * num_heads * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let k = Tensor::from_vec(
            k_data.clone(),
            (new_tokens, num_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();
        let v = Tensor::from_vec(
            k_data.clone(),
            (new_tokens, num_heads, head_dim),
            &Device::Cpu,
        )
        .unwrap();

        // Write 3 tokens to block 0 (slots 0, 1, 2)
        let slot_mapping = vec![0, 1, 2];

        engine.write_batch(&k, &v, &slot_mapping).unwrap();

        // Read back from block 0 with 3 tokens
        let (k_out, _) = engine.read(&[0], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        // read() returns [num_blocks=1, kv_heads=2, seq_len=3, head_dim=8]
        // write_batch writes [new_tokens=3, kv_heads=2, head_dim=8]
        // The layout differs: write is token-major, read is head-major
        // Just verify the data can be read back (shapes match)
        let k_read: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k_read.len(), 3 * 2 * 8);
    }

    #[test]
    fn quantized_preserves_zero() {
        // Test that zero values are preserved through quantization
        let config_fp8 = test_config_fp8(8);
        let mut engine = CacheEngine::new(&config_fp8).unwrap();

        // Create tensor with zeros
        let k = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((2, 3, 8), DType::F32, &Device::Cpu).unwrap();

        engine.write(&k, &v, &[0, 1, 2]).unwrap();

        let (k_out, v_out) = engine.read(&[0], 3).unwrap();
        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let v_flat: Vec<f32> = v_out.flatten_all().unwrap().to_vec1().unwrap();

        // All values should be zero
        assert!(k_flat.iter().all(|&x| x == 0.0), "K should preserve zeros");
        assert!(v_flat.iter().all(|&x| x == 0.0), "V should preserve zeros");
    }

    #[test]
    fn quantized_handles_negative_values() {
        let config_int8 = test_config_int8(8);
        let mut engine = CacheEngine::new(&config_int8).unwrap();

        // Create tensor with negative values
        let k_data: Vec<f32> = (0..2 * 3 * 8)
            .map(|i: usize| {
                if i.is_multiple_of(2) {
                    i as f32 * 0.1
                } else {
                    -(i as f32) * 0.1
                }
            })
            .collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();

        engine.write(&k, &v, &[0, 1, 2]).unwrap();

        let (k_out, _) = engine.read(&[0], 3).unwrap();
        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();

        // Check signs are preserved
        for (orig, &read) in k_data.iter().zip(k_flat.iter()) {
            assert_eq!(
                orig.signum(),
                read.signum(),
                "Sign should be preserved: orig={}, read={}",
                orig,
                read
            );
        }
    }

    // ─── HND layout tests ───────────────────────────────────────────────────

    fn test_config_hnd(num_blocks: usize) -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn hnd_write_read_roundtrip() {
        let config = test_config_hnd(8);
        let mut engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();
        assert_eq!(engine.layout(), KVCacheLayout::HND);
        assert_eq!(engine.k_cache.dims(), &[8, 2, 4, 8]); // [blocks, heads, block_size, head_dim]

        // Write 3 tokens to block 2 (slots 8, 9, 10)
        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![8, 9, 10];

        engine.write(&k, &v, &slot_mapping).unwrap();

        // Cache should still be HND after write
        assert_eq!(engine.k_cache.dims(), &[8, 2, 4, 8]);

        let (k_out, _v_out) = engine.read(&[2], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let k_orig_flat: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k_read_flat, k_orig_flat);
    }

    #[test]
    fn hnd_write_batch_roundtrip() {
        let config = test_config_hnd(8);
        let mut engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        // write_batch: [new_tokens, kv_heads, head_dim]
        let k_data: Vec<f32> = (0..3 * 2 * 8).map(|i| (i + 1) as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (3, 2, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (3, 2, 8), &Device::Cpu).unwrap();

        engine.write_batch(&k, &v, &[0, 1, 2]).unwrap();

        let (k_out, _) = engine.read(&[0], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);
    }

    #[test]
    fn hnd_write_non_contiguous_slots() {
        let config = test_config_hnd(8);
        let mut engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        // Write 2 tokens: slot 1 (block 0, offset 1) and slot 12 (block 3, offset 0)
        let k_data: Vec<f32> = (0..2 * 2 * 8).map(|i| (i + 1) as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 2, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 2, 8), &Device::Cpu).unwrap();

        engine.write(&k, &v, &[1, 12]).unwrap();

        // Read block 0 (2 tokens): slot 0 = zeros, slot 1 = first token
        let (k_out, _) = engine.read(&[0], 2).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 2, 8]);

        let k_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        // head 0, token 0: positions 0..8 should be zeros
        assert!(k_flat[0..8].iter().all(|&x| x == 0.0));
        // head 1, token 0: positions 16..24 should be zeros
        assert!(k_flat[16..24].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hnd_read_multi_block_partial() {
        let config = test_config_hnd(8);
        let mut engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        // Write 6 tokens: 4 in block 1 + 2 in block 5
        let k_data: Vec<f32> = (0..2 * 6 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 6, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 6, 8), &Device::Cpu).unwrap();
        let slot_mapping = vec![4, 5, 6, 7, 20, 21];

        engine.write(&k, &v, &slot_mapping).unwrap();

        let (k_out, _) = engine.read(&[1, 5], 6).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 6, 8]);

        let k_read_flat: Vec<f32> = k_out.flatten_all().unwrap().to_vec1().unwrap();
        let k_orig_flat: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(k_read_flat, k_orig_flat);
    }

    #[test]
    fn hnd_read_contiguous_multi() {
        let config = test_config_hnd(8);
        let mut engine = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        // Write data into blocks 0 and 2
        let k1_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k1 = Tensor::from_vec(k1_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v1 = Tensor::from_vec(k1_data, (2, 3, 8), &Device::Cpu).unwrap();
        engine.write(&k1, &v1, &[0, 1, 2]).unwrap();

        let k2_data: Vec<f32> = (100..100 + 2 * 2 * 8).map(|i| i as f32).collect();
        let k2 = Tensor::from_vec(k2_data.clone(), (2, 2, 8), &Device::Cpu).unwrap();
        let v2 = Tensor::from_vec(k2_data, (2, 2, 8), &Device::Cpu).unwrap();
        engine.write(&k2, &v2, &[8, 9]).unwrap();

        let (k_out, _) = engine
            .read_contiguous_multi(&[(&[0usize], 3), (&[2], 2)])
            .unwrap();
        assert_eq!(k_out.dims(), &[5, 2, 8]); // 3 + 2 tokens
    }

    // ─── NHD vs HND equivalence ─────────────────────────────────────────────

    #[test]
    fn nhd_hnd_equivalence_write_read() {
        // Write identical data into NHD and HND engines, read back, verify same result
        let config = test_config(8);
        let mut nhd = CacheEngine::new(&config).unwrap();
        let mut hnd = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        let k_data: Vec<f32> = (0..2 * 6 * 8).map(|i| (i as f32) * 0.1).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 6, 8), &Device::Cpu).unwrap();
        let v_data: Vec<f32> = (0..2 * 6 * 8).map(|i| (i as f32) * -0.05 + 1.0).collect();
        let v = Tensor::from_vec(v_data.clone(), (2, 6, 8), &Device::Cpu).unwrap();

        // Non-contiguous slots across multiple blocks
        let slot_mapping = vec![4, 5, 6, 7, 20, 21];

        nhd.write(&k, &v, &slot_mapping).unwrap();
        hnd.write(&k, &v, &slot_mapping).unwrap();

        // Read back from both
        let (k_nhd, v_nhd) = nhd.read(&[1, 5], 6).unwrap();
        let (k_hnd, v_hnd) = hnd.read(&[1, 5], 6).unwrap();

        let k_nhd_flat: Vec<f32> = k_nhd.flatten_all().unwrap().to_vec1().unwrap();
        let k_hnd_flat: Vec<f32> = k_hnd.flatten_all().unwrap().to_vec1().unwrap();
        let v_nhd_flat: Vec<f32> = v_nhd.flatten_all().unwrap().to_vec1().unwrap();
        let v_hnd_flat: Vec<f32> = v_hnd.flatten_all().unwrap().to_vec1().unwrap();

        assert_eq!(
            k_nhd_flat, k_hnd_flat,
            "K values must match between NHD and HND"
        );
        assert_eq!(
            v_nhd_flat, v_hnd_flat,
            "V values must match between NHD and HND"
        );
    }

    #[test]
    fn nhd_hnd_equivalence_write_batch() {
        let config = test_config(8);
        let mut nhd = CacheEngine::new(&config).unwrap();
        let mut hnd = CacheEngine::with_layout(&config, KVCacheLayout::HND).unwrap();

        // write_batch input: [new_tokens, kv_heads, head_dim]
        let k_data: Vec<f32> = (0..4 * 2 * 8).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let k = Tensor::from_vec(k_data.clone(), (4, 2, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (4, 2, 8), &Device::Cpu).unwrap();

        // Scatter across blocks
        let slot_mapping = vec![1, 5, 10, 25];

        nhd.write_batch(&k, &v, &slot_mapping).unwrap();
        hnd.write_batch(&k, &v, &slot_mapping).unwrap();

        // Read individual blocks and compare
        for block_id in [0usize, 1, 2, 6] {
            let (k_nhd, v_nhd) = nhd.read(&[block_id], 4).unwrap();
            let (k_hnd, v_hnd) = hnd.read(&[block_id], 4).unwrap();

            let k_nhd_flat: Vec<f32> = k_nhd.flatten_all().unwrap().to_vec1().unwrap();
            let k_hnd_flat: Vec<f32> = k_hnd.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(k_nhd_flat, k_hnd_flat, "K mismatch in block {block_id}");

            let v_nhd_flat: Vec<f32> = v_nhd.flatten_all().unwrap().to_vec1().unwrap();
            let v_hnd_flat: Vec<f32> = v_hnd.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(v_nhd_flat, v_hnd_flat, "V mismatch in block {block_id}");
        }
    }
}
