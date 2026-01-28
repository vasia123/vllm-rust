//! MLA Cache Engine for DeepSeek V2/V3.
//!
//! Stores compressed KV latent representations instead of full K/V tensors,
//! achieving 42x memory reduction for DeepSeek models.

use candle_core::{DType, Module, Tensor};
use candle_nn::Linear;

use super::block_pool::BlockId;
use super::error::CacheError;
use super::mla_cache_config::MLACacheConfig;
use super::quantization::{
    compute_int8_scale, dequantize_fp8, dequantize_int8, quantize_fp8, quantize_int8, KVCacheDtype,
};

type Result<T> = std::result::Result<T, CacheError>;

/// MLA Cache Engine for compressed KV storage.
///
/// Instead of storing full K [num_heads, head_dim] and V [num_heads, v_head_dim],
/// MLA stores:
/// - kv_c_cache: Compressed latent [kv_lora_rank]
/// - k_pe_cache: RoPE key component [qk_rope_head_dim]
///
/// This reduces memory from ~49KB/token to ~1.2KB/token for DeepSeek V3.
pub struct MLACacheEngine {
    /// Compressed KV latent: [num_blocks, block_size, kv_lora_rank]
    kv_c_cache: Tensor,
    /// RoPE key component: [num_blocks, block_size, qk_rope_head_dim]
    k_pe_cache: Tensor,
    /// Configuration
    config: MLACacheConfig,
    /// Scales for quantized mode
    scales: Option<MLAScales>,
}

/// Scales for quantized MLA cache.
#[derive(Debug, Clone)]
pub struct MLAScales {
    /// Scale for kv_c cache
    pub kv_c_scale: Tensor,
    /// Scale for k_pe cache
    pub k_pe_scale: Tensor,
}

impl MLAScales {
    /// Create new scales initialized to 1.0.
    pub fn new(device: &candle_core::Device) -> candle_core::Result<Self> {
        Ok(Self {
            kv_c_scale: Tensor::ones(1, DType::F32, device)?,
            k_pe_scale: Tensor::ones(1, DType::F32, device)?,
        })
    }

    /// Update scales based on observed data.
    pub fn calibrate(&mut self, kv_c: &Tensor, k_pe: &Tensor) -> candle_core::Result<()> {
        let abs_kv_c = kv_c.abs()?.to_dtype(DType::F32)?;
        let abs_k_pe = k_pe.abs()?.to_dtype(DType::F32)?;

        let max_kv_c = abs_kv_c.flatten_all()?.max(0)?;
        let max_k_pe = abs_k_pe.flatten_all()?.max(0)?;

        // FP8 E4M3 max = 448.0
        const FP8_MAX: f64 = 448.0;
        const SCALE_MIN: f64 = 1e-12;

        self.kv_c_scale = (max_kv_c / FP8_MAX)?.maximum(SCALE_MIN)?;
        self.k_pe_scale = (max_k_pe / FP8_MAX)?.maximum(SCALE_MIN)?;

        Ok(())
    }
}

impl MLACacheEngine {
    /// Create a new MLA cache engine.
    pub fn new(config: &MLACacheConfig) -> Result<Self> {
        let storage_dtype = config.storage_dtype();

        // kv_c_cache: [num_blocks, block_size, kv_lora_rank]
        let kv_c_cache = Tensor::zeros(
            (config.num_blocks, config.block_size, config.kv_lora_rank),
            storage_dtype,
            &config.device,
        )?;

        // k_pe_cache: [num_blocks, block_size, qk_rope_head_dim]
        let k_pe_cache = Tensor::zeros(
            (
                config.num_blocks,
                config.block_size,
                config.qk_rope_head_dim,
            ),
            storage_dtype,
            &config.device,
        )?;

        let scales = if config.is_quantized() {
            Some(MLAScales::new(&config.device)?)
        } else {
            None
        };

        Ok(Self {
            kv_c_cache,
            k_pe_cache,
            config: config.clone(),
            scales,
        })
    }

    /// Write compressed latent and RoPE key to cache.
    ///
    /// # Arguments
    /// * `kv_c` - Compressed KV latent [num_tokens, kv_lora_rank]
    /// * `k_pe` - RoPE key component [num_tokens, qk_rope_head_dim]
    /// * `slot_mapping` - Physical slot IDs for each token
    pub fn write(&mut self, kv_c: &Tensor, k_pe: &Tensor, slot_mapping: &[usize]) -> Result<()> {
        let num_tokens = slot_mapping.len();
        let total_slots = self.config.num_blocks * self.config.block_size;

        // Reshape caches to flat [total_slots, dim]
        let kv_c_flat = self
            .kv_c_cache
            .reshape((total_slots, self.config.kv_lora_rank))?;
        let k_pe_flat = self
            .k_pe_cache
            .reshape((total_slots, self.config.qk_rope_head_dim))?;

        // Ensure input is contiguous
        let kv_c = kv_c.contiguous()?;
        let k_pe = k_pe.contiguous()?;

        // Apply quantization if enabled
        let (kv_c_src, k_pe_src) = match self.config.kv_cache_dtype {
            KVCacheDtype::Auto => (kv_c, k_pe),
            KVCacheDtype::Fp8E4m3 => {
                let scales = self.scales.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("FP8 mode requires scales".to_string())
                })?;
                scales.calibrate(&kv_c, &k_pe)?;

                let kv_c_quant = quantize_fp8(&kv_c, &scales.kv_c_scale)?;
                let k_pe_quant = quantize_fp8(&k_pe, &scales.k_pe_scale)?;

                (kv_c_quant, k_pe_quant)
            }
            KVCacheDtype::Int8 => {
                let scales = self.scales.as_mut().ok_or_else(|| {
                    candle_core::Error::Msg("INT8 mode requires scales".to_string())
                })?;

                scales.kv_c_scale = compute_int8_scale(&kv_c)?;
                scales.k_pe_scale = compute_int8_scale(&k_pe)?;

                let kv_c_quant = quantize_int8(&kv_c, &scales.kv_c_scale)?;
                let k_pe_quant = quantize_int8(&k_pe, &scales.k_pe_scale)?;

                (kv_c_quant, k_pe_quant)
            }
        };

        // Build scatter indices
        let kv_c_indices = Tensor::from_vec(
            slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            (num_tokens,),
            kv_c_flat.device(),
        )?;
        let kv_c_indices = kv_c_indices
            .reshape((num_tokens, 1))?
            .expand((num_tokens, self.config.kv_lora_rank))?
            .contiguous()?;

        let k_pe_indices = Tensor::from_vec(
            slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            (num_tokens,),
            k_pe_flat.device(),
        )?;
        let k_pe_indices = k_pe_indices
            .reshape((num_tokens, 1))?
            .expand((num_tokens, self.config.qk_rope_head_dim))?
            .contiguous()?;

        // Scatter write
        kv_c_flat.scatter_set(&kv_c_indices, &kv_c_src, 0)?;
        k_pe_flat.scatter_set(&k_pe_indices, &k_pe_src, 0)?;

        Ok(())
    }

    /// Read raw compressed latent and RoPE key from cache.
    ///
    /// # Arguments
    /// * `block_ids` - Block IDs to read from
    /// * `num_tokens` - Number of valid tokens to read
    ///
    /// # Returns
    /// (kv_c, k_pe) each with shape [num_tokens, dim]
    pub fn read_raw(&self, block_ids: &[BlockId], num_tokens: usize) -> Result<(Tensor, Tensor)> {
        let num_blocks_used = block_ids.len();

        let indices = Tensor::from_vec(
            block_ids.iter().map(|&b| b as u32).collect::<Vec<_>>(),
            (num_blocks_used,),
            self.kv_c_cache.device(),
        )?;

        // Index select blocks
        let kv_c_blocks = self.kv_c_cache.index_select(&indices, 0)?;
        let k_pe_blocks = self.k_pe_cache.index_select(&indices, 0)?;

        // Reshape to flat
        let total_capacity = num_blocks_used * self.config.block_size;
        let kv_c_flat = kv_c_blocks.reshape((total_capacity, self.config.kv_lora_rank))?;
        let k_pe_flat = k_pe_blocks.reshape((total_capacity, self.config.qk_rope_head_dim))?;

        // Apply dequantization if needed
        let (kv_c, k_pe) = match self.config.kv_cache_dtype {
            KVCacheDtype::Auto => (kv_c_flat, k_pe_flat),
            KVCacheDtype::Fp8E4m3 => {
                let scales = self.scales.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("FP8 mode requires scales".to_string())
                })?;

                let kv_c_dequant =
                    dequantize_fp8(&kv_c_flat, &scales.kv_c_scale, self.config.dtype)?;
                let k_pe_dequant =
                    dequantize_fp8(&k_pe_flat, &scales.k_pe_scale, self.config.dtype)?;

                (kv_c_dequant, k_pe_dequant)
            }
            KVCacheDtype::Int8 => {
                let scales = self.scales.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("INT8 mode requires scales".to_string())
                })?;

                let kv_c_dequant =
                    dequantize_int8(&kv_c_flat, &scales.kv_c_scale, self.config.dtype)?;
                let k_pe_dequant =
                    dequantize_int8(&k_pe_flat, &scales.k_pe_scale, self.config.dtype)?;

                (kv_c_dequant, k_pe_dequant)
            }
        };

        // Narrow to actual tokens
        let kv_c = kv_c.narrow(0, 0, num_tokens)?;
        let k_pe = k_pe.narrow(0, 0, num_tokens)?;

        Ok((kv_c, k_pe))
    }

    /// Read and expand KV for prefill attention.
    ///
    /// Expands the compressed latent through kv_b_proj to get full K/V tensors.
    /// This is compute-friendly for prefill where we need full attention.
    ///
    /// # Arguments
    /// * `block_ids` - Block IDs to read from
    /// * `num_tokens` - Number of valid tokens
    /// * `kv_b_proj` - Projection layer to expand kv_c
    ///
    /// # Returns
    /// (k_nope, k_pe, v) where:
    /// - k_nope: [num_tokens, num_heads, qk_nope_head_dim]
    /// - k_pe: [num_tokens, qk_rope_head_dim] (broadcast to heads in attention)
    /// - v: [num_tokens, num_heads, v_head_dim]
    pub fn read_expand_prefill(
        &self,
        block_ids: &[BlockId],
        num_tokens: usize,
        kv_b_proj: &Linear,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (kv_c, k_pe) = self.read_raw(block_ids, num_tokens)?;

        // Expand kv_c through projection: [num_tokens, kv_lora_rank] -> [num_tokens, num_heads * (qk_nope + v)]
        let kv_expanded = kv_b_proj.forward(&kv_c)?;

        // Split into k_nope and v
        let _total_dim =
            self.config.num_heads * (self.config.qk_nope_head_dim + self.config.v_head_dim);
        let kv_expanded = kv_expanded.reshape((
            num_tokens,
            self.config.num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        ))?;

        let k_nope = kv_expanded.narrow(2, 0, self.config.qk_nope_head_dim)?;
        let v = kv_expanded.narrow(2, self.config.qk_nope_head_dim, self.config.v_head_dim)?;

        Ok((k_nope, k_pe, v))
    }

    /// Get the configuration.
    pub fn config(&self) -> &MLACacheConfig {
        &self.config
    }

    /// Get the raw kv_c cache tensor.
    pub fn kv_c_cache(&self) -> &Tensor {
        &self.kv_c_cache
    }

    /// Get the raw k_pe cache tensor.
    pub fn k_pe_cache(&self) -> &Tensor {
        &self.k_pe_cache
    }

    /// Returns true if this cache uses quantization.
    pub fn is_quantized(&self) -> bool {
        self.config.is_quantized()
    }

    /// Get kv_c scale for quantized mode.
    pub fn kv_c_scale(&self) -> Option<&Tensor> {
        self.scales.as_ref().map(|s| &s.kv_c_scale)
    }

    /// Get k_pe scale for quantized mode.
    pub fn k_pe_scale(&self) -> Option<&Tensor> {
        self.scales.as_ref().map(|s| &s.k_pe_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_config() -> MLACacheConfig {
        MLACacheConfig::new(
            32, // kv_lora_rank (small for testing)
            8,  // qk_rope_head_dim
            16, // qk_nope_head_dim
            16, // v_head_dim
            4,  // num_heads
            4,  // block_size
            8,  // num_blocks
            1,  // num_layers
            DType::F32,
            Device::Cpu,
        )
    }

    #[test]
    fn test_mla_cache_creation() {
        let config = test_config();
        let engine = MLACacheEngine::new(&config).unwrap();

        assert_eq!(engine.kv_c_cache.dims(), &[8, 4, 32]);
        assert_eq!(engine.k_pe_cache.dims(), &[8, 4, 8]);
        assert!(!engine.is_quantized());
    }

    #[test]
    fn test_mla_cache_write_read() {
        let config = test_config();
        let mut engine = MLACacheEngine::new(&config).unwrap();

        // Write 3 tokens to block 1 (slots 4, 5, 6)
        let kv_c_data: Vec<f32> = (0..3 * 32).map(|i| i as f32).collect();
        let k_pe_data: Vec<f32> = (0..3 * 8).map(|i| (i as f32) * 0.1).collect();

        let kv_c = Tensor::from_vec(kv_c_data.clone(), (3, 32), &Device::Cpu).unwrap();
        let k_pe = Tensor::from_vec(k_pe_data.clone(), (3, 8), &Device::Cpu).unwrap();

        engine.write(&kv_c, &k_pe, &[4, 5, 6]).unwrap();

        // Read back
        let (kv_c_out, k_pe_out) = engine.read_raw(&[1], 3).unwrap();

        assert_eq!(kv_c_out.dims(), &[3, 32]);
        assert_eq!(k_pe_out.dims(), &[3, 8]);

        // Verify data matches
        let kv_c_read: Vec<f32> = kv_c_out.flatten_all().unwrap().to_vec1().unwrap();
        let k_pe_read: Vec<f32> = k_pe_out.flatten_all().unwrap().to_vec1().unwrap();

        for (orig, read) in kv_c_data.iter().zip(kv_c_read.iter()) {
            assert!(
                (orig - read).abs() < 1e-5,
                "kv_c mismatch: {} vs {}",
                orig,
                read
            );
        }
        for (orig, read) in k_pe_data.iter().zip(k_pe_read.iter()) {
            assert!(
                (orig - read).abs() < 1e-5,
                "k_pe mismatch: {} vs {}",
                orig,
                read
            );
        }
    }

    #[test]
    fn test_mla_cache_quantized() {
        let config = test_config().with_kv_cache_dtype(KVCacheDtype::Fp8E4m3);
        let mut engine = MLACacheEngine::new(&config).unwrap();

        assert!(engine.is_quantized());

        // Write data
        let kv_c_data: Vec<f32> = (0..3 * 32).map(|i| (i as f32) * 0.1).collect();
        let k_pe_data: Vec<f32> = (0..3 * 8).map(|i| (i as f32) * 0.1).collect();

        let kv_c = Tensor::from_vec(kv_c_data.clone(), (3, 32), &Device::Cpu).unwrap();
        let k_pe = Tensor::from_vec(k_pe_data.clone(), (3, 8), &Device::Cpu).unwrap();

        engine.write(&kv_c, &k_pe, &[4, 5, 6]).unwrap();

        // Read back - should be dequantized
        let (kv_c_out, _k_pe_out) = engine.read_raw(&[1], 3).unwrap();

        // FP8 has limited precision, allow some error
        let kv_c_read: Vec<f32> = kv_c_out.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, read) in kv_c_data.iter().zip(kv_c_read.iter()) {
            let error = (orig - read).abs();
            let rel_error = if orig.abs() > 1e-6 {
                error / orig.abs()
            } else {
                error
            };
            assert!(
                rel_error < 0.2 || error < 0.5,
                "FP8 error too large: orig={}, read={}",
                orig,
                read
            );
        }
    }

    #[test]
    fn test_mla_cache_multi_block() {
        let config = test_config();
        let mut engine = MLACacheEngine::new(&config).unwrap();

        // Write 6 tokens across blocks 0 and 2
        let kv_c = Tensor::randn(0f32, 1f32, (6, 32), &Device::Cpu).unwrap();
        let k_pe = Tensor::randn(0f32, 1f32, (6, 8), &Device::Cpu).unwrap();

        // Slots: 0,1,2,3 (block 0) + 8,9 (block 2)
        engine.write(&kv_c, &k_pe, &[0, 1, 2, 3, 8, 9]).unwrap();

        // Read back blocks [0, 2] with 6 tokens
        let (kv_c_out, k_pe_out) = engine.read_raw(&[0, 2], 6).unwrap();

        assert_eq!(kv_c_out.dims(), &[6, 32]);
        assert_eq!(k_pe_out.dims(), &[6, 8]);
    }
}
