use candle_core::Tensor;

use super::block_pool::BlockId;
use super::config::CacheConfig;
use super::error::CacheError;

/// Owns pre-allocated GPU tensors for one layer's KV cache.
/// Performs block-level read/write via scatter_set and index_select.
///
/// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
/// This layout allows direct reshape to [total_slots, kv_heads, head_dim] for scatter/gather.
pub struct CacheEngine {
    k_cache: Tensor,
    v_cache: Tensor,
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl CacheEngine {
    /// Pre-allocate cache tensors, filled with zeros.
    pub fn new(config: &CacheConfig) -> Result<Self, CacheError> {
        let shape = (
            config.num_blocks,
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        );
        let k_cache = Tensor::zeros(shape, config.dtype, &config.device)?;
        let v_cache = Tensor::zeros(shape, config.dtype, &config.device)?;
        Ok(Self {
            k_cache,
            v_cache,
            num_blocks: config.num_blocks,
            block_size: config.block_size,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        })
    }

    /// Raw K cache tensor: [num_blocks, block_size, num_kv_heads, head_dim].
    pub fn k_cache(&self) -> &Tensor {
        &self.k_cache
    }

    /// Raw V cache tensor: [num_blocks, block_size, num_kv_heads, head_dim].
    pub fn v_cache(&self) -> &Tensor {
        &self.v_cache
    }

    /// Write K, V for new tokens into their assigned slots.
    ///
    /// k, v shape: [num_kv_heads, new_tokens, head_dim]
    /// slot_mapping: physical slot IDs (length = new_tokens)
    pub fn write(&self, k: &Tensor, v: &Tensor, slot_mapping: &[usize]) -> Result<(), CacheError> {
        let new_tokens = slot_mapping.len();
        let total_slots = self.num_blocks * self.block_size;

        // Cache [num_blocks, block_size, kv_heads, head_dim] → [total_slots, kv_heads, head_dim]
        let k_cache_flat = self
            .k_cache
            .reshape((total_slots, self.num_kv_heads, self.head_dim))?;
        let v_cache_flat = self
            .v_cache
            .reshape((total_slots, self.num_kv_heads, self.head_dim))?;

        // Input [kv_heads, new_tokens, head_dim] → [new_tokens, kv_heads, head_dim]
        let k_src = k.transpose(0, 1)?.contiguous()?;
        let v_src = v.transpose(0, 1)?.contiguous()?;

        // Index: [new_tokens] → expand to [new_tokens, kv_heads, head_dim]
        let indices = Tensor::from_vec(
            slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            (new_tokens,),
            k_cache_flat.device(),
        )?;
        let indices = indices
            .reshape((new_tokens, 1, 1))?
            .expand((new_tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;

        k_cache_flat.scatter_set(&indices, &k_src, 0)?;
        v_cache_flat.scatter_set(&indices, &v_src, 0)?;

        Ok(())
    }

    /// Write K, V tokens in [new_tokens, kv_heads, head_dim] layout.
    /// Used by batched decode where tokens from different sequences are concatenated.
    pub fn write_batch(
        &self,
        k: &Tensor,
        v: &Tensor,
        slot_mapping: &[usize],
    ) -> Result<(), CacheError> {
        let new_tokens = slot_mapping.len();
        let total_slots = self.num_blocks * self.block_size;

        let k_cache_flat = self
            .k_cache
            .reshape((total_slots, self.num_kv_heads, self.head_dim))?;
        let v_cache_flat = self
            .v_cache
            .reshape((total_slots, self.num_kv_heads, self.head_dim))?;

        // Input already in [new_tokens, kv_heads, head_dim]
        let k_src = k.contiguous()?;
        let v_src = v.contiguous()?;

        let indices = Tensor::from_vec(
            slot_mapping.iter().map(|&s| s as u32).collect::<Vec<_>>(),
            (new_tokens,),
            k_cache_flat.device(),
        )?;
        let indices = indices
            .reshape((new_tokens, 1, 1))?
            .expand((new_tokens, self.num_kv_heads, self.head_dim))?
            .contiguous()?;

        k_cache_flat.scatter_set(&indices, &k_src, 0)?;
        v_cache_flat.scatter_set(&indices, &v_src, 0)?;

        Ok(())
    }

    /// Read K, V for multiple sequences concatenated contiguously.
    /// Returns (k, v) each with shape [total_tokens, num_kv_heads, head_dim].
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

            let k = self.k_cache.index_select(&indices, 0)?;
            let v = self.v_cache.index_select(&indices, 0)?;

            let total_capacity = num_blocks_used * self.block_size;
            let k = k.reshape((total_capacity, self.num_kv_heads, self.head_dim))?;
            let v = v.reshape((total_capacity, self.num_kv_heads, self.head_dim))?;

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

        // index_select on dim 0: [num_blocks_used, block_size, kv_heads, head_dim]
        let k = self.k_cache.index_select(&indices, 0)?;
        let v = self.v_cache.index_select(&indices, 0)?;

        // Reshape to merge blocks: [num_blocks_used * block_size, kv_heads, head_dim]
        let total_capacity = num_blocks_used * self.block_size;
        let k = k.reshape((total_capacity, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((total_capacity, self.num_kv_heads, self.head_dim))?;

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
        }
    }

    #[test]
    fn new_allocates_correct_shape() {
        let config = test_config(8);
        let engine = CacheEngine::new(&config).unwrap();
        // Layout: [num_blocks, block_size, kv_heads, head_dim]
        assert_eq!(engine.k_cache.dims(), &[8, 4, 2, 8]);
        assert_eq!(engine.v_cache.dims(), &[8, 4, 2, 8]);
    }

    #[test]
    fn write_read_roundtrip() {
        let config = test_config(8);
        let engine = CacheEngine::new(&config).unwrap();

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
    fn write_non_contiguous_slots() {
        let config = test_config(8);
        let engine = CacheEngine::new(&config).unwrap();

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
        let engine = CacheEngine::new(&config).unwrap();

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
}
