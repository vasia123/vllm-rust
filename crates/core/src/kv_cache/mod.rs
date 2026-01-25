mod block_pool;
mod block_table;
mod cache_engine;
pub mod config;
mod error;
pub mod prefix_cache;

pub use block_pool::BlockId;
pub use block_table::BlockTable;
pub use cache_engine::CacheEngine;
pub use config::CacheConfig;
pub use error::CacheError;

use block_pool::BlockPool;

/// Top-level coordinator: BlockPool + per-layer CacheEngines.
pub struct KVCacheManager {
    block_pool: BlockPool,
    engines: Vec<CacheEngine>,
    block_size: usize,
}

impl KVCacheManager {
    pub fn new(config: &CacheConfig) -> Result<Self, CacheError> {
        let mut engines = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            engines.push(CacheEngine::new(config)?);
        }
        Ok(Self {
            block_pool: BlockPool::new(config.num_blocks),
            engines,
            block_size: config.block_size,
        })
    }

    /// Allocate blocks needed for `new_tokens` additional tokens.
    pub fn allocate_for_request(
        &mut self,
        block_table: &mut BlockTable,
        new_tokens: usize,
    ) -> Result<(), CacheError> {
        let needed = block_table.blocks_needed(new_tokens);
        if needed > 0 {
            let ids = self.block_pool.allocate(needed)?;
            block_table.append_blocks(&ids);
        }
        Ok(())
    }

    /// Free all blocks owned by a request.
    pub fn free_request(&mut self, block_table: &mut BlockTable) -> Result<(), CacheError> {
        let ids = block_table.release();
        if !ids.is_empty() {
            self.block_pool.free(&ids)?;
        }
        Ok(())
    }

    /// Get the CacheEngine for a specific layer.
    pub fn engine(&self, layer_idx: usize) -> &CacheEngine {
        &self.engines[layer_idx]
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free_blocks(&self) -> usize {
        self.block_pool.num_free()
    }

    /// Free specific blocks back to the pool (used during speculative decode rollback).
    pub fn free_blocks(&mut self, block_ids: &[BlockId]) -> Result<(), CacheError> {
        if !block_ids.is_empty() {
            self.block_pool.free(block_ids)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn test_config() -> CacheConfig {
        CacheConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    #[test]
    fn allocate_and_free_lifecycle() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        assert_eq!(mgr.num_free_blocks(), 16);

        let mut table = BlockTable::new(config.block_size);
        mgr.allocate_for_request(&mut table, 10).unwrap(); // needs 3 blocks (ceil(10/4))
        assert_eq!(mgr.num_free_blocks(), 13);

        mgr.free_request(&mut table).unwrap();
        assert_eq!(mgr.num_free_blocks(), 16);
    }

    #[test]
    fn write_read_through_manager() {
        let config = test_config();
        let mgr = KVCacheManager::new(&config).unwrap();

        let k_data: Vec<f32> = (0..2 * 3 * 8).map(|i| i as f32).collect();
        let k = Tensor::from_vec(k_data.clone(), (2, 3, 8), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(k_data, (2, 3, 8), &Device::Cpu).unwrap();

        // Write to layer 0, slots 0,1,2 (block 0)
        mgr.engine(0).write(&k, &v, &[0, 1, 2]).unwrap();

        // Read back
        let (k_out, _) = mgr.engine(0).read(&[0], 3).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 8]);

        // Layer 1 should be independent (all zeros still)
        let (k_layer1, _) = mgr.engine(1).read(&[0], 3).unwrap();
        let flat: Vec<f32> = k_layer1.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn free_blocks_returns_to_pool() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        mgr.allocate_for_request(&mut table, 10).unwrap(); // 3 blocks
        table.advance(10);
        assert_eq!(mgr.num_free_blocks(), 13);

        let freed = table.trim_to(5); // keep 2 blocks, free 1
        assert_eq!(freed.len(), 1);
        mgr.free_blocks(&freed).unwrap();
        assert_eq!(mgr.num_free_blocks(), 14);
    }

    #[test]
    fn incremental_allocation() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config).unwrap();
        let mut table = BlockTable::new(config.block_size);

        // Prefill: 5 tokens → need 2 blocks
        mgr.allocate_for_request(&mut table, 5).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(5);

        // Decode: 1 token → fits in existing block (5 < 8)
        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2); // no new block
        table.advance(1);

        // Decode more: fill up to 8, then 9th needs new block
        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(1);

        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 2);
        table.advance(1); // 8 tokens, last block full

        mgr.allocate_for_request(&mut table, 1).unwrap();
        assert_eq!(table.block_ids().len(), 3); // new block!
        table.advance(1);

        assert_eq!(table.num_tokens(), 9);
    }
}
