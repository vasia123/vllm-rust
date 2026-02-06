use candle_core::{DType, Device, Tensor};

use crate::engine::{DecodeSequenceMetadata, ModelForward};
use crate::kv_cache::{BlockTable, KVCacheManager};

/// A mock model for testing engine logic without real GPU computation.
/// Returns zero logits of the correct shape, enabling deterministic
/// and memory-efficient tests.
pub struct MockModelForward {
    vocab_size: usize,
    device: Device,
}

impl MockModelForward {
    pub fn new(vocab_size: usize, device: Device) -> Self {
        Self { vocab_size, device }
    }

    pub fn cpu(vocab_size: usize) -> Self {
        Self::new(vocab_size, Device::Cpu)
    }
}

impl ModelForward for MockModelForward {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> candle_core::Result<Tensor> {
        let batch_size = input_ids.dim(0)?;
        Tensor::zeros((batch_size, self.vocab_size), DType::F32, &self.device)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> candle_core::Result<Tensor> {
        let batch_size = input_ids.dim(0)?;
        Tensor::zeros((batch_size, self.vocab_size), DType::F32, &self.device)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model_forward_shape() {
        let model = MockModelForward::cpu(256);
        let input = Tensor::zeros((2, 5), DType::U32, &Device::Cpu).unwrap();

        let config = crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&config).unwrap();
        let block_table = BlockTable::new(16);
        let slot_mapping: Vec<usize> = vec![];

        let output = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(output.dims(), &[2, 256]);
    }

    #[test]
    fn test_mock_model_decode_batch_shape() {
        let model = MockModelForward::cpu(128);
        let input = Tensor::zeros((3, 1), DType::U32, &Device::Cpu).unwrap();

        let config = crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&config).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 0,
                seqlen_offset: 10,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 20,
                block_ids: vec![],
                slot_mapping: vec![],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 30,
                block_ids: vec![],
                slot_mapping: vec![],
            },
        ];

        let output = model
            .forward_decode_batch(&input, &sequences, &mut kv_cache_mgr)
            .unwrap();

        assert_eq!(output.dims(), &[3, 128]);
    }

    #[test]
    fn test_mock_model_device() {
        let model = MockModelForward::cpu(256);
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_mock_model_returns_zeros() {
        let model = MockModelForward::cpu(4);
        let input = Tensor::zeros((1, 1), DType::U32, &Device::Cpu).unwrap();

        let config = crate::kv_cache::config::CacheConfig {
            block_size: 16,
            num_blocks: 64,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: crate::kv_cache::KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&config).unwrap();
        let block_table = BlockTable::new(16);
        let slot_mapping: Vec<usize> = vec![];

        let output = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .unwrap();

        let values: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }
}
