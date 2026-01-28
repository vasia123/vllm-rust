//! FlashInfer metadata construction for paged KV cache.
//!
//! Converts vllm-rust's PagedAttentionMetadata to FlashInfer's paged KV format.

use candle_core::{Device, Result, Tensor};

use crate::kv_cache::BlockId;

/// Metadata for FlashInfer paged KV cache operations.
///
/// FlashInfer uses a ragged tensor format:
/// - `paged_kv_indptr`: Cumulative block counts per sequence [batch_size + 1]
/// - `paged_kv_indices`: Flattened block IDs [total_blocks]
/// - `paged_kv_last_page_len`: Valid tokens in last block per sequence [batch_size]
#[derive(Debug)]
pub struct FlashInferMetadata {
    /// Cumulative block indices: [batch_size + 1]
    /// indptr[i] = start index into paged_kv_indices for sequence i
    pub paged_kv_indptr: Tensor,
    /// Flattened block IDs: [total_blocks]
    pub paged_kv_indices: Tensor,
    /// Number of valid tokens in the last page for each sequence: [batch_size]
    pub paged_kv_last_page_len: Tensor,
}

impl FlashInferMetadata {
    /// Build metadata from paged attention block IDs and KV lengths.
    ///
    /// # Arguments
    /// * `block_ids` - Block IDs for each sequence
    /// * `kv_lengths` - KV cache length for each sequence
    /// * `block_size` - Number of tokens per block
    /// * `device` - Target device for tensors
    pub fn from_paged_attention(
        block_ids: &[&[BlockId]],
        kv_lengths: &[usize],
        block_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let batch_size = block_ids.len();
        assert_eq!(
            kv_lengths.len(),
            batch_size,
            "block_ids and kv_lengths must have same length"
        );

        // Build indptr: cumulative block counts
        let mut indptr = Vec::with_capacity(batch_size + 1);
        indptr.push(0i64);

        let mut total_blocks = 0usize;
        for seq_blocks in block_ids.iter() {
            total_blocks += seq_blocks.len();
            indptr.push(total_blocks as i64);
        }

        // Build indices: flattened block IDs
        let mut indices = Vec::with_capacity(total_blocks);
        for seq_blocks in block_ids.iter() {
            for &block_id in *seq_blocks {
                indices.push(block_id as u32);
            }
        }

        // Build last_page_len: valid tokens in last block
        let mut last_page_len = Vec::with_capacity(batch_size);
        for &kv_len in kv_lengths {
            let len_in_last = kv_len % block_size;
            // If len_in_last == 0 and kv_len > 0, the last block is full
            let last_len = if len_in_last == 0 && kv_len > 0 {
                block_size
            } else {
                len_in_last
            };
            last_page_len.push(last_len as u32);
        }

        let paged_kv_indptr = Tensor::from_vec(indptr, (batch_size + 1,), device)?;
        let paged_kv_indices = Tensor::from_vec(indices, (total_blocks,), device)?;
        let paged_kv_last_page_len = Tensor::from_vec(last_page_len, (batch_size,), device)?;

        Ok(Self {
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        })
    }

    /// Build metadata for a single sequence (prefill case).
    pub fn from_single_sequence(
        block_ids: &[BlockId],
        kv_length: usize,
        block_size: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::from_paged_attention(&[block_ids], &[kv_length], block_size, device)
    }

    /// Get total number of blocks across all sequences.
    pub fn total_blocks(&self) -> Result<usize> {
        self.paged_kv_indices.dim(0)
    }

    /// Get batch size.
    pub fn batch_size(&self) -> Result<usize> {
        self.paged_kv_last_page_len.dim(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_single_sequence() {
        let device = Device::Cpu;
        let block_ids = vec![0, 1, 2];
        let kv_length = 10; // 10 tokens across 3 blocks with block_size=4
        let block_size = 4;

        let metadata =
            FlashInferMetadata::from_single_sequence(&block_ids, kv_length, block_size, &device)
                .unwrap();

        // indptr: [0, 3]
        let indptr: Vec<i64> = metadata.paged_kv_indptr.to_vec1().unwrap();
        assert_eq!(indptr, vec![0, 3]);

        // indices: [0, 1, 2]
        let indices: Vec<u32> = metadata.paged_kv_indices.to_vec1().unwrap();
        assert_eq!(indices, vec![0, 1, 2]);

        // last_page_len: 10 % 4 = 2
        let last_page_len: Vec<u32> = metadata.paged_kv_last_page_len.to_vec1().unwrap();
        assert_eq!(last_page_len, vec![2]);
    }

    #[test]
    fn test_metadata_multiple_sequences() {
        let device = Device::Cpu;
        let block_ids: Vec<&[BlockId]> = vec![
            &[0, 1],    // seq 0: 2 blocks
            &[2, 3, 4], // seq 1: 3 blocks
            &[5],       // seq 2: 1 block
        ];
        let kv_lengths = vec![8, 10, 3]; // block_size=4
        let block_size = 4;

        let metadata =
            FlashInferMetadata::from_paged_attention(&block_ids, &kv_lengths, block_size, &device)
                .unwrap();

        // indptr: [0, 2, 5, 6]
        let indptr: Vec<i64> = metadata.paged_kv_indptr.to_vec1().unwrap();
        assert_eq!(indptr, vec![0, 2, 5, 6]);

        // indices: [0, 1, 2, 3, 4, 5]
        let indices: Vec<u32> = metadata.paged_kv_indices.to_vec1().unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);

        // last_page_len: [4, 2, 3] (8%4=0->4, 10%4=2, 3%4=3)
        let last_page_len: Vec<u32> = metadata.paged_kv_last_page_len.to_vec1().unwrap();
        assert_eq!(last_page_len, vec![4, 2, 3]);

        assert_eq!(metadata.total_blocks().unwrap(), 6);
        assert_eq!(metadata.batch_size().unwrap(), 3);
    }

    #[test]
    fn test_metadata_full_last_block() {
        let device = Device::Cpu;
        let block_ids = vec![0, 1];
        let kv_length = 8; // Exactly 2 full blocks
        let block_size = 4;

        let metadata =
            FlashInferMetadata::from_single_sequence(&block_ids, kv_length, block_size, &device)
                .unwrap();

        // last_page_len should be block_size (4), not 0
        let last_page_len: Vec<u32> = metadata.paged_kv_last_page_len.to_vec1().unwrap();
        assert_eq!(last_page_len, vec![4]);
    }

    #[test]
    fn test_metadata_empty_batch() {
        let device = Device::Cpu;
        let block_ids: Vec<&[BlockId]> = vec![];
        let kv_lengths: Vec<usize> = vec![];
        let block_size = 4;

        let metadata =
            FlashInferMetadata::from_paged_attention(&block_ids, &kv_lengths, block_size, &device)
                .unwrap();

        assert_eq!(metadata.total_blocks().unwrap(), 0);
        assert_eq!(metadata.batch_size().unwrap(), 0);
    }
}
