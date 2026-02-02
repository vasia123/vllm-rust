//! Embedding model trait for text embedding generation.
//!
//! This module provides the `ModelForEmbedding` trait for models that generate
//! text embeddings (e.g., for RAG, semantic search, clustering).
//!
//! # Pooling Strategies
//!
//! Different embedding models use different pooling strategies:
//! - `Mean`: Average of all token embeddings (e.g., Sentence-BERT)
//! - `Cls`: Use the [CLS] token embedding (e.g., BERT)
//! - `LastToken`: Use the last token embedding (e.g., decoder-only models)
//!
//! # Example
//!
//! ```ignore
//! use vllm_core::engine::{ModelForEmbedding, PoolingStrategy};
//!
//! let embeddings = model.embed(&input_ids, Some(&attention_mask))?;
//! let pooled = model.pool(&embeddings, &attention_mask)?;
//! ```

use candle_core::{Device, Result, Tensor};

/// Pooling strategy for aggregating token embeddings into sentence embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    /// Average pooling over all tokens (weighted by attention mask).
    /// Common for Sentence-BERT, BGE, E5 models.
    #[default]
    Mean,
    /// Use the [CLS] token embedding (first token).
    /// Common for BERT-based models.
    Cls,
    /// Use the last non-padding token embedding.
    /// Common for decoder-only models like GPT, Llama.
    LastToken,
    /// Use the [EOS] token embedding.
    /// Some models use EOS for sentence representation.
    Eos,
}

impl PoolingStrategy {
    /// Parse pooling strategy from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "mean" | "average" => Some(Self::Mean),
            "cls" | "first" => Some(Self::Cls),
            "last" | "last_token" => Some(Self::LastToken),
            "eos" => Some(Self::Eos),
            _ => None,
        }
    }

    /// Get the string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Cls => "cls",
            Self::LastToken => "last_token",
            Self::Eos => "eos",
        }
    }
}

/// Trait for models that generate text embeddings.
///
/// This trait is separate from `ModelForward` because embedding models:
/// 1. Don't need KV cache (no autoregressive generation)
/// 2. Return embeddings instead of logits
/// 3. Often use bidirectional attention (encoder-only)
pub trait ModelForEmbedding: Send + 'static {
    /// Generate token embeddings for the input.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch_size, seq_len]
    /// * `attention_mask` - Optional attention mask [batch_size, seq_len]
    ///   Values: 1 for real tokens, 0 for padding
    ///
    /// # Returns
    /// Token embeddings [batch_size, seq_len, hidden_size]
    fn embed(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;

    /// Get the pooling strategy for this model.
    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::Mean
    }

    /// Pool token embeddings into sentence embeddings.
    ///
    /// # Arguments
    /// * `token_embeddings` - Token embeddings [batch_size, seq_len, hidden_size]
    /// * `attention_mask` - Attention mask [batch_size, seq_len]
    ///
    /// # Returns
    /// Sentence embeddings [batch_size, hidden_size]
    fn pool(
        &self,
        token_embeddings: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        pool_embeddings(token_embeddings, attention_mask, self.pooling_strategy())
    }

    /// Generate sentence embeddings (embed + pool).
    ///
    /// Convenience method that combines embedding and pooling.
    fn encode(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let token_embeddings = self.embed(input_ids, attention_mask)?;

        // Create default attention mask if not provided
        let mask = if let Some(mask) = attention_mask {
            mask.clone()
        } else {
            Tensor::ones(input_ids.dims(), input_ids.dtype(), input_ids.device())?
        };

        self.pool(&token_embeddings, &mask)
    }

    /// Get the embedding dimension.
    fn embedding_dim(&self) -> usize;

    /// Get the maximum sequence length supported.
    fn max_seq_len(&self) -> usize;

    /// Get the device.
    fn device(&self) -> &Device;

    /// Whether this model supports normalization.
    fn supports_normalize(&self) -> bool {
        true
    }

    /// Normalize embeddings to unit length (L2 normalization).
    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor> {
        // L2 normalize along last dimension
        let norm = embeddings.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
        embeddings.broadcast_div(&norm)
    }
}

/// Pool token embeddings into sentence embeddings.
///
/// Standalone function for pooling that can be used without the trait.
pub fn pool_embeddings(
    token_embeddings: &Tensor,
    attention_mask: &Tensor,
    strategy: PoolingStrategy,
) -> Result<Tensor> {
    match strategy {
        PoolingStrategy::Mean => mean_pooling(token_embeddings, attention_mask),
        PoolingStrategy::Cls => cls_pooling(token_embeddings),
        PoolingStrategy::LastToken => last_token_pooling(token_embeddings, attention_mask),
        PoolingStrategy::Eos => last_token_pooling(token_embeddings, attention_mask),
    }
}

/// Mean pooling over token embeddings.
fn mean_pooling(token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Expand attention mask to match embedding dimensions
    // [batch, seq_len] -> [batch, seq_len, 1]
    let mask_expanded = attention_mask.unsqueeze(2)?;
    let mask_expanded = mask_expanded.to_dtype(token_embeddings.dtype())?;

    // Broadcast mask to [batch, seq_len, hidden_size]
    let mask_expanded = mask_expanded.broadcast_as(token_embeddings.shape())?;

    // Sum of embeddings weighted by mask
    let sum_embeddings = (token_embeddings * &mask_expanded)?.sum(1)?;

    // Sum of mask (number of real tokens)
    let sum_mask = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;

    // Mean = sum / count
    sum_embeddings.broadcast_div(&sum_mask)
}

/// CLS token pooling (first token).
fn cls_pooling(token_embeddings: &Tensor) -> Result<Tensor> {
    // Take first token: [batch, seq_len, hidden] -> [batch, hidden]
    token_embeddings.narrow(1, 0, 1)?.squeeze(1)
}

/// Last token pooling (last non-padding token).
fn last_token_pooling(token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, _hidden_size) = token_embeddings.dims3()?;

    // Find last non-zero position in attention mask for each batch item
    let mask_vec: Vec<Vec<f32>> = attention_mask.to_dtype(candle_core::DType::F32)?.to_vec2()?;

    let mut result = Vec::with_capacity(batch_size);
    let embeddings_vec: Vec<Vec<Vec<f32>>> = token_embeddings
        .to_dtype(candle_core::DType::F32)?
        .to_vec3()?;

    for (batch_idx, mask_row) in mask_vec.iter().enumerate() {
        // Find last non-zero position
        let last_pos = mask_row
            .iter()
            .rposition(|&x| x > 0.5)
            .unwrap_or(seq_len - 1);

        result.push(embeddings_vec[batch_idx][last_pos].clone());
    }

    Tensor::new(result, token_embeddings.device())?.to_dtype(token_embeddings.dtype())
}

/// Embedding output with optional normalization.
#[derive(Debug, Clone)]
pub struct EmbeddingOutput {
    /// Raw embeddings [batch_size, hidden_size].
    pub embeddings: Tensor,
    /// Token count per input (for usage tracking).
    pub token_counts: Vec<usize>,
}

impl EmbeddingOutput {
    /// Create new embedding output.
    pub fn new(embeddings: Tensor, token_counts: Vec<usize>) -> Self {
        Self {
            embeddings,
            token_counts,
        }
    }

    /// Get the number of embeddings.
    pub fn len(&self) -> usize {
        self.token_counts.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.token_counts.is_empty()
    }

    /// Total tokens processed.
    pub fn total_tokens(&self) -> usize {
        self.token_counts.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_pooling_strategy_from_str() {
        assert_eq!(PoolingStrategy::from_str("mean"), Some(PoolingStrategy::Mean));
        assert_eq!(PoolingStrategy::from_str("MEAN"), Some(PoolingStrategy::Mean));
        assert_eq!(PoolingStrategy::from_str("cls"), Some(PoolingStrategy::Cls));
        assert_eq!(PoolingStrategy::from_str("last_token"), Some(PoolingStrategy::LastToken));
        assert_eq!(PoolingStrategy::from_str("unknown"), None);
    }

    #[test]
    fn test_pooling_strategy_as_str() {
        assert_eq!(PoolingStrategy::Mean.as_str(), "mean");
        assert_eq!(PoolingStrategy::Cls.as_str(), "cls");
        assert_eq!(PoolingStrategy::LastToken.as_str(), "last_token");
    }

    #[test]
    fn test_mean_pooling() {
        let device = Device::Cpu;

        // [batch=2, seq_len=3, hidden=4]
        let embeddings = Tensor::new(
            vec![
                vec![vec![1.0f32, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], vec![9.0, 10.0, 11.0, 12.0]],
                vec![vec![1.0f32, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0], vec![0.0, 0.0, 0.0, 0.0]],
            ],
            &device,
        )
        .unwrap();

        // Mask: first batch all tokens, second batch only first two
        let mask = Tensor::new(
            vec![vec![1.0f32, 1.0, 1.0], vec![1.0f32, 1.0, 0.0]],
            &device,
        )
        .unwrap();

        let pooled = mean_pooling(&embeddings, &mask).unwrap();
        assert_eq!(pooled.dims(), &[2, 4]);

        let pooled_vec: Vec<Vec<f32>> = pooled.to_vec2().unwrap();

        // First batch: mean of all three tokens
        // (1+5+9)/3=5, (2+6+10)/3=6, (3+7+11)/3=7, (4+8+12)/3=8
        assert!((pooled_vec[0][0] - 5.0).abs() < 1e-5);
        assert!((pooled_vec[0][1] - 6.0).abs() < 1e-5);

        // Second batch: mean of first two tokens only
        // (1+2)/2=1.5, (1+2)/2=1.5, (1+2)/2=1.5, (1+2)/2=1.5
        assert!((pooled_vec[1][0] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_cls_pooling() {
        let device = Device::Cpu;

        let embeddings = Tensor::new(
            vec![
                vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                vec![vec![7.0f32, 8.0, 9.0], vec![10.0, 11.0, 12.0]],
            ],
            &device,
        )
        .unwrap();

        let pooled = cls_pooling(&embeddings).unwrap();
        assert_eq!(pooled.dims(), &[2, 3]);

        let pooled_vec: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
        assert_eq!(pooled_vec[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(pooled_vec[1], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_last_token_pooling() {
        let device = Device::Cpu;

        let embeddings = Tensor::new(
            vec![
                vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
                vec![vec![7.0f32, 8.0], vec![9.0, 10.0], vec![0.0, 0.0]],
            ],
            &device,
        )
        .unwrap();

        let mask = Tensor::new(
            vec![vec![1.0f32, 1.0, 1.0], vec![1.0f32, 1.0, 0.0]],
            &device,
        )
        .unwrap();

        let pooled = last_token_pooling(&embeddings, &mask).unwrap();
        assert_eq!(pooled.dims(), &[2, 2]);

        let pooled_vec: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
        // First batch: last token (index 2)
        assert_eq!(pooled_vec[0], vec![5.0, 6.0]);
        // Second batch: last non-padding token (index 1)
        assert_eq!(pooled_vec[1], vec![9.0, 10.0]);
    }

    #[test]
    fn test_embedding_output() {
        let device = Device::Cpu;
        let embeddings = Tensor::zeros((3, 768), DType::F32, &device).unwrap();
        let token_counts = vec![10, 15, 8];

        let output = EmbeddingOutput::new(embeddings, token_counts);

        assert_eq!(output.len(), 3);
        assert!(!output.is_empty());
        assert_eq!(output.total_tokens(), 33);
    }
}
