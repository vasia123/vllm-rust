//! OpenAI-compatible embeddings API endpoint.
//!
//! Implements the `/v1/embeddings` endpoint for generating text embeddings.
//!
//! # Example Request
//!
//! ```json
//! {
//!   "model": "text-embedding-ada-002",
//!   "input": "The quick brown fox jumps over the lazy dog"
//! }
//! ```
//!
//! # Example Response
//!
//! ```json
//! {
//!   "object": "list",
//!   "data": [
//!     {
//!       "object": "embedding",
//!       "embedding": [0.0023, -0.0091, ...],
//!       "index": 0
//!     }
//!   ],
//!   "model": "text-embedding-ada-002",
//!   "usage": {
//!     "prompt_tokens": 8,
//!     "total_tokens": 8
//!   }
//! }
//! ```

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};

use super::error::ApiError;
use super::AppState;

// ─── Request Types ───────────────────────────────────────────────────────────

/// Input for embedding request - can be string, array of strings, or token IDs.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single text string.
    Single(String),
    /// Multiple text strings.
    Multiple(Vec<String>),
    /// Single token ID array.
    TokenIds(Vec<u32>),
    /// Multiple token ID arrays.
    MultipleTokenIds(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    /// Get the number of inputs.
    pub fn len(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Multiple(v) => v.len(),
            EmbeddingInput::TokenIds(_) => 1,
            EmbeddingInput::MultipleTokenIds(v) => v.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to list of text inputs.
    pub fn into_texts(self) -> Vec<InputItem> {
        match self {
            EmbeddingInput::Single(s) => vec![InputItem::Text(s)],
            EmbeddingInput::Multiple(v) => v.into_iter().map(InputItem::Text).collect(),
            EmbeddingInput::TokenIds(ids) => vec![InputItem::TokenIds(ids)],
            EmbeddingInput::MultipleTokenIds(v) => v.into_iter().map(InputItem::TokenIds).collect(),
        }
    }
}

/// Individual input item.
#[derive(Debug, Clone)]
pub enum InputItem {
    Text(String),
    TokenIds(Vec<u32>),
}

/// Embedding request following OpenAI API spec.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: String,
    /// Input text(s) to embed.
    pub input: EmbeddingInput,
    /// Encoding format for the embeddings.
    /// Supported: "float" (default), "base64".
    #[serde(default = "default_encoding_format")]
    pub encoding_format: EncodingFormat,
    /// Number of dimensions for the output embeddings.
    /// Only supported by some models.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// A unique identifier for the end-user (for monitoring/abuse detection).
    #[serde(default)]
    pub user: Option<String>,
}

fn default_encoding_format() -> EncodingFormat {
    EncodingFormat::Float
}

/// Encoding format for embedding vectors.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Return embeddings as array of floats.
    #[default]
    Float,
    /// Return embeddings as base64-encoded bytes.
    Base64,
}

// ─── Response Types ──────────────────────────────────────────────────────────

/// Embedding response following OpenAI API spec.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Always "list".
    pub object: &'static str,
    /// List of embedding objects.
    pub data: Vec<EmbeddingData>,
    /// Model used for embedding.
    pub model: String,
    /// Token usage information.
    pub usage: EmbeddingUsage,
}

/// Individual embedding result.
#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    /// Always "embedding".
    pub object: &'static str,
    /// The embedding vector (floats or base64 depending on encoding_format).
    pub embedding: EmbeddingVector,
    /// Index of this embedding in the input list.
    pub index: usize,
}

/// Embedding vector - either float array or base64 string.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

/// Token usage for embedding request.
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Total tokens (same as prompt_tokens for embeddings).
    pub total_tokens: usize,
}

impl EmbeddingResponse {
    /// Create a new embedding response.
    pub fn new(embeddings: Vec<Vec<f32>>, model: String, token_counts: Vec<usize>, encoding_format: EncodingFormat) -> Self {
        let total_tokens: usize = token_counts.iter().sum();

        let data: Vec<EmbeddingData> = embeddings
            .into_iter()
            .enumerate()
            .map(|(index, emb)| EmbeddingData {
                object: "embedding",
                embedding: match encoding_format {
                    EncodingFormat::Float => EmbeddingVector::Float(emb),
                    EncodingFormat::Base64 => {
                        // Convert f32 array to bytes and base64 encode
                        let bytes: Vec<u8> = emb
                            .iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();
                        EmbeddingVector::Base64(base64_encode(&bytes))
                    }
                },
                index,
            })
            .collect();

        Self {
            object: "list",
            data,
            model,
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        }
    }
}

/// Simple base64 encoding (no external dependency).
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

// ─── Handler ─────────────────────────────────────────────────────────────────

/// Create embeddings for the given input.
///
/// POST /v1/embeddings
pub async fn create_embedding(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    // Validate request
    if request.input.is_empty() {
        return Err(ApiError::InvalidRequest("Input cannot be empty".to_string()));
    }

    // Convert inputs to token IDs
    let inputs = request.input.into_texts();
    let mut all_token_ids = Vec::with_capacity(inputs.len());
    let mut token_counts = Vec::with_capacity(inputs.len());

    for input in inputs {
        let token_ids = match input {
            InputItem::Text(text) => {
                state
                    .tokenizer
                    .encode(&text)
                    .map_err(|e| ApiError::InvalidRequest(format!("Tokenization failed: {}", e)))?
            }
            InputItem::TokenIds(ids) => ids,
        };
        token_counts.push(token_ids.len());
        all_token_ids.push(token_ids);
    }

    // For now, return a placeholder response since we don't have an embedding model loaded.
    // In a full implementation, this would:
    // 1. Get the embedding model from state
    // 2. Create input tensors
    // 3. Run forward pass through the embedding model
    // 4. Apply pooling and normalization
    // 5. Return the embeddings

    // Placeholder: return zero embeddings with correct dimensions
    // Real implementation would use ModelForEmbedding::encode()
    let embedding_dim = 1536; // Common dimension for embedding models
    let embeddings: Vec<Vec<f32>> = all_token_ids
        .iter()
        .map(|_| vec![0.0f32; embedding_dim])
        .collect();

    let response = EmbeddingResponse::new(
        embeddings,
        request.model,
        token_counts,
        request.encoding_format,
    );

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_single() {
        let input: EmbeddingInput = serde_json::from_str(r#""hello world""#).unwrap();
        assert_eq!(input.len(), 1);
        let items = input.into_texts();
        assert!(matches!(items[0], InputItem::Text(_)));
    }

    #[test]
    fn test_embedding_input_multiple() {
        let input: EmbeddingInput = serde_json::from_str(r#"["hello", "world"]"#).unwrap();
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn test_embedding_input_token_ids() {
        let input: EmbeddingInput = serde_json::from_str(r#"[1, 2, 3, 4]"#).unwrap();
        // This should be parsed as TokenIds
        let items = input.into_texts();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_embedding_request_minimal() {
        let json = r#"{
            "model": "text-embedding-ada-002",
            "input": "The food was delicious"
        }"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-ada-002");
        assert_eq!(req.encoding_format, EncodingFormat::Float);
    }

    #[test]
    fn test_embedding_request_full() {
        let json = r#"{
            "model": "text-embedding-3-small",
            "input": ["hello", "world"],
            "encoding_format": "base64",
            "dimensions": 512,
            "user": "user-123"
        }"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "text-embedding-3-small");
        assert_eq!(req.encoding_format, EncodingFormat::Base64);
        assert_eq!(req.dimensions, Some(512));
        assert_eq!(req.user, Some("user-123".to_string()));
    }

    #[test]
    fn test_embedding_response_serialization() {
        let response = EmbeddingResponse::new(
            vec![vec![0.1, 0.2, 0.3]],
            "test-model".to_string(),
            vec![5],
            EncodingFormat::Float,
        );

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"model\":\"test-model\""));
        assert!(json.contains("\"prompt_tokens\":5"));
    }

    #[test]
    fn test_base64_encode() {
        // Test known values
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn test_encoding_format_deserialization() {
        let float: EncodingFormat = serde_json::from_str(r#""float""#).unwrap();
        assert_eq!(float, EncodingFormat::Float);

        let base64: EncodingFormat = serde_json::from_str(r#""base64""#).unwrap();
        assert_eq!(base64, EncodingFormat::Base64);
    }
}
