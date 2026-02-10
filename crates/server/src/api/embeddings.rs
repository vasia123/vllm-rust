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
    pub fn new(
        embeddings: Vec<Vec<f32>>,
        model: String,
        token_counts: Vec<usize>,
        encoding_format: EncodingFormat,
    ) -> Self {
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
                        let bytes: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
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
        return Err(ApiError::InvalidRequest(
            "Input cannot be empty".to_string(),
        ));
    }

    // Convert inputs to token IDs
    let inputs = request.input.into_texts();
    let mut all_token_ids = Vec::with_capacity(inputs.len());
    let mut token_counts = Vec::with_capacity(inputs.len());

    for input in inputs {
        let token_ids = match input {
            InputItem::Text(text) => state
                .tokenizer
                .encode(&text)
                .map_err(|e| ApiError::InvalidRequest(format!("Tokenization failed: {}", e)))?,
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

// ─── Pooling Types ──────────────────────────────────────────────────────────

/// Request for the pooling endpoint.
#[derive(Debug, Deserialize)]
pub struct PoolingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub truncate_prompt_tokens: Option<usize>,
}

/// Individual pooling result.
#[derive(Debug, Serialize)]
pub struct PoolingData {
    pub object: &'static str,
    pub data: Vec<f32>,
    pub index: usize,
}

/// Pooling response.
#[derive(Debug, Serialize)]
pub struct PoolingResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<PoolingData>,
    pub usage: EmbeddingUsage,
}

/// POST /pooling — Pool text inputs into fixed-size representations.
pub async fn pooling(
    State(state): State<AppState>,
    Json(request): Json<PoolingRequest>,
) -> Result<Json<PoolingResponse>, ApiError> {
    if request.input.is_empty() {
        return Err(ApiError::InvalidRequest(
            "Input cannot be empty".to_string(),
        ));
    }

    let inputs = request.input.into_texts();
    let mut total_tokens = 0;

    for input in &inputs {
        let count = match input {
            InputItem::Text(text) => state.tokenizer.encode(text).map(|t| t.len()).unwrap_or(0),
            InputItem::TokenIds(ids) => ids.len(),
        };
        total_tokens += count;
    }

    // Placeholder: return zero vectors
    let embedding_dim = 1536;
    let data: Vec<PoolingData> = (0..inputs.len())
        .map(|i| PoolingData {
            object: "embedding",
            data: vec![0.0f32; embedding_dim],
            index: i,
        })
        .collect();

    let response = PoolingResponse {
        id: format!("pool-{}", uuid::Uuid::new_v4()),
        object: "list",
        created: super::types::timestamp_now(),
        model: request.model,
        data,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response))
}

// ─── Classify Types ─────────────────────────────────────────────────────────

/// Request for the classify endpoint.
#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub user: Option<String>,
}

/// Individual classification result.
#[derive(Debug, Serialize)]
pub struct ClassifyData {
    pub index: usize,
    pub object: &'static str,
    pub label: String,
    pub score: f32,
}

/// Classification response.
#[derive(Debug, Serialize)]
pub struct ClassifyResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<ClassifyData>,
    pub usage: EmbeddingUsage,
}

/// POST /classify — Classify text inputs.
pub async fn classify(
    State(state): State<AppState>,
    Json(request): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, ApiError> {
    if request.input.is_empty() {
        return Err(ApiError::InvalidRequest(
            "Input cannot be empty".to_string(),
        ));
    }

    let inputs = request.input.into_texts();
    let mut total_tokens = 0;

    for input in &inputs {
        let count = match input {
            InputItem::Text(text) => state.tokenizer.encode(text).map(|t| t.len()).unwrap_or(0),
            InputItem::TokenIds(ids) => ids.len(),
        };
        total_tokens += count;
    }

    // Placeholder: return neutral classification
    let data: Vec<ClassifyData> = (0..inputs.len())
        .map(|i| ClassifyData {
            index: i,
            object: "classify",
            label: "neutral".to_string(),
            score: 0.0,
        })
        .collect();

    let response = ClassifyResponse {
        id: format!("cls-{}", uuid::Uuid::new_v4()),
        object: "list",
        created: super::types::timestamp_now(),
        model: request.model,
        data,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response))
}

// ─── Score Types ────────────────────────────────────────────────────────────

/// Input for scoring - can be a single string or list of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ScoreInput {
    Single(String),
    Multiple(Vec<String>),
}

impl ScoreInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            ScoreInput::Single(s) => vec![s],
            ScoreInput::Multiple(v) => v,
        }
    }
}

/// Score request — compute similarity between text_1 and text_2.
#[derive(Debug, Deserialize)]
pub struct ScoreRequest {
    pub model: String,
    /// First text(s) — typically the query.
    pub text_1: ScoreInput,
    /// Second text(s) — typically the documents.
    pub text_2: ScoreInput,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub truncate_prompt_tokens: Option<usize>,
}

/// Individual score result.
#[derive(Debug, Serialize)]
pub struct ScoreResponseData {
    pub index: usize,
    pub object: &'static str,
    pub score: f32,
}

/// Score response.
#[derive(Debug, Serialize)]
pub struct ScoreResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<ScoreResponseData>,
    pub usage: EmbeddingUsage,
}

/// POST /score — Compute similarity scores between text pairs.
pub async fn score(
    State(state): State<AppState>,
    Json(request): Json<ScoreRequest>,
) -> Result<Json<ScoreResponse>, ApiError> {
    let texts_1 = request.text_1.into_vec();
    let texts_2 = request.text_2.into_vec();

    if texts_1.is_empty() || texts_2.is_empty() {
        return Err(ApiError::InvalidRequest(
            "text_1 and text_2 cannot be empty".to_string(),
        ));
    }

    // Determine pairing: 1:N, N:1, or N:N
    let pairs: Vec<(usize, &str, &str)> = if texts_1.len() == 1 {
        // 1:N — score single query against all documents
        texts_2
            .iter()
            .enumerate()
            .map(|(i, d)| (i, texts_1[0].as_str(), d.as_str()))
            .collect()
    } else if texts_2.len() == 1 {
        // N:1 — score all queries against single document
        texts_1
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.as_str(), texts_2[0].as_str()))
            .collect()
    } else if texts_1.len() == texts_2.len() {
        // N:N — pairwise scoring
        texts_1
            .iter()
            .zip(texts_2.iter())
            .enumerate()
            .map(|(i, (q, d))| (i, q.as_str(), d.as_str()))
            .collect()
    } else {
        return Err(ApiError::InvalidRequest(
            "text_1 and text_2 must have the same length, or one must be a single string"
                .to_string(),
        ));
    };

    // Tokenize all texts to count tokens
    let mut total_tokens = 0;
    for &(_, q, d) in &pairs {
        total_tokens += state.tokenizer.encode(q).map(|t| t.len()).unwrap_or(0);
        total_tokens += state.tokenizer.encode(d).map(|t| t.len()).unwrap_or(0);
    }

    // Placeholder: return zero scores (real implementation uses embedding model)
    let data: Vec<ScoreResponseData> = pairs
        .iter()
        .map(|&(i, _, _)| ScoreResponseData {
            index: i,
            object: "score",
            score: 0.0,
        })
        .collect();

    let response = ScoreResponse {
        id: format!("score-{}", uuid::Uuid::new_v4()),
        object: "list",
        created: super::types::timestamp_now(),
        model: request.model,
        data,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response))
}

// ─── Rerank Types ───────────────────────────────────────────────────────────

/// Rerank request — re-order documents by relevance to a query.
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocumentInput>,
    /// Number of top results to return (0 = all).
    #[serde(default)]
    pub top_n: usize,
    #[serde(default)]
    pub user: Option<String>,
}

/// Input document for reranking — either plain text or structured.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RerankDocumentInput {
    Text(String),
    Structured { text: String },
}

impl RerankDocumentInput {
    fn as_str(&self) -> &str {
        match self {
            RerankDocumentInput::Text(s) => s,
            RerankDocumentInput::Structured { text } => text,
        }
    }
}

/// Single rerank result.
#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub document: RerankDocument,
    pub relevance_score: f32,
}

/// Document in a rerank result.
#[derive(Debug, Serialize)]
pub struct RerankDocument {
    pub text: String,
}

/// Usage for rerank response.
#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Rerank response.
#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub id: String,
    pub model: String,
    pub usage: RerankUsage,
    pub results: Vec<RerankResult>,
}

/// POST /rerank — Re-rank documents by relevance to a query.
pub async fn rerank(
    State(state): State<AppState>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, ApiError> {
    if request.query.is_empty() {
        return Err(ApiError::InvalidRequest(
            "query cannot be empty".to_string(),
        ));
    }
    if request.documents.is_empty() {
        return Err(ApiError::InvalidRequest(
            "documents cannot be empty".to_string(),
        ));
    }

    // Tokenize to count tokens
    let query_tokens = state
        .tokenizer
        .encode(&request.query)
        .map(|t| t.len())
        .unwrap_or(0);
    let mut total_tokens = query_tokens;

    for doc in &request.documents {
        total_tokens += state
            .tokenizer
            .encode(doc.as_str())
            .map(|t| t.len())
            .unwrap_or(0);
    }

    // Placeholder: return documents in original order with zero scores
    // Real implementation: encode query + docs, compute similarity, sort descending
    let mut results: Vec<RerankResult> = request
        .documents
        .iter()
        .enumerate()
        .map(|(i, doc)| RerankResult {
            index: i,
            document: RerankDocument {
                text: doc.as_str().to_string(),
            },
            relevance_score: 0.0,
        })
        .collect();

    // Sort by relevance_score descending (real implementation would have actual scores)
    results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

    // Apply top_n filter
    if request.top_n > 0 && request.top_n < results.len() {
        results.truncate(request.top_n);
    }

    let response = RerankResponse {
        id: format!("rerank-{}", uuid::Uuid::new_v4()),
        model: request.model,
        usage: RerankUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
        results,
    };

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

    // ─── Score types ────────────────────────────────────────────────────────

    #[test]
    fn score_request_text_pair() {
        let json = r#"{
            "model": "cross-encoder",
            "text_1": "query text",
            "text_2": ["doc 1", "doc 2"]
        }"#;
        let req: ScoreRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "cross-encoder");
        let t1 = req.text_1.into_vec();
        assert_eq!(t1.len(), 1);
        let t2 = req.text_2.into_vec();
        assert_eq!(t2.len(), 2);
    }

    #[test]
    fn score_response_serialization() {
        let response = ScoreResponse {
            id: "score-123".to_string(),
            object: "list",
            created: 0,
            model: "test".to_string(),
            data: vec![ScoreResponseData {
                index: 0,
                object: "score",
                score: 0.85,
            }],
            usage: EmbeddingUsage {
                prompt_tokens: 10,
                total_tokens: 10,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "list");
        assert!((json["data"][0]["score"].as_f64().unwrap() - 0.85).abs() < 0.001);
        assert_eq!(json["data"][0]["object"], "score");
    }

    // ─── Rerank types ───────────────────────────────────────────────────────

    #[test]
    fn rerank_request_deserialization() {
        let json = r#"{
            "model": "reranker",
            "query": "what is deep learning",
            "documents": ["Deep learning is...", "Machine learning is..."],
            "top_n": 1
        }"#;
        let req: RerankRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "reranker");
        assert_eq!(req.query, "what is deep learning");
        assert_eq!(req.documents.len(), 2);
        assert_eq!(req.top_n, 1);
    }

    #[test]
    fn rerank_request_structured_documents() {
        let json = r#"{
            "model": "reranker",
            "query": "query",
            "documents": [{"text": "doc 1"}, {"text": "doc 2"}]
        }"#;
        let req: RerankRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.documents[0].as_str(), "doc 1");
        assert_eq!(req.documents[1].as_str(), "doc 2");
    }

    #[test]
    fn rerank_response_serialization() {
        let response = RerankResponse {
            id: "rerank-123".to_string(),
            model: "test".to_string(),
            usage: RerankUsage {
                prompt_tokens: 20,
                total_tokens: 20,
            },
            results: vec![RerankResult {
                index: 0,
                document: RerankDocument {
                    text: "relevant doc".to_string(),
                },
                relevance_score: 0.95,
            }],
        };
        let json = serde_json::to_value(&response).unwrap();
        assert!((json["results"][0]["relevance_score"].as_f64().unwrap() - 0.95).abs() < 0.001);
        assert_eq!(json["results"][0]["document"]["text"], "relevant doc");
        assert_eq!(json["usage"]["prompt_tokens"], 20);
    }

    #[test]
    fn score_input_single_string() {
        let input: ScoreInput = serde_json::from_str(r#""hello""#).unwrap();
        let vec = input.into_vec();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec[0], "hello");
    }

    #[test]
    fn score_input_multiple_strings() {
        let input: ScoreInput = serde_json::from_str(r#"["a", "b", "c"]"#).unwrap();
        let vec = input.into_vec();
        assert_eq!(vec.len(), 3);
    }

    // ─── Pooling types ─────────────────────────────────────────────────────

    #[test]
    fn pooling_request_deserialization() {
        let json = r#"{
            "model": "pool-model",
            "input": "hello world"
        }"#;
        let req: PoolingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "pool-model");
        assert!(req.truncate_prompt_tokens.is_none());
    }

    #[test]
    fn pooling_response_serialization() {
        let response = PoolingResponse {
            id: "pool-123".to_string(),
            object: "list",
            created: 0,
            model: "test".to_string(),
            data: vec![PoolingData {
                object: "embedding",
                data: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            usage: EmbeddingUsage {
                prompt_tokens: 5,
                total_tokens: 5,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "embedding");
        assert_eq!(json["data"][0]["data"].as_array().unwrap().len(), 3);
        assert_eq!(json["usage"]["prompt_tokens"], 5);
    }

    // ─── Classify types ────────────────────────────────────────────────────

    #[test]
    fn classify_request_deserialization() {
        let json = r#"{
            "model": "cls-model",
            "input": ["positive text", "negative text"]
        }"#;
        let req: ClassifyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "cls-model");
        assert!(req.user.is_none());
    }

    #[test]
    fn classify_response_serialization() {
        let response = ClassifyResponse {
            id: "cls-123".to_string(),
            object: "list",
            created: 0,
            model: "test".to_string(),
            data: vec![ClassifyData {
                index: 0,
                object: "classify",
                label: "positive".to_string(),
                score: 0.92,
            }],
            usage: EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "classify");
        assert_eq!(json["data"][0]["label"], "positive");
        assert!((json["data"][0]["score"].as_f64().unwrap() - 0.92).abs() < 0.001);
        assert_eq!(json["usage"]["prompt_tokens"], 3);
    }
}
