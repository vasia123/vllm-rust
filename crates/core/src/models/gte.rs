//! GTE / Nomic embedding models with rotary position embeddings.
//!
//! Architecture: BertWithRope — BERT encoder with RoPE instead of absolute position embeddings.
//!
//! Key differences from standard BERT:
//! - RoPE (rotary position embeddings) instead of absolute
//! - Supports gated MLP (GEGLU) activation
//! - Optional token type embeddings (type_vocab_size may be 0)
//! - Pre-norm (LayerNorm before attention/MLP, residual after)
//!
//! Covers: GteNewModel, NomicBertModel, GteNewForSequenceClassification,
//! JinaRobertaModel (XLMRobertaModel)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::engine::{ModelForEmbedding, PoolingStrategy};
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::RotaryEmbedding;

// ─── Config Helpers ──────────────────────────────────────────────────────────

/// Extract GTE-specific config from extra fields.
struct GteConfig {
    hidden_act: String,
    layer_norm_eps: f64,
    type_vocab_size: usize,
    rotary_emb_dim: usize,
    rope_theta: f64,
    bias: bool,
}

impl GteConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let hidden_act = cfg
            .extra
            .get("hidden_act")
            .and_then(|v| v.as_str())
            .unwrap_or("geglu")
            .to_string();

        let layer_norm_eps = cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-12);

        let type_vocab_size = cfg
            .extra
            .get("type_vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        // rotary_emb_dim: from config, or from rotary_kwargs, or default to head_dim
        let rotary_emb_dim = cfg
            .extra
            .get("rotary_emb_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                cfg.extra
                    .get("rotary_kwargs")
                    .and_then(|rk| rk.get("dim"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
            .unwrap_or(head_dim);

        let rope_theta = cfg
            .extra
            .get("rotary_kwargs")
            .and_then(|rk| rk.get("base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rope_theta);

        let bias = cfg
            .extra
            .get("bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Self {
            hidden_act,
            layer_norm_eps,
            type_vocab_size,
            rotary_emb_dim,
            rope_theta,
            bias,
        }
    }

    fn is_gated(&self) -> bool {
        matches!(self.hidden_act.as_str(), "silu" | "geglu")
    }
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

struct GteEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
}

impl GteEmbeddings {
    fn new(cfg: &ModelConfig, gte_cfg: &GteConfig, vb: VarBuilder) -> Result<Self> {
        let word_embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;

        let token_type_embeddings = if gte_cfg.type_vocab_size > 0 {
            Some(embedding(
                gte_cfg.type_vocab_size,
                cfg.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };

        let layer_norm = layer_norm(cfg.hidden_size, gte_cfg.layer_norm_eps, vb.pp("LayerNorm"))?;

        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut embeddings = self.word_embeddings.forward(input_ids)?;

        if let Some(ref tte) = self.token_type_embeddings {
            let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, input_ids.device())?;
            let type_emb = tte.forward(&token_type_ids)?;
            embeddings = (embeddings + type_emb)?;
        }

        self.layer_norm.forward(&embeddings)
    }
}

// ─── Self-Attention with RoPE ────────────────────────────────────────────────

struct GteSelfAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GteSelfAttention {
    fn new(cfg: &ModelConfig, gte_cfg: &GteConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        let qkv_proj = linear(hidden_size, 3 * hidden_size, vb.pp("qkv_proj"))?;
        let out_proj = if gte_cfg.bias {
            linear(hidden_size, hidden_size, vb.pp("out_proj"))
        } else {
            candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("out_proj"))
        }?;

        let partial_factor = gte_cfg.rotary_emb_dim as f64 / head_dim as f64;
        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            gte_cfg.rope_theta,
            partial_factor,
            true, // neox style
            DType::F32,
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            out_proj,
            rotary_emb,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        let qkv = self.qkv_proj.forward(hidden_states)?;
        let q_size = self.num_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, q_size)?;
        let v = qkv.narrow(2, 2 * q_size, q_size)?;

        // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, 0)?;

        // Scaled dot-product attention (bidirectional: no causal mask)
        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

// ─── MLP (gated or ungated) ──────────────────────────────────────────────────

enum GteMlp {
    /// Gated MLP (GEGLU/SiLU-gated): gate_up_proj → act_and_mul → down_proj
    Gated {
        gate_up_proj: Linear,
        down_proj: Linear,
        use_silu: bool,
    },
    /// Standard MLP: up_proj → act → down_proj
    Standard { up_proj: Linear, down_proj: Linear },
}

impl GteMlp {
    fn new_gated(
        cfg: &ModelConfig,
        gte_cfg: &GteConfig,
        no_gate_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_up_proj = if no_gate_bias {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                vb.pp("gate_up_proj"),
            )
        } else if gte_cfg.bias {
            linear(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                vb.pp("gate_up_proj"),
            )
        } else {
            candle_nn::linear_no_bias(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                vb.pp("gate_up_proj"),
            )
        }?;

        let down_proj = if gte_cfg.bias {
            linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
        } else {
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
        }?;

        let use_silu = gte_cfg.hidden_act == "silu";

        Ok(Self::Gated {
            gate_up_proj,
            down_proj,
            use_silu,
        })
    }

    fn new_standard(cfg: &ModelConfig, gte_cfg: &GteConfig, vb: VarBuilder) -> Result<Self> {
        let up_proj = if gte_cfg.bias {
            linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
        }?;

        let down_proj = if gte_cfg.bias {
            linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
        } else {
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
        }?;

        Ok(Self::Standard { up_proj, down_proj })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gated {
                gate_up_proj,
                down_proj,
                use_silu,
            } => {
                let gate_up = gate_up_proj.forward(hidden_states)?;
                let chunks = gate_up.chunk(2, candle_core::D::Minus1)?;
                let gate = &chunks[0];
                let up = &chunks[1];

                let activated = if *use_silu {
                    (candle_nn::ops::silu(gate)? * up)?
                } else {
                    // GEGLU: gelu(gate) * up
                    (gate.gelu_erf()? * up)?
                };

                down_proj.forward(&activated)
            }
            Self::Standard { up_proj, down_proj } => {
                let hidden = up_proj.forward(hidden_states)?.gelu_erf()?;
                down_proj.forward(&hidden)
            }
        }
    }
}

// ─── Encoder Block ───────────────────────────────────────────────────────────

struct GteBlock {
    attn: GteSelfAttention,
    mlp: GteMlp,
    attn_ln: LayerNorm,
    mlp_ln: LayerNorm,
}

impl GteBlock {
    fn new(
        cfg: &ModelConfig,
        gte_cfg: &GteConfig,
        no_gate_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = GteSelfAttention::new(cfg, gte_cfg, vb.pp("attention"))?;

        let mlp = if gte_cfg.is_gated() {
            GteMlp::new_gated(cfg, gte_cfg, no_gate_bias, vb.pp("mlp"))?
        } else {
            GteMlp::new_standard(cfg, gte_cfg, vb.pp("mlp"))?
        };

        let attn_ln = layer_norm(cfg.hidden_size, gte_cfg.layer_norm_eps, vb.pp("attn_ln"))?;
        let mlp_ln = layer_norm(cfg.hidden_size, gte_cfg.layer_norm_eps, vb.pp("mlp_ln"))?;

        Ok(Self {
            attn,
            mlp,
            attn_ln,
            mlp_ln,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm residual pattern: attn_ln(x + attn(x)), mlp_ln(x + mlp(x))
        let attn_output = self.attn.forward(hidden_states)?;
        let hidden_states = self.attn_ln.forward(&(hidden_states + &attn_output)?)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        self.mlp_ln.forward(&(&hidden_states + &mlp_output)?)
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

struct GteEncoder {
    layers: Vec<GteBlock>,
}

impl GteEncoder {
    fn new(
        cfg: &ModelConfig,
        gte_cfg: &GteConfig,
        no_gate_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(GteBlock::new(
                cfg,
                gte_cfg,
                no_gate_bias,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, mut hidden_states: Tensor) -> Result<Tensor> {
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

// ─── Pooler ──────────────────────────────────────────────────────────────────

struct GtePooler {
    dense: Linear,
}

impl GtePooler {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let cls_token = hidden_states.narrow(1, 0, 1)?.squeeze(1)?;
        self.dense.forward(&cls_token)?.tanh()
    }
}

// ─── GTE New Model ───────────────────────────────────────────────────────────

/// GTE-New embedding model (Alibaba-NLP) with RoPE and GEGLU.
///
/// Architecture: BertWithRope variant used for GteNewModel.
/// Also covers NomicBertModel and JinaRobertaModel (same base, different config).
pub struct GteNewForEmbedding {
    embeddings: GteEmbeddings,
    encoder: GteEncoder,
    pooler: Option<GtePooler>,
    pooling: PoolingStrategy,
    hidden_size: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl GteNewForEmbedding {
    /// Create a GTE-New model.
    ///
    /// Weight prefix: expects `encoder.layers.{i}.{attention,mlp,attn_ln,mlp_ln}.*`
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gte_cfg = GteConfig::from_model_config(cfg);

        let embeddings = GteEmbeddings::new(cfg, &gte_cfg, vb.pp("embeddings"))?;

        // GteNewModel: gate_up_proj has no bias
        let no_gate_bias = true;
        let encoder = GteEncoder::new(cfg, &gte_cfg, no_gate_bias, vb.pp("encoder"))?;

        let pooler = GtePooler::new(cfg.hidden_size, vb.pp("pooler")).ok();

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pooling: PoolingStrategy::Cls,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Create a NomicBert model (same arch, bias on gate_up_proj).
    pub fn new_nomic(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gte_cfg = GteConfig::from_model_config(cfg);

        let embeddings = GteEmbeddings::new(cfg, &gte_cfg, vb.pp("embeddings"))?;

        // NomicBert: gate_up_proj has bias
        let no_gate_bias = false;
        let encoder = GteEncoder::new(cfg, &gte_cfg, no_gate_bias, vb.pp("encoder"))?;

        let pooler = GtePooler::new(cfg.hidden_size, vb.pp("pooler")).ok();

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pooling: PoolingStrategy::Cls,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    /// Create a JinaRoberta model (standard GELU, mean pooling, bias on all layers).
    ///
    /// Jina-v3 uses LoRA task adapters that should be merged at weight load time.
    /// This constructor expects already-merged weights.
    pub fn new_jina(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let gte_cfg = GteConfig::from_model_config(cfg);

        let embeddings = GteEmbeddings::new(cfg, &gte_cfg, vb.pp("embeddings"))?;

        // Jina: standard GELU (not gated), bias on all layers
        let no_gate_bias = false;
        let encoder = GteEncoder::new(cfg, &gte_cfg, no_gate_bias, vb.pp("encoder"))?;

        // Jina doesn't use a pooler layer
        let pooler = GtePooler::new(cfg.hidden_size, vb.pp("pooler")).ok();

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            pooling: PoolingStrategy::Mean,
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            device: vb.device().clone(),
        })
    }

    fn encode_hidden(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(input_ids)?;
        self.encoder.forward(embeddings)
    }
}

impl crate::engine::ModelForward for GteNewForEmbedding {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for GteNewForEmbedding {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.encode_hidden(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        self.pooling
    }

    fn pool(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        if self.pooling == PoolingStrategy::Cls {
            if let Some(ref pooler) = self.pooler {
                return pooler.forward(token_embeddings);
            }
        }
        crate::engine::pool_embeddings(token_embeddings, attention_mask, self.pooling)
    }

    fn embedding_dim(&self) -> usize {
        self.hidden_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── GTE-New for Sequence Classification (Cross-Encoder / Reranker) ──────────

/// GTE-New cross-encoder for reranking / sequence classification.
///
/// Wraps `GteNewForEmbedding` with a classification head.
pub struct GteNewForSequenceClassification {
    model: GteNewForEmbedding,
    classifier: Linear,
    num_labels: usize,
    device: Device,
}

impl GteNewForSequenceClassification {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_labels = cfg
            .extra
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        // Build the underlying GTE model with pooler
        let model = GteNewForEmbedding::new(cfg, vb.pp("new"))?;
        let classifier = linear(cfg.hidden_size, num_labels, vb.pp("classifier"))?;

        Ok(Self {
            model,
            classifier,
            num_labels,
            device: vb.device().clone(),
        })
    }

    /// Run classification: returns [batch_size, num_labels] scores.
    pub fn classify(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.model.encode_hidden(input_ids)?;

        // Pool using CLS token
        let pooled = if let Some(ref pooler) = self.model.pooler {
            pooler.forward(&hidden_states)?
        } else {
            hidden_states.narrow(1, 0, 1)?.squeeze(1)?
        };

        self.classifier.forward(&pooled)
    }
}

impl crate::engine::ModelForward for GteNewForSequenceClassification {
    fn forward(
        &self,
        input_ids: &Tensor,
        _seqlen_offset: usize,
        _kv_cache_mgr: &mut KVCacheManager,
        _block_table: &BlockTable,
        _slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.classify(input_ids)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        _sequences: &[DecodeSequenceMetadata],
        _kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.classify(input_ids)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl ModelForEmbedding for GteNewForSequenceClassification {
    fn embed(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.model.encode_hidden(input_ids)
    }

    fn pooling_strategy(&self) -> PoolingStrategy {
        PoolingStrategy::Cls
    }

    fn pool(&self, token_embeddings: &Tensor, _attention_mask: &Tensor) -> Result<Tensor> {
        // For classification: pool → classify
        let pooled = if let Some(ref pooler) = self.model.pooler {
            pooler.forward(token_embeddings)?
        } else {
            token_embeddings.narrow(1, 0, 1)?.squeeze(1)?
        };
        self.classifier.forward(&pooled)
    }

    fn embedding_dim(&self) -> usize {
        self.num_labels
    }

    fn max_seq_len(&self) -> usize {
        self.model.max_position_embeddings
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_gte_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::json!(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::json!(0));
        extra.insert("hidden_act".to_string(), serde_json::json!("geglu"));
        extra.insert("rotary_emb_dim".to_string(), serde_json::json!(32));
        extra.insert("bias".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["GteNewModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 32,
            hidden_act: "geglu".to_string(),
            rms_norm_eps: 1e-12,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    fn tiny_nomic_config() -> ModelConfig {
        let mut cfg = tiny_gte_config();
        cfg.architectures = vec!["NomicBertModel".to_string()];
        cfg.extra
            .insert("type_vocab_size".to_string(), serde_json::json!(2));
        cfg.extra
            .insert("hidden_act".to_string(), serde_json::json!("geglu"));
        cfg
    }

    fn tiny_classifier_config() -> ModelConfig {
        let mut cfg = tiny_gte_config();
        cfg.architectures = vec!["GteNewForSequenceClassification".to_string()];
        cfg.extra
            .insert("num_labels".to_string(), serde_json::json!(1));
        cfg
    }

    #[test]
    fn test_gte_construction() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = GteNewForEmbedding::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GteNewForEmbedding should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 512);
        assert_eq!(model.encoder.layers.len(), 2);
    }

    #[test]
    fn test_gte_forward_shape() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let batch_size = 2;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();

        let hidden_states = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(
            hidden_states.dims(),
            &[batch_size, seq_len, cfg.hidden_size]
        );
    }

    #[test]
    fn test_gte_embed_shape() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let embeddings = model.embed(&input_ids, None).unwrap();
        assert_eq!(embeddings.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_gte_encode_produces_sentence_embeddings() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 6), DType::U32, &device).unwrap();
        let mask = Tensor::ones((2, 6), DType::F32, &device).unwrap();

        let sentence_embeddings = model.encode(&input_ids, Some(&mask)).unwrap();
        assert_eq!(sentence_embeddings.dims(), &[2, 64]);
    }

    #[test]
    fn test_gte_pooling_strategy() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Cls
        );
    }

    #[test]
    fn test_gte_embedding_dim_and_max_seq_len() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        assert_eq!(ModelForEmbedding::embedding_dim(&model), 64);
        assert_eq!(ModelForEmbedding::max_seq_len(&model), 512);
    }

    #[test]
    fn test_gte_normalize() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let embeddings = Tensor::new(vec![vec![3.0f32, 4.0]], &device).unwrap();
        let normalized = model.normalize(&embeddings).unwrap();
        let vals: Vec<Vec<f32>> = normalized.to_vec2().unwrap();
        assert!((vals[0][0] - 0.6).abs() < 1e-5);
        assert!((vals[0][1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_nomic_construction() {
        let cfg = tiny_nomic_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = GteNewForEmbedding::new_nomic(&cfg, vb);
        assert!(
            model.is_ok(),
            "NomicBert should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_nomic_with_token_type_embeddings() {
        let cfg = tiny_nomic_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_nomic(&cfg, vb).unwrap();

        assert!(model.embeddings.token_type_embeddings.is_some());
    }

    #[test]
    fn test_nomic_forward_shape() {
        let cfg = tiny_nomic_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_nomic(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 5), DType::U32, &device).unwrap();
        let hidden = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(hidden.dims(), &[2, 5, 64]);
    }

    #[test]
    fn test_classifier_construction() {
        let cfg = tiny_classifier_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = GteNewForSequenceClassification::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GteNewForSequenceClassification should construct: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_classifier_forward_shape() {
        let cfg = tiny_classifier_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForSequenceClassification::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((3, 8), DType::U32, &device).unwrap();
        let scores = model.classify(&input_ids).unwrap();
        assert_eq!(scores.dims(), &[3, 1]); // 3 batch items, 1 label
    }

    #[test]
    fn test_gte_model_forward_trait() {
        use crate::engine::ModelForward;
        use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 16,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
        let output = model
            .forward(&input_ids, 0, &mut kv_cache_mgr, &block_table, &[])
            .unwrap();
        assert_eq!(output.dims(), &[1, 4, cfg.hidden_size]);
    }

    #[test]
    fn test_gte_no_token_type_embeddings() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        // type_vocab_size = 0, so no token type embeddings
        assert!(model.embeddings.token_type_embeddings.is_none());
    }

    #[test]
    fn test_gte_single_token_input() {
        let cfg = tiny_gte_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).unwrap();
        let hidden = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(hidden.dims(), &[1, 1, 64]);
    }

    #[test]
    fn test_gte_config_parsing() {
        let cfg = tiny_gte_config();
        let gte_cfg = GteConfig::from_model_config(&cfg);

        assert_eq!(gte_cfg.hidden_act, "geglu");
        assert!((gte_cfg.layer_norm_eps - 1e-12).abs() < 1e-15);
        assert_eq!(gte_cfg.type_vocab_size, 0);
        assert_eq!(gte_cfg.rotary_emb_dim, 32);
        assert!(gte_cfg.is_gated());
    }

    #[test]
    fn test_gte_config_silu_is_gated() {
        let mut cfg = tiny_gte_config();
        cfg.extra
            .insert("hidden_act".to_string(), serde_json::json!("silu"));
        let gte_cfg = GteConfig::from_model_config(&cfg);
        assert!(gte_cfg.is_gated());
    }

    #[test]
    fn test_gte_config_gelu_not_gated() {
        let mut cfg = tiny_gte_config();
        cfg.extra
            .insert("hidden_act".to_string(), serde_json::json!("gelu"));
        let gte_cfg = GteConfig::from_model_config(&cfg);
        assert!(!gte_cfg.is_gated());
    }

    // ─── Jina (XLMRobertaModel) Tests ────────────────────────────────────────

    fn tiny_jina_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("layer_norm_eps".to_string(), serde_json::json!(1e-12));
        extra.insert("type_vocab_size".to_string(), serde_json::json!(0));
        extra.insert("hidden_act".to_string(), serde_json::json!("gelu"));
        extra.insert("rotary_emb_dim".to_string(), serde_json::json!(32));
        extra.insert("bias".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["XLMRobertaModel".to_string()],
            hidden_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 8194,
            head_dim: 32,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-12,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 0,
            eos_token_id: 0,
            sliding_window: None,
            attention_bias: Some(true),
            extra,
        }
    }

    #[test]
    fn test_jina_construction() {
        let cfg = tiny_jina_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = GteNewForEmbedding::new_jina(&cfg, vb);
        assert!(
            model.is_ok(),
            "JinaRoberta should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.hidden_size, 64);
        assert_eq!(model.max_position_embeddings, 8194);
        assert_eq!(model.encoder.layers.len(), 2);
    }

    #[test]
    fn test_jina_pooling_strategy_is_mean() {
        let cfg = tiny_jina_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_jina(&cfg, vb).unwrap();

        assert_eq!(
            ModelForEmbedding::pooling_strategy(&model),
            PoolingStrategy::Mean
        );
    }

    #[test]
    fn test_jina_forward_shape() {
        let cfg = tiny_jina_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_jina(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 5), DType::U32, &device).unwrap();
        let hidden = model.encode_hidden(&input_ids).unwrap();
        assert_eq!(hidden.dims(), &[2, 5, 64]);
    }

    #[test]
    fn test_jina_encode_mean_pooling() {
        let cfg = tiny_jina_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_jina(&cfg, vb).unwrap();

        let input_ids = Tensor::zeros((2, 6), DType::U32, &device).unwrap();
        let mask = Tensor::ones((2, 6), DType::F32, &device).unwrap();

        let embeddings = model.encode(&input_ids, Some(&mask)).unwrap();
        assert_eq!(embeddings.dims(), &[2, 64]);
    }

    #[test]
    fn test_jina_no_token_type_embeddings() {
        let cfg = tiny_jina_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = GteNewForEmbedding::new_jina(&cfg, vb).unwrap();

        // type_vocab_size = 0
        assert!(model.embeddings.token_type_embeddings.is_none());
    }

    #[test]
    fn test_jina_uses_standard_gelu_mlp() {
        let cfg = tiny_jina_config();
        let gte_cfg = GteConfig::from_model_config(&cfg);

        // hidden_act = "gelu" → not gated → Standard MLP
        assert!(!gte_cfg.is_gated());
        assert_eq!(gte_cfg.hidden_act, "gelu");
    }
}
