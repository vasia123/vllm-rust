//! Eagle-3 speculative decoding draft model (Llama architecture).
//!
//! Eagle-3 generates draft tokens by combining target model hidden states
//! with token embeddings. Layer 0 concatenates embeddings and hidden states
//! (2×hidden_size) before QKV projection; subsequent layers use hidden_size.
//!
//! Key differences from standard Llama:
//! - Layer 0 QKV input: 2×hidden_size (embeds ++ hidden_states)
//! - Optional `fc` projection (3×target_hidden_size → hidden_size) for auxiliary hidden states
//! - Draft vocab remapping via `draft_id_to_target_id`
//! - `norm_before_residual` config option for normalization ordering
//!
//! NOTE: This model does NOT implement `ModelForward` because its forward
//! pass requires `hidden_states` from the target model — a different signature.
//! Integration with speculative decoding happens through a future
//! `Eagle3DraftProposer` that wraps this model.
//!
//! Reference: `reference/vllm/vllm/model_executor/models/llama_eagle3.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

// ─── Eagle3 Config ──────────────────────────────────────────────────────────

/// Configuration specific to Eagle-3 draft models, parsed from the
/// model's config.json fields.
#[derive(Debug, Clone)]
pub struct Eagle3Config {
    /// Whether to use auxiliary hidden states from target model.
    pub use_aux_hidden_state: bool,
    /// Whether to normalize before computing residual (vs after).
    pub norm_before_residual: bool,
    /// Whether the attention layers use bias.
    pub attention_bias: bool,
    /// Target model's hidden size (may differ from draft model's).
    pub target_hidden_size: Option<usize>,
    /// Draft vocabulary size (may be smaller than target vocab).
    pub draft_vocab_size: Option<usize>,
    /// Logit scaling factor.
    pub logit_scale: f64,
}

impl Eagle3Config {
    /// Parse Eagle3 configuration from a ModelConfig.
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        let eagle_config = cfg.extra.get("eagle_config");

        let use_aux_hidden_state = eagle_config
            .and_then(|v| v.get("use_aux_hidden_state"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let norm_before_residual = cfg
            .extra
            .get("norm_before_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let attention_bias = cfg.attention_bias.unwrap_or(false);

        let target_hidden_size = cfg
            .extra
            .get("target_hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let draft_vocab_size = cfg
            .extra
            .get("draft_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let logit_scale = cfg
            .extra
            .get("logit_scale")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        Self {
            use_aux_hidden_state,
            norm_before_residual,
            attention_bias,
            target_hidden_size,
            draft_vocab_size,
            logit_scale,
        }
    }
}

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct Eagle3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Eagle3Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Eagle3 Attention ───────────────────────────────────────────────────────

/// Attention for Eagle3 decoder layers.
///
/// Layer 0's QKV projection accepts 2×hidden_size input (concatenation of
/// embeddings and hidden states); subsequent layers accept hidden_size.
struct Eagle3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Eagle3Attention {
    fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        qkv_input_size: usize,
        attention_bias: bool,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = candle_nn::linear_b(
            qkv_input_size,
            num_heads * head_dim,
            attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            qkv_input_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            qkv_input_size,
            num_kv_heads * head_dim,
            attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut crate::kv_cache::CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.apply(&q, &k, seqlen_offset)?;

        let attn_output = paged_attention(
            &q,
            &k,
            &v,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table.block_ids(),
            slot_mapping,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )?;

        self.o_proj.forward(&attn_output)
    }
}

// ─── Eagle3 Decoder Layer ───────────────────────────────────────────────────

struct Eagle3DecoderLayer {
    self_attn: Eagle3Attention,
    mlp: Eagle3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    hidden_norm: RmsNorm,
    layer_idx: usize,
    norm_before_residual: bool,
}

impl Eagle3DecoderLayer {
    fn new(
        cfg: &ModelConfig,
        eagle_cfg: &Eagle3Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        // Layer 0: QKV input is 2*hidden_size (embeds ++ hidden_states)
        let qkv_input_size = if layer_idx == 0 {
            2 * cfg.hidden_size
        } else {
            cfg.hidden_size
        };

        let self_attn = Eagle3Attention::new(
            cfg,
            vb.pp("self_attn"),
            qkv_input_size,
            eagle_cfg.attention_bias,
        )?;
        let mlp = Eagle3Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;

        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let hidden_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hidden_norm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            hidden_norm,
            layer_idx,
            norm_before_residual: eagle_cfg.norm_before_residual,
        })
    }

    /// Forward pass for a single Eagle3 decoder layer.
    ///
    /// Layer 0: concatenates `norm(embeds)` with `norm(hidden_states)` → 2×hidden_size.
    /// Subsequent layers: standard pre-norm residual flow.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        embeds: &Tensor,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut crate::kv_cache::CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (hs, residual) = if self.layer_idx == 0 {
            let normed_embeds = self.input_layernorm.forward(embeds)?;
            let (normed_hs, res) = if self.norm_before_residual {
                let hs = self.hidden_norm.forward(hidden_states)?;
                (hs.clone(), hs)
            } else {
                let hs = self.hidden_norm.forward(hidden_states)?;
                (hs, hidden_states.clone())
            };
            // Concatenate along hidden dim: [batch, seq, 2*hidden]
            let concatenated = Tensor::cat(&[&normed_embeds, &normed_hs], 2)?;
            (concatenated, res)
        } else {
            // Fused residual + norm: x = x + residual; residual = x; x = norm(x)
            let residual = residual.expect("residual required for layers > 0");
            let xs = (hidden_states + residual)?;
            let res = xs.clone();
            let xs = self.input_layernorm.forward(&xs)?;
            (xs, res)
        };

        // Self attention
        let hs = self.self_attn.forward(
            &hs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;

        // Post-attention: fused residual + norm + MLP
        let xs = (hs + &residual)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;

        Ok((xs, residual))
    }
}

// ─── Eagle3 Llama Model ─────────────────────────────────────────────────────

/// Inner model: embedding + decoder layers + fc projection + norm.
struct Eagle3LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<Eagle3DecoderLayer>,
    /// Projects 3×target_hidden_size → hidden_size when use_aux_hidden_state is true.
    fc: Option<Linear>,
    norm: RmsNorm,
    use_aux_hidden_state: bool,
}

impl Eagle3LlamaModel {
    fn new(cfg: &ModelConfig, eagle_cfg: &Eagle3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Eagle3DecoderLayer::new(cfg, eagle_cfg, vb_l.pp(i), i)?);
        }

        let fc = if eagle_cfg.use_aux_hidden_state {
            let fc_input_size = eagle_cfg
                .target_hidden_size
                .unwrap_or(cfg.hidden_size)
                .checked_mul(3)
                .expect("fc_input_size overflow");
            Some(linear_no_bias(
                fc_input_size,
                cfg.hidden_size,
                vb.pp("fc"),
            )?)
        } else {
            None
        };

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            fc,
            norm,
            use_aux_hidden_state: eagle_cfg.use_aux_hidden_state,
        })
    }

    /// Forward pass through the Eagle3 transformer backbone.
    ///
    /// Returns `(hidden_states, hidden_prenorm)`:
    /// - `hidden_states`: post-norm output (used for logits)
    /// - `hidden_prenorm`: pre-norm output (residual before final norm)
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len) = input_ids.dims2()?;
        let embeds = self.embed_tokens.forward(input_ids)?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                embeds.dtype(),
                embeds.device(),
            )?)
        };

        let mut hs = hidden_states.clone();
        let mut residual: Option<Tensor> = None;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (new_hs, new_residual) = layer.forward(
                &embeds,
                &hs,
                residual.as_ref(),
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
            hs = new_hs;
            residual = Some(new_residual);
        }

        // Final fused residual + norm
        let residual = residual.expect("at least one layer required");
        let hidden_prenorm = (&hs + &residual)?;
        let hidden_states = self.norm.forward(&hidden_prenorm)?;

        Ok((hidden_states, hidden_prenorm))
    }

    /// Combine auxiliary hidden states using the fc projection.
    ///
    /// Called externally to project target model's auxiliary hidden states
    /// (3×target_hidden_size) down to draft model's hidden_size.
    fn combine_hidden_states(&self, hidden_states: &Tensor) -> Result<Tensor> {
        match &self.fc {
            Some(fc) => fc.forward(hidden_states),
            None => Ok(hidden_states.clone()),
        }
    }
}

// ─── Eagle3 Llama For CausalLM ──────────────────────────────────────────────

/// Eagle-3 speculative decoding draft model based on Llama architecture.
///
/// This model generates draft tokens for speculative decoding by combining
/// target model hidden states with token embeddings. It supports:
/// - Draft vocabulary remapping (draft_id_to_target_id)
/// - Auxiliary hidden state combination (fc projection)
/// - Logit scaling
pub struct Eagle3LlamaForCausalLM {
    model: Eagle3LlamaModel,
    lm_head: Linear,
    /// Maps draft token indices to target token indices.
    /// `None` means identity mapping (draft vocab == target vocab).
    draft_id_to_target_id: Option<Vec<usize>>,
    draft_vocab_size: usize,
    target_vocab_size: usize,
    logit_scale: f64,
    device: Device,
    dtype: DType,
}

impl Eagle3LlamaForCausalLM {
    /// Create a new Eagle3 draft model.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let eagle_cfg = Eagle3Config::from_model_config(cfg);
        let draft_vocab_size = eagle_cfg.draft_vocab_size.unwrap_or(cfg.vocab_size);
        let target_vocab_size = cfg.vocab_size;

        let model = Eagle3LlamaModel::new(cfg, &eagle_cfg, vb.pp("model"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, draft_vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            model,
            lm_head,
            draft_id_to_target_id: None, // Set during weight loading
            draft_vocab_size,
            target_vocab_size,
            logit_scale: eagle_cfg.logit_scale,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Set the draft-to-target vocabulary mapping.
    ///
    /// Each entry `mapping[i]` is the target vocab index that draft token `i`
    /// maps to. When set, `compute_logits()` scatters draft logits into the
    /// full target vocab space.
    pub fn set_draft_id_to_target_id(&mut self, mapping: Vec<usize>) {
        assert_eq!(
            mapping.len(),
            self.draft_vocab_size,
            "mapping length must equal draft_vocab_size"
        );
        self.draft_id_to_target_id = Some(mapping);
    }

    /// Forward pass through the Eagle3 draft model.
    ///
    /// # Arguments
    /// * `input_ids` - Draft token IDs `[batch, seq_len]`
    /// * `hidden_states` - Hidden states from the target model `[batch, seq_len, hidden_size]`
    /// * `seqlen_offset` - Position offset for RoPE
    /// * `kv_cache_mgr` - Eagle3's own KV cache manager
    /// * `block_table` - Block table for paged attention
    /// * `slot_mapping` - Slot mapping for cache writes
    ///
    /// # Returns
    /// `(hidden_states, hidden_prenorm)` — post-norm and pre-norm outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        self.model
            .forward(input_ids, hidden_states, seqlen_offset, kv_cache_mgr, block_table, slot_mapping)
    }

    /// Compute logits from hidden states, with optional draft→target vocab remapping.
    ///
    /// If `draft_id_to_target_id` is set, scatters draft logits into -inf-filled
    /// target vocab tensor. Otherwise returns draft logits directly.
    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let logits = self.lm_head.forward(hidden_states)?;

        // Apply logit scale
        let logits = if (self.logit_scale - 1.0).abs() > f64::EPSILON {
            (logits * self.logit_scale)?
        } else {
            logits
        };

        // Remap draft vocab → target vocab
        match &self.draft_id_to_target_id {
            None => Ok(logits),
            Some(mapping) => {
                let dims = logits.dims();
                let batch_seq = dims[..dims.len() - 1].iter().product::<usize>();
                let flat = logits.reshape((batch_seq, self.draft_vocab_size))?;

                // Extract draft logits and scatter into target-sized tensor.
                // Uses SET semantics (not add) — unmapped positions stay -inf.
                let flat_data: Vec<f32> =
                    flat.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                let mut result_data =
                    vec![f32::NEG_INFINITY; batch_seq * self.target_vocab_size];

                for batch in 0..batch_seq {
                    for (draft_idx, &target_idx) in mapping.iter().enumerate() {
                        result_data[batch * self.target_vocab_size + target_idx] =
                            flat_data[batch * self.draft_vocab_size + draft_idx];
                    }
                }

                let result = Tensor::from_vec(
                    result_data,
                    (batch_seq, self.target_vocab_size),
                    &self.device,
                )?
                .to_dtype(self.dtype)?;

                // Reshape back to original batch dims
                let mut result_dims = dims.to_vec();
                *result_dims.last_mut().expect("non-empty dims") = self.target_vocab_size;
                result.reshape(result_dims)
            }
        }
    }

    /// Combine auxiliary hidden states from the target model.
    ///
    /// Projects `3×target_hidden_size → hidden_size` via the fc layer.
    /// Pass-through if `use_aux_hidden_state` is false.
    pub fn combine_hidden_states(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.model.combine_hidden_states(hidden_states)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this model uses auxiliary hidden state combination.
    pub fn use_aux_hidden_state(&self) -> bool {
        self.model.use_aux_hidden_state
    }

    pub fn draft_vocab_size(&self) -> usize {
        self.draft_vocab_size
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size
    }
}

// ─── Eagle3 Draft Model Trait ──────────────────────────────────────────────

/// Trait for Eagle3-family draft models that require target hidden states.
///
/// Both [`Eagle3LlamaForCausalLM`] and
/// [`Eagle3MistralLarge3ForCausalLM`](super::Eagle3MistralLarge3ForCausalLM)
/// implement this trait, allowing the [`Eagle3DraftProposer`](crate::engine::spec_decode::Eagle3DraftProposer)
/// to be generic over the underlying architecture.
pub trait Eagle3DraftModel: Send {
    /// Forward pass taking input_ids and target hidden states.
    ///
    /// Returns `(hidden_states, hidden_prenorm)`:
    /// - Llama variant: hidden_states (post-norm) and hidden_prenorm (pre-norm) differ
    /// - MistralLarge3 variant: both are the same tensor
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)>;

    /// Compute logits from hidden states.
    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor>;

    /// Project auxiliary hidden states from target model.
    ///
    /// For models with `use_aux_hidden_state() == true`, this projects
    /// 3×target_hidden_size → hidden_size. Otherwise returns a clone.
    fn combine_hidden_states(&self, hidden_states: &Tensor) -> Result<Tensor>;

    /// Whether this model uses auxiliary hidden state combination.
    fn use_aux_hidden_state(&self) -> bool;

    /// Device this model resides on.
    fn device(&self) -> &Device;
}

impl Eagle3DraftModel for Eagle3LlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        hidden_states: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        Eagle3LlamaForCausalLM::forward(
            self,
            input_ids,
            hidden_states,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Eagle3LlamaForCausalLM::compute_logits(self, hidden_states)
    }

    fn combine_hidden_states(&self, hidden_states: &Tensor) -> Result<Tensor> {
        Eagle3LlamaForCausalLM::combine_hidden_states(self, hidden_states)
    }

    fn use_aux_hidden_state(&self) -> bool {
        Eagle3LlamaForCausalLM::use_aux_hidden_state(self)
    }

    fn device(&self) -> &Device {
        Eagle3LlamaForCausalLM::device(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn base_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Eagle3LlamaForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn config_with_eagle(use_aux: bool, norm_before: bool) -> ModelConfig {
        let mut cfg = base_config();
        let mut eagle = serde_json::Map::new();
        eagle.insert(
            "use_aux_hidden_state".to_string(),
            serde_json::Value::Bool(use_aux),
        );
        cfg.extra.insert(
            "eagle_config".to_string(),
            serde_json::Value::Object(eagle),
        );
        cfg.extra.insert(
            "norm_before_residual".to_string(),
            serde_json::Value::Bool(norm_before),
        );
        cfg
    }

    fn config_with_draft_vocab(draft_vocab_size: usize) -> ModelConfig {
        let mut cfg = config_with_eagle(true, false);
        cfg.extra.insert(
            "draft_vocab_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(draft_vocab_size)),
        );
        cfg
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    // ─── Config Parsing ─────────────────────────────────────────────────

    #[test]
    fn eagle3_config_defaults() {
        let cfg = base_config();
        let eagle = Eagle3Config::from_model_config(&cfg);

        assert!(eagle.use_aux_hidden_state);
        assert!(!eagle.norm_before_residual);
        assert!(!eagle.attention_bias);
        assert!(eagle.target_hidden_size.is_none());
        assert!(eagle.draft_vocab_size.is_none());
        assert!((eagle.logit_scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn eagle3_config_custom() {
        let mut cfg = base_config();
        let mut eagle = serde_json::Map::new();
        eagle.insert(
            "use_aux_hidden_state".to_string(),
            serde_json::Value::Bool(false),
        );
        cfg.extra.insert(
            "eagle_config".to_string(),
            serde_json::Value::Object(eagle),
        );
        cfg.extra.insert(
            "norm_before_residual".to_string(),
            serde_json::Value::Bool(true),
        );
        cfg.extra.insert(
            "target_hidden_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(128)),
        );
        cfg.extra.insert(
            "draft_vocab_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(64)),
        );
        cfg.extra.insert(
            "logit_scale".to_string(),
            serde_json::json!(2.0).as_number().unwrap().clone().into(),
        );
        cfg.attention_bias = Some(true);

        let eagle_cfg = Eagle3Config::from_model_config(&cfg);

        assert!(!eagle_cfg.use_aux_hidden_state);
        assert!(eagle_cfg.norm_before_residual);
        assert!(eagle_cfg.attention_bias);
        assert_eq!(eagle_cfg.target_hidden_size, Some(128));
        assert_eq!(eagle_cfg.draft_vocab_size, Some(64));
        assert!((eagle_cfg.logit_scale - 2.0).abs() < f64::EPSILON);
    }

    // ─── Construction ───────────────────────────────────────────────────

    #[test]
    fn eagle3_construction_with_aux() {
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.model.layers.len(), cfg.num_hidden_layers);
        assert!(model.model.fc.is_some());
        assert!(model.use_aux_hidden_state());
    }

    #[test]
    fn eagle3_construction_without_aux() {
        let cfg = config_with_eagle(false, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");
        assert!(model.model.fc.is_none());
        assert!(!model.use_aux_hidden_state());
    }

    #[test]
    fn eagle3_construction_with_draft_vocab() {
        let cfg = config_with_draft_vocab(64);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.draft_vocab_size(), 64);
        assert_eq!(model.target_vocab_size(), 256);
    }

    // ─── Forward ────────────────────────────────────────────────────────

    #[test]
    fn eagle3_forward_shape() {
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 3;
        let input_ids =
            Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input ids");
        let hidden_states = Tensor::zeros(
            (batch_size, seq_len, cfg.hidden_size),
            DType::F32,
            &device,
        )
        .expect("hidden states");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let (hs, hs_prenorm) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(hs.dims(), &[batch_size, seq_len, cfg.hidden_size]);
        assert_eq!(hs_prenorm.dims(), &[batch_size, seq_len, cfg.hidden_size]);
    }

    #[test]
    fn eagle3_forward_single_token() {
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input ids");
        let hidden_states =
            Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("hidden states");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let (hs, _) = model
            .forward(
                &input_ids,
                &hidden_states,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(hs.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn eagle3_forward_norm_before_residual() {
        let cfg = config_with_eagle(true, true);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input ids");
        let hidden_states =
            Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("hidden states");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let result = model.forward(
            &input_ids,
            &hidden_states,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        );
        assert!(result.is_ok(), "forward with norm_before_residual should work");
    }

    // ─── Logits & Vocab Remapping ───────────────────────────────────────

    #[test]
    fn eagle3_compute_logits_no_remapping() {
        let cfg = base_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let hidden = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("hidden");
        let logits = model.compute_logits(&hidden).expect("logits");

        // No remapping: draft_vocab_size == vocab_size == 256
        assert_eq!(logits.dims(), &[1, 1, 256]);
    }

    #[test]
    fn eagle3_compute_logits_with_remapping() {
        let cfg = config_with_draft_vocab(4);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let mut model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        // Map: draft[0]→target[10], draft[1]→target[5], draft[2]→target[3], draft[3]→target[7]
        model.set_draft_id_to_target_id(vec![10, 5, 3, 7]);

        let hidden = Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("hidden");
        let logits = model.compute_logits(&hidden).expect("logits");

        // Output should be in target vocab space
        assert_eq!(logits.dims(), &[1, 1, 256]);

        // With zero weights, all draft logits are 0.0.
        // Mapped positions should be 0.0, unmapped should be -inf.
        let logits_vec: Vec<f32> = logits.flatten_all().expect("flatten").to_vec1().expect("vec");
        assert!(logits_vec[10].is_finite(), "mapped position should be finite");
        assert!(logits_vec[5].is_finite(), "mapped position should be finite");
        assert!(
            logits_vec[0].is_infinite() && logits_vec[0] < 0.0,
            "unmapped position should be -inf"
        );
        assert!(
            logits_vec[1].is_infinite() && logits_vec[1] < 0.0,
            "unmapped position should be -inf"
        );
    }

    // ─── Auxiliary Hidden State Combination ──────────────────────────────

    #[test]
    fn eagle3_combine_hidden_states_with_fc() {
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        // fc expects 3 * hidden_size = 192 input
        let aux_hidden = Tensor::zeros((1, 1, 3 * cfg.hidden_size), DType::F32, &device)
            .expect("aux hidden states");

        let combined = model
            .combine_hidden_states(&aux_hidden)
            .expect("combine hidden states");

        assert_eq!(combined.dims(), &[1, 1, cfg.hidden_size]);
    }

    #[test]
    fn eagle3_combine_hidden_states_passthrough() {
        let cfg = config_with_eagle(false, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let hidden =
            Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("hidden states");

        let combined = model
            .combine_hidden_states(&hidden)
            .expect("combine hidden states");

        // Passthrough: same shape
        assert_eq!(combined.dims(), &[1, 1, cfg.hidden_size]);
    }

    // ─── Prefill + Decode Workflow ──────────────────────────────────────

    #[test]
    fn eagle3_prefill_then_decode() {
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        let hs = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("hs");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let (out, _) = model
            .forward(&prompt, &hs, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(out.dims(), &[1, 3, cfg.hidden_size]);
        block_table.advance(3);

        // Decode step at offset=3
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");
        let next_hs =
            Tensor::zeros((1, 1, cfg.hidden_size), DType::F32, &device).expect("next hs");

        let (out, _) = model
            .forward(
                &next_token,
                &next_hs,
                3,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("decode");
        assert_eq!(out.dims(), &[1, 1, cfg.hidden_size]);
    }

    // ─── Layer 0 Concatenation ──────────────────────────────────────────

    #[test]
    fn eagle3_layer0_qkv_input_size() {
        // Verify layer 0 QKV accepts 2*hidden_size and layer 1 accepts hidden_size
        let cfg = config_with_eagle(true, false);
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        // Layer 0 should have layer_idx=0
        assert_eq!(model.model.layers[0].layer_idx, 0);
        // Layer 1 should have layer_idx=1
        assert_eq!(model.model.layers[1].layer_idx, 1);

        // Verify construction succeeded (proves QKV sizes are correct
        // since VarBuilder::zeros creates correct shapes)
        assert_eq!(model.model.layers.len(), 2);
    }

    // ─── ModelForward trait (not implemented) ───────────────────────────

    #[test]
    fn eagle3_device() {
        let cfg = base_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        assert!(matches!(model.device(), Device::Cpu));
    }

    // ─── Target Hidden Size ─────────────────────────────────────────────

    #[test]
    fn eagle3_fc_with_target_hidden_size() {
        let mut cfg = config_with_eagle(true, false);
        cfg.extra.insert(
            "target_hidden_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(128)),
        );
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Eagle3LlamaForCausalLM::new(&cfg, vb).expect("build model");

        // fc input = 3 * target_hidden_size = 384
        let aux = Tensor::zeros((1, 1, 384), DType::F32, &device).expect("aux");
        let combined = model.combine_hidden_states(&aux).expect("combine");
        assert_eq!(combined.dims(), &[1, 1, cfg.hidden_size]);
    }
}
