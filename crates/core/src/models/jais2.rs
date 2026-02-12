use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, rms_norm, LayerNorm, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── ReLU-squared activation ─────────────────────────────────────────────────
// relu2(x) = max(0, x)^2

fn relu_squared(xs: &Tensor) -> Result<Tensor> {
    xs.relu()?.sqr()
}

// ─── Normalization ───────────────────────────────────────────────────────────
// Jais2 uses LayerNorm; Arcee uses RMSNorm. Both share the same model structure
// otherwise, so we abstract the norm layer.

enum NormLayer {
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
}

impl NormLayer {
    fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self::LayerNorm(layer_norm(size, eps, vb)?))
    }

    fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self::RmsNorm(rms_norm(size, eps, vb)?))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(ln) => ln.forward(xs),
            Self::RmsNorm(rn) => rn.forward(xs),
        }
    }
}

/// Which normalization layer to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard LayerNorm (Jais2)
    LayerNorm,
    /// RMSNorm (Arcee / Llama-style)
    RmsNorm,
}

// ─── ReLU² MLP ───────────────────────────────────────────────────────────────
// Unlike SwiGLU (gate_proj * up_proj), relu2 MLP is: down(relu2(up(x)))
// No gate_proj needed.

struct Relu2Mlp {
    up_proj: TpLinear,
    down_proj: TpLinear,
}

impl Relu2Mlp {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let bias = cfg
            .extra
            .get("mlp_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let up_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            cfg.intermediate_size,
            bias,
            false,
            vb.pp("up_proj"),
            pg,
        )?;
        let down_proj = TpLinear::row_parallel(
            cfg.intermediate_size,
            cfg.hidden_size,
            bias,
            true,
            vb.pp("down_proj"),
            pg,
        )?;

        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let xs = self.up_proj.forward(xs, tp_ctx)?;
        let xs = relu_squared(&xs)?;
        self.down_proj.forward(&xs, tp_ctx)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct Jais2Attention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    o_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Jais2Attention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();
        let bias = cfg.attention_bias.unwrap_or(false);

        if world_size > 1 {
            if !num_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
                )));
            }
            if !num_kv_heads.is_multiple_of(world_size) {
                return Err(candle_core::Error::Msg(format!(
                    "num_kv_heads ({num_kv_heads}) must be divisible by world_size ({world_size})"
                )));
            }
        }

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            bias,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            bias,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let o_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            bias,
            true,
            vb.pp("o_proj"),
            pg,
        )?;

        let num_heads_per_gpu = num_heads / world_size;
        let num_kv_heads_per_gpu = num_kv_heads / world_size;

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
            num_heads: num_heads_per_gpu,
            num_kv_heads: num_kv_heads_per_gpu,
            head_dim,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        cache_engine: &mut CacheEngine,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

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

        self.o_proj.forward(&attn_output, tp_ctx)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let q = self.q_proj.forward(xs, tp_ctx)?;
        let k = self.k_proj.forward(xs, tp_ctx)?;
        let v = self.v_proj.forward(xs, tp_ctx)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.rotary_emb.apply_varlen(&q, &k, &positions)?;

            let all_slot_mapping: Vec<usize> = sequences
                .iter()
                .flat_map(|s| s.slot_mapping.iter().copied())
                .collect();
            cache_engine
                .write_batch(&k, &v, &all_slot_mapping)
                .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?;

            let max_blocks_per_seq = sequences
                .iter()
                .map(|s| s.block_ids.len())
                .max()
                .unwrap_or(1);
            let mut bt_data = vec![0u32; batch_size * max_blocks_per_seq];
            for (i, seq) in sequences.iter().enumerate() {
                for (j, &block_id) in seq.block_ids.iter().enumerate() {
                    bt_data[i * max_blocks_per_seq + j] = block_id as u32;
                }
            }
            let block_tables =
                Tensor::from_vec(bt_data, (batch_size, max_blocks_per_seq), q.device())?;

            let seq_lens_data: Vec<u32> = sequences
                .iter()
                .map(|s| (s.seqlen_offset + 1) as u32)
                .collect();
            let max_seq_len = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
            let seq_lens = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;

            let scale = 1.0 / (self.head_dim as f32).sqrt();

            let attn_output = crate::cuda_kernels::paged_attention_cuda(
                &q,
                cache_engine.k_cache(),
                cache_engine.v_cache(),
                &block_tables,
                &seq_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                max_blocks_per_seq,
                max_seq_len,
                self.head_dim,
                cache_engine.block_size(),
            )?;

            self.o_proj.forward(&attn_output, tp_ctx)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = self.rotary_emb.apply(&q_i, &k_i, seq.seqlen_offset)?;

                let attn_out = paged_attention(
                    &q_i,
                    &k_i,
                    &v_i,
                    None,
                    seq.seqlen_offset,
                    cache_engine,
                    &seq.block_ids,
                    &seq.slot_mapping,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                )?;
                outputs.push(attn_out);
            }

            let attn_output = Tensor::cat(&outputs, 0)?;
            self.o_proj.forward(&attn_output, tp_ctx)
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct Jais2DecoderLayer {
    self_attn: Jais2Attention,
    mlp: Relu2Mlp,
    input_layernorm: NormLayer,
    post_attention_layernorm: NormLayer,
}

impl Jais2DecoderLayer {
    fn new_with_tp(
        cfg: &ModelConfig,
        norm_type: NormType,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
    ) -> Result<Self> {
        let self_attn = Jais2Attention::new_with_tp(cfg, vb.pp("self_attn"), pg)?;
        let mlp = Relu2Mlp::new_with_tp(cfg, vb.pp("mlp"), pg)?;

        let norm_eps = norm_eps_for_type(cfg, norm_type);

        let input_layernorm = make_norm(
            norm_type,
            cfg.hidden_size,
            norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = make_norm(
            norm_type,
            cfg.hidden_size,
            norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        tp_ctx: &TpContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?, tp_ctx)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Jais2 / Arcee causal language model.
///
/// Both Jais2 and Arcee are Llama-like architectures that replace SwiGLU with
/// relu-squared activation (no gate_proj). The only structural difference is
/// the norm layer: Jais2 uses LayerNorm, Arcee uses RMSNorm.
pub struct Jais2ForCausalLM {
    embed_tokens: TpEmbedding,
    layers: Vec<Jais2DecoderLayer>,
    norm: NormLayer,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl Jais2ForCausalLM {
    /// Create a Jais2 model with LayerNorm (single GPU).
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_norm_and_tp(
            cfg,
            NormType::LayerNorm,
            vb,
            &LocalProcessGroup::new(),
            TpContext::single_gpu(),
        )
    }

    /// Create a Jais2 model with LayerNorm and tensor parallelism.
    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        Self::new_with_norm_and_tp(cfg, NormType::LayerNorm, vb, pg, tp_ctx)
    }

    /// Create model with explicit norm type (single GPU).
    pub fn new_with_norm(cfg: &ModelConfig, norm_type: NormType, vb: VarBuilder) -> Result<Self> {
        Self::new_with_norm_and_tp(
            cfg,
            norm_type,
            vb,
            &LocalProcessGroup::new(),
            TpContext::single_gpu(),
        )
    }

    /// Create model with explicit norm type and tensor parallelism.
    pub fn new_with_norm_and_tp(
        cfg: &ModelConfig,
        norm_type: NormType,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let world_size = pg.world_size();

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Jais2DecoderLayer::new_with_tp(
                cfg,
                norm_type,
                vb_l.pp(i),
                pg,
            )?);
        }

        let norm_eps = norm_eps_for_type(cfg, norm_type);
        let norm = make_norm(norm_type, cfg.hidden_size, norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(crate::layers::causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tp_context(&self) -> &TpContext {
        &self.tp_ctx
    }
}

impl crate::engine::ModelForward for Jais2ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Jais2ForCausalLM::forward(
            self,
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs, &self.tp_ctx)?;
        Ok(logits)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn norm_eps_for_type(cfg: &ModelConfig, norm_type: NormType) -> f64 {
    match norm_type {
        NormType::LayerNorm => cfg
            .extra
            .get("layer_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(cfg.rms_norm_eps),
        NormType::RmsNorm => cfg.rms_norm_eps,
    }
}

fn make_norm(
    norm_type: NormType,
    size: usize,
    eps: f64,
    vb: VarBuilder,
) -> Result<NormLayer> {
    match norm_type {
        NormType::LayerNorm => NormLayer::layer_norm(size, eps, vb),
        NormType::RmsNorm => NormLayer::rms_norm(size, eps, vb),
    }
}

// ─── Arcee constructors ──────────────────────────────────────────────────────
// Arcee is architecturally identical to Jais2 but uses RMSNorm.
// Exposed as free functions rather than a newtype to avoid clippy's
// `new_ret_no_self` warning, since the returned type is `Jais2ForCausalLM`.

/// Create an Arcee (AFM) model with RMSNorm (single GPU).
///
/// Identical to Jais2 but uses RMSNorm instead of LayerNorm. Weight names and
/// projection structure are the same (up_proj, down_proj with relu-squared).
pub fn new_arcee(cfg: &ModelConfig, vb: VarBuilder) -> Result<Jais2ForCausalLM> {
    Jais2ForCausalLM::new_with_norm(cfg, NormType::RmsNorm, vb)
}

/// Create an Arcee (AFM) model with RMSNorm and tensor parallelism.
pub fn new_arcee_with_tp(
    cfg: &ModelConfig,
    vb: VarBuilder,
    pg: &dyn ProcessGroup,
    tp_ctx: TpContext,
) -> Result<Jais2ForCausalLM> {
    Jais2ForCausalLM::new_with_norm_and_tp(cfg, NormType::RmsNorm, vb, pg, tp_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Jais2ForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "relu2".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn arcee_config() -> ModelConfig {
        let mut cfg = test_config();
        cfg.architectures = vec!["ArceeForCausalLM".to_string()];
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

    // ─── relu_squared tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_squared_positive() {
        let device = Device::Cpu;
        let xs = Tensor::new(&[2.0_f32, 3.0, 0.5], &device).expect("tensor");
        let result = relu_squared(&xs).expect("relu2");
        let values: Vec<f32> = result.to_vec1().expect("to_vec");
        assert!((values[0] - 4.0).abs() < 1e-6, "relu2(2) = 4");
        assert!((values[1] - 9.0).abs() < 1e-6, "relu2(3) = 9");
        assert!((values[2] - 0.25).abs() < 1e-6, "relu2(0.5) = 0.25");
    }

    #[test]
    fn test_relu_squared_negative() {
        let device = Device::Cpu;
        let xs = Tensor::new(&[-1.0_f32, -0.5, 0.0], &device).expect("tensor");
        let result = relu_squared(&xs).expect("relu2");
        let values: Vec<f32> = result.to_vec1().expect("to_vec");
        assert!((values[0]).abs() < 1e-6, "relu2(-1) = 0");
        assert!((values[1]).abs() < 1e-6, "relu2(-0.5) = 0");
        assert!((values[2]).abs() < 1e-6, "relu2(0) = 0");
    }

    #[test]
    fn test_relu_squared_mixed() {
        let device = Device::Cpu;
        let xs = Tensor::new(&[-2.0_f32, 0.0, 1.0, -0.1, 4.0], &device).expect("tensor");
        let result = relu_squared(&xs).expect("relu2");
        let values: Vec<f32> = result.to_vec1().expect("to_vec");
        let expected = [0.0, 0.0, 1.0, 0.0, 16.0];
        for (i, (&v, &e)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((v - e).abs() < 1e-6, "index {i}: got {v}, expected {e}");
        }
    }

    // ─── Jais2 construction tests ────────────────────────────────────────────

    #[test]
    fn test_jais2_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Jais2ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Jais2ForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais2_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_jais2_single_token_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 1), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 1);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_jais2_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill with 3 tokens
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode step with seqlen_offset=3
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = model
            .forward(
                &next_token,
                3,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_jais2_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.device(), Device::Cpu),
            "device() should return the construction device"
        );
    }

    #[test]
    fn test_jais2_gqa_configuration() {
        let cfg = test_config();
        let gqa_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(gqa_groups, 2, "test config uses GQA with 2 groups");
    }

    #[test]
    fn test_jais2_tied_embeddings() {
        let cfg = test_config();
        assert!(cfg.tie_word_embeddings);

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_jais2_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    // ─── Jais2 uses LayerNorm ────────────────────────────────────────────────

    #[test]
    fn test_jais2_uses_layernorm() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        assert!(
            matches!(model.norm, NormLayer::LayerNorm(_)),
            "Jais2 should use LayerNorm"
        );
        for layer in &model.layers {
            assert!(
                matches!(layer.input_layernorm, NormLayer::LayerNorm(_)),
                "decoder layer input_layernorm should be LayerNorm"
            );
            assert!(
                matches!(layer.post_attention_layernorm, NormLayer::LayerNorm(_)),
                "decoder layer post_attention_layernorm should be LayerNorm"
            );
        }
    }

    // ─── Arcee construction tests ────────────────────────────────────────────

    #[test]
    fn test_arcee_construction() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = new_arcee(&cfg, vb);
        assert!(
            model.is_ok(),
            "ArceeForCausalLM should construct: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_arcee_uses_rmsnorm() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = new_arcee(&cfg, vb).expect("build model");

        assert!(
            matches!(model.norm, NormLayer::RmsNorm(_)),
            "Arcee should use RMSNorm"
        );
        for layer in &model.layers {
            assert!(
                matches!(layer.input_layernorm, NormLayer::RmsNorm(_)),
                "decoder layer input_layernorm should be RMSNorm"
            );
            assert!(
                matches!(layer.post_attention_layernorm, NormLayer::RmsNorm(_)),
                "decoder layer post_attention_layernorm should be RMSNorm"
            );
        }
    }

    #[test]
    fn test_arcee_forward_shape() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = new_arcee(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 4;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, seq_len);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward");

        assert_eq!(
            logits.dims(),
            &[batch_size, seq_len, cfg.vocab_size],
            "logits shape should be [batch, seq_len, vocab_size]"
        );
    }

    #[test]
    fn test_arcee_prefill_then_decode() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = new_arcee(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).expect("next token");

        let logits = model
            .forward(
                &next_token,
                3,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    // ─── No gate_proj verification ───────────────────────────────────────────

    #[test]
    fn test_relu2_mlp_has_no_gate_proj() {
        // Verify the MLP structure: only up_proj + down_proj, no gate_proj.
        // This is the key difference from Llama's SwiGLU MLP.
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let pg = LocalProcessGroup::new();

        let mlp = Relu2Mlp::new_with_tp(&cfg, vb.pp("mlp"), &pg).expect("build mlp");

        // Verify forward works (up_proj -> relu2 -> down_proj)
        let xs = Tensor::zeros((1, 3, cfg.hidden_size), DType::F32, &device).expect("input");
        let tp_ctx = TpContext::single_gpu();
        let result = mlp.forward(&xs, &tp_ctx);
        assert!(result.is_ok(), "Relu2Mlp forward should work");
        let result = result.unwrap();
        assert_eq!(
            result.dims(),
            &[1, 3, cfg.hidden_size],
            "MLP output should preserve hidden_size"
        );
    }

    // ─── TP tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_jais2_tp_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = Jais2ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Jais2 should construct with TP=2: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_jais2_tp_heads_divisibility_check() {
        let mut cfg = test_config();
        cfg.num_key_value_heads = 3; // Not divisible by 2

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let result = Jais2ForCausalLM::new_with_tp(&cfg, vb, &pg, tp_ctx);
        match result {
            Ok(_) => panic!("Should fail when num_kv_heads is not divisible by world_size"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("divisible"),
                    "Error should mention divisibility: {err_msg}",
                );
            }
        }
    }

    #[test]
    fn test_arcee_tp_construction() {
        let cfg = arcee_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let pg = LocalProcessGroup::with_rank(0, 2);
        let tp_ctx = TpContext::mock_multi_gpu(0, 2);

        let model = new_arcee_with_tp(&cfg, vb, &pg, tp_ctx);
        assert!(
            model.is_ok(),
            "Arcee should construct with TP=2: {:?}",
            model.err()
        );

        let model = model.unwrap();
        assert!(
            matches!(model.norm, NormLayer::RmsNorm(_)),
            "Arcee TP model should use RMSNorm"
        );
    }

    // ─── ModelForward trait ──────────────────────────────────────────────────

    #[test]
    fn test_jais2_model_forward_trait() {
        use crate::engine::ModelForward;

        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Jais2ForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        // Use trait method
        let logits = ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("trait forward");

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    // ─── norm_eps_for_type tests ─────────────────────────────────────────────

    #[test]
    fn test_norm_eps_layernorm_from_extra() {
        let mut cfg = test_config();
        cfg.extra.insert(
            "layer_norm_eps".to_string(),
            serde_json::Value::from(1e-5),
        );

        let eps = norm_eps_for_type(&cfg, NormType::LayerNorm);
        assert!((eps - 1e-5).abs() < 1e-12, "should use layer_norm_eps from extra");
    }

    #[test]
    fn test_norm_eps_layernorm_fallback() {
        let cfg = test_config();
        let eps = norm_eps_for_type(&cfg, NormType::LayerNorm);
        assert!(
            (eps - cfg.rms_norm_eps).abs() < 1e-12,
            "should fall back to rms_norm_eps"
        );
    }

    #[test]
    fn test_norm_eps_rmsnorm() {
        let cfg = test_config();
        let eps = norm_eps_for_type(&cfg, NormType::RmsNorm);
        assert!(
            (eps - cfg.rms_norm_eps).abs() < 1e-12,
            "should use rms_norm_eps"
        );
    }
}
