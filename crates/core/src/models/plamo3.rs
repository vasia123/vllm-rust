//! PLaMo-3 model architecture.
//!
//! PLaMo-3 (Preferred Networks) is a dense transformer with:
//! - SiLU + gate MLP (SwiGLU)
//! - QK normalization via per-head RMSNorm
//! - RoPE positional embeddings
//! - Custom 4-norm residual connection pattern:
//!   pre_mixer_norm -> mixer -> post_mixer_norm -> pre_mlp_norm -> MLP -> post_mlp_norm
//! - Weight offsets on RMSNorm (loaded weights += offset)
//!
//! Architecture:
//! ```text
//! Embedding -> [Plamo3DecoderLayer x N] -> RMSNorm -> LM Head
//!
//! Plamo3DecoderLayer:
//!   pre_mixer_norm -> Attention -> post_mixer_norm -> pre_mlp_norm -> MLP -> post_mlp_norm
//! ```

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, paged_attention, RotaryEmbedding};

// ─── SwiGLU MLP ─────────────────────────────────────────────────────────────

struct Plamo3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Plamo3Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_up_proj.gate"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_up_proj.up"))?;
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

// ─── Attention ───────────────────────────────────────────────────────────────

struct Plamo3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Plamo3Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // QKV as separate projections (PLaMo3 uses merged qkv_proj in Python,
        // but separate loads are weight-compatible via weight name mapping)
        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        // Per-head QK normalization
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
            q_norm,
            k_norm,
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
        cache_engine: &mut CacheEngine,
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

        // Per-head QK normalization
        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

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

// ─── Decoder Layer ───────────────────────────────────────────────────────────
//
// PLaMo3 uses a 4-norm residual pattern:
//   residual flow: input -> pre_mixer_norm -> mixer -> post_mixer_norm -> pre_mlp_norm -> MLP -> post_mlp_norm

struct Plamo3DecoderLayer {
    mixer: Plamo3Attention,
    mlp: Plamo3Mlp,
    pre_mixer_norm: RmsNorm,
    post_mixer_norm: RmsNorm,
    pre_mlp_norm: RmsNorm,
    post_mlp_norm: RmsNorm,
}

impl Plamo3DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let mixer = Plamo3Attention::new(cfg, vb.pp("mixer"))?;
        let mlp = Plamo3Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
        let pre_mixer_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_mixer_norm"))?;
        let post_mixer_norm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_mixer_norm"))?;
        let pre_mlp_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_mlp_norm"))?;
        let post_mlp_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_mlp_norm"))?;
        Ok(Self {
            mixer,
            mlp,
            pre_mixer_norm,
            post_mixer_norm,
            pre_mlp_norm,
            post_mlp_norm,
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
    ) -> Result<Tensor> {
        // Pre-mixer norm + attention
        let residual = xs;
        let xs = self.pre_mixer_norm.forward(xs)?;
        let xs = self.mixer.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            cache_engine,
            block_table,
            slot_mapping,
        )?;
        let xs = self.post_mixer_norm.forward(&xs)?;

        // Residual connection after mixer
        let xs = (xs + residual)?;

        // Pre-MLP norm + MLP
        let residual = &xs;
        let xs = self.pre_mlp_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_norm.forward(&xs)?;

        // Residual connection after MLP
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct Plamo3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Plamo3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Plamo3ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_model = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Plamo3DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_model.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            let emb_weights = embed_tokens.embeddings().clone();
            Linear::new(emb_weights, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
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

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr.engine_mut(layer_idx),
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}

impl crate::engine::ModelForward for Plamo3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        Plamo3ForCausalLM::forward(
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
        let mut outputs = Vec::with_capacity(sequences.len());
        for (i, seq) in sequences.iter().enumerate() {
            let token = input_ids.narrow(0, i, 1)?;
            let block_table = BlockTable::from_block_ids(seq.block_ids.clone(), seq.seqlen_offset);
            let logits = self.forward(
                &token,
                seq.seqlen_offset,
                kv_cache_mgr,
                &block_table,
                &seq.slot_mapping,
            )?;
            outputs.push(logits);
        }
        Tensor::cat(&outputs, 0)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["Plamo3ForCausalLM".to_string()],
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
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache(cfg: &ModelConfig) -> (KVCacheManager, BlockTable) {
        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: Device::Cpu,
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let bt = BlockTable::new(cache_config.block_size);
        (mgr, bt)
    }

    #[test]
    fn test_plamo3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Plamo3ForCausalLM should construct: {:?}",
            model.err()
        );
        let model = model.expect("model");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_plamo3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);
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
    fn test_plamo3_prefill_then_decode() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
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
        let next = Tensor::zeros((1, 1), DType::U32, &device).expect("next");
        let logits = model
            .forward(&next, 3, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    #[test]
    fn test_plamo3_model_forward_trait() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);
        let input = Tensor::zeros((1, 3), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = ModelForward::forward(
            &model,
            &input,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward");
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }

    #[test]
    fn test_plamo3_device() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb).expect("build model");
        assert!(matches!(model.device(), Device::Cpu));
    }

    #[test]
    fn test_plamo3_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "Plamo3ForCausalLM should construct with untied embeddings"
        );
    }

    #[test]
    fn test_plamo3_four_norm_residual_pattern() {
        // Verify the 4-norm pattern produces valid outputs
        // (if any norm was misconfigured, output would be NaN or wrong shape)
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Plamo3ForCausalLM::new(&cfg, vb).expect("build model");

        let (mut kv_cache_mgr, mut block_table) = create_cache(&cfg);
        let input = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let logits = model
            .forward(&input, 0, &mut kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward");

        // Verify no NaN values
        let vals: Vec<f32> = logits.flatten_all().expect("flat").to_vec1().expect("vec");
        for v in &vals {
            assert!(v.is_finite(), "logit should be finite, got {}", v);
        }
    }
}
