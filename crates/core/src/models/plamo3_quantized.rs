//! Quantized PLaMo-3 model implementation (Preferred Networks).
//!
//! PLaMo-3-specific features preserved in the quantized path:
//! - **4-norm residual pattern**: pre_mixer_norm → attn → post_mixer_norm → pre_mlp_norm → MLP → post_mlp_norm
//! - **QK normalization**: Per-head RMSNorm on Q and K (after reshape to heads)
//! - **SwiGLU MLP**: gate_up_proj.gate + gate_up_proj.up + down_proj
//! - **No bias** on projections
//! - Weight prefix: model.layers.layers.X (note double layers)

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, rms_norm, RmsNorm, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Tied Embedding Head ─────────────────────────────────────────────────────

struct TiedEmbeddingHead {
    weight: Tensor,
}

impl TiedEmbeddingHead {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = match x.dims().len() {
            3 => self.weight.broadcast_left(x.dim(0)?)?,
            _ => self.weight.clone(),
        };
        x.matmul(&w.t()?)
    }

    fn load_weights(&mut self, _weights: &HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }

    fn weight_dtype(&self) -> DType {
        self.weight.dtype()
    }

    fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }

    fn has_bias(&self) -> bool {
        false
    }
}

// ─── Quantized SwiGLU MLP ───────────────────────────────────────────────────
// PLaMo3 uses gate_up_proj.gate / gate_up_proj.up naming

struct QuantizedPlamo3Mlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedPlamo3Mlp {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{}.gate_up_proj.gate", vb.prefix()),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let up_proj = loader.load_linear(
            &format!("{}.gate_up_proj.up", vb.prefix()),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{}.down_proj", vb.prefix()),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = (gate.silu()? * up)?;
        self.down_proj.forward(&hidden)
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

struct QuantizedPlamo3Attention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedPlamo3Attention {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = loader.load_linear(
            &format!("{}.q_proj", vb.prefix()),
            cfg.hidden_size,
            num_heads * head_dim,
            false,
        )?;
        let k_proj = loader.load_linear(
            &format!("{}.k_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let v_proj = loader.load_linear(
            &format!("{}.v_proj", vb.prefix()),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            false,
        )?;
        let o_proj = loader.load_linear(
            &format!("{}.o_proj", vb.prefix()),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

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

        // Per-head QK normalization (after reshape to heads)
        let q = crate::layers::apply_per_head_norm(&q, &self.q_norm)?;
        let k = crate::layers::apply_per_head_norm(&k, &self.k_norm)?;

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

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = crate::layers::apply_per_head_norm(&q, &self.q_norm)?;
        let k = crate::layers::apply_per_head_norm(&k, &self.k_norm)?;

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
        self.o_proj.forward(&attn_output)
    }
}

// ─── Decoder Layer (4-norm residual pattern) ─────────────────────────────────

struct QuantizedPlamo3DecoderLayer {
    mixer: QuantizedPlamo3Attention,
    mlp: QuantizedPlamo3Mlp,
    pre_mixer_norm: RmsNorm,
    post_mixer_norm: RmsNorm,
    pre_mlp_norm: RmsNorm,
    post_mlp_norm: RmsNorm,
}

impl QuantizedPlamo3DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder, loader: &dyn QuantizedWeightLoader) -> Result<Self> {
        let mixer = QuantizedPlamo3Attention::new(cfg, vb.pp("mixer"), loader)?;
        let mlp = QuantizedPlamo3Mlp::new(cfg, vb.pp("mlp"), loader)?;

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
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        // Pre-mixer norm + attention + post-mixer norm + residual
        let residual = xs;
        let xs = self.pre_mixer_norm.forward(xs)?;
        let xs = self.mixer.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = self.post_mixer_norm.forward(&xs)?;
        let xs = (xs + residual)?;

        // Pre-MLP norm + MLP + post-MLP norm + residual
        let residual = &xs;
        let xs = self.pre_mlp_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_norm.forward(&xs)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.pre_mixer_norm.forward(xs)?;
        let xs =
            self.mixer
                .forward_decode_batch(&xs, sequences, kv_cache_mgr.engine_mut(layer_idx))?;
        let xs = self.post_mixer_norm.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.pre_mlp_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_norm.forward(&xs)?;
        residual + xs
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct QuantizedPlamo3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedPlamo3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedPlamo3ForCausalLM {
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        // PLaMo3 uses double "layers" prefix: model.layers.layers.X
        let vb_l = vb_m.pp("layers").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedPlamo3DecoderLayer::new(cfg, vb_l.pp(i), loader)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head: Box<dyn QuantizedLinear> = if cfg.tie_word_embeddings {
            Box::new(TiedEmbeddingHead::new(embed_tokens.embeddings().clone()))
        } else {
            loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
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
}

impl crate::engine::ModelForward for QuantizedPlamo3ForCausalLM {
    fn forward(
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
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }
        xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

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

    #[test]
    fn test_quantized_plamo3_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedPlamo3ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedPlamo3ForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), 2);
    }

    #[test]
    fn test_quantized_plamo3_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedPlamo3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let seq_len = 5;
        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).expect("input");

        kv_cache_mgr
            .allocate_for_request(&mut block_table, seq_len)
            .expect("allocate");

        let slot_mapping: Vec<usize> = (0..seq_len).collect();
        let logits = crate::engine::ModelForward::forward(
            &model,
            &input_ids,
            0,
            &mut kv_cache_mgr,
            &block_table,
            &slot_mapping,
        )
        .expect("forward");

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_quantized_plamo3_untied_embeddings() {
        let mut cfg = test_config();
        cfg.tie_word_embeddings = false;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedPlamo3ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "Untied should construct: {:?}", model.err());
    }

    #[test]
    fn test_quantized_plamo3_gptq_loader() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig {
            method: crate::quantization::QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            ..Default::default()
        };
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedPlamo3ForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(model.is_ok(), "GPTQ should construct: {:?}", model.err());
    }
}
