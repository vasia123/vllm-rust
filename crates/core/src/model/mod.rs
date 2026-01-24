mod attention;
mod mlp;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::kv_cache::{BlockTable, KVCacheManager};

use self::attention::Qwen3Attention;
use self::mlp::Qwen3Mlp;

struct Qwen3DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Qwen3Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
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
        kv_cache_mgr: &KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .post_attention_layernorm
            .forward(&xs)?
            .apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Qwen3ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    device: Device,
    dtype: DType,
}

impl Qwen3ForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
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
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, seqlen_offset)?)
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
        let xs = self.norm.forward(&xs)?;
        let logits = xs.apply(&self.lm_head)?;
        Ok(logits)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn causal_mask(&self, seq_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let total_len = seq_len + seqlen_offset;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j > i + seqlen_offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, total_len), &self.device)?;
        mask.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for Qwen3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::CacheConfig;
    use crate::loader;

    #[test]
    #[ignore] // requires downloaded model
    fn forward_pass_produces_correct_logits_shape() {
        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let device = Device::Cpu;
        let vb = loader::load_weights(&files.weights, DType::F32, &device).expect("load weights");
        let model = Qwen3ForCausalLM::new(&files.config, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 32,
            num_layers: files.config.num_hidden_layers,
            num_kv_heads: files.config.num_key_value_heads,
            head_dim: files.config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).expect("input tensor");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 5)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 5);
        let logits = model
            .forward(&input_ids, 0, &kv_cache_mgr, &block_table, &slot_mapping)
            .expect("forward pass");
        block_table.advance(5);

        assert_eq!(logits.dims(), &[1, 5, files.config.vocab_size]);
    }

    #[test]
    #[ignore] // requires downloaded model
    fn paged_cache_decode_step() {
        let files = loader::fetch_model("Qwen/Qwen3-0.6B").expect("fetch model");
        let device = Device::Cpu;
        let vb = loader::load_weights(&files.weights, DType::F32, &device).expect("load weights");
        let model = Qwen3ForCausalLM::new(&files.config, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 32,
            num_layers: files.config.num_hidden_layers,
            num_kv_heads: files.config.num_key_value_heads,
            head_dim: files.config.head_dim,
            dtype: DType::F32,
            device: device.clone(),
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        // Prefill
        let prompt = Tensor::new(&[[1u32, 2, 3]], &device).expect("prompt");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .expect("allocate prefill");
        let slot_mapping = block_table.slot_mapping(0, 3);
        let _ = model
            .forward(&prompt, 0, &kv_cache_mgr, &block_table, &slot_mapping)
            .expect("prefill");
        block_table.advance(3);

        // Decode step
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 1)
            .expect("allocate decode");
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::new(&[[4u32]], &device).expect("next token");
        let logits = model
            .forward(&next_token, 3, &kv_cache_mgr, &block_table, &slot_mapping)
            .expect("decode step");
        block_table.advance(1);

        assert_eq!(logits.dims(), &[1, 1, files.config.vocab_size]);
    }
}
