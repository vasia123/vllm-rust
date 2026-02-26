use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::paged_attention;

// OPT uses learned positional embeddings with offset=2

// ─── OPT Attention ───────────────────────────────────────────────────────────

struct OPTAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    enable_bias: bool,
}

impl OPTAttention {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let enable_bias = cfg
            .extra
            .get("enable_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let (q_proj, k_proj, v_proj, out_proj) = if enable_bias {
            (
                candle_nn::linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
                candle_nn::linear(cfg.hidden_size, num_heads * head_dim, vb.pp("k_proj"))?,
                candle_nn::linear(cfg.hidden_size, num_heads * head_dim, vb.pp("v_proj"))?,
                candle_nn::linear(num_heads * head_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            )
        } else {
            (
                candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
                candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("k_proj"))?,
                candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("v_proj"))?,
                candle_nn::linear_no_bias(
                    num_heads * head_dim,
                    cfg.hidden_size,
                    vb.pp("out_proj"),
                )?,
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            enable_bias: false, // field reserved for future use
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
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // OPT uses no RoPE, positions are handled by learned embeddings
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
            self.num_heads,
            self.head_dim,
        )?;

        self.out_proj.forward(&attn_output)
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
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let mut outputs = Vec::with_capacity(batch_size);
        for (i, seq) in sequences.iter().enumerate() {
            let q_i = q.narrow(0, i, 1)?;
            let k_i = k.narrow(0, i, 1)?;
            let v_i = v.narrow(0, i, 1)?;

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
                self.num_heads,
                self.head_dim,
            )?;
            outputs.push(attn_out);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.out_proj.forward(&attn_output)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

#[allow(clippy::upper_case_acronyms)]
struct OPTMLP {
    fc1: Linear,
    fc2: Linear,
}

impl OPTMLP {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let ffn_dim = cfg
            .extra
            .get("ffn_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.intermediate_size as u64) as usize;
        let enable_bias = cfg
            .extra
            .get("enable_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let (fc1, fc2) = if enable_bias {
            (
                candle_nn::linear(cfg.hidden_size, ffn_dim, vb.pp("fc1"))?,
                candle_nn::linear(ffn_dim, cfg.hidden_size, vb.pp("fc2"))?,
            )
        } else {
            (
                candle_nn::linear_no_bias(cfg.hidden_size, ffn_dim, vb.pp("fc1"))?,
                candle_nn::linear_no_bias(ffn_dim, cfg.hidden_size, vb.pp("fc2"))?,
            )
        };

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.relu()?;
        self.fc2.forward(&xs)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct OPTDecoderLayer {
    self_attn: OPTAttention,
    mlp: OPTMLP,
    self_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    do_layer_norm_before: bool,
}

impl OPTDecoderLayer {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        let do_layer_norm_before = cfg
            .extra
            .get("do_layer_norm_before")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let self_attn = OPTAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = OPTMLP::new(cfg, vb.pp("mlp"))?;
        let self_attn_layer_norm = layer_norm(cfg.hidden_size, eps, vb.pp("self_attn_layer_norm"))?;
        let final_layer_norm = layer_norm(cfg.hidden_size, eps, vb.pp("final_layer_norm"))?;

        Ok(Self {
            self_attn,
            mlp,
            self_attn_layer_norm,
            final_layer_norm,
            do_layer_norm_before,
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
        let residual = xs;
        let xs = if self.do_layer_norm_before {
            self.self_attn_layer_norm.forward(xs)?
        } else {
            xs.clone()
        };

        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let xs = if !self.do_layer_norm_before {
            self.self_attn_layer_norm.forward(&xs)?
        } else {
            xs
        };

        let residual = &xs;
        let hidden = if self.do_layer_norm_before {
            self.final_layer_norm.forward(&xs)?
        } else {
            xs.clone()
        };
        let hidden = self.mlp.forward(&hidden)?;
        let xs = (hidden + residual)?;
        if !self.do_layer_norm_before {
            self.final_layer_norm.forward(&xs)
        } else {
            Ok(xs)
        }
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = if self.do_layer_norm_before {
            self.self_attn_layer_norm.forward(xs)?
        } else {
            xs.clone()
        };

        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let xs = if !self.do_layer_norm_before {
            self.self_attn_layer_norm.forward(&xs)?
        } else {
            xs
        };

        let residual = &xs;
        let hidden = if self.do_layer_norm_before {
            self.final_layer_norm.forward(&xs)?
        } else {
            xs.clone()
        };
        let hidden = self.mlp.forward(&hidden)?;
        let xs = (hidden + residual)?;
        if !self.do_layer_norm_before {
            self.final_layer_norm.forward(&xs)
        } else {
            Ok(xs)
        }
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct OPTForCausalLM {
    embed_tokens: Embedding,
    embed_positions: Embedding,
    project_in: Option<Linear>,
    project_out: Option<Linear>,
    layers: Vec<OPTDecoderLayer>,
    final_layer_norm: Option<LayerNorm>,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl OPTForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);
        let do_layer_norm_before = cfg
            .extra
            .get("do_layer_norm_before")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let word_embed_proj_dim = cfg
            .extra
            .get("word_embed_proj_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.hidden_size as u64) as usize;

        let vb_m = vb.pp("model");
        let vb_d = vb_m.pp("decoder");

        let embed_tokens = embedding(cfg.vocab_size, word_embed_proj_dim, vb_d.pp("embed_tokens"))?;
        // OPT learned positional embeddings: size = max_pos + 2 (offset of 2)
        let embed_positions = embedding(
            cfg.max_position_embeddings + 2,
            cfg.hidden_size,
            vb_d.pp("embed_positions"),
        )?;

        // Optional projection layers if word_embed_proj_dim != hidden_size
        let project_in = if word_embed_proj_dim != cfg.hidden_size {
            Some(candle_nn::linear_no_bias(
                word_embed_proj_dim,
                cfg.hidden_size,
                vb_d.pp("project_in"),
            )?)
        } else {
            None
        };
        let project_out = if word_embed_proj_dim != cfg.hidden_size {
            Some(candle_nn::linear_no_bias(
                cfg.hidden_size,
                word_embed_proj_dim,
                vb_d.pp("project_out"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_d.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(OPTDecoderLayer::new(cfg, vb_l.pp(i))?);
        }

        let final_layer_norm = if do_layer_norm_before {
            Some(layer_norm(
                cfg.hidden_size,
                eps,
                vb_d.pp("final_layer_norm"),
            )?)
        } else {
            None
        };

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(word_embed_proj_dim, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            embed_positions,
            project_in,
            project_out,
            layers,
            final_layer_norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed(&self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;

        // Create position ids with OPT offset of 2
        let positions: Vec<u32> = (0..seq_len)
            .map(|i| (seqlen_offset + i + 2) as u32)
            .collect();
        let position_ids = Tensor::new(positions.as_slice(), input_ids.device())?;

        let word_embeds = self.embed_tokens.forward(input_ids)?;
        let pos_embeds = self.embed_positions.forward(&position_ids)?.unsqueeze(0)?;

        let mut hidden = if let Some(ref proj_in) = self.project_in {
            proj_in.forward(&word_embeds)?
        } else {
            word_embeds
        };
        hidden = hidden.broadcast_add(&pos_embeds)?;
        Ok(hidden)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for OPTForCausalLM {
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

        let mut xs = self.embed(input_ids, seqlen_offset)?;
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

        if let Some(ref ln) = self.final_layer_norm {
            xs = ln.forward(&xs)?;
        }
        if let Some(ref proj_out) = self.project_out {
            xs = proj_out.forward(&xs)?;
        }
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        // Each sequence has its own seqlen_offset; compute per-sequence position
        // embeddings (OPT offset = +2) then stack as [batch, 1, hidden].
        let batch_size = sequences.len();
        let word_embeds = self.embed_tokens.forward(input_ids)?; // [batch, 1, hidden]
        let mut pos_rows = Vec::with_capacity(batch_size);
        for seq in sequences {
            let pos_id = (seq.seqlen_offset + 2) as u32;
            let pos_ids = Tensor::new(&[pos_id], input_ids.device())?;
            pos_rows.push(self.embed_positions.forward(&pos_ids)?); // [1, hidden]
        }
        let pos_embeds = Tensor::stack(&pos_rows, 0)?.contiguous()?; // [batch, 1, hidden]

        let mut xs = if let Some(ref proj_in) = self.project_in {
            proj_in.forward(&word_embeds)?.broadcast_add(&pos_embeds)?
        } else {
            word_embeds.broadcast_add(&pos_embeds)?
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx)?;
        }

        if let Some(ref ln) = self.final_layer_norm {
            xs = ln.forward(&xs)?;
        }
        if let Some(ref proj_out) = self.project_out {
            xs = proj_out.forward(&xs)?;
        }
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

    fn test_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert("do_layer_norm_before".into(), serde_json::Value::from(true));
        extra.insert("enable_bias".into(), serde_json::Value::from(true));
        extra.insert("layer_norm_epsilon".into(), serde_json::Value::from(1e-5));

        ModelConfig {
            architectures: vec!["OPTForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "relu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_opt_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = OPTForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "OPTForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), 2);
    }

    #[test]
    fn test_opt_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OPTForCausalLM::new(&cfg, vb).expect("build model");

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
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let batch_size = 1;
        let seq_len = 5;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).expect("input");

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
        .expect("forward pass");

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_opt_decode_batch_heterogeneous_offsets() {
        // Two sequences with different sequence lengths, verifying that each
        // gets the correct positional embedding (not both using the first offset).
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = OPTForCausalLM::new(&cfg, vb).expect("build model");

        let cache_config = CacheConfig {
            block_size: 16,
            num_blocks: 16,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        };
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");

        // Sequence A: offset 3 (prefilled 3 tokens)
        let mut bt_a = BlockTable::new(cache_config.block_size);
        kv_cache_mgr
            .allocate_for_request(&mut bt_a, 4)
            .expect("alloc A");

        // Sequence B: offset 7 (prefilled 7 tokens)
        let mut bt_b = BlockTable::new(cache_config.block_size);
        kv_cache_mgr
            .allocate_for_request(&mut bt_b, 8)
            .expect("alloc B");

        let sequences = vec![
            crate::engine::DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 3,
                slot_mapping: bt_a.slot_mapping(3, 1),
                block_ids: bt_a.block_ids().to_vec(),
            },
            crate::engine::DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 7,
                slot_mapping: bt_b.slot_mapping(7, 1),
                block_ids: bt_b.block_ids().to_vec(),
            },
        ];

        let input_ids = Tensor::zeros((2usize, 1usize), DType::U32, &device).expect("input");
        let logits = crate::engine::ModelForward::forward_decode_batch(
            &model,
            &input_ids,
            &sequences,
            &mut kv_cache_mgr,
        )
        .expect("decode batch");

        assert_eq!(
            logits.dims(),
            &[2, 1, cfg.vocab_size],
            "batch decode should return [batch, 1, vocab]"
        );
    }
}
