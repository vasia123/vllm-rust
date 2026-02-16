//! Qwen3-VL-MoE vision-language model implementation.
//!
//! Combines the Qwen3-VL vision transformer with a Mixture of Experts language
//! model backbone. Uses MRoPE for 3D positional encoding and sparse MoE blocks
//! for the feed-forward layers.
//!
//! Architecture:
//! - Vision encoder: Qwen3VisionTransformer (same as Qwen3-VL)
//! - Language model: Qwen3-style attention with MRoPE + MoE sparse FFN
//!
//! Reference: Qwen3-VL-MoE (https://github.com/QwenLM/Qwen3-VL)

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{rms_norm, Linear, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{apply_per_head_norm, causal_mask, paged_attention};
use crate::moe::{MoERouter, RouterConfig, ScoringFunc, TopKRouter};
use crate::multimodal::MultimodalInputs;

use super::qwen2_vl::MRoPE;
use super::qwen3_vl::{Qwen3VLConfig, Qwen3VisionTransformer};

// ─── MoE Expert ──────────────────────────────────────────────────────────────

struct MoEExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MoEExpert {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
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

// ─── Sparse MoE Block ───────────────────────────────────────────────────────

struct SparseMoeBlock {
    router: TopKRouter,
    experts: Vec<MoEExpert>,
    num_experts: usize,
}

impl SparseMoeBlock {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_experts().unwrap_or(64);
        let top_k = cfg.num_experts_per_tok().unwrap_or(8);
        let hidden_size = cfg.hidden_size;
        let moe_intermediate_size = cfg.moe_intermediate_size().unwrap_or(cfg.intermediate_size);
        let renormalize = cfg.norm_topk_prob();

        let router_config = RouterConfig {
            hidden_size,
            num_experts,
            top_k,
            renormalize,
            scoring_func: ScoringFunc::Softmax,
            ..Default::default()
        };
        let router = TopKRouter::new(router_config, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..num_experts {
            experts.push(MoEExpert::new(
                hidden_size,
                moe_intermediate_size,
                vb_experts.pp(i),
            )?);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_shape = xs.dims().to_vec();
        let hidden_dim = *orig_shape.last().unwrap();
        let xs_2d = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_2d.dim(0)?;

        let (routing_weights, selected_experts) = self.router.route(&xs_2d)?;
        let mut final_output = Tensor::zeros((num_tokens, hidden_dim), xs.dtype(), xs.device())?;

        for token_idx in 0..num_tokens {
            let token_input = xs_2d.narrow(0, token_idx, 1)?;
            let token_weights = routing_weights.narrow(0, token_idx, 1)?;
            let token_experts = selected_experts.narrow(0, token_idx, 1)?;

            let expert_indices: Vec<u32> = token_experts.flatten_all()?.to_vec1()?;
            let weights: Vec<f32> = token_weights
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1()?;

            let mut token_output = Tensor::zeros((1, hidden_dim), xs.dtype(), xs.device())?;
            for (k, &expert_idx) in expert_indices.iter().enumerate() {
                let expert_idx = expert_idx as usize;
                if expert_idx < self.num_experts {
                    let expert_out = self.experts[expert_idx].forward(&token_input)?;
                    let weighted = expert_out.affine(weights[k] as f64, 0.0)?;
                    token_output = (token_output + weighted)?;
                }
            }

            let indices = Tensor::new(&[token_idx as u32], xs.device())?;
            final_output = final_output.index_add(&indices, &token_output, 0)?;
        }

        final_output.reshape(orig_shape)
    }
}

// ─── MLP Variant ────────────────────────────────────────────────────────────

enum MlpVariant {
    Dense {
        gate_proj: Linear,
        up_proj: Linear,
        down_proj: Linear,
    },
    Sparse(SparseMoeBlock),
}

impl MlpVariant {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            MlpVariant::Dense {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = candle_nn::ops::silu(&gate_proj.forward(xs)?)?;
                let up = up_proj.forward(xs)?;
                down_proj.forward(&(gate * up)?)
            }
            MlpVariant::Sparse(moe) => moe.forward(xs),
        }
    }
}

// ─── Attention (MRoPE + QK Norm) ────────────────────────────────────────────

struct Qwen3VLMoeAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    mrope: MRoPE,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen3VLMoeAttention {
    fn new(cfg: &ModelConfig, mrope_section: &[usize], vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj =
            candle_nn::linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let mrope = MRoPE::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            mrope_section.to_vec(),
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
            mrope,
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
        position_ids: Option<&Tensor>,
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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        let (q, k) = if let Some(pos_ids) = position_ids {
            self.mrope.apply(&q, &k, pos_ids)?
        } else {
            self.mrope.apply_scalar(&q, &k, seqlen_offset)?
        };

        paged_attention(
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
        )
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

        let q = apply_per_head_norm(&q, &self.q_norm)?;
        let k = apply_per_head_norm(&k, &self.k_norm)?;

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            let positions: Vec<usize> = sequences.iter().map(|s| s.seqlen_offset).collect();
            let (q, k) = self.mrope.apply_varlen(&q, &k, &positions)?;

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

            self.o_proj.forward(&attn_output)?.unsqueeze(1)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let mut outputs = Vec::with_capacity(batch_size);
            for (i, seq) in sequences.iter().enumerate() {
                let q_i = q.narrow(0, i, 1)?;
                let k_i = k.narrow(0, i, 1)?;
                let v_i = v.narrow(0, i, 1)?;

                let (q_i, k_i) = self.mrope.apply_scalar(&q_i, &k_i, seq.seqlen_offset)?;

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
}

// ─── Decoder Layer ──────────────────────────────────────────────────────────

struct Qwen3VLMoeDecoderLayer {
    self_attn: Qwen3VLMoeAttention,
    mlp: MlpVariant,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3VLMoeDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        mrope_section: &[usize],
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Qwen3VLMoeAttention::new(cfg, mrope_section, vb.pp("self_attn"))?;

        let decoder_sparse_step = cfg.decoder_sparse_step().unwrap_or(1);
        let mlp_only_layers = cfg.mlp_only_layers();
        let num_experts = cfg.num_experts().unwrap_or(0);

        let is_moe_layer = !mlp_only_layers.contains(&layer_idx)
            && num_experts > 0
            && (layer_idx + 1).is_multiple_of(decoder_sparse_step);

        let mlp_vb = vb.pp("mlp");
        let mlp = if is_moe_layer {
            MlpVariant::Sparse(SparseMoeBlock::new(cfg, mlp_vb)?)
        } else {
            let gate_proj = candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp_vb.pp("gate_proj"),
            )?;
            let up_proj = candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                mlp_vb.pp("up_proj"),
            )?;
            let down_proj = candle_nn::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                mlp_vb.pp("down_proj"),
            )?;
            MlpVariant::Dense {
                gate_proj,
                up_proj,
                down_proj,
            }
        };

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
        position_ids: Option<&Tensor>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            position_ids,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
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
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        residual + xs
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Qwen3-VL-MoE model for conditional generation.
///
/// Combines the Qwen3 vision transformer with a Mixture of Experts language
/// model. Uses MRoPE for 3D positional encoding (temporal/height/width) and
/// sparse MoE blocks for the feed-forward layers.
pub struct Qwen3VLMoeForConditionalGeneration {
    #[allow(dead_code)]
    visual: Qwen3VisionTransformer,
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3VLMoeDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    config: Qwen3VLConfig,
    device: Device,
    dtype: DType,
}

impl Qwen3VLMoeForConditionalGeneration {
    pub fn new(cfg: &Qwen3VLConfig, vb: VarBuilder) -> Result<Self> {
        let visual = Qwen3VisionTransformer::new(&cfg.vision_config, vb.pp("visual"))?;

        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            cfg.model_config.vocab_size,
            cfg.model_config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(cfg.model_config.num_hidden_layers);
        for i in 0..cfg.model_config.num_hidden_layers {
            layers.push(Qwen3VLMoeDecoderLayer::new(
                &cfg.model_config,
                &cfg.mrope_section,
                i,
                vb_m.pp("layers").pp(i),
            )?);
        }

        let norm = rms_norm(
            cfg.model_config.hidden_size,
            cfg.model_config.rms_norm_eps,
            vb_m.pp("norm"),
        )?;

        let lm_head = if cfg.model_config.tie_word_embeddings {
            let emb_w = vb_m.pp("embed_tokens").get(
                (cfg.model_config.vocab_size, cfg.model_config.hidden_size),
                "weight",
            )?;
            Linear::new(emb_w, None)
        } else {
            candle_nn::linear_no_bias(
                cfg.model_config.hidden_size,
                cfg.model_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };

        Ok(Self {
            visual,
            embed_tokens,
            layers,
            norm,
            lm_head,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn from_model_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let qwen3vl_cfg = Qwen3VLConfig::from_model_config(cfg);
        Self::new(&qwen3vl_cfg, vb)
    }

    fn compute_position_ids(
        &self,
        input_ids: &[u32],
        mm_inputs: Option<&MultimodalInputs>,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let mut positions = vec![vec![0u32; seq_len]; 3];

        if mm_inputs.is_none() || !mm_inputs.is_some_and(|m| m.has_images()) {
            for (i, pos) in positions[0].iter_mut().enumerate() {
                *pos = i as u32;
            }
            positions[1] = positions[0].clone();
            positions[2] = positions[0].clone();
        } else {
            let merge = self.config.vision_config.spatial_merge_size;
            let mut pos = 0u32;
            let mut i = 0;
            while i < seq_len {
                if input_ids[i] == self.config.image_token_id {
                    let grid_info = mm_inputs.and_then(|m| self.find_image_grid_at(m, i));
                    let (grid_h, grid_w) = grid_info.unwrap_or((1, 1));
                    let merged_h = grid_h / merge;
                    let merged_w = grid_w / merge;
                    let num_image_tokens = merged_h * merged_w;

                    for t in 0..num_image_tokens {
                        if i + t >= seq_len {
                            break;
                        }
                        let h = t / merged_w;
                        let w = t % merged_w;
                        positions[0][i + t] = pos;
                        positions[1][i + t] = pos + h as u32;
                        positions[2][i + t] = pos + w as u32;
                    }
                    let max_dim = merged_h.max(merged_w) as u32;
                    pos += max_dim;
                    i += num_image_tokens;
                } else {
                    positions[0][i] = pos;
                    positions[1][i] = pos;
                    positions[2][i] = pos;
                    pos += 1;
                    i += 1;
                }
            }
        }

        let flat: Vec<u32> = positions.into_iter().flatten().collect();
        Tensor::from_vec(flat, (3, seq_len), &self.device)
    }

    fn find_image_grid_at(
        &self,
        mm_inputs: &MultimodalInputs,
        pos: usize,
    ) -> Option<(usize, usize)> {
        for (img_pos, processed) in &mm_inputs.image_embeddings {
            if *img_pos == pos {
                return processed.grid_size;
            }
        }
        None
    }

    fn merge_multimodal(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _hidden) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed) in &mm_inputs.image_embeddings {
            let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in emb_vec.iter().enumerate() {
                let target = start_pos + i;
                if target < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }
}

impl crate::engine::ModelForward for Qwen3VLMoeForConditionalGeneration {
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
            Some(causal_mask(
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
                None,
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
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
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;

        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
        let position_ids = self.compute_position_ids(&input_ids_vec, multimodal_inputs)?;

        let text_embeddings = self.embed_tokens.forward(input_ids)?;
        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            self.merge_multimodal(&text_embeddings, mm_inputs)?
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                Some(&position_ids),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            serde_json::json!({
                "depth": 2,
                "hidden_size": 64,
                "num_heads": 4,
                "intermediate_size": 128,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 64
            }),
        );
        extra.insert(
            "rope_scaling".to_string(),
            serde_json::json!({ "mrope_section": [2, 3, 3] }),
        );
        extra.insert("image_token_id".to_string(), serde_json::json!(151655));
        // MoE config
        extra.insert("num_experts".to_string(), serde_json::json!(4));
        extra.insert("num_experts_per_tok".to_string(), serde_json::json!(2));
        extra.insert("moe_intermediate_size".to_string(), serde_json::json!(64));
        extra.insert("decoder_sparse_step".to_string(), serde_json::json!(1));
        extra.insert("norm_topk_prob".to_string(), serde_json::json!(true));

        ModelConfig {
            architectures: vec!["Qwen3VLMoeForConditionalGeneration".to_string()],
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
            rope_theta: 1000000.0,
            tie_word_embeddings: true,
            bos_token_id: 151643,
            eos_token_id: 151645,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn test_vlmoe_config() -> Qwen3VLConfig {
        Qwen3VLConfig::from_model_config(&test_model_config())
    }

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_vlmoe_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_from_model_config() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::from_model_config(&cfg, vb);
        assert!(model.is_ok(), "from_model_config failed: {:?}", model.err());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_vlmoe_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_vlmoe_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_multimodal_forward_text_only() {
        let device = Device::Cpu;
        let cfg = test_vlmoe_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward_multimodal(
                &input_ids,
                None,
                0,
                &mut kv_cache,
                &block_table,
                &slot_mapping,
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.model_config.vocab_size]);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_vlmoe_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Qwen3VLMoeForConditionalGeneration::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg.model_config, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        // Prefill
        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.model_config.vocab_size]);
        block_table.advance(3);

        // Decode
        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.model_config.vocab_size]);
    }
}
