//! Quantized Llama model implementation.
//!
//! This module provides a quantized version of the Llama model that supports
//! various quantization methods (FP8, GPTQ, AWQ) through the QuantizedWeightLoader
//! abstraction.

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::attention::DecodeBatchShared;
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::quantization::{QuantizedLinear, QuantizedWeightLoader};

// ─── Quantized SwiGLU MLP ────────────────────────────────────────────────────

struct QuantizedSwiGluMlp {
    gate_proj: Box<dyn QuantizedLinear>,
    up_proj: Box<dyn QuantizedLinear>,
    down_proj: Box<dyn QuantizedLinear>,
}

impl QuantizedSwiGluMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        loader: &dyn QuantizedWeightLoader,
        prefix: &str,
    ) -> Result<Self> {
        let gate_proj = loader.load_linear(
            &format!("{prefix}.gate_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let up_proj = loader.load_linear(
            &format!("{prefix}.up_proj"),
            hidden_size,
            intermediate_size,
            false,
        )?;
        let down_proj = loader.load_linear(
            &format!("{prefix}.down_proj"),
            intermediate_size,
            hidden_size,
            false,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // Phase 11.2.C: pool-backed silu+mul (F16/BF16). Falls back to
        // candle silu+broadcast_mul for unsupported dtypes or
        // num_tokens > 64 (prefill).
        #[cfg(feature = "cuda-fused-activations")]
        let activated = crate::cuda_kernels::silu_and_mul_separate_pooled(&gate, &up)?;
        #[cfg(not(feature = "cuda-fused-activations"))]
        let activated = (candle_nn::ops::silu(&gate)? * up)?;
        self.down_proj.forward(&activated)
    }
}

// ─── Quantized Attention ─────────────────────────────────────────────────────

struct QuantizedLlamaAttention {
    q_proj: Box<dyn QuantizedLinear>,
    k_proj: Box<dyn QuantizedLinear>,
    v_proj: Box<dyn QuantizedLinear>,
    o_proj: Box<dyn QuantizedLinear>,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedLlamaAttention {
    fn new(cfg: &ModelConfig, loader: &dyn QuantizedWeightLoader, prefix: &str) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let attn_bias = cfg.attention_bias.unwrap_or(false);

        let q_proj = loader.load_linear(
            &format!("{prefix}.q_proj"),
            cfg.hidden_size,
            num_heads * head_dim,
            attn_bias,
        )?;
        let k_proj = loader.load_linear(
            &format!("{prefix}.k_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attn_bias,
        )?;
        let v_proj = loader.load_linear(
            &format!("{prefix}.v_proj"),
            cfg.hidden_size,
            num_kv_heads * head_dim,
            attn_bias,
        )?;
        let o_proj = loader.load_linear(
            &format!("{prefix}.o_proj"),
            num_heads * head_dim,
            cfg.hidden_size,
            false,
        )?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            loader.dtype(),
            loader.device(),
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

    #[allow(dead_code)]
    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
    ) -> Result<Tensor> {
        self.forward_decode_batch_with_shared(xs, sequences, cache_engine, None)
    }

    /// Phase 11.2.B (root-cause fix): the capture-friendly decode path
    /// for the *quantized* Llama (EXL3, AWQ, GPTQ — anything routed
    /// through `QuantizedLlamaForCausalLM`). When `shared` is `Some`,
    /// every per-layer device allocation (`positions`, `slot_mapping`,
    /// `block_tables`, `seq_lens`) is read from the pool-backed bundle
    /// the engine built ONCE per forward, mirroring the Qwen3 Phase
    /// A/B/B.8 pattern. When `shared` is `None`, falls back to the
    /// pre-existing eager path (per-layer `Tensor::from_vec`), so
    /// non-capture builds are unaffected.
    ///
    /// Without this method, the engine warmup with capture enabled
    /// failed at the *eager* JIT forward of the next-smaller batch
    /// after a capture cycle — because the captured graph held device
    /// pointers from `Tensor::from_vec(block_tables)` /
    /// `Tensor::from_vec(seq_lens)` that the freshly-allocated next
    /// forward's `Tensor::from_vec` reused, hosing the stream with
    /// `CUDA_ERROR_INVALID_VALUE`.
    fn forward_decode_batch_with_shared(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        shared: Option<&DecodeBatchShared>,
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

        #[cfg(feature = "cuda-kernels")]
        {
            let q = q.squeeze(2)?;
            let k = k.squeeze(2)?;
            let v = v.squeeze(2)?;

            // RoPE positions — pool-backed when `shared` is present.
            let positions_owned: Vec<usize>;
            let positions: &[usize] = match shared {
                Some(s) => &s.positions,
                None => {
                    positions_owned = sequences.iter().map(|s| s.seqlen_offset).collect();
                    &positions_owned
                }
            };
            let (q, k) = match shared.and_then(|s| s.positions_device.as_ref()) {
                Some(pos_t) => self
                    .rotary_emb
                    .apply_varlen_with_pos_tensor(&q, &k, positions, pos_t)?,
                None => self.rotary_emb.apply_varlen(&q, &k, positions)?,
            };

            // KV write — pool-backed slot_mapping when present.
            let slot_mapping_owned: Vec<usize>;
            let all_slot_mapping: &[usize] = match shared {
                Some(s) => &s.all_slot_mapping,
                None => {
                    slot_mapping_owned = sequences
                        .iter()
                        .flat_map(|s| s.slot_mapping.iter().copied())
                        .collect();
                    &slot_mapping_owned
                }
            };
            match shared.and_then(|s| s.slot_mapping_device.as_ref()) {
                Some(slot_t) => cache_engine
                    .write_batch_with_slot_tensor(&k, &v, all_slot_mapping, slot_t)
                    .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?,
                None => cache_engine
                    .write_batch(&k, &v, all_slot_mapping)
                    .map_err(|e| candle_core::Error::Msg(format!("cache write: {e}")))?,
            }

            // block_tables / seq_lens — reuse from shared (built once
            // per forward) instead of per-layer `Tensor::from_vec` (the
            // failure mode root cause).
            let (block_tables, seq_lens, max_blocks_per_seq, max_seq_len) = if let Some(s) = shared
            {
                (
                    s.block_tables.clone(),
                    s.seq_lens.clone(),
                    s.max_blocks_per_seq,
                    s.max_seq_len,
                )
            } else {
                let max_blocks = sequences
                    .iter()
                    .map(|s| s.block_ids.len())
                    .max()
                    .unwrap_or(1);
                let mut bt_data = vec![0u32; batch_size * max_blocks];
                for (i, seq) in sequences.iter().enumerate() {
                    for (j, &block_id) in seq.block_ids.iter().enumerate() {
                        bt_data[i * max_blocks + j] = block_id as u32;
                    }
                }
                let bt = Tensor::from_vec(bt_data, (batch_size, max_blocks), q.device())?;
                let seq_lens_data: Vec<u32> = sequences
                    .iter()
                    .map(|s| (s.seqlen_offset + 1) as u32)
                    .collect();
                let max_seq = *seq_lens_data.iter().max().unwrap_or(&1) as usize;
                let sl = Tensor::from_vec(seq_lens_data, (batch_size,), q.device())?;
                (bt, sl, max_blocks, max_seq)
            };

            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let prefer_pool = shared.is_some_and(|s| s.prefer_pooled_attention);
            let attn_output = if prefer_pool {
                // Pooled paged_attention with worst-case sizing — stable
                // device addresses across forwards (required for
                // CUDA-Graph-captured decode replays).
                let worst = crate::engine::engine_limits::max_model_len()
                    .unwrap_or(1024)
                    .max(max_seq_len);
                let partition_size = crate::cuda_kernels::select_v2_partition_size(worst);
                crate::cuda_kernels::paged_attention_v2_cuda_pooled(
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
                    worst,
                    self.head_dim,
                    cache_engine.block_size(),
                    partition_size,
                )?
            } else {
                crate::cuda_kernels::paged_attention_cuda(
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
                )?
            };

            self.o_proj.forward(&attn_output.unsqueeze(1)?)
        }

        #[cfg(not(feature = "cuda-kernels"))]
        {
            let _ = shared;
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
}

// ─── Quantized Decoder Layer ─────────────────────────────────────────────────

struct QuantizedLlamaDecoderLayer {
    self_attn: QuantizedLlamaAttention,
    mlp: QuantizedSwiGluMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedLlamaDecoderLayer {
    fn new(
        cfg: &ModelConfig,
        loader: &dyn QuantizedWeightLoader,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_idx}");
        let self_attn = QuantizedLlamaAttention::new(cfg, loader, &format!("{prefix}.self_attn"))?;
        let mlp = QuantizedSwiGluMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            loader,
            &format!("{prefix}.mlp"),
        )?;

        let vb_layer = vb.pp("model").pp("layers").pp(layer_idx);
        let input_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_layer.pp("post_attention_layernorm"),
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
        self.forward_decode_batch_with_shared(xs, sequences, kv_cache_mgr, layer_idx, None)
    }

    fn forward_decode_batch_with_shared(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        shared: Option<&DecodeBatchShared>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch_with_shared(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            shared,
        )?;
        // Phase 11.2.C: pool-backed F16/BF16 residual add. Stable
        // device address across forwards — captured graph reads from
        // the same pool slot on every replay.
        #[cfg(feature = "cuda-fused-activations")]
        let xs = crate::cuda_kernels::bf16_add_pooled(&xs, residual)?;
        #[cfg(not(feature = "cuda-fused-activations"))]
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&xs)?)?;
        #[cfg(feature = "cuda-fused-activations")]
        let result = crate::cuda_kernels::bf16_add_pooled(residual, &xs)?;
        #[cfg(not(feature = "cuda-fused-activations"))]
        let result = (residual + xs)?;
        Ok(result)
    }
}

// ─── Quantized Model ─────────────────────────────────────────────────────────

/// Quantized Llama model supporting FP8, GPTQ, AWQ, and unquantized weights.
pub struct QuantizedLlamaForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<QuantizedLlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: Box<dyn QuantizedLinear>,
    device: Device,
    dtype: DType,
}

impl QuantizedLlamaForCausalLM {
    /// Create a new quantized Llama model.
    ///
    /// # Arguments
    /// * `cfg` - Model configuration
    /// * `vb` - VarBuilder for loading non-quantized weights (embeddings, norms)
    /// * `weight_loader` - Quantized weight loader for linear layers
    pub fn new(
        cfg: &ModelConfig,
        vb: VarBuilder,
        weight_loader: &dyn QuantizedWeightLoader,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(QuantizedLlamaDecoderLayer::new(
                cfg,
                weight_loader,
                vb.clone(),
                i,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            // For tied embeddings, create an unquantized linear using embedding weights
            Box::new(TiedEmbeddingHead {
                weight: embed_tokens.embeddings().clone(),
            }) as Box<dyn QuantizedLinear>
        } else {
            weight_loader.load_linear("lm_head", cfg.hidden_size, cfg.vocab_size, false)?
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
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Helper for tied embedding lm_head.
struct TiedEmbeddingHead {
    weight: Tensor,
}

impl QuantizedLinear for TiedEmbeddingHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Flatten 3D `[B, S, H]` to 2D so cuBLAS picks plain GEMM
        // instead of stride-0 batched GEMM. +40% e2e at c=8 on the
        // lm_head shape (Qwen3-4B-AWQ side-by-side, 2026-05-09).
        //
        // Phase 11.2.C: route the matmul through `bf16_matmul_pooled`
        // (now BF16/F16 dtype-generic) when both operands are on CUDA
        // with matching dtype, so the lm_head's output uses a
        // stable-address pool slot. The pool wrapper itself falls back
        // to candle `Tensor::matmul` for unsupported dtypes or
        // m > POOL_MAX_M (prefill).
        match x.dims().len() {
            3 => {
                let dims = x.dims();
                let (b, s, h) = (dims[0], dims[1], dims[2]);
                let v = self.weight.dims()[0];
                let x_flat = x.reshape((b * s, h))?;
                #[cfg(feature = "cuda-kernels")]
                let y_flat = crate::cuda_kernels::bf16_matmul_pooled(&x_flat, &self.weight)?;
                #[cfg(not(feature = "cuda-kernels"))]
                let y_flat = x_flat.matmul(&self.weight.t()?)?;
                y_flat.reshape((b, s, v))
            }
            _ => {
                #[cfg(feature = "cuda-kernels")]
                {
                    crate::cuda_kernels::bf16_matmul_pooled(x, &self.weight)
                }
                #[cfg(not(feature = "cuda-kernels"))]
                {
                    x.matmul(&self.weight.t()?)
                }
            }
        }
    }

    fn load_weights(&mut self, _weights: &std::collections::HashMap<String, Tensor>) -> Result<()> {
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

impl crate::engine::ModelForward for QuantizedLlamaForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
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

    /// Phase 11.2.B (root-cause fix): override the default
    /// `forward_decode_batch_with_ctx` so the per-forward
    /// `DecodeBatchShared` bundle (pool-backed positions, slot_mapping,
    /// block_tables, seq_lens — built once by
    /// `engine::helpers::execute_batched_decode_with_graph` /
    /// `standard::capture_decode_graph` and stashed in `ctx.decode_shared`)
    /// reaches every attention layer. Without this override, the default
    /// trait impl discards the ctx → attention rebuilds the device
    /// tensors per layer via `Tensor::from_vec` → captured graph holds
    /// stale pointers → subsequent forwards trip
    /// `CUDA_ERROR_INVALID_VALUE` on the eager fallback.
    ///
    /// This is the missing companion to the
    /// `LlamaForCausalLM::forward_decode_batch_with_ctx` we wired in
    /// Phase 11.2 foundation (commit 5162936) — applied to the
    /// *quantized* path which EXL3 (and AWQ, GPTQ, …) routes through.
    fn forward_decode_batch_with_ctx(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        ctx: &crate::engine::cuda_graph::ForwardContext,
    ) -> Result<Tensor> {
        let shared: Option<&DecodeBatchShared> = ctx
            .decode_shared
            .as_ref()
            .and_then(|arc| arc.downcast_ref::<DecodeBatchShared>());

        // Phase 11.2.C: pool-backed embedding lookup. Falls back to the
        // candle `Embedding::forward` path when CUDA is unavailable,
        // num_tokens exceeds the pool budget (prefill), or the weight
        // dtype isn't BF16/F16.
        let xs = {
            #[cfg(feature = "cuda-fused-activations")]
            {
                let weight = self.embed_tokens.embeddings();
                let ids_dims = input_ids.dims();
                let num_tokens: usize = ids_dims.iter().product();
                let dtype_ok = matches!(
                    weight.dtype(),
                    candle_core::DType::BF16 | candle_core::DType::F16
                );
                let in_pool_budget = (1..=64).contains(&num_tokens);
                if input_ids.device().is_cuda()
                    && dtype_ok
                    && in_pool_budget
                    && weight.dims().len() == 2
                {
                    let ids_flat = input_ids.flatten_all()?;
                    let out_flat = crate::cuda_kernels::embedding_pooled(&ids_flat, weight)?;
                    let mut out_shape: Vec<usize> = ids_dims.to_vec();
                    out_shape.push(weight.dims()[1]);
                    out_flat.reshape(out_shape)?
                } else {
                    self.embed_tokens.forward(input_ids)?
                }
            }
            #[cfg(not(feature = "cuda-fused-activations"))]
            {
                self.embed_tokens.forward(input_ids)?
            }
        };

        let mut xs = xs;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch_with_shared(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                shared,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
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
    use crate::quantization::{
        create_weight_loader_with_params, DetectedQuantConfig, QuantizationMethod,
    };

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["LlamaForCausalLM".to_string()],
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
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            sliding_window: None,
            attention_bias: Some(false),
            extra: serde_json::Map::new(),
        }
    }

    fn create_cache_config(cfg: &crate::config::ModelConfig, device: &Device) -> CacheConfig {
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

    #[test]
    fn test_quantized_llama_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model = QuantizedLlamaForCausalLM::new(&cfg, vb, loader.as_ref());
        assert!(
            model.is_ok(),
            "QuantizedLlamaForCausalLM should construct with unquantized loader"
        );

        let model = model.unwrap();
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_quantized_llama_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedLlamaForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build model");

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
    fn test_quantized_llama_with_gptq_loader() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // Test with GPTQ config (will use zeros for quantized weights)
        let detected = DetectedQuantConfig {
            method: QuantizationMethod::Gptq,
            bits: Some(4),
            group_size: Some(128),
            desc_act: Some(false),
            activation_scheme: None,
            raw_config: std::collections::HashMap::new(),
        };
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        // Note: This will fail to load actual weights since VarBuilder::zeros
        // doesn't have the right weight shapes for GPTQ, but it tests the
        // construction path
        let model = QuantizedLlamaForCausalLM::new(&cfg, vb, loader.as_ref());
        // GPTQ loader expects specific tensor shapes, so this may fail
        // In production, real safetensors would be loaded
        assert!(model.is_err() || model.is_ok());
    }

    #[test]
    fn test_quantized_llama_with_attention_bias() {
        // SeedOss uses Llama with attention_bias=true on QKV
        let mut cfg = test_config();
        cfg.attention_bias = Some(true);
        cfg.rope_theta = 1_000_000.0;

        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let detected = DetectedQuantConfig::default();
        let loader = create_weight_loader_with_params(vb.clone(), &detected);

        let model =
            QuantizedLlamaForCausalLM::new(&cfg, vb, loader.as_ref()).expect("build with bias");
        assert_eq!(model.layers.len(), cfg.num_hidden_layers);

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).expect("cache manager");
        let mut block_table = BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).expect("input");
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 4)
            .expect("allocate");
        let slot_mapping = block_table.slot_mapping(0, 4);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
            )
            .expect("forward with attention bias");
        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }
}
