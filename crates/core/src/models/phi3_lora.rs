//! Phi-3 model with LoRA adapter support.
//!
//! Phi-3 uses fused QKV projection (qkv_proj) and fused gate+up projection
//! (gate_up_proj). LoRA is applied to these fused projections as well as the
//! output projection (o_proj) and down projection (down_proj).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, RmsNorm, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};
use crate::lora::{LinearWithLora, LoraContext, LoraModel};

// ─── Fused SwiGLU MLP with LoRA ─────────────────────────────────────────────
//
// Phi-3 fuses gate_proj and up_proj into a single gate_up_proj weight.
// After projection, we split the output: first half is gate, second is up.

struct Phi3FusedSwiGluMlpWithLora {
    gate_up_proj: LinearWithLora,
    down_proj: LinearWithLora,
    intermediate_size: usize,
}

impl Phi3FusedSwiGluMlpWithLora {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // gate_up_proj: [hidden_size, 2 * intermediate_size] (fused)
        let gate_up_proj =
            LinearWithLora::new(hidden_size, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = LinearWithLora::new(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, layer_lora: &LoraModel, layer_prefix: &str) {
        let gate_up_key = format!("{}.gate_up_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&gate_up_key) {
            self.gate_up_proj
                .register_adapter(adapter_name, adapter.clone());
        }

        let down_key = format!("{}.down_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&down_key) {
            self.down_proj
                .register_adapter(adapter_name, adapter.clone());
        }
    }

    fn forward(&self, xs: &Tensor, lora_ctx: &LoraContext) -> Result<Tensor> {
        let adapter = lora_ctx.adapter_name();
        let fused = self.gate_up_proj.forward_with_lora(xs, adapter)?;
        let gate = fused.narrow(candle_core::D::Minus1, 0, self.intermediate_size)?;
        let up = fused.narrow(
            candle_core::D::Minus1,
            self.intermediate_size,
            self.intermediate_size,
        )?;
        let hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward_with_lora(&hidden, adapter)
    }
}

// ─── Attention with LoRA ─────────────────────────────────────────────────────
//
// Phi-3 uses fused QKV projection instead of separate Q/K/V.

struct Phi3AttentionWithLora {
    qkv_proj: LinearWithLora,
    o_proj: LinearWithLora,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Phi3AttentionWithLora {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Fused QKV: output dim = (num_heads + 2 * num_kv_heads) * head_dim
        let qkv_out_dim = (num_heads + 2 * num_kv_heads) * head_dim;
        let qkv_proj = LinearWithLora::new(cfg.hidden_size, qkv_out_dim, vb.pp("qkv_proj"))?;
        let o_proj = LinearWithLora::new(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn register_lora(&mut self, adapter_name: &str, layer_lora: &LoraModel, layer_prefix: &str) {
        let qkv_key = format!("{}.qkv_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&qkv_key) {
            self.qkv_proj
                .register_adapter(adapter_name, adapter.clone());
        }

        let o_key = format!("{}.o_proj", layer_prefix);
        if let Some(adapter) = layer_lora.get_adapter(&o_key) {
            self.o_proj.register_adapter(adapter_name, adapter.clone());
        }
    }

    /// Split fused QKV output into separate Q, K, V tensors.
    fn split_qkv(
        &self,
        qkv: &Tensor,
        b_sz: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv.narrow(candle_core::D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(candle_core::D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(candle_core::D::Minus1, q_dim + kv_dim, kv_dim)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        Ok((q, k, v))
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
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let adapter = lora_ctx.adapter_name();

        let qkv = self.qkv_proj.forward_with_lora(xs, adapter)?;
        let (q, k, v) = self.split_qkv(&qkv, b_sz, q_len)?;

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

        self.o_proj.forward_with_lora(&attn_output, adapter)
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        cache_engine: &mut CacheEngine,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let batch_size = sequences.len();
        let adapter = lora_ctx.adapter_name();

        let qkv = self.qkv_proj.forward_with_lora(xs, adapter)?;
        let (q, k, v) = self.split_qkv(&qkv, batch_size, 1)?;

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
            )?;

            self.o_proj
                .forward_with_lora(&attn_output, adapter)?
                .unsqueeze(1)
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
            self.o_proj.forward_with_lora(&attn_output, adapter)
        }
    }
}

// ─── Decoder Layer with LoRA ─────────────────────────────────────────────────

struct Phi3DecoderLayerWithLora {
    self_attn: Phi3AttentionWithLora,
    mlp: Phi3FusedSwiGluMlpWithLora,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Phi3DecoderLayerWithLora {
    fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Phi3AttentionWithLora::new(cfg, vb.pp("self_attn"))?;
        let mlp =
            Phi3FusedSwiGluMlpWithLora::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?;
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

    fn register_lora(&mut self, adapter_name: &str, lora_model: &LoraModel, layer_idx: usize) {
        let attn_prefix = format!("layers.{}.self_attn", layer_idx);
        self.self_attn
            .register_lora(adapter_name, lora_model, &attn_prefix);

        let mlp_prefix = format!("layers.{}.mlp", layer_idx);
        self.mlp
            .register_lora(adapter_name, lora_model, &mlp_prefix);
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
        lora_ctx: &LoraContext,
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
            lora_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, lora_ctx)?;
        residual + xs
    }

    fn forward_decode_batch(
        &self,
        xs: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        layer_idx: usize,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            lora_ctx,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs, lora_ctx)?;
        residual + xs
    }
}

// ─── Model with LoRA ─────────────────────────────────────────────────────────

/// Phi-3 model with LoRA adapter support.
///
/// Uses fused QKV projection and fused gate+up projection.
/// LoRA is applied to fused projections and output/down projections.
pub struct Phi3WithLora {
    embed_tokens: Embedding,
    layers: Vec<Phi3DecoderLayerWithLora>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    device: Device,
    dtype: DType,
}

impl Phi3WithLora {
    /// Create a new Phi-3 model with LoRA support.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Phi3DecoderLayerWithLora::new(cfg, vb_l.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
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

    /// Register a LoRA adapter with the model.
    pub fn register_lora(&mut self, lora_model: &LoraModel) {
        let adapter_name = &lora_model.name;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            layer.register_lora(adapter_name, lora_model, layer_idx);
        }
    }

    /// Forward pass with optional LoRA adapter.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
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
                lora_ctx,
            )?;
        }
        let xs = self.norm.forward(&xs)?;
        xs.apply(&self.lm_head)
    }

    /// Batched decode forward with optional LoRA adapter.
    pub fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(&xs, sequences, kv_cache_mgr, layer_idx, lora_ctx)?;
        }

        let xs = self.norm.forward(&xs)?;
        xs.apply(&self.lm_head)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

// ─── ModelForward implementation ─────────────────────────────────────────────

impl crate::engine::ModelForward for Phi3WithLora {
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
            &LoraContext::none(),
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, &LoraContext::none())
    }

    fn supports_lora(&self) -> bool {
        true
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── ModelForwardWithLora implementation ─────────────────────────────────────

impl crate::models::ModelForwardWithLora for Phi3WithLora {
    fn register_lora(&mut self, lora_model: &LoraModel) {
        Phi3WithLora::register_lora(self, lora_model)
    }

    fn forward_with_lora(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
            lora_ctx,
        )
    }

    fn forward_decode_batch_with_lora(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
        lora_ctx: &LoraContext,
    ) -> Result<Tensor> {
        self.forward_decode_batch(input_ids, sequences, kv_cache_mgr, lora_ctx)
    }

    fn lora_adapters(&self) -> Vec<String> {
        if let Some(first_layer) = self.layers.first() {
            first_layer
                .self_attn
                .qkv_proj
                .adapter_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::lora::LoraAdapter;

    fn test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            architectures: vec!["Phi3ForCausalLM".to_string()],
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
    fn test_phi3_with_lora_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = Phi3WithLora::new(&cfg, vb);
        assert!(model.is_ok(), "Phi3WithLora should construct");
        assert_eq!(model.unwrap().layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_phi3_with_lora_forward_no_adapter() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Phi3WithLora::new(&cfg, vb).unwrap();

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 5), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 5)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, 5);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::none(),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
    }

    #[test]
    fn test_phi3_with_lora_register_and_forward() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let mut model = Phi3WithLora::new(&cfg, vb).unwrap();

        let mut lora_model = LoraModel::new("test-adapter", 1, 8, 16.0);

        // Phi3 uses fused qkv_proj
        let qkv_out_dim = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim;
        let lora_a = Tensor::randn(0.0f32, 0.1, (8, cfg.hidden_size), &device).unwrap();
        let lora_b = Tensor::randn(0.0f32, 0.1, (qkv_out_dim, 8), &device).unwrap();
        lora_model.add_adapter(
            "layers.0.self_attn.qkv_proj",
            LoraAdapter::new(lora_a, lora_b, 8, 16.0),
        );

        model.register_lora(&lora_model);

        let adapters = model.layers[0].self_attn.qkv_proj.adapter_names();
        assert!(
            adapters.contains(&"test-adapter"),
            "Adapter should be registered"
        );

        let cache_config = create_cache_config(&cfg, &device);
        let mut kv_cache_mgr = KVCacheManager::new(&cache_config).unwrap();
        let mut block_table = crate::kv_cache::BlockTable::new(cache_config.block_size);

        let input_ids = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache_mgr
            .allocate_for_request(&mut block_table, 3)
            .unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(
                &input_ids,
                0,
                &mut kv_cache_mgr,
                &block_table,
                &slot_mapping,
                &LoraContext::with_adapter("test-adapter"),
            )
            .unwrap();

        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    }
}
