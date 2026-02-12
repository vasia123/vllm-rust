use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, CacheEngine, KVCacheManager};
use crate::layers::{paged_attention, RotaryEmbedding};

use super::tp_layers::{TpContext, TpEmbedding, TpLinear};

// ─── Attention ───────────────────────────────────────────────────────────────

struct GPTJAttention {
    q_proj: TpLinear,
    k_proj: TpLinear,
    v_proj: TpLinear,
    out_proj: TpLinear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl GPTJAttention {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let world_size = pg.world_size();

        let rotary_dim = cfg
            .extra
            .get("rotary_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(head_dim as u64) as usize;
        let partial_factor = rotary_dim as f64 / head_dim as f64;

        let q_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("q_proj"),
            pg,
        )?;
        let k_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("k_proj"),
            pg,
        )?;
        let v_proj = TpLinear::column_parallel(
            cfg.hidden_size,
            num_heads * head_dim,
            false,
            false,
            vb.pp("v_proj"),
            pg,
        )?;
        let out_proj = TpLinear::row_parallel(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            true,
            vb.pp("out_proj"),
            pg,
        )?;

        let rotary_emb = RotaryEmbedding::new_partial(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            partial_factor,
            false, // GPT-J style (not NeoX)
            vb.dtype(),
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rotary_emb,
            num_heads: num_heads / world_size,
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
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
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
            self.num_heads,
            self.head_dim,
        )?;

        self.out_proj.forward(&attn_output, tp_ctx)
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
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-sequence: RoPE + cache write/read + attention
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
                self.num_heads,
                self.head_dim,
            )?;
            outputs.push(attn_out);
        }

        let attn_output = Tensor::cat(&outputs, 0)?;
        self.out_proj.forward(&attn_output, tp_ctx)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

#[allow(clippy::upper_case_acronyms)]
struct GPTJMLP {
    fc_in: TpLinear,
    fc_out: TpLinear,
}

impl GPTJMLP {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let intermediate = cfg
            .extra
            .get("n_inner")
            .and_then(|v| v.as_u64())
            .unwrap_or(cfg.intermediate_size as u64) as usize;
        let intermediate = if intermediate == 0 {
            4 * cfg.hidden_size
        } else {
            intermediate
        };

        let fc_in = TpLinear::column_parallel(
            cfg.hidden_size,
            intermediate,
            true,
            false,
            vb.pp("fc_in"),
            pg,
        )?;
        let fc_out = TpLinear::row_parallel(
            intermediate,
            cfg.hidden_size,
            true,
            true,
            vb.pp("fc_out"),
            pg,
        )?;
        Ok(Self { fc_in, fc_out })
    }

    fn forward(&self, xs: &Tensor, tp_ctx: &TpContext) -> Result<Tensor> {
        let xs = self.fc_in.forward(xs, tp_ctx)?;
        let xs = xs.gelu_erf()?;
        self.fc_out.forward(&xs, tp_ctx)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

struct GPTJDecoderLayer {
    self_attn: GPTJAttention,
    mlp: GPTJMLP,
    ln: candle_nn::LayerNorm,
}

impl GPTJDecoderLayer {
    fn new_with_tp(cfg: &ModelConfig, vb: VarBuilder, pg: &dyn ProcessGroup) -> Result<Self> {
        let eps = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let self_attn = GPTJAttention::new_with_tp(cfg, vb.pp("attn"), pg)?;
        let mlp = GPTJMLP::new_with_tp(cfg, vb.pp("mlp"), pg)?;
        let ln = layer_norm(cfg.hidden_size, eps, vb.pp("ln_1"))?;

        Ok(Self { self_attn, mlp, ln })
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
        let xs = self.ln.forward(xs)?;
        // GPT-J parallel residual: attn(ln(x)) + mlp(ln(x)) + x
        let attn_output = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offset,
            kv_cache_mgr.engine_mut(layer_idx),
            block_table,
            slot_mapping,
            tp_ctx,
        )?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        (&attn_output + &mlp_output)? + residual
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
        let xs = self.ln.forward(xs)?;
        let attn_output = self.self_attn.forward_decode_batch(
            &xs,
            sequences,
            kv_cache_mgr.engine_mut(layer_idx),
            tp_ctx,
        )?;
        let mlp_output = self.mlp.forward(&xs, tp_ctx)?;
        (&attn_output + &mlp_output)? + residual
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

pub struct GPTJForCausalLM {
    wte: TpEmbedding,
    layers: Vec<GPTJDecoderLayer>,
    ln_f: candle_nn::LayerNorm,
    lm_head: Linear,
    tp_ctx: TpContext,
    device: Device,
    dtype: DType,
}

impl GPTJForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let eps = cfg
            .extra
            .get("layer_norm_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let vb_t = vb.pp("transformer");
        let wte = TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_t.pp("wte"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_t.pp("h");
        for i in 0..cfg.num_hidden_layers {
            layers.push(GPTJDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let ln_f = layer_norm(cfg.hidden_size, eps, vb_t.pp("ln_f"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            layers,
            ln_f,
            lm_head,
            tp_ctx,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::engine::ModelForward for GPTJForCausalLM {
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

        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;
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
        let xs = self.ln_f.forward(&xs)?;
        self.lm_head.forward(&xs)
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let mut xs = self.wte.forward(input_ids, &self.tp_ctx)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }
        let xs = self.ln_f.forward(&xs)?;
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
        extra.insert("rotary_dim".into(), serde_json::Value::from(8));
        extra.insert("layer_norm_epsilon".into(), serde_json::Value::from(1e-5));

        ModelConfig {
            architectures: vec!["GPTJForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 256,
            vocab_size: 256,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "gelu".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    #[test]
    fn test_gptj_construction() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        let model = GPTJForCausalLM::new(&cfg, vb);
        assert!(
            model.is_ok(),
            "GPTJForCausalLM should construct: {:?}",
            model.err()
        );
        assert_eq!(model.unwrap().layers.len(), 2);
    }

    #[test]
    fn test_gptj_forward_shape() {
        let cfg = test_config();
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = GPTJForCausalLM::new(&cfg, vb).expect("build model");

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
}
