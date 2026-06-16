//! Bench for the embedding forward path (`ModelForward::forward_hidden`).
//!
//! `forward_hidden` is the new prefill-style pass that backs `/v1/embeddings`
//! on a generative model (ADR: serve generation + embeddings from one loaded
//! model). It is `forward` minus the last-token narrow and minus the `lm_head`
//! projection — so it must never be *more* expensive than a generation prefill.
//! This bench records the new path's baseline and contrasts it with `forward`
//! (which still pays the vocab projection) for both quantized architectures we
//! wired: Gemma 4 (PLE + sliding-window) and Qwen3.
//!
//! Run on CPU with toy dimensions: WSL2 blocks GPU profilers (nsys/ncu), and
//! the relative cost of "skip lm_head + skip narrow" is stable on CPU. Weights
//! are zero (unquantized loader) — we measure the forward *shape* cost, not
//! kernel numerics.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use vllm_core::config::ModelConfig;
use vllm_core::kv_cache::config::CacheConfig;
use vllm_core::kv_cache::{BlockTable, KVCacheDtype, KVCacheManager};
use vllm_core::models::gemma4_quantized::QuantizedGemma4ForCausalLM;
use vllm_core::models::qwen3_quantized::QuantizedQwen3ForCausalLM;
use vllm_core::quantization::{create_weight_loader_with_params, DetectedQuantConfig};

const SEQ_LENS: &[usize] = &[32, 128];
const BLOCK_SIZE: usize = 16;
const NUM_BLOCKS: usize = 64; // 64 * 16 = 1024 tokens — covers SEQ_LENS

fn qwen3_config() -> ModelConfig {
    ModelConfig {
        architectures: vec!["Qwen3ForCausalLM".to_string()],
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

fn gemma4_config() -> ModelConfig {
    let mut extra = serde_json::Map::new();
    extra.insert("sliding_window_pattern".to_string(), serde_json::json!(2));
    extra.insert(
        "hidden_size_per_layer_input".to_string(),
        serde_json::json!(16),
    );
    extra.insert(
        "final_logit_softcapping".to_string(),
        serde_json::json!(30.0),
    );
    ModelConfig {
        architectures: vec!["Gemma4ForCausalLM".to_string()],
        hidden_size: 64,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_hidden_layers: 4,
        intermediate_size: 128,
        vocab_size: 256,
        max_position_embeddings: 512,
        head_dim: 16,
        hidden_act: "gelu_pytorch_tanh".to_string(),
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        tie_word_embeddings: true,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        sliding_window: Some(256),
        attention_bias: Some(false),
        extra,
    }
}

fn cache_for(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
    let cache_config = CacheConfig {
        block_size: BLOCK_SIZE,
        num_blocks: NUM_BLOCKS,
        num_layers: cfg.num_hidden_layers,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        dtype: DType::F32,
        device: device.clone(),
        kv_cache_dtype: KVCacheDtype::Auto,
        cpu_offload: None,
    };
    KVCacheManager::new(&cache_config).expect("cache manager")
}

fn alloc_input(
    kv: &mut KVCacheManager,
    seq_len: usize,
    device: &Device,
) -> (Tensor, BlockTable, Vec<usize>) {
    let mut block_table = BlockTable::new(BLOCK_SIZE);
    kv.allocate_for_request(&mut block_table, seq_len)
        .expect("allocate");
    let slot_mapping = block_table.slot_mapping(0, seq_len);
    let input = Tensor::zeros((1, seq_len), DType::U32, device).expect("input");
    (input, block_table, slot_mapping)
}

fn bench_embed_forward(c: &mut Criterion) {
    let device = Device::Cpu;
    let detected = DetectedQuantConfig::default();

    // ── Qwen3 ──────────────────────────────────────────────────────────────
    {
        let cfg = qwen3_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedQwen3ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("qwen3");

        let mut group = c.benchmark_group("embed_forward/qwen3");
        for &seq_len in SEQ_LENS {
            let mut kv = cache_for(&cfg, &device);
            let (input, block_table, slot_mapping) = alloc_input(&mut kv, seq_len, &device);

            group.bench_with_input(
                BenchmarkId::new("forward_hidden", seq_len),
                &seq_len,
                |b, _| {
                    b.iter(|| {
                        let h = model
                            .forward_hidden(
                                black_box(&input),
                                0,
                                &mut kv,
                                &block_table,
                                &slot_mapping,
                            )
                            .expect("forward_hidden");
                        black_box(h)
                    })
                },
            );
            group.bench_with_input(BenchmarkId::new("forward", seq_len), &seq_len, |b, _| {
                b.iter(|| {
                    let l = model
                        .forward(black_box(&input), 0, &mut kv, &block_table, &slot_mapping)
                        .expect("forward");
                    black_box(l)
                })
            });
        }
        group.finish();
    }

    // ── Gemma 4 (PLE + sliding window) ───────────────────────────────────────
    {
        let cfg = gemma4_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let loader = create_weight_loader_with_params(vb.clone(), &detected);
        let model = QuantizedGemma4ForCausalLM::new(&cfg, vb, loader.as_ref()).expect("gemma4");

        let mut group = c.benchmark_group("embed_forward/gemma4");
        for &seq_len in SEQ_LENS {
            let mut kv = cache_for(&cfg, &device);
            let (input, block_table, slot_mapping) = alloc_input(&mut kv, seq_len, &device);

            group.bench_with_input(
                BenchmarkId::new("forward_hidden", seq_len),
                &seq_len,
                |b, _| {
                    b.iter(|| {
                        let h = model
                            .forward_hidden(
                                black_box(&input),
                                0,
                                &mut kv,
                                &block_table,
                                &slot_mapping,
                            )
                            .expect("forward_hidden");
                        black_box(h)
                    })
                },
            );
            group.bench_with_input(BenchmarkId::new("forward", seq_len), &seq_len, |b, _| {
                b.iter(|| {
                    let l = model
                        .forward(black_box(&input), 0, &mut kv, &block_table, &slot_mapping)
                        .expect("forward");
                    black_box(l)
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_embed_forward);
criterion_main!(benches);
