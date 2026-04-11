use serde::{Deserialize, Serialize};

use crate::perf_estimate::gpu_profile::GpuHardwareProfile;
use crate::perf_estimate::model_profile::ModelProfile;
use crate::perf_estimate::vram_fitness::WeightDtype;

/// Empirical efficiency factor for memory bandwidth utilization.
/// Real-world LLM inference achieves ~60-80% of theoretical bandwidth
/// due to kernel launch overhead, memory allocation, and CUDA graph overhead.
const BANDWIDTH_EFFICIENCY: f64 = 0.70;

/// User-configurable parameters that affect performance estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimationConfig {
    #[serde(default = "default_weight_dtype")]
    pub weight_dtype: WeightDtype,
    /// KV cache data type: "auto" uses model dtype, "fp8" halves KV memory.
    #[serde(default = "default_kv_cache_dtype")]
    pub kv_cache_dtype: String,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_max_context")]
    pub max_context_length: usize,
    #[serde(default = "default_gpu_utilization")]
    pub gpu_memory_utilization: f32,
    #[serde(default = "default_avg_input")]
    pub avg_input_length: usize,
    #[serde(default = "default_avg_output")]
    pub avg_output_length: usize,
}

fn default_weight_dtype() -> WeightDtype {
    WeightDtype::Bf16
}
fn default_kv_cache_dtype() -> String {
    "auto".to_string()
}
fn default_max_concurrent() -> usize {
    8
}
fn default_max_context() -> usize {
    4096
}
fn default_gpu_utilization() -> f32 {
    0.9
}
fn default_avg_input() -> usize {
    512
}
fn default_avg_output() -> usize {
    256
}

/// Memory breakdown of GPU VRAM usage.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryBreakdown {
    /// Total GPU memory in bytes.
    pub total_gpu_memory: u64,
    /// Usable GPU memory after utilization factor.
    pub usable_gpu_memory: u64,
    /// Model weights in bytes.
    pub model_weights: u64,
    /// KV cache allocation in bytes.
    pub kv_cache: u64,
    /// Estimated activation memory in bytes.
    pub activations: u64,
    /// CUDA context and overhead in bytes.
    pub overhead: u64,
    /// Remaining free memory (can be negative = doesn't fit).
    pub remaining: i64,
}

/// Whether the bottleneck is memory bandwidth or compute.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    MemoryBandwidth,
    Compute,
}

/// Complete performance estimation result.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceEstimate {
    /// Whether the model fits in GPU memory.
    pub model_fits: bool,
    /// Error message if model doesn't fit.
    pub fit_error: Option<String>,
    /// Memory usage breakdown.
    pub memory: MemoryBreakdown,

    /// Single-request decode tokens/sec.
    pub single_request_decode_tps: f64,
    /// Per-request decode tokens/sec at configured batch size.
    pub batched_decode_tps_per_request: f64,
    /// Aggregate decode tokens/sec at configured batch size.
    pub batched_decode_tps_aggregate: f64,
    /// Batch size where decode transitions from memory-bound to compute-bound.
    pub critical_batch_size: usize,

    /// Estimated time-to-first-token in milliseconds.
    pub ttft_ms: f64,
    /// Inter-token latency at batch=1 in milliseconds.
    pub itl_ms_single: f64,
    /// Inter-token latency at configured batch in milliseconds.
    pub itl_ms_batched: f64,

    /// Maximum concurrent requests given KV cache budget.
    pub max_concurrent_possible: usize,
    /// Number of KV cache blocks available.
    pub num_kv_blocks: usize,

    /// Whether decode is memory-bound or compute-bound at configured batch.
    pub decode_bottleneck: Bottleneck,
    /// Whether prefill is memory-bound or compute-bound.
    pub prefill_bottleneck: Bottleneck,
}

/// Estimate performance using the roofline model.
pub fn estimate(
    gpu: &GpuHardwareProfile,
    model: &ModelProfile,
    config: &EstimationConfig,
) -> PerformanceEstimate {
    let bandwidth_bps = gpu.memory_bandwidth_gbs * 1e9;
    let compute_fps = gpu.fp16_tflops * 1e12;

    // Weight size for selected dtype
    let model_weight_bytes =
        (model.total_params as f64 * config.weight_dtype.bytes_per_param()) as u64;

    // Memory breakdown
    let usable = (gpu.total_vram_bytes as f64 * config.gpu_memory_utilization as f64) as u64;
    let overhead = usable / 20; // 5%
    let activation_bytes = estimate_activation_memory(model, config.max_concurrent_requests);

    let kv_elem_size: usize = if config.kv_cache_dtype == "fp8" {
        1
    } else {
        2 // BF16
    };

    // KV cache budget
    let fixed_usage = model_weight_bytes + overhead + activation_bytes;
    let kv_budget = usable.saturating_sub(fixed_usage);

    // KV cache blocks
    let block_size = 16_usize;
    let kv_bytes_per_token_adjusted = if kv_elem_size == 1 {
        model.kv_bytes_per_token / 2 // FP8 KV cache halves the size
    } else {
        model.kv_bytes_per_token
    };
    let bytes_per_block = kv_bytes_per_token_adjusted.max(1) * block_size;
    let num_kv_blocks = kv_budget as usize / bytes_per_block.max(1);
    let total_kv_tokens = num_kv_blocks * block_size;

    // Max concurrent requests
    let max_concurrent_possible = if config.max_context_length > 0 {
        total_kv_tokens / config.max_context_length
    } else {
        0
    };
    let effective_batch = max_concurrent_possible.min(config.max_concurrent_requests);

    // KV cache actual usage (for memory breakdown display)
    let kv_cache_used = effective_batch * config.max_context_length * kv_bytes_per_token_adjusted;

    let model_fits = model_weight_bytes < usable.saturating_sub(overhead);
    let fit_error = if !model_fits {
        Some(format!(
            "Model weights ({:.1}GB) exceed available VRAM ({:.1}GB)",
            model_weight_bytes as f64 / 1e9,
            usable.saturating_sub(overhead) as f64 / 1e9
        ))
    } else {
        None
    };

    let memory = MemoryBreakdown {
        total_gpu_memory: gpu.total_vram_bytes,
        usable_gpu_memory: usable,
        model_weights: model_weight_bytes,
        kv_cache: kv_cache_used as u64,
        activations: activation_bytes,
        overhead,
        remaining: usable as i64
            - model_weight_bytes as i64
            - kv_cache_used as i64
            - activation_bytes as i64
            - overhead as i64,
    };

    if !model_fits {
        return PerformanceEstimate {
            model_fits: false,
            fit_error,
            memory,
            single_request_decode_tps: 0.0,
            batched_decode_tps_per_request: 0.0,
            batched_decode_tps_aggregate: 0.0,
            critical_batch_size: 0,
            ttft_ms: 0.0,
            itl_ms_single: 0.0,
            itl_ms_batched: 0.0,
            max_concurrent_possible,
            num_kv_blocks,
            decode_bottleneck: Bottleneck::MemoryBandwidth,
            prefill_bottleneck: Bottleneck::Compute,
        };
    }

    // Roofline: decode throughput
    // Single request: always memory-bandwidth bound
    let single_decode_tps = bandwidth_bps * BANDWIDTH_EFFICIENCY / model_weight_bytes as f64;

    // Critical batch size: crossover from memory-bound to compute-bound
    let flops_per_token = model.flops_per_token_decode as f64;
    let critical_batch = if flops_per_token > 0.0 {
        (model_weight_bytes as f64 * compute_fps / (bandwidth_bps * flops_per_token)).ceil()
            as usize
    } else {
        1
    };

    // Batched decode
    let (batched_tps_aggregate, decode_bottleneck) = if effective_batch == 0 {
        (0.0, Bottleneck::MemoryBandwidth)
    } else if (effective_batch as f64) < critical_batch as f64 {
        // Memory-bound: throughput scales linearly with batch
        (
            effective_batch as f64 * single_decode_tps,
            Bottleneck::MemoryBandwidth,
        )
    } else {
        // Compute-bound: throughput saturates
        (compute_fps / flops_per_token.max(1.0), Bottleneck::Compute)
    };

    let batched_tps_per_request = if effective_batch > 0 {
        batched_tps_aggregate / effective_batch as f64
    } else {
        0.0
    };

    // Prefill / TTFT
    let input_len = config.avg_input_length as f64;
    let attn_quadratic_flops = model.num_layers as f64
        * model.num_attention_heads as f64
        * model.head_dim as f64
        * input_len
        * input_len
        * 2.0;
    let prefill_flops =
        input_len * model.flops_per_token_prefill_linear as f64 + attn_quadratic_flops;

    let prefill_time_compute = prefill_flops / compute_fps;
    let prefill_time_memory = model_weight_bytes as f64 / bandwidth_bps;
    let ttft_sec = prefill_time_compute.max(prefill_time_memory);
    let ttft_ms = ttft_sec * 1000.0;

    let prefill_bottleneck = if prefill_time_compute > prefill_time_memory {
        Bottleneck::Compute
    } else {
        Bottleneck::MemoryBandwidth
    };

    // Inter-token latency
    let itl_single = if single_decode_tps > 0.0 {
        1000.0 / single_decode_tps
    } else {
        f64::INFINITY
    };
    let itl_batched = if batched_tps_per_request > 0.0 {
        1000.0 / batched_tps_per_request
    } else {
        f64::INFINITY
    };

    PerformanceEstimate {
        model_fits: true,
        fit_error: None,
        memory,
        single_request_decode_tps: single_decode_tps,
        batched_decode_tps_per_request: batched_tps_per_request,
        batched_decode_tps_aggregate: batched_tps_aggregate,
        critical_batch_size: critical_batch,
        ttft_ms,
        itl_ms_single: itl_single,
        itl_ms_batched: itl_batched,
        max_concurrent_possible,
        num_kv_blocks,
        decode_bottleneck,
        prefill_bottleneck,
    }
}

fn estimate_activation_memory(model: &ModelProfile, batch_size: usize) -> u64 {
    // Peak activation: one layer at a time
    // QKV projections + FFN intermediates, BF16 (2 bytes)
    let per_token = (model.hidden_size * 4 + model.intermediate_size * 2) * 2;
    (batch_size * per_token) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::perf_estimate::gpu_profile::gpu_profile_from_values;

    fn llama_8b_profile() -> ModelProfile {
        let config = ModelConfig {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            intermediate_size: 14336,
            vocab_size: 128256,
            head_dim: 128,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
            ..Default::default()
        };
        ModelProfile::from_config("meta-llama/Llama-3.1-8B", &config, WeightDtype::Bf16)
    }

    fn rtx4090_gpu() -> GpuHardwareProfile {
        gpu_profile_from_values(
            "NVIDIA GeForce RTX 4090",
            24 * 1024 * 1024 * 1024,
            20 * 1024 * 1024 * 1024,
        )
    }

    #[test]
    fn test_estimate_llama_8b_rtx4090() {
        let model = llama_8b_profile();
        let gpu = rtx4090_gpu();
        let config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 1,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let result = estimate(&gpu, &model, &config);

        assert!(result.model_fits);
        assert!(result.fit_error.is_none());

        // Llama-3.1-8B BF16 on RTX 4090: expect ~80-150 tok/s single decode
        assert!(
            result.single_request_decode_tps > 30.0,
            "Decode TPS too low: {:.1}",
            result.single_request_decode_tps
        );
        assert!(
            result.single_request_decode_tps < 300.0,
            "Decode TPS too high: {:.1}",
            result.single_request_decode_tps
        );

        // TTFT for 512 tokens should be < 1000ms
        assert!(
            result.ttft_ms < 1000.0,
            "TTFT too high: {:.1}ms",
            result.ttft_ms
        );
        assert!(result.ttft_ms > 0.0);

        assert!(result.num_kv_blocks > 0);
        assert!(result.max_concurrent_possible > 0);
    }

    #[test]
    fn test_estimate_model_doesnt_fit() {
        let gpu =
            gpu_profile_from_values("Small GPU", 4 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024);
        let model = llama_8b_profile(); // ~16GB BF16
        let config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 1,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let result = estimate(&gpu, &model, &config);

        assert!(!result.model_fits);
        assert!(result.fit_error.is_some());
        assert_eq!(result.single_request_decode_tps, 0.0);
    }

    #[test]
    fn test_estimate_fp8_doubles_capacity() {
        let gpu = rtx4090_gpu();
        let model = llama_8b_profile();

        let bf16_config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 8,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let fp8_config = EstimationConfig {
            weight_dtype: WeightDtype::Fp8,
            ..bf16_config.clone()
        };

        let bf16_result = estimate(&gpu, &model, &bf16_config);
        let fp8_result = estimate(&gpu, &model, &fp8_config);

        // FP8 should have more KV blocks (less memory used by weights)
        assert!(fp8_result.num_kv_blocks > bf16_result.num_kv_blocks);

        // FP8 single decode should be ~2x faster (half the weight bytes to load)
        let speedup = fp8_result.single_request_decode_tps / bf16_result.single_request_decode_tps;
        assert!(
            speedup > 1.5,
            "Expected ~2x speedup with FP8, got {speedup:.1}x"
        );
    }

    #[test]
    fn test_estimate_memory_breakdown() {
        let gpu = rtx4090_gpu();
        let model = llama_8b_profile();
        let config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 4,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let result = estimate(&gpu, &model, &config);

        assert_eq!(result.memory.total_gpu_memory, 24 * 1024 * 1024 * 1024);
        assert!(result.memory.model_weights > 0);
        assert!(result.memory.overhead > 0);
        assert!(result.memory.activations > 0);
    }

    #[test]
    fn test_estimate_bottleneck_detection() {
        let gpu = rtx4090_gpu();
        let model = llama_8b_profile();

        // Batch=1: should be memory-bandwidth bound
        let config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 1,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };
        let result = estimate(&gpu, &model, &config);
        assert!(matches!(
            result.decode_bottleneck,
            Bottleneck::MemoryBandwidth
        ));
    }

    #[test]
    fn test_estimate_higher_batch_improves_aggregate() {
        let gpu = rtx4090_gpu();
        let model = llama_8b_profile();

        let config1 = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 1,
            max_context_length: 2048,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let config4 = EstimationConfig {
            max_concurrent_requests: 4,
            ..config1.clone()
        };

        let result1 = estimate(&gpu, &model, &config1);
        let result4 = estimate(&gpu, &model, &config4);

        // Aggregate TPS should increase with batch
        assert!(result4.batched_decode_tps_aggregate >= result1.batched_decode_tps_aggregate);
    }

    #[test]
    fn test_estimate_kv_fp8_more_blocks() {
        let gpu = rtx4090_gpu();
        let model = llama_8b_profile();

        let auto_config = EstimationConfig {
            weight_dtype: WeightDtype::Bf16,
            kv_cache_dtype: "auto".to_string(),
            max_concurrent_requests: 8,
            max_context_length: 4096,
            gpu_memory_utilization: 0.9,
            avg_input_length: 512,
            avg_output_length: 256,
        };

        let fp8_kv_config = EstimationConfig {
            kv_cache_dtype: "fp8".to_string(),
            ..auto_config.clone()
        };

        let auto_result = estimate(&gpu, &model, &auto_config);
        let fp8_kv_result = estimate(&gpu, &model, &fp8_kv_config);

        // FP8 KV cache should give more blocks
        assert!(fp8_kv_result.num_kv_blocks > auto_result.num_kv_blocks);
    }
}
