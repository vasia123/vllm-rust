export interface KVCacheMetrics {
  allocations: number
  blocks_allocated: number
  blocks_freed: number
  blocks_evicted: number
  cache_hits: number
  cache_misses: number
  cache_queries: number
  hit_rate: number | null
}

export interface PrefixCacheStats {
  cached_blocks: number
  evictable_blocks: number
}

export interface AdminMetrics {
  kv_cache: KVCacheMetrics
  running_requests: number
  waiting_requests: number
  model_id: string
  uptime_seconds: number
  accepting_requests: boolean
  timestamp_ms: number
  num_free_blocks: number
  num_total_blocks: number
  block_size: number
  prefix_cache_stats?: PrefixCacheStats
}

export interface RuntimeConfig {
  model: string
  draft_model?: string
  num_speculative_tokens: number
  num_blocks: number
  block_size: number
  max_requests: number
  max_tokens_per_step: number
  enable_chunked_prefill: boolean
  multi_step_count: number
  enable_prefix_caching: boolean
  dtype: string
  device: string
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  model_id: string
  uptime_seconds: number
}

export type RestartStatus =
  | { status: 'idle' }
  | { status: 'draining'; active_requests: number }
  | { status: 'shutting_down' }
  | { status: 'loading'; model: string }
  | { status: 'ready' }
  | { status: 'failed'; error: string }

// ─── GPU Info ────────────────────────────────────────────────────────────────

export interface GpuInfo {
  name: string
  compute_capability: number
  memory_bandwidth_gbs: number
  fp16_tflops: number
  total_vram_bytes: number
  free_vram_bytes: number
  sm_count: number
  is_known_gpu: boolean
}

// ─── Model Search ────────────────────────────────────────────────────────────

export interface VariantFitness {
  dtype: string
  weight_bytes: number
  weight_gb: number
  fits: boolean
  remaining_bytes: number
  max_context_at_batch1: number
}

export interface VramFitness {
  model_id: string
  total_params: number
  variants: VariantFitness[]
  any_fits: boolean
  recommended_dtype: string | null
}

export interface HfModelResult {
  id: string
  downloads: number
  likes: number
  pipeline_tag: string | null
  tags: string[]
  total_params: number | null
  vram_fitness: VramFitness | null
}

// ─── Download ────────────────────────────────────────────────────────────────

export interface DownloadProgress {
  status: 'downloading' | 'complete' | 'error'
  model_id: string
  file?: string
  message?: string
}

// ─── Performance Estimates ───────────────────────────────────────────────────

export interface MemoryBreakdown {
  total_gpu_memory: number
  usable_gpu_memory: number
  model_weights: number
  kv_cache: number
  activations: number
  overhead: number
  remaining: number
}

export interface PerformanceEstimate {
  model_fits: boolean
  fit_error: string | null
  memory: MemoryBreakdown
  single_request_decode_tps: number
  batched_decode_tps_per_request: number
  batched_decode_tps_aggregate: number
  critical_batch_size: number
  ttft_ms: number
  itl_ms_single: number
  itl_ms_batched: number
  max_concurrent_possible: number
  num_kv_blocks: number
  decode_bottleneck: 'memory_bandwidth' | 'compute'
  prefill_bottleneck: 'memory_bandwidth' | 'compute'
}

export interface EstimateRequest {
  model_id: string
  revision?: string
  weight_dtype?: string
  kv_cache_dtype: string
  max_concurrent_requests: number
  max_context_length: number
  gpu_memory_utilization: number
  avg_input_length: number
  avg_output_length: number
}
