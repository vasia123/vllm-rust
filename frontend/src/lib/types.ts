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
