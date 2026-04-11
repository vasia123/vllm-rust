import { ref, computed, watch, type Ref } from 'vue'
import type { EstimateRequest, GpuInfo } from '@/lib/types'

export function useEngineConfig(modelId: Ref<string>, gpuInfo: Ref<GpuInfo | null>) {
  const maxContextLength = ref(4096)
  const maxConcurrentRequests = ref(8)
  const kvCacheDtype = ref<'auto' | 'fp8'>('auto')
  const gpuMemoryUtilization = ref(0.9)
  const weightDtype = ref('bf16')
  const avgInputLength = ref(512)
  const avgOutputLength = ref(256)

  // Set smart defaults when GPU info arrives
  watch(
    gpuInfo,
    (info) => {
      if (!info) return
      const vramGb = info.total_vram_bytes / (1024 * 1024 * 1024)
      if (vramGb >= 40) {
        maxConcurrentRequests.value = 16
        maxContextLength.value = 8192
      } else if (vramGb >= 16) {
        maxConcurrentRequests.value = 8
        maxContextLength.value = 4096
      } else {
        maxConcurrentRequests.value = 4
        maxContextLength.value = 2048
      }
    },
    { immediate: true },
  )

  const estimateRequest = computed<EstimateRequest | null>(() => {
    if (!modelId.value) return null
    return {
      model_id: modelId.value,
      weight_dtype: weightDtype.value,
      kv_cache_dtype: kvCacheDtype.value,
      max_concurrent_requests: maxConcurrentRequests.value,
      max_context_length: maxContextLength.value,
      gpu_memory_utilization: gpuMemoryUtilization.value,
      avg_input_length: avgInputLength.value,
      avg_output_length: avgOutputLength.value,
    }
  })

  return {
    maxContextLength,
    maxConcurrentRequests,
    kvCacheDtype,
    gpuMemoryUtilization,
    weightDtype,
    avgInputLength,
    avgOutputLength,
    estimateRequest,
  }
}
