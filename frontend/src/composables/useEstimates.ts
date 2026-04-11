import { ref, watch, type Ref } from 'vue'
import type { EstimateRequest, PerformanceEstimate } from '@/lib/types'
import { fetchEstimates } from '@/lib/api'

export function useEstimates(request: Ref<EstimateRequest | null>) {
  const estimates = ref<PerformanceEstimate | null>(null)
  const loading = ref(false)
  const error = ref<Error | null>(null)

  let debounceTimer: ReturnType<typeof setTimeout> | null = null
  let abortController: AbortController | null = null

  async function refresh() {
    if (!request.value) return

    if (abortController) abortController.abort()
    abortController = new AbortController()

    loading.value = true
    try {
      estimates.value = await fetchEstimates(request.value)
      error.value = null
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        error.value = e as Error
      }
    } finally {
      loading.value = false
    }
  }

  function debouncedRefresh() {
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(refresh, 150)
  }

  watch(request, debouncedRefresh, { deep: true })

  return { estimates, loading, error, refresh }
}
