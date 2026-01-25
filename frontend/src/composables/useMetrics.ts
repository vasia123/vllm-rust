import { ref, onMounted, onUnmounted } from 'vue'
import type { AdminMetrics } from '@/lib/types'
import { fetchMetrics, createMetricsStream } from '@/lib/api'

export function useMetrics() {
  const metrics = ref<AdminMetrics | null>(null)
  const error = ref<Error | null>(null)
  const loading = ref(true)
  let cleanup: (() => void) | null = null

  async function refresh() {
    try {
      loading.value = true
      metrics.value = await fetchMetrics()
      error.value = null
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  function startStream() {
    cleanup = createMetricsStream(
      (newMetrics) => {
        metrics.value = newMetrics
        error.value = null
        loading.value = false
      },
      (e) => {
        error.value = e
      }
    )
  }

  onMounted(() => {
    refresh()
    startStream()
  })

  onUnmounted(() => {
    cleanup?.()
  })

  return {
    metrics,
    error,
    loading,
    refresh,
  }
}
