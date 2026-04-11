import { ref, onMounted } from 'vue'
import type { GpuInfo } from '@/lib/types'
import { fetchGpuInfo } from '@/lib/api'

export function useGpuInfo() {
  const gpuInfo = ref<GpuInfo | null>(null)
  const loading = ref(true)
  const error = ref<Error | null>(null)

  async function load() {
    loading.value = true
    try {
      gpuInfo.value = await fetchGpuInfo()
      error.value = null
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  onMounted(load)

  return { gpuInfo, loading, error, reload: load }
}
