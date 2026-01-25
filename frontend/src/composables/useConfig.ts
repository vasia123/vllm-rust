import { ref, onMounted } from 'vue'
import type { RuntimeConfig } from '@/lib/types'
import { fetchConfig, saveConfig as apiSaveConfig, type ConfigSaveResponse } from '@/lib/api'

export function useConfig() {
  const config = ref<RuntimeConfig | null>(null)
  const error = ref<Error | null>(null)
  const loading = ref(true)
  const saving = ref(false)
  const saveResult = ref<ConfigSaveResponse | null>(null)

  async function refresh() {
    try {
      loading.value = true
      config.value = await fetchConfig()
      error.value = null
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  async function save(updatedConfig: RuntimeConfig) {
    try {
      saving.value = true
      error.value = null
      saveResult.value = null
      saveResult.value = await apiSaveConfig(updatedConfig)
    } catch (e) {
      error.value = e as Error
    } finally {
      saving.value = false
    }
  }

  onMounted(refresh)

  return {
    config,
    error,
    loading,
    saving,
    saveResult,
    refresh,
    save,
  }
}
