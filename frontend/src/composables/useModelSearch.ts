import { ref, watch } from 'vue'
import type { HfModelResult } from '@/lib/types'
import { searchModels } from '@/lib/api'

export function useModelSearch() {
  const query = ref('')
  const fitsOnly = ref(true)
  const results = ref<HfModelResult[]>([])
  const loading = ref(false)
  const error = ref<Error | null>(null)

  let debounceTimer: ReturnType<typeof setTimeout> | null = null

  async function search() {
    if (!query.value.trim()) {
      results.value = []
      return
    }
    loading.value = true
    try {
      results.value = await searchModels(query.value, fitsOnly.value)
      error.value = null
    } catch (e) {
      error.value = e as Error
    } finally {
      loading.value = false
    }
  }

  function debouncedSearch() {
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(search, 300)
  }

  watch(query, debouncedSearch)
  watch(fitsOnly, search)

  return { query, fitsOnly, results, loading, error, search }
}
