import { ref, computed, onUnmounted } from 'vue'
import type { DownloadProgress } from '@/lib/types'
import { startModelDownload, createDownloadProgressStream } from '@/lib/api'

export function useModelDownload() {
  const progress = ref<DownloadProgress | null>(null)
  const isDownloading = ref(false)
  const error = ref<Error | null>(null)
  let cleanup: (() => void) | null = null

  const isComplete = computed(() => progress.value?.status === 'complete')
  const isFailed = computed(() => progress.value?.status === 'error')

  async function download(modelId: string) {
    isDownloading.value = true
    error.value = null
    progress.value = {
      status: 'downloading',
      model_id: modelId,
      message: 'Starting download...',
    }

    try {
      await startModelDownload(modelId)
      startProgressStream()
    } catch (e) {
      error.value = e as Error
      isDownloading.value = false
      progress.value = {
        status: 'error',
        model_id: modelId,
        message: (e as Error).message,
      }
    }
  }

  function startProgressStream() {
    stopProgressStream()
    cleanup = createDownloadProgressStream(
      (event) => {
        progress.value = event
        if (event.status === 'complete' || event.status === 'error') {
          isDownloading.value = false
          stopProgressStream()
        }
      },
      (e) => {
        error.value = e
      },
    )
  }

  function stopProgressStream() {
    cleanup?.()
    cleanup = null
  }

  onUnmounted(stopProgressStream)

  return { progress, isDownloading, isComplete, isFailed, error, download }
}
