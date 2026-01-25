import { ref, computed, onUnmounted } from 'vue'
import type { RestartStatus, RuntimeConfig } from '@/lib/types'
import { requestRestart, createRestartStatusStream } from '@/lib/api'

export function useRestart() {
  const status = ref<RestartStatus>({ status: 'idle' })
  const error = ref<Error | null>(null)
  let cleanup: (() => void) | null = null

  const isRestarting = computed(() => {
    const s = status.value.status
    return s !== 'idle' && s !== 'ready' && s !== 'failed'
  })

  const isIdle = computed(() => status.value.status === 'idle')
  const isReady = computed(() => status.value.status === 'ready')
  const isFailed = computed(() => status.value.status === 'failed')

  const statusMessage = computed(() => {
    const s = status.value
    switch (s.status) {
      case 'idle':
        return 'Server is running'
      case 'draining':
        return `Draining ${s.active_requests} active request(s)...`
      case 'shutting_down':
        return 'Shutting down old engine...'
      case 'loading':
        return `Loading model: ${s.model}...`
      case 'ready':
        return 'Restart complete'
      case 'failed':
        return `Restart failed: ${s.error}`
      default:
        return 'Unknown status'
    }
  })

  async function restart(config?: Partial<RuntimeConfig>) {
    try {
      error.value = null
      await requestRestart(config)
      startStatusStream()
    } catch (e) {
      error.value = e as Error
      status.value = { status: 'failed', error: (e as Error).message }
    }
  }

  function startStatusStream() {
    stopStatusStream()
    cleanup = createRestartStatusStream(
      (newStatus) => {
        status.value = newStatus
        error.value = null
        if (newStatus.status === 'ready' || newStatus.status === 'failed') {
          stopStatusStream()
        }
      },
      (e) => {
        error.value = e
      }
    )
  }

  function stopStatusStream() {
    cleanup?.()
    cleanup = null
  }

  function reset() {
    stopStatusStream()
    status.value = { status: 'idle' }
    error.value = null
  }

  onUnmounted(() => {
    stopStatusStream()
  })

  return {
    status,
    error,
    isRestarting,
    isIdle,
    isReady,
    isFailed,
    statusMessage,
    restart,
    reset,
  }
}
