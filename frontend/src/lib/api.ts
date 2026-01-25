import type { AdminMetrics, RuntimeConfig, HealthResponse, RestartStatus } from './types'

const BASE_URL = '/admin'

export async function fetchMetrics(): Promise<AdminMetrics> {
  const response = await fetch(`${BASE_URL}/metrics`)
  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`)
  }
  return response.json()
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${BASE_URL}/health`)
  if (!response.ok) {
    throw new Error(`Failed to fetch health: ${response.statusText}`)
  }
  return response.json()
}

export async function fetchConfig(): Promise<RuntimeConfig> {
  const response = await fetch(`${BASE_URL}/config`)
  if (!response.ok) {
    throw new Error(`Failed to fetch config: ${response.statusText}`)
  }
  return response.json()
}

export interface ConfigSaveResponse {
  success: boolean
  path: string
  message: string
}

export async function saveConfig(config: RuntimeConfig): Promise<ConfigSaveResponse> {
  const response = await fetch(`${BASE_URL}/config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ config }),
  })
  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Failed to save config: ${errorText || response.statusText}`)
  }
  return response.json()
}

export async function requestRestart(config?: Partial<RuntimeConfig>): Promise<void> {
  const response = await fetch(`${BASE_URL}/restart`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ config }),
  })
  if (!response.ok) {
    throw new Error(`Failed to request restart: ${response.statusText}`)
  }
}

export function createMetricsStream(onMessage: (metrics: AdminMetrics) => void, onError?: (error: Error) => void): () => void {
  const eventSource = new EventSource(`${BASE_URL}/metrics/stream`)

  eventSource.addEventListener('metrics', (event) => {
    try {
      const metrics = JSON.parse(event.data) as AdminMetrics
      onMessage(metrics)
    } catch (e) {
      onError?.(e as Error)
    }
  })

  eventSource.onerror = () => {
    onError?.(new Error('SSE connection error'))
  }

  return () => eventSource.close()
}

export function createRestartStatusStream(onMessage: (status: RestartStatus) => void, onError?: (error: Error) => void): () => void {
  const eventSource = new EventSource(`${BASE_URL}/restart/status`)

  eventSource.addEventListener('status', (event) => {
    try {
      const status = JSON.parse(event.data) as RestartStatus
      onMessage(status)
    } catch (e) {
      onError?.(e as Error)
    }
  })

  eventSource.onerror = () => {
    onError?.(new Error('Restart status SSE connection error'))
  }

  return () => eventSource.close()
}
