import type {
  AdminMetrics,
  RuntimeConfig,
  HealthResponse,
  RestartStatus,
  GpuInfo,
  HfModelResult,
  DownloadProgress,
  EstimateRequest,
  PerformanceEstimate,
} from './types'

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

// ─── GPU Info ────────────────────────────────────────────────────────────────

export async function fetchGpuInfo(): Promise<GpuInfo> {
  const response = await fetch(`${BASE_URL}/gpu/info`)
  if (!response.ok) {
    throw new Error(`Failed to fetch GPU info: ${response.statusText}`)
  }
  return response.json()
}

// ─── Model Search ────────────────────────────────────────────────────────────

export async function searchModels(
  query: string,
  fitsOnly: boolean = true,
  limit: number = 50,
): Promise<HfModelResult[]> {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
    fits_only: String(fitsOnly),
  })
  const response = await fetch(`${BASE_URL}/models/search?${params}`)
  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`)
  }
  return response.json()
}

// ─── Model Download ──────────────────────────────────────────────────────────

export async function startModelDownload(modelId: string): Promise<void> {
  const response = await fetch(`${BASE_URL}/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Download failed: ${text || response.statusText}`)
  }
}

export function createDownloadProgressStream(
  onMessage: (progress: DownloadProgress) => void,
  onError?: (error: Error) => void,
): () => void {
  const eventSource = new EventSource(`${BASE_URL}/models/download/progress`)

  eventSource.addEventListener('progress', (event) => {
    try {
      const progress = JSON.parse(event.data) as DownloadProgress
      onMessage(progress)
    } catch (e) {
      onError?.(e as Error)
    }
  })

  eventSource.onerror = () => {
    onError?.(new Error('Download progress SSE error'))
  }

  return () => eventSource.close()
}

// ─── Performance Estimates ───────────────────────────────────────────────────

export async function fetchEstimates(
  request: EstimateRequest,
): Promise<PerformanceEstimate> {
  const response = await fetch(`${BASE_URL}/models/estimate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    throw new Error(`Estimate failed: ${response.statusText}`)
  }
  return response.json()
}
