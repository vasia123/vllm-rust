<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { fetchHealth } from '@/lib/api'
import type { HealthResponse } from '@/lib/types'

const health = ref<HealthResponse | null>(null)

async function checkHealth() {
  try {
    health.value = await fetchHealth()
  } catch {
    health.value = null
  }
}

onMounted(() => {
  checkHealth()
  setInterval(checkHealth, 5000)
})

const statusColors = {
  healthy: 'bg-green-500',
  degraded: 'bg-yellow-500',
  unhealthy: 'bg-red-500',
}
</script>

<template>
  <header class="h-16 bg-gray-800 border-b border-gray-700 flex items-center justify-between px-6">
    <div class="flex items-center gap-4">
      <h2 class="text-lg font-semibold text-white">
        {{ $route.path === '/admin' ? 'Dashboard' : 'Settings' }}
      </h2>
    </div>
    <div class="flex items-center gap-4">
      <div v-if="health" class="flex items-center gap-2">
        <span
          class="w-2.5 h-2.5 rounded-full animate-pulse"
          :class="statusColors[health.status]"
        />
        <span class="text-sm text-gray-300 capitalize">{{ health.status }}</span>
        <span class="text-sm text-gray-500">|</span>
        <span class="text-sm text-gray-400">{{ health.model_id }}</span>
      </div>
      <div v-else class="flex items-center gap-2">
        <span class="w-2.5 h-2.5 rounded-full bg-gray-500" />
        <span class="text-sm text-gray-500">Connecting...</span>
      </div>
    </div>
  </header>
</template>
