<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  modelId: string
  uptime: number
  device: string
  dtype: string
}>()

const formattedUptime = computed(() => {
  const hours = Math.floor(props.uptime / 3600)
  const minutes = Math.floor((props.uptime % 3600) / 60)
  const seconds = props.uptime % 60
  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
})
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <h3 class="text-sm font-medium text-gray-400 mb-4">System Information</h3>
    <div class="space-y-3">
      <div class="flex justify-between">
        <span class="text-gray-500">Model</span>
        <span class="text-gray-200 font-mono text-sm">{{ modelId }}</span>
      </div>
      <div class="flex justify-between">
        <span class="text-gray-500">Device</span>
        <span class="text-gray-200 font-mono text-sm">{{ device }}</span>
      </div>
      <div class="flex justify-between">
        <span class="text-gray-500">Precision</span>
        <span class="text-gray-200 font-mono text-sm">{{ dtype }}</span>
      </div>
      <div class="flex justify-between">
        <span class="text-gray-500">Uptime</span>
        <span class="text-gray-200 font-mono text-sm">{{ formattedUptime }}</span>
      </div>
    </div>
  </div>
</template>
