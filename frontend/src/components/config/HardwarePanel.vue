<script setup lang="ts">
import { computed } from 'vue'
import type { GpuInfo } from '@/lib/types'

const props = defineProps<{
  gpu: GpuInfo | null
  loading?: boolean
}>()

const vramTotal = computed(() => {
  if (!props.gpu) return '-'
  return `${(props.gpu.total_vram_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
})

const vramFree = computed(() => {
  if (!props.gpu) return '-'
  return `${(props.gpu.free_vram_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
})
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <h3 class="text-sm text-gray-400 mb-4">GPU Hardware</h3>
    <div v-if="loading" class="text-gray-500 text-sm">Detecting GPU...</div>
    <div v-else-if="!gpu" class="text-red-400 text-sm">No GPU detected</div>
    <div v-else class="grid grid-cols-2 gap-3">
      <div>
        <p class="text-xs text-gray-500">GPU</p>
        <p class="text-sm text-white font-medium">{{ gpu.name }}</p>
      </div>
      <div>
        <p class="text-xs text-gray-500">VRAM</p>
        <p class="text-sm text-white">{{ vramTotal }} ({{ vramFree }} free)</p>
      </div>
      <div>
        <p class="text-xs text-gray-500">Bandwidth</p>
        <p class="text-sm text-white">{{ gpu.memory_bandwidth_gbs.toFixed(0) }} GB/s</p>
      </div>
      <div>
        <p class="text-xs text-gray-500">FP16 Compute</p>
        <p class="text-sm text-white">{{ gpu.fp16_tflops.toFixed(1) }} TFLOPS</p>
      </div>
      <div v-if="!gpu.is_known_gpu" class="col-span-2">
        <p class="text-xs text-amber-400">Unrecognized GPU - estimates may be approximate</p>
      </div>
    </div>
  </div>
</template>
