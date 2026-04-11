<script setup lang="ts">
import { computed } from 'vue'
import type { HfModelResult } from '@/lib/types'

const props = defineProps<{
  model: HfModelResult
  isDownloading?: boolean
}>()

const emit = defineEmits<{
  select: [modelId: string]
  download: [modelId: string]
}>()

const paramLabel = computed(() => {
  if (!props.model.total_params) return '?'
  const p = props.model.total_params
  if (p >= 1e9) return `${(p / 1e9).toFixed(1)}B`
  if (p >= 1e6) return `${(p / 1e6).toFixed(0)}M`
  return `${p}`
})

const downloadsLabel = computed(() => {
  const d = props.model.downloads
  if (d >= 1e6) return `${(d / 1e6).toFixed(1)}M`
  if (d >= 1e3) return `${(d / 1e3).toFixed(1)}K`
  return `${d}`
})

function dtypeBadgeClass(fits: boolean, isRecommended: boolean): string {
  if (isRecommended) return 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/50'
  if (fits) return 'bg-green-500/20 text-green-300'
  return 'bg-red-500/20 text-red-400'
}
</script>

<template>
  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0 flex-1">
        <h3 class="text-white font-medium truncate">{{ model.id }}</h3>
        <div class="flex items-center gap-3 mt-1 text-sm text-gray-400">
          <span class="font-mono">{{ paramLabel }}</span>
          <span>{{ downloadsLabel }} downloads</span>
        </div>
      </div>
      <button
        @click="emit('select', model.id)"
        :disabled="isDownloading"
        class="shrink-0 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
        :class="isDownloading
          ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
          : 'bg-blue-600 text-white hover:bg-blue-500'"
      >
        {{ isDownloading ? 'Downloading...' : 'Select' }}
      </button>
    </div>

    <!-- VRAM fitness badges -->
    <div v-if="model.vram_fitness" class="flex flex-wrap gap-1.5 mt-3">
      <span
        v-for="v in model.vram_fitness.variants"
        :key="v.dtype"
        class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono"
        :class="dtypeBadgeClass(v.fits, v.dtype === model.vram_fitness!.recommended_dtype)"
      >
        {{ v.dtype.toUpperCase() }}
        <span v-if="v.dtype === model.vram_fitness!.recommended_dtype" class="text-yellow-400">*</span>
        <span class="text-gray-500">{{ v.weight_gb.toFixed(1) }}G</span>
      </span>
    </div>
  </div>
</template>
