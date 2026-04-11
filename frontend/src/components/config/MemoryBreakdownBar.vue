<script setup lang="ts">
import { computed } from 'vue'
import type { MemoryBreakdown } from '@/lib/types'

const props = defineProps<{
  breakdown: MemoryBreakdown | null
}>()

function formatBytes(bytes: number): string {
  const gb = Math.abs(bytes) / (1024 * 1024 * 1024)
  return `${gb.toFixed(1)} GB`
}

const segments = computed(() => {
  if (!props.breakdown) return []
  const total = props.breakdown.total_gpu_memory
  if (total === 0) return []
  const pct = (v: number) => Math.max(0, (v / total) * 100)
  return [
    { label: 'Weights', bytes: props.breakdown.model_weights, pct: pct(props.breakdown.model_weights), color: 'bg-blue-500' },
    { label: 'KV Cache', bytes: props.breakdown.kv_cache, pct: pct(props.breakdown.kv_cache), color: 'bg-purple-500' },
    { label: 'Activations', bytes: props.breakdown.activations, pct: pct(props.breakdown.activations), color: 'bg-orange-500' },
    { label: 'Overhead', bytes: props.breakdown.overhead, pct: pct(props.breakdown.overhead), color: 'bg-gray-500' },
    { label: 'Free', bytes: Math.max(0, props.breakdown.remaining), pct: pct(Math.max(0, props.breakdown.remaining)), color: 'bg-gray-700' },
  ]
})
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <div class="flex justify-between items-center mb-4">
      <p class="text-sm text-gray-400">Memory Breakdown</p>
      <p v-if="breakdown" class="text-sm text-gray-300">{{ formatBytes(breakdown.total_gpu_memory) }} total</p>
    </div>

    <div v-if="breakdown" class="h-6 bg-gray-700 rounded-full overflow-hidden flex">
      <div
        v-for="seg in segments"
        :key="seg.label"
        :class="seg.color"
        :style="{ width: `${seg.pct}%` }"
        class="h-full transition-all duration-300"
      />
    </div>

    <div v-if="breakdown" class="flex flex-wrap gap-x-4 gap-y-1 mt-3">
      <div v-for="seg in segments" :key="seg.label" class="flex items-center gap-1.5">
        <div :class="seg.color" class="w-2.5 h-2.5 rounded-sm" />
        <span class="text-xs text-gray-400">{{ seg.label }}: {{ formatBytes(seg.bytes) }}</span>
      </div>
    </div>
  </div>
</template>
