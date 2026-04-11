<script setup lang="ts">
import { computed } from 'vue'
import type { PerformanceEstimate } from '@/lib/types'

const props = defineProps<{
  estimates: PerformanceEstimate | null
  loading?: boolean
}>()

function fmt(n: number, decimals = 1): string {
  if (!isFinite(n)) return '-'
  return n.toFixed(decimals)
}

const metrics = computed(() => {
  if (!props.estimates) return null
  const e = props.estimates
  return [
    {
      label: 'Decode (1 req)',
      value: `${fmt(e.single_request_decode_tps)} tok/s`,
      sub: `${fmt(e.itl_ms_single)} ms/token`,
    },
    {
      label: `Decode (${e.max_concurrent_possible} req)`,
      value: `${fmt(e.batched_decode_tps_aggregate)} tok/s`,
      sub: `${fmt(e.batched_decode_tps_per_request)} per req`,
    },
    {
      label: 'Time to First Token',
      value: `${fmt(e.ttft_ms, 0)} ms`,
      sub: e.prefill_bottleneck === 'compute' ? 'compute-bound' : 'bandwidth-bound',
    },
    {
      label: 'KV Cache Blocks',
      value: `${e.num_kv_blocks}`,
      sub: `max ${e.max_concurrent_possible} concurrent`,
    },
  ]
})
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700" :class="{ 'opacity-60': loading }">
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-sm text-gray-400">Performance Estimates</h3>
      <div v-if="loading" class="w-3 h-3 bg-blue-400 rounded-full animate-pulse" />
    </div>

    <div v-if="estimates && !estimates.model_fits" class="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
      <p class="text-red-400 text-sm font-medium">Model doesn't fit</p>
      <p class="text-red-300 text-xs mt-1">{{ estimates.fit_error }}</p>
    </div>

    <div v-else-if="metrics" class="grid grid-cols-2 gap-4">
      <div v-for="m in metrics" :key="m.label">
        <p class="text-xs text-gray-500">{{ m.label }}</p>
        <p class="text-xl font-bold text-white mt-1">{{ m.value }}</p>
        <p class="text-xs text-gray-500 mt-0.5">{{ m.sub }}</p>
      </div>
    </div>

    <div v-else class="text-gray-500 text-sm text-center py-4">
      Select a model to see estimates
    </div>

    <div v-if="estimates?.model_fits" class="mt-4 pt-3 border-t border-gray-700">
      <p class="text-xs text-gray-500">
        Decode bottleneck:
        <span :class="estimates.decode_bottleneck === 'compute' ? 'text-orange-400' : 'text-blue-400'">
          {{ estimates.decode_bottleneck === 'compute' ? 'Compute-bound' : 'Bandwidth-bound' }}
        </span>
        &middot; Critical batch: {{ estimates.critical_batch_size }}
      </p>
    </div>
  </div>
</template>
