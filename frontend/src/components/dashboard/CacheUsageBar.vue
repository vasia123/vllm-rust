<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  used: number
  total: number
}>()

const percentage = computed(() => {
  if (props.total === 0) return 0
  return Math.round(((props.total - props.used) / props.total) * 100)
})

const colorClass = computed(() => {
  if (percentage.value > 75) return 'bg-green-500'
  if (percentage.value > 50) return 'bg-yellow-500'
  if (percentage.value > 25) return 'bg-orange-500'
  return 'bg-red-500'
})
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <div class="flex justify-between items-center mb-4">
      <p class="text-sm text-gray-400">KV Cache Usage</p>
      <p class="text-sm text-gray-300">{{ used }} / {{ total }} blocks</p>
    </div>
    <div class="h-4 bg-gray-700 rounded-full overflow-hidden">
      <div
        class="h-full transition-all duration-500 rounded-full"
        :class="colorClass"
        :style="{ width: `${100 - percentage}%` }"
      />
    </div>
    <div class="flex justify-between mt-2">
      <span class="text-xs text-gray-500">Used</span>
      <span class="text-xs text-gray-400">{{ 100 - percentage }}% used</span>
    </div>
  </div>
</template>
