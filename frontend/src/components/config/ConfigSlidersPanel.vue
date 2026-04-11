<script setup lang="ts">
import ConfigSlider from './ConfigSlider.vue'

const maxContextLength = defineModel<number>('maxContextLength', { required: true })
const maxConcurrentRequests = defineModel<number>('maxConcurrentRequests', { required: true })
const gpuMemoryUtilization = defineModel<number>('gpuMemoryUtilization', { required: true })
const kvCacheDtype = defineModel<string>('kvCacheDtype', { required: true })
const weightDtype = defineModel<string>('weightDtype', { required: true })
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 space-y-5">
    <h3 class="text-sm text-gray-400 mb-2">Configuration</h3>

    <ConfigSlider
      v-model="maxContextLength"
      :min="512"
      :max="131072"
      label="Max Context Length"
      :log-scale="true"
    />

    <ConfigSlider
      v-model="maxConcurrentRequests"
      :min="1"
      :max="64"
      :step="1"
      label="Max Concurrent Requests"
    />

    <ConfigSlider
      v-model="gpuMemoryUtilization"
      :min="0.5"
      :max="0.95"
      :step="0.05"
      label="GPU Memory Utilization"
      :format-value="(v: number) => `${(v * 100).toFixed(0)}%`"
    />

    <!-- Weight dtype selector -->
    <div>
      <label class="text-sm text-gray-300 block mb-1.5">Weight Precision</label>
      <div class="flex gap-2">
        <button
          v-for="dt in ['bf16', 'fp8', 'int8', 'int4']"
          :key="dt"
          @click="weightDtype = dt"
          class="px-3 py-1.5 rounded-lg text-xs font-mono transition-colors"
          :class="weightDtype === dt
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
        >
          {{ dt.toUpperCase() }}
        </button>
      </div>
    </div>

    <!-- KV cache dtype toggle -->
    <div>
      <label class="text-sm text-gray-300 block mb-1.5">KV Cache Precision</label>
      <div class="flex gap-2">
        <button
          v-for="dt in ['auto', 'fp8']"
          :key="dt"
          @click="kvCacheDtype = dt"
          class="px-3 py-1.5 rounded-lg text-xs font-mono transition-colors"
          :class="kvCacheDtype === dt
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
        >
          {{ dt === 'auto' ? 'Auto (FP16)' : 'FP8' }}
        </button>
      </div>
    </div>
  </div>
</template>
