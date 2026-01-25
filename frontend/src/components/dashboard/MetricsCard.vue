<script setup lang="ts">
defineProps<{
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
}>()
</script>

<template>
  <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <div class="flex items-start justify-between">
      <div>
        <p class="text-sm text-gray-400">{{ title }}</p>
        <p class="text-3xl font-bold text-white mt-2">{{ value }}</p>
        <p v-if="subtitle" class="text-sm text-gray-500 mt-1">{{ subtitle }}</p>
      </div>
      <div
        v-if="trend"
        class="w-10 h-10 rounded-lg flex items-center justify-center"
        :class="{
          'bg-green-500/20 text-green-400': trend === 'up',
          'bg-red-500/20 text-red-400': trend === 'down',
          'bg-gray-500/20 text-gray-400': trend === 'neutral',
        }"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            v-if="trend === 'up'"
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M7 17l9.5-9.5M17 17V7H7"
          />
          <path
            v-else-if="trend === 'down'"
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M17 7l-9.5 9.5M7 7v10h10"
          />
          <path
            v-else
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M5 12h14"
          />
        </svg>
      </div>
    </div>
  </div>
</template>
