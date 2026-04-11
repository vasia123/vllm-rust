<script setup lang="ts">
import type { DownloadProgress } from '@/lib/types'

defineProps<{
  progress: DownloadProgress | null
}>()
</script>

<template>
  <div v-if="progress" class="bg-gray-800 rounded-xl p-4 border border-gray-700">
    <div class="flex justify-between items-center mb-2">
      <p class="text-sm text-gray-400">
        <span v-if="progress.status === 'downloading'">Downloading</span>
        <span v-else-if="progress.status === 'complete'" class="text-green-400">Complete</span>
        <span v-else-if="progress.status === 'error'" class="text-red-400">Error</span>
      </p>
      <p class="text-sm text-gray-300 font-mono truncate max-w-[200px]">{{ progress.model_id }}</p>
    </div>

    <div v-if="progress.status === 'downloading'" class="h-2 bg-gray-700 rounded-full overflow-hidden">
      <div class="h-full bg-blue-500 rounded-full animate-pulse" style="width: 100%" />
    </div>
    <div v-else-if="progress.status === 'complete'" class="h-2 bg-gray-700 rounded-full overflow-hidden">
      <div class="h-full bg-green-500 rounded-full" style="width: 100%" />
    </div>

    <p v-if="progress.message" class="text-xs text-gray-500 mt-2">{{ progress.message }}</p>
  </div>
</template>
