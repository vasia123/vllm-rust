<script setup lang="ts">
import type { HfModelResult } from '@/lib/types'
import ModelResultCard from './ModelResultCard.vue'

defineProps<{
  models: HfModelResult[]
  downloadingModelId?: string | null
}>()

const emit = defineEmits<{
  select: [modelId: string]
  download: [modelId: string]
}>()
</script>

<template>
  <div class="space-y-2 max-h-[600px] overflow-y-auto pr-1">
    <ModelResultCard
      v-for="model in models"
      :key="model.id"
      :model="model"
      :is-downloading="model.id === downloadingModelId"
      @select="emit('select', $event)"
      @download="emit('download', $event)"
    />
    <p v-if="models.length === 0" class="text-center text-gray-500 py-8">
      No models found
    </p>
  </div>
</template>
