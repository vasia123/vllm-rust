<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useModelSearch } from '@/composables/useModelSearch'
import { useModelDownload } from '@/composables/useModelDownload'
import ModelSearchBar from '@/components/models/ModelSearchBar.vue'
import ModelResultList from '@/components/models/ModelResultList.vue'
import DownloadProgressBar from '@/components/models/DownloadProgressBar.vue'

const router = useRouter()
const { query, fitsOnly, results, loading, error } = useModelSearch()
const { progress, isDownloading, download } = useModelDownload()

async function handleSelect(modelId: string) {
  if (!isDownloading.value) {
    await download(modelId)
  }
  router.push({ path: '/admin/configure', query: { model: modelId } })
}
</script>

<template>
  <div class="space-y-6">
    <div>
      <h2 class="text-2xl font-bold text-white">Model Selection</h2>
      <p class="text-gray-400 text-sm mt-1">Search HuggingFace for models that fit your GPU</p>
    </div>

    <ModelSearchBar v-model="query" :loading="loading" />

    <div class="flex items-center gap-4">
      <label class="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
        <input
          type="checkbox"
          v-model="fitsOnly"
          class="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
        />
        Show only models that fit
      </label>
      <span v-if="results.length > 0" class="text-sm text-gray-500">
        {{ results.length }} results
      </span>
    </div>

    <DownloadProgressBar :progress="progress" />

    <div v-if="error" class="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
      <p class="text-red-400 text-sm">{{ error.message }}</p>
    </div>

    <ModelResultList
      :models="results"
      :downloading-model-id="isDownloading ? progress?.model_id ?? null : null"
      @select="handleSelect"
    />
  </div>
</template>
