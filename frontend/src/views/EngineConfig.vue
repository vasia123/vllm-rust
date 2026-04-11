<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useGpuInfo } from '@/composables/useGpuInfo'
import { useEngineConfig } from '@/composables/useEngineConfig'
import { useEstimates } from '@/composables/useEstimates'
import { useRestart } from '@/composables/useRestart'
import HardwarePanel from '@/components/config/HardwarePanel.vue'
import ConfigSlidersPanel from '@/components/config/ConfigSlidersPanel.vue'
import MemoryBreakdownBar from '@/components/config/MemoryBreakdownBar.vue'
import EstimatesPanel from '@/components/config/EstimatesPanel.vue'

const route = useRoute()
const router = useRouter()
const modelId = computed(() => (route.query.model as string) || '')

const { gpuInfo, loading: gpuLoading } = useGpuInfo()
const config = useEngineConfig(modelId, gpuInfo)
const { estimates, loading: estimateLoading } = useEstimates(config.estimateRequest)
const { restart, isRestarting, statusMessage, isReady } = useRestart()

async function handleLaunch() {
  await restart({
    model: modelId.value,
    max_requests: config.maxConcurrentRequests.value,
    dtype: config.weightDtype.value,
  })
}
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-white">Configure Engine</h2>
        <p v-if="modelId" class="text-gray-400 text-sm mt-1 font-mono">{{ modelId }}</p>
        <p v-else class="text-gray-500 text-sm mt-1">
          No model selected.
          <button @click="router.push('/admin/models')" class="text-blue-400 hover:underline">
            Select one
          </button>
        </p>
      </div>
      <button
        v-if="modelId && estimates?.model_fits"
        @click="handleLaunch"
        :disabled="isRestarting"
        class="px-5 py-2.5 rounded-lg font-medium transition-colors"
        :class="isRestarting
          ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
          : 'bg-green-600 text-white hover:bg-green-500'"
      >
        {{ isRestarting ? statusMessage : 'Apply & Launch' }}
      </button>
    </div>

    <div v-if="isReady" class="bg-green-500/20 border border-green-500/50 rounded-lg p-4">
      <p class="text-green-400 text-sm font-medium">Engine started successfully</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Left column: hardware + sliders -->
      <div class="space-y-6">
        <HardwarePanel :gpu="gpuInfo" :loading="gpuLoading" />

        <ConfigSlidersPanel
          v-if="modelId"
          v-model:maxContextLength="config.maxContextLength.value"
          v-model:maxConcurrentRequests="config.maxConcurrentRequests.value"
          v-model:gpuMemoryUtilization="config.gpuMemoryUtilization.value"
          v-model:kvCacheDtype="config.kvCacheDtype.value"
          v-model:weightDtype="config.weightDtype.value"
        />
      </div>

      <!-- Right column: estimates + memory -->
      <div class="space-y-6">
        <EstimatesPanel :estimates="estimates" :loading="estimateLoading" />
        <MemoryBreakdownBar :breakdown="estimates?.memory ?? null" />
      </div>
    </div>
  </div>
</template>
