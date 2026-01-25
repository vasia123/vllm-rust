<script setup lang="ts">
import { ref, watch } from 'vue'
import { useConfig } from '@/composables/useConfig'
import type { RuntimeConfig } from '@/lib/types'

const { config, loading, error, saving, saveResult, save } = useConfig()
const showAdvanced = ref(false)

// Local editable copy of config
const editableConfig = ref<RuntimeConfig | null>(null)

// Watch for config changes to initialize editable copy
watch(config, (newConfig) => {
  if (newConfig && !editableConfig.value) {
    editableConfig.value = { ...newConfig }
  }
}, { immediate: true })

const formatValue = (value: unknown) => {
  if (typeof value === 'boolean') return value ? 'Enabled' : 'Disabled'
  if (value === null || value === undefined) return 'Not set'
  return String(value)
}

async function handleSave() {
  if (editableConfig.value) {
    await save(editableConfig.value)
  }
}
</script>

<template>
  <div>
    <!-- Error message -->
    <div v-if="error" class="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-6">
      <p class="text-red-400">{{ error.message }}</p>
    </div>

    <!-- Success message -->
    <div v-if="saveResult?.success" class="bg-green-500/20 border border-green-500 rounded-lg p-4 mb-6">
      <p class="text-green-400 font-medium">{{ saveResult.message }}</p>
      <p class="text-green-400/80 text-sm mt-1">Saved to: {{ saveResult.path }}</p>
    </div>

    <div v-if="loading && !config" class="flex items-center justify-center h-64">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
    </div>

    <template v-else-if="editableConfig">
      <!-- Model Configuration -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
        <h3 class="text-lg font-medium text-white mb-4">Model Configuration</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label class="block text-sm text-gray-400 mb-2">Current Model</label>
            <input
              v-model="editableConfig.model"
              type="text"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 font-mono text-sm border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">HuggingFace model identifier</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Draft Model</label>
            <input
              v-model="editableConfig.draft_model"
              type="text"
              placeholder="Not configured"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 font-mono text-sm border border-gray-700 focus:border-blue-500 focus:outline-none placeholder-gray-600"
            />
            <p class="text-xs text-gray-500 mt-1">For speculative decoding</p>
          </div>
        </div>
      </div>

      <!-- Basic Settings -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
        <h3 class="text-lg font-medium text-white mb-4">Basic Settings</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label class="block text-sm text-gray-400 mb-2">Max Concurrent Requests</label>
            <input
              v-model.number="editableConfig.max_requests"
              type="number"
              min="1"
              max="64"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">Maximum parallel requests</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">KV Cache Blocks</label>
            <input
              v-model.number="editableConfig.num_blocks"
              type="number"
              min="64"
              max="8192"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">Total memory blocks</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Block Size</label>
            <div class="bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700">
              {{ editableConfig.block_size }} tokens
            </div>
            <p class="text-xs text-gray-500 mt-1">Tokens per block (read-only)</p>
          </div>
        </div>
      </div>

      <!-- Advanced Settings -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
        <button
          @click="showAdvanced = !showAdvanced"
          class="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
        >
          <svg
            class="w-4 h-4 transition-transform"
            :class="{ 'rotate-90': showAdvanced }"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
          <span class="text-lg font-medium">Advanced Settings</span>
        </button>

        <div v-show="showAdvanced" class="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div>
            <label class="block text-sm text-gray-400 mb-2">Max Tokens per Step</label>
            <input
              v-model.number="editableConfig.max_tokens_per_step"
              type="number"
              min="256"
              max="16384"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">Scheduler token budget</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Multi-Step Count</label>
            <input
              v-model.number="editableConfig.multi_step_count"
              type="number"
              min="1"
              max="16"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">Decode steps per scheduling</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Speculative Tokens</label>
            <input
              v-model.number="editableConfig.num_speculative_tokens"
              type="number"
              min="1"
              max="16"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-gray-200 border border-gray-700 focus:border-blue-500 focus:outline-none"
            />
            <p class="text-xs text-gray-500 mt-1">Tokens to speculate</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Chunked Prefill</label>
            <button
              @click="editableConfig.enable_chunked_prefill = !editableConfig.enable_chunked_prefill"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-left border border-gray-700 hover:border-gray-600 transition-colors"
              :class="editableConfig.enable_chunked_prefill ? 'text-green-400' : 'text-gray-500'"
            >
              {{ formatValue(editableConfig.enable_chunked_prefill) }}
            </button>
            <p class="text-xs text-gray-500 mt-1">Split long prompts</p>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Prefix Caching</label>
            <button
              @click="editableConfig.enable_prefix_caching = !editableConfig.enable_prefix_caching"
              class="w-full bg-gray-900 rounded-lg px-4 py-3 text-left border border-gray-700 hover:border-gray-600 transition-colors"
              :class="editableConfig.enable_prefix_caching ? 'text-green-400' : 'text-gray-500'"
            >
              {{ formatValue(editableConfig.enable_prefix_caching) }}
            </button>
            <p class="text-xs text-gray-500 mt-1">Reuse cached prefixes</p>
          </div>
        </div>
      </div>

      <!-- System Info -->
      <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-6">
        <h3 class="text-lg font-medium text-white mb-4">System Information</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label class="block text-sm text-gray-400 mb-2">Device</label>
            <div class="bg-gray-900 rounded-lg px-4 py-3 text-gray-200 font-mono border border-gray-700">
              {{ editableConfig.device }}
            </div>
          </div>
          <div>
            <label class="block text-sm text-gray-400 mb-2">Data Type</label>
            <div class="bg-gray-900 rounded-lg px-4 py-3 text-gray-200 font-mono border border-gray-700">
              {{ editableConfig.dtype }}
            </div>
          </div>
        </div>
      </div>

      <!-- Save button -->
      <div class="flex items-center gap-4">
        <button
          @click="handleSave"
          :disabled="saving"
          class="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 disabled:cursor-not-allowed text-white font-medium px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
        >
          <svg v-if="saving" class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>{{ saving ? 'Saving...' : 'Save Configuration' }}</span>
        </button>
        <p class="text-sm text-gray-400">
          Changes will be applied on next server restart.
        </p>
      </div>
    </template>
  </div>
</template>
