<script setup lang="ts">
import { computed } from 'vue'
import MetricsCard from '@/components/dashboard/MetricsCard.vue'
import CacheUsageBar from '@/components/dashboard/CacheUsageBar.vue'
import SystemInfo from '@/components/dashboard/SystemInfo.vue'
import { useMetrics } from '@/composables/useMetrics'
import { useConfig } from '@/composables/useConfig'

const { metrics, loading, error } = useMetrics()
const { config } = useConfig()

const hitRate = computed(() => {
  if (!metrics.value?.kv_cache.hit_rate) return 'N/A'
  return `${(metrics.value.kv_cache.hit_rate * 100).toFixed(1)}%`
})
</script>

<template>
  <div>
    <div v-if="error" class="bg-red-500/20 border border-red-500 rounded-lg p-4 mb-6">
      <p class="text-red-400">{{ error.message }}</p>
    </div>

    <div v-if="loading && !metrics" class="flex items-center justify-center h-64">
      <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
    </div>

    <template v-else-if="metrics">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <MetricsCard
          title="Running Requests"
          :value="metrics.running_requests"
          subtitle="Currently processing"
          trend="neutral"
        />
        <MetricsCard
          title="Waiting Requests"
          :value="metrics.waiting_requests"
          subtitle="In queue"
          :trend="metrics.waiting_requests > 0 ? 'up' : 'neutral'"
        />
        <MetricsCard
          title="Cache Hit Rate"
          :value="hitRate"
          subtitle="Prefix cache efficiency"
          :trend="(metrics.kv_cache.hit_rate || 0) > 0.5 ? 'up' : 'neutral'"
        />
        <MetricsCard
          title="Blocks Evicted"
          :value="metrics.kv_cache.blocks_evicted"
          subtitle="Total evictions"
          :trend="metrics.kv_cache.blocks_evicted > 0 ? 'down' : 'neutral'"
        />
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2">
          <CacheUsageBar
            :used="metrics.num_total_blocks - metrics.num_free_blocks"
            :total="metrics.num_total_blocks"
          />
        </div>
        <SystemInfo
          :model-id="metrics.model_id"
          :uptime="metrics.uptime_seconds"
          :device="config?.device || 'cuda:0'"
          :dtype="config?.dtype || 'bf16'"
        />
      </div>

      <div class="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 class="text-sm font-medium text-gray-400 mb-4">Allocation Statistics</h3>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-gray-500">Total Allocations</span>
              <span class="text-gray-200">{{ metrics.kv_cache.allocations }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">Blocks Allocated</span>
              <span class="text-gray-200">{{ metrics.kv_cache.blocks_allocated }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">Blocks Freed</span>
              <span class="text-gray-200">{{ metrics.kv_cache.blocks_freed }}</span>
            </div>
          </div>
        </div>

        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 class="text-sm font-medium text-gray-400 mb-4">Cache Statistics</h3>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-gray-500">Cache Queries</span>
              <span class="text-gray-200">{{ metrics.kv_cache.cache_queries }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">Cache Hits</span>
              <span class="text-green-400">{{ metrics.kv_cache.cache_hits }}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500">Cache Misses</span>
              <span class="text-red-400">{{ metrics.kv_cache.cache_misses }}</span>
            </div>
            <template v-if="metrics.prefix_cache_stats">
              <div class="flex justify-between">
                <span class="text-gray-500">Cached Blocks</span>
                <span class="text-gray-200">{{ metrics.prefix_cache_stats.cached_blocks }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-500">Evictable Blocks</span>
                <span class="text-gray-200">{{ metrics.prefix_cache_stats.evictable_blocks }}</span>
              </div>
            </template>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
