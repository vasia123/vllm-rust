<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    modelValue: number
    min: number
    max: number
    step?: number
    label: string
    unit?: string
    logScale?: boolean
    formatValue?: (v: number) => string
  }>(),
  {
    step: 1,
    unit: '',
    logScale: false,
  },
)

const emit = defineEmits<{
  'update:modelValue': [value: number]
}>()

// For log scale: map linear [0, 100] to exponential [min, max]
const SLIDER_STEPS = 100

const sliderValue = computed(() => {
  if (!props.logScale) return props.modelValue
  if (props.min <= 0 || props.max <= 0) return 50
  const logMin = Math.log2(props.min)
  const logMax = Math.log2(props.max)
  return ((Math.log2(props.modelValue) - logMin) / (logMax - logMin)) * SLIDER_STEPS
})

function onInput(event: Event) {
  const raw = Number((event.target as HTMLInputElement).value)
  if (props.logScale) {
    const logMin = Math.log2(props.min)
    const logMax = Math.log2(props.max)
    const logVal = logMin + (raw / SLIDER_STEPS) * (logMax - logMin)
    // Snap to nearest power of 2
    const snapped = Math.pow(2, Math.round(logVal))
    emit('update:modelValue', Math.min(Math.max(snapped, props.min), props.max))
  } else {
    emit('update:modelValue', raw)
  }
}

const displayValue = computed(() => {
  if (props.formatValue) return props.formatValue(props.modelValue)
  if (props.modelValue >= 1024 && props.unit === '') {
    return `${(props.modelValue / 1024).toFixed(0)}K`
  }
  return `${props.modelValue}`
})
</script>

<template>
  <div>
    <div class="flex justify-between items-center mb-1.5">
      <label class="text-sm text-gray-300">{{ label }}</label>
      <span class="text-sm font-mono text-white">{{ displayValue }}{{ unit ? ` ${unit}` : '' }}</span>
    </div>
    <input
      type="range"
      :value="logScale ? sliderValue : modelValue"
      :min="logScale ? 0 : min"
      :max="logScale ? SLIDER_STEPS : max"
      :step="logScale ? 1 : step"
      @input="onInput"
      class="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
    />
    <div class="flex justify-between text-xs text-gray-500 mt-0.5">
      <span>{{ logScale && min >= 1024 ? `${min / 1024}K` : min }}</span>
      <span>{{ logScale && max >= 1024 ? `${max / 1024}K` : max }}</span>
    </div>
  </div>
</template>
