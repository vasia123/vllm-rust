# Анализ Feature Gap: vLLM vs vllm-rust

## Резюме

Проект vllm-rust реализует ~40% функциональности vLLM. Основные пробелы: квантизация, расширенные модели, multimodal, продвинутые оптимизации.

---

## Текущее состояние vllm-rust (реализовано)

| Категория | Реализовано |
|-----------|-------------|
| **Core** | PagedAttention, continuous batching, FCFS scheduler, chunked prefill, preemption |
| **Speculative** | Draft model + verification, multi-step decode |
| **KV Cache** | Block management, prefix caching, COW, eviction |
| **Sampling** | Greedy, top-k, top-p, temperature, repetition penalty, min-p, logprobs |
| **Models** | Qwen3, Llama (2 модели) |
| **Distributed** | NCCL bindings, tensor parallelism, pipeline parallelism (framework) |
| **API** | OpenAI-compatible (completions, chat), SSE streaming |
| **Admin** | Vue.js panel, config persistence, graceful restart |

---

## P0 - Критические (без них не production-ready)

| # | Фича | Описание | Сложность | Влияние |
|---|------|----------|-----------|---------|
| 1 | **FP8 Quantization** | 8-bit float, экономия памяти 2x | High | Позволяет запускать 70B+ модели на consumer GPU |
| 2 | **GPTQ + Marlin Kernels** | 4-bit quantization с быстрыми kernels | Very High | Самый популярный формат квантизации |
| 3 | **AWQ + Marlin** | Альтернативная 4-bit квантизация | High | Хороший баланс качество/скорость |
| 4 | **CUDA Graph Optimization** | Захват и воспроизведение GPU графов | High | 10-30% снижение latency в decode |
| 5 | **Structured Output** | JSON Schema, regex, grammar constraints | Medium | Enterprise requirement для интеграций |
| 6 | **Tool/Function Calling** | Парсинг и исполнение function calls | Medium | Критично для AI агентов |

---

## P1 - Важные (конкурентный паритет)

| # | Фича | Описание | Сложность | Влияние |
|---|------|----------|-----------|---------|
| 7 | **Mistral/Mixtral Models** | MoE архитектура | High | Очень популярные модели |
| 8 | **DeepSeek V2/V3** | MoE + MLA attention | Very High | Лидеры по reasoning |
| 9 | **Gemma/Gemma2** | Google open models | Medium | Высокий спрос |
| 10 | **LoRA Adapters** | Динамическая загрузка адаптеров | High | Multiple fine-tunes с одной базы |
| 11 | **FlashInfer Backend** | Оптимизированное внимание (default в vLLM v1) | Very High | Лучшая производительность |
| 12 | **KV Cache Quantization** | INT8/FP8 для KV cache | Medium | До 4x больше контекста |
| 13 | **Sliding Window Attention** | Для очень длинных контекстов | Medium | Config уже парсится |
| 14 | **Beam Search** | Multi-hypothesis decoding | Medium | Требуется для некоторых задач |
| 15 | **Fused MoE Kernels** | Оптимизированные kernels для MoE | Very High | Prerequisite для Mixtral/DeepSeek |

---

## P2 - Средние (market differentiation)

| # | Фича | Описание | Сложность |
|---|------|----------|-----------|
| 16 | **GGUF Format** | Загрузка llama.cpp моделей | High |
| 17 | **Phi Models** | Microsoft efficient models | Medium |
| 18 | **Yi Models** | Мультиязычные модели | Medium |
| 19 | **Falcon Models** | TII open models | Medium |
| 20 | **CPU Offloading** | KV cache на CPU | High |
| 21 | **gRPC API** | High-performance RPC | Medium |
| 22 | **Multimodal: Images** | Vision-language (LLaVA, etc.) | Very High |
| 23 | **MLA Attention** | DeepSeek efficient attention | Very High |
| 24 | **Tree Attention** | Для beam search | High |
| 25 | **INT4/INT8 Quantization** | Дополнительные форматы | High |

---

## P3 - Низкие (future roadmap)

| # | Фича | Описание | Сложность |
|---|------|----------|-----------|
| 26 | **BitsAndBytes** | 8-bit optimizer-compatible | Medium |
| 27 | **BitBLAS** | Alternative backend | High |
| 28 | **Multimodal: Video** | Video understanding | Very High |
| 29 | **Multimodal: Audio** | Audio understanding | Very High |
| 30 | **Ray Executor** | Cluster management | High |
| 31 | **Disaggregated Inference** | Separate prefill/decode | Very High |
| 32 | **Embedding Models** | BERT-style encoders | Medium |
| 33 | **Reasoning Parsers** | DeepSeek R1 thinking extraction | Medium |
| 34 | **Mamba/SSM** | State-space models | High |
| 35 | **Linear Attention** | Alternative mechanism | Medium |
| 36 | **200+ Additional Models** | Complete model zoo | Ongoing |

---

## Рекомендуемый порядок реализации

### Phase 1: Quantization Foundation (P0)
```
1. FP8 Quantization → GPTQ + Marlin → AWQ + Marlin
```
**Результат**: Возможность запускать большие модели на consumer hardware

### Phase 2: Performance & Enterprise (P0-P1)
```
2. CUDA Graph → Structured Output → Tool Calling
```
**Результат**: Production-ready latency и enterprise features

### Phase 3: Model Coverage (P1)
```
3. Fused MoE Kernels → Mistral/Mixtral → DeepSeek → Gemma
```
**Результат**: Покрытие самых востребованных моделей

### Phase 4: Advanced Features (P1-P2)
```
4. LoRA → FlashInfer → KV Quantization → Multimodal
```
**Результат**: Feature parity с vLLM

---

## Ключевые зависимости

```
Fused MoE Kernels ─────┬──→ Mixtral
                       ├──→ DeepSeek V2/V3
                       └──→ Qwen-MoE

Marlin Kernels ────────┬──→ GPTQ
                       └──→ AWQ

Vision Encoders ───────┬──→ LLaVA
                       ├──→ Qwen-VL
                       └──→ InternVL

MLA Implementation ────────→ DeepSeek V2/V3
```

---

## Сложность реализации (reference)

| Компонент | vLLM размер | Оценка для Rust |
|-----------|-------------|-----------------|
| FP8 | ~47K chars | 2-3 недели |
| GPTQ Marlin | ~35K chars | 3-4 недели |
| FlashInfer | ~69K chars | 4-6 недель |
| MLA | ~82K chars | 4-6 недель |
| Fused MoE | Multiple files | 3-4 недели |
| Structured Output | ~20K chars | 1-2 недели |

---

## Критические файлы для модификации

- `crates/core/src/layers/attention.rs` - новые attention backends
- `crates/core/src/models/mod.rs` - registry новых моделей
- `crates/core/src/engine.rs` - CUDA Graph интеграция
- `crates/core/src/distributed/mod.rs` - MoE routing
- Новая директория: `crates/core/src/quantization/` - методы квантизации

---

## Метрики успеха

| Milestone | Критерий |
|-----------|----------|
| MVP Quantization | FP8 + GPTQ работают с Llama 70B |
| Production Ready | CUDA Graph + Structured Output |
| Model Parity | Top-10 HuggingFace моделей поддержаны |
| Feature Parity | LoRA + Multimodal images |
