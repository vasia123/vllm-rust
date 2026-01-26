# Анализ архитектурной готовности vllm-rust

## Резюме

Проект **на 70% готов к расширению**, но требует **P0 рефакторинга** в engine.rs и attention.rs **ПЕРЕД** добавлением ключевых фич.

---

## Матрица готовности компонентов

| Компонент | Состояние | Готовность | Рефакторинг? |
|-----------|-----------|------------|--------------|
| **Models Registry** | Good | 7/10 | Не срочно |
| **Attention Backend** | Basic | 5/10 | **ДА (P0)** |
| **Quantization** | None | 0/10 | **ДА (P0)** |
| **KV Cache** | Excellent | 9/10 | Минорный |
| **Engine Loop** | Works | 4/10 | **ДА (P0)** |
| **CUDA Graphs** | None | 0/10 | **ДА (P0)** |
| **Distributed** | Framework | 5/10 | **ДА (P1)** |
| **Tool Calling** | None | 0/10 | **ДА (P1)** |
| **Structured Output** | None | 0/10 | **ДА (P1)** |

---

## P0 Рефакторинг (обязателен перед фичами)

### 1. Engine Loop Extraction

**Проблема:** `engine.rs` (2966 LOC) содержит `engine_loop()` и `engine_loop_speculative()` с 80% дублирования кода.

**Решение:**
```rust
// Новый модуль: crates/core/src/engine/executor.rs

pub struct ExecutionContext {
    pub scheduler: Scheduler,
    pub requests: HashMap<RequestId, ActiveRequest>,
    pub kv_cache_mgr: KVCacheManager,
}

pub trait ExecutionStrategy: Send {
    async fn step(&self, ctx: &mut ExecutionContext) -> Result<SchedulerOutput>;
}

pub struct StandardExecution;
pub struct SpeculativeExecution { draft_model: M }

impl ExecutionStrategy for StandardExecution { /* текущий engine_loop */ }
impl ExecutionStrategy for SpeculativeExecution { /* текущий speculative_loop */ }
```

**Файлы:** `crates/core/src/engine.rs` → split into `engine/mod.rs`, `engine/executor.rs`, `engine/context.rs`

---

### 2. Attention Backend Abstraction

**Проблема:** `attention.rs` (490 LOC) — монолитная реализация, невозможно добавить FlashInfer/MLA без хирургии.

**Решение:**
```rust
// Новый модуль: crates/core/src/layers/attention/backend.rs

pub trait AttentionBackend: Send + Sync {
    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        config: &AttentionConfig,
    ) -> Result<Tensor>;

    fn batched_decode(
        &self,
        q: &Tensor,
        kv_cache: &KVCache,
        block_tables: &[Vec<BlockId>],
    ) -> Result<Tensor>;
}

// Implementations:
pub struct PagedAttentionBackend;      // текущая реализация
pub struct FlashInferBackend;          // future
pub struct MlaBackend;                 // future (DeepSeek)
```

**Файлы:** `crates/core/src/layers/attention.rs` → `layers/attention/mod.rs`, `layers/attention/paged.rs`, `layers/attention/backend.rs`

---

### 3. Quantization Infrastructure

**Проблема:** Полностью отсутствует инфраструктура для квантизации.

**Решение:**
```rust
// config.rs — добавить
#[derive(Deserialize, Clone, Default)]
pub struct QuantizationConfig {
    pub method: String,        // "fp8", "gptq", "awq"
    pub bits: usize,           // 4, 8
    pub group_size: Option<usize>,
    pub desc_act: bool,        // GPTQ specific
}

// ModelConfig — добавить поле
#[serde(default)]
pub quantization: Option<QuantizationConfig>,
```

```rust
// Новый модуль: crates/core/src/quantization/mod.rs

pub trait QuantizedLinear: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

pub struct FP8Linear { ... }
pub struct GptqLinear { ... }
pub struct AwqLinear { ... }

pub fn load_quantized_weights(
    config: &QuantizationConfig,
    tensors: &SafeTensors,
) -> Result<Box<dyn QuantizedLinear>>;
```

**Новые файлы:**
- `crates/core/src/quantization/mod.rs`
- `crates/core/src/quantization/fp8.rs`
- `crates/core/src/quantization/gptq.rs`
- `crates/core/src/quantization/awq.rs`

---

## P1 Рефакторинг (перед enterprise фичами)

### 4. Model Composition

**Проблема:** Qwen3 и Llama имеют ~60% идентичного кода с hardcoded layers.

**Решение:**
```rust
// Новый общий блок
pub struct TransformerBlock {
    attention: Box<dyn AttentionLayer>,
    mlp: Box<dyn MLPLayer>,
    pre_norm: RmsNorm,
    post_norm: RmsNorm,
}

pub struct TransformerModel {
    embeddings: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    lm_head: Linear,
}

// Конфигурация определяет attention/mlp типы
impl TransformerModel {
    pub fn from_config(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let attention_type = cfg.attention_type(); // "standard", "mla", "moe"
        let mlp_type = cfg.mlp_type();             // "swiglu", "moe"
        // ...
    }
}
```

---

### 5. Constraint Sampling Framework

**Проблема:** Нет инфраструктуры для structured output и tool calling.

**Решение:**
```rust
// Новый модуль: crates/core/src/sampling/constraint.rs

pub trait SamplingConstraint: Send + Sync {
    /// Фильтрует logits на позиции position
    fn filter_logits(&self, logits: &mut Tensor, position: usize) -> Result<()>;

    /// Проверяет, завершена ли генерация
    fn is_complete(&self, tokens: &[u32]) -> bool;
}

pub struct JsonSchemaConstraint {
    schema: serde_json::Value,
    state_machine: JsonStateMachine,
}

pub struct RegexConstraint {
    pattern: regex::Regex,
    dfa: DFA,
}

pub struct ToolCallingConstraint {
    tools: Vec<ToolDefinition>,
    state: ToolCallState,
}
```

---

### 6. Distributed Integration

**Проблема:** Distributed framework существует, но не интегрирован с engine.

**Решение:**
```rust
// engine/context.rs — добавить
pub struct ExecutionContext {
    // existing...
    pub process_group: Option<ProcessGroup>,
    pub tensor_parallel_size: usize,
    pub pipeline_parallel_size: usize,
}

// models — использовать distributed layers
impl Qwen3ForCausalLM {
    fn new_distributed(
        cfg: &ModelConfig,
        vb: VarBuilder,
        tp_group: &ProcessGroup,
    ) -> Result<Self> {
        // Использовать ColumnParallelLinear, RowParallelLinear
    }
}
```

---

## Архитектурные решения (ADR)

### ADR-001: Attention Backend Strategy

**Контекст:** Нужна поддержка FlashInfer, MLA, Mamba attention.

**Решение:** Trait-based polymorphism с runtime dispatch.

**Обоснование:**
- Compile-time generics приведут к code bloat
- Runtime dispatch позволяет config-based выбор backend
- Незначительный overhead (~1-2%)

**Когда:** Перед FlashInfer или MLA реализацией

---

### ADR-002: Execution Strategy Pattern

**Контекст:** Engine loop дублируется для speculative decoding.

**Решение:** Strategy pattern с shared ExecutionContext.

**Обоснование:**
- Единая точка входа для всех execution modes
- Легче добавить: chunked prefill, disaggregated, etc.
- Тестируемость: можно мокать strategies

**Когда:** При следующем изменении engine.rs

---

### ADR-003: Quantization as First-Class Citizen

**Контекст:** Квантизация критична для production (70B+ модели).

**Решение:** Отдельный модуль `quantization/` с trait QuantizedLinear.

**Обоснование:**
- Квантизация влияет на: loader, layers, config
- Изоляция позволяет добавлять новые методы без касания core
- Унификация API для всех методов

**Когда:** P0 — до любой работы над фичами

---

## План рефакторинга

### Phase 0: Подготовка (2-3 недели)

| Неделя | Задача | Файлы |
|--------|--------|-------|
| 1 | Extraction ExecutionStrategy | `engine.rs` → split |
| 1-2 | AttentionBackend trait | `attention.rs` → split |
| 2-3 | Quantization infrastructure | New `quantization/` |

### После рефакторинга

| Неделя | Фича | Зависит от |
|--------|------|------------|
| 4-5 | FP8 Quantization | Quantization infra |
| 6 | CUDA Graph | Engine refactor |
| 7-8 | GPTQ + Marlin | Quantization infra |
| 9-10 | FlashInfer | AttentionBackend |

---

## Что хорошо (не трогать)

| Компонент | Причина |
|-----------|---------|
| **KV Cache** | Production-ready, хорошая абстракция |
| **Scheduler** | Clean, extensible, хорошо протестирован |
| **Request handling** | Well-structured state machine |
| **Distributed primitives** | NCCL bindings работают |

---

## Риски

| Риск | Митигация |
|------|-----------|
| Рефакторинг сломает существующее | Полное test coverage перед рефакторингом |
| Рефакторинг затянется | Фиксированный scope, 2-недельные sprints |
| Новая архитектура не масштабируется | ADR review перед реализацией |

---

## Вывод

**Рекомендация:** Начать с Phase 0 (рефакторинг) перед добавлением P0 фич.

Без рефакторинга:
- FP8: возможно, но грязно (hardcode в loader)
- FlashInfer: невозможно без AttentionBackend
- Mixtral: невозможно без MoE infrastructure
- Tool Calling: невозможно без Constraint Sampling

С рефакторингом (3 недели инвестиции):
- Все P0-P1 фичи реализуемы
- Код maintainable
- Легче онбордить контрибьюторов
