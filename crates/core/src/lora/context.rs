//! LoRA context for passing adapter information through forward passes.

/// Context for LoRA-enabled forward passes.
///
/// This struct carries the adapter name through the model's forward pass,
/// allowing each layer to apply the appropriate LoRA adapter.
#[derive(Debug, Clone, Default)]
pub struct LoraContext {
    /// Name of the LoRA adapter to use, if any.
    pub adapter_name: Option<String>,
}

impl LoraContext {
    /// Create a context with no adapter (base model only).
    pub fn none() -> Self {
        Self { adapter_name: None }
    }

    /// Create a context with a specific adapter.
    pub fn with_adapter(name: impl Into<String>) -> Self {
        Self {
            adapter_name: Some(name.into()),
        }
    }

    /// Get the adapter name as a string slice, if set.
    pub fn adapter_name(&self) -> Option<&str> {
        self.adapter_name.as_deref()
    }

    /// Check if an adapter is specified.
    pub fn has_adapter(&self) -> bool {
        self.adapter_name.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_context_none() {
        let ctx = LoraContext::none();
        assert!(!ctx.has_adapter());
        assert!(ctx.adapter_name().is_none());
    }

    #[test]
    fn test_lora_context_with_adapter() {
        let ctx = LoraContext::with_adapter("sql-adapter");
        assert!(ctx.has_adapter());
        assert_eq!(ctx.adapter_name(), Some("sql-adapter"));
    }

    #[test]
    fn test_lora_context_default() {
        let ctx = LoraContext::default();
        assert!(!ctx.has_adapter());
    }
}
