//! Structured logging configuration for production.
//!
//! Supports two modes:
//! - Development: Pretty-printed human-readable logs
//! - Production: JSON-formatted logs for log aggregation
//!
//! Set `VLLM_LOG_FORMAT=json` for production JSON logs.
//! Set `RUST_LOG` to control log levels (default: info).

use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Log format mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Pretty-printed human-readable logs (default).
    Pretty,
    /// JSON-formatted logs for production.
    Json,
}

impl LogFormat {
    /// Detect log format from environment.
    pub fn from_env() -> Self {
        match std::env::var("VLLM_LOG_FORMAT")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "json" => Self::Json,
            _ => Self::Pretty,
        }
    }
}

/// Initialize the logging subsystem.
///
/// Respects the following environment variables:
/// - `RUST_LOG`: Log level filter (default: "info")
/// - `VLLM_LOG_FORMAT`: "json" for JSON output, anything else for pretty output
pub fn init() {
    init_with_level_and_format(None, LogFormat::from_env());
}

/// Initialize the logging subsystem with an explicit log level.
///
/// `level` overrides `RUST_LOG` when provided (e.g. `"debug"`, `"warn"`).
pub fn init_with_level(level: &str) {
    init_with_level_and_format(Some(level), LogFormat::from_env());
}

/// Initialize the logging subsystem with a specific format.
pub fn init_with_format(format: LogFormat) {
    init_with_level_and_format(None, format);
}

/// Initialize the logging subsystem with an explicit level and format.
pub fn init_with_level_and_format(level: Option<&str>, format: LogFormat) {
    let env_filter = if let Some(lvl) = level {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(lvl))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
    };

    match format {
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(false)
                        .with_file(false)
                        .with_line_number(false),
                )
                .init();
        }
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(
                    fmt::layer()
                        .json()
                        .with_span_events(FmtSpan::CLOSE)
                        .with_current_span(true)
                        .with_target(true)
                        .with_file(true)
                        .with_line_number(true),
                )
                .init();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_format_from_env() {
        // Default should be Pretty
        std::env::remove_var("VLLM_LOG_FORMAT");
        assert_eq!(LogFormat::from_env(), LogFormat::Pretty);

        // JSON format
        std::env::set_var("VLLM_LOG_FORMAT", "json");
        assert_eq!(LogFormat::from_env(), LogFormat::Json);

        // Case insensitive
        std::env::set_var("VLLM_LOG_FORMAT", "JSON");
        assert_eq!(LogFormat::from_env(), LogFormat::Json);

        // Cleanup
        std::env::remove_var("VLLM_LOG_FORMAT");
    }
}
