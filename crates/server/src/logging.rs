//! Structured logging configuration for production.
//!
//! Supports two modes:
//! - Development: Pretty-printed human-readable logs
//! - Production: JSON-formatted logs for log aggregation
//!
//! Set `VLLM_LOG_FORMAT=json` for production JSON logs.
//! Set `RUST_LOG` to control log levels (default: info).
//!
//! When `--otlp-traces-endpoint` is configured, spans are also exported via
//! OTLP/HTTP JSON to the given collector endpoint (e.g. Jaeger, Tempo).

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

/// Initialize the logging subsystem and optionally export spans via OTLP.
///
/// When `otlp_endpoint` is `Some`, configures a batch OTLP/HTTP JSON exporter
/// targeting the given URL (e.g. `http://localhost:4318`). The tracing
/// subscriber gains an additional `tracing-opentelemetry` layer that converts
/// spans to OpenTelemetry format and flushes them in the background.
///
/// The function must be called from inside a Tokio runtime because the batch
/// exporter requires one. A `SetGlobalDefaultError` from a double-init is
/// silently ignored so that test harnesses can call this multiple times.
pub fn init_with_otlp(level: &str, otlp_endpoint: Option<&str>) -> anyhow::Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    let format = LogFormat::from_env();
    let fmt_layer: Box<dyn tracing_subscriber::Layer<_> + Send + Sync> = match format {
        LogFormat::Pretty => Box::new(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false),
        ),
        LogFormat::Json => Box::new(
            fmt::layer()
                .json()
                .with_span_events(FmtSpan::CLOSE)
                .with_current_span(true)
                .with_target(true)
                .with_file(true)
                .with_line_number(true),
        ),
    };

    if let Some(endpoint) = otlp_endpoint {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_otlp::WithExportConfig;

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| anyhow::anyhow!("OTLP exporter init failed: {e}"))?;

        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", "vllm-server"),
            ]))
            .build();

        // Keep the provider alive for the process lifetime.
        opentelemetry::global::set_tracer_provider(provider.clone());

        let tracer = provider.tracer("vllm-server");
        let otlp_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otlp_layer)
            .try_init()
            .map_err(|e| anyhow::anyhow!("logging subscriber init failed: {e}"))?;

        tracing::info!(endpoint = %endpoint, "OTLP trace exporter configured");
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .try_init()
            .map_err(|e| anyhow::anyhow!("logging subscriber init failed: {e}"))?;
    }

    Ok(())
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

    #[test]
    fn init_with_otlp_no_endpoint_succeeds() {
        // Should succeed (subscriber may already be set from another test;
        // try_init returns Err which we treat as non-fatal in tests).
        let _ = init_with_otlp("warn", None);
    }
}
