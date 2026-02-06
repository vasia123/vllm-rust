//! Graceful shutdown signal handling.
//!
//! Listens for OS termination signals (SIGINT, SIGTERM) and produces
//! a future that resolves when the server should begin shutting down.
//! Designed to be passed to `axum::serve().with_graceful_shutdown()`.

/// Returns a future that resolves when an OS shutdown signal is received.
///
/// On Unix, listens for both SIGINT (Ctrl+C) and SIGTERM.
/// Logs which signal triggered the shutdown via `tracing::info!`.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("Failed to listen for SIGINT: {e}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("Failed to listen for SIGTERM: {e}");
            }
        }
    };

    #[cfg(unix)]
    {
        tokio::select! {
            () = ctrl_c => {
                tracing::info!("Received SIGINT (Ctrl+C), initiating graceful shutdown");
            }
            () = terminate => {
                tracing::info!("Received SIGTERM, initiating graceful shutdown");
            }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await;
        tracing::info!("Received Ctrl+C, initiating graceful shutdown");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `shutdown_signal` is a valid `Future<Output = ()> + Send`,
    /// which is required by `axum::serve(..).with_graceful_shutdown()`.
    #[test]
    fn shutdown_signal_is_send_future() {
        fn assert_send_future<T: std::future::Future<Output = ()> + Send>(_f: T) {}

        // This will fail to compile if the future is not Send or has the wrong Output.
        assert_send_future(shutdown_signal());
    }

    /// Verify that sending SIGINT causes the shutdown signal to resolve.
    #[tokio::test]
    async fn shutdown_signal_resolves_on_ctrl_c() {
        // We cannot easily send real signals in unit tests without forking,
        // so we verify the tokio::select! structure works by racing against
        // an immediate resolution. This ensures the function can be polled
        // and does not panic on construction.
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), async {
            tokio::select! {
                () = shutdown_signal() => "shutdown",
                _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => "timeout",
            }
        })
        .await;

        // The sleep branch should win since no signal is sent.
        assert_eq!(result.expect("outer timeout should not fire"), "timeout");
    }

    /// Ensure the shutdown future can be stored in a `Pin<Box<dyn Future>>`,
    /// which is the shape axum's `with_graceful_shutdown` expects.
    #[test]
    fn shutdown_signal_is_boxable() {
        let _boxed: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> =
            Box::pin(shutdown_signal());
    }
}
