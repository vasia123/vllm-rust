//! EngineHandle - public interface for interacting with the engine.

use tokio::sync::{mpsc, oneshot};

use super::types::{
    EngineCommand, EngineError, EngineStats, GenerationRequest, GenerationResult, PauseMode,
    StreamEvent,
};

/// Handle to the inference engine, cloneable for sharing across tasks.
#[derive(Clone)]
pub struct EngineHandle {
    pub(crate) cmd_tx: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    /// Submit a generation request and wait for the complete result.
    pub async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Generate {
                request,
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)?
    }

    /// Submit a generation request and receive a stream of tokens.
    ///
    /// Returns the engine-internal request ID alongside the token stream.
    /// The caller can use this ID with [`abort`] to cancel the request
    /// early (e.g., when a client disconnects from an SSE stream).
    pub async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<(crate::request::RequestId, mpsc::Receiver<StreamEvent>), EngineError> {
        let (stream_tx, stream_rx) = mpsc::channel(64);
        let (id_tx, id_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::GenerateStream {
                request,
                stream_tx,
                request_id_tx: id_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        let request_id = id_rx.await.map_err(|_| EngineError::Shutdown)?;
        Ok((request_id, stream_rx))
    }

    /// Abort a running request, freeing its GPU resources.
    ///
    /// This is used when a client disconnects before generation completes.
    /// The engine will remove the request from the scheduler and free its
    /// KV cache blocks. If the request is not found (already completed or
    /// never existed), this is a no-op.
    pub async fn abort(&self, request_id: crate::request::RequestId) -> Result<(), EngineError> {
        self.cmd_tx
            .send(EngineCommand::Abort { request_id })
            .await
            .map_err(|_| EngineError::Shutdown)
    }

    /// Shutdown the engine.
    pub async fn shutdown(&self) -> Result<(), EngineError> {
        self.cmd_tx
            .send(EngineCommand::Shutdown)
            .await
            .map_err(|_| EngineError::Shutdown)
    }

    /// Get current engine statistics for monitoring.
    pub async fn get_stats(&self) -> Result<EngineStats, EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::GetStats {
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)
    }

    /// Pause the engine with the specified mode.
    ///
    /// - `Abort`: abort all in-flight requests immediately.
    /// - `Wait`: reject new requests, let existing ones finish.
    /// - `Keep`: freeze the scheduler; requests resume on `resume()`.
    pub async fn pause(&self, mode: PauseMode) -> Result<(), EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Pause {
                mode,
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)?
    }

    /// Resume a paused engine, accepting new requests again.
    pub async fn resume(&self) -> Result<(), EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::Resume {
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)?
    }

    /// Query whether the engine is currently paused.
    pub async fn is_paused(&self) -> Result<bool, EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::IsPaused {
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)
    }

    /// Reset (clear) the prefix cache, returning the number of evicted blocks.
    ///
    /// Evicted blocks are returned to the free pool. Returns 0 if prefix
    /// caching is not enabled.
    pub async fn reset_prefix_cache(&self) -> Result<usize, EngineError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.cmd_tx
            .send(EngineCommand::ResetPrefixCache {
                response_tx: resp_tx,
            })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        resp_rx.await.map_err(|_| EngineError::Shutdown)?
    }
}
