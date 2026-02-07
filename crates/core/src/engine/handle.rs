//! EngineHandle - public interface for interacting with the engine.

use tokio::sync::{mpsc, oneshot};

use super::types::{
    EngineCommand, EngineError, EngineStats, GenerationRequest, GenerationResult, StreamEvent,
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
    pub async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<StreamEvent>, EngineError> {
        let (stream_tx, stream_rx) = mpsc::channel(64);
        self.cmd_tx
            .send(EngineCommand::GenerateStream { request, stream_tx })
            .await
            .map_err(|_| EngineError::Shutdown)?;
        Ok(stream_rx)
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
}
