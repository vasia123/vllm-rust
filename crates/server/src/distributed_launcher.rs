//! Multi-process launcher for pipeline parallelism.
//!
//! When `--pipeline-parallel-size N` is requested, rank 0 spawns N-1 worker
//! processes (ranks 1..N-1) that run the pipeline worker loop instead of the
//! HTTP server. Each worker is a re-execution of the same binary with the same
//! CLI arguments, distinguished by the standard distributed environment variables.
//!
//! # Environment Variables (standard distributed training convention)
//!
//! | Variable      | Set by launcher | Consumed by |
//! |---------------|----------------|-------------|
//! | `RANK`        | 1..N-1         | `DistributedConfig::from_env()` |
//! | `WORLD_SIZE`  | N              | `DistributedConfig::from_env()` |
//! | `LOCAL_RANK`  | 1..N-1         | `DistributedConfig::from_env()` |
//! | `MASTER_ADDR` | 127.0.0.1      | NCCL bootstrap TCP |
//! | `MASTER_PORT` | 29500          | NCCL bootstrap TCP |
//!
//! Workers detect their role via `is_worker_process()` which returns `true` when
//! `RANK > 0`.

use std::process::{Child, Command};

/// Return `true` when this process is a pipeline worker (RANK > 0).
///
/// Called at the top of `run_server` to branch into the worker code path
/// instead of the full HTTP server path.
// Only called from #[cfg(feature = "cuda")] code; suppress the warning in
// non-cuda builds where the call site is compiled out.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn is_worker_process() -> bool {
    std::env::var("RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|r| r > 0)
        .unwrap_or(false)
}

/// Spawn `world_size - 1` worker processes (ranks 1..world_size).
///
/// Each worker is a re-execution of the current binary with the same CLI
/// arguments plus the standard distributed environment variables. Workers
/// detect their role via `is_worker_process()` and call
/// `pipeline_worker_loop` instead of starting the HTTP server.
///
/// # Arguments
/// * `world_size`   — total number of pipeline stages (= pp_size)
/// * `master_port`  — TCP port used by NCCL for unique-ID broadcast
///
/// # Returns
/// Handles to the spawned child processes so the coordinator can wait for
/// them on shutdown or propagate signals.
pub fn spawn_pipeline_workers(world_size: usize, master_port: u16) -> anyhow::Result<Vec<Child>> {
    assert!(world_size > 1, "no workers to spawn for world_size=1");

    let current_exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("failed to determine current executable: {e}"))?;

    // Forward the same CLI arguments to every worker so they parse the same
    // model ID, dtype, cache config, etc.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

    let mut workers = Vec::with_capacity(world_size - 1);
    for rank in 1..world_size {
        tracing::info!(rank, world_size, master_port, "spawning pipeline worker");

        let child = Command::new(&current_exe)
            .args(&args)
            .env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            // Assumes single-node (all ranks on same machine). LOCAL_RANK == RANK
            // maps each pipeline stage to a distinct GPU device ordinal.
            .env("LOCAL_RANK", rank.to_string())
            .env("MASTER_ADDR", "127.0.0.1")
            .env("MASTER_PORT", master_port.to_string())
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn worker rank {rank}: {e}"))?;

        workers.push(child);
    }

    tracing::info!(count = world_size - 1, "all pipeline workers spawned");
    Ok(workers)
}

/// Wait for all worker processes to exit and collect their exit statuses.
///
/// Called by the coordinator on shutdown to ensure workers are torn down
/// gracefully. Non-zero exit codes are logged as warnings but do not cause
/// an error here — the coordinator's own shutdown already propagated the
/// shutdown signal via `SIGNAL_SHUTDOWN`.
pub fn wait_for_workers(mut workers: Vec<Child>) {
    for (i, child) in workers.iter_mut().enumerate() {
        match child.wait() {
            Ok(status) if status.success() => {
                tracing::debug!(rank = i + 1, "pipeline worker exited cleanly");
            }
            Ok(status) => {
                tracing::warn!(
                    rank = i + 1,
                    ?status,
                    "pipeline worker exited with non-zero status"
                );
            }
            Err(e) => {
                tracing::warn!(rank = i + 1, error = %e, "error waiting for pipeline worker");
            }
        }
    }
}
