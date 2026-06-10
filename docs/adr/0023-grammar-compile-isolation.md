# 0023 — Grammar-compile isolation: async path, single-flight, cooperative deadline (patched upstream)

Date: 2026-06-10
Status: accepted

## Context

A production incident (dnd-llm repro, `repro-vllm-rust-repetition-perf.sh`)
showed that one pathological EBNF grammar can take the whole server down
until restart:

- A chain of ~40 optional rules (`strchar? ×40`) explodes xgrammar's
  token-mask precompute to **37.7s** on a 262k-token vocabulary (the cost
  is FSM-states × vocab; the state count grows ~O(n²) in the number of
  optionals). The same grammar compiles in microseconds on a 256-token
  vocab — the blow-up is invisible in small-vocab tests.
- The server compiled grammars with `compile_sync` **inside the async
  axum handler**: the tokio worker is pinned for the whole compile (a
  future cannot be cancelled while `poll` is inside FFI), the client
  timeout cancels nothing, and upstream xgrammar runs each compile on
  its own 8-thread pool. A handful of timed-out retries left the server
  at 1500% CPU with an empty queue and every request degraded ~3.5×.
- Upstream `xgrammar::GrammarCompiler` has **no cancellation API**: once
  `MultiThreadCompileGrammar` starts, it runs to completion no matter
  what the caller does.

## Decision

Defense in depth, three layers:

1. **Server/core (vllm-rust):** `GrammarCompiler::compile` (async) is the
   only production path. It provides: template cache by spec hash;
   **single-flight** (concurrent identical specs await one compile via a
   per-hash `tokio::sync::OnceCell`); a **2-slot semaphore** bounding
   concurrent compiles; the vocabulary marshal and the compile both run
   in `spawn_blocking`. `compile_sync` is demoted to test/diagnostic use.

2. **Cooperative deadline inside upstream (xgrammar-rs):** we patch the
   pinned upstream source at build time
   (`xgrammar-rs/patches/0001-compile-deadline.patch`, applied by
   build.rs after checkout — the tree is reset to the pristine pin first,
   so the step is idempotent; a non-applying patch fails the build
   loudly). The patch reads a `thread_local` deadline at the start of
   `MultiThreadCompileGrammar`, snapshots it into the worker lambdas, and
   makes per-state tasks **skip** their work past the deadline (the
   queued tasks drain in microseconds); if anything was skipped the
   compile **throws** instead of caching a partial mask set. The shim
   exposes `xgr_set_compile_deadline_ms`; the Rust API arms it via an
   RAII guard in `compile_*_with_timeout`.

   Why thread-local: upstream's compile cache computes values on the
   calling thread (`ThreadSafeLRUCache` runs the packaged task inline),
   so a TLS value set by the shim immediately before `xgr_compile_*` is
   visible exactly where it must be, with zero API churn through the
   cache's key/computer plumbing.

3. **Bounded compiler resources:** the shim now exposes the upstream
   constructor parameters (`xgr_compiler_new_ex`); vllm-core builds the
   compiler with `max_threads = 4` (upstream default 8) and
   `max_memory_bytes = 256MB` for the upstream caches (a single
   pathological grammar's masks reach tens of MB at a 262k vocab —
   "unlimited" is an OOM vector on an 8-10GB host).

The end-to-end budget is `--grammar-compile-timeout-secs` (default 15s):
it bounds the semaphore queue wait, is armed as the C++ deadline, and a
`+5s` hard-cap watchdog in the async layer covers the cooperative
mechanism failing to fire.

## Consequences

- A pathological grammar now costs at most one timeout window of
  bounded CPU (≤ 2 slots × 4 threads), returns a clear HTTP 400, and the
  threads are actually reclaimed — no permanent CPU burn, no restart.
- A deadline-aborted spec leaves an exception in upstream's
  `shared_future` cache: identical retries fail **instantly** instead of
  recompiling. This is deliberate (recompiling would explode again); the
  flip side is that raising the server timeout for a borderline grammar
  requires a process restart for that spec to be retried.
- The deadline only bounds a compile initiated by the calling thread;
  waiting on another thread's in-progress compile of the same spec is
  unbounded upstream. Our single-flight layer makes that case unreachable
  in the server.
- Bumping `XGRAMMAR_COMMIT` requires re-validating the patch (build
  fails loudly on `git apply`). The patch is ~30 lines and marked with
  `[xgrammar-rs patch: compile deadline]` comments.
- Overshoot granularity is one token-mask state task per worker thread
  (~20-25ms each at 262k vocab) — observed abort latency is within
  seconds of the deadline, not minutes.
