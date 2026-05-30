//! Rust harness over `scripts/test_grammar_strictness_e2e.sh` so a
//! single `cargo test` invocation can run the end-to-end xgrammar
//! enforcement smoke against a real model.
//!
//! The test is `#[ignore]` by default — running it spins up the
//! release server binary and a real HF model. Opt in via:
//!
//! ```bash
//! VLLM_E2E_GRAMMAR=1 cargo test -p vllm-server --release \
//!     --features cuda-full --test grammar_strictness_e2e \
//!     -- --ignored --nocapture
//! ```
//!
//! Per the shell script's contract: missing model in the HF cache is
//! a SKIP (exit 0) unless `VLLM_E2E_GRAMMAR_STRICT=1`. So this test
//! turns "script exit 0" into pass; non-zero into fail. Stdout and
//! stderr stream through `--nocapture`.

use std::path::PathBuf;
use std::process::Command;

/// Locate the runner script. Relative to the workspace root, which
/// `cargo test` invokes the binary from.
fn script_path() -> PathBuf {
    // CARGO_MANIFEST_DIR is .../crates/server; walk up to workspace root.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("scripts/test_grammar_strictness_e2e.sh");
    p
}

#[test]
#[ignore = "spins up release server + real model; opt in via `cargo test -- --ignored` AND set VLLM_E2E_GRAMMAR=1"]
fn xgrammar_strictness_end_to_end() {
    if std::env::var("VLLM_E2E_GRAMMAR").ok().as_deref() != Some("1") {
        // Belt-and-suspenders: even with --ignored someone might run
        // this without realising the cost. Demand the explicit env.
        eprintln!(
            "skipping: set VLLM_E2E_GRAMMAR=1 to actually run the e2e \
             grammar strictness check (it boots a release server)."
        );
        return;
    }

    let script = script_path();
    assert!(
        script.exists(),
        "expected runner script at {} — was scripts/test_grammar_strictness_e2e.sh moved?",
        script.display()
    );

    let status = Command::new("bash")
        .arg(&script)
        .status()
        .expect("failed to spawn bash runner");

    assert!(
        status.success(),
        "{} exited with {:?}",
        script.display(),
        status.code()
    );
}
