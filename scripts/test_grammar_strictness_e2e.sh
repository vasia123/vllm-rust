#!/usr/bin/env bash
# End-to-end smoke test for xgrammar JSON-Schema enforcement.
#
# Spins up the release vllm-server binary against a real model,
# fires the adversarial /tmp/grammar_strictness_check.py reproducer
# (or a copy in the repo), and asserts 5/5 strict JSON conformance.
#
# Skips (exit 0) when the configured model isn't present in the
# local HF cache — so CI on a fresh machine doesn't get a spurious
# fail. Set `VLLM_E2E_GRAMMAR_STRICT=1` to upgrade "model missing"
# to a hard fail.
#
# Env knobs:
#   VLLM_E2E_MODEL          model id (default: turboderp/Qwen3-8B-exl3)
#   VLLM_E2E_REVISION       revision  (default: 4.0bpw)
#   VLLM_E2E_PORT           HTTP port (default: 8111)
#   VLLM_E2E_BINARY         path to vllm-server (default: latest v* in target/release/)
#   VLLM_E2E_CHECK_SCRIPT   override path to grammar_strictness_check.py
#                           (default: /tmp/grammar_strictness_check.py)
#   VLLM_E2E_GRAMMAR_STRICT 1 → "model missing" is a fail (default skip)

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${VLLM_E2E_MODEL:-turboderp/Qwen3-8B-exl3}"
REVISION="${VLLM_E2E_REVISION:-4.0bpw}"
PORT="${VLLM_E2E_PORT:-8111}"
CHECK_SCRIPT="${VLLM_E2E_CHECK_SCRIPT:-/tmp/grammar_strictness_check.py}"
STRICT="${VLLM_E2E_GRAMMAR_STRICT:-0}"

# Pick the latest versioned binary if not overridden.
BINARY="${VLLM_E2E_BINARY:-}"
if [[ -z "$BINARY" ]]; then
    candidate="$(ls -1 "$REPO_ROOT"/target/release/vllm-server-v*.* 2>/dev/null \
        | sort -V | tail -n 1)"
    if [[ -z "$candidate" ]]; then
        candidate="$REPO_ROOT/target/release/vllm-server"
    fi
    BINARY="$candidate"
fi

log() { printf '[e2e] %s\n' "$*" >&2; }

skip_or_fail() {
    local reason="$1"
    if [[ "$STRICT" == "1" ]]; then
        log "FAIL: $reason"
        exit 1
    fi
    log "SKIP: $reason (set VLLM_E2E_GRAMMAR_STRICT=1 to upgrade to fail)"
    exit 0
}

# ── Preflight ────────────────────────────────────────────────────────
if [[ ! -x "$BINARY" ]]; then
    skip_or_fail "binary not found or not executable: $BINARY"
fi
if [[ ! -f "$CHECK_SCRIPT" ]]; then
    skip_or_fail "check script not found: $CHECK_SCRIPT"
fi
if ! command -v python3 >/dev/null 2>&1; then
    skip_or_fail "python3 not on PATH"
fi
if ! command -v curl >/dev/null 2>&1; then
    skip_or_fail "curl not on PATH"
fi

# Probe HF cache for the model snapshot.
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_DIR_FRAG="$(printf '%s' "$MODEL" | tr '/' '-')"
CACHE_HIT="$(find "$HF_HOME_DIR/hub" -maxdepth 2 -type d -name "models--*${MODEL_DIR_FRAG#*-}*" 2>/dev/null | head -n 1)"
if [[ -z "$CACHE_HIT" ]]; then
    skip_or_fail "model snapshot not in HF cache for $MODEL — refusing to download \
in an E2E smoke. Pre-fetch with 'huggingface-cli download $MODEL --revision $REVISION'."
fi

# Free the port (defensive).
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    skip_or_fail "port $PORT is already in use"
fi

# ── Start server ─────────────────────────────────────────────────────
LOG_FILE="$(mktemp -t vllm-e2e-grammar-XXXXXX.log)"
log "starting server: $BINARY (model=$MODEL@$REVISION, port=$PORT)"
log "server log: $LOG_FILE"
"$BINARY" serve \
    --model "$MODEL" \
    --revision "$REVISION" \
    --port "$PORT" \
    --num-gpu-blocks-override 128 \
    --kv-cache-dtype fp8_e4m3 \
    --enforce-eager \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
    log "shutting down server (pid=$SERVER_PID)"
    kill -INT "$SERVER_PID" 2>/dev/null || true
    # Wait up to 10s for graceful exit, then SIGKILL.
    for _ in {1..10}; do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Wait for ready (poll /v1/models up to 120s) ──────────────────────
log "waiting for server to become ready"
READY=0
for i in $(seq 1 120); do
    if curl -sS -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        READY=1
        log "server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "FAIL: server process exited during startup"
        tail -40 "$LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$READY" == "0" ]]; then
    log "FAIL: server did not become ready within 120s"
    tail -40 "$LOG_FILE" >&2
    exit 1
fi

# ── Run the strictness check ─────────────────────────────────────────
log "running $CHECK_SCRIPT"
CHECK_OUT="$(mktemp -t vllm-e2e-grammar-check-XXXXXX.out)"
if ! python3 "$CHECK_SCRIPT" > "$CHECK_OUT" 2>&1; then
    log "FAIL: check script returned non-zero"
    cat "$CHECK_OUT" >&2
    exit 1
fi
cat "$CHECK_OUT"

# Last line of /tmp/grammar_strictness_check.py output is
#   "Result: N/5 trials produced valid schema-conforming JSON"
PASSED="$(grep -E '^Result: [0-9]+/5' "$CHECK_OUT" | tail -n 1 | sed -E 's/Result: ([0-9]+)\/5.*/\1/')"
if [[ -z "$PASSED" ]]; then
    log "FAIL: could not parse 'Result: N/5' from check output"
    exit 1
fi

log "passed $PASSED / 5"
if [[ "$PASSED" != "5" ]]; then
    log "FAIL: xgrammar enforcement leaked — expected 5/5 valid schema JSON"
    exit 1
fi

log "OK — xgrammar enforced 5/5 strictness on $MODEL"
exit 0
