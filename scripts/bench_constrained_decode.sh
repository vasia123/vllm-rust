#!/usr/bin/env bash
# Wrapper around scripts/bench_constrained_decode.py.
#
# Spins up a single vllm-server binary, runs the benchmark, prints the
# JSON result, shuts the server down. Use twice with different
# --binary args to compare two builds (baseline vs candidate).
#
# Env knobs:
#   BENCH_MODEL          model id (default: turboderp/Qwen3-8B-exl3)
#   BENCH_REVISION       revision  (default: 4.0bpw)
#   BENCH_PORT           HTTP port (default: 8121)
#   BENCH_CONCURRENCY    space-separated list (default: "1 4")
#   BENCH_RUNS           runs per (concurrency, mode) (default: 5)
#   BENCH_MAX_TOKENS     decode length per request (default: 80)
#   BENCH_BINARY         server binary (default: latest v* in target/release/)
#   BENCH_LABEL          label for JSON output (default: $BENCH_BINARY basename)
#   BENCH_OUT_JSON       where to write JSON (default: stdout)
#   BENCH_PRETTY         1 → print table to stderr (default: 1)

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${BENCH_MODEL:-turboderp/Qwen3-8B-exl3}"
REVISION="${BENCH_REVISION:-4.0bpw}"
PORT="${BENCH_PORT:-8121}"
CONCURRENCY="${BENCH_CONCURRENCY:-1 4}"
RUNS="${BENCH_RUNS:-5}"
MAX_TOKENS="${BENCH_MAX_TOKENS:-80}"
OUT_JSON="${BENCH_OUT_JSON:-}"
PRETTY="${BENCH_PRETTY:-1}"

BINARY="${BENCH_BINARY:-}"
if [[ -z "$BINARY" ]]; then
    candidate="$(ls -1 "$REPO_ROOT"/target/release/vllm-server-v*.* 2>/dev/null \
        | sort -V | tail -n 1)"
    if [[ -z "$candidate" ]]; then
        candidate="$REPO_ROOT/target/release/vllm-server"
    fi
    BINARY="$candidate"
fi
LABEL="${BENCH_LABEL:-$(basename "$BINARY")}"

log() { printf '[bench] %s\n' "$*" >&2; }

if [[ ! -x "$BINARY" ]]; then
    log "FAIL: binary missing or not executable: $BINARY"
    exit 1
fi
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    log "FAIL: port $PORT already in use — pass BENCH_PORT or free it"
    exit 1
fi

LOG_FILE="$(mktemp -t vllm-bench-constrained-XXXXXX.log)"
log "starting $BINARY (model=$MODEL@$REVISION, port=$PORT)"
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
    for _ in {1..15}; do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

log "waiting for server to become ready"
READY=0
for i in $(seq 1 180); do
    if curl -sS -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        READY=1
        log "server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "FAIL: server exited during startup"
        tail -60 "$LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$READY" == "0" ]]; then
    log "FAIL: server not ready after 180s"
    tail -60 "$LOG_FILE" >&2
    exit 1
fi

extra=()
if [[ "$PRETTY" == "1" ]]; then
    extra+=(--pretty)
fi

OUT_FILE="$(mktemp -t vllm-bench-constrained-out-XXXXXX.json)"
log "running constrained decode bench (runs=$RUNS, max_tokens=$MAX_TOKENS, concurrency='$CONCURRENCY')"
# shellcheck disable=SC2086
python3 "$REPO_ROOT/scripts/bench_constrained_decode.py" \
    --base-url "http://localhost:$PORT/v1" \
    --model "$MODEL" \
    --label "$LABEL" \
    --concurrency $CONCURRENCY \
    --runs "$RUNS" \
    --max-tokens "$MAX_TOKENS" \
    "${extra[@]}" \
    > "$OUT_FILE"
rc=$?
if [[ $rc -ne 0 ]]; then
    log "FAIL: bench script exited with $rc"
    tail -20 "$OUT_FILE" >&2 || true
    exit $rc
fi

if [[ -n "$OUT_JSON" ]]; then
    cp "$OUT_FILE" "$OUT_JSON"
    log "wrote $OUT_JSON"
else
    cat "$OUT_FILE"
fi
exit 0
