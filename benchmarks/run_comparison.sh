#!/usr/bin/env bash
set -euo pipefail

# Benchmark comparison: vllm-rust vs Python vLLM
# Runs both servers sequentially on the same GPU to avoid memory contention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (override via environment variables)
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
RUST_PORT="${RUST_PORT:-8100}"
VLLM_PORT="${VLLM_PORT:-8200}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"
WARMUP="${WARMUP:-3}"
CONCURRENCY="${CONCURRENCY:-1,2,4}"
MAX_TOKENS="${MAX_TOKENS:-64,128}"
PROMPT_LENGTH="${PROMPT_LENGTH:-short,medium}"

# Use venv python if available, otherwise system python3
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

RUST_RESULTS="$RESULTS_DIR/results_rust.json"
VLLM_RESULTS="$RESULTS_DIR/results_vllm.json"

# --- Helper functions ---

wait_for_server() {
    local url="$1"
    local name="$2"
    local timeout="${3:-60}"
    local elapsed=0

    echo "  Waiting for $name at $url/v1/models ..."
    while ! curl -sf "$url/v1/models" > /dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "  ERROR: $name did not start within ${timeout}s"
            return 1
        fi
    done
    echo "  $name is ready (took ${elapsed}s)"
}

kill_server() {
    local pid="$1"
    local name="$2"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping $name (PID $pid) ..."
        kill "$pid"
        wait "$pid" 2>/dev/null || true
    fi
}

run_bench() {
    local base_url="$1"
    local output="$2"

    "$PYTHON" "$SCRIPT_DIR/bench_serving.py" \
        --base-url "$base_url" \
        --model "$MODEL" \
        --prompts-file "$SCRIPT_DIR/prompts.json" \
        --prompt-length "$PROMPT_LENGTH" \
        --warmup "$WARMUP" \
        --num-requests "$NUM_REQUESTS" \
        --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_TOKENS" \
        --output "$output"
}

# --- Main ---

echo "=== Benchmark: vllm-rust vs Python vLLM ==="
echo "Model: $MODEL"
echo "Requests: $NUM_REQUESTS (warmup: $WARMUP)"
echo "Concurrency: $CONCURRENCY"
echo "Max tokens: $MAX_TOKENS"
echo "Prompt lengths: $PROMPT_LENGTH"
echo ""

# Step 1: Build vllm-rust
echo "--- Step 1: Building vllm-rust (release) ---"
cargo build --release -p vllm-server --manifest-path "$PROJECT_ROOT/Cargo.toml"
echo ""

# Step 2: Run vllm-rust benchmark
echo "--- Step 2: Benchmarking vllm-rust ---"
"$PROJECT_ROOT/target/release/vllm-server" serve \
    --model "$MODEL" \
    --port "$RUST_PORT" &
RUST_PID=$!

if wait_for_server "http://localhost:$RUST_PORT" "vllm-rust" 120; then
    run_bench "http://localhost:$RUST_PORT" "$RUST_RESULTS"
else
    kill_server $RUST_PID "vllm-rust"
    exit 1
fi

kill_server $RUST_PID "vllm-rust"
echo ""

# Step 3: Run Python vLLM benchmark
echo "--- Step 3: Benchmarking Python vLLM ---"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --max-model-len 2048 &
VLLM_PID=$!

if wait_for_server "http://localhost:$VLLM_PORT" "Python vLLM" 180; then
    run_bench "http://localhost:$VLLM_PORT" "$VLLM_RESULTS"
else
    kill_server $VLLM_PID "Python vLLM"
    exit 1
fi

kill_server $VLLM_PID "Python vLLM"
echo ""

# Step 4: Comparison
echo "--- Step 4: Comparison ---"
"$PYTHON" -c "
import json, sys

with open('$RUST_RESULTS') as f:
    rust = json.load(f)
with open('$VLLM_RESULTS') as f:
    vllm = json.load(f)

print()
print('=== Side-by-side Comparison (ratio = rust/vllm, <1.0 = rust faster) ===')
print()
header = f'{\"Scenario\":<28} {\"TTFT\":>8} {\"Thr/req\":>9} {\"ThrTot\":>8} {\"Latency\":>9}'
print(header)
print('-' * len(header))

rust_scenarios = {s['name']: s for s in rust['scenarios']}
vllm_scenarios = {s['name']: s for s in vllm['scenarios']}

for name in rust_scenarios:
    if name not in vllm_scenarios:
        continue
    rs = rust_scenarios[name]
    vs = vllm_scenarios[name]

    ttft_r = rs['ttft_p50_ms'] / vs['ttft_p50_ms'] if vs['ttft_p50_ms'] > 0 else 0
    thr_r = rs['throughput_mean_tps'] / vs['throughput_mean_tps'] if vs['throughput_mean_tps'] > 0 else 0
    thr_tot_r = rs['throughput_total_tps'] / vs['throughput_total_tps'] if vs['throughput_total_tps'] > 0 else 0
    lat_r = rs['latency_p50_ms'] / vs['latency_p50_ms'] if vs['latency_p50_ms'] > 0 else 0

    print(f'{name:<28} {ttft_r:>7.2f}x {thr_r:>8.2f}x {thr_tot_r:>7.2f}x {lat_r:>8.2f}x')

print()
print(f'Results saved to: $RESULTS_DIR/')
"

echo ""
echo "Done. Raw results:"
echo "  Rust: $RUST_RESULTS"
echo "  vLLM: $VLLM_RESULTS"
