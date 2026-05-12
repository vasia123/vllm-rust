#!/usr/bin/env bash
# Phase 11.2.D diagnostic harness.
#
# Launches the server twice on the same prompt — once in pure eager mode,
# once with CUDA-graph capture enabled. Each run dumps per-forward decode
# logits (and input token IDs) into a separate directory. Afterwards
# `diff -r /tmp/eager_dump /tmp/capture_dump` localizes the first forward
# at which the captured replay diverges from eager.
#
# Usage: bash scripts/diag_exl3_replay.sh [PROMPT]

set -euo pipefail

PROMPT="${1:-The capital of France is}"
MODEL="${MODEL:-turboderp/Llama-3.2-1B-Instruct-exl3}"
REVISION="${REVISION:-3.0bpw}"
MAX_TOKENS="${MAX_TOKENS:-24}"
PORT="${PORT:-18080}"
SERVER_BIN="${SERVER_BIN:-target/release/vllm-server}"
export HF_HOME="${HF_HOME:-/home/vasis/projects_hobby/vllm-rust/model_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EAGER_DIR="${EAGER_DIR:-/tmp/exl3_diag_eager}"
CAP_DIR="${CAP_DIR:-/tmp/exl3_diag_capture}"

run_pass() {
    local label="$1"
    local capture_m="$2"
    local dump_dir="$3"

    rm -rf "$dump_dir"
    mkdir -p "$dump_dir"

    local layer_dir="$dump_dir/layers"
    mkdir -p "$layer_dir"
    # D10: force pooled-V2 path in eager so its dumps share the same op
    # sequence as capture (where it engages by default via graph_runner).
    local force_pooled=""
    if [[ "$capture_m" == "0" ]]; then
        force_pooled="VLLM_FORCE_POOLED=1"
    fi
    echo "=== [$label] launching server (VLLM_EXL3_CAPTURE_MAX_M=$capture_m, dump=$dump_dir, layers=$layer_dir)"
    # D10: VLLM_CAPTURE_SINGLE_SIZE=1 captures batch_size=1 specifically;
    # without it the default single-capture policy targets the largest
    # configured batch size (16), and batch=1 real requests fall through
    # to eager — we'd be diffing two eager runs and learn nothing.
    env $force_pooled VLLM_EXL3_DUMP_DIR="$dump_dir" \
        VLLM_LAYER_DUMP_DIR="$layer_dir" \
        VLLM_LAYER_DUMP_AFTER_N="${LAYER_DUMP_AFTER_N:-0}" \
        VLLM_EXL3_CAPTURE_MAX_M="$capture_m" \
        VLLM_CAPTURE_SINGLE_SIZE=1 \
        VLLM_LOG_LEVEL=warn \
        "$SERVER_BIN" serve --model "$MODEL" --revision "$REVISION" --port "$PORT" --host 127.0.0.1 \
        >/tmp/exl3_diag_${label}.log 2>&1 &
    local pid=$!

    # Wait for server to be ready — model load + warmup can take 90+ s
    # on first run (downloads kernels, allocates KV blocks). Longer
    # timeout covers worst-case capture-warmup compilation.
    local ready=0
    for i in $(seq 1 300); do
        if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200; then
            ready=1
            echo "=== [$label] server ready after ${i}s"
            break
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "=== [$label] server exited before becoming ready (log tail):"
            tail -20 "/tmp/exl3_diag_${label}.log"
            return 1
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "=== [$label] server NEVER became ready in 300s (log tail):"
        tail -20 "/tmp/exl3_diag_${label}.log"
        kill "$pid" 2>/dev/null || true
        return 1
    fi

    echo "=== [$label] sending prompt: $PROMPT"
    local resp
    resp=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys; print(json.dumps({'model': '$MODEL', 'prompt': '$PROMPT', 'max_tokens': $MAX_TOKENS, 'temperature': 0.0}))")")
    echo "=== [$label] response:"
    python3 -c "import sys, json; d=json.loads(sys.argv[1]); print(d['choices'][0]['text'])" "$resp" || echo "[parse error] $resp"

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

echo "Phase 11.2.D diagnostic — eager vs capture replay"
echo ""

run_pass eager   0  "$EAGER_DIR"
sleep 2
run_pass capture 16 "$CAP_DIR"

echo ""
echo "=== diffing dumps"
diff -r "$EAGER_DIR" "$CAP_DIR" --brief || true
echo ""
echo "=== first decode_logits divergence (binary)"
for f in $(ls "$EAGER_DIR"/decode_logits.*.bin 2>/dev/null | sort); do
    base=$(basename "$f")
    if [[ -f "$CAP_DIR/$base" ]] && ! cmp -s "$f" "$CAP_DIR/$base"; then
        echo "DIVERGE: $base"
        break
    fi
    echo "match:   $base"
done

echo ""
echo "=== D10 per-layer divergence ==="
shopt -s nullglob
layer_eager_files=("$EAGER_DIR/layers"/*.bin)
shopt -u nullglob
if [[ ${#layer_eager_files[@]} -eq 0 ]]; then
    echo "(no layer dumps in $EAGER_DIR/layers — VLLM_LAYER_DUMP_DIR may not have triggered)"
else
    for f in $(printf '%s\n' "${layer_eager_files[@]}" | sort); do
        base=$(basename "$f")
        cap="$CAP_DIR/layers/$base"
        if [[ ! -f "$cap" ]]; then
            echo "MISSING capture: $base"
            continue
        fi
        if cmp -s "$f" "$cap"; then
            echo "match:   $base"
        else
            first_diff=$(cmp "$f" "$cap" 2>/dev/null | head -1 || true)
            echo "DIVERGE: $base  ($first_diff)"
        fi
    done
fi
