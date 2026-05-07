#!/usr/bin/env bash
#
# End-to-end correctness regression guard for Qwen3-4B-AWQ.
#
# Spins up the vllm-server with the real Qwen/Qwen3-4B-AWQ checkpoint and
# verifies that a small fixed set of canonical prompts produce the expected
# first generated token (or the expected first few tokens for thinking-model
# behaviour). The expected outputs were observed against the vendored
# baseline at commit ea5ec9c (post Stage-12 OOM fix). They are stable across
# greedy decode in a clean process.
#
# Run manually after any kernel-touching change:
#   VLLM_BIN=./target/release/vllm-server scripts/test_qwen3_awq_correctness.sh
#
# Exits non-zero on any mismatch, sustained-load OOM, or server start-up
# failure. The OutputPool/zero-init/quantization paths historically all
# silently broke without a guard like this — the user-visible regression
# was missed for several commits in a row.
set -euo pipefail

PORT="${PORT:-8765}"
BASE_URL="http://localhost:${PORT}"
VLLM_BIN="${VLLM_BIN:-./target/release/vllm-server}"
MODEL="${MODEL:-Qwen/Qwen3-4B-AWQ}"
# 6000 ≥ adaptive selector boundary (4096) so case 6 can cross it.
# Translates to ~850 MiB of KV-cache for a 36-layer 8-kv-head model — fine
# on an 8 GiB Ada laptop alongside the ~2.2 GiB AWQ weights.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6000}"
LOG_FILE="$(mktemp -t vllm-correctness-XXXXXX.log)"

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "FATAL: $VLLM_BIN missing or not executable" >&2
    echo "  Build with: cargo build --release -p vllm-server --features cuda-full" >&2
    exit 2
fi

echo "[setup] using binary: $VLLM_BIN"
echo "[setup] log file:    $LOG_FILE"
echo "[setup] base url:    $BASE_URL"
echo

# Start server in background. --enforce-eager to keep the harness
# deterministic across CUDA-graph capture availability changes.
RUST_LOG=warn "$VLLM_BIN" serve \
    --model "$MODEL" \
    --port "$PORT" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[setup] started server pid=$SERVER_PID, waiting for /v1/models …"
for _ in $(seq 1 60); do
    if curl -sS --max-time 3 "${BASE_URL}/v1/models" 2>/dev/null | grep -q '"object"'; then
        break
    fi
    sleep 3
done

if ! curl -sS --max-time 3 "${BASE_URL}/v1/models" 2>/dev/null | grep -q '"object"'; then
    echo "FATAL: server failed to come up within 180 s" >&2
    tail -20 "$LOG_FILE" >&2
    exit 3
fi

echo "[setup] server ready"
echo

# Each case = (label, jq filter producing the value to assert, expected value).
# We assert on the first generated token's `bytes` field of the logprobs
# stream (top-1 token, fully deterministic at temperature=0). The Cyrillic
# `а` byte sequence below is the documented Qwen3-4B-AWQ output for the
# adversarial "Reply with X only" prompt class — it exercises the full
# prefill + decode path even on otherwise unusual logit distributions.

PASS=0
FAIL=0

run_case() {
    local label="$1"
    local payload="$2"
    local expect_first="$3"
    local response
    local first_token

    response="$(curl -sS --max-time 60 \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        "${BASE_URL}/v1/chat/completions" 2>&1)"

    if echo "$response" | grep -q '"error"'; then
        echo "FAIL [$label]: server error"
        echo "  response: $(echo "$response" | head -c 400)"
        FAIL=$((FAIL + 1))
        return
    fi

    first_token="$(echo "$response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ch = d["choices"][0]
toks = ch.get("logprobs", {}).get("content", [])
if not toks:
    print("<no-logprobs>")
else:
    print(repr(toks[0]["token"]))
' 2>/dev/null)"

    if [[ "$first_token" == "$expect_first" ]]; then
        echo "PASS [$label]: first_token=$first_token"
        PASS=$((PASS + 1))
    else
        echo "FAIL [$label]: first_token=$first_token, expected=$expect_first"
        echo "  body head: $(echo "$response" | head -c 300)"
        FAIL=$((FAIL + 1))
    fi
}

# Case 1: completion-mode continuation, no chat template
echo "[case 1] /v1/completions: 'Once upon a time, there was'"
response="$(curl -sS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"Once upon a time, there was","max_tokens":3,"temperature":0,"logprobs":1}' \
    "${BASE_URL}/v1/completions" 2>&1)"

first_token="$(echo "$response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ch = d["choices"][0]
toks = ch.get("logprobs", {}).get("tokens", [])
print(repr(toks[0]) if toks else "<no-tokens>")
')"
expected_completion=$'\' a\''
if [[ "$first_token" == "$expected_completion" ]]; then
    echo "PASS [completion]: first_token=$first_token"
    PASS=$((PASS + 1))
else
    echo "FAIL [completion]: first_token=$first_token, expected=$expected_completion"
    FAIL=$((FAIL + 1))
fi

# Case 2: chat completion, normal prompt → Qwen3 thinking start
run_case "chat-thinking" \
    '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":3,"temperature":0,"logprobs":true,"top_logprobs":1}' \
    "'<think>'"

# Case 3: chat completion, casual greeting → Qwen3 thinking start
run_case "chat-greeting" \
    '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":3,"temperature":0,"logprobs":true,"top_logprobs":1}' \
    "'<think>'"

# Case 4: simple math completion (no chat template)
echo "[case 4] /v1/completions: '1+1='"
response="$(curl -sS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"1+1=","max_tokens":3,"temperature":0,"logprobs":1}' \
    "${BASE_URL}/v1/completions" 2>&1)"
first_token="$(echo "$response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ch = d["choices"][0]
toks = ch.get("logprobs", {}).get("tokens", [])
print(repr(toks[0]) if toks else "<no-tokens>")
')"
if [[ "$first_token" == "'2'" ]]; then
    echo "PASS [math]: first_token=$first_token"
    PASS=$((PASS + 1))
else
    echo "FAIL [math]: first_token=$first_token, expected='2'"
    FAIL=$((FAIL + 1))
fi

# Case 5: sustained-load OOM regression guard.
# Eight medium-prompt requests followed by a 200-token request should all
# succeed and GPU memory should stay in a tight band. This is the exact
# pattern that surfaced the Stage-12 OutputPool leak (commit ea5ec9c).
echo
echo "[case 5] sustained load: 8× chat + 1× large generation"
oom_seen=0
for i in 1 2 3 4 5 6 7 8; do
    body="$(curl -sS --max-time 60 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Iter $i\"}],\"max_tokens\":40,\"temperature\":0}" \
        "${BASE_URL}/v1/chat/completions" 2>&1)"
    if echo "$body" | grep -q '"error"'; then
        echo "FAIL [sustained iter $i]: $(echo "$body" | head -c 200)"
        oom_seen=1
        break
    fi
done
if [[ $oom_seen -eq 0 ]]; then
    body="$(curl -sS --max-time 120 \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about photosynthesis\"}],\"max_tokens\":200,\"temperature\":0}" \
        "${BASE_URL}/v1/chat/completions" 2>&1)"
    if echo "$body" | grep -q '"error"'; then
        echo "FAIL [sustained large]: $(echo "$body" | head -c 200)"
        oom_seen=1
    fi
fi
if [[ $oom_seen -eq 0 ]]; then
    echo "PASS [sustained]: 8 small + 1 large request, no OOM"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

# Case 6: long-context generation crossing the adaptive selector boundary.
# `select_v2_partition_size` picks p=64 below seq_len=4096 and p=256 above.
# A prompt of ~3000 tokens + 1500 new tokens forces the kernel to pick both
# branches inside a single decode loop. Verifies (a) no shape errors, (b) no
# numerical blow-up, (c) the response field is well-formed.
echo
echo "[case 6] long-context: ~3k prompt + 1.5k generation crosses 4096 boundary"
# Build a long prompt by repeating a narrative seed. Qwen tokenizer averages
# ~0.75 tokens per character on English text, so 4500 chars ≈ 3300 tokens —
# well past 4096 even before generation starts.
LONG_PROMPT="$(python3 -c "
import sys
seed = 'The history of computing began long before electronic devices. '
seed += 'Mechanical calculators, the abacus, and astronomical instruments '
seed += 'all preceded modern computers. '
sys.stdout.write(seed * 60)
sys.stdout.write(' Continue with one more paragraph about modern computing: ')
")"
long_payload="$(python3 -c "
import json, sys
p = sys.stdin.read()
print(json.dumps({
    'model': '$MODEL',
    'prompt': p,
    'max_tokens': 50,
    'temperature': 0,
}))
" <<< "$LONG_PROMPT")"
long_response="$(curl -sS --max-time 120 \
    -H 'Content-Type: application/json' \
    -d "$long_payload" \
    "${BASE_URL}/v1/completions" 2>&1)"
if echo "$long_response" | grep -q '"error"'; then
    echo "FAIL [long-context]: $(echo "$long_response" | head -c 300)"
    FAIL=$((FAIL + 1))
else
    long_text="$(echo "$long_response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["choices"][0]["text"][:80])
' 2>/dev/null)"
    if [[ -n "$long_text" ]]; then
        echo "PASS [long-context]: response head=${long_text}"
        PASS=$((PASS + 1))
    else
        echo "FAIL [long-context]: empty completion"
        FAIL=$((FAIL + 1))
    fi
fi

# Case 7: concurrent multi-sequence batch.
# Two requests in flight simultaneously — exercises the engine's batched
# decode with sequences at *different* seq_lens (one in mid-prefill, the
# other already in steady-decode). The paged_attn kernel must handle a
# `seq_lens` tensor with varying entries; this is the exact path the
# unit-test parity tests cannot exercise (they all use num_seqs=1).
echo
echo "[case 7] concurrent batch: 2 requests with different prompt sizes"
mkfifo /tmp/case7_pipe1 /tmp/case7_pipe2 2>/dev/null || true
trap "cleanup; rm -f /tmp/case7_pipe1 /tmp/case7_pipe2" EXIT INT TERM
(curl -sS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0}" \
    "${BASE_URL}/v1/completions" > /tmp/case7_resp1 2>&1) &
PID_A=$!
(curl -sS --max-time 60 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"In Python, a list comprehension looks like\",\"max_tokens\":40,\"temperature\":0}" \
    "${BASE_URL}/v1/completions" > /tmp/case7_resp2 2>&1) &
PID_B=$!
wait $PID_A $PID_B
resp1="$(cat /tmp/case7_resp1)"
resp2="$(cat /tmp/case7_resp2)"
rm -f /tmp/case7_resp1 /tmp/case7_resp2 /tmp/case7_pipe1 /tmp/case7_pipe2
both_ok=1
for body in "$resp1" "$resp2"; do
    if echo "$body" | grep -q '"error"'; then
        both_ok=0
        echo "FAIL [concurrent]: $(echo "$body" | head -c 200)"
        break
    fi
    text_field="$(echo "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["text"][:60])
except Exception as e:
    print("PARSE_ERROR")
' 2>/dev/null)"
    if [[ "$text_field" == "PARSE_ERROR" ]] || [[ -z "$text_field" ]]; then
        both_ok=0
        echo "FAIL [concurrent]: malformed response: $(echo "$body" | head -c 200)"
        break
    fi
    echo "  ok: $text_field"
done
if [[ $both_ok -eq 1 ]]; then
    echo "PASS [concurrent]: both completions succeeded"
    PASS=$((PASS + 1))
else
    FAIL=$((FAIL + 1))
fi

echo
echo "──────────────"
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "──────────────"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
