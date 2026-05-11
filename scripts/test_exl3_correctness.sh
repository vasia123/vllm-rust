#!/usr/bin/env bash
# EXL3 correctness sweep — sanity check that Exl3Linear::forward produces
# coherent output across a curated prompt set.
set -euo pipefail
URL="${1:-http://127.0.0.1:18080/v1/completions}"
MODEL="${2:-turboderp/Llama-3.2-1B-Instruct-exl3}"
PROMPTS=(
  "The capital of France is"
  "Q: What is 2+2?\nA:"
  "Once upon a time"
  "The largest planet in our solar system is"
  "In Python, the print function is used to"
  "The first president of the United States was"
  "Water freezes at"
)
for p in "${PROMPTS[@]}"; do
  out=$(curl -s -X POST "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$(printf '%s' "$p" | sed 's/"/\\"/g')\",\"max_tokens\":24,\"temperature\":0.0}")
  text=$(echo "$out" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "[ERROR: $(echo "$out" | head -c 200)]")
  printf '> %s\n  %s\n\n' "$p" "$text"
done
