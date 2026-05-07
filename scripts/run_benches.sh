#!/usr/bin/env bash
# Run all criterion benches and snapshot results to docs/perf/bench-history/.
#
# Usage:
#   scripts/run_benches.sh [--quick] [--filter <regex>] [--label <name>]
#
# --quick   pass through to criterion (single sample, ~5x faster)
# --filter  pass through to cargo bench (e.g. "compute_schedule")
# --label   suffix on the snapshot file name (default: git short SHA)
#
# Output: docs/perf/bench-history/<YYYY-MM-DD>-<label>.json
# Each entry: {bench, group, name, mean_ns, std_dev_ns}.
#
# Run before and after a perf-sensitive change. Diff the two JSONs to
# spot regressions early — much cheaper than discovering them via
# Python-vLLM side-by-side.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

QUICK=""
FILTER=""
LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)  QUICK="--quick"; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        --label)  LABEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$LABEL" ]]; then
    LABEL="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi
DATE="$(date +%Y-%m-%d)"
OUT_DIR="docs/perf/bench-history"
OUT_FILE="$OUT_DIR/${DATE}-${LABEL}.json"
mkdir -p "$OUT_DIR"

# Benches with required-features=cuda-kernels need the GPU build set.
# All others compile under the default (no CUDA) build, but enabling
# `cuda-kernels` everywhere is harmless on a host with CUDA available
# and keeps the command simple.
BENCH_FEATURES="cuda-kernels,marlin"

CARGO_BENCH=(cargo bench --features "$BENCH_FEATURES" -p vllm-core)
if [[ -n "$FILTER" ]]; then
    CARGO_BENCH+=(--bench "$FILTER")
fi

echo "==> running criterion benches (features=$BENCH_FEATURES, label=$LABEL, quick=${QUICK:-no})"

# Trigger the run; criterion writes raw samples under target/criterion/.
# Pipe stderr+stdout through tee so the user sees progress live.
"${CARGO_BENCH[@]}" -- ${QUICK:+$QUICK} 2>&1 | tee /tmp/run_benches.$$.log

# Collect the per-bench estimates.json files criterion drops at
# target/criterion/<group>/<bench>/<id>/new/estimates.json. Each holds
# the mean / median / std-dev of the most recent sample run.
python3 - <<PY > "$OUT_FILE"
import json, os, sys
root = "target/criterion"
out = []
if not os.path.isdir(root):
    print("[]"); sys.exit(0)
for group in sorted(os.listdir(root)):
    gdir = os.path.join(root, group)
    if not os.path.isdir(gdir) or group == "report":
        continue
    # Walk one or two levels (group/[id/]new/estimates.json).
    for entry in sorted(os.listdir(gdir)):
        edir = os.path.join(gdir, entry)
        if not os.path.isdir(edir):
            continue
        cands = []
        for sub in os.listdir(edir):
            sdir = os.path.join(edir, sub)
            est = os.path.join(sdir, "new", "estimates.json")
            if os.path.isfile(est):
                cands.append((sub, est))
            elif sub == "new" and os.path.isfile(os.path.join(sdir, "estimates.json")):
                cands.append((entry, os.path.join(sdir, "estimates.json")))
        if not cands:
            est = os.path.join(edir, "new", "estimates.json")
            if os.path.isfile(est):
                cands.append((entry, est))
        for name, path in cands:
            try:
                with open(path) as f:
                    d = json.load(f)
            except Exception:
                continue
            mean = d.get("mean", {}).get("point_estimate")
            stdv = d.get("std_dev", {}).get("point_estimate")
            if mean is None:
                continue
            out.append({
                "group": group,
                "name": name,
                "mean_ns": mean,
                "std_dev_ns": stdv,
            })
print(json.dumps(out, indent=2))
PY

rm -f /tmp/run_benches.$$.log
echo "==> wrote $OUT_FILE  ($(wc -l < "$OUT_FILE") lines)"
echo
echo "compare against a previous snapshot with:"
echo "    diff <(jq -S . $OUT_DIR/<old>.json) <(jq -S . $OUT_FILE)"
