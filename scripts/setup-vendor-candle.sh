#!/usr/bin/env bash
# Vendors candle-core 0.9.1 into vendor/candle/ and applies the
# perf branch patch (CudaDevice uses `new_stream()` instead of the legacy
# null `default_stream()`, unblocking CUDA Graph capture).
#
# Run once after checkout. Idempotent — re-running is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/candle"

if [[ -d "$VENDOR_DIR/.git" ]]; then
    echo "vendor/candle already present — skipping clone"
else
    mkdir -p "$REPO_ROOT/vendor"
    git clone --depth 1 --branch 0.9.1 \
        https://github.com/huggingface/candle.git "$VENDOR_DIR"
fi

cd "$VENDOR_DIR"

if git rev-parse --verify perf/non-default-stream >/dev/null 2>&1; then
    git checkout perf/non-default-stream
else
    git checkout -b perf/non-default-stream
fi

# The patch is a one-line surgical change. Detect whether it's already
# applied to keep the script idempotent.
DEVICE_RS="candle-core/src/cuda_backend/device.rs"
if grep -q "let stream = context.new_stream().w()?;" "$DEVICE_RS"; then
    echo "patch already applied"
    exit 0
fi

if ! grep -q "let stream = context.default_stream();" "$DEVICE_RS"; then
    echo "ERROR: candle source layout has changed — expected line not found in $DEVICE_RS" >&2
    exit 1
fi

# Use a sed surgical edit (single-line, single-file) so the script does
# not need to ship a binary patch.  Idempotency is guarded by the grep
# above; this branch only runs when the unpatched line is present.
sed -i 's|let stream = context.default_stream();|let stream = context.new_stream().w()?;|' "$DEVICE_RS"

git add "$DEVICE_RS"
git -c user.email="vllm-rust@local" -c user.name="vllm-rust" \
    commit -m "perf(cuda): use non-default stream so cuStreamBeginCapture works

cudarc 0.16's CudaContext::default_stream() returns a CudaStream whose
underlying handle is the legacy null stream. CUDA stream capture is
unconditionally unsupported on the null stream — switching CudaDevice::new
to context.new_stream() lets vllm-rust build CUDA graphs over candle
kernel launches."

echo "vendor/candle ready on branch perf/non-default-stream"
