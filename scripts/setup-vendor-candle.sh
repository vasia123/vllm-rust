#!/usr/bin/env bash
# Vendors a pinned candle commit into vendor/candle/ and applies the
# `perf/non-default-stream` patch (CudaDevice uses `new_stream()` instead
# of the legacy null `default_stream()`, unblocking part 1 of CUDA Graph
# capture; capture is V3-blocked separately, see Stage 13-A verdict).
#
# Why pin a commit, not a tag/branch?
#   `--branch 0.9.1` resolves to the *branch head* of the upstream
#   `0.9.1` maintenance branch, which moves as HuggingFace lands
#   post-tag fixes (e.g. memcpy_stod / memcpy_dtov / extra DType
#   variants we now depend on). Cloning a branch is non-deterministic;
#   our codebase wants the exact post-0.9.1 API tree we built against.
#   So we fetch the specific commit SHA below and check it out.
#
# Run once after checkout (or in CI before cargo). Idempotent — re-running
# is safe (the patch detection short-circuits).

set -euo pipefail

# Pinned candle commit. Bumping this is a deliberate change — verify
# vllm-core compiles after retargeting.
CANDLE_SHA="cd96fa80da255e34f7b16b4ff98b6a31d557201b"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/candle"

if [[ -d "$VENDOR_DIR/.git" ]]; then
    echo "vendor/candle already present — verifying SHA"
    cd "$VENDOR_DIR"
    HAVE="$(git rev-parse HEAD)"
    if [[ "$HAVE" != "$CANDLE_SHA" ]] && \
       ! git rev-parse --verify perf/non-default-stream >/dev/null 2>&1; then
        echo "ERROR: vendor/candle is at $HAVE but expected $CANDLE_SHA"  >&2
        echo "       remove vendor/ and re-run this script to re-pin."    >&2
        exit 1
    fi
else
    mkdir -p "$REPO_ROOT/vendor"
    # `git clone` followed by an explicit checkout is the minimal-bandwidth
    # way to land on a specific SHA without depending on it being on a
    # branch. `--no-checkout` skips the initial worktree population so the
    # subsequent fetch+checkout produces exactly one materialised tree.
    git clone --no-checkout https://github.com/huggingface/candle.git "$VENDOR_DIR"
    cd "$VENDOR_DIR"
    git fetch origin "$CANDLE_SHA" --depth 1
    git checkout --quiet "$CANDLE_SHA"
fi

cd "$VENDOR_DIR"

# Always work on the perf branch so subsequent re-runs see a stable name.
if git rev-parse --verify perf/non-default-stream >/dev/null 2>&1; then
    git checkout perf/non-default-stream
else
    git checkout -b perf/non-default-stream
fi

# The patch is a one-line surgical change to the standard
# `CudaDevice::new()` constructor (line ~261 in the pinned commit).
# Upstream already has a sibling `new_with_stream()` that uses
# `new_stream()`; we can't grep the whole file for `new_stream` to
# decide idempotency because that match also fires on the upstream
# function. Instead, idempotency = "the unpatched `default_stream()`
# call is *gone*". The pinned commit ships exactly one such call
# (verified above) — once it's been replaced, this branch exits.
DEVICE_RS="candle-core/src/cuda_backend/device.rs"
if ! grep -q "let stream = context.default_stream();" "$DEVICE_RS"; then
    echo "vendor/candle ready (patch already applied at $(git rev-parse --short HEAD))"
    exit 0
fi

# Use a sed surgical edit (single-line, single-file) so the script does
# not need to ship a binary patch.  Idempotency is guarded by the grep
# above; this branch only runs when the unpatched line is present.
sed -i 's|let stream = context.default_stream();|let stream = context.new_stream().w()?;|' "$DEVICE_RS"

git add "$DEVICE_RS"
git -c user.email="vllm-rust@local" -c user.name="vllm-rust" \
    commit --quiet -m "perf(cuda): use non-default stream so cuStreamBeginCapture works

cudarc 0.16's CudaContext::default_stream() returns a CudaStream whose
underlying handle is the legacy null stream. CUDA stream capture is
unconditionally unsupported on the null stream — switching CudaDevice::new
to context.new_stream() lets vllm-rust build CUDA graphs over candle
kernel launches."

echo "vendor/candle ready on branch perf/non-default-stream @ $(git rev-parse --short HEAD)"
