#!/usr/bin/env bash
#
# Build the release binary locally and publish it as a GitHub Release asset.
#
# CUDA binaries are too heavy and too environment-specific for free GitHub
# Actions runners (the FlashInfer template kernels peak ptxas memory well past
# a hosted runner's RAM), so we build on this machine and upload the artifact.
# The asset name records the platform, CUDA toolkit version and GPU arch the
# cubins target; PTX is forward-compatible so it JITs to newer archs, but the
# CUDA runtime major version must match.
#
# Usage:
#   scripts/publish-binary.sh [TAG] [--features <set>] [--no-upload]
#
#   TAG          release tag to attach to (default: latest tag, `git describe`)
#   --features   cargo features (default: cuda-full; e.g. --features cuda-kernels,
#                or a CPU build with --features "" )
#   --no-upload  build + package into dist/ only; skip the GitHub upload
#
# See docs/RELEASING.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
die() { echo "publish-binary: $*" >&2; exit 1; }

TAG="" FEATURES="cuda-full" UPLOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --features) FEATURES="${2:-}"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    -*) die "unknown flag: $1" ;;
    *) [ -z "$TAG" ] || die "tag given twice"; TAG="$1"; shift ;;
  esac
done
[ -n "$TAG" ] || TAG="$(git describe --tags --abbrev=0 2>/dev/null)" || die "no tag given and no tags exist"
git rev-parse "$TAG" >/dev/null 2>&1 || die "tag '$TAG' does not exist"

# ---- platform / toolchain metadata for the asset name ----
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
SUFFIX="$OS-$ARCH"
if [[ ",$FEATURES," == *",cuda"* ]] || [[ "$FEATURES" == cuda* ]]; then
  CUDAVER="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
  SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  SUFFIX="$SUFFIX-cuda${CUDAVER:-unknown}-sm${SM:-unknown}"
else
  SUFFIX="$SUFFIX-cpu"
fi

echo "publish-binary: tag=$TAG features='$FEATURES' -> $SUFFIX"

# ---- build ----
echo "publish-binary: building (this is slow; .cargo/config.toml pins jobs=1)..."
if [ -n "$FEATURES" ]; then
  cargo build --release -p vllm-server --features "$FEATURES" --bin vllm-server
else
  cargo build --release -p vllm-server --bin vllm-server
fi
BIN="target/release/vllm-server"
[ -x "$BIN" ] || die "build produced no $BIN"

# ---- package ----
mkdir -p dist
NAME="vllm-server-$TAG-$SUFFIX"
STAGE="$(mktemp -d)"
cp "$BIN" "$STAGE/vllm-server"
{
  echo "vllm-server $TAG"
  echo "built: $(date -u +%FT%TZ)"
  echo "platform: $OS/$ARCH"
  echo "features: ${FEATURES:-<default (cpu)>}"
  echo "commit: $(git rev-parse HEAD)"
} > "$STAGE/BUILD-INFO.txt"
ASSET="dist/$NAME.tar.gz"
tar -czf "$ASSET" -C "$STAGE" vllm-server BUILD-INFO.txt
rm -rf "$STAGE"
SHA="$(sha256sum "$ASSET" | awk '{print $1}')"
echo "$SHA  $NAME.tar.gz" > "dist/$NAME.tar.gz.sha256"
echo "publish-binary: $ASSET ($(du -h "$ASSET" | cut -f1), sha256 $SHA)"

# ---- upload ----
if [ "$UPLOAD" = 0 ]; then
  echo "publish-binary: --no-upload set; artifact in dist/"
  exit 0
fi
command -v gh >/dev/null || die "upload needs the gh CLI (or pass --no-upload)"
if ! gh release view "$TAG" >/dev/null 2>&1; then
  echo "publish-binary: release $TAG not found — creating from tag"
  gh release create "$TAG" --title "$TAG" --notes "Binaries are attached as assets. See CHANGELOG.md for changes."
fi
gh release upload "$TAG" "$ASSET" "dist/$NAME.tar.gz.sha256" --clobber
echo "publish-binary: uploaded $NAME.tar.gz to release $TAG"
