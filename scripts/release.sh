#!/usr/bin/env bash
#
# Cut a release: bump the workspace version, roll the changelog, commit, tag,
# and push. The single version source is `[workspace.package] version` in the
# root Cargo.toml; every crate inherits it via `version.workspace = true`, so
# `vllm-server --version` and the `/version` endpoint track this number.
#
# Usage:
#   scripts/release.sh <X.Y.Z | patch | minor | major> [flags]
#
# Flags:
#   --no-push      Commit + tag locally, do not push (review first).
#   --gh-release   After pushing, create a GitHub Release (notes from the
#                  changelog) via `gh`. Binaries are built/attached by CI on
#                  the tag, not here.
#   --dry-run      Print what would happen; change nothing.
#
# The release commit goes through the pre-commit hook (fmt + clippy -D
# warnings), so a release can never ship a lint-dirty tree. See
# docs/RELEASING.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() { echo "release: $*" >&2; exit 1; }

PUSH=1 GH_RELEASE=0 DRY=0 BUMP=""
for arg in "$@"; do
  case "$arg" in
    --no-push) PUSH=0 ;;
    --gh-release) GH_RELEASE=1 ;;
    --dry-run) DRY=1 ;;
    -*) die "unknown flag: $arg" ;;
    *) [ -z "$BUMP" ] || die "version given twice"; BUMP="$arg" ;;
  esac
done
[ -n "$BUMP" ] || die "usage: scripts/release.sh <X.Y.Z|patch|minor|major> [--no-push] [--gh-release] [--dry-run]"

run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$*"; fi; }

# ---- current version (single source) ----
CUR="$(sed -n 's/^version = "\([0-9]*\.[0-9]*\.[0-9]*\)".*/\1/p' Cargo.toml | head -1)"
[ -n "$CUR" ] || die "could not read current version from [workspace.package] in Cargo.toml"
IFS=. read -r MA MI PA <<<"$CUR"

case "$BUMP" in
  major) NEW="$((MA + 1)).0.0" ;;
  minor) NEW="${MA}.$((MI + 1)).0" ;;
  patch) NEW="${MA}.${MI}.$((PA + 1))" ;;
  [0-9]*.[0-9]*.[0-9]*) NEW="$BUMP" ;;
  *) die "version must be X.Y.Z or patch|minor|major (got: $BUMP)" ;;
esac
TAG="v$NEW"
echo "release: $CUR -> $NEW  (tag $TAG)"

# ---- preconditions ----
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "must release from main (on '$BRANCH')"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty — commit or stash first"
git rev-parse "$TAG" >/dev/null 2>&1 && die "tag $TAG already exists"
DATE="$(date +%F)"

# ---- 1) bump the single version source + refresh the lockfile ----
run "sed -i 's/^version = \"$CUR\"/version = \"$NEW\"/' Cargo.toml"
# `--offline` updates only the workspace members' versions in Cargo.lock
# without reaching the registry (so it can't drift third-party deps).
run "cargo update --workspace --offline"

# ---- 2) roll the changelog: [Unreleased] -> [NEW] - DATE, fresh Unreleased ----
if [ "$DRY" = 1 ]; then
  echo "+ changelog: rename [Unreleased] -> [$NEW] - $DATE, add empty [Unreleased]"
else
  awk -v ver="$NEW" -v date="$DATE" '
    !done && /^## \[Unreleased\]/ {
      print "## [Unreleased]"; print ""; print "## [" ver "] - " date; done=1; next
    }
    { print }
  ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
  # Link refs at the bottom: point Unreleased at the new tag, add the tag link.
  sed -i "s|^\[Unreleased\]:.*|[Unreleased]: https://github.com/vasia123/vllm-rust/compare/$TAG...HEAD|" CHANGELOG.md
  if ! grep -q "^\[$NEW\]:" CHANGELOG.md; then
    sed -i "/^\[Unreleased\]:.*/a [$NEW]: https://github.com/vasia123/vllm-rust/releases/tag/$TAG" CHANGELOG.md
  fi
fi

# ---- 3) commit (pre-commit hook runs fmt + clippy) + annotated tag ----
run "git add Cargo.toml Cargo.lock CHANGELOG.md"
run "git commit -m 'chore(release): $TAG'"
run "git tag -a '$TAG' -m '$TAG'"

# ---- 4) push ----
if [ "$PUSH" = 1 ]; then
  run "git push origin main"
  run "git push origin '$TAG'"
else
  echo "release: --no-push set; push with: git push origin main && git push origin $TAG"
fi

# ---- 5) optional GitHub Release (notes from changelog section) ----
if [ "$GH_RELEASE" = 1 ]; then
  command -v gh >/dev/null || die "--gh-release needs the gh CLI"
  if [ "$DRY" = 1 ]; then
    echo "+ gh release create $TAG --title $TAG --notes <changelog section>"
  else
    NOTES="$(awk -v ver="$NEW" '
      $0 ~ "^## \\[" ver "\\]" {grab=1; next}
      grab && /^## \[/ {exit}
      grab {print}
    ' CHANGELOG.md)"
    gh release create "$TAG" --title "$TAG" --notes "$NOTES"
  fi
fi

echo "release: done — $TAG"
