# Releasing

How a release is cut. The version lives in **one** place — `[workspace.package]
version` in the root `Cargo.toml` — and every crate inherits it via
`version.workspace = true`, so `vllm-server --version` and the `/version`
endpoint always match the tag.

## Versioning policy

[Semantic Versioning](https://semver.org). While we are on `0.x`:

- **minor** (`0.N.0`) — new features, and any breaking change to the HTTP API,
  CLI flags, config, or on-disk/cache formats. Pre-1.0, minors are allowed to
  break.
- **patch** (`0.N.P`) — bug fixes and perf work with no interface change.

At `1.0.0` we switch to strict semver (breaking changes ⇒ major).

Keep `CHANGELOG.md` current as you work: add entries under `## [Unreleased]`
(Added / Changed / Fixed / Removed). The release script promotes that section
to the new version.

## Cut a release

From a clean `main`:

```sh
scripts/release.sh minor          # or: patch | major | an explicit X.Y.Z
```

This bumps `[workspace.package] version`, refreshes `Cargo.lock`, promotes
`## [Unreleased]` → `## [X.Y.Z] - <date>`, commits `chore(release): vX.Y.Z`
(the pre-commit hook re-runs `fmt` + `clippy -D warnings`, so a release can
never ship a lint-dirty tree), tags `vX.Y.Z` (annotated), and pushes the
commit and tag.

Useful flags: `--no-push` (review before pushing), `--dry-run` (print only),
`--gh-release` (also open a GitHub Release with notes from the changelog).

## Publish binaries

We build binaries **locally** and upload them, rather than in CI: the
`cuda-full` build pulls in FlashInfer's template-heavy kernels whose `ptxas`
memory peaks exceed a free GitHub Actions runner (locally the build pins
`jobs = 1` and leans on swap — see `.cargo/config.toml`). CUDA *compilation*
needs no GPU, so a self-hosted or larger runner could do it later; until then,
local is the path.

After tagging:

```sh
scripts/publish-binary.sh v0.2.0                      # cuda-full (default)
scripts/publish-binary.sh v0.2.0 --features cuda-kernels   # lighter CUDA build
scripts/publish-binary.sh v0.2.0 --features "" --no-upload # CPU-only, local only
```

This builds the release binary, packages it as
`dist/vllm-server-<tag>-<os>-<arch>-cuda<ver>-sm<arch>.tar.gz` (plus a
`.sha256`), and uploads it to the GitHub Release for the tag (creating the
release if needed). The asset name records the CUDA toolkit version and GPU
arch the cubins target — PTX is forward-compatible (it JITs to newer archs at
load), but the CUDA runtime **major** version on the host must match.

Re-run with a different `--features` / on a different machine to add more
binary variants to the same release. `dist/` is git-ignored.

## One command, end to end

```sh
scripts/release.sh minor && scripts/publish-binary.sh "$(git describe --tags --abbrev=0)"
```
