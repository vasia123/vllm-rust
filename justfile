# vLLM Rust - Development Commands
# Install just: cargo install just
# Run: just <command>

# Default command - show help
default:
    @just --list

# === Development ===

# Run frontend dev server (hot reload)
frontend-dev:
    cd frontend && npm run dev

# Run backend server
backend-dev *ARGS:
    cargo run -p vllm-server -- serve {{ARGS}}

# Run both frontend and backend in parallel (requires terminal multiplexer or two terminals)
dev:
    #!/usr/bin/env bash
    echo "Starting development servers..."
    echo "Frontend: http://localhost:5173"
    echo "Backend:  http://localhost:8000"
    echo "Admin:    http://localhost:8000/admin/"
    echo ""
    echo "Press Ctrl+C to stop"
    trap 'kill $(jobs -p) 2>/dev/null' EXIT
    (cd frontend && npm run dev) &
    cargo run -p vllm-server -- serve
    wait

# Run backend only (no frontend build needed for API development)
serve *ARGS:
    cargo run -p vllm-server -- serve {{ARGS}}

# Generate text from CLI
generate *ARGS:
    cargo run -p vllm-server -- generate {{ARGS}}

# === Building ===

# Build frontend for production
build-frontend:
    cd frontend && npm run build

# Build backend (release mode)
build-backend:
    cargo build --release -p vllm-server

# Build everything (frontend + backend)
build: build-frontend build-backend
    @echo "Build complete: target/release/vllm-server"

# Build with CUDA kernels
build-cuda: build-frontend
    cargo build --release -p vllm-server --features cuda-kernels

# === Testing ===

# Run all tests
test:
    cargo test

# Run server tests only
test-server:
    cargo test -p vllm-server

# Run core tests only
test-core:
    cargo test -p vllm-core

# Run tests with output
test-verbose:
    cargo test -- --nocapture

# === Code Quality ===

# Format code
fmt:
    cargo fmt
    cd frontend && npm run format 2>/dev/null || true

# Check formatting (CI)
fmt-check:
    cargo fmt --check

# Run clippy linter
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run all checks (format + clippy + test)
check: fmt-check clippy test
    @echo "All checks passed!"

# === Frontend ===

# Install frontend dependencies
frontend-install:
    cd frontend && npm install

# Lint frontend
frontend-lint:
    cd frontend && npm run lint 2>/dev/null || echo "No lint script configured"

# Type check frontend
frontend-typecheck:
    cd frontend && npx vue-tsc --noEmit

# === Utilities ===

# Clean build artifacts
clean:
    cargo clean
    rm -rf frontend/dist frontend/node_modules/.vite

# Update dependencies
update:
    cargo update
    cd frontend && npm update

# Show dependency tree
deps:
    cargo tree -p vllm-server

# Watch for changes and rebuild (requires cargo-watch)
watch:
    cargo watch -x 'check -p vllm-server'

# === Docker ===

# Build docker image
docker-build:
    docker build -t vllm-rust .

# Run in docker
docker-run *ARGS:
    docker run --gpus all -p 8000:8000 vllm-rust {{ARGS}}

# === Benchmarks ===

# Run benchmarks
bench:
    cargo bench

# === Documentation ===

# Generate and open docs
docs:
    cargo doc --open --no-deps

# === Release ===

# Create release build with all optimizations
release: build-frontend
    RUSTFLAGS="-C target-cpu=native" cargo build --release -p vllm-server
    @echo "Binary: target/release/vllm-server"
    @ls -lh target/release/vllm-server
