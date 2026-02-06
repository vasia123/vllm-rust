# vllm-rust Docker image
# Multi-stage build for optimized production image

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

# ==================== BUILDER STAGE ====================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create app directory
WORKDIR /app

# Copy only dependency files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/
COPY crates/server/Cargo.toml crates/server/

# Create dummy source files to build dependencies
RUN mkdir -p crates/core/src crates/server/src && \
    echo "fn main() {}" > crates/server/src/main.rs && \
    echo "pub fn dummy() {}" > crates/core/src/lib.rs && \
    echo "pub fn dummy() {}" > crates/server/src/lib.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release -p vllm-server 2>/dev/null || true

# Remove dummy source files
RUN rm -rf crates/core/src crates/server/src

# Copy actual source code
COPY crates/ crates/
COPY kernels/ kernels/
COPY build.rs ./

# Touch source files to invalidate the cache
RUN touch crates/core/src/lib.rs crates/server/src/main.rs

# Build the actual application
RUN cargo build --release -p vllm-server

# ==================== RUNTIME STAGE ====================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 vllm
USER vllm

# Create directories for model cache and config
RUN mkdir -p /home/vllm/.cache/huggingface /home/vllm/.config/vllm-server

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/vllm-server /app/vllm-server

# Environment variables
ENV HF_HOME=/home/vllm/.cache/huggingface
ENV RUST_LOG=info

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/admin/live || exit 1

# Default command
ENTRYPOINT ["/app/vllm-server"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
