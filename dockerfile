FROM rust:slim-bullseye AS builder

WORKDIR /usr/src/dynamic-agent

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && \
    echo "fn main() { /* dummy main for dependency caching */ }" > src/main.rs && \
    echo "pub fn lib_placeholder() { /* dummy lib for dependency caching */ }" > src/lib.rs && \
    cargo build --release && \
    rm -f target/release/dynamic-agent && \
    find target/release -maxdepth 1 -type f -name "libdynamic_agent.*" -delete && \
    rm -rf target/release/.fingerprint/dynamic-agent-* && \
    rm -rf target/release/build/dynamic-agent-* && \
    rm -rf target/release/incremental/dynamic-agent-*

COPY . .

# Build the application with the real source code
RUN cargo build --release

# Stage 2: Create the runtime image
FROM rust:slim-bullseye AS runner

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /usr/src/dynamic-agent/target/release/dynamic-agent /app/
# Copy JSON configuration directory
COPY --from=builder /usr/src/dynamic-agent/json /app/json/

# Create directory for environment files
RUN mkdir -p /app/config

# Set environment variables with defaults
ENV SERVER_ADDR=0.0.0.0:4000 \
    CHAT_LLM_TYPE=ollama \
    EMBEDDING_LLM_TYPE=ollama

# Expose the WebSocket server port
EXPOSE 4000

# Command to run the application
CMD ["/app/dynamic-agent"]