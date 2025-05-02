# Stage 1: Build the application
FROM rust:slim-bullseye as builder

WORKDIR /usr/src/dynamic-agent

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only Cargo files first to take advantage of Docker layer caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -f target/release/deps/dynamic_agent*

# Now copy the real source code
COPY . .

# Build the application
RUN cargo build --release

# Stage 2: Create the runtime image
FROM rust:slim-bullseye as runner

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