# Dynamic Agent

[![Rust](https://img.shields.io/badge/rust-stable-blue.svg)](https://www.rust-lang.org/)
<!-- Add other badges as needed, e.g., license, build status -->

Dynamic Agent is a flexible and configurable AI agent framework built in Rust. It provides a foundation for creating Retrieval-Augmented Generation (RAG) agents that can interact with users over WebSockets, leveraging multiple LLM providers and vector stores.

## Key Features

*   **Multi-LLM Support:** Integrates with various Large Language Model providers for chat completion, text embedding, and query generation.
    *   Supported: Ollama, OpenAI, Anthropic, Gemini, DeepSeek, XAI, Groq.
    *   Easily configurable via environment variables or CLI arguments.
*   **Multi-Vector Store Support:** Leverages the `vector-nexus` crate to connect to different vector databases for RAG.
    *   Supported: Redis, Qdrant, Chroma, Milvus, SurrealDB, Pinecone.
*   **Configurable RAG Pipeline:**
    *   Define agent behavior, intents, and prompt templates using `json/prompts.json`.
    *   Specify the structure of your vector data using `json/index_schema.json`.
    *   Supports LLM-driven intent classification and dynamic RAG query generation.
*   **Conversation History:** Persists conversation history using Redis, Qdrant, or in-memory storage.
*   **Caching Layer:** Optional Redis (exact match) and Qdrant (semantic match) caching to improve performance and reduce LLM costs.
*   **Flexible Configuration:** Configure all aspects using a `.env` file, with overrides possible via command-line arguments.
*   **WebSocket Interface:** Communicates with clients via a WebSocket server, supporting optional TLS (WSS) for secure connections.
*   **Authentication:** Optional API key authentication for securing the WebSocket server endpoint.

## Prerequisites

*   **Rust:** Install the Rust toolchain via [rustup](https://rustup.rs/).
*   **LLM Provider:** Access to at least one supported LLM (e.g., Ollama running locally, an OpenAI API key).
*   **Vector Store:** Access to a running instance of your chosen vector store (e.g., Redis, Qdrant).
*   **Redis (Optional):** Required if using Redis for history persistence or caching.

## Configuration

1.  **Environment Variables (`.env`)**
    *   Create a `.env` file in the project root (you can copy `.env.example` if one exists).
    *   Configure connection details and credentials for your chosen LLM providers, vector store, history store, and caching layer.
    *   Set the server address and optional API key.
    *   Example essential variables:
        ```dotenv
        # --- LLMs ---
        CHAT_LLM_TYPE=ollama
        CHAT_BASE_URL="http://localhost:11434" # Or OpenAI URL, etc.
        CHAT_API_KEY="" # Required for OpenAI, Anthropic, etc.
        CHAT_MODEL="llama3.2"

        EMBEDDING_LLM_TYPE=ollama
        EMBEDDING_BASE_URL="http://localhost:11434"
        EMBEDDING_API_KEY=""
        EMBEDDING_MODEL="nomic-embed-text"

        # --- Vector Store (Example: Qdrant) ---
        VECTOR_TYPE=qdrant
        VECTOR_HOST=http://localhost:6333
        VECTOR_SECRET="" # API Key if needed
        VECTOR_INDEX_NAME=my_documents
        VECTOR_DIMENSION=768 # Match your embedding model

        # --- History (Example: Redis) ---
        HISTORY_TYPE=redis
        HISTORY_HOST=redis://127.0.0.1:6379

        # --- Server ---
        SERVER_ADDR=127.0.0.1:4000
        # SERVER_API_KEY=your-secret-key # Uncomment to enable auth
        # TLS_CERT_PATH=/path/to/cert.pem # Uncomment for WSS
        # TLS_KEY_PATH=/path/to/key.pem   # Uncomment for WSS

        # --- Caching (Optional) ---
        ENABLE_CACHE=true
        CACHE_REDIS_URL=redis://127.0.0.1:6379/1
        CACHE_QDRANT_URL=http://localhost:6334
        # CACHE_QDRANT_API_KEY= # Qdrant cache API key if needed
        CACHE_QDRANT_COLLECTION=agent_cache
        CACHE_SIMILARITY_THRESHOLD=0.95
        CACHE_REDIS_TTL=3600
        ```
    *   Refer to the `src/cli/mod.rs` file or run with `--help` for a full list of environment variables and their corresponding CLI flags.

2.  **JSON Configuration Files**
    *   **`json/prompts.json`:** Define agent intents (e.g., `PROFILE_INFO`, `GENERAL_CHAT`), the action associated with each intent (`call_rag_tool`, `general_llm_call`), and the prompt templates used for tasks like intent classification and RAG response generation.
    *   **`json/index_schema.json`:** Describe the structure of your data within the vector store. List each index/collection name and the fields it contains. This is crucial for the RAG process to understand where to search.
    *   **`json/query/*.json`:** (Optional) Contains function schemas specific to vector stores, potentially used for advanced query generation (though currently marked as unused in the RAG engine).

## Building

```bash
cargo build --release
```

## Running

Execute the compiled binary. Configuration options can be provided via command-line arguments, which will override values set in the `.env` file.

```bash
# Run with defaults from .env
./target/release/dynamic-agent

# Override server address and vector store type
./target/release/dynamic-agent --server-addr 0.0.0.0:8080 --vector-type qdrant --vector-host http://localhost:6333

# See all available options
./target/release/dynamic-agent --help
```

The server will start listening on the configured `SERVER_ADDR`.

## Connecting & Interacting

1.  **Connect:** Use a WebSocket client to connect to the agent at the address specified by `SERVER_ADDR`.
    *   Use `ws://<host>:<port>` for standard connections.
    *   Use `wss://<host>:<port>` if TLS is enabled via `TLS_CERT_PATH` and `TLS_KEY_PATH`.
2.  **Authentication:** If `SERVER_API_KEY` is set in the configuration, your WebSocket client must provide this key during the connection handshake (the exact mechanism depends on the client library, often via a header like `Authorization: Bearer <key>` or `X-Api-Key: <key>`).
3.  **Send Messages:** Send user queries as messages over the WebSocket connection. The expected format is typically a simple text message or a JSON structure depending on the client implementation (e.g., `{"type": "message", "payload": "Tell me about your experience."}`).
4.  **Receive Responses:** The agent will process the message, potentially performing RAG and LLM calls, and send the response back over the same WebSocket connection.

## Contributing

<!-- Add contribution guidelines if desired -->
Contributions are welcome! Please open an issue or submit a pull request.

## License

<!-- Specify your license, e.g., MIT, Apache 2.0 -->
This project is licensed under the [Your License Name] License - see the LICENSE file for details.