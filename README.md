# Dynamic Agent

[![Rust](https://img.shields.io/badge/rust-stable-blue.svg)](https://www.rust-lang.org/)


Dynamic Agent is a flexible and configurable AI agent framework built in Rust. It provides a foundation for creating Retrieval-Augmented Generation (RAG) agents that can interact with users over WebSockets, leveraging multiple LLM providers and vector stores.

**Live Demo:** Experience Dynamic Agent in action at [thanon.dev/chat](https://thanon.dev/chat) (powered by the [Leptos Portfolio Admin](https://github.com/DevsHero/leptos_portfolio_admin) frontend).

## Key Features

*   **Multi-LLM Support:** Integrates with various Large Language Model providers for chat completion, text embedding, and query generation.
    *   Supported: Ollama, OpenAI, Anthropic, Gemini, DeepSeek, XAI, Groq.
    *   Easily configurable via environment variables or CLI arguments.
*   **Multi-Vector Store Support:** Leverages the `vector-nexus` crate to connect to different vector databases for RAG.
    *   Supported: Redis, Qdrant, Chroma, Milvus, SurrealDB, Pinecone.
*   **Configurable RAG Pipeline:**
    *   Define agent behavior, intents, and prompt templates using local JSON files or Firebase Remote Config.
    *   `vector-nexus` automatically detects your vector data structure, eliminating manual `index_schema.json` creation.
    *   Supports LLM-driven intent classification and dynamic RAG query generation.
*   **Dynamic Prompt Management:**
    *   Supports loading prompts from local files and/or Firebase Remote Config.
    *   Includes an HTTP webhook API to manually trigger prompt reloads without restarting the agent.
*   **Conversation History:** Persists conversation history using Redis, Qdrant, or in-memory storage.
*   **Two-Tier Caching System:** Implements a hybrid caching approach combining Redis (for exact matches) and Qdrant (for semantic similarity matches), reducing LLM costs and improving response times.
*   **Dynamic Topic Resolution:** Uses a cascading prompt system to determine the most relevant data indexes for queries, with primary and fallback resolution mechanisms for handling ambiguous queries.
*   **Flexible Configuration:** Configure all aspects using a `.env` file, with overrides possible via command-line arguments.
*   **WebSocket Interface:** Communicates with clients via a WebSocket server, supporting optional TLS (WSS) for secure connections.
*   **WebSocket Authentication:** Optional API key authentication (HMAC-based) for securing the WebSocket server endpoint.
*   **Frontend Integration:** Designed to work seamlessly with frontend applications, such as the [Leptos Portfolio Admin](https://github.com/DevsHero/leptos_portfolio_admin) project.

## Prerequisites

*   **Rust:** Install the Rust toolchain via [rustup](https://rustup.rs/).
*   **LLM Provider:** Access to at least one supported LLM (e.g., Ollama running locally, an OpenAI API key).
*   **Vector Store:** Access to a running instance of your chosen vector store (e.g., Qdrant).
*   **Redis:** Access to a running Redis instance (required for history persistence and/or caching if configured).

## Getting Data Ready (Integration with db2vec & vector-nexus)

Dynamic Agent works seamlessly with `db2vec` for data ingestion and relies on `vector-nexus` (which is included as a dependency) to interact with your vector store and understand its structure. Here's the typical workflow:

1.  **Ingest Data with `db2vec`:**
    *   Use the [db2vec](https://github.com/DevsHero/db2vec) tool to dump your source data (from databases, files, etc.) into your chosen vector store. `db2vec` handles connecting to data sources, generating embeddings using a specified model, and indexing the data.
    *   Ensure the embedding model used in `db2vec` matches the `EMBEDDING_MODEL` configured for Dynamic Agent.

2.  **Automatic Schema Detection via `vector-nexus`:**
    *   Dynamic Agent uses the [vector-nexus](https://github.com/DevsHero/vector-nexus) library internally.
    *   **You do not need to manually create `json/index_schema.json`.** `vector-nexus` will automatically inspect your vector store (based on the connection details provided) to determine the structure (indexes/collections and their fields) of the data you ingested with `db2vec`.
    *   This schema information is then used internally by Dynamic Agent for RAG operations, including dynamic hybrid search query generation driven by the LLM.

3.  **Configure Dynamic Agent:**
    *   Set up your `.env` file or use CLI arguments to point Dynamic Agent to the same vector store instance used by `db2vec`.
    *   Ensure the `VECTOR_TYPE`, `VECTOR_HOST`, `VECTOR_INDEX_NAME`, etc., match your setup so `vector-nexus` can connect and inspect the correct store.

With these steps completed, Dynamic Agent, powered by `vector-nexus`, will be ready to perform RAG queries against your data.

## Configuration

Configuration is primarily handled via environment variables, with JSON files for prompt and query structures.

1.  **Environment Variables (`.env` or `.env-agent`)**
    *   Create a `.env` file in the project root for native runs, or a `.env-agent` file for Docker Compose setups. You can copy from `.env.example` as a starting point.
    *   This file configures LLM providers, vector store connections, history store, caching, server address, API keys, prompt sources, etc.
    *   **Key Variables to Set:**
        *   `CHAT_LLM_TYPE`, `CHAT_BASE_URL`, `CHAT_MODEL`
        *   `EMBEDDING_LLM_TYPE`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
        *   `VECTOR_TYPE`, `VECTOR_HOST`, `VECTOR_INDEX_NAME`, `VECTOR_DIMENSION`
        *   `HISTORY_TYPE`, `HISTORY_HOST`
        *   `SERVER_ADDR`
        *   `PROMPTS_PATH` (for local prompts, default: `json/prompts.json`)
        *   (Optional) `SERVER_API_KEY` (for WebSocket authentication)
        *   (Optional) `ENABLE_CACHE`, `CACHE_REDIS_URL`, `CACHE_QDRANT_URL`, etc.
        *   (Optional) `HTTP_PORT` (for the prompt reload webhook, default: `4200`)
        *   (Optional, for Firebase Remote Config) `ENABLE_REMOTE_PROMPTS`, `REMOTE_PROMPTS_PROJECT_ID`, `REMOTE_PROMPTS_SA_KEY_PATH`
    *   Refer to `.env.example` for a comprehensive list of all available variables and their descriptions.
    *   Values set directly as environment variables in Docker Compose or via CLI arguments will override those in the `.env` or `.env-agent` file.

2.  **JSON Configuration Files**
    *   **Local Prompts (`PROMPTS_PATH`, e.g., `json/prompts.json`):** Defines agent intents, actions, and core prompt templates if not using or to supplement remote prompts.
    *   **Firebase Remote Config:** (If enabled) Provides a dynamic way to manage prompt configurations. See "Dynamic Prompt Management" section for details.
    *   **`json/query/*.json`:** (Optional) Schemas for advanced vector store query generation.

## Dynamic Prompt Management

Dynamic Agent supports flexible prompt management, allowing you to update agent behavior without restarting the application. Prompts can be sourced locally and/or from Firebase Remote Config, and reloaded on demand via an API.

### Prompt Sources

1.  **Local Prompts:**
    *   Defined in a JSON file specified by the `PROMPTS_PATH` environment variable (default: `json/prompts.json`).
    *   Changes to this file can be reloaded into the running agent using the webhook API.

2.  **Firebase Remote Config (Recommended for Dynamic Updates):**
    *   Provides a secure and centralized way to manage and update your prompt configurations.
    *   Updates can be published through the Firebase console and then reloaded into the agent via the webhook.
    *   **Configuration via `.env` or CLI:**
        *   `ENABLE_REMOTE_PROMPTS=true` (or `--enable-remote-prompts`)
        *   `REMOTE_PROMPTS_PROJECT_ID="your-firebase-project-id"` (or `--remote-prompts-project-id`)
        *   `REMOTE_PROMPTS_SA_KEY_PATH="path/to/your/firebase-sa.json"` (or `--remote-prompts-sa-key-path`)
    *   **Data Format in Firebase Remote Config:**
        In the Firebase Remote Config console, you will typically define **one primary parameter** to hold your entire prompt configuration. Let's assume you name this parameter `prompts` (you can choose another name, but ensure your agent's fetching logic matches).

        1.  **Create a Parameter:** In the Firebase Remote Config console, create a new parameter. For example, name it `prompts`.
        2.  **Set Value Type:** Set this parameter's "Value type" to "JSON".
        3.  **Set Default Value:** The "Default value" for this `prompts` parameter will be a **JSON string**. This string *itself* must be the **entire content of your local `json/prompts.json` file, stringified**.

        **Example:**

        If your `json/prompts.json` file looks like this:
        ```json
        {
          "intents": {
            "greeting": {
              "keywords": ["hello", "hi"],
              "actions": ["greet_user"]
            }
          },
          "core_prompts": {
            "system_message": "You are a helpful AI.",
            "greeting": "Hello! How can I assist you today?"
          },
          "query_templates": {
            "default_rag": "Context: {context}\nQuestion: {query}\nAnswer:"
          },
          "response_templates": {
            "greet_user": "Hello there! How may I help you?"
          }
        }
        ```

        Then, in the Firebase console, for your `prompts` parameter:
        *   **Parameter key:** `prompts`
        *   **Value type:** JSON
        *   **Default value:** You would paste the *stringified version* of the entire JSON content above. It would look like this (all on one line, or properly escaped if your editor requires it for a multi-line string input):
            ```
            "{\"intents\":{\"greeting\":{\"keywords\":[\"hello\",\"hi\"],\"actions\":[\"greet_user\"]}},\"core_prompts\":{\"system_message\":\"You are a helpful AI.\",\"greeting\":\"Hello! How can I assist you today?\"},\"query_templates\":{\"default_rag\":\"Context: {context}\\nQuestion: {query}\\nAnswer:\"},\"response_templates\":{\"greet_user\":\"Hello there! How may I help you?\"}}"
            ```

        The Dynamic Agent will then fetch the value of this single `prompts` parameter. The fetched string is then parsed as JSON to load the entire prompt configuration. This approach simplifies managing your prompts in Firebase, as your entire prompt structure is contained within a single Remote Config parameter.

        **(Developer Note:** Ensure the agent's `RemoteConfigClient::fetch_config` method in `src/config/remote_config.rs` is adapted to fetch and use the value of this single, all-encompassing parameter directly.)

### Webhook API for Reloading Prompts

An HTTP GET endpoint is available to manually trigger a reload of prompt configurations from their configured sources (local file and/or Firebase Remote Config).

*   **Endpoint:** `GET /api/reload-prompts`
*   **Authentication:** None (this endpoint is unauthenticated by default).
*   **Port:** Configured by `HTTP_PORT` (default `4200`).
*   **Query Parameter:**
    *   `source`: (Optional) Specifies which prompts to reload.
        *   `local`: Reloads only from the local file specified by `PROMPTS_PATH`.
        *   `remote`: Reloads only from Firebase Remote Config (if `ENABLE_REMOTE_PROMPTS` is true).
       

**How to Use the Webhook:**

Assuming the agent is running and the HTTP webhook server is enabled on port `4201` (`HTTP_PORT=4201`):

1.  **Reload only local prompts:**
    ```bash
    curl "http://localhost:4201/api/reload-prompts?source=local"
    ```

2.  **Reload only remote prompts (from Firebase):**
    ```bash
    curl "http://localhost:4201/api/reload-prompts?source=remote"
    ```

The API will respond with a JSON object indicating the success status and details of the reload operation, for example:
```json
{
  "success": true,
  "message": "Reload operation completed successfully",
  "details": [
    "Local prompts reloaded successfully",
    "Remote prompts reloaded successfully"
  ]
}
```
If a source is not configured (e.g., remote prompts are disabled), the details will reflect that.

## Advanced Features

### Two-Tier Caching System

Dynamic Agent implements a sophisticated two-tier caching strategy to improve performance and reduce LLM costs:

1. **Redis Cache (Tier 1):**
   * Provides fast exact-match lookups for previously seen queries.
   * Configured with a TTL to automatically expire cache entries.
   * Extremely fast response time when exact matches are found.

2. **Qdrant Semantic Cache (Tier 2):**
   * Used as a fallback when exact matches aren't found in Redis.
   * Stores embeddings of previous queries for semantic similarity matching.
   * Can respond to questions with the same meaning but different wording.
   * Uses a configurable similarity threshold to ensure relevant responses.

**Cache Processing Flow:**
1. User query is normalized and checked against Redis for an exact match.
2. If no exact match, the query is embedded and checked against Qdrant for a semantic match.
3. If no matches are found in either cache, the query proceeds to the LLM.
4. The LLM's response is then stored in both Redis (for exact match caching) and Qdrant (for semantic caching) to benefit future queries.

**Configuration:**
```dotenv
# Enable/disable the cache system
ENABLE_CACHE=true

# Redis cache configuration
CACHE_REDIS_URL=redis://127.0.0.1:6379 # Or redis://host.docker.internal:6379 from Docker
CACHE_REDIS_TTL=3600  # TTL in seconds

# Qdrant semantic cache configuration
CACHE_QDRANT_URL=http://localhost:6334 # Or http://host.docker.internal:6334 from Docker
CACHE_QDRANT_API_KEY=  # Leave empty if no auth required
CACHE_QDRANT_COLLECTION=prompt_response_cache
CACHE_SIMILARITY_THRESHOLD=0.85  # 0.0-1.0, higher is more strict
```

### Dynamic Topic Resolution

The agent uses a sophisticated prompt-based system to determine which data index is most relevant to a user's query:

1. **Primary Topic Inference:**
   * Uses an LLM to analyze the user question and available schema.
   * Attempts to match the question to the most relevant index.

2. **Fallback Resolution:**
   * If primary inference returns "None" or an invalid index, a fallback mechanism is triggered.
   * Uses a specialized prompt that focuses on indirect relationships and contextual understanding.
   * Maps implied concepts to actual indexes (e.g., "age" → profile index, which contains `birth_date`).

**Customizing Topic Resolution:**
You can customize how the agent resolves topics by modifying prompt templates in `json/prompts.json`:
```json
{
  "query_templates": {
    "rag_topic_inference": "You are given a JSON schema that defines an array `indexes`...",
    "fallback_topic_resolver": "You are helping with database topic selection when our primary classifier returns 'None'..."
    // ... other templates
  }
```
This prompt-based approach makes the agent truly dynamic, allowing it to adapt to different schemas and query types without code changes.

## Building

```bash
cargo build --release
```

## Running

### Natively

Execute the compiled binary. Use a `.env` file for configuration. CLI arguments can override `.env` settings.

```bash
# Run with .env settings
./target/release/dynamic-agent

# Override specific settings
./target/release/dynamic-agent --server-addr 0.0.0.0:8080 --chat-model "new-model"

# See all CLI options
./target/release/dynamic-agent --help
```

### With Docker Compose

We offer two Docker Compose setups:

1.  **All-in-One (`docker-compose-full.yml`):** Includes Dynamic Agent, Redis, and Qdrant. Recommended for easy development.
2.  **Standalone Agent (`docker-compose.yml`):** Runs only the Dynamic Agent. Use if Redis/Qdrant are managed externally.

#### Option 1: All-in-One Setup (`docker-compose-full.yml`)

**Prerequisites:** Docker & Docker Compose.

**Steps:**

1.  **Configure via `.env-agent`:**
    Create a `.env-agent` file. This is the primary way to configure the agent and its connections to the Redis/Qdrant services within the Docker network.
    *   **Example crucial settings in `.env-agent` for `docker-compose-full.yml`:**
        ```dotenv
        # LLMs (e.g., Ollama on host)
        CHAT_BASE_URL="http://host.docker.internal:11434"
        EMBEDDING_BASE_URL="http://host.docker.internal:11434"

        # Vector Store (uses 'qdrant' service from docker-compose-full.yml)
        VECTOR_HOST=http://qdrant:6333

        # History (uses 'redis' service from docker-compose-full.yml)
        HISTORY_HOST=redis://redis:6379

        # Caching (uses 'redis' & 'qdrant' services)
        ENABLE_CACHE=true
        CACHE_REDIS_URL=redis://redis:6379/1
        CACHE_QDRANT_URL=http://qdrant:6333
        ```
    *   *Refer to `.env.example` for all other variables.*

2.  **Run Docker Compose:**
    The `docker-compose-full.yml` defines the `dynamic-agent`, `qdrant`, and `redis` services.
    ```bash
    docker-compose -f docker-compose-full.yml up -d
    ```
    *   To override settings from `.env-agent` directly in `docker-compose-full.yml` (less common for full setup):
        ```dockercompose
        # In docker-compose-full.yml under dynamic-agent service:
        # environment:
        #   CHAT_MODEL: "override_model_here"
        #   VECTOR_INDEX_NAME: "override_index_name"
        ```

#### Option 2: Standalone Agent (`docker-compose.yml`)

**Prerequisites:** Docker & Docker Compose; external Redis, Qdrant, and LLM.

**Steps:**

1.  **Configure via `.env-agent`:**
    Create `.env-agent` pointing to your externally managed services.
    *   **Example crucial settings in `.env-agent` for standalone `docker-compose.yml`:**
        ```dotenv
        # LLMs (e.g., Ollama on host)
        CHAT_BASE_URL="http://host.docker.internal:11434" # Or actual external IP/hostname
        VECTOR_HOST=http://host.docker.internal:6333 # Or actual external IP/hostname
        HISTORY_HOST=redis://host.docker.internal:6379 # Or actual external IP/hostname
        ```
    *   *Refer to `.env.example` for all other variables.*

2.  **Run Docker Compose:**
    The `docker-compose.yml` defines only the `dynamic-agent` service.
    ```bash
    docker-compose -f docker-compose.yml up -d
    # Or simply: docker-compose up -d
    ```
    *   To override settings from `.env-agent` directly in `docker-compose.yml`:
        ```dockercompose
        # In docker-compose.yml under dynamic-agent service:
        # environment:
        #   CHAT_MODEL: "override_model_for_standalone"
        #   SERVER_API_KEY: "direct_api_key_if_not_in_env_agent"
        ```

## Connecting & Interacting

1.  **Connect:** Use a WebSocket client to connect to the agent.
    *   If running natively: `ws://<SERVER_ADDR>` (e.g., `ws://127.0.0.1:4000`).
    *   If running with Docker: `ws://localhost:4000` (since port 4000 is mapped).
    *   Use `wss://` if TLS is enabled.

2.  **WebSocket Authentication:** The server uses HMAC-based authentication for secure WebSocket connections. Clients must include the following query parameters in the WebSocket URL:
    *   `ts`: A UNIX timestamp (in seconds).
    *   `sig`: An HMAC-SHA256 signature of the `ts` value, generated using the shared secret key (`SERVER_API_KEY` from your `.env` or `.env-agent` file).

    Example WebSocket URL:
    ```
    ws://localhost:4000/?ts=1746639884&sig=24efed14e1616e403435034f77899c10441218083d9d047f8aa2435901d486d5
    ```

    **Steps to Generate the HMAC Signature:**
    1.  Compute the current UNIX timestamp (e.g., `ts = Math.floor(Date.now() / 1000)` in JavaScript).
    2.  Use the shared secret key (`SERVER_API_KEY`) to compute the HMAC-SHA256 signature of the `ts` value.
    3.  Encode the resulting HMAC as a hexadecimal string and include it as the `sig` parameter.

    Example JavaScript Code:
    ```javascript
    const ts = Math.floor(Date.now() / 1000).toString();
    const secret = "your-server-api-key";
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
        "raw",
        encoder.encode(secret),
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign"]
    );
    const signature = await crypto.subtle.sign(
        "HMAC",
        key,
        encoder.encode(ts)
    );
    const sig = Array.from(new Uint8Array(signature))
        .map(b => b.toString(16).padStart(2, "0"))
        .join("");

    const ws = new WebSocket(`ws://localhost:4000/?ts=${ts}&sig=${sig}`); 
    ```
    **Note on Webhook Authentication:** The `/api/reload-prompts` webhook endpoint for reloading prompts is currently unauthenticated. Ensure appropriate network security if exposing this endpoint publicly.

3.  **Send Messages:** Send user queries as messages over the WebSocket connection. The expected format is typically a simple text message or a JSON structure depending on the client implementation (e.g., `{"type": "message", "payload": "Tell me about your experience."}`).

4.  **Receive Responses:** The agent will process the message, potentially performing RAG and LLM calls, and send the response back over the same WebSocket connection.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT