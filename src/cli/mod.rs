use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    // --- History Store Args ---
    /// History chat store type (redis, qdrant, memory)
    #[arg(long, env = "HISTORY_TYPE", default_value = "redis")]
    pub history_type: String,

    /// History chat store host endpoint (e.g., redis://127.0.0.1:6379)
    #[arg(long, env = "HISTORY_HOST", default_value = "redis://127.0.0.1:6379")]
    pub history_host: String,

    /// Prefix for Redis history keys.
    #[arg(long, env = "HISTORY_REDIS_PREFIX", default_value = "history:")]
    pub history_redis_prefix: String,

    /// Batch size for Redis SCAN command when listing history.
    #[arg(long, env = "HISTORY_REDIS_SCAN_COUNT", default_value = "100")]
    pub history_redis_scan_count: usize,

    // --- Chat LLM Provider Args ---
    /// Type of LLM provider for chat completion (ollama, openai, anthropic)
    #[arg(long, env = "CHAT_LLM_TYPE", default_value = "ollama")]
    pub chat_llm_type: String,

    /// Base URL for the Chat LLM provider API (e.g., http://localhost:11434 for Ollama)
    #[arg(long, env = "CHAT_BASE_URL")] // No default, let adapters handle defaults if None
    pub chat_base_url: Option<String>,

    /// API Key for the Chat LLM provider (e.g., OpenAI, Anthropic)
    #[arg(long, env = "CHAT_API_KEY", default_value = "")]
    pub chat_api_key: String,

    /// Model name for chat completion (e.g., gpt-4o, llama3, claude-3-opus-20240229)
    #[arg(long, env = "CHAT_MODEL")] // No default, rely on adapter defaults if None
    pub chat_model: Option<String>,

    // --- Embedding LLM Provider Args ---
    /// Type of LLM provider for text embedding (ollama, openai, anthropic)
    #[arg(long, env = "EMBEDDING_LLM_TYPE", default_value = "ollama")]
    pub embedding_llm_type: String,

    /// Base URL for the Embedding LLM provider API (e.g., http://localhost:11434 for Ollama)
    #[arg(long, env = "EMBEDDING_BASE_URL")] // No default, let adapters handle defaults if None
    pub embedding_base_url: Option<String>,

    /// API Key for the Embedding LLM provider (e.g., OpenAI, Anthropic)
    #[arg(long, env = "EMBEDDING_API_KEY", default_value = "")]
    pub embedding_api_key: String,

    /// Model name for text embedding (e.g., text-embedding-3-small, nomic-embed-text)
    #[arg(long, env = "EMBEDDING_MODEL")] // No default, rely on adapter defaults if None
    pub embedding_model: Option<String>,

    // --- Query Generation LLM Provider Args (Optional) ---
    /// Type of LLM provider for query generation (ollama, openai, etc.). Defaults to CHAT_LLM_TYPE if not set.
    #[arg(long, env = "QUERY_LLM_TYPE")]
    pub query_llm_type: Option<String>,

    /// Base URL for the Query Generation LLM provider API. Defaults to CHAT_BASE_URL if not set.
    #[arg(long, env = "QUERY_BASE_URL")]
    pub query_base_url: Option<String>,

    /// API Key for the Query Generation LLM provider. Defaults to CHAT_API_KEY if not set.
    #[arg(long, env = "QUERY_API_KEY")]
    pub query_api_key: Option<String>, // Allow empty string to signify using chat key

    /// Model name for query generation. Defaults to CHAT_MODEL if not set.
    #[arg(long, env = "QUERY_MODEL")]
    pub query_model: Option<String>,

    // --- Vector Store Args ---
    /// Vector database type (redis, chroma, milvus, qdrant, surreal, pinecone)
    #[arg(short = 't', long, env = "VECTOR_TYPE", default_value = "redis")]
    pub vector_type: String,

    /// Vector database URL/host endpoint (e.g., redis://127.0.0.1:6379)
    #[arg(long, env = "VECTOR_HOST", default_value = "redis://127.0.0.1:6379")]
    pub host: String,

    /// Enable authentication for the vector database
    #[arg(long, env = "VECTOR_AUTH", default_value = "false")]
    pub use_auth: bool,

    /// Username for vector database authentication (Milvus, SurrealDB)
    #[arg(short = 'u', env = "VECTOR_USER", long, default_value = "root")]
    pub user: String,

    /// Password for vector database authentication (Milvus, SurrealDB, Redis)
    #[arg(short = 'p', env = "VECTOR_PASS", long, default_value = "")]
    pub pass: String,

    /// API key/token for vector database authentication (Chroma, Qdrant, Pinecone)
    #[arg(short = 'k', env = "VECTOR_SECRET", long, default_value = "")]
    pub secret: String,

    /// Target database name for vector store
    #[arg(long, env = "VECTOR_DATABASE", default_value = "default_database")]
    pub database: String,

    /// Index/Collection name for vector store (Pinecone, Chroma, Qdrant, etc.)
    #[arg(long, env = "VECTOR_INDEX_NAME", default_value = "default_index")]
    pub indexes: String,

    /// Tenant name for multi-tenant vector databases (Chroma)
    #[arg(long, env = "VECTOR_TENANT", default_value = "default_tenant")]
    pub tenant: String,

    /// Namespace for vector databases that support it (SurrealDB)
    #[arg(long, env = "VECTOR_NAMESPACE", default_value = "default_namespace")]
    pub namespace: String,

    /// Vector dimension size
    #[arg(long, env = "VECTOR_DIMENSION", default_value = "768")]
    pub dimension: usize,

    /// Distance metric for vector similarity (l2, ip, cosine, euclidean, dotproduct)
    #[arg(long, env = "VECTOR_METRIC", default_value = "cosine")]
    pub metric: String,

    // --- Pinecone Specific Vector Args ---
    /// Cloud provider for Pinecone (aws, azure, gcp)
    #[arg(long, env = "PINECONE_CLOUD", default_value = "aws")]
    pub cloud: String,

    /// Cloud region for Pinecone (us-east-1, us-west-1, etc.)
    #[arg(long, env = "PINECONE_REGION", default_value = "us-east-1")]
    pub region: String,

    // --- General App Args ---
    /// Auto generate/reload schema for the vector database on startup
    #[arg(long, env = "AUTO_SCHEMA", default_value = "false")]
    pub auto_schema: bool,

    /// Enable debug logging/output
    #[arg(long, env = "DEBUG", default_value = "false")]
    pub debug: bool,

    /// Path to the vector store schema definition file.
    #[arg(long, env = "SCHEMA_PATH", default_value = "json/index_schema.json")]
    pub schema_path: String,

    /// Path to the prompt configuration file.
    #[arg(long, env = "PROMPTS_PATH", default_value = "json/prompts.json")]
    pub prompts_path: String,

    /// Default number of results to retrieve in RAG queries.
    #[arg(long, env = "RAG_DEFAULT_LIMIT", default_value = "20")]
    pub rag_default_limit: usize,

    /// Host address and port for the server to listen on.
    #[arg(long, env = "SERVER_ADDR", default_value = "127.0.0.1:4000")]
    pub server_addr: String,

    /// Optional API Key required for clients to connect to the WebSocket server. If set, clients must provide this key.
    #[arg(long, env = "SERVER_API_KEY")]
    pub server_api_key: Option<String>,

    /// Directory containing vector store function schema definition files (e.g., qdrant.json, redis.json).
    #[arg(long, env = "FUNCTION_SCHEMA_DIR", default_value = "json/query")]
    pub function_schema_dir: String,

    /// Use an LLM to generate vector search queries that specify relevant fields, helping to reduce less relevant results.
    /// (e.g., "title, description, content").
    /// However, since the accuracy depends on the LLM's understanding, the results are not guaranteed to be fully reliable.
    #[arg(long, env = "LLM_QUERY", default_value = "false")]
    pub llm_query: bool,

    // --- Caching Args ---
    /// Enable caching layer (Redis exact + Qdrant semantic).
    #[arg(long, env = "ENABLE_CACHE", default_value = "false")]
    pub enable_cache: bool,

    /// Redis URL for the caching layer.
    #[arg(long, env = "CACHE_REDIS_URL", default_value = "redis://127.0.0.1:6379/1")] // Use DB 1 to avoid collision
    pub cache_redis_url: String,

    /// Qdrant URL for the semantic caching layer.
    #[arg(long, env = "CACHE_QDRANT_URL", default_value = "http://localhost:6334")]
    pub cache_qdrant_url: String,

    /// Optional API Key for the Qdrant cache instance.
    #[arg(long, env = "CACHE_QDRANT_API_KEY")]
    pub cache_qdrant_api_key: Option<String>,

    /// Qdrant collection name for caching prompts and responses.
    #[arg(long, env = "CACHE_QDRANT_COLLECTION", default_value = "prompt_response_cache")]
    pub cache_qdrant_collection: String,

    /// Cosine similarity threshold for considering a Qdrant cache hit valid (0.0 to 1.0).
    #[arg(long, env = "CACHE_SIMILARITY_THRESHOLD", default_value = "0.5")]
    pub cache_similarity_threshold: f32,

    /// Time-to-live (TTL) in seconds for Redis cache entries. 0 means no TTL.
    #[arg(long, env = "CACHE_REDIS_TTL", default_value = "3600")] // 1 hour
    pub cache_redis_ttl: usize,

    /// Optional path to the TLS certificate file (PEM format) for enabling WSS. Requires --tls-key.
    #[arg(long, env = "TLS_CERT_PATH")]
    pub tls_cert_path: Option<String>,

    /// Optional path to the TLS private key file (PEM format) for enabling WSS. Requires --tls-cert.
    #[arg(long, env = "TLS_KEY_PATH")]
    pub tls_key_path: Option<String>,

    #[arg(long, env = "ENABLE_TLS", default_value = "false")]
    pub enable_tls: bool,
}
