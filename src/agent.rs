use crate::history::{ format_history_for_prompt, initialize_history_store, HistoryStore };
use crate::rag::rag::{ RagEngine, RagQueryArgs };

use vector_nexus::db::{
    VectorStore,
    get_store_type as get_vector_store_type,
    create_vector_store,
    VectorStoreConfig,
};
use vector_nexus::schema::SchemaFile;

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{ Distance, PointStruct, SearchPointsBuilder, VectorParams };
use qdrant_client::qdrant::vectors_config::Config as VectorsConfig;
use qdrant_client::qdrant::CreateCollectionBuilder;

use serde_json::Value as JsonValue;
use serde::{ Deserialize, Serialize };
use uuid::Uuid;

use redis::Client as RedisClient;
use redis::AsyncCommands;

use crate::cli::Args;
use crate::config::prompt::{ self, PromptConfig };
use crate::llm::{ parse_llm_type, LlmConfig };
use crate::llm::chat::{ ChatClient, new_client as new_chat_client };
use crate::llm::embedding::{ EmbeddingClient, new_client as new_embedding_client };

use log::{ info, warn, error };
use std::error::Error;
use std::sync::Arc;
use std::fs;
use std::time::SystemTime;
use std::path::PathBuf;

const HISTORY_FOR_PROMPT_LEN: usize = 6;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CachePayload {
    normalized_prompt: String,
    response: String,
}

#[derive(Clone)]
pub struct AIAgent {
    chat_client: Arc<dyn ChatClient>,
    embedding_client: Arc<dyn EmbeddingClient>,
    query_generation_client: Arc<dyn ChatClient>,
    rag_tool: RagEngine,
    prompt_config: Arc<PromptConfig>,
    vector_store: Arc<dyn VectorStore>,
    history_store: Arc<dyn HistoryStore>,
    schema_last_reload: Option<SystemTime>,
    rag_default_limit: usize,
    vector_type: String,
    enable_cache: bool,
    cache_redis_conn: Option<Arc<tokio::sync::Mutex<redis::aio::MultiplexedConnection>>>,
    cache_qdrant_client: Option<Arc<Qdrant>>,
    cache_qdrant_collection: String,
    cache_similarity_threshold: f32,
    cache_redis_ttl: usize,
}

impl AIAgent {
    async fn initialize_llm_clients(
        args: &Args
    ) -> Result<
        (Arc<dyn ChatClient>, Arc<dyn EmbeddingClient>, Arc<dyn ChatClient>),
        Box<dyn Error + Send + Sync>
    > {
        let chat_llm_type = parse_llm_type(&args.chat_llm_type)?;
        let chat_api_key = if !args.chat_api_key.is_empty() {
            Some(args.chat_api_key.clone())
        } else {
            None
        };
        let chat_config = LlmConfig {
            llm_type: chat_llm_type,
            base_url: args.chat_base_url.clone(),
            api_key: chat_api_key,
            completion_model: args.chat_model.clone(),
            embedding_model: None,
        };
        let chat_client = new_chat_client(&chat_config)?;
        info!(
            "Chat client configured: Type={}, Model={:?}, BaseURL={:?}",
            args.chat_llm_type,
            chat_config.completion_model.as_deref().unwrap_or("adapter default"),
            chat_config.base_url.as_deref().unwrap_or("adapter default")
        );

        let embedding_llm_type = parse_llm_type(&args.embedding_llm_type)?;
        let embedding_api_key = if !args.embedding_api_key.is_empty() {
            Some(args.embedding_api_key.clone())
        } else {
            None
        };
        let embedding_config = LlmConfig {
            llm_type: embedding_llm_type,
            base_url: args.embedding_base_url.clone(),
            api_key: embedding_api_key,
            embedding_model: args.embedding_model.clone(),
            completion_model: None,
        };
        let embedding_client = new_embedding_client(&embedding_config)?;
        info!(
            "Embedding client configured: Type={}, Model={:?}, BaseURL={:?}",
            args.embedding_llm_type,
            embedding_config.embedding_model.as_deref().unwrap_or("adapter default"),
            embedding_config.base_url.as_deref().unwrap_or("adapter default")
        );

        let query_llm_type_str = match &args.query_llm_type {
            Some(s) if !s.trim().is_empty() => s.as_str(),
            _ => &args.chat_llm_type,
        };
        let query_llm_type = parse_llm_type(query_llm_type_str)?;
        let query_api_key_str = args.query_api_key.as_deref().unwrap_or(&args.chat_api_key);
        let query_api_key = {
            let s = query_api_key_str.to_string();
            if !s.is_empty() {
                Some(s)
            } else {
                None
            }
        };
        let query_config = LlmConfig {
            llm_type: query_llm_type,
            base_url: args.query_base_url.clone().or_else(|| args.chat_base_url.clone()),
            api_key: query_api_key,
            completion_model: args.query_model.clone().or_else(|| args.chat_model.clone()),
            embedding_model: None,
        };
        let query_generation_client = new_chat_client(&query_config)?;
        info!(
            "Query Generation client configured: Type={}, Model={:?}, BaseURL={:?}",
            query_llm_type_str,
            query_config.completion_model.as_deref().unwrap_or("adapter default"),
            query_config.base_url.as_deref().unwrap_or("adapter default")
        );

        Ok((chat_client, embedding_client, query_generation_client))
    }

    async fn initialize_vector_store(
        args: &Args
    ) -> Result<Arc<dyn VectorStore>, Box<dyn Error + Send + Sync>> {
        info!("Connecting to vector store at: {}", args.host);
        let vector_store_type = get_vector_store_type(args.vector_type.as_str()).map_err(|e|
            format!("Failed to get vector store type: {}", e)
        )?;
        let vector_store_config = VectorStoreConfig {
            store_type: vector_store_type,
            host: args.host.clone(),
            api_key: Some(args.secret.clone()),
            tenant: Some(args.tenant.clone()),
            database: Some(args.database.clone()),
            namespace: Some(args.namespace.clone()),
            index_name: Some(args.indexes.clone()),
            user: Some(args.user.clone()),
            pass: Some(args.pass.clone()),
            dimension: Some(args.dimension.clone()),
            metric: Some(args.metric.clone()),
        };
        create_vector_store(vector_store_config.clone()).await
    }

    async fn load_configs_and_schemas(
        args: &Args,
        vector_store: &Arc<dyn VectorStore>
    ) -> Result<(SchemaFile, Arc<PromptConfig>, JsonValue), Box<dyn Error + Send + Sync>> {
        let schema_path = &args.schema_path;
        let prompts_path = &args.prompts_path;
        let function_schema_dir = &args.function_schema_dir;
        let schemas = vector_store.generate_schema(schema_path).await?;
        let schema_file = SchemaFile { indexes: schemas };
        let prompt_config_arc = prompt::load_prompts(prompts_path)?;
        let function_schema_path = PathBuf::from(function_schema_dir).join(
            format!("{}.json", args.vector_type)
        );
        let function_schema_str = fs
            ::read_to_string(&function_schema_path)
            .map_err(|e|
                format!(
                    "Failed to read function schema file {}: {}",
                    function_schema_path.display(),
                    e
                )
            )?;
        let function_schema: JsonValue = serde_json
            ::from_str(&function_schema_str)
            .map_err(|e|
                format!(
                    "Failed to parse function schema from {}: {}",
                    function_schema_path.display(),
                    e
                )
            )?;
        info!(
            "Loaded function schema for type '{}' from: {}",
            args.vector_type,
            function_schema_path.display()
        );

        Ok((schema_file, prompt_config_arc, function_schema))
    }

    async fn initialize_cache_clients(
        args: &Args
    ) -> Result<
        (Option<Arc<tokio::sync::Mutex<redis::aio::MultiplexedConnection>>>, Option<Arc<Qdrant>>),
        Box<dyn Error + Send + Sync>
    > {
        if !args.enable_cache {
            info!("Cache disabled.");
            return Ok((None, None));
        }

        info!("Cache enabled. Initializing cache clients...");
        let mut cache_redis_conn = None;
        let mut cache_qdrant_client = None;

        match RedisClient::open(args.cache_redis_url.as_str()) {
            Ok(redis_client) => {
                match redis_client.get_multiplexed_async_connection().await {
                    Ok(conn) => {
                        cache_redis_conn = Some(Arc::new(tokio::sync::Mutex::new(conn)));
                        info!("✅ Cache Redis client connected to {}", args.cache_redis_url);
                    }
                    Err(e) => error!("Failed to get Redis cache connection: {}", e),
                }
            }
            Err(e) => error!("Failed to create Redis cache client: {}", e),
        }

        match
            Qdrant::from_url(&args.cache_qdrant_url)
                .api_key(args.cache_qdrant_api_key.clone())
                .build()
        {
            Ok(qdrant_client_instance) => {
                let qdrant_arc = Arc::new(qdrant_client_instance);
                cache_qdrant_client = Some(Arc::clone(&qdrant_arc));
                info!("✅ Cache Qdrant client connected to {}", args.cache_qdrant_url);

                let collection_name = &args.cache_qdrant_collection;
                if qdrant_arc.collection_info(collection_name).await.is_err() {
                    info!("Qdrant cache collection '{}' not found. Attempting to create...", collection_name);
                    let embed_dim = args.dimension;
                    let create_collection = CreateCollectionBuilder::new(collection_name.clone())
                        .vectors_config(
                            VectorsConfig::Params(VectorParams {
                                size: embed_dim as u64,
                                distance: Distance::Cosine.into(),
                                ..Default::default()
                            })
                        )
                        .build();
                    match qdrant_arc.create_collection(create_collection).await {
                        Ok(_) =>
                            info!(
                                "✅ Successfully created Qdrant cache collection '{}' with dimension {}.",
                                collection_name,
                                embed_dim
                            ),
                        Err(e) =>
                            error!(
                                "Failed to create Qdrant cache collection '{}': {}",
                                collection_name,
                                e
                            ),
                    }
                } else {
                    info!("✅ Qdrant cache collection '{}' already exists.", collection_name);
                }
            }
            Err(e) => error!("Failed to create Qdrant cache client: {}", e),
        }

        Ok((cache_redis_conn, cache_qdrant_client))
    }

    pub async fn new(args: Args) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let (chat_client, embedding_client, query_generation_client) = Self::initialize_llm_clients(
            &args
        ).await?;
        let vector_store = Self::initialize_vector_store(&args).await?;
        let history_store = initialize_history_store(&args)?;
        let (schema_file, prompt_config_arc, function_schema) = Self::load_configs_and_schemas(
            &args,
            &vector_store
        ).await?;
        let (cache_redis_conn, cache_qdrant_client) = Self::initialize_cache_clients(&args).await?;

        let rag_tool = RagEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&chat_client),
            Arc::clone(&embedding_client),
            Arc::clone(&query_generation_client),
            schema_file.indexes,
            Arc::clone(&prompt_config_arc),
            function_schema,
            args.vector_type.clone(),
            args.rag_default_limit,
            args.llm_query
        );

        Ok(Self {
            chat_client,
            embedding_client,
            query_generation_client,
            rag_tool,
            prompt_config: prompt_config_arc,
            vector_store,
            history_store,
            schema_last_reload: Some(SystemTime::now()),
            rag_default_limit: args.rag_default_limit,
            vector_type: args.vector_type.clone(),
            enable_cache: args.enable_cache,
            cache_redis_conn,
            cache_qdrant_client,
            cache_qdrant_collection: args.cache_qdrant_collection,
            cache_similarity_threshold: args.cache_similarity_threshold,
            cache_redis_ttl: args.cache_redis_ttl,
        })
    }

    async fn check_redis_cache(
        &self,
        normalized_key: &str
    ) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
        if !self.enable_cache || self.cache_redis_conn.is_none() {
            return Ok(None);
        }

        let mut conn_guard = self.cache_redis_conn.as_ref().unwrap().lock().await;
        match conn_guard.get::<_, String>(normalized_key).await {
            Ok(cached_response) => {
                info!("✅ Cache Hit (Redis Exact on normalized key)");
                Ok(Some(cached_response))
            }
            Err(redis::RedisError { .. }) => Ok(None),
        }
    }

    async fn check_qdrant_cache(
        &self,
        normalized_message: &str
    ) -> Result<Option<(String, Vec<f32>)>, Box<dyn Error + Send + Sync>> {
        if !self.enable_cache || self.cache_qdrant_client.is_none() {
            return Ok(None);
        }

        let embed_resp = self.embedding_client.embed(normalized_message).await?;
        let vec_f32 = embed_resp.embedding;
        let qdrant_client = self.cache_qdrant_client.as_ref().unwrap();
        let search_response = qdrant_client.search_points(
            SearchPointsBuilder::new(
                &self.cache_qdrant_collection,
                vec_f32.clone(),
                1
            ).score_threshold(self.cache_similarity_threshold)
        ).await?;

        if let Some(point) = search_response.result.first() {
            if point.score >= self.cache_similarity_threshold {
                let mut map = serde_json::Map::new();
                for (k, v) in point.payload.clone() {
                    match serde_json::to_value(v) {
                        Ok(val) => {
                            map.insert(k, val);
                        }
                        Err(err) => warn!("Skipping field '{}' in cache payload: {}", k, err),
                    }
                }
                let obj = JsonValue::Object(map);
                match serde_json::from_value::<CachePayload>(obj) {
                    Ok(cached) => {
                        info!("✅ Cache Hit (Qdrant Semantic, Score: {:.4})", point.score);
                        return Ok(Some((cached.response, vec_f32)));
                    }
                    Err(err) => {
                        warn!("Malformed cache payload, skipping hit: {}", err);
                        return Ok(None);
                    }
                }
            }
        }

        Ok(None)
    }

    async fn prime_redis_cache(&self, key: &str, value: &str) {
        if let Some(redis_conn_arc) = &self.cache_redis_conn {
            let mut conn_guard = redis_conn_arc.lock().await;
            let ttl = self.cache_redis_ttl;
            let set_cmd = if ttl > 0 {
                conn_guard.set_ex::<_, _, ()>(key, value, ttl as u64)
            } else {
                conn_guard.set::<_, _, ()>(key, value)
            };
            if let Err(e) = set_cmd.await {
                error!("Failed to prime Redis cache for key '{}': {}", key, e);
            } else {
                info!("📝 Cached response in Redis (key: '{}')", key);
            }
        }
    }

    async fn execute_llm_interaction(
        &self,
        conversation_id: &str,
        message: &str
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let conversation = self.history_store.get_conversation(
            conversation_id,
            HISTORY_FOR_PROMPT_LEN
        ).await?;
        let history_str = format_history_for_prompt(&conversation);
        let intent_prompt = prompt::get_intent_prompt(&self.prompt_config, message)?;
        let intent_response = self.chat_client.complete(&intent_prompt).await?;
        let intent_name = intent_response.response.trim();
        let intent_definition = self.prompt_config.intents
            .get(intent_name)
            .ok_or_else(|| prompt::PromptError::IntentNotFound(intent_name.to_string()))?;

        match intent_definition.action.as_str() {
            "call_rag_tool" => {
                let rag_args = RagQueryArgs {
                    query: message.to_string(),
                    limit: Some(self.rag_default_limit),
                };
                self.rag_tool.query_and_answer(rag_args, message).await
            }
            "general_llm_call" => {
                let prompt_with_history = format!("{}\n\nUser: {}", history_str, message);
                let resp = self.chat_client.complete(&prompt_with_history).await?;
                Ok(resp.response)
            }
            unknown_action => {
                Err(
                    Box::new(
                        prompt::PromptError::ActionError(
                            format!("Action '{}' is not implemented.", unknown_action)
                        )
                    )
                )
            }
        }
    }

    async fn update_qdrant_cache(
        &self,
        normalized_prompt: &str,
        response: &str,
        embedding: Vec<f32>
    ) {
        if !self.enable_cache || self.cache_qdrant_client.is_none() || embedding.is_empty() {
            return;
        }

        let qdrant_client_arc = self.cache_qdrant_client.as_ref().unwrap();
        let payload = CachePayload {
            normalized_prompt: normalized_prompt.to_string(),
            response: response.to_string(),
        };

        match serde_json::to_value(payload) {
            Ok(JsonValue::Object(map)) => {
                let point_id = Uuid::new_v4().to_string();
                let point = PointStruct::new(point_id, embedding, map);
                let upsert_op = qdrant_client::qdrant::UpsertPointsBuilder
                    ::new(&self.cache_qdrant_collection, vec![point])
                    .build();

                match qdrant_client_arc.upsert_points(upsert_op).await {
                    Ok(_) => info!("📝 Cached response in Qdrant (normalized key vector)"),
                    Err(e) => error!("Failed to update Qdrant cache: {}", e),
                }
            }
            Ok(_) | Err(_) => warn!("Failed to serialize/convert cache payload for Qdrant"),
        }
    }

    pub async fn process_message(
        &self,
        conversation_id: &str,
        message: &str
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let normalized_message = message.trim().to_lowercase();
        info!("ℹ️ Normalized message for cache lookup: '{}'", normalized_message);

        if let Some(cached_response) = self.check_redis_cache(&normalized_message).await? {
            if let Err(e) = self.history_store.add_message(conversation_id, "user", message).await {
                warn!("History write (user) failed: {}", e);
            }
            if
                let Err(e) = self.history_store.add_message(
                    conversation_id,
                    "assistant",
                    &cached_response
                ).await
            {
                warn!("History write (assistant) failed: {}", e);
            }
            return Ok(cached_response);
        }

        let qdrant_hit = self.check_qdrant_cache(&normalized_message).await?;
        let mut maybe_embedding = None;

        if let Some((cached_response, embedding_used)) = qdrant_hit {
            if !cached_response.is_empty() {
                self.prime_redis_cache(&normalized_message, &cached_response).await;
                if
                    let Err(e) = self.history_store.add_message(
                        conversation_id,
                        "user",
                        message
                    ).await
                {
                    warn!("History write (user) failed: {}", e);
                }
                if
                    let Err(e) = self.history_store.add_message(
                        conversation_id,
                        "assistant",
                        &cached_response
                    ).await
                {
                    warn!("History write (assistant) failed: {}", e);
                }
                return Ok(cached_response);
            } else {
                maybe_embedding = Some(embedding_used);
            }
        }
        info!("ℹ️ Cache Miss. Proceeding with LLM call...");
        let response_content = self
            .execute_llm_interaction(conversation_id, message).await
            .map_err(|e| {
                error!("LLM interaction error: {}", e);
                e
            })?;

        if self.enable_cache {
            self.prime_redis_cache(&normalized_message, &response_content).await;
            let embedding_to_use = if let Some(e) = maybe_embedding {
                e
            } else {
                self.embedding_client.embed(&normalized_message).await?.embedding
            };
            self.update_qdrant_cache(
                &normalized_message,
                &response_content,
                embedding_to_use
            ).await;
        }

        self.history_store.add_message(conversation_id, "user", message).await?;
        self.history_store.add_message(conversation_id, "assistant", &response_content).await?;

        Ok(response_content)
    }

    pub async fn reload_prompts_if_changed(
        &mut self,
        args: &Args
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let prompts_path = &args.prompts_path;
        let schema_path = &args.schema_path;
        let function_schema_dir = &args.function_schema_dir;

        let result = prompt::reload_prompts_if_changed(prompts_path, &self.prompt_config)?;

        if let Some(new_config) = result {
            let schema_text = fs::read_to_string(schema_path)?;
            let schema_file: SchemaFile = serde_json::from_str(&schema_text)?;
            let function_schema_path = PathBuf::from(function_schema_dir).join(
                format!("{}.json", args.vector_type)
            );
            let function_schema: JsonValue = match fs::read_to_string(&function_schema_path) {
                Ok(text) => serde_json::from_str(&text)?,
                Err(_) => {
                    warn!(
                        "Function schema not found during reload at {}. Using empty schema.",
                        function_schema_path.display()
                    );
                    JsonValue::Object(serde_json::Map::new())
                }
            };

            self.prompt_config = Arc::clone(&new_config);

            self.rag_tool = RagEngine::new(
                Arc::clone(&self.vector_store),
                Arc::clone(&self.chat_client),
                Arc::clone(&self.embedding_client),
                Arc::clone(&self.query_generation_client),
                schema_file.indexes.clone(),
                new_config,
                function_schema,
                self.vector_type.clone(),
                args.rag_default_limit,
                args.llm_query
            );

            info!("Prompts and function schema successfully reloaded");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn reload_schema_if_needed(
        &mut self,
        args: &Args
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let schema_path = &args.schema_path;
        let function_schema_dir = &args.function_schema_dir;
        let schemas = self.vector_store.generate_schema(schema_path).await?;
        let function_schema_path = PathBuf::from(function_schema_dir).join(
            format!("{}.json", args.vector_type)
        );
        let function_schema: JsonValue = match fs::read_to_string(&function_schema_path) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(_) => {
                warn!(
                    "Function schema not found during schema reload at {}. Using empty schema.",
                    function_schema_path.display()
                );
                JsonValue::Object(serde_json::Map::new())
            }
        };

        self.rag_tool = RagEngine::new(
            Arc::clone(&self.vector_store),
            Arc::clone(&self.chat_client),
            Arc::clone(&self.embedding_client),
            Arc::clone(&self.query_generation_client),
            schemas,
            Arc::clone(&self.prompt_config),
            function_schema,
            self.vector_type.clone(),
            args.rag_default_limit,
            args.llm_query
        );

        self.schema_last_reload = Some(SystemTime::now());
        info!("Schema successfully reloaded");
        Ok(true)
    }
}
