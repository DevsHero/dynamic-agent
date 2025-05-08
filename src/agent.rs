use crate::history::{ format_history_for_prompt, initialize_history_store, HistoryStore };
use crate::rag::rag::{ RagEngine, RagQueryArgs };

use vector_nexus::db::{
    VectorStore,
    get_store_type as get_vector_store_type,
    create_vector_store,
    VectorStoreConfig,
};
use vector_nexus::schema::SchemaFile;

use serde_json::Value as JsonValue;
use serde::{ Deserialize, Serialize };

use crate::cli::Args;
use crate::config::prompt::{ self, PromptConfig };
use crate::llm::{ parse_llm_type, LlmConfig };
use crate::llm::chat::{ ChatClient, new_client as new_chat_client };
use crate::llm::embedding::{ EmbeddingClient, new_client as new_embedding_client };

use crate::cache::{self, CacheClients};

use log::{ info, warn };
use std::error::Error;
use std::sync::Arc;
use std::fs;
use std::time::SystemTime;
use std::path::PathBuf;
use tokio::sync::RwLock;

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
    prompt_config: Arc<RwLock<Arc<PromptConfig>>>,
    vector_store: Arc<dyn VectorStore>,
    history_store: Arc<dyn HistoryStore>,
    schema_last_reload: Option<SystemTime>,
    rag_default_limit: usize,
    vector_type: String,
    enable_cache: bool,
    cache: CacheClients,
    prompts_path: String, 
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
    ) -> Result<(SchemaFile, JsonValue), Box<dyn Error + Send + Sync>> {
        let schema_path = &args.schema_path;
        let function_schema_dir = &args.function_schema_dir;
        let schemas = vector_store.generate_schema(schema_path).await?;
        let schema_file = SchemaFile { indexes: schemas };
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

        Ok((schema_file, function_schema))
    }

    pub async fn new(
        args: Args, 
        shared_prompt_config: Arc<RwLock<Arc<PromptConfig>>>
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let (chat_client, embedding_client, query_generation_client) = Self::initialize_llm_clients(
            &args
        ).await?;
        let vector_store = Self::initialize_vector_store(&args).await?;
        let history_store = initialize_history_store(&args)?;
        let (schema_file, function_schema) = Self::load_configs_and_schemas(
            &args,
            &vector_store
        ).await?;
        let cache = cache::init(&args).await;

        let current_prompt_config = shared_prompt_config.read().await.clone();

        let rag_tool = RagEngine::new(
            Arc::clone(&vector_store),
            Arc::clone(&chat_client),
            Arc::clone(&embedding_client),
            Arc::clone(&query_generation_client),
            schema_file.indexes,
            current_prompt_config,
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
            prompt_config: shared_prompt_config,
            vector_store,
            history_store,
            schema_last_reload: Some(SystemTime::now()),
            rag_default_limit: args.rag_default_limit,
            vector_type: args.vector_type.clone(),
            enable_cache: args.enable_cache,
            cache,
            prompts_path: args.prompts_path.clone(), 
        })
    }

    async fn execute_llm_interaction(
        &self,
        conversation_id: &str,
        message: &str
    ) -> Result<String, Box<dyn Error + Send + Sync>> {

        if let Ok(true) = prompt::check_local_prompt_file_changed(&self.prompts_path) {
            info!("Local prompts file changed, reloading...");
            if let Ok(new_config) = prompt::load_prompts_from_str(&self.prompts_path) {
                let mut write_lock = self.prompt_config.write().await;
                *write_lock = new_config;
                info!("Local prompts reloaded successfully");
            }
        }
        
        let conversation = self.history_store.get_conversation(
            conversation_id,
            HISTORY_FOR_PROMPT_LEN
        ).await?;
        let history_str = format_history_for_prompt(&conversation);
        let current_prompt_config = self.prompt_config.read().await;
        let intent_prompt = prompt::get_intent_prompt(&current_prompt_config, message)?;
        let intent_response = self.chat_client.complete(&intent_prompt).await?;
        let intent_name = intent_response.response.trim();
        let intent_definition = current_prompt_config.intents
            .get(intent_name)
            .ok_or_else(|| prompt::PromptError::IntentNotFound(intent_name.to_string()))?;

        match intent_definition.action.as_str() {
            "call_rag_tool" => {
                let rag_args = RagQueryArgs {
                    query: message.to_string(),
                    limit: Some(self.rag_default_limit),
                };
                
                let (documents, topic, schema_json) = self.rag_tool.get_documents_for_query(rag_args).await?;
                
                let docs_text = documents.iter()
                    .map(|doc| doc.to_string())
                    .collect::<Vec<_>>()
                    .join("\n");
                
                let final_prompt = prompt::get_rag_final_prompt(
                    &current_prompt_config, 
                    &schema_json,
                    &topic,
                    &docs_text,
                    message
                )?;
                
                drop(current_prompt_config);
                let resp = self.chat_client.complete(&final_prompt).await?;
                Ok(resp.response)
            }
            "general_llm_call" => {
                let prompt_with_history = format!("{}\n\nUser: {}", history_str, message);
                drop(current_prompt_config);
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

    pub async fn process_message(
        &self,
        conversation_id: &str,
        message: &str
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let normalized = message.trim().to_lowercase();
        info!("ℹ️ Normalized message for cache lookup: '{}'", normalized);

        if self.enable_cache {
            if let Some((resp, _emb)) =
                cache::check(&self.cache, &normalized, &*self.embedding_client).await?
            {
                info!("✅ Cache Hit");
                self.history_store.add_message(conversation_id, "user", message).await?;
                self.history_store.add_message(conversation_id, "assistant", &resp).await?;
                return Ok(resp.to_string());
            }
        }

        info!("ℹ️ Cache Miss. Proceeding with LLM call…");
        let reply = self.execute_llm_interaction(conversation_id, message).await?;

        if self.enable_cache {
            let emb_to_use = self.embedding_client.embed(&normalized).await?.embedding;
            cache::update(&self.cache, &normalized, &reply, emb_to_use).await?;
        }

        self.history_store.add_message(conversation_id, "user", message).await?;
        self.history_store.add_message(conversation_id, "assistant", &reply).await?;

        Ok(reply)
    }

    pub async fn reload_prompts_if_changed(
        &mut self,
        args: &Args
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        let prompts_path = &args.prompts_path;
        let schema_path = &args.schema_path;
        let function_schema_dir = &args.function_schema_dir;

        let current_prompt_config = self.prompt_config.read().await.clone();
        let result = prompt::reload_prompts_if_changed(prompts_path, &current_prompt_config)?;

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

            let mut prompt_write = self.prompt_config.write().await;
            *prompt_write = Arc::clone(&new_config);
            drop(prompt_write);

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

        let current_prompt_config = self.prompt_config.read().await.clone();

        self.rag_tool = RagEngine::new(
            Arc::clone(&self.vector_store),
            Arc::clone(&self.chat_client),
            Arc::clone(&self.embedding_client),
            Arc::clone(&self.query_generation_client),
            schemas,
            current_prompt_config,
            function_schema,
            self.vector_type.clone(),
            args.rag_default_limit,
            args.llm_query
        );

        self.schema_last_reload = Some(SystemTime::now());
        info!("Schema successfully reloaded");
        Ok(true)
    }

    pub async fn force_refresh_remote_prompts(
        &mut self,
        args: &Args
    ) -> Result<bool, Box<dyn Error + Send + Sync>> {
        if !args.enable_remote_prompts {
            return Ok(false);
        }
        
        let project_id = args.remote_prompts_project_id.as_deref().ok_or_else(|| {
            "Missing REMOTE_PROMPTS_PROJECT_ID".to_string()
        })?;
        
        let sa_key_path = args.remote_prompts_sa_key_path.as_deref().ok_or_else(|| {
            "Missing REMOTE_PROMPTS_SA_KEY_PATH".to_string()
        })?;
        
        let remote_client = crate::config::remote_config::RemoteConfigClient::new();
        
        match remote_client.fetch_config(project_id, sa_key_path).await {
            Ok(Some(json_str)) => {  
                match crate::config::prompt::load_prompts_from_str(&json_str) {
                    Ok(new_config) => {
                        let mut w = self.prompt_config.write().await;
                        *w = new_config;
                        drop(w);
                        
                        info!("Remote prompts successfully refreshed via webhook");
                        Ok(true)
                    }
                    Err(e) => Err(Box::new(e)),
                }
            },
            Ok(None) => {
                warn!("No remote prompts found for project ID: {}", project_id);
                Ok(false)
            },
            Err(e) => Err(Box::new(e)),
        }
    }
}
