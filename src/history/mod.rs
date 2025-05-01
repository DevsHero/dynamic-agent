mod qdrant;
mod redis;
use async_trait::async_trait;
use log::info;
use std::error::Error;
use crate::cli::Args;
use std::sync::Arc;
use crate::models::chat::Conversation;
use crate::llm::embedding::new_client as new_embedding_client;
use crate::llm::LlmConfig;

#[async_trait]
pub trait HistoryStore: Send + Sync {
    async fn add_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    async fn get_conversation(
        &self,
        conversation_id: &str,
        limit: usize
    ) -> Result<Conversation, Box<dyn Error + Send + Sync>>;
}

pub fn create_history_store(
    args: &Args
) -> Result<Arc<dyn HistoryStore>, Box<dyn Error + Send + Sync>> {
    match args.history_type.to_lowercase().as_str() {
        "redis" => {
            let store = redis::RedisHistoryStore::new(args.clone())?;
            Ok(Arc::new(store))
        }
        "qdrant" => {
            let embedding_config = LlmConfig {
                llm_type: args.embedding_llm_type
                    .parse()
                    .map_err(|e| format!("Invalid embedding LLM type: {}", e))?,
                base_url: args.embedding_base_url.clone(),
                api_key: Some(args.embedding_api_key.clone()).filter(|k| !k.is_empty()),
                completion_model: None,
                embedding_model: args.embedding_model.clone(),
            };
            let embedding_client = new_embedding_client(&embedding_config)?;
            let store = qdrant::QdrantHistoryStore::new(args.clone(), embedding_client)?;

            Ok(Arc::new(store))
        }
        _ =>
            Err(
                Box::new(
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("Unsupported history store type: {}", args.history_type)
                    )
                )
            ),
    }
}

pub fn initialize_history_store(
    args: &Args
) -> Result<Arc<dyn HistoryStore>, Box<dyn Error + Send + Sync>> {
    info!("Chat history will be stored in: {} at {}", args.history_type, args.history_host);
    create_history_store(&args)
}

pub fn format_history_for_prompt(conversation: &Conversation) -> String {
    if conversation.messages.is_empty() {
        return String::new();
    }
    let mut result = String::from("Previous conversation:\n");
    for msg in &conversation.messages {
        let role_display = match msg.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };

        result.push_str(&format!("{}: {}\n", role_display, msg.content));
    }

    result
}
