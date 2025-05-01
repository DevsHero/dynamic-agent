pub mod ollama;
pub mod openai;
pub mod gemini;
pub mod anthropic;
pub mod deepseek;
pub mod xai;
pub mod groq;

use async_trait::async_trait;
use std::error::Error as StdError;
use std::sync::Arc;
use log::warn;

use super::{ LlmConfig, LlmType };
use self::ollama::OllamaEmbeddingClient;
use self::openai::OpenAIEmbeddingClient;
use self::gemini::GoogleEmbeddingClient as GeminiEmbeddingClient;
use self::anthropic::AnthropicEmbeddingClient;
use self::deepseek::DeepSeekEmbeddingClient;
use self::xai::XAIEmbeddingClient;
use self::groq::GroqEmbeddingClient;

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
}

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>>;
}

pub fn new_client(
    config: &LlmConfig
) -> Result<Arc<dyn EmbeddingClient>, Box<dyn StdError + Send + Sync>> {
    let client: Arc<dyn EmbeddingClient> = match config.llm_type {
        LlmType::Ollama => {
            let specific_client = OllamaEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::OpenAI => {
            let specific_client = OpenAIEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Gemini => {
            let specific_client = GeminiEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Anthropic => {
            warn!(
                "WARNING: Creating Anthropic embedding client. This backend likely does not support embeddings."
            );
            let specific_client = AnthropicEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::DeepSeek => {
            if
                config.embedding_model.is_none() ||
                config.embedding_model.as_deref() == Some("deepseek-chat")
            {
                warn!(
                    "WARNING: Using default/chat model for DeepSeek embeddings. Verify the correct embedding model name."
                );
            }
            let specific_client = DeepSeekEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::XAI => {
            warn!(
                "WARNING: Creating XAI/Grok embedding client. This backend likely does not support embeddings."
            );
            let specific_client = XAIEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Groq => {
            warn!(
                "WARNING: Creating Groq embedding client. This backend likely does not support embeddings."
            );
            let specific_client = GroqEmbeddingClient::from_config(config)?;
            Arc::new(specific_client)
        }
    };
    Ok(client)
}
