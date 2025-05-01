pub mod ollama;
pub mod openai;
pub mod gemini;
pub mod anthropic;
pub mod deepseek;
pub mod groq;
pub mod xai;

use async_trait::async_trait;
use serde::Deserialize;
use std::error::Error as StdError;
use std::sync::Arc;
use super::{ LlmConfig, LlmType };
use self::ollama::OllamaClient;
use self::openai::OpenAIChatClient;
use self::gemini::GeminiChatClient;
use self::anthropic::AnthropicChatClient;
use self::deepseek::DeepSeekChatClient;
use self::groq::GroqChatClient;
use self::xai::XAIChatClient;

#[derive(Deserialize, Debug, Clone)]
pub struct CompletionResponse {
    pub response: String,
}

#[async_trait]
pub trait ChatClient: Send + Sync {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>>;
}

pub fn new_client(
    config: &LlmConfig
) -> Result<Arc<dyn ChatClient>, Box<dyn StdError + Send + Sync>> {
    let client: Arc<dyn ChatClient> = match config.llm_type {
        LlmType::Ollama => {
            let specific_client = OllamaClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::OpenAI => {
            let specific_client = OpenAIChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Gemini => {
            let specific_client = GeminiChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Anthropic => {
            let specific_client = AnthropicChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::DeepSeek => {
            let specific_client = DeepSeekChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::Groq => {
            let specific_client = GroqChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
        LlmType::XAI => {
            let specific_client = XAIChatClient::from_config(config)?;
            Arc::new(specific_client)
        }
    };
    Ok(client)
}
