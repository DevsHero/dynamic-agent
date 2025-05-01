use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ ChatClient, CompletionResponse };
use crate::llm::LlmConfig;
use rllm::{
    builder::{ LLMBackend, LLMBuilder },
    chat::{ ChatMessage, ChatRole, MessageType },
    LLMProvider,
};

pub struct XAIChatClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl XAIChatClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let chat_model = model.unwrap_or_else(|| "grok-1".to_string());

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::XAI)
            .api_key(api_key)
            .model(&chat_model)
            .stream(false);

        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        if let Some(tokens) = max_tokens {
            builder = builder.max_tokens(tokens);
        }
        if let Some(temp) = temperature {
            builder = builder.temperature(temp);
        }

        let llm_provider = builder.build()?;

        Ok(Self { llm: llm_provider })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "XAI API key is required for XAIChatClient".to_string())?;
        let model = config.completion_model.clone();
        let base_url = config.base_url.clone();
        let max_tokens = None;
        let temperature = None;

        Self::new(api_key, model, base_url, max_tokens, temperature)
    }
}

#[async_trait]
impl ChatClient for XAIChatClient {
    async fn complete(
        &self,
        prompt: &str
    ) -> Result<CompletionResponse, Box<dyn StdError + Send + Sync>> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.to_string(),
            message_type: MessageType::Text,
        }];

        let response_text = self.llm.chat(&messages).await?;

        Ok(CompletionResponse { response: response_text.to_string() })
    }
}
