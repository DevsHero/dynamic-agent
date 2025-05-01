use async_trait::async_trait;
use std::error::Error as StdError;
use super::{ EmbeddingClient, EmbeddingResponse };
use crate::llm::LlmConfig;

use rllm::{ builder::{ LLMBackend, LLMBuilder }, LLMProvider };

pub struct DeepSeekEmbeddingClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl DeepSeekEmbeddingClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let embedding_model = model.unwrap_or_else(|| "deepseek-chat".to_string());

        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::DeepSeek)
            .api_key(api_key)
            .model(&embedding_model);

        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }

        let llm_provider = builder.build()?;

        Ok(Self { llm: llm_provider })
    }

    pub fn from_config(config: &LlmConfig) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let api_key = config.api_key
            .clone()
            .ok_or_else(|| "DeepSeek API key is required for DeepSeekEmbeddingClient".to_string())?;
        let model = config.embedding_model.clone();
        let base_url = config.base_url.clone();

        Self::new(api_key, model, base_url)
    }
}

#[async_trait]
impl EmbeddingClient for DeepSeekEmbeddingClient {
    async fn embed(
        &self,
        text: &str
    ) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>> {
        let mut embeddings = self.llm.embed(vec![text.to_string()]).await?;

        let embedding = embeddings
            .pop()
            .ok_or_else(|| "DeepSeek embedding generation returned no results".to_string())?;

        Ok(EmbeddingResponse { embedding })
    }
}
