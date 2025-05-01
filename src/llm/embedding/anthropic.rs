use async_trait::async_trait;
use log::warn;
use std::error::Error as StdError;
use super::{ EmbeddingClient, EmbeddingResponse };
use crate::llm::LlmConfig;
use rllm::{ builder::{ LLMBackend, LLMBuilder }, LLMProvider };

pub struct AnthropicEmbeddingClient {
    llm: Box<dyn LLMProvider + Send + Sync>,
}

impl AnthropicEmbeddingClient {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>
    ) -> Result<Self, Box<dyn StdError + Send + Sync>> {
        let embedding_model = model.unwrap_or_else(|| "claude-3-haiku-20240307".to_string());
        let mut builder = LLMBuilder::new()
            .backend(LLMBackend::Anthropic)
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
            .ok_or_else(||
                "Anthropic API key is required for AnthropicEmbeddingClient".to_string()
            )?;
        let model = config.embedding_model.clone();
        let base_url = config.base_url.clone();

        Self::new(api_key, model, base_url)
    }
}

#[async_trait]
impl EmbeddingClient for AnthropicEmbeddingClient {
    async fn embed(
        &self,
        text: &str
    ) -> Result<EmbeddingResponse, Box<dyn StdError + Send + Sync>> {
        warn!(
            "WARNING: Attempting to generate embeddings using Anthropic via rllm. This is likely non-functional."
        );
        let mut embeddings = self.llm.embed(vec![text.to_string()]).await?;

        let embedding = embeddings
            .pop()
            .ok_or_else(||
                "Anthropic embedding generation returned no results (or failed)".to_string()
            )?;

        Ok(EmbeddingResponse { embedding })
    }
}
